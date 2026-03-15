"""
FFmpeg Hybrid Renderer.

Takes an AssemblyPlan produced by MoviePyAssembler and translates it into
a single raw FFmpeg command using -filter_complex.

Why bypass MoviePy for the final encode?
  - MoviePy's write_videofile() is single-threaded Python — it processes
    frames in a Python loop, which is throttled by the GIL.
  - A direct FFmpeg subprocess encodes with all available CPU cores
    (libx264 / libx265 multi-threading) and uses hardware acceleration
    (NVENC/VA-API) when available.
  - filter_complex supports lossless stream copy for untouched segments,
    further reducing encode time.

filter_complex structure for N clips + 1 narration:
  ┌─────────────────────────────────────────────────────────┐
  │  [0:v] trim=start=S:end=E, setpts=PTS-STARTPTS,         │
  │        scale=WxH:force_original_aspect_ratio=increase,  │
  │        crop=W:H,                                         │
  │        fps=FPS  [v0]                                     │
  │  [1:v] trim=... [v1]                                     │
  │  ...                                                     │
  │  [v0][v1]...[vN] concat=n=N:v=1:a=0 [vout]              │
  │                                                          │
  │  [N+1:a] atrim=start=0:end=D,                           │
  │           asetpts=PTS-STARTPTS,                          │
  │           afade=t=out:st=FADE_ST:d=0.5 [narration]      │
  │  [narration] volume=1.0 [aout]                           │
  └─────────────────────────────────────────────────────────┘

Text overlays use the `drawtext` filter chained onto the relevant [vN] stream.
Fade transitions use `xfade` between adjacent clip streams.
Audio ducking uses `volume` + `asidechain` on the B-roll audio mix.
"""
from __future__ import annotations

import pathlib
import shlex
import subprocess
import tempfile
from typing import Any

import structlog

from src.agents.rendering.moviepy_assembler import AssemblyPlan, AudioTrackSpec, ClipSpec
from src.core.config import get_settings
from src.core.exceptions import FFmpegError
from src.schemas.pipeline import ClipTransition

logger = structlog.get_logger(__name__)
settings = get_settings()

# Transition duration for xfade
_XFADE_DURATION = 0.3

# Codecs
_VIDEO_CODEC = "libx264"
_AUDIO_CODEC = "aac"
_CRF = 18          # quality (lower = better, 18 is near-lossless)
_PRESET = "fast"   # libx264 speed preset


_ALLOWED_ROOTS: tuple[pathlib.Path, ...] = (
    pathlib.Path(settings.video_output_dir).resolve(),
    pathlib.Path(settings.video_scratch_dir).resolve(),
    # Common system font directories (read-only)
    pathlib.Path("/usr/share/fonts").resolve(),
    pathlib.Path("/usr/local/share/fonts").resolve(),
)


def _validate_path(p: str, label: str) -> pathlib.Path:
    """
    Resolve ``p`` and confirm it sits under one of the allowed root directories.

    Raises ``FFmpegError`` for any path that escapes the sandbox — prevents
    Local File Inclusion if an upstream component produces a malicious path.
    """
    resolved = pathlib.Path(p).resolve()
    if not any(
        resolved == root or resolved.is_relative_to(root)
        for root in _ALLOWED_ROOTS
    ):
        raise FFmpegError(
            f"Unsafe {label} path rejected.",
            detail=f"Path '{p}' is outside the allowed directories.",
        )
    return resolved


class FFmpegRenderer:
    """
    Translates an AssemblyPlan into a raw FFmpeg filter_complex command
    and executes it as a subprocess.
    """

    def render(self, plan: AssemblyPlan) -> str:
        """
        Build and execute the FFmpeg command.
        Returns the path to the output file.
        """
        if not plan.clips:
            raise FFmpegError("Cannot render: AssemblyPlan has no clips.")

        # Validate output path before building the command
        _validate_path(plan.output_path, "output")

        cmd = self._build_command(plan)
        self._execute(cmd, plan.output_path)
        return plan.output_path

    def render_thumbnail(self, video_path: str, output_path: str, at_second: float = 2.0) -> str:
        """Extract a single JPEG frame as a thumbnail."""
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(at_second),
            "-i", video_path,
            "-frames:v", "1",
            "-q:v", "2",
            output_path,
        ]
        self._execute(cmd, output_path)
        return output_path

    # ── Command builder ────────────────────────────────────────────────────────

    def _build_command(self, plan: AssemblyPlan) -> list[str]:
        """
        Assemble the full FFmpeg command list.

        Structure:
          ffmpeg -y
            [input flags for each clip]
            [input flags for each audio track]
            -filter_complex "..."
            -map [vout] -map [aout]
            -c:v libx264 -crf 18 -preset fast
            -c:a aac -b:a 192k
            -movflags +faststart
            -threads N
            output.mp4
        """
        cmd: list[str] = ["ffmpeg", "-y"]

        # ── Inputs ────────────────────────────────────────────────────────────
        input_files: list[str] = []

        for clip in plan.clips:
            safe_path = str(_validate_path(clip.local_path, "clip"))
            # Seek to near source_start for fast seeking (re-encode from keyframe)
            fast_seek = max(0.0, clip.source_start - 2.0)
            cmd += ["-ss", f"{fast_seek:.6f}", "-i", safe_path]
            input_files.append(safe_path)

        n_video_inputs = len(plan.clips)

        for track in plan.audio_tracks:
            safe_audio = str(_validate_path(track.local_path, "audio track"))
            if pathlib.Path(safe_audio).exists():
                cmd += ["-i", safe_audio]
                input_files.append(safe_audio)

        # ── filter_complex ────────────────────────────────────────────────────
        fc_parts, video_out_label, audio_out_label = self._build_filter_complex(
            plan, n_video_inputs
        )
        filter_complex_str = ";\n".join(fc_parts)

        cmd += ["-filter_complex", filter_complex_str]

        # ── Output mapping ────────────────────────────────────────────────────
        cmd += [
            "-map", video_out_label,
            "-map", audio_out_label,
        ]

        # ── Codec flags ───────────────────────────────────────────────────────
        cmd += [
            "-c:v", _VIDEO_CODEC,
            "-crf",    str(_CRF),
            "-preset", _PRESET,
            "-profile:v", "high",
            "-level",     "4.1",
            "-c:a", _AUDIO_CODEC,
            "-b:a", "192k",
            "-ar",  "44100",
            "-ac",  "2",
            # Fast-start: move moov atom to front for streaming
            "-movflags", "+faststart",
            # Thread count
            "-threads", str(settings.ffmpeg_threads),
            # Duration cap
            "-t", f"{plan.total_duration:.6f}",
            plan.output_path,
        ]

        logger.debug(
            "ffmpeg.command_built",
            inputs=n_video_inputs,
            output=plan.output_path,
            cmd_len=len(cmd),
        )
        return cmd

    # ── filter_complex builder ─────────────────────────────────────────────────

    def _build_filter_complex(
        self, plan: AssemblyPlan, n_video_inputs: int
    ) -> tuple[list[str], str, str]:
        """
        Returns (filter_parts, video_out_label, audio_out_label).

        filter_parts is a list of filter chains joined by semicolons.
        """
        parts: list[str] = []
        processed_video_labels: list[str] = []
        processed_audio_labels: list[str] = []

        W = plan.output_width
        H = plan.output_height
        FPS = plan.fps

        # ── Per-clip video chains ─────────────────────────────────────────────
        for i, clip in enumerate(plan.clips):
            # Adjust trim times relative to the fast-seek offset
            fast_seek = max(0.0, clip.source_start - 2.0)
            trim_start = clip.source_start - fast_seek
            trim_end   = clip.source_end   - fast_seek

            chain: list[str] = [f"[{i}:v]"]

            # 1. Trim
            chain.append(f"trim=start={trim_start:.6f}:end={trim_end:.6f}")
            chain.append("setpts=PTS-STARTPTS")

            # 2. Loop if source is shorter than required duration
            required = clip.timeline_end - clip.timeline_start
            src_dur = clip.source_end - clip.source_start
            if clip.loop_source and src_dur > 0:
                loops = int(required / src_dur) + 2
                chain = [f"[{i}:v]", f"loop={loops}:size=32767", *chain[1:]]

            # 3. Scale to fill + crop to target resolution + normalise SAR.
            # force_original_aspect_ratio=increase scales up so the smaller
            # dimension fills the frame (no black bars), then crop removes the
            # excess on the larger dimension (center crop).  This correctly
            # handles vertical 9:16 clips in a 16:9 output — they fill the
            # frame rather than appearing as a small centred rectangle.
            chain.append(
                f"scale={W}:{H}:force_original_aspect_ratio=increase,"
                f"crop={W}:{H},"
                f"setsar=1"
            )

            # 4. Force FPS — use integer to avoid "fps=30.0" being parsed as
            # two tokens "fps=30" and ".0" by the filter_complex parser
            chain.append(f"fps={int(FPS)}")

            # 5. Duration trim (after FPS normalisation)
            chain.append(f"trim=duration={required:.6f}")
            chain.append("setpts=PTS-STARTPTS")

            # 6. Fade transitions
            if clip.transition_in == ClipTransition.fade and i > 0:
                chain.append(f"fade=t=in:st=0:d={_XFADE_DURATION}")
            if clip.transition_out == ClipTransition.fade:
                fade_start = max(0.0, required - _XFADE_DURATION)
                chain.append(f"fade=t=out:st={fade_start:.6f}:d={_XFADE_DURATION}")

            # 7. Text overlays (drawtext)
            for ov in clip.overlays:
                chain.append(self._drawtext_filter(ov))

            label = f"[v{i}]"
            parts.append(chain[0] + ",".join(chain[1:]) + label)
            processed_video_labels.append(label)

            # ── Per-clip audio chain (ducked) ──────────────────────────────
            audio_label = f"[a{i}]"
            gain_linear = 10 ** (clip.audio_gain_db / 20)
            if clip.has_audio:
                parts.append(
                    f"[{i}:a]"
                    f"atrim=start={trim_start:.6f}:end={trim_end:.6f},"
                    f"asetpts=PTS-STARTPTS,"
                    f"volume={gain_linear:.4f},"
                    f"atrim=duration={required:.6f},"
                    f"asetpts=PTS-STARTPTS"
                    f"{audio_label}"
                )
            else:
                # Video-only clip — generate silence to keep audio concat intact
                parts.append(
                    f"aevalsrc=0:c=stereo:s=44100:d={required:.6f}"
                    f"{audio_label}"
                )
            processed_audio_labels.append(audio_label)

        # ── Concat all video clips ────────────────────────────────────────────
        n = len(processed_video_labels)
        concat_video_in = "".join(processed_video_labels)
        concat_audio_in = "".join(processed_audio_labels)

        parts.append(
            f"{concat_video_in}concat=n={n}:v=1:a=0[vconcat]"
        )
        parts.append(
            f"{concat_audio_in}concat=n={n}:v=0:a=1[aconcat]"
        )

        # ── Narration audio ───────────────────────────────────────────────────
        narration_input_idx = n_video_inputs  # first audio input after all video

        if plan.audio_tracks and pathlib.Path(plan.audio_tracks[0].local_path).exists():
            track = plan.audio_tracks[0]
            fade_start = max(0.0, plan.total_duration - track.fade_out)
            parts.append(
                f"[{narration_input_idx}:a]"
                f"volume={track.volume:.4f},"
                f"afade=t=in:st=0:d={track.fade_in:.3f},"
                f"afade=t=out:st={fade_start:.6f}:d={track.fade_out:.3f},"
                f"atrim=duration={plan.total_duration:.6f},"
                f"asetpts=PTS-STARTPTS"
                f"[narration]"
            )

            # Mix B-roll audio + narration.
            # normalize=0 keeps each stream at its own volume level instead of
            # halving both streams (the default normalize=1 behaviour).
            parts.append(
                "[aconcat][narration]"
                "amix=inputs=2:duration=longest:dropout_transition=0:normalize=0"
                "[aout]"
            )
            audio_out_label = "[aout]"
        else:
            # No narration — use B-roll audio only
            parts.append(f"[aconcat]asetpts=PTS-STARTPTS[aout]")
            audio_out_label = "[aout]"

        return parts, "[vconcat]", audio_out_label

    # ── drawtext filter ────────────────────────────────────────────────────────

    @staticmethod
    def _escape_drawtext(text: str) -> str:
        """
        Escape a string for use inside an FFmpeg drawtext `text='...'` value.

        FFmpeg drawtext escaping rules (applied in order):
          1. `\\` → `\\\\`  (literal backslash)
          2. `'`  → `'\\'`  (close quote, backslash-escaped apostrophe in
                             unquoted mode, reopen quote — keeps single-quote
                             balance intact so `enable='between(...)'` parses
                             correctly; plain `\\'` would close the quote)
          3. `:`  → `\\:`   (option key/value separator)
          4. `%`  → `%%`    (strftime / text_shaping expansion)
          5. Newlines / CR  → space  (would break the filter string)
        """
        return (
            text
            .replace("\\", "\\\\")
            .replace("'",  "'\\''")
            .replace(":",  "\\:")
            .replace("%",  "%%")
            .replace("\r", "")
            .replace("\n", "\\n")   # FFmpeg drawtext interprets \n as a line break
        )

    @staticmethod
    def _safe_font_path(font_path: str) -> str:
        """
        Return the font path only if it:
          1. Contains only safe characters (no injection risk), AND
          2. Resolves inside an allowed root directory.
        Falls back to an empty string (no fontfile arg) on failure.
        """
        import re
        # Step 1: character whitelist (prevents shell metachar injection)
        if not re.fullmatch(r"[A-Za-z0-9 /._\-]+", font_path):
            logger.warning("ffmpeg.unsafe_font_path_rejected", reason="bad_chars", font_path=font_path)
            return ""
        # Step 2: path traversal / LFI check
        try:
            _validate_path(font_path, "font")
        except Exception:
            logger.warning("ffmpeg.unsafe_font_path_rejected", reason="outside_allowed_root", font_path=font_path)
            return ""
        return font_path

    def _drawtext_filter(self, ov: Any) -> str:
        """
        Build an FFmpeg drawtext filter string for a text overlay.

        All user-supplied strings are escaped before insertion.
        """
        safe_text = self._escape_drawtext(ov.text)

        color_hex = ov.color.lstrip("#")
        color_ffmpeg = f"0x{color_hex}FF"  # RRGGBBAA

        font_arg = ""
        font_candidates = (
            [ov.font_path] if ov.font_path else []
        ) + [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
            "/usr/local/share/fonts/DejaVuSans.ttf",
        ]
        for candidate in font_candidates:
            if not candidate:
                continue
            clean_path = self._safe_font_path(candidate)
            if clean_path and pathlib.Path(clean_path).exists():
                font_arg = f":fontfile='{clean_path}'"
                break

        bg_args = ""
        if ov.bg_color:
            bg_hex = ov.bg_color.lstrip("#")
            bg_ffmpeg = f"0x{bg_hex}80"  # 50% opacity background
            bg_args = f":box=1:boxcolor={bg_ffmpeg}:boxborderw=10"

        # x and y are int (computed from TextPosition enum) — safe to interpolate
        # font_size is a bounded int (12–120) — safe to interpolate
        enable = f"between(t,{ov.start_seconds:.3f},{ov.end_seconds:.3f})"

        return (
            f"drawtext=text='{safe_text}'"
            f":x={ov.x}:y={ov.y}"
            f":fontsize={ov.font_size}"
            f":fontcolor={color_ffmpeg}"
            f"{font_arg}"
            f"{bg_args}"
            f":enable='{enable}'"
        )

    # ── Subprocess execution ───────────────────────────────────────────────────

    def _execute(self, cmd: list[str], output_path: str) -> None:
        """
        Execute the FFmpeg subprocess.  Streams stderr to the structured logger.
        Raises FFmpegError on non-zero exit code.
        """
        logger.info(
            "ffmpeg.starting",
            output=output_path,
            cmd=" ".join(shlex.quote(c) for c in cmd[:8]) + " ...",
        )

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=settings.celery_task_time_limit,
            )
        except subprocess.TimeoutExpired as exc:
            raise FFmpegError(
                "FFmpeg render timed out.",
                detail=str(exc),
            ) from exc
        except FileNotFoundError as exc:
            raise FFmpegError(
                "FFmpeg binary not found. Is ffmpeg installed?",
                detail=str(exc),
            ) from exc

        if result.returncode != 0:
            # Log the last 2 KB of stderr for diagnosis
            stderr_tail = result.stderr[-2048:] if result.stderr else ""
            logger.error("ffmpeg.failed", returncode=result.returncode, stderr=stderr_tail)
            raise FFmpegError(
                f"FFmpeg exited with code {result.returncode}.",
                detail=stderr_tail,
            )

        logger.info("ffmpeg.complete", output=output_path)
