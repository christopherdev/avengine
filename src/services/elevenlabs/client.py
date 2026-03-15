"""
ElevenLabs TTS Service Client.

Generates narration audio for the full video script and extracts
word-level and character-level timestamps from the alignment response.

ElevenLabs `with_timestamps` endpoint returns:
  {
    "audio_base64": "...",
    "alignment": {
      "characters":                    ["H","e","l","l","o"," ","w",...],
      "character_start_times_seconds": [0.0, 0.07, 0.13, ...],
      "character_end_times_seconds":   [0.07, 0.13, 0.19, ...]
    },
    "normalized_alignment": {
      "characters":                    ["Hello", " ", "world",...],
      "character_start_times_seconds": [...],
      "character_end_times_seconds":   [...]
    }
  }

We post-process the character-level alignment into WordTimestamp objects
so the timeline calculator can assign precise B-roll cut points.
"""
from __future__ import annotations

import base64
import pathlib
from dataclasses import dataclass, field
from typing import Any

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.config import get_settings
from src.core.exceptions import ElevenLabsError

logger = structlog.get_logger(__name__)
settings = get_settings()

_BASE_URL = "https://api.elevenlabs.io/v1"
_TIMEOUT = httpx.Timeout(120.0, connect=10.0)


# ── Data models ────────────────────────────────────────────────────────────────

@dataclass
class WordTimestamp:
    """A single word with its precise start/end time in the audio."""
    word: str
    start_seconds: float
    end_seconds: float

    @property
    def duration(self) -> float:
        return self.end_seconds - self.start_seconds


@dataclass
class SceneAudio:
    """
    TTS output for one ScriptScene.

    Contains:
      - The local path to the synthesised .mp3 file
      - Word-level timestamps for frame-accurate B-roll cut points
      - Character-level timestamps for fine-grained subtitle rendering
    """
    scene_id: str
    local_path: str
    duration_seconds: float
    word_timestamps: list[WordTimestamp] = field(default_factory=list)
    character_timestamps: list[WordTimestamp] = field(default_factory=list)
    raw_alignment: dict[str, Any] = field(default_factory=dict)


@dataclass
class NarrationAudio:
    """Full narration audio for the entire script."""
    task_id: str
    local_path: str           # merged narration file
    total_duration: float
    scenes: list[SceneAudio] = field(default_factory=list)


# ── Client ────────────────────────────────────────────────────────────────────

class ElevenLabsClient:
    """
    ElevenLabs TTS client.

    Uses the `/text-to-speech/{voice_id}/with-timestamps` endpoint to
    get audio + alignment data in a single request.
    """

    def __init__(self) -> None:
        if not settings.elevenlabs_api_key:
            raise ElevenLabsError("ELEVENLABS_API_KEY is not configured.")
        self._api_key = settings.elevenlabs_api_key
        self._voice_id = settings.elevenlabs_voice_id
        self._client = httpx.Client(
            base_url=_BASE_URL,
            headers={
                "xi-api-key": self._api_key,
                "Content-Type": "application/json",
            },
            timeout=_TIMEOUT,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def synthesise_script(
        self,
        task_id: str,
        scenes: list[dict[str, Any]],
        voice_id: str | None = None,
    ) -> NarrationAudio:
        """
        Synthesise TTS audio for each scene separately.

        Generating per-scene allows precise timestamp alignment per-scene
        without accumulated drift from a single long synthesis request.
        Returns a NarrationAudio containing per-scene audio files and
        merged narration.
        """
        scratch = pathlib.Path(settings.video_scratch_dir) / task_id / "audio"
        scratch.mkdir(parents=True, exist_ok=True)

        voice = voice_id or self._voice_id
        scene_audios: list[SceneAudio] = []

        for scene_dict in scenes:
            scene_id: str = scene_dict.get("scene_id", "S00")
            narration: str = scene_dict.get("narration", "")
            if not narration.strip():
                continue

            logger.info("elevenlabs.synthesising_scene", scene_id=scene_id)
            audio_path = scratch / f"{scene_id}.mp3"

            scene_audio = self._synthesise_scene(
                scene_id=scene_id,
                text=narration,
                voice_id=voice,
                output_path=audio_path,
            )
            scene_audios.append(scene_audio)
            logger.info(
                "elevenlabs.scene_done",
                scene_id=scene_id,
                duration=round(scene_audio.duration_seconds, 2),
                words=len(scene_audio.word_timestamps),
            )

        # Merge all per-scene MP3s into one narration file with ffmpeg
        merged_path = scratch / "narration.mp3"
        total_duration = self._merge_audio(
            [sa.local_path for sa in scene_audios],
            output_path=merged_path,
        )

        return NarrationAudio(
            task_id=task_id,
            local_path=str(merged_path),
            total_duration=total_duration,
            scenes=scene_audios,
        )

    # ── Core synthesis ─────────────────────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=15))
    def _synthesise_scene(
        self,
        scene_id: str,
        text: str,
        voice_id: str,
        output_path: pathlib.Path,
    ) -> SceneAudio:
        """
        Call the ElevenLabs `with-timestamps` endpoint for one scene.
        Writes the audio file and returns a SceneAudio with parsed timestamps.
        """
        payload = {
            "text": text,
            "model_id": "eleven_turbo_v2_5",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "style": 0.0,
                "use_speaker_boost": True,
            },
            "output_format": "mp3_44100_128",
        }

        response = self._client.post(
            f"/text-to-speech/{voice_id}/with-timestamps",
            json=payload,
        )

        if response.status_code != 200:
            raise ElevenLabsError(
                f"ElevenLabs API error {response.status_code} for scene {scene_id}",
                detail=response.text[:500],
            )

        data: dict[str, Any] = response.json()
        audio_b64: str = data.get("audio_base64", "")
        alignment: dict[str, Any] = data.get("alignment", {})

        # Write audio file
        audio_bytes = base64.b64decode(audio_b64)
        output_path.write_bytes(audio_bytes)

        # Parse timestamps
        word_timestamps = _parse_word_timestamps(alignment)
        char_timestamps = _parse_character_timestamps(alignment)

        # Infer duration from last character timestamp
        duration = 0.0
        if alignment.get("character_end_times_seconds"):
            duration = alignment["character_end_times_seconds"][-1]
        elif word_timestamps:
            duration = word_timestamps[-1].end_seconds

        return SceneAudio(
            scene_id=scene_id,
            local_path=str(output_path),
            duration_seconds=duration,
            word_timestamps=word_timestamps,
            character_timestamps=char_timestamps,
            raw_alignment=alignment,
        )

    # ── Audio merging ──────────────────────────────────────────────────────────

    def _merge_audio(self, paths: list[str], output_path: pathlib.Path) -> float:
        """
        Concatenate per-scene MP3 files into one narration track using FFmpeg.
        Returns the total duration in seconds.
        """
        import subprocess
        import tempfile

        if not paths:
            return 0.0

        if len(paths) == 1:
            import shutil
            shutil.copy(paths[0], output_path)
        else:
            # Write a concat list file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as f:
                for p in paths:
                    f.write(f"file '{p}'\n")
                concat_file = f.name

            cmd = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_file,
                "-c", "copy",
                str(output_path),
            ]
            subprocess.run(cmd, check=True, capture_output=True)

        return _probe_duration(str(output_path))


# ── Timestamp parsers ─────────────────────────────────────────────────────────

def _parse_word_timestamps(alignment: dict[str, Any]) -> list[WordTimestamp]:
    """
    Group character-level timestamps into word-level timestamps.

    Algorithm:
      Walk the characters array.  Accumulate characters into a buffer until
      a space or end-of-string is reached.  The word's start time is the
      first character's start; end time is the last character's end.
    """
    chars: list[str] = alignment.get("characters", [])
    starts: list[float] = alignment.get("character_start_times_seconds", [])
    ends: list[float] = alignment.get("character_end_times_seconds", [])

    if not chars or len(chars) != len(starts):
        return []

    words: list[WordTimestamp] = []
    buf: list[str] = []
    word_start = 0.0
    word_end = 0.0

    for i, ch in enumerate(chars):
        if ch == " " or ch == "\n":
            if buf:
                words.append(WordTimestamp(
                    word="".join(buf),
                    start_seconds=word_start,
                    end_seconds=word_end,
                ))
                buf = []
        else:
            if not buf:
                word_start = starts[i]
            buf.append(ch)
            word_end = ends[i]

    # Flush trailing word
    if buf:
        words.append(WordTimestamp(
            word="".join(buf),
            start_seconds=word_start,
            end_seconds=word_end,
        ))

    return words


def _parse_character_timestamps(alignment: dict[str, Any]) -> list[WordTimestamp]:
    """Return raw character-level timestamps as WordTimestamp objects."""
    chars: list[str] = alignment.get("characters", [])
    starts: list[float] = alignment.get("character_start_times_seconds", [])
    ends: list[float] = alignment.get("character_end_times_seconds", [])

    return [
        WordTimestamp(word=ch, start_seconds=s, end_seconds=e)
        for ch, s, e in zip(chars, starts, ends, strict=False)
        if ch.strip()
    ]


def _probe_duration(path: str) -> float:
    """Use ffprobe to get audio file duration in seconds."""
    import subprocess
    import json as _json

    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = _json.loads(result.stdout)
        for stream in data.get("streams", []):
            if "duration" in stream:
                return float(stream["duration"])
    except Exception:  # noqa: BLE001
        pass
    return 0.0


# ── Singleton ─────────────────────────────────────────────────────────────────

_client: ElevenLabsClient | None = None


def get_elevenlabs_client() -> ElevenLabsClient:
    global _client  # noqa: PLW0603
    if _client is None:
        _client = ElevenLabsClient()
    return _client
