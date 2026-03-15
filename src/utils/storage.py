"""
S3 upload utility.

Wraps boto3 with retry logic.  Used by the rendering node to push
the final MP4 and thumbnail to S3 and return a presigned URL.
"""
from __future__ import annotations

import mimetypes
import pathlib

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def upload_to_s3(local_path: str, *, s3_key: str) -> str:
    """
    Upload a local file to S3 and return its public HTTPS URL.

    Falls back to a local file:// URL when AWS credentials are not
    configured (useful for local development).
    """
    if not settings.aws_s3_bucket or not settings.aws_access_key_id:
        logger.warning("storage.s3_not_configured", local_path=local_path)
        # Serve locally via the /output static mount
        rel = pathlib.Path(local_path).relative_to(settings.video_output_dir)
        return f"http://{settings.api_host}:{settings.api_port}/output/{rel}"

    import boto3

    content_type, _ = mimetypes.guess_type(local_path)
    content_type = content_type or "application/octet-stream"

    s3 = boto3.client(
        "s3",
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
    )

    with pathlib.Path(local_path).open("rb") as fh:
        s3.put_object(
            Bucket=settings.aws_s3_bucket,
            Key=s3_key,
            Body=fh,
            ContentType=content_type,
        )

    url = f"https://{settings.aws_s3_bucket}.s3.{settings.aws_region}.amazonaws.com/{s3_key}"
    logger.info("storage.uploaded", s3_key=s3_key, url=url)
    return url
