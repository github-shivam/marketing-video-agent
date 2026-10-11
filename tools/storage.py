"""S3 upload and presigned URLs.

Instagram fetches the video from a URL we supply, so the rendered MP4 has to
be reachable over HTTPS for the duration of the publish.
"""

from __future__ import annotations

import logging
import mimetypes
from pathlib import Path

import boto3
from botocore.config import Config as BotoConfig

from config import Settings

logger = logging.getLogger(__name__)


def _client(settings: Settings):
    """Build an S3 client with a generous timeout for large uploads.

    Signature version and addressing style are pinned. Left to defaults, a
    bucket outside us-east-1 can yield a presigned URL on the global host that
    S3 rejects with SignatureDoesNotMatch, and Instagram then cannot fetch the
    video.
    """
    return boto3.client(
        "s3",
        region_name=settings.s3_region,
        config=BotoConfig(
            read_timeout=300,
            retries={"max_attempts": 3},
            signature_version="s3v4",
            s3={"addressing_style": "virtual"},
        ),
    )


def build_key(settings: Settings, path: Path) -> str:
    """Return the S3 object key for a local file."""
    prefix = settings.s3_prefix.strip("/")
    return f"{prefix}/{path.name}" if prefix else path.name


def upload_and_presign(settings: Settings, path: Path) -> str:
    """Upload ``path`` to S3 and return a presigned HTTPS URL.

    Args:
        settings: Supplies the bucket, prefix, region and expiry.
        path: The local file to upload.

    Returns:
        A presigned URL valid for ``PRESIGN_EXPIRY`` seconds.

    Raises:
        ValueError: If no bucket is configured or the file is missing.
    """
    if not settings.s3_bucket:
        raise ValueError("S3_BUCKET is not set - Instagram needs a public video URL")
    if not path.exists():
        raise ValueError(f"file not found: {path}")

    key = build_key(settings, path)
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    client = _client(settings)

    logger.info("Uploading %s to s3://%s/%s", path.name, settings.s3_bucket, key)
    client.upload_file(
        str(path), settings.s3_bucket, key, ExtraArgs={"ContentType": content_type}
    )
    url = client.generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.s3_bucket, "Key": key},
        ExpiresIn=settings.presign_expiry,
    )
    logger.info("Presigned URL valid for %ss", settings.presign_expiry)
    return url
