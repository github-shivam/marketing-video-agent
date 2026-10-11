"""Instagram Graph API content publishing. Deterministic code, no LLM.

Reels flow, per Meta's documentation:

1. ``POST /<IG_USER_ID>/media`` with ``media_type=REELS`` and ``video_url``
   creates a container.
2. ``GET /<CONTAINER_ID>?fields=status_code`` is polled until ``FINISHED``.
   Meta advises checking about once a minute for no more than five minutes.
3. ``POST /<IG_USER_ID>/media_publish`` with ``creation_id`` publishes it.

Accounts are limited to 100 API-published posts per rolling 24 hours, enforced
on ``media_publish``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

import httpx

from config import Settings

logger = logging.getLogger(__name__)

#: Facebook Login for Business tokens use this host. Instagram Login tokens use
#: ``https://graph.instagram.com``, selected with ``IG_GRAPH_HOST``.
GRAPH_HOST = "https://graph.facebook.com"

#: Resumable uploads go to a different host than the rest of the Graph API.
RUPLOAD_HOST = "https://rupload.facebook.com"

#: Meta rejects reels larger than this, resumable or not.
MAX_UPLOAD_BYTES = 300 * 1024 * 1024

#: Only this container state may be published.
STATUS_READY = "FINISHED"

#: Terminal failure states. Anything else means keep polling.
STATUS_FAILED = frozenset({"ERROR", "EXPIRED"})


class InstagramError(RuntimeError):
    """Raised on an API error, a failed container, or a publish timeout."""


def base_url(settings: Settings) -> str:
    """Return the versioned Graph API base URL."""
    return f"{settings.ig_graph_host.rstrip('/')}/{settings.ig_api_version}"


def _raise_for_api_error(response: httpx.Response, action: str) -> dict[str, Any]:
    """Return the JSON body, or raise with the API's own error text.

    Meta puts the useful diagnosis in the response body rather than the status
    line, so the body is always surfaced in the exception.
    """
    try:
        payload = response.json()
    except ValueError:
        raise InstagramError(
            f"{action} failed: HTTP {response.status_code} with a non-JSON body: "
            f"{response.text[:500]}"
        ) from None

    if response.status_code >= 400 or "error" in payload:
        detail = payload.get("error", payload)
        raise InstagramError(
            f"{action} failed: HTTP {response.status_code} - {detail}"
        )
    return payload


async def create_reel_container(
    client: httpx.AsyncClient, settings: Settings, video_url: str, caption: str
) -> str:
    """Create a REELS media container and return its ID."""
    response = await client.post(
        f"{base_url(settings)}/{settings.ig_user_id}/media",
        data={
            "media_type": "REELS",
            "video_url": video_url,
            "caption": caption,
            "access_token": settings.ig_access_token,
        },
    )
    payload = _raise_for_api_error(response, "Container creation")
    container_id = payload.get("id")
    if not container_id:
        raise InstagramError(f"Container creation returned no id: {payload}")
    logger.info("Created container %s", container_id)
    return str(container_id)


async def create_resumable_container(
    client: httpx.AsyncClient, settings: Settings, caption: str
) -> str:
    """Open a resumable upload session and return the container ID.

    Unlike the ``video_url`` flow, this never asks Meta to fetch anything, so
    the video does not need to be hosted anywhere.
    """
    response = await client.post(
        f"{base_url(settings)}/{settings.ig_user_id}/media",
        data={
            "media_type": "REELS",
            "upload_type": "resumable",
            "caption": caption,
            "access_token": settings.ig_access_token,
        },
    )
    payload = _raise_for_api_error(response, "Resumable session creation")
    container_id = payload.get("id")
    if not container_id:
        raise InstagramError(f"Resumable session returned no id: {payload}")
    logger.info("Opened resumable session %s", container_id)
    return str(container_id)


async def upload_video_bytes(
    client: httpx.AsyncClient, settings: Settings, container_id: str, path: Path
) -> None:
    """Upload a local video file into an open resumable session.

    Args:
        client: The HTTP client.
        settings: Supplies the access token and API version.
        container_id: The container returned by the session creation call.
        path: The local MP4 to send.

    Raises:
        InstagramError: If the file is missing, too large, or the upload fails.
    """
    if not path.exists():
        raise InstagramError(f"video file not found: {path}")

    size = path.stat().st_size
    if size > MAX_UPLOAD_BYTES:
        raise InstagramError(
            f"video is {size / 1024 / 1024:.0f}MB; Instagram's limit is "
            f"{MAX_UPLOAD_BYTES // 1024 // 1024}MB"
        )

    logger.info("Uploading %.1fMB to the resumable session", size / 1024 / 1024)
    response = await client.post(
        f"{RUPLOAD_HOST}/ig-api-upload/{settings.ig_api_version}/{container_id}",
        headers={
            "Authorization": f"OAuth {settings.ig_access_token}",
            "offset": "0",
            "file_size": str(size),
        },
        content=path.read_bytes(),
    )
    payload = _raise_for_api_error(response, "Resumable upload")
    if not payload.get("success"):
        raise InstagramError(f"Resumable upload did not succeed: {payload}")
    logger.info("Upload complete")


async def wait_for_container(
    client: httpx.AsyncClient, settings: Settings, container_id: str
) -> None:
    """Poll a container until it reports FINISHED.

    Raises:
        InstagramError: If the container errors, expires, or never finishes
            within ``IG_POLL_TIMEOUT``.
    """
    deadline = time.monotonic() + settings.ig_poll_timeout
    while True:
        response = await client.get(
            f"{base_url(settings)}/{container_id}",
            params={
                "fields": "status_code",
                "access_token": settings.ig_access_token,
            },
        )
        payload = _raise_for_api_error(response, "Container status check")
        status = payload.get("status_code")

        if status == STATUS_READY:
            logger.info("Container %s is ready", container_id)
            return
        if status in STATUS_FAILED:
            raise InstagramError(
                f"Container {container_id} reported {status}: {payload}"
            )
        if time.monotonic() > deadline:
            raise InstagramError(
                f"Container {container_id} still {status} after "
                f"{settings.ig_poll_timeout}s"
            )
        await asyncio.sleep(settings.ig_poll_interval)


async def publish_container(
    client: httpx.AsyncClient, settings: Settings, container_id: str
) -> str:
    """Publish a finished container and return the published media ID."""
    response = await client.post(
        f"{base_url(settings)}/{settings.ig_user_id}/media_publish",
        data={
            "creation_id": container_id,
            "access_token": settings.ig_access_token,
        },
    )
    payload = _raise_for_api_error(response, "Publish")
    media_id = payload.get("id")
    if not media_id:
        raise InstagramError(f"Publish returned no id: {payload}")
    return str(media_id)


async def publish_reel(
    settings: Settings,
    caption: str,
    *,
    video_url: str | None = None,
    local_path: Path | None = None,
) -> str:
    """Publish a reel end to end and return the media ID.

    Two upload strategies, selected by ``UPLOAD_STRATEGY``:

    * ``resumable`` (default) sends the local file's bytes straight to Meta.
      Nothing needs to be publicly hosted. Requires an app using Facebook
      Login for Business.
    * ``url`` hands Meta a public HTTPS URL to fetch, which is what S3 is for.

    Honours ``DRY_RUN``: when set, nothing is sent to Meta and the string
    ``"dry-run"`` is returned instead.

    Args:
        settings: Credentials, API version and upload strategy.
        caption: The caption, hashtags included.
        video_url: Publicly fetchable URL, for the ``url`` strategy.
        local_path: Local MP4, for the ``resumable`` strategy.

    Returns:
        The published media ID, or ``"dry-run"``.

    Raises:
        InstagramError: On missing credentials, a missing source, or any API
            failure.
    """
    resumable = settings.upload_strategy == "resumable"

    if settings.dry_run:
        source = local_path if resumable else video_url
        logger.info("DRY_RUN=true - skipping the Instagram API call")
        logger.info("Would publish via %s: %s", settings.upload_strategy, source)
        logger.info("Caption: %.120s", caption)
        return "dry-run"

    if not settings.ig_user_id or not settings.ig_access_token:
        raise InstagramError(
            "IG_USER_ID and IG_ACCESS_TOKEN are required to publish"
        )

    async with httpx.AsyncClient(timeout=300) as client:
        if resumable:
            if not local_path:
                raise InstagramError(
                    "UPLOAD_STRATEGY=resumable needs a local file to upload"
                )
            container_id = await create_resumable_container(
                client, settings, caption
            )
            await upload_video_bytes(client, settings, container_id, local_path)
        else:
            if not video_url:
                raise InstagramError(
                    "UPLOAD_STRATEGY=url needs a publicly fetchable video_url"
                )
            container_id = await create_reel_container(
                client, settings, video_url, caption
            )

        await wait_for_container(client, settings, container_id)
        media_id = await publish_container(client, settings, container_id)
        logger.info("Published reel %s", media_id)
        return media_id
