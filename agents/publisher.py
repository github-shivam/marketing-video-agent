"""The video and publish nodes.

Both are deterministic code rather than agents. Video generation and the
Graph API are places where a wrong token costs real money or corrupts a post,
so no LLM makes decisions here - the screenplay was already rendered into
prompts by :func:`agents.render.render_prompts` before the human approved it.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from agents.common import plan_for_run
from config import Settings
from state import Media, PostState
from tools.instagram import publish_reel
from tools.media import trim_video
from tools.storage import upload_and_presign

logger = logging.getLogger(__name__)

#: The video models the pipeline can write prompts for.
VIDEO_MODELS = ("veo", "seedance")


def select_video_provider(settings: Settings):
    """Pick the video provider for this run.

    Veo 3.1 on Vertex AI is the live generator (Google Cloud credentials, a
    project and a bucket). Seedance 2.5 has its prompt format and duration
    handling in place, but its API is not wired in yet, so it runs against the
    no-cost stub. Veo falls back to the stub when it is not configured, so the
    graph can be exercised end to end without spending money.
    """
    if settings.video_model == "seedance":
        from providers.stub import StubVideoProvider

        logger.warning(
            "Seedance 2.5 API is not wired in yet - using the stub video provider. "
            "The prompt above is what Seedance would receive."
        )
        return StubVideoProvider(settings)

    if settings.veo_live:
        from providers.veo import VeoVideoProvider

        return VeoVideoProvider(settings)

    from providers.stub import StubVideoProvider

    logger.warning(
        "GOOGLE_CLOUD_PROJECT and VEO_GCS_BUCKET not set - using the stub video provider"
    )
    return StubVideoProvider(settings)


def save_prompts(prompts: list[str], video_path: Path) -> Path:
    """Write the prompts that produced a video next to it.

    The prompt is the real deliverable of the writing stages, and a video with
    no record of what made it cannot be iterated on.
    """
    target = video_path.with_suffix(".prompt.txt")
    body = "\n\n".join(
        f"=== Prompt {i} ===\n{prompt}" for i, prompt in enumerate(prompts)
    )
    target.write_text(body + "\n", encoding="utf-8")
    return target


def video_node(settings: Settings, provider: Any = None):
    """Build the video node.

    Args:
        settings: Supplies the Veo and S3 configuration.
        provider: Optional provider override, used by the tests.
    """

    async def run(state: PostState) -> dict[str, Any]:
        prompts = state.get("prompts") or []
        if not prompts:
            return {"error": "no prompts to generate from"}
        plan = plan_for_run(settings, state)
        brief = state.get("asset_brief")
        references = [
            Path(p) for p in (brief.reference_files if brief else []) if Path(p).is_file()
        ]
        active = provider or select_video_provider(settings)
        logger.info(
            "Generating a %ds video with %r (%d prompt(s), %ds generated)",
            plan.seconds, active.name, len(prompts), plan.generated_seconds,
        )

        try:
            # Only passed when there are photos, so a provider that knows
            # nothing about them is called exactly as before.
            extra = {"references": references} if references else {}
            local_path = await active.generate(prompts, settings.output_dir, plan, **extra)
        except Exception as exc:  # noqa: BLE001 - reported to the CLI, not raised
            logger.error("Video generation failed: %s", exc)
            return {"error": f"video generation failed: {exc}"}

        # Veo makes fixed lengths, so a 20s video was generated as 22. Cut it to
        # the length that was asked for. If that cannot be done the file is left
        # whole and its real length is what gets reported, rather than a promise
        # the video does not keep.
        kept_seconds = plan.seconds
        if plan.needs_trim and not await trim_video(local_path, plan.seconds):
            kept_seconds = plan.generated_seconds

        prompt_file = save_prompts(prompts, local_path)

        # With the resumable strategy the bytes go straight to Meta, so the
        # file never needs to be hosted anywhere.
        urls: list[str] = []
        if settings.upload_strategy == "resumable":
            urls.append(local_path.resolve().as_uri())
        elif settings.s3_bucket:
            # boto3 is blocking; keep the event loop free for other chats.
            urls.append(await asyncio.to_thread(upload_and_presign, settings, local_path))
        else:
            logger.warning(
                "UPLOAD_STRATEGY=url but S3_BUCKET is not set - no public URL. "
                "A real publish will fail; DRY_RUN runs are unaffected."
            )
            urls.append(local_path.resolve().as_uri())

        return {
            "media": Media(
                media_type="REELS",
                urls=urls,
                local_files=[str(local_path)],
                prompt_file=str(prompt_file),
                duration_seconds=kept_seconds,
                notes=list(getattr(active, "notes", None) or []),
            )
        }

    return run


def build_caption(state: PostState) -> str:
    """Join the caption and hashtags into the text Instagram will display."""
    dialogue = state.get("dialogue")
    if not dialogue:
        return ""
    tags = " ".join(dialogue.hashtags)
    return f"{dialogue.caption}\n\n{tags}".strip()


def publish_node(settings: Settings):
    """Build the publish node.

    Returns ``post_id`` on success, or records the failure in ``error`` so a
    failed publish ends the run cleanly instead of crashing it.
    """

    async def run(state: PostState) -> dict[str, Any]:
        media = state.get("media")
        if not media or not (media.urls or media.local_files):
            return {"error": "no media to publish"}

        caption = build_caption(state)
        local_path = Path(media.local_files[0]) if media.local_files else None
        try:
            post_id = await publish_reel(
                settings,
                caption,
                video_url=media.urls[0] if media.urls else None,
                local_path=local_path,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the CLI, not swallowed
            logger.error("Publish failed: %s", exc)
            return {"error": str(exc)}

        return {"post_id": post_id}

    return run
