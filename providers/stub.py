"""A no-cost stand-in for the video provider.

Used automatically when Veo is not configured (no ``GOOGLE_CLOUD_PROJECT`` and
``VEO_GCS_BUCKET``) so that the whole graph
can be exercised end to end without spending money. It produces a real,
playable MP4 via ffmpeg when that binary is available, and falls back to an
empty placeholder file when it is not.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
import uuid
from pathlib import Path

from config import Settings
from video_plan import VideoPlan

logger = logging.getLogger(__name__)


class StubVideoProvider:
    """Writes a placeholder clip instead of calling a paid API."""

    name = "stub"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def generate(
        self,
        prompts: list[str],
        out_dir: Path,
        plan: VideoPlan | None = None,
        references: list[Path] | None = None,
    ) -> Path:
        """Produce a placeholder video and return its path.

        Args:
            prompts: Ignored beyond being logged, so a run shows what *would*
                have been sent to Veo.
            out_dir: Directory to write the file into.
            plan: Sets the placeholder's length, as a real provider's would. The
                placeholder is as long as the plan generates, so trimming can be
                exercised offline too.

        Returns:
            Path to the placeholder MP4.
        """
        plan = plan or self.settings.plan()
        if references:
            logger.info("[stub] would send %d reference photo(s) to Veo", len(references))
        out_dir.mkdir(parents=True, exist_ok=True)
        destination = out_dir / f"reel-stub-{int(time.time())}-{uuid.uuid4().hex[:8]}.mp4"

        for i, prompt in enumerate(prompts):
            logger.info("[stub] beat %d prompt: %.160s", i, prompt)

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            await self._render_placeholder(ffmpeg, destination, plan.generated_seconds)
        else:
            destination.touch()
            logger.warning("ffmpeg not found - wrote an empty placeholder")

        logger.info("[stub] wrote %s (no API call, no cost)", destination)
        return destination

    async def _render_placeholder(
        self, ffmpeg: str, destination: Path, seconds: int
    ) -> None:
        """Render a silent vertical colour bar clip matching the real output."""
        process = await asyncio.create_subprocess_exec(
            ffmpeg,
            "-y",
            "-f", "lavfi",
            "-i", f"testsrc=size=720x1280:rate=30:duration={seconds}",
            "-f", "lavfi",
            "-i", f"anullsrc=channel_layout=stereo:sample_rate=44100:d={seconds}",
            "-c:v", "libx264",
            "-preset", "ultrafast",  # a placeholder: speed matters, quality does not
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-shortest",
            str(destination),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        if process.returncode != 0:
            logger.warning(
                "ffmpeg placeholder failed (%s) - writing an empty file",
                stderr.decode(errors="replace")[-300:],
            )
            destination.touch()
