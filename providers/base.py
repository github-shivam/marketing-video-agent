"""Provider protocols for media generation.

Keeping generation behind a protocol is deliberate: three video models were
retired within one week of each other in September 2026 (OpenAI's Sora 2,
Amazon's Nova Reel and Nova Canvas). Whatever we wire in today should be
replaceable without touching the graph.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from video_plan import VideoPlan


@runtime_checkable
class VideoProvider(Protocol):
    """Generates a single vertical video file for a reel."""

    name: str

    async def generate(
        self,
        prompts: list[str],
        out_dir: Path,
        plan: VideoPlan | None = None,
        references: list[Path] | None = None,
    ) -> Path:
        """Generate one video and return the path to the local file.

        Args:
            prompts: One prompt per segment. The first is the base clip; each
                subsequent prompt extends the same continuous shot.
            out_dir: Directory to write the finished file into.
            plan: The generations to make, which fix the length. Defaults to
                the default length.
            references: Photos of a person to feature. A provider that cannot use
                them ignores them.

        Returns:
            Path to the generated MP4. It is as long as the plan *generates*;
            the caller trims it to the length the plan keeps.
        """
        ...


@runtime_checkable
class ImageProvider(Protocol):
    """Generates a single still. Unused in phase 1; kept for carousels."""

    name: str

    async def generate(self, prompt: str, out_dir: Path) -> Path:
        """Generate one image and return the path to the local file."""
        ...
