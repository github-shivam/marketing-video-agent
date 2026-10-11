"""How long the video is, and how that length is built from Veo generations.

A person asks for a length ("make it 20 seconds"). Veo cannot make an arbitrary
length: it makes a base clip of 4, 6 or 8 seconds and extends it in steps of
exactly 7. So a requested length becomes a *plan*, a list of generations, each
with the seconds Veo is asked to make and the seconds of it that are kept. The
screenplay is written to the kept seconds, so the story finishes inside the
requested time, and the finished clip is trimmed to it.

This module is pure: no model, no network, no settings. The same functions
parse a chat message, clamp the answer and build the plan.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MIN_SECONDS = 4
MAX_SECONDS = 30

#: Veo's base clip lengths. Anything longer than the largest is built by
#: extending an 8-second base.
VEO_BASE_CHOICES = (4, 6, 8)
VEO_EXTENSION_SECONDS = 7

#: When the last extension would keep only this many seconds or fewer, it is
#: left out and the video ends that much short. Paying for seven seconds of
#: generation to keep one is not worth it, and a video that finishes a second
#: early still finishes inside the requested time.
MAX_SHORTFALL = 1


@dataclass(frozen=True)
class SegmentPlan:
    """One Veo generation."""

    generate: int  #: seconds Veo is asked to make (billed)
    use: int  #: seconds of it the screenplay covers and the video keeps


@dataclass(frozen=True)
class VideoPlan:
    """The generations that make up one video."""

    segments: tuple[SegmentPlan, ...]

    @property
    def seconds(self) -> int:
        """Length of the finished video."""
        return sum(s.use for s in self.segments)

    @property
    def generated_seconds(self) -> int:
        """Seconds of video Veo is asked to make, which is what is billed."""
        return sum(s.generate for s in self.segments)

    @property
    def extensions(self) -> int:
        """How many 7-second extensions follow the base clip."""
        return len(self.segments) - 1

    @property
    def needs_trim(self) -> bool:
        """True when more is generated than is kept."""
        return self.seconds < self.generated_seconds

    @property
    def use_seconds(self) -> list[int]:
        """Kept seconds per segment: the windows the screenplay must tile."""
        return [s.use for s in self.segments]


def clamp_seconds(seconds: int) -> int:
    """Bring a length into the range that can be made."""
    return min(MAX_SECONDS, max(MIN_SECONDS, int(seconds)))


def plan_for(
    seconds: int, video_model: str = "veo", *, fixed_base: bool = False
) -> VideoPlan:
    """Work out the generations for a video of about ``seconds``.

    Args:
        seconds: The requested length. Clamped to 4-30.
        video_model: ``veo`` builds it from a base clip and extensions.
            ``seedance`` renders the whole length in one pass.
        fixed_base: True when the base clip must be 8 seconds whatever the
            length asked for. Veo requires that when it is given reference
            images of a person, so a short video is made as 8s and trimmed.

    Returns:
        The plan. Its ``seconds`` is the requested length, or one second less
        when a final extension would have kept only that one second.
    """
    seconds = clamp_seconds(seconds)

    if video_model == "seedance":
        return VideoPlan((SegmentPlan(seconds, seconds),))

    if seconds <= max(VEO_BASE_CHOICES):
        base = (
            max(VEO_BASE_CHOICES)
            if fixed_base
            else next(choice for choice in VEO_BASE_CHOICES if choice >= seconds)
        )
        return VideoPlan((SegmentPlan(base, seconds),))

    base = max(VEO_BASE_CHOICES)
    full, rest = divmod(seconds - base, VEO_EXTENSION_SECONDS)
    segments = [SegmentPlan(base, base)]
    segments += [SegmentPlan(VEO_EXTENSION_SECONDS, VEO_EXTENSION_SECONDS)] * full
    if rest > MAX_SHORTFALL:
        segments.append(SegmentPlan(VEO_EXTENSION_SECONDS, rest))
    return VideoPlan(tuple(segments))


# ── Reading a length out of a message ─────────────────────────────────────

#: The lookbehind keeps a number that is part of a word or a longer number
#: ("v3s", "1.5.2s") from being read as a length.
_NUMBER = r"(?<![\w.])(\d+(?:\.\d+)?)"

#: "20 seconds", "20 sec", "20-second", "20s", "20 सेकंड". A bare "s" counts
#: only when it is glued to the number, so "5 stages" is not five seconds.
_SECONDS = re.compile(
    rf"{_NUMBER}\s*-?\s*(?:seconds?\b|secs?\b|sec\.|सेकंड|सेकेंड|सेकण्ड)"
    rf"|{_NUMBER}s\b",
    re.IGNORECASE,
)
_MINUTES = re.compile(rf"{_NUMBER}\s*-?\s*(?:minutes?\b|mins?\b)", re.IGNORECASE)
_HALF_MINUTE = re.compile(r"\bhalf\s+(?:a\s+)?minute\b", re.IGNORECASE)
_A_MINUTE = re.compile(r"\b(?:a|one)\s+(?:whole\s+)?minute\b", re.IGNORECASE)


#: A number after one of these is the length of a part of the video, not of the
#: whole thing: "the first 3 seconds", "every 2 seconds".
_PART_OF_THE_VIDEO = re.compile(r"\b(?:first|last|next|every|each)\s+$", re.IGNORECASE)


def _first_length(pattern: re.Pattern[str], text: str, scale: float) -> int | None:
    """The first match of ``pattern`` that is a length for the whole video."""
    for found in pattern.finditer(text):
        if _PART_OF_THE_VIDEO.search(text[max(0, found.start() - 12) : found.start()]):
            continue
        value = float(found.group(1) or found.group(2) or 0) * scale
        if value >= 1:
            return round(value)
    return None


def parse_seconds(text: str) -> int | None:
    """Read a requested length from a message, with no model involved.

    The chat bot uses this for messages that carry a link, which never reach the
    reply interpreter. Everything else is read by the interpreter, which can tell
    "20 seconds long" from "a hook in the first 3 seconds" by understanding the
    sentence; this only skips the obvious case of a number that follows "first",
    "last", "every" or "each".

    Args:
        text: The message as typed.

    Returns:
        Whole seconds, or None when the message states no length. The value is
        not clamped: "2 minutes" is 120, and the caller decides what to do.
    """
    seconds = _first_length(_SECONDS, text, 1)
    if seconds is not None:
        return seconds

    minutes = _first_length(_MINUTES, text, 60)
    if minutes is not None:
        return minutes

    if _HALF_MINUTE.search(text):
        return 30
    if _A_MINUTE.search(text):
        return 60
    return None


def resolve_seconds(requested: int | None, default: int) -> tuple[int, str]:
    """Settle on a length, and say so when the request had to change.

    Args:
        requested: What the person asked for, or None.
        default: The length to use when they did not say.

    Returns:
        ``(seconds, note)``. ``note`` is a sentence for the person when their
        request was outside the range and was clamped, otherwise empty.
    """
    if requested is None or requested <= 0:
        return clamp_seconds(default), ""
    if requested > MAX_SECONDS:
        return MAX_SECONDS, (
            f"The longest video I can make is {MAX_SECONDS} seconds, "
            f"so I'll make it {MAX_SECONDS}."
        )
    if requested < MIN_SECONDS:
        return MIN_SECONDS, (
            f"The shortest video I can make is {MIN_SECONDS} seconds, "
            f"so I'll make it {MIN_SECONDS}."
        )
    return requested, ""
