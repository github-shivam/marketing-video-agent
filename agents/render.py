"""Turn a model-agnostic :class:`~state.Screenplay` into video-model prompts.

Prompt syntax is applied here, in code, rather than asked of the LLM. The
screenwriter supplies content (what the camera sees, what is said, what is
heard); this module supplies the exact format each model documents:

* **Veo 3.1** (Google): ``[00:00-00:02]`` timestamped shots, dialogue as
  ``X says, "..."``, and ``SFX:`` / ``Ambient noise:`` labels. One prompt per
  generation call, because the 8s base clip and each 7s extension are
  separate calls.
* **Seedance 2.5** (ByteDance): ``Shot 1 [0-3s]:`` labels, ``<sound effect>``,
  ``{dialogue}`` and ``(music)`` brackets, and constraints (no subtitles, no
  watermark) at the end. One prompt for the whole clip.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Iterable

from config import Settings
from state import Brief, Screenplay, Shot

logger = logging.getLogger(__name__)

#: Veo rejects prompts naming a real person: "Sorry, we can't create videos
#: with real people's names or likenesses." The writer is told not to use
#: names, but a model instruction is advice, not a guarantee — and the failure
#: costs a whole run — so names are also scrubbed deterministically here.
NAME_REPLACEMENT = "the speaker"

#: Veo caps a prompt at 1,024 tokens. Words are a safe proxy.
VEO_PROMPT_WORD_WARNING = 700

#: Fields that describe what is SEEN. Spoken dialogue is never scrubbed: a
#: person saying their own name is normal, and replacing it would produce
#: nonsense like "Hi, the speaker here."
_VISUAL_FIELDS = ("camera", "subject", "action", "setting", "speaker")


def scrub_personal_names(prompt: str, names: Iterable[str]) -> str:
    """Replace real personal names in a prompt with a neutral descriptor.

    Args:
        prompt: Text to clean.
        names: Names to remove — typically the full name plus each word of it,
            since models often use just a first name.

    Returns:
        The text with every name occurrence replaced.
    """
    cleaned = prompt
    # Longest first, so "Jane Doe" is replaced before "Jane".
    for name in sorted({n.strip() for n in names if n.strip()}, key=len, reverse=True):
        cleaned = re.sub(
            rf"\b{re.escape(name)}\b", NAME_REPLACEMENT, cleaned, flags=re.IGNORECASE
        )
    return cleaned


def candidate_names(brief: Any) -> list[str]:
    """Return the personal names to scrub from visual descriptions.

    Each full name is expanded to its individual words, because models often
    use only a first name. Very short fragments are skipped so that a name
    like "Al" cannot mangle ordinary words.
    """
    names: list[str] = []
    for full in getattr(brief, "person_names", None) or []:
        full = full.strip()
        if not full:
            continue
        names.append(full)
        names.extend(part for part in full.split() if len(part) > 2)
    return names


def _sentence(text: str, *, capitalise: bool = True) -> str:
    """Return ``text`` stripped, capitalised and ending in terminal punctuation.

    ``capitalise`` is off for text that follows a label such as ``SFX:``,
    where the original casing reads better.
    """
    text = text.strip()
    if not text:
        return ""
    if capitalise:
        text = text[0].upper() + text[1:]
    return text if text[-1] in ".!?\"'" else f"{text}."


def _capitalise(text: str) -> str:
    """Upper-case the first letter only, leaving the rest as written."""
    return text[:1].upper() + text[1:]


def _subject_action(shot: Shot) -> str:
    """Join subject and action into one sentence.

    The subject is meant to be a noun phrase, but models sometimes write a
    full sentence into it, which would leave a stray full stop mid-sentence
    ("sits beside her. slaps the book"). Trailing punctuation is stripped so
    the join always reads as one clause.
    """
    subject = shot.subject.strip().rstrip(".;:,!? ")
    return _sentence(f"{subject} {shot.action.strip()}")


def _scrub(shot: Shot, names: list[str]) -> Shot:
    """Return a copy of ``shot`` with names removed from its visual fields."""
    if not names:
        return shot
    return shot.model_copy(
        update={f: scrub_personal_names(getattr(shot, f), names) for f in _VISUAL_FIELDS}
    )


def _language_clause(settings: Settings) -> str:
    """Name the spoken language when it is not English, as both models advise."""
    language = settings.script_language.strip()
    if language.lower() in {"english", "en", ""}:
        return ""
    return f" in {language}"


def _timecode(seconds: int) -> str:
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


# ── Veo 3.1 ───────────────────────────────────────────────────────────────


def _veo_shot(shot: Shot, number: int, settings: Settings) -> str:
    head = (
        f"[{_timecode(shot.start)}-{_timecode(shot.end)}]"
        if settings.veo_timestamps
        else f"Shot {number}:"
    )
    parts = [
        head,
        _sentence(shot.camera),
        _subject_action(shot),
        _sentence(shot.setting),
    ]
    if shot.dialogue.strip():
        speaker = _capitalise(shot.speaker.strip() or "The speaker")
        delivery = f" {shot.delivery.strip()}" if shot.delivery.strip() else ""
        parts.append(
            f'{speaker} says{delivery}{_language_clause(settings)}, '
            f'"{shot.dialogue.strip()}"'
        )
    if shot.sfx.strip():
        parts.append(f"SFX: {_sentence(shot.sfx, capitalise=False)}")
    if shot.ambient.strip():
        parts.append(f"Ambient noise: {_sentence(shot.ambient, capitalise=False)}")
    if shot.music.strip():
        parts.append(f"Music: {_sentence(shot.music, capitalise=False)}")
    return " ".join(p for p in parts if p)


def render_veo(
    screenplay: Screenplay, settings: Settings, names: list[str]
) -> list[str]:
    """Render one Veo prompt per generation call (base clip, then extensions)."""
    prompts: list[str] = []
    for segment in screenplay.segments:
        lines: list[str] = []
        if segment.index > 0:
            lines.append(
                "Continue the same continuous shot: same location, people, "
                "light and style, with no cut to a new scene."
            )
        lines.append(_sentence(f"Overall look: {scrub_personal_names(screenplay.style, names)}"))
        lines.append(_sentence(scrub_personal_names(screenplay.continuity, names)))
        for number, shot in enumerate(segment.shots, 1):
            lines.append(_veo_shot(_scrub(shot, names), number, settings))
        prompt = "\n\n".join(line for line in lines if line)

        words = len(prompt.split())
        if words > VEO_PROMPT_WORD_WARNING:
            logger.warning(
                "segment %d prompt is %d words; Veo's limit is 1,024 tokens",
                segment.index, words,
            )
        prompts.append(prompt)
    return prompts


# ── Seedance 2.5 ──────────────────────────────────────────────────────────


def _seedance_shot(shot: Shot, number: int, offset: int, settings: Settings) -> str:
    parts = [
        f"Shot {number} [{offset + shot.start}-{offset + shot.end}s]:",
        _sentence(shot.camera),
        _subject_action(shot),
        _sentence(shot.setting),
    ]
    if shot.sfx.strip():
        parts.append(f"<{shot.sfx.strip().rstrip('.')}>")
    if shot.ambient.strip():
        parts.append(f"<{shot.ambient.strip().rstrip('.')}>")
    if shot.music.strip():
        parts.append(f"({shot.music.strip().rstrip('.')})")
    if shot.dialogue.strip():
        speaker = _capitalise(shot.speaker.strip() or "The speaker")
        delivery = f" {shot.delivery.strip()}" if shot.delivery.strip() else ""
        parts.append(
            f"{speaker} says{delivery}{_language_clause(settings)}: "
            f"{{{shot.dialogue.strip()}}}"
        )
    return " ".join(p for p in parts if p)


def render_seedance(
    screenplay: Screenplay, settings: Settings, names: list[str]
) -> list[str]:
    """Render a single Seedance prompt covering the whole clip."""
    lines = [
        _sentence(scrub_personal_names(screenplay.style, names)),
        _sentence(scrub_personal_names(screenplay.continuity, names)),
    ]
    number = 0
    offset = 0
    has_music = False
    for segment in screenplay.segments:
        for shot in segment.shots:
            number += 1
            has_music = has_music or bool(shot.music.strip())
            lines.append(_seedance_shot(_scrub(shot, names), number, offset, settings))
        offset += max(s.end for s in segment.shots)

    # Seedance adds stray subtitles, watermarks and music unless told not to.
    constraints = ["No subtitles.", "No watermarks or logos.", "No duplicated people."]
    if not has_music:
        constraints.append("No BGM, no music, no score.")
    lines.append(" ".join(constraints))
    return ["\n\n".join(line for line in lines if line)]


# ── Entry point ───────────────────────────────────────────────────────────


def render_prompts(
    screenplay: Screenplay, settings: Settings, brief: Brief | None = None
) -> list[str]:
    """Render the screenplay into the prompts for ``settings.video_model``.

    Args:
        screenplay: The finished screenplay.
        settings: Selects the target model and carries language and format flags.
        brief: When given, real people it names are scrubbed from the visuals.

    Returns:
        One prompt per generation call: one per segment for Veo, a single
        prompt for Seedance.
    """
    names = candidate_names(brief)
    if settings.video_model == "seedance":
        return render_seedance(screenplay, settings, names)
    return render_veo(screenplay, settings, names)
