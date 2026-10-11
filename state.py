"""Shared graph state and the structured outputs each agent produces.

The graph passes a single ``PostState`` dict between nodes. Every agent
returns a Pydantic model so that malformed LLM output fails loudly at the
node boundary rather than silently downstream.

The central artefact is the :class:`Screenplay`: a model-agnostic description
of the video (shots, camera, action, dialogue, sound). It is rendered into the
exact prompt syntax each video model wants by :mod:`agents.render`, so format
rules live in code instead of depending on the LLM remembering them.
"""

from __future__ import annotations

import re
from typing import Literal, TypedDict

from pydantic import BaseModel, Field, field_validator

MediaType = Literal["REELS", "IMAGE", "CAROUSEL"]
InputKind = Literal["topic", "url"]

#: What the user is asking for. Each intent gets its own structural guidance
#: in :mod:`agents.prompt_guides`.
VideoIntent = Literal[
    "product_promo",       # sell a product or service
    "explainer",           # make one idea click
    "educational_howto",   # teach steps
    "brand_story",         # who a company or person is and why it matters
    "announcement",        # news, launch, event
    "entertainment",       # a story, joke or mood piece
]


class Brief(BaseModel):
    """What the user asked for, classified. Drives the whole run."""

    request: str = Field(default="", description="Set by code. Leave blank.")
    input_kind: InputKind = Field(
        default="topic", description="Set by code. Leave as 'topic'."
    )
    url: str | None = Field(default=None, description="Set by code. Leave null.")

    intent: VideoIntent = Field(description="The kind of video being asked for")
    subject: str = Field(description="What the video is about, in one sentence")
    audience: str = Field(description="Who it is for")
    tone: str = Field(description="Brand or creator voice, e.g. 'warm, plain-spoken'")
    key_message: str = Field(description="The single thing a viewer should remember")
    cta: str = Field(
        default="", description="The one action to take. Empty if there is none."
    )
    visual_style: str = Field(
        description="Look and feel that fits the subject, e.g. 'warm documentary'"
    )
    dialogue_led: bool = Field(
        default=False,
        description=(
            "True when a person speaking to camera should carry the video; "
            "false when visuals, sound effects and music carry it."
        ),
    )
    person_names: list[str] = Field(
        default_factory=list,
        description=(
            "Real individuals named in the request. Veo rejects prompts naming "
            "real people, so these are scrubbed from the visual description."
        ),
    )


class Shot(BaseModel):
    """One camera setup inside a segment, with its own timing and sound."""

    # Deliberately unconstrained: gaps, overlaps and zero-length shots are
    # repaired by ``retime`` rather than costing the whole draft.
    start: int = Field(description="Second the shot starts, within its segment")
    end: int = Field(description="Second the shot ends, within its segment")
    camera: str = Field(
        description="Shot size and ONE camera move, e.g. 'Medium close-up, slow dolly in'"
    )
    subject: str = Field(
        description=(
            "ONE short noun phrase naming who or what is on screen, e.g. 'the "
            "woman in the blue linen shirt'. No verb and no full stop: full "
            "descriptions belong in the screenplay's continuity field."
        )
    )
    action: str = Field(
        description=(
            "A verb phrase that continues the subject, e.g. "
            "'pours steamed milk in a slow spiral'"
        )
    )
    setting: str = Field(default="", description="Place, time of day and light")
    speaker: str = Field(
        default="",
        description="Who speaks, as the same short noun phrase used for the subject",
    )
    delivery: str = Field(default="", description="How they speak, e.g. 'warmly and briskly'")
    dialogue: str = Field(default="", description="The exact words spoken, without quotes")
    sfx: str = Field(default="", description="Specific sound effects")
    ambient: str = Field(default="", description="The ambient soundscape")
    music: str = Field(default="", description="Music, if any")

    @field_validator("start", "end", mode="before")
    @classmethod
    def _whole_seconds(cls, v):
        """Models often emit 2.0 or 2.5. Veo and Seedance both want whole seconds."""
        try:
            return round(float(v))
        except (TypeError, ValueError):
            return v

    def spoken_words(self) -> int:
        """Count the words that will be spoken in this shot."""
        return len(self.dialogue.split())


class Segment(BaseModel):
    """One generation call: the 8s base clip, a 7s extension, or a whole Seedance clip."""

    index: int = Field(ge=0, description="0 = base clip, 1+ = extensions")
    shots: list[Shot] = Field(min_length=1, max_length=15)

    def spoken_words(self) -> int:
        return sum(s.spoken_words() for s in self.shots)


class Screenplay(BaseModel):
    """The complete, model-agnostic description of the video."""

    title: str
    logline: str = Field(description="One sentence: what happens and why it lands")
    hook: str = Field(description="What stops the scroll in the first second")
    style: str = Field(
        description="Global look: film stock, colour grade, lighting, mood. Applies to every shot."
    )
    continuity: str = Field(
        description=(
            "Fixed descriptions of every recurring person, product and location, "
            "written once so they can be repeated identically in each segment."
        )
    )
    segments: list[Segment] = Field(min_length=1, max_length=21)
    cta: str = Field(default="", description="The closing call to action")


class Critique(BaseModel):
    """The critic's structured verdict. Scored strictly."""

    score: float = Field(ge=0, le=10)
    hook: str = Field(description="Critique of the hook specifically")
    intent_fit: str = Field(description="Does it deliver what the user asked for?")
    prompt_craft: str = Field(
        description="Are the shots concrete, filmable and within the model's rules?"
    )
    feedback: str = Field(description="Concrete, actionable rewrite guidance")


class Dialogue(BaseModel):
    """Caption and hashtags for the post itself."""

    caption: str
    hashtags: list[str] = Field(min_length=5, max_length=12)

    @field_validator("caption")
    @classmethod
    def _caption_under_150_words(cls, v: str) -> str:
        """Instagram truncates long captions; keep them scannable."""
        if len(v.split()) > 150:
            raise ValueError("caption must be under 150 words")
        return v

    @field_validator("hashtags")
    @classmethod
    def _normalise_hashtags(cls, v: list[str]) -> list[str]:
        """Ensure every hashtag carries exactly one leading '#'."""
        return [f"#{tag.lstrip('#')}" for tag in v]


AssetKind = Literal["image", "video", "audio"]


class Asset(BaseModel):
    """A file a person uploaded to steer the video: a photo, a clip or a sound."""

    path: str
    kind: AssetKind
    mime: str = ""
    caption: str = ""


# The next three are what the model that reads the uploads returns. Every field
# is required, with no defaults: the schema is sent to Gemini, which rejects
# some default values, and an empty string or list says "nothing" just as well.


class PersonLook(BaseModel):
    """A person seen in the uploads, described so they can be drawn again."""

    label: str = Field(
        description=(
            "A short noun phrase naming them, e.g. 'the woman in the green jacket'. "
            "Never a real name."
        )
    )
    description: str = Field(
        description=(
            "Age range, build, hair, skin tone, clothing, glasses, anything "
            "distinctive. Only what is visible. Never a name, and never a guess "
            "at who they are."
        )
    )
    item_indexes: list[int] = Field(
        description="Which numbered images or frames show them"
    )
    is_main: bool = Field(description="True for the person the material is about")


class AudioLook(BaseModel):
    """What an uploaded sound is like."""

    kind: Literal["music", "speech", "ambient", "mixed", "none"] = Field(
        description="'none' when no audio was uploaded"
    )
    mood: str = Field(description="e.g. 'warm, nostalgic'. Empty if no audio")
    genre: str = Field(description="Musical genre or style. Empty if not music")
    tempo: str = Field(description="e.g. 'slow, about 70 bpm'. Empty if not music")
    instruments: str = Field(description="What is playing. Empty if not music")
    language: str = Field(description="Language of any words. Empty if none")
    transcript: str = Field(
        description="Words spoken or sung, as heard, at most 400 characters"
    )
    summary: str = Field(description="One sentence on what the audio is and feels like")


class UploadAnalysis(BaseModel):
    """What the uploaded material is like. Returned by the model that reads it."""

    theme: str = Field(description="What the material is about, in one sentence")
    mood: str = Field(description="The feeling it gives, e.g. 'calm, sunlit'")
    setting: str = Field(description="Where and when it is set")
    visual_style: str = Field(
        description="The look: lighting, film feel, composition, e.g. 'soft natural light'"
    )
    color_words: str = Field(
        description="The colours in plain words, e.g. 'warm amber, cream and deep teal'"
    )
    motion: str = Field(
        description="Pacing and camera movement if there was video, else empty"
    )
    people: list[PersonLook] = Field(description="Every distinct person visible")
    audio: AudioLook
    reference_items: list[int] = Field(
        description=(
            "Up to 3 numbered images or frames that best show the main person's "
            "face clearly and on their own. Empty if no person is clearly visible."
        )
    )


class AssetBrief(BaseModel):
    """What was learned from the uploads, plus the parts that code works out."""

    analysis: UploadAnalysis
    palette: list[str] = Field(
        default_factory=list,
        description="Dominant colours as #RRGGBB, measured from the pixels",
    )
    reference_files: list[str] = Field(
        default_factory=list,
        description="Photos of the main person to send to Veo as references",
    )
    n_images: int = 0
    n_videos: int = 0
    n_audio: int = 0


class Media(BaseModel):
    """The rendered asset and where it lives."""

    media_type: MediaType = "REELS"
    urls: list[str] = Field(default_factory=list, description="Public/presigned URLs")
    local_files: list[str] = Field(default_factory=list)
    prompt_file: str = Field(default="", description="The prompts sent to the model")
    duration_seconds: int = 0
    notes: list[str] = Field(
        default_factory=list,
        description="Things the person should be told about how the video was made",
    )


class PostState(TypedDict, total=False):
    """State threaded through every node of the graph."""

    request: str
    #: Length of the finished video, in seconds. Per run, so each chat can ask
    #: for its own. Absent means ``VIDEO_SECONDS_DEFAULT``.
    duration_seconds: int
    #: Files the person uploaded to steer the video. Absent for most runs.
    assets: list[Asset]
    asset_brief: AssetBrief | None
    #: Why the uploads could not be read, so the person can be told.
    asset_error: str | None
    brief: Brief
    source_text: str
    research_notes: str
    screenplay: Screenplay
    prompts: list[str]
    critique: Critique
    score: float
    iterations: int
    human_feedback: str | None
    dialogue: Dialogue
    media: Media
    approved: bool
    post_id: str | None
    error: str | None


# ── Input parsing ─────────────────────────────────────────────────────────

_URL = re.compile(r"(?:https?://|www\.)[^\s<>\"']+", re.IGNORECASE)
_BARE_DOMAIN = re.compile(
    r"^(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,}(?:/\S*)?$", re.IGNORECASE
)


def extract_url(request: str) -> str | None:
    """Return the URL in a request, or None if it is a plain topic.

    Done in code rather than by the classifier: whether something is a link is
    not a judgement call, and a wrong guess would skip fetching the page.

    Args:
        request: The raw text the user typed.

    Returns:
        An ``http(s)`` URL, or None.
    """
    text = request.strip()
    found = _URL.search(text)
    candidate = found.group(0).rstrip(".,;:!?)") if found else None
    if candidate is None and _BARE_DOMAIN.match(text):
        candidate = text
    if candidate is None:
        return None
    if not candidate.lower().startswith(("http://", "https://")):
        candidate = f"https://{candidate}"
    return candidate
