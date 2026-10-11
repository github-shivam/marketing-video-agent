"""Understand what a person means when they reply in a chat.

The approval gates were built for ``y`` / ``n``. In WhatsApp people write
"looks great, go for it", "make the hook punchier", "haan post kar do" or
"hmm, maybe later". This node reads the reply *in the context of the question
that was just asked* and returns one of a small set of intents, which code then
maps onto the graph's existing approve / reject decisions.

Two of those decisions cost something that cannot be undone: approving the
prompts spends Veo money, and approving the video posts to Instagram. So the
model is told to treat anything hedged as ``unclear`` and ask again, and code
rejects any intent that is not valid for the question that was asked.
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field

from config import Settings

from .common import structured

logger = logging.getLogger(__name__)

Stage = Literal["script", "video", "idle"]
Intent = Literal["approve", "change", "decline", "request", "chat", "unclear"]

#: Which intents make sense for each question. Anything else is treated as
#: ``unclear``, so a confused model can never post a video nobody asked to post.
VALID_INTENTS: dict[str, frozenset[str]] = {
    "script": frozenset({"approve", "change", "decline", "unclear"}),
    "video": frozenset({"approve", "decline", "unclear"}),
    "idle": frozenset({"request", "chat"}),
}


class Reply(BaseModel):
    """What the user's message means."""

    intent: Intent = Field(
        description="What the user wants, from the allowed list for this question"
    )
    feedback: str = Field(
        default="",
        description=(
            "Only for 'change': what they want different, in their own words, "
            "as an instruction to the screenwriter. Empty otherwise."
        ),
    )
    message: str = Field(
        default="",
        description=(
            "Only for 'unclear' and 'chat': one or two short, friendly sentences "
            "to send back. Empty otherwise."
        ),
    )
    video_seconds: int | None = Field(
        default=None,
        description=(
            "Only for 'request' and 'change': how long the WHOLE video should be, "
            "in seconds, if the person said so. Null when they did not."
        ),
    )


CONVERSE_SYSTEM = """You read one WhatsApp message from a person and work out what
they mean. You are the ear of a bot that makes short marketing videos. The
person may write in English, Hindi, Hinglish or another language, casually, with
typos, slang or emoji. Understand meaning, not keywords.

The bot has just asked the person a question. Interpret the message as the
answer to THAT question. The question is given below.

{question}

Choose exactly one intent:
{intents}

Rules:
- Money and publishing cannot be undone. Choose "approve" ONLY when the person
  clearly agrees to go ahead NOW. Hedging ("maybe", "I guess", "not sure",
  "after I check", "later", "let me think") is "unclear", not "approve".
- If a message both agrees and asks for a change ("yes but make it shorter"),
  it is "change". The change must happen before anything proceeds.
- A question, a complaint or anything off-topic is "unclear" (or "chat" when
  idle). Answer it briefly and kindly in "message", and remind them what you
  are waiting for.
- "change": copy what they want different into "feedback", phrased as an
  instruction ("Make the opening line punchier and keep it under 10 words").
  Keep every specific they gave. Do not invent changes they did not ask for.
- "message" is plain text for a chat: short, warm, no markdown headings, no
  lists of options longer than three.
- "video_seconds": set it only when the person states how long the WHOLE video
  should be, converting to seconds ("half a minute" is 30, "1 minute" is 60,
  "20 sec ka video" is 20). A length for one PART of the video ("a hook in the
  first 3 seconds", "keep the intro under 5 seconds", "each shot 2 seconds") is
  not the length of the video: leave it null. Never guess a length they did not
  state, and leave it null when they did not mention one."""

_SCRIPT_QUESTION = """THE QUESTION JUST ASKED: the bot showed the person the exact
prompts for a video (title: {title!r}, hook: {hook!r}) and asked whether to
generate it. Generating costs real money, so only a clear yes counts."""

_SCRIPT_INTENTS = """- approve : go ahead and generate the video as shown
- change  : they want something different; the prompts should be rewritten
- decline : they want to stop completely and do not want a rewrite (cancel,
            never mind, drop it)
- unclear : you cannot tell, or they only said "no" with no reason (a bare "no"
            could be "cancel" or "change it", so ask which)"""

_VIDEO_QUESTION = """THE QUESTION JUST ASKED: the bot sent the finished video and
asked whether to post it to Instagram. Posting is public and cannot be undone,
so only a clear yes counts. Caption that would be posted: {caption!r}"""

_VIDEO_INTENTS = """- approve : post it to Instagram now
- decline : do not post it (no, skip, not now, keep it, I'll do it myself)
- unclear : you cannot tell, they want a change to the video itself (tell them
            they can send a new request to make another version), or they
            hedged"""

_IDLE_QUESTION = """THERE IS NO OPEN QUESTION. The person is starting a
conversation. The bot makes a short vertical video from a topic or a link."""

_IDLE_INTENTS = """- request : they are giving a topic, idea or link to make a video about
- chat    : a greeting, a thank-you, a question about what the bot does, or
            anything that is not a topic. Reply in "message": say you turn a
            topic or link into a video, and ask what they would like to make."""


def _prompt_for(stage: str, context: dict[str, str]) -> str:
    if stage == "script":
        question, intents = (
            _SCRIPT_QUESTION.format(
                title=context.get("title", ""), hook=context.get("hook", "")
            ),
            _SCRIPT_INTENTS,
        )
    elif stage == "video":
        question, intents = (
            _VIDEO_QUESTION.format(caption=context.get("caption", "")),
            _VIDEO_INTENTS,
        )
    else:
        question, intents = _IDLE_QUESTION, _IDLE_INTENTS
    return CONVERSE_SYSTEM.format(question=question, intents=intents)


async def interpret(
    settings: Settings, stage: str, text: str, context: dict[str, str] | None = None
) -> Reply:
    """Work out what ``text`` means as an answer at ``stage``.

    Args:
        settings: Model configuration.
        stage: ``script`` (approve the prompts), ``video`` (approve posting) or
            ``idle`` (no open question).
        text: The user's message, verbatim.
        context: Details of the open question, for the model to resolve "it".

    Returns:
        The interpretation. An intent that is not valid for ``stage`` becomes
        ``unclear`` (or ``chat`` when idle), and a model failure does too, so
        the person is simply asked again rather than guessed at.
    """
    fallback = "chat" if stage == "idle" else "unclear"
    try:
        reply = await structured(
            settings,
            "converse",
            Reply,
            _prompt_for(stage, context or {}),
            f"THE PERSON'S MESSAGE\n<message>\n{text}\n</message>",
            temperature=0.0,
        )
    except Exception as exc:  # noqa: BLE001 - never guess at a decision on a failure
        logger.warning("Could not interpret a chat reply (%s) - asking again", exc)
        return Reply(intent=fallback)

    if reply.intent not in VALID_INTENTS.get(stage, frozenset()):
        logger.warning("Intent %r is not valid at stage %r - asking again", reply.intent, stage)
        return Reply(intent=fallback, message=reply.message)
    if reply.intent == "change" and not reply.feedback.strip():
        # A change with nothing to change gives the writer nothing to act on.
        return Reply(intent="unclear", message=reply.message)
    return reply
