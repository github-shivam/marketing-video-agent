"""Graph wiring: nodes, conditional edges and the two human-approval gates.

The node implementations are injected via :class:`Nodes` so that the wiring
can be tested with fakes and without a model, a network or an API key.

Flow::

    START -> [analyze ->] classify -> research -> writer <-> critic -> dialogue
             -> approve_script -> video -> confirm_video -> publish -> END

``analyze`` runs only when the person uploaded photos, video or audio. It reads them
into the look, the people and the sound the video should have.

``classify`` turns the user's topic or link into a :class:`state.Brief`.
``writer`` produces the screenplay and renders it into the video model's
prompt. ``approve_script`` is the cheap gate: the human reads the exact
prompts for free, and only approved prompts are allowed to spend money on a
generation. ``confirm_video`` is the final check on the rendered MP4, and
approving it is what posts to Instagram.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from config import Settings
from agents.common import uploads_summary
from state import PostState

NodeFn = Callable[[PostState], Awaitable[dict[str, Any]]]

#: Pydantic models stored in graph state. LangGraph refuses to deserialise
#: unregistered types from a checkpoint (today a warning, an error in a later
#: version), so they are allowlisted explicitly rather than implicitly trusted.
STATE_MODELS = (
    "Brief", "Shot", "Segment", "Screenplay", "Critique", "Dialogue", "Media",
    "Asset", "PersonLook", "AudioLook", "UploadAnalysis", "AssetBrief",
)


def make_serializer():
    """Build a checkpoint serializer that recognises our state models."""
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    return JsonPlusSerializer(
        allowed_msgpack_modules=[("state", name) for name in STATE_MODELS]
    )

# Node names, centralised so the wiring and the tests cannot drift apart.
ANALYZE = "analyze"
CLASSIFY = "classify"
RESEARCH = "research"
WRITER = "writer"
CRITIC = "critic"
DIALOGUE = "dialogue"
APPROVE_SCRIPT = "approve_script"
VIDEO = "video"
CONFIRM_VIDEO = "confirm_video"
PUBLISH = "publish"


@dataclass(frozen=True)
class Nodes:
    """The agent implementations the graph should call."""

    classify: NodeFn
    research: NodeFn
    writer: NodeFn
    critic: NodeFn
    dialogue: NodeFn
    video: NodeFn
    publish: NodeFn
    #: Reads uploaded files. Optional: without it, uploads are ignored.
    analyze: NodeFn | None = None


# ── Human-in-the-loop gates ───────────────────────────────────────────────
# NOTE: LangGraph re-runs a node from its start when resuming, so neither of
# these may perform a side effect before calling interrupt().


def approve_script_node(state: PostState) -> dict[str, Any]:
    """Pause for human review of the exact prompts, caption and critic score."""
    brief = state.get("brief")
    screenplay = state.get("screenplay")
    dialogue = state.get("dialogue")
    decision = interrupt(
        {
            "stage": "script",
            "duration_seconds": state.get("duration_seconds"),
            "intent": brief.intent if brief else None,
            "subject": brief.subject if brief else None,
            "title": screenplay.title if screenplay else None,
            "hook": screenplay.hook if screenplay else None,
            "cta": screenplay.cta if screenplay else None,
            "prompts": list(state.get("prompts") or []),
            "caption": dialogue.caption if dialogue else None,
            "hashtags": dialogue.hashtags if dialogue else [],
            "score": state.get("score"),
            "uploads": uploads_summary(state),
            "uploads_error": state.get("asset_error"),
            "critique": (
                state["critique"].model_dump() if state.get("critique") else None
            ),
        }
    )
    return _decision_updates(decision)


def confirm_video_node(state: PostState) -> dict[str, Any]:
    """Pause for a final ship/don't-ship check on the rendered video."""
    media = state.get("media")
    dialogue = state.get("dialogue")
    decision = interrupt(
        {
            "stage": "video",
            "media_type": media.media_type if media else None,
            "urls": media.urls if media else [],
            "files": media.local_files if media else [],
            "prompt_file": media.prompt_file if media else "",
            "duration_seconds": media.duration_seconds if media else 0,
            "notes": list(media.notes) if media else [],
            "caption": dialogue.caption if dialogue else None,
            "hashtags": dialogue.hashtags if dialogue else [],
        }
    )
    return _decision_updates(decision)


def _decision_updates(decision: Any) -> dict[str, Any]:
    """Normalise a ``Command(resume=...)`` payload into state updates.

    A rejection carrying feedback resets the iteration count so the writer
    gets a full set of fresh rounds to act on it. A decision may also carry a new
    ``duration_seconds``, for a person who asks for a different length while
    reviewing the prompts; the writer then rewrites to that length.
    """
    if not isinstance(decision, dict):
        decision = {"approved": bool(decision)}
    approved = bool(decision.get("approved"))
    feedback = (decision.get("feedback") or "").strip() or None

    updates: dict[str, Any] = {"approved": approved, "human_feedback": feedback}
    if not approved and feedback:
        updates["iterations"] = 0
    seconds = decision.get("duration_seconds")
    if isinstance(seconds, int) and not isinstance(seconds, bool) and seconds > 0:
        updates["duration_seconds"] = seconds
    return updates


# ── Routing ───────────────────────────────────────────────────────────────
# Pure functions, so they can be unit-tested without building a graph.


def route_start(state: PostState) -> str:
    """Read the uploads first when there are any, else go straight to classifying."""
    return ANALYZE if state.get("assets") else CLASSIFY


def route_after_classify(state: PostState) -> str:
    """Stop on an unusable request rather than researching nothing."""
    return END if state.get("error") else RESEARCH


def route_after_critic(state: PostState, settings: Settings) -> str:
    """Loop back to the writer while the score is low and rounds remain."""
    score = state.get("score") or 0.0
    iterations = state.get("iterations") or 0
    if score < settings.score_threshold and iterations < settings.max_iterations:
        return WRITER
    return DIALOGUE


def route_after_script_approval(state: PostState) -> str:
    """Approved -> generate video; rejected with notes -> rewrite; else stop."""
    if state.get("approved"):
        return VIDEO
    if state.get("human_feedback"):
        return WRITER
    return END


def route_after_video(state: PostState) -> str:
    """Skip the confirmation gate when generation failed.

    Asking a human to approve a video that does not exist is worse than
    ending the run and showing them the error.
    """
    return END if state.get("error") else CONFIRM_VIDEO


def route_after_video_confirm(state: PostState) -> str:
    """Approved -> publish; anything else ends the run without posting."""
    return PUBLISH if state.get("approved") else END


def build_graph(nodes: Nodes, settings: Settings, checkpointer: Any = None):
    """Wire the nodes into a compiled graph.

    Args:
        nodes: The agent implementations to call.
        settings: Supplies the loop threshold and iteration cap.
        checkpointer: Required for ``interrupt()`` to survive a restart.

    Returns:
        The compiled LangGraph application.
    """
    builder = StateGraph(PostState)

    if nodes.analyze:
        builder.add_node(ANALYZE, nodes.analyze)
    builder.add_node(CLASSIFY, nodes.classify)
    builder.add_node(RESEARCH, nodes.research)
    builder.add_node(WRITER, nodes.writer)
    builder.add_node(CRITIC, nodes.critic)
    builder.add_node(DIALOGUE, nodes.dialogue)
    builder.add_node(APPROVE_SCRIPT, approve_script_node)
    builder.add_node(VIDEO, nodes.video)
    builder.add_node(CONFIRM_VIDEO, confirm_video_node)
    builder.add_node(PUBLISH, nodes.publish)

    if nodes.analyze:
        builder.add_conditional_edges(
            START, route_start, {ANALYZE: ANALYZE, CLASSIFY: CLASSIFY}
        )
        builder.add_edge(ANALYZE, CLASSIFY)
    else:
        builder.add_edge(START, CLASSIFY)
    builder.add_conditional_edges(
        CLASSIFY,
        route_after_classify,
        {RESEARCH: RESEARCH, END: END},
    )
    builder.add_edge(RESEARCH, WRITER)
    builder.add_edge(WRITER, CRITIC)
    builder.add_conditional_edges(
        CRITIC,
        lambda s: route_after_critic(s, settings),
        {WRITER: WRITER, DIALOGUE: DIALOGUE},
    )
    builder.add_edge(DIALOGUE, APPROVE_SCRIPT)
    builder.add_conditional_edges(
        APPROVE_SCRIPT,
        route_after_script_approval,
        {VIDEO: VIDEO, WRITER: WRITER, END: END},
    )
    builder.add_conditional_edges(
        VIDEO,
        route_after_video,
        {CONFIRM_VIDEO: CONFIRM_VIDEO, END: END},
    )
    builder.add_conditional_edges(
        CONFIRM_VIDEO,
        route_after_video_confirm,
        {PUBLISH: PUBLISH, END: END},
    )
    builder.add_edge(PUBLISH, END)

    return builder.compile(checkpointer=checkpointer)
