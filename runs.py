"""Runs the graph in the background and reports where each run is.

A video takes minutes, so a request cannot wait for it. Each run is a LangGraph
thread driven by a background task; its status is read back from the
checkpointer rather than tracked separately, so it is always what the graph
itself says. The same mechanism that lets the CLI resume a paused run lets an
API client answer an approval gate from a later request.

No HTTP lives here. :mod:`api` maps these methods and exceptions to routes.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from langgraph.types import Command

from agents.publisher import VIDEO_MODELS
from config import Settings
from graph import Nodes, build_graph
from state import Asset
from nodes import build_nodes

logger = logging.getLogger(__name__)

#: Ids look like ``veo-3f9a1c2b7d4e``. The model prefix says which graph owns the
#: run, which keeps it resumable after a restart without a separate store.
_ID = re.compile(rf"^({'|'.join(VIDEO_MODELS)})-[0-9a-f]{{12}}$")

#: Which stage a running run is in, keyed by the next node to complete.
_STAGE = {
    "analyze": "reading_uploads",
    "classify": "classifying",
    "research": "researching",
    "writer": "writing",
    "critic": "writing",
    "dialogue": "writing",
    "video": "generating",
    "publish": "publishing",
    # A gate answered a moment ago: its checkpoint has not moved on yet.
    "approve_script": "working",
    "confirm_video": "working",
}

_GATE_STATUS = {
    "script": "awaiting_prompt_approval",
    "video": "awaiting_post_decision",
}


class RunError(Exception):
    """Base class for errors the API turns into a response."""


class UnknownRun(RunError):
    """No run has this id."""


class RunBusy(RunError):
    """The run is still working and cannot take a decision."""


class NotAtGate(RunError):
    """The run is not waiting for a decision."""


class NoVideo(RunError):
    """The run has produced no video."""


@dataclass
class _Run:
    """Bookkeeping for a run driven by this process."""

    request: str
    task: asyncio.Task | None = None
    error: str | None = None

    @property
    def busy(self) -> bool:
        return self.task is not None and not self.task.done()


class RunManager:
    """Starts runs, answers their approval gates and reports their status."""

    def __init__(
        self,
        cfg: Settings,
        checkpointer: Any,
        nodes_for: Callable[[Settings], Nodes] = build_nodes,
        on_settled: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        """Compile one graph per video model.

        The writer's prompt and the video provider are fixed by the model at
        build time, so each model needs its own compiled graph. All of them
        share the one checkpointer.

        Args:
            cfg: Base settings.
            checkpointer: A LangGraph checkpointer.
            nodes_for: Builds the node implementations for a settings object.
                Overridden by tests to inject fakes.
            on_settled: Awaited with the run id each time a run stops driving
                itself: at an approval gate, on finishing, or on failing. It is
                how a chat front end learns there is something to tell the
                user, without polling.
        """
        self._cfg = cfg
        self._on_settled = on_settled
        self._notifying: set[asyncio.Task] = set()
        self._graphs: dict[str, Any] = {}
        for model in VIDEO_MODELS:
            model_cfg = dataclasses.replace(cfg, video_model=model)
            self._graphs[model] = build_graph(
                nodes_for(model_cfg), model_cfg, checkpointer=checkpointer
            )
        self._runs: dict[str, _Run] = {}

    # ── Public API ────────────────────────────────────────────────────────

    def _seconds(self, model: str, seconds: int | None) -> int:
        """The length that will really be made, for a video model.

        A request is turned into the generations that make it, and the length
        stored is the one those give, which can be a second short of the ask.
        """
        return dataclasses.replace(self._cfg, video_model=model).plan(seconds).seconds

    async def create(
        self,
        request: str,
        model: str | None = None,
        seconds: int | None = None,
        assets: list[Asset] | None = None,
    ) -> str:
        """Start a run and return its id without waiting for it.

        Args:
            request: The topic or link the user gave.
            model: ``veo`` or ``seedance``; defaults to ``VIDEO_MODEL``.
            seconds: How long the video should be; defaults to
                ``VIDEO_SECONDS_DEFAULT``. Clamped to 4-30.
            assets: Photos, video or audio the person uploaded, to steer the video.

        Returns:
            The new run's id.

        Raises:
            ValueError: If ``model`` is not a known video model.
        """
        model = model or self._cfg.video_model
        if model not in self._graphs:
            raise ValueError(f"unknown video model {model!r}; use one of {VIDEO_MODELS}")

        video_id = f"{model}-{uuid.uuid4().hex[:12]}"
        run = _Run(request=request)
        self._runs[video_id] = run
        state: dict[str, Any] = {
            "request": request,
            "iterations": 0,
            "duration_seconds": self._seconds(model, seconds),
        }
        if assets:
            state["assets"] = list(assets)
        self._launch(video_id, run, state)
        return video_id

    async def decide(
        self,
        video_id: str,
        approved: bool,
        feedback: str | None = None,
        seconds: int | None = None,
    ) -> None:
        """Answer the gate a run is paused at.

        Args:
            video_id: The run.
            approved: Yes or no.
            feedback: What to change, on a rejection.
            seconds: A new length, for a person who changed their mind about it
                while reviewing the prompts. The script is rewritten to it.

        Raises:
            UnknownRun: If there is no such run.
            RunBusy: If the run is already working on a previous answer.
            NotAtGate: If the run is not waiting for a decision.
        """
        snapshot = await self._snapshot(video_id)
        run = self._runs.setdefault(video_id, _Run(request=snapshot.values.get("request", "")))
        if run.busy:
            raise RunBusy("the run is still working; poll its status")
        if not _pending(snapshot):
            raise NotAtGate("the run is not waiting for a decision")

        run.error = None
        resume: dict[str, Any] = {"approved": approved, "feedback": feedback or ""}
        if seconds:
            model = self._model_of(video_id)
            resume["duration_seconds"] = self._seconds(model, seconds)
        self._launch(video_id, run, Command(resume=resume))

    async def status(self, video_id: str) -> dict[str, Any]:
        """Describe where a run is, and what it needs or produced."""
        snapshot = await self._snapshot(video_id)
        run = self._runs.get(video_id)
        values = snapshot.values or {}
        pending = _pending(snapshot)
        busy = bool(run and run.busy)

        out: dict[str, Any] = {
            "video_id": video_id,
            "model": video_id.split("-", 1)[0],
            "request": values.get("request") or (run.request if run else None),
            "duration_seconds": values.get("duration_seconds"),
        }
        brief = values.get("brief")
        if brief:
            out["classification"] = {
                "intent": brief.intent,
                "subject": brief.subject,
                "audience": brief.audience,
                "tone": brief.tone,
                "key_message": brief.key_message,
                "dialogue_led": brief.dialogue_led,
            }

        if run and run.error:
            return {**out, "status": "failed", "error": run.error}

        if busy:
            if not values:
                out["status"] = "classifying"
            elif snapshot.next:
                out["status"] = _STAGE.get(snapshot.next[0], "working")
            else:
                out["status"] = "finishing"
            return out

        if pending:
            payload = pending[0].value
            out["status"] = _GATE_STATUS.get(payload.get("stage"), "awaiting_decision")
            out["awaiting"] = payload
            return out

        if snapshot.next:
            # Nothing is driving it and it is not at a gate: the process that
            # was running it stopped (a restart, most likely).
            return {**out, "status": "interrupted"}

        if values.get("error"):
            return {**out, "status": "failed", "error": values["error"]}

        media = values.get("media")
        if media and media.local_files:
            out["status"] = "posted" if values.get("post_id") else "completed"
            out["result"] = {
                "video_url": f"/videos/{video_id}/video",
                "duration_seconds": media.duration_seconds,
                "prompts": values.get("prompts") or [],
                "post_id": values.get("post_id"),
            }
            return out

        out["status"] = "ended"  # declined at the prompt gate; nothing was made
        return out

    async def video_path(self, video_id: str) -> Path:
        """Return the local path of a run's finished video.

        Raises:
            UnknownRun: If there is no such run.
            NoVideo: If the run has not produced a video, or it is missing.
        """
        snapshot = await self._snapshot(video_id)
        media = (snapshot.values or {}).get("media")
        if not media or not media.local_files:
            raise NoVideo("this run has not produced a video")
        path = Path(media.local_files[0])
        if not path.is_file():
            raise NoVideo("the video file is no longer on disk")
        return path

    async def close(self) -> None:
        """Cancel any runs still in flight, for a clean shutdown."""
        tasks = [r.task for r in self._runs.values() if r.busy]
        tasks += list(self._notifying)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    # ── Internals ─────────────────────────────────────────────────────────

    def _model_of(self, video_id: str) -> str:
        match = _ID.match(video_id)
        if not match:
            raise UnknownRun(video_id)
        return match.group(1)

    def _graph(self, video_id: str):
        return self._graphs[self._model_of(video_id)]

    async def _snapshot(self, video_id: str):
        snapshot = await self._graph(video_id).aget_state(_config(video_id))
        if not snapshot.values and video_id not in self._runs:
            raise UnknownRun(video_id)
        return snapshot

    def _launch(self, video_id: str, run: _Run, payload: Any) -> None:
        run.task = asyncio.create_task(self._drive(video_id, run, payload))
        if self._on_settled:
            # A done-callback rather than a call at the end of _drive: inside
            # _drive the task is still alive, so status() would report it busy.
            run.task.add_done_callback(lambda task: self._settled(video_id, task))

    def _settled(self, video_id: str, task: asyncio.Task) -> None:
        if task.cancelled():
            return
        notify = asyncio.create_task(self._notify(video_id))
        self._notifying.add(notify)
        notify.add_done_callback(self._notifying.discard)

    async def _notify(self, video_id: str) -> None:
        try:
            await self._on_settled(video_id)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - a chat hiccup must not break the run
            logger.exception("Settle notification for %s failed", video_id)

    async def _drive(self, video_id: str, run: _Run, payload: Any) -> None:
        """Run the graph until it finishes or reaches the next gate."""
        try:
            await self._graph(video_id).ainvoke(payload, _config(video_id))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - recorded for the status endpoint
            logger.exception("Run %s failed", video_id)
            run.error = f"{type(exc).__name__}: {exc}"[:500]


def _config(video_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": video_id}}


def _pending(snapshot: Any) -> list[Any]:
    """The interrupts a paused run is waiting on."""
    return [i for task in snapshot.tasks for i in task.interrupts]
