"""HTTP API around the video pipeline.

Run it::

    uvicorn api:create_app --factory --port 8000

A video takes minutes and pauses twice for a human, so this is a job API
rather than one request that returns a file::

    POST /videos                    start a run          -> 202 {video_id}
    GET  /videos/{id}               status, and the gate's payload if paused
    POST /videos/{id}/approve       answer the current gate
    GET  /videos/{id}/video         download the finished MP4

Every route except ``/health`` needs an ``X-API-Key`` header.
"""

from __future__ import annotations

import json
import logging
import secrets
from contextlib import asynccontextmanager
from typing import Annotated, Any, Callable, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

from agents.publisher import VIDEO_MODELS
from config import Settings, settings as default_settings
from conversations import ConversationStore
from graph import Nodes, make_serializer
from nodes import build_nodes
from runs import NoVideo, NotAtGate, RunBusy, RunManager, UnknownRun
from tools.whatsapp import WhatsAppClient, parse_incoming, verify_signature
from video_plan import MAX_SECONDS, MIN_SECONDS
from whatsapp_bot import WhatsAppBot

logger = logging.getLogger(__name__)


def get_manager(request: Request) -> RunManager:
    """The run manager, created by the app's lifespan."""
    return request.app.state.manager


#: Module level, because FastAPI resolves route annotations against module
#: globals and ``from __future__ import annotations`` stringifies them.
Manager = Annotated[RunManager, Depends(get_manager)]


class CreateVideo(BaseModel):
    """Body of ``POST /videos``."""

    request: str = Field(
        min_length=1,
        max_length=2000,
        description="A topic or idea, a link, or both",
        examples=["explain how RAG works to non-technical people"],
    )
    model: Literal["veo", "seedance"] | None = Field(
        default=None, description="Video model; defaults to VIDEO_MODEL"
    )
    seconds: int | None = Field(
        default=None,
        ge=MIN_SECONDS,
        le=MAX_SECONDS,
        description=(
            f"How long the video should be, {MIN_SECONDS}-{MAX_SECONDS} seconds. "
            "Defaults to VIDEO_SECONDS_DEFAULT."
        ),
    )


class Decision(BaseModel):
    """Body of ``POST /videos/{id}/approve``."""

    approved: bool
    feedback: str | None = Field(
        default=None,
        max_length=2000,
        description=(
            "At the prompt gate, a rejection with feedback sends the prompts "
            "back to the writer; without it the run ends. Ignored at the "
            "post gate."
        ),
    )
    seconds: int | None = Field(
        default=None,
        ge=MIN_SECONDS,
        le=MAX_SECONDS,
        description=(
            "A new video length, for a rejection with feedback. The script is "
            "rewritten to it. Ignored at the post gate."
        ),
    )


def create_app(
    cfg: Settings | None = None,
    nodes_for: Callable[[Settings], Nodes] = build_nodes,
    checkpointer: Any = None,
) -> FastAPI:
    """Build the application.

    Args:
        cfg: Settings; defaults to the environment's.
        nodes_for: Builds the agent nodes. Overridden by tests with fakes.
        checkpointer: A LangGraph checkpointer. When omitted the app opens the
            SQLite file named by ``CHECKPOINT_DB`` for its lifetime.

    Raises:
        RuntimeError: If ``API_KEY`` is not set. Every approved request spends
            money, so the service fails closed rather than starting open.
    """
    cfg = cfg or default_settings
    if not cfg.api_key:
        raise RuntimeError(
            "API_KEY is not set. Generate one with: "
            'python -c "import secrets; print(secrets.token_urlsafe(32))"'
        )

    whatsapp_on = cfg.whatsapp_configured
    if whatsapp_on and (missing := cfg.whatsapp_missing()):
        # Half-configured is worse than off: the webhook would accept messages
        # it cannot authenticate, or run for anyone.
        raise RuntimeError(
            "WhatsApp is partly configured; also set: " + ", ".join(missing)
            + ". Use WHATSAPP_ALLOWED_NUMBERS=* only if anyone may use the bot."
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.bot = None

        async def notify(video_id: str) -> None:
            if app.state.bot:
                await app.state.bot.on_run_settled(video_id)

        @asynccontextmanager
        async def serve(saver: Any):
            manager = RunManager(
                cfg, saver, nodes_for, on_settled=notify if whatsapp_on else None
            )
            app.state.manager = manager
            if whatsapp_on:
                app.state.bot = WhatsAppBot(
                    cfg,
                    manager,
                    ConversationStore(cfg.conversations_db),
                    WhatsAppClient(cfg),
                )
            try:
                yield
            finally:
                if app.state.bot:
                    await app.state.bot.close()
                await manager.close()

        if checkpointer is not None:
            async with serve(checkpointer):
                yield
            return

        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        async with AsyncSqliteSaver.from_conn_string(cfg.checkpoint_db) as saver:
            saver.serde = make_serializer()
            async with serve(saver):
                yield

    app = FastAPI(
        title="Marketing Video Agent",
        description="Turn a topic or a link into a generated video.",
        lifespan=lifespan,
    )

    def require_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
        # compare_digest so the check takes the same time however many leading
        # characters of a guess are right.
        if not x_api_key or not secrets.compare_digest(
            x_api_key.encode(), cfg.api_key.encode()
        ):
            raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")

    Auth = Depends(require_key)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        """Liveness check. Needs no key."""
        return {"status": "ok", "models": list(VIDEO_MODELS)}

    @app.post("/videos", status_code=202, dependencies=[Auth])
    async def create_video(body: CreateVideo, runs: Manager) -> dict[str, str]:
        """Start a run. Returns at once; poll ``status_url`` for progress."""
        video_id = await runs.create(body.request.strip(), body.model, body.seconds)
        return {"video_id": video_id, "status_url": f"/videos/{video_id}"}

    @app.get("/videos/{video_id}", dependencies=[Auth])
    async def get_video(video_id: str, runs: Manager) -> dict[str, Any]:
        """Where the run is, and what it needs or produced.

        ``status`` is one of: classifying, researching, writing,
        awaiting_prompt_approval, generating, awaiting_post_decision,
        publishing, completed, posted, ended, failed, interrupted. At a gate,
        ``awaiting`` carries what a person needs to decide.
        """
        try:
            return await runs.status(video_id)
        except UnknownRun:
            raise HTTPException(status_code=404, detail="no such video") from None

    @app.post("/videos/{video_id}/approve", status_code=202, dependencies=[Auth])
    async def approve(video_id: str, body: Decision, runs: Manager) -> dict[str, str]:
        """Answer the gate the run is paused at, then poll for what happens next."""
        try:
            await runs.decide(video_id, body.approved, body.feedback, body.seconds)
        except UnknownRun:
            raise HTTPException(status_code=404, detail="no such video") from None
        except (RunBusy, NotAtGate) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None
        return {"video_id": video_id, "status_url": f"/videos/{video_id}"}

    @app.get("/videos/{video_id}/video", dependencies=[Auth])
    async def download(video_id: str, runs: Manager) -> FileResponse:
        """Download the finished MP4."""
        try:
            path = await runs.video_path(video_id)
        except UnknownRun:
            raise HTTPException(status_code=404, detail="no such video") from None
        except NoVideo as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None
        return FileResponse(path, media_type="video/mp4", filename=path.name)

    if whatsapp_on:
        # Meta cannot send our X-API-Key, so these two routes are authenticated
        # by Meta's own mechanisms instead: the verify token on setup, and an
        # HMAC signature of the body on every delivery.

        @app.get("/whatsapp/webhook", response_class=PlainTextResponse)
        async def whatsapp_verify(
            mode: Annotated[str | None, Query(alias="hub.mode")] = None,
            token: Annotated[str | None, Query(alias="hub.verify_token")] = None,
            challenge: Annotated[str | None, Query(alias="hub.challenge")] = None,
        ) -> str:
            """Meta's one-off handshake when the webhook URL is registered."""
            if (
                mode == "subscribe"
                and token
                and secrets.compare_digest(token.encode(), cfg.whatsapp_verify_token.encode())
            ):
                return challenge or ""
            raise HTTPException(status_code=403, detail="verification failed")

        @app.post("/whatsapp/webhook")
        async def whatsapp_receive(
            request: Request,
            x_hub_signature_256: Annotated[str | None, Header()] = None,
        ) -> dict[str, str]:
            """Receive inbound messages. Always answers fast; work happens behind."""
            body = await request.body()
            if not verify_signature(cfg.whatsapp_app_secret, body, x_hub_signature_256):
                raise HTTPException(status_code=401, detail="bad signature")
            try:
                payload = json.loads(body)
            except ValueError:
                raise HTTPException(status_code=400, detail="not JSON") from None
            for message in parse_incoming(payload):
                request.app.state.bot.spawn(message)
            return {"status": "ok"}

    return app
