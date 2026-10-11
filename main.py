"""CLI entry point.

Usage::

    python main.py "explain how RAG works to non-technical people"
    python main.py https://example.com/product
    python main.py "a promo for this" https://example.com --model seedance
    python main.py "explain RAG" --thread my-run-id

The run pauses twice for human approval: once to read the exact prompts
before any money is spent, and once to look at the finished video and decide
whether to post it. Because state is checkpointed to SQLite, a paused run
survives the process exiting: rerun the same command with the same
``--thread`` to pick up where it stopped.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import logging
import sys
import uuid
from typing import Any

from langgraph.types import Command

from agents.publisher import VIDEO_MODELS
from config import Settings, settings
from graph import build_graph, make_serializer
from nodes import build_nodes
from video_plan import MAX_SECONDS, MIN_SECONDS, resolve_seconds

logger = logging.getLogger("marketing-agent")

RULE = "=" * 72


def force_utf8_output() -> None:
    """Make stdout/stderr UTF-8 safe.

    Windows consoles default to cp1252, which cannot encode emoji. Captions
    routinely contain them, and an un-encodable character in a ``print`` would
    otherwise abort the run at an approval gate — after the work was done.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            pass  # already wrapped, or not a real console


def render_script_review(payload: dict[str, Any]) -> None:
    """Print the first gate: the exact prompts the video model will receive."""
    print(f"\n{RULE}\n  PROMPT REVIEW  (nothing has been generated or spent yet)\n{RULE}")
    print(f"\nClassified as : {payload.get('intent')}")
    print(f"Subject       : {payload.get('subject')}")
    print(f"Title         : {payload.get('title')}")
    print(f"Hook          : {payload.get('hook')}")
    print(f"CTA           : {payload.get('cta') or '-'}")

    for i, prompt in enumerate(payload.get("prompts", [])):
        print(f"\n{'-' * 72}\nPROMPT {i}  (sent to the video model exactly as written)\n{'-' * 72}")
        print(prompt)

    print(f"\n{'-' * 72}\nCaption (used only if you post):\n{payload.get('caption')}")
    print(f"\nHashtags: {' '.join(payload.get('hashtags') or [])}")

    critique = payload.get("critique") or {}
    print(f"\nCritic score: {payload.get('score')}/10")
    if critique.get("feedback"):
        print(f"Critic notes: {critique['feedback']}")


def render_video_review(payload: dict[str, Any], cfg: Settings) -> None:
    """Print the final gate, after the video exists."""
    print(f"\n{RULE}\n  VIDEO REVIEW  (the video is saved whatever you answer)\n{RULE}")
    print(f"\nType     : {payload.get('media_type')}")
    print(f"Duration : {payload.get('duration_seconds')}s")
    for path in payload.get("files", []):
        print(f"File     : {path}")
    if payload.get("prompt_file"):
        print(f"Prompts  : {payload['prompt_file']}")
    print(f"\nCaption:\n{payload.get('caption')}")
    print(f"\nHashtags: {' '.join(payload.get('hashtags') or [])}")
    print("\nApproving POSTS this video to Instagram. Answer n to keep it without posting.")
    if cfg.dry_run:
        print("DRY_RUN=true - approving will NOT post to Instagram.")
    else:
        print("*** DRY_RUN=false - approving WILL post this to Instagram. ***")


def ask_human(stage: str) -> dict[str, Any]:
    """Prompt for a decision and return a ``Command(resume=...)`` payload.

    Rejecting the prompts with feedback sends them back to the writer;
    rejecting with no feedback ends the run.
    """
    prompt = "\nApprove? [y]es / [n]o + feedback / [q]uit without feedback: "
    if stage == "video":
        prompt = "\nPost to Instagram? [y]es / [n]o (keep the video, do not post): "
    while True:
        answer = input(prompt).strip().lower()
        if answer in {"y", "yes"}:
            return {"approved": True}
        if answer in {"q", "quit", ""}:
            return {"approved": False}
        if answer in {"n", "no"}:
            if stage == "video":
                return {"approved": False}
            feedback = input("What should change? ").strip()
            return {"approved": False, "feedback": feedback}
        print("Please answer y, n or q.")


def report_outputs(result: dict[str, Any]) -> None:
    """Say where the video and its prompts ended up."""
    media = result.get("media")
    if not media or not media.local_files:
        return
    print(f"\nVideo   : {media.local_files[0]}")
    if media.prompt_file:
        print(f"Prompts : {media.prompt_file}")


async def run(
    request: str, thread_id: str, cfg: Settings, seconds: int | None = None
) -> int:
    """Drive the graph, pausing at each approval gate.

    Args:
        request: The topic or link.
        thread_id: Reuse it to resume a paused run.
        cfg: Settings.
        seconds: How long the video should be; ``VIDEO_SECONDS_DEFAULT`` if None.

    Returns:
        A process exit code.
    """
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    plan = cfg.plan(seconds)
    print(f"\nRequest  : {request}")
    print(f"Thread   : {thread_id}")
    print(f"Text LLM : {cfg.model_for('writer')} (writer) @ {cfg.bedrock_text_region}")
    print(f"Video    : {cfg.video_model}  {plan.seconds}s")
    print(f"DRY_RUN  : {cfg.dry_run}")

    async with AsyncSqliteSaver.from_conn_string(cfg.checkpoint_db) as checkpointer:
        # from_conn_string takes no serde argument, so set it after construction.
        checkpointer.serde = make_serializer()
        app = build_graph(build_nodes(cfg), cfg, checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}

        # A thread already paused at a gate must be RESUMED. Invoking it with
        # fresh input would restart from the first node and repeat every LLM
        # call, so pick the pending question back up instead.
        snapshot = await app.aget_state(config)
        pending = [i for task in snapshot.tasks for i in task.interrupts]
        if pending:
            print("\nResuming a paused run.")
            result: dict[str, Any] = {"__interrupt__": pending}
        else:
            result = await app.ainvoke(
                {
                    "request": request,
                    "iterations": 0,
                    "duration_seconds": plan.seconds,
                },
                config,
            )

        while result.get("__interrupt__"):
            payload = result["__interrupt__"][0].value
            stage = payload.get("stage")
            if stage == "script":
                render_script_review(payload)
            else:
                render_video_review(payload, cfg)

            result = await app.ainvoke(Command(resume=ask_human(stage)), config)

    brief = result.get("brief")
    if brief:
        print(f"\nClassified as {brief.intent}: {brief.subject}")
    report_outputs(result)

    if result.get("error"):
        print(f"\nFinished with an error: {result['error']}")
        return 1
    if result.get("post_id"):
        print(f"\nPublished. post_id = {result['post_id']}")
        if result["post_id"] == "dry-run":
            print("That was a dry run. Set DRY_RUN=false in .env to post for real.")
        return 0

    print("\nRun ended without posting.")
    return 0


def main() -> int:
    """Parse arguments and run the graph."""
    parser = argparse.ArgumentParser(
        description="Turn a topic or a link into a generated video."
    )
    parser.add_argument(
        "request",
        nargs="+",
        help="A topic or idea, a link, or both",
    )
    parser.add_argument(
        "--model",
        choices=VIDEO_MODELS,
        default=None,
        help=f"Video model to write prompts for (default: VIDEO_MODEL from .env, "
        f"currently {settings.video_model})",
    )
    parser.add_argument(
        "--seconds",
        type=int,
        default=None,
        help=f"How long the video should be, {MIN_SECONDS}-{MAX_SECONDS} seconds "
        f"(default: VIDEO_SECONDS_DEFAULT, currently {settings.video_seconds_default})",
    )
    parser.add_argument(
        "--thread",
        default=None,
        help="Thread ID. Reuse it to resume a paused run (default: random).",
    )
    args = parser.parse_args()

    force_utf8_output()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(levelname)-8s %(name)s: %(message)s",
    )

    cfg = settings
    if args.model:
        cfg = dataclasses.replace(cfg, video_model=args.model)
    if cfg.video_model not in VIDEO_MODELS:
        raise SystemExit(
            f"VIDEO_MODEL={cfg.video_model!r} is not one of: {', '.join(VIDEO_MODELS)}"
        )

    request = " ".join(args.request).strip()
    thread_id = args.thread or f"run-{uuid.uuid4().hex[:8]}"

    seconds, note = resolve_seconds(args.seconds, cfg.video_seconds_default)
    if note:
        print(note)

    try:
        return asyncio.run(run(request, thread_id, cfg, seconds))
    except KeyboardInterrupt:
        print(f"\nInterrupted. Resume with:  python main.py \"{request}\" "
              f"--thread {thread_id}")
        return 130


if __name__ == "__main__":
    sys.exit(main())
