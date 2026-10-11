"""The real nodes, wired through the real graph and a real SQLite checkpointer.

Only the LLM boundary is faked (``structured`` and the research model) and the
video provider is the offline stub. This is the test that proves the new state
models survive checkpointing and that the pieces actually fit together:
classify -> research -> writer -> critic -> dialogue -> gate -> video -> gate.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from graph import build_graph, make_serializer
from main import build_nodes
from state import Brief, Critique

from conftest import make_dialogue, make_screenplay

REQUEST = "explain how RAG works to non-technical people"


@pytest.fixture
def wired(monkeypatch, settings, tmp_path):
    """Real nodes with the LLM boundary replaced by canned structured outputs."""
    seen: dict[str, list] = {"schemas": []}

    async def fake_structured(cfg, node, schema, system, user, **kwargs):
        seen["schemas"].append(schema.__name__)
        if schema is Brief:
            return Brief(
                intent="explainer",
                subject="How retrieval-augmented generation works",
                audience="non-technical professionals",
                tone="friendly, plain-spoken",
                key_message="RAG lets a model look things up before it answers",
                visual_style="bright, clean explainer",
                dialogue_led=True,
            )
        if schema is Critique:
            return Critique(
                score=9.0, hook="ok", intent_fit="ok", prompt_craft="ok", feedback="none"
            )
        if schema.__name__ == "Screenplay":
            return make_screenplay()
        return make_dialogue()

    class FakeModel:
        async def ainvoke(self, messages):
            return SimpleNamespace(content="Research notes.")

    for module in ("classify", "research_script", "dialogue"):
        monkeypatch.setattr(f"agents.{module}.structured", fake_structured)
    monkeypatch.setattr("agents.research_script.get_model", lambda *a, **k: FakeModel())

    cfg = dataclasses.replace(
        settings,
        output_dir=tmp_path / "out",
        checkpoint_db=str(tmp_path / "checkpoints.sqlite"),
    )
    return cfg, seen


async def test_topic_to_video_through_a_real_checkpointer(wired):
    cfg, seen = wired
    config = {"configurable": {"thread_id": "pipeline"}}

    async with AsyncSqliteSaver.from_conn_string(cfg.checkpoint_db) as saver:
        saver.serde = make_serializer()
        app = build_graph(build_nodes(cfg), cfg, checkpointer=saver)

        first = await app.ainvoke({"request": REQUEST, "iterations": 0}, config)
        gate = first["__interrupt__"][0].value

        assert gate["stage"] == "script"
        assert gate["intent"] == "explainer"
        assert len(gate["prompts"]) == 2, "one prompt per Veo generation call"
        assert "[00:00-00:04]" in gate["prompts"][0]
        assert seen["schemas"][:2] == ["Brief", "Screenplay"]

        second = await app.ainvoke(Command(resume={"approved": True}), config)
        video_gate = second["__interrupt__"][0].value

        assert video_gate["stage"] == "video"
        video = Path(video_gate["files"][0])
        assert video.exists() and video.suffix == ".mp4"
        prompts = Path(video_gate["prompt_file"]).read_text(encoding="utf-8")
        assert gate["prompts"][0] in prompts, "the file records what was sent"

        # Declining to post still leaves the finished video on disk.
        final = await app.ainvoke(Command(resume={"approved": False}), config)
        assert final.get("post_id") is None
        assert video.exists()


async def test_a_paused_run_resumes_from_a_new_process(wired):
    """The state models must round-trip through SQLite, not just memory."""
    cfg, _ = wired
    config = {"configurable": {"thread_id": "resume"}}

    async with AsyncSqliteSaver.from_conn_string(cfg.checkpoint_db) as saver:
        saver.serde = make_serializer()
        app = build_graph(build_nodes(cfg), cfg, checkpointer=saver)
        await app.ainvoke({"request": REQUEST, "iterations": 0}, config)

    async with AsyncSqliteSaver.from_conn_string(cfg.checkpoint_db) as saver:
        saver.serde = make_serializer()
        app = build_graph(build_nodes(cfg), cfg, checkpointer=saver)
        resumed = await app.ainvoke(Command(resume={"approved": True}), config)

    assert resumed["__interrupt__"][0].value["stage"] == "video"


async def test_seedance_runs_end_to_end_on_the_stub(wired):
    cfg, _ = wired
    cfg = dataclasses.replace(cfg, video_model="seedance")
    config = {"configurable": {"thread_id": "seedance"}}

    async with AsyncSqliteSaver.from_conn_string(cfg.checkpoint_db) as saver:
        saver.serde = make_serializer()
        app = build_graph(build_nodes(cfg), cfg, checkpointer=saver)

        first = await app.ainvoke({"request": REQUEST, "iterations": 0}, config)
        prompts = first["__interrupt__"][0].value["prompts"]
        assert len(prompts) == 1 and "Shot 1 [0-" in prompts[0]

        second = await app.ainvoke(Command(resume={"approved": True}), config)
        assert second["__interrupt__"][0].value["duration_seconds"] == 15


async def test_rerunning_a_paused_thread_resumes_instead_of_restarting(wired, monkeypatch):
    """Regression: invoking a paused thread with fresh input restarted it from
    classify and repeated every LLM call, so --thread never actually resumed."""
    from main import run

    cfg, seen = wired

    def quit_the_process(_prompt):
        raise KeyboardInterrupt  # the user closes the terminal at the first gate

    monkeypatch.setattr("builtins.input", quit_the_process)
    with pytest.raises(KeyboardInterrupt):
        await run(REQUEST, "same-thread", cfg)
    assert seen["schemas"].count("Brief") == 1

    answers = iter(["y", "n"])  # approve the prompts, then decline to post
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    code = await run(REQUEST, "same-thread", cfg)

    assert code == 0
    assert seen["schemas"].count("Brief") == 1, "the classifier must not run again"
    assert seen["schemas"].count("Screenplay") == 1, "nor the writer"


# ── Video length ──────────────────────────────────────────────────────────

import shutil  # noqa: E402
import subprocess  # noqa: E402

needs_ffmpeg = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
    reason="needs ffmpeg and ffprobe on PATH",
)


def seconds_in(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    return float(out.strip())


@needs_ffmpeg
async def test_a_requested_length_runs_through_the_whole_pipeline(wired):
    """20s is generated as 22 (8+7+7) and cut: three prompts, a 20s file."""
    cfg, _ = wired
    config = {"configurable": {"thread_id": "length-20"}}

    async with AsyncSqliteSaver.from_conn_string(cfg.checkpoint_db) as saver:
        saver.serde = make_serializer()
        app = build_graph(build_nodes(cfg), cfg, checkpointer=saver)

        first = await app.ainvoke(
            {"request": REQUEST, "iterations": 0, "duration_seconds": 20}, config
        )
        gate = first["__interrupt__"][0].value
        assert gate["duration_seconds"] == 20
        assert len(gate["prompts"]) == 3, "8s base + 7s + 7s, of which the last keeps 5s"

        second = await app.ainvoke(Command(resume={"approved": True}), config)
        video_gate = second["__interrupt__"][0].value

    assert video_gate["duration_seconds"] == 20
    assert seconds_in(Path(video_gate["files"][0])) == pytest.approx(20, abs=0.2)


async def test_changing_the_length_at_the_review_rewrites_the_script_to_it(wired):
    """The length changes the structure: 15s is two prompts, 8s is one."""
    cfg, seen = wired
    config = {"configurable": {"thread_id": "length-change"}}

    async with AsyncSqliteSaver.from_conn_string(cfg.checkpoint_db) as saver:
        saver.serde = make_serializer()
        app = build_graph(build_nodes(cfg), cfg, checkpointer=saver)

        first = await app.ainvoke({"request": REQUEST, "iterations": 0}, config)
        gate = first["__interrupt__"][0].value
        assert gate["duration_seconds"] is None, "this run never named a length"
        assert len(gate["prompts"]) == 2, "the default is 15s"

        second = await app.ainvoke(
            Command(
                resume={
                    "approved": False,
                    "feedback": "make it shorter",
                    "duration_seconds": 8,
                }
            ),
            config,
        )
        rewritten = second["__interrupt__"][0].value

    assert rewritten["duration_seconds"] == 8
    assert len(rewritten["prompts"]) == 1, "8s is a single base clip"
    assert seen["schemas"].count("Screenplay") == 2, "the writer ran again"
