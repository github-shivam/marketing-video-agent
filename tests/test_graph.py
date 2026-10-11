"""End-to-end graph runs driven entirely by fakes.

Every path through both approval gates is exercised, including the rewrite
loop that a rejection-with-feedback is supposed to trigger.
"""

from __future__ import annotations

import dataclasses

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from graph import build_graph

THREAD = {"configurable": {"thread_id": "test-thread"}}
REQUEST = "a promo for Ridgeline Coffee"


@pytest.fixture
def app(agents, settings):
    """A compiled graph wired to the fake agents and an in-memory checkpointer."""
    return build_graph(agents.as_nodes(), settings, checkpointer=InMemorySaver())


async def start(app):
    """Run from START until the first interrupt."""
    return await app.ainvoke({"request": REQUEST, "iterations": 0}, THREAD)


class TestHappyPath:
    async def test_runs_to_publish_with_both_approvals(self, app, agents):
        first = await start(app)
        assert "__interrupt__" in first, "should pause at the script gate"

        payload = first["__interrupt__"][0].value
        assert payload["stage"] == "script"
        assert payload["hook"] == "Your coffee deserves better"
        assert payload["score"] == 9.0
        assert payload["intent"] == "product_promo"
        assert payload["prompts"] == ["PROMPT ZERO", "PROMPT ONE"], (
            "the gate must show the exact prompts the video model will get"
        )
        assert agents.calls == ["classify", "research", "writer", "critic", "dialogue"]
        assert "video" not in agents.calls, "no money spent before approval"

        second = await app.ainvoke(Command(resume={"approved": True}), THREAD)
        assert "__interrupt__" in second, "should pause at the video gate"
        assert second["__interrupt__"][0].value["stage"] == "video"
        assert second["__interrupt__"][0].value["duration_seconds"] == 15

        final = await app.ainvoke(Command(resume={"approved": True}), THREAD)
        assert final["post_id"] == "dry-run"
        assert agents.calls == [
            "classify", "research", "writer", "critic", "dialogue", "video", "publish",
        ]


class TestCriticLoopEndToEnd:
    async def test_loops_until_score_clears_threshold(self, agents, settings):
        """Two failing verdicts, then a pass: three writer rounds total."""
        agents.scores = [4.0, 6.0, 9.0]
        app = build_graph(agents.as_nodes(), settings, checkpointer=InMemorySaver())

        await start(app)
        assert agents.calls.count("writer") == 3
        assert agents.calls.count("critic") == 3
        assert agents.calls.count("dialogue") == 1

    async def test_stops_at_max_iterations_despite_low_scores(
        self, agents, settings
    ):
        """A persistently bad script must still reach the gate, not spin."""
        agents.scores = [1.0] * 10
        app = build_graph(agents.as_nodes(), settings, checkpointer=InMemorySaver())

        result = await start(app)
        assert agents.calls.count("writer") == settings.max_iterations
        assert "__interrupt__" in result
        assert result["__interrupt__"][0].value["score"] == 1.0


class TestRejection:
    async def test_feedback_triggers_rewrite_then_publishes(
        self, app, agents
    ):
        """The documented loop: reject with notes, rewrite, approve, ship."""
        await start(app)
        writers_before = agents.calls.count("writer")

        await app.ainvoke(
            Command(resume={"approved": False, "feedback": "Make the hook sharper"}),
            THREAD,
        )
        assert agents.calls.count("writer") > writers_before, "should rewrite"
        assert "Make the hook sharper" in agents.writer_saw_feedback, (
            "the writer must actually receive the human's feedback"
        )
        assert "video" not in agents.calls

        await app.ainvoke(Command(resume={"approved": True}), THREAD)
        final = await app.ainvoke(Command(resume={"approved": True}), THREAD)
        assert final["post_id"] == "dry-run"

    async def test_rejection_without_feedback_ends_without_publishing(
        self, app, agents
    ):
        await start(app)
        result = await app.ainvoke(Command(resume={"approved": False}), THREAD)

        assert "publish" not in agents.calls
        assert "video" not in agents.calls
        assert result.get("post_id") is None

    async def test_rejecting_the_video_does_not_publish(self, app, agents):
        """Money already spent on the video is no reason to post it."""
        await start(app)
        await app.ainvoke(Command(resume={"approved": True}), THREAD)
        result = await app.ainvoke(Command(resume={"approved": False}), THREAD)

        assert "video" in agents.calls, "the video was generated"
        assert "publish" not in agents.calls, "but it was not posted"
        assert result.get("post_id") is None


class TestCheckpointing:
    async def test_paused_run_survives_a_new_app_instance(
        self, agents, settings
    ):
        """A shared checkpointer lets a different process resume the interrupt."""
        saver = InMemorySaver()
        first_app = build_graph(agents.as_nodes(), settings, checkpointer=saver)
        await first_app.ainvoke({"request": REQUEST, "iterations": 0}, THREAD)

        resumed_app = build_graph(agents.as_nodes(), settings, checkpointer=saver)
        await resumed_app.ainvoke(Command(resume={"approved": True}), THREAD)
        final = await resumed_app.ainvoke(Command(resume={"approved": True}), THREAD)

        assert final["post_id"] == "dry-run"


class TestClassification:
    async def test_unusable_request_stops_before_any_research(self, agents, settings):
        """An empty request must end the run, not research nothing."""

        async def rejects(state):
            agents.calls.append("classify")
            return {"error": "empty request: give a topic or a link"}

        nodes = dataclasses.replace(agents.as_nodes(), classify=rejects)
        app = build_graph(nodes, settings, checkpointer=InMemorySaver())

        result = await app.ainvoke({"request": "", "iterations": 0}, THREAD)

        assert agents.calls == ["classify"]
        assert "__interrupt__" not in result
        assert "empty request" in result["error"]


class TestVideoFailure:
    async def test_failed_generation_skips_the_confirm_gate(
        self, agents, settings
    ):
        """A human must not be asked to approve a video that does not exist."""

        async def failing_video(state):
            agents.calls.append("video")
            return {"error": "video generation failed: quota exceeded"}

        nodes = dataclasses.replace(agents.as_nodes(), video=failing_video)
        app = build_graph(nodes, settings, checkpointer=InMemorySaver())

        await app.ainvoke({"request": REQUEST, "iterations": 0}, THREAD)
        result = await app.ainvoke(Command(resume={"approved": True}), THREAD)

        assert "__interrupt__" not in result, "should not pause on a failed video"
        assert "publish" not in agents.calls
        assert "quota exceeded" in result["error"]
