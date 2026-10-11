"""Tests for transient-failure handling.

Nova Pro intermittently emits a malformed tool-use block, which Bedrock
surfaces as ``ModelErrorException: Model produced invalid sequence as part of
ToolUse``. That killed a live run at the critic node after research and
drafting had already completed, so both a retry and a graceful fallback are
covered here.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field, ValidationError

from agents.common import _is_retryable, structured
from agents.research_script import critic_node
from conftest import make_screenplay


class Demo(BaseModel):
    """Minimal schema for exercising ``structured``."""

    value: int = Field(ge=0, le=10)


class FakeModelError(Exception):
    """Stands in for botocore's ModelErrorException."""


class FlakyChain:
    """Raises a scripted sequence of errors, then returns a good value."""

    def __init__(self, failures: list[Exception]) -> None:
        self.failures = failures
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return Demo(value=7)


def patch_chain(monkeypatch, chain) -> None:
    """Make ``structured`` use ``chain`` instead of a real Bedrock model."""

    class FakeModel:
        def with_structured_output(self, schema):
            return chain

    monkeypatch.setattr("agents.common.get_model", lambda *a, **k: FakeModel())


class TestRetryClassification:
    def test_validation_errors_are_retryable(self):
        try:
            Demo(value=99)
        except ValidationError as exc:
            assert _is_retryable(exc)

    @pytest.mark.parametrize(
        "message",
        [
            "Model produced invalid sequence as part of ToolUse",
            "An error occurred (ThrottlingException) when calling Converse",
            "ServiceUnavailable",
        ],
    )
    def test_transient_bedrock_errors_are_retryable(self, message):
        assert _is_retryable(FakeModelError(message))

    def test_class_name_alone_is_enough(self):
        exc = type("ModelErrorException", (Exception,), {})("opaque")
        assert _is_retryable(exc)

    @pytest.mark.parametrize(
        "exc",
        [
            PermissionError("AccessDeniedException: not authorized"),
            ValueError("bad config"),
        ],
    )
    def test_real_errors_are_not_retryable(self, exc):
        assert not _is_retryable(exc)


class TestStructuredRetry:
    async def test_recovers_from_a_malformed_tool_use(self, settings, monkeypatch):
        chain = FlakyChain(
            [FakeModelError("Model produced invalid sequence as part of ToolUse")]
        )
        patch_chain(monkeypatch, chain)

        result = await structured(settings, "critic", Demo, "sys", "user")

        assert result.value == 7
        assert chain.calls == 2

    async def test_gives_up_after_the_attempt_budget(self, settings, monkeypatch):
        msg = "Model produced invalid sequence as part of ToolUse"
        chain = FlakyChain([FakeModelError(msg) for _ in range(10)])
        patch_chain(monkeypatch, chain)

        with pytest.raises(FakeModelError):
            await structured(settings, "critic", Demo, "sys", "user", attempts=3)

        assert chain.calls == 3, "should stop at the budget, not retry forever"

    async def test_non_transient_errors_are_not_retried(self, settings, monkeypatch):
        chain = FlakyChain([PermissionError("AccessDeniedException")])
        patch_chain(monkeypatch, chain)

        with pytest.raises(PermissionError):
            await structured(settings, "critic", Demo, "sys", "user")

        assert chain.calls == 1, "a real error must fail fast, not burn retries"


class TestCriticFallback:
    async def test_critic_failure_does_not_kill_the_run(
        self, settings, brief, monkeypatch
    ):
        """A flaked critic hands off to the human gate instead of crashing."""

        async def always_fails(*args, **kwargs):
            raise FakeModelError("ToolUse exploded")

        monkeypatch.setattr("agents.research_script.structured", always_fails)

        result = await critic_node(settings)(
            {"brief": brief, "screenplay": make_screenplay(), "prompts": ["p"]}
        )

        assert result["score"] == settings.score_threshold, (
            "scoring at the threshold exits the loop and reaches human review"
        )
        assert "critic unavailable" in result["critique"].hook.lower()
        assert "ToolUse exploded" in result["critique"].feedback

    async def test_healthy_critic_is_unaffected(self, settings, brief, monkeypatch):
        from state import Critique

        async def ok(*args, **kwargs):
            return Critique(
                score=4.0,
                hook="weak",
                intent_fit="off",
                prompt_craft="vague",
                feedback="fix it",
            )

        monkeypatch.setattr("agents.research_script.structured", ok)
        result = await critic_node(settings)(
            {"brief": brief, "screenplay": make_screenplay(), "prompts": ["p"]}
        )

        assert result["score"] == 4.0
        assert result["critique"].feedback == "fix it"


class TestSpeechPacing:
    """The word budget must scale with clip length and configured pace."""

    def test_budget_scales_with_duration(self, settings):
        assert settings.words_for(8) > settings.words_for(7)

    def test_budget_leaves_a_tail_so_speech_finishes(self, settings):
        """The target must fit in LESS than the clip, so nothing is cut off."""
        assert settings.words_for(8) == round(175 / 60 * (8 - 1.0))
        assert settings.words_for(8) < settings.words_ceiling(8)

    def test_ceiling_is_the_full_clip(self, settings):
        """Beyond the ceiling the clip ends before the sentence does."""
        assert settings.words_ceiling(8) == round(175 / 60 * 8)

    def test_faster_pace_allows_more_words(self, settings):
        import dataclasses

        fast = dataclasses.replace(settings, speech_pace_wpm=220)
        assert fast.words_for(8) > settings.words_for(8)

    def test_the_default_plan_is_a_base_clip_then_one_extension(self, settings):
        plan = settings.plan()
        assert plan.use_seconds == [8, 7]
        assert sum(plan.use_seconds) == plan.seconds == 15

    def test_total_budget_well_above_the_old_fixed_cap(self, settings):
        """The old hard-coded 10-16 words/beat produced 88 wpm - far too slow."""
        total = sum(settings.words_for(s) for s in settings.plan().use_seconds)
        assert total >= 35, f"15s should carry ~38 words, got {total}"

    def test_speech_ends_before_the_clip_does(self, settings):
        """Every beat's target must leave at least the configured tail."""
        for secs in settings.plan().use_seconds:
            spoken_seconds = settings.words_for(secs) / (settings.speech_pace_wpm / 60)
            assert spoken_seconds <= secs - 0.8, (
                f"{secs}s beat would speak for {spoken_seconds:.1f}s"
            )


class TestPacingWarnings:
    """Over- and under-budget dialogue must be visible, not silent."""

    def _screenplay(self, first: str, second: str = ""):
        """Segment 0 is two 4s shots carrying the given dialogue."""
        play = make_screenplay()
        shots = play.segments[0].shots
        shots[0].dialogue = first
        shots[1].dialogue = second
        return play

    def test_over_budget_warns(self, settings, caplog):
        from agents.research_script import _warn_on_pacing

        # 39 words in a 4s shot - far past what can be spoken.
        with caplog.at_level("WARNING"):
            _warn_on_pacing(
                self._screenplay(" ".join(["word"] * 39)), settings, settings.plan()
            )
        assert "rushed or cut off" in caplog.text

    def test_under_budget_warns(self, settings, caplog):
        from agents.research_script import _warn_on_pacing

        with caplog.at_level("WARNING"):
            _warn_on_pacing(self._screenplay("Too short."), settings, settings.plan())
        assert "dead air" in caplog.text

    def test_on_budget_is_silent(self, settings, caplog):
        from agents.research_script import _warn_on_pacing

        half = " ".join(["word"] * (settings.words_for(8) // 2))
        with caplog.at_level("WARNING"):
            _warn_on_pacing(self._screenplay(half, half), settings, settings.plan())
        assert "spoken words" not in caplog.text

    def test_silent_segments_are_not_flagged(self, settings, caplog):
        """A visual-led video has no dialogue, which is not dead air."""
        from agents.research_script import _warn_on_pacing

        with caplog.at_level("WARNING"):
            _warn_on_pacing(self._screenplay(""), settings, settings.plan())
        assert "dead air" not in caplog.text


class TestPerNodeModels:
    """Each node names its own model; MODEL is only an optional shared fallback."""

    NODES = ("classify", "research", "writer", "critic", "dialogue", "converse")

    def test_blank_overrides_fall_back_to_the_shared_model(self, settings):
        for node in self.NODES:
            assert settings.model_for(node) == settings.model

    def test_every_node_can_be_set_without_any_shared_model(self, settings):
        """The setup with no MODEL at all: each node is named explicitly."""
        import dataclasses

        cfg = dataclasses.replace(
            settings,
            model="",
            model_classify="c", model_research="r", model_writer="w",
            model_critic="k", model_dialogue="d", model_converse="v",
        )
        assert [cfg.model_for(n) for n in self.NODES] == ["c", "r", "w", "k", "d", "v"]

    def test_a_missing_model_fails_loudly_instead_of_using_a_hidden_default(
        self, settings
    ):
        import dataclasses

        cfg = dataclasses.replace(settings, model="", model_writer="w")
        assert cfg.model_for("writer") == "w"
        with pytest.raises(ValueError, match="MODEL_CONVERSE"):
            cfg.model_for("converse")

    def test_there_is_no_built_in_default_model(self, monkeypatch):
        """Regression: MODEL used to default to Nova Pro when unset."""
        from config import Settings

        for name in ("MODEL", "MODEL_WRITER"):
            monkeypatch.setenv(name, "")
        cfg = Settings.from_env()
        assert cfg.model == ""
        with pytest.raises(ValueError, match="No text model configured"):
            cfg.model_for("writer")

    def test_each_node_reads_its_own_override(self, settings):
        import dataclasses

        cfg = dataclasses.replace(
            settings,
            model_classify="classify-model",
            model_writer="writer-model",
        )
        assert cfg.model_for("classify") == "classify-model"
        assert cfg.model_for("writer") == "writer-model"
        assert cfg.model_for("critic") == settings.model, "unset nodes keep the default"

    def test_unknown_nodes_use_the_default(self, settings):
        assert settings.model_for("something-else") == settings.model
