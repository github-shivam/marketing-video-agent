"""Unit tests for the pure routing functions.

These need no graph, no checkpointer and no event loop — they assert the
branch decisions directly.
"""

from __future__ import annotations

import pytest
from langgraph.graph import END

from graph import (
    DIALOGUE,
    PUBLISH,
    VIDEO,
    WRITER,
    _decision_updates,
    route_after_critic,
    route_after_script_approval,
    route_after_video_confirm,
)


class TestCriticLoop:
    """The writer<->critic loop must exit on score OR on the iteration cap."""

    def test_low_score_with_rounds_left_loops_back(self, settings):
        state = {"score": 5.0, "iterations": 1}
        assert route_after_critic(state, settings) == WRITER

    def test_score_at_threshold_exits(self, settings):
        """>= threshold exits; the check is strict less-than."""
        state = {"score": 8.0, "iterations": 1}
        assert route_after_critic(state, settings) == DIALOGUE

    def test_score_above_threshold_exits(self, settings):
        state = {"score": 9.5, "iterations": 1}
        assert route_after_critic(state, settings) == DIALOGUE

    def test_max_iterations_exits_even_with_low_score(self, settings):
        """A bad script must not loop forever."""
        state = {"score": 2.0, "iterations": settings.max_iterations}
        assert route_after_critic(state, settings) == DIALOGUE

    def test_iterations_beyond_cap_exits(self, settings):
        state = {"score": 2.0, "iterations": settings.max_iterations + 5}
        assert route_after_critic(state, settings) == DIALOGUE

    def test_missing_keys_default_to_looping(self, settings):
        """An empty state means nothing has been scored yet."""
        assert route_after_critic({}, settings) == WRITER


class TestScriptApproval:
    """All three branches of the cheap (pre-video) gate."""

    def test_approved_goes_to_video(self):
        assert route_after_script_approval({"approved": True}) == VIDEO

    def test_rejected_with_feedback_goes_back_to_writer(self):
        state = {"approved": False, "human_feedback": "Sharpen the hook"}
        assert route_after_script_approval(state) == WRITER

    def test_rejected_without_feedback_ends(self):
        state = {"approved": False, "human_feedback": None}
        assert route_after_script_approval(state) == END

    def test_empty_feedback_string_ends(self):
        """Whitespace-only feedback is treated as no feedback."""
        state = {"approved": False, "human_feedback": ""}
        assert route_after_script_approval(state) == END

    def test_approval_wins_even_if_feedback_present(self):
        state = {"approved": True, "human_feedback": "nice work"}
        assert route_after_script_approval(state) == VIDEO


class TestVideoConfirm:
    """The final ship/don't-ship gate never falls through to publish."""

    def test_approved_publishes(self):
        assert route_after_video_confirm({"approved": True}) == PUBLISH

    def test_rejected_ends(self):
        assert route_after_video_confirm({"approved": False}) == END

    def test_missing_approval_ends(self):
        """Absent approval must never be read as consent to post."""
        assert route_after_video_confirm({}) == END


class TestDecisionUpdates:
    """Normalising the Command(resume=...) payload."""

    def test_approval(self):
        assert _decision_updates({"approved": True}) == {
            "approved": True,
            "human_feedback": None,
        }

    def test_rejection_with_feedback_resets_iterations(self):
        """Feedback earns a full set of fresh writer rounds."""
        out = _decision_updates({"approved": False, "feedback": "Sharpen the hook"})
        assert out["approved"] is False
        assert out["human_feedback"] == "Sharpen the hook"
        assert out["iterations"] == 0

    def test_rejection_without_feedback_does_not_reset(self):
        out = _decision_updates({"approved": False})
        assert "iterations" not in out
        assert out["human_feedback"] is None

    def test_a_decision_can_carry_a_new_length(self):
        out = _decision_updates(
            {"approved": False, "feedback": "longer", "duration_seconds": 25}
        )
        assert out["duration_seconds"] == 25

    @pytest.mark.parametrize("bad", [None, 0, -5, "25", 2.5, True])
    def test_anything_that_is_not_a_positive_whole_number_of_seconds_is_ignored(self, bad):
        out = _decision_updates({"approved": False, "feedback": "x", "duration_seconds": bad})
        assert "duration_seconds" not in out

    def test_no_length_in_the_decision_leaves_the_length_alone(self):
        assert "duration_seconds" not in _decision_updates({"approved": True})

    def test_whitespace_feedback_is_treated_as_none(self):
        out = _decision_updates({"approved": False, "feedback": "   "})
        assert out["human_feedback"] is None
        assert "iterations" not in out

    @pytest.mark.parametrize("raw,expected", [(True, True), (False, False)])
    def test_bare_boolean_payload(self, raw, expected):
        """A CLI that resumes with a plain bool still works."""
        assert _decision_updates(raw)["approved"] is expected
