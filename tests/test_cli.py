"""CLI rendering and decision-parsing tests.

The emoji cases are regression tests for a real failure: a caption containing
U+1F331 aborted a live run on a cp1252 Windows console, at the approval gate,
after all the upstream work had already been done.
"""

from __future__ import annotations

import pytest

from main import (
    ask_human,
    build_nodes,
    force_utf8_output,
    render_script_review,
    render_video_review,
    report_outputs,
)

EMOJI_CAPTION = "Roasted to order 🌱 tasted at origin ☕ shipped today 📦"


def script_payload(**overrides) -> dict:
    """A script-gate interrupt payload."""
    base = {
        "stage": "script",
        "intent": "product_promo",
        "subject": "single-origin coffee",
        "title": "Roasted to order",
        "hook": "Your coffee deserves better",
        "cta": "Order a bag",
        "prompts": ["[00:00-00:04] Close-up of beans cracking.", "Continue the same shot."],
        "caption": EMOJI_CAPTION,
        "hashtags": ["#coffee", "#roastery"],
        "score": 7.5,
        "critique": {"feedback": "Sharpen the hook 🔥"},
    }
    return {**base, **overrides}


class TestRenderingSurvivesEmoji:
    def test_script_review_renders_emoji(self, capsys):
        force_utf8_output()
        render_script_review(script_payload())

        out = capsys.readouterr().out
        assert "🌱" in out
        assert "Your coffee deserves better" in out

    def test_video_review_renders_emoji(self, capsys, settings):
        force_utf8_output()
        render_video_review(
            {
                "stage": "video",
                "media_type": "REELS",
                "duration_seconds": 15,
                "files": ["output/reel.mp4"],
                "prompt_file": "output/reel.prompt.txt",
                "caption": EMOJI_CAPTION,
                "hashtags": ["#coffee"],
            },
            settings,
        )
        assert "🌱" in capsys.readouterr().out

    def test_force_utf8_is_idempotent(self):
        """Calling it twice, or on a non-console stream, must not raise."""
        force_utf8_output()
        force_utf8_output()


class TestRenderingHandlesMissingFields:
    def test_empty_payload_does_not_crash(self, capsys):
        render_script_review({"stage": "script"})
        assert "PROMPT REVIEW" in capsys.readouterr().out

    def test_every_prompt_is_printed_verbatim(self, capsys):
        """The gate must show exactly what the video model will receive."""
        payload = script_payload()
        render_script_review(payload)

        out = capsys.readouterr().out
        for prompt in payload["prompts"]:
            assert prompt in out

    def test_classification_is_shown(self, capsys):
        render_script_review(script_payload())
        out = capsys.readouterr().out
        assert "product_promo" in out
        assert "single-origin coffee" in out


class TestDryRunWarning:
    def test_live_mode_warns_loudly(self, capsys, settings):
        import dataclasses

        live = dataclasses.replace(settings, dry_run=False)
        render_video_review({"stage": "video", "urls": []}, live)
        assert "WILL post" in capsys.readouterr().out

    def test_dry_run_says_it_will_not_post(self, capsys, settings):
        render_video_review({"stage": "video", "urls": []}, settings)
        assert "will NOT post" in capsys.readouterr().out


class TestAskHuman:
    @pytest.mark.parametrize("answer", ["y", "yes", "Y", "  YES  "])
    def test_approval_variants(self, answer, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: answer)
        assert ask_human("script") == {"approved": True}

    @pytest.mark.parametrize("answer", ["q", "quit", ""])
    def test_quit_variants_reject_without_feedback(self, answer, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: answer)
        assert ask_human("script") == {"approved": False}

    def test_rejection_collects_feedback_at_the_script_gate(self, monkeypatch):
        answers = iter(["n", "Sharpen the hook"])
        monkeypatch.setattr("builtins.input", lambda _: next(answers))
        assert ask_human("script") == {
            "approved": False,
            "feedback": "Sharpen the hook",
        }

    def test_declining_to_post_does_not_ask_for_feedback(self, monkeypatch):
        """There is no rewrite path from the video gate, so do not pretend."""
        calls: list[str] = []

        def fake_input(prompt: str) -> str:
            calls.append(prompt)
            return "n"

        monkeypatch.setattr("builtins.input", fake_input)
        assert ask_human("video") == {"approved": False}
        assert len(calls) == 1, "should not prompt for feedback"

    def test_invalid_answer_reprompts(self, monkeypatch):
        answers = iter(["maybe", "nonsense", "y"])
        monkeypatch.setattr("builtins.input", lambda _: next(answers))
        assert ask_human("script") == {"approved": True}


class TestVideoGatePrompt:
    def test_the_question_is_about_posting(self, monkeypatch):
        """The video already exists; the only open decision is whether to post."""
        asked: list[str] = []

        def fake_input(prompt: str) -> str:
            asked.append(prompt)
            return "y"

        monkeypatch.setattr("builtins.input", fake_input)
        assert ask_human("video") == {"approved": True}
        assert "Post to Instagram" in asked[0]

    def test_video_review_says_where_the_video_is(self, capsys, settings):
        render_video_review(
            {
                "stage": "video",
                "files": ["output/reel.mp4"],
                "prompt_file": "output/reel.prompt.txt",
            },
            settings,
        )
        out = capsys.readouterr().out
        assert "output/reel.mp4" in out
        assert "output/reel.prompt.txt" in out
        assert "keep it without posting" in out


class TestReportOutputs:
    def test_prints_the_video_and_prompt_paths(self, capsys):
        from state import Media

        report_outputs(
            {"media": Media(local_files=["a.mp4"], prompt_file="a.prompt.txt")}
        )
        out = capsys.readouterr().out
        assert "a.mp4" in out and "a.prompt.txt" in out

    def test_silent_when_no_video_was_made(self, capsys):
        report_outputs({})
        assert capsys.readouterr().out == ""


class TestNodeWiring:
    def test_build_nodes_supplies_every_node(self, settings):
        nodes = build_nodes(settings)
        for field in (
            "classify", "research", "writer", "critic", "dialogue", "video", "publish",
        ):
            assert callable(getattr(nodes, field))


class TestSecondsOption:
    """``--seconds`` sets the video length, clamped to what can be made."""

    @pytest.fixture
    def captured(self, monkeypatch):
        import main

        seen: dict = {}

        async def fake_run(request, thread_id, cfg, seconds=None):
            seen.update(request=request, seconds=seconds)
            return 0

        monkeypatch.setattr(main, "run", fake_run)
        return seen

    def run_main(self, monkeypatch, *argv):
        import main

        monkeypatch.setattr("sys.argv", ["main.py", *argv])
        return main.main()

    def test_the_default_is_the_configured_length(self, monkeypatch, captured):
        from config import settings

        self.run_main(monkeypatch, "explain RAG")
        assert captured["seconds"] == settings.video_seconds_default

    def test_a_length_is_passed_through(self, monkeypatch, captured):
        self.run_main(monkeypatch, "explain RAG", "--seconds", "22")
        assert captured["seconds"] == 22

    def test_too_long_is_clamped_and_the_person_is_told(self, monkeypatch, captured, capsys):
        self.run_main(monkeypatch, "explain RAG", "--seconds", "120")

        assert captured["seconds"] == 30
        assert "longest video I can make is 30 seconds" in capsys.readouterr().out

    def test_too_short_is_clamped_and_the_person_is_told(self, monkeypatch, captured, capsys):
        self.run_main(monkeypatch, "explain RAG", "--seconds", "1")

        assert captured["seconds"] == 4
        assert "shortest" in capsys.readouterr().out
