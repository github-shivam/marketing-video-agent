"""Tests for the intake side: fetching a page, classifying the request, and
the screenwriter node that consumes the result.

No model and no network: ``structured`` is replaced with a recording fake and
HTTP is mocked with respx.
"""

from __future__ import annotations

import dataclasses

import httpx
import pytest
import respx

from agents.classify import classify_node
from agents.prompt_guides import INTENT_GUIDES, SEEDANCE_GUIDE, VEO_GUIDE
from agents.research_script import writer_node
from state import Brief
from tools.webpage import WebpageError, fetch_page_text, html_to_text

from conftest import make_screenplay

PAGE = """<html><head><title>Ridgeline Coffee</title>
<meta name="description" content="Roasted to order, shipped the same day.">
<style>body { color: red }</style></head>
<body><nav>Home Shop</nav><h1>Single-origin coffee</h1>
<p>Roasted to order.</p><script>alert('x')</script><p>Roasted to order.</p>
</body></html>"""


class TestHtmlToText:
    def test_keeps_title_and_visible_text(self):
        text = html_to_text(PAGE)
        assert "Title: Ridgeline Coffee" in text
        assert "Single-origin coffee" in text

    def test_drops_scripts_and_styles(self):
        text = html_to_text(PAGE)
        assert "alert" not in text
        assert "color: red" not in text

    def test_keeps_the_meta_description(self):
        assert "Roasted to order, shipped the same day." in html_to_text(PAGE)

    def test_output_is_truncated(self):
        assert len(html_to_text("<p>" + "word " * 5000 + "</p>", max_chars=100)) <= 100


class TestFetchPage:
    @respx.mock
    async def test_returns_readable_text(self):
        respx.get("https://example.com/").mock(
            return_value=httpx.Response(
                200, text=PAGE, headers={"content-type": "text/html; charset=utf-8"}
            )
        )
        assert "Single-origin coffee" in await fetch_page_text("https://example.com/")

    @respx.mock
    async def test_http_errors_raise(self):
        respx.get("https://example.com/").mock(return_value=httpx.Response(404))
        with pytest.raises(WebpageError, match="404"):
            await fetch_page_text("https://example.com/")

    @respx.mock
    async def test_network_errors_raise(self):
        respx.get("https://example.com/").mock(side_effect=httpx.ConnectError("down"))
        with pytest.raises(WebpageError, match="could not fetch"):
            await fetch_page_text("https://example.com/")

    @respx.mock
    async def test_non_html_raises(self):
        respx.get("https://example.com/a.pdf").mock(
            return_value=httpx.Response(
                200, content=b"%PDF", headers={"content-type": "application/pdf"}
            )
        )
        with pytest.raises(WebpageError, match="not a web page"):
            await fetch_page_text("https://example.com/a.pdf")

    @respx.mock
    async def test_a_page_with_no_text_raises(self):
        respx.get("https://example.com/").mock(
            return_value=httpx.Response(
                200,
                text="<html><body><script>app()</script></body></html>",
                headers={"content-type": "text/html"},
            )
        )
        with pytest.raises(WebpageError, match="no readable text"):
            await fetch_page_text("https://example.com/")

    async def test_non_http_schemes_are_refused(self):
        for url in ("file:///etc/passwd", "ftp://example.com/x"):
            with pytest.raises(WebpageError, match="unsupported"):
                await fetch_page_text(url)


class FakeStructured:
    """Records each ``structured`` call and returns a canned result."""

    def __init__(self, result):
        self.result = result
        self.calls: list[dict] = []

    async def __call__(self, settings, node, schema, system, user, **kwargs):
        self.calls.append({"node": node, "schema": schema, "system": system, "user": user})
        return self.result


def llm_brief(**overrides) -> Brief:
    """What the classifier LLM returns, including wrongly filled code fields."""
    base = {
        "intent": "product_promo",
        "subject": "Ridgeline's single-origin coffee",
        "audience": "home brewers",
        "tone": "warm",
        "key_message": "Roasted to order",
        "visual_style": "warm editorial",
        "request": "model wrote this",
        "input_kind": "topic",
        "url": "https://wrong.invalid",
    }
    return Brief(**{**base, **overrides})


class TestClassifyNode:
    async def test_a_topic_is_classified_without_fetching(self, settings, monkeypatch):
        fake = FakeStructured(llm_brief())
        monkeypatch.setattr("agents.classify.structured", fake)

        async def boom(url):
            raise AssertionError("a topic must not trigger a page fetch")

        monkeypatch.setattr("agents.classify.fetch_page_text", boom)

        out = await classify_node(settings)({"request": "explain how RAG works"})

        assert out["brief"].input_kind == "topic"
        assert out["brief"].url is None
        assert out["source_text"] == ""

    async def test_code_owns_the_request_kind_and_url(self, settings, monkeypatch):
        """The LLM is told to leave these blank; whatever it writes is overridden."""
        monkeypatch.setattr("agents.classify.structured", FakeStructured(llm_brief()))

        async def page(url):
            return "Title: Ridgeline\n\nRoasted to order."

        monkeypatch.setattr("agents.classify.fetch_page_text", page)

        out = await classify_node(settings)({"request": "https://ridgeline.coffee"})

        assert out["brief"].input_kind == "url"
        assert out["brief"].url == "https://ridgeline.coffee"
        assert out["brief"].request == "https://ridgeline.coffee"

    async def test_page_text_reaches_the_classifier_and_state(self, settings, monkeypatch):
        fake = FakeStructured(llm_brief())
        monkeypatch.setattr("agents.classify.structured", fake)

        async def page(url):
            return "Roasted to order."

        monkeypatch.setattr("agents.classify.fetch_page_text", page)

        out = await classify_node(settings)({"request": "https://ridgeline.coffee"})

        assert "Roasted to order." in fake.calls[0]["user"]
        assert out["source_text"] == "Roasted to order."

    async def test_an_unreadable_page_is_not_fatal(self, settings, monkeypatch):
        """The user's own words and the URL still say a lot."""
        fake = FakeStructured(llm_brief())
        monkeypatch.setattr("agents.classify.structured", fake)

        async def down(url):
            raise WebpageError("HTTP 403")

        monkeypatch.setattr("agents.classify.fetch_page_text", down)

        out = await classify_node(settings)(
            {"request": "a promo for https://ridgeline.coffee"}
        )

        assert out["brief"].intent == "product_promo"
        assert out["source_text"] == ""
        assert "could not be read" in fake.calls[0]["user"]

    async def test_an_empty_request_is_an_error_without_calling_the_model(
        self, settings, monkeypatch
    ):
        fake = FakeStructured(llm_brief())
        monkeypatch.setattr("agents.classify.structured", fake)

        out = await classify_node(settings)({"request": "   "})

        assert "empty request" in out["error"]
        assert fake.calls == []

    async def test_the_prompt_treats_page_content_as_untrusted(self, settings, monkeypatch):
        fake = FakeStructured(llm_brief())
        monkeypatch.setattr("agents.classify.structured", fake)

        await classify_node(settings)({"request": "explain RAG"})

        assert "untrusted" in fake.calls[0]["system"]

    async def test_every_intent_is_described_to_the_classifier(self, settings, monkeypatch):
        fake = FakeStructured(llm_brief())
        monkeypatch.setattr("agents.classify.structured", fake)

        await classify_node(settings)({"request": "explain RAG"})

        for intent in INTENT_GUIDES:
            assert intent in fake.calls[0]["system"]


class TestWriterNode:
    @pytest.fixture
    def fake(self, monkeypatch):
        fake = FakeStructured(make_screenplay())
        monkeypatch.setattr("agents.research_script.structured", fake)
        return fake

    async def test_returns_the_screenplay_and_the_rendered_prompts(
        self, settings, brief, fake
    ):
        out = await writer_node(settings)({"brief": brief, "iterations": 0})

        assert out["iterations"] == 1
        assert len(out["prompts"]) == 2, "one prompt per Veo generation call"
        assert "[00:00-00:04]" in out["prompts"][0]

    async def test_prompts_are_in_the_target_models_format(self, settings, brief, fake):
        cfg = dataclasses.replace(settings, video_model="seedance")
        out = await writer_node(cfg)({"brief": brief})

        assert len(out["prompts"]) == 1
        assert "Shot 1 [0-" in out["prompts"][0]
        assert len(out["screenplay"].segments) == 1

    async def test_the_veo_guide_is_used_for_veo(self, settings, brief, fake):
        await writer_node(settings)({"brief": brief})
        system = fake.calls[0]["system"]
        assert VEO_GUIDE in system
        assert SEEDANCE_GUIDE not in system

    async def test_the_seedance_guide_is_used_for_seedance(self, settings, brief, fake):
        cfg = dataclasses.replace(settings, video_model="seedance")
        await writer_node(cfg)({"brief": brief})
        system = fake.calls[0]["system"]
        assert SEEDANCE_GUIDE in system
        assert VEO_GUIDE not in system

    async def test_the_guide_for_the_classified_intent_is_used(
        self, settings, brief, fake
    ):
        explainer = brief.model_copy(update={"intent": "explainer"})
        await writer_node(settings)({"brief": explainer})

        system = fake.calls[0]["system"]
        assert INTENT_GUIDES["explainer"] in system
        assert INTENT_GUIDES["product_promo"] not in system

    async def test_segment_windows_follow_the_model(self, settings, brief, fake):
        await writer_node(settings)({"brief": brief})
        assert "segment 0: 8 seconds" in fake.calls[0]["system"]
        assert "segment 1: 7 seconds" in fake.calls[0]["system"]

    async def test_the_structure_follows_the_length_of_this_run(self, settings, brief, fake):
        out = await writer_node(settings)({"brief": brief, "duration_seconds": 22})

        system = fake.calls[0]["system"]
        for window in ("segment 0: 8 seconds", "segment 1: 7 seconds", "segment 2: 7 seconds"):
            assert window in system
        assert "exactly 22 seconds long" in system
        assert "22-second video" in fake.calls[0]["user"]
        assert len(out["prompts"]) == 3, "one prompt per Veo generation"

    async def test_the_last_segment_is_partial_when_the_length_does_not_fit(
        self, settings, brief, fake
    ):
        """20s is generated as 8+7+7 and cut: the last 7s clip only keeps 5."""
        out = await writer_node(settings)({"brief": brief, "duration_seconds": 20})

        system = fake.calls[0]["system"]
        assert "segment 2: 5 seconds -> shots must tile 0 to 5" in system
        assert "only the first 5s are kept" in system
        assert "finished by second 5" in system
        last = out["screenplay"].segments[2]
        assert [(s.start, s.end) for s in last.shots][-1][1] == 5, "shots tile the kept 5s"
        assert "00:05]" in out["prompts"][2]

    async def test_a_short_video_is_a_single_base_clip(self, settings, brief, fake):
        out = await writer_node(settings)({"brief": brief, "duration_seconds": 6})

        assert "segment 0: 6 seconds" in fake.calls[0]["system"]
        assert "segment 1" not in fake.calls[0]["system"]
        assert len(out["prompts"]) == 1

    async def test_a_run_that_names_no_length_gets_the_default(self, settings, brief, fake):
        await writer_node(settings)({"brief": brief})

        assert "exactly 15 seconds long" in fake.calls[0]["system"]

    async def test_two_runs_with_different_lengths_do_not_share_a_structure(
        self, settings, brief, fake
    ):
        """The writer is built once; the length must be read per run."""
        writer = writer_node(settings)
        await writer({"brief": brief, "duration_seconds": 8})
        await writer({"brief": brief, "duration_seconds": 29})

        assert "exactly 8 seconds long" in fake.calls[0]["system"]
        assert "exactly 29 seconds long" in fake.calls[1]["system"]

    async def test_seedance_writes_one_segment_of_the_requested_length(
        self, settings, brief, fake
    ):
        cfg = dataclasses.replace(settings, video_model="seedance")
        out = await writer_node(cfg)({"brief": brief, "duration_seconds": 25})

        assert "segment 0: 25 seconds" in fake.calls[0]["system"]
        assert len(out["prompts"]) == 1

    async def test_the_word_budget_is_for_the_seconds_that_are_kept(
        self, settings, brief, fake
    ):
        talking = brief.model_copy(update={"dialogue_led": True})
        await writer_node(settings)({"brief": talking, "duration_seconds": 20})

        system = fake.calls[0]["system"]
        assert f"segment 2 (5s): about {settings.words_for(5)} spoken words" in system

    async def test_dialogue_led_briefs_get_a_word_budget(self, settings, brief, fake):
        talking = brief.model_copy(update={"dialogue_led": True})
        await writer_node(settings)({"brief": talking})

        system = fake.calls[0]["system"]
        assert "WORD BUDGET" in system
        assert f"about {settings.words_for(8)} spoken words" in system

    async def test_dialogue_is_limited_per_shot_not_just_per_segment(
        self, settings, brief, fake
    ):
        """Regression: Sonnet put 13 spoken words into a 3-second shot because
        it was only told a budget for the whole segment."""
        talking = brief.model_copy(update={"dialogue_led": True})
        await writer_node(settings)({"brief": talking})

        system = fake.calls[0]["system"]
        assert "PER-SHOT LIMIT" in system
        assert "2.9 spoken words for every second" in system
        assert f"holds at most {settings.words_ceiling(3)} words" in system

    async def test_visual_led_briefs_are_not_told_to_fill_the_clip_with_speech(
        self, settings, brief, fake
    ):
        await writer_node(settings)({"brief": brief})
        assert "visual-led" in fake.calls[0]["system"]
        assert "WORD BUDGET" not in fake.calls[0]["system"]

    async def test_source_page_is_passed_as_untrusted_data(self, settings, brief, fake):
        await writer_node(settings)(
            {"brief": brief, "source_text": "Roasted to order."}
        )
        user = fake.calls[0]["user"]
        assert "Roasted to order." in user
        assert "untrusted data" in user

    async def test_human_feedback_reaches_the_rewrite(self, settings, brief, fake):
        await writer_node(settings)(
            {
                "brief": brief,
                "screenplay": make_screenplay(),
                "human_feedback": "make the opening funnier",
            }
        )
        user = fake.calls[0]["user"]
        assert "make the opening funnier" in user
        assert "PREVIOUS DRAFT" in user

    async def test_real_names_are_scrubbed_from_the_prompts(self, settings, brief, fake):
        person = brief.model_copy(update={"person_names": ["Barista"]})
        out = await writer_node(settings)({"brief": person})

        assert "Barista" not in out["prompts"][0]
        assert "the speaker" in out["prompts"][0]
