"""Tests for the state models' validation and the request parsing.

The distinction being enforced here: presentational constraints coerce (so a
run is never abandoned over a cosmetic detail), while constraints that would
produce a broken post still reject.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from state import Brief, Dialogue, Media, Screenplay, Segment, Shot, extract_url


def shot(**overrides) -> Shot:
    """A valid shot with optional field overrides."""
    base = {
        "start": 0,
        "end": 4,
        "camera": "Close-up",
        "subject": "a bottle",
        "action": "rotates slowly",
    }
    return Shot(**{**base, **overrides})


class TestExtractUrl:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("https://example.com/product", "https://example.com/product"),
            ("http://example.com", "http://example.com"),
            ("www.example.com", "https://www.example.com"),
            ("example.com", "https://example.com"),
            ("shop.example.co.uk/item?id=4", "https://shop.example.co.uk/item?id=4"),
        ],
    )
    def test_recognises_links(self, text, expected):
        assert extract_url(text) == expected

    def test_finds_a_link_inside_a_sentence(self):
        assert (
            extract_url("make a promo for https://example.com/shop for students")
            == "https://example.com/shop"
        )

    def test_trailing_punctuation_is_not_part_of_the_link(self):
        assert extract_url("see https://example.com/a, then decide.") == (
            "https://example.com/a"
        )

    @pytest.mark.parametrize(
        "text",
        [
            "explain how RAG works",
            "make a video about i.e. something",
            "what is the best way to learn python",
            "",
        ],
    )
    def test_plain_topics_are_not_links(self, text):
        assert extract_url(text) is None

    def test_a_bare_domain_inside_a_sentence_is_not_guessed(self):
        """Only a whole-input bare domain counts; prose with dots does not."""
        assert extract_url("I love python. Teach it") is None


class TestShot:
    def test_fractional_seconds_are_rounded_not_rejected(self):
        """Models emit 2.0 or 2.5; a run must not die over it."""
        s = shot(start=0.0, end=2.4)
        assert (s.start, s.end) == (0, 2)

    def test_spoken_words_counts_only_the_dialogue(self):
        assert shot(dialogue="Hello there friend", sfx="door slam").spoken_words() == 3

    def test_no_dialogue_counts_zero(self):
        assert shot().spoken_words() == 0


class TestScreenplay:
    def _segment(self, index: int = 0) -> Segment:
        return Segment(index=index, shots=[shot()])

    def test_segment_requires_at_least_one_shot(self):
        with pytest.raises(ValidationError):
            Segment(index=0, shots=[])

    def test_requires_at_least_one_segment(self):
        with pytest.raises(ValidationError):
            Screenplay(
                title="t", logline="l", hook="h", style="s", continuity="c", segments=[]
            )

    def test_accepts_the_veo_maximum_of_21_segments(self):
        """8s base plus 20 extensions is Veo's documented ceiling."""
        segments = [self._segment(i) for i in range(21)]
        play = Screenplay(
            title="t", logline="l", hook="h", style="s", continuity="c", segments=segments
        )
        assert len(play.segments) == 21

    def test_segment_sums_spoken_words_across_shots(self):
        seg = Segment(
            index=0, shots=[shot(dialogue="one two"), shot(dialogue="three four five")]
        )
        assert seg.spoken_words() == 5


class TestBrief:
    def test_code_owned_fields_have_safe_defaults(self):
        b = Brief(
            intent="explainer",
            subject="s",
            audience="a",
            tone="t",
            key_message="k",
            visual_style="v",
        )
        assert b.input_kind == "topic"
        assert b.url is None
        assert b.dialogue_led is False
        assert b.person_names == []

    def test_unknown_intent_is_rejected(self):
        with pytest.raises(ValidationError):
            Brief(
                intent="vlog",
                subject="s",
                audience="a",
                tone="t",
                key_message="k",
                visual_style="v",
            )


class TestDialogue:
    def test_hashtags_are_normalised_to_a_single_hash(self):
        d = Dialogue(
            caption="x", hashtags=["coffee", "#roastery", "##double", "brew", "beans"]
        )
        assert d.hashtags == ["#coffee", "#roastery", "#double", "#brew", "#beans"]

    def test_too_few_hashtags_rejected(self):
        """Fewer than five would ship a weak post, so this one does reject."""
        with pytest.raises(ValidationError):
            Dialogue(caption="x", hashtags=["one", "two"])

    def test_too_many_hashtags_rejected(self):
        with pytest.raises(ValidationError):
            Dialogue(caption="x", hashtags=[f"tag{i}" for i in range(13)])

    def test_overlong_caption_rejected(self):
        with pytest.raises(ValidationError, match="150 words"):
            Dialogue(caption=" ".join(["word"] * 151), hashtags=["a"] * 5)


class TestMedia:
    def test_defaults_to_reels(self):
        assert Media().media_type == "REELS"
