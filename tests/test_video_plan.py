"""The rules that turn "make it 20 seconds" into Veo generations."""

from __future__ import annotations

import pytest

from video_plan import (
    MAX_SECONDS,
    MIN_SECONDS,
    SegmentPlan,
    parse_seconds,
    plan_for,
    resolve_seconds,
)


class TestPlanForVeo:
    @pytest.mark.parametrize(
        ("asked", "generated", "kept"),
        [
            (4, [4], [4]),
            (5, [6], [5]),
            (6, [6], [6]),
            (7, [8], [7]),
            (8, [8], [8]),
            (10, [8, 7], [8, 2]),
            (15, [8, 7], [8, 7]),
            (20, [8, 7, 7], [8, 7, 5]),
            (22, [8, 7, 7], [8, 7, 7]),
            (25, [8, 7, 7, 7], [8, 7, 7, 3]),
            (29, [8, 7, 7, 7], [8, 7, 7, 7]),
        ],
    )
    def test_the_generations_that_make_each_length(self, asked, generated, kept):
        plan = plan_for(asked)
        assert [s.generate for s in plan.segments] == generated
        assert plan.use_seconds == kept
        assert plan.seconds == asked, "the video is exactly as long as asked"

    @pytest.mark.parametrize(("asked", "made"), [(9, 8), (16, 15), (23, 22), (30, 29)])
    def test_a_last_extension_that_would_keep_one_second_is_left_out(self, asked, made):
        """Paying for 7s to keep 1s is not worth it: end one second early."""
        plan = plan_for(asked)
        assert plan.seconds == made
        assert asked - plan.seconds == 1
        assert not plan.needs_trim, "nothing is generated only to be thrown away"

    def test_a_video_is_never_longer_than_asked(self):
        for asked in range(MIN_SECONDS, MAX_SECONDS + 1):
            assert plan_for(asked).seconds <= asked

    def test_it_is_never_more_than_a_second_short(self):
        for asked in range(MIN_SECONDS, MAX_SECONDS + 1):
            assert asked - plan_for(asked).seconds <= 1

    def test_the_default_is_a_base_clip_and_one_extension(self):
        plan = plan_for(15)
        assert plan.segments == (SegmentPlan(8, 8), SegmentPlan(7, 7))
        assert plan.extensions == 1
        assert plan.generated_seconds == 15

    def test_more_is_generated_than_kept_only_for_a_partial_last_segment(self):
        assert plan_for(20).needs_trim
        assert plan_for(20).generated_seconds == 22
        assert not plan_for(22).needs_trim

    def test_only_the_last_segment_is_ever_partial(self):
        for asked in range(MIN_SECONDS, MAX_SECONDS + 1):
            *earlier, _ = plan_for(asked).segments
            assert all(s.generate == s.use for s in earlier)

    def test_every_extension_is_seven_seconds(self):
        for asked in range(9, MAX_SECONDS + 1):
            assert all(s.generate == 7 for s in plan_for(asked).segments[1:])

    def test_extending_always_starts_from_an_eight_second_base(self):
        for asked in range(9, MAX_SECONDS + 1):
            assert plan_for(asked).segments[0].generate == 8

    def test_out_of_range_lengths_are_clamped_not_rejected(self):
        assert plan_for(1).seconds == MIN_SECONDS
        assert plan_for(500).seconds == plan_for(MAX_SECONDS).seconds

    def test_the_longest_video_is_four_generations(self):
        assert len(plan_for(MAX_SECONDS).segments) == 4


class TestPlanForSeedance:
    def test_it_is_one_pass_of_the_requested_length(self):
        plan = plan_for(27, "seedance")
        assert plan.segments == (SegmentPlan(27, 27),)
        assert plan.seconds == 27 and not plan.needs_trim

    def test_it_is_clamped_like_veo(self):
        assert plan_for(99, "seedance").seconds == MAX_SECONDS


class TestParseSeconds:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("make a 20 second video about RAG", 20),
            ("explain RAG in 30 seconds", 30),
            ("a 12 sec reel", 12),
            ("15s teaser for my shop", 15),
            ("a 25-second explainer", 25),
            ("make it 10 secs long", 10),
            ("video of 8 Seconds please", 8),
            ("a 1.5 second clip", 2),
            ("20 सेकंड का वीडियो बनाओ", 20),
            ("1 minute video", 60),
            ("2 minutes", 120),
            ("a half minute ad", 30),
            ("make a minute long video", 60),
            ("https://example.com make it 20 seconds", 20),
        ],
    )
    def test_reads_a_stated_length(self, text, expected):
        assert parse_seconds(text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "explain how RAG works",
            "5 steps to better coffee",
            "top 10 stages of a startup",
            "a section about v3s and 2.5.1s",
            "https://example.com/page",
            "make it 0 seconds",
            "",
        ],
    )
    def test_says_nothing_when_no_length_is_stated(self, text):
        assert parse_seconds(text) is None

    def test_the_first_length_wins(self):
        assert parse_seconds("a 20 second video, not 30 seconds") == 20

    def test_seconds_beat_minutes_when_both_appear(self):
        assert parse_seconds("30 seconds, not 2 minutes") == 30


class TestResolveSeconds:
    def test_no_request_uses_the_default_silently(self):
        assert resolve_seconds(None, 15) == (15, "")

    def test_a_request_in_range_is_kept(self):
        assert resolve_seconds(22, 15) == (22, "")

    def test_the_limits_themselves_are_fine(self):
        assert resolve_seconds(MIN_SECONDS, 15) == (MIN_SECONDS, "")
        assert resolve_seconds(MAX_SECONDS, 15) == (MAX_SECONDS, "")

    def test_too_long_is_clamped_and_the_person_is_told(self):
        seconds, note = resolve_seconds(90, 15)
        assert seconds == MAX_SECONDS
        assert "30 seconds" in note and "longest" in note

    def test_too_short_is_clamped_and_the_person_is_told(self):
        seconds, note = resolve_seconds(2, 15)
        assert seconds == MIN_SECONDS
        assert "shortest" in note

    def test_a_bad_default_is_clamped_too(self):
        assert resolve_seconds(None, 600) == (MAX_SECONDS, "")
