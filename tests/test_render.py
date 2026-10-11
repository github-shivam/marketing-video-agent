"""Tests for turning a screenplay into each video model's prompt syntax.

These pin the documented formats: Veo's timestamped shots with ``SFX:`` and
``Ambient noise:`` labels, and Seedance's ``Shot N [a-bs]`` labels with
``<sfx>``, ``{dialogue}`` and ``(music)`` brackets.
"""

from __future__ import annotations

import dataclasses

from agents.render import (
    candidate_names,
    render_prompts,
    render_seedance,
    render_veo,
    scrub_personal_names,
)
from agents.research_script import fit_screenplay, retime
from state import Brief, Screenplay, Segment, Shot

from conftest import make_screenplay, make_shots


def talking_shot(**overrides) -> Shot:
    base = {
        "start": 0,
        "end": 4,
        "camera": "Medium close-up, static",
        "subject": "a presenter in a navy shirt",
        "action": "looks into the lens",
        "speaker": "The presenter",
        "delivery": "warmly",
        "dialogue": "Here is one idea.",
    }
    return Shot(**{**base, **overrides})


def one_shot_play(shot: Shot, seconds: int = 8) -> Screenplay:
    shot = shot.model_copy(update={"start": 0, "end": seconds})
    return Screenplay(
        title="t",
        logline="l",
        hook="h",
        style="warm documentary",
        continuity="A presenter in a navy shirt.",
        segments=[Segment(index=0, shots=[shot])],
    )


class TestVeoRendering:
    def test_one_prompt_per_generation_call(self, settings):
        prompts = render_veo(make_screenplay(segments=2), settings, [])
        assert len(prompts) == 2, "8s base + 7s extension are separate calls"

    def test_shots_carry_veo_timestamps(self, settings):
        prompt = render_veo(make_screenplay(), settings, [])[0]
        assert "[00:00-00:04]" in prompt
        assert "[00:04-00:08]" in prompt

    def test_extension_segments_continue_the_scene(self, settings):
        base, extension = render_veo(make_screenplay(segments=2), settings, [])
        assert "Continue the same continuous shot" not in base
        assert "Continue the same continuous shot" in extension

    def test_extension_timestamps_restart_at_zero(self, settings):
        """Each extension is its own 7s call, so its clock starts again."""
        extension = render_veo(make_screenplay(segments=2), settings, [])[1]
        assert "[00:00-00:03]" in extension
        assert "[00:03-00:07]" in extension

    def test_dialogue_is_quoted_with_a_speaker(self, settings):
        """Google: put speech in quotes, attributed to someone."""
        prompt = render_veo(one_shot_play(talking_shot()), settings, [])[0]
        assert 'The presenter says warmly, "Here is one idea."' in prompt

    def test_dialogue_without_a_speaker_gets_a_default_one(self, settings):
        """A bare quote makes Veo move the mouth and invent the words."""
        prompt = render_veo(
            one_shot_play(talking_shot(speaker="", delivery="")), settings, []
        )[0]
        assert 'The speaker says, "Here is one idea."' in prompt

    def test_a_lowercase_speaker_starts_the_sentence_capitalised(self, settings):
        prompt = render_veo(
            one_shot_play(talking_shot(speaker="the host")), settings, []
        )[0]
        assert 'The host says warmly, "Here is one idea."' in prompt

    def test_audio_uses_the_documented_labels(self, settings):
        shot = talking_shot(
            dialogue="", sfx="a kettle clicks off", ambient="quiet kitchen hum"
        )
        prompt = render_veo(one_shot_play(shot), settings, [])[0]
        assert "SFX: a kettle clicks off." in prompt
        assert "Ambient noise: quiet kitchen hum." in prompt

    def test_empty_audio_fields_are_omitted(self, settings):
        shot = talking_shot(dialogue="")
        prompt = render_veo(one_shot_play(shot), settings, [])[0]
        assert "SFX:" not in prompt
        assert "Ambient noise:" not in prompt
        assert "Music:" not in prompt

    def test_cinematography_comes_before_the_subject(self, settings):
        """Google's formula starts each shot with the cinematography."""
        prompt = render_veo(one_shot_play(talking_shot()), settings, [])[0]
        shot_line = prompt.split("[00:00-00:08]")[1]
        assert shot_line.index("Medium close-up") < shot_line.index("A presenter in a navy shirt")

    def test_timestamps_can_be_switched_off(self, settings):
        cfg = dataclasses.replace(settings, veo_timestamps=False)
        prompt = render_veo(make_screenplay(), cfg, [])[0]
        assert "[00:" not in prompt
        assert "Shot 1:" in prompt

    def test_non_english_speech_names_the_language(self, settings):
        cfg = dataclasses.replace(settings, script_language="Hindi")
        prompt = render_veo(one_shot_play(talking_shot()), cfg, [])[0]
        assert 'says warmly in Hindi, "Here is one idea."' in prompt

    def test_overlong_prompt_warns(self, settings, caplog):
        shot = talking_shot(action=" ".join(["word"] * 800))
        with caplog.at_level("WARNING"):
            render_veo(one_shot_play(shot), settings, [])
        assert "1,024 tokens" in caplog.text


class TestSubjectActionJoin:
    """Regression: a model wrote a whole sentence into `subject`, leaving
    "...sits beside her. slaps the book" in the prompt."""

    def prompt(self, settings, model, **kw):
        cfg = dataclasses.replace(settings, video_model=model)
        shot = talking_shot(dialogue="", **kw)
        return render_prompts(one_shot_play(shot), cfg)[0]

    def test_a_trailing_full_stop_on_the_subject_does_not_split_the_sentence(self, settings):
        for model in ("veo", "seedance"):
            prompt = self.prompt(
                settings, model,
                subject="A woman at a desk. A book sits open beside her.",
                action="slaps the book with her palm",
            )
            assert "beside her slaps the book" in prompt
            assert ". slaps" not in prompt

    def test_a_normal_noun_phrase_is_unchanged(self, settings):
        prompt = self.prompt(settings, "veo", subject="the woman in blue", action="waves")
        assert "The woman in blue waves." in prompt


class TestSeedanceRendering:
    @staticmethod
    def seedance(settings):
        return dataclasses.replace(settings, video_model="seedance")

    def test_whole_clip_is_one_prompt(self, settings):
        prompts = render_seedance(make_screenplay(segments=1), self.seedance(settings), [])
        assert len(prompts) == 1

    def test_shots_are_labelled_with_second_ranges(self, settings):
        prompt = render_seedance(make_screenplay(segments=1), self.seedance(settings), [])[0]
        assert "Shot 1 [0-4s]:" in prompt
        assert "Shot 2 [4-8s]:" in prompt

    def test_bracket_syntax_for_each_audio_kind(self, settings):
        """Seedance: <> sound effects, {} dialogue, () music."""
        shot = talking_shot(
            sfx="a chair scrapes", ambient="", music="soft piano", dialogue="Hello."
        )
        prompt = render_seedance(one_shot_play(shot), self.seedance(settings), [])[0]
        assert "<a chair scrapes>" in prompt
        assert "(soft piano)" in prompt
        assert "The presenter says warmly: {Hello.}" in prompt

    def test_exclusions_close_the_prompt(self, settings):
        prompt = render_seedance(make_screenplay(segments=1), self.seedance(settings), [])[0]
        assert prompt.rstrip().endswith("No BGM, no music, no score.")
        assert "No subtitles." in prompt
        assert "No watermarks or logos." in prompt

    def test_bgm_exclusion_is_dropped_when_music_is_wanted(self, settings):
        prompt = render_seedance(
            one_shot_play(talking_shot(music="ukulele")), self.seedance(settings), []
        )[0]
        assert "No BGM" not in prompt
        assert "No subtitles." in prompt

    def test_non_english_speech_names_the_language(self, settings):
        cfg = dataclasses.replace(self.seedance(settings), script_language="Hindi")
        prompt = render_seedance(one_shot_play(talking_shot()), cfg, [])[0]
        assert "says warmly in Hindi: {Here is one idea.}" in prompt


class TestEntryPoint:
    def test_veo_is_the_default_target(self, settings):
        assert len(render_prompts(make_screenplay(segments=2), settings)) == 2

    def test_seedance_setting_selects_the_seedance_format(self, settings):
        cfg = dataclasses.replace(settings, video_model="seedance")
        prompts = render_prompts(make_screenplay(segments=1), cfg)
        assert "Shot 1 [0-4s]:" in prompts[0]


class TestPersonalNameScrubbing:
    """Veo rejects prompts naming real people; names must not reach it."""

    def brief(self, *names: str) -> Brief:
        return Brief(
            intent="explainer",
            subject="s",
            audience="a",
            tone="t",
            key_message="k",
            visual_style="v",
            person_names=list(names),
        )

    def test_names_are_expanded_to_their_words(self):
        assert candidate_names(self.brief("Jane Doe")) == ["Jane Doe", "Jane", "Doe"]

    def test_short_fragments_are_not_scrubbed(self):
        """A two-letter name part must not mangle ordinary words."""
        assert candidate_names(self.brief("Al Smith")) == ["Al Smith", "Smith"]

    def test_no_names_means_no_scrubbing(self):
        assert candidate_names(self.brief()) == []
        assert candidate_names(None) == []

    def test_longest_name_replaced_first(self):
        """'Jane Doe' must not become 'the speaker the speaker'."""
        out = scrub_personal_names("Jane Doe walks", ["Jane Doe", "Jane", "Doe"])
        assert out == "the speaker walks"

    def test_visual_fields_are_scrubbed_in_both_formats(self, settings):
        shot = talking_shot(subject="Jane Doe", action="waves", dialogue="Hi.")
        play = one_shot_play(shot)
        brief = self.brief("Jane Doe")
        for model in ("veo", "seedance"):
            cfg = dataclasses.replace(settings, video_model=model)
            prompt = render_prompts(play, cfg, brief)[0]
            assert "Jane" not in prompt
            assert "the speaker" in prompt.lower()

    def test_spoken_dialogue_keeps_the_name(self, settings):
        """Replacing a name in speech gives nonsense like 'Hi, the speaker here.'"""
        shot = talking_shot(dialogue="Hi, Jane here.")
        prompt = render_prompts(one_shot_play(shot), settings, self.brief("Jane Doe"))[0]
        assert '"Hi, Jane here."' in prompt

    def test_global_style_and_continuity_are_scrubbed(self, settings):
        play = one_shot_play(talking_shot()).model_copy(
            update={"continuity": "Jane Doe, a presenter in a navy shirt."}
        )
        prompt = render_prompts(play, settings, self.brief("Jane Doe"))[0]
        assert "Jane" not in prompt


class TestRetime:
    """Models leave gaps, overlap shots and overshoot; the clock must be fixed."""

    def shots(self, spans):
        return [
            Shot(start=a, end=b, camera="c", subject="s", action="a") for a, b in spans
        ]

    def tiles(self, shots, total):
        clock = 0
        for s in shots:
            assert s.start == clock and s.end > s.start
            clock = s.end
        return clock == total

    def test_a_good_timeline_is_unchanged(self):
        out = retime(self.shots([(0, 3), (3, 8)]), 8)
        assert [(s.start, s.end) for s in out] == [(0, 3), (3, 8)]

    def test_gaps_are_closed(self):
        out = retime(self.shots([(0, 2), (5, 7)]), 8)
        assert self.tiles(out, 8)

    def test_overlaps_are_resolved(self):
        out = retime(self.shots([(0, 5), (3, 8)]), 8)
        assert self.tiles(out, 8)

    def test_overshoot_is_scaled_down(self):
        out = retime(self.shots([(0, 6), (6, 12)]), 8)
        assert self.tiles(out, 8)
        assert out[-1].end == 8

    def test_undershoot_is_scaled_up(self):
        out = retime(self.shots([(0, 2), (2, 4)]), 8)
        assert self.tiles(out, 8)

    def test_relative_length_is_preserved(self):
        out = retime(self.shots([(0, 1), (1, 4)]), 8)
        assert (out[1].end - out[1].start) > (out[0].end - out[0].start)

    def test_more_shots_than_seconds_are_dropped(self):
        spans = [(i, i + 1) for i in range(10)]
        out = retime(self.shots(spans), 7)
        assert len(out) == 7
        assert self.tiles(out, 7)

    def test_zero_length_shots_get_a_second(self):
        out = retime(self.shots([(0, 0), (0, 0), (0, 0)]), 6)
        assert self.tiles(out, 6)

    def test_other_fields_survive(self):
        shots = self.shots([(0, 8)])
        shots[0].dialogue = "keep me"
        assert retime(shots, 8)[0].dialogue == "keep me"


class TestFitScreenplay:
    def test_truncates_extra_segments(self):
        fitted = fit_screenplay(make_screenplay(segments=3), [8, 7])
        assert len(fitted.segments) == 2

    def test_repeats_the_last_segment_when_short(self):
        fitted = fit_screenplay(make_screenplay(segments=1), [8, 7])
        assert [s.index for s in fitted.segments] == [0, 1]

    def test_each_segment_is_retimed_to_its_own_length(self):
        fitted = fit_screenplay(make_screenplay(segments=2), [8, 7])
        assert [seg.shots[-1].end for seg in fitted.segments] == [8, 7]

    def test_seedance_collapses_to_a_single_long_segment(self):
        fitted = fit_screenplay(make_screenplay(segments=2), [15])
        assert len(fitted.segments) == 1
        assert fitted.segments[0].shots[-1].end == 15

    def test_make_shots_helper_tiles(self):
        shots = make_shots(0, 8)
        assert shots[0].start == 0 and shots[-1].end == 8
