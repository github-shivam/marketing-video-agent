"""Reading uploaded photos, video and audio, and using what was read.

The files are real and ffmpeg and Pillow really process them. Only the call to
the model that reads them is replaced, so nothing here touches Google Cloud.
"""

from __future__ import annotations

import dataclasses
import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from agents.assets import (
    MAX_REFERENCE_IMAGES,
    MAX_TRANSCRIPT_CHARS,
    PreparedUploads,
    UploadsUnreadable,
    analyze_node,
    prepare_uploads,
    tidy,
)
from agents.classify import classify_node
from agents.common import describe_uploads, plan_for_run, uploads_summary
from agents.research_script import critic_node, research_node, writer_node
from graph import ANALYZE, CLASSIFY, Nodes, build_graph, route_start
from state import Asset, AssetBrief, Critique, PersonLook

from conftest import FakeAgents, make_asset_brief, make_screenplay

needs_ffmpeg = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
    reason="needs ffmpeg and ffprobe on PATH",
)


# ── real files ────────────────────────────────────────────────────────────


def make_photo(path: Path, colour=(200, 120, 40), size=(320, 480)) -> Asset:
    Image.new("RGB", size, colour).save(path, "JPEG")
    return Asset(path=str(path), kind="image", mime="image/jpeg")


def make_clip(path: Path, seconds: int = 3) -> Asset:
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", f"color=c=0x306090:s=160x288:r=12:d={seconds}",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(path),
        ],
        check=True,
    )
    return Asset(path=str(path), kind="video", mime="video/mp4")


def make_sound(path: Path, seconds: int = 2) -> Asset:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
         f"sine=frequency=330:duration={seconds}", str(path)],
        check=True,
    )
    return Asset(path=str(path), kind="audio", mime="audio/wav")


class FakeReader:
    """Stands in for the Gemini call. Records what it was asked to read."""

    def __init__(self, analysis=None, error: Exception | None = None):
        self.analysis = analysis or make_asset_brief().analysis
        self.error = error
        self.seen: list[PreparedUploads] = []

    async def __call__(self, settings, prepared):
        self.seen.append(prepared)
        if self.error:
            raise self.error
        return self.analysis


# ── preparing the files ───────────────────────────────────────────────────


@needs_ffmpeg
class TestPrepareUploads:
    async def test_photos_a_video_and_audio_are_all_made_ready(self, tmp_path):
        assets = [
            make_photo(tmp_path / "a.jpg"),
            make_clip(tmp_path / "v.mp4"),
            make_sound(tmp_path / "s.wav"),
        ]
        prepared = await prepare_uploads(assets, tmp_path / "work")

        assert (prepared.n_images, prepared.n_videos, prepared.n_audio) == (1, 1, 1)
        assert prepared.skipped == []
        assert len(prepared.item_files) == 1 + 3, "the photo, then 3 stills from the video"
        assert all(f.suffix == ".jpg" and f.exists() for f in prepared.item_files)

    async def test_everything_the_model_sees_is_numbered_and_labelled(self, tmp_path):
        assets = [make_photo(tmp_path / "a.jpg"), make_clip(tmp_path / "v.mp4"), make_sound(tmp_path / "s.wav")]
        prepared = await prepare_uploads(assets, tmp_path / "work")

        labels = [c for c in prepared.contents if isinstance(c, str)]
        assert labels[0].startswith("Image 0")
        assert any(label.startswith("Video 1") and "with its sound" in label for label in labels)
        assert any("a still from video 1" in label for label in labels)
        assert any(label.startswith("Audio 1") for label in labels)
        assert "Image 3" in " ".join(labels), "the numbers run on across photos and stills"

    async def test_the_colours_are_measured_from_the_pixels(self, tmp_path):
        assets = [make_photo(tmp_path / "a.jpg", colour=(220, 20, 20))]
        prepared = await prepare_uploads(assets, tmp_path / "work")

        (red,) = prepared.palette
        assert int(red[1:3], 16) > 180 and int(red[3:5], 16) < 60

    async def test_the_video_is_cut_down_before_it_is_sent(self, tmp_path):
        clip = make_clip(tmp_path / "v.mp4")
        prepared = await prepare_uploads([clip], tmp_path / "work")

        small = tmp_path / "work" / "video-1.mp4"
        assert small.exists()
        parts = [c for c in prepared.contents if not isinstance(c, str)]
        assert any(p.inline_data.mime_type == "video/mp4" for p in parts)

    async def test_audio_becomes_a_small_wav(self, tmp_path):
        prepared = await prepare_uploads([make_sound(tmp_path / "s.wav")], tmp_path / "work")

        parts = [c for c in prepared.contents if not isinstance(c, str)]
        assert [p.inline_data.mime_type for p in parts] == ["audio/wav"]

    async def test_a_caption_goes_to_the_model_as_something_the_person_wrote(self, tmp_path):
        photo = make_photo(tmp_path / "a.jpg")
        photo = photo.model_copy(update={"caption": "this is my bakery"})
        prepared = await prepare_uploads([photo], tmp_path / "work")

        assert 'the person wrote: "this is my bakery"' in prepared.contents[0]

    async def test_a_file_that_cannot_be_read_is_skipped_and_the_rest_carry_on(self, tmp_path):
        bad = tmp_path / "bad.jpg"
        bad.write_text("not a picture")
        assets = [Asset(path=str(bad), kind="image"), make_photo(tmp_path / "ok.jpg")]

        prepared = await prepare_uploads(assets, tmp_path / "work")

        assert prepared.skipped == ["image"]
        assert prepared.n_images == 1

    async def test_if_nothing_can_be_read_it_says_so(self, tmp_path):
        bad = tmp_path / "bad.jpg"
        bad.write_text("not a picture")
        broken_audio = tmp_path / "bad.ogg"
        broken_audio.write_bytes(b"\x00\x01")

        with pytest.raises(UploadsUnreadable):
            await prepare_uploads(
                [Asset(path=str(bad), kind="image"), Asset(path=str(broken_audio), kind="audio")],
                tmp_path / "work",
            )

    async def test_a_missing_file_is_skipped_not_a_crash(self, tmp_path):
        gone = Asset(path=str(tmp_path / "gone.jpg"), kind="image")
        prepared = await prepare_uploads([gone, make_photo(tmp_path / "ok.jpg")], tmp_path / "work")
        assert prepared.skipped == ["image"] and prepared.n_images == 1


# ── making what the model said safe to act on ─────────────────────────────


class TestTidy:
    def prepared(self, n_items: int) -> PreparedUploads:
        return PreparedUploads(item_files=[Path(f"item-{i}.jpg") for i in range(n_items)])

    def analysis(self, **over):
        return make_asset_brief().analysis.model_copy(update=over)

    def test_reference_photos_that_do_not_exist_are_dropped(self):
        out = tidy(self.analysis(reference_items=[0, 7, -1, 2]), self.prepared(3))
        assert out.reference_items == [0, 2]

    def test_at_most_three_references_go_to_veo(self):
        out = tidy(self.analysis(reference_items=[0, 1, 2, 3, 4]), self.prepared(5))
        assert out.reference_items == [0, 1, 2]
        assert len(out.reference_items) == MAX_REFERENCE_IMAGES

    def test_a_photo_named_twice_is_sent_once(self):
        out = tidy(self.analysis(reference_items=[1, 1, 1]), self.prepared(3))
        assert out.reference_items == [1]

    def test_no_person_means_no_reference_photos_whatever_the_model_picked(self):
        """The model must not be able to send a photo of nobody as 'the person'."""
        out = tidy(self.analysis(people=[], reference_items=[0, 1]), self.prepared(3))
        assert out.reference_items == []

    def test_a_person_seen_in_a_photo_that_does_not_exist_is_trimmed(self):
        person = PersonLook(label="the baker", description="x", item_indexes=[0, 9], is_main=True)
        out = tidy(self.analysis(people=[person]), self.prepared(2))
        assert out.people[0].item_indexes == [0]

    def test_a_long_transcript_is_cut(self):
        long_audio = make_asset_brief().analysis.audio.model_copy(update={"transcript": "la " * 500})
        out = tidy(self.analysis(audio=long_audio), self.prepared(1))
        assert len(out.audio.transcript) == MAX_TRANSCRIPT_CHARS

    def test_what_is_fine_is_left_alone(self):
        before = self.analysis(reference_items=[0])
        assert tidy(before, self.prepared(2)) == before


# ── the analyze node ──────────────────────────────────────────────────────


@needs_ffmpeg
class TestAnalyzeNode:
    @pytest.fixture
    def assets(self, tmp_path):
        folder = tmp_path / "batch"
        folder.mkdir()
        return [make_photo(folder / "a.jpg"), make_photo(folder / "b.jpg", colour=(30, 90, 160))]

    async def test_it_builds_a_brief_from_what_the_model_read(self, settings, assets):
        reader = FakeReader()
        out = await analyze_node(settings, reader)({"assets": assets})

        brief: AssetBrief = out["asset_brief"]
        assert out["asset_error"] is None
        assert brief.analysis.color_words == "warm amber, cream and deep teal"
        assert brief.n_images == 2
        assert brief.palette and all(c.startswith("#") for c in brief.palette)

    async def test_the_palette_is_measured_not_taken_from_the_model(self, settings, assets):
        """The model's analysis has no colour codes; they come from the pixels."""
        out = await analyze_node(settings, FakeReader())({"assets": assets})

        assert out["asset_brief"].palette != make_asset_brief().palette, (
            "not the fixture's made-up codes: these are measured"
        )

    async def test_the_reference_photos_are_files_that_exist(self, settings, assets):
        out = await analyze_node(settings, FakeReader())({"assets": assets})

        (reference,) = out["asset_brief"].reference_files
        assert Path(reference).is_file() and reference.endswith(".jpg")

    async def test_scratch_files_go_inside_the_uploads_folder(self, settings, assets):
        """So they are deleted along with the uploads when the run ends."""
        out = await analyze_node(settings, FakeReader())({"assets": assets})

        folder = Path(assets[0].path).parent
        assert Path(out["asset_brief"].reference_files[0]).parent.parent == folder

    async def test_no_uploads_means_nothing_happens(self, settings):
        reader = FakeReader()
        assert await analyze_node(settings, reader)({}) == {}
        assert await analyze_node(settings, reader)({"assets": []}) == {}
        assert reader.seen == []

    async def test_a_failing_model_does_not_fail_the_run(self, settings, assets):
        reader = FakeReader(error=RuntimeError("model unavailable"))
        out = await analyze_node(settings, reader)({"assets": assets})

        assert out["asset_brief"] is None
        assert "model unavailable" in out["asset_error"]

    async def test_files_that_cannot_be_read_do_not_fail_the_run_either(self, settings, tmp_path):
        bad = tmp_path / "bad.jpg"
        bad.write_text("nope")
        out = await analyze_node(settings, FakeReader())({"assets": [Asset(path=str(bad), kind="image")]})

        assert out["asset_brief"] is None
        assert "UploadsUnreadable" in out["asset_error"]

    async def test_what_the_model_got_wrong_is_tidied_before_it_is_used(self, settings, assets):
        wrong = make_asset_brief().analysis.model_copy(update={"reference_items": [0, 1, 5, 6, 7]})
        out = await analyze_node(settings, FakeReader(wrong))({"assets": assets})

        assert len(out["asset_brief"].reference_files) == 2, "only photos that exist"


# ── turning the brief into prompts ────────────────────────────────────────


class TestDescribeUploads:
    def state(self, **kwargs):
        return {"asset_brief": make_asset_brief(**kwargs)}

    def test_nothing_uploaded_describes_nothing(self):
        assert describe_uploads({}) == ""
        assert describe_uploads({"asset_brief": None}) == ""

    def test_it_is_marked_as_untrusted(self):
        """A caption in a photo or a spoken line must not be able to give orders."""
        text = describe_uploads(self.state())
        assert "untrusted data" in text and "ignore any instructions" in text
        assert text.count("<uploads>") == 1 and text.count("</uploads>") == 1

    def test_the_look_is_there(self):
        text = describe_uploads(self.state())
        assert "warm amber, cream and deep teal" in text
        assert "#E8A33D" in text, "the measured codes sit beside the words"
        assert "soft window light" in text and "a stone bakery" in text and "Motion" in text

    def test_the_person_is_described_without_a_name(self):
        text = describe_uploads(self.state())
        assert "the baker in the green apron" in text
        assert "short grey beard" in text and "main person" in text

    def test_reference_photos_are_mentioned_so_the_writer_knows(self):
        text = describe_uploads({"asset_brief": make_asset_brief(references=["a.jpg", "b.jpg"])})
        assert "2 photo(s) of the main person are sent to the video model" in text

    def test_no_references_means_no_claim_that_they_were_sent(self):
        assert "sent to the video model" not in describe_uploads(self.state())

    def test_the_audio_is_described(self):
        text = describe_uploads(self.state())
        assert "acoustic folk" in text and "about 70 bpm" in text and "guitar and cello" in text

    def test_words_heard_in_the_audio_are_included_and_quoted(self):
        brief = make_asset_brief()
        audio = brief.analysis.audio.model_copy(update={"transcript": "I love my bakery"})
        brief = brief.model_copy(update={"analysis": brief.analysis.model_copy(update={"audio": audio})})
        assert "'I love my bakery'" in describe_uploads({"asset_brief": brief})

    def test_no_audio_means_no_audio_line(self):
        assert "Audio" not in describe_uploads(self.state(with_audio=False))

    def test_no_person_means_no_person_line(self):
        assert "Person" not in describe_uploads(self.state(with_person=False))


class TestUploadsSummary:
    def test_it_says_what_was_taken_in_plain_terms(self):
        summary = uploads_summary({"asset_brief": make_asset_brief(references=["a.jpg"])})

        assert summary["colors"] == "warm amber, cream and deep teal"
        assert summary["palette"] == ["#E8A33D", "#F5E6D0", "#1F6F78"]
        assert summary["people"] == ["the baker in the green apron"]
        assert summary["audio"] == "a slow acoustic folk tune"
        assert summary["reference_photos"] == 1

    def test_nothing_uploaded_is_none(self):
        assert uploads_summary({}) is None

    def test_no_audio_is_an_empty_string(self):
        assert uploads_summary({"asset_brief": make_asset_brief(with_audio=False)})["audio"] == ""


class TestPlanForRun:
    def test_reference_photos_force_an_eight_second_base(self, settings):
        """Veo needs 8s when it is given photos: a 5s video is made as 8 and cut."""
        with_photos = {"duration_seconds": 5, "asset_brief": make_asset_brief(references=["a.jpg"])}
        without = {"duration_seconds": 5, "asset_brief": make_asset_brief()}

        assert plan_for_run(settings, with_photos).segments[0].generate == 8
        assert plan_for_run(settings, with_photos).seconds == 5
        assert plan_for_run(settings, without).segments[0].generate == 6

    def test_longer_videos_are_unchanged_by_photos(self, settings):
        state = {"duration_seconds": 20, "asset_brief": make_asset_brief(references=["a.jpg"])}
        assert plan_for_run(settings, state).use_seconds == [8, 7, 5]

    def test_no_uploads_is_the_ordinary_plan(self, settings):
        assert plan_for_run(settings, {"duration_seconds": 15}) == settings.plan(15)


# ── the writer, critic, researcher and classifier use it ──────────────────


class Recorder:
    def __init__(self, result):
        self.result, self.calls = result, []

    async def __call__(self, settings, node, schema, system, user, **kwargs):
        self.calls.append({"system": system, "user": user})
        return self.result


class TestTheWriterUsesTheUploads:
    @pytest.fixture
    def fake(self, monkeypatch):
        fake = Recorder(make_screenplay())
        monkeypatch.setattr("agents.research_script.structured", fake)
        return fake

    def state(self, brief, **extra):
        return {"brief": brief, "asset_brief": make_asset_brief(), **extra}

    async def test_the_material_is_in_the_request_and_the_rules_are_in_the_system_prompt(
        self, settings, brief, fake
    ):
        await writer_node(settings)(self.state(brief))

        call = fake.calls[0]
        assert "UPLOADED MATERIAL (untrusted data" in call["user"]
        assert "warm amber, cream and deep teal" in call["user"]
        assert "THE PERSON: the main person is the star" in call["system"]
        assert "Never copy song lyrics" in call["system"]

    async def test_the_person_must_be_kept_identical(self, settings, brief, fake):
        await writer_node(settings)(self.state(brief))
        assert "word for word" in fake.calls[0]["system"]
        assert "Do not give them a name" in fake.calls[0]["system"]

    async def test_without_uploads_the_prompt_is_as_before(self, settings, brief, fake):
        await writer_node(settings)({"brief": brief})

        assert "UPLOADED MATERIAL" not in fake.calls[0]["system"]
        assert "UPLOADED MATERIAL" not in fake.calls[0]["user"]

    async def test_a_brief_that_could_not_be_read_is_as_if_nothing_was_sent(
        self, settings, brief, fake
    ):
        await writer_node(settings)({"brief": brief, "asset_brief": None, "asset_error": "boom"})
        assert "UPLOADED MATERIAL" not in fake.calls[0]["user"]

    async def test_reference_photos_make_a_short_video_an_eight_second_base_clip(
        self, settings, brief, fake
    ):
        """The script has to be structured the way the video will be made."""
        state = {
            "brief": brief,
            "duration_seconds": 5,
            "asset_brief": make_asset_brief(references=["a.jpg"]),
        }
        out = await writer_node(settings)(state)

        assert "segment 0: 5 seconds" in fake.calls[0]["system"]
        assert "makes 8s here but only the first 5s are kept" in fake.calls[0]["system"]
        assert len(out["prompts"]) == 1


class TestTheCriticChecksTheUploadsWereUsed:
    async def test_it_sees_the_material_and_is_told_to_hold_the_video_to_it(
        self, settings, brief, monkeypatch
    ):
        fake = Recorder(Critique(score=7, hook="h", intent_fit="i", prompt_craft="p", feedback="f"))
        monkeypatch.setattr("agents.research_script.structured", fake)

        await critic_node(settings)(
            {
                "brief": brief,
                "screenplay": make_screenplay(),
                "prompts": ["p"],
                "asset_brief": make_asset_brief(),
            }
        )

        assert "UPLOADED MATERIAL" in fake.calls[0]["user"]
        assert "does the video really use it" in fake.calls[0]["system"]

    async def test_without_uploads_there_is_nothing_extra(self, settings, brief, monkeypatch):
        fake = Recorder(Critique(score=7, hook="h", intent_fit="i", prompt_craft="p", feedback="f"))
        monkeypatch.setattr("agents.research_script.structured", fake)

        await critic_node(settings)({"brief": brief, "screenplay": make_screenplay(), "prompts": ["p"]})

        assert "UPLOADED MATERIAL" not in fake.calls[0]["user"]


class TestTheClassifierAndResearcherSeeThem:
    async def test_the_classifier_is_given_the_material(self, settings, brief, monkeypatch):
        fake = Recorder(brief)
        monkeypatch.setattr("agents.classify.structured", fake)

        await classify_node(settings)({"request": "an ad", "asset_brief": make_asset_brief()})

        assert "UPLOADED MATERIAL" in fake.calls[0]["user"]
        assert "untrusted data too" in fake.calls[0]["system"]

    async def test_the_researcher_is_given_the_material(self, settings, brief, monkeypatch):
        seen = {}

        class Model:
            async def ainvoke(self, messages):
                seen["user"] = messages[1]["content"]
                return type("R", (), {"content": "notes"})()

        monkeypatch.setattr("agents.research_script.get_model", lambda *a, **k: Model())
        monkeypatch.setattr("agents.research_script.load_tools", _no_tools)

        await research_node(settings)({"brief": brief, "asset_brief": make_asset_brief()})

        assert "UPLOADED MATERIAL" in seen["user"]


async def _no_tools(settings, server):
    return []


# ── the graph ─────────────────────────────────────────────────────────────


class TestTheGraph:
    def test_uploads_are_read_first(self):
        assert route_start({"assets": [Asset(path="a.jpg", kind="image")]}) == ANALYZE

    @pytest.mark.parametrize("state", [{}, {"assets": []}])
    def test_no_uploads_goes_straight_to_classifying(self, state):
        assert route_start(state) == CLASSIFY

    async def test_the_analyze_step_runs_only_when_there_are_uploads(self, settings):
        from langgraph.checkpoint.memory import InMemorySaver

        agents = FakeAgents()
        app = build_graph(agents.as_nodes(), settings, checkpointer=InMemorySaver())

        await app.ainvoke({"request": "x", "iterations": 0}, {"configurable": {"thread_id": "plain"}})
        assert "analyze" not in agents.calls

        await app.ainvoke(
            {"request": "x", "iterations": 0, "assets": [Asset(path="a.jpg", kind="image")]},
            {"configurable": {"thread_id": "with-uploads"}},
        )
        assert agents.calls.index("analyze") < agents.calls.index("classify", agents.calls.index("analyze"))
        assert len(agents.analyzed) == 1

    async def test_the_review_shows_what_was_taken_from_the_uploads(self, settings):
        from langgraph.checkpoint.memory import InMemorySaver

        app = build_graph(FakeAgents().as_nodes(), settings, checkpointer=InMemorySaver())
        result = await app.ainvoke(
            {"request": "x", "iterations": 0, "assets": [Asset(path="a.jpg", kind="image")]},
            {"configurable": {"thread_id": "review"}},
        )

        gate = result["__interrupt__"][0].value
        assert gate["uploads"]["colors"] == "warm amber, cream and deep teal"
        assert gate["uploads"]["people"] == ["the baker in the green apron"]
        assert gate["uploads_error"] is None

    async def test_a_failed_reading_is_reported_at_the_review_and_the_run_goes_on(self, settings):
        from langgraph.checkpoint.memory import InMemorySaver

        agents = FakeAgents()
        agents.analysis_error = "model unavailable"
        app = build_graph(agents.as_nodes(), settings, checkpointer=InMemorySaver())
        result = await app.ainvoke(
            {"request": "x", "iterations": 0, "assets": [Asset(path="a.jpg", kind="image")]},
            {"configurable": {"thread_id": "failed"}},
        )

        gate = result["__interrupt__"][0].value
        assert gate["uploads"] is None
        assert gate["uploads_error"] == "model unavailable"
        assert gate["prompts"], "a script was still written, from the request alone"

    async def test_a_graph_without_an_analyze_node_ignores_uploads(self, settings):
        from langgraph.checkpoint.memory import InMemorySaver

        agents = FakeAgents()
        nodes = dataclasses.replace(agents.as_nodes(), analyze=None)
        app = build_graph(nodes, settings, checkpointer=InMemorySaver())

        result = await app.ainvoke(
            {"request": "x", "iterations": 0, "assets": [Asset(path="a.jpg", kind="image")]},
            {"configurable": {"thread_id": "none"}},
        )

        assert "analyze" not in agents.calls
        assert result["__interrupt__"][0].value["stage"] == "script"

    async def test_the_uploads_survive_a_checkpoint(self, settings, tmp_path):
        """The new state models must round-trip through SQLite, not just memory."""
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        from graph import make_serializer

        config = {"configurable": {"thread_id": "persist"}}
        db = str(tmp_path / "checkpoints.sqlite")
        async with AsyncSqliteSaver.from_conn_string(db) as saver:
            saver.serde = make_serializer()
            app = build_graph(FakeAgents().as_nodes(), settings, checkpointer=saver)
            await app.ainvoke(
                {"request": "x", "iterations": 0, "assets": [Asset(path="a.jpg", kind="image")]}, config
            )

        async with AsyncSqliteSaver.from_conn_string(db) as saver:
            saver.serde = make_serializer()
            app = build_graph(FakeAgents().as_nodes(), settings, checkpointer=saver)
            values = (await app.aget_state(config)).values

        assert values["asset_brief"].analysis.color_words == "warm amber, cream and deep teal"
        assert values["assets"][0].path == "a.jpg"
