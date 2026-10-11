"""Tests for the video node, provider selection and the Veo provider config.

The Veo provider is never actually invoked against Google here; only the
request configuration it would send is asserted, because that is where the
documented constraints live (720p-only extensions, an int duration, and
``allow_all`` for text-to-video).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from agents.publisher import build_caption, publish_node, select_video_provider, video_node
from providers.base import VideoProvider
from providers.stub import StubVideoProvider
from providers.veo import VeoError, VeoVideoProvider

from conftest import make_dialogue

PROMPTS = ["first prompt", "second prompt"]


async def _trim_that_works(path: Path, seconds: int) -> bool:
    return True


class RecordingProvider:
    """A provider that records its prompts and writes an empty file."""

    name = "recording"

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.plan = None

    async def generate(self, prompts: list[str], out_dir: Path, plan=None) -> Path:
        self.prompts = list(prompts)
        self.plan = plan
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "recorded.mp4"
        path.touch()
        return path


class TestVeoProviderConfig:
    def test_requires_a_project_and_a_bucket(self, settings):
        with pytest.raises(VeoError, match="GOOGLE_CLOUD_PROJECT"):
            VeoVideoProvider(settings)

    @pytest.fixture
    def provider(self, live):
        return VeoVideoProvider(live)

    def test_satisfies_the_provider_protocol(self, provider):
        assert isinstance(provider, VideoProvider)

    def test_base_config_matches_the_documented_constraints(self, provider):
        config = provider._base_config()
        assert config.aspect_ratio == "9:16"
        assert config.resolution == "720p"
        assert config.number_of_videos == 1
        # Text-to-video accepts allow_all only; allow_adult is image-to-video.
        assert config.person_generation == "allow_all"

    def test_duration_is_an_int_not_a_string(self, provider):
        """The REST API documents a string enum, but the SDK wants an int."""
        assert provider._base_config().duration_seconds == 8
        assert isinstance(provider._base_config().duration_seconds, int)

    def test_extension_config_is_always_720p_and_7s(self, provider):
        """Extension is 720p-only, and Vertex only accepts a 7-second extension."""
        config = provider._extension_config()
        assert config.resolution == "720p"
        assert config.duration_seconds == 7

    async def test_empty_prompts_raise(self, provider, tmp_path):
        with pytest.raises(VeoError, match="at least one prompt"):
            await provider.generate([], tmp_path)


class TestProviderSelection:
    def test_falls_back_to_the_stub_when_veo_is_not_configured(self, settings):
        assert isinstance(select_video_provider(settings), StubVideoProvider)

    def test_uses_veo_when_a_project_and_bucket_are_set(self, live):
        assert isinstance(select_video_provider(live), VeoVideoProvider)

    def test_seedance_uses_the_stub_until_its_api_is_wired(self, live, caplog):
        """A configured Veo must not receive a Seedance-format prompt."""
        cfg = dataclasses.replace(live, video_model="seedance")
        with caplog.at_level("WARNING"):
            provider = select_video_provider(cfg)

        assert isinstance(provider, StubVideoProvider)
        assert "not wired in yet" in caplog.text


class TestDurations:
    def test_the_default_is_fifteen_seconds(self, settings):
        assert settings.video_seconds_default == 15
        assert settings.plan().seconds == 15
        assert settings.plan().use_seconds == [8, 7]

    def test_a_requested_length_overrides_the_default(self, settings):
        assert settings.plan(22).use_seconds == [8, 7, 7]

    def test_seedance_is_a_single_pass(self, settings):
        cfg = dataclasses.replace(settings, video_model="seedance")
        plan = cfg.plan(20)
        assert plan.use_seconds == [20] and plan.seconds == 20

    def test_the_default_is_clamped_into_range(self, monkeypatch):
        from config import Settings

        monkeypatch.setenv("VIDEO_SECONDS_DEFAULT", "999")
        assert Settings.from_env().video_seconds_default == 30
        monkeypatch.setenv("VIDEO_SECONDS_DEFAULT", "1")
        assert Settings.from_env().video_seconds_default == 4


class TestVideoNode:
    async def test_sends_the_rendered_prompts_unchanged(self, settings, tmp_path):
        """What the human approved at the gate is what the model receives."""
        provider = RecordingProvider()
        node = video_node(dataclasses.replace(settings, output_dir=tmp_path), provider)

        await node({"prompts": PROMPTS})

        assert provider.prompts == PROMPTS

    async def test_reports_the_configured_duration(self, settings, tmp_path):
        node = video_node(
            dataclasses.replace(settings, output_dir=tmp_path), RecordingProvider()
        )
        result = await node({"prompts": PROMPTS})

        assert result["media"].media_type == "REELS"
        assert result["media"].duration_seconds == 15, "8s base + one 7s extension"

    async def test_the_length_comes_from_the_run_not_the_settings(
        self, settings, tmp_path, monkeypatch
    ):
        """Each chat can ask for its own length."""
        monkeypatch.setattr("agents.publisher.trim_video", _trim_that_works)
        provider = RecordingProvider()
        node = video_node(dataclasses.replace(settings, output_dir=tmp_path), provider)

        result = await node({"prompts": PROMPTS, "duration_seconds": 22})

        assert provider.plan.seconds == 22
        assert provider.plan.use_seconds == [8, 7, 7]
        assert result["media"].duration_seconds == 22

    async def test_a_video_generated_longer_than_asked_is_trimmed_to_the_ask(
        self, settings, tmp_path, monkeypatch
    ):
        trimmed: list[tuple[Path, int]] = []

        async def fake_trim(path: Path, seconds: int) -> bool:
            trimmed.append((path, seconds))
            return True

        monkeypatch.setattr("agents.publisher.trim_video", fake_trim)
        node = video_node(
            dataclasses.replace(settings, output_dir=tmp_path), RecordingProvider()
        )
        result = await node({"prompts": PROMPTS, "duration_seconds": 12})

        assert trimmed == [(Path(result["media"].local_files[0]), 12)]
        assert result["media"].duration_seconds == 12

    async def test_nothing_is_trimmed_when_the_plan_fits_exactly(
        self, settings, tmp_path, monkeypatch
    ):
        async def never(path, seconds):  # pragma: no cover - must not be called
            raise AssertionError("trimmed a video that was already the right length")

        monkeypatch.setattr("agents.publisher.trim_video", never)
        node = video_node(
            dataclasses.replace(settings, output_dir=tmp_path), RecordingProvider()
        )
        result = await node({"prompts": PROMPTS, "duration_seconds": 15})

        assert result["media"].duration_seconds == 15

    async def test_a_failed_trim_reports_the_real_length_not_the_promised_one(
        self, settings, tmp_path, monkeypatch
    ):
        """If the cut cannot be made, do not claim a length the file does not have."""

        async def cannot_trim(path: Path, seconds: int) -> bool:
            return False

        monkeypatch.setattr("agents.publisher.trim_video", cannot_trim)
        node = video_node(
            dataclasses.replace(settings, output_dir=tmp_path), RecordingProvider()
        )
        result = await node({"prompts": PROMPTS, "duration_seconds": 12})

        assert result["media"].duration_seconds == 15, "generated 8+7, could not cut"

    async def test_the_default_length_is_used_when_the_run_names_none(
        self, settings, tmp_path
    ):
        provider = RecordingProvider()
        node = video_node(dataclasses.replace(settings, output_dir=tmp_path), provider)

        result = await node({"prompts": PROMPTS})

        assert provider.plan.seconds == 15
        assert result["media"].duration_seconds == 15

    async def test_saves_the_prompts_next_to_the_video(self, settings, tmp_path):
        node = video_node(
            dataclasses.replace(settings, output_dir=tmp_path), RecordingProvider()
        )
        result = await node({"prompts": PROMPTS})

        saved = Path(result["media"].prompt_file)
        assert saved.parent == tmp_path
        text = saved.read_text(encoding="utf-8")
        assert "first prompt" in text and "second prompt" in text

    async def test_no_prompts_is_an_error_not_a_crash(self, settings, tmp_path):
        node = video_node(
            dataclasses.replace(settings, output_dir=tmp_path), RecordingProvider()
        )
        assert (await node({}))["error"] == "no prompts to generate from"

    async def test_a_failing_provider_is_reported_not_raised(self, settings, tmp_path):
        class Failing:
            name = "failing"

            async def generate(self, prompts, out_dir, plan=None):
                raise RuntimeError("quota exceeded")

        node = video_node(dataclasses.replace(settings, output_dir=tmp_path), Failing())
        result = await node({"prompts": PROMPTS})

        assert "quota exceeded" in result["error"]
        assert "media" not in result

    async def test_resumable_strategy_uses_a_local_file_uri(self, settings, tmp_path):
        cfg = dataclasses.replace(settings, output_dir=tmp_path, s3_bucket="")
        result = await video_node(cfg, RecordingProvider())({"prompts": PROMPTS})

        assert result["media"].urls[0].startswith("file://")
        assert result["media"].local_files

    async def test_uploads_and_presigns_when_a_bucket_is_configured(
        self, settings, tmp_path, monkeypatch
    ):
        """With a bucket set, the node must hand Instagram an HTTPS URL."""
        calls: list[Path] = []

        def fake_upload(_cfg, path: Path) -> str:
            calls.append(path)
            return "https://s3.example.invalid/signed/reel.mp4"

        monkeypatch.setattr("agents.publisher.upload_and_presign", fake_upload)
        cfg = dataclasses.replace(
            settings, output_dir=tmp_path, s3_bucket="a-real-bucket", upload_strategy="url"
        )

        result = await video_node(cfg, RecordingProvider())({"prompts": PROMPTS})

        assert len(calls) == 1, "the rendered file should be uploaded exactly once"
        assert result["media"].urls == ["https://s3.example.invalid/signed/reel.mp4"]


class TestCaption:
    def test_joins_caption_and_hashtags(self):
        caption = build_caption({"dialogue": make_dialogue()})
        assert caption.startswith("Roasted to order")
        assert "#coffee" in caption

    def test_missing_dialogue_yields_empty_string(self):
        assert build_caption({}) == ""


class TestPublishNode:
    async def test_dry_run_returns_the_sentinel(self, settings):
        from state import Media

        node = publish_node(settings)
        result = await node(
            {
                "media": Media(urls=["https://example.invalid/r.mp4"]),
                "dialogue": make_dialogue(),
            }
        )
        assert result["post_id"] == "dry-run"

    async def test_missing_media_is_an_error_not_a_crash(self, settings):
        result = await publish_node(settings)({})
        assert result["error"] == "no media to publish"
        assert "post_id" not in result


class TestStubProvider:
    async def test_writes_a_file_without_calling_an_api(self, settings, tmp_path):
        provider = StubVideoProvider(settings)
        path = await provider.generate(["a prompt"], tmp_path)

        assert path.exists()
        assert path.suffix == ".mp4"
