"""Sending the person's photos to Veo, and carrying on without them if Veo says no.

The person chose this: try the photos as subject references so the video looks like
them, and if Veo will not take them, make the video from a description instead of
failing. These tests pin both halves, and pin which failures are allowed to fall
back (a refusal of the photos) and which are not (running out of quota).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from google.genai import errors, types

from agents.publisher import video_node
from providers.stub import StubVideoProvider
from providers.veo import VeoError, VeoVideoProvider
from video_plan import plan_for

from conftest import make_asset_brief
from test_veo_flow import FakeClient, FakeEmptyOperation, FakeOperation


@pytest.fixture(autouse=True)
def _no_real_storage(storage):
    """Every test here runs against the fake Cloud Storage."""


@pytest.fixture
def photos(tmp_path):
    files = []
    for i in range(5):
        path = tmp_path / f"ref-{i}.jpg"
        path.write_bytes(f"photo-{i}".encode())
        files.append(path)
    return files


def provider_with(cfg, client) -> VeoVideoProvider:
    provider = VeoVideoProvider(cfg)
    provider._client = lambda: client
    return provider


class RefusingClient(FakeClient):
    """Fails any generation that carries reference photos, in a chosen way."""

    def __init__(self, how: str):
        super().__init__()
        self.how = how

    async def generate_videos(self, **kwargs):
        if kwargs["config"].reference_images:
            self.generate_calls.append(kwargs)
            if self.how == "blocked":
                return FakeEmptyOperation()
            if self.how == "400":
                raise errors.ClientError(
                    400, {"error": {"code": 400, "message": "not supported", "status": "INVALID_ARGUMENT"}}
                )
            if self.how == "403":
                raise errors.ClientError(403, {"error": {"code": 403, "message": "denied"}})
            if self.how == "invalid_argument":
                return FakeOperation("gs://x/y.mp4", error={"code": 3, "message": "bad references"})
            if self.how == "quota":
                raise errors.ClientError(429, {"error": {"code": 429, "message": "quota"}})
            if self.how == "server":
                raise errors.ServerError(500, {"error": {"code": 500, "message": "boom"}})
            raise AssertionError(self.how)
        return await super().generate_videos(**kwargs)


class TestSendingThePhotos:
    async def test_the_photos_go_to_veo_as_subject_references(self, live, photos, tmp_path):
        client = FakeClient()
        await provider_with(live, client).generate(["a"], tmp_path, plan_for(8, fixed_base=True), photos[:2])

        config = client.generate_calls[0]["config"]
        assert len(config.reference_images) == 2
        assert all(r.reference_type == types.VideoGenerationReferenceType.ASSET for r in config.reference_images)
        assert [r.image.image_bytes for r in config.reference_images] == [b"photo-0", b"photo-1"]

    async def test_at_most_three_photos_are_sent(self, live, photos, tmp_path):
        client = FakeClient()
        await provider_with(live, client).generate(["a"], tmp_path, plan_for(8, fixed_base=True), photos)

        assert len(client.generate_calls[0]["config"].reference_images) == 3

    async def test_photos_need_allow_adult_and_an_eight_second_base(self, live, photos, tmp_path):
        """Documented: reference images are allow_adult only, and 8s only."""
        client = FakeClient()
        await provider_with(live, client).generate(["a"], tmp_path, plan_for(5, fixed_base=True), photos[:1])

        config = client.generate_calls[0]["config"]
        assert config.person_generation == "allow_adult"
        assert config.duration_seconds == 8

    async def test_a_prompt_and_no_first_frame_go_with_them(self, live, photos, tmp_path):
        client = FakeClient()
        await provider_with(live, client).generate(["the prompt"], tmp_path, plan_for(8, fixed_base=True), photos[:1])

        source = client.generate_calls[0]["source"]
        assert source.prompt == "the prompt" and source.image is None

    async def test_the_extension_does_not_carry_the_photos_but_keeps_allow_all(
        self, live, photos, tmp_path
    ):
        """The photos start the clip; the extension continues the clip."""
        client = FakeClient()
        await provider_with(live, client).generate(
            ["a", "b"], tmp_path, plan_for(15, fixed_base=True), photos[:2]
        )

        base, extension = client.generate_calls
        assert base["config"].reference_images and not extension["config"].reference_images
        assert extension["config"].person_generation == "allow_all"
        assert extension["source"].video is not None

    async def test_a_first_frame_in_the_settings_gives_way_to_the_photos(
        self, live, photos, tmp_path
    ):
        """What this person just sent outranks the one photo fixed in the settings."""
        fixed = tmp_path / "fixed.jpg"
        fixed.write_bytes(b"fixed")
        cfg = dataclasses.replace(live, veo_image=str(fixed))
        client = FakeClient()

        await provider_with(cfg, client).generate(["a"], tmp_path, plan_for(8, fixed_base=True), photos[:1])

        call = client.generate_calls[0]
        assert call["source"].image is None
        assert call["config"].reference_images

    async def test_no_photos_is_an_ordinary_run(self, live, tmp_path):
        client = FakeClient()
        provider = provider_with(live, client)
        await provider.generate(["a"], tmp_path, plan_for(8))

        assert not client.generate_calls[0]["config"].reference_images
        assert provider.notes == []

    async def test_a_photo_that_has_gone_is_skipped_not_fatal(self, live, photos, tmp_path):
        photos[0].unlink()
        client = FakeClient()
        await provider_with(live, client).generate(["a"], tmp_path, plan_for(8, fixed_base=True), photos[:2])

        sent = client.generate_calls[0]["config"].reference_images
        assert [r.image.image_bytes for r in sent] == [b"photo-1"]

    async def test_if_every_photo_has_gone_it_is_a_plain_text_run(self, live, photos, tmp_path):
        for photo in photos:
            photo.unlink()
        client = FakeClient()
        provider = provider_with(live, client)

        await provider.generate(["a"], tmp_path, plan_for(8, fixed_base=True), photos)

        assert not client.generate_calls[0]["config"].reference_images
        assert provider.notes == [], "nothing was refused, so nothing to explain"


class TestWhenVeoWillNotTakeThePhotos:
    @pytest.mark.parametrize("how", ["blocked", "400", "403", "invalid_argument"])
    async def test_the_video_is_made_from_a_description_instead(self, live, photos, tmp_path, how):
        client = RefusingClient(how)
        provider = provider_with(live, client)

        path = await provider.generate(["a"], tmp_path, plan_for(8, fixed_base=True), photos[:2])

        assert path.exists(), "the run still produces a video"
        *with_photos, without = client.generate_calls
        assert with_photos and all(c["config"].reference_images for c in with_photos)
        assert not without["config"].reference_images, "the last try has no photos"

    async def test_a_blocked_generation_is_retried_with_the_photos_first(
        self, live, photos, tmp_path
    ):
        """A block is intermittent and not billed, so it gets its free retries
        before the photos are given up on."""
        client = RefusingClient("blocked")
        await provider_with(live, client).generate(["a"], tmp_path, plan_for(8, fixed_base=True), photos[:1])

        with_photos = [c for c in client.generate_calls if c["config"].reference_images]
        assert len(with_photos) == live.veo_block_retries + 1

    @pytest.mark.parametrize("how", ["blocked", "400", "invalid_argument"])
    async def test_the_person_is_told_what_happened(self, live, photos, tmp_path, how):
        provider = provider_with(live, RefusingClient(how))
        await provider.generate(["a"], tmp_path, plan_for(8, fixed_base=True), photos[:1])

        (note,) = provider.notes
        assert "couldn't use your photos as references" in note
        assert "description" in note and "may not look exactly like them" in note

    async def test_the_reason_is_given(self, live, photos, tmp_path):
        blocked = provider_with(live, RefusingClient("blocked"))
        await blocked.generate(["a"], tmp_path, plan_for(8, fixed_base=True), photos[:1])
        refused = provider_with(live, RefusingClient("400"))
        await refused.generate(["a"], tmp_path, plan_for(8, fixed_base=True), photos[:1])

        assert "safety check" in blocked.notes[0]
        assert "does not accept them for this video" in refused.notes[0]

    async def test_the_second_try_keeps_the_planned_length(self, live, photos, tmp_path):
        client = RefusingClient("400")
        await provider_with(live, client).generate(
            ["a", "b"], tmp_path, plan_for(15, fixed_base=True), photos[:1]
        )

        assert [c["config"].duration_seconds for c in client.generate_calls] == [8, 8, 7]

    async def test_a_refusal_in_the_extension_is_not_blamed_on_the_photos(
        self, live, photos, tmp_path
    ):
        """Only the first call carries photos, so only it may fall back."""

        class ExtensionFails(FakeClient):
            async def generate_videos(self, **kwargs):
                if kwargs["source"].video is not None:
                    raise errors.ClientError(400, {"error": {"code": 400, "message": "nope"}})
                return await super().generate_videos(**kwargs)

        provider = provider_with(live, ExtensionFails())
        with pytest.raises(errors.ClientError):
            await provider.generate(["a", "b"], tmp_path, plan_for(15, fixed_base=True), photos[:1])
        assert provider.notes == []

    @pytest.mark.parametrize("how", ["quota", "server"])
    async def test_other_failures_are_not_retried_without_the_photos(
        self, live, photos, tmp_path, how
    ):
        """Running out of quota fails the same way again, and would bill nothing
        but waste a minute. It must surface, not be hidden behind a fallback."""
        client = RefusingClient(how)
        provider = provider_with(live, client)

        with pytest.raises((errors.ClientError, errors.ServerError)):
            await provider.generate(["a"], tmp_path, plan_for(8, fixed_base=True), photos[:1])

        assert len(client.generate_calls) == 1, "no second attempt"
        assert provider.notes == []

    async def test_a_run_without_photos_never_falls_back(self, live, tmp_path):
        class Quota(FakeClient):
            async def generate_videos(self, **kwargs):
                raise errors.ClientError(400, {"error": {"code": 400, "message": "nope"}})

        with pytest.raises(errors.ClientError):
            await provider_with(live, Quota()).generate(["a"], tmp_path, plan_for(8))

    async def test_notes_do_not_leak_from_one_video_to_the_next(self, live, photos, tmp_path):
        provider = provider_with(live, RefusingClient("400"))
        await provider.generate(["a"], tmp_path, plan_for(8, fixed_base=True), photos[:1])
        assert provider.notes

        provider._client = lambda: FakeClient()
        await provider.generate(["a"], tmp_path, plan_for(8))

        assert provider.notes == []


class TestTheVideoNodePassesThemOn:
    @pytest.fixture
    def cfg(self, settings, tmp_path):
        return dataclasses.replace(settings, output_dir=tmp_path / "out")

    async def test_the_reference_photos_reach_the_provider(self, cfg, photos):
        seen = {}

        class Provider:
            name = "p"
            notes: list[str] = []

            async def generate(self, prompts, out_dir, plan=None, references=None):
                seen.update(references=references, plan=plan)
                out_dir.mkdir(parents=True, exist_ok=True)
                path = out_dir / "v.mp4"
                path.touch()
                return path

        brief = make_asset_brief(references=[str(p) for p in photos[:2]])
        await video_node(cfg, Provider())({"prompts": ["a"], "duration_seconds": 8, "asset_brief": brief})

        assert seen["references"] == photos[:2]
        assert seen["plan"].segments[0].generate == 8

    async def test_a_provider_that_knows_nothing_about_photos_is_called_as_before(self, cfg):
        """Passing a keyword it does not have would break it, so only do so with photos."""

        class Old:
            name = "old"

            async def generate(self, prompts, out_dir, plan=None):
                out_dir.mkdir(parents=True, exist_ok=True)
                path = out_dir / "v.mp4"
                path.touch()
                return path

        result = await video_node(cfg, Old())({"prompts": ["a"], "asset_brief": make_asset_brief()})
        assert result["media"].local_files

    async def test_what_the_provider_reports_reaches_the_person(self, cfg, photos):
        class Provider:
            name = "p"
            notes = ["Veo couldn't use your photos as references (...)."]

            async def generate(self, prompts, out_dir, plan=None, references=None):
                out_dir.mkdir(parents=True, exist_ok=True)
                path = out_dir / "v.mp4"
                path.touch()
                return path

        brief = make_asset_brief(references=[str(photos[0])])
        result = await video_node(cfg, Provider())({"prompts": ["a"], "asset_brief": brief})

        assert result["media"].notes == ["Veo couldn't use your photos as references (...)."]

    async def test_photos_that_are_gone_are_not_passed(self, cfg, photos):
        seen = {}

        class Provider:
            name = "p"
            notes: list[str] = []

            async def generate(self, prompts, out_dir, plan=None, references=None):
                seen["references"] = references
                out_dir.mkdir(parents=True, exist_ok=True)
                path = out_dir / "v.mp4"
                path.touch()
                return path

        brief = make_asset_brief(references=[str(photos[0]), str(photos[0].parent / "gone.jpg")])
        await video_node(cfg, Provider())({"prompts": ["a"], "asset_brief": brief})

        assert seen["references"] == [photos[0]]

    async def test_the_free_stub_accepts_photos_and_ignores_them(self, cfg, photos):
        provider = StubVideoProvider(cfg)
        path = await provider.generate(["a"], cfg.output_dir, plan_for(4), photos[:1])
        assert path.exists()
