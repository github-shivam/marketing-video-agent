"""Drives the Veo provider end to end against a fake Vertex AI client.

The point is to assert the exact request shapes before a real generation is
ever paid for: the base call, the extension call (which must reference the
previous generation's clip in Cloud Storage), and the download.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from providers.veo import VeoError, VeoVideoProvider
from video_plan import plan_for

PLAN_8 = plan_for(8)    # a base clip alone
PLAN_15 = plan_for(15)  # a base clip and one extension


class FakeVideo:
    """Stands in for ``types.Video`` - a ``gs://`` URI, no bytes."""

    def __init__(self, uri: str) -> None:
        self.uri = uri
        self.video_bytes = None


class FakeOperation:
    """Stands in for a ``GenerateVideosOperation``."""

    def __init__(self, uri: str, done: bool = True, error: Any = None) -> None:
        self.done = done
        self.error = error
        self.response = type("R", (), {"generated_videos": [
            type("G", (), {"video": FakeVideo(uri)})()
        ]})()


class FakeEmptyOperation:
    """A finished operation that returned no video, as a safety block does."""

    def __init__(self) -> None:
        self.done = True
        self.error = None
        self.response = type("R", (), {
            "generated_videos": [],
            "rai_media_filtered_count": 1,
            "rai_media_filtered_reasons": [
                "We encountered an issue with the audio for your prompt"
            ],
        })()


class FakeClient:
    """Records every call the provider makes.

    Like Vertex, each generation is written under the folder the request named
    in ``output_gcs_uri``, and the response carries the ``gs://`` URI.
    """

    def __init__(self, pending_polls: int = 0) -> None:
        self.generate_calls: list[dict[str, Any]] = []
        self.poll_count = 0
        self._pending = pending_polls
        self.aio = self

    # -- models ----------------------------------------------------------
    @property
    def models(self):
        return self

    async def generate_videos(self, **kwargs):
        self.generate_calls.append(kwargs)
        done = self._pending == 0
        folder = kwargs["config"].output_gcs_uri
        return FakeOperation(f"{folder}sample_0.mp4", done=done)

    # -- operations ------------------------------------------------------
    @property
    def operations(self):
        return self

    async def get(self, operation):
        self.poll_count += 1
        self._pending = max(0, self._pending - 1)
        operation.done = self._pending == 0
        return operation


@pytest.fixture(autouse=True)
def _no_real_storage(storage):
    """Every test in this module runs against the fake Cloud Storage."""


@pytest.fixture
def landscape(live):
    """16:9 with one extension."""
    return dataclasses.replace(live, veo_aspect_ratio="16:9")


def provider_with(cfg, client) -> VeoVideoProvider:
    """Build a provider whose client is the supplied fake."""
    provider = VeoVideoProvider(cfg)
    provider._client = lambda: client
    return provider


class TestBaseGeneration:
    async def test_single_beat_makes_one_call(self, live, tmp_path):
        client = FakeClient()
        path = await provider_with(live, client).generate(["a prompt"], tmp_path, PLAN_8)

        assert len(client.generate_calls) == 1
        assert path.exists() and path.read_bytes() == b"vertex mp4 payload"

    async def test_base_call_uses_the_configured_model_and_prompt(
        self, live, tmp_path
    ):
        client = FakeClient()
        await provider_with(live, client).generate(["the hook beat"], tmp_path, PLAN_8)

        call = client.generate_calls[0]
        assert call["model"] == "veo-3.1-fast-generate-001"
        assert call["source"].prompt == "the hook beat"
        assert call["config"].aspect_ratio == "9:16"
        assert call["config"].person_generation == "allow_all"
        assert call["source"].video is None, "base call must not reference a prior video"


class TestExtension:
    async def test_extension_references_the_previous_generation(
        self, landscape, tmp_path
    ):
        """Veo extends a clip already in Cloud Storage, not a local file."""
        client = FakeClient()
        await provider_with(landscape, client).generate(["beat one", "beat two"], tmp_path, PLAN_15)

        assert len(client.generate_calls) == 2
        extension = client.generate_calls[1]
        assert extension["source"].prompt == "beat two"
        assert extension["source"].video is not None
        assert extension["source"].video.uri.endswith("/base/sample_0.mp4"), (
            "must chain off the base clip"
        )

    async def test_extension_always_requests_720p(self, landscape, tmp_path):
        """Extension is 720p-only, so the extension call always says so."""
        client = FakeClient()
        await provider_with(landscape, client).generate(["one", "two"], tmp_path, PLAN_15)

        assert client.generate_calls[1]["config"].resolution == "720p"

    async def test_extra_prompts_beyond_the_plan_are_ignored(self, landscape, tmp_path):
        """The plan caps spend; a long prompt list must not overrun it."""
        client = FakeClient()
        await provider_with(landscape, client).generate(
            ["a", "b", "c", "d"], tmp_path, PLAN_15
        )

        assert len(client.generate_calls) == 2, "1 base + 1 extension, as planned"

    async def test_a_short_video_makes_only_the_base_call(self, landscape, tmp_path):
        client = FakeClient()
        await provider_with(landscape, client).generate(["a", "b"], tmp_path, PLAN_8)

        assert len(client.generate_calls) == 1


class TestThePlanDrivesTheCalls:
    """How long the video is decides how many generations are made, and how long."""

    @pytest.mark.parametrize(
        ("seconds", "calls"), [(5, 1), (8, 1), (15, 2), (22, 3), (29, 4), (30, 4)]
    )
    async def test_one_call_per_generation_in_the_plan(
        self, live, tmp_path, seconds, calls
    ):
        client = FakeClient()
        prompts = [f"p{i}" for i in range(5)]
        await provider_with(live, client).generate(prompts, tmp_path, plan_for(seconds))

        assert len(client.generate_calls) == calls

    @pytest.mark.parametrize(("seconds", "base"), [(4, 4), (5, 6), (6, 6), (7, 8), (8, 8)])
    async def test_a_short_video_asks_for_the_smallest_base_clip_that_fits(
        self, live, tmp_path, seconds, base
    ):
        client = FakeClient()
        await provider_with(live, client).generate(["a"], tmp_path, plan_for(seconds))

        assert client.generate_calls[0]["config"].duration_seconds == base

    async def test_a_video_that_extends_always_starts_from_an_eight_second_base(
        self, live, tmp_path
    ):
        client = FakeClient()
        await provider_with(live, client).generate(["a", "b"], tmp_path, plan_for(10))

        assert client.generate_calls[0]["config"].duration_seconds == 8

    async def test_every_extension_asks_for_seven_seconds(self, live, tmp_path):
        client = FakeClient()
        await provider_with(live, client).generate(
            ["a", "b", "c", "d"], tmp_path, plan_for(29)
        )

        extensions = client.generate_calls[1:]
        assert len(extensions) == 3
        assert all(c["config"].duration_seconds == 7 for c in extensions)

    async def test_each_extension_chains_off_the_one_before(self, live, tmp_path):
        client = FakeClient()
        await provider_with(live, client).generate(
            ["a", "b", "c", "d"], tmp_path, plan_for(29)
        )

        uris = [c["source"].video.uri for c in client.generate_calls[1:]]
        assert [u.rsplit("/", 2)[-2] for u in uris] == ["base", "ext1", "ext2"]

    async def test_the_final_clip_is_downloaded_from_the_last_extension(
        self, live, storage, tmp_path
    ):
        client = FakeClient()
        await provider_with(live, client).generate(
            ["a", "b", "c", "d"], tmp_path, plan_for(29)
        )

        assert storage.downloads[0]["blob"].endswith("/ext3/sample_0.mp4")

    async def test_too_few_prompts_fail_before_anything_is_billed(self, live, tmp_path):
        client = FakeClient()
        with pytest.raises(VeoError, match="needs 3 prompts, but only 2"):
            await provider_with(live, client).generate(["a", "b"], tmp_path, plan_for(22))

        assert client.generate_calls == []

    async def test_with_no_plan_the_default_length_is_used(self, live, tmp_path):
        client = FakeClient()
        await provider_with(live, client).generate(["a", "b"], tmp_path)

        assert len(client.generate_calls) == 2, "the default is 15s: base + one extension"


class TestPolling:
    async def test_polls_until_done(self, live, tmp_path):
        client = FakeClient(pending_polls=3)
        await provider_with(live, client).generate(["a prompt"], tmp_path, PLAN_8)

        assert client.poll_count == 3

    async def test_timeout_raises(self, live, tmp_path):
        client = FakeClient(pending_polls=10_000)
        cfg = dataclasses.replace(live, veo_timeout=0)
        with pytest.raises(VeoError, match="timed out"):
            await provider_with(cfg, client).generate(["a prompt"], tmp_path, PLAN_8)

    async def test_operation_error_raises(self, live, tmp_path):
        client = FakeClient()

        async def failing(**kwargs):
            return FakeOperation("gs://x/y.mp4", done=True, error={"message": "quota"})

        provider = provider_with(live, client)
        client.generate_videos = failing
        with pytest.raises(VeoError, match="quota"):
            await provider.generate(["a prompt"], tmp_path, PLAN_8)

    async def test_blocked_generation_is_reported_clearly(self, live, tmp_path):
        """An empty result means a safety filter - and is not billed."""
        client = FakeClient()

        async def blocked(**kwargs):
            return FakeEmptyOperation()

        provider = provider_with(live, client)
        client.generate_videos = blocked
        with pytest.raises(VeoError, match="safety filter"):
            await provider.generate(["a prompt"], tmp_path, PLAN_8)


class TestDownload:
    async def test_the_clip_is_fetched_from_cloud_storage(self, live, storage, tmp_path):
        client = FakeClient()
        path = await provider_with(live, client).generate(["a prompt"], tmp_path, PLAN_8)

        (download,) = storage.downloads
        assert download["bucket"] == "my-veo-bucket"
        assert download["blob"].endswith("/base/sample_0.mp4")
        assert download["filename"] == str(path)

    async def test_downloads_the_final_clip_not_the_base(
        self, landscape, storage, tmp_path
    ):
        client = FakeClient()
        await provider_with(landscape, client).generate(["one", "two"], tmp_path, PLAN_15)

        (download,) = storage.downloads
        assert download["blob"].endswith("/ext1/sample_0.mp4"), (
            "must download the extended clip, not the 8s base"
        )

    async def test_storage_uses_the_configured_project(self, live, storage, tmp_path):
        await provider_with(live, FakeClient()).generate(["a"], tmp_path, PLAN_8)
        assert storage.projects == ["my-project"]

    async def test_empty_download_raises(self, live, storage, tmp_path):
        storage.payload = b""
        with pytest.raises(VeoError, match="no data"):
            await provider_with(live, FakeClient()).generate(["a prompt"], tmp_path, PLAN_8)


class TestExtensionAspectRatio:
    """Regression: the extension call must restate the source aspect ratio.

    Omitting aspect_ratio made the Gemini API default to 16:9 and reject a 9:16
    input with "Aspect ratio of the input video must be 16:9, but got: 9:16",
    which reads like vertical video cannot be extended. It can: the requested
    output and the input simply have to match. Vertex is sent the same value.
    """

    async def test_extension_restates_the_vertical_aspect_ratio(self, live, tmp_path):
        client = FakeClient()
        cfg = dataclasses.replace(
            live, veo_aspect_ratio="9:16", veo_resolution="720p"
        )
        await provider_with(cfg, client).generate(["a", "b"], tmp_path, PLAN_15)

        assert client.generate_calls[1]["config"].aspect_ratio == "9:16", (
            "omitting this makes the API default to 16:9 and reject the input"
        )

    async def test_extension_restates_a_landscape_aspect_ratio(self, live, tmp_path):
        client = FakeClient()
        cfg = dataclasses.replace(
            live, veo_aspect_ratio="16:9", veo_resolution="720p"
        )
        await provider_with(cfg, client).generate(["a", "b"], tmp_path, PLAN_15)

        assert client.generate_calls[1]["config"].aspect_ratio == "16:9"

    async def test_base_and_extension_agree(self, live, tmp_path):
        """A mismatch between the two calls is what triggered the failure."""
        client = FakeClient()
        cfg = dataclasses.replace(live, veo_resolution="720p")
        await provider_with(cfg, client).generate(["a", "b"], tmp_path, PLAN_15)

        base, extension = client.generate_calls
        assert base["config"].aspect_ratio == extension["config"].aspect_ratio


class TestExtensionPreconditions:
    """Fail before billing a base clip that cannot then be extended."""

    async def test_lite_model_cannot_extend(self, live, tmp_path):
        client = FakeClient()
        cfg = dataclasses.replace(live, veo_model="veo-3.1-lite-generate-001")

        with pytest.raises(VeoError, match="does not support video extension"):
            await provider_with(cfg, client).generate(["a", "b"], tmp_path, PLAN_15)
        assert client.generate_calls == [], "must not bill an unusable base clip"

    async def test_non_720p_cannot_extend(self, live, tmp_path):
        """Extension is 720p-only, so 1080p + extensions is a contradiction."""
        client = FakeClient()
        cfg = dataclasses.replace(live, veo_resolution="1080p")

        with pytest.raises(VeoError, match="720p-only"):
            await provider_with(cfg, client).generate(["a", "b"], tmp_path, PLAN_15)
        assert client.generate_calls == []

    async def test_no_extensions_skips_every_precondition(self, live, tmp_path):
        """1080p/8s is a valid shape - it just cannot be extended."""
        client = FakeClient()
        cfg = dataclasses.replace(live, veo_resolution="1080p")

        await provider_with(cfg, client).generate(["a"], tmp_path, PLAN_8)
        assert len(client.generate_calls) == 1


class TestDocumentedParameterCompliance:
    """Assert both configs match the published Veo parameter table."""

    @pytest.fixture
    def configs(self, live):
        provider = VeoVideoProvider(live)
        return provider._base_config(), provider._extension_config()

    def test_duration_is_7_for_extension(self, configs):
        """Vertex: 'supported durations are [7] for feature video_extension'.

        Found live: the Gemini API wanted 8 here and Vertex rejects it.
        """
        _, extension = configs
        assert extension.duration_seconds == 7

    def test_extension_is_720p(self, configs):
        """'720p only for extension'."""
        _, extension = configs
        assert extension.resolution == "720p"

    def test_person_generation_set_on_both_calls(self, configs):
        """'Text-to-video & Extension: "allow_all" only'."""
        base, extension = configs
        assert base.person_generation == "allow_all"
        assert extension.person_generation == "allow_all", (
            "extension is documented as allow_all-only, same as text-to-video"
        )

    def test_one_video_per_request(self, configs):
        """'Videos per request: 1'."""
        for config in configs:
            assert config.number_of_videos == 1

    def test_no_reference_images_on_a_text_to_video_run(self, configs):
        """Reference images would force person_generation to allow_adult."""
        for config in configs:
            assert config.reference_images is None


class TestImageToVideo:
    """A supplied first frame changes both the call shape and the rules."""

    @pytest.fixture
    def photo(self, tmp_path):
        path = tmp_path / "me.jpg"
        path.write_bytes(b"not-a-real-jpeg-but-enough-for-from_file")
        return path

    async def test_text_to_video_sends_no_image(self, live, tmp_path):
        client = FakeClient()
        await provider_with(live, client).generate(["a"], tmp_path, PLAN_8)
        assert client.generate_calls[0]["source"].image is None

    async def test_image_is_attached_to_the_base_call(self, live, photo, tmp_path):
        client = FakeClient()
        cfg = dataclasses.replace(live, veo_image=str(photo))
        await provider_with(cfg, client).generate(["a"], tmp_path, PLAN_8)

        assert client.generate_calls[0]["source"].image is not None

    async def test_image_forces_allow_adult(self, live, photo, tmp_path):
        """Image-to-video is documented as allow_adult-only."""
        client = FakeClient()
        cfg = dataclasses.replace(live, veo_image=str(photo))
        await provider_with(cfg, client).generate(["a"], tmp_path, PLAN_8)

        assert client.generate_calls[0]["config"].person_generation == "allow_adult"

    async def test_extension_after_an_image_reverts_to_allow_all(
        self, live, photo, tmp_path
    ):
        """The extension is video-to-video, so it needs allow_all again."""
        client = FakeClient()
        cfg = dataclasses.replace(
            live, veo_image=str(photo), veo_resolution="720p"
        )
        await provider_with(cfg, client).generate(["a", "b"], tmp_path, PLAN_15)

        base, extension = client.generate_calls
        assert base["config"].person_generation == "allow_adult"
        assert extension["config"].person_generation == "allow_all"
        assert extension["source"].image is None, "only the base frame is seeded"

    async def test_missing_image_fails_before_generating(self, live, tmp_path):
        client = FakeClient()
        cfg = dataclasses.replace(live, veo_image=str(tmp_path / "nope.jpg"))

        with pytest.raises(VeoError, match="does not exist"):
            await provider_with(cfg, client).generate(["a"], tmp_path, PLAN_8)
        assert client.generate_calls == []


class BlockingClient(FakeClient):
    """Blocks the first ``block_times`` generations, then succeeds."""

    def __init__(self, block_times: int) -> None:
        super().__init__()
        self.block_times = block_times

    async def generate_videos(self, **kwargs):
        self.generate_calls.append(kwargs)
        if self.block_times > 0:
            self.block_times -= 1
            return FakeEmptyOperation()
        return FakeOperation(f"{kwargs['config'].output_gcs_uri}sample_0.mp4")


class TestSafetyBlockRetry:
    """Blocked generations are free, so retry rather than failing the run."""

    async def test_recovers_from_a_single_block(self, live, tmp_path):
        client = BlockingClient(block_times=1)
        path = await provider_with(live, client).generate(["a"], tmp_path, PLAN_8)

        assert len(client.generate_calls) == 2
        assert path.exists()

    async def test_gives_up_after_the_retry_budget(self, live, tmp_path):
        client = BlockingClient(block_times=99)
        cfg = dataclasses.replace(live, veo_block_retries=2)

        with pytest.raises(VeoError, match="safety filter"):
            await provider_with(cfg, client).generate(["a"], tmp_path, PLAN_8)

        assert len(client.generate_calls) == 3, "1 initial attempt + 2 retries"

    async def test_zero_retries_fails_immediately(self, live, tmp_path):
        client = BlockingClient(block_times=99)
        cfg = dataclasses.replace(live, veo_block_retries=0)

        with pytest.raises(VeoError):
            await provider_with(cfg, client).generate(["a"], tmp_path, PLAN_8)
        assert len(client.generate_calls) == 1

    async def test_block_reason_is_surfaced(self, live, tmp_path):
        """The remedy differs by reason, so do not swallow it."""
        client = BlockingClient(block_times=99)
        cfg = dataclasses.replace(live, veo_block_retries=0)

        with pytest.raises(VeoError) as excinfo:
            await provider_with(cfg, client).generate(["a"], tmp_path, PLAN_8)
        assert "not billed" in str(excinfo.value)
