"""What is specific to running Veo on Vertex AI: Google Cloud auth and a bucket.

``test_veo_flow`` covers the generation flow. This file pins down the parts
that differ from a plain API-key service: no key anywhere, an explicit project
and location, one output folder per generation, and the Cloud Storage handoff.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest

from agents.publisher import select_video_provider
from providers.stub import StubVideoProvider
from providers.veo import VeoError, VeoVideoProvider
from video_plan import plan_for

from conftest import BUCKET
from test_veo_flow import FakeClient, FakeVideo


def provider_with(cfg, client) -> VeoVideoProvider:
    provider = VeoVideoProvider(cfg)
    provider._client = lambda: client
    return provider


class TestConfiguration:
    def test_needs_a_project_and_a_bucket(self, settings):
        with pytest.raises(VeoError, match="GOOGLE_CLOUD_PROJECT and VEO_GCS_BUCKET"):
            VeoVideoProvider(settings)

    def test_names_only_what_is_missing(self, live):
        cfg = dataclasses.replace(live, veo_gcs_bucket="")
        with pytest.raises(VeoError) as excinfo:
            VeoVideoProvider(cfg)
        assert "VEO_GCS_BUCKET" in str(excinfo.value)
        assert "GOOGLE_CLOUD_PROJECT" not in str(excinfo.value)

    def test_there_is_no_api_key_setting(self, settings):
        """Auth is Application Default Credentials; no key can be configured."""
        assert not hasattr(settings, "gemini_api_key")

    def test_the_client_is_a_vertex_client_for_the_project(self, live, monkeypatch):
        seen: dict[str, Any] = {}
        monkeypatch.setattr("google.genai.Client", lambda **kw: seen.update(kw))

        VeoVideoProvider(live)._client()

        assert seen == {
            "vertexai": True,
            "project": "my-project",
            "location": "us-central1",
        }, "no api_key: the SDK must use Application Default Credentials"


class TestRequestShape:
    def test_both_calls_ask_for_audio_and_an_output_folder(self, live):
        provider = VeoVideoProvider(live)
        base = provider._base_config(output_gcs_uri="gs://b/p/base/")
        extension = provider._extension_config(output_gcs_uri="gs://b/p/ext1/")

        assert base.generate_audio is True and extension.generate_audio is True
        assert base.output_gcs_uri == "gs://b/p/base/"
        assert extension.output_gcs_uri == "gs://b/p/ext1/"

    async def test_every_call_writes_to_its_own_folder_in_the_bucket(
        self, live, storage, tmp_path
    ):
        client = FakeClient()
        await provider_with(live, client).generate(["one", "two"], tmp_path)

        base, extension = (c["config"].output_gcs_uri for c in client.generate_calls)
        assert base.startswith(f"gs://{BUCKET}/marketing-agent/")
        assert base.endswith("/base/") and extension.endswith("/ext1/")
        assert base.rsplit("/base/", 1)[0] == extension.rsplit("/ext1/", 1)[0], (
            "one run, one folder: the base and its extension sit together"
        )

    async def test_two_runs_never_share_a_folder(self, live, storage, tmp_path):
        first, second = FakeClient(), FakeClient()
        await provider_with(live, first).generate(["a"], tmp_path, plan_for(8))
        await provider_with(live, second).generate(["a"], tmp_path, plan_for(8))

        assert (
            first.generate_calls[0]["config"].output_gcs_uri
            != second.generate_calls[0]["config"].output_gcs_uri
        )


class TestStorageHandoff:
    async def test_inline_bytes_are_written_without_touching_storage(
        self, live, storage, tmp_path
    ):
        destination = tmp_path / "out.mp4"
        video = FakeVideo("")
        video.video_bytes = b"inline"

        await VeoVideoProvider(live)._download_from_gcs(video, destination)

        assert destination.read_bytes() == b"inline"
        assert storage.downloads == []

    async def test_a_response_with_no_location_is_an_error(
        self, live, storage, tmp_path
    ):
        with pytest.raises(VeoError, match="neither bytes nor a gs://"):
            await VeoVideoProvider(live)._download_from_gcs(
                FakeVideo("https://elsewhere"), tmp_path / "x"
            )

    async def test_the_bucket_and_object_are_split_from_the_uri(
        self, live, storage, tmp_path
    ):
        await VeoVideoProvider(live)._download_from_gcs(
            FakeVideo("gs://some-bucket/a/b/c.mp4"), tmp_path / "x.mp4"
        )

        (download,) = storage.downloads
        assert (download["bucket"], download["blob"]) == ("some-bucket", "a/b/c.mp4")


class TestSelection:
    def test_a_project_and_bucket_make_veo_live(self, live):
        assert live.veo_live
        assert isinstance(select_video_provider(live), VeoVideoProvider)

    def test_nothing_configured_falls_back_to_the_stub(self, settings, caplog):
        assert not settings.veo_live
        with caplog.at_level("WARNING"):
            provider = select_video_provider(settings)

        assert isinstance(provider, StubVideoProvider)
        assert "VEO_GCS_BUCKET" in caplog.text

    def test_a_project_without_a_bucket_is_not_live(self, live):
        """Without a bucket Vertex has nowhere to write, so use the stub."""
        cfg = dataclasses.replace(live, veo_gcs_bucket="")
        assert not cfg.veo_live
        assert isinstance(select_video_provider(cfg), StubVideoProvider)


class TestTwoPeoplesVideosNeverShareAFile:
    """Regression: files were named from the time in whole seconds, so two videos
    finishing in the same second got one path and one user's video replaced the
    other's, and both runs then pointed at it."""

    async def test_two_veo_videos_finished_in_the_same_second_get_different_files(
        self, live, storage, tmp_path, monkeypatch
    ):
        import asyncio

        monkeypatch.setattr("providers.veo.time.time", lambda: 1_700_000_000.4)
        first, second = FakeClient(), FakeClient()

        a, b = await asyncio.gather(
            provider_with(live, first).generate(["a"], tmp_path, plan_for(8)),
            provider_with(live, second).generate(["b"], tmp_path, plan_for(8)),
        )

        assert a != b
        assert a.exists() and b.exists() and len(list(tmp_path.glob("reel-*.mp4"))) == 2

    async def test_two_stub_videos_finished_in_the_same_second_get_different_files(
        self, settings, tmp_path, monkeypatch
    ):
        import asyncio

        monkeypatch.setattr("providers.stub.time.time", lambda: 1_700_000_000.4)
        cfg = dataclasses.replace(settings, output_dir=tmp_path)
        stub = StubVideoProvider(cfg)

        a, b = await asyncio.gather(
            stub.generate(["a"], tmp_path, plan_for(4)), stub.generate(["b"], tmp_path, plan_for(4))
        )

        assert a != b and a.exists() and b.exists()

    async def test_the_prompt_file_of_each_video_is_its_own(self, settings, tmp_path, monkeypatch):
        """The prompts are saved beside the video, so they must not collide either."""
        import asyncio

        from agents.publisher import video_node

        monkeypatch.setattr("providers.stub.time.time", lambda: 1_700_000_000.4)
        cfg = dataclasses.replace(settings, output_dir=tmp_path)
        node = video_node(cfg, StubVideoProvider(cfg))

        one, two = await asyncio.gather(
            node({"prompts": ["PROMPT FOR ALICE"], "duration_seconds": 4}),
            node({"prompts": ["PROMPT FOR BOB"], "duration_seconds": 4}),
        )

        alice = Path(one["media"].prompt_file).read_text(encoding="utf-8")
        bob = Path(two["media"].prompt_file).read_text(encoding="utf-8")
        assert "ALICE" in alice and "BOB" not in alice
        assert "BOB" in bob and "ALICE" not in bob
        assert one["media"].local_files != two["media"].local_files
