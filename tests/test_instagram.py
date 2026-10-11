"""Instagram Graph API tests. Every HTTP call is mocked with respx.

Nothing in this file may reach Meta. The live-publish path is only ever
exercised against mocks, with ``DRY_RUN`` explicitly disabled so that the
real request shapes are asserted.
"""

from __future__ import annotations

import dataclasses

import httpx
import pytest
import respx

from tools.instagram import (
    RUPLOAD_HOST,
    InstagramError,
    create_resumable_container,
    base_url,
    create_reel_container,
    publish_container,
    publish_reel,
    upload_video_bytes,
    wait_for_container,
)

VIDEO_URL = "https://example.invalid/presigned/reel.mp4"
CAPTION = "Roasted to order.\n\n#coffee #roastery"


@pytest.fixture
def live(settings):
    """URL-strategy settings with DRY_RUN disabled, for real request shapes."""
    return dataclasses.replace(settings, dry_run=False, upload_strategy="url")


@pytest.fixture
def resumable(settings):
    """Resumable-strategy settings with DRY_RUN disabled."""
    return dataclasses.replace(settings, dry_run=False, upload_strategy="resumable")


@pytest.fixture
def video_file(tmp_path):
    """A small stand-in MP4 on disk."""
    path = tmp_path / "reel.mp4"
    path.write_bytes(b"fake mp4 bytes" * 64)
    return path


@pytest.fixture
def urls(live):
    """The three endpoints the reels flow touches."""
    root = base_url(live)
    return {
        "create": f"{root}/{live.ig_user_id}/media",
        "status": f"{root}/container-1",
        "publish": f"{root}/{live.ig_user_id}/media_publish",
    }


class TestDryRun:
    async def test_dry_run_makes_no_http_call(self, settings):
        """DRY_RUN must short-circuit before any network access."""
        with respx.mock(assert_all_called=False) as mock:
            catch_all = mock.route(host="graph.facebook.com")
            post_id = await publish_reel(settings, CAPTION, video_url=VIDEO_URL)

        assert post_id == "dry-run"
        assert not catch_all.called, "DRY_RUN must not touch the Graph API"

    async def test_dry_run_does_not_require_credentials(self, settings):
        bare = dataclasses.replace(settings, ig_user_id="", ig_access_token="")
        assert await publish_reel(bare, CAPTION, video_url=VIDEO_URL) == "dry-run"

    async def test_live_run_without_credentials_raises(self, live):
        bare = dataclasses.replace(live, ig_user_id="", ig_access_token="")
        with pytest.raises(InstagramError, match="IG_USER_ID"):
            await publish_reel(bare, CAPTION, video_url=VIDEO_URL)


class TestContainerCreation:
    @respx.mock
    async def test_sends_the_documented_reels_parameters(self, live, urls):
        route = respx.post(urls["create"]).mock(
            return_value=httpx.Response(200, json={"id": "container-1"})
        )
        async with httpx.AsyncClient() as client:
            container_id = await create_reel_container(
                client, live, VIDEO_URL, CAPTION
            )

        assert container_id == "container-1"
        body = dict(
            pair.split("=", 1)
            for pair in route.calls[0].request.content.decode().split("&")
        )
        assert body["media_type"] == "REELS"
        assert "video_url" in body
        assert "access_token" in body

    @respx.mock
    async def test_api_error_body_is_surfaced(self, live, urls):
        """Meta puts the real diagnosis in the body, so it must reach the user."""
        respx.post(urls["create"]).mock(
            return_value=httpx.Response(
                400,
                json={
                    "error": {
                        "message": "The video file is not accessible",
                        "code": 9004,
                    }
                },
            )
        )
        async with httpx.AsyncClient() as client:
            with pytest.raises(InstagramError) as excinfo:
                await create_reel_container(client, live, VIDEO_URL, CAPTION)

        assert "not accessible" in str(excinfo.value)
        assert "9004" in str(excinfo.value)

    @respx.mock
    async def test_missing_id_raises(self, live, urls):
        respx.post(urls["create"]).mock(return_value=httpx.Response(200, json={}))
        async with httpx.AsyncClient() as client:
            with pytest.raises(InstagramError, match="no id"):
                await create_reel_container(client, live, VIDEO_URL, CAPTION)


class TestStatusPolling:
    @respx.mock
    async def test_polls_until_finished(self, live, urls):
        respx.get(urls["status"]).mock(
            side_effect=[
                httpx.Response(200, json={"status_code": "IN_PROGRESS"}),
                httpx.Response(200, json={"status_code": "IN_PROGRESS"}),
                httpx.Response(200, json={"status_code": "FINISHED"}),
            ]
        )
        async with httpx.AsyncClient() as client:
            await wait_for_container(client, live, "container-1")

    @pytest.mark.parametrize("status", ["ERROR", "EXPIRED"])
    @respx.mock
    async def test_terminal_failure_states_raise(self, live, urls, status):
        respx.get(urls["status"]).mock(
            return_value=httpx.Response(200, json={"status_code": status})
        )
        async with httpx.AsyncClient() as client:
            with pytest.raises(InstagramError, match=status):
                await wait_for_container(client, live, "container-1")

    @respx.mock
    async def test_timeout_raises_rather_than_hanging(self, live, urls):
        respx.get(urls["status"]).mock(
            return_value=httpx.Response(200, json={"status_code": "IN_PROGRESS"})
        )
        stingy = dataclasses.replace(live, ig_poll_timeout=0, ig_poll_interval=0)
        async with httpx.AsyncClient() as client:
            with pytest.raises(InstagramError, match="still IN_PROGRESS"):
                await wait_for_container(client, stingy, "container-1")


class TestPublish:
    @respx.mock
    async def test_publish_sends_creation_id(self, live, urls):
        route = respx.post(urls["publish"]).mock(
            return_value=httpx.Response(200, json={"id": "17900000000000000"})
        )
        async with httpx.AsyncClient() as client:
            media_id = await publish_container(client, live, "container-1")

        assert media_id == "17900000000000000"
        assert "creation_id=container-1" in route.calls[0].request.content.decode()

    @respx.mock
    async def test_full_reels_flow_in_order(self, live, urls):
        """Create, poll to FINISHED, then publish - in that order."""
        create = respx.post(urls["create"]).mock(
            return_value=httpx.Response(200, json={"id": "container-1"})
        )
        status = respx.get(urls["status"]).mock(
            side_effect=[
                httpx.Response(200, json={"status_code": "IN_PROGRESS"}),
                httpx.Response(200, json={"status_code": "FINISHED"}),
            ]
        )
        publish = respx.post(urls["publish"]).mock(
            return_value=httpx.Response(200, json={"id": "17900000000000000"})
        )

        media_id = await publish_reel(live, CAPTION, video_url=VIDEO_URL)

        assert media_id == "17900000000000000"
        assert create.call_count == 1
        assert status.call_count == 2
        assert publish.call_count == 1

    @respx.mock
    async def test_does_not_publish_a_failed_container(self, live, urls):
        respx.post(urls["create"]).mock(
            return_value=httpx.Response(200, json={"id": "container-1"})
        )
        respx.get(urls["status"]).mock(
            return_value=httpx.Response(200, json={"status_code": "ERROR"})
        )
        publish = respx.post(urls["publish"]).mock(
            return_value=httpx.Response(200, json={"id": "nope"})
        )

        with pytest.raises(InstagramError, match="ERROR"):
            await publish_reel(live, CAPTION, video_url=VIDEO_URL)

        assert not publish.called, "a failed container must never be published"


class TestResumableUpload:
    """The default strategy: bytes straight to Meta, nothing publicly hosted."""

    @respx.mock
    async def test_session_creation_sets_upload_type(self, resumable, urls):
        route = respx.post(urls["create"]).mock(
            return_value=httpx.Response(200, json={"id": "container-1"})
        )
        async with httpx.AsyncClient() as client:
            container_id = await create_resumable_container(
                client, resumable, CAPTION
            )

        assert container_id == "container-1"
        body = route.calls[0].request.content.decode()
        assert "upload_type=resumable" in body
        assert "media_type=REELS" in body
        assert "video_url" not in body, "resumable must not ask Meta to fetch"

    @respx.mock
    async def test_upload_sends_the_documented_headers(
        self, resumable, video_file
    ):
        route = respx.post(
            f"{RUPLOAD_HOST}/ig-api-upload/{resumable.ig_api_version}/container-1"
        ).mock(return_value=httpx.Response(200, json={"success": True}))

        async with httpx.AsyncClient() as client:
            await upload_video_bytes(client, resumable, "container-1", video_file)

        headers = route.calls[0].request.headers
        assert headers["authorization"] == f"OAuth {resumable.ig_access_token}"
        assert headers["offset"] == "0"
        assert headers["file_size"] == str(video_file.stat().st_size)
        assert route.calls[0].request.content == video_file.read_bytes()

    @respx.mock
    async def test_unsuccessful_upload_raises(self, resumable, video_file):
        respx.post(
            f"{RUPLOAD_HOST}/ig-api-upload/{resumable.ig_api_version}/container-1"
        ).mock(
            return_value=httpx.Response(
                200, json={"debug_info": {"retriable": False, "type": "BadRequest"}}
            )
        )
        async with httpx.AsyncClient() as client:
            with pytest.raises(InstagramError, match="did not succeed"):
                await upload_video_bytes(
                    client, resumable, "container-1", video_file
                )

    async def test_missing_file_raises(self, resumable, tmp_path):
        async with httpx.AsyncClient() as client:
            with pytest.raises(InstagramError, match="not found"):
                await upload_video_bytes(
                    client, resumable, "container-1", tmp_path / "nope.mp4"
                )

    async def test_oversized_file_is_rejected_before_uploading(
        self, resumable, video_file, monkeypatch
    ):
        """300MB is Meta's ceiling; fail fast rather than after the upload."""
        monkeypatch.setattr("tools.instagram.MAX_UPLOAD_BYTES", 10)
        async with httpx.AsyncClient() as client:
            with pytest.raises(InstagramError, match="limit is"):
                await upload_video_bytes(
                    client, resumable, "container-1", video_file
                )

    @respx.mock
    async def test_full_resumable_flow_never_needs_a_url(
        self, resumable, urls, video_file
    ):
        create = respx.post(urls["create"]).mock(
            return_value=httpx.Response(200, json={"id": "container-1"})
        )
        upload = respx.post(
            f"{RUPLOAD_HOST}/ig-api-upload/{resumable.ig_api_version}/container-1"
        ).mock(return_value=httpx.Response(200, json={"success": True}))
        respx.get(urls["status"]).mock(
            return_value=httpx.Response(200, json={"status_code": "FINISHED"})
        )
        publish = respx.post(urls["publish"]).mock(
            return_value=httpx.Response(200, json={"id": "17900000000000000"})
        )

        media_id = await publish_reel(
            resumable, CAPTION, local_path=video_file, video_url=None
        )

        assert media_id == "17900000000000000"
        assert create.called and upload.called and publish.called

    async def test_resumable_without_a_local_file_raises(self, resumable):
        with pytest.raises(InstagramError, match="needs a local file"):
            await publish_reel(resumable, CAPTION, video_url=VIDEO_URL)

    async def test_url_strategy_without_a_url_raises(self, live, video_file):
        with pytest.raises(InstagramError, match="publicly fetchable"):
            await publish_reel(live, CAPTION, local_path=video_file)


class TestInstagramLoginHost:
    """A token from Instagram Login must be sent to graph.instagram.com."""

    @pytest.fixture
    def ig_login(self, live):
        return dataclasses.replace(live, ig_graph_host="https://graph.instagram.com")

    def test_the_default_host_is_unchanged(self, live):
        assert base_url(live) == "https://graph.facebook.com/v25.0"

    def test_the_host_comes_from_settings(self, ig_login):
        assert base_url(ig_login) == "https://graph.instagram.com/v25.0"

    def test_a_trailing_slash_in_the_host_is_harmless(self, live):
        slashed = dataclasses.replace(live, ig_graph_host="https://graph.instagram.com/")
        assert base_url(slashed) == "https://graph.instagram.com/v25.0"

    @respx.mock
    async def test_a_full_publish_never_touches_the_facebook_host(self, ig_login):
        root = base_url(ig_login)
        respx.post(f"{root}/{ig_login.ig_user_id}/media").mock(
            return_value=httpx.Response(200, json={"id": "c1"})
        )
        respx.get(f"{root}/c1").mock(
            return_value=httpx.Response(200, json={"status_code": "FINISHED"})
        )
        respx.post(f"{root}/{ig_login.ig_user_id}/media_publish").mock(
            return_value=httpx.Response(200, json={"id": "999"})
        )
        facebook = respx.route(host="graph.facebook.com")

        assert await publish_reel(ig_login, CAPTION, video_url=VIDEO_URL) == "999"
        assert not facebook.called
