"""HTTP API tests: real FastAPI app and run manager, fake agents.

The graph, checkpointer, background tasks and routes are all real; only the
agent nodes are fakes, so no model, network or Instagram is touched.
"""

from __future__ import annotations

import asyncio
import dataclasses
import re
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from api import create_app
from state import Media

from conftest import FakeAgents

KEY = {"X-API-Key": "test-api-key"}
ID = re.compile(r"^(veo|seedance)-[0-9a-f]{12}$")


@pytest.fixture
def agents_with_video(tmp_path):
    """Fake agents whose video node writes a real file, so it can be downloaded."""
    agents = FakeAgents()

    async def video(state):
        agents.calls.append("video")
        path = tmp_path / "reel.mp4"
        path.write_bytes(b"fake mp4 payload")
        return {
            "media": Media(
                local_files=[str(path)],
                prompt_file=str(tmp_path / "reel.prompt.txt"),
                duration_seconds=15,
            )
        }

    agents.video = video
    return agents


def make_client(agents, settings, checkpointer=None):
    app = create_app(
        settings, lambda cfg: agents.as_nodes(), checkpointer or InMemorySaver()
    )
    return TestClient(app, headers=KEY)


def wait_for(client, video_id, status, timeout=5.0):
    """Poll until the run reaches ``status``, returning the last body."""
    deadline = time.monotonic() + timeout
    body = {}
    while time.monotonic() < deadline:
        body = client.get(f"/videos/{video_id}").json()
        if body.get("status") == status:
            return body
        time.sleep(0.02)
    raise AssertionError(f"never reached {status!r}; last body: {body}")


def start(client, request="explain how RAG works", **extra):
    response = client.post("/videos", json={"request": request, **extra})
    assert response.status_code == 202, response.text
    return response.json()["video_id"]


class TestAuth:
    def test_health_needs_no_key(self, settings, agents):
        with TestClient(create_app(settings, lambda c: agents.as_nodes(), InMemorySaver())) as c:
            assert c.get("/health").json()["status"] == "ok"

    @pytest.mark.parametrize(
        "method,path",
        [
            ("post", "/videos"),
            ("get", "/videos/veo-aaaaaaaaaaaa"),
            ("post", "/videos/veo-aaaaaaaaaaaa/approve"),
            ("get", "/videos/veo-aaaaaaaaaaaa/video"),
        ],
    )
    def test_every_route_rejects_a_missing_or_wrong_key(self, settings, agents, method, path):
        app = create_app(settings, lambda c: agents.as_nodes(), InMemorySaver())
        with TestClient(app) as client:
            assert getattr(client, method)(path).status_code == 401
            wrong = getattr(client, method)(path, headers={"X-API-Key": "nope"})
            assert wrong.status_code == 401

    def test_the_service_refuses_to_start_without_a_key(self, settings):
        with pytest.raises(RuntimeError, match="API_KEY"):
            create_app(dataclasses.replace(settings, api_key=""))


class TestStartingARun:
    def test_returns_202_and_an_id_immediately(self, settings, agents):
        with make_client(agents, settings) as client:
            response = client.post("/videos", json={"request": "explain RAG"})

        assert response.status_code == 202
        body = response.json()
        assert ID.match(body["video_id"])
        assert body["status_url"] == f"/videos/{body['video_id']}"

    def test_reaches_the_prompt_gate_with_the_exact_prompts(self, settings, agents):
        with make_client(agents, settings) as client:
            body = wait_for(client, start(client), "awaiting_prompt_approval")

        assert body["awaiting"]["stage"] == "script"
        assert body["awaiting"]["prompts"] == ["PROMPT ZERO", "PROMPT ONE"]
        assert body["classification"]["intent"] == "product_promo"
        assert "video" not in agents.calls, "nothing is spent before approval"

    def test_the_model_chooses_the_graph_and_prefixes_the_id(self, settings, agents):
        with make_client(agents, settings) as client:
            assert start(client, model="seedance").startswith("seedance-")
            assert start(client).startswith("veo-")

    @pytest.mark.parametrize(
        "body",
        [{}, {"request": ""}, {"request": "x" * 2001}, {"request": "ok", "model": "sora"}],
    )
    def test_bad_input_is_a_422(self, settings, agents, body):
        with make_client(agents, settings) as client:
            assert client.post("/videos", json=body).status_code == 422


class TestVideoLength:
    def test_a_run_can_ask_for_its_length(self, settings, agents):
        with make_client(agents, settings) as client:
            video_id = start(client, seconds=22)
            body = wait_for(client, video_id, "awaiting_prompt_approval")

        assert body["duration_seconds"] == 22
        assert body["awaiting"]["duration_seconds"] == 22
        assert agents.writer_saw_duration == [22]

    def test_the_default_is_fifteen_seconds(self, settings, agents):
        with make_client(agents, settings) as client:
            body = wait_for(client, start(client), "awaiting_prompt_approval")

        assert body["duration_seconds"] == 15

    def test_a_length_the_plan_cannot_make_exactly_is_stored_as_made(self, settings, agents):
        """30s is made as 29: the status reports the length the video will be."""
        with make_client(agents, settings) as client:
            body = wait_for(client, start(client, seconds=30), "awaiting_prompt_approval")

        assert body["duration_seconds"] == 29

    @pytest.mark.parametrize("bad", [0, 3, 31, 120, -1])
    def test_a_length_out_of_range_is_a_422(self, settings, agents, bad):
        with make_client(agents, settings) as client:
            response = client.post("/videos", json={"request": "x", "seconds": bad})

        assert response.status_code == 422
        assert agents.calls == [], "nothing ran"

    def test_a_rejection_can_change_the_length(self, settings, agents):
        with make_client(agents, settings) as client:
            video_id = start(client, seconds=15)
            wait_for(client, video_id, "awaiting_prompt_approval")
            response = client.post(
                f"/videos/{video_id}/approve",
                json={"approved": False, "feedback": "make it longer", "seconds": 25},
            )
            assert response.status_code == 202
            deadline = time.monotonic() + 5
            while len(agents.writer_saw_duration) < 2 and time.monotonic() < deadline:
                time.sleep(0.02)
            body = wait_for(client, video_id, "awaiting_prompt_approval")

        assert agents.writer_saw_duration == [15, 25]
        assert body["duration_seconds"] == 25

    def test_a_decision_with_a_length_out_of_range_is_a_422(self, settings, agents):
        with make_client(agents, settings) as client:
            video_id = start(client)
            wait_for(client, video_id, "awaiting_prompt_approval")
            response = client.post(
                f"/videos/{video_id}/approve", json={"approved": False, "seconds": 99}
            )

        assert response.status_code == 422


class TestGates:
    def test_full_flow_to_a_downloadable_video(self, settings, agents_with_video):
        with make_client(agents_with_video, settings) as client:
            vid = start(client)
            wait_for(client, vid, "awaiting_prompt_approval")

            assert client.post(f"/videos/{vid}/approve", json={"approved": True}).status_code == 202
            gate = wait_for(client, vid, "awaiting_post_decision")
            assert gate["awaiting"]["stage"] == "video"
            assert gate["awaiting"]["duration_seconds"] == 15

            client.post(f"/videos/{vid}/approve", json={"approved": False})
            done = wait_for(client, vid, "completed")

            assert done["result"]["video_url"] == f"/videos/{vid}/video"
            assert done["result"]["prompts"] == ["PROMPT ZERO", "PROMPT ONE"]
            assert done["result"]["post_id"] is None
            video = client.get(done["result"]["video_url"])
            assert video.status_code == 200
            assert video.headers["content-type"] == "video/mp4"
            assert video.content == b"fake mp4 payload"
        assert "publish" not in agents_with_video.calls

    def test_approving_the_post_gate_publishes(self, settings, agents_with_video):
        with make_client(agents_with_video, settings) as client:
            vid = start(client)
            wait_for(client, vid, "awaiting_prompt_approval")
            client.post(f"/videos/{vid}/approve", json={"approved": True})
            wait_for(client, vid, "awaiting_post_decision")
            client.post(f"/videos/{vid}/approve", json={"approved": True})

            assert wait_for(client, vid, "posted")["result"]["post_id"] == "dry-run"

    def test_rejecting_with_feedback_sends_it_back_to_the_writer(self, settings, agents):
        with make_client(agents, settings) as client:
            vid = start(client)
            wait_for(client, vid, "awaiting_prompt_approval")
            client.post(
                f"/videos/{vid}/approve",
                json={"approved": False, "feedback": "make the hook sharper"},
            )

            wait_for(client, vid, "awaiting_prompt_approval")
        assert agents.calls.count("writer") == 2
        assert "make the hook sharper" in agents.writer_saw_feedback
        assert "video" not in agents.calls

    def test_rejecting_without_feedback_ends_the_run(self, settings, agents):
        with make_client(agents, settings) as client:
            vid = start(client)
            wait_for(client, vid, "awaiting_prompt_approval")
            client.post(f"/videos/{vid}/approve", json={"approved": False})

            ended = wait_for(client, vid, "ended")
        assert "result" not in ended
        assert "video" not in agents.calls

    def test_answering_when_not_at_a_gate_is_a_409(self, settings, agents_with_video):
        with make_client(agents_with_video, settings) as client:
            vid = start(client)
            wait_for(client, vid, "awaiting_prompt_approval")
            client.post(f"/videos/{vid}/approve", json={"approved": False})
            wait_for(client, vid, "ended")

            assert client.post(f"/videos/{vid}/approve", json={"approved": True}).status_code == 409

    def test_answering_while_the_run_is_busy_is_a_409(self, settings, agents_with_video):
        release = threading.Event()
        original = agents_with_video.video

        async def slow_video(state):
            while not release.is_set():
                await asyncio.sleep(0.01)
            return await original(state)

        agents_with_video.video = slow_video
        with make_client(agents_with_video, settings) as client:
            vid = start(client)
            wait_for(client, vid, "awaiting_prompt_approval")
            client.post(f"/videos/{vid}/approve", json={"approved": True})

            assert wait_for(client, vid, "generating")["status"] == "generating"
            twice = client.post(f"/videos/{vid}/approve", json={"approved": True})
            release.set()

            assert twice.status_code == 409
            wait_for(client, vid, "awaiting_post_decision")


class TestFailures:
    def test_a_crashing_node_is_reported_as_failed(self, settings, agents):
        async def boom(state):
            raise RuntimeError("bedrock exploded")

        agents.research = boom
        with make_client(agents, settings) as client:
            body = wait_for(client, start(client), "failed")

        assert "bedrock exploded" in body["error"]

    def test_a_failed_video_is_reported_not_offered_for_approval(self, settings, agents):
        async def failing_video(state):
            return {"error": "video generation failed: quota exceeded"}

        agents.video = failing_video
        with make_client(agents, settings) as client:
            vid = start(client)
            wait_for(client, vid, "awaiting_prompt_approval")
            client.post(f"/videos/{vid}/approve", json={"approved": True})

            assert "quota exceeded" in wait_for(client, vid, "failed")["error"]

    def test_an_unusable_request_fails_without_researching(self, settings, agents):
        async def reject(state):
            return {"error": "empty request: give a topic or a link"}

        agents.classify = reject
        with make_client(agents, settings) as client:
            assert "empty request" in wait_for(client, start(client), "failed")["error"]
        assert "research" not in agents.calls


class TestLookups:
    @pytest.mark.parametrize(
        "video_id",
        ["veo-000000000000", "nonsense", "../etc/passwd", "sora-aaaaaaaaaaaa"],
    )
    def test_unknown_or_malformed_ids_are_404(self, settings, agents, video_id):
        with make_client(agents, settings) as client:
            assert client.get(f"/videos/{video_id}").status_code == 404

    def test_downloading_before_a_video_exists_is_a_409(self, settings, agents):
        with make_client(agents, settings) as client:
            vid = start(client)
            wait_for(client, vid, "awaiting_prompt_approval")

            assert client.get(f"/videos/{vid}/video").status_code == 409

    def test_a_missing_file_is_a_409_not_a_500(self, settings, agents_with_video):
        with make_client(agents_with_video, settings) as client:
            vid = start(client)
            wait_for(client, vid, "awaiting_prompt_approval")
            client.post(f"/videos/{vid}/approve", json={"approved": True})
            wait_for(client, vid, "awaiting_post_decision")
            Path(client.get(f"/videos/{vid}").json()["awaiting"]["files"][0]).unlink()

            assert client.get(f"/videos/{vid}/video").status_code == 409


class TestSurvivingARestart:
    def test_a_run_paused_in_one_process_is_answered_by_another(
        self, settings, agents_with_video
    ):
        """State lives in the checkpointer, not in the server's memory."""
        saver = InMemorySaver()
        with make_client(agents_with_video, settings, saver) as first:
            vid = start(first)
            wait_for(first, vid, "awaiting_prompt_approval")

        with make_client(agents_with_video, settings, saver) as second:
            assert second.get(f"/videos/{vid}").json()["status"] == "awaiting_prompt_approval"
            assert second.post(f"/videos/{vid}/approve", json={"approved": True}).status_code == 202
            wait_for(second, vid, "awaiting_post_decision")
        assert agents_with_video.calls.count("classify") == 1, "must not start over"
