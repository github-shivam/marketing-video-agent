"""Shared fixtures and fakes.

Nothing here touches a model, the network, AWS or Instagram. Every agent is
replaced by a deterministic async stub that records how it was called.
"""

from __future__ import annotations

import dataclasses
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Settings  # noqa: E402
from graph import Nodes  # noqa: E402
from state import (  # noqa: E402
    AssetBrief,
    AudioLook,
    Brief,
    Critique,
    Dialogue,
    Media,
    PersonLook,
    PostState,
    Screenplay,
    Segment,
    Shot,
    UploadAnalysis,
)


@pytest.fixture
def settings() -> Settings:
    """Settings with the documented defaults and no credentials."""
    return Settings(
        dry_run=True,
        model="test-model",
        bedrock_text_region="us-east-1",
        model_temperature=0.7,
        model_classify="",
        model_research="",
        model_writer="",
        model_critic="",
        model_dialogue="",
        model_converse="",
        script_language="English",
        speech_pace_wpm=175,
        score_threshold=8.0,
        max_iterations=3,
        video_model="veo",
        video_seconds_default=15,
        gcp_project="",
        gcp_location="us-central1",
        veo_gcs_bucket="",
        veo_model="veo-3.1-fast-generate-001",
        veo_resolution="720p",
        veo_timestamps=True,
        veo_aspect_ratio="9:16",
        veo_person_generation="allow_all",
        veo_image="",
        veo_block_retries=2,
        veo_poll_interval=0,
        veo_timeout=5,
        asset_model="test-asset-model",
        uploads_dir=Path("uploads"),
        brightdata_api_token="",
        brightdata_groups="",
        brightdata_tools="search_engine,scrape_as_markdown",
        brightdata_rate_limit="100/1h",
        brightdata_max_retries=2,
        upload_strategy="resumable",
        s3_bucket="",  # empty by default so no test can reach AWS
        s3_prefix="test",
        s3_region="us-east-1",
        presign_expiry=3600,
        ig_user_id="17841400000000000",
        ig_access_token="test-token",
        ig_api_version="v25.0",
        ig_graph_host="https://graph.facebook.com",
        ig_poll_interval=0,
        ig_poll_timeout=5,
        # Blank so the WhatsApp routes do not exist unless a test opts in.
        whatsapp_access_token="",
        whatsapp_phone_number_id="",
        whatsapp_app_secret="",
        whatsapp_verify_token="",
        whatsapp_api_version="v25.0",
        whatsapp_allowed_numbers="",
        whatsapp_max_runs_per_day=10,
        api_key="test-api-key",
        checkpoint_db=":memory:",
        conversations_db=":memory:",
        output_dir=Path("output"),
        log_level="INFO",
    )


BUCKET = "my-veo-bucket"


@pytest.fixture
def live(settings: Settings) -> Settings:
    """Settings with Veo configured (a project and a bucket), no poll delay."""
    return dataclasses.replace(
        settings,
        gcp_project="my-project",
        gcp_location="us-central1",
        veo_gcs_bucket=BUCKET,
        veo_poll_interval=0,
    )


class FakeStorageClient:
    """Stands in for ``google.cloud.storage.Client``.

    Writes ``payload`` to whatever file it is asked to download into.
    """

    downloads: list[dict[str, Any]] = []
    projects: list[str | None] = []
    payload: bytes = b"vertex mp4 payload"

    def __init__(self, project: str | None = None) -> None:
        type(self).projects.append(project)
        self._bucket = ""
        self._blob = ""

    def bucket(self, name: str):
        self._bucket = name
        return self

    def blob(self, name: str):
        self._blob = name
        return self

    def download_to_filename(self, filename: str) -> None:
        type(self).downloads.append(
            {"bucket": self._bucket, "blob": self._blob, "filename": filename}
        )
        with open(filename, "wb") as handle:
            handle.write(type(self).payload)


@pytest.fixture
def storage(monkeypatch):
    """Replace Cloud Storage with a recorder, so no test can reach GCP."""
    FakeStorageClient.downloads = []
    FakeStorageClient.projects = []
    FakeStorageClient.payload = b"vertex mp4 payload"
    monkeypatch.setattr("google.cloud.storage.Client", FakeStorageClient)
    return FakeStorageClient


@pytest.fixture
def brief() -> Brief:
    """A classified request, as the classify node would produce it."""
    return Brief(
        request="a promo for Ridgeline Coffee",
        intent="product_promo",
        subject="Ridgeline Coffee's roasted-to-order single-origin beans",
        audience="home brewers aged 25-40",
        tone="warm, plain-spoken, no hype",
        key_message="Coffee roasted to order tastes different",
        cta="Order a bag at ridgeline.coffee",
        visual_style="warm editorial product film",
    )


def make_shots(start: int = 0, end: int = 8) -> list[Shot]:
    """Two valid shots that tile ``start``-``end`` seconds."""
    mid = (start + end) // 2
    return [
        Shot(
            start=start,
            end=mid,
            camera="Close-up, slow dolly in",
            subject="roasted coffee beans in a copper cooling tray",
            action="tumble as a wooden paddle stirs them",
            setting="a sunlit roastery at dawn",
            ambient="the soft rattle of beans and a distant roaster hum",
        ),
        Shot(
            start=mid,
            end=end,
            camera="Medium shot, static",
            subject="a barista in a green apron",
            action="pours hot water over a pour-over dripper",
            setting="a wooden counter in warm morning light",
            sfx="water splashing onto grounds",
        ),
    ]


def make_screenplay(hook: str = "Your coffee deserves better", segments: int = 2) -> Screenplay:
    """A valid screenplay with ``segments`` generation calls (8s, then 7s)."""
    lengths = [8] + [7] * (segments - 1)
    return Screenplay(
        title="Roasted to order",
        logline="Beans go from roaster to cup in one unbroken morning.",
        hook=hook,
        style="warm editorial product film, soft natural light, shallow depth of field",
        continuity="A barista in her thirties with short dark hair wearing a green apron.",
        segments=[
            Segment(index=i, shots=make_shots(0, n)) for i, n in enumerate(lengths)
        ],
        cta="Order a bag today",
    )


def make_dialogue() -> Dialogue:
    """A valid caption and hashtag set."""
    return Dialogue(
        caption="Roasted to order, shipped the same day.",
        hashtags=["coffee", "coldbrew", "specialtycoffee", "roastery", "homebrew"],
    )


def make_asset_brief(
    *, with_person: bool = True, with_audio: bool = True, references: list[str] | None = None
) -> AssetBrief:
    """What the analyze node would learn from a few uploads."""
    return AssetBrief(
        analysis=UploadAnalysis(
            theme="a small bakery at dawn",
            mood="warm, calm",
            setting="a stone bakery, early morning",
            visual_style="soft window light, film grain",
            color_words="warm amber, cream and deep teal",
            motion="slow, handheld",
            people=[
                PersonLook(
                    label="the baker in the green apron",
                    description="a man in his forties, short grey beard, green linen apron",
                    item_indexes=[0],
                    is_main=True,
                )
            ]
            if with_person
            else [],
            audio=AudioLook(
                kind="music", mood="nostalgic", genre="acoustic folk",
                tempo="slow, about 70 bpm", instruments="guitar and cello",
                language="", transcript="", summary="a slow acoustic folk tune",
            )
            if with_audio
            else AudioLook(
                kind="none", mood="", genre="", tempo="", instruments="",
                language="", transcript="", summary="",
            ),
            reference_items=[0] if with_person else [],
        ),
        palette=["#E8A33D", "#F5E6D0", "#1F6F78"],
        reference_files=references or [],
        n_images=2,
        n_audio=1 if with_audio else 0,
    )


@dataclass
class FakeAgents:
    """Deterministic stand-ins for the seven agent nodes.

    ``scores`` is consumed one entry per critic call, so a test can drive the
    writer/critic loop through an exact sequence of verdicts.
    """

    scores: list[float] = field(default_factory=lambda: [9.0])
    calls: list[str] = field(default_factory=list)
    writer_saw_feedback: list[str | None] = field(default_factory=list)
    writer_saw_duration: list[int | None] = field(default_factory=list)
    analyzed: list[list] = field(default_factory=list)
    analysis_error: str | None = None
    publish_called_with: dict[str, Any] = field(default_factory=dict)

    async def classify(self, state: PostState) -> dict[str, Any]:
        self.calls.append("classify")
        return {
            "brief": Brief(
                request=state["request"],
                intent="product_promo",
                subject="Ridgeline Coffee",
                audience="home brewers",
                tone="warm",
                key_message="Roasted to order",
                visual_style="warm editorial",
            ),
            "source_text": "",
        }

    async def analyze(self, state: PostState) -> dict[str, Any]:
        self.calls.append("analyze")
        self.analyzed.append(list(state.get("assets") or []))
        if self.analysis_error:
            return {"asset_brief": None, "asset_error": self.analysis_error}
        return {"asset_brief": make_asset_brief(), "asset_error": None}

    async def research(self, state: PostState) -> dict[str, Any]:
        self.calls.append("research")
        return {"research_notes": "Competitors post short, quiet product reels."}

    async def writer(self, state: PostState) -> dict[str, Any]:
        self.calls.append("writer")
        self.writer_saw_feedback.append(state.get("human_feedback"))
        self.writer_saw_duration.append(state.get("duration_seconds"))
        return {
            "screenplay": make_screenplay(),
            "prompts": ["PROMPT ZERO", "PROMPT ONE"],
            "iterations": (state.get("iterations") or 0) + 1,
        }

    async def critic(self, state: PostState) -> dict[str, Any]:
        self.calls.append("critic")
        score = self.scores.pop(0) if self.scores else 9.0
        return {
            "score": score,
            "critique": Critique(
                score=score,
                hook="Hook assessment",
                intent_fit="Intent fit assessment",
                prompt_craft="Craft assessment",
                feedback="Tighten the opening line",
            ),
        }

    async def dialogue(self, state: PostState) -> dict[str, Any]:
        self.calls.append("dialogue")
        return {"dialogue": make_dialogue()}

    async def video(self, state: PostState) -> dict[str, Any]:
        self.calls.append("video")
        return {
            "media": Media(
                media_type="REELS",
                urls=["https://example.invalid/presigned/reel.mp4"],
                local_files=["output/reel.mp4"],
                prompt_file="output/reel.prompt.txt",
                duration_seconds=15,
            )
        }

    async def publish(self, state: PostState) -> dict[str, Any]:
        self.calls.append("publish")
        self.publish_called_with = dict(state)
        return {"post_id": "dry-run"}

    def as_nodes(self) -> Nodes:
        """Expose the stubs in the shape ``build_graph`` expects."""
        return Nodes(
            classify=self.classify,
            research=self.research,
            writer=self.writer,
            critic=self.critic,
            dialogue=self.dialogue,
            video=self.video,
            publish=self.publish,
            analyze=self.analyze,
        )


@pytest.fixture
def agents() -> FakeAgents:
    """A fresh set of fake agents per test."""
    return FakeAgents()
