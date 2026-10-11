"""WhatsApp chat front end.

The run manager, graph and checkpointer are real. The agents are the usual
fakes, the WhatsApp client is a recorder, and the interpreter is scripted, so
no model, no network and no Meta account is touched.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import hmac
import json
import time
from pathlib import Path

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

import api as api_module
from agents import converse
from agents.converse import Reply, interpret
from api import create_app
from conversations import ConversationStore
from graph import Nodes
from runs import RunManager
from state import Media
from tools.whatsapp import (
    Incoming,
    WhatsAppClient,
    WhatsAppError,
    chunk_text,
    is_allowed,
    parse_incoming,
    verify_signature,
)
from whatsapp_bot import WhatsAppBot

from conftest import FakeAgents

ALICE = "919800000001"
BOB = "919800000002"
STRANGER = "919811111111"
SECRET = "wa-app-secret"


def enable_whatsapp(settings, allowed=f"{ALICE},{BOB}", **extra):
    return dataclasses.replace(
        settings,
        whatsapp_access_token="wa-token",
        whatsapp_phone_number_id="1055550000",
        whatsapp_app_secret=SECRET,
        whatsapp_verify_token="wa-verify",
        whatsapp_allowed_numbers=allowed,
        **extra,
    )


def sign(body: bytes, secret: str = SECRET) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


# ── Pure helpers ──────────────────────────────────────────────────────────


class TestSignature:
    def test_accepts_a_correct_signature(self):
        body = b'{"object":"whatsapp_business_account"}'
        assert verify_signature(SECRET, body, sign(body))

    @pytest.mark.parametrize("header", [None, "", "sha256=", "sha256=deadbeef", "md5=abc"])
    def test_rejects_a_missing_or_wrong_signature(self, header):
        assert not verify_signature(SECRET, b"{}", header)

    def test_rejects_a_body_that_was_changed_after_signing(self):
        assert not verify_signature(SECRET, b'{"a":2}', sign(b'{"a":1}'))

    def test_rejects_everything_when_no_secret_is_configured(self):
        assert not verify_signature("", b"{}", sign(b"{}", ""))


class TestParseIncoming:
    def payload(self, *messages):
        return {"entry": [{"changes": [{"value": {"messages": list(messages)}}]}]}

    def test_reads_a_text_message(self):
        got = parse_incoming(
            self.payload({"from": ALICE, "id": "wamid.1", "type": "text", "text": {"body": "hi"}})
        )
        assert got == [Incoming(ALICE, "wamid.1", "text", "hi")]

    def test_keeps_the_kind_of_a_message_it_cannot_read(self):
        got = parse_incoming(self.payload({"from": ALICE, "id": "wamid.2", "type": "audio"}))
        assert got[0].kind == "audio" and got[0].text == ""

    @pytest.mark.parametrize("kind", ["image", "video", "audio"])
    def test_reads_the_file_a_media_message_carries(self, kind):
        got = parse_incoming(
            self.payload(
                {
                    "from": ALICE, "id": "wamid.m", "type": kind,
                    kind: {"id": "MEDIA9", "mime_type": f"{kind}/x", "caption": "my shop"},
                }
            )
        )
        assert got == [Incoming(ALICE, "wamid.m", kind, "", "MEDIA9", f"{kind}/x", "my shop")]

    def test_a_photo_with_no_caption_has_an_empty_one(self):
        (got,) = parse_incoming(
            self.payload({"from": ALICE, "id": "1", "type": "image", "image": {"id": "M", "mime_type": "image/jpeg"}})
        )
        assert got.caption == "" and got.media_id == "M"

    def test_a_voice_note_is_audio(self):
        (got,) = parse_incoming(
            self.payload(
                {"from": ALICE, "id": "1", "type": "audio",
                 "audio": {"id": "V", "mime_type": "audio/ogg; codecs=opus", "voice": True}}
            )
        )
        assert got.kind == "audio" and got.mime_type == "audio/ogg; codecs=opus"

    @pytest.mark.parametrize("media", [None, "oops", [], 5])
    def test_a_malformed_media_block_does_not_crash(self, media):
        (got,) = parse_incoming(self.payload({"from": ALICE, "id": "1", "type": "image", "image": media}))
        assert got.kind == "image" and got.media_id == ""

    def test_ignores_delivery_receipts(self):
        receipt = {"entry": [{"changes": [{"value": {"statuses": [{"id": "x"}]}}]}]}
        assert parse_incoming(receipt) == []

    def test_batches_several_messages(self):
        got = parse_incoming(
            self.payload(
                {"from": ALICE, "id": "1", "type": "text", "text": {"body": "a"}},
                {"from": BOB, "id": "2", "type": "text", "text": {"body": "b"}},
            )
        )
        assert [m.sender for m in got] == [ALICE, BOB]

    @pytest.mark.parametrize("junk", [{}, {"entry": None}, {"entry": [{}]}, {"entry": [{"changes": [{}]}]}])
    def test_survives_malformed_payloads(self, junk):
        assert parse_incoming(junk) == []

    def test_skips_a_message_with_no_sender_or_id(self):
        assert parse_incoming(self.payload({"type": "text", "text": {"body": "x"}})) == []


class TestAllowlist:
    def test_listed_numbers_pass_and_others_do_not(self, settings):
        cfg = enable_whatsapp(settings)
        assert is_allowed(cfg, ALICE) and is_allowed(cfg, BOB)
        assert not is_allowed(cfg, STRANGER)

    def test_formatting_in_the_list_does_not_matter(self, settings):
        cfg = enable_whatsapp(settings, allowed="+91 98000-00001")
        assert is_allowed(cfg, ALICE)

    def test_a_star_opens_it_to_everyone(self, settings):
        assert is_allowed(enable_whatsapp(settings, allowed="*"), STRANGER)

    def test_an_empty_list_lets_nobody_in(self, settings):
        assert not is_allowed(enable_whatsapp(settings, allowed=""), ALICE)


class TestChunkText:
    def test_short_text_is_one_message(self):
        assert chunk_text("hello") == ["hello"]

    def test_empty_text_sends_nothing(self):
        assert chunk_text("   ") == []

    def test_long_text_splits_under_the_limit_without_losing_words(self):
        text = "\n\n".join(f"Paragraph {i} " + "word " * 80 for i in range(30))
        parts = chunk_text(text, limit=1000)
        assert len(parts) > 1
        assert all(len(p) <= 1000 for p in parts)
        assert " ".join(" ".join(parts).split()) == " ".join(text.split())

    def test_an_unbroken_string_is_still_split(self):
        parts = chunk_text("x" * 2500, limit=1000)
        assert [len(p) for p in parts] == [1000, 1000, 500]


# ── The Cloud API client ──────────────────────────────────────────────────


class TestCloudApiClient:
    URL = "https://graph.facebook.com/v25.0/1055550000"

    @pytest.fixture
    def wa(self, settings):
        return WhatsAppClient(enable_whatsapp(settings))

    @respx.mock
    async def test_send_text_posts_to_the_messages_endpoint(self, wa):
        route = respx.post(f"{self.URL}/messages").respond(json={"messages": [{"id": "m"}]})
        await wa.send_text(ALICE, "hello")

        sent = route.calls.last.request
        assert sent.headers["authorization"] == "Bearer wa-token"
        body = json.loads(sent.content)
        assert body["to"] == ALICE and body["text"]["body"] == "hello"
        assert body["messaging_product"] == "whatsapp"

    @respx.mock
    @pytest.mark.parametrize(
        ("code", "must_mention"),
        [
            (190, "invalid or has expired"),
            (131005, "restart the server"),
        ],
    )
    async def test_a_refused_token_comes_with_what_to_do_about_it(self, wa, code, must_mention):
        respx.post(f"{self.URL}/messages").respond(
            403, json={"error": {"message": "(#131005) Access denied", "code": code}}
        )
        with pytest.raises(WhatsAppError) as excinfo:
            await wa.send_text(ALICE, "hello")

        assert must_mention in str(excinfo.value)
        assert "System User token never expires" in str(excinfo.value) or code == 190

    @respx.mock
    async def test_other_errors_get_no_token_advice(self, wa):
        """Advice about tokens on an unrelated error would send people the wrong way."""
        respx.post(f"{self.URL}/messages").respond(
            400, json={"error": {"message": "recipient not in allowed list", "code": 131030}}
        )
        with pytest.raises(WhatsAppError) as excinfo:
            await wa.send_text(ALICE, "hello")

        assert "allowed list" in str(excinfo.value) and "token" not in str(excinfo.value).lower()

    @respx.mock
    async def test_a_long_message_goes_out_as_several(self, wa):
        route = respx.post(f"{self.URL}/messages").respond(json={})
        await wa.send_text(ALICE, ("word " * 400 + "\n\n") * 5)
        assert route.call_count > 1

    @respx.mock
    async def test_meta_error_text_is_surfaced(self, wa):
        respx.post(f"{self.URL}/messages").respond(
            400, json={"error": {"message": "outside the 24 hour window"}}
        )
        with pytest.raises(WhatsAppError, match="24 hour window"):
            await wa.send_text(ALICE, "hello")

    @respx.mock
    async def test_send_video_uploads_then_sends_by_media_id(self, wa, tmp_path):
        clip = tmp_path / "reel.mp4"
        clip.write_bytes(b"mp4 bytes")
        upload = respx.post(f"{self.URL}/media").respond(json={"id": "MEDIA42"})
        send = respx.post(f"{self.URL}/messages").respond(json={})

        await wa.send_video(ALICE, clip, "Here you go")

        assert upload.called and b"mp4 bytes" in upload.calls.last.request.content
        body = json.loads(send.calls.last.request.content)
        assert body["type"] == "video"
        assert body["video"] == {"id": "MEDIA42", "caption": "Here you go"}

    @respx.mock
    async def test_a_video_over_16mb_is_refused_before_uploading(self, wa, tmp_path):
        clip = tmp_path / "big.mp4"
        clip.write_bytes(b"\0" * (16 * 1024 * 1024 + 1))
        upload = respx.post(f"{self.URL}/media").respond(json={"id": "x"})

        with pytest.raises(WhatsAppError, match="limit"):
            await wa.send_video(ALICE, clip)
        assert not upload.called

    async def test_a_missing_video_is_an_error(self, wa, tmp_path):
        with pytest.raises(WhatsAppError, match="not found"):
            await wa.send_video(ALICE, tmp_path / "nope.mp4")


class TestDownloadMedia:
    LOOKUP = "https://graph.facebook.com/v25.0/MEDIA1"
    FILE = "https://lookaside.fbsbx.com/whatsapp_business/attachments/?mid=MEDIA1"

    @pytest.fixture
    def wa(self, settings):
        return WhatsAppClient(enable_whatsapp(settings))

    @respx.mock
    async def test_it_looks_the_id_up_then_fetches_the_file(self, wa):
        lookup = respx.get(self.LOOKUP).respond(
            json={"url": self.FILE, "mime_type": "image/jpeg", "file_size": 11}
        )
        fetch = respx.get(self.FILE).respond(content=b"jpeg-bytes")

        data, mime = await wa.download_media("MEDIA1", 1000)

        assert (data, mime) == (b"jpeg-bytes", "image/jpeg")
        assert lookup.called and fetch.called

    @respx.mock
    async def test_both_requests_carry_the_token(self, wa):
        lookup = respx.get(self.LOOKUP).respond(json={"url": self.FILE, "mime_type": "image/png"})
        fetch = respx.get(self.FILE).respond(content=b"x")

        await wa.download_media("MEDIA1", 1000)

        assert lookup.calls.last.request.headers["authorization"] == "Bearer wa-token"
        assert fetch.calls.last.request.headers["authorization"] == "Bearer wa-token"

    @respx.mock
    async def test_a_file_reported_as_too_big_is_not_downloaded(self, wa):
        respx.get(self.LOOKUP).respond(json={"url": self.FILE, "file_size": 50 * 1024 * 1024})
        fetch = respx.get(self.FILE).respond(content=b"x")

        with pytest.raises(WhatsAppError, match="50MB"):
            await wa.download_media("MEDIA1", 25 * 1024 * 1024)
        assert not fetch.called

    @respx.mock
    async def test_a_file_that_turns_out_bigger_than_reported_is_refused(self, wa):
        """Do not trust the size the lookup claimed."""
        respx.get(self.LOOKUP).respond(json={"url": self.FILE, "file_size": 10})
        respx.get(self.FILE).respond(content=b"x" * 5000)

        with pytest.raises(WhatsAppError, match="larger"):
            await wa.download_media("MEDIA1", 1000)

    @respx.mock
    async def test_a_download_link_that_is_not_https_is_never_used(self, wa):
        """The token would be sent to it."""
        respx.get(self.LOOKUP).respond(json={"url": "http://evil.example/steal"})
        stolen = respx.get("http://evil.example/steal").respond(content=b"x")

        with pytest.raises(WhatsAppError, match="no usable download link"):
            await wa.download_media("MEDIA1", 1000)
        assert not stolen.called

    @respx.mock
    async def test_a_lookup_with_no_link_is_an_error(self, wa):
        respx.get(self.LOOKUP).respond(json={"mime_type": "image/jpeg"})
        with pytest.raises(WhatsAppError, match="no usable download link"):
            await wa.download_media("MEDIA1", 1000)

    @respx.mock
    async def test_metas_own_error_is_surfaced(self, wa):
        respx.get(self.LOOKUP).respond(404, json={"error": {"message": "media not found"}})
        with pytest.raises(WhatsAppError, match="media not found"):
            await wa.download_media("MEDIA1", 1000)

    @respx.mock
    async def test_a_failed_download_is_an_error(self, wa):
        respx.get(self.LOOKUP).respond(json={"url": self.FILE})
        respx.get(self.FILE).respond(500)
        with pytest.raises(WhatsAppError, match="HTTP 500"):
            await wa.download_media("MEDIA1", 1000)


# ── Reading a reply ───────────────────────────────────────────────────────


class TestInterpret:
    @pytest.fixture
    def returns(self, monkeypatch):
        """Make the model return a fixed reply, and record what it was asked."""
        seen: dict = {}

        def install(reply_or_exc):
            async def fake(settings, node, schema, system, user, temperature=None, attempts=3):
                seen.update(node=node, system=system, user=user, temperature=temperature)
                if isinstance(reply_or_exc, Exception):
                    raise reply_or_exc
                return reply_or_exc

            monkeypatch.setattr(converse, "structured", fake)
            return seen

        return install

    async def test_passes_the_message_and_the_open_question_to_the_model(self, settings, returns):
        seen = returns(Reply(intent="approve"))
        await interpret(settings, "script", "go ahead", {"title": "Roasted", "hook": "Wake up"})

        assert "go ahead" in seen["user"]
        assert "Roasted" in seen["system"] and "Wake up" in seen["system"]
        assert seen["temperature"] == 0.0, "a decision must not be sampled"

    async def test_a_clear_yes_is_approve(self, settings, returns):
        returns(Reply(intent="approve"))
        assert (await interpret(settings, "video", "yes post it")).intent == "approve"

    async def test_a_change_keeps_the_feedback(self, settings, returns):
        returns(Reply(intent="change", feedback="Shorter hook"))
        reply = await interpret(settings, "script", "shorter hook")
        assert (reply.intent, reply.feedback) == ("change", "Shorter hook")

    async def test_a_change_with_nothing_to_change_is_unclear(self, settings, returns):
        returns(Reply(intent="change", feedback="  "))
        assert (await interpret(settings, "script", "change it")).intent == "unclear"

    @pytest.mark.parametrize(
        "stage,intent,expected",
        [
            ("video", "change", "unclear"),    # nothing to rewrite after the video exists
            ("video", "request", "unclear"),
            ("script", "request", "unclear"),
            ("idle", "approve", "chat"),       # there is nothing to approve
            ("idle", "decline", "chat"),
        ],
    )
    async def test_an_intent_the_question_cannot_take_is_not_acted_on(
        self, settings, returns, stage, intent, expected
    ):
        returns(Reply(intent=intent, feedback="x"))
        assert (await interpret(settings, stage, "whatever")).intent == expected

    @pytest.mark.parametrize("stage,expected", [("script", "unclear"), ("video", "unclear"), ("idle", "chat")])
    async def test_a_model_failure_never_becomes_a_decision(self, settings, returns, stage, expected):
        returns(RuntimeError("bedrock is down"))
        assert (await interpret(settings, stage, "yes")).intent == expected

    @pytest.mark.parametrize(("stage", "intent"), [("idle", "request"), ("script", "change")])
    async def test_a_stated_video_length_is_passed_through(
        self, settings, returns, stage, intent
    ):
        returns(Reply(intent=intent, feedback="make it longer", video_seconds=22))
        reply = await interpret(settings, stage, "make it 22 seconds")
        assert reply.video_seconds == 22

    async def test_no_stated_length_is_none(self, settings, returns):
        returns(Reply(intent="request"))
        assert (await interpret(settings, "idle", "explain RAG")).video_seconds is None

    async def test_the_model_is_told_to_ignore_the_length_of_a_part(self, settings, returns):
        """'A hook in the first 3 seconds' is not a 3-second video."""
        seen = returns(Reply(intent="request"))
        await interpret(settings, "idle", "explain RAG with a hook in the first 3 seconds")

        assert "WHOLE video" in seen["system"]
        assert "first 3 seconds" in seen["system"], "the example of what to ignore"

    async def test_the_system_prompt_treats_hedging_as_unclear(self, settings, returns):
        seen = returns(Reply(intent="unclear"))
        await interpret(settings, "video", "maybe")
        assert "Hedging" in seen["system"] and "unclear" in seen["system"]


class TestUploadRecords:
    @pytest.fixture
    def store(self):
        return ConversationStore(":memory:")

    def test_a_file_waits_for_a_request(self, store):
        store.stage_asset(ALICE, "/u/a.jpg", "image", "image/jpeg", "my shop")

        (asset,) = store.staged_assets(ALICE)
        assert (asset.path, asset.kind, asset.mime, asset.caption) == ("/u/a.jpg", "image", "image/jpeg", "my shop")

    def test_files_come_back_in_the_order_they_were_sent(self, store):
        for name in ("1", "2", "3"):
            store.stage_asset(ALICE, f"/u/{name}.jpg", "image", "image/jpeg")
        assert [a.path for a in store.staged_assets(ALICE)] == ["/u/1.jpg", "/u/2.jpg", "/u/3.jpg"]

    def test_phones_do_not_see_each_others_files(self, store):
        store.stage_asset(ALICE, "/u/a.jpg", "image", "image/jpeg")
        store.stage_asset(BOB, "/u/b.jpg", "image", "image/jpeg")

        assert [a.path for a in store.staged_assets(ALICE)] == ["/u/a.jpg"]
        assert [a.path for a in store.staged_assets(BOB)] == ["/u/b.jpg"]

    def test_claiming_hands_the_files_to_a_run(self, store):
        store.stage_asset(ALICE, "/u/a.jpg", "image", "image/jpeg")
        store.claim_assets(ALICE, "veo-1")

        assert store.staged_assets(ALICE) == [], "no longer waiting"
        assert store.release_assets("veo-1") == ["/u/a.jpg"]

    def test_claiming_takes_only_that_phones_files(self, store):
        store.stage_asset(ALICE, "/u/a.jpg", "image", "image/jpeg")
        store.stage_asset(BOB, "/u/b.jpg", "image", "image/jpeg")
        store.claim_assets(ALICE, "veo-1")

        assert [a.path for a in store.staged_assets(BOB)] == ["/u/b.jpg"]

    def test_files_sent_after_a_claim_wait_for_the_next_request(self, store):
        store.stage_asset(ALICE, "/u/a.jpg", "image", "image/jpeg")
        store.claim_assets(ALICE, "veo-1")
        store.stage_asset(ALICE, "/u/b.jpg", "image", "image/jpeg")

        assert [a.path for a in store.staged_assets(ALICE)] == ["/u/b.jpg"]

    def test_releasing_a_run_returns_its_paths_once(self, store):
        store.stage_asset(ALICE, "/u/a.jpg", "image", "image/jpeg")
        store.claim_assets(ALICE, "veo-1")

        assert store.release_assets("veo-1") == ["/u/a.jpg"]
        assert store.release_assets("veo-1") == [], "nothing left to delete twice"

    def test_releasing_staged_files_leaves_a_runs_files_alone(self, store):
        store.stage_asset(ALICE, "/u/run.jpg", "image", "image/jpeg")
        store.claim_assets(ALICE, "veo-1")
        store.stage_asset(ALICE, "/u/waiting.jpg", "image", "image/jpeg")

        assert store.release_staged(ALICE) == ["/u/waiting.jpg"]
        assert store.release_assets("veo-1") == ["/u/run.jpg"]

    def test_old_files_are_forgotten_and_new_ones_are_not(self, store):
        store.stage_asset(ALICE, "/u/old.jpg", "image", "image/jpeg")
        store._db.execute("UPDATE assets SET created_at = created_at - 7200")
        store.stage_asset(ALICE, "/u/new.jpg", "image", "image/jpeg")

        assert store.stale_assets(staged_after=3600, claimed_after=86400) == ["/u/old.jpg"]
        assert [a.path for a in store.staged_assets(ALICE)] == ["/u/new.jpg"]

    def test_a_runs_files_outlive_the_short_wait_but_not_the_long_one(self, store):
        store.stage_asset(ALICE, "/u/a.jpg", "image", "image/jpeg")
        store.claim_assets(ALICE, "veo-1")
        store._db.execute("UPDATE assets SET created_at = created_at - 7200")

        assert store.stale_assets(staged_after=3600, claimed_after=86400) == [], "2h is fine for a run"
        store._db.execute("UPDATE assets SET created_at = created_at - 86400")
        assert store.stale_assets(staged_after=3600, claimed_after=86400) == ["/u/a.jpg"]

    def test_the_records_survive_a_restart(self, tmp_path):
        path = str(tmp_path / "c.sqlite")
        first = ConversationStore(path)
        first.stage_asset(ALICE, "/u/a.jpg", "image", "image/jpeg")
        first.close()

        assert [a.path for a in ConversationStore(path).staged_assets(ALICE)] == ["/u/a.jpg"]


# ── Conversation store ────────────────────────────────────────────────────


class TestStore:
    def test_a_phone_has_one_active_run_and_the_newest_wins(self):
        store = ConversationStore(":memory:")
        store.start(ALICE, "veo-1")
        store.start(ALICE, "veo-2")
        assert store.active_run(ALICE) == "veo-2"

    def test_finishing_a_run_frees_the_phone_but_keeps_the_owner(self):
        store = ConversationStore(":memory:")
        store.start(ALICE, "veo-1")
        store.finish("veo-1")
        assert store.active_run(ALICE) is None
        assert store.phone_for("veo-1") == ALICE

    def test_phones_do_not_share_runs(self):
        store = ConversationStore(":memory:")
        store.start(ALICE, "veo-1")
        store.start(BOB, "veo-2")
        assert store.active_run(ALICE) == "veo-1" and store.active_run(BOB) == "veo-2"

    def test_a_run_from_outside_chat_has_no_owner(self):
        assert ConversationStore(":memory:").phone_for("veo-api") is None

    def test_a_message_is_only_new_once(self):
        store = ConversationStore(":memory:")
        assert store.first_sighting("wamid.1") is True
        assert store.first_sighting("wamid.1") is False
        assert store.first_sighting("wamid.2") is True

    def test_counts_runs_in_the_last_day_only(self):
        store = ConversationStore(":memory:")
        store.start(ALICE, "veo-1")
        store.start(ALICE, "veo-2")
        store._db.execute("UPDATE runs SET created_at = created_at - 90000 WHERE video_id = 'veo-1'")
        assert store.runs_in_last_day(ALICE) == 1

    def test_survives_a_restart(self, tmp_path):
        path = str(tmp_path / "c.sqlite")
        ConversationStore(path).start(ALICE, "veo-1")
        assert ConversationStore(path).active_run(ALICE) == "veo-1"


# ── The bot, end to end ───────────────────────────────────────────────────


class FakeClient:
    """Records what would have been sent to WhatsApp."""

    def __init__(self):
        self.texts: list[tuple[str, str]] = []
        self.videos: list[tuple[str, object, str]] = []
        self.video_error: WhatsAppError | None = None
        self.media: dict[str, tuple[bytes, str]] = {}
        self.media_error: WhatsAppError | None = None
        self.downloads: list[tuple[str, int]] = []

    async def download_media(self, media_id, max_bytes):
        self.downloads.append((media_id, max_bytes))
        if self.media_error:
            raise self.media_error
        return self.media.get(media_id, (b"file-bytes", "image/jpeg"))

    async def send_text(self, to, body):
        self.texts.append((to, body))

    async def send_video(self, to, path, caption=""):
        if self.video_error:
            raise self.video_error
        self.videos.append((to, path, caption))

    async def aclose(self):
        pass

    def to(self, phone):
        return [body for who, body in self.texts if who == phone]


class ScriptedInterpreter:
    """Returns queued replies and records how it was asked."""

    def __init__(self):
        self.replies: list[Reply] = []
        self.calls: list[tuple[str, str, dict]] = []

    def will_say(self, *replies: Reply) -> None:
        self.replies.extend(replies)

    async def __call__(self, cfg, stage, text, context=None):
        self.calls.append((stage, text, context or {}))
        assert self.replies, f"unexpected interpretation of {text!r} at {stage!r}"
        return self.replies.pop(0)


async def wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition never became true")


class Env:
    def __init__(self, cfg, agents, client, interp, store, runs, bot):
        self.cfg, self.agents, self.client = cfg, agents, client
        self.interp, self.store, self.runs, self.bot = interp, store, runs, bot
        self._n = 0

    async def say(self, phone, text, kind="text", message_id=None):
        self._n += 1
        await self.bot.handle(Incoming(phone, message_id or f"wamid.{self._n}", kind, text))

    async def at_script_gate(self, phone=ALICE, request="https://example.com/product"):
        before = len([b for b in self.client.to(phone) if "Should I generate it?" in b])
        await self.say(phone, request)
        await wait_until(
            lambda: len([b for b in self.client.to(phone) if "Should I generate it?" in b]) > before
        )

    async def at_video_gate(self, phone=ALICE):
        await self.at_script_gate(phone, "https://example.com/product")
        self.interp.will_say(Reply(intent="approve"))
        await self.say(phone, "looks good, go")
        await wait_until(lambda: any("post this to Instagram" in b for b in self.client.to(phone)))


@pytest.fixture
async def env(settings, tmp_path):
    cfg = enable_whatsapp(settings, uploads_dir=tmp_path / "uploads")
    agents = FakeAgents()
    clip = tmp_path / "reel.mp4"
    clip.write_bytes(b"clip")

    async def video(state):
        agents.calls.append("video")
        return {"media": Media(local_files=[str(clip)], duration_seconds=15)}

    agents.video = video

    client, interp, store = FakeClient(), ScriptedInterpreter(), ConversationStore(":memory:")
    holder: dict[str, WhatsAppBot] = {}

    async def settled(video_id):
        await holder["bot"].on_run_settled(video_id)

    def delegate(name):
        # Looked up on every call, so a test can swap an agent after the graph
        # is compiled (as_nodes() would have bound the originals for good).
        async def node(state):
            return await getattr(agents, name)(state)

        return node

    def nodes_for(_cfg):
        return Nodes(**{n: delegate(n) for n in Nodes.__dataclass_fields__})

    runs = RunManager(cfg, InMemorySaver(), nodes_for, on_settled=settled)
    bot = holder["bot"] = WhatsAppBot(cfg, runs, store, client, interp)
    yield Env(cfg, agents, client, interp, store, runs, bot)
    await runs.close()


class TestStartingARun:
    async def test_a_link_starts_a_run_without_asking_the_model(self, env):
        await env.at_script_gate(request="https://example.com/product")

        assert env.interp.calls == [], "a link is a request; no interpretation needed"
        assert env.agents.calls[:2] == ["classify", "research"]
        assert any("On it" in b for b in env.client.to(ALICE))

    async def test_a_topic_is_read_by_the_model_first(self, env):
        env.interp.will_say(Reply(intent="request"))
        await env.at_script_gate(request="explain how RAG works")
        assert env.interp.calls[0][:2] == ("idle", "explain how RAG works")

    async def test_the_users_own_words_are_the_request(self, env):
        env.interp.will_say(Reply(intent="request"))
        await env.at_script_gate(request="  explain how RAG works  ")
        status = await env.runs.status(env.store.active_run(ALICE))
        assert status["request"] == "explain how RAG works"

    async def test_a_greeting_does_not_start_a_run(self, env):
        env.interp.will_say(Reply(intent="chat", message="Hi! What would you like to make?"))
        await env.say(ALICE, "hi")

        assert env.client.to(ALICE) == ["Hi! What would you like to make?"]
        assert env.store.active_run(ALICE) is None and env.agents.calls == []

    @pytest.mark.parametrize("kind", ["document", "sticker", "location", "contacts"])
    async def test_a_kind_of_message_it_cannot_use_gets_a_clear_reply(self, env, kind):
        await env.say(ALICE, "", kind=kind)

        assert "text, photos, video and audio" in env.client.to(ALICE)[0]
        assert env.store.active_run(ALICE) is None

    async def test_the_daily_limit_stops_a_user_starting_more(self, env):
        env.bot._cfg = dataclasses.replace(env.cfg, whatsapp_max_runs_per_day=1)
        await env.at_script_gate(request="https://example.com/a")
        env.store.finish(env.store.active_run(ALICE))

        await env.say(ALICE, "https://example.com/b")
        assert "limit" in env.client.to(ALICE)[-1]
        assert env.agents.calls.count("classify") == 1


class TestVideoLength:
    """A person can say how long the video should be: 4-30 seconds, default 15."""

    async def start(self, env, text, *, says=None):
        if says is not None:
            env.interp.will_say(says)
        await env.at_script_gate(request=text)
        return await env.runs.status(env.store.active_run(ALICE))

    async def test_the_default_is_fifteen_seconds(self, env):
        status = await self.start(env, "https://example.com/product")

        assert status["duration_seconds"] == 15
        assert env.agents.writer_saw_duration == [15]
        assert any("15-second video" in b for b in env.client.to(ALICE))

    async def test_a_length_read_from_free_text_reaches_the_writer(self, env):
        status = await self.start(
            env, "explain RAG in 22 seconds", says=Reply(intent="request", video_seconds=22)
        )

        assert status["duration_seconds"] == 22
        assert env.agents.writer_saw_duration == [22]
        assert any("22-second video" in b for b in env.client.to(ALICE))

    async def test_a_link_message_can_carry_a_length_too(self, env):
        """A link never reaches the model, so its length is read in code."""
        status = await self.start(env, "https://example.com/product make it 20 seconds")

        assert env.interp.calls == []
        assert status["duration_seconds"] == 20
        assert env.agents.writer_saw_duration == [20]

    async def test_more_than_thirty_seconds_is_cut_to_the_maximum_and_said_so(self, env):
        status = await self.start(
            env, "a 2 minute video about RAG", says=Reply(intent="request", video_seconds=120)
        )

        said = env.client.to(ALICE)[0]
        assert "longest video I can make is 30 seconds" in said
        assert status["duration_seconds"] == 29, "30 is made as 29: see video_plan"
        assert "29-second video" in said

    async def test_less_than_four_seconds_is_raised_to_the_minimum_and_said_so(self, env):
        status = await self.start(
            env, "a 2 second clip", says=Reply(intent="request", video_seconds=2)
        )

        assert "shortest video I can make is 4 seconds" in env.client.to(ALICE)[0]
        assert status["duration_seconds"] == 4

    async def test_a_length_the_model_did_not_find_means_the_default(self, env):
        """'A hook in the first 3 seconds' is for the model to rule out, not us."""
        status = await self.start(
            env,
            "explain RAG with a hook in the first 3 seconds",
            says=Reply(intent="request", video_seconds=None),
        )

        assert status["duration_seconds"] == 15
        assert "longest" not in env.client.to(ALICE)[0]

    async def test_a_link_with_the_length_of_a_part_is_not_a_length(self, env):
        status = await self.start(
            env, "https://example.com/product hook in the first 3 seconds"
        )
        assert status["duration_seconds"] == 15

    async def test_the_prompt_review_says_how_long_it_is(self, env):
        await self.start(
            env, "explain RAG in 22 seconds", says=Reply(intent="request", video_seconds=22)
        )
        assert any("Length: 22 seconds" in b for b in env.client.to(ALICE))

    async def test_a_new_length_at_the_review_rewrites_the_script_to_it(self, env):
        await self.start(env, "https://example.com/product")
        env.interp.will_say(
            Reply(intent="change", feedback="Make it longer", video_seconds=25)
        )
        await env.say(ALICE, "make it 25 seconds")
        await wait_until(
            lambda: len([b for b in env.client.to(ALICE) if "Should I generate it?" in b]) == 2
        )

        assert env.agents.writer_saw_duration == [15, 25]
        assert any("25-second video" in b for b in env.client.to(ALICE))
        status = await env.runs.status(env.store.active_run(ALICE))
        assert status["duration_seconds"] == 25
        assert any("Length: 25 seconds" in b for b in env.client.to(ALICE))

    async def test_a_change_that_names_no_length_keeps_the_one_it_has(self, env):
        await self.start(
            env, "explain RAG in 22 seconds", says=Reply(intent="request", video_seconds=22)
        )
        env.interp.will_say(Reply(intent="change", feedback="Shorter hook"))
        await env.say(ALICE, "shorter hook")
        await wait_until(
            lambda: len([b for b in env.client.to(ALICE) if "Should I generate it?" in b]) == 2
        )

        assert env.agents.writer_saw_duration == [22, 22]
        assert any("Rewriting the script with your changes" in b for b in env.client.to(ALICE))

    async def test_an_out_of_range_length_at_the_review_is_clamped_and_said_so(self, env):
        await self.start(env, "https://example.com/product")
        env.interp.will_say(Reply(intent="change", feedback="longer", video_seconds=90))
        await env.say(ALICE, "make it 90 seconds")
        await wait_until(
            lambda: len([b for b in env.client.to(ALICE) if "Should I generate it?" in b]) == 2
        )

        assert any("longest video I can make is 30 seconds" in b for b in env.client.to(ALICE))
        assert env.agents.writer_saw_duration == [15, 29]


class TestBigVideosFitTheChat:
    """WhatsApp takes 16 MB; a 30-second reel can be close, so the chat gets a copy."""

    async def test_the_chat_gets_the_smaller_copy_and_the_original_is_kept(
        self, env, tmp_path, monkeypatch
    ):
        from tools.whatsapp import MAX_VIDEO_BYTES

        small = tmp_path / "reel.chat.mp4"
        small.write_bytes(b"small")
        asked: list[tuple] = []

        async def fake_shrink(path, max_bytes, seconds):
            asked.append((path.name, max_bytes, seconds))
            return small

        monkeypatch.setattr("whatsapp_bot.shrink_to_fit", fake_shrink)
        await env.at_video_gate()

        (to, sent, _caption), = env.client.videos
        assert sent == small, "the compressed copy goes into the chat"
        assert asked == [("reel.mp4", MAX_VIDEO_BYTES, 15)], "the full-size file, the cap, the length"
        status = await env.runs.status(env.store.active_run(ALICE))
        assert status["awaiting"]["files"][0].endswith("reel.mp4"), "Instagram still gets the original"

    async def test_a_video_that_already_fits_is_sent_as_it_is(self, env):
        await env.at_video_gate()

        (to, sent, _caption), = env.client.videos
        assert sent.name == "reel.mp4"


class TestUploads:
    """Photos, video and audio sent to the bot steer the video that follows."""

    async def send(self, env, kind="image", *, media_id=None, caption="", phone=ALICE, mime=None):
        env._n += 1
        mid = media_id or f"MEDIA{env._n}"
        if mime:
            env.client.media[mid] = (b"file-bytes", mime)
        await env.bot.handle(
            Incoming(phone, f"wamid.u{env._n}", kind, "", mid, mime or "", caption)
        )

    def on_disk(self, env) -> list:
        root = env.cfg.uploads_dir
        return sorted(p for p in root.rglob("*") if p.is_file()) if root.exists() else []

    # ── receiving ──

    async def test_a_photo_is_saved_and_the_person_is_told_what_to_do_next(self, env):
        await self.send(env, "image")

        (asset,) = env.store.staged_assets(ALICE)
        assert asset.kind == "image" and asset.path.startswith(str(env.cfg.uploads_dir))
        assert asset.mime == "image/jpeg"
        reply = env.client.to(ALICE)[0]
        assert "Got your photo" in reply and "tell me what to make" in reply
        assert env.store.active_run(ALICE) is None, "a photo alone does not start a run"

    async def test_the_file_is_really_written_with_the_bytes_sent(self, env):
        env.client.media["M1"] = (b"the actual bytes", "image/png")
        await self.send(env, "image", media_id="M1")

        (asset,) = env.store.staged_assets(ALICE)
        assert Path(asset.path).read_bytes() == b"the actual bytes"
        assert asset.path.endswith(".png")

    async def test_the_download_is_capped(self, env):
        from whatsapp_bot import MAX_UPLOAD_BYTES

        await self.send(env, "image", media_id="M1")
        assert env.client.downloads == [("M1", MAX_UPLOAD_BYTES)]

    async def test_a_name_the_sender_chose_never_becomes_a_path(self, env):
        """Nothing the sender wrote may end up in a file path."""
        await env.bot.handle(
            Incoming(ALICE, "wamid.x", "image", "", "../../etc/passwd", "image/jpeg", "")
        )
        (asset,) = env.store.staged_assets(ALICE)

        path = Path(asset.path).resolve()
        assert env.cfg.uploads_dir.resolve() in path.parents
        assert "passwd" not in path.name and ".." not in path.parts

    @pytest.mark.parametrize(
        ("kind", "mime", "word"),
        [("image", "image/jpeg", "photo"), ("video", "video/mp4", "video"), ("audio", "audio/ogg; codecs=opus", "audio clip")],
    )
    async def test_each_kind_is_understood(self, env, kind, mime, word):
        await self.send(env, kind, mime=mime)

        (asset,) = env.store.staged_assets(ALICE)
        assert asset.kind == kind
        assert f"Got your {word}" in env.client.to(ALICE)[0]

    async def test_a_voice_note_has_a_usable_extension(self, env):
        await self.send(env, "audio", mime="audio/ogg; codecs=opus")
        (asset,) = env.store.staged_assets(ALICE)
        assert Path(asset.path).suffix in {".ogg", ".oga", ".opus"}

    async def test_a_kind_with_no_known_mime_still_gets_an_extension(self, env):
        env.client.media["M9"] = (b"x", "")
        await env.bot.handle(Incoming(ALICE, "wamid.9", "video", "", "M9", "", ""))
        (asset,) = env.store.staged_assets(ALICE)
        assert asset.path.endswith(".mp4")

    async def test_several_files_get_one_answer_not_one_each(self, env):
        for _ in range(3):
            await self.send(env, "image")

        assert len(env.store.staged_assets(ALICE)) == 3
        assert len(env.client.to(ALICE)) == 1, "three photos sent together, one reply"

    async def test_files_sent_together_share_one_folder(self, env):
        for _ in range(3):
            await self.send(env, "image")
        assert len({Path(a.path).parent for a in env.store.staged_assets(ALICE)}) == 1

    async def test_two_people_get_separate_folders(self, env):
        await self.send(env, "image", phone=ALICE)
        await self.send(env, "image", phone=BOB)

        (a,), (b,) = env.store.staged_assets(ALICE), env.store.staged_assets(BOB)
        assert Path(a.path).parent != Path(b.path).parent

    async def test_at_most_six_files_are_taken(self, env):
        from whatsapp_bot import MAX_UPLOADS

        for _ in range(MAX_UPLOADS):
            await self.send(env, "image")
        env.client.texts.clear()
        await self.send(env, "image")

        assert len(env.store.staged_assets(ALICE)) == MAX_UPLOADS
        assert "most I can use at once" in env.client.to(ALICE)[0]
        assert len(env.client.downloads) == MAX_UPLOADS, "the seventh was not even downloaded"

    async def test_a_download_that_fails_says_so_and_saves_nothing(self, env):
        env.client.media_error = WhatsAppError("the file is 80MB; the most I can take is 25MB")
        await self.send(env, "video")

        assert env.store.staged_assets(ALICE) == []
        assert "couldn't get that file" in env.client.to(ALICE)[0]
        assert "80MB" in env.client.to(ALICE)[0]
        assert self.on_disk(env) == []

    async def test_a_message_with_no_media_id_is_not_a_crash(self, env):
        await env.bot.handle(Incoming(ALICE, "wamid.n", "image", "", "", "", ""))
        assert "couldn't read that file" in env.client.to(ALICE)[0]
        assert env.client.downloads == []

    # ── using them ──

    async def test_the_next_request_uses_the_files(self, env):
        await self.send(env, "image")
        await self.send(env, "audio", mime="audio/ogg")
        await env.at_script_gate()

        assert env.agents.calls[0] == "analyze", "the files are read before anything else"
        (assets,) = env.agents.analyzed
        assert [a.kind for a in assets] == ["image", "audio"]
        assert all(Path(a.path).is_file() for a in assets)

    async def test_the_person_is_told_what_will_be_used(self, env):
        await self.send(env, "image")
        await self.send(env, "image")
        await self.send(env, "audio", mime="audio/ogg")
        await env.at_script_gate()

        said = " ".join(env.client.to(ALICE))
        assert "your 2 photos and audio clip" in said
        assert "take the look, the person and the sound" in said

    async def test_a_request_with_no_files_is_unchanged(self, env):
        await env.at_script_gate()

        assert "analyze" not in env.agents.calls
        assert not any("From your files" in b for b in env.client.to(ALICE))

    async def test_a_caption_is_the_request_so_one_message_starts_a_run(self, env):
        env.interp.will_say(Reply(intent="request", video_seconds=20))
        await self.send(env, "image", caption="a 20 second ad for my bakery")
        await wait_until(lambda: any("Should I generate it?" in b for b in env.client.to(ALICE)))

        assert env.interp.calls[0][:2] == ("idle", "a 20 second ad for my bakery")
        (assets,) = env.agents.analyzed
        assert [a.kind for a in assets] == ["image"]
        assert any("20-second video" in b and "your photo" in b for b in env.client.to(ALICE))

    async def test_a_caption_that_is_not_a_request_keeps_the_photo_for_later(self, env):
        env.interp.will_say(Reply(intent="chat", message="Hi! What shall I make?"))
        await self.send(env, "image", caption="hello")

        assert len(env.store.staged_assets(ALICE)) == 1, "not lost"
        assert env.store.active_run(ALICE) is None

    async def test_a_chat_message_does_not_use_up_the_files(self, env):
        await self.send(env, "image")
        env.interp.will_say(Reply(intent="chat", message="Happy to help!"))
        await env.say(ALICE, "thanks")

        assert len(env.store.staged_assets(ALICE)) == 1

    async def test_the_files_belong_to_one_run_only(self, env):
        await self.send(env, "image")
        await env.at_script_gate(request="https://example.com/one")
        env.interp.will_say(Reply(intent="decline"))
        await env.say(ALICE, "cancel")
        await wait_until(lambda: env.store.active_run(ALICE) is None)

        await env.at_script_gate(request="https://example.com/two")

        assert len(env.agents.analyzed) == 1, "the second request had no files"
        assert env.agents.calls.count("analyze") == 1

    async def test_files_are_not_shared_between_people(self, env):
        await self.send(env, "image", phone=ALICE)
        await env.at_script_gate(phone=BOB, request="https://example.com/bob")

        assert "analyze" not in env.agents.calls, "Bob used none of Alice's files"
        assert len(env.store.staged_assets(ALICE)) == 1

    async def test_what_was_taken_is_shown_at_the_review(self, env):
        await self.send(env, "image")
        await env.at_script_gate()

        said = "\n".join(env.client.to(ALICE))
        assert "*From your files*" in said
        assert "Colours: warm amber, cream and deep teal (#E8A33D #F5E6D0 #1F6F78)" in said
        assert "Mood: warm, calm" in said
        assert "Person: the baker in the green apron" in said
        assert "Audio: a slow acoustic folk tune" in said

    async def test_the_review_says_when_the_photos_go_to_the_video_model(self, env, monkeypatch):
        from conftest import make_asset_brief

        async def analyze(state):
            return {"asset_brief": make_asset_brief(references=["a.jpg"]), "asset_error": None}

        env.agents.analyze = analyze
        await self.send(env, "image")
        await env.at_script_gate()

        assert "your photos go to the video model, so it can match them" in " ".join(env.client.to(ALICE))

    async def test_a_failed_reading_is_said_and_the_run_goes_on(self, env):
        env.agents.analysis_error = "model unavailable"
        await self.send(env, "image")
        await env.at_script_gate()

        said = "\n".join(env.client.to(ALICE))
        assert "I couldn't read your files" in said and "written request alone" in said
        assert "*From your files*" not in said
        assert "Should I generate it?" in said, "the person can still go ahead"

    async def test_a_note_about_the_video_reaches_the_person(self, env, tmp_path):
        clip = tmp_path / "reel.mp4"

        async def video(state):
            return {
                "media": Media(
                    local_files=[str(clip)],
                    duration_seconds=15,
                    notes=["Veo couldn't use your photos as references (Veo's safety check blocked them)."],
                )
            }

        env.agents.video = video
        await self.send(env, "image")
        await env.at_video_gate()

        assert any("couldn't use your photos as references" in b for b in env.client.to(ALICE))

    # ── while a run is going ──

    async def test_files_sent_during_a_run_are_not_taken(self, env):
        await env.at_script_gate()
        await self.send(env, "image")

        assert env.store.staged_assets(ALICE) == []
        assert "still on your last video" in env.client.to(ALICE)[-1]
        assert env.client.downloads == [], "not even downloaded"

    async def test_a_finished_run_does_not_block_new_files(self, env):
        await env.at_video_gate()
        env.interp.will_say(Reply(intent="decline"))
        await env.say(ALICE, "no")
        await wait_until(lambda: env.store.active_run(ALICE) is None)

        await self.send(env, "image")
        assert len(env.store.staged_assets(ALICE)) == 1

    # ── deleting them ──

    async def test_the_files_are_deleted_when_a_run_is_cancelled(self, env):
        await self.send(env, "image")
        await env.at_script_gate()
        assert self.on_disk(env), "there are files while the run is going"

        env.interp.will_say(Reply(intent="decline"))
        await env.say(ALICE, "never mind")
        await wait_until(lambda: env.store.active_run(ALICE) is None)

        assert self.on_disk(env) == []
        assert not list(env.cfg.uploads_dir.iterdir()), "the folder goes too"

    async def test_the_files_are_deleted_when_a_video_is_made_and_not_posted(self, env):
        await self.send(env, "image")
        await env.at_video_gate()
        env.interp.will_say(Reply(intent="decline"))
        await env.say(ALICE, "no")
        await wait_until(lambda: env.store.active_run(ALICE) is None)

        assert self.on_disk(env) == []

    async def test_the_files_are_deleted_when_a_video_is_posted(self, env):
        await self.send(env, "image")
        await env.at_video_gate()
        env.interp.will_say(Reply(intent="approve"))
        await env.say(ALICE, "yes post it")
        await wait_until(lambda: env.store.active_run(ALICE) is None)

        assert self.on_disk(env) == []

    async def test_the_files_are_deleted_when_a_run_fails(self, env):
        async def broken(state):
            raise RuntimeError("boom")

        env.agents.research = broken
        await self.send(env, "image")
        await env.say(ALICE, "https://example.com/x")
        await wait_until(lambda: env.store.active_run(ALICE) is None)

        assert self.on_disk(env) == []
        assert env.store.release_assets("anything") == []

    async def test_scratch_files_made_while_reading_are_deleted_too(self, env):
        """The analyze step writes into the same folder, so it goes with it."""

        async def analyze(state):
            work = Path(state["assets"][0].path).parent / "work"
            work.mkdir()
            (work / "item-0.jpg").write_bytes(b"scratch")
            return {"asset_brief": None, "asset_error": None}

        env.agents.analyze = analyze
        await self.send(env, "image")
        await env.at_script_gate()
        assert any("work" in p.parts for p in self.on_disk(env))

        env.interp.will_say(Reply(intent="decline"))
        await env.say(ALICE, "cancel")
        await wait_until(lambda: env.store.active_run(ALICE) is None)

        assert self.on_disk(env) == []

    async def test_files_nobody_came_back_for_are_deleted(self, env, monkeypatch):
        await self.send(env, "image")
        assert self.on_disk(env)

        monkeypatch.setattr("whatsapp_bot.STAGED_SECONDS", -1)  # everything is now old
        env.interp.will_say(Reply(intent="chat", message="hi"))
        await env.say(ALICE, "hi")

        assert self.on_disk(env) == []
        assert env.store.staged_assets(ALICE) == []

    async def test_only_folders_inside_the_uploads_directory_are_ever_removed(self, env, tmp_path):
        """A bad record must not be able to point the cleanup at anything else."""
        outside = tmp_path / "precious"
        outside.mkdir()
        keep = outside / "keep.txt"
        keep.write_text("important")
        victim = outside / "file.jpg"
        victim.write_bytes(b"x")

        env.bot._delete_files([str(victim)])

        assert not victim.exists(), "the named file goes"
        assert keep.exists() and outside.exists(), "but its folder and neighbours do not"

    async def test_deleting_a_file_that_is_already_gone_is_fine(self, env):
        env.bot._delete_files([str(env.cfg.uploads_dir / "nope" / "gone.jpg")])


class TestDescribingFiles:
    """The sentence that tells the person which of their files will be used."""

    @pytest.mark.parametrize(
        ("kinds", "expected"),
        [
            (["image"], "photo"),
            (["image", "image"], "2 photos"),
            (["video"], "video"),
            (["audio"], "audio clip"),
            (["audio", "audio"], "2 audio clips"),
            (["image", "audio"], "photo and audio clip"),
            (["image", "image", "audio"], "2 photos and audio clip"),
            (["image", "video", "audio"], "photo, video and audio clip"),
            (["image", "image", "video", "audio", "audio"], "2 photos, video and 2 audio clips"),
        ],
    )
    def test_it_reads_naturally_after_your(self, kinds, expected):
        assets = [type("A", (), {"kind": k})() for k in kinds]
        text = WhatsAppBot._describe_files(assets)

        assert text == expected
        assert f"your {text}" and " a " not in f" {text} ", "no stray article after 'your'"


class TestHelp:
    def test_the_help_text_mentions_uploads(self):
        from whatsapp_bot import HELP

        assert "photos, a video or a voice note" in HELP


class TestPromptApproval:
    async def test_sends_every_prompt_and_asks_before_spending(self, env):
        await env.at_script_gate()
        text = "\n".join(env.client.to(ALICE))

        assert "PROMPT ZERO" in text and "PROMPT ONE" in text
        assert "Prompt 1 of 2" in text and "Prompt 2 of 2" in text
        assert "Roasted to order" in text, "the title"
        assert "#coffee" in text, "the hashtags"
        assert "video" not in env.agents.calls, "nothing is generated before approval"

    async def test_the_model_is_told_what_was_asked(self, env):
        await env.at_script_gate()
        env.interp.will_say(Reply(intent="approve"))
        await env.say(ALICE, "yes")

        stage, text, context = env.interp.calls[-1]
        assert stage == "script" and text == "yes"
        assert context == {"title": "Roasted to order", "hook": "Your coffee deserves better"}

    async def test_a_yes_generates_the_video_and_sends_it_into_the_chat(self, env, tmp_path):
        await env.at_video_gate()

        assert "video" in env.agents.calls
        (to, path, _caption), = env.client.videos
        assert to == ALICE and path.name == "reel.mp4"
        assert any("Generating" in b or "generating" in b for b in env.client.to(ALICE))

    async def test_a_change_goes_back_to_the_writer_in_the_users_words(self, env):
        await env.at_script_gate()
        env.interp.will_say(Reply(intent="change", feedback="Make the hook shorter"))
        await env.say(ALICE, "hook is too long")

        await wait_until(
            lambda: len([b for b in env.client.to(ALICE) if "Should I generate it?" in b]) == 2
        )
        assert env.agents.writer_saw_feedback == [None, "Make the hook shorter"]
        assert "video" not in env.agents.calls, "a rewrite is still unapproved"

    async def test_a_cancel_ends_the_run_and_frees_the_chat(self, env):
        await env.at_script_gate()
        env.interp.will_say(Reply(intent="decline"))
        await env.say(ALICE, "never mind")

        await wait_until(lambda: env.store.active_run(ALICE) is None)
        assert "video" not in env.agents.calls
        assert any("cancelled" in b for b in env.client.to(ALICE))

    async def test_an_unclear_answer_asks_again_and_decides_nothing(self, env):
        await env.at_script_gate()
        env.interp.will_say(Reply(intent="unclear", message="Do you want me to generate it?"))
        await env.say(ALICE, "hmm")
        await asyncio.sleep(0.1)

        assert env.client.to(ALICE)[-1] == "Do you want me to generate it?"
        assert "video" not in env.agents.calls
        status = await env.runs.status(env.store.active_run(ALICE))
        assert status["status"] == "awaiting_prompt_approval"

    async def test_an_unclear_answer_without_a_model_message_still_gets_a_prompt(self, env):
        await env.at_script_gate()
        env.interp.will_say(Reply(intent="unclear"))
        await env.say(ALICE, "hmm")
        assert "Should I generate this video?" in env.client.to(ALICE)[-1]


class TestPostApproval:
    async def test_asks_whether_to_post_and_waits(self, env):
        await env.at_video_gate()

        assert "publish" not in env.agents.calls
        assert any("Roasted to order, shipped the same day." in b for b in env.client.to(ALICE))
        assert any("DRY_RUN is on" in b for b in env.client.to(ALICE))

    async def test_yes_posts_and_reports_the_outcome(self, env):
        await env.at_video_gate()
        env.interp.will_say(Reply(intent="approve"))
        await env.say(ALICE, "yes post it")

        await wait_until(lambda: "publish" in env.agents.calls)
        await wait_until(lambda: env.store.active_run(ALICE) is None)
        assert "dry run" in env.client.to(ALICE)[-1]

    async def test_a_real_post_reports_the_post_id(self, env):
        async def publish(state):
            env.agents.calls.append("publish")
            return {"post_id": "1789"}

        env.agents.publish = publish
        await env.at_video_gate()
        env.interp.will_say(Reply(intent="approve"))
        await env.say(ALICE, "post it")

        await wait_until(lambda: any("1789" in b for b in env.client.to(ALICE)))

    async def test_no_keeps_the_video_and_never_posts(self, env):
        await env.at_video_gate()
        env.interp.will_say(Reply(intent="decline"))
        await env.say(ALICE, "not now")

        await wait_until(lambda: env.store.active_run(ALICE) is None)
        assert "publish" not in env.agents.calls
        assert "won't post" in env.client.to(ALICE)[-1]

    async def test_a_hedge_does_not_post(self, env):
        await env.at_video_gate()
        env.interp.will_say(Reply(intent="unclear", message="Should I post it now?"))
        await env.say(ALICE, "maybe later?")
        await asyncio.sleep(0.1)

        assert "publish" not in env.agents.calls
        assert env.client.to(ALICE)[-1] == "Should I post it now?"

    async def test_a_video_that_cannot_be_sent_still_asks_the_question(self, env):
        env.client.video_error = WhatsAppError("video is 20.0MB; WhatsApp's limit is 16MB")
        await env.at_video_gate()

        text = "\n".join(env.client.to(ALICE))
        assert "couldn't send it here" in text and "20.0MB" in text
        assert "post this to Instagram" in text

    async def test_a_new_message_after_a_finished_run_starts_a_new_one(self, env):
        await env.at_video_gate()
        env.interp.will_say(Reply(intent="decline"))
        await env.say(ALICE, "no")
        await wait_until(lambda: env.store.active_run(ALICE) is None)

        await env.at_script_gate(request="https://example.com/second")
        assert env.agents.calls.count("classify") == 2


class TestWhileWorking:
    async def test_a_message_mid_run_gets_a_status_and_is_not_a_decision(self, env):
        release = asyncio.Event()

        async def research(state):
            env.agents.calls.append("research")
            await release.wait()
            return {"research_notes": "notes"}

        env.agents.research = research
        await env.say(ALICE, "https://example.com/x")
        await wait_until(lambda: "research" in env.agents.calls)

        await env.say(ALICE, "yes")
        assert "researching" in env.client.to(ALICE)[-1]
        assert env.interp.calls == [], "nothing is being asked, so nothing is read"
        release.set()
        await wait_until(lambda: any("Should I generate it?" in b for b in env.client.to(ALICE)))

    async def test_a_crashed_run_tells_the_user_and_frees_the_chat(self, env):
        async def boom(state):
            raise RuntimeError("bedrock exploded")

        env.agents.writer = boom
        await env.say(ALICE, "https://example.com/x")

        await wait_until(lambda: any("went wrong" in b for b in env.client.to(ALICE)))
        assert "bedrock exploded" in env.client.to(ALICE)[-1]
        await wait_until(lambda: env.store.active_run(ALICE) is None)


class TestSafety:
    async def test_a_stranger_is_ignored_completely(self, env):
        await env.say(STRANGER, "https://example.com/x")

        assert env.client.texts == []
        assert env.agents.calls == []
        assert env.store.active_run(STRANGER) is None

    async def test_a_redelivered_message_is_handled_once(self, env):
        await env.say(ALICE, "https://example.com/x", message_id="wamid.same")
        await env.say(ALICE, "https://example.com/x", message_id="wamid.same")
        await wait_until(lambda: any("Should I generate it?" in b for b in env.client.to(ALICE)))
        assert env.agents.calls.count("classify") == 1

    async def test_users_cannot_answer_each_others_gates(self, env):
        await env.at_script_gate(ALICE)
        env.interp.will_say(Reply(intent="chat", message="Send me a topic."))
        await env.say(BOB, "yes")

        await asyncio.sleep(0.1)
        assert "video" not in env.agents.calls
        status = await env.runs.status(env.store.active_run(ALICE))
        assert status["status"] == "awaiting_prompt_approval"
        assert env.client.to(BOB) == ["Send me a topic."]

    async def test_two_users_run_side_by_side(self, env):
        await env.at_script_gate(ALICE, "https://example.com/a")
        await env.at_script_gate(BOB, "https://example.com/b")
        assert env.store.active_run(ALICE) != env.store.active_run(BOB)

    async def test_a_failure_while_handling_still_answers(self, env):
        async def broken(*args, **kwargs):
            raise RuntimeError("model unreachable")

        env.bot._interpret = broken
        await env.say(ALICE, "make me a video")
        assert "went wrong" in env.client.to(ALICE)[-1]

    async def test_runs_from_the_http_api_are_not_announced_in_chat(self, env):
        video_id = await env.runs.create("explain RAG")
        await wait_until(lambda: env.agents.calls.count("dialogue") == 1)
        await asyncio.sleep(0.05)

        assert env.client.texts == []
        assert env.store.phone_for(video_id) is None


# ── The webhook ───────────────────────────────────────────────────────────


def webhook_body(sender, text, message_id="wamid.hook"):
    return json.dumps(
        {
            "object": "whatsapp_business_account",
            "entry": [
                {
                    "changes": [
                        {
                            "value": {
                                "messages": [
                                    {"from": sender, "id": message_id, "type": "text", "text": {"body": text}}
                                ]
                            }
                        }
                    ]
                }
            ],
        }
    ).encode()


@pytest.fixture
def wa_app(settings, agents, monkeypatch):
    """The real app with WhatsApp on and the outbound client replaced."""
    fake = FakeClient()
    monkeypatch.setattr(api_module, "WhatsAppClient", lambda cfg: fake)
    cfg = enable_whatsapp(settings)
    app = create_app(cfg, lambda c: agents.as_nodes(), InMemorySaver())
    with TestClient(app) as client:
        yield client, fake, agents


class TestWebhook:
    def test_the_handshake_echoes_the_challenge(self, wa_app):
        client, _, _ = wa_app
        response = client.get(
            "/whatsapp/webhook",
            params={"hub.mode": "subscribe", "hub.verify_token": "wa-verify", "hub.challenge": "12345"},
        )
        assert response.status_code == 200 and response.text == "12345"

    @pytest.mark.parametrize("token", ["wrong", ""])
    def test_the_handshake_rejects_a_wrong_token(self, wa_app, token):
        client, _, _ = wa_app
        response = client.get(
            "/whatsapp/webhook",
            params={"hub.mode": "subscribe", "hub.verify_token": token, "hub.challenge": "1"},
        )
        assert response.status_code == 403

    def test_a_delivery_with_a_bad_signature_is_rejected(self, wa_app):
        client, fake, agents = wa_app
        body = webhook_body(ALICE, "https://example.com/x")
        response = client.post(
            "/whatsapp/webhook", content=body, headers={"X-Hub-Signature-256": sign(body, "wrong")}
        )
        assert response.status_code == 401
        assert agents.calls == [] and fake.texts == []

    def test_a_delivery_with_no_signature_is_rejected(self, wa_app):
        client, _, agents = wa_app
        response = client.post("/whatsapp/webhook", content=webhook_body(ALICE, "hi"))
        assert response.status_code == 401 and agents.calls == []

    def test_a_signed_message_starts_a_run_and_the_prompts_come_back(self, wa_app):
        client, fake, agents = wa_app
        body = webhook_body(ALICE, "https://example.com/product")
        response = client.post(
            "/whatsapp/webhook", content=body, headers={"X-Hub-Signature-256": sign(body)}
        )
        assert response.status_code == 200

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not any(
            "Should I generate it?" in b for b in fake.to(ALICE)
        ):
            time.sleep(0.02)
        assert any("PROMPT ZERO" in b for b in fake.to(ALICE))
        assert "video" not in agents.calls

    def test_a_signed_message_from_a_stranger_is_ignored(self, wa_app):
        client, fake, agents = wa_app
        body = webhook_body(STRANGER, "https://example.com/product")
        response = client.post(
            "/whatsapp/webhook", content=body, headers={"X-Hub-Signature-256": sign(body)}
        )
        time.sleep(0.2)
        assert response.status_code == 200
        assert agents.calls == [] and fake.texts == []

    def test_signed_non_json_is_a_400(self, wa_app):
        client, _, _ = wa_app
        body = b"not json"
        response = client.post(
            "/whatsapp/webhook", content=body, headers={"X-Hub-Signature-256": sign(body)}
        )
        assert response.status_code == 400

    def test_the_existing_routes_still_need_the_api_key(self, wa_app):
        client, _, _ = wa_app
        assert client.post("/videos", json={"request": "x"}).status_code == 401


class TestWhatsAppOptIn:
    def test_the_routes_do_not_exist_when_whatsapp_is_not_configured(self, settings, agents):
        app = create_app(settings, lambda c: agents.as_nodes(), InMemorySaver())
        with TestClient(app) as client:
            assert client.get("/whatsapp/webhook").status_code == 404
            assert client.post("/whatsapp/webhook", content=b"{}").status_code == 404

    def test_the_existing_api_works_unchanged_without_whatsapp(self, settings, agents):
        app = create_app(settings, lambda c: agents.as_nodes(), InMemorySaver())
        with TestClient(app, headers={"X-API-Key": "test-api-key"}) as client:
            assert client.post("/videos", json={"request": "explain RAG"}).status_code == 202

    def test_refuses_to_start_half_configured(self, settings, agents):
        half = dataclasses.replace(settings, whatsapp_access_token="wa-token")
        with pytest.raises(RuntimeError, match="WHATSAPP_APP_SECRET"):
            create_app(half, lambda c: agents.as_nodes(), InMemorySaver())

    def test_refuses_to_start_without_an_allowlist(self, settings, agents):
        open_door = enable_whatsapp(settings, allowed="")
        with pytest.raises(RuntimeError, match="WHATSAPP_ALLOWED_NUMBERS"):
            create_app(open_door, lambda c: agents.as_nodes(), InMemorySaver())


class TestFailuresAreExplained:
    @respx.mock
    async def test_an_expired_token_says_so(self, settings):
        respx.post("https://graph.facebook.com/v25.0/1055550000/messages").respond(
            401, json={"error": {"message": "Session has expired", "code": 190}}
        )
        wa = WhatsAppClient(enable_whatsapp(settings))
        with pytest.raises(WhatsAppError, match="expired.*System User"):
            await wa.send_text(ALICE, "hello")

    async def test_a_handler_failure_survives_a_second_failure_reporting_it(self, env):
        async def broken(*args, **kwargs):
            raise RuntimeError("model unreachable")

        async def cannot_send(to, body):
            raise WhatsAppError("token expired")

        env.bot._interpret = broken
        env.client.send_text = cannot_send
        await env.say(ALICE, "make me a video")  # must not raise
