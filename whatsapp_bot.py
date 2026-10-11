"""The WhatsApp front end: a chat in place of the CLI's two ``y/n`` prompts.

A user messages a topic or a link. The bot starts a run, sends the exact prompts
to approve, sends the finished video into the chat and asks whether to post it.
Replies are free text; :func:`agents.converse.interpret` turns them into the
approve / change / decline decisions the graph already understands.

Nothing here touches the graph directly. :class:`runs.RunManager` drives runs
and reports where each one is; this module only translates between that and a
conversation. It reaches the user two ways:

* a message arrives -> :meth:`WhatsAppBot.handle` (answers a gate, or starts a run)
* a run reaches a gate or finishes -> :meth:`WhatsAppBot.on_run_settled`
"""

from __future__ import annotations

import asyncio
import logging
import mimetypes
import shutil
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable

from agents.converse import Reply, interpret
from config import Settings
from conversations import ConversationStore
from runs import NotAtGate, RunBusy, RunManager, UnknownRun
from state import Asset, extract_url
from tools.media import shrink_to_fit
from tools.whatsapp import (
    MAX_VIDEO_BYTES,
    MEDIA_KINDS,
    Incoming,
    WhatsAppClient,
    WhatsAppError,
    is_allowed,
)
from video_plan import parse_seconds, resolve_seconds

logger = logging.getLogger(__name__)

Interpreter = Callable[..., Awaitable[Reply]]

#: What a run still working is doing, in words a person would use.
_BUSY = {
    "reading_uploads": "reading your files",
    "classifying": "reading your request",
    "researching": "researching the topic",
    "writing": "writing the script",
    "generating": "generating the video, which takes a few minutes",
    "publishing": "posting to Instagram",
}

_GATES = {"awaiting_prompt_approval", "awaiting_post_decision"}

HELP = (
    "I turn a topic or a link into a short video. Send me an idea, like "
    "\"explain how RAG works to non-technical people\", or paste a link. You can "
    "also send photos, a video or a voice note first, and I'll take the look, the "
    "person and the sound from them."
)

#: A request can use at most this many uploaded files.
MAX_UPLOADS = 6

#: The largest single file taken from a chat.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

#: How long files wait for a request, and how long a run's files may outlive it.
STAGED_SECONDS = 60 * 60
RUN_FILES_SECONDS = 24 * 60 * 60

_FILE_WORDS = {"image": "photo", "video": "video", "audio": "audio clip"}
_FALLBACK_EXTENSION = {"image": ".jpg", "video": ".mp4", "audio": ".ogg"}
UNCLEAR_SCRIPT = (
    "I didn't catch that. Should I generate this video? Say yes to go ahead, "
    "tell me what to change, or say cancel."
)
UNCLEAR_VIDEO = (
    "I didn't catch that. Should I post this video to Instagram? "
    "Say yes to post it or no to keep it without posting."
)


class WhatsAppBot:
    """Runs a conversation per phone number on top of :class:`RunManager`."""

    def __init__(
        self,
        cfg: Settings,
        runs: RunManager,
        store: ConversationStore,
        client: WhatsAppClient,
        interpreter: Interpreter = interpret,
    ) -> None:
        """
        Args:
            cfg: Settings; supplies the allowlist and the daily cap.
            runs: Starts runs and answers their gates.
            store: Phone number to run mapping.
            client: Sends WhatsApp messages. Replaced by a recorder in tests.
            interpreter: Reads a reply. Replaced by a stub in tests.
        """
        self._cfg = cfg
        self._runs = runs
        self._store = store
        self._client = client
        self._interpret = interpreter
        self._locks: dict[str, asyncio.Lock] = {}
        self._pending: set[asyncio.Task] = set()

    # ── Entry points ──────────────────────────────────────────────────────

    def spawn(self, message: Incoming) -> None:
        """Handle a webhook message in the background.

        Meta wants a ``200`` within seconds and redelivers when it does not get
        one, while understanding a reply takes a model call.
        """
        task = asyncio.create_task(self.handle(message))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def close(self) -> None:
        """Cancel work in flight and release the client, for a clean shutdown."""
        for task in list(self._pending):
            task.cancel()
        await asyncio.gather(*self._pending, return_exceptions=True)
        await self._client.aclose()

    async def handle(self, message: Incoming) -> None:
        """Process one inbound message from start to finish."""
        if not is_allowed(self._cfg, message.sender):
            # No reply: answering strangers only confirms the number is live.
            logger.warning("Ignoring a message from a number that is not allowed")
            return
        if not self._store.first_sighting(message.message_id):
            return  # a redelivery

        async with self._lock(message.sender):
            try:
                await self._handle(message)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - the user must hear something
                logger.exception("Failed handling a message from %s", message.sender)
                try:
                    await self._say(
                        message.sender,
                        "Sorry, something went wrong on my side. Please try again.",
                    )
                except Exception:  # noqa: BLE001 - e.g. the token is the problem
                    logger.error("Could not tell %s about the failure either", message.sender)

    async def on_run_settled(self, video_id: str) -> None:
        """Tell the user what a run needs from them, or how it ended.

        Called by the run manager when a run reaches a gate, finishes or fails.
        Runs that did not start in a chat are ignored.
        """
        phone = self._store.phone_for(video_id)
        if not phone:
            return

        # Same lock as handle(), so an acknowledgement for the answer that
        # woke this run is always sent before what the run says next.
        async with self._lock(phone):
            status = await self._runs.status(video_id)
            state = status["status"]
            if state == "awaiting_prompt_approval":
                await self._send_script_review(phone, status)
            elif state == "awaiting_post_decision":
                await self._send_video(phone, status)
            elif state == "posted":
                await self._say(phone, self._posted_message(status))
                self._finish(video_id)
            elif state == "failed":
                await self._say(
                    phone,
                    f"Something went wrong: {status.get('error', 'unknown error')}\n\n"
                    "Send me the topic again to retry.",
                )
                self._finish(video_id)
            elif state in {"completed", "ended", "interrupted"}:
                # completed/ended follow a "no" the user was already answered
                # for, so there is nothing new to say.
                self._finish(video_id)

    # ── Inbound ───────────────────────────────────────────────────────────

    def _lock(self, phone: str) -> asyncio.Lock:
        return self._locks.setdefault(phone, asyncio.Lock())

    async def _handle(self, message: Incoming) -> None:
        phone = message.sender
        if message.kind in MEDIA_KINDS:
            await self._receive_upload(message)
            return

        text = message.text.strip()
        if message.kind != "text" or not text:
            await self._say(
                phone,
                "I can read text, photos, video and audio, but not that. " + HELP,
            )
            return

        video_id = self._store.active_run(phone)
        status: dict[str, Any] = {}
        if video_id:
            try:
                status = await self._runs.status(video_id)
            except UnknownRun:
                self._finish(video_id)
                video_id = None

        if video_id:
            state = status["status"]
            if state in _GATES:
                await self._answer_gate(phone, video_id, status, text)
                return
            if state in _BUSY or state in {"working", "finishing"}:
                doing = _BUSY.get(state, "working on it")
                await self._say(
                    phone,
                    f"Still {doing}. I'll message you the moment it's ready.",
                )
                return
            if state == "interrupted":
                self._finish(video_id)
                await self._say(
                    phone,
                    "Your earlier video was interrupted by a restart on my side. "
                    "Send the topic again and I'll start it over.",
                )
                return
            self._finish(video_id)  # finished runs never block a new one

        await self._start_run(phone, text)

    # ── Uploads ───────────────────────────────────────────────────────────

    async def _receive_upload(self, message: Incoming) -> None:
        """Take a photo, video or audio file the person sent.

        Files come before the request, in separate messages, so each is staged
        for the phone until a request claims them. A caption counts as the
        request, so one photo with "make a 20 second ad" starts a run at once.
        """
        phone = message.sender
        self._purge_old_uploads()

        if not message.media_id:
            await self._say(phone, "I couldn't read that file. Try sending it again.")
            return

        video_id = self._store.active_run(phone)
        if video_id:
            try:
                state = (await self._runs.status(video_id))["status"]
            except UnknownRun:
                state = "gone"
            if state in _GATES or state in _BUSY or state in {"working", "finishing"}:
                await self._say(
                    phone,
                    "I'm still on your last video. Send your files together with "
                    "your next request, once this one is done.",
                )
                return
            self._finish(video_id)  # a finished run never blocks new files

        staged = self._store.staged_assets(phone)
        if len(staged) >= MAX_UPLOADS:
            await self._say(
                phone,
                f"That's the most I can use at once ({MAX_UPLOADS} files). "
                "Tell me what to make from them.",
            )
            return

        try:
            data, mime = await self._client.download_media(
                message.media_id, MAX_UPLOAD_BYTES
            )
        except WhatsAppError as exc:
            logger.warning("Could not download an upload from %s: %s", phone, exc)
            await self._say(
                phone, f"I couldn't get that file ({exc}). Try sending it again."
            )
            return

        mime = mime or message.mime_type
        folder = Path(staged[0].path).parent if staged else (
            self._cfg.uploads_dir / uuid.uuid4().hex
        )
        folder.mkdir(parents=True, exist_ok=True)
        extension = mimetypes.guess_extension(mime.split(";")[0].strip()) or (
            _FALLBACK_EXTENSION[message.kind]
        )
        # The name is made up here: nothing a sender wrote ever becomes a path.
        path = folder / f"{uuid.uuid4().hex[:12]}{extension}"
        path.write_bytes(data)
        self._store.stage_asset(phone, str(path), message.kind, mime, message.caption)

        caption = message.caption.strip()
        if caption:
            await self._start_run(phone, caption)
        elif not staged:
            # Only the first file is answered: three photos sent together should
            # not get three replies.
            await self._say(
                phone,
                f"Got your {_FILE_WORDS[message.kind]}. Send more if you like, then "
                "tell me what to make, for example \"a 20 second video about my "
                "bakery\". I'll take the look, the person and the sound from "
                "your files.",
            )

    def _finish(self, video_id: str) -> None:
        """End a run for the chat, and delete the files it was given."""
        self._store.finish(video_id)
        self._delete_files(self._store.release_assets(video_id))

    def _purge_old_uploads(self) -> None:
        """Delete files nobody came back for."""
        self._delete_files(self._store.stale_assets(STAGED_SECONDS, RUN_FILES_SECONDS))

    def _delete_files(self, paths: list[str]) -> None:
        """Delete uploaded files and the folder they were kept in.

        A folder is only removed if it is inside the uploads directory, so a bad
        record can never point this at anything else.
        """
        root = self._cfg.uploads_dir.resolve()
        for raw in paths:
            path = Path(raw).resolve()
            path.unlink(missing_ok=True)
            if root in path.parent.parents:
                shutil.rmtree(path.parent, ignore_errors=True)

    @staticmethod
    def _describe_files(assets: list[Any]) -> str:
        """"photo", "2 photos and audio clip": what the person sent, for "your ...".

        No article, so it reads after "your" whatever the mix: "your photo",
        "your 2 photos and audio clip".
        """
        parts = []
        for kind in ("image", "video", "audio"):
            n = sum(1 for a in assets if a.kind == kind)
            if n:
                word = _FILE_WORDS[kind]
                parts.append(f"{n} {word}s" if n > 1 else word)
        if len(parts) <= 2:
            return " and ".join(parts)
        return ", ".join(parts[:-1]) + f" and {parts[-1]}"

    async def _start_run(self, phone: str, text: str) -> None:
        self._purge_old_uploads()
        # A link is a request by definition; only free text needs reading. The
        # length comes from the interpreter, which understands the sentence, or
        # for a link (which never reaches it) from a plain pattern match.
        if extract_url(text):
            reply = Reply(intent="request", video_seconds=parse_seconds(text))
        else:
            reply = await self._interpret(self._cfg, "idle", text)
        if reply.intent != "request":
            await self._say(phone, reply.message or HELP)
            return

        limit = self._cfg.whatsapp_max_runs_per_day
        if self._store.runs_in_last_day(phone) >= limit:
            await self._say(
                phone,
                f"You've reached today's limit of {limit} videos. "
                "Try again tomorrow.",
            )
            return

        seconds, note = resolve_seconds(
            reply.video_seconds, self._cfg.video_seconds_default
        )
        made = self._cfg.plan(seconds).seconds  # can be a second under the ask

        staged = self._store.staged_assets(phone)
        assets = [Asset(path=a.path, kind=a.kind, mime=a.mime, caption=a.caption) for a in staged]
        video_id = await self._runs.create(
            text[:2000], seconds=seconds, assets=assets or None
        )
        self._store.start(phone, video_id)
        self._store.claim_assets(phone, video_id)
        using = (
            f" I'll take the look, the person and the sound from your "
            f"{self._describe_files(staged)}."
            if staged
            else ""
        )
        await self._say(
            phone,
            (f"{note} " if note else "")
            + f"On it. I'll make a {made}-second video.{using} I'll write the "
            "script and send you the exact prompts to approve before anything is "
            "generated.",
        )

    async def _answer_gate(
        self, phone: str, video_id: str, status: dict[str, Any], text: str
    ) -> None:
        awaiting = status["awaiting"]
        stage = awaiting["stage"]
        context = (
            {"title": awaiting.get("title") or "", "hook": awaiting.get("hook") or ""}
            if stage == "script"
            else {"caption": awaiting.get("caption") or ""}
        )
        reply = await self._interpret(self._cfg, stage, text, context)

        decision: tuple[Any, ...]
        if reply.intent == "approve":
            decision = (True, None)
            ack = (
                "Great, generating your video now. This usually takes a few minutes."
                if stage == "script"
                else "Posting it to Instagram now."
            )
        elif reply.intent == "change":
            # "Make it 30 seconds" changes the plan, not just the wording, so the
            # length is handed over alongside the feedback, not left in it.
            seconds, note = (
                resolve_seconds(reply.video_seconds, self._cfg.video_seconds_default)
                if reply.video_seconds
                else (None, "")
            )
            decision = (False, reply.feedback, seconds)
            ack = (
                (f"{note} " if note else "")
                + (
                    f"Got it. Rewriting the script as a "
                    f"{self._cfg.plan(seconds).seconds}-second video. One moment."
                    if seconds
                    else "Got it. Rewriting the script with your changes. One moment."
                )
            )
        elif reply.intent == "decline":
            decision = (False, None)
            ack = (
                "Okay, cancelled. Nothing was generated."
                if stage == "script"
                else "Okay, I won't post it. The video is saved, and you can "
                "send me a new idea any time."
            )
        else:
            fallback = UNCLEAR_SCRIPT if stage == "script" else UNCLEAR_VIDEO
            await self._say(phone, reply.message or fallback)
            return

        try:
            await self._runs.decide(video_id, *decision)
        except (RunBusy, NotAtGate):
            await self._say(phone, "One moment, I'm still working on the last step.")
            return
        await self._say(phone, ack)

    # ── Outbound ──────────────────────────────────────────────────────────

    async def _say(self, phone: str, text: str) -> None:
        await self._client.send_text(phone, text)

    async def _send_script_review(self, phone: str, status: dict[str, Any]) -> None:
        awaiting = status["awaiting"]
        prompts = awaiting.get("prompts") or []
        tags = " ".join(awaiting.get("hashtags") or [])
        seconds = awaiting.get("duration_seconds")
        length_line = f"Length: {seconds} seconds\n" if seconds else ""
        uploads_line = self._uploads_text(awaiting)

        await self._say(
            phone,
            f"*{awaiting.get('title') or 'Your video'}*\n"
            f"Type: {awaiting.get('intent')}\n"
            f"{length_line}"
            f"{uploads_line}"
            f"Hook: {awaiting.get('hook')}\n"
            f"Call to action: {awaiting.get('cta') or 'none'}\n\n"
            f"Here are the exact prompts I'll send to the video model "
            f"({len(prompts)} part{'s' if len(prompts) != 1 else ''}).",
        )
        for i, prompt in enumerate(prompts, 1):
            await self._say(phone, f"*Prompt {i} of {len(prompts)}*\n\n{prompt}")

        critique = awaiting.get("critique") or {}
        notes = f"\nCritic notes: {critique['feedback']}" if critique.get("feedback") else ""
        costly = status.get("model") == "veo" and self._cfg.veo_live
        await self._say(
            phone,
            f"*Caption (used only if you post)*\n{awaiting.get('caption')}\n{tags}\n\n"
            f"Critic score: {awaiting.get('score')}/10{notes}\n\n"
            + ("Generating the video uses paid Veo credits. " if costly else "")
            + "Should I generate it? Say yes to go ahead, tell me what to "
            "change, or say cancel.",
        )

    @staticmethod
    def _uploads_text(awaiting: dict[str, Any]) -> str:
        """What was taken from the person's files, so they can check it."""
        if awaiting.get("uploads_error"):
            return (
                "I couldn't read your files, so this video is made from your "
                "written request alone.\n"
            )
        uploads = awaiting.get("uploads")
        if not uploads:
            return ""
        lines = ["*From your files*"]
        if uploads.get("colors"):
            palette = " ".join(uploads.get("palette") or [])
            lines.append(f"Colours: {uploads['colors']}" + (f" ({palette})" if palette else ""))
        if uploads.get("mood"):
            lines.append(f"Mood: {uploads['mood']}")
        if uploads.get("people"):
            who = ", ".join(uploads["people"])
            photos = uploads.get("reference_photos") or 0
            lines.append(
                f"Person: {who}"
                + (f" (your photos go to the video model, so it can match them)" if photos else "")
            )
        if uploads.get("audio"):
            lines.append(f"Audio: {uploads['audio']}")
        return "\n".join(lines) + "\n\n"

    async def _send_video(self, phone: str, status: dict[str, Any]) -> None:
        awaiting = status["awaiting"]
        files = awaiting.get("files") or []
        tags = " ".join(awaiting.get("hashtags") or [])

        try:
            if not files:
                raise WhatsAppError("no video file was produced")
            # WhatsApp takes at most 16 MB and a 30-second reel can come close,
            # so a copy is compressed for the chat. The original stays as it is,
            # for Instagram.
            to_send = await shrink_to_fit(
                Path(files[0]), MAX_VIDEO_BYTES, awaiting.get("duration_seconds") or 0
            )
            await self._client.send_video(phone, to_send, "Here's your video")
        except WhatsAppError as exc:
            logger.error("Could not send the video to chat: %s", exc)
            await self._say(
                phone,
                f"Your video is ready, but I couldn't send it here ({exc}). "
                f"It's saved at {files[0] if files else 'the output folder'}.",
            )

        notes = "".join(f"\n{note}" for note in awaiting.get("notes") or [])
        dry = (
            "\n\n(DRY_RUN is on, so saying yes won't really post it.)"
            if self._cfg.dry_run
            else ""
        )
        await self._say(
            phone,
            f"*Caption*\n{awaiting.get('caption')}\n{tags}\n{notes}\n\n"
            f"Should I post this to Instagram?{dry}",
        )

    @staticmethod
    def _posted_message(status: dict[str, Any]) -> str:
        post_id = (status.get("result") or {}).get("post_id")
        if post_id == "dry-run":
            return (
                "That was a dry run: nothing was sent to Instagram. Turn "
                "DRY_RUN off to post for real."
            )
        return f"Posted to Instagram. (post id {post_id})"
