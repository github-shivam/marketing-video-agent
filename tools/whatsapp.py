"""WhatsApp Cloud API: receive messages, send text and video. No LLM.

Meta calls our webhook for every inbound message and expects a fast ``200``;
replies go out through ``POST /<PHONE_NUMBER_ID>/messages``.

Limits that shape this module:

* A free-form message can only be sent within 24 hours of the user's last
  message. A chat that is actively answering approval gates stays inside it.
* A text body is capped at 4096 characters, so long prompts are split.
* A video is capped at 16 MB, which a 15s 720p reel is well under.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from config import Settings

logger = logging.getLogger(__name__)

GRAPH_HOST = "https://graph.facebook.com"

#: WhatsApp rejects a text body over 4096 characters; leave headroom.
MAX_TEXT_CHARS = 3800

#: WhatsApp rejects a video over 16 MB.
MAX_VIDEO_BYTES = 16 * 1024 * 1024

#: A video caption is capped at 1024 characters.
MAX_CAPTION_CHARS = 1000


#: The kinds of message that carry a file the bot can use.
MEDIA_KINDS = ("image", "video", "audio")


class WhatsAppError(RuntimeError):
    """Raised when the Cloud API rejects a request."""


@dataclass(frozen=True)
class Incoming:
    """One inbound message, reduced to what the bot needs."""

    sender: str
    message_id: str
    kind: str  # "text", "image", "video", "audio", or another WhatsApp type
    text: str = ""
    media_id: str = ""  # to download an image, video or audio message
    mime_type: str = ""
    caption: str = ""  # the words a person typed with a photo or video


# ── Inbound ───────────────────────────────────────────────────────────────


def verify_signature(app_secret: str, body: bytes, header: str | None) -> bool:
    """Check Meta's ``X-Hub-Signature-256`` header against the raw request body.

    Anyone can POST to a public webhook, and a forged message would start runs
    that spend money, so every delivery must prove it came from Meta.

    Args:
        app_secret: The Meta app secret.
        body: The request body exactly as received, before any JSON parsing.
        header: The header value, ``sha256=<hex>``.

    Returns:
        True only if the signature matches.
    """
    if not app_secret or not header or not header.startswith("sha256="):
        return False
    expected = hmac.new(app_secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header.removeprefix("sha256="))


def parse_incoming(payload: dict[str, Any]) -> list[Incoming]:
    """Pull the user messages out of a webhook payload.

    Delivery receipts (``statuses``) and anything malformed are skipped; a
    single payload can batch several messages from several users.
    """
    found: list[Incoming] = []
    for entry in payload.get("entry") or []:
        for change in entry.get("changes") or []:
            for message in (change.get("value") or {}).get("messages") or []:
                sender = str(message.get("from") or "")
                message_id = str(message.get("id") or "")
                if not sender or not message_id:
                    continue
                kind = str(message.get("type") or "unknown")
                text = media_id = mime_type = caption = ""
                if kind == "text":
                    text = str((message.get("text") or {}).get("body") or "")
                else:
                    media = message.get(kind)
                    if isinstance(media, dict):
                        media_id = str(media.get("id") or "")
                        mime_type = str(media.get("mime_type") or "")
                        caption = str(media.get("caption") or "")
                found.append(
                    Incoming(sender, message_id, kind, text, media_id, mime_type, caption)
                )
    return found


def normalise_number(number: str) -> str:
    """Reduce a phone number to digits, which is how WhatsApp reports senders."""
    return re.sub(r"\D", "", number)


def is_allowed(settings: Settings, sender: str) -> bool:
    """Whether ``sender`` may use the bot.

    Every approved request spends Veo money and a ``yes`` can post to the
    company's Instagram, so the bot answers a named list of numbers. ``*``
    opens it to everyone and must be written out deliberately.
    """
    entries = [e.strip() for e in settings.whatsapp_allowed_numbers.split(",") if e.strip()]
    if "*" in entries:
        return True
    return normalise_number(sender) in {normalise_number(e) for e in entries}


# ── Outbound ──────────────────────────────────────────────────────────────


def chunk_text(text: str, limit: int = MAX_TEXT_CHARS) -> list[str]:
    """Split ``text`` into messages of at most ``limit`` characters.

    Breaks at paragraph boundaries where it can, then at line breaks, then at
    spaces, so a long prompt is never cut in the middle of a word.
    """
    text = text.strip()
    if len(text) <= limit:
        return [text] if text else []

    chunks: list[str] = []
    rest = text
    while len(rest) > limit:
        window = rest[:limit]
        cut = max(window.rfind("\n\n"), window.rfind("\n"), window.rfind(" "))
        if cut < limit // 2:  # no sensible break; hard cut rather than loop forever
            cut = limit
        chunks.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    if rest:
        chunks.append(rest)
    return chunks


class WhatsAppClient:
    """Sends messages through the Cloud API."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self._client = client or httpx.AsyncClient(timeout=120)
        self._owns_client = client is None

    @property
    def _base(self) -> str:
        s = self.settings
        return f"{GRAPH_HOST}/{s.whatsapp_api_version}/{s.whatsapp_phone_number_id}"

    @property
    def _auth(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.settings.whatsapp_access_token}"}

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    @staticmethod
    def _check(response: httpx.Response, action: str) -> dict[str, Any]:
        """Return the JSON body, or raise with Meta's own error text."""
        try:
            payload = response.json()
        except ValueError:
            raise WhatsAppError(
                f"{action} failed: HTTP {response.status_code} with a non-JSON body: "
                f"{response.text[:300]}"
            ) from None
        if response.status_code >= 400 or "error" in payload:
            error = payload.get("error", payload)
            hint = ""
            code = error.get("code") if isinstance(error, dict) else None
            if code == 190:
                # The most common first-day failure: the dashboard's temporary
                # token lasts about an hour.
                hint = (
                    " [WHATSAPP_ACCESS_TOKEN is invalid or has expired. Generate "
                    "a new one, update .env and restart the server. A System User "
                    "token does not expire.]"
                )
            elif code == 131005:
                # "Access denied". The token is refused, or lacks a permission, or
                # is not allowed to use this phone number.
                hint = (
                    " [Meta refused WHATSAPP_ACCESS_TOKEN. Most often it is a "
                    "temporary token from the getting-started page that has "
                    "expired or been replaced, or the server is still running "
                    "with an older one: .env is read only at startup, so restart "
                    "the server after changing it. Otherwise check the token has "
                    "whatsapp_business_messaging and that its System User has "
                    "this WhatsApp account assigned. A System User token never "
                    "expires.]"
                )
            raise WhatsAppError(
                f"{action} failed: HTTP {response.status_code} - {error}{hint}"
            )
        return payload

    async def download_media(self, media_id: str, max_bytes: int) -> tuple[bytes, str]:
        """Download a file a person sent.

        WhatsApp gives a message a media id, not the file: the id is looked up for
        a short-lived URL, and the URL is fetched with the same token.

        Args:
            media_id: The id from the incoming message.
            max_bytes: The largest file to accept. Checked against the size WhatsApp
                reports before downloading, and again against what arrives.

        Returns:
            ``(bytes, mime_type)``.

        Raises:
            WhatsAppError: If the file cannot be found, is too large, or the
                download fails.
        """
        s = self.settings
        lookup = await self._client.get(
            f"{GRAPH_HOST}/{s.whatsapp_api_version}/{media_id}", headers=self._auth
        )
        info = self._check(lookup, "Media lookup")
        url = str(info.get("url") or "")
        if not url.startswith("https://"):
            # The token goes to this URL, so it must at least be a secure one.
            raise WhatsAppError("Media lookup returned no usable download link")

        size = int(info.get("file_size") or 0)
        if size > max_bytes:
            raise WhatsAppError(
                f"the file is {size / 1024 / 1024:.0f}MB; the most I can take is "
                f"{max_bytes // 1024 // 1024}MB"
            )

        response = await self._client.get(url, headers=self._auth)
        if response.status_code >= 400:
            raise WhatsAppError(f"Media download failed: HTTP {response.status_code}")
        if len(response.content) > max_bytes:
            raise WhatsAppError("the file is larger than I can take")
        return response.content, str(info.get("mime_type") or "")

    async def send_text(self, to: str, body: str) -> None:
        """Send ``body`` to ``to``, split across messages if it is long."""
        for part in chunk_text(body):
            response = await self._client.post(
                f"{self._base}/messages",
                headers=self._auth,
                json={
                    "messaging_product": "whatsapp",
                    "to": to,
                    "type": "text",
                    "text": {"body": part, "preview_url": False},
                },
            )
            self._check(response, "Send message")

    async def send_video(self, to: str, path: Path, caption: str = "") -> None:
        """Upload a local MP4 and send it to ``to``.

        Raises:
            WhatsAppError: If the file is missing, over 16 MB, or Meta rejects it.
        """
        if not path.is_file():
            raise WhatsAppError(f"video file not found: {path}")
        size = path.stat().st_size
        if size > MAX_VIDEO_BYTES:
            raise WhatsAppError(
                f"video is {size / 1024 / 1024:.1f}MB; WhatsApp's limit is "
                f"{MAX_VIDEO_BYTES // 1024 // 1024}MB"
            )

        mime = mimetypes.guess_type(path.name)[0] or "video/mp4"
        upload = await self._client.post(
            f"{self._base}/media",
            headers=self._auth,
            data={"messaging_product": "whatsapp", "type": mime},
            files={"file": (path.name, path.read_bytes(), mime)},
        )
        media_id = self._check(upload, "Media upload").get("id")
        if not media_id:
            raise WhatsAppError("Media upload returned no id")

        response = await self._client.post(
            f"{self._base}/messages",
            headers=self._auth,
            json={
                "messaging_product": "whatsapp",
                "to": to,
                "type": "video",
                "video": {"id": media_id, "caption": caption[:MAX_CAPTION_CHARS]},
            },
        )
        self._check(response, "Send video")
