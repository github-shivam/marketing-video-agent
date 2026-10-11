"""Fetch a web page and reduce it to readable text.

Used when the user gives a link instead of a topic, so the classifier and the
screenwriter work from what the page actually says rather than guessing from
the URL. Plain ``httpx`` plus the standard-library HTML parser: no scraping
service or token is needed for this.
"""

from __future__ import annotations

import logging
import re
from html.parser import HTMLParser

import httpx

logger = logging.getLogger(__name__)

#: Enough to capture a landing page's offer without flooding the prompt.
MAX_CHARS = 6000

#: Pages are a few MB at most; refuse to buffer anything larger.
MAX_BYTES = 3 * 1024 * 1024

_SKIP_TAGS = frozenset({"script", "style", "noscript", "svg", "template", "iframe"})
_BLOCK_TAGS = frozenset(
    {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "section",
     "article", "tr", "header", "footer", "main"}
)


class WebpageError(RuntimeError):
    """Raised when a page cannot be fetched or has no readable text."""


class _TextExtractor(HTMLParser):
    """Collects visible text and the page title."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.parts: list[str] = []
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")
        elif tag == "meta":
            fields = dict(attrs)
            name = (fields.get("name") or fields.get("property") or "").lower()
            if name in {"description", "og:description"} and fields.get("content"):
                self.parts.append(f"\n{fields['content']}\n")

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS and self._skip:
            self._skip -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip:
            self.parts.append(data)


def html_to_text(html: str, max_chars: int = MAX_CHARS) -> str:
    """Reduce an HTML document to its title and visible text.

    Args:
        html: The raw document.
        max_chars: Truncation limit for the returned text.

    Returns:
        The title followed by the page text, whitespace-collapsed.
    """
    extractor = _TextExtractor()
    extractor.feed(html)
    extractor.close()

    lines = []
    for raw in "".join(extractor.parts).splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        if line and (not lines or line != lines[-1]):
            lines.append(line)
    body = "\n".join(lines)

    title = re.sub(r"\s+", " ", extractor.title).strip()
    text = f"Title: {title}\n\n{body}" if title else body
    return text[:max_chars].strip()


async def fetch_page_text(url: str, max_chars: int = MAX_CHARS) -> str:
    """Download ``url`` and return its readable text.

    Args:
        url: An ``http(s)`` URL.
        max_chars: Truncation limit for the returned text.

    Returns:
        The page title and visible text.

    Raises:
        WebpageError: On a non-http scheme, a network or HTTP error, a
            non-HTML response, or a page with no readable text.
    """
    if not url.lower().startswith(("http://", "https://")):
        raise WebpageError(f"unsupported URL scheme: {url}")

    try:
        async with httpx.AsyncClient(
            timeout=20,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; marketing-agent/1.0)"},
        ) as client:
            response = await client.get(url)
    except httpx.HTTPError as exc:
        raise WebpageError(f"could not fetch {url}: {type(exc).__name__}: {exc}") from exc

    if response.status_code >= 400:
        raise WebpageError(f"{url} returned HTTP {response.status_code}")
    content_type = response.headers.get("content-type", "")
    if content_type and "html" not in content_type and "text" not in content_type:
        raise WebpageError(f"{url} is {content_type}, not a web page")
    if len(response.content) > MAX_BYTES:
        raise WebpageError(f"{url} is larger than {MAX_BYTES // 1024 // 1024}MB")

    text = html_to_text(response.text, max_chars)
    if not text:
        raise WebpageError(f"{url} has no readable text (it may need JavaScript)")
    logger.info("Fetched %d chars of text from %s", len(text), url)
    return text
