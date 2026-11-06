"""The facts a link card shows, extracted from fetched BYTES only.

Three rules hold this module together, and each one is a bug found the hard way:

* **Decode with the DECLARED charset.** A blind ``latin-1`` fallback (the obvious
  shortcut, because it never raises) mojibakes EVERY non-ASCII title: a Russian,
  Greek or Japanese page renders as question-mark soup that survives every unit
  test written against ASCII fixtures. The declared charset is the header's
  parameter, then the document's own ``<meta charset>``, then UTF-8.
* **Bound the parse.** Markup comes from a stranger. The parse reads a capped
  prefix and stops after a capped number of tags, so a 40 MB page of ``<div>``s
  cannot turn into an unbounded walk, and nothing here ever decodes an image.
* **Never trust ``Content-Type`` on an image.** A server can label an SVG
  ``image/png``; an SVG is a DOCUMENT — it carries script and external
  references, so it is rejected outright rather than served from this backend's
  origin. Raster is decided by magic bytes.
"""

from __future__ import annotations

import codecs
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Final
from urllib.parse import urljoin, urlsplit

#: Markup we are willing to look at. ``<head>`` metadata lives in the first few
#: KB of every real page; 512 KB is generous for a page whose generator wrote a
#: wall of inline JSON before the tags we need, and small enough that the parse
#: stays cheap inside a UI-facing deadline.
MAX_MARKUP_BYTES: Final = 512 * 1024

#: Tags examined before the walk stops. A real page's head is < 200 tags; the cap
#: exists so a page built to be hostile costs a bounded amount of work.
_MAX_TAGS: Final = 4000

MAX_TITLE_CHARS: Final = 512
MAX_DESCRIPTION_CHARS: Final = 2048
MAX_SITE_NAME_CHARS: Final = 256
MAX_URL_CHARS: Final = 4096

#: The ``charset`` parameter of a Content-Type, and of a ``<meta>`` tag inside the
#: document. Scanned as BYTES over the first :data:`MAX_MARKUP_BYTES` through a
#: byte regex: a charset probe must not have to decode the document it is trying
#: to find the encoding for.
_HEADER_CHARSET = re.compile(r"charset\s*=\s*[\"']?([A-Za-z0-9_.:-]+)", re.IGNORECASE)
_META_CHARSET = re.compile(
    rb"<meta[^>]{0,200}?charset\s*=\s*[\"']?([A-Za-z0-9_.:-]+)", re.IGNORECASE
)

#: Charsets every Python build names, so the fallback chain below can try each
#: without a LookupError dance in the hot path.
_UTF8: Final = "utf-8"

_HTML_MIME_TYPES: Final = frozenset({"text/html", "application/xhtml+xml"})

_ICON_RELS: Final = frozenset(
    {"icon", "shortcut", "apple-touch-icon", "apple-touch-icon-precomposed"}
)
_APPLE_ICON_RELS: Final = frozenset({"apple-touch-icon", "apple-touch-icon-precomposed"})


@dataclass(frozen=True)
class PageMetadata:
    """What a card renders, plus the two URLs whose BYTES are fetched next."""

    title: str = ""
    description: str = ""
    site_name: str = ""
    icon_url: str | None = None
    image_url: str | None = None

    @property
    def is_empty(self) -> bool:
        """True when the page said nothing a card could show."""
        return not (self.title or self.description or self.icon_url or self.image_url)


@dataclass(frozen=True)
class Raster:
    """A sniffed image: its real mime, and its dimensions when the header says.

    ``width``/``height`` are ``None`` when the format's header does not carry
    them cheaply (ICO stores a byte per axis in its directory entry, which is a
    different question than an image's pixel size). They are NEVER obtained by
    decoding pixels: a 2 MB image is not worth a decoder on a UI thread.
    """

    mime: str
    width: int | None = None
    height: int | None = None


def media_type(content_type: str) -> str:
    """The bare mime type of a ``Content-Type`` header, lowercased.

    ``"text/html; charset=UTF-8"`` → ``"text/html"``. An empty header stays
    empty, which callers must read as "the origin did not say".
    """
    return content_type.split(";", 1)[0].strip().lower()


def is_html_type(content_type: str) -> bool:
    return media_type(content_type) in _HTML_MIME_TYPES


def looks_like_markup(body: bytes) -> bool:
    """Whether unlabelled bytes start with an HTML tag.

    Only consulted when the origin declared no usable type: many origins serve a
    page with no ``Content-Type`` at all, and refusing those would lose real
    previews. A leading ``<`` after whitespace/BOM is the cheap, honest test.
    """
    head = body[:64].lstrip(b"\xef\xbb\xbf").lstrip()
    return head.startswith(b"<")


def decode_markup(body: bytes, content_type: str) -> str:
    """Decode a page with the charset it DECLARED, falling back to UTF-8.

    Order, and why: a byte-order mark is authoritative (the spec says so and a
    BOM-bearing file is unambiguous); then the header's ``charset`` parameter;
    then the document's first ``<meta charset>``; then UTF-8. Never latin-1
    unless something declared it — the "it never raises" convenience is exactly
    what mojibake'd every non-ASCII title in the earlier implementation.
    """
    body = body[:MAX_MARKUP_BYTES]
    for name in _charset_candidates(body, content_type):
        try:
            codec = codecs.lookup(name)
        except LookupError:
            # A declared-but-unknown name is a broken declaration, not a reason
            # to guess latin-1: keep walking the chain.
            continue
        return body.decode(codec.name, errors="replace")
    return body.decode(_UTF8, errors="replace")


def _charset_candidates(body: bytes, content_type: str) -> list[str]:
    candidates: list[str] = []
    if body.startswith(codecs.BOM_UTF8):
        candidates.append("utf-8-sig")
    elif body.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        candidates.append("utf-16")
    header = _HEADER_CHARSET.search(content_type or "")
    if header:
        candidates.append(header.group(1))
    meta = _META_CHARSET.search(body[:4096])
    if meta:
        candidates.append(meta.group(1).decode("ascii", errors="ignore"))
    return candidates


def extract_metadata(markup: str, *, base_url: str) -> PageMetadata:
    """Parse the bounded prefix of ``markup`` into card facts.

    ``base_url`` resolves relative ``href``/``content`` values. It is the URL the
    body was ACTUALLY served from — after redirects — because a site that 301s to
    ``/en/`` may reference ``./icon.png`` relative to where it really lives.
    """
    parser = _MetadataParser(base_url=base_url)
    try:
        parser.feed(markup)
        parser.close()
    except Exception:  # noqa: BLE001 — a malformed document must not fail a card
        # ``html.parser`` is lenient by design, but a pathological document can
        # still walk into an assertion inside it. Whatever was collected before
        # that point is a valid, partial answer and is strictly better than an
        # error for a preview.
        pass
    return parser.result()


class _MetadataParser(HTMLParser):
    """Collects title/description/site name/icon/image, within bounds."""

    def __init__(self, *, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self._base_url = base_url
        self._tags = 0
        self._in_title = False
        self._title_parts: list[str] = []
        self._og_title = ""
        self._og_description = ""
        self._description = ""
        self._site_name = ""
        self._icon: str | None = None
        self._apple_icon: str | None = None
        self._image: str | None = None

    # -- the bounded walk ------------------------------------------------
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._tags += 1
        if self._tags > _MAX_TAGS:
            return
        name = tag.lower()
        if name == "title":
            self._in_title = True
            return
        values = {key.lower(): (value or "") for key, value in attrs}
        if name == "meta":
            self._meta(values)
        elif name == "link":
            self._link(values)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title and self._tags <= _MAX_TAGS:
            self._title_parts.append(data)

    # -- attribute families ----------------------------------------------
    def _meta(self, values: dict[str, str]) -> None:
        key = (values.get("property") or values.get("name") or "").strip().lower()
        content = _bounded(values.get("content", ""), MAX_DESCRIPTION_CHARS)
        if not key or not content:
            return
        if key == "og:title" and not self._og_title:
            self._og_title = _bounded(content, MAX_TITLE_CHARS)
        elif key == "og:description" and not self._og_description:
            self._og_description = content
        elif key == "description" and not self._description:
            self._description = content
        elif key == "og:site_name" and not self._site_name:
            self._site_name = _bounded(content, MAX_SITE_NAME_CHARS)
        elif key == "og:image" and not self._image:
            self._image = _resolve(self._base_url, content)

    def _link(self, values: dict[str, str]) -> None:
        rels = {token.lower() for token in values.get("rel", "").split()}
        if not rels & _ICON_RELS:
            return
        # ``mask-icon`` is an SVG by definition and is not in the set; only the
        # two families we can actually serve get here.
        href = _resolve(self._base_url, values.get("href", ""))
        if href is None:
            return
        if rels & _APPLE_ICON_RELS:
            # Preferred when both are offered: a touch icon is a real raster app
            # icon, while a bare ``rel=icon`` is often a 16 px favicon that looks
            # like a smudge in a message card.
            if self._apple_icon is None:
                self._apple_icon = href
        elif self._icon is None:
            self._icon = href

    def result(self) -> PageMetadata:
        title = self._og_title or _bounded("".join(self._title_parts), MAX_TITLE_CHARS)
        return PageMetadata(
            title=_collapse(title),
            description=_collapse(self._og_description or self._description),
            site_name=_collapse(self._site_name),
            icon_url=self._apple_icon or self._icon,
            image_url=self._image,
        )


def _bounded(value: str, limit: int) -> str:
    return value[:limit]


def _collapse(value: str) -> str:
    """One line, no runs of whitespace — a card renders a single-line title."""
    return " ".join(value.split())


def _resolve(base_url: str, raw: str) -> str | None:
    """Absolute http(s) URL for ``raw`` against ``base_url``, or ``None``.

    Only http/https survive: a ``data:`` or ``javascript:`` value in an
    ``og:image`` is not something this backend will fetch or serve, and dropping
    it here means the fetcher never sees it.
    """
    candidate = (raw or "").strip()
    if not candidate or len(candidate) > MAX_URL_CHARS:
        return None
    absolute = urljoin(base_url, candidate)
    parsed = urlsplit(absolute)
    if parsed.scheme.lower() not in ("http", "https") or not parsed.hostname:
        return None
    return absolute


# ---------------------------------------------------------------------------
# Raster sniffing
# ---------------------------------------------------------------------------

_PNG_SIGNATURE: Final = b"\x89PNG\r\n\x1a\n"
_GIF_SIGNATURES: Final = (b"GIF87a", b"GIF89a")
_ICO_SIGNATURE: Final = b"\x00\x00\x01\x00"

#: JPEG start-of-frame markers that carry the frame header. Deliberately the
#: closed set: C4 (Huffman tables), C8 (reserved) and CC (arithmetic conditioning)
#: sit inside the same range and are NOT frames, and reading them as one yields
#: nonsense dimensions from unrelated bytes.
#: How far into a JPEG the segment walk may look for its start-of-frame header.
#:
#: The walk is a Python loop over ATTACKER-SUPPLIED bytes, so it has to be bounded
#: by something other than the body length: a 2 MiB body of ``FF D8 FF`` followed by
#: repeated ``FF D0`` (restart markers, no SOF ever) measured **1.75 s** of pure
#: loop before this cap, and the bytes were then stored as an image with unknown
#: dimensions. 64 KiB is comfortably past every real encoder's APP0/APP1 block, so
#: a legitimate photo still reports its dimensions; a stream that has not named a
#: frame by then answers "unknown" — the contract allows ``None`` per axis, so the
#: cap costs an optional field rather than the card.
_JPEG_SOF_SCAN_BYTES: Final = 64 * 1024

_JPEG_SOF: Final = frozenset(
    {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
)

#: The ONLY types this backend will ever serve back. Everything else — including
#: SVG, which is a document — is refused at the sniff, whatever the origin's
#: ``Content-Type`` claimed.
RASTER_MIME_TYPES: Final = frozenset(
    {
        "image/png",
        "image/jpeg",
        "image/webp",
        "image/gif",
        "image/vnd.microsoft.icon",
    }
)


def sniff_raster(body: bytes) -> Raster | None:
    """Identify raster bytes by their magic number; ``None`` for anything else.

    Dimensions come from the format's own header — never by decoding pixels.
    """
    if len(body) < 12:
        # Shorter than every signature and every header we read below. A 1×1 GIF
        # is 35 bytes, an ICO is at least 22, so there is no real image here.
        return None
    if body.startswith(_PNG_SIGNATURE):
        width, height = _png_size(body)
        return Raster("image/png", width, height)
    if body.startswith(b"\xff\xd8\xff"):
        width, height = _jpeg_size(body)
        return Raster("image/jpeg", width, height)
    if body.startswith(_GIF_SIGNATURES):
        width = int.from_bytes(body[6:8], "little")
        height = int.from_bytes(body[8:10], "little")
        return Raster("image/gif", _sane(width), _sane(height))
    if body[:4] == b"RIFF" and body[8:12] == b"WEBP":
        width, height = _webp_size(body)
        return Raster("image/webp", width, height)
    if body[:4] == _ICO_SIGNATURE:
        # ICO carries its pixel sizes as single bytes in the directory entry
        # (``0`` means 256). They describe the container's slot, not a decoded
        # frame, which is exactly what a card needs.
        width = body[6] or 256
        height = body[7] or 256
        return Raster("image/vnd.microsoft.icon", width, height)
    return None


def _sane(value: int) -> int | None:
    return value if 0 < value <= 65535 else None


def _png_size(body: bytes) -> tuple[int | None, int | None]:
    # IHDR is mandated first: 8 signature bytes, 4 length, 4 type, then 4+4.
    if len(body) < 24 or body[12:16] != b"IHDR":
        return None, None
    return _sane(int.from_bytes(body[16:20], "big")), _sane(int.from_bytes(body[20:24], "big"))


def _jpeg_size(body: bytes) -> tuple[int | None, int | None]:
    """Walk the segment chain to the first start-of-frame header.

    Bounded by the buffer: every segment carries its own 2-byte length, and a
    marker whose length runs past the end simply ends the walk with no
    dimensions — never a wrong pair read from arbitrary bytes.
    """
    index = 2
    # The cap, not the body: see :data:`_JPEG_SOF_SCAN_BYTES`. Every bound below
    # uses it, so no read can walk past it either.
    limit = min(len(body), _JPEG_SOF_SCAN_BYTES)
    while index + 4 <= limit:
        if body[index] != 0xFF:
            index += 1
            continue
        marker = body[index + 1]
        # Pad bytes (0xFF fill) are legal between segments.
        if marker == 0xFF:
            index += 1
            continue
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            index += 2
            continue
        if index + 4 > limit:
            return None, None
        length = int.from_bytes(body[index + 2 : index + 4], "big")
        if length < 2:
            return None, None
        if marker in _JPEG_SOF:
            # precision(1) height(2) width(2) start at index+4.
            if index + 9 > limit:
                return None, None
            height = int.from_bytes(body[index + 5 : index + 7], "big")
            width = int.from_bytes(body[index + 7 : index + 9], "big")
            return _sane(width), _sane(height)
        index += 2 + length
    return None, None


def _webp_size(body: bytes) -> tuple[int | None, int | None]:
    """Three WebP flavours, three different header layouts."""
    chunk = body[12:16]
    if chunk == b"VP8 ":
        # Lossy: 3-byte frame tag, then the 3-byte sync code, then 14-bit dims.
        if len(body) < 30 or body[23:26] != b"\x9d\x01\x2a":
            return None, None
        width = int.from_bytes(body[26:28], "little") & 0x3FFF
        height = int.from_bytes(body[28:30], "little") & 0x3FFF
        return _sane(width), _sane(height)
    if chunk == b"VP8L":
        # Lossless: signature byte 0x2F, then 14 bits width-1 and 14 bits
        # height-1 packed little-endian into the next 4 bytes.
        if len(body) < 25 or body[20] != 0x2F:
            return None, None
        packed = int.from_bytes(body[21:25], "little")
        width = (packed & 0x3FFF) + 1
        height = ((packed >> 14) & 0x3FFF) + 1
        return _sane(width), _sane(height)
    if chunk == b"VP8X":
        # Extended: 24-bit canvas dimensions minus one, at 24 and 27.
        if len(body) < 30:
            return None, None
        width = int.from_bytes(body[24:27], "little") + 1
        height = int.from_bytes(body[27:30], "little") + 1
        return _sane(width), _sane(height)
    return None, None
