"""A streamed ``multipart/form-data`` body built from in-memory parts.

WHY NOT ``httpx``'s ``files=``. Two RFC constraints (§6.5, §7.2): uploads are
STREAMED, and a submission body is never written anywhere — no temp file, no
spill, no log. httpx's multipart encoder is fine on the wire, but the body this
client sends has to be re-sendable for the one forced-refresh retry after a 401
(the pinned-credential rule), and the image parts are already bounded bytes the
caller holds (≤ 4 × 5 MiB, validated before this runs). So this is a tiny
encoder whose ``__aiter__`` yields a FRESH chunk stream every time it is
iterated: the retry re-streams the same memory instead of buffering a joined
copy, and nothing in it ever touches the filesystem.

The ``Content-Length`` is computed up front, so httpx sends a sized body rather
than chunked transfer encoding (some ingress proxies buffer chunked uploads to
disk to learn their length, which is exactly the spill the RFC forbids).

FILE NAMES ARE NEVER THE USER'S (canon §7: "server-generated file names, never
user filenames"): the part name is ``images`` and the filename a fixed
``image-<n>.<ext>``, so a local path or a revealing name cannot travel.
"""

from __future__ import annotations

import secrets
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

#: Upload slice size. Small enough that a slow link interleaves with other
#: requests on the loop; large enough that a 5 MiB image is ~80 yields.
CHUNK_BYTES = 64 * 1024

_CRLF = b"\r\n"


@dataclass(frozen=True)
class TextPart:
    name: str
    value: str


@dataclass(frozen=True)
class FilePart:
    name: str
    filename: str
    content_type: str
    data: bytes


Part = TextPart | FilePart


def _quote(value: str) -> str:
    # Names here are this module's own constants (never user input), but quote
    # defensively so a future caller cannot break the header grammar.
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\r", "").replace("\n", "")


class StreamedMultipart:
    """A re-iterable, sized multipart body. Construct once; iterate per attempt."""

    def __init__(self, parts: Sequence[Part], *, boundary: str | None = None) -> None:
        self._boundary = boundary or ("lo-community-" + secrets.token_hex(16))
        self._segments: list[tuple[bytes, bytes]] = []
        for part in parts:
            if isinstance(part, FilePart):
                head = (
                    f'Content-Disposition: form-data; name="{_quote(part.name)}"; '
                    f'filename="{_quote(part.filename)}"\r\n'
                    f"Content-Type: {part.content_type}\r\n\r\n"
                ).encode("ascii")
                payload = part.data
            else:
                head = (
                    f'Content-Disposition: form-data; name="{_quote(part.name)}"\r\n\r\n'
                ).encode("ascii")
                payload = part.value.encode("utf-8")
            self._segments.append((b"--" + self._boundary.encode("ascii") + _CRLF + head, payload))
        self._closing = b"--" + self._boundary.encode("ascii") + b"--" + _CRLF

    @property
    def content_type(self) -> str:
        return f"multipart/form-data; boundary={self._boundary}"

    @property
    def content_length(self) -> int:
        total = len(self._closing)
        for head, payload in self._segments:
            total += len(head) + len(payload) + len(_CRLF)
        return total

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self._stream()

    async def _stream(self) -> AsyncIterator[bytes]:
        for head, payload in self._segments:
            yield head
            view = memoryview(payload)
            for start in range(0, len(view), CHUNK_BYTES):
                yield bytes(view[start : start + CHUNK_BYTES])
            yield _CRLF
        yield self._closing


__all__ = ["CHUNK_BYTES", "FilePart", "StreamedMultipart", "TextPart"]
