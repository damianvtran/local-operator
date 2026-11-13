"""Streamed file bodies with single-range support: ONE implementation for every route.

Lifted from the Library content route (``routes/desktop_library.py``) when the
session attachment routes began serving a peer's cached bytes
(docs/design/remote-files-parity.md §1): both answer ``Range`` the same way, and
two parsers would be two definitions of a satisfiable range.

RAM is flat by construction: a body is a generator of ``CHUNK`` reads, so a 64 MiB
file never exists in memory whole.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import BinaryIO, Iterator

from fastapi.responses import StreamingResponse

#: One read. 64 KiB is the Library route's measured value; the generator is
#: iterated in Starlette's threadpool, so a smaller chunk only costs more hops.
CHUNK = 64 * 1024


def stream_path(path: Path, start: int, length: int) -> Iterator[bytes]:
    """``length`` bytes of ``path`` from ``start``, opened when iteration begins."""
    with Path(path).open("rb") as handle:
        yield from stream_handle(handle, start, length)


def stream_handle(handle: BinaryIO, start: int, length: int) -> Iterator[bytes]:
    """``length`` bytes from an ALREADY-OPEN handle, closed when the body ends.

    Opening before the route answers is what makes serving safe against a
    concurrent cache eviction: the unlink removes the name, the open fd keeps the
    inode (POSIX), so the body completes and the space is freed after.
    """
    try:
        handle.seek(start)
        remaining = length
        while remaining > 0:
            chunk = handle.read(min(CHUNK, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk
    finally:
        handle.close()


def parse_range(header: str, size: int) -> tuple[int, int] | None:
    """``(start, end)`` inclusive for one satisfiable byte range, else ``None``.

    Only a single ``bytes=`` range is honoured. Multi-range answers are legal
    multipart responses, but no surface asks for one and a wrong multipart body is
    worse than a clean 200 whole-file answer.
    """
    if not header.startswith("bytes=") or "," in header:
        return None
    spec = header[6:].strip()
    if "-" not in spec:
        return None
    raw_start, raw_end = spec.split("-", 1)
    try:
        if raw_start == "":
            suffix = int(raw_end)
            if suffix <= 0:
                return None
            start = max(0, size - suffix)
            return start, size - 1
        start = int(raw_start)
        end = int(raw_end) if raw_end else size - 1
    except ValueError:
        return None
    if start < 0 or start > end or start >= size:
        return None
    return start, min(end, size - 1)


def handle_response(
    handle: BinaryIO,
    *,
    media_type: str,
    headers: dict[str, str],
    range_header: str | None,
) -> StreamingResponse:
    """200 with ``Accept-Ranges``, 206 for a satisfiable ``Range``, 416 otherwise."""
    size = os.fstat(handle.fileno()).st_size
    headers = {**headers, "Accept-Ranges": "bytes"}
    if range_header:
        span = parse_range(range_header, size)
        if span is None:
            handle.close()
            return StreamingResponse(
                iter(()),
                status_code=416,
                headers={**headers, "Content-Range": f"bytes */{size}"},
            )
        start, end = span
        length = end - start + 1
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        headers["Content-Length"] = str(length)
        return StreamingResponse(
            stream_handle(handle, start, length),
            status_code=206,
            media_type=media_type,
            headers=headers,
        )
    headers["Content-Length"] = str(size)
    return StreamingResponse(stream_handle(handle, 0, size), media_type=media_type, headers=headers)
