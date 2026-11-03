"""Frame SSE bytes without mistaking transport boundaries for event boundaries.

Why: both translator bridges must suppress search-call events even when upstream
uses CRLF or CR, including a CR/LF or UTF-8 codepoint split across chunks. Keep
raw frames alongside joined data so unrelated events, errors, [DONE], and an
unterminated EOF tail survive byte-for-byte; decode only complete JSON payloads
in the caller. This is framing only, not an SSE client or reconnect policy.
"""

from collections.abc import AsyncIterable, AsyncIterator


async def iter_sse(chunks: AsyncIterable[bytes]) -> AsyncIterator[tuple[bytes, bytes | None]]:
    """Yield original frame bytes and multiline data; None means no data field."""
    pending = bytearray()
    frame = bytearray()
    data: list[bytes] = []
    scan = 0
    async for chunk in chunks:
        pending.extend(chunk)
        start = 0
        while scan < len(pending):
            marker = pending[scan]
            if marker not in (10, 13):
                scan += 1
                continue
            # A trailing CR may be half of CRLF: defer it until the next chunk.
            if marker == 13 and scan + 1 == len(pending):
                break
            end = scan + (2 if marker == 13 and pending[scan + 1] == 10 else 1)
            line = bytes(pending[start:scan])
            frame.extend(pending[start:end])
            if not line:
                yield bytes(frame), b"\n".join(data) if data else None
                frame.clear()
                data.clear()
            elif line == b"data" or line.startswith(b"data:"):
                value = line[5:]
                data.append(value[1:] if value.startswith(b" ") else value)
            start = scan = end
        if start:
            del pending[:start]
            scan -= start
    # EOF resolves an ambiguous final CR as a line ending, not half a CRLF.
    if pending.endswith(b"\r"):
        line = bytes(pending[:-1])
        frame.extend(pending)
        pending.clear()
        if not line:
            yield bytes(frame), b"\n".join(data) if data else None
            frame.clear()
    if frame or pending:
        yield bytes(frame + pending), None
