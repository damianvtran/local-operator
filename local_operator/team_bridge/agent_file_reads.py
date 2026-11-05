"""Agent reads of generic files — the receiving runtime's verification half.

WHY THIS MODULE EXISTS. ``agent_files.py`` is the SEND half (stage, bind, post); this
is the half the receiving agent needs: the commitment it must match, the shape of the
descriptor it may see, and the bytes it may put in front of a model. It is deliberately
image-free — the image lane (``attachments.py``) keeps its own ``attachments_hash``,
its own sniffing digests and its own model bound.

THE HUB CONTRACT (agent-server ``docs/bridge-files-agent-read.md``, the contract of
record). A message's generic files are committed NAME-FREE:

    BridgeFilesAdmissionHashV1 =
        sha256("bridge-files-admission-v1\\n" || compact-json([[id, class, size, sha256], …]))

computed in SEND ORDER over the descriptors the hub showed. The name is excluded on
purpose: an agent is never shown the uploader's filename, so a commitment folding it in
could only be TRUSTED, never RECOMPUTED, and the receiving runtime's whole half of the
admission invariant — "the set I screened is the set the hub bound" — would be lost.
:func:`files_admission_hash` therefore re-derives the hub's own number from the
descriptor list, and a mismatch means the runtime admits nothing.

The domain separator and the JSON bytes are the hub's, not ours: ``encoding/json``
compacts and HTML-escapes (``<``, ``>``, ``&``, U+2028, U+2029 -> ``\\uXXXX``; controls
other than ``\\n``/``\\r``/``\\t`` as ``\\u00XX``, i.e. NOT Python's ``\\b``/``\\f``).
:func:`_go_json_string` reproduces those exact bytes because the hub's golden vectors
are the parity test; ``json.dumps`` differs on those escapes and would silently produce
a different commitment for an ID containing one.

TEXT ONLY. The hub serves only a ``text``-class row to an agent (every other class is
the uniform 404), so a non-text descriptor is reported — descriptor and "bytes were not
read" — and never fetched. What reaches a model is untrusted submission data: the
caller must lead it with the bridge untrusted clause, which is why this module renders
content and coverage but never a model-facing preamble of its own.

Nothing here persists or logs bytes, ids or hashes — the same no-body-cache rule as
message text (``attachments.py``).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


__all__ = [
    "FILE_CLASSES",
    "FILE_MAX_BYTES",
    "FILE_READ_MAX_BYTES_PER_FILE",
    "FILE_READ_MAX_BYTES_TOTAL",
    "FILE_READ_MAX_FILES",
    "FILE_SERVED_MIME",
    "FILES_HASH_DOMAIN",
    "FILES_MAX",
    "FileDescriptor",
    "FileRejected",
    "FileText",
    "bound_text",
    "context_caption",
    "digest_file",
    "fetch_plan",
    "files_admission_hash",
    "parse_file_descriptors",
    "screen_listing",
    "withheld_marker",
]

#: The hub's closed class set (``models.BridgeFileClasses``). A LABEL, never a safety
#: claim: the hub byte-sniffs it at upload and never recomputes it on read.
FILE_CLASSES = ("text", "pdf", "container", "other")
#: The ONE class an agent may read (decided: text-only; every other class needs a
#: screened projection that does not exist yet).
READABLE_FILE_CLASS = "text"
#: The hub's per-message bound across images AND files (``BridgeFileMaxPerMessage``);
#: a longer descriptor list is malformed, not truncated.
FILES_MAX = 10
#: The hub's per-class UPLOAD caps (``models.BridgeFileClassMaxBytes``). They are upload
#: limits, not safety ratings: text is cheap to screen, containers are the largest
#: legitimate sharing case. A descriptor claiming more than its class cap is malformed.
FILE_CLASS_MAX_BYTES = {
    "pdf": 20 * 1024 * 1024,
    "text": 2 * 1024 * 1024,
    "container": 25 * 1024 * 1024,
    "other": 25 * 1024 * 1024,
}
#: The transport bound (``BridgeFileMaxBytes``): no descriptor may claim more at all.
FILE_MAX_BYTES = 25 * 1024 * 1024
#: What the hub SERVES for a generic file: the class never becomes a content type, so
#: a browser is never invited to render one.
FILE_SERVED_MIME = "application/octet-stream"

#: Context-read budgets (decided in the slice brief): three files, 64 KiB per file and
#: 128 KiB per read. The hub deliberately leaves sizing to the client — ``ServeContent``
#: stays range-enabled and ``HISTORY_IMAGES_MAX`` is the image precedent — so these are
#: the runtime's own ceilings, applied BEFORE anything is decoded or handed on.
FILE_READ_MAX_FILES = 3
FILE_READ_MAX_BYTES_PER_FILE = 64 * 1024
FILE_READ_MAX_BYTES_TOTAL = 128 * 1024

#: The commitment's domain separator, byte-for-byte (``models.BridgeFilesAdmissionHashV1``).
FILES_HASH_DOMAIN = b"bridge-files-admission-v1\n"

_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
#: A generic file id is OPAQUE, not ``batt_``-shaped: the hub promises nothing about its
#: spelling ("File IDs are opaque, exact UTF-8, with no prefix or case claims") and mints
#: ``batt_``+UUID in practice. So the shape is NOT a prefix check — but it is not a
#: free-for-all either: the id becomes a URL path segment, so this refuses every
#: character that could reshape the path (``/``, ``?``, ``#``, ``%``), every control or
#: whitespace byte, and anything past the hub's own 64-byte commitment bound. An id
#: outside it is reported as unreadable rather than interpolated into a request.
_FILE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class FileRejected(ValueError):
    """One file failed verification. ``code`` is closed and content-free."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class FileDescriptor:
    """The hub's metadata for one bound generic file. No bytes; safe to render.

    The field set is the agent-visible one: ``id``, ``class``, ``size``, ``sha256``,
    ``state`` and the served ``mime``. ``display_name`` is NEVER part of it — an agent
    has no use for a filename and the commitment does not cover one.
    """

    id: str
    file_class: str
    size: int
    sha256: str
    state: str = "bound"
    mime: str = FILE_SERVED_MIME

    @property
    def fetchable(self) -> bool:
        """Whether the hub still holds bytes (a purged row is ``expired``)."""

        return self.state == "bound"

    @property
    def readable(self) -> bool:
        """Whether an agent may fetch this class at all (text only)."""

        return self.file_class == READABLE_FILE_CLASS

    def commitment_row(self) -> tuple[str, str, int, str]:
        """The ``[id, class, size, sha256]`` row the hub commits to."""

        return (self.id, self.file_class, self.size, self.sha256)


@dataclass(frozen=True)
class FileText:
    """One bounded, decoded text file: what a model is shown and how much of it."""

    descriptor: FileDescriptor
    text: str
    shown_bytes: int
    truncated: bool


def _non_negative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def parse_file_descriptors(raw: Any) -> tuple[FileDescriptor, ...] | None:
    """The validated file descriptor list, ``()`` when absent, ``None`` when malformed.

    Malformed is ANY entry failing the closed shape. The caller then treats the whole
    set as unverifiable rather than admitting a subset: the commitment covers the set,
    so a partial list can never reproduce it — and screening a subset as if complete
    would bind a verdict to bytes the screener never saw.
    """

    if raw is None:
        return ()
    if not isinstance(raw, list) or len(raw) > FILES_MAX:
        return None
    out: list[FileDescriptor] = []
    for entry in raw:
        if not isinstance(entry, Mapping):
            return None
        file_id = entry.get("id")
        file_class = entry.get("class")
        sha = entry.get("sha256")
        size = _non_negative_int(entry.get("size"))
        mime = entry.get("mime", FILE_SERVED_MIME)
        state = entry.get("state", "bound")
        if (
            not isinstance(file_id, str)
            or _FILE_ID.fullmatch(file_id) is None
            or file_class not in FILE_CLASSES
            or not isinstance(sha, str)
            or _SHA256_HEX.fullmatch(sha) is None
            or size is None
            or size > FILE_MAX_BYTES
            or size > FILE_CLASS_MAX_BYTES[str(file_class)]
            or not isinstance(state, str)
            or not isinstance(mime, str)
        ):
            return None
        out.append(
            FileDescriptor(
                id=file_id,
                file_class=str(file_class),
                size=size,
                sha256=sha,
                state=state,
                mime=mime,
            )
        )
    if len({d.id for d in out}) != len(out):
        return None
    return tuple(out)


def require_file_id(file_id: object) -> None:
    """Refuse a file id that could reshape the upstream path or exceed the wire bound.

    An id is opaque, so this is a bounded-shape check rather than a minted-prefix one —
    but it is NOT a weaker check than the image lane's: nothing may carry a ``/``, a
    ``?`` or a path-reshaping byte into a URL segment, and the hub's 64-byte bound is the
    same one its commitment enforces.
    """


def _go_json_string(value: str) -> str:
    """One Go ``encoding/json`` string literal, byte-for-byte.

    WHY NOT ``json.dumps``: the hub's commitment is ``json.Marshal`` output, which
    HTML-escapes ``<``, ``>`` and ``&`` and writes every control character except
    ``\\n``, ``\\r`` and ``\\t`` as ``\\u00XX`` — Python uses ``\\b``/``\\f`` for two of
    those and leaves ``<>&`` literal. A file id is opaque UTF-8, so the two encoders
    disagree on ids the hub accepts, and the runtime would refuse a set it had in fact
    been shown. Lone surrogates raise: ``utf8.ValidString`` refuses such an id at the
    hub, so it can never be in a commitment either.
    """

    out: list[str] = ['"']
    for char in value:
        point = ord(char)
        if char == '"':
            out.append('\\"')
        elif char == "\\":
            out.append("\\\\")
        elif point < 0x20:
            if char == "\n":
                out.append("\\n")
            elif char == "\r":
                out.append("\\r")
            elif char == "\t":
                out.append("\\t")
            else:
                out.append(f"\\u{point:04x}")
        elif char in "<>&" or point in (0x2028, 0x2029):
            out.append(f"\\u{point:04x}")
        else:
            out.append(char)
    out.append('"')
    return "".join(out)


def files_admission_hash(descriptors: Sequence[FileDescriptor]) -> str:
    """``BridgeFilesAdmissionHashV1`` over the descriptors, in SEND ORDER.

    The empty set has a defined hash on purpose, so "the message had files at screen
    time, none at context time" is a MISMATCH and never a silent pass. Order is
    significant: the screener is told the files follow in send order, and a reordered
    set is a different set.

    Returns the hex digest; the caller compares it with the hub's ``files_hash`` (or
    sends it on an allow) and refuses on any difference. Duplicate ids are refused here
    because the hub refuses them: a set whose meaning is ambiguous cannot be committed.
    """

    if len(descriptors) > FILES_MAX:
        raise FileRejected("files_too_many")
    seen: set[str] = set()
    rows: list[str] = []
    for descriptor in descriptors:
        if descriptor.id in seen:
            raise FileRejected("file_duplicate_id")
        seen.add(descriptor.id)
        encoded = ",".join(
            (
                _go_json_string(descriptor.id),
                _go_json_string(descriptor.file_class),
                str(descriptor.size),
                _go_json_string(descriptor.sha256),
            )
        )
        rows.append(f"[{encoded}]")
    payload = FILES_HASH_DOMAIN + ("[" + ",".join(rows) + "]").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def bound_text(data: bytes, limit: int = FILE_READ_MAX_BYTES_PER_FILE) -> tuple[bytes, bool]:
    """The longest complete-rune prefix of ``data`` within ``limit`` bytes.

    A cut mid-rune would hand a model a replacement character that was never in the
    file, so the truncation point backs up to the last complete UTF-8 sequence. The
    hub guarantees a readable row is UTF-8 without NUL; anything else is refused by the
    caller's verification, not repaired here.
    """

    if len(data) <= limit:
        return data, False
    cut = data[:limit]
    # Back up over a partial multi-byte sequence: at most three continuation bytes.
    for back in range(0, min(4, len(cut)) + 1):
        tail = cut[: len(cut) - back] if back else cut
        try:
            tail.decode("utf-8")
        except UnicodeDecodeError:
            continue
        return tail, True
    return b"", True


def digest_file(descriptor: FileDescriptor, data: bytes) -> FileText:
    """Verify a whole file against its descriptor, then bound it for the model.

    Two checks, both from the descriptor the hub showed: the byte count and the
    content hash. A file whose bytes do not hash to what the commitment covers can
    never be presented as that file, so a mismatch is a refusal (the caller reports
    the descriptor and states the bytes were not used) rather than a best-effort body.

    ``truncated`` then limits what a MODEL is shown to the display ceiling: a 200 KiB
    text file read at 64 KiB is truncated, and saying so is what keeps a partial view
    from being read as the whole document.
    """

    if len(data) != descriptor.size:
        raise FileRejected("file_size_mismatch")
    if hashlib.sha256(data).hexdigest() != descriptor.sha256:
        raise FileRejected("file_sha256_mismatch")
    bounded, cut = bound_text(data)
    try:
        text = bounded.decode("utf-8")
    except UnicodeDecodeError:
        raise FileRejected("file_not_text") from None
    if "\x00" in text:
        raise FileRejected("file_not_text")
    return FileText(
        descriptor=descriptor,
        text=text,
        shown_bytes=len(bounded),
        truncated=cut or len(bounded) < descriptor.size,
    )


def fetch_plan(
    descriptors: Sequence[FileDescriptor], *, max_files: int = FILE_READ_MAX_FILES
) -> tuple[tuple[FileDescriptor, ...], tuple[FileDescriptor, ...]]:
    """Split descriptors into ``(fetchable text files to fetch, the rest)``.

    The budget is three files per read, applied in SEND ORDER so the same message yields
    the same three files on every read (a set that varies per read would make a
    long-truncated read look complete). Unreadable classes, expired rows and anything
    past the budget are returned in the second element and are REPORTED by the caller —
    never fetched, never silently dropped.
    """

    to_fetch: list[FileDescriptor] = []
    rest: list[FileDescriptor] = []
    for descriptor in descriptors:
        if descriptor.fetchable and descriptor.readable and len(to_fetch) < max_files:
            to_fetch.append(descriptor)
        else:
            rest.append(descriptor)
    return tuple(to_fetch), tuple(rest)


def screen_listing(
    descriptors: Sequence[FileDescriptor], texts: Mapping[str, "FileText"]
) -> str:
    """The v4 ``{{files}}`` slot: one listing line, then the text that follows as data.

    The hub's placeholder contract (``bridge.ScreenFiles``) is "one line per file as
    class and size, e.g. ``Files: 2 (1: text 4310 bytes; 2: pdf 2301994 bytes)``, and
    for a TEXT-class file its first bytes follow that line, inside the same nonce
    wrapper". Server-sourced metadata only: never a display name, id, hash or MIME —
    the name is content the admitting runtime is never shown.

    A text file's body is the SAME bounded view the context read shows, so the screen
    that admits a file set always screened exactly the bytes a later read can put in
    front of the agent ("one truncation, two consumers"). Coverage is stated whenever
    the body was cut: a screener must never be told a partial file is whole, and the
    absent/over-budget cases say so by name rather than vanishing from the listing.
    """

    parts = "; ".join(
        f"{index}: {d.file_class} {d.size} bytes" for index, d in enumerate(descriptors, start=1)
    )
    lines = [f"Files: {len(descriptors)} ({parts})"]
    for index, descriptor in enumerate(descriptors, start=1):
        if not descriptor.readable:
            lines.append(
                f"File {index} ({descriptor.file_class}, {descriptor.size} bytes): "
                "contents not read."
            )
            continue
        item = texts.get(descriptor.id)
        if item is None:
            lines.append(
                f"File {index} (text, {descriptor.size} bytes): contents not read."
            )
            continue
        coverage = (
            f"first {item.shown_bytes} of {descriptor.size} bytes"
            if item.truncated
            else f"all {item.shown_bytes} bytes"
        )
        lines.append(f"File {index} (text, {descriptor.size} bytes; {coverage}) follows as data:")
        lines.append(item.text)
    return "\n".join(lines)


def context_caption(
    files: Iterable[FileText], unread: Iterable[FileDescriptor], *, source: str
) -> str:
    """The descriptor lines that head a context read — never the bytes themselves.

    Kept apart from :func:`screen_listing` because the two consumers owe the model
    different things: the screener must SEE the bytes to judge them, while the agent's
    context gets provenance (which file, which class, how much of it) around content
    rendered by the caller. A non-text class is stated as explicitly not read: the
    agent asked, and "the hub refused this class to agents" is the fact, not a gap.
    """

    lines = [source]
    for index, item in enumerate(files, start=1):
        coverage = (
            f"first {item.shown_bytes} of {item.descriptor.size} bytes"
            if item.truncated
            else f"all {item.shown_bytes} bytes"
        )
        lines.append(
            f"{index}. file {item.descriptor.id} "
            f"({item.descriptor.file_class}, {item.descriptor.size} bytes) — {coverage}"
        )
    for descriptor in unread:
        if not descriptor.readable:
            lines.append(
                f"- file {descriptor.id} ({descriptor.file_class}, {descriptor.size} bytes) "
                f"— bytes not read: only text-class files are readable by an agent"
            )
        elif not descriptor.fetchable:
            lines.append(
                f"- file {descriptor.id} ({descriptor.file_class}, {descriptor.size} bytes) "
                f"— bytes unavailable: {descriptor.state}"
            )
        else:
            lines.append(
                f"- file {descriptor.id} ({descriptor.file_class}, {descriptor.size} bytes) "
                f"— bytes not read: over this read's 3-file budget"
            )
    return "\n".join(lines)


#: The closed reasons a file set stays unread, and the one sentence each becomes. A model
#: reading a card or a history page gets the CAUSE, never a bare code: "too large to
#: screen" is something a person can act on, ``too_large`` is not.
_WITHHELD_REASONS = {
    "unverified": "could not be verified against the hub's commitment",
    "unadmitted": "not admitted by screening",
    "too_large": "too large to screen in one read",
    "unavailable": "not available to this agent",
}


def withheld_marker(count: int, reason: str) -> str:
    """The honest refusal line: files exist on a message that admitted none to us.

    ``reason`` is the closed cause the caller determined (``unadmitted``: no grant;
    ``unverified``: descriptor set malformed or hash mismatch; ``unavailable``: an
    expired row). The count is metadata the hub serves to every reader, so stating it
    discloses nothing about content.
    """

    noun = "file" if count == 1 else "files"
    return f"[{count} {noun} withheld: {_WITHHELD_REASONS.get(reason, reason)}]"
