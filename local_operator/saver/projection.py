"""The ONE Saver upload serialiser, and its inverse for verbatim epochs (§3.1, §3.2).

WHY ONE FUNCTION. Everything Radient receives from this client goes through
:func:`saver_projection`. It is an ALLOWLIST over lop's ``Message`` model, never
a ``model_dump`` and never ``ChatRequest.wire_messages()``: a dump would carry
``provider_payload`` (native reasoning replay, including
``native_replay.credential_scope``), ``usage``, row ids, dictation provenance and
attachment paths to a third party, and a later field added to ``Message`` would
leak by default. An allowlist leaks nothing it was not told about (S-29).

THE FORMAT is ``openai.chat``, because lop's provider-neutral rows already have
that shape (``user``/``assistant``/``tool`` roles, string tool arguments, tool
rows keyed by ``tool_call_id``). Per message, only these keys exist:

- ``role``; ``content`` as a list of ``{"type":"text","text":…}`` blocks and
  ``image_ref`` blocks;
- assistant: ``tool_calls`` as ``[{"id", "function": {"name", "arguments"}}]``
  with ``arguments`` the replayable argument STRING;
- tool: ``tool_call_id``, and ``is_error: true`` only when set.

REDACTION IS RE-APPLIED HERE (C-S2-3): ``redact`` (registered values AND the
credential-shape table) on every text and tool-result string, and
``redact_arguments`` (the high-confidence floor that never rewrites code) on
tool-call arguments. A row recorded before its secret was registered is
therefore uploaded without it, and the digest covers the redacted bytes.

MEDIA NEVER TRAVELS. Every media block becomes an ``image_ref`` whose ``sha256``
is taken over the DECODED media bytes (the contract's rev3 correction), so the
upload is a few hundred bytes per image instead of megabytes of base64.

WHY A CACHE. Redaction measured ~0.45 s per MB of history on this host, and the
chain re-digests its frozen prefix every turn (§3.2 step 1). Projections are
memoised per message content and invalidated when the redactor learns a new
value, so a turn pays only for rows it has not projected yet.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
import struct
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from local_operator.harness.types import (
    AttachmentContent,
    AudioContent,
    DocumentContent,
    ImageContent,
    Message,
    TextContent,
    ToolCall,
)
from local_operator.saver.canonical import canonical_bytes

Redactor = Callable[[str], str]


@dataclass(frozen=True)
class Redaction:
    """The session's two redaction passes, and the value set that versions them.

    ``values`` returns the exact values the ``redact`` pass scrubs (the
    session's ``VariableStore.redaction_values``); a change there is what makes
    a cached projection stale.
    """

    redact: Redactor
    redact_arguments: Redactor
    values: Callable[[], Iterable[str]] = lambda: ()


#: How long a provider-secret snapshot is reused before the stores are re-read.
_PROVIDER_SECRET_TTL_S = 30.0


def with_provider_secrets(redaction: Redaction, secrets: Callable[[], Iterable[str]]) -> Redaction:
    """``redaction`` plus an exact-value pass over the stored provider secrets (S-3).

    The S-32 sentence promises Saver never sends the user's AI-provider
    credentials to Radient. The bearer half is the forbidden-set check; this is
    the BODY half: a provider key printed bare in a tool result (``cat``,
    ``env``, a config dump) has no credential shape the shape table knows, and
    the session's store never registered it, so without this it was uploaded
    (security round 1, p21). The values come from the same
    :func:`~local_operator.saver.client.stored_provider_secrets` reader, are
    scrubbed in every spelling by the shared ``scrub_values`` loop, and join
    ``values`` so the projection cache re-projects when one is added.

    They are deliberately NOT registered into the session's ``VariableStore``:
    that store also drives the model-visible redaction, whose behaviour this
    feature must not change for a user who never turns Saver on.
    """
    import time

    from local_operator.redaction_shapes import scrub_values

    # A SNAPSHOT, refreshed when the cache asks for ``values`` (once before and
    # once after each projection) and at most every ``_PROVIDER_SECRET_TTL_S``:
    # the reader opens the auth store and the encrypted store, which is far too
    # costly per text block, and ``redact`` is called per block.
    snapshot: list[str] = []
    read_at = [float("-inf")]

    def _refresh() -> list[str]:
        if time.monotonic() - read_at[0] >= _PROVIDER_SECRET_TTL_S:
            try:
                snapshot[:] = sorted({value for value in secrets() if value})
            except Exception:  # noqa: BLE001 — a failing reader keeps the last snapshot
                pass
            read_at[0] = time.monotonic()
        return snapshot

    def _wrap(inner: Redactor) -> Redactor:
        def _redact(text: str) -> str:
            return inner(scrub_values(text, snapshot)) if snapshot else inner(text)

        return _redact

    def _values() -> list[str]:
        return [*redaction.values(), *_refresh()]

    return Redaction(_wrap(redaction.redact), _wrap(redaction.redact_arguments), _values)


def shape_only_redaction() -> Redaction:
    """The shape-table floor for a stream no session has bound yet.

    Never weaker than the shape pass: a stream used outside a ``Session`` (a
    test host, a future embedder) still uploads nothing a credential SHAPE
    gives away. A real session replaces this with its ``VariableStore``.
    """
    from local_operator.redaction_shapes import scrub_shapes, scrub_shapes_for_arguments

    return Redaction(scrub_shapes, scrub_shapes_for_arguments)


def _decode_b64(data: str) -> bytes:
    if not data:
        return b""
    try:
        return base64.b64decode(data, validate=False)
    except (binascii.Error, ValueError):
        # Not base64 after all: its identity is its own text, which still never
        # travels (only the hash does).
        return data.encode("utf-8", "surrogatepass")


#: JPEG start-of-frame markers that carry the frame's dimensions.
_JPEG_SOF_MARKERS = frozenset(
    {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
)


def image_dimensions(raw: bytes) -> tuple[int, int] | None:
    """Width and height from a PNG, GIF or JPEG header (the reference's rule)."""
    if raw[:8] == b"\x89PNG\r\n\x1a\n" and len(raw) >= 24:
        width, height = struct.unpack(">II", raw[16:24])
        return int(width), int(height)
    if raw[:6] in (b"GIF87a", b"GIF89a") and len(raw) >= 10:
        width, height = struct.unpack("<HH", raw[6:10])
        return int(width), int(height)
    if raw[:2] == b"\xff\xd8":
        index = 2
        while index + 9 < len(raw):
            if raw[index] != 0xFF:
                return None
            marker = raw[index + 1]
            size = struct.unpack(">H", raw[index + 2 : index + 4])[0]
            if marker in _JPEG_SOF_MARKERS:
                height, width = struct.unpack(">HH", raw[index + 5 : index + 9])
                return int(width), int(height)
            index += 2 + size
    return None


def _ref_from_bytes(raw: bytes, mime: str | None) -> dict[str, Any]:
    ref: dict[str, Any] = {
        "type": "image_ref",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }
    dims = image_dimensions(raw)
    if dims:
        ref["width"], ref["height"] = dims
    if mime:
        ref["mime"] = mime
    return ref


def _ref_from_identity(identity: str, mime: str | None) -> dict[str, Any]:
    # No bytes on this client (an attachment-store reference, a document held by
    # path): its identity is hashed and ``bytes`` is 0, the contract's URL rule.
    # The identity itself never travels, and neither does a provider URL — a
    # hosted artifact's URL can carry a signed token (S-31).
    ref: dict[str, Any] = {
        "type": "image_ref",
        "sha256": hashlib.sha256(identity.encode("utf-8", "surrogatepass")).hexdigest(),
        "bytes": 0,
    }
    if mime:
        ref["mime"] = mime
    return ref


def _media_ref(block: Any) -> dict[str, Any] | None:
    if isinstance(block, (ImageContent, AudioContent)):
        return _ref_from_bytes(_decode_b64(block.data), block.mime_type)
    if isinstance(block, DocumentContent):
        if block.data:
            return _ref_from_bytes(_decode_b64(block.data), block.mime_type)
        return _ref_from_identity("document:" + (block.digest or block.filename), block.mime_type)
    if isinstance(block, AttachmentContent):
        return _ref_from_identity(
            "attachment:" + (block.attachment or block.source_url or ""), block.content_type
        )
    return None


def _blocks(message: Message, redact: Redactor) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for block in message.content:
        if isinstance(block, TextContent):
            out.append({"type": "text", "text": redact(block.text)})
            continue
        ref = _media_ref(block)
        if ref is not None:
            out.append(ref)
    return out


def _arguments(call: ToolCall) -> str:
    # The SAME replayable string the provider wire sends (salvaging a truncated
    # fragment), so the frozen record describes the call the model really sees.
    from local_operator.providers.clients import _replayable_tool_arguments_json

    return _replayable_tool_arguments_json(call)


def project_message(message: Message, redaction: Redaction) -> dict[str, Any]:
    """One message's allowlisted, redacted upload form (see the module docstring)."""
    projected: dict[str, Any] = {
        "role": message.role,
        "content": _blocks(message, redaction.redact),
    }
    if message.role == "assistant" and message.tool_calls:
        projected["tool_calls"] = [
            {
                "id": call.id,
                "function": {
                    "name": call.name,
                    "arguments": redaction.redact_arguments(_arguments(call)),
                },
            }
            for call in message.tool_calls
        ]
    if message.role == "tool":
        projected["tool_call_id"] = message.tool_call_id or ""
        if message.is_error:
            projected["is_error"] = True
    return projected


def saver_projection(
    messages: Sequence[Message], redaction: Redaction | None = None
) -> list[dict[str, Any]]:
    """THE Saver upload serialiser (§3.1). Nothing else may build an upload."""
    active = redaction or shape_only_redaction()
    return [project_message(message, active) for message in messages]


def _content_key(message: Message) -> tuple[Any, ...]:
    """What the projection of ``message`` depends on, cheaply.

    Content-addressed rather than id-keyed: pruning blanks a row IN PLACE under
    the same id, and that must re-project. The key holds the strings themselves,
    never their ``hash()``: a hash collision would upload ANOTHER row's
    projection, and a tuple of references copies nothing.
    """
    blocks: list[tuple[Any, ...]] = []
    for block in message.content:
        if isinstance(block, TextContent):
            blocks.append(("text", block.text))
        elif isinstance(block, (ImageContent, AudioContent)):
            blocks.append((block.type, block.data, block.mime_type))
        elif isinstance(block, DocumentContent):
            blocks.append(("document", block.data, block.digest, block.filename))
        elif isinstance(block, AttachmentContent):
            blocks.append(("attachment", block.attachment, block.source_url, block.content_type))
    calls = tuple(
        (
            call.id,
            call.name,
            call.raw_arguments,
            json.dumps(call.arguments, sort_keys=True, default=str),
        )
        for call in message.tool_calls
    )
    return (
        message.role,
        tuple(blocks),
        calls,
        message.tool_call_id,
        message.is_error,
    )


def _raw_strings(message: Message) -> tuple[str, ...]:
    return tuple(block.text for block in message.content if isinstance(block, TextContent))


@dataclass
class _Entry:
    projected: dict[str, Any]
    canonical: bytes
    raw: tuple[str, ...]
    #: The row carries tool calls, whose arguments are JSON: a registered value
    #: holding ``"``, ``\\`` or non-ASCII is spelled ESCAPED there, so a
    #: substring test on the raw text cannot see it (S-11).
    has_calls: bool = False


@dataclass
class ProjectionCache:
    """Per-chain memo of projected rows and their canonical bytes.

    Invalidation follows the redactor: a newly registered value evicts the text
    rows whose raw text contains it, and EVERY row that carries tool calls (the
    shape pass is a pure function of the text, so only the exact-value pass can
    change an old row's projection); a REMOVED value evicts everything, because
    the redacted copy no longer shows where it was. Tool-call rows are evicted
    wholesale because their arguments are JSON, where a value's escaped
    spelling defeats a substring test (S-11); text rows carry most of the bytes,
    so the cache stays warm where it pays.
    """

    limit: int = 20_000
    _entries: dict[tuple[Any, ...], _Entry] = field(default_factory=dict)
    _values: frozenset[str] = frozenset()

    def _sync(self, redaction: Redaction) -> None:
        try:
            current = frozenset(value for value in redaction.values() if value)
        except Exception:  # noqa: BLE001 — a failing reader must not pin stale bytes
            self._entries.clear()
            self._values = frozenset()
            return
        if current == self._values:
            return
        if self._values - current:
            self._entries.clear()
        else:
            added = current - self._values
            stale = [
                key
                for key, entry in self._entries.items()
                if entry.has_calls or any(value in text for value in added for text in entry.raw)
            ]
            for key in stale:
                del self._entries[key]
        self._values = current

    def project(
        self, messages: Sequence[Message], redaction: Redaction
    ) -> tuple[list[dict[str, Any]], list[bytes]]:
        """The projection of ``messages`` and each row's canonical bytes."""
        self._sync(redaction)
        projected: list[dict[str, Any]] = []
        encoded: list[bytes] = []
        for message in messages:
            key = _content_key(message)
            entry = self._entries.get(key)
            if entry is None:
                row = project_message(message, redaction)
                entry = _Entry(
                    row,
                    canonical_bytes(row),
                    _raw_strings(message),
                    has_calls=bool(message.tool_calls),
                )
                if len(self._entries) >= self.limit:
                    self._entries.clear()
                self._entries[key] = entry
            projected.append(entry.projected)
            encoded.append(entry.canonical)
        # The redact pass registers the shape hits it finds, so projecting new
        # rows can itself add values; resync once so the NEXT turn starts clean.
        self._sync(redaction)
        return projected, encoded

    def clear(self) -> None:
        self._entries.clear()
        self._values = frozenset()


def span_bytes(encoded: Sequence[bytes]) -> bytes:
    """Canonical bytes of a span from its rows' canonical bytes (a JSON array)."""
    return b"[" + b",".join(encoded) + b"]"


# ---------------------------------------------------------------------------
# verbatim_wire: the client's own projection back to provider-neutral rows
# ---------------------------------------------------------------------------

_MIME_RE = re.compile(r"[a-z]{1,20}/[A-Za-z0-9.+-]{1,60}")
_URL_SAFE_RE = re.compile(r"https?://[A-Za-z0-9._~:/?#@!$&'()*+,;=%-]{1,300}")
_SHA_RE = re.compile(r"[0-9a-f]{12,64}")


def _small_int(value: Any, limit: int = 100_000) -> bool:
    # bool is an int subclass here and a distinct type in Go; refuse it so both
    # ports read the field the same way (the reference's rule).
    if isinstance(value, bool):
        return False
    if isinstance(value, str) and value.isdigit():
        value = int(value)
    return isinstance(value, int) and 0 < value <= limit


def _bytes_label(size: int) -> str:
    if size < 1000:
        return f"{size} B"
    if size < 1_000_000:
        return f"{size / 1000:.1f} kB"
    return f"{size / 1_000_000:.1f} MB"


def media_receipt(ref: dict[str, Any]) -> str:
    """The reducer's first-sight media receipt (template M2), byte-equal to
    the reference's ``Packer(Context()).media_receipt`` for one block.

    Model-facing text, deliberately English and deliberately not catalogued: it
    is part of the method's wire format, which the server and the Python
    reference produce identically (§3.3 template M2).
    """
    sha = str(ref.get("sha256", ""))
    sha = sha[:12] if _SHA_RE.fullmatch(sha) else "invalid"
    mime = str(ref.get("mime") or "")
    kind = mime if _MIME_RE.fullmatch(mime) else "media"
    width, height = ref.get("width"), ref.get("height")
    dims = f" {width}x{height}" if _small_int(width) and _small_int(height) else ""
    raw_size = ref.get("bytes")
    size = raw_size if isinstance(raw_size, int) and _small_int(raw_size, 1 << 40) else 0
    what = "image" if kind.startswith("image") or kind == "media" else kind
    url = str(ref.get("url") or "")
    url_note = f" url={url}" if _URL_SAFE_RE.fullmatch(url) else ""
    sense = "audible" if kind.startswith("audio") else "visible"
    size_note = _bytes_label(size) if size else ""
    return (
        f"[radient:{what} omitted — {kind}{dims} {size_note} sha256={sha}{url_note}; "
        f"it was {sense} when first sent and is not {sense} now; "
        "describe it only from what was said about it]"
    ).replace("  ", " ")


def _text_blocks(content: Any) -> list[TextContent]:
    if isinstance(content, str):
        return [TextContent(text=content)]
    out: list[TextContent] = []
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "image_ref":
                out.append(TextContent(text=media_receipt(block)))
            elif block.get("type") == "text":
                out.append(TextContent(text=str(block.get("text", ""))))
    return out


def verbatim_wire(span: Sequence[dict[str, Any]], *, id_prefix: str) -> list[Message]:
    """A ``verbatim`` epoch's wire rows: the client's OWN projection, mapped back.

    Never the raw ``Message``s and never server bytes (S2-5): the span is sent
    redacted, without ``provider_payload`` and without media bytes, exactly as a
    packed span would be. A pure function of the digested bytes, so it is the
    same on every turn. ``id_prefix`` gives each row a stable id, which the
    context tracker keys on.

    A tool row's ``tool_name`` is filled from its call in the same span: the
    span is pairing-safe (§6.1 check 9), so the call is always there, and the
    Google wire needs the name to render a function response.
    """
    names: dict[str, str] = {}
    out: list[Message] = []
    for index, row in enumerate(span):
        role = row.get("role")
        message_id = f"{id_prefix}:{index}"
        if role == "assistant":
            calls: list[ToolCall] = []
            for call in row.get("tool_calls") or []:
                function = call.get("function") or {}
                raw = str(function.get("arguments", ""))
                try:
                    parsed = json.loads(raw) if raw else {}
                except ValueError:
                    parsed = {}
                name = str(function.get("name", ""))
                call_id = str(call.get("id", ""))
                names[call_id] = name
                calls.append(
                    ToolCall(
                        id=call_id,
                        name=name,
                        arguments=parsed if isinstance(parsed, dict) else {},
                        raw_arguments=raw,
                    )
                )
            out.append(
                Message(
                    id=message_id,
                    role="assistant",
                    content=list(_text_blocks(row.get("content"))),
                    tool_calls=calls,
                )
            )
        elif role == "tool":
            call_id = str(row.get("tool_call_id", ""))
            out.append(
                Message(
                    id=message_id,
                    role="tool",
                    tool_call_id=call_id,
                    tool_name=names.get(call_id),
                    content=list(_text_blocks(row.get("content"))),
                    is_error=bool(row.get("is_error")),
                )
            )
        elif role == "user":
            out.append(
                Message(id=message_id, role="user", content=list(_text_blocks(row.get("content"))))
            )
        # system/developer cannot come from lop's own projection (MessageRole has
        # neither), so a verbatim span here never carries one.
    return out
