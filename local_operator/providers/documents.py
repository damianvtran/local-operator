"""Per-provider document delivery: the capability table and the fallback.

A user-attached document is not like an image: every provider that takes
images accepts a base64 image part, but only SOME accept a PDF, and each one
that does spells the part differently. This module owns the one table that
decides which wire form a document may take, and the extraction that carries
it when the answer is "none of them".

WHY A TABLE RATHER THAN A FLAG ON ``ModelSpec``. The decision is not a
property of the model alone. The same OpenAI model can be reached through
Chat Completions or the Responses API depending on credentials and config,
and OracleRouter/Mistral ride the OpenAI-compatible client with their own
part shape. Amendment 6 of the attachment contract is explicit: the policy is
computed at REQUEST time from the actual endpoint/wire/model, including a
failover attempt, because a ``/model`` switch must never send a PDF natively
to a provider that cannot read it. The serializers therefore call
:func:`document_form` with the wire they are about to speak.

SOURCES (capability table verified 2026-10-23 from official docs):

- Anthropic PDF support (``document`` block, base64 source, 32 MB request,
  600 pages; 100 pages under a 1M-token context):
  https://docs.anthropic.com/en/docs/build-with-claude/pdf-support
- OpenAI file inputs (``input_file`` with a data URL; <50 MB/file, 50 MB
  combined on the Responses API):
  https://platform.openai.com/docs/guides/pdf-files
- Gemini document processing (``inline_data``; 50 MB or 1000 pages):
  https://ai.google.dev/gemini-api/docs/document-processing
- OpenRouter PDFs (``file`` part with ``file_data``/``filename``; parsing is
  server-side and available to any model):
  https://openrouter.ai/docs/features/multimodal/pdfs
- Mistral document QnA (``document_url`` part; 50 MB / 1000 pages):
  https://docs.mistral.ai/capabilities/document/

ASSUMED rather than read verbatim from those pages, stated so a later reader
knows what to re-check before changing a row:

- OpenAI Chat Completions' ``file`` part is documented on the same file-inputs
  guide but its exact acceptance varies by model; we gate it on
  ``ModelSpec.supports_images``, the closest capability signal the spec
  carries.
- The byte/page ceilings copied from each page are per REQUEST for Anthropic
  and per FILE for the others; this module treats them all as per-file,
  which is the conservative reading.

Every row defaults to the FALLBACK: an unknown provider, an unknown wire, or
a payload outside the row's limits gets extracted text or a reference block,
never a part the provider would reject inside a paid turn.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:  # pragma: no cover - typing only
    from local_operator.harness.types import DocumentContent

logger = logging.getLogger(__name__)

__all__ = [
    "DOCUMENT_FORMS",
    "DocumentForm",
    "admission_wire",
    "document_form",
    "document_injection",
    "document_native_ok",
    "extract_document_text",
    "fallback_document_text",
    "pdf_page_count",
]

#: Bounds for the fallback extraction. 200 pages is the first GPT-4.1-era
#: ceiling (the tightest document page limit among the providers we fall back
#: FROM) and 400k characters keeps one document under the context of every
#: model this harness routes to. The truncation is announced, never silent:
#: a model told "this is all there was" would answer confidently about a
#: document it only half read.
MAX_FALLBACK_PAGES = 200
MAX_FALLBACK_CHARS = 400_000

#: How much of a text-ish file the fallback keeps. The same 400k character
#: bound: text has no page structure to count, and the notice names the cut.
MAX_FALLBACK_TEXT_CHARS = 400_000

#: A PDF's page count costs a parse of the xref table. Cached per digest for
#: the life of the process: the bytes are content-addressed, so the answer
#: cannot change for a digest, and a session resends its history on every
#: turn.
_PAGE_COUNT_CACHE: dict[str, int | None] = {}


@dataclass(frozen=True)
class DocumentForm:
    """One provider row: the wire spelling and the limits it must fit.

    ``wire`` is the spelling a serializer switches on, not the provider id:
    the serializer already knows which API it is building, and letting the
    table name it keeps provider renames out of the serializers.
    """

    wire: Literal["anthropic", "openai_responses", "openai_chat", "google", "openrouter", "mistral"]
    max_bytes: int
    max_pages: int | None
    provider_id: str


#: The table. Keys are the provider ids the model registry actually uses;
#: aliases (``anthropic-key``) resolve in :func:`document_form` so a second
#: spelling cannot silently miss a row.
DOCUMENT_FORMS: dict[str, DocumentForm] = {
    "anthropic": DocumentForm("anthropic", 32_000_000, 600, "anthropic"),
    "anthropic-key": DocumentForm("anthropic", 32_000_000, 600, "anthropic"),
    "openai": DocumentForm("openai_chat", 32_000_000, None, "openai"),
    "google": DocumentForm("google", 50_000_000, 1000, "google"),
    "openrouter": DocumentForm("openrouter", 32_000_000, None, "openrouter"),
    "mistral": DocumentForm("mistral", 50_000_000, 1000, "mistral"),
}

#: OpenAI's Responses API is a different wire for the SAME provider id, so it
#: cannot live in the id-keyed map above.
_OPENAI_RESPONSES = DocumentForm("openai_responses", 50_000_000, None, "openai")

#: MIME prefixes/suffixes the fallback can read as text. PDF is handled by
#: extraction first; these are the "text-ish" second arm of contract §3.
_TEXT_MIMES = frozenset(
    {
        "application/json",
        "application/xml",
        "application/yaml",
    }
)


def _is_text_mime(mime_type: str) -> bool:
    return mime_type.startswith("text/") or mime_type in _TEXT_MIMES


def document_form(provider: str, wire: str, *, supports_images: bool = True) -> DocumentForm | None:
    """The native form for this provider ON THIS WIRE, or ``None`` for fallback.

    ``wire`` names the API the caller is building (``openai-chat``,
    ``openai-responses``, ``anthropic-messages``, ``google-content``), not a
    provider. ``supports_images`` is the capability gate the table names for
    the OpenAI rows: a text-only model under the same provider id must not be
    handed a file part.
    """
    form = DOCUMENT_FORMS.get(provider)
    if provider == "openai" and wire == "openai-responses":
        return _OPENAI_RESPONSES if supports_images else None
    if form is None or form.wire not in _WIRE_FOR_REQUEST:
        return None
    if _WIRE_FOR_REQUEST[form.wire] != wire:
        return None
    if form.wire in {"openai_chat", "openai_responses"} and not supports_images:
        return None
    return form


#: Which request wire each table form belongs to. ``openrouter`` and
#: ``mistral`` ride the OpenAI Chat Completions client but are NOT the OpenAI
#: row: their part shapes differ, which is why the mapping is explicit.
_WIRE_FOR_REQUEST = {
    "anthropic": "anthropic-messages",
    "openai_responses": "openai-responses",
    "openai_chat": "openai-chat",
    "google": "google-content",
    "openrouter": "openai-chat",
    "mistral": "openai-chat",
}


def admission_wire(provider: str, *, supports_responses_api: bool = False) -> str:
    """The request wire a provider is MOST LIKELY to be called on.

    Used ONLY to stamp ``DocumentContent.injection`` at admission, which is
    historical metadata (contract §3 / amendment 6). The actual serialization
    re-decides from the wire it is about to speak; a label that guessed wrong
    tells the transcript something imprecise, never changes what is sent.

    OpenAI is the only provider with two wires this harness ships, so it is
    the only one that needs the spec's Responses capability to choose.
    """
    if provider in {"anthropic", "anthropic-key"}:
        return "anthropic-messages"
    if provider == "google":
        return "google-content"
    if provider == "openai" and supports_responses_api:
        return "openai-responses"
    return "openai-chat"


def pdf_page_count(raw: bytes, digest: str | None = None) -> int | None:
    """Pages in a PDF, or ``None`` when the document cannot be parsed.

    ``None`` is not an error: a scanned/encrypted/corrupt PDF still has bytes
    and can still be sent to a provider that accepts PDFs (Anthropic,
    Google). It only means the page LIMIT cannot be checked, which is why the
    delivery decision treats an unknown count as "inside" for providers whose
    only meaningful limit is bytes, and extraction announces its own bound.
    """
    key = digest or hashlib.sha256(raw).hexdigest()[:32]
    if key in _PAGE_COUNT_CACHE:
        return _PAGE_COUNT_CACHE[key]
    try:
        from pypdf import PdfReader  # lazy: only a document turn pays for it

        count: int | None = len(PdfReader(io.BytesIO(raw), strict=False).pages)
    except Exception:  # noqa: BLE001 — an unparseable PDF is a normal document
        logger.debug("document: page count unavailable", exc_info=True)
        count = None
    _PAGE_COUNT_CACHE[key] = count
    return count


def _decoded(block: "DocumentContent") -> bytes | None:
    try:
        return base64.b64decode(block.data, validate=False)
    except (ValueError, binascii.Error):
        return None


def document_native_ok(block: "DocumentContent", form: DocumentForm | None) -> bool:
    """Whether ``block`` may take ``form``'s NATIVE shape, limits included.

    The single eligibility test every serializer consults, so the admission
    label (``document_injection``) and the request-time render cannot drift
    about what "native" means. ``False`` for a non-PDF: the table's native rows
    are document-input rows for PDFs, and a text file sent as a "document" part
    is the send-and-hope this contract forbids.
    """
    if form is None or block.mime_type != "application/pdf":
        return False
    raw = _decoded(block)
    if raw is None or len(raw) > form.max_bytes:
        return False
    pages = pdf_page_count(raw, block.digest)
    return pages is None or form.max_pages is None or pages <= form.max_pages


def document_injection(
    block: "DocumentContent", form: DocumentForm | None
) -> Literal["native", "extracted", "reference"]:
    """How a document WILL be delivered: ``native``/``extracted``/``reference``.

    Stamped on the block at admission as historical metadata (contract §3):
    the transcript can then state how the turn actually reached the model,
    without claiming a later request used the same path.
    """
    if document_native_ok(block, form):
        return "native"
    if block.mime_type == "application/pdf" or _is_text_mime(block.mime_type):
        return "extracted"
    return "reference"


def _cached_text_path(digest: str) -> Path:
    from local_operator.session.attachments import AttachmentStore

    return AttachmentStore().root / f"{digest}.txt"


def extract_document_text(block: "DocumentContent") -> str:
    """The fallback text for ``block``, with page markers and a truncation note.

    PDFs are extracted with ``pypdf`` (lazily imported: it is a base
    dependency, but a conversation with no documents must not pay its import).
    The output is cached beside the blob as ``<digest>.txt``, so a session
    that resends its history re-reads a small text file instead of reparsing
    the PDF on every turn — and the cache is content-addressed, so it cannot
    go stale against the bytes. A cache write failure is ordinary (read-only
    home): the extraction is recomputed, nothing is lost.

    Text-ish MIME is decoded directly, bounded. Anything else yields the
    REFERENCE sentence — a model must never receive nothing for a file the
    user believes it can see.
    """
    raw = _decoded(block)
    if raw is None:
        return _reference_text(block, reason="its stored bytes could not be decoded")
    if block.mime_type == "application/pdf":
        return _extract_pdf(block, raw)
    if _is_text_mime(block.mime_type):
        text = raw.decode("utf-8", errors="replace")
        if len(text) > MAX_FALLBACK_TEXT_CHARS:
            dropped = len(text) - MAX_FALLBACK_TEXT_CHARS
            text = (
                text[:MAX_FALLBACK_TEXT_CHARS]
                + f"\n[... {dropped} characters truncated from {block.filename}]"
            )
        return f"[Document: {block.filename} ({block.mime_type})]\n```\n{text}\n```"
    return _reference_text(block)


def _extract_pdf(block: "DocumentContent", raw: bytes) -> str:
    digest = block.digest or hashlib.sha256(raw).hexdigest()[:32]
    cached = _read_cached_text(digest, block)
    if cached is not None:
        return cached
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(raw), strict=False)
        total = len(reader.pages)
        chunks: list[str] = []
        used = 0
        pages_read = 0
        for index, page in enumerate(reader.pages):
            if pages_read >= MAX_FALLBACK_PAGES:
                break
            try:
                page_text = page.extract_text() or ""
            except Exception:  # noqa: BLE001 — one bad page must not lose the rest
                logger.debug("document: page %d extraction failed", index + 1, exc_info=True)
                page_text = ""
            marker = f"--- page {index + 1} ---"
            chunk = f"{marker}\n{page_text.strip()}"
            if used + len(chunk) > MAX_FALLBACK_CHARS:
                remaining = MAX_FALLBACK_CHARS - used
                if remaining > len(marker) + 1:
                    chunks.append(chunk[:remaining])
                used = MAX_FALLBACK_CHARS
                break
            chunks.append(chunk)
            used += len(chunk) + 1
            pages_read = index + 1
    except Exception:  # noqa: BLE001 — a corrupt PDF degrades to its metadata
        logger.warning("document: PDF extraction failed for %s", block.filename, exc_info=True)
        return _reference_text(block, reason="it is a PDF the text extractor could not read")
    body = "\n".join(chunks)
    notice = ""
    if total > pages_read:
        notice = f"\n[... only the first {pages_read} of {total} pages were read]"
    text = f"[Document: {block.filename} ({block.mime_type}, {block.bytes} bytes)]\n{body}{notice}"
    _write_cached_text(digest, text)
    return text


def _read_cached_text(digest: str, block: "DocumentContent") -> str | None:
    try:
        return _cached_text_path(digest).read_text(encoding="utf-8")
    except OSError:
        return None


def _write_cached_text(digest: str, text: str) -> None:
    path = _cached_text_path(digest)
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        logger.debug("document: cannot cache extraction for %s: %s", digest, exc)


def _reference_text(block: "DocumentContent", *, reason: str | None = None) -> str:
    """The last-resort arm: metadata the model can act on, never an empty part.

    The path is included because the contract's reference arm exists to let a
    model with file tools open the file itself. Confinement is the SESSION's
    policy and is applied by the path hint that follows every user turn
    (``session/attachment_hints.py``); this sentence is the same facts a hint
    would carry, and a confined session's notice tells the model not to
    search for the file.
    """
    where = f"; original path: {block.path}" if block.path else ""
    why = f" ({reason})" if reason else ""
    return (
        f"[Document reference: {block.filename} ({block.mime_type}, {block.bytes} bytes)"
        f"{where}. The full contents were not inlined{why}; open it with a tool if the "
        "session has one.]"
    )


def fallback_document_text(block: "DocumentContent") -> str:
    """One entry point for the serializers' fallback arm."""
    return extract_document_text(block)


@lru_cache(maxsize=64)
def _has_pypdf() -> bool:
    try:
        import pypdf  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False
