"""Per-provider document delivery: the capability table and the fallback.

A user-attached document is not like an image: every provider that takes
images accepts a base64 image part, but only SOME accept a PDF, and each one
that does spells the part differently. This module owns the one table that
decides which wire form a document may take, and the extraction that carries
it when the answer is "none of them".

WHY A TABLE RATHER THAN A FLAG ON ``ModelSpec``. The decision is not a
property of the model alone. The same OpenAI model can be reached through
Chat Completions or the Responses API depending on credentials and config,
and OpenRouter/Mistral ride the OpenAI-compatible client with their own
part shape. Amendment 6 of the attachment contract is explicit: the policy is
computed at REQUEST time from the actual endpoint/wire/model, including a
failover attempt, because a ``/model`` switch must never send a PDF natively
to a provider that cannot read it. The serializers therefore call
:func:`document_form` with the wire they are about to speak.

SOURCES (each URL was fetched and read on 2026-10-23; the row it supports is
named beside it):

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
  https://docs.mistral.ai/studio/document-processing/document_qna

ASSUMED rather than read verbatim from those pages, stated so a later reader
knows what to re-check before changing a row:

- OpenAI Chat Completions' ``file`` part is documented on the same file-inputs
  guide but its exact acceptance varies by model; we gate it on
  ``ModelSpec.supports_images``, the closest capability signal the spec
  carries.
- Mistral's base64 ``document_url`` spelling (a ``data:`` URL rather than the
  public/signed https URL the document-QnA page shows) is documented on the
  basic-OCR page, which states "encoding them in base64 and passing them as a
  data url": https://docs.mistral.ai/studio/document-processing/basic_ocr
- WHERE EACH CEILING APPLIES, because getting this backwards is permissive
  rather than conservative: Anthropic's 32 MB is the WHOLE REQUEST (its page
  says "both limits are on the entire request payload, including any other
  content sent alongside PDFs"), and OpenAI's 50 MB is the COMBINED total
  across files as well as the per-file bound. `document_native_ok` therefore
  compares the ENCODED length (base64 inflates 4/3) and a per-request
  :class:`DocumentBudget` sums every document a request will send; a block
  that no longer fits falls back to extraction.

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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:  # pragma: no cover - typing only
    from local_operator.harness.types import DocumentContent

logger = logging.getLogger(__name__)

__all__ = [
    "DOCUMENT_FORMS",
    "DOCUMENT_REQUEST_HEADROOM_BYTES",
    "DocumentBudget",
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

#: Anthropic documents are capped at 600 pages, but only 100 when the request's
#: context window is under 1M tokens (its PDF page states both). A model whose
#: window is unknown is treated as SMALL: the conservative direction is the
#: one that falls back to extraction rather than a paid 400.
ANTHROPIC_LARGE_CONTEXT_WINDOW = 1_000_000
ANTHROPIC_SMALL_CONTEXT_MAX_PAGES = 100

#: Reserved out of a request-scoped limit for everything that is not a
#: document: text, images, tool schemas and the JSON envelope. 1 MiB is a
#: stated allowance, not a measurement — the session's own wire-byte budget
#: keeps the whole request smaller than this in practice, and the point here
#: is that documents never claim the provider's entire ceiling.
DOCUMENT_REQUEST_HEADROOM_BYTES = 1_048_576


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
    # 32 MB here is the STORE'S UPLOAD CEILING, not a documented OpenAI limit:
    # ``MAX_ATTACHMENT_UPLOAD_BYTES`` refuses a larger file before it can ever
    # reach this row, so a bigger number would be unreachable. The file-inputs
    # guide's <50 MB/file, 50 MB combined (the responses row below) is the
    # provider side of the same gate; this row states the conservative
    # per-request reading the contract's table carries.
    "openai": DocumentForm("openai_chat", 32_000_000, None, "openai"),
    "google": DocumentForm("google", 50_000_000, 1000, "google"),
    "openrouter": DocumentForm("openrouter", 32_000_000, None, "openrouter"),
    "mistral": DocumentForm("mistral", 50_000_000, 1000, "mistral"),
}

#: OpenAI's Responses API is a different wire for the SAME provider id, so it
#: cannot live in the id-keyed map above.
_OPENAI_RESPONSES = DocumentForm("openai_responses", 50_000_000, None, "openai")


@dataclass
class DocumentBudget:
    """The request-scoped byte room documents may still claim.

    Lives for ONE render pass (one provider body build), because the limits it
    models are per request. ``take`` both checks and consumes, so two documents
    that each fit the per-file bound cannot together exceed the combined one:
    the second falls back to extraction instead of putting both on one
    oversized wire.
    """

    remaining: int

    def take(self, encoded: int) -> bool:
        if encoded <= 0 or encoded > self.remaining:
            return False
        self.remaining -= encoded
        return True


def document_budget(form: DocumentForm | None) -> DocumentBudget | None:
    """A fresh per-request budget for ``form``, or ``None`` without a row."""
    if form is None:
        return None
    return DocumentBudget(max(0, form.max_bytes - DOCUMENT_REQUEST_HEADROOM_BYTES))


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


def document_form(
    provider: str,
    wire: str,
    *,
    supports_images: bool = True,
    context_window: int | None = None,
) -> DocumentForm | None:
    """The native form for this provider ON THIS WIRE, or ``None`` for fallback.

    ``wire`` names the API the caller is building (``openai-chat``,
    ``openai-responses``, ``anthropic-messages``, ``google-content``), not a
    provider. ``supports_images`` is the capability gate the table names for
    the OpenAI rows: a text-only model under the same provider id must not be
    handed a file part.

    ``context_window`` narrows Anthropic's 600-page ceiling to the documented
    100-page sub-limit when the request's model is under 1M tokens. ``None``
    (a caller that cannot name the model, e.g. the legacy ``/v1/chat`` door)
    takes the SMALL limit: unknown is not evidence of a large window, and the
    cost of guessing wrong in the other direction is a paid provider 400.
    """
    form = DOCUMENT_FORMS.get(provider)
    if provider == "openai" and wire == "openai-responses":
        form = _OPENAI_RESPONSES if supports_images else None
        return form
    if form is None or form.wire not in _WIRE_FOR_REQUEST:
        return None
    if _WIRE_FOR_REQUEST[form.wire] != wire:
        return None
    if form.wire in {"openai_chat", "openai_responses"} and not supports_images:
        return None
    if form.wire == "anthropic" and (context_window or 0) < ANTHROPIC_LARGE_CONTEXT_WINDOW:
        form = replace(form, max_pages=ANTHROPIC_SMALL_CONTEXT_MAX_PAGES)
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


def encoded_document_bytes(block: "DocumentContent") -> int:
    """The bytes this block occupies ON THE WIRE (base64), not decoded size.

    The provider ceilings are about the request payload, and base64 inflates a
    PDF by 4/3: a 25 MB file is 33.3 MB encoded and no longer fits Anthropic's
    32 MB request cap. Comparing the decoded size — the first version of this
    check did — accepts exactly the payload the provider refuses.
    """
    raw = _decoded(block)
    if raw is None:
        return 0
    return 4 * ((len(raw) + 2) // 3)


def document_native_ok(
    block: "DocumentContent",
    form: DocumentForm | None,
    *,
    budget: DocumentBudget | None = None,
) -> bool:
    """Whether ``block`` may take ``form``'s NATIVE shape, limits included.

    The single eligibility test every serializer consults, so the admission
    label (``document_injection``) and the request-time render cannot drift
    about what "native" means. ``False`` for a non-PDF: the table's native rows
    are document-input rows for PDFs, and a text file sent as a "document" part
    is the send-and-hope this contract forbids.

    ``budget`` is the REQUEST's remaining document room. A successful check
    CONSUMES the encoded size, so every caller must pass the same budget for
    every document of one body build and call this exactly once per block.
    Without one, only the per-block ceiling applies (a single-document
    question, which is what the admission label asks).
    """
    if form is None or block.mime_type != "application/pdf":
        return False
    raw = _decoded(block)
    if raw is None:
        return False
    encoded = encoded_document_bytes(block)
    if encoded > form.max_bytes:
        return False
    if form.max_pages is not None:
        pages = pdf_page_count(raw, block.digest)
        # An UNCOUNTABLE PDF is not native: if the page structure cannot be
        # read here, a provider's own parser is at least as likely to refuse
        # it, and "send and hope" is exactly what the contract forbids. The
        # fallback then extracts (or degrades to a reference), which never
        # spends a paid request on a shape the provider rejects.
        if pages is None or pages > form.max_pages:
            return False
    if budget is not None and not budget.take(encoded):
        return False
    return True


def document_injection(
    block: "DocumentContent", form: DocumentForm | None
) -> Literal["native", "extracted", "reference"]:
    """How a document WILL be delivered: ``native``/``extracted``/``reference``.

    Stamped on the block at admission as historical metadata (contract §3):
    the transcript can then state how the turn actually reached the model,
    without claiming a later request used the same path. The label is asked
    as a ONE-DOCUMENT request (a fresh budget), which is the closest answer
    available before the real body exists.
    """
    if document_native_ok(block, form, budget=document_budget(form)):
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
    # The cache holds the EXTRACTION only, never the display header: the same
    # bytes can be re-attached under a new filename, and a cached header would
    # then name a file the user is no longer sending.
    digest = block.digest or hashlib.sha256(raw).hexdigest()[:32]
    header = f"[Document: {block.filename} ({block.mime_type}, {block.bytes} bytes)]"
    cached = _read_cached_text(digest)
    if cached is not None:
        return f"{header}\n{cached}"
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
    if total > pages_read:
        body += f"\n[... only the first {pages_read} of {total} pages were read]"
    _write_cached_text(digest, body)
    return f"{header}\n{body}"


def _read_cached_text(digest: str) -> str | None:
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

    NO PATH HERE, deliberately (agent review round 1, M3). This sentence is
    rendered by a serializer, which has no session and therefore no
    confinement boundary; the contract's reference arm used to name the origin
    path here, and a confined session then told the model the controller's
    absolute path and invited it to open a file its own tools must refuse.
    Paths are the hint's job: ``session/attachment_hints.py`` renders the
    retained copy only when it is inside the session's world and the origin
    path under the same test. The reference block states the facts that are
    true everywhere; the trailing note says where the file actually is.
    """
    why = f" ({reason})" if reason else ""
    return (
        f"[Document reference: {block.filename} ({block.mime_type}, {block.bytes} bytes). "
        f"The full contents were not inlined{why}. The attachment note after this message "
        "names the retained copy when it is reachable from this session.]"
    )


def fallback_document_text(block: "DocumentContent") -> str:
    """One entry point for the serializers' fallback arm."""
    return extract_document_text(block)
