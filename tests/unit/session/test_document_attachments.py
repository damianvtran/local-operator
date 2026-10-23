"""Document attachments: the core half of the composer attachment contract.

The defect these lock down (reproduced 2026-10-23, session ``439818272d84``):
a PDF staged on the composer reached the turn as ``[text, image/png]`` — or
not at all — because the content union had no document member, ``Prompt`` had
no file field, and no serializer had a document arm. Every test below names
the hop it covers so a regression points at the layer that broke.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from local_operator.compaction.marker import (
    ATTACHMENTS_MANIFEST_KEY,
    build_compaction_marker,
    collect_attachment_manifest,
    render_attachment_manifest,
    render_compaction_marker,
    with_attachment_manifest,
)
from local_operator.harness.types import (
    ChatRequest,
    DocumentContent,
    ImageContent,
    Message,
    ModelSpec,
    TextContent,
    coerce_content_blocks,
)
from local_operator.providers.clients import (
    AnthropicClient,
    GoogleClient,
    _message_to_openai,
    _messages_to_openai_responses,
)
from local_operator.providers.documents import (
    document_form,
    document_injection,
    extract_document_text,
)
from local_operator.session.attachment_hints import with_attachment_hints
from local_operator.session.attachments import AttachmentStore, sniff_document_mime
from local_operator.session.errors import (
    AttachmentMissing,
    DocumentCarriageUnsupported,
    RemoteDocumentUnsupported,
    admission_error,
)
from local_operator.session.runtime.server import document_blocks
from local_operator.session.transcript import (
    _resolve_attachments,
    encode_message_payload,
)


def _pdf_raw(pages: int = 2) -> bytes:
    from pypdf import PdfWriter

    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)
    # A trailing comment keeps the archive valid and pushes the base64 past
    # the 1 KiB externalization floor, so the store tests exercise the real
    # reference row rather than the inline arm.
    return buffer.getvalue() + b"\n%" + b"p" * 3000


def _pdf_block(raw: bytes, *, filename: str = "report.pdf", path: str = "/tmp/report.pdf"):
    return DocumentContent(
        data=base64.b64encode(raw).decode("ascii"),
        mime_type="application/pdf",
        filename=filename,
        path=path,
        bytes=len(raw),
    )


@pytest.fixture()
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AttachmentStore:
    """An isolated store, and the process default pointed at it.

    ``AttachmentStore()`` with no root resolves ``config_dir()``; several
    production seams (the hint renderer, the document cache) build one that
    way, so the default has to be relocated rather than merely avoided.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "cfg"))
    return AttachmentStore(tmp_path / "cfg" / "attachments")


# ---------------------------------------------------------------------------
# A. the model
# ---------------------------------------------------------------------------


def test_document_blocks_route_by_type_and_by_legacy_shape() -> None:
    explicit = coerce_content_blocks(
        [{"type": "document", "filename": "a.pdf", "mime_type": "application/pdf", "data": "eA=="}]
    )
    assert isinstance(explicit[0], DocumentContent)
    # The durable encoder drops ``type`` (it is the model default), so a reader
    # must route the reference shape by the facts a document row carries.
    legacy = coerce_content_blocks(
        [{"filename": "a.pdf", "mime_type": "application/pdf", "attachment": "a" * 32}]
    )
    assert isinstance(legacy[0], DocumentContent)
    # An image row never carries filename+mime_type together, so it must not
    # be stolen by the document arm.
    image = coerce_content_blocks([{"attachment": "b" * 32, "mime_type": "image/png"}])
    assert not isinstance(image[0], DocumentContent)


def test_image_origin_path_is_persisted_and_absent_when_unset() -> None:
    assert ImageContent(data="eA==").model_dump() == {
        "type": "image",
        "data": "eA==",
        "mime_type": "image/png",
    }
    dumped = ImageContent(data="eA==", path="/tmp/a.png").model_dump()
    assert dumped["path"] == "/tmp/a.png"


# ---------------------------------------------------------------------------
# B. upload classification (the store-side rules; the route test lives in
#    tests/unit/server/test_desktop_attachment_upload.py)
# ---------------------------------------------------------------------------


def test_mime_comes_from_content_before_suffix() -> None:
    assert sniff_document_mime(_pdf_raw(), "actually.txt") == "application/pdf"
    assert sniff_document_mime(b"# notes\n", "notes.md") == "text/markdown"
    assert sniff_document_mime(b'{"a": 1}', "data.json") == "application/json"
    assert sniff_document_mime(b"\x00\x01\x02", "blob.bin") == "application/octet-stream"


# ---------------------------------------------------------------------------
# C. Prompt.files -> stored bytes (the exact reproduction)
# ---------------------------------------------------------------------------


def test_a_pdf_sent_as_files_reaches_message_content_as_a_document(store) -> None:
    raw = _pdf_raw()
    ref = store.put_bytes(raw, "application/pdf")
    assert ref is not None
    # What the wire sends: a reference, not the bytes.
    from local_operator.server.routes.desktop_sessions import FileRef, Prompt

    prompt = Prompt(
        request_id="11111111-2222-3333-4444-555555555555",
        text="Summarise the agreement",
        files=[
            FileRef(
                digest=ref.digest,
                name="Complete_With_Docusign.pdf",
                mime_type="application/pdf",
                bytes=len(raw),
                path="/Users/victor/Documents/Complete_With_Docusign.pdf",
            )
        ],
    )
    blocks = document_blocks([f.model_dump() for f in prompt.files])
    message = Message.user(prompt.text, documents=blocks)
    documents = [b for b in message.content if isinstance(b, DocumentContent)]
    assert len(documents) == 1
    block = documents[0]
    assert block.filename == "Complete_With_Docusign.pdf"
    assert block.mime_type == "application/pdf"
    assert block.bytes == len(raw)
    assert block.path == "/Users/victor/Documents/Complete_With_Docusign.pdf"
    assert block.digest == ref.digest
    assert base64.b64decode(block.data) == raw


def test_a_missing_digest_refuses_the_send_and_names_the_file(store) -> None:
    with pytest.raises(AttachmentMissing) as raised:
        document_blocks([{"digest": "f" * 32, "name": "gone.pdf"}])
    assert "gone.pdf" in str(raised.value)
    rebuilt = admission_error("attachment_missing", filename="gone.pdf")
    assert isinstance(rebuilt, AttachmentMissing)
    assert "gone.pdf" in str(rebuilt)


def test_file_ref_digest_shape_is_validated_at_the_wire() -> None:
    from local_operator.server.routes.desktop_sessions import FileRef, Prompt

    with pytest.raises(ValidationError):
        FileRef(digest="../../etc/passwd")
    with pytest.raises(ValidationError):
        Prompt(request_id="r", text="x", files=[FileRef(digest="a" * 32)] * 9)


# ---------------------------------------------------------------------------
# D. carriage refusals
# ---------------------------------------------------------------------------


def test_a_peer_owned_session_refuses_documents_before_the_frame() -> None:
    from local_operator.session.attached import AttachedSession

    session = object.__new__(AttachedSession)
    session._owner = SimpleNamespace(placement=SimpleNamespace(is_local=False))
    with pytest.raises(RemoteDocumentUnsupported):
        import asyncio

        asyncio.run(
            session.admit_prompt(
                "read this", command_id="c-1", images=[], files=[{"digest": "a" * 32}]
            )
        )


def test_a_handle_without_the_files_keyword_refuses_an_attachment_bearing_frame() -> None:
    from local_operator.session.runtime.server import RuntimeServer

    class OldHandle:
        async def prompt(self, text: str, images=None, command_id=None):
            return "admitted"

        async def steer(self, text: str, images=None):
            return "queued"

    server = object.__new__(RuntimeServer)
    server._handle = OldHandle()
    import asyncio

    with pytest.raises(DocumentCarriageUnsupported):
        asyncio.run(
            server._dispatch(
                "prompt",
                {"op": "prompt", "text": "hi", "files": [{"digest": "a" * 32}]},
            )
        )
    with pytest.raises(DocumentCarriageUnsupported):
        asyncio.run(
            server._dispatch(
                "steer",
                {"op": "steer", "text": "hi", "files": [{"digest": "a" * 32}]},
            )
        )


# ---------------------------------------------------------------------------
# E. transcript: externalize, resolve, degrade
# ---------------------------------------------------------------------------


def test_document_rows_externalize_and_resolve_with_their_digest(store) -> None:
    raw = _pdf_raw()
    block = _pdf_block(raw)
    payload = encode_message_payload(Message.user("read", documents=[block]), store)
    row = payload["content"][1]
    assert row["type"] == "document"
    assert "data" not in row
    assert row["attachment"]
    assert row["filename"] == "report.pdf"

    resolved = {"role": "user", "content": [dict(row)]}
    _resolve_attachments(resolved, store)
    assert base64.b64decode(resolved["content"][0]["data"]) == raw
    assert resolved["content"][0]["digest"] == row["attachment"]

    # The reloaded row parses as a document, not an image.
    parsed = coerce_content_blocks(resolved["content"])
    assert isinstance(parsed[0], DocumentContent)
    assert parsed[0].digest == row["attachment"]


def test_a_small_document_stays_inline(store) -> None:
    raw = b"%PDF-" + b"tiny"
    payload = encode_message_payload(Message.user("read", documents=[_pdf_block(raw)]), store)
    assert payload["content"][1]["data"]
    assert "attachment" not in payload["content"][1]


def test_a_missing_document_blob_becomes_a_named_notice(store) -> None:
    row = {
        "type": "document",
        "attachment": "c" * 32,
        "mime_type": "application/pdf",
        "filename": "lost.pdf",
        "bytes": 10,
    }
    resolved = {"role": "user", "content": [row]}
    _resolve_attachments(resolved, store)
    assert resolved["content"][0]["type"] == "text"
    assert "lost.pdf" in resolved["content"][0]["text"]
    assert resolved["content"][0]["text"].strip()


def test_an_image_row_stays_byte_identical(store) -> None:
    image = ImageContent(data=base64.b64encode(b"x" * 4096).decode(), mime_type="image/png")
    payload = encode_message_payload(Message.user("look", images=[image]), store)
    row = payload["content"][1]
    assert "type" not in row
    assert row["attachment"]


# ---------------------------------------------------------------------------
# F. the path hint
# ---------------------------------------------------------------------------


def test_the_hint_names_documents_origin_and_delivery(store) -> None:
    raw = _pdf_raw()
    ref = store.put_bytes(raw, "application/pdf")
    block = _pdf_block(raw, path="/home/u/report.pdf")
    block.digest = ref.digest
    block.injection = "extracted"
    noted = with_attachment_hints([Message.user("read", documents=[block])])
    note = noted[0].content[-1]
    assert isinstance(note, TextContent)
    assert "report.pdf" in note.text
    assert "retained at" in note.text
    assert "/home/u/report.pdf" in note.text
    assert "extracted text" in note.text


def test_a_confined_hint_withholds_an_outside_origin_path(store, tmp_path: Path) -> None:
    raw = _pdf_raw()
    ref = store.put_bytes(raw, "application/pdf")
    block = _pdf_block(raw, path="/outside/world/report.pdf")
    block.digest = ref.digest
    world = tmp_path / "world"
    world.mkdir()
    note = with_attachment_hints([Message.user("read", documents=[block])], world_root=world)[
        0
    ].content[-1]
    assert isinstance(note, TextContent)
    assert "/outside/world/report.pdf" not in note.text
    assert "no workspace file" in note.text


# ---------------------------------------------------------------------------
# G. compaction manifest
# ---------------------------------------------------------------------------


def test_the_manifest_merges_across_two_compactions() -> None:
    raw = b"x" * 40
    first = Message.user(
        "one",
        documents=[
            DocumentContent(
                mime_type="application/pdf",
                filename="one.pdf",
                bytes=len(raw),
                digest="a" * 32,
                path="/tmp/one.pdf",
            )
        ],
    )
    payload = with_attachment_manifest({"snapcompact": {"text": "x"}}, [first])
    assert payload is not None
    entries = payload[ATTACHMENTS_MANIFEST_KEY]
    assert [entry["filename"] for entry in entries] == ["one.pdf"]

    marker = build_compaction_marker("summary", payload)
    rendered = render_compaction_marker(marker)
    assert "Files attached earlier in this conversation:" in rendered.text
    assert "one.pdf" in rendered.text

    second = Message.user(
        "two",
        documents=[
            DocumentContent(
                mime_type="text/markdown",
                filename="two.md",
                bytes=12,
                digest="b" * 32,
            )
        ],
    )
    merged = with_attachment_manifest({ATTACHMENTS_MANIFEST_KEY: entries}, [rendered, second])
    assert merged is not None
    names = [entry["filename"] for entry in merged[ATTACHMENTS_MANIFEST_KEY]]
    assert names == ["one.pdf", "two.md"]


def test_the_manifest_renders_on_the_snapcompact_branch() -> None:
    preserve = {
        "snapcompact": {"text_head": "head", "text_tail": "tail"},
        ATTACHMENTS_MANIFEST_KEY: [
            {"digest": "a" * 32, "filename": "a.pdf", "mime_type": "application/pdf", "bytes": 5}
        ],
    }
    rendered = render_compaction_marker(build_compaction_marker(" ", preserve))
    assert "Files attached earlier in this conversation:" in rendered.text
    assert "a.pdf" in rendered.text


def test_an_empty_manifest_renders_nothing_and_keeps_old_bytes() -> None:
    assert render_attachment_manifest(None) == ""
    assert render_attachment_manifest({"snapcompact": {}}) == ""
    plain = render_compaction_marker(build_compaction_marker("summary"))
    assert plain.text == "<previous-context-summary>\nsummary\n</previous-context-summary>"
    assert with_attachment_manifest(None, []) is None


def test_the_manifest_caps_entries_and_drops_the_oldest() -> None:
    messages = [
        Message.user(
            "x",
            documents=[
                DocumentContent(
                    mime_type="application/pdf",
                    filename=f"{index}.pdf",
                    bytes=1,
                    digest=f"{index:032x}",
                )
            ],
        )
        for index in range(25)
    ]
    entries = collect_attachment_manifest(messages)
    # Both caps apply: at most twenty entries AND ~2 kB of JSON, so a long
    # list loses its OLDEST attachments first.
    assert len(entries) <= 20
    assert entries[-1]["filename"] == "24.pdf"
    assert "0.pdf" not in {entry["filename"] for entry in entries}
    assert entries[0]["filename"] == "10.pdf"


# ---------------------------------------------------------------------------
# H. serializers: native shape per wire, fallback otherwise
# ---------------------------------------------------------------------------


def test_openai_chat_sends_a_file_part_for_a_pdf() -> None:
    form = document_form("openai", "openai-chat", supports_images=True)
    message = Message.user("read", documents=[_pdf_block(_pdf_raw())])
    part = _message_to_openai(message, form)["content"][1]
    assert part["type"] == "file"
    assert part["file"]["filename"] == "report.pdf"
    assert part["file"]["file_data"].startswith("data:application/pdf;base64,")


def test_openai_responses_sends_an_input_file_for_a_pdf() -> None:
    model = ModelSpec(provider="openai", model_id="gpt-4o", supports_images=True)
    message = Message.user("read", documents=[_pdf_block(_pdf_raw())])
    item = _messages_to_openai_responses([message], model)[0]
    assert item["content"][1]["type"] == "input_file"
    assert item["content"][1]["file_data"].startswith("data:application/pdf;base64,")


def test_anthropic_sends_a_base64_document_block() -> None:
    form = document_form("anthropic", "anthropic-messages")
    message = Message.user("read", documents=[_pdf_block(_pdf_raw())])
    block = AnthropicClient._message_blocks(message, form)[1]
    assert block["type"] == "document"
    assert block["source"]["type"] == "base64"
    assert block["source"]["media_type"] == "application/pdf"


def test_google_sends_inline_data_for_a_pdf() -> None:
    model = ModelSpec(provider="google", model_id="gemini-2.5-pro", supports_images=True)
    request = ChatRequest(
        model=model, messages=[Message.user("read", documents=[_pdf_block(_pdf_raw())])]
    )
    body = GoogleClient()._build_body(request)
    part = body["contents"][0]["parts"][1]
    assert part["inline_data"]["mime_type"] == "application/pdf"


def test_mistral_and_openrouter_get_their_own_spellings() -> None:
    message = Message.user("read", documents=[_pdf_block(_pdf_raw())])
    mistral = _message_to_openai(
        message, document_form("mistral", "openai-chat", supports_images=True)
    )["content"][1]
    assert mistral["type"] == "document_url"
    openrouter = _message_to_openai(
        message, document_form("openrouter", "openai-chat", supports_images=True)
    )["content"][1]
    assert openrouter["type"] == "file"


def test_a_provider_without_a_row_gets_extracted_text_never_a_dropped_block() -> None:
    message = Message.user("read", documents=[_pdf_block(_pdf_raw())])
    part = _message_to_openai(message, None)["content"][1]
    assert part["type"] == "text"
    assert "--- page 1 ---" in part["text"]
    assert "[Document: report.pdf" in part["text"]


def test_the_policy_is_decided_per_request_not_by_the_admission_label() -> None:
    block = _pdf_block(_pdf_raw())
    # A turn admitted while an Anthropic model was selected...
    block.injection = "native"
    message = Message.user("read", documents=[block])
    # ...re-rendered after a switch to a provider with no document row.
    part = _message_to_openai(message, None)["content"][1]
    assert part["type"] == "text"
    # And the same block still renders natively for an Anthropic request.
    native = AnthropicClient._message_blocks(
        message, document_form("anthropic", "anthropic-messages")
    )[1]
    assert native["type"] == "document"


def test_injection_labels_come_from_the_form_and_limits() -> None:
    raw = _pdf_raw()
    block = _pdf_block(raw)
    assert document_injection(block, document_form("anthropic", "anthropic-messages")) == "native"
    assert document_injection(block, None) == "extracted"
    binary = DocumentContent(
        data=base64.b64encode(b"\x00\x01").decode(),
        mime_type="application/octet-stream",
        filename="x.bin",
        bytes=2,
    )
    assert document_injection(binary, None) == "reference"


def test_a_pdf_extraction_is_page_marked_and_cached(store) -> None:
    raw = _pdf_raw(pages=3)
    block = _pdf_block(raw)
    block.digest = "d" * 32
    text = extract_document_text(block)
    assert text.count("--- page ") == 3
    cached = store.root / f"{'d' * 32}.txt"
    assert cached.exists()
    # The cache holds the EXTRACTION, not the display header: the same bytes
    # re-attached under another filename must not replay the old name.
    assert cached.read_text(encoding="utf-8") in text
    assert "--- page 1 ---" in cached.read_text(encoding="utf-8")
    # Same bytes under a new name: the cached extraction is reused but the
    # header names the filename this send actually carries.
    renamed = DocumentContent(
        data=base64.b64encode(raw).decode("ascii"),
        mime_type="application/pdf",
        filename="renamed.pdf",
        bytes=len(raw),
        digest="d" * 32,
    )
    assert extract_document_text(renamed).startswith("[Document: renamed.pdf")
    # A malformed PDF degrades to a reference naming the file, never nothing.
    broken = DocumentContent(
        data=base64.b64encode(b"%PDF-1.4 not really").decode(),
        mime_type="application/pdf",
        filename="broken.pdf",
        bytes=20,
        path="/tmp/broken.pdf",
    )
    degraded = extract_document_text(broken)
    assert "broken.pdf" in degraded
    assert degraded.strip()


def test_a_resolved_row_writes_back_to_the_same_reference(store) -> None:
    """Second write round trip: resolve -> parse -> encode must not churn.

    A resumed session re-persists what it replayed (compaction rewrites, a
    branch copy). If the second encode produced a different reference, the
    store would slowly accumulate a new blob per replay.
    """
    raw = _pdf_raw()
    first = encode_message_payload(Message.user("read", documents=[_pdf_block(raw)]), store)
    resolved = {"role": "user", "content": [dict(first["content"][1])]}
    _resolve_attachments(resolved, store)
    parsed = coerce_content_blocks(resolved["content"])[0]
    second = encode_message_payload(Message.user("read", documents=[parsed]), store)
    assert second["content"][1]["attachment"] == first["content"][1]["attachment"]


def test_prompt_bounds_the_file_list_and_validates_the_digest() -> None:
    from local_operator.server.routes.desktop_sessions import FileRef, Prompt

    def prompt(count: int) -> Prompt:
        return Prompt(
            request_id="11111111-2222-3333-4444-555555555555",
            text="",
            files=[FileRef(digest=f"{index:032x}") for index in range(count)],
        )

    assert len(prompt(8).files) == 8
    with pytest.raises(ValidationError):
        prompt(9)
    with pytest.raises(ValidationError):
        Prompt(
            request_id="11111111-2222-3333-4444-555555555555",
            text="",
            files=[FileRef(digest="not-a-digest")],
        )
    # An empty send with only files is a message: the composer may send a file
    # with no text.
    assert prompt(1).text == ""


def test_the_missing_file_refusal_decodes_by_name_not_by_prose() -> None:
    error = admission_error("attachment_missing", filename="Complete_With_Docusign.pdf")
    assert isinstance(error, AttachmentMissing)
    assert "Complete_With_Docusign.pdf" in str(error)
    # An older runtime's frame carries no name: the sentence degrades, never
    # renders owner prose.
    bare = admission_error("attachment_missing")
    assert isinstance(bare, AttachmentMissing)
    assert "attached file" in str(bare)


@pytest.mark.asyncio
async def test_the_viewer_door_sends_file_references_on_the_frame() -> None:
    from local_operator.session.attached import AttachedSession

    class FakeClient:
        connected = True
        _document_attachments_supported = True

        def __init__(self) -> None:
            self.frames: list[tuple[str, dict]] = []

        async def request_ack_with_duplicate(self, op, **fields):
            self.frames.append((op, fields))
            return ("admitted", False)

    session = object.__new__(AttachedSession)
    session._client = FakeClient()

    async def _bound() -> None:
        return None

    session._ensure_bound = _bound  # type: ignore[method-assign]
    reference = {"digest": "a" * 32, "name": "a.pdf", "bytes": 1}
    await session.admit_prompt("read", command_id="c-1", images=[], files=[reference])
    op, fields = session._client.frames[0]
    assert op == "prompt"
    assert fields["files"] == [reference]
    # No files -> no key at all, so a legacy frame stays byte-identical.
    await session.admit_prompt("hello", command_id="c-2", images=[])
    assert "files" not in session._client.frames[1][1]

    # An owner that never advertised the capability REFUSES the send rather
    # than accepting the text without the file (contract amendment 1).
    session._client._document_attachments_supported = False
    with pytest.raises(DocumentCarriageUnsupported):
        await session.admit_prompt("read", command_id="c-3", images=[], files=[reference])


def test_an_image_wire_entry_carries_its_origin_path() -> None:
    """``Prompt.images[].origin_path`` -> ``ImageContent.path`` (contract §1.3).

    The declared key is optional, so an older client's entry still produces a
    block with no path (``None``) rather than an extra null durable field.
    """
    import io as _io

    from PIL import Image as _PILImage

    from local_operator.session.runtime.server import image_blocks

    buffer = _io.BytesIO()
    _PILImage.new("RGB", (8, 8), "red").save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    with_path = image_blocks(
        [{"data_b64": payload, "mime_type": "image/png", "origin_path": "/tmp/i.png"}]
    )[0]
    assert with_path.path == "/tmp/i.png"
    without = image_blocks([{"data_b64": payload, "mime_type": "image/png"}])[0]
    assert without.path is None


@pytest.mark.asyncio
async def test_a_pdf_files_send_reaches_message_content_end_to_end(tmp_path: Path, store) -> None:
    """THE REPRODUCTION, through the real owner handle and a real session.

    A composer upload stores a PDF, ``Prompt.files`` names its digest, and the
    user message the model receives must carry a ``DocumentContent`` with the
    filename, MIME, byte count and absolute origin path — the exact shape the
    operator's session ``439818272d84`` lacked when the chip showed a PDF and
    the turn carried an image instead.
    """
    import asyncio

    from local_operator.providers.clients import MockClient
    from local_operator.session.runtime.serving import ServingSessionHandle
    from tests.e2e.harness import build_session

    raw = _pdf_raw()
    ref = store.put_bytes(raw, "application/pdf")
    assert ref is not None
    session = build_session(tmp_path / "session", MockClient().stream)
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    try:
        receipt = await handle.prompt(
            "Summarise the agreement",
            files=[
                {
                    "digest": ref.digest,
                    "name": "Complete_With_Docusign.pdf",
                    "mime_type": "application/pdf",
                    "bytes": len(raw),
                    "path": "/Users/victor/Documents/Complete_With_Docusign.pdf",
                }
            ],
            command_id="c-pdf-1",
            # Wait for the TURN, not just the admission: ``_context.messages``
            # is updated by the pipeline, while the admission future resolves
            # at the durable row append.
            wait_complete=True,
        )
        assert receipt == "prompt admitted"
        user = next(
            message
            for message in session.history()
            if isinstance(message, Message) and message.role == "user"
        )
        document = next(block for block in user.content if isinstance(block, DocumentContent))
        assert document.filename == "Complete_With_Docusign.pdf"
        assert document.mime_type == "application/pdf"
        assert document.bytes == len(raw)
        assert document.path == "/Users/victor/Documents/Complete_With_Docusign.pdf"
        assert document.digest == ref.digest
        assert base64.b64decode(document.data) == raw

        # The durable row externalises the bytes and keeps the reference.
        rows = [
            json.loads(line)
            for line in Path(session.transcript.path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        document_row = next(
            block
            for row in rows
            for block in (row.get("payload") or {}).get("content") or []
            if block.get("type") == "document"
        )
        assert document_row["attachment"] == ref.digest
        assert "data" not in document_row
    finally:
        await session.dispose()
