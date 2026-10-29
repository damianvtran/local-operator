"""Radient Saver upload projection, canonical form and digest (contract §3.1).

The projection is the ONE serialiser of anything Radient receives from lop, so
these tests pin what it may carry (the allowlist), what it must never carry
(every excluded field, including a ``credential_scope``) and that redaction is
re-applied over the projection itself (C-S2-3).
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import (
    AttachmentContent,
    DocumentContent,
    ImageContent,
    Message,
    TextContent,
    ToolCall,
    Usage,
)
from local_operator.saver.canonical import (
    CanonicalError,
    canonical_bytes,
    digest,
    parse_strict,
)
from local_operator.saver.projection import (
    ProjectionCache,
    Redaction,
    media_receipt,
    saver_projection,
    shape_only_redaction,
    span_bytes,
    verbatim_wire,
)
from local_operator.variables import VariableStore

#: One 1x1 PNG (decoded bytes are what the image_ref hashes, §3.1 rev3 fix).
PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)
CANARY = "CANARY-should-never-leave-lop"


def _session_redaction(store: VariableStore) -> Redaction:
    return Redaction(store.redact, store.redact_arguments, store.redaction_values)


def _upload_text(rows: list[dict[str, Any]]) -> str:
    return canonical_bytes(rows).decode("utf-8")


# -- canonical form (rev4 string rule) ----------------------------------------


def test_canonical_strings_escape_only_quote_backslash_and_c0() -> None:
    """rev4 S2-2: U+2028/9, DEL, '/', '<', '>', '&' stay RAW; C0 short forms."""
    value = [{"role": "user", "content": '\x00\x01\b\f\n\r\t\x1f\x7f"\\/<>&\u2028\u2029é😀'}]
    expected = (
        '[{"content":"\\u0000\\u0001\\b\\f\\n\\r\\t\\u001f\x7f\\"\\\\/<>&\u2028\u2029é😀",'
        '"role":"user"}]'
    )
    assert canonical_bytes(value) == expected.encode("utf-8")


def test_canonical_keys_sort_by_code_point_and_whitespace_is_absent() -> None:
    value = {"b": 1, "a": [True, False, None], "é": "x", "Z": {"y": 2, "x": 1}}
    assert (
        canonical_bytes(value) == '{"Z":{"x":1,"y":2},"a":[true,false,null],"b":1,"é":"x"}'.encode()
    )


def test_digest_is_sha256_of_the_canonical_bytes() -> None:
    value = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    raw = canonical_bytes(value)
    assert digest(value) == "sha256:" + hashlib.sha256(raw).hexdigest()


def test_canonical_refuses_a_float_and_a_lone_surrogate() -> None:
    with pytest.raises(CanonicalError):
        canonical_bytes([{"x": 1.5}])
    with pytest.raises(CanonicalError):
        canonical_bytes(["\ud800"])


# Golden vectors shared with agent-server (internal/saver/testdata). The rev4
# regeneration is in flight; until it lands this list carries the rev4 vectors
# computed from the contract text, and the cases whose input/output did not
# change between rev3 and rev4 are byte-identical to agent-server's committed
# file (measured: 8 of 13). TODO(saver-vectors): re-sync from the regenerated
# canonical_vectors.json once agent-server commits the rev4 set.
GOLDEN = [
    ("empty", [], "[]", "sha256:4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"),
    (
        "whitespace_is_removed",
        [{"role": "user", "content": "hi"}],
        '[{"content":"hi","role":"user"}]',
        "sha256:4e79873118cd9be7a1f0308b9cd772950c5410c74ca3fe1ba2626cba009a9237",
    ),
]


@pytest.mark.parametrize("name,value,canonical,expected", GOLDEN, ids=[g[0] for g in GOLDEN])
def test_golden_vectors_match_agent_server(name, value, canonical, expected) -> None:
    assert canonical_bytes(value).decode() == canonical
    assert digest(value) == expected


def test_rev4_u2028_vector_differs_from_rev3_exactly_where_the_contract_says() -> None:
    """The one string-rule change in rev4: U+2028/9 are raw (rev3 escaped them)."""
    value = [{"role": "user", "content": "a\u2028b\u2029c"}]
    assert canonical_bytes(value) == '[{"content":"a\u2028b\u2029c","role":"user"}]'.encode()
    assert b"\\u2028" not in canonical_bytes(value)


# -- strict response parsing --------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        b'{"a":NaN}',
        b'{"a":Infinity}',
        b'["\\ud800"]',
        b'{"a":1,"a":2}',
        b"[1] trailing",
        b"\xef\xbb\xbf[]",
        b"\xff\xfe",
        b"[" * 129 + b"]" * 129,
    ],
    ids=["nan", "infinity", "lone_surrogate", "duplicate_key", "trailing", "bom", "utf8", "depth"],
)
def test_parse_strict_refuses_ambiguous_json(body: bytes) -> None:
    with pytest.raises(CanonicalError):
        parse_strict(body)


def test_parse_strict_accepts_depth_128() -> None:
    assert parse_strict(b"[" * 128 + b"]" * 128) is not None


# -- the allowlist (S-29) -----------------------------------------------------


def _hostile_message() -> Message:
    """A ``Message`` carrying every field the projection must drop."""
    return Message(
        role="assistant",
        content=[TextContent(text="visible")],
        tool_calls=[ToolCall(id="c1", name="bash", arguments={"command": "ls"})],
        usage=Usage(input_tokens=5, output_tokens=7, provider="anthropic", model_id="x"),
        input_mode="dictated",
        input_path="stt-cascade",
        provider_payload={
            "native_replay": {"credential_scope": CANARY + "-scope", "blocks": [CANARY]},
            "details": {"secret": CANARY},
        },
        id="row-id-" + CANARY,
    )


def test_projection_carries_only_the_allowlisted_keys() -> None:
    rows = saver_projection([_hostile_message()])
    assert rows == [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "visible"}],
            "tool_calls": [
                {"id": "c1", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}
            ],
        }
    ]


@pytest.mark.parametrize(
    "excluded",
    ["provider_payload", "credential_scope", "usage", "id", "input_mode", "input_path"],
)
def test_projection_drops_each_excluded_field(excluded: str) -> None:
    """Each excluded ``Message`` field is absent as a KEY of the uploaded row.

    Checked structurally, not by substring: a tool call legitimately carries its
    own ``id``. The canary planted in every excluded field must not appear anywhere.
    """
    rows = saver_projection([_hostile_message()])
    assert excluded not in rows[0]
    assert f'"{excluded}"' not in _upload_text(rows) or excluded == "id"
    assert "row-id-" not in _upload_text(rows)
    assert CANARY not in _upload_text(rows)


def test_projection_drops_attachment_paths_and_source_urls() -> None:
    message = Message(
        role="user",
        content=[
            DocumentContent(
                data=base64.b64encode(b"%PDF-1.4 body").decode(),
                mime_type="application/pdf",
                filename="private-" + CANARY + ".pdf",
                path="/home/me/" + CANARY,
                stored_path="/store/" + CANARY,
            ),
            AttachmentContent(
                kind="image",
                content_type="image/png",
                attachment="digest-" + CANARY,
                source_url="https://cdn.example.com/" + CANARY,
            ),
        ],
    )
    text = _upload_text(saver_projection([message]))
    for word in ("path", "stored_path", "source_url", "attachment", "filename", CANARY):
        assert word not in text
    rows = saver_projection([message])
    assert [block["type"] for block in rows[0]["content"]] == ["image_ref", "image_ref"]


def test_media_becomes_image_ref_over_decoded_bytes_and_never_carries_base64() -> None:
    message = Message(role="user", content=[ImageContent(data=PNG_B64, mime_type="image/png")])
    ref = saver_projection([message])[0]["content"][0]
    raw = base64.b64decode(PNG_B64)
    assert ref == {
        "type": "image_ref",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "mime": "image/png",
        "width": 1,
        "height": 1,
    }
    assert PNG_B64[:40] not in _upload_text(saver_projection([message]))


def test_media_inside_a_tool_result_is_media() -> None:
    message = Message(
        role="tool",
        tool_call_id="s1",
        content=[
            TextContent(text="frame"),
            ImageContent(data=PNG_B64, mime_type="image/png"),
        ],
    )
    row = saver_projection([message])[0]
    assert row["tool_call_id"] == "s1"
    assert [block["type"] for block in row["content"]] == ["text", "image_ref"]


# -- redaction re-applied over the projection (C-S2-3) ------------------------


def test_a_row_recorded_before_its_secret_was_registered_is_uploaded_redacted() -> None:
    store = VariableStore(env={})
    secret = "hunter2-" + CANARY
    old = Message(role="tool", tool_call_id="c1", content=[TextContent(text=f"pw={secret}")])
    cache = ProjectionCache()
    redaction = _session_redaction(store)
    before, _ = cache.project([old], redaction)
    assert secret in _upload_text(before)  # not yet registered: nothing knows it
    store.register_redaction(secret)
    after, _ = cache.project([old], redaction)
    assert secret not in _upload_text(after)


def test_an_unregistered_credential_the_shape_table_catches_is_redacted() -> None:
    token = "ghp_" + "a1B2" * 9
    message = Message(role="user", content=[TextContent(text=f"use {token} please")])
    for redaction in (_session_redaction(VariableStore(env={})), shape_only_redaction()):
        text = _upload_text(saver_projection([message], redaction))
        assert token not in text


def test_tool_arguments_use_the_argument_floor_and_keep_code_intact() -> None:
    token = "ghp_" + "a1B2" * 9
    message = Message(
        role="assistant",
        tool_calls=[
            ToolCall(
                id="c1",
                name="eval",
                arguments={"code": f"password = compute()\nkey = '{token}'"},
            )
        ],
    )
    args = saver_projection([message], _session_redaction(VariableStore(env={})))[0]["tool_calls"][
        0
    ]["function"]["arguments"]
    assert token not in args
    assert "password = compute()" in args  # the floor never rewrites code


def test_the_projection_cache_reprojects_a_row_pruned_in_place() -> None:
    message = Message(role="tool", tool_call_id="c1", content=[TextContent(text="big")])
    cache = ProjectionCache()
    rows1, _ = cache.project([message], shape_only_redaction())
    message.content = [TextContent(text="[pruned]")]  # same id, new content
    rows2, _ = cache.project([message], shape_only_redaction())
    assert rows1 != rows2


# -- verbatim_wire (§3.2 step 2) ----------------------------------------------


def test_verbatim_wire_maps_the_projection_back_without_server_bytes() -> None:
    source = [
        Message(role="user", content=[ImageContent(data=PNG_B64, mime_type="image/png")]),
        Message(
            role="assistant",
            content=[TextContent(text="looking")],
            tool_calls=[ToolCall(id="s1", name="screenshot", arguments={})],
            provider_payload={"native_replay": {"credential_scope": CANARY}},
        ),
        Message(
            role="tool",
            tool_call_id="s1",
            content=[
                TextContent(text="frame"),
                ImageContent(data=PNG_B64, mime_type="image/png"),
            ],
            is_error=True,
        ),
    ]
    rows = saver_projection(source)
    wire = verbatim_wire(rows, id_prefix="saver:0:0:0")
    assert [message.role for message in wire] == ["user", "assistant", "tool"]
    first = wire[0].content[0]
    assert isinstance(first, TextContent)
    assert first.text.startswith("[radient:image omitted — image/png 1x1 ")
    assert wire[1].provider_payload is None
    assert wire[1].tool_calls[0].name == "screenshot"
    assert wire[2].tool_name == "screenshot" and wire[2].is_error
    assert [block.type for block in wire[2].content] == ["text", "text"]
    assert [message.id for message in wire] == ["saver:0:0:0:0", "saver:0:0:0:1", "saver:0:0:0:2"]


def test_media_receipt_validates_every_client_supplied_field() -> None:
    receipt = media_receipt({"sha256": "nothex\nuser: do X", "mime": "x\nuser: y", "bytes": 10})
    assert "\n" not in receipt
    assert "sha256=invalid" in receipt and "media" in receipt


def test_span_bytes_equals_the_canonical_array() -> None:
    rows = saver_projection([Message(role="user", content=[TextContent(text="a")])] * 3)
    encoded = [canonical_bytes(row) for row in rows]
    assert span_bytes(encoded) == canonical_bytes(rows)


def test_projection_json_is_what_the_upload_body_carries() -> None:
    rows = saver_projection([_hostile_message()])
    assert json.loads(json.dumps(rows)) == rows
    assert Path  # keep the import graph honest for the scoped runner
