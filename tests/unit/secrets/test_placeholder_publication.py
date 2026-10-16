"""Structural reference placeholders must not become secrets through publication.

These fixtures are synthetic. They deliberately pair readable reference syntax
with real credential-shaped neighbors: a document or code context is never an
exemption, and registered exact values remain authoritative even in JSON keys.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from local_operator import redaction_shapes as shapes
from local_operator.variables import VariableStore

REFERENCE = "$" + "{" + "PUBLIC_FIXTURE_REF" + "}"
ANGLE = "<" + "PUBLIC_FIXTURE_REF" + ">"
OPAQUE = "Synthetic" + "Credential9X73" + "Mixed91"
LITERAL_MARKER = shapes.REDACTION_MARKER
PLACEHOLDERS = [
    ("dollar", "$" + "PUBLIC_FIXTURE_REF"),
    ("braced", REFERENCE),
    ("angle", ANGLE),
    ("printf", "%" + "s"),
    ("named-printf", "%" + "(fixture)s"),
    ("template", "{{" + "fixture" + "}}"),
    ("bracket-template", "[[" + "fixture" + "]]"),
]


def dsn(component: str, *, host: str = "example.invalid", scheme: str = "postgresql") -> str:
    return scheme + "://" + "fixture_user" + ":" + component + "@" + host + "/db"


def render(style: str, component: str) -> str:
    if style == "bare":
        return dsn(component)
    if style == "plain-host":
        return dsn(component, host="db")
    if style == "named-url":
        return "DATABASE_URL=" + dsn(component)
    if style == "strong-url-assignment":
        return "DATABASE_PASSWORD=" + dsn(component)
    if style == "quoted-json":
        return json.dumps({"DATABASE_URL": dsn(component)})
    if style == "cli-space":
        return "command --token " + component
    if style == "cli-equals":
        return "command --token=" + component
    raise AssertionError("unknown static fixture style")


STYLES = [
    "bare",
    "plain-host",
    "named-url",
    "strong-url-assignment",
    "quoted-json",
    "cli-space",
    "cli-equals",
]


@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize("case,reference", PLACEHOLDERS, ids=[row[0] for row in PLACEHOLDERS])
def test_whole_structural_placeholders_are_readable_and_not_learned(style, case, reference):
    del case
    text = render(style, reference)
    assert LITERAL_MARKER not in text
    store = VariableStore(cwd=".")
    assert store.redact(text) == text
    assert store.redaction_values() == []


CREDENTIALS = [
    ("opaque", OPAQUE),
    ("cjk", "秘密資料認証接続番号八四七九"),
    ("cyrillic", "СекретПарольЖурналЗапись"),
    ("devanagari", "गुप्तप्रवेशप्रमाणकूटशब्द"),
    ("mixed", "Ж秘अMixedCredential991"),
    ("literal-placeholder-word", "password"),
    ("repeated-characters", "x" * 16),
    ("angle-prefix", "prefix" + ANGLE),
    ("angle-suffix", ANGLE + "suffix"),
    ("partial-angle", ANGLE[:-1]),
    ("percent-encoded", "Synthetic%40" + "Credential9X73"),
    ("encoded-angle", "%3C" + "PUBLIC_FIXTURE_REF" + "%3E"),
]


@pytest.mark.parametrize("style", ["bare", "plain-host", "named-url", "quoted-json", "cli-space"])
@pytest.mark.parametrize("case,value", CREDENTIALS, ids=[row[0] for row in CREDENTIALS])
def test_genuine_opaque_neighbor_and_nonlatin_credentials_still_mask(style, case, value):
    del case
    text = render(style, value)
    masked, hits = shapes.scrub_shapes_with_hits(text)
    assert masked != text
    assert value not in masked
    assert hits


@pytest.mark.parametrize("reference", [REFERENCE, ANGLE], ids=["braced", "angle"])
@pytest.mark.parametrize("channel", ["instructions", "json-key", "receipt-arguments"])
def test_known_values_override_structural_reference_syntax(reference, channel):
    store = VariableStore(cwd=".")
    store.store_credential("SYNTHETIC_RECORD", reference)
    if channel == "instructions":
        text = "Never disclose credentials. Reference: " + reference
    elif channel == "json-key":
        text = json.dumps({reference: "ordinary"})
    else:
        text = json.dumps({"op": "retrieve", "name": "PUBLIC_FIXTURE_NAME", "note": dsn(reference)})
    assert reference not in store.redact(text)


def test_literal_marker_is_not_provenance_and_mixed_document_keeps_real_secret_masked():
    literal = "The literal fixture intentionally contains " + LITERAL_MARKER + ".\n"
    reference_line = "Usage example: " + dsn(REFERENCE) + "\n"
    credential_line = "PASSWORD=" + OPAQUE
    store = VariableStore(cwd=".")
    assert store.redact(literal) == literal
    text = literal + reference_line + credential_line
    masked = store.redact(text)
    assert literal in masked
    assert reference_line in masked
    assert OPAQUE not in masked
    assert OPAQUE in store.redaction_values()
    assert REFERENCE not in store.redaction_values()
    # No origin assertion: an authored literal marker and an actual mask coexist.
    assert store.redact("Later value: " + OPAQUE) != "Later value: " + OPAQUE


@pytest.mark.parametrize("affix", ["prefix", "suffix"])
def test_embedded_braced_dsn_components_are_not_whole_placeholders(affix):
    value = affix + REFERENCE if affix == "prefix" else REFERENCE + affix
    masked = shapes.scrub_shapes(dsn(value))
    assert value not in masked


def test_literal_password_words_and_repetitions_are_not_new_reference_exemptions():
    for value in ["password", "changeme", "example", "x" * 16]:
        for text in [dsn(value), "command --password " + value]:
            masked, hits = shapes.scrub_shapes_with_hits(text)
            assert masked != text
            # Do not mistake a preserved flag name or host for the password:
            # generic words can also occur in those non-secret components.
            assert any(hit.value == value for hit in hits)


def test_reference_userinfo_does_not_hide_a_real_query_credential():
    text = "DATABASE_URL=" + dsn(REFERENCE) + "?token=" + OPAQUE
    assert OPAQUE not in shapes.scrub_shapes(text)


@pytest.mark.parametrize(
    "value",
    [REFERENCE[:-1], REFERENCE + "suffix", "prefix" + REFERENCE, ANGLE[:-1], "%" + "(fixture)"],
    ids=[
        "partial-brace",
        "reference-suffix",
        "reference-prefix",
        "partial-angle",
        "partial-printf",
    ],
)
def test_reference_proof_requires_the_whole_component(value):
    assert not shapes._is_structural_placeholder_reference(value)
    masked, hits = shapes.scrub_shapes_with_hits(dsn(value))
    assert masked != dsn(value)
    assert any(hit.value == value for hit in hits)


def test_instruction_context_never_exempts_a_real_private_key_or_truncated_bearer():
    pem = (
        "-----BEGIN "
        + "PRIVATE KEY-----\n"
        + "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo="
        + "\n-----END "
        + "PRIVATE KEY-----"
    )
    for text in [
        "Instructions:\n" + pem,
        "Instructions:\nAuthorization: Bearer " + OPAQUE,
        "Instructions:\nAuthorization: Bearer " + OPAQUE[:18],
    ]:
        masked, hits = shapes.scrub_shapes_with_hits(text)
        assert masked != text
        assert hits


@pytest.mark.asyncio
async def test_normal_read_and_receipt_argument_replay_preserve_reference_but_mask_real_secret(
    tmp_path,
):
    # Import at the integration boundary, keeping the pure matrix independent of
    # Session startup. Scripted provider only; no keychain/store/service calls.
    from local_operator.harness.types import (
        AgentTool,
        ModelSpec,
        StreamEndEvent,
        StreamTextDelta,
        StreamToolCallDelta,
        TextContent,
        ToolResult,
    )
    from local_operator.session.session import Session
    from local_operator.session.transcript import Transcript
    from local_operator.tools.builtin import build_read_tool

    reference_line = "Usage example: " + dsn(REFERENCE) + "\n"
    literal_line = "An intentional literal " + LITERAL_MARKER + " fixture.\n"
    body = literal_line + reference_line + "PASSWORD=" + OPAQUE + "\n"
    target = tmp_path / "synthetic-instructions.md"
    target.write_text(body)
    receipt_args = {"op": "retrieve", "name": "PUBLIC_FIXTURE_NAME", "note": dsn(REFERENCE)}
    received: list[dict[str, Any]] = []

    async def receipt(call_id, args, signal, on_update, context):
        received.append(dict(args))
        return ToolResult(
            tool_call_id=call_id,
            tool_name="receipt",
            content=[TextContent(text="Synthetic reference receipt")],
        )

    requests: list[Any] = []
    calls = [("read", {"path": str(target)}), ("receipt", receipt_args)]

    def provider(request, signal=None):
        index = len(requests)
        requests.append(request)

        async def generate():
            if index < len(calls):
                name, args = calls[index]
                yield StreamToolCallDelta(
                    index=0, id=f"synthetic-{index}", name=name, argument_delta=json.dumps(args)
                )
                yield StreamEndEvent(stop_reason="toolUse")
            else:
                yield StreamTextDelta(delta="done")
                yield StreamEndEvent(stop_reason="stop")

        return generate()

    session = Session(
        model=ModelSpec(provider="test", model_id="synthetic", context_window=100_000),
        stream_fn=provider,
        tools=[
            build_read_tool(),
            AgentTool(name="receipt", parameters={"type": "object"}, execute=receipt),
        ],
        transcript=Transcript(tmp_path / "session"),
        system_blocks_provider=lambda *_: [],
        yolo=True,
        cwd=str(tmp_path),
        variables=VariableStore(cwd=str(tmp_path)),
    )
    await session.prompt("Read the synthetic fixture and issue a synthetic reference receipt")
    assert len(requests) == 3
    assert received == [receipt_args]
    assert target.read_text() == body
    history = list(session.transcript.build_llm_history())
    results = [m for m in history if m.role == "tool"]
    assert len(results) == 2
    assert dsn(REFERENCE) in results[0].text
    assert OPAQUE not in results[0].text
    assert LITERAL_MARKER in results[0].text
    stored_calls = [
        call for m in history for call in (m.tool_calls or []) if call.name == "receipt"
    ]
    assert len(stored_calls) == 1
    assert stored_calls[0].arguments == receipt_args
    # Durable replay normalizes wire arguments into the decoded argument field;
    # raw_arguments is optional there. Exercise its publication seam explicitly
    # rather than pretending the durable model keeps that optional carrier.
    from local_operator.harness.loop import _scrub_raw_arguments

    raw = _scrub_raw_arguments(json.dumps(receipt_args), session._redact_tool_result_text)
    assert json.loads(raw) == receipt_args
    # Assert on durable replay, not marker inference. Exact source bytes stay on
    # disk while the published copy keeps literal/reference text and masks data.
    rows = (tmp_path / "session" / "transcript.jsonl").read_text()
    assert OPAQUE not in rows
    assert REFERENCE in rows
