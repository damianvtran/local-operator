"""The push payload and its two keys, asserted against the frozen fixture contract.

Push/ack-sync S3 (ADR 0006 §3.2 payload, §3.4 keys — pinned at
``damianvtran/local-operator-mobile`` @ ``b03aeb15``). ``fixtures/push/`` is the
contract all three repos build against, so these cells compare the core's own
builder against the FILED literals instead of restating the shapes: a field added
to a builder fails an equality here, a field removed fails it too, and a forbidden
field fails the deny-list scan wherever it is injected. That bidirectionality is
the point of the slice — a contract test that only ever agrees with the code it
tests is a second copy of the code.

Nothing here touches a daemon or a store: this is the payload half of the freeze
and it is pure data. The registry half — the real builders, the real routes —
lives in ``test_push_wire_contract.py``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from local_operator.mobile import push_credentials, push_handles, push_payload
from local_operator.mobile.push_payload import (
    EMIT_ROUTE,
    IDEMPOTENCY_HEADER,
    PAYLOAD_VERSION,
    PUSH_KINDS,
    TYPE_ATTENTION,
    TYPE_COMPLETION,
    attention_emit_key,
    attention_payload,
    completion_emit_key,
    completion_payload,
    emit_body,
)

#: ``tests/unit/mobile/<this file>`` -> the repo root, where the shared
#: contract tree lives. Derived from the file rather than the cwd so the suite
#: reads the fixtures of the checkout it is running in.
FIXTURES = Path(__file__).resolve().parents[3] / "fixtures" / "push"

#: The fixture alphabet -> the Python type each name asserts. ``object`` is a
#: JSON object; the two list spellings differ only in what a reader is told about
#: the items, which is what the fixtures are for.
_TYPES: dict[str, type] = {
    "str": str,
    "int": int,
    "bool": bool,
    "object": dict,
    "list[str]": list,
    "list[object]": list,
}


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _assert_type(where: str, value: object, type_name: str) -> None:
    expected = _TYPES[type_name]
    if type_name == "int":
        # ``bool`` is a subclass of ``int``, so a bare isinstance would accept
        # ``"count": true`` — a payload the app would render as "1 unread" or
        # crash on. The freeze says int, so this says int.
        assert isinstance(value, int) and not isinstance(
            value, bool
        ), f"{where}: expected int, got {type(value).__name__}"
        return
    if type_name == "list[str]":
        # The ITEMS, not just the list: a bare ``list`` would accept
        # ``exclude: [1, 2]``, and ``exclude`` carries device ids the cloud
        # matches against its registry (review round 1, N3).
        assert isinstance(value, list), f"{where}: expected list, got {type(value).__name__}"
        offenders = [item for item in value if not isinstance(item, str)]
        assert not offenders, f"{where}: non-string item(s) {offenders!r}"
        return
    assert isinstance(value, expected), f"{where}: expected {type_name}, got {type(value).__name__}"


def _walk_values(value: object) -> list[str]:
    """Every string at every depth, for the value-level half of the deny scan.

    Keys alone are not enough: the rule the file states is about what a raw
    session id or a conversation name must not REACH, and a value is where one
    would travel (review round 1, N3 — a substring test on the serialized body
    would also fail a correct payload whose value merely contained the text).
    """
    if isinstance(value, dict):
        return [item for entry in value.values() for item in _walk_values(entry)]
    if isinstance(value, list):
        return [item for entry in value for item in _walk_values(entry)]
    return [value] if isinstance(value, str) else []


def _check_shape(
    value: dict[str, Any], required: dict[str, str], optional: dict[str, str], *, where: str
) -> None:
    """The allow-list check: exactly the named fields, each of the named type.

    ``required`` must all be present; nothing outside ``required | optional`` may
    appear at all — an unlisted field is a contract change, not an addition, which
    is what ``extra="forbid"`` means on the cloud side of this wire.
    """
    assert isinstance(value, dict), f"{where}: expected an object"
    allowed = set(required) | set(optional)
    unlisted = sorted(set(value) - allowed)
    assert not unlisted, f"{where}: unlisted field(s) {unlisted}"
    for name in sorted(required):
        assert name in value, f"{where}: required field {name!r} is missing"
    for name, type_name in {**required, **optional}.items():
        if name in value:
            _assert_type(f"{where}.{name}", value[name], type_name)


def _walk_keys(value: object) -> list[str]:
    """Every key at every depth — the deny list is scanned, not just the top level."""
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            found.append(str(key))
            found.extend(_walk_keys(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_walk_keys(item))
    return found


def _forbidden() -> dict[str, str]:
    return _fixture("payload-forbidden-fields.json")["forbidden"]


#: One report-block row, the emit cells' INPUT.
#:
#: THE TEST OWNS THE INPUT; THE FILE OWNS THE EXPECTED OUTPUT (review round 1,
#: M1). Feeding ``fixture["body"]["devices"]`` into :func:`emit_body` and
#: comparing the result to that same file made the comparison a tautology: a row
#: edited in the fixture moved both sides at once, which is why the block's
#: declared allow-list could be broken without a red cell. Written here, either
#: side drifting fails — a field dropped from this row is missing from the built
#: body, and a field dropped from the file is missing from the expected one.
DEVICE_ROWS: list[dict[str, Any]] = [
    {
        "device_id": "3f1c9a7b2d4e506182a3b4c5d6e7f809",
        "credential_live": True,
        "credential_expires_at": 1791592000,
        "last_authenticated_at": 1789099000,
    }
]

#: The completion emit's input, spelled here rather than read back from
#: ``emit-completion.json`` for the same reason as :data:`DEVICE_ROWS`.
COMPLETION_INPUT: dict[str, Any] = {
    "computer": "VZ8kQ1mT4pR7sW2xY6bN9c",
    "conversation": "jH3kL9mN2pQ5rS8tU1vW4x",
    "completion_token": "9f5d1d6e-6b1a-4c6e-9b3a-7a1c2f3d4e5f",
    "kind": "complete",
    "emit_id": "6b1e2d3c4a5f60718293a4b5c6d7e8f9",
    "count": 2,
}

#: The attention emit's input: one field fewer (no ``conversation``, no
#: ``completion_token``, no ``kind``) and an ``exclude`` naming one device.
ATTENTION_INPUT: dict[str, Any] = {
    "computer": "VZ8kQ1mT4pR7sW2xY6bN9c",
    "count": 1,
    "emit_id": "1a2b3c4d5e6f708192a3b4c5d6e7f809",
}


def _check_block(rows: object, fixture: dict[str, Any], *, where: str) -> None:
    """Every report-block row against the block's OWN allow-list.

    ``report_block_required`` / ``report_block_optional`` are the §3.2 block's
    declaration, and before this they were read by nothing: the block was only
    ever compared as a pass-through, so an extra ``platform``, a retyped
    ``last_authenticated_at`` or a row missing ``credential_live`` travelled
    unguarded (review round 1, M1).
    """
    assert isinstance(rows, list) and rows, f"{where}: expected a non-empty block"
    for index, row in enumerate(rows):
        _check_shape(
            row,
            fixture["report_block_required"],
            fixture["report_block_optional"],
            where=f"{where}[{index}]",
        )


def _completion_built() -> dict[str, Any]:
    """The builder's output for :data:`COMPLETION_INPUT`, not for the file's.

    The fixture is the EXPECTED side of this comparison; taking the input from it
    too would make the cell agree with whatever the file says (review round 1,
    M1).
    """
    return completion_payload(**COMPLETION_INPUT)


def test_the_completion_payload_and_emit_body_equal_the_filed_literals() -> None:
    fixture = _fixture("emit-completion.json")
    payload = _completion_built()

    _check_shape(
        payload,
        fixture["payload_required"],
        fixture["payload_optional"],
        where="emit-completion.payload",
    )
    assert payload == fixture["payload"], "the builder and the filed payload have diverged"
    assert payload["v"] == PAYLOAD_VERSION
    assert payload["type"] == TYPE_COMPLETION
    assert set(payload) == set(fixture["payload_required"]), (
        "the completion form carries no optional field: a new key must be added to the "
        "fixture's allow-list in the same change"
    )

    body = emit_body(payload, DEVICE_ROWS)
    # Both sides of the block, against the block's own allow-list FIRST, so a
    # block-shaped defect reports which field was wrong rather than the blunter
    # "these two objects differ": the rows this test built, and the literal the
    # next repos copy.
    _check_block(body["devices"], fixture, where="emit-completion.body.devices")
    _check_block(fixture["body"]["devices"], fixture, where="emit-completion.filed.devices")
    assert body == fixture["body"], "the emit body is the payload plus the report block"
    assert set(body) - set(payload) == {push_payload.REPORT_DEVICES_FIELD}


def test_the_attention_payload_equals_the_filed_literals_with_and_without_exclude() -> None:
    fixture = _fixture("emit-attention.json")
    payload = fixture["payload"]

    with_exclude = attention_payload(**ATTENTION_INPUT, exclude=[DEVICE_ROWS[0]["device_id"]])
    _check_shape(
        with_exclude,
        fixture["payload_required"],
        fixture["payload_optional"],
        where="emit-attention.payload",
    )
    assert with_exclude == payload
    assert with_exclude["type"] == TYPE_ATTENTION

    # ``exclude`` absent is a whole, valid payload — not an empty list. A
    # tick-detected change never knows who acknowledged, and the cloud reads one
    # spelling of "exclude nobody".
    without = attention_payload(**ATTENTION_INPUT)
    assert without == fixture["exclude_absent_payload"]
    assert "exclude" not in without

    body = emit_body(with_exclude, DEVICE_ROWS)
    _check_block(body["devices"], fixture, where="emit-attention.body.devices")
    _check_block(fixture["body"]["devices"], fixture, where="emit-attention.filed.devices")
    assert body == fixture["body"], "the emit body is the payload plus the report block"


def test_the_report_block_allow_list_rejects_an_extra_or_mistyped_row() -> None:
    """The block's declaration bites — the mutation the review round filed as M1.

    A pure negative: each of these rows is what the guard must refuse, so the
    cell fails if a later edit loosens ``_check_shape`` or drops the block's
    declaration.
    """
    fixture = _fixture("emit-completion.json")
    good = DEVICE_ROWS[0]

    with pytest.raises(AssertionError, match="unlisted field"):
        _check_block([{**good, "platform": "ios"}], fixture, where="mutated.devices")
    with pytest.raises(AssertionError, match="required field 'credential_live' is missing"):
        _check_block(
            [{key: value for key, value in good.items() if key != "credential_live"}],
            fixture,
            where="mutated.devices",
        )
    with pytest.raises(AssertionError, match="expected int"):
        _check_block(
            [{**good, "last_authenticated_at": "whenever"}], fixture, where="mutated.devices"
        )
    with pytest.raises(AssertionError, match="expected bool"):
        _check_block([{**good, "credential_live": "true"}], fixture, where="mutated.devices")

    # The dropping direction is the emit cells': the block row is the test's own
    # input there, so a row that loses a field the filed body carries cannot
    # equal ``fixture["body"]``. Asserted here on the file side so the pair is
    # visible in one place.
    assert set(fixture["report_block_required"]) | set(fixture["report_block_optional"]) == set(
        fixture["body"]["devices"][0]
    ), "the filed row carries exactly the declared fields"


def test_the_attention_form_carries_no_completion_field() -> None:
    """A tap on an attention push must not deep-link anywhere (ADR §3.2)."""
    payload = attention_payload(computer="c" * 22, count=1, emit_id="e")
    for name in ("conversation", "completion_token", "kind"):
        assert name not in payload, f"the attention form must not carry {name}"


def test_an_unlisted_field_fails_the_allow_list_in_both_directions() -> None:
    """The guard is bidirectional: an extra field and a missing one both refuse.

    This is the cell that proves the instrument, and it is deliberately written
    against a mutated copy of the filed payload rather than the builder: the
    builder-side mutation (add a field to the module, watch this suite fail) is the
    slice's hand-run evidence, and it is the same check.
    """
    fixture = _fixture("emit-completion.json")
    required = fixture["payload_required"]
    optional = fixture["payload_optional"]

    germane = dict(fixture["payload"], revision=2)
    with pytest.raises(AssertionError, match="unlisted field"):
        _check_shape(germane, required, optional, where="mutated")

    missing = {k: v for k, v in fixture["payload"].items() if k != "kind"}
    with pytest.raises(AssertionError, match="required field 'kind' is missing"):
        _check_shape(missing, required, optional, where="mutated")

    mistyped = dict(fixture["payload"], count=True)
    with pytest.raises(AssertionError, match="expected int"):
        _check_shape(mistyped, required, optional, where="mutated")


@pytest.mark.parametrize("name", ["emit-completion.json", "emit-attention.json"])
def test_no_forbidden_field_appears_anywhere_in_a_filed_shape(name: str) -> None:
    """The deny list is scanned at every depth, including the nested report block.

    Scanned over the shape's LITERALS (``payload`` / ``body`` /
    ``exclude_absent_payload``) and not over the whole file: the file's own metadata
    legitimately names a forbidden field (``payload-forbidden-fields.json`` keys its
    reasons by the field it forbids), and the deny list is about what travels on the
    wire. The two emits are the scope because that is what the file's ``applies_to``
    says — the list response carries a device ``name``, which is a label and not a
    conversation, and its own ``forbidden`` array is asserted in the registry suite.
    """
    fixture = _fixture(name)
    assert set(fixture["provenance"].get("sections", [])) & {"§3.2"}
    forbidden = set(_forbidden())
    assert set(_fixture("payload-forbidden-fields.json")["applies_to"]) >= {name}
    for part in ("payload", "body", "exclude_absent_payload"):
        if part in fixture:
            present = set(_walk_keys(fixture[part])) & forbidden
            assert not present, f"{name}.{part} carries forbidden field(s): {sorted(present)}"


def test_the_two_payloads_carry_nothing_the_cloud_must_not_hold() -> None:
    """The machine's own output, scanned against the deny list and for a raw id."""
    forbidden = set(_forbidden())

    session_id = "session-2f1c9a7b"
    handle = push_handles.handle_for(b"k" * 32, session_id)

    completion = completion_payload(
        computer="c" * 22,
        conversation=handle,
        completion_token="7c1b2a39-4d5e-4f60-8172-93a4b5c6d7e8",
        kind="complete",
        emit_id="aa11bb22cc33dd44ee55ff6677889900",
        count=1,
    )
    bodies = [
        emit_body(completion, DEVICE_ROWS),
        emit_body(
            attention_payload(
                computer="c" * 22,
                count=1,
                emit_id="99887766554433221100ffeeddccbbaa",
                exclude=["d" * 32],
            ),
            DEVICE_ROWS,
        ),
    ]

    for body in bodies:
        assert not (set(_walk_keys(body)) & forbidden), "a forbidden key reached the wire"
        # Values, not the serialized text: a substring test on ``json.dumps`` would
        # also fire on a correct payload whose value merely contained the text, and
        # a raw session id smuggled INSIDE a value still leaks — so the check is a
        # substring test over every string the body carries.
        assert not any(
            session_id in value for value in _walk_values(body)
        ), "a raw session id reached the wire"
        assert "aps" not in _walk_keys(
            body
        ), "the APNs envelope is the cloud's to build, not the machine's"
        # The handle is the wire identity: 22 base64url characters (§4's mint).
        assert re.fullmatch(r"[A-Za-z0-9_-]{22}", completion["conversation"])


def test_the_filed_handle_is_the_adr_shape() -> None:
    """The example handle obeys §4's mint, so a reader copies a legal value."""
    for name in ("emit-completion.json", "registry-list-response.json"):
        fixture = _fixture(name)
        assert fixture["provenance"]["kind"] == "synthetic"
    handle = _fixture("emit-completion.json")["payload"]["conversation"]
    assert re.fullmatch(r"[A-Za-z0-9_-]{22}", handle), handle


def test_the_kind_vocabulary_is_the_stores_and_refuses_the_composers_gaps() -> None:
    """``closed`` is real and keyable; the composer's own kinds are not the wire's."""
    assert set(PUSH_KINDS) == set(_fixture("emit-completion.json")["kinds"])
    # The composer's set is the trap: it carries the gate kinds and not ``closed``.
    message = (
        "NotificationKind does not list 'closed'; a builder keyed to it would drop a real outcome"
    )
    assert "closed" in PUSH_KINDS, message
    kwargs: dict[str, Any] = {
        "computer": "c" * 22,
        "conversation": "h" * 22,
        "completion_token": "t",
        "emit_id": "e",
        "count": 1,
    }
    for kind in PUSH_KINDS:
        assert completion_payload(kind=kind, **kwargs)["kind"] == kind
    for unknown in ("ask", "approval", "queued", "Complete"):
        with pytest.raises(ValueError, match="unknown completion kind"):
            completion_payload(kind=unknown, **kwargs)


def test_the_route_and_header_are_the_frozen_ones() -> None:
    completion = _fixture("emit-completion.json")
    attention = _fixture("emit-attention.json")
    assert EMIT_ROUTE == completion["route"] == attention["route"]
    assert IDEMPOTENCY_HEADER in completion["required_headers"]
    assert IDEMPOTENCY_HEADER in attention["required_headers"]
    assert completion["payload_version"] == attention["payload_version"] == PAYLOAD_VERSION


def test_the_report_block_key_is_single_sourced() -> None:
    """One wire key, one home (review round 1, M3, closed once #1881 merged).

    The block's rows are S4c's, and so is the name of the key they travel under:
    ``push_payload`` imports ``push_credentials.REPORT_DEVICES_FIELD`` rather than
    declaring a second string, so the two cannot drift. Tested rather than trusted
    — the import is one line and a later edit could as easily re-add a literal —
    and the value is asserted against the filed fixtures' own extra key.
    """
    assert push_payload.REPORT_DEVICES_FIELD == push_credentials.REPORT_DEVICES_FIELD
    assert push_payload.REPORT_DEVICES_FIELD == "devices"

    for name in ("emit-completion.json", "emit-attention.json"):
        fixture = _fixture(name)
        extra = set(fixture["body"]) - set(fixture["payload"])
        assert extra == {
            push_payload.REPORT_DEVICES_FIELD
        }, f"{name}: the filed body's only extra key is the report block's"

    body = emit_body(_completion_built(), DEVICE_ROWS)
    assert push_payload.REPORT_DEVICES_FIELD in body
    assert set(body) - set(_completion_built()) == {push_payload.REPORT_DEVICES_FIELD}


def test_the_idempotency_key_vectors_are_reproducible() -> None:
    """§3.4's completion recipe, recomputed from the filed vectors.

    The digests are the one value in ``fixtures/push/`` that is not transcription,
    so they are pinned by recomputation here: a change to the recipe — a separator,
    a reordered part, a different digest — fails this cell rather than quietly
    diverging from the cloud's derivation.
    """
    fixture = _fixture("emit-idempotency-keys.json")
    assert fixture["header"] == IDEMPOTENCY_HEADER
    message = "the concatenation reading is the one thing both sides must agree on"
    assert "no separator" in fixture["recipes"]["completion"].lower(), message

    for vector in fixture["completion_vectors"]:
        key = completion_emit_key(vector["completion_token"], vector["anchor_id"], vector["kind"])
        assert key == vector["key"], f"{vector['kind']}/{vector['anchor_id']}: key drifted"
        assert re.fullmatch(r"[0-9a-f]{64}", key), "the key is lowercase hex sha256"

    keys = [v["key"] for v in fixture["completion_vectors"]]
    assert len(set(keys)) == len(keys), "the filed vectors must not collide"


def test_a_heal_mints_a_new_key_and_the_attention_sequence_never_collides() -> None:
    """§3.4: a heal is a new delivery; an ack is not a heal and is not keyed like one."""
    token, anchor = "9f5d1d6e-6b1a-4c6e-9b3a-7a1c2f3d4e5f", "entry-1043"
    assert completion_emit_key(token, anchor, "complete") != completion_emit_key(
        token, anchor, "interrupted"
    ), "a heal would be swallowed by idempotency if the correction kept the key"
    assert completion_emit_key(token, anchor, "complete") != completion_emit_key(
        token, "completion-9f5d1d6e", "complete"
    ), "the anchor is part of the recipe: a provisional record's supersede must differ"

    for sequence in range(1, 512):
        key = attention_emit_key(sequence)
        assert re.fullmatch(r"attention-\d+", key)
        assert not re.fullmatch(r"[0-9a-f]{64}", key), "the key spaces are disjoint by shape"
    assert attention_emit_key(1) != attention_emit_key(2)

    fixture = _fixture("emit-idempotency-keys.json")
    assert [v["key"] for v in fixture["attention_vectors"]] == [
        attention_emit_key(v["sequence"]) for v in fixture["attention_vectors"]
    ]
