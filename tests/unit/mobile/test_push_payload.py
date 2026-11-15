"""The push payload and its two keys, asserted against the frozen fixture contract.

Push/ack-sync S3 (ADR 0006 §3.2 payload, §3.4 keys — pinned at
``damianvtran/local-operator-mobile`` @ ``b03aeb15``). ``tests/fixtures/push/`` is the
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

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest

from local_operator.mobile import push_credentials, push_handles, push_payload
from local_operator.mobile.push_payload import (
    ALERT_BODY_FIELD,
    ALERT_FIELD,
    ALERT_TITLE_FIELD,
    EMIT_ROUTE,
    IDEMPOTENCY_HEADER,
    PAYLOAD_VERSION,
    PUSH_KINDS,
    SOUND_DEFAULT,
    SOUND_FIELD,
    SOUND_NONE,
    TYPE_ATTENTION,
    TYPE_COMPLETION,
    TYPE_DIGEST,
    attention_emit_key,
    attention_payload,
    completion_alert,
    completion_emit_key,
    completion_payload,
    digest_alert,
    digest_emit_key,
    digest_payload,
    emit_body,
    push_sound,
)
from local_operator.notifications import compose
from local_operator.notifications.sound import DEFAULT_SOUND_ID, sound_ids
from local_operator.tui.notify import (
    APP_NAME,
    BACKGROUND_FALLBACK_TITLE,
    BODIES,
    BODY_COMPLETE,
    CONTEXT_MIXED,
    CONTEXTS,
    digest_subtitle,
)

#: ``tests/unit/mobile/<this file>`` -> the repo root, where the shared
#: contract tree lives. Derived from the file rather than the cwd so the suite
#: reads the fixtures of the checkout it is running in.
FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "push"

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


@pytest.fixture(autouse=True)
def _names_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every cell runs with ``display.notification_session_name`` OFF by default.

    Since 2026-11-14 the completion alert is the desktop composer's output, which
    reads that switch; the filed literals are the PRIVATE form (app name + house
    sentence), the one form that is a fixed string. Pinning the switch here keeps
    the goldens independent of the developer's own settings. Cells about the
    named form opt back in with :func:`_names_on`.
    """
    monkeypatch.setattr("local_operator.tui.notify.settings_get", lambda key, default=None: False)


def _names_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("local_operator.tui.notify.settings_get", lambda key, default=None: True)


def _session(root: Path, *, assistant: str = "", title: str = "") -> Path:
    """A session directory in the shapes ``resume.py`` reads (as test_compose's).

    Real files rather than a stubbed composer, so these cells prove the push says
    what the desktop composer says for the SAME directory.
    """
    session = root / "sessions" / "abcdef123456"
    session.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = [
        {
            "id": "user-1",
            "ts": 1.0,
            "type": "message",
            "payload": {"kind": "message", "role": "user", "content": [{"text": "do the thing"}]},
        }
    ]
    if assistant:
        rows.append(
            {
                "id": "assistant-1",
                "ts": 2.0,
                "type": "message",
                "payload": {
                    "kind": "message",
                    "role": "assistant",
                    "content": [{"text": assistant}],
                },
            }
        )
    (session / "transcript.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    if title:
        (session / "title.json").write_text(
            json.dumps({"text": title, "user_set": True, "names": [title]}), encoding="utf-8"
        )
    return session


def _incident(session: Path, raw: str) -> None:
    """Append the ``session_incident`` record a classified failure journals.

    The shape ``incidents.py`` writes (as test_compose's helper): compose reads
    ``details.raw`` as an ``error`` banner's failure line, which can quote a
    prompt fragment, a path or an account id — the riskiest text a push can
    carry (security review r1, S-4).
    """
    row = {
        "id": "incident-1",
        "ts": 3.0,
        "type": "message",
        "payload": {
            "kind": "custom",
            "custom_type": "session_incident",
            "details": {"text": f"rendered: {raw}", "raw": raw},
        },
    }
    with (session / "transcript.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


#: Every string the PRIVATE alert may carry: the house vocabulary, nothing else.
def _house_strings() -> set[str]:
    return (
        set(BODIES.values())
        | set(CONTEXTS.values())
        | {CONTEXT_MIXED, BACKGROUND_FALLBACK_TITLE, APP_NAME}
    )


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


def _without_alert(value: dict[str, Any]) -> dict[str, Any]:
    """A body with the machine's alert object removed, for the deny-list scan.

    The deny list is scanned by KEY NAME and forbids ``title``/``body`` as payload
    fields. Since the lane's ruling the machine composes the user-visible alert,
    and that object is the one place those two keys legitimately appear. Its
    CONTENT stays checked: :func:`_check_alert` asserts a filed literal is the
    private form (app name + house sentence), so removing it from the name scan
    does not loosen what the fixtures promise when the privacy switch is off.
    """
    return {key: item for key, item in value.items() if key != ALERT_FIELD}


def _check_alert(value: dict[str, Any], fixture: dict[str, Any], *, where: str) -> None:
    """The alert's own allow-list, and the two facts about its text."""
    alert = value[ALERT_FIELD]
    _check_shape(alert, fixture["alert_required"], {}, where=f"{where}.{ALERT_FIELD}")
    assert alert[ALERT_TITLE_FIELD] == APP_NAME, f"{where}: the private title is the app name"
    assert alert[ALERT_BODY_FIELD] in _house_strings(), f"{where}: the private body is house text"


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
    "sound": "soft-pulse",
}

#: The attention emit's input: one field fewer (no ``conversation``, no
#: ``completion_token``, no ``kind``) and an ``exclude`` naming one device.
ATTENTION_INPUT: dict[str, Any] = {
    "computer": "VZ8kQ1mT4pR7sW2xY6bN9c",
    "count": 1,
    "emit_id": "1a2b3c4d5e6f708192a3b4c5d6e7f809",
}

#: The digest emit's input: the same three fields as the attention form, and
#: deliberately its OWN literal rather than a reuse of the one above — the two
#: forms share a field set and nothing else, and a test that shared their input
#: could not tell a copy-paste from a shape.
#:
#: ``kinds`` is the batch's member kinds, which is what the alert's leading
#: phrase is composed from (the wire carries no per-member kind: the batch is a
#: set). A uniform batch of four, so the filed alert reads "Complete · 4 …".
DIGEST_INPUT: dict[str, Any] = {
    "computer": "VZ8kQ1mT4pR7sW2xY6bN9c",
    "count": 4,
    "emit_id": "5c7e9a1b3d5f708192a3b4c5d6e7f809",
    "kinds": ["complete", "complete", "complete", "complete"],
    "sound": "soft-pulse",
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
    _check_alert(payload, fixture, where="emit-completion")
    assert payload[ALERT_FIELD] == completion_alert(COMPLETION_INPUT["kind"])

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

    # The attention form is the SILENT one, so it carries no alert at all — and
    # the absence is declared (absent_fields) rather than left to the equality
    # above, so a later edit that "tidy-ups" an alert onto it fails here.
    # ``sound_custom`` rides the visible forms only (ADR amendment 2026-11-04),
    # so the silent form declares it absent alongside ``sound``.
    assert fixture["absent_fields"] == [ALERT_FIELD, SOUND_FIELD, "sound_custom"]
    for key in ("payload", "exclude_absent_payload", "body"):
        for absent in fixture["absent_fields"]:
            assert absent not in fixture[key], f"the attention form must not carry {key}.{absent}"
            assert absent not in with_exclude
            assert absent not in without

    body = emit_body(with_exclude, DEVICE_ROWS)
    _check_block(body["devices"], fixture, where="emit-attention.body.devices")
    _check_block(fixture["body"]["devices"], fixture, where="emit-attention.filed.devices")
    assert body == fixture["body"], "the emit body is the payload plus the report block"


def test_the_digest_payload_equals_the_filed_literals_with_and_without_exclude() -> None:
    """The third form, against its own filed literal in both variants.

    The digest is a VISIBLE alert (the lane's ruling of 2026-10-01): its type is
    what tells the cloud to compose a banner from it, so this cell pins the type
    and the field set — a digest that drifted into the attention form's shape or
    its type would arrive with no banner at all.
    """
    fixture = _fixture("emit-digest.json")
    payload = fixture["payload"]

    with_exclude = digest_payload(**DIGEST_INPUT, exclude=[DEVICE_ROWS[0]["device_id"]])
    _check_shape(
        with_exclude,
        fixture["payload_required"],
        fixture["payload_optional"],
        where="emit-digest.payload",
    )
    assert with_exclude == payload
    assert with_exclude["type"] == TYPE_DIGEST

    without = digest_payload(**DIGEST_INPUT)
    assert without == fixture["exclude_absent_payload"]
    assert "exclude" not in without
    _check_alert(with_exclude, fixture, where="emit-digest")
    assert with_exclude[ALERT_FIELD] == digest_alert(DIGEST_INPUT["kinds"])

    body = emit_body(with_exclude, DEVICE_ROWS)
    _check_block(body["devices"], fixture, where="emit-digest.body.devices")
    _check_block(fixture["body"]["devices"], fixture, where="emit-digest.filed.devices")
    assert body == fixture["body"], "the emit body is the payload plus the report block"
    assert set(body) - set(with_exclude) == {push_payload.REPORT_DEVICES_FIELD}

    # The three forms are told apart by the TYPE alone, so nothing a tap could
    # resolve to may appear on this one either (ADR §3.2).
    for name in ("conversation", "completion_token", "kind"):
        assert name not in with_exclude, f"the digest form must not carry {name}"
    assert set(with_exclude) - {"exclude"} == set(fixture["payload_required"])


def test_the_digest_allow_list_bites_in_both_directions() -> None:
    """The digest's declaration refuses an extra field and a missing one.

    Written as the negative pair so the cell fails if a later edit loosens the
    builder's field set or the fixture's declaration: a digest that grew a field
    the cloud validates with ``extra="forbid"`` would be refused on the wire, and
    one that lost a required field would be refused for the other reason.
    """
    fixture = _fixture("emit-digest.json")
    required = fixture["payload_required"]
    optional = fixture["payload_optional"]
    built = digest_payload(**DIGEST_INPUT)

    assert set(built) <= set(required) | set(optional)
    assert set(required) <= set(
        fixture["exclude_absent_payload"]
    ), "the filed literal carries every required field"

    with pytest.raises(AssertionError, match="unlisted field"):
        _check_shape({**built, "thread": "x"}, required, optional, where="mutated")
    with pytest.raises(AssertionError, match="required field 'emit_id' is missing"):
        _check_shape(
            {k: v for k, v in built.items() if k != "emit_id"},
            required,
            optional,
            where="mutated",
        )


def test_the_alert_allow_list_bites_in_both_directions() -> None:
    """The alert's own declaration, with the same negative pair as the payload's.

    Both emits that carry an alert are checked, because both builders compose it
    through the same two helpers: a field added to :func:`completion_alert` or
    :func:`digest_alert` must be declared in the fixture before the cloud will
    accept it, and one dropped must be dropped there too.
    """
    for name, alert in (
        ("emit-completion.json", completion_alert("complete")),
        ("emit-digest.json", digest_alert(["complete", "complete"])),
    ):
        fixture = _fixture(name)
        required = fixture["alert_required"]
        _check_shape(alert, required, {}, where=f"{name}.alert")
        assert set(alert) == set(required), f"{name}: the alert's fields are the declared ones"

        with pytest.raises(AssertionError, match="unlisted field"):
            _check_shape({**alert, "subtitle": "x"}, required, {}, where="mutated")
        with pytest.raises(AssertionError, match="required field 'body' is missing"):
            _check_shape({ALERT_TITLE_FIELD: APP_NAME}, required, {}, where="mutated")


def test_a_named_completion_is_the_desktop_composers_title_and_preview(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The operator's request (2026-11-14): say which session, preview the turn.

    Asserted as EQUALITY with what the desktop composer returns for the same
    directory, so the phone and the desktop cannot drift into two formatters.
    """
    _names_on(monkeypatch)
    session = _session(tmp_path, assistant="Migrated 14 tables.", title="Quota reporting fix")
    composed = compose("complete", session_dir=session)

    alert = completion_alert("complete", session_dir=session, previews=True)

    assert alert == {ALERT_TITLE_FIELD: composed.title, ALERT_BODY_FIELD: composed.body}
    assert alert == {
        ALERT_TITLE_FIELD: "Quota reporting fix",
        ALERT_BODY_FIELD: "Migrated 14 tables.",
    }
    assert _fixture("emit-completion.json")["alert_named_example"] == alert
    # No count phrase any more — the count stays a payload field only.
    assert "need" not in alert[ALERT_BODY_FIELD]


#: Text the fixture session carries in every place a banner could quote it:
#: the name, the last reply, and an ``error``'s incident ``details.raw``.
_LEAK_NAME = "Project Falcon"
_LEAK_REPLY = "Secret plan: rotate the keys"
_LEAK_FAILURE = "quota exhausted for acct-7731 while reading /Users/alice/payroll.xlsx"


def _leaky_session(root: Path) -> Path:
    session = _session(root, assistant=_LEAK_REPLY, title=_LEAK_NAME)
    _incident(session, _LEAK_FAILURE)
    return session


def _emit_bytes(session: Path, kind: str, *, previews: bool) -> tuple[dict[str, Any], bytes]:
    payload = completion_payload(
        **{**COMPLETION_INPUT, "kind": kind, "session_dir": session, "previews": previews}
    )
    raw = json.dumps(emit_body(payload, DEVICE_ROWS), ensure_ascii=False).encode("utf-8")
    return payload, raw


def _assert_generic(payload: dict[str, Any], raw: bytes, kind: str) -> None:
    """The P2 guard, re-scoped (security review r1, S-1/S-4): GENERIC unless allowed.

    The alert is the house constants for the kind — the product name over the
    house sentence, no count line — and no name, reply snippet, failure text or
    session id appears anywhere in the serialised emit.
    """
    assert payload[ALERT_FIELD] == {
        ALERT_TITLE_FIELD: APP_NAME,
        ALERT_BODY_FIELD: BODIES.get(kind, BODY_COMPLETE),
    }
    leaks = (_LEAK_NAME, _LEAK_REPLY, "rotate the keys", _LEAK_FAILURE, "acct-7731", "abcdef123456")
    for leaked in leaks:
        assert leaked.encode("utf-8") not in raw, (kind, leaked)


@pytest.mark.parametrize("kind", PUSH_KINDS)
def test_with_previews_off_no_session_text_reaches_the_emit_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """A phone with previews off: generic, even with the desktop switch ON.

    ``previews=False`` is what the worker passes when any live phone resolves
    ``preferences.previews`` to false; the desktop flag cannot override it.
    """
    _names_on(monkeypatch)
    payload, raw = _emit_bytes(_leaky_session(tmp_path), kind, previews=False)
    _assert_generic(payload, raw, kind)


@pytest.mark.parametrize("kind", PUSH_KINDS)
def test_with_names_off_no_session_text_reaches_the_emit_bytes(tmp_path: Path, kind: str) -> None:
    """The desktop switch off (the autouse default here): generic, even with previews on.

    Every kind, and ``error`` in particular: its failure line comes from the
    incident's ``details.raw`` and is gated by the same desktop switch.
    """
    session = _leaky_session(tmp_path)
    payload, raw = _emit_bytes(session, kind, previews=True)
    _assert_generic(payload, raw, kind)
    composed = compose(kind, session_dir=session)  # type: ignore[arg-type]
    assert payload[ALERT_FIELD] == {
        ALERT_TITLE_FIELD: composed.title,
        ALERT_BODY_FIELD: composed.body,
    }
    if kind == "complete":
        assert _fixture("emit-completion.json")["alert_private"] == payload[ALERT_FIELD]


def test_previews_default_to_generic_for_a_caller_that_does_not_claim_consent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``previews`` defaults False on both builders, so forgetting it cannot leak."""
    _names_on(monkeypatch)
    session = _leaky_session(tmp_path)
    payload = completion_payload(**{**COMPLETION_INPUT, "session_dir": session})
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    _assert_generic(payload, raw, COMPLETION_INPUT["kind"])
    assert completion_alert("complete", session_dir=session) == {
        ALERT_TITLE_FIELD: APP_NAME,
        ALERT_BODY_FIELD: BODY_COMPLETE,
    }


def test_with_both_switches_on_an_error_carries_its_failure_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The positive control for the matrix above: the leak strings DO reach the
    bytes when both consents hold, so their absence there is the gate's doing."""
    _names_on(monkeypatch)
    _payload, raw = _emit_bytes(_leaky_session(tmp_path), "error", previews=True)
    assert _LEAK_NAME.encode() in raw
    assert _LEAK_FAILURE.encode() in raw


def test_the_generic_alert_is_the_house_text_and_nothing_else() -> None:
    """The house-constant cell, restored for the generic form (S-1).

    Every string a generic alert carries is the product's own: the app name and
    the house sentence for the outcome. No count line (operator, 2026-11-14) —
    the count rides the payload's ``count`` field only.
    """
    for kind in PUSH_KINDS:
        alert = completion_alert(kind)
        assert alert == {
            ALERT_TITLE_FIELD: APP_NAME,
            ALERT_BODY_FIELD: BODIES.get(kind, BODY_COMPLETE),
        }
        assert set(alert.values()) <= _house_strings()
        assert "need" not in alert[ALERT_BODY_FIELD]
    assert completion_alert("not-a-kind")[ALERT_BODY_FIELD] == BODY_COMPLETE
    payload = completion_payload(**{**COMPLETION_INPUT, "count": None})
    assert payload["count"] == 0, "an unknown count travels as 0 (wire_count)"


#: A vendor-prefixed OpenAI-style key and a bearer token, assembled at runtime so
#: this file does not itself carry a credential-shaped literal.
_FAKE_KEY = "sk-" + "proj-" + "Z9" * 24
_FAKE_BEARER = "eyJhbGciOiJIUzI1NiJ9." + "a1B2c3D4" * 5


def test_credential_shaped_text_is_scrubbed_before_it_leaves_the_machine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S-2: a reply quoting a key or a Bearer header never reaches the emit bytes."""
    _names_on(monkeypatch)
    session = _session(
        tmp_path,
        assistant=f"Set OPENAI_API_KEY={_FAKE_KEY} then send Authorization: Bearer {_FAKE_BEARER}",
        title=f"Key {_FAKE_KEY}",
    )
    payload, raw = _emit_bytes(session, "complete", previews=True)
    assert _FAKE_KEY.encode() not in raw
    assert _FAKE_BEARER.encode() not in raw
    assert "[redacted]" in payload[ALERT_FIELD][ALERT_BODY_FIELD]
    # The failure line goes through the same pass.
    failing = _session(tmp_path / "e", assistant="x", title="Deploy")
    _incident(failing, f"401 from provider: Authorization: Bearer {_FAKE_BEARER}")
    _payload, raw = _emit_bytes(failing, "error", previews=True)
    assert _FAKE_BEARER.encode() not in raw
    # Scrubbed BEFORE the empty-check: a string that is only a secret still
    # yields a non-empty alert string.
    assert push_payload._bounded_alert_text(_FAKE_KEY, "house")


def test_bidi_and_zero_width_characters_are_stripped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S-3: no reordering/invisible format characters on a lock screen."""
    spoofers = "\u200b\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069\ufeff"
    cleaned = push_payload._bounded_alert_text(f"Title \u202eevil\u202c{spoofers}x", "house")
    assert cleaned == "Title evilx"
    assert not set(cleaned) & set(spoofers)
    # ZWNJ (Urdu/Hindi orthography) and ZWJ (emoji sequences) are kept.
    assert push_payload._bounded_alert_text("a\u200cb\u200dc", "house") == "a\u200cb\u200dc"
    # Only spoofers: falls back rather than sending an empty string.
    assert push_payload._bounded_alert_text(spoofers, "house") == "house"
    _names_on(monkeypatch)
    session = _session(tmp_path, assistant="done \u202egnp.exe\u202c", title="Name\u200b\u202e")
    alert = completion_alert("complete", session_dir=session, previews=True)
    assert not set("".join(alert.values())) & set(spoofers)


def test_a_long_final_message_is_clamped_well_formed_and_under_the_cloud_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 2,000-char reply: the body stays inside validateAlert's 1-200 code points."""
    _names_on(monkeypatch)
    long_reply = ("word " * 400).strip()
    assert len(long_reply) >= 1999
    session = _session(tmp_path, assistant=long_reply, title="T" * 500)

    alert = completion_alert("complete", session_dir=session, previews=True)

    for field in (ALERT_TITLE_FIELD, ALERT_BODY_FIELD):
        text = alert[field]
        assert 1 <= len(text) <= push_payload.ALERT_TEXT_MAX_CHARS, (field, len(text))
        assert text == text.strip()
        assert not any(ord(ch) < 32 for ch in text)
    assert alert[ALERT_BODY_FIELD].startswith("word word")


def test_the_clamp_cuts_on_a_grapheme_boundary_with_an_ellipsis() -> None:
    """Above 200 code points: cut whole graphemes, end with one ellipsis.

    Compose's own budgets (80/120) keep its output under 200 today, so the clamp
    is exercised directly: a family emoji (5 code points, one grapheme) sits
    across the cut and must survive whole or not at all.
    """
    family = "\U0001f468\u200d\U0001f469\u200d\U0001f467"
    text = "a" * 197 + family + "tail"
    clamped = push_payload._bounded_alert_text(text, "fallback")

    assert len(clamped) <= push_payload.ALERT_TEXT_MAX_CHARS
    assert clamped.endswith("\u2026")
    assert clamped == "a" * 197 + "\u2026", "the family emoji is dropped whole, never split"
    assert not clamped[:-1].endswith("\u200d")
    # Within the bound: untouched.
    assert push_payload._bounded_alert_text("short", "fallback") == "short"


@pytest.mark.parametrize("preview", ["", "   ", "\x00\x01"])
def test_an_empty_or_missing_preview_falls_back_and_is_never_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preview: str
) -> None:
    """Empty text is refused by the cloud, so every string falls back to house text."""
    _names_on(monkeypatch)
    session = _session(tmp_path, assistant=preview)
    alert = completion_alert("complete", session_dir=session, previews=True)
    assert alert[ALERT_BODY_FIELD] == BODY_COMPLETE
    assert alert[ALERT_TITLE_FIELD] == BACKGROUND_FALLBACK_TITLE

    # No directory at all: compose's nameless fallback, still non-empty.
    missing = completion_alert(
        "complete", session_dir=tmp_path / "sessions" / "gone", previews=True
    )
    assert all(missing.values())
    # And the clamp itself never returns "" whatever it is handed.
    assert push_payload._bounded_alert_text(preview, "") == APP_NAME
    assert push_payload._bounded_alert_text(preview, "house") == "house"


def test_a_compose_failure_degrades_to_the_house_alert(monkeypatch: pytest.MonkeyPatch) -> None:
    """compose promises never to raise; a push must not fail even if it does."""

    def boom(*_a: object, **_k: object) -> None:
        raise RuntimeError("compose exploded")

    monkeypatch.setattr("local_operator.notifications.compose", boom)
    assert completion_alert("error", previews=True) == {
        ALERT_TITLE_FIELD: APP_NAME,
        ALERT_BODY_FIELD: BODIES["error"],
    }


def test_other_kinds_take_whatever_compose_produces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """error / interrupted / closed / retired: compose's own text, unedited."""
    _names_on(monkeypatch)
    session = _session(tmp_path, assistant="All 412 tests pass.", title="CI fix")
    for kind in PUSH_KINDS:
        composed = compose(kind, session_dir=session)  # type: ignore[arg-type]
        alert = completion_alert(kind, session_dir=session, previews=True)
        assert alert == {ALERT_TITLE_FIELD: composed.title, ALERT_BODY_FIELD: composed.body}
        if kind != "complete":
            # Snippet is complete-only (compose's rule): an errored turn's last
            # line must not read as a success on a lock screen.
            assert "412" not in alert[ALERT_BODY_FIELD]
    # An unknown kind takes the composer's own default rather than inventing one.
    assert completion_alert("not-a-kind", previews=True)[ALERT_BODY_FIELD] == BODY_COMPLETE


def test_a_worst_case_emit_stays_well_under_the_apns_4kb_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both strings at their 200-code-point cap, in 4-byte characters, plus the block.

    APNs refuses a payload over 4,096 bytes; the cloud wraps this alert in its
    envelope, so the machine's whole emit must leave generous headroom.
    """
    _names_on(monkeypatch)
    session = _session(tmp_path, assistant="\U0001f600" * 2000, title="\U0001f600" * 2000)
    payload = completion_payload(
        **{
            **COMPLETION_INPUT,
            "session_dir": session,
            "previews": True,
            "sound_custom": "c_" + "0" * 16,
        }
    )
    size = len(json.dumps(emit_body(payload, DEVICE_ROWS), ensure_ascii=False).encode("utf-8"))
    assert size < 2048, size


def test_the_digest_alert_says_the_sets_state_and_never_a_fixed_one() -> None:
    """A coalesced batch speaks for a SET, so it names no session and prints no count."""
    uniform = digest_alert(["error"] * 3)
    assert uniform == {
        ALERT_TITLE_FIELD: APP_NAME,
        ALERT_BODY_FIELD: digest_subtitle(["error"] * 3),
    }

    mixed = digest_alert(["complete", "error"])[ALERT_BODY_FIELD]
    assert mixed == CONTEXT_MIXED

    # Kinds that could not be read make NO outcome claim (review round 2, R2-4):
    # "A session finished" is true of any digest, "Task complete" is not.
    unreadable = digest_alert([])
    assert unreadable[ALERT_BODY_FIELD] == BACKGROUND_FALLBACK_TITLE
    assert not unreadable[ALERT_BODY_FIELD].startswith(BODY_COMPLETE)
    assert _fixture("emit-digest.json")["alert_unreadable"] == unreadable
    for alert in (uniform, unreadable):
        assert "need" not in alert[ALERT_BODY_FIELD]


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
    # ``sound_custom_ready`` is per-EMIT, not credential, state: it is on a row
    # only when the emit names a custom sound, so the built-in literal (``body``)
    # carries the credential fields and the custom literal adds exactly the bit.
    assert set(fixture["report_block_required"]) | set(fixture["report_block_optional"]) == set(
        fixture["body"]["devices"][0]
    ) | {push_credentials.REPORT_SOUND_READY_FIELD}, "the filed row carries the declared fields"


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


@pytest.mark.parametrize(
    "name", ["emit-completion.json", "emit-attention.json", "emit-digest.json"]
)
def test_no_forbidden_field_appears_anywhere_in_a_filed_shape(name: str) -> None:
    """The deny list is scanned at every depth, including the nested report block.

    scanned over the shape's LITERALS (``payload`` / ``body`` /
    ``exclude_absent_payload``) and not over the whole file: the file's own metadata
    legitimately names a forbidden field (``payload-forbidden-fields.json`` keys its
    reasons by the field it forbids), and the deny list is about what travels on the
    wire. The three emits are the scope because that is what the file's ``applies_to``
    says — the list response carries a device ``name``, which is a label and not a
    conversation, and its own ``forbidden`` array is asserted in the registry suite.
    """
    fixture = _fixture(name)
    assert set(fixture["provenance"].get("sections", [])) & {"§3.2"}
    forbidden = set(_forbidden())
    assert set(_fixture("payload-forbidden-fields.json")["applies_to"]) >= {name}
    # The ``alert`` object is scanned for its CONTENT rather than by key name: the
    # machine composes the user-visible text (the lane's ruling of 2026-10-01), and
    # ``title``/``body`` are the two keys it is allowed to hold there. _check_alert
    # is the cell that keeps the exception narrow.
    for part in ("payload", "body", "exclude_absent_payload"):
        if part in fixture:
            present = set(_walk_keys(_without_alert(fixture[part]))) & forbidden
            assert not present, f"{name}.{part} carries forbidden field(s): {sorted(present)}"
            if ALERT_FIELD in fixture[part]:
                _check_alert(fixture[part], fixture, where=f"{name}.{part}")


def test_the_three_payloads_carry_nothing_the_cloud_must_not_hold() -> None:
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
        sound="soft-pulse",
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
        emit_body(
            digest_payload(
                computer="c" * 22,
                count=3,
                emit_id="00112233445566778899aabbccddeeff",
                kinds=["complete", "complete", "complete"],
                sound="soft-pulse",
                exclude=["d" * 32],
            ),
            DEVICE_ROWS,
        ),
    ]

    for body in bodies:
        # The alert is the one object allowed to carry ``title``/``body``, and its
        # text is asserted to be the house constants in the cells above; the name
        # scan therefore runs over the body without it.
        assert not (
            set(_walk_keys(_without_alert(body))) & forbidden
        ), "a forbidden key reached the wire"
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
        "sound": "soft-pulse",
    }
    for kind in PUSH_KINDS:
        assert completion_payload(kind=kind, **kwargs)["kind"] == kind
    for unknown in ("ask", "approval", "queued", "Complete"):
        with pytest.raises(ValueError, match="unknown completion kind"):
            completion_payload(kind=unknown, **kwargs)


def test_the_route_and_header_are_the_frozen_ones() -> None:
    completion = _fixture("emit-completion.json")
    attention = _fixture("emit-attention.json")
    digest = _fixture("emit-digest.json")
    assert EMIT_ROUTE == completion["route"] == attention["route"] == digest["route"]
    for fixture in (completion, attention, digest):
        assert IDEMPOTENCY_HEADER in fixture["required_headers"]
    assert (
        completion["payload_version"]
        == attention["payload_version"]
        == digest["payload_version"]
        == PAYLOAD_VERSION
    )
    assert {
        completion["payload"]["type"],
        attention["payload"]["type"],
        digest["payload"]["type"],
    } == {
        TYPE_COMPLETION,
        TYPE_ATTENTION,
        TYPE_DIGEST,
    }, "one route, three types, and the type is the only discriminator"


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

    for name in ("emit-completion.json", "emit-attention.json", "emit-digest.json"):
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

    The digests are the one value in ``tests/fixtures/push/`` that is not transcription,
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

    # The digest recipe (the lane's ruling, 2026-10-01) is recomputed the same
    # way: the id is what the machine mints at window-close, so the vector is the
    # only place the derivation is written down outside the builder.
    assert "no separator" in fixture["recipes"]["digest"].lower(), message
    digest_keys: list[str] = []
    for vector in fixture["digest_vectors"]:
        key = digest_emit_key(vector["emit_id"], vector["computer"])
        assert key == vector["key"], f"digest {vector['emit_id']}: key drifted"
        assert re.fullmatch(r"[0-9a-f]{64}", key), "the key is lowercase hex sha256"
        digest_keys.append(key)
    assert len(set(digest_keys)) == len(digest_keys), "the filed digest vectors must not collide"
    assert digest_keys[0] != digest_keys[1], (
        "the COMPUTER is part of the recipe: the same emit id on another machine is a "
        "different key"
    )


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


# --- data.sound (notification-sounds design §3.6, PR C3) -----------------------


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        # No file, no section, an empty section: every default, so Soft pulse.
        (None, "soft-pulse"),
        ({}, "soft-pulse"),
        ({"notifications": {"sound": {}}}, "soft-pulse"),
        ({"notifications": {"sound": {"id": "felt-marimba"}}}, "felt-marimba"),
        ({"notifications": {"sound": {"enabled": True, "id": "tape-warble"}}}, "tape-warble"),
        # ``system`` is "the platform's own sound", which on the phone is its default.
        ({"notifications": {"sound": {"id": "system"}}}, SOUND_DEFAULT),
        # Off wins over any id, ``system`` included.
        ({"notifications": {"sound": {"enabled": False, "id": "felt-marimba"}}}, SOUND_NONE),
        ({"notifications": {"sound": {"enabled": False, "id": "system"}}}, SOUND_NONE),
        # A custom sound (C2) sends the built-in fallback in data.sound (its file is
        # named by the reserved sound_custom, from C5), as does a typo'd id.
        ({"notifications": {"sound": {"id": "custom-3f9a2c"}}}, "soft-pulse"),
        ({"notifications": {"sound": {"id": "soft-puls"}}}, "soft-pulse"),
        # A hand-written off reads as off, exactly as the settings page paints
        # it (``strict_bool`` via ``sound.configured``); garbage keeps the default.
        ({"notifications": {"sound": {"enabled": "off", "id": "hum-swell"}}}, SOUND_NONE),
        ({"notifications": {"sound": {"enabled": "maybe", "id": "hum-swell"}}}, "hum-swell"),
        # The computer's volume does not travel: 0 mutes local speakers only.
        ({"notifications": {"sound": {"id": "hum-swell", "volume": 0}}}, "hum-swell"),
        # A malformed section is the defaults, never an exception in the worker.
        ({"notifications": "loud"}, "soft-pulse"),
        ({"notifications": {"sound": ["soft-pulse"]}}, "soft-pulse"),
    ],
)
def test_push_sound_maps_the_computers_setting_to_what_the_phone_can_play(
    values: dict[str, Any] | None, expected: str
) -> None:
    assert push_sound(values) == expected


def test_every_built_in_sound_travels_by_its_own_id_and_the_fixture_lists_them() -> None:
    """Each shipped sound reaches the phone unchanged, and the filed value list is
    the catalog's: a sound added to core without a phone file and channel would
    be a push the app cannot play (mobile M1 bundles exactly this list)."""
    built_ins = [sound_id for sound_id in sound_ids() if sound_id != "system"]
    assert len(built_ins) == 10
    assert DEFAULT_SOUND_ID == "soft-pulse"
    for sound_id in built_ins:
        assert push_sound({"notifications": {"sound": {"id": sound_id}}}) == sound_id
    for name in ("emit-completion.json", "emit-digest.json"):
        fixture = _fixture(name)
        assert fixture["sound_values"]["builtin"] == built_ins, name
        assert fixture["payload_required"][SOUND_FIELD] == "str", name
        assert {fixture["payload"][SOUND_FIELD], fixture["body"][SOUND_FIELD]} == {"soft-pulse"}


def test_both_visible_builders_carry_the_sound_and_refuse_one_the_phone_lacks() -> None:
    for sound in ("soft-pulse", "bubble-pulse", SOUND_DEFAULT, SOUND_NONE):
        assert completion_payload(**{**COMPLETION_INPUT, "sound": sound})[SOUND_FIELD] == sound
        assert digest_payload(**{**DIGEST_INPUT, "sound": sound})[SOUND_FIELD] == sound
    # The builders validate rather than map: a custom id that reached them
    # bypassed :func:`push_sound`, and the cloud has no file or channel for it.
    for bad in ("custom-3f9a2c", "system", "", "Soft-Pulse"):
        with pytest.raises(ValueError, match="unknown push sound"):
            completion_payload(**{**COMPLETION_INPUT, "sound": bad})
        with pytest.raises(ValueError, match="unknown push sound"):
            digest_payload(**{**DIGEST_INPUT, "sound": bad})


# --- sound_custom / sound_custom_ready (ADR 2026-11-04, sent from C5) ---------

#: A custom selection's stem and two report rows: one device that reported the
#: file cached, one that did not. The test's own input, as with DEVICE_ROWS.
CUSTOM_STEM = "c_3b9f2e7a51c04d86"
CUSTOM_DEVICE_ROWS: list[dict[str, Any]] = [
    {**DEVICE_ROWS[0], "sound_custom_ready": True},
    {
        "device_id": "7a2e4c6b8d0f12345678a9b0c1d2e3f4",
        "credential_live": True,
        "credential_expires_at": 1791592000,
        "sound_custom_ready": False,
    },
]


@pytest.mark.parametrize("name", ["emit-completion.json", "emit-digest.json"])
def test_a_custom_selection_equals_the_filed_custom_literals(name: str) -> None:
    """``sound_custom`` beside the soft-pulse fallback, and readiness per device.

    The visible forms carry the optional field only when the caller passes it,
    so the built-in literal stays the pre-C5 shape (the equality cells above)
    and the custom literal is this one. Both rows of the block pass the block's
    own allow-list, which now declares ``sound_custom_ready``.
    """
    fixture = _fixture(name)
    if name == "emit-completion.json":
        built = completion_payload(**COMPLETION_INPUT, sound_custom=CUSTOM_STEM)
    else:
        built = digest_payload(
            **DIGEST_INPUT, exclude=["3f1c9a7b2d4e506182a3b4c5d6e7f809"], sound_custom=CUSTOM_STEM
        )
    _check_shape(
        built, fixture["payload_required"], fixture["payload_optional"], where=f"{name}.custom"
    )
    assert built == fixture["custom_payload"]
    assert built[SOUND_FIELD] == DEFAULT_SOUND_ID, "the fallback every not-ready device plays"
    assert built[push_payload.SOUND_CUSTOM_FIELD] == CUSTOM_STEM
    body = emit_body(built, CUSTOM_DEVICE_ROWS)
    _check_block(body["devices"], fixture, where=f"{name}.custom_body.devices")
    _check_block(fixture["custom_body"]["devices"], fixture, where=f"{name}.filed.custom")
    assert body == fixture["custom_body"]
    assert "reserved_not_sent" not in fixture, "C5 sends both fields; nothing is reserved now"


def test_the_custom_sound_field_is_declared_where_it_may_travel_and_nowhere_else() -> None:
    for name in ("emit-completion.json", "emit-digest.json", "emit-attention.json"):
        fixture = _fixture(name)
        assert fixture["report_block_optional"]["sound_custom_ready"] == "bool"
        if name == "emit-attention.json":
            assert "sound_custom" not in fixture["payload_optional"]
            assert "sound_custom" in fixture["absent_fields"]
        else:
            assert fixture["payload_optional"]["sound_custom"] == "str"
            assert "sound_custom" not in fixture["payload_required"]
            # The built-in literal carries neither field.
            assert "sound_custom" not in fixture["body"]
            assert all("sound_custom_ready" not in row for row in fixture["body"]["devices"])


@pytest.mark.parametrize(
    "bad", ["c-3b9f2e7a51c04d86", "c_3B9F2E7A51C04D86", "c_123", "custom-x", ""]
)
def test_a_custom_sound_that_is_not_the_frozen_stem_is_refused(bad: str) -> None:
    with pytest.raises(ValueError, match="unknown custom push sound"):
        completion_payload(**COMPLETION_INPUT, sound_custom=bad)
    with pytest.raises(ValueError, match="unknown custom push sound"):
        digest_payload(**DIGEST_INPUT, sound_custom=bad)


def test_a_custom_sound_never_rides_a_silenced_banner() -> None:
    with pytest.raises(ValueError, match="silenced"):
        completion_payload(**{**COMPLETION_INPUT, "sound": SOUND_NONE}, sound_custom=CUSTOM_STEM)


def test_the_custom_sound_name_is_the_phones_file_stem() -> None:
    """``c_<sha16>``, lowercase, underscore: the wire value IS the phone's file stem.

    Android ``res/raw`` names must match ``[a-z][a-z0-9_]*`` and expo-notifications
    applies one ``sounds:`` list to both platforms (M1 prebuild), so a hyphen or
    an uppercase hex digit would need a second, cloud-side rename; with this shape
    iOS ``aps.sound = "<sound_custom>.wav"`` is the identity. The cloud splices the
    value into ``aps.sound`` and a channel id, so the freeze pins the SHAPE
    (``custom_pattern``), not only ``str`` (review round 3, m1).

    The rule is derived here from real bytes, not read from the prose: the name of
    a WAV is ``"c_" + sha256(bytes).hexdigest()[:16]``, it matches the filed
    pattern, and it is a legal ``res/raw`` stem; near-misses are refused.
    """
    raw_name = re.compile(r"^[a-z][a-z0-9_]*$")
    wav = b"RIFF\x24\x00\x00\x00WAVEfmt "  # any bytes: the rule is over content
    name = "c_" + hashlib.sha256(wav).hexdigest()[:16]
    for fixture_name in ("emit-completion.json", "emit-digest.json"):
        values = _fixture(fixture_name)["sound_values"]
        pattern = re.compile(values["custom_pattern"])
        assert values["custom_pattern"] == r"^c_[0-9a-f]{16}$", fixture_name
        assert values["custom"].startswith("c_<sha256[:16]>"), values["custom"]
        assert pattern.fullmatch(name) and raw_name.fullmatch(name), name
        for bad in (name.replace("c_", "c-"), name.upper(), name[:-1], name + "0", "c_" + "g" * 16):
            assert not pattern.fullmatch(bad), bad


def test_no_filed_push_shape_carries_a_hyphenated_ios_file_name() -> None:
    """Every filed push note uses the bundled file name, never ``"<id>.wav"``.

    The phone bundles ``<id with - → _>.wav`` (``soft_pulse.wav``), so a fixture
    that told S7 ``aps.sound = "<id>.wav"`` would make every built-in play the iOS
    default (review round 3, M1). Mirrors the mobile ADR verifier's check so the
    two contract homes cannot disagree on that literal again.
    """
    for path in sorted(FIXTURES.glob("*.json")):
        text = path.read_text(encoding="utf-8")
        assert '"<id>.wav' not in text and "soft-pulse.wav" not in text, path.name
        assert "c-<sha256" not in text, path.name
