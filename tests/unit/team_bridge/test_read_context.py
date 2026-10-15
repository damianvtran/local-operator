"""The READ-ONLY read context: the two standings, the grammars, and no writes.

WHY THIS FILE EXISTS BESIDE THE ROUTE CELLS. The route tests drive the read
through the ASGI app, which is what a caller does; these pin the decisions the
route cannot show — WHICH standing the read resolved (two DISTINCT context
types, so a surface can never present retained history as a live inbox), and the
ID/cursor grammars that keep a caller-supplied value out of an upstream path.

The store here is the REAL one (``commit_enrollment`` is the seam H4 commits
through): the two standings differ by the DOCUMENT on disk, not by an argument,
and a fixture that faked either would test nothing about the difference.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

from local_operator.team_bridge.read_context import (
    CURSOR_PATTERN,
    EXCERPT_MAX_CHARS,
    MESSAGE_ID_PATTERN,
    HistoryReadContext,
    ReadContext,
    ReadInbox,
    ReadRefused,
    ReceiveReadContext,
    admit_read_content,
    detail_view,
    inbox_view,
    read_inbox_page,
    read_message_detail,
    resolve_read_context,
)
from local_operator.team_bridge.store import OwnerReceipt, OwnerSnapshot, Store
from tests.unit.server.test_desktop_team_bridge import (
    _access,
    _owner,
    _plant_proof,
    _wire_item,
)


def _enroll(root: Path, *, enabled: bool) -> OwnerSnapshot:
    owner = _owner()
    Store(root).commit_enrollment(owner, OwnerReceipt("bridge_consent_v1", "ref-1"))
    (root / "config.yml").write_text(
        yaml.safe_dump({"values": {"team_bridge": {"enabled": enabled}}})
    )
    return owner


def _patch_client(monkeypatch: pytest.MonkeyPatch) -> None:
    from local_operator.team_bridge import client as tb_client
    from tests.unit.server.test_desktop_team_bridge import FakeClient

    FakeClient.behavior = {}
    FakeClient.instances = []
    FakeClient.behavior["access"] = _access()
    monkeypatch.setattr(tb_client, "TeamBridgeClient", FakeClient)


@pytest.mark.asyncio
async def test_an_enabled_machine_resolves_receive_standing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Enabled + consent + own credential row = the receipt/receive context."""

    _patch_client(monkeypatch)
    owner = _enroll(tmp_path, enabled=True)

    context = await resolve_read_context(tmp_path, expected_account_id="acc_1")

    assert isinstance(context, ReceiveReadContext)
    assert context.mode == "receive"
    assert context.read_only is False
    assert context.owner == owner
    assert not hasattr(context, "reason")


@pytest.mark.asyncio
async def test_a_paused_machine_resolves_retained_history_standing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The SAME namespace, a weaker standing: reads only, and it SAYS so.

    The distinction is a type, not a flag a caller may forget to read: a surface
    that receives a ``HistoryReadContext`` cannot render it as a live inbox, and
    the view names the reason so the honest sentence is available to it.
    """

    _patch_client(monkeypatch)
    _enroll(tmp_path, enabled=False)

    context = await resolve_read_context(tmp_path, expected_account_id="acc_1")

    assert isinstance(context, HistoryReadContext)
    assert isinstance(context, ReadContext)
    assert context.mode == "history"
    assert context.read_only is True
    assert context.reason == "bridge_off"
    view = inbox_view(ReadInbox(context=context))
    assert view["mode"] == "history"
    assert view["read_only"] is True
    assert view["history_reason"] == "bridge_off"
    assert view["body_included"] is False


@pytest.mark.asyncio
async def test_a_declared_account_this_machine_is_not_signed_into_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The captured-account guard is the RESOLVER's, before any wire call."""

    _patch_client(monkeypatch)
    _enroll(tmp_path, enabled=True)
    from tests.unit.server.test_desktop_team_bridge import FakeClient

    FakeClient.behavior["access"] = _access("acc_other")

    with pytest.raises(ReadRefused) as refusal:
        await resolve_read_context(tmp_path, expected_account_id="acc_1")
    assert refusal.value.code == "account_mismatch"
    assert FakeClient.behavior.get("last_inbox_kwargs") is None


def test_the_id_and_cursor_grammars_reject_injection_shapes() -> None:
    """The probe's shapes: ``..``, ``?``, ``/`` and a fragment never match.

    These values are interpolated into an upstream path or query by the
    transport, so this is the gate that keeps a caller-supplied id from
    rewriting the request — before any path is built.
    """

    accepted: dict[str, list[str]] = {
        MESSAGE_ID_PATTERN: ["bmsg_1", "bmsg_AbC-9_0"],
        CURSOR_PATTERN: ["MjU=", "b64+/_-="],
    }
    refused: dict[str, list[str]] = {
        MESSAGE_ID_PATTERN: [
            "bmsg_1/../x",
            "bmsg_1?y=1",
            "../../etc",
            "bmsg_1#f",
            "msg_1",
            # A TRAILING NEWLINE (review R-5): with a ``$`` anchor ``re.match``
            # ACCEPTS it — ``$`` also matches before a final newline, so such a
            # value reached the transport. ``fullmatch`` is the semantics the
            # route's validator uses and the one this cell pins.
            "bmsg_1\n",
        ],
        CURSOR_PATTERN: ["a b", "a?b=1", "a#b", "../x", "MjU=\n"],
    }
    for pattern, values in accepted.items():
        for value in values:
            assert re.fullmatch(pattern, value), (pattern, value)
    for pattern, values in refused.items():
        for value in values:
            assert re.fullmatch(pattern, value) is None, (pattern, value)


def test_the_view_carries_no_field_a_body_could_ride() -> None:
    """The view's own contract: metadata flags, stated, and no content key."""

    context = HistoryReadContext(owner=_owner(), access=_access(), reason="bridge_off")
    view: Any = inbox_view(ReadInbox(context=context))
    assert view["screening_detail_included"] is False
    assert view["local_events"] is False
    assert "items" in view and view["items"] == []


# --- M1a: the shared content-admission predicate (S-RSA-3) --------------------


def _admission(root: Path, owner: OwnerSnapshot, item: Mapping[str, Any]) -> Any:
    """The predicate's verdict, fed by the REAL read-only snapshot."""

    context = ReceiveReadContext(owner=owner, access=_access())
    proof = Store(root).content_proof_readonly(owner, [str(item["message"]["id"])])
    return admit_read_content(context, item, proof[str(item["message"]["id"])])


def test_the_predicate_admits_the_body_only_on_the_whole_tuple(tmp_path: Path) -> None:
    """The positive control every refusal cell is measured against."""

    owner = _enroll(tmp_path, enabled=True)
    item = _wire_item()
    _plant_proof(tmp_path, owner, item)

    admission = _admission(tmp_path, owner, item)

    assert admission.admitted is True
    assert admission.reason is None
    assert admission.text == item["message"]["body"]["text"]
    assert admission.envelope is not None
    assert admission.envelope.envelope_digest


@pytest.mark.parametrize(
    ("case", "overrides", "expected_reason"),
    [
        ("receipt-absent", {"receipt": False}, "proof_missing"),
        ("policy-absent", {"policy": False}, "proof_missing"),
        ("read-intent-absent", {"intent": None}, "read_intent_missing"),
        ("read-intent-pending", {"intent": "pending"}, "read_intent_mismatch"),
        ("read-intent-conflict", {"intent": "conflict"}, "read_intent_mismatch"),
        (
            "read-intent-different-tuple",
            {"intent_body": "b" * 64},
            "read_intent_mismatch",
        ),
        (
            "verdict-reject",
            {"verdict": "reject", "categories": ("spam",), "intent": None},
            "not_allowed",
        ),
        # A policy that MOVED on from the receipt's tuple: the verdict is evidence
        # of what the policy of its moment asked for, not a licence that outlives
        # it (S-RSA-3.4). Three separate cells, one per field the server pins.
        ("template-changed", {"policy_template": "ac" * 32}, "receipt_stale"),
        ("model-changed", {"policy_model": "m-2"}, "receipt_stale"),
        ("admission-revision-changed", {"policy_revision": "r-2"}, "receipt_stale"),
        ("policy-epoch-changed", {"epoch_offset": -1}, "receipt_stale"),
        ("creator-generation-bumped", {"generation": 4}, "receipt_stale"),
        ("allow-with-categories", {"categories": ("spam",)}, "proof_corrupt"),
        ("categories-not-a-list", {"categories_text": "not-json"}, "proof_corrupt"),
        ("body-hash-mismatch", {"receipt_body": "c" * 64}, "body_mismatch"),
        ("envelope-digest-mismatch", {"receipt_envelope": "d" * 64}, "envelope_mismatch"),
    ],
)
def test_each_false_term_denies_content(
    tmp_path: Path, case: str, overrides: dict[str, Any], expected_reason: str
) -> None:
    """One broken term per cell: no generic "privacy" assertion.

    A cell that is refused must also say WHY it was refused with the closed code
    for the term it broke — otherwise a predicate that denied everything would
    pass this file, and the next surface would inherit a denial it cannot explain.
    """

    owner = _enroll(tmp_path, enabled=True)
    item = _wire_item()
    _plant_proof(tmp_path, owner, item, **overrides)

    admission = _admission(tmp_path, owner, item)

    assert admission.admitted is False, case
    assert admission.text is None, case
    assert admission.reason == expected_reason, case


@pytest.mark.parametrize("state", ["unread", "blocked"])
def test_an_unread_or_blocked_delivery_shows_no_content(tmp_path: Path, state: str) -> None:
    """The upstream state gates content even beside a perfect local proof (S-RSA-3.6).

    ``unread`` means nobody took the message up and ``blocked`` is a native
    refusal; a local receipt may not overrule either, and the state is read from
    the AUTHENTICATED answer rather than inferred from the row.
    """

    owner = _enroll(tmp_path, enabled=True)
    item = _wire_item(state=state)
    _plant_proof(tmp_path, owner, item)

    admission = _admission(tmp_path, owner, item)

    assert admission.admitted is False
    assert admission.reason == "delivery_not_admitted"


def test_a_native_hold_outranks_a_receipt(tmp_path: Path) -> None:
    """A hold is this machine's record that the item never reached a session."""

    owner = _enroll(tmp_path, enabled=True)
    item = _wire_item()
    _plant_proof(tmp_path, owner, item, hold="envelope_refused:whatever")

    admission = _admission(tmp_path, owner, item)

    assert admission.admitted is False
    assert admission.reason == "held"


def test_a_sender_change_under_the_same_body_denies_content(tmp_path: Path) -> None:
    """Envelope digest, not just the body hash (spec L199).

    The receipt is minted for Priya Raman's note; the item arrives with the same
    bytes under another sender. The body hash still matches, so a predicate that
    compared only bytes would serve it.
    """

    owner = _enroll(tmp_path, enabled=True)
    minted = _wire_item()
    _plant_proof(tmp_path, owner, minted)
    arrived = _wire_item(sender_name="Someone Else")

    admission = _admission(tmp_path, owner, arrived)

    assert admission.admitted is False
    assert admission.reason == "envelope_mismatch"


def test_a_body_whose_served_hash_does_not_match_is_refused(tmp_path: Path) -> None:
    """The envelope is authenticated BEFORE any stored row is consulted."""

    owner = _enroll(tmp_path, enabled=True)
    item = _wire_item()
    _plant_proof(tmp_path, owner, item)
    # The proof is planted for the item the server SHOULD have sent; the wire
    # answer then carries a hash the bytes do not produce. Nothing stored may be
    # consulted first: a receipt is evidence about bytes, so an unauthenticated
    # body is refused before the receipt is even looked at.
    item["message"]["body_hash"] = "e" * 64

    admission = _admission(tmp_path, owner, item)

    assert admission.admitted is False
    assert admission.text is None
    assert admission.reason == "body_hash_mismatch"


def test_the_proof_snapshot_creates_no_store_when_the_machine_has_none(tmp_path: Path) -> None:
    """Zero writes is structural: an absent store is "nothing enrolled" (M-9/M-10).

    A reader that created the database to answer would turn a GET into a write,
    and the absence of proof would become indistinguishable from an empty store.
    """

    owner = _owner()
    assert not (tmp_path / "team_bridge").exists()

    snapshot = Store(tmp_path).content_proof_readonly(owner, ["bmsg_1", "bmsg_2"])

    assert snapshot == {
        "bmsg_1": {"policy": None, "receipt": None, "intent": None, "hold": None},
        "bmsg_2": {"policy": None, "receipt": None, "intent": None, "hold": None},
    }
    assert not list(tmp_path.rglob("*.sqlite3")), "the read created a database"
    assert not (tmp_path / "team_bridge").exists()


def test_the_snapshot_returns_the_planted_rows_and_their_own_values(tmp_path: Path) -> None:
    """What the predicate reads is the store's own row, decoded exactly once."""

    owner = _enroll(tmp_path, enabled=True)
    item = _wire_item()
    _plant_proof(tmp_path, owner, item, categories=())

    snapshot = Store(tmp_path).content_proof_readonly(owner, ["bmsg_1"])["bmsg_1"]

    assert snapshot["policy"] is not None
    assert snapshot["receipt"] is not None and snapshot["receipt"]["verdict"] == "allow"
    assert snapshot["intent"] is not None and snapshot["intent"]["state"] == "confirmed"
    assert snapshot["intent"]["categories"] == []
    assert snapshot["hold"] is None


@pytest.mark.asyncio
async def test_the_inbox_row_carries_an_excerpt_only_where_the_predicate_admits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The list window is the wake's own bound, and only under the same decision."""

    owner = _enroll(tmp_path, enabled=True)
    _patch_client(monkeypatch)
    from tests.unit.server.test_desktop_team_bridge import FakeClient, _page

    long_body = "x" * (EXCERPT_MAX_CHARS + 25)
    admitted = _wire_item(message_id="bmsg_1", body=long_body)
    refused = _wire_item(message_id="bmsg_2", body="not admitted")
    _plant_proof(tmp_path, owner, admitted)
    FakeClient.behavior["inbox"] = _page(admitted, refused)
    context = ReceiveReadContext(owner=owner, access=_access())

    page = await read_inbox_page(context, config_dir=tmp_path)
    rows = inbox_view(page)["items"]

    assert rows[0]["content"]["excerpt"] == long_body[:EXCERPT_MAX_CHARS]
    assert rows[0]["content"]["truncated"] is True
    assert rows[0]["content"]["content_trust"] == "untrusted_external"
    assert rows[0]["content"]["content_format"] == "plain_text"
    assert rows[1]["content"] is None, "a row without proof carried text"
    assert inbox_view(page)["body_included"] is True


@pytest.mark.asyncio
async def test_the_detail_read_serves_the_full_body_through_the_predicate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Detail = the whole body; list = the excerpt; SAME decision behind both."""

    owner = _enroll(tmp_path, enabled=True)
    _patch_client(monkeypatch)
    from tests.unit.server.test_desktop_team_bridge import FakeClient

    body = "y" * (EXCERPT_MAX_CHARS + 25)
    item = _wire_item(body=body)
    _plant_proof(tmp_path, owner, item)
    FakeClient.behavior["message"] = item
    context = ReceiveReadContext(owner=owner, access=_access())

    detail = await read_message_detail(
        context, config_dir=tmp_path, message_id="bmsg_1", tenant_id="tnt_1"
    )
    view: Any = detail_view(detail)

    assert view["content"]["text"] == body
    assert view["content"]["truncated"] is False
    assert view["body_included"] is True
    assert view["screening_detail_included"] is True
    assert view["local_events"] is False
    assert view["item"]["content"]["text"] == body


@pytest.mark.asyncio
async def test_the_detail_read_is_metadata_only_without_a_confirmed_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An allow receipt with a PENDING intent is not an admission (S-RSA-3.5)."""

    owner = _enroll(tmp_path, enabled=True)
    _patch_client(monkeypatch)
    from tests.unit.server.test_desktop_team_bridge import FakeClient

    item = _wire_item(private_reason="the model said the sender was pushy")
    _plant_proof(tmp_path, owner, item, intent="pending")
    FakeClient.behavior["message"] = item
    context = ReceiveReadContext(owner=owner, access=_access())

    view: Any = detail_view(
        await read_message_detail(
            context, config_dir=tmp_path, message_id="bmsg_1", tenant_id="tnt_1"
        )
    )

    assert view["content"] is None
    assert view["body_included"] is False
    assert "the model said the sender was pushy" not in json.dumps(view)
    assert view["item"]["screening"]["state"] == "unreviewed"


@pytest.mark.asyncio
async def test_a_detail_answer_for_another_message_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The id rides the path: an answer for a different id is not this read."""

    owner = _enroll(tmp_path, enabled=True)
    _patch_client(monkeypatch)
    from tests.unit.server.test_desktop_team_bridge import FakeClient

    FakeClient.behavior["message"] = _wire_item(message_id="bmsg_other")
    context = ReceiveReadContext(owner=owner, access=_access())

    with pytest.raises(ReadRefused) as refusal:
        await read_message_detail(
            context, config_dir=tmp_path, message_id="bmsg_1", tenant_id="tnt_1"
        )

    assert refusal.value.code == "malformed_page"
