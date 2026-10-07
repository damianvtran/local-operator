"""S4 cells: the wipe — ends reachable copies, by provenance, with a confirmation.

WHAT THIS FILE PINS (design §4.3, §5.5c):

- a revoke wipes the member's copy and CLOSES the ledger — the member deletes
  and confirms, and the owner's row reads ``wiped``;
- an OFFLINE member is not exempt: nothing is queued at the member, and the
  same notice completes on its NEXT contact (catch-up, never invalidate);
- MEMBER REMOVAL carries the ending itself, through the real verb: the relay's
  ``member rm`` runs the bounded ending exchange BEFORE the tombstone (after it
  the dial is refused by design — an inactive member is never contacted again),
  so the copies are deleted while the member is still contactable and the
  receipt says so; unreachable at that instant, the ledger row stays OPEN and
  the receipt names the count plus the ceiling sentence (§2.3's discipline:
  an ending that cannot complete says so);
- the confirmation rides the REPLY to the owner's dialled announce — an
  un-approved member cannot open a frame (its ``broker_credential`` capability
  left with the grant; measured, the transport refuses ``net_broker`` from a
  deactivated row), so a fresh-request ack would be refused at the owner's door;
- a wipe that cannot confirm blocks ITS OWN key only: a member that answers
  receipt-only (a pre-S4 build) still receives every OTHER key's updates on
  the same pass (review round 1, M2 — the old wipes-only return starved them);
- a ``local-only`` mark ends existing copies the same way;
- a wipe spares rows another owner or another key wrote — deletion is bounded
  by the provenance marker, not by the notice;
- a RE-SHARE after a wipe delivers again (a wiped ledger row must not match
  the old generation+digest, or the re-shared copy would stay undelivered).

The delivery runs two real relays on loopback roots, exactly as
``test_secret_copies`` does.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.network import store
from local_operator.network.credentials import sync
from local_operator.network.credentials.sync import SyncState
from tests.unit.network import conftest as net_fixtures
from tests.unit.network.test_credentials_real_link import (  # noqa: F401
    _lop_network,
    _pull,
)
from tests.unit.network.test_relay_e2e import devices  # noqa: F401 — fixtures by import
from tests.unit.network.test_secret_copies import (  # noqa: F401 — fixtures by import
    KEY,
    SECRET_NAME,
    VALUE_1,
    VALUE_2,
    _member_has,
    _member_value,
    _secret_store,
    secret_mesh,
)


@pytest.fixture()
def copied_mesh(request: pytest.FixtureRequest) -> Any:
    """``secret_mesh`` with A's copy ALREADY delivered to B (one real exchange)."""
    mesh = request.getfixturevalue("secret_mesh")
    code = sync.credentials_sync_step(mesh.a, mesh.member)
    assert code in ("scheduled", "in_sync"), code
    assert net_fixtures.wait_for(lambda: _member_has(mesh.b.root, SECRET_NAME))
    return mesh


def _ledger_wiped(mesh: Any) -> bool:
    row = SyncState.load(mesh.network_id, mesh.a.root).ack_for(mesh.member, KEY) or {}
    return bool(row.get("wiped"))


def test_a_revoke_wipes_the_member_copy_and_the_ledger_closes(copied_mesh: Any) -> None:
    mesh = copied_mesh
    assert _lop_network("credential", "revoke", KEY, "--from", mesh.b.identity.name) == 0
    code = sync.credentials_sync_step(mesh.a, mesh.member)
    assert code in ("scheduled", "in_sync"), code
    assert net_fixtures.wait_for(
        lambda: not _member_has(mesh.b.root, SECRET_NAME)
    ), "the member's copy was not wiped"
    assert net_fixtures.wait_for(
        lambda: _ledger_wiped(mesh)
    ), "the ending was not confirmed to the owner"
    # NOTHING WAS LEFT BEHIND: neither the value nor a record under the name.
    assert not _member_has(mesh.b.root, SECRET_NAME)


def test_an_offline_member_is_wiped_on_reconnect(copied_mesh: Any) -> None:
    """The unreachable member: nothing is queued at either end, and the SAME
    notice recomputes and completes on the next contact (§5.5c, (a)'s catch-up).

    Reachability is simulated at the endpoint (a dead port), not by stopping a
    relay: ``stop``/``start`` is not a supported cycle in this build, and the
    product property under test is the DIAL, not the process.
    """
    mesh = copied_mesh
    with store.mutate(mesh.network_id, mesh.a.root) as record:
        member_row = record.member(mesh.member)
        assert member_row is not None
        member_row.endpoints = ["127.0.0.1:1"]  # a dead endpoint: unreachable now
        store.save(record, mesh.a.root)
    assert _lop_network("credential", "revoke", KEY, "--from", mesh.b.identity.name) == 0
    code = sync.credentials_sync_step(mesh.a, mesh.member)
    assert code in ("scheduled", "in_sync"), code
    assert _member_has(mesh.b.root, SECRET_NAME), "no contact, no wipe"
    assert not _ledger_wiped(mesh)

    with store.mutate(mesh.network_id, mesh.a.root) as record:
        member_row = record.member(mesh.member)
        assert member_row is not None
        member_row.endpoints = [f"127.0.0.1:{mesh.b.settings.port}"]
        store.save(record, mesh.a.root)

    def _landed() -> bool:
        # The dead-endpoint dial may still hold the per-member in-flight guard,
        # so the cell RETRIES the tick rather than assuming one step lands it —
        # which is also how the definitions syncer drives it in the product.
        code = sync.credentials_sync_step(mesh.a, mesh.member)
        assert code in ("scheduled", "in_sync", "in_flight"), code
        return not _member_has(mesh.b.root, SECRET_NAME)

    assert net_fixtures.wait_for(_landed), "the notice did not complete on reconnect"
    assert net_fixtures.wait_for(lambda: _ledger_wiped(mesh))


def test_member_rm_delivers_the_ending_while_the_member_is_contactable(
    copied_mesh: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """Un-approve through the REAL verb: the removal path carries the ending.

    Q1 of review round 1: the definitions tick only visits ACTIVE members, so
    the old cell — deactivate, then drive ``credentials_sync_step`` directly —
    proved a path no product call ever takes, and the product's ``member rm``
    left every copy usable on the removed device with no wipe attempt and no
    sentence. This drives the verb: the relay's handler runs the bounded ending
    exchange BEFORE the tombstone, the member deletes and the ledger closes,
    and the receipt says the ending was confirmed.
    """
    mesh = copied_mesh
    record = store.load(mesh.network_id, mesh.a.root)
    assert _lop_network("member", "rm", record.name, mesh.b.identity.name) == 0
    out = capsys.readouterr().out
    assert net_fixtures.wait_for(
        lambda: not _member_has(mesh.b.root, SECRET_NAME)
    ), "the removal did not deliver the ending"
    assert net_fixtures.wait_for(
        lambda: _ledger_wiped(mesh)
    ), "the wipe was not confirmed to the owner"
    assert "were deleted (the ending is confirmed)" in out, out
    # The tombstone is still the verb's observable side effect, and the removed
    # member is never ticked again — no later contact can re-deliver anything.
    removed = store.load(mesh.network_id, mesh.a.root).member(mesh.member)
    assert removed is not None and not removed.active


def test_member_rm_without_contact_records_the_open_ending(
    copied_mesh: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """Unreachable at removal: the receipt and the ledger carry the OPEN ending.

    A removed member is never contacted again — an inactive row is refused the
    dial by design — so there is no next-contact catch-up to defer to. §2.3's
    discipline applies instead: the exchange runs, fails, and SAYS SO, so a
    removal with live copies never reads as a clean sweep.

    The copy exchange leaves a live link (``LINK_IDLE_S`` is 120 s, far past a
    cell), and a live link would carry the ending however dead the endpoint
    reads — measured while writing this cell. A genuinely unreachable peer has
    none, so the cell closes what the idle reaper eventually would.
    """
    from local_operator.network.credentials.messages import COPY_CEILING_SENTENCE

    mesh = copied_mesh
    for link in list(mesh.a.links.values()):
        if getattr(link, "device_id", "") == mesh.member:
            link.close("test: the member becomes unreachable")
    with store.mutate(mesh.network_id, mesh.a.root) as record:
        member_row = record.member(mesh.member)
        assert member_row is not None
        member_row.endpoints = ["127.0.0.1:1"]  # a dead endpoint: unreachable now
        store.save(record, mesh.a.root)
    record = store.load(mesh.network_id, mesh.a.root)
    assert _lop_network("member", "rm", record.name, mesh.b.identity.name) == 0
    out = capsys.readouterr().out
    assert _member_has(mesh.b.root, SECRET_NAME), "no contact, no wipe"
    assert not _ledger_wiped(mesh), "nothing confirmed it, so it must stay open"
    assert "could NOT be confirmed deleted" in out, out
    assert COPY_CEILING_SENTENCE in out, out


def test_an_unconfirmable_wipe_does_not_starve_other_keys(
    copied_mesh: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M2 (review round 1): a wipe that never confirms blocks ITS key only.

    A pre-S4 member answers receipt-only, so the owner's ledger row for the
    revoked key stays un-``wiped`` forever. The old ``_pending_announces``
    returned wipes-ONLY whenever any wipe was owed, so every other key's future
    value updates were withheld from that member silently and unboundedly.
    The fix gates per key: the ending rides every exchange AND the positives
    flow in the same pass, each kind under its own cap. Both arms are pinned:
    the unconfirmable wipe's key stays withheld (the copy is gone and the row
    is open), and the other key's update reaches the member anyway.
    """
    mesh = copied_mesh
    second_name = "STORE_COPY_TOKEN_TWO"
    second_key = f"secret:{second_name}"
    mesh.store_a.set(second_name, VALUE_1, description="M2's second key")
    assert _lop_network("credential", "share", second_key, "--with", mesh.b.identity.name) == 0
    pulled = _pull(SimpleNamespace(b=mesh.b, borrower=mesh.member))
    assert second_key in pulled.get("changed", []), pulled
    code = sync.credentials_sync_step(mesh.a, mesh.member)
    assert code in ("scheduled", "in_sync"), code
    assert net_fixtures.wait_for(lambda: _member_has(mesh.b.root, second_name))

    # The member becomes one that CANNOT confirm a wipe: the reply is answered
    # (the copy is deleted) but the owner never records it — the pre-S4 shape.
    assert sync.sync_for_relay(mesh.a) is mesh.engine_a
    monkeypatch.setattr(mesh.engine_a, "_record_wipe_reply", lambda *a, **k: None)
    assert _lop_network("credential", "revoke", KEY, "--from", mesh.b.identity.name) == 0
    code = sync.credentials_sync_step(mesh.a, mesh.member)
    assert code in ("scheduled", "in_sync"), code
    assert net_fixtures.wait_for(lambda: not _member_has(mesh.b.root, SECRET_NAME))
    assert not _ledger_wiped(mesh), "the unconfirmable wipe must stay open"

    # The OTHER key changes at the source: the same pass must still carry it.
    mesh.store_a.update(second_name, VALUE_2)
    _net, frames = mesh.engine_a._pending_announces(mesh.member)
    kinds = [(f.get("key"), f.get("value_state")) for f in frames]
    assert (KEY, sync.VALUE_STATE_ABSENT) in kinds, kinds
    assert (second_key, sync.VALUE_STATE_PRESENT) in kinds, kinds
    assert kinds[0][0] == KEY, f"the ending must lead the frames, got {kinds}"

    code = sync.credentials_sync_step(mesh.a, mesh.member)
    assert code in ("scheduled", "in_sync"), code
    assert net_fixtures.wait_for(
        lambda: _member_value(mesh.b.root, second_name) == VALUE_2
    ), "the other key's update was starved behind the unconfirmable wipe"
    assert not _ledger_wiped(mesh), "and the open row is still tracked, not dropped"


def test_marking_local_only_ends_existing_copies(copied_mesh: Any) -> None:
    """§4.2's kill switch, second half: the mark ends a copy that already exists."""
    mesh = copied_mesh
    assert _lop_network("credential", "mark", SECRET_NAME, "local-only") == 0
    code = sync.credentials_sync_step(mesh.a, mesh.member)
    assert code in ("scheduled", "in_sync"), code
    assert net_fixtures.wait_for(
        lambda: not _member_has(mesh.b.root, SECRET_NAME)
    ), "a local-only key must end copies it already has"
    assert net_fixtures.wait_for(lambda: _ledger_wiped(mesh))


def test_a_wipe_spares_rows_another_owner_or_key_wrote(request: pytest.FixtureRequest) -> None:
    """The provenance bound: deletion is matched on BOTH the sender and the key."""
    mesh = request.getfixturevalue("secret_mesh")
    handle = _secret_store(mesh.b.root, create=True)
    handle.set("LOCAL_TOKEN", b"a local row")
    handle.set(
        "OTHER_OWNERS_ROW",
        b"not from A",
        origin={"owner_device": "d_" + "f" * 32, "key": KEY, "gen": 1},
    )
    handle.set(
        "OTHER_KEY_ROW",
        b"from A, another key",
        origin={"owner_device": mesh.owner, "key": "secret:OTHER_KEY", "gen": 1},
    )

    code = sync.credentials_sync_step(mesh.a, mesh.member)
    assert code in ("scheduled", "in_sync"), code
    assert net_fixtures.wait_for(lambda: _member_has(mesh.b.root, SECRET_NAME))

    assert _lop_network("credential", "revoke", KEY, "--from", mesh.b.identity.name) == 0
    assert sync.credentials_sync_step(mesh.a, mesh.member) in ("scheduled", "in_sync")
    assert net_fixtures.wait_for(lambda: not _member_has(mesh.b.root, SECRET_NAME))

    assert _member_has(mesh.b.root, "LOCAL_TOKEN"), "a local row is not this owner's to delete"
    assert _member_has(mesh.b.root, "OTHER_OWNERS_ROW"), "another owner's row must survive"
    assert _member_has(mesh.b.root, "OTHER_KEY_ROW"), "another key's row must survive"


def test_copy_use_revoke_wipe_in_one_run(request: pytest.FixtureRequest) -> None:
    """The whole sequence in one run: copy lands, the member USES it, revoke wipes.

    "Use" is the member's own read path (``get`` — the ordinary read a session
    performs, which also counts as a use), not a re-read of the frame. The
    closing canary re-scans the member's root: after the wipe, the value's
    plaintext appears nowhere, and neither does a record under the name.
    """
    mesh = request.getfixturevalue("secret_mesh")
    code = sync.credentials_sync_step(mesh.a, mesh.member)
    assert code in ("scheduled", "in_sync"), code
    assert net_fixtures.wait_for(lambda: _member_has(mesh.b.root, SECRET_NAME))

    handle = _secret_store(mesh.b.root)
    assert handle.get(SECRET_NAME) == VALUE_1  # the member uses its copy

    assert _lop_network("credential", "revoke", KEY, "--from", mesh.b.identity.name) == 0
    assert sync.credentials_sync_step(mesh.a, mesh.member) in ("scheduled", "in_sync")
    assert net_fixtures.wait_for(lambda: not _member_has(mesh.b.root, SECRET_NAME))
    assert net_fixtures.wait_for(lambda: _ledger_wiped(mesh))

    for path in Path(mesh.b.root).rglob("*"):
        if path.is_file():
            assert VALUE_1 not in path.read_bytes(), path


def test_the_revoke_and_mark_receipts_say_which_ending_applies(
    copied_mesh: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """The receipt sentences the design commits to (§2.3, §4.3) are what prints."""
    mesh = copied_mesh
    assert _lop_network("credential", "mark", SECRET_NAME, "sync") == 0
    out = capsys.readouterr().out
    assert f"'{SECRET_NAME}' is marked sync" in out

    # THE LISTING RENDERS IT ON THE ROW (one document, so the mark cannot
    # disagree with the holders it governs).
    assert _lop_network("credentials") == 0
    out = capsys.readouterr().out
    assert "marked sync" in out

    assert _lop_network("credential", "revoke", KEY, "--from", mesh.b.identity.name) == 0
    out = capsys.readouterr().out
    assert f"a wipe notice for '{KEY}' is queued for" in out
    assert "This removed the copies it could reach." in out


def test_a_re_share_after_a_wipe_delivers_again(copied_mesh: Any) -> None:
    """The wiped ledger row is an ENDING, not a match: a re-share must announce."""
    mesh = copied_mesh
    assert _lop_network("credential", "revoke", KEY, "--from", mesh.b.identity.name) == 0
    assert sync.credentials_sync_step(mesh.a, mesh.member) in ("scheduled", "in_sync")
    assert net_fixtures.wait_for(lambda: not _member_has(mesh.b.root, SECRET_NAME))
    assert net_fixtures.wait_for(lambda: _ledger_wiped(mesh))

    assert _lop_network("credential", "share", KEY, "--with", mesh.b.identity.name) == 0
    pulled = _pull(SimpleNamespace(b=mesh.b, borrower=mesh.member))
    assert KEY in pulled.get("changed", []), pulled
    code = sync.credentials_sync_step(mesh.a, mesh.member)
    assert code in ("scheduled", "in_sync"), code
    assert net_fixtures.wait_for(
        lambda: _member_has(mesh.b.root, SECRET_NAME)
    ), "a re-shared copy stayed undelivered"
    row = SyncState.load(mesh.network_id, mesh.a.root).ack_for(mesh.member, KEY) or {}
    assert not row.get("wiped"), "a live copy must not read wiped"
