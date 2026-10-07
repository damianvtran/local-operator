"""S4 cells: the wipe — ends reachable copies, by provenance, with a confirmation.

WHAT THIS FILE PINS (design §4.3, §5.5c):

- a revoke wipes the member's copy and CLOSES the ledger — the member deletes
  and confirms, and the owner's row reads ``wiped``;
- an OFFLINE member is not exempt: nothing is queued at the member, and the
  same notice completes on its NEXT contact (catch-up, never invalidate);
- the ending reaches a REMOVED member, where the confirmation rides the REPLY
  to the owner's dialled announce — an un-approved member cannot open a frame
  (its ``broker_credential`` capability left with the grant; measured, the
  transport refuses ``net_broker`` from a deactivated row), so a fresh-request
  ack would be refused at the owner's door;
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
    _member_has,
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


def test_the_ending_reaches_a_removed_member(request: pytest.FixtureRequest) -> None:
    """Un-approve: the copy is withheld AND the ending still lands, reply-confirmed.

    B is deactivated outright, so its capability is gone and a frame it opened
    would be refused at A's door. The wipe still reaches it (A dialled), and the
    confirmation rides the REPLY — which is why the ledger closes at all.
    """
    mesh = request.getfixturevalue("secret_mesh")
    code = sync.credentials_sync_step(mesh.a, mesh.member)
    assert code in ("scheduled", "in_sync"), code
    assert net_fixtures.wait_for(lambda: _member_has(mesh.b.root, SECRET_NAME))

    with store.mutate(mesh.network_id, mesh.a.root) as copy:
        member_row = copy.member(mesh.member)
        assert member_row is not None
        member_row.lifecycle = "expired"
        store.save(copy, mesh.a.root)

    code = sync.credentials_sync_step(mesh.a, mesh.member)
    assert code in ("scheduled", "in_sync"), code
    assert net_fixtures.wait_for(
        lambda: not _member_has(mesh.b.root, SECRET_NAME)
    ), "the removed member's copy was not wiped"
    assert net_fixtures.wait_for(
        lambda: _ledger_wiped(mesh)
    ), "the wipe was not confirmed to the owner"


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
