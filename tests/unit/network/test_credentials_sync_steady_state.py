"""Steady state costs nothing: an unchanged value is never re-copied or re-numbered.

WHY THIS MODULE EXISTS (measured on the live mesh, 2026-10-21). The owner Mac's
``audit.jsonl`` carried 5,491 ``credential.copy`` rows in 24 h: the SAME five keys
re-copied to one member every ~15 s, while ``sync.json`` showed each key's
generation advancing by exactly one per tick with its digest and ``updated_at``
constant and the member's ack always equal to it. The value never moved.

THE MECHANISM these cells pin, one per half:

* the member's placement pull (``enqueue_placement_pull`` on every definitions
  tick) called ``_pull_blocking`` with an EMPTY announced digest — which can
  never equal the held one — so every tick sent ``copy`` for every held key;
* the owner's ``_served_generation`` answers ``held + 1`` whenever the member's
  counter is at or past its own, which is precisely the steady state, so every
  request bumped the generation and re-served the value;
* the member re-applied it (insert a new row, sweep the old), and the new row id
  read to every session's binding recorder as a different credential — one
  ``session_credential_binding_notice.v1`` row per tick.

Each cell drives the product's own functions; the fake-link rig wires the
member's link straight into the owner's real handlers (``owner_copy`` /
``owner_ack``) over two isolated roots, so both ledgers and both stores are real.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.network.credentials import sync
from local_operator.network.credentials.sync import SyncState
from tests.unit.network import conftest as net_fixtures
from tests.unit.network.test_credentials_sync import (  # noqa: F401 — fixtures by import
    KEY,
    MEMBER,
    NETWORK,
    OWNER,
    VALUE_1,
    VALUE_2,
    _declare_owner,
    _engine,
    _FakeServer,
    _member_rows,
    _open_member_store,
    _payload,
    _run_exchange,
    _seed_network,
    _wait_for_owner_ack,
    devices,
    sync_mesh,
)

#: Ticks per steady-state cell. Enough that a per-tick bump is unmistakable
#: (the live symptom was one bump per tick, so N ticks = N rows), small enough to
#: stay a unit cell.
TICKS = 6


class _Audit:
    """Collects the broker's ``AuditEvent`` records (the owner's evidence row)."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    def record(self, event: Any) -> None:
        self.events.append(event)

    def count(self, name: str) -> int:
        return sum(1 for event in self.events if getattr(event, "event", "") == name)


class _Rig:
    """Owner broker + member engine on two isolated roots, linked in-process."""

    def __init__(self, root: Path) -> None:
        from local_operator.network.credentials.owner import MeshCredentialBroker

        self.owner_root = root / "owner"
        self.member_root = root / "member"
        self.owner_root.mkdir()
        self.member_root.mkdir()
        _declare_owner(self.owner_root)
        _declare_owner(self.member_root)
        _seed_network(self.owner_root)
        self.owner_store = _open_member_store(self.owner_root)
        self.owner_store.upsert_credential(KEY, _payload(VALUE_1))
        self.audit = _Audit()
        self.broker = MeshCredentialBroker(
            root=self.owner_root,
            self_device=OWNER,
            self_device_name="owner",
            network_id=NETWORK,
            audit=self.audit,
            auth_store=self.owner_store,
        )
        self.server = _FakeServer(self.member_root, MEMBER, "member")
        self.server.reply = self._dispatch
        self.engine = _engine(self.member_root, self.server, member_root=self.member_root)

    def _dispatch(self, frame: dict[str, Any]) -> dict[str, Any]:
        link = SimpleNamespace(device_id=MEMBER)
        kind = frame.get("kind")
        if kind == "copy":
            return {"op": "ack", "detail": sync.owner_copy(self.broker, link, frame)}
        if kind == "ack":
            return {"op": "ack", "detail": sync.owner_ack(self.broker, link, frame)}
        raise AssertionError(f"unexpected frame kind {kind!r}")

    def tick(self) -> None:
        """The member half of one definitions tick, run inline."""
        self.engine._pull_placement_blocking(OWNER, [KEY])

    def copy_frames(self) -> int:
        return sum(1 for frame in self.server.frames if frame.get("kind") == "copy")

    def owner_gen(self) -> int:
        row = SyncState.load(NETWORK, self.owner_root).generation(KEY)
        return int((row or {}).get("gen") or 0)

    def member_row_ids(self) -> list[int]:
        handle = _open_member_store(self.member_root)
        try:
            return [int(row.id) for row in handle.list_credentials(KEY, include_disabled=True)]
        finally:
            handle.close()

    def close(self) -> None:
        self.owner_store.close()


@pytest.fixture()
def rig(root: Path) -> Any:
    built = _Rig(root)
    try:
        yield built
    finally:
        built.close()


def test_steady_state_makes_zero_copy_requests_over_many_ticks(rig: _Rig) -> None:
    """The headline: after the first delivery, N unchanged ticks copy NOTHING.

    Fails on the pre-fix head with ``copy_frames == 1 + TICKS`` and the owner's
    generation at ``1 + TICKS``: every tick requested a copy and every request
    bumped — the live mesh's shape, one row per key per tick.
    """
    rig.tick()
    assert rig.copy_frames() == 1, "the first tick delivers the copy once"
    assert rig.audit.count("credential.copy") == 1
    first_gen = rig.owner_gen()
    first_rows = rig.member_row_ids()
    assert len(first_rows) == 1

    for _ in range(TICKS):
        rig.tick()

    assert rig.copy_frames() == 1, "an unchanged value must not be requested again"
    assert rig.audit.count("credential.copy") == 1, "the owner served no further copy"
    assert rig.owner_gen() == first_gen, "the generation moves only on a value change"
    assert rig.member_row_ids() == first_rows, "the member's row (and its id) is untouched"


def test_a_real_change_still_moves_the_generation_exactly_once(rig: _Rig) -> None:
    """The guard is not a freeze: a changed value bumps once and converges.

    The change itself is the OWNER's announce to deliver (the placement pull
    only fills keys the member does not hold), so it is driven through
    ``_pull_blocking`` with the announced generation+digest — the announce
    handler's own call.
    """
    rig.tick()
    gen_1 = rig.owner_gen()
    rig.owner_store.upsert_credential(KEY, _payload(VALUE_2))
    meta = sync.read_copy_value(rig.owner_store, KEY, rig.broker._entry(KEY))
    assert meta is not None
    with sync.mutate(NETWORK, rig.owner_root) as state:
        gen_2 = sync._ensure_generation(state, KEY, meta)
    assert gen_2 == gen_1 + 1
    assert rig.engine._pull_blocking(OWNER, KEY, gen_2, meta["digest"]) == "applied"
    handle = _open_member_store(rig.member_root)
    try:
        assert [row.data["key"] for row in handle.list_credentials(KEY)] == [VALUE_2]
    finally:
        handle.close()
    for _ in range(TICKS):
        rig.tick()
    assert rig.owner_gen() == gen_2, "nothing after the change moves the counter again"
    assert rig.audit.count("credential.copy") == 2, "one copy per real value, no more"


def test_a_copy_deleted_from_the_member_store_is_re_served(rig: _Rig) -> None:
    """The guard's safety half: the ledger alone never suppresses a pull.

    An operator who deletes the copied row by hand leaves an ``applied`` row
    behind, and the owner's ack says synced, so the owner will not announce
    again. The placement pull must see the row is gone and pull it back.
    """
    rig.tick()
    handle = _open_member_store(rig.member_root)
    try:
        for row in handle.list_credentials(KEY, include_disabled=True):
            handle.delete_credential(int(row.id))
    finally:
        handle.close()
    rig.tick()
    assert rig.copy_frames() == 2, "a missing copy must be requested again"
    handle = _open_member_store(rig.member_root)
    try:
        assert [row.data["key"] for row in handle.list_credentials(KEY)] == [VALUE_1]
    finally:
        handle.close()


def test_owner_adopts_rather_than_re_serves_an_equal_held_digest(rig: _Rig) -> None:
    """The defensive owner rule: held digest == current digest never bumps.

    A copy request that proves (``held_digest``) the member already holds the
    current bytes is answered ``in_sync`` — no value, no ``credential.copy``
    row, no generation move — however its ``held`` counter compares. Pre-fix
    this request served ``held + 1`` and bumped.
    """
    rig.tick()
    gen = rig.owner_gen()
    digest = sync.fingerprint(_payload(VALUE_1))
    link = SimpleNamespace(device_id=MEMBER)
    for held in (gen, gen + 5):
        detail = sync.owner_copy(
            rig.broker,
            link,
            {"kind": "copy", "key": KEY, "gen": 0, "held": held, "held_digest": digest},
        )
        assert detail == {"kind": "in_sync", "key": KEY, "gen": gen, "digest": digest}, detail
        assert "value" not in detail
    assert rig.owner_gen() == gen
    assert rig.audit.count("credential.copy") == 1


def test_owner_reads_a_legacy_members_ack_as_its_held_digest(rig: _Rig) -> None:
    """A member build without ``held_digest`` (0.84.5) stops the bump too.

    The owner's own ack ledger for exactly the generation the member says it
    holds is the evidence. An old member cannot read ``in_sync`` (its apply
    calls anything but ``copy`` a refusal), so it is served at the CURRENT
    generation — never ``held + 1`` — and its own monotonicity rule drops that
    reply before applying. A frame that DOES carry the field, even empty, is
    taken at its word — that is how a current member whose copy was deleted
    gets it re-served — and an ack for a DIFFERENT generation proves nothing.
    """
    rig.tick()
    gen = rig.owner_gen()
    link = SimpleNamespace(device_id=MEMBER)
    legacy = {"kind": "copy", "key": KEY, "gen": 0, "held": gen}
    for _ in range(TICKS):
        detail = sync.owner_copy(rig.broker, link, dict(legacy))
        assert detail["kind"] == "copy" and detail["gen"] == gen, detail
        # The old member's own rule, exercised: at-or-below held is dropped.
        assert rig.engine.apply_copy_reply(KEY, detail, owner=OWNER, held_gen=gen) == "in_sync"
    assert rig.owner_gen() == gen, "a legacy steady-state request must not bump"

    # An empty field is a claim of "nothing held": served, as before the fix.
    served = sync.owner_copy(rig.broker, link, {**legacy, "held_digest": ""})
    assert served["kind"] == "copy" and "value" in served

    # A held counter the ack ledger does not name is not evidence.
    stale = sync.owner_copy(rig.broker, link, {**legacy, "held": served["gen"] + 3})
    assert stale["kind"] == "copy"


def test_member_keeps_its_row_when_a_legacy_owner_re_serves_the_same_bytes(
    root: Path,
) -> None:
    """The member's half of "adopt, don't re-apply", against an older owner.

    An owner without the adopt arm re-serves an equal digest at ``held + 1``.
    The member adopts the counter and acks it, but does NOT rewrite the store:
    a rewrite inserts a new row and sweeps the old, and the new id is what every
    session's binding recorder read as an account change.
    """
    member_root = root / "member"
    member_root.mkdir()
    _declare_owner(member_root)
    server = _FakeServer(member_root, MEMBER, "member")
    engine = _engine(member_root, server, member_root=member_root)
    payload = _payload(VALUE_1)
    digest = sync.fingerprint(payload)

    def _legacy_owner(frame: dict[str, Any]) -> dict[str, Any]:
        if frame.get("kind") != "copy":
            return {"op": "ack", "detail": {"kind": "ack", "action": "noted"}}
        return {
            "op": "ack",
            "detail": {
                "kind": "copy",
                "key": KEY,
                "gen": int(frame.get("held") or 0) + 1,
                "digest": digest,
                "value_state": "present",
                "value": payload,
                "provenance": {"owner_device": OWNER, "owner_device_name": "owner"},
            },
        }

    server.reply = _legacy_owner
    assert engine._pull_blocking(OWNER, KEY, 0, "") == "applied"
    handle = _open_member_store(member_root)
    try:
        ids = [int(row.id) for row in handle.list_credentials(KEY, include_disabled=True)]
    finally:
        handle.close()
    for expected_gen in (2, 3):
        assert engine._pull_blocking(OWNER, KEY, 0, "") == "in_sync"
        applied = SyncState.load(NETWORK, member_root).applied_for(KEY)
        assert applied is not None and int(applied["gen"]) == expected_gen
    handle = _open_member_store(member_root)
    try:
        assert [int(row.id) for row in handle.list_credentials(KEY, include_disabled=True)] == ids
    finally:
        handle.close()
    acks = [frame for frame in server.frames if frame.get("kind") == "ack"]
    assert [int(frame["gen"]) for frame in acks] == [1, 2, 3]


def test_two_relays_steady_state_writes_no_copy_audit_rows(sync_mesh: Any) -> None:  # noqa: F811
    """Real relays, real links, the real tick step on the MEMBER side.

    ``credentials_sync_step(member, owner)`` is what the member's definitions
    syncer runs every 15 s; the owner's ``audit.jsonl`` is what the live
    evidence counted. After the first delivery, N member ticks must add zero
    ``credential.copy`` rows and leave the owner's generation where it was.
    """
    mesh = sync_mesh
    _run_exchange(mesh)
    _wait_for_owner_ack(mesh, VALUE_1)

    def _copies() -> int:
        return sum(
            1
            for row in mesh.a.audit.tail(10_000, network_id=mesh.network_id)
            if row.get("event") == "credential.copy"
        )

    def _owner_gen() -> int:
        row = SyncState.load(mesh.network_id, mesh.a.root).generation(KEY)
        return int((row or {}).get("gen") or 0)

    baseline_copies = _copies()
    baseline_gen = _owner_gen()
    rows_before = [int(row.id) for row in _member_rows(mesh)]
    token = ("placement", mesh.owner)
    for _ in range(TICKS):
        sync.credentials_sync_step(mesh.b, mesh.owner)
        assert net_fixtures.wait_for(
            lambda: token not in mesh.engine_b._pulling
        ), "the member's placement pull never settled"
    assert _copies() == baseline_copies, "steady-state ticks must not copy"
    assert _owner_gen() == baseline_gen, "steady-state ticks must not bump the generation"
    assert [int(row.id) for row in _member_rows(mesh)] == rows_before
