"""The credential sync engine: generations, acks, and the copy path (design S3).

WHY THESE CELLS, AND WHY THEY ARE SHAPED THIS WAY. The sync engine is the first
secret-COPYING code in the tree (``mesh-consent-provisioning.md`` §9.2's S3 row;
the slice holds for the operator's C5 sign-off). The properties the design pins
by name (§8.3) are, one per class:

* ``generation_monotonic`` — an older generation never overwrites a newer copy,
  and a counter reset on the owner converges by DIGEST, never by a value guess;
* ``copy_requires_active_holder`` — no copy to a non-member or to a removed
  member: the same check the broker reads, not a parallel one;
* ``sync_never_blocks_use`` — nothing on a use path awaits the sync, and the
  tick step ENQUEUES: the dial happens off the caller's thread, once per member;
* the ack ledger is per-peer — one holder's silence never moves another's chip,
  and it never moves the owner's own rows at all.

THE LAYERS, AND WHY EACH EXISTS. State/vocabulary cells need no relay; the
fake-link cells drive ``SyncEngine`` and the two handler entry points
(``owner_copy``/``member_announce``) with a scripted link, which is the only way
to pin a REFUSAL arm (real links refuse the same way, but constructing the bad
state through them would take a whole second device); and the two-relay cells
drive the product's own paths end to end — the share verb, the placement pull,
the tick step, the wire kinds — over two real relays with isolated roots, which
is the design's cheap primary rig (``mesh-credentials.md`` §9.2; the real
two-machine topology is S5's drill).

EVERY CELL BUILDS ITS OWN CONFIG ROOTS (the package conftest's rule): the
modules take ``root`` explicitly, and the one ambient read left — ``AuthStore``'s
default database path, which production resolves per relay process — is pinned
to the owner's root by the fixture, exactly as ``test_credentials_real_link``
does, while the member's store is reached through ``SyncEngine``'s
``auth_store_factory`` seam with an explicit ``db_path``.
"""

from __future__ import annotations

import stat
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.network import definitions, store
from local_operator.network.credentials import placement as placement_mod
from local_operator.network.credentials import sync
from local_operator.network.credentials.sync import SyncState
from local_operator.network.credentials.types import peer_int
from tests.unit.network import conftest as net_fixtures
from tests.unit.network.test_credentials_real_link import (  # noqa: F401
    _lop_network,
    _pull,
)
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — fixtures by import
    _pair,
    devices,
)

#: The provider the copy path is exercised with: any name works — the share verb
#: classifies it from the store row, and ``api-key-static`` is the copy-eligible
#: class S3 serves.
KEY = "stub-sync-api"
VALUE_1 = "sync-value-one"
VALUE_2 = "sync-value-two"

OWNER = "d_" + "a" * 32
MEMBER = "d_" + "b" * 32
OTHER = "d_" + "c" * 32
NETWORK = "n_" + "d" * 24


def _payload(value: str) -> dict[str, Any]:
    return {"type": "api_key", "source": "login", "key": value}


def _open_member_store(root: Path) -> Any:
    """The member's own store, with an EXPLICIT path — one process, two roots.

    Production resolves each relay's database from its own process environment,
    so this seam exists for the same reason ``test_credentials_real_link`` keeps
    the ambient root on the owner: two config roots in one process need one of
    the two stores addressed by its path, and the member is the one that is.
    """
    from local_operator.providers.auth_store import AuthStore

    return AuthStore(db_path=root / "auth.db", config_dir=root)


# ---------------------------------------------------------------------------
# State and vocabulary
# ---------------------------------------------------------------------------


def test_generation_counts_value_changes_not_reads(root: Path) -> None:
    """``gen`` moves when the VALUE moves — and only then.

    A digest change bumps; the same digest at a moved ``updated_at`` bumps too
    (the design's diff is over the cheap metadata, and a store touch is a fact
    the owner observed — the member's digest comparison keeps such an announce
    from transferring anything). Re-reading an unchanged value never bumps: a
    counter that moved on every tick would re-announce forever.
    """
    with sync.mutate(NETWORK, root) as state:
        first = sync._ensure_generation(state, KEY, {"digest": "sha256:aaa", "updated_at": 100})
        assert first == 1
        again = sync._ensure_generation(state, KEY, {"digest": "sha256:aaa", "updated_at": 100})
        assert again == 1, "an unchanged read must not bump the generation"
        moved = sync._ensure_generation(state, KEY, {"digest": "sha256:bbb", "updated_at": 100})
        assert moved == 2, "a changed digest must bump"
        touched = sync._ensure_generation(state, KEY, {"digest": "sha256:bbb", "updated_at": 101})
        assert touched == 3, "a moved updated_at is the diff the design names"


def test_the_ack_ledger_is_per_peer_and_never_regresses(root: Path) -> None:
    """``{device: {key: {gen, digest, at}}}``: one holder's facts never move another's."""
    with sync.mutate(NETWORK, root) as state:
        state.record_ack(MEMBER, KEY, gen=2, digest="sha256:one", at=100.0)
        state.record_ack(OTHER, KEY, gen=1, digest="sha256:one", at=100.0)
        # A late ack for an older generation must not regress the ledger — the
        # owner would re-announce work the member has already done.
        state.record_ack(MEMBER, KEY, gen=1, digest="sha256:zero", at=101.0)
        member_row = state.ack_for(MEMBER, KEY)
        other_row = state.ack_for(OTHER, KEY)
        assert member_row is not None and peer_int(member_row["gen"]) == 2
        assert other_row is not None and peer_int(other_row["gen"]) == 1
        assert state.ack_for(MEMBER, "another-key") is None


def test_the_segment_speaks_the_design_vocabulary() -> None:
    """§5.2's two forms, plus the never-acked one; nothing is invented."""
    current = {"gen": 7, "digest": "sha256:seven"}
    assert (
        sync.sync_segment(acked={"gen": 7, "digest": "sha256:seven", "at": 0}, current=current)
        == "synced (gen 7)"
    )
    at = time.time()
    stamp = time.strftime("%H:%M", time.localtime(at))
    assert (
        sync.sync_segment(acked={"gen": 6, "digest": "sha256:six", "at": at}, current=current)
        == f"stale (gen 6 of 7, last acked {stamp})"
    )
    assert sync.sync_segment(acked=None, current=current) == "not yet synced (gen 7)"
    assert sync.sync_segment(acked=None, current=None) == ""
    assert sync.sync_segment(acked=None, current={"gen": 0}) == ""


def test_copies_by_class_is_the_one_table() -> None:
    """The mechanism mapping of §2.1: static classes copy; rotating ones broker.

    A kind outside the copy table must NOT copy — the closed direction — because
    the rotating classes are broker-only by measured construction (one rotating
    refresh token raced by two hosts is the PR-24 failure the design cites).
    """
    assert sync.copies_by_class("api-key-static")
    for kind in ("oauth-rotating", "mcp-rotating", "github-app", "radient", ""):
        assert not sync.copies_by_class(kind), kind


def test_the_sync_document_lands_private(root: Path) -> None:
    with sync.mutate(NETWORK, root) as state:
        state.record_ack(MEMBER, KEY, gen=1, digest="sha256:one", at=1.0)
    path = sync.sync_path(NETWORK, root)
    assert path.exists()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600, oct(path.stat().st_mode)


# ---------------------------------------------------------------------------
# The engine with a scripted link
# ---------------------------------------------------------------------------


class _FakeLink:
    """One scripted link: records request frames, answers with ``reply``."""

    def __init__(self, server: "_FakeServer") -> None:
        self._server = server

    def request(self, frame: dict[str, Any], timeout: float = 0) -> Any:
        self._server.frames.append(frame)
        if self._server.gate is not None:
            self._server.gate.wait(10.0)
        reply = self._server.reply
        if callable(reply):
            return reply(frame)
        return reply


class _FakeServer:
    """The seams ``SyncEngine`` uses, with a scripted link instead of a relay."""

    def __init__(self, root: Path, device_id: str, name: str) -> None:
        self.root = root
        self.identity = SimpleNamespace(device_id=device_id, name=name)
        self.audit: Any = None
        self.frames: list[dict[str, Any]] = []
        self.reply: Any = None
        self.gate: threading.Event | None = None
        self.dial_threads: list[str] = []
        self._req = 0

    def _ensure_link(self, device_id: str) -> Any:
        self.dial_threads.append(threading.current_thread().name)
        return _FakeLink(self)

    def _next_relay_req(self) -> int:
        self._req += 1
        return self._req


def _engine(
    root: Path,
    server: _FakeServer,
    *,
    device: str = MEMBER,
    member_root: Path | None = None,
) -> sync.SyncEngine:
    return sync.SyncEngine(
        server,
        root=root,
        self_device=device,
        self_device_name="member",
        audit=server.audit,
        auth_store_factory=(
            (lambda: _open_member_store(member_root)) if member_root is not None else None
        ),
    )


def _declare_owner(
    root: Path, *, kind: str = "api-key-static", holders: tuple[str, ...] = (MEMBER,)
) -> None:
    with placement_mod.mutate(NETWORK, root, self_device=OWNER) as document:
        document.declare(KEY, owner_device=OWNER, provider=KEY, kind=kind, by=OWNER)
        for holder in holders:
            document.grant(KEY, holder, scope="device", by=OWNER)


def _seed_network(root: Path, *, member_lifecycle: Any = "active") -> None:
    """A minimal network record on ``root``: self=OWNER (admin), MEMBER (drive).

    The announce side reads the member table (the §8.1 withholding rule), so a
    cell that expects the exchange to DIAL needs an active member row; the
    two-relay fixture gets its rows from a real pairing instead.
    """
    from local_operator.network import types as net_types

    record = net_types.NetworkRecord(
        network_id=NETWORK,
        name="sync-test",
        created_by=OWNER,
        self_device_id=OWNER,
        self_role="admin",
        self_capabilities=sorted(net_types.capabilities_for_role("admin")),
        listen={"address": "127.0.0.1", "port": 1, "advertised": []},
    )
    record.members.append(
        net_types.MemberRecord(
            device_id=MEMBER,
            name="member",
            kind="device",
            lifecycle=member_lifecycle,
            role="drive",
            capabilities=sorted(net_types.capabilities_for_role("drive")),
            endpoints=["127.0.0.1:1"],
        )
    )
    store.save(record, root)


def test_a_newer_generation_applies_and_the_store_holds_the_value(root: Path) -> None:
    """The apply path, end to end minus transport: pull, verify, write, ack."""
    owner_root = root / "owner"
    member_root = root / "member"
    owner_root.mkdir()
    member_root.mkdir()
    _declare_owner(member_root)
    server = _FakeServer(member_root, MEMBER, "member")
    engine = _engine(member_root, server, member_root=member_root)
    digest = sync.fingerprint(_payload(VALUE_1))
    server.reply = {
        "op": "ack",
        "detail": {
            "kind": "copy",
            "key": KEY,
            "gen": 4,
            "digest": digest,
            "value_state": "present",
            "value": _payload(VALUE_1),
            "provenance": {"owner_device": OWNER, "owner_device_name": "owner"},
        },
    }
    engine._pull_blocking(OWNER, KEY, 4, digest)
    store = _open_member_store(member_root)
    try:
        rows = store.list_credentials(KEY)
        assert len(rows) == 1, "exactly one copy row after one apply"
        assert rows[0].data.get("key") == VALUE_1
        assert rows[0].data.get(sync.MESH_ORIGIN_KEY, {}).get("owner_device") == OWNER
    finally:
        store.close()
    state = SyncState.load(NETWORK, member_root)
    applied = state.applied_for(KEY)
    assert applied is not None and peer_int(applied["gen"]) == 4
    # The member acked what it holds — the second frame on the same link.
    ack_frames = [frame for frame in server.frames if frame.get("kind") == "ack"]
    assert ack_frames and peer_int(ack_frames[0]["gen"]) == 4


def test_replacing_a_copy_never_accumulates_rows(root: Path) -> None:
    """Change → converge: the superseded copy row is swept, not left behind."""
    member_root = root / "member"
    member_root.mkdir()
    _declare_owner(member_root)
    server = _FakeServer(member_root, MEMBER, "member")
    engine = _engine(member_root, server, member_root=member_root)
    value_live = {"current": VALUE_1}

    def _reply(frame: dict[str, Any]) -> dict[str, Any]:
        if frame.get("kind") != "copy":
            return {"op": "ack", "detail": {"kind": "ack", "action": "noted"}}
        payload = _payload(value_live["current"])
        return {
            "op": "ack",
            "detail": {
                "kind": "copy",
                "key": KEY,
                "gen": 1 if value_live["current"] == VALUE_1 else 2,
                "digest": sync.fingerprint(payload),
                "value_state": "present",
                "value": payload,
                "provenance": {"owner_device": OWNER, "owner_device_name": "owner"},
            },
        }

    server.reply = _reply
    engine._pull_blocking(OWNER, KEY, 1, sync.fingerprint(_payload(VALUE_1)))
    value_live["current"] = VALUE_2
    engine._pull_blocking(OWNER, KEY, 2, sync.fingerprint(_payload(VALUE_2)))
    store = _open_member_store(member_root)
    try:
        rows = store.list_credentials(KEY, include_disabled=True)
        assert len(rows) == 1, "a replaced copy must not leave its predecessor"
        assert rows[0].data.get("key") == VALUE_2
    finally:
        store.close()


def test_an_older_served_generation_never_rolls_a_copy_back(root: Path) -> None:
    """``generation_monotonic``: served <= held is dropped, whatever the digest says."""
    member_root = root / "member"
    member_root.mkdir()
    _declare_owner(member_root)
    server = _FakeServer(member_root, MEMBER, "member")
    engine = _engine(member_root, server, member_root=member_root)
    with sync.mutate(NETWORK, member_root) as state:
        state.record_applied(
            KEY,
            gen=5,
            digest=sync.fingerprint(_payload(VALUE_1)),
            owner_device=OWNER,
            row_id=0,
            at=time.time(),
        )
    payload = _payload(VALUE_2)
    server.reply = {
        "op": "ack",
        "detail": {
            "kind": "copy",
            "key": KEY,
            "gen": 5,  # NOT newer than held: a rollback attempt
            "digest": sync.fingerprint(payload),
            "value_state": "present",
            "value": payload,
            "provenance": {"owner_device": OWNER, "owner_device_name": "owner"},
        },
    }
    engine._pull_blocking(OWNER, KEY, 6, sync.fingerprint(payload))
    store = _open_member_store(member_root)
    try:
        assert store.list_credentials(KEY) == [], "an older generation must not apply"
    finally:
        store.close()
    applied = SyncState.load(NETWORK, member_root).applied_for(KEY)
    assert applied is not None and peer_int(applied["gen"]) == 5


def test_a_digest_mismatch_is_refused_and_audited(root: Path) -> None:
    """The digest PIN: a reply whose bytes do not match its digest is not applied."""
    member_root = root / "member"
    member_root.mkdir()
    _declare_owner(member_root)
    server = _FakeServer(member_root, MEMBER, "member")
    audit_rows: list[Any] = []

    class _Audit:
        # The real ``AuditLog.record`` takes ONE ``AuditEvent`` object (the
        # broker's ``_audit`` passes one), so the stub models that shape rather
        # than a kwargs call nothing in the tree makes.
        def record(self, entry: Any) -> None:
            audit_rows.append(entry)

    server.audit = _Audit()
    engine = _engine(member_root, server, member_root=member_root)
    server.reply = {
        "op": "ack",
        "detail": {
            "kind": "copy",
            "key": KEY,
            "gen": 2,
            "digest": "sha256:" + "0" * 64,  # not the payload's digest
            "value_state": "present",
            "value": _payload(VALUE_1),
            "provenance": {"owner_device": OWNER, "owner_device_name": "owner"},
        },
    }
    engine._pull_blocking(OWNER, KEY, 2, "sha256:" + "0" * 64)
    store = _open_member_store(member_root)
    try:
        assert store.list_credentials(KEY) == [], "digest mismatch must not write"
    finally:
        store.close()
    assert any(
        str(getattr(row, "event", "")) == "credential.copy_refused"
        and str((getattr(row, "detail", None) or {}).get("reason")) == "digest_mismatch"
        for row in audit_rows
    ), audit_rows


def test_an_equal_digest_adopts_the_owners_counter_without_a_transfer(root: Path) -> None:
    """The reset edge: equal bytes adopt the owner's counter DOWN, never the value up.

    An owner whose sync state was lost renumbers from 1 while the member holds 5.
    Equal digests prove the values are identical, so the member moves its counter
    to the owner's — a counter resynchronisation, not a value rollback — acks it,
    and the next real change converges normally.
    """
    member_root = root / "member"
    member_root.mkdir()
    _declare_owner(member_root)
    server = _FakeServer(member_root, MEMBER, "member")
    engine = _engine(member_root, server, member_root=member_root)
    digest = sync.fingerprint(_payload(VALUE_1))
    with sync.mutate(NETWORK, member_root) as state:
        state.record_applied(
            KEY, gen=5, digest=digest, owner_device=OWNER, row_id=0, at=time.time()
        )
    engine._pull_blocking(OWNER, KEY, 2, digest)
    assert not [
        frame for frame in server.frames if frame.get("kind") == "copy"
    ], "equal digests must not transfer the value again"
    applied = SyncState.load(NETWORK, member_root).applied_for(KEY)
    assert applied is not None and peer_int(applied["gen"]) == 2
    acks = [frame for frame in server.frames if frame.get("kind") == "ack"]
    assert acks and peer_int(acks[0]["gen"]) == 2


def test_the_tick_step_enqueues_off_the_caller_and_guards_one_exchange(root: Path) -> None:
    """§5.4 (2): the step never dials on the caller's thread, and never stacks.

    The caller here is the definitions syncer's own thread in production; the
    property is that the dial happens on the engine's loop (whose executor
    threads are named ``mesh-broker-io``-prefixed), and that a second step for
    the same member while one exchange is in flight answers WITHOUT a second
    dial — the in-flight guard that stands in for a second set of floors.
    """
    owner_root = root / "owner"
    owner_root.mkdir()
    _declare_owner(owner_root)
    _seed_network(owner_root)
    server = _FakeServer(owner_root, OWNER, "owner")
    engine = _engine(owner_root, server, device=OWNER)

    class _StoreRow:
        id = 1
        credential_type = "api_key"
        data = _payload(VALUE_1)
        updated_at = 100

    class _Store:
        def list_credentials(self, provider: str, include_disabled: bool = False) -> list[Any]:
            return [_StoreRow()]

        def close(self) -> None:  # pragma: no cover - nothing to close
            pass

    engine._auth_store_factory = lambda: _Store()
    gate = threading.Event()
    server.gate = gate
    server.reply = {"op": "ack", "detail": {"kind": "ack", "key": KEY, "action": "noted"}}
    first = engine.enqueue_owner_exchange(MEMBER)
    assert first == "scheduled"
    assert net_fixtures.wait_for(lambda: bool(server.dial_threads)), "no dial was enqueued"
    second = engine.enqueue_owner_exchange(MEMBER)
    assert second == "in_flight", "a second exchange must not stack while one is running"
    assert server.dial_threads[0] != threading.current_thread().name
    gate.set()
    # Once the gated exchange settles, the in-flight guard clears: the same
    # member is schedulable again, which is what "no second set of floors"
    # means — the definitions syncer's own floors pace the next call.
    assert net_fixtures.wait_for(
        lambda: engine.enqueue_owner_exchange(MEMBER) == "scheduled"
    ), "the in-flight guard never cleared after the exchange settled"


def test_the_step_is_a_no_op_for_a_device_with_nothing_to_announce(root: Path) -> None:
    owner_root = root / "owner"
    owner_root.mkdir()
    server = _FakeServer(owner_root, OWNER, "owner")
    engine = _engine(owner_root, server, device=OWNER)
    assert engine.enqueue_owner_exchange(MEMBER) == "in_sync"
    assert server.dial_threads == []


def test_the_copy_is_withheld_from_a_removed_member(root: Path) -> None:
    """§8.3: ``copy_requires_active_holder`` — no record, no active member, no copy.

    Without a network record this device cannot verify the caller is an ACTIVE
    member, and the copy path refuses rather than trusting the link alone. The
    positive arm (an active holder IS served) is the two-relay cell below.
    """
    from local_operator.network.credentials.owner import MeshCredentialBroker

    owner_root = root / "owner"
    owner_root.mkdir()
    _declare_owner(owner_root)
    store_handle = _open_member_store(owner_root)
    store_handle.upsert_credential(KEY, _payload(VALUE_1))
    broker = MeshCredentialBroker(
        root=owner_root,
        self_device=OWNER,
        self_device_name="owner",
        network_id=NETWORK,
        audit=None,
        auth_store=store_handle,
    )
    link = SimpleNamespace(device_id=MEMBER)
    frame = {"kind": "copy", "key": KEY, "gen": 1, "held": 0, "from_device": MEMBER}
    detail = sync.owner_copy(broker, link, frame)
    assert detail.get("kind") == "error" and detail.get("code") == "member_not_active", detail
    store_handle.close()


def test_member_refusals_are_named(root: Path) -> None:
    """The member-side announce arms, each by its own code."""
    member_root = root / "member"
    member_root.mkdir()
    server = _FakeServer(member_root, MEMBER, "member")
    engine = _engine(member_root, server, member_root=member_root)
    digest = sync.fingerprint(_payload(VALUE_1))

    def _announce(link: Any = None, **fields: Any) -> dict[str, Any]:
        frame = {
            "kind": "announce",
            "from_device": OWNER,
            "key": KEY,
            "gen": 1,
            "digest": digest,
            "value_state": "present",
        }
        frame.update(fields)
        return engine.on_announce(link or SimpleNamespace(device_id=OWNER), frame)

    assert _announce().get("code") == "unknown_key"  # no placement at all
    _declare_owner(member_root)
    assert _announce(key="not-shared").get("code") == "unknown_key"
    # An announcement from a device that does not own the key HERE: the link's
    # authenticated id is what is checked, never the frame's claim.
    other_link = SimpleNamespace(device_id=OTHER)
    assert _announce(link=other_link, from_device=OTHER).get("code") == "not_owner"
    # Copyable but this device is not a holder of it.
    with placement_mod.mutate(NETWORK, member_root, self_device=OWNER) as document:
        document.declare(
            "other-key",
            owner_device=OWNER,
            provider="other-key",
            kind="api-key-static",
            by=OWNER,
        )
    assert _announce(key="other-key").get("code") == "not_a_holder"
    # Held but NOT copy-eligible by class.
    with placement_mod.mutate(NETWORK, member_root, self_device=OWNER) as document:
        document.declare(
            "rotating-key",
            owner_device=OWNER,
            provider="rotating-key",
            kind="oauth-rotating",
            by=OWNER,
        )
        document.grant("rotating-key", MEMBER, scope="device", by=OWNER)
    assert _announce(key="rotating-key").get("code") == "not_copyable"


# ---------------------------------------------------------------------------
# Two relays, isolated roots: change -> usable, ack ledger, offline catch-up
# ---------------------------------------------------------------------------


@pytest.fixture()
def sync_mesh(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Owner A (admin, holds the api key) and member B (drive), both relays up."""
    from local_operator.providers.auth_store import AuthStore

    pair_devices = request.getfixturevalue("devices")
    server_a, server_b, host, port = pair_devices
    record, _host, _port = _pair(pair_devices, monkeypatch, role="drive")
    # The ambient root is the OWNER's from here on (see the module docstring).
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_a.root))
    # B must be able to DIAL A: the pairing leaves A's row on B with no endpoint.
    with store.mutate(record.network_id, server_b.root) as copy:
        owner_row = copy.member(server_a.identity.device_id)
        assert owner_row is not None
        owner_row.endpoints = [f"{host}:{port}"]
        store.save(copy, server_b.root)
    server_b.start()
    # A must be able to DIAL B for the announce push: B joined before it was
    # listening, so A's record of B carries whatever B advertised then.
    with store.mutate(record.network_id, server_a.root) as copy:
        member_row = copy.member(server_b.identity.device_id)
        assert member_row is not None
        member_row.endpoints = [f"127.0.0.1:{server_b.settings.port}"]
        store.save(copy, server_a.root)

    auth_a = AuthStore(config_dir=server_a.root)
    auth_a.upsert_credential(KEY, _payload(VALUE_1))

    assert _lop_network("credential", "share", KEY, "--with", server_b.identity.name) == 0
    pulled = _pull(SimpleNamespace(b=server_b, borrower=server_b.identity.device_id))
    assert KEY in pulled.get("changed", []), pulled

    engine_b = sync.SyncEngine(
        server_b,
        root=server_b.root,
        self_device=server_b.identity.device_id,
        self_device_name=server_b.identity.name,
        audit=server_b.audit,
        auth_store_factory=lambda: _open_member_store(server_b.root),
    )
    setattr(server_b, sync._SYNC_ATTR, engine_b)
    engine_a = sync.sync_for_relay(server_a)
    assert engine_a is not None
    mesh = SimpleNamespace(
        a=server_a,
        b=server_b,
        engine_a=engine_a,
        engine_b=engine_b,
        network_id=record.network_id,
        auth_a=auth_a,
        owner=server_a.identity.device_id,
        member=server_b.identity.device_id,
    )
    try:
        yield mesh
    finally:
        auth_a.close()
        server_b.stop()


def _member_rows(mesh: Any) -> list[Any]:
    store_handle = _open_member_store(mesh.b.root)
    try:
        return store_handle.list_credentials(KEY)
    finally:
        store_handle.close()


def _run_exchange(mesh: Any) -> None:
    """One tick step, as the definitions syncer would run it, then wait for the pull."""
    code = sync.credentials_sync_step(mesh.a, mesh.member)
    assert code in ("scheduled", "in_sync", "in_flight"), code
    assert net_fixtures.wait_for(
        lambda: SyncState.load(mesh.network_id, mesh.b.root).applied_for(KEY) is not None
    ), "the member never recorded an applied copy"


def test_a_change_reaches_the_holder_and_the_ack_ledger_records_it(sync_mesh: Any) -> None:
    """The headline (§5.3): change -> usable within the budget; the ack ledger follows.

    The budget here is one exchange, not one minute: both devices are in one
    process, in the same host, but the WORK is the product's own — the share
    verb's placement, the tick step's announce, the member's real dial, the
    apply into B's own store, the ack back. The number the design quotes (≈75 s
    p95 on a reachable member) is the CADENCE's; a manual tick is strictly
    faster and the assertion below is deliberately about convergence, not about
    wall time (see AGENTS.md, "Wait on the event, never on the clock").
    """
    mesh = sync_mesh
    _run_exchange(mesh)

    rows = _member_rows(mesh)
    assert (
        len(rows) == 1 and rows[0].data.get("key") == VALUE_1
    ), "the member must hold the copy, usable, after one exchange"

    ledger = SyncState.load(mesh.network_id, mesh.a.root)
    current = ledger.generation(KEY)
    acked = ledger.ack_for(mesh.member, KEY)
    assert current is not None and current.get("digest") == sync.fingerprint(_payload(VALUE_1))
    assert acked is not None, "the owner never recorded the member's ack"
    assert peer_int(acked["gen"]) == peer_int(current["gen"])
    assert acked.get("digest") == current.get("digest")

    # A second exchange with nothing changed re-announces nothing and writes no
    # second copy: the ledger says synced, and the member stays at one row.
    sync.credentials_sync_step(mesh.a, mesh.member)
    time.sleep(0.2)
    assert len(_member_rows(mesh)) == 1

    # CHANGE -> converge. The owner's stored value moves; one exchange delivers
    # (the generation is computed BY the exchange, so there is nothing to wait on
    # before it runs).
    mesh.auth_a.upsert_credential(KEY, _payload(VALUE_2))
    _run_exchange_gen2(mesh)

    rows = _member_rows(mesh)
    assert len(rows) == 1, "the replaced copy must not accumulate rows"
    assert rows[0].data.get("key") == VALUE_2
    ledger = SyncState.load(mesh.network_id, mesh.a.root)
    current = ledger.generation(KEY)
    acked = ledger.ack_for(mesh.member, KEY)
    assert current is not None and acked is not None
    assert current.get("gen") == 2, current
    assert peer_int(acked["gen"]) == 2 and acked.get("digest") == current.get("digest")


def _run_exchange_gen2(mesh: Any) -> None:
    """One more tick, then wait for the member's record to show the NEW digest."""
    expected = sync.fingerprint(_payload(VALUE_2))
    code = sync.credentials_sync_step(mesh.a, mesh.member)
    assert code in ("scheduled", "in_sync", "in_flight"), code
    assert net_fixtures.wait_for(
        lambda: (
            (SyncState.load(mesh.network_id, mesh.b.root).applied_for(KEY) or {}).get("digest")
        )
        == expected
    ), "the member never converged on the new generation"


def test_offline_catch_up_never_invalidates_the_member_copy(sync_mesh: Any) -> None:
    """§5.5(a): a peer offline when the value changes keeps its copy and catches up.

    Nothing is queued and nothing is invalidated: while B is unreachable the
    announce simply fails and is recomputed on the next contact, and B's own
    store keeps serving the previous copy. On contact it converges — the same
    one exchange, no replay machinery.
    """
    mesh = sync_mesh
    _run_exchange(mesh)
    assert _member_rows(mesh)[0].data.get("key") == VALUE_1

    # B goes UNREACHABLE for A: the endpoint A would dial is dead and the live
    # link is closed, so the announce simply fails — the same state a sleeping
    # node is in.
    with store.mutate(mesh.network_id, mesh.a.root) as copy:
        member_row = copy.member(mesh.member)
        assert member_row is not None
        member_row.endpoints = ["127.0.0.1:9"]
        store.save(copy, mesh.a.root)
    for link in list(mesh.a.links.values()):
        if str(getattr(link, "device_id", "")) == mesh.member:
            link.close("test-offline")
    mesh.auth_a.upsert_credential(KEY, _payload(VALUE_2))
    code = sync.credentials_sync_step(mesh.a, mesh.member)
    assert code in ("scheduled", "in_sync", "in_flight"), code
    time.sleep(1.0)
    assert (
        _member_rows(mesh)[0].data.get("key") == VALUE_1
    ), "the offline member's copy must not be invalidated by the change"

    # B comes back: the endpoint it answers on is restored, and the next contact
    # recomputes the work — no queue, no replay.
    with store.mutate(mesh.network_id, mesh.a.root) as copy:
        member_row = copy.member(mesh.member)
        assert member_row is not None
        member_row.endpoints = [f"127.0.0.1:{mesh.b.settings.port}"]
        store.save(copy, mesh.a.root)
    _run_exchange_gen2(mesh)
    rows = _member_rows(mesh)
    assert len(rows) == 1 and rows[0].data.get("key") == VALUE_2


def test_a_removed_member_is_not_announced_to_and_is_refused(sync_mesh: Any) -> None:
    """§8.1/§8.3: the sync path withholds from a non-active member, both ways —
    and §5.5c/§4.3 (S4): the ENDING reaches the same member.

    The announce side: once B is deactivated in A's record no new generation is
    delivered. The ending side: a wipe notice is NOT a copy, so it does reach
    the removed member — B deletes its copy by provenance and acks, and A's
    ledger records the ending. The copy side: a copy request from the removed
    member is refused, audited, and changes nothing.
    """
    mesh = sync_mesh
    _run_exchange(mesh)
    before = SyncState.load(mesh.network_id, mesh.b.root).applied_for(KEY)
    assert before is not None

    with store.mutate(mesh.network_id, mesh.a.root) as copy:
        member_row = copy.member(mesh.member)
        assert member_row is not None
        member_row.lifecycle = "expired"
        store.save(copy, mesh.a.root)
    mesh.auth_a.upsert_credential(KEY, _payload(VALUE_2))
    code = sync.credentials_sync_step(mesh.a, mesh.member)
    assert code in ("scheduled", "in_sync"), code
    # THE ENDING, NOT THE NEW VALUE: B's copy is deleted by provenance and its
    # applied sidecar cleared; the generation A bumped never arrives.
    assert net_fixtures.wait_for(
        lambda: SyncState.load(mesh.network_id, mesh.b.root).applied_for(KEY) is None
    ), "the removed member's copy was not wiped"
    # And the wipe-ack lands in A's ledger as an ending, not a held copy.
    assert net_fixtures.wait_for(
        lambda: bool(
            (SyncState.load(mesh.network_id, mesh.a.root).ack_for(mesh.member, KEY) or {}).get(
                "wiped"
            )
        )
    ), "the wipe was not acknowledged to the owner"

    # The refusal arm: a copy request from the removed member is refused.
    from local_operator.network.credentials import owner as owner_mod

    broker = owner_mod.broker_for_relay(mesh.a)
    assert broker is not None
    link = SimpleNamespace(device_id=mesh.member)
    frame = {
        "kind": "copy",
        "key": KEY,
        "gen": 2,
        "held": 1,
        "from_device": mesh.member,
    }
    detail = sync.owner_copy(broker, link, frame)
    assert detail.get("code") == "member_not_active", detail


def test_the_sync_step_rides_the_definitions_seam(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The carrier is the seam, not a second cadence: construction registers it."""
    from local_operator.network import audit as audit_mod
    from local_operator.network import identity as identity_mod
    from tests.unit.network.test_relay_e2e import serve_shaped_relay

    root_a = root / "seam"
    ident = identity_mod.mint(root_a, name="seam-a")
    serve_shaped_relay(root_a, monkeypatch, identity=ident, audit=audit_mod.AuditLog(root_a))
    assert (
        sync.credentials_sync_step in definitions._tick_steps()
    ), "credentials.install must register the step on the definitions syncer"


# ---------------------------------------------------------------------------
# Review round 1's remediation (R1's ceiling, Q-2's ledger, N2's rows)
# ---------------------------------------------------------------------------


def test_a_ceiling_held_is_refused_and_never_adopted(root: Path) -> None:
    """R1 (review round 1): a forged ``held`` at the ceiling is refused by name.

    Reproduced on the pre-fix head: one ``copy`` frame with ``held: 2**53``
    recorded the owner's generation AT the ceiling; every later bump clamped
    there and every reply was dropped by a member at the ceiling — a permanent
    liveness break for one key from one frame. The fix refuses (and records
    nothing), so the key keeps bumping; the corrupt-state half is the load cell
    below.
    """
    from local_operator.network.credentials.owner import MeshCredentialBroker

    owner_root = root / "owner"
    owner_root.mkdir()
    _declare_owner(owner_root)
    _seed_network(owner_root)
    store_handle = _open_member_store(owner_root)
    store_handle.upsert_credential(KEY, _payload(VALUE_1))
    broker = MeshCredentialBroker(
        root=owner_root,
        self_device=OWNER,
        self_device_name="owner",
        network_id=NETWORK,
        audit=None,
        auth_store=store_handle,
    )
    link = SimpleNamespace(device_id=MEMBER)
    for forged in (2**53, sync.GEN_SAFE_MAX):
        detail = sync.owner_copy(
            broker,
            link,
            {"kind": "copy", "key": KEY, "gen": 1, "held": forged, "from_device": MEMBER},
        )
        assert detail.get("code") == "generation_out_of_range", detail
    assert (
        SyncState.load(NETWORK, owner_root).generation(KEY) is None
    ), "a refused held must record nothing"
    # The key is NOT frozen: a normal request serves gen 1, and a later change bumps.
    detail = sync.owner_copy(
        broker, link, {"kind": "copy", "key": KEY, "gen": 0, "held": 0, "from_device": MEMBER}
    )
    assert detail.get("kind") == "copy" and detail.get("gen") == 1, detail
    store_handle.upsert_credential(KEY, _payload(VALUE_2))
    detail = sync.owner_copy(
        broker, link, {"kind": "copy", "key": KEY, "gen": 1, "held": 1, "from_device": MEMBER}
    )
    assert detail.get("kind") == "copy" and detail.get("gen") == 2, detail
    store_handle.close()


def test_a_corrupt_generation_row_is_dropped_not_obeyed(root: Path) -> None:
    """R1's corrupt-state path: rows at/above the bound load as ABSENT.

    ``peer_int`` clamps an inflated value to ``2**53`` rather than rejecting
    it, so a corrupt ``applied`` row made the member drop every reply, and a
    corrupt generation row froze the owner. The load now drops such rows (the
    load docstring's "unreadable falls back to a fresh one"), and the member
    heals by RE-PULLING from zero — both halves asserted here.
    """
    owner_root = root / "owner"
    owner_root.mkdir()
    with sync.mutate(NETWORK, owner_root) as state:
        state.record_generation(KEY, sync.GEN_SAFE_MAX, "sha256:" + "a" * 64, 1)
    assert SyncState.load(NETWORK, owner_root).generation(KEY) is None

    member_root = root / "member"
    member_root.mkdir()
    _declare_owner(member_root)
    with sync.mutate(NETWORK, member_root) as state:
        state.record_applied(
            KEY,
            gen=sync.GEN_CEILING,
            digest="sha256:" + "b" * 64,
            owner_device=OWNER,
            row_id=7,
            at=1.0,
        )
        state.record_ack(MEMBER, KEY, gen=2**53, digest="sha256:" + "c" * 64, at=1.0)
    corrupt = SyncState.load(NETWORK, member_root)
    assert corrupt.applied_for(KEY) is None
    assert corrupt.ack_for(MEMBER, KEY) is None

    # The member heals by re-pulling from zero (the dropped row reads as held 0).
    server = _FakeServer(member_root, MEMBER, "member")
    engine = _engine(member_root, server, member_root=member_root)
    payload = _payload(VALUE_1)
    digest = sync.fingerprint(payload)
    server.reply = {
        "op": "ack",
        "detail": {
            "kind": "copy",
            "key": KEY,
            "gen": 1,
            "digest": digest,
            "value_state": "present",
            "value": payload,
            "provenance": {"owner_device": OWNER, "owner_device_name": "owner"},
        },
    }
    engine._pull_blocking(OWNER, KEY, 1, digest)
    applied = SyncState.load(NETWORK, member_root).applied_for(KEY)
    assert applied is not None and applied.get("gen") == 1, applied
    store = _open_member_store(member_root)
    try:
        assert store.list_credentials(KEY)[0].data.get("key") == VALUE_1
    finally:
        store.close()


def test_an_equal_digest_ack_follows_the_reset_counter_down(root: Path) -> None:
    """Q-2 (review round 1): the ledger follows the member DOWN on a reset.

    After the owner's generation row is lost while acks survive, the member
    adopts the fresh counter down and acks it. Refusing that ack (the old
    monotonic rule) left ``stale (gen 4 of 1)`` and a doctor FAIL forever, and
    re-announced every cycle; an equal digest is proof no value moved, so the
    ledger follows — and a lower gen with a DIFFERENT digest still cannot
    regress it.
    """
    digest = "sha256:" + "d" * 64
    with sync.mutate(NETWORK, root) as state:
        state.record_generation(KEY, 1, digest, 5)
        state.record_ack(MEMBER, KEY, gen=4, digest=digest, at=100.0)
        state.record_ack(MEMBER, KEY, gen=1, digest=digest, at=200.0)
        acked = state.ack_for(MEMBER, KEY)
        assert acked is not None and acked.get("gen") == 1, acked
        assert sync._ack_matches(
            acked, 1, digest
        ), "the announce gate must read the reset as synced"
        state.record_ack(MEMBER, KEY, gen=0, digest="sha256:" + "e" * 64, at=300.0)
        kept = state.ack_for(MEMBER, KEY)
        assert kept is not None and kept.get("gen") == 1, "a different digest must not regress"


def test_member_audit_rows_land_with_actor_and_network(root: Path) -> None:
    """N2 (review round 1): the member's rows carry actor/network ON DISK.

    ``_audit_row`` used to leave ``actor`` at the dataclass default ("self")
    and pass no network id, where the owner's ``credential.copy`` row carries
    both. Read from the real ``audit.jsonl`` rather than a fake recorder: the
    writer's per-event whitelist drops a field it does not know, so a recorder
    would agree with a row that never landed.
    """
    from local_operator.network.audit import AuditLog

    member_root = root / "member"
    member_root.mkdir()
    _declare_owner(member_root)
    server = _FakeServer(member_root, MEMBER, "member")
    server.audit = AuditLog(root=member_root)
    engine = _engine(member_root, server, member_root=member_root)
    payload = _payload(VALUE_1)
    digest = sync.fingerprint(payload)
    server.reply = {
        "op": "ack",
        "detail": {
            "kind": "copy",
            "key": KEY,
            "gen": 2,
            "digest": "sha256:" + "0" * 64,
            "value_state": "present",
            "value": payload,
            "provenance": {"owner_device": OWNER, "owner_device_name": "owner"},
        },
    }
    engine._pull_blocking(OWNER, KEY, 2, "sha256:" + "0" * 64)
    server.reply = {
        "op": "ack",
        "detail": {
            "kind": "copy",
            "key": KEY,
            "gen": 2,
            "digest": digest,
            "value_state": "present",
            "value": payload,
            "provenance": {"owner_device": OWNER, "owner_device_name": "owner"},
        },
    }
    engine._pull_blocking(OWNER, KEY, 2, digest)
    rows = [
        row
        for row in server.audit.tail(50, network_id=NETWORK)
        if str(row.get("event") or "").startswith("credential.copy")
    ]
    events = {row["event"]: row for row in rows}
    assert set(events) == {"credential.copy_refused", "credential.copy_applied"}, rows
    refused = events["credential.copy_refused"]
    assert refused["actor"] == MEMBER and refused["network_id"] == NETWORK, refused
    assert refused["detail"] == {
        "credential_key": KEY,
        "act": MEMBER,
        "sub": OWNER,
        "reason": "digest_mismatch",
    }, refused
    applied = events["credential.copy_applied"]
    assert applied["actor"] == MEMBER and applied["network_id"] == NETWORK, applied
    assert applied["detail"]["act"] == MEMBER and applied["detail"]["sub"] == OWNER
    assert applied["detail"]["gen"] == 2, applied
    server.audit.close()
