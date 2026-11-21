"""G9-B follow-up: the notice across rotations, and one refusal path for every non-member.

#4394 shipped the "membership ended" notice with two known gaps, both closed here:

* P3 (security S2 / review R1-1): the listener's epoch check refused a member hello
  BEFORE the challenge, so a removed device two or more rotations behind never
  reached the tombstone probe and got silence until its offline hold ended. The
  check now runs at step 4 of ``verify_auth``, after the same one probe.
* The unknown-vs-known id timing oracle (#4394 QA O1): a never-seen id was refused
  ~6x slower than a tombstoned one. Measured root cause: ``handshake_refused`` is a
  DURABLE (fsynced) audit event, the refusal coalescer writes one only for a claimed
  id it has not seen in its window, and the row was written BEFORE the close — so
  the fsync landed inside the time a dialer observes. The relay now closes first.

Every cell runs A's REAL inbound handler (``RelayServer._run_inbound_handshake``)
over loopback TCP with synthetic identities; nothing touches a live mesh.
"""

from __future__ import annotations

import ast
import inspect
import logging
import socket
import statistics
import textwrap
import threading
import time
from pathlib import Path
from secrets import token_bytes
from typing import Any, cast

import pytest

from local_operator.network import handshake as hs_mod
from local_operator.network import identity, relay, store, wire
from tests.unit.network.test_removed_member_notice import (  # noqa: F401
    BSide,
    Net,
    _audit_rows,
    _DropSpy,
    _frame,
    _Lines,
    b_side,
    drops,
    net,
    verifies,
)

# ---------------------------------------------------------------------------
# Rig: a second rotation on A, and a dialer that can time A's answer
# ---------------------------------------------------------------------------


def _rotate_again(net: Net) -> None:  # noqa: F811
    """A removes D too: A moves 2 -> 3, so B (still at epoch 1) is TWO rotations behind."""
    record = store.load(net.network_id, net.root)
    # The first removal's rotation lock (30 s) still covers the record; the second
    # removal is "later inside the notice window", so step past it.
    record.rotation_lock_until = 0.0
    state = store.require_secrets(net.network_id, net.root)
    relay.remove_member(
        record,
        state,
        device_id=net.d_identity.device_id,
        by=net.a_identity.device_id,
        root=net.root,
    )
    assert store.load(net.network_id, net.root).epoch == 3


def _timed_dial(
    net: Net, signer: Any, *, claim: str = "", epoch: int = 1  # noqa: F811
) -> tuple[float, bytes, threading.Thread]:
    """Dial A's handler; milliseconds from our auth frame to A's EOF, and A's raw reply.

    The clock stops at EOF as the DIALER sees it — the only clock an attacker has —
    and A's thread is NOT joined first, so anything A does after its close (the audit
    row, now) is outside the measurement, as it is outside the attacker's.
    """
    with socket.create_server(("127.0.0.1", 0)) as listener:
        ours = socket.create_connection(listener.getsockname()[:2], timeout=5.0)
        theirs, _ = listener.accept()
    thread = threading.Thread(
        target=net.server._run_inbound_handshake,  # noqa: SLF001
        args=(theirs, ("127.0.0.1", 40000)),
        daemon=True,
    )
    thread.start()
    hs = hs_mod.Handshake.new(
        role="dialer",
        identity=signer,
        network_id=net.network_id,
        epoch=epoch,
        instance_id="i_dialer",
        session_protocol=relay._session_protocol(),  # noqa: SLF001
        capabilities=list(wire.LINK_CAPABILITIES),
    )
    if claim:
        hs.self_device_id = claim
    try:
        hs.send_hello(ours)
        reader = wire.FrameReader(ours)
        hs.read_challenge(reader, wire.deadline_in(5.0))
        key = wire.epoch_key(net.secret_e1, net.network_id, 1)
        ours.settimeout(5.0)
        started = time.perf_counter()
        hs.send_auth(ours, hs_mod.Credential("epoch", 1, key))
        chunks = [bytes(reader._buffer)]  # noqa: SLF001 — anything already buffered
        while True:
            chunk = ours.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
    finally:
        ours.close()
    return elapsed_ms, b"".join(chunks), thread


# ---------------------------------------------------------------------------
# P3: the notice reaches a device two or more rotations behind
# ---------------------------------------------------------------------------


def test_a_removed_device_two_rotations_behind_is_still_told(net: Net) -> None:  # noqa: F811
    """`rm laptop` then `rm phone`: the laptop, still at epoch 1, must learn at reconnect."""
    _rotate_again(net)
    hs, raw, error = net.dial(net.b_identity, epoch=1)
    assert error is None
    frame = _frame(raw)
    assert frame["net"] == "removed" and frame["epoch"] == 3
    # B verifies it exactly as it would a one-rotation notice.
    with pytest.raises(hs_mod.MembershipEnded) as ended:
        hs.read_welcome(cast(wire.FrameReader, _Lines([frame])), 0.0, net.expect())
    assert ended.value.epoch == 3
    # A's local record of the refusal is the stale-epoch one it always was.
    assert [code for code, _ in net.refusals] == [hs_mod.REASON_EPOCH]


@pytest.mark.parametrize("case", ["stranger", "tombstoned_id_random_key", "stale_live_member"])
def test_a_stale_non_proven_dialer_gets_silence_on_the_same_probe(
    net: Net, verifies: list[str], case: str  # noqa: F811
) -> None:
    """The reorder adds no oracle: every stale dialer that cannot prove a tombstone is silent."""
    _rotate_again(net)
    live = identity.mint(net.root.parent / "live-e", name="live-e")
    if case == "stranger":
        _hs, raw, _ = net.dial(net.c_identity, epoch=1)
    elif case == "tombstoned_id_random_key":
        _hs, raw, _ = net.dial(net.c_identity, claim=net.b_identity.device_id, epoch=1)
    else:
        with store.mutate(net.network_id, net.root) as record:
            relay.admit(
                record,
                device_id=live.device_id,
                public_key=live.public_key,
                name=live.name,
                role="drive",
                capabilities=["link"],
                added_by=net.a_identity.device_id,
                added_via="invite",
                root=net.root,
                persist=False,
            )
            store.save(record, net.root)
        # A real live member that missed two rotations: its own key, a stale epoch.
        _hs, raw, _ = net.dial(live, epoch=1)
    assert raw == b"", raw
    assert verifies == ["lop-mesh-auth-v1"], verifies
    assert [code for code, _ in net.refusals] == [hs_mod.REASON_EPOCH]
    assert net.server._removal_notices == {}  # noqa: SLF001


def test_the_stale_notice_keeps_the_rate_limit(net: Net) -> None:  # noqa: F811
    _rotate_again(net)
    _hs, first, _ = net.dial(net.b_identity, epoch=1)
    assert _frame(first)["net"] == "removed"
    _hs, second, _ = net.dial(net.b_identity, epoch=1)
    assert second == b"", "the per-key rate limit let a second stale notice out"


def test_the_stale_notice_keeps_the_seven_day_window(net: Net) -> None:  # noqa: F811
    _rotate_again(net)
    with store.mutate(net.network_id, net.root) as record:
        row = record.member(net.b_identity.device_id)
        assert row is not None and row.removed_at is not None
        row.removed_at = time.time() - relay.REMOVAL_NOTICE_WINDOW_S - 60
        store.save(record, net.root)
    _hs, raw, _ = net.dial(net.b_identity, epoch=1)
    assert raw == b""


# ---------------------------------------------------------------------------
# One refusal path: structure
# ---------------------------------------------------------------------------


def test_every_member_refusal_after_the_auth_frame_verifies_exactly_once(
    net: Net, verifies: list[str]  # noqa: F811
) -> None:
    """Every class an attacker can present runs ONE auth-label Ed25519 verify, then refuses.

    Covers both probe sites (step 4 ``epoch_stale``, step 6 ``not_a_member``) and the
    live-member wrong key (step 7 ``bad_signature``, the same verify function against
    the stored key). A class that skipped or repeated the verify would be measurably
    faster or slower than the rest.
    """
    fresh = identity.mint(net.root.parent / "fresh-r", name="fresh-r")
    cases: list[tuple[str, Any, str, int, str]] = [
        ("unknown_id", net.c_identity, "d_" + token_bytes(8).hex(), 1, hs_mod.REASON_MEMBER),
        ("random_key", fresh, "", 1, hs_mod.REASON_MEMBER),
        (
            "tombstoned_random_key",
            net.c_identity,
            net.b_identity.device_id,
            1,
            hs_mod.REASON_MEMBER,
        ),
        (
            "tombstoned_wrong_key",
            net.d_identity,
            net.b_identity.device_id,
            1,
            hs_mod.REASON_MEMBER,
        ),
        ("live_id_wrong_key", net.c_identity, net.d_identity.device_id, 1, hs_mod.REASON_SIGNATURE),
        ("stale_unknown_id", net.c_identity, "", 0, hs_mod.REASON_EPOCH),
        ("stale_tombstoned_random_key", net.c_identity, net.b_identity.device_id, 0, "epoch_stale"),
    ]
    # Spend the notice slot so the proven key, too, is a silent refusal below.
    net.dial(net.b_identity)
    cases.append(("tombstoned_proven_silent", net.b_identity, "", 1, hs_mod.REASON_MEMBER))
    for name, signer, claim, epoch, expected in cases:
        verifies.clear()
        net.refusals.clear()
        _hs, raw, error = net.dial(signer, claim=claim, epoch=epoch)
        assert error is None, (name, error)
        assert raw == b"", (name, raw)
        assert verifies == ["lop-mesh-auth-v1"], (name, verifies)
        assert [code for code, _ in net.refusals] == [expected], (name, net.refusals)


def test_the_unknown_id_path_reads_a_row_slot_and_verifies_against_the_process_key(
    net: Net, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """The stand-in key is ONE per process, and an unknown id verifies against exactly it."""
    keys: list[str] = []
    original = hs_mod._verify_signature  # noqa: SLF001

    def spy(public_key: str, signature: str, digest: bytes, **kwargs: Any) -> None:
        keys.append(public_key)
        original(public_key, signature, digest, **kwargs)

    monkeypatch.setattr(hs_mod, "_verify_signature", spy)
    net.dial(net.c_identity, claim="d_" + token_bytes(8).hex())
    net.dial(net.c_identity)
    dummy = hs_mod._dummy_public_key()  # noqa: SLF001
    assert keys == [dummy, dummy]
    assert hs_mod.prime_dummy_public_key() == dummy


def test_the_listener_closes_before_it_writes_the_refusal_row(net: Net) -> None:  # noqa: F811
    """The durable audit row (an fsync) must land AFTER the dialer's EOF.

    A blocked audit writer stands in for a slow disk: the dialer must still see its
    silent close, and the row must still be written once the writer is released.
    """
    release = threading.Event()
    entered = threading.Event()
    original = net.server.audit.record

    def slow_record(event: Any) -> None:
        if event.event == "handshake_refused":
            entered.set()
            release.wait(10)
        original(event)

    net.server.audit.record = slow_record  # type: ignore[method-assign]
    try:
        elapsed_ms, raw, thread = _timed_dial(net, net.c_identity, claim="d_" + "ab" * 8)
        assert raw == b""
        assert entered.wait(5), "the refusal was never audited"
        assert thread.is_alive(), "the audit write finished before the dialer's EOF"
        assert elapsed_ms < 5000
    finally:
        release.set()
    thread.join(10)
    net.server.audit.flush()
    rows = _audit_rows(net.root, "handshake_refused")
    assert len(rows) == 1 and rows[0]["detail"].get("declared_device") == "d_" + "ab" * 8


def test_the_refusal_handler_answers_before_it_audits() -> None:
    """Structural: in the ``MeshRefusal`` handler the close precedes the audit call."""
    source = textwrap.dedent(inspect.getsource(relay.RelayServer._run_inbound_handshake))
    tree = ast.parse(source)
    handler = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ExceptHandler)
        and isinstance(node.type, ast.Name)
        and node.type.id == "MeshRefusal"
    )
    calls = [
        ast.unparse(node.func)
        for node in sorted(
            (n for n in ast.walk(handler) if isinstance(n, ast.Call)),
            key=lambda n: (n.lineno, n.col_offset),
        )
    ]
    assert "self._send_removal_notice" in calls
    cleanup = next(node for node in handler.body if isinstance(node, ast.Try) and node.finalbody)
    final_calls = [
        ast.unparse(node.value.func)
        for node in cleanup.finalbody
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
    ]
    assert final_calls == ["_close_quietly", "self._audit_handshake_refusal"]


# ---------------------------------------------------------------------------
# Historical statistical check (not a proof of uniform work)
# ---------------------------------------------------------------------------

#: A modelled durable-write cost, added to every ``AuditLog.sync`` (the fsync behind
#: every ``handshake_refused`` row). CI runners often sit on tmpfs or a page cache
#: where fsync is ~free, which would let the pre-fix order pass by accident; a fixed
#: added cost makes "is the fsync inside the dialer's window?" visible on any disk.
#: 8 ms sits well above the ~1.6-2.5 ms an fsync measured on EBS here and in #4394 QA,
#: so the signal is far outside scheduler noise.
_MODELLED_FSYNC_S = 0.008
_SAMPLES = 25


def test_refusal_timing_does_not_separate_unknown_from_known_ids(
    net: Net, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]  # noqa: F811
) -> None:
    """Four dialer classes, interleaved; medians must sit together, generously.

    * ``unknown_id``: a never-seen id per sample (so the coalescer writes a row each
      time — the class #4394 QA measured at ~6x);
    * ``tombstoned_unproven``: B's tombstoned id, signed by a fresh random key;
    * ``tombstoned_wrong_key``: B's tombstoned id, signed by another member's REAL key;
    * ``random_key``: a freshly minted identity dialling as itself.

    This historical heuristic can mask the injected effect under load: its
    tolerance grows with the fastest median, and workers overlap samples. It is
    not a constant-time or flake-free guarantee; structural/ordering guards carry
    the defensive invariant. Do not infer uniformity from a passing statistic.
    """
    original_sync = net.server.audit.sync

    def costly_sync() -> None:
        original_sync()
        time.sleep(_MODELLED_FSYNC_S)

    monkeypatch.setattr(net.server.audit, "sync", costly_sync)
    randoms = [identity.mint(net.root.parent / f"rnd{i}", name=f"rnd{i}") for i in range(_SAMPLES)]
    signers = [identity.mint(net.root.parent / f"sig{i}", name=f"sig{i}") for i in range(_SAMPLES)]
    samples: dict[str, list[float]] = {
        "unknown_id": [],
        "tombstoned_unproven": [],
        "tombstoned_wrong_key": [],
        "random_key": [],
    }
    threads: list[threading.Thread] = []
    # Warm-up (imports, first keygen, first rows for the repeating classes) is unmeasured.
    for warm in (
        lambda: _timed_dial(net, net.c_identity, claim=net.b_identity.device_id),
        lambda: _timed_dial(net, net.d_identity, claim=net.b_identity.device_id),
        lambda: _timed_dial(net, net.c_identity, claim="d_" + token_bytes(8).hex()),
    ):
        threads.append(warm()[2])
    for i in range(_SAMPLES):
        for name in samples:
            if name == "unknown_id":
                result = _timed_dial(net, net.c_identity, claim="d_" + token_bytes(8).hex())
            elif name == "tombstoned_unproven":
                result = _timed_dial(net, signers[i], claim=net.b_identity.device_id)
            elif name == "tombstoned_wrong_key":
                result = _timed_dial(net, net.d_identity, claim=net.b_identity.device_id)
            else:
                result = _timed_dial(net, randoms[i])
            elapsed_ms, raw, thread = result
            threads.append(thread)
            assert raw == b"", (name, raw)
            samples[name].append(elapsed_ms)
    for thread in threads:
        thread.join(10)
    medians = {name: statistics.median(values) for name, values in samples.items()}
    with capsys.disabled():
        for name, values in samples.items():
            q = statistics.quantiles(values, n=4)
            print(
                f"\n[timing] {name:22s} median {medians[name]:.3f} ms "
                f"p25 {q[0]:.3f} p75 {q[2]:.3f} min {min(values):.3f} max {max(values):.3f}"
            )
    fastest = min(medians.values())
    bound = max(_MODELLED_FSYNC_S * 1000.0 / 2, 0.75 * fastest)
    spread = max(medians.values()) - fastest
    assert spread < bound, (medians, bound)


# ---------------------------------------------------------------------------
# Adjacent #4394 round-2 deferrals (R2-2, R2-3, R2-4, S4)
# ---------------------------------------------------------------------------


def test_importing_the_handshake_module_mints_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """R2-3: import is a CLI path; the stand-in key belongs to relay start."""
    import importlib.util
    import sys

    name = "local_operator.network._handshake_import_probe_r2_3"
    spec = importlib.util.spec_from_file_location(name, hs_mod.__file__)
    assert spec is not None and spec.loader is not None
    fresh = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, fresh)
    spec.loader.exec_module(fresh)
    assert fresh._DUMMY_PUBLIC_KEY == ""  # noqa: SLF001
    minted = fresh.prime_dummy_public_key()
    assert len(wire.unb64u(minted)) == 32
    assert fresh.prime_dummy_public_key() == minted == fresh._dummy_public_key()  # noqa: SLF001


def test_relay_start_mints_the_key_before_it_accepts(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R2-3 / R1-5: ``start`` primes the key, so no inbound probe pays for a keygen."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root / "starter"))
    order: list[str] = []
    monkeypatch.setattr(relay, "prime_dummy_public_key", lambda: order.append("prime") or "")
    server = relay.RelayServer(
        settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1"),
        identity=identity.mint(root / "starter", name="starter"),
    )
    original = server._accept_loop  # noqa: SLF001

    def accept_loop() -> None:
        order.append("accept")
        original()

    monkeypatch.setattr(server, "_accept_loop", accept_loop)
    try:
        server.start(background=True)
        deadline = time.monotonic() + 5
        while "accept" not in order and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        server.stop()
    assert order[:2] == ["prime", "accept"], order


def test_a_failed_record_write_is_audited_once_as_unrecorded_then_recovers(
    b_side: BSide,  # noqa: F811
    drops: _DropSpy,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """R2-2 + S4: a drop that cannot be recorded still leaves ONE audit row, no secret text,
    and the first notice after the store recovers writes the real transition row."""
    real_save = relay.store.save

    def broken_save(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("disk full at /secret/path")

    monkeypatch.setattr(relay.store, "save", broken_save)
    with caplog.at_level(logging.WARNING, logger="local_operator.network.relay"):
        assert b_side.dial_scripted(b_side.a_identity) == "membership_ended"
        assert b_side.dial_scripted(b_side.a_identity) == "membership_ended"
    assert drops.calls == 2
    # S4: the log names the class only; the exception's message never reaches it.
    assert any("could not record a verified removal" in r.getMessage() for r in caplog.records)
    assert "OSError" in caplog.text
    assert "disk full" not in caplog.text and "/secret/path" not in caplog.text
    assert all(r.exc_info is None for r in caplog.records)
    b_side.server.audit.flush()
    rows = _audit_rows(b_side.root, "membership_ended")
    assert len(rows) == 1, "an unrecordable drop must be audited exactly once"
    assert rows[0]["detail"]["recorded"] is False
    assert rows[0]["detail"]["credentials_dropped"] is True
    assert b_side.self_active(), "the injected failure did not stop the save"

    # S4: the store recovers; the next verified notice records and audits the flip.
    monkeypatch.setattr(relay.store, "save", real_save)
    assert b_side.dial_scripted(b_side.a_identity) == "membership_ended"
    assert not b_side.self_active()
    b_side.server.audit.flush()
    rows = _audit_rows(b_side.root, "membership_ended")
    assert len(rows) == 2
    assert "recorded" not in rows[1]["detail"]
    assert drops.calls == 3


def _run_redial_briefly(b_side: BSide, monkeypatch: pytest.MonkeyPatch) -> list[str]:  # noqa: F811
    dialled: list[str] = []
    monkeypatch.setattr(relay, "REDIAL_TICK_S", 0.01)
    monkeypatch.setattr(
        b_side.server,
        "_ensure_link_with_reason",
        lambda device_id, **_k: dialled.append(device_id) or (None, "unreachable"),
    )
    monkeypatch.setattr(b_side.server, "_redial_candidates", lambda _member: ["127.0.0.1:9"])
    loop = threading.Thread(target=b_side.server._redial_loop, daemon=True)  # noqa: SLF001
    loop.start()
    time.sleep(0.4)
    b_side.server._stop.set()  # noqa: SLF001
    loop.join(5)
    assert not loop.is_alive()
    return dialled


def test_a_self_removed_device_stops_redialling_the_network(
    b_side: BSide, drops: _DropSpy, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """R2-4: once a verified notice tombstoned our own row, the redial loop lets the peer go."""
    assert b_side.dial_scripted(b_side.a_identity) == "membership_ended"
    assert not b_side.self_active()
    a_id = b_side.a_identity.device_id
    b_side.server.redial.seed(a_id)
    assert _run_redial_briefly(b_side, monkeypatch) == []
    assert a_id not in b_side.server.redial.tracked()


def test_a_live_device_still_redials_its_peers(
    b_side: BSide, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """Negative control for R2-4: the same loop, self row active, does dial."""
    assert b_side.self_active()
    b_side.server.redial.seed(b_side.a_identity.device_id)
    assert b_side.a_identity.device_id in _run_redial_briefly(b_side, monkeypatch)


def test_tombstone_lookup_stays_bound_to_its_network_snapshot(net: Net) -> None:  # noqa: F811
    """A stored removal in one network supplies no key to another network's probe."""
    policy = relay.ServerPolicy(net.server, net.network_id)
    removed = net.b_identity.device_id
    assert policy.removed_member_public_key(net.network_id, removed) == net.b_identity.public_key
    assert policy.removed_member_public_key("n_other", removed) is None
    # A rejected cross-network lookup must not consume/change the legitimate row.
    assert policy.removed_member_public_key(net.network_id, removed) == net.b_identity.public_key


@pytest.mark.parametrize("present", [True, False])
def test_tombstone_probe_scans_all_slots_and_checks_dummy_eligibility(
    net: Net, monkeypatch: pytest.MonkeyPatch, present: bool  # noqa: F811
) -> None:
    """Observe work shape directly, not elapsed time or an inferred timing bound."""
    policy = relay.ServerPolicy(net.server, net.network_id)
    record = policy._record  # noqa: SLF001
    assert record is not None
    visits: list[tuple[str, Any]] = []

    class Slots(list[Any]):
        def __init__(self, label: str, values: Any) -> None:
            super().__init__(values)
            self.label = label

        def __iter__(self) -> Any:
            for value in super().__iter__():
                visits.append((self.label, value))
                yield value

    # Nonempty aliases ensure a direct-id match cannot short-circuit alias work.
    for row in record.members:
        row.previous_ids = Slots("alias", ["old_" + row.device_id])
    expected_rows = list(record.members)
    expected_aliases = [alias for row in record.members for alias in row.previous_ids]
    record.members = Slots("member", record.members)
    record.removed_ids = Slots("revoked", [*record.removed_ids, "other_removed"])
    expected_revocations = list(record.removed_ids)
    visits.clear()
    ticks: list[bool] = []
    original_time = relay.time.time

    def now() -> float:
        ticks.append(True)
        return original_time()

    monkeypatch.setattr(relay.time, "time", now)
    # A private policy snapshot only: neither case sends any network traffic.
    probe = hs_mod.Handshake.new(
        role="listener",
        identity=net.a_identity,
        network_id=net.network_id,
        epoch=policy.current_epoch,
        instance_id="i_structural",
        session_protocol=relay._session_protocol(),  # noqa: SLF001
    )
    probe.peer_device_id = net.b_identity.device_id if present else "missing_slot"
    digest = hs_mod.transcript_digest(
        hs_mod.transcript(probe.hello, probe.challenge, probe.auth_core)
    )
    signature = wire.b64u(net.b_identity.sign(b"lop-mesh-auth-v1\x00" + digest))
    checked_keys: list[str] = []
    original_verify = hs_mod._verify_signature  # noqa: SLF001

    def verify(key: str, *args: Any, **kwargs: Any) -> None:
        checked_keys.append(key)
        original_verify(key, *args, **kwargs)

    monkeypatch.setattr(hs_mod, "_verify_signature", verify)
    # Exclude setup/key generation from the eligibility clock spy.
    ticks.clear()
    probe._probe_tombstone(policy, signature)  # noqa: SLF001
    assert [v for label, v in visits if label == "member"] == expected_rows
    assert [v for label, v in visits if label == "alias"] == expected_aliases
    assert [v for label, v in visits if label == "revoked"] == expected_revocations
    assert ticks == [True], "misses must evaluate age eligibility too"
    assert checked_keys == [
        net.b_identity.public_key if present else hs_mod._dummy_public_key()
    ]  # noqa: SLF001
    assert probe.removal_proven is present


@pytest.mark.parametrize(
    "error", [TypeError("synthetic I/O seam"), RuntimeError("synthetic I/O seam")]
)
def test_unexpected_notice_error_still_closes_and_attempts_refusal_audit(
    net: Net, monkeypatch: pytest.MonkeyPatch, error: Exception  # noqa: F811
) -> None:
    """A legitimate removed member's courtesy reply cannot own refusal cleanup."""
    events: list[str] = []
    original_close = relay._close_quietly  # noqa: SLF001
    original_audit = net.server._audit_handshake_refusal  # noqa: SLF001

    def broken_notice(*_args: Any, **_kwargs: Any) -> bool:
        events.append("notice")
        raise error

    def close(sock: socket.socket) -> None:
        original_close(sock)
        assert sock.fileno() == -1
        events.append("close")

    def audit(*args: Any, **kwargs: Any) -> None:
        assert events == ["notice", "close"]
        events.append("audit")
        original_audit(*args, **kwargs)

    monkeypatch.setattr(net.server, "_send_removal_notice", broken_notice)
    monkeypatch.setattr(relay, "_close_quietly", close)
    monkeypatch.setattr(net.server, "_audit_handshake_refusal", audit)
    _hs, raw, failure = net.dial(net.b_identity)
    assert failure is None
    assert raw == b""
    assert events == ["notice", "close", "audit"]
    net.server.audit.flush()
    rows = _audit_rows(net.root, "handshake_refused")
    assert len(rows) == 1
    assert rows[0]["detail"]["declared_device"] == net.b_identity.device_id


def test_overlapping_failed_removals_claim_one_audit_attempt_outside_lock(
    b_side: BSide, drops: _DropSpy, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """Overlap at a blocked sink; require atomic claims without serializing I/O."""
    server = b_side.server
    # Also supplies the lock to the old product in the RED control, so absence
    # of the attribute cannot be mistaken for observing the check/claim defect.
    lock = threading.Lock()
    server._unrecorded_removals_lock = lock  # noqa: SLF001

    class Claims(set[str]):
        def __contains__(self, value: object) -> bool:
            assert lock.locked(), "dedup check must hold the claim lock"
            return super().__contains__(value)

        def add(self, value: str) -> None:
            assert lock.locked(), "dedup claim must hold the claim lock"
            super().add(value)

    server._unrecorded_removals = Claims()  # noqa: SLF001
    entered = threading.Event()
    release = threading.Event()
    errors: list[BaseException] = []
    attempts: list[Any] = []

    def broken_save(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("synthetic save failure")

    def blocked_audit(event: Any) -> None:
        if event.event == "membership_ended":
            assert not lock.locked(), "audit I/O must not hold the claim lock"
            attempts.append(event)
            entered.set()
            assert release.wait(5)

    monkeypatch.setattr(relay.store, "save", broken_save)
    monkeypatch.setattr(server.audit, "record", blocked_audit)
    ended = hs_mod.MembershipEnded(
        "verified synthetic notice", epoch=2, signer=b_side.a_identity.device_id
    )

    def apply() -> None:
        try:
            server._apply_verified_removal(b_side.network_id, ended)  # noqa: SLF001
        except BaseException as exc:  # noqa: BLE001 — report worker failures
            errors.append(exc)
            entered.set()

    first = threading.Thread(target=apply)
    second = threading.Thread(target=apply)
    first.start()
    try:
        assert entered.wait(5)
        assert not errors, errors
        second.start()
        second.join(5)
        assert not second.is_alive(), "second notice blocked on audit I/O"
        assert not errors, errors
        assert len(attempts) == 1
        assert attempts[0].detail["recorded"] is False
        assert drops.calls == 2
    finally:
        release.set()
        first.join(5)
        if second.ident is not None:
            second.join(5)
    assert not first.is_alive()
    assert not errors, errors
