"""The move's two 2026-10-25 failures, each pinned by the cell that fails before it.

Measured on the operator's mesh: ``lop sessions move 81b9f802851d --to local`` against
``lop-remote-node-2`` answered ``busy`` ("the conversation changed…") for a turn that
had finished 30 s earlier, and the retry then read ``unreachable`` — "the source
stopped answering during the copy" — about a relay that was up throughout (PID
unchanged, NRestarts=0). Both ends logged ``link_closed cause=error`` a few
milliseconds apart,
40.4 s after the source prepared, and the source stayed parked in ``prepared`` with no
recovery but a relay restart.

Two independent defects, one per section below:

* the OUTBOUND PLAN was 13.67 MB against the link's 8 MiB record ceiling, because
  ``held_report`` reported and hashed the whole SHARED attachment store (122,101
  blobs, 5.2 GB, 80.8 s) instead of the blobs this copy's transcript references. The
  writer thread refused the frame and closed the link;
* a copy that fails for any reason rolled back ONE side only, so the source kept its
  ``prepared`` entry — and a live relay cannot clear its own (the instance rule that
  protects a live handoff), so nothing could ever release it.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from local_operator.network import mobility, sync, wire
from local_operator.session.placement import (
    HANDOFF_PHASE_HANDING_OFF,
    HANDOFF_PHASE_PREPARED,
    read_handoff_journal,
    write_handoff_entry,
)
from tests.unit.network.conftest import wait_for
from tests.unit.network.test_mobility import (  # noqa: F401 — `devices` and `pair` are fixtures
    SESSION,
    Devices,
    _move,
    _owned_session,
    devices,
    pair,
)
from tests.unit.network.test_relay_e2e import _pair_settled

# ``SESSION`` comes from the rig: one name for the id both files move, so the cells
# here and the cells there cannot drift apart.


def _blob(store: Path, ref: str, payload: bytes) -> None:
    store.mkdir(parents=True, exist_ok=True)
    (store / f"{ref}.bin").write_bytes(payload)
    (store / f"{ref}.json").write_text(json.dumps({"name": "x"}), encoding="utf-8")


@pytest.fixture()
def shared_store(tmp_path: Path) -> Path:
    """A store shaped like the operator's: a few blobs this session uses, many it does not."""
    store = tmp_path / "store" / "attachments"
    for index in range(40):
        _blob(store, f"{index:032x}", b"unreferenced" * 8)
    return tmp_path / "store"


def _own_session(root: Path, refs: list[str]) -> Path:
    directory = root / "sessions" / SESSION
    directory.mkdir(parents=True)
    rows = [{"id": "e0", "type": "user", "content": "hello"}]
    rows += [
        {"id": f"e{i + 1}", "type": "assistant", "attachment": ref} for i, ref in enumerate(refs)
    ]
    directory.joinpath("transcript.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return directory


# ---------------------------------------------------------------------------
# 1 — the outbound plan is bounded by the copy, not by the store
# ---------------------------------------------------------------------------


def test_held_report_reports_only_the_blobs_this_copy_references(
    tmp_path: Path, shared_store: Path
) -> None:
    """THE DEVICE'S WHOLE STORE IS NOT THIS COPY'S BUSINESS (#2918 lane, live 2026-10-25).

    ``held_report`` answers "what do I already hold of the set being copied". A move
    points ``attachments_root`` at the install's SHARED content-addressed store, so
    globbing it reported every attachment of every conversation on the device — and
    hashed each one. On the operator's Mac that was 122,101 blobs and 5.2 GB in
    80.8 s, and the resulting plan frame was 13.67 MB against the 8 MiB
    ``wire.MAX_RECORD_BYTES``: ``PeerCodec.seal`` refused it inside the writer thread,
    the link closed with ``cause=error``, and the move reported the source as silent.

    This cell fails on the old code (41 entries: the whole store) and passes when the
    report is the referenced set (2: the one blob and its sidecar).
    """
    used = f"{0:032x}"
    directory = _own_session(tmp_path, [used])
    _blob(shared_store / "attachments", used, b"the one this conversation uses")

    report = sync.held_report(directory, shared_store)

    assert sorted(report["attachments"]) == [used, f"{used}.json"], sorted(report["attachments"])
    # AND THE FRAME IT PRODUCES FITS THE LINK. A report the codec refuses is not a
    # report: this is the property the old shape violated, asserted where it broke.
    frame = {
        "req": 1,
        "op": "net_sync",
        "phase": "plan",
        "session_id": SESSION,
        "have": report,
        "purpose": "move",
    }
    size = len(json.dumps(frame, separators=(",", ":")).encode())
    assert size < wire.MAX_RECORD_BYTES, size


def test_held_report_is_bounded_however_many_blobs_the_store_holds(tmp_path: Path) -> None:
    """The report's size follows the TRANSCRIPT, not the store's history.

    Four hundred unreferenced blobs change nothing about what a copy of this session
    holds, so the frame a destination sends is the same size. This is the guard that
    keeps a bigger store (or a longer-lived one) from re-breaking a move.
    """
    store = tmp_path / "store" / "attachments"
    for index in range(400):
        _blob(store, f"{index:032x}", b"x" * 64)
    directory = _own_session(tmp_path, [])

    report = sync.held_report(directory, tmp_path / "store")

    assert report["attachments"] == {}, report["attachments"]


# ---------------------------------------------------------------------------
# 2 — a prepared entry is released when its link dies with no replacement
# ---------------------------------------------------------------------------


def _prepared(
    root: Path,
    *,
    to_device: str,
    instance: str,
    phase: str = HANDOFF_PHASE_PREPARED,
    link_id: str = "",
) -> None:
    write_handoff_entry(
        root,
        SESSION,
        {
            "role": "source",
            "phase": phase,
            "to_device": to_device,
            "lease_epoch": "e_1",
            "mode": "move",
            "requester": to_device,
            "instance_id": instance,
            # THE LINK THE MOVE RIDES (QA round 1, Q1). Empty means "written before
            # the binding existed", which the reaper can only answer device-wide.
            **({"link_id": link_id} if link_id else {}),
        },
    )


def test_a_dead_link_releases_the_prepared_entry_it_stranded(tmp_path: Path) -> None:
    """§6.5's missing row: the copy cannot outlive the link it travels over.

    Every phase of a move arrives as a frame on one link, so once that link is closed
    and no replacement dial exists, nothing can advance the entry — not ``ready``, not
    the destination's ``rollback``, not ``done``. Leaving it parks the conversation:
    the engage guard refuses it and every retry answers ``in_progress`` naming a phase
    nobody will advance. Measured 2026-09-26 and again 2026-10-25, where the source sat
    ``prepared`` for hours with its relay healthy.

    Fails on the old code (nothing clears it; ``expire_for_peer`` did not exist).
    """
    _prepared(tmp_path, to_device="d_away", instance="i_mine")

    cleared = mobility.expire_for_peer(tmp_path, "", instance="i_mine", to_device="d_away")

    assert cleared == [SESSION], cleared
    assert read_handoff_journal(tmp_path) == {}


@pytest.mark.parametrize(
    "case",
    [
        # NOT OURS TO ABANDON: another relay instance's in-flight move.
        {"instance": "i_other", "device": "d_away", "phase": HANDOFF_PHASE_PREPARED},
        # MONOTONE: past ``prepared`` the destination may be the only holder.
        {"instance": "i_mine", "device": "d_away", "phase": HANDOFF_PHASE_HANDING_OFF},
        # A DIFFERENT PEER's link, not this one.
        {"instance": "i_mine", "device": "d_other", "phase": HANDOFF_PHASE_PREPARED},
    ],
    ids=["another-instance", "handing-off", "another-device"],
)
def test_release_refuses_every_entry_it_cannot_prove_is_this_move(
    tmp_path: Path, case: dict[str, str]
) -> None:
    """Three conditions, each a way the release could otherwise do damage."""
    _prepared(tmp_path, to_device=case["device"], instance=case["instance"], phase=case["phase"])

    cleared = mobility.expire_for_peer(tmp_path, "", instance="i_mine", to_device="d_away")

    assert cleared == [], cleared
    assert SESSION in read_handoff_journal(tmp_path)


def test_link_close_keeps_the_descriptor_until_its_reader_exits(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A shutdown is not descriptor ownership transfer (R3-1 / Q6).

    Hold a real link's reader just after recv reports EOF. Closing from another
    thread must wake that read without releasing its fd; only the reader's finally
    may close it. This pins the ordering independently of the macOS timed-recv race
    that left the re-dial test's evicted reader alive on an already-closed socket.
    """
    p: Devices = request.getfixturevalue("pair")
    server_a, server_b, _host, _port = p
    _pair_settled(p, monkeypatch, role="admin")
    observed = threading.Event()
    release = threading.Event()
    owner: dict[str, Any] = {}
    real = wire.FrameReader.read_record_payload

    def read_until_shutdown(reader: Any, deadline: float | None) -> bytes:
        try:
            return real(reader, deadline)
        except (ConnectionError, OSError):
            if reader._sock is owner.get("socket"):
                observed.set()
                release.wait(timeout=5)
            raise

    monkeypatch.setattr(wire.FrameReader, "read_record_payload", read_until_shutdown)
    server_b._ensure_link(server_a.identity.device_id)
    assert wait_for(lambda: server_a._link_for(server_b.identity.device_id) is not None)
    link = server_a._link_for(server_b.identity.device_id)
    assert link is not None and link._reader is not None
    owner["socket"] = link.sock
    try:
        link.close("we-closed", flush_s=0.0)
        assert observed.wait(timeout=5), "shutdown did not wake the retained reader"
        assert link._reader.is_alive(), "the gate must hold the reader before its finally"
        assert link.sock.fileno() >= 0, "the closing thread released the reader's descriptor"
    finally:
        release.set()
        link.close("we-closed", flush_s=0.0)
        link._reader.join(timeout=5)
    assert not link._reader.is_alive(), link._reader.name
    assert link.sock.fileno() == -1, "the reader must release its descriptor on exit"


def test_a_re_dial_during_a_copy_releases_the_source_and_the_retry_completes(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """QA ROUND 2, Q4: a re-dial does not resume the evicted link's move.

    The destination's ``LinkTransport`` holds the old ``PeerLink``. The identity
    fence fails that link's outstanding fetch, and the destination's rollback also
    tries the same dead transport. Rebinding the source's entry to the new link (the
    round-1 response) therefore left ``prepared`` with no move capable of finishing,
    and a retry told the operator to restart a healthy relay. ``replaced`` is a
    death for this move: release its entry, keep the source bytes, and let a fresh
    prepare complete.

    This drives an ACTUAL copy, not a planted journal entry. On 3c00f490 the journal
    is rebound and the final retry answers ``in_progress``; here it is empty and the
    retry completes over two real relays.
    """
    p: Devices = request.getfixturevalue("pair")
    server_a, server_b, host, port = p
    record, _h, _p = _pair_settled(p, monkeypatch, role="admin")
    _owned_session(server_a)
    a_dev = server_a.identity.device_id
    server_b._ensure_link(a_dev)
    real = mobility.LinkTransport.ask
    fetches = 0
    redialed = False

    def ask(self_: Any, frame: dict[str, Any], *, timeout: float | None = None) -> Any:
        nonlocal fetches, redialed
        if frame.get("op") == "net_sync" and frame.get("phase") == "fetch":
            fetches += 1
            if fetches == 2:
                # Match QA's in-flight drive: one fetch has already crossed, and
                # this next fetch is about to use the transport the fence evicts.
                redialed = True
                # Eviction removes both old links from the relay registries before
                # their readers finish the close hooks. The fixture's stop cannot
                # join those removed links; retain and reap only this dial's pair.
                evicted = [self_.link, server_a.links[self_.link.link_id]]

                def reap_evicted_readers() -> None:
                    # Stop the pair and join BOTH retained readers before asserting:
                    # eviction has removed them from the fixture's registry, but
                    # they still own their sockets until the read loops unwind.
                    server_a.stop()
                    server_b.stop()
                    survivors = []
                    for old_link in evicted:
                        reader = old_link._reader  # noqa: SLF001 — exact fixture-owned thread
                        if reader is not None:
                            reader.join(timeout=5)
                            if reader.is_alive():
                                survivors.append(reader.name)
                    assert not survivors, survivors

                request.addfinalizer(reap_evicted_readers)
                replacement, reason = server_b.dial(
                    record.network_id, host=f"{host}:{port}", epoch=record.epoch
                )
                assert replacement is not None, reason
        return real(self_, frame, timeout=timeout)

    monkeypatch.setattr(mobility.LinkTransport, "ask", ask)
    failed = _move(server_b, SESSION, monkeypatch=monkeypatch)
    assert failed["ok"] is False, failed
    assert redialed, "the copy never reached its fetch"
    # WAIT FOR THE RELEASE FIRST (#2950 R-C1): a one-shot read before the wait made
    # a slow-but-correct release take the diagnostic branch and false-fail. Only a
    # journal still wedged after the wait drives the retry, so the old rebind
    # control fails on the user-facing in_progress refusal rather than on a bare
    # journal assertion; the fixed path still requires the journal to clear.
    released = wait_for(lambda: read_handoff_journal(server_a.root) == {}, 15)
    monkeypatch.setattr(mobility.LinkTransport, "ask", real)
    if not released:
        wedged_retry = _move(server_b, SESSION, monkeypatch=monkeypatch)
        assert wedged_retry["ok"] is True, wedged_retry
        assert wedged_retry["state"] == "completed", wedged_retry
    assert released, read_handoff_journal(server_a.root)

    moved = _move(server_b, SESSION, monkeypatch=monkeypatch)
    assert moved["ok"] is True, moved
    assert moved["state"] == "completed", moved
    assert (server_b.root / "sessions" / SESSION).is_dir()
    assert not (server_a.root / "sessions" / SESSION).exists()


def test_the_copy_links_own_death_releases_the_entry_even_beside_a_second_link(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """QA ROUND 1, Q1(b): two links to one peer are normal; the copy rides ONE.

    A device-level guard kept the ``prepared`` entry whenever ANY link to the peer was
    alive, but the copy's frames ride exactly one of them — so when the copy's own link
    died the destination's ``rollback`` had already gone over that dead link and nothing
    ever cleared the source, which then refused every retry ``in_progress`` … "restart
    the relay on device-a". That is the wedge this branch exists to remove, reached on
    a healthy relay (QA's 3g).

    Fails on 7586450d07 (journal keeps ``prepared``, retry refused); passes here, and
    the retry completes — which is the recovery, not just the empty journal.
    """
    p: Devices = request.getfixturevalue("pair")
    server_a, server_b, _host, _port = p
    record, _h, _p = _pair_settled(p, monkeypatch, role="admin")
    _owned_session(server_a)
    a_dev, b_dev = server_a.identity.device_id, server_b.identity.device_id
    server_b._ensure_link(a_dev)
    host_b, port_b = server_b._listener.getsockname()[:2]  # type: ignore[union-attr]
    second, reason = server_a.dial(record.network_id, host=f"{host_b}:{port_b}", epoch=record.epoch)
    assert second is not None, reason
    two = wait_for(
        lambda: len([lk for lk in server_b.links.values() if lk.device_id == a_dev and lk.alive])
        >= 2
        and len([lk for lk in server_a.links.values() if lk.device_id == b_dev and lk.alive]) >= 2,
        15,
    )
    assert two, "the second link did not come up on both ends"

    real = mobility.LinkTransport.ask
    state = {"fetch": 0}

    def ask(self_: Any, frame: dict[str, Any], *, timeout: float | None = None) -> Any:
        if frame.get("op") == "net_sync" and frame.get("phase") == "fetch":
            state["fetch"] += 1
            if state["fetch"] == 2:
                # THE COPY'S OWN LINK, mid-copy — the live failure's branch.
                self_.link.close("error")
        return real(self_, frame, timeout=timeout)

    monkeypatch.setattr(mobility.LinkTransport, "ask", ask)
    failed = _move(server_b, SESSION, monkeypatch=monkeypatch)
    assert failed["ok"] is False, failed
    released = wait_for(lambda: read_handoff_journal(server_a.root) == {}, 15)
    assert released, read_handoff_journal(server_a.root)
    # THE OTHER LINK IS STILL THERE: the release was keyed to the copy's link, not to
    # "no link left".
    assert server_a._link_for(b_dev) is not None  # noqa: SLF001

    monkeypatch.setattr(mobility.LinkTransport, "ask", real)
    retry = _move(server_b, SESSION, monkeypatch=monkeypatch)
    assert retry["ok"] is True, retry
    assert retry["state"] == "completed", retry


def test_a_rollback_without_an_epoch_cannot_clear_a_live_move(root: Path) -> None:
    """REVIEW ROUND 1 m1 = QA ROUND 1 Q2: an absent epoch is not a wildcard.

    ``if epoch and epoch != entry_epoch`` let a frame that carried NO epoch through,
    and an empty epoch was reachable: a ``--keep`` prepare answered ``lease_epoch: ""``
    and both failure branches of ``_destination_move`` report a rollback for copies
    too. A keep and a move of one id to one device at once therefore let the keep's
    report empty the live move's journal — the docstring's "a stale report can never
    clear a newer handoff" did not hold.

    The source requires a match, so no sender has to special-case "no epoch". A
    ``--keep`` copy writes no journal entry of its own, so its epoch-less report was
    only ever able to clear a MOVE's entry — the collision this closes.

    Fails on 7586450d07 (``cleared``); passes here.
    """
    _prepared(root, to_device="d_away", instance="i_mine", link_id="l_live")
    server = SimpleNamespace(
        root=root, instance_id="i_mine", identity=SimpleNamespace(device_id="d_me")
    )
    link = SimpleNamespace(device_id="d_away", link_id="l_live")

    for frame_epoch in ("", None):
        answer = mobility._source_rollback(  # noqa: SLF001
            cast(Any, server),
            cast(Any, link),
            {"session_id": SESSION, "lease_epoch": frame_epoch, "code": "unreachable"},
        )
        assert answer["result"] == "kept", (frame_epoch, answer)
        assert SESSION in read_handoff_journal(root), frame_epoch

    # THE CONTROL: the right epoch still clears, so the guard is a comparison and not
    # a refusal of every report.
    cleared = mobility._source_rollback(  # noqa: SLF001
        cast(Any, server),
        cast(Any, link),
        {"session_id": SESSION, "lease_epoch": "e_1", "code": "unreachable"},
    )
    assert cleared["result"] == "cleared", cleared
    assert read_handoff_journal(root) == {}


def test_a_link_death_is_logged_once_and_a_deliberate_close_is_never_logged(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """REVIEW ROUND 1 m3 = QA ROUND 1 Q3: one line, and only when something failed.

    ``close`` sets ``_closed`` BEFORE it shuts the socket down, so a reader that loses
    the race with its own ``sock.close()`` raised EBADF and logged a SECOND line after
    the writer's real cause — and the same misleading line appeared on a deliberate
    ``replaced``/``we-closed`` close where nothing had failed at all. QA measured both
    on the rig: two lines for one death on the closing side, and a line for a clean
    close.

    The guard is pinned by DRIVING THE TWO CALLS rather than by racing the threads:
    the EBADF line depends on which side notices the teardown first, so a timing-based
    cell is a coin flip (it passed once on the old code while this branch was being
    written). What is not a coin flip is the rule: once ``_closed`` is set, nothing
    more is logged.

    Fails on 7586450d07 (the post-close call logs an EBADF line); passes here.
    """
    p: Devices = request.getfixturevalue("pair")
    server_a, server_b, _host, _port = p
    _pair_settled(p, monkeypatch, role="admin")
    a_dev = server_a.identity.device_id
    server_b._ensure_link(a_dev)
    link = server_b._ensure_link(a_dev)
    assert link is not None

    caplog.set_level(logging.WARNING, logger="local_operator.network.relay")
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        # THE ORIGINATING CAUSE: logged, once.
        link._log_link_error(  # noqa: SLF001
            "writer", wire.LinkCryptoError("a frame exceeds the limit", kind="limit")
        )
        # THE TEARDOWN'S OWN ARTEFACT: ``close`` has begun, so this is the socket
        # this side just took away — exactly the line that used to follow the real
        # cause (and that a deliberate close produced with no cause at all).
        link.close("we-closed")
        link._log_link_error("reader", OSError(9, "Bad file descriptor"))  # noqa: SLF001
        time.sleep(0.3)

    messages = [record.getMessage() for record in caplog.records]
    caused = [m for m in messages if "closed by its writer" in m and link.link_id in m]
    assert len(caused) == 1, messages
    assert "LinkCryptoError" in caused[0], messages
    noise = [m for m in messages if "Bad file descriptor" in m]
    assert noise == [], noise


# ---------------------------------------------------------------------------
# 2b — the hook is wired, and it releases
# ---------------------------------------------------------------------------


def test_the_link_closed_hook_is_registered_and_releases_a_bound_entry(root: Path) -> None:
    """The hook is wired on the relay and does the release, on the closing thread.

    Registered by ``mobility.install`` rather than re-derived per call, so the release
    cannot be forgotten by a relay whose slice installed without it. The release is
    ASSERTED, not merely returned: an earlier draft of this cell called the hook and
    checked ``is None``, which is true of a hook that did nothing at all (review round
    1, n2).
    """
    hooks: dict[str, Any] = {}
    _prepared(root, to_device="d_away", instance="i_mine", link_id="l_doomed")
    server = SimpleNamespace(
        root=root,
        instance_id="i_mine",
        _link_for=lambda device_id: None,
        register_ops=lambda *a, **k: hooks.update(k.get("on_link_closed") or {}),
    )
    mobility.install(cast(Any, server))

    assert "mobility-stranded-move" in hooks, hooks
    link = SimpleNamespace(device_id="d_away", link_id="l_doomed")
    hooks["mobility-stranded-move"](link, "error")

    assert read_handoff_journal(root) == {}


# ---------------------------------------------------------------------------
# 3 — the retire's own exit writes are not "a concurrent writer"
# ---------------------------------------------------------------------------


def test_the_retired_runtimes_own_exit_writes_do_not_refuse_the_move(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE SPURIOUS ``busy`` (live 2026-10-25, ``81b9f802851d``).

    The prepare digests the copy set, retires the runtime, then re-checks one ``stat``
    per member. The runtime's exit path writes ``runtime-exit.json`` and appends a
    closing ``frontend_state_checkpoint_v1`` row — the row's mtime is restored but its
    SIZE is not — so the re-check saw a change the move had itself caused and refused
    ``busy`` ("the conversation changed while this device was preparing to hand it
    over…") for a session whose turn had finished 30 s earlier. The operator retried,
    which is how the second attempt reached the copy and failed there instead.

    A change is absorbed only when the runtime THIS CALL RETIRED left its exit record;
    ``test_a_source_that_changes_while_it_is_being_prepared_is_refused`` pins the
    other side — a change nothing accounts for is still refused.
    """
    pair_devices: Devices = request.getfixturevalue("devices")
    server_a, server_b, _host, _port = pair_devices
    server_b.bind_control()
    server_b.start()
    _pair_settled(pair_devices, monkeypatch, role="admin")
    source = _owned_session(server_a)
    real_retire = mobility._retire_local_runtime

    def retire_writing_its_own_exit(*args: Any, **kwargs: Any) -> Any:
        """A runtime that retires and then leaves its own durable trace."""
        (source / "runtime-exit.json").write_text(
            json.dumps(
                {
                    "v": 1,
                    "kind": "runtime-exit",
                    "session_id": SESSION,
                    "pid": 4242,
                    "started_at": 1700000000.0,
                    "at": 1700000100.0,
                    "reason": "socket-stop",
                    "detail": "",
                    "requested": True,
                    "build": "0.90.3",
                }
            ),
            encoding="utf-8",
        )
        with (source / "transcript.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "id": "e99",
                        "type": "custom",
                        "payload": {"custom_type": "frontend_state_checkpoint_v1"},
                    }
                )
                + "\n"
            )
        return {"result": "retired", "sentence": "", "pid": 4242}

    monkeypatch.setattr(mobility, "_retire_local_runtime", retire_writing_its_own_exit)

    moved = _move(server_b, SESSION, monkeypatch=monkeypatch)

    # THE MOVE PROCEEDS. On the old code this answered ``busy`` with "the conversation
    # changed while this device was preparing to hand it over".
    assert moved["ok"] is True, moved
    assert moved["state"] == "completed", moved
    assert not source.exists()
    assert (server_b.root / "sessions" / SESSION).is_dir()
    monkeypatch.setattr(mobility, "_retire_local_runtime", real_retire)


def test_a_write_nothing_here_caused_is_absorbed_once_the_retire_is_confirmed(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE LOOSER HALF OF THE ABSORPTION (review round 1, m2), pinned on purpose.

    ``_departed_runtime_explains`` asks only "did the runtime THIS CALL RETIRED leave
    its exit record" — it never asks what changed. So a turn that lands inside the
    retire window is absorbed too, not just the exit path's own writes. That is safe
    for a reason the journal cannot express (the manifest is rebuilt from the
    re-stamped set and the commit compares against a fresh digest under the epoch),
    but the behaviour is looser than "unattributable changes are refused", and this
    cell exists so a later reader does not tighten it back into the spurious ``busy``
    the operator hit on 2026-10-25.

    The refusal side is pinned separately by
    ``test_a_source_that_changes_while_it_is_being_prepared_is_refused`` — that session
    is ``cold``, so nothing explains the change at all.
    """
    pair_devices: Devices = request.getfixturevalue("devices")
    server_a, server_b, _host, _port = pair_devices
    server_b.bind_control()
    server_b.start()
    _pair_settled(pair_devices, monkeypatch, role="admin")
    source = _owned_session(server_a)

    def retire_while_a_turn_lands(*args: Any, **kwargs: Any) -> Any:
        """The exit record, plus a row this call did not write."""
        (source / "runtime-exit.json").write_text(
            json.dumps(
                {
                    "v": 1,
                    "kind": "runtime-exit",
                    "session_id": SESSION,
                    "pid": 4242,
                    "started_at": 1700000000.0,
                    "at": 1700000100.0,
                    "reason": "socket-stop",
                    "detail": "",
                    "requested": True,
                    "build": "0.90.3",
                }
            ),
            encoding="utf-8",
        )
        with (source / "transcript.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"id": "e98", "type": "user", "content": "a late turn"}) + "\n")
        return {"result": "retired", "sentence": "", "pid": 4242}

    monkeypatch.setattr(mobility, "_retire_local_runtime", retire_while_a_turn_lands)

    moved = _move(server_b, SESSION, monkeypatch=monkeypatch)

    # ABSORBED, and the row travels with the copy rather than being silently dropped:
    # the destination's transcript is what the source's bytes now contain.
    assert moved["ok"] is True, moved
    assert moved["state"] == "completed", moved
    arrived = (server_b.root / "sessions" / SESSION / "transcript.jsonl").read_text("utf-8")
    assert "a late turn" in arrived, arrived


def test_a_restarted_relay_is_what_clears_a_parked_entry_on_the_shipped_build(
    root: Path,
) -> None:
    """THE SUPPORTED RECOVERY FOR AN ALREADY-WEDGED DEVICE (v0.90.3 and older).

    ``reconcile`` skips an entry written by the relay that is running (the instance
    rule that stops a slow peer losing a live handoff), and a shipped build has no
    self-clearing path — so an entry a dead copy left behind survives every retry,
    every ``lop sessions move`` and every engage. What clears it is a relay whose
    instance id differs, which is exactly what ``lop network restart`` produces:
    ``recover_on_start`` then reconciles the entry as another instance's leftover and
    rolls it back at ``prepared``.

    This cell is the shipped build's behaviour, unchanged by this branch: it is the
    command the report gives for the device wedged before the fix.
    """
    write_handoff_entry(
        root,
        SESSION,
        {
            "role": "source",
            "phase": HANDOFF_PHASE_PREPARED,
            "to_device": "d_away",
            "lease_epoch": "e_1",
            "mode": "move",
            "instance_id": "i_running_relay",
        },
    )

    # THE RUNNING RELAY CANNOT CLEAR ITS OWN ENTRY.
    own = mobility.reconcile(root, only=SESSION, own_instance="i_running_relay")
    assert [row["action"] for row in own] == ["in_flight"], own
    assert SESSION in read_handoff_journal(root)

    # A RESTARTED RELAY IS A DIFFERENT INSTANCE, AND ROLLS IT BACK.
    after_restart = mobility.reconcile(root, only=SESSION, own_instance="i_after_restart")
    assert [row["action"] for row in after_restart] == ["rolled_back"], after_restart
    assert read_handoff_journal(root) == {}


def test_a_relay_that_is_stopping_does_not_release_its_entries() -> None:
    """Every link goes at shutdown; that is not a dead copy.

    Releasing on the way down would turn an ordinary restart into a rollback of a move
    that was merely interrupted, and the recovery for that already exists: the next
    instance reconciles the entry as another instance's leftover.
    """
    stopping = SimpleNamespace(is_set=lambda: True)
    server = SimpleNamespace(root=Path("/nonexistent"), instance_id="i_mine", _stop=stopping)
    link = SimpleNamespace(device_id="d_away", link_id="l_stopping")

    assert mobility._on_link_closed(cast(Any, server), link, "we-closed") is None  # noqa: SLF001
