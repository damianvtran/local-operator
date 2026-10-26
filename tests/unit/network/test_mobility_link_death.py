"""The move's two 2026-10-25 failures, each pinned by the cell that fails before it.

Measured on the operator's mesh: ``lop sessions move 81b9f802851d --to local`` against
``lop-remote-node-2`` answered ``busy`` ("the conversation changed…") for a turn that
had finished 30 s earlier, and the retry then read ``unreachable`` — "the source
stopped answering during the copy" — about a relay that was up throughout (PID
unchanged, NRestarts=0). Both ends logged ``link_closed cause=error`` 1.3 ms apart,
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
from tests.unit.network.test_mobility import (  # noqa: F401 — `devices` is a fixture
    SESSION as _RIG_SESSION,
    Devices,
    _move,
    _owned_session,
    devices,
)
from tests.unit.network.test_relay_e2e import _pair_settled

SESSION = "9f3ac1e0b7d2"
assert _RIG_SESSION == SESSION, "the rig and this file must name one session"


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
    root: Path, *, to_device: str, instance: str, phase: str = HANDOFF_PHASE_PREPARED
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

    cleared = mobility.expire_for_peer(
        tmp_path, "", instance="i_mine", to_device="d_away"
    )

    assert cleared == [], cleared
    assert SESSION in read_handoff_journal(tmp_path)


def test_the_link_closed_hook_releases_and_is_registered() -> None:
    """The hook is wired on the relay and does the release, on the closing thread.

    Registered by ``mobility.install`` rather than re-derived per call, so the release
    cannot be forgotten by a relay whose slice installed without it.
    """
    hooks: dict[str, Any] = {}

    class _Server:
        root = Path("/nonexistent")
        instance_id = "i_mine"
        _link_for = staticmethod(lambda device_id: None)  # noqa: SLF001 — the seam

    server = SimpleNamespace(
        root=Path("/nonexistent"),
        instance_id="i_mine",
        _link_for=lambda device_id: None,
        register_ops=lambda *a, **k: hooks.update(k.get("on_link_closed") or {}),
    )
    mobility.install(cast(Any, server))

    assert "mobility-stranded-move" in hooks, hooks
    link = SimpleNamespace(device_id="d_away")
    assert mobility._on_link_closed(cast(Any, server), link, "error") is None  # noqa: SLF001


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
    link = SimpleNamespace(device_id="d_away")

    assert mobility._on_link_closed(cast(Any, server), link, "we-closed") is None  # noqa: SLF001
