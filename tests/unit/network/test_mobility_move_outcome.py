"""A move reports only what it verified (``docs/design/mesh-move-outcome.md``).

Measured from the operator's screen: a conversation "moved" from ``cloud-node-1``
to the local device, the surface entered its moving state, and the conversation was
still on the remote — ``On cloud-node-1``, cwd ``/home/ec2-user`` — with
``cloud-node-1`` not answering the last two mesh probes. Each cell below pins one
half of the contract that failure broke:

* ``state`` is ``completed`` only when the destination holds a promoted copy and the
  source no longer holds a directory for the id (cell 1);
* a source that cannot take its directory out of the store REFUSES, changes
  nothing, and a retry then works (cell 2);
* an owner that still lists a committed id is a contradiction, not a success
  (cell 3);
* a queued receipt is ``in_progress`` and never renders as "Moved" (cell 4);
* an unreachable owner is NAMED instead of reported as "nobody holds it" (cell 5);
* a cwd that does not exist on the destination is mapped before the indexes are
  rebuilt, and the transcript is untouched (cell 6);
* an offload to a dead peer refuses and changes nothing (cell 7).

The pair and the helpers come from ``test_mobility`` on purpose: these cells must
run over the same two-relay rig, with the protocol driven the way the CLI drives
it, or they would be testing a second implementation of the move.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import mobility, sync
from local_operator.network.projection import read_tombstones
from local_operator.session.placement import read_handoff_journal
from tests.unit.network.test_mobility import (  # noqa: F401 — `devices` is a fixture
    SESSION,
    Devices,
    _move,
    _owned_session,
    _transcript,
    devices,
)
from tests.unit.network.test_relay_e2e import _pair_settled


@pytest.fixture()
def pair(request: pytest.FixtureRequest) -> Devices:
    """Two relays with BOTH ends able to serve a control socket.

    The same shape ``test_mobility.pair`` builds (and for the same reason: a move is
    issued by the device that will HOLD the conversation, so B has to answer its own
    CLI). Defined here rather than imported as a fixture name so this module does not
    shadow the shared one in its own signatures.
    """
    value: Devices = request.getfixturevalue("devices")
    server_a, server_b, _host, _port = value
    server_b.bind_control()
    server_b.start()
    return value


#: A wake row for the carry cell: a far-future due time so nothing can fire while
#: the test runs, and one row so ``rebuild_indexes`` has an index to WRITE (an
#: empty schedule set removes the entry, which is the store's own contract).
_WAKE_ROW: dict[str, Any] = {
    "id": "w1",
    "message": "carried",
    "next_due_at": 9_999_999_999_999,
    "every_ms": 86_400_000,
    "fired_count": 1,
    "created_at": 1,
    "notify": False,
}


def _custom_entry(custom_type: str, details: dict[str, Any]) -> str:
    return json.dumps(
        {
            "id": f"e-{custom_type}",
            "ts": 1.0,
            "type": "custom",
            "payload": {"custom_type": custom_type, "details": details},
        },
        separators=(",", ":"),
    )


def _own_session_with_wake_and_cwd(server: Any, cwd: str, *, session_id: str = SESSION) -> Path:
    """A session this device owns, carrying one wake and a desktop marker cwd."""
    directory = _owned_session(server, session_id)
    with directory.joinpath("transcript.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(_custom_entry("wake_schedules", {"schedules": [_WAKE_ROW]}) + "\n")
    directory.joinpath("desktop.json").write_text(json.dumps({"cwd": cwd}), encoding="utf-8")
    return directory


def _own_session_with_wake_index_cwd(server: Any, cwd: str, *, session_id: str = SESSION) -> Path:
    """A session with a wake INDEX cwd and NO ``desktop.json`` — a TUI-created one.

    This is the shape the owner reported: the transcript carries the wake schedule
    (so the destination's rebuild has an index to write), the index this device's
    supervisor reads names the working directory, and the session has no marker of
    its own for the copy set to carry.
    """
    from local_operator.wakes import store as wake_store

    directory = _owned_session(server, session_id)
    with directory.joinpath("transcript.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(_custom_entry("wake_schedules", {"schedules": [_WAKE_ROW]}) + "\n")
    wake_store.write_entry(Path(server.root), session_id, cwd=cwd, schedules=[dict(_WAKE_ROW)])
    assert not (directory / "desktop.json").exists(), "this cell needs a marker-less session"
    return directory


def _losing_ready_ask(
    monkeypatch: pytest.MonkeyPatch,
    *,
    code: str = "unreachable",
    message: str = "the reply was lost after the commit",
) -> dict[str, bool]:
    """Let the owner REALLY commit, then lose its answer — a lost reply, not a refusal.

    ``LinkTransport.ask`` raises ``Moved('unreachable')`` when the reply never
    arrives (a timeout, a peer that died mid-answer) and ``Moved('refused')`` when the
    peer answered an ``op: error`` frame instead of an ack (a raise inside its
    handler). Both are the destination's "this device cannot tell", so ``code``
    selects which spelling this cell drives. The deterministic form is to deliver the
    frame to the real transport first — so the source takes its directory out of the
    store and writes its tombstone, exactly as it does in the field — then raise once.
    """
    real_ask = mobility.LinkTransport.ask
    state = {"lost": False}

    def losing_ask(self: Any, frame: dict[str, Any], *, timeout: float | None = None):
        if str(frame.get("phase") or "") == "ready" and not state["lost"]:
            real_ask(self, frame, timeout=timeout)
            state["lost"] = True
            raise mobility.Moved(code, message)
        return real_ask(self, frame, timeout=timeout)

    monkeypatch.setattr(mobility.LinkTransport, "ask", losing_ask)
    return state


def _holders(root_a: Path, root_b: Path, session_id: str) -> list[Path]:
    """Every device holding a TRANSCRIPT for ``session_id`` — INV-1's own count."""
    return [
        root / "sessions" / session_id
        for root in (root_a, root_b)
        if (root / "sessions" / session_id / "transcript.jsonl").is_file()
    ]


# ---------------------------------------------------------------------------
# 1 — completed means verified, and exactly one holder
# ---------------------------------------------------------------------------


def test_a_recall_reports_completed_and_leaves_exactly_one_holder(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    server_a, server_b, _host, _port = pair
    _pair_settled(pair, monkeypatch, role="admin")
    source = _owned_session(server_a)
    before = _transcript(server_a.root, SESSION)

    result = _move(server_b, SESSION, monkeypatch=monkeypatch)

    assert result["ok"] is True, result
    # THE FIELD THE REPORTED FAILURE WAS MISSING: this was a real, verified move.
    assert result["state"] == "completed", result
    assert result["phase"] == "done"
    # EXACTLY ONE HOLDER, holding the whole conversation byte for byte.
    assert not source.exists()
    assert _transcript(server_b.root, SESSION) == before
    # The source is tombstoned to the destination, and its journal is clear.
    assert read_tombstones(server_a.root)[SESSION]["device_id"] == server_b.identity.device_id
    assert read_handoff_journal(server_a.root) == {}
    # Nothing is left in the quarantine: the commit deleted what it renamed away.
    leftover = list((server_a.root / "network" / "handed-away").glob(f"{SESSION}.*"))
    assert leftover == [], leftover


# ---------------------------------------------------------------------------
# 2 — a source that cannot let go refuses, and changes nothing
# ---------------------------------------------------------------------------


def test_a_source_that_cannot_hand_the_directory_away_refuses_and_changes_nothing(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.session import cleanup as cleanup_mod

    server_a, server_b, _host, _port = pair
    _pair_settled(pair, monkeypatch, role="admin")
    source = _owned_session(server_a)
    before = _transcript(server_a.root, SESSION)

    real = cleanup_mod.hand_off_session_dir
    monkeypatch.setattr(
        cleanup_mod,
        "hand_off_session_dir",
        lambda target, **kwargs: cleanup_mod.HandOffOutcome(
            False, "the store guard refused: patched by the test"
        ),
    )

    refused = _move(server_b, SESSION, monkeypatch=monkeypatch)

    assert refused["ok"] is False, refused
    assert refused["code"] == "cleanup_refused", refused
    assert refused["changed"] is False, refused
    # THE CAUSE IS IN THE SENTENCE (QA round 1, Q-4), not only in the relay log.
    assert "the store guard refused: patched by the test" in refused["message"], refused
    # NOTHING MOVED: no tombstone, the source's directory and transcript intact.
    assert SESSION not in read_tombstones(server_a.root)
    assert source.is_dir()
    assert _transcript(server_a.root, SESSION) == before
    assert read_handoff_journal(server_a.root) == {}
    # ... and the destination rolled its staging back rather than keeping bytes that
    # describe a handoff nobody committed.
    assert not (server_b.root / "sessions" / SESSION).exists()
    assert not sync.staging_dir(server_b.root, SESSION).exists()

    # THE RETRY IS SAFE AND COMPLETES: a refused move changed nothing, which is
    # exactly what ``changed: false`` promises a front end.
    monkeypatch.setattr(cleanup_mod, "hand_off_session_dir", real)
    retried = _move(server_b, SESSION, monkeypatch=monkeypatch)
    assert retried["ok"] is True, retried
    assert retried["state"] == "completed", retried
    assert _transcript(server_b.root, SESSION) == before
    assert not source.exists()


# ---------------------------------------------------------------------------
# 3 — a committed id the owner still lists is a contradiction
# ---------------------------------------------------------------------------


def test_a_committed_source_whose_owner_still_lists_it_is_not_a_success(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    server_a, server_b, _host, _port = pair
    _pair_settled(pair, monkeypatch, role="admin")
    _owned_session(server_a)
    before = _transcript(server_a.root, SESSION)

    real_ask = mobility.LinkTransport.ask
    status_asks: list[dict[str, Any]] = []

    def contradictory_ask(self: Any, frame: dict[str, Any], *, timeout: float | None = None):
        """Answer the FIRST status honestly, then report the owner still holding it.

        The first status is the move's own handshake; from the second on, the only
        status ask is the post-commit contradiction check, and that is the one this
        cell makes lie. The transport is otherwise the real one, so the transfer
        itself is untouched.
        """
        if str(frame.get("phase") or "") == "status":
            status_asks.append(frame)
            if len(status_asks) > 1:
                return {
                    "result": "state",
                    "owner": True,
                    "busy": False,
                    "archived": False,
                    "pending": None,
                    "queue": None,
                    "session_id": SESSION,
                }
        return real_ask(self, frame, timeout=timeout)

    monkeypatch.setattr(mobility.LinkTransport, "ask", contradictory_ask)

    result = _move(server_b, SESSION, monkeypatch=monkeypatch)

    assert len(status_asks) > 1, "the post-commit verification status was never asked"
    assert result["ok"] is False, result
    assert result["code"] == "duplicate", result
    assert result["changed"] is True, result
    assert result["phase_reached"] == "committed", result
    # THE POST-PROMOTE WORDING IS PRESERVED for the live path (R2-2's default).
    assert "The copy here is committed" in result["message"], result
    # The contradiction is a fact about the OWNER's record: the promote really did
    # land here, which is why ``changed`` is True and a retry is not the remedy.
    assert _transcript(server_b.root, SESSION) == before


# ---------------------------------------------------------------------------
# 4 — a queued receipt is in progress, and its words say so
# ---------------------------------------------------------------------------


def test_a_queued_receipt_is_in_progress_not_completed() -> None:
    from local_operator import cli

    from_block = mobility.MoveDevice(device_id="d_here", name="here")
    to_block = mobility.MoveDevice(device_id="d_peer", name="pixel-8")
    queued = mobility._queued_receipt(
        SESSION, {"phase": "queued"}, keep=False, from_block=from_block, to_block=to_block
    )

    assert queued["state"] == "in_progress", queued
    lines = cli._sessions_move_words(dict(queued), session_id=SESSION, to="pixel-8")
    assert not any("Moved" in line for line in lines), lines
    assert "nothing has moved yet" in lines[0], lines
    assert "phase: queued" in lines[0], lines
    assert any("--cancel-queued" in line for line in lines), lines

    # THE COPY HAS STARTED: from ``paused``/``copying`` the cancel hint is a lie
    # (``move_queue.cancel`` answers ``too_late``), and the sentence says so.
    copying = mobility._queued_receipt(
        SESSION, {"phase": "copying"}, keep=False, from_block=from_block, to_block=to_block
    )
    copying_lines = cli._sessions_move_words(dict(copying), session_id=SESSION, to="pixel-8")
    assert "the copy has started" in copying_lines[0], copying_lines
    assert not any("--cancel-queued" in line for line in copying_lines), copying_lines

    # AND THE SAME RULE WITHOUT A QUEUE BLOCK: an in-progress receipt never says
    # "Moved", whatever other fields it carries.
    in_flight = dict(queued)
    del in_flight["queue"]
    in_flight["phase"] = "handing_off"
    in_flight_lines = cli._sessions_move_words(in_flight, session_id=SESSION, to="pixel-8")
    assert not any("Moved" in line for line in in_flight_lines), in_flight_lines
    # NOT A QUEUED MOVE EITHER (QA round 1, Q-3): no queue block means no queue, and
    # the sentence names the step it is actually at.
    assert "still in progress" in in_flight_lines[0], in_flight_lines
    assert "nothing has landed yet" in in_flight_lines[0], in_flight_lines
    assert "Queued" not in " ".join(in_flight_lines), in_flight_lines

    # AND THE DESKTOP/PHONE PLANE AGREES: a queued receipt never retires the source,
    # and the model accepts the shape it produces (the brief was "no surface can say
    # 'the copy here is gone' about a queued one").
    from local_operator.server.models.desktop_mesh import TransferReceipt
    from local_operator.server.utils.desktop_mesh import transfer_receipt

    desktop = transfer_receipt(dict(queued), session_id=SESSION, keep=False, to="pixel-8")
    assert desktop["state"] == "in_progress", desktop
    assert desktop["source_retired"] is False, desktop
    TransferReceipt(**{**desktop, "replayed": False})


# ---------------------------------------------------------------------------
# 5 — the silent peer is named
# ---------------------------------------------------------------------------


def test_an_unreachable_owner_is_named(pair: Devices, monkeypatch: pytest.MonkeyPatch) -> None:
    server_a, server_b, _host, _port = pair
    _pair_settled(pair, monkeypatch, role="admin")
    source = _owned_session(server_a)
    before = _transcript(server_a.root, SESSION)

    # §8.3's shape for a peer that stopped answering: NO rows for it, one
    # ``reachable: false`` block. Crafted to the contract because the live fan-out
    # emits no rows for such a peer (a listing must not show phantom rows).
    silent = {
        "device_id": server_a.identity.device_id,
        "name": "cloud-node-1",
        "reachable": False,
        "age_s": 42,
    }
    monkeypatch.setattr(
        server_b,
        "federated_rows",
        lambda: {"sessions": [], "peers": {silent["device_id"]: silent}},
    )

    refused = _move(server_b, SESSION, monkeypatch=monkeypatch)

    assert refused["ok"] is False, refused
    assert refused["code"] == "unreachable", refused
    message = str(refused["message"])
    assert "cloud-node-1" in message, message
    assert SESSION in message, message
    assert "did not answer" in message and "retry when it is back" in message, message
    assert refused["changed"] is False, refused
    assert _transcript(server_a.root, SESSION) == before
    assert source.is_dir(), "the owner still holds it"
    assert not (server_b.root / "sessions" / SESSION).exists()


# ---------------------------------------------------------------------------
# 6 — the cwd is rewritten for this host, and only the marker changes
# ---------------------------------------------------------------------------


def test_a_moved_session_rewrites_a_cwd_that_does_not_exist_here(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    server_a, server_b, _host, _port = pair
    _pair_settled(pair, monkeypatch, role="admin")
    remote_cwd = f"/home/ec2-user/lop-cwd-{uuid.uuid4().hex[:8]}"
    _own_session_with_wake_and_cwd(server_a, remote_cwd)
    before = _transcript(server_a.root, SESSION)
    # THE MAPPING'S FIRST STEP, made available: the same trailing component under
    # this user's home (the isolated HOME every unit test runs with).
    mapped = Path(os.path.expanduser("~")) / Path(remote_cwd).name
    mapped.mkdir(parents=True, exist_ok=True)

    result = _move(server_b, SESSION, monkeypatch=monkeypatch)

    assert result["ok"] is True, result
    assert result["state"] == "completed", result
    landed = server_b.root / "sessions" / SESSION
    marker = json.loads((landed / "desktop.json").read_text(encoding="utf-8"))
    assert marker["cwd"] == str(mapped), marker
    # THE REBUILT INDEX CARRIES THE REWRITTEN PATH, which is the reason the rewrite
    # happens before ``carry.rebuild_indexes`` reads the marker.
    index = json.loads((server_b.root / "wakes" / f"{SESSION}.json").read_text(encoding="utf-8"))
    assert index["cwd"] == str(mapped), index
    # THE CONVERSATION IS UNTOUCHED: a cwd is host state, not transcript content.
    assert _transcript(server_b.root, SESSION) == before
    # ... and the receipt names both paths.
    assert (result.get("carry") or {}).get("cwd") == {"from": remote_cwd, "to": str(mapped)}, result


def test_a_moved_session_keeps_a_cwd_that_exists_here(
    pair: Devices, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    server_a, server_b, _host, _port = pair
    _pair_settled(pair, monkeypatch, role="admin")
    here = str(tmp_path)
    _own_session_with_wake_and_cwd(server_a, here)

    result = _move(server_b, SESSION, monkeypatch=monkeypatch)

    assert result["ok"] is True, result
    landed = server_b.root / "sessions" / SESSION
    marker = json.loads((landed / "desktop.json").read_text(encoding="utf-8"))
    assert marker["cwd"] == here, marker
    assert "cwd" not in (result.get("carry") or {}), result


# ---------------------------------------------------------------------------
# 7 — the reverse direction: an offload to a peer that is not there
# ---------------------------------------------------------------------------


def test_offload_to_a_dead_peer_fails_and_changes_nothing(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    devices_value: Devices = request.getfixturevalue("devices")
    server_a, server_b, _host, _port = devices_value
    # B is dial-only in the shared fixture (it never starts), which is exactly the
    # dead peer an offload must refuse instead of waiting out: pairing succeeds,
    # nothing listens afterwards.
    _pair_settled(devices_value, monkeypatch, role="admin", settings=server_b.settings)
    source = _owned_session(server_a)
    before = _transcript(server_a.root, SESSION)

    refused = _move(server_a, SESSION, to=server_b.identity.device_id, monkeypatch=monkeypatch)

    assert refused["ok"] is False, refused
    assert refused["code"] == "unreachable", refused
    assert refused["changed"] is False, refused
    # NOTHING MOVED, ON EITHER SIDE.
    assert source.is_dir()
    assert _transcript(server_a.root, SESSION) == before
    assert SESSION not in read_tombstones(server_a.root)
    assert read_handoff_journal(server_a.root) == {}
    assert not (server_b.root / "sessions" / SESSION).exists()


# ---------------------------------------------------------------------------
# 8 — a lost reply is not a refusal: the verified copy survives
# ---------------------------------------------------------------------------


def test_a_lost_ready_reply_keeps_the_verified_copy_and_reports_unconfirmed(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review R1-2 / QA Q-1: a lost ``ready`` reply must not delete the only copy.

    With the new commit order the owner takes its directory out of the store and
    deletes the quarantine BEFORE it answers, so a reply that never arrives is not a
    refusal: the source has already committed. The old branch treated any ``Moved``
    as a refusal, cleared the destination's entry and swept the verified staging —
    zero copies, reported as ``changed: false``. This cell drives the owner's real
    commit and then loses the answer, which is the deterministic spelling of a
    timeout.
    """
    server_a, server_b, _host, _port = pair
    _pair_settled(pair, monkeypatch, role="admin")
    source = _owned_session(server_a)
    before = _transcript(server_a.root, SESSION)
    staging = sync.staging_dir(server_b.root, SESSION)

    state = _losing_ready_ask(monkeypatch)
    result = _move(server_b, SESSION, monkeypatch=monkeypatch)

    assert state["lost"] is True, "the ready frame never reached the owner"
    assert result["ok"] is False, result
    assert result["code"] == "unconfirmed", result
    # A FAILED MOVE THAT CHANGED EVERYTHING IS STILL ``changed: True``: a caller must
    # not retry as though nothing had happened.
    assert result["changed"] is True, result
    assert result["phase_reached"] == "handing_off", result
    # THE SENTENCE NAMES WHERE THE ONLY COPY IS and says not to retry blindly.
    assert str(staging) in result["message"], result
    assert "Do not retry as though nothing happened" in result["message"], result

    # THE SOURCE REALLY COMMITTED: out of the store, tombstoned to this device.
    assert not source.exists()
    assert read_tombstones(server_a.root)[SESSION]["device_id"] == server_b.identity.device_id
    assert read_handoff_journal(server_a.root) == {}
    # ... and NOTHING live holds the transcript at this instant: the staging copy is
    # the only one left, which is exactly why it must not be swept.
    assert _holders(server_a.root, server_b.root, SESSION) == []
    # THE DESTINATION KEPT IT, and kept the journal entry that describes it.
    assert (staging / "ready.json").is_file()
    entry = read_handoff_journal(server_b.root)[SESSION]
    assert entry["role"] == "destination", entry
    assert entry["phase"] == "handing-off", entry
    assert not (server_b.root / "sessions" / SESSION).exists()

    # THE RECOVERY THE SENTENCE PROMISES: a later relay reconciles this device
    # against the owner's own record, and the verified copy is adopted.
    report = mobility.reconcile(server_b.root, server=server_b, own_instance="restarted-instance-b")
    assert [row["action"] for row in report] == ["promoted"], report
    assert _holders(server_a.root, server_b.root, SESSION) == [server_b.root / "sessions" / SESSION]
    assert _transcript(server_b.root, SESSION) == before
    assert read_handoff_journal(server_b.root) == {}
    assert not (server_b.root / "sessions" / SESSION / "ready.json").exists()
    assert not staging.exists()


def test_a_source_whose_tombstone_write_fails_still_commits_and_keeps_one_holder(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review R2-1(a): the rename is the commit point; bookkeeping cannot undo it.

    Before this cell, a ``write_tombstone`` raise AFTER the quarantine rename escaped
    ``_source_commit`` to the relay's ``op: error`` frame; ``LinkTransport.ask`` reads
    that as ``Moved("refused")``, and the destination answers a refusal by clearing
    its entry and sweeping the verified staging — zero copies, reported as
    ``changed: false``. A full disk is the realistic spelling of the raise. The answer
    must be ``committed``: the destination keeps the only copy, the source's journal
    stays ``handing-off`` (the phase forbids a rollback), and a later reconcile on the
    source writes the tombstone and clears the entry.
    """
    from local_operator.network import projection

    server_a, server_b, _host, _port = pair
    _pair_settled(pair, monkeypatch, role="admin")
    source = _owned_session(server_a)
    before = _transcript(server_a.root, SESSION)

    real_write = projection.write_tombstone
    calls = {"n": 0}

    def failing_write(session_id: str, **kwargs: Any) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError(28, "No space left on device")
        real_write(session_id, **kwargs)

    monkeypatch.setattr(projection, "write_tombstone", failing_write)

    result = _move(server_b, SESSION, monkeypatch=monkeypatch)

    # ONLY THE SOURCE'S WRITE WAS TRIED, and it raised: the destination's promote
    # never needs a tombstone of its own.
    assert calls["n"] == 1, calls
    # THE MOVE IS REPORTED FOR WHAT IT IS: a real, verified commit.
    assert result["ok"] is True, result
    assert result["state"] == "completed", result
    assert not source.exists()
    assert _holders(server_a.root, server_b.root, SESSION) == [server_b.root / "sessions" / SESSION]
    assert _transcript(server_b.root, SESSION) == before
    # THE SOURCE'S ID IS NOT DURABLY THE DESTINATION'S YET, and it says so: the
    # journal stays at ``handing-off``, which is the phase reconcile completes.
    assert SESSION not in read_tombstones(server_a.root)
    entry = read_handoff_journal(server_a.root)[SESSION]
    assert entry["phase"] == "handing-off", entry

    # THE RECOVERY THE JOURNAL PROMISES: a restarted relay finishes the bookkeeping.
    report = mobility.reconcile(server_a.root, server=server_a, own_instance="restarted-a")
    assert [row["action"] for row in report] == ["completed"], report
    assert read_tombstones(server_a.root)[SESSION]["device_id"] == server_b.identity.device_id
    assert read_handoff_journal(server_a.root) == {}
    # ... and the id is still held exactly once.
    assert _holders(server_a.root, server_b.root, SESSION) == [server_b.root / "sessions" / SESSION]


def test_a_queued_move_whose_tombstone_fails_folds_its_record_resumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review R3-1: the queue fold belongs to the COMMIT, not to the tombstone.

    ``_settle_source_commit`` returned on a failed tombstone BEFORE its derived-state
    step, and that step is the only caller of ``move_queue.note_committed`` — the fold
    from ``copying`` to ``resumed``. A queued move whose tombstone write kept failing
    (a full disk is the realistic spelling) therefore stayed live, and the driver's
    tombstone-only watch folded it terminally ``failed`` after ``COPY_WATCH_S`` with
    "this device still holds the conversation" — while this device's directory was
    gone and the destination held and ran the only copy. Terminal is write-once, so
    the reconcile that later wrote the tombstone could not repair the record.

    Driven on a source root in exactly the post-rename state the commit leaves: the
    directory is out of ``sessions/``, the journal says ``handing-off``, the queue
    record is live at ``copying``. The queue fold and the F5 prune must happen
    whatever the tombstone does; the journal is what stays held back.
    """
    from local_operator.network import move_queue, projection
    from local_operator.session.placement import write_handoff_entry
    from local_operator.wakes import store as wake_store

    # THE QUEUED MOVE'S RECORD, in the phase the driver leaves while it copies.
    record, created = move_queue.enqueue(
        tmp_path,
        SESSION,
        to_device="device-b",
        to_name="pixel-8",
        request_id="req-r3-1",
        requested_by="device-b",
    )
    assert created, record
    move_queue.transition(tmp_path, SESSION, move_queue.QUEUE_PHASE_COPYING, writer="relay")
    # Derived state the commit must prune: a wake index entry for the moved session.
    wake_store.write_entry(tmp_path, SESSION, cwd="/tmp/here", schedules=[dict(_WAKE_ROW)])
    assert wake_store.read_entry(tmp_path, SESSION) is not None

    # THE POST-RENAME STATE: the directory has left ``sessions/`` and the journal
    # says ``handing-off``, so only the commit's durable tail remains to run.
    write_handoff_entry(
        tmp_path,
        SESSION,
        {
            "role": "source",
            "phase": "handing-off",
            "to_device": "device-b",
            "to_name": "pixel-8",
            "network_id": "net-1",
            "lease_epoch": "1",
            "instance_id": "dead-relay",
        },
    )

    def refusing_write(session_id: str, **kwargs: Any) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(projection, "write_tombstone", refusing_write)

    settled = mobility._settle_source_commit(
        tmp_path,
        SESSION,
        device_id="device-b",
        device_name="pixel-8",
        network_id="net-1",
    )

    assert settled is False, "the tombstone did not land, and the answer must say so"
    # THE FOLD THE DRIVER'S WATCH READS: terminal ``resumed``, never a live record
    # the watch would fail after COPY_WATCH_S.
    folded = move_queue.read_record(tmp_path, SESSION) or {}
    assert folded.get("phase") == move_queue.QUEUE_PHASE_RESUMED, folded
    assert move_queue.is_terminal(folded), folded
    assert not folded.get("code"), folded
    assert [stamp.get("phase") for stamp in folded.get("phases") or []] == [
        "queued",
        "copying",
        "resumed",
    ], folded
    # THE F5 PRUNE RAN ON THE SAME STEP, so this device fires none of the moved
    # session's wakes any more.
    assert wake_store.read_entry(tmp_path, SESSION) is None
    # THE JOURNAL IS HELD BACK, because it is how the repair finds the tombstone
    # write that did not happen.
    assert read_handoff_journal(tmp_path)[SESSION]["phase"] == "handing-off"
    assert SESSION not in read_tombstones(tmp_path)

    # THE REPAIR: a restarted relay writes the tombstone and finishes the handoff...
    monkeypatch.undo()
    report = mobility.reconcile(tmp_path, own_instance="restarted-a")
    assert [row["action"] for row in report] == ["completed"], report
    assert read_tombstones(tmp_path)[SESSION]["device_id"] == "device-b"
    assert read_handoff_journal(tmp_path) == {}
    # ... and the terminal record is STILL the commit's own word: write-once means
    # the failed-tombstone window cannot be turned into the driver's failed report
    # by anything that runs later.
    final = move_queue.read_record(tmp_path, SESSION) or {}
    assert final.get("phase") == move_queue.QUEUE_PHASE_RESUMED, final
    assert move_queue.is_terminal(final), final


def test_an_error_frame_after_the_commit_keeps_the_verified_copy_too(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review R2-1(b): an ``op: error`` answer is not proof of a refusal.

    ``Moved("refused")`` is ``LinkTransport``'s word for a peer that raised inside its
    handler — here the real commit ran and only the answer turned into an error frame.
    Every GENUINE refusal travels inside the ack with its own code (``busy``,
    ``digest_mismatch``, ``cleanup_refused``, ...), so this shape joins ``unreachable``
    in "this device cannot tell": the staging and the entry stay, the receipt is
    ``unconfirmed``/``changed: true``, and the reconcile adopts the copy.
    """
    server_a, server_b, _host, _port = pair
    _pair_settled(pair, monkeypatch, role="admin")
    source = _owned_session(server_a)
    before = _transcript(server_a.root, SESSION)
    staging = sync.staging_dir(server_b.root, SESSION)

    state = _losing_ready_ask(
        monkeypatch,
        code="refused",
        message="net_session_move failed on this device: [Errno 28] No space left on device",
    )
    result = _move(server_b, SESSION, monkeypatch=monkeypatch)

    assert state["lost"] is True, "the ready frame never reached the owner"
    assert result["ok"] is False, result
    assert result["code"] == "unconfirmed", result
    assert result["changed"] is True, result
    assert result["phase_reached"] == "handing_off", result
    # THE SENTENCE NAMES THE FAILURE THAT ACTUALLY HAPPENED, not the silence shape.
    assert str(staging) in result["message"], result
    assert "No space left on device" in result["message"], result
    assert "did not answer" not in result["message"], result

    # THE SOURCE REALLY COMMITTED BEFORE THE ERROR FRAME (the ask was delegated first).
    assert not source.exists()
    assert read_tombstones(server_a.root)[SESSION]["device_id"] == server_b.identity.device_id
    # ... and with nothing live holding the transcript, the staging is the only copy.
    assert _holders(server_a.root, server_b.root, SESSION) == []
    assert (staging / "ready.json").is_file()
    entry = read_handoff_journal(server_b.root)[SESSION]
    assert entry["role"] == "destination", entry
    assert entry["phase"] == "handing-off", entry
    assert not (server_b.root / "sessions" / SESSION).exists()

    # THE RECOVERY: one holder, the same bytes.
    report = mobility.reconcile(server_b.root, server=server_b, own_instance="restarted-b")
    assert [row["action"] for row in report] == ["promoted"], report
    assert _holders(server_a.root, server_b.root, SESSION) == [server_b.root / "sessions" / SESSION]
    assert _transcript(server_b.root, SESSION) == before
    assert read_handoff_journal(server_b.root) == {}


def test_a_recovery_whose_tombstone_cannot_be_written_reports_cleanup_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review R2-1(a): the crash-recovery half must not raise either.

    ``_complete_handoff`` runs behind :func:`reconcile` with no request of its own, so
    a raise here does not become an error frame — it aborts the recovery table, where
    the later entries are other moves a person is waiting on. The phase ``handing-off``
    forbids a rollback, so a tombstone that cannot be written leaves the journal alone
    and reports ``cleanup_pending``: the next attempt writes it. The test then does
    exactly that and finishes the handoff.
    """
    from local_operator.network import projection
    from local_operator.session.placement import write_handoff_entry

    entry = {
        "role": "source",
        "phase": "handing-off",
        "to_device": "device-b",
        "to_name": "pixel-8",
        "network_id": "net-1",
        "lease_epoch": "1",
        "instance_id": "dead-relay",
    }
    write_handoff_entry(tmp_path, SESSION, entry)

    def refusing_write(session_id: str, **kwargs: Any) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(projection, "write_tombstone", refusing_write)

    answer = mobility._complete_handoff(tmp_path, SESSION, entry)

    assert answer["action"] == "cleanup_pending", answer
    assert SESSION not in read_tombstones(tmp_path)
    assert read_handoff_journal(tmp_path)[SESSION]["phase"] == "handing-off"

    # THE NEXT ATTEMPT FINISHES IT, which is what ``cleanup_pending`` promises.
    monkeypatch.undo()
    finished = mobility._complete_handoff(tmp_path, SESSION, entry)
    assert finished["action"] == "completed", finished
    assert read_tombstones(tmp_path)[SESSION]["device_id"] == "device-b"
    assert read_handoff_journal(tmp_path) == {}


def test_the_tombstone_path_checks_the_contradiction_before_promoting(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review R1-5: the contradiction check must run BEFORE the promote.

    The startup path adopts from the owner's tombstone — the owner's own statement
    that the id left — so an owner that still answers ``owner: true`` is
    contradicting the record being acted on. Checking after the promote minted the
    second live copy this refusal reports, and cleared the entry for a handoff that
    had not finished. Here nothing is promoted, the staging and the entry survive,
    and the refusal says which phase it stopped at.
    """
    server_a, server_b, _host, _port = pair
    _pair_settled(pair, monkeypatch, role="admin")
    _owned_session(server_a)
    before = _transcript(server_a.root, SESSION)
    staging = sync.staging_dir(server_b.root, SESSION)
    _losing_ready_ask(monkeypatch)
    assert _move(server_b, SESSION, monkeypatch=monkeypatch)["code"] == "unconfirmed"

    real_ask = mobility.LinkTransport.ask

    def owner_still_lists(self: Any, frame: dict[str, Any], *, timeout: float | None = None):
        if str(frame.get("phase") or "") == "status":
            return {
                "result": "state",
                "owner": True,
                "busy": False,
                "archived": False,
                "pending": None,
                "queue": None,
                "session_id": SESSION,
            }
        return real_ask(self, frame, timeout=timeout)

    monkeypatch.setattr(mobility.LinkTransport, "ask", owner_still_lists)

    result, refusal, _target = mobility._finish_from_tombstone(
        server_b, SESSION, owner_device=server_a.identity.device_id
    )

    assert result is None, result
    assert refusal is not None, "the contradiction was not reported"
    assert refusal["code"] == "duplicate", refusal
    assert refusal["phase_reached"] == "handing_off", refusal
    assert refusal["changed"] is True, refusal
    # THE SENTENCE MATCHES WHAT HAPPENED (review round 2, R2-2): nothing was promoted
    # here, so it must not claim a committed copy or two live copies.
    assert "did not promote" in refusal["message"], refusal
    assert "still staged here" in refusal["message"], refusal
    assert "The copy here is committed" not in refusal["message"], refusal
    assert "both devices now list" not in refusal["message"], refusal
    # NOTHING WAS PROMOTED, and both artefacts of the unfinished handoff survive.
    assert not (server_b.root / "sessions" / SESSION).exists()
    assert (staging / "ready.json").is_file()
    assert read_handoff_journal(server_b.root)[SESSION]["role"] == "destination"

    # THE NORMAL TOMBSTONE PATH STILL PROMOTES the same copy once the owner answers
    # its own record, and leaves exactly one holder holding the same bytes.
    monkeypatch.setattr(mobility.LinkTransport, "ask", real_ask)
    report = mobility.reconcile(server_b.root, server=server_b, own_instance="restarted-instance-b")
    assert [row["action"] for row in report] == ["promoted"], report
    assert _holders(server_a.root, server_b.root, SESSION) == [server_b.root / "sessions" / SESSION]
    assert _transcript(server_b.root, SESSION) == before


# ---------------------------------------------------------------------------
# 9 — the cwd a marker-less session has nowhere else to record
# ---------------------------------------------------------------------------


def test_a_marker_less_session_carries_the_cwd_its_wake_index_records(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The owner's clause 4: a TUI-created session has no ``desktop.json``.

    Its cwd lives in this device's wake index, which the destination cannot see and
    the copy set does not carry. Without the prepare's additive ``cwd`` the moved
    conversation opened at the session directory — a path that exists everywhere and
    means nothing.
    """
    server_a, server_b, _host, _port = pair
    _pair_settled(pair, monkeypatch, role="admin")
    remote_cwd = f"/home/ec2-user/lop-wake-{uuid.uuid4().hex[:8]}"
    _own_session_with_wake_index_cwd(server_a, remote_cwd)
    before = _transcript(server_a.root, SESSION)
    # THE MAPPING'S FIRST STEP, made available: the same trailing component under
    # this user's home (the isolated HOME every unit test runs with).
    mapped = Path(os.path.expanduser("~")) / Path(remote_cwd).name
    mapped.mkdir(parents=True, exist_ok=True)

    result = _move(server_b, SESSION, monkeypatch=monkeypatch)

    assert result["ok"] is True, result
    assert result["state"] == "completed", result
    landed = server_b.root / "sessions" / SESSION
    # THE MARKER IS INVENTED FROM THE SOURCE'S RECORD, at the mapped local path.
    marker = json.loads((landed / "desktop.json").read_text(encoding="utf-8"))
    assert marker["cwd"] == str(mapped), marker
    # ... and the index the supervisor reads agrees with it, because the marker was
    # written before the rebuild.
    index = json.loads((server_b.root / "wakes" / f"{SESSION}.json").read_text(encoding="utf-8"))
    assert index["cwd"] == str(mapped), index
    # THE CONVERSATION IS UNTOUCHED: a cwd is host state, not transcript content.
    assert _transcript(server_b.root, SESSION) == before
    assert (result.get("carry") or {}).get("cwd") == {"from": remote_cwd, "to": str(mapped)}, result


def test_a_move_with_no_known_cwd_invents_no_marker(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing is known, so nothing is written — and the move still completes."""
    server_a, server_b, _host, _port = pair
    _pair_settled(pair, monkeypatch, role="admin")
    source = _owned_session(server_a)
    before = _transcript(server_a.root, SESSION)

    result = _move(server_b, SESSION, monkeypatch=monkeypatch)

    assert result["ok"] is True, result
    assert result["state"] == "completed", result
    landed = server_b.root / "sessions" / SESSION
    assert not (landed / "desktop.json").exists(), "a marker was invented for an unknown cwd"
    assert "cwd" not in (result.get("carry") or {}), result
    assert _transcript(server_b.root, SESSION) == before
    assert not source.exists()


def test_an_explicit_refusal_after_the_copy_still_rolls_the_staging_back(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other half of the lost-reply branch: a real refusal is still a refusal.

    An owner that ANSWERS ``ready`` with a refusal has not committed — it still holds
    its directory — so the destination must sweep its verified staging and release the
    id, exactly as before. Keeping the bytes here would leave a copy that describes a
    handoff nobody will make, and a retry would resume against a plan the source has
    already abandoned.
    """
    server_a, server_b, _host, _port = pair
    _pair_settled(pair, monkeypatch, role="admin")
    source = _owned_session(server_a)
    before = _transcript(server_a.root, SESSION)
    staging = sync.staging_dir(server_b.root, SESSION)
    real_ask = mobility.LinkTransport.ask

    def refusing_ask(self: Any, frame: dict[str, Any], *, timeout: float | None = None):
        if str(frame.get("phase") or "") == "ready":
            # NOT DELEGATED: the owner never commits, it refuses the commit.
            raise mobility.Moved("busy", "another front end is attached to this conversation")
        return real_ask(self, frame, timeout=timeout)

    monkeypatch.setattr(mobility.LinkTransport, "ask", refusing_ask)

    refused = _move(server_b, SESSION, monkeypatch=monkeypatch)

    assert refused["ok"] is False, refused
    assert refused["code"] == "busy", refused
    assert refused["changed"] is False, refused
    # THE SOURCE IS UNTOUCHED, because it never committed.
    assert source.is_dir()
    assert _transcript(server_a.root, SESSION) == before
    assert SESSION not in read_tombstones(server_a.root)
    # ... AND THE DESTINATION ROLLED BACK, which is what a retry relies on.
    assert not staging.exists()
    assert read_handoff_journal(server_b.root) == {}
    assert _holders(server_a.root, server_b.root, SESSION) == [source]


def test_an_error_frame_before_the_commit_keeps_the_staging_and_loses_the_claim(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review R3-2: ``Moved('refused')`` raised BEFORE the rename still means
    "this device cannot tell", and the owner's own answer is what settles it.

    ``refused`` is the transport's word for a peer that threw inside its handler, and
    the R2-1(b) arm keeps the verified staging for it whether the throw came after
    the commit (the cell above drives the real commit, then loses the answer) or
    before it. This cell pins the BEFORE flavour: the owner never receives ``ready``,
    so its directory and transcript stay, there is no tombstone, and the receipt is
    ``unconfirmed`` with ``changed: true`` — a caller must not retry blind. The
    destination still cannot adopt anything, so its reconcile asks the owner, hears
    "mine", drops the claim and KEEPS the verified bytes: a retry resumes them, and
    exactly one device still holds the conversation.
    """
    server_a, server_b, _host, _port = pair
    _pair_settled(pair, monkeypatch, role="admin")
    source = _owned_session(server_a)
    before = _transcript(server_a.root, SESSION)
    staging = sync.staging_dir(server_b.root, SESSION)
    real_ask = mobility.LinkTransport.ask
    seen = {"ready": 0}

    def refusing_ask(self: Any, frame: dict[str, Any], *, timeout: float | None = None):
        if str(frame.get("phase") or "") == "ready":
            # NOT DELEGATED: the owner throws before it commits, so it still owns.
            seen["ready"] += 1
            raise mobility.Moved("refused", "the owner raised before it committed")
        return real_ask(self, frame, timeout=timeout)

    monkeypatch.setattr(mobility.LinkTransport, "ask", refusing_ask)

    result = _move(server_b, SESSION, monkeypatch=monkeypatch)

    assert seen["ready"] == 1, seen
    assert result["ok"] is False, result
    assert result["code"] == "unconfirmed", result
    assert result["changed"] is True, result
    assert result["phase_reached"] == "handing_off", result
    # THE SOURCE STILL OWNS IT: the commit never ran, so the directory, the
    # transcript and the id are all where they were.
    assert source.is_dir()
    assert _transcript(server_a.root, SESSION) == before
    assert SESSION not in read_tombstones(server_a.root)
    assert _holders(server_a.root, server_b.root, SESSION) == [source]
    # ... AND THE DESTINATION KEPT WHAT IT VERIFIED, plus the entry that describes
    # it — sweeping either here is how a lost reply used to lose the only copy.
    assert (staging / "ready.json").is_file()
    entry = read_handoff_journal(server_b.root)[SESSION]
    assert entry["role"] == "destination", entry
    assert entry["phase"] == "handing-off", entry
    assert not (server_b.root / "sessions" / SESSION).exists()

    # THE RECOVERY: the owner answers "still mine", so this device releases the id
    # and keeps the bytes — nothing is promoted, and no second holder appears.
    report = mobility.reconcile(server_b.root, server=server_b, own_instance="restarted-b")
    assert [row["action"] for row in report] == ["rolled_back"], report
    assert read_handoff_journal(server_b.root) == {}
    assert (staging / "ready.json").is_file(), "the verified bytes are a retry's head start"
    assert not (server_b.root / "sessions" / SESSION).exists()
    assert _holders(server_a.root, server_b.root, SESSION) == [source]
    assert SESSION not in read_tombstones(server_a.root)
