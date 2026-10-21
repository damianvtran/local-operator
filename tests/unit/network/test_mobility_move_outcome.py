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
        lambda target, **kwargs: False,  # the directory is still there
    )

    refused = _move(server_b, SESSION, monkeypatch=monkeypatch)

    assert refused["ok"] is False, refused
    assert refused["code"] == "cleanup_refused", refused
    assert refused["changed"] is False, refused
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
    # The contradiction is a fact about the OWNER's record: the promote really did
    # land here, which is why ``changed`` is True and a retry is not the remedy.
    assert _transcript(server_b.root, SESSION) == before


# ---------------------------------------------------------------------------
# 4 — a queued receipt is in progress, and its words say so
# ---------------------------------------------------------------------------


def test_a_queued_receipt_is_in_progress_not_completed() -> None:
    from local_operator import cli

    from_block = {"device_id": "d_here", "name": "here"}
    to_block = {"device_id": "d_peer", "name": "pixel-8"}
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
    assert "nothing has moved yet" in in_flight_lines[0], in_flight_lines

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
