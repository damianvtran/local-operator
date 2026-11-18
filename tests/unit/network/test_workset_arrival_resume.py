"""The active-move resume (docs/design/move-working-set.md §7.4, operator request 2026-11-18).

A conversation that was ACTIVE when it moved (a turn within the keep-alive window, a
queued move resuming, a running subagent) is engaged on arrival BY DEFAULT and handed
exactly ONE continuation note: which worktrees it used and that they stayed on the
source (PR A carries no files), what was left behind, what this device lacks. Cells:

* active → engaged with ``reason: "active"`` and one waking inbox row, marker written;
* the note names the worktree and the device that kept it;
* retry of the same move and a reconcile pass → still ONE row;
* ``no_engage_on_arrival`` → cold, no row;
* an idle conversation (last turn outside the window) → cold, no row (today's arrival);
* a ``keep`` copy of an active conversation → never resumed (the source keeps running);
* an OFFLOAD's receipt (the source's) records the resume it REQUESTED in
  ``resume_requested`` — ``active`` / ``asked`` — because the destination's engage
  outcome never comes back (UI #1978 QA Q-1); keep and opt-out carry none.

Driven through ``mobility.request_move`` on the in-process two-relay rig (recall: B
pulls from A, so ``_destination_move`` runs on B and its receipt carries the
engagement). Only the runtime spawn is stood in for (``engage_session`` on B).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import mobility
from local_operator.session.runtime.inbox import peek_inbox
from tests.unit.network.test_mobility import (  # noqa: F401 — `devices` is a fixture
    SESSION,
    Devices,
    _owned_session,
    devices,
)
from tests.unit.network.test_relay_e2e import _pair_settled
from tests.unit.network.workset_rig import Rig, make_rig, point_session_at


@pytest.fixture()
def pair(request: pytest.FixtureRequest) -> Devices:
    value: Devices = request.getfixturevalue("devices")
    _server_a, server_b, _host, _port = value
    server_b.bind_control()
    server_b.start()
    return value


@pytest.fixture()
def engaged(pair: Devices, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Stand in for the runtime spawn on the destination; record what was engaged."""
    _server_a, server_b, _host, _port = pair
    calls: list[str] = []

    def fake(session_id: str, *, cwd: str = "") -> dict[str, Any]:
        del cwd
        calls.append(session_id)
        return {"engaged": True, "detail": "runtime joining", "session_id": session_id}

    monkeypatch.setattr(server_b, "engage_session", fake)
    return calls


def _last_turn(source: Path, *, ago_s: float) -> None:
    """Append one transcript row whose ``ts`` is ``ago_s`` seconds in the past."""
    row = {"id": "turn-last", "ts": time.time() - ago_s, "type": "message", "payload": {}}
    with (source / "transcript.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def _session(pair: Devices, monkeypatch: pytest.MonkeyPatch, *, ago_s: float) -> Path:
    server_a, server_b, _host, _port = pair
    _pair_settled(pair, monkeypatch, role="admin", settings=server_b.settings)
    source = _owned_session(server_a)
    _last_turn(source, ago_s=ago_s)
    return source


def _move(server: Any, monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> dict[str, Any]:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server.root))
    return dict(mobility.request_move(SESSION, root=server.root, **kwargs))


def _rows(server: Any) -> list[Any]:
    return peek_inbox(Path(server.root) / "sessions" / SESSION)


def test_an_active_move_engages_and_hands_exactly_one_note(
    pair: Devices, monkeypatch: pytest.MonkeyPatch, engaged: list[str]
) -> None:
    _server_a, server_b, _host, _port = pair
    _session(pair, monkeypatch, ago_s=30)

    moved = _move(server_b, monkeypatch, to="local")

    assert moved["ok"] is True, moved
    assert engaged == [SESSION]
    assert moved["engagement"]["reason"] == "active"
    assert moved["engagement"]["arrival_note"] == "delivered"
    # A RECALL carries the outcome itself, so no request record (UI #1978 QA Q-1).
    assert "resume_requested" not in moved
    rows = _rows(server_b)
    assert len(rows) == 1, rows
    assert rows[0].wake is True
    assert rows[0].message_id.startswith("peer-")
    assert rows[0].text.startswith("[session move]")
    marker = Path(server_b.root) / "sessions" / SESSION / mobility.ARRIVAL_NOTE_MARKER
    assert json.loads(marker.read_text(encoding="utf-8"))["message_id"] == rows[0].message_id

    # RETRY of the same move (same lease epoch) and a RECONCILE pass: still one row.
    epoch = json.loads(marker.read_text(encoding="utf-8"))["lease_epoch"]
    again = mobility._arrival_resume_note(
        server_b, SESSION, lease_epoch=epoch, block=None, source="device-a"
    )
    assert again == "already_delivered"
    mobility.reconcile(server_b.root, server=None, only=SESSION, own_instance="other")
    assert len(_rows(server_b)) == 1


def test_the_note_names_the_worktree_and_the_device_that_kept_it(
    pair: Devices, monkeypatch: pytest.MonkeyPatch, engaged: list[str], tmp_path: Path
) -> None:
    _server_a, server_b, _host, _port = pair
    rig: Rig = make_rig(tmp_path / "ws")
    monkeypatch.setenv("HOME", str(rig.home))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    source = _session(pair, monkeypatch, ago_s=10)
    point_session_at(source, rig.feat)
    digest = _move(server_b, monkeypatch, to="local", dry_run=True)["working_set"]["scope_digest"]

    moved = _move(server_b, monkeypatch, to="local", confirm_working_set=digest)

    assert moved["ok"] is True, moved
    assert engaged == [SESSION]
    (row,) = _rows(server_b)
    source_name = moved["from_device"]["name"]
    assert "~/proj/.worktrees/feat (branch feat): NOT on this device" in row.text, row.text
    assert f"it stayed on {source_name}" in row.text
    assert f"lop sessions move {SESSION} --to {source_name}" in row.text


def test_the_opt_out_keeps_the_cold_arrival(
    pair: Devices, monkeypatch: pytest.MonkeyPatch, engaged: list[str]
) -> None:
    _server_a, server_b, _host, _port = pair
    _session(pair, monkeypatch, ago_s=30)

    moved = _move(server_b, monkeypatch, to="local", no_engage_on_arrival=True)

    assert moved["ok"] is True, moved
    assert "engagement" not in moved
    assert engaged == []
    assert _rows(server_b) == []


def test_an_idle_conversation_still_arrives_cold(
    pair: Devices, monkeypatch: pytest.MonkeyPatch, engaged: list[str]
) -> None:
    _server_a, server_b, _host, _port = pair
    # Outside DEFAULT_KEEP_ALIVE_SECONDS (300 s): not active.
    _session(pair, monkeypatch, ago_s=3600)

    moved = _move(server_b, monkeypatch, to="local")

    assert moved["ok"] is True, moved
    assert "engagement" not in moved
    assert engaged == []
    assert _rows(server_b) == []


def test_a_keep_copy_of_an_active_conversation_is_never_resumed(
    pair: Devices, monkeypatch: pytest.MonkeyPatch, engaged: list[str]
) -> None:
    _server_a, server_b, _host, _port = pair
    _session(pair, monkeypatch, ago_s=5)

    copied = _move(server_b, monkeypatch, to="local", keep=True)

    assert copied["ok"] is True, copied
    assert "engagement" not in copied
    assert engaged == []


# ---------------------------------------------------------------------------
# The offload's receipt: what the SOURCE requested (UI #1978 QA Q-1)
# ---------------------------------------------------------------------------
#
# On an offload A builds the receipt and B engages AFTER ``done``, so A cannot report
# the outcome: ``engagement`` stays absent (unknown). The desktop dialog read that
# absence as "Nothing is running on <B> yet" while B was resuming the conversation.
# The receipt now says what A asked for, from the reading A's prepare took.


def _offload(server_a: Any, server_b: Any, monkeypatch: pytest.MonkeyPatch, **kwargs: Any):
    return _move(server_a, monkeypatch, to=server_b.identity.device_id, **kwargs)


def test_an_active_offload_through_the_desktop_route_reports_the_requested_resume(
    pair: Devices, monkeypatch: pytest.MonkeyPatch, engaged: list[str]
) -> None:
    """The REAL desktop route on A, over the two-relay rig: the dialog's own answer."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from local_operator.config import ConfigManager
    from local_operator.server.routes import desktop_mesh

    server_a, server_b, _host, _port = pair
    _session(pair, monkeypatch, ago_s=30)
    token = "t" * 32
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", token)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_a.root))
    app = FastAPI()
    app.state.config_manager = ConfigManager(Path(server_a.root))
    app.include_router(desktop_mesh.router)

    with TestClient(app, headers={"Authorization": f"Bearer {token}"}) as client:
        response = client.post(
            f"/v1/desktop/sessions/{SESSION}/transfer",
            json={"to": server_b.identity.device_id},
        )

    assert response.status_code == 200, response.text
    receipt = response.json()["result"]
    assert receipt["state"] == "completed", receipt
    assert receipt["resume_requested"] == {"reason": "active", "session_id": SESSION}
    # The OUTCOME stays unknown on the source's receipt — never a faked ``engaged``.
    assert receipt["engagement"] is None, receipt


def test_an_offload_asked_to_engage_records_the_request_as_asked(
    pair: Devices, monkeypatch: pytest.MonkeyPatch, engaged: list[str]
) -> None:
    server_a, server_b, _host, _port = pair
    _session(pair, monkeypatch, ago_s=3600)  # idle: only the flag asks

    moved = _offload(server_a, server_b, monkeypatch, engage_on_arrival=True)

    assert moved["ok"] is True, moved
    assert moved["resume_requested"] == {"reason": "asked", "session_id": SESSION}
    assert "engagement" not in moved


@pytest.mark.parametrize(
    ("kwargs", "ago_s"),
    [
        ({"keep": True}, 30),  # a keep copy is never resumed
        ({"no_engage_on_arrival": True}, 30),  # the opt-out
        ({}, 3600),  # an idle conversation arrives cold
    ],
    ids=["keep", "no_engage_on_arrival", "idle"],
)
def test_an_offload_that_requested_no_resume_carries_no_request(
    pair: Devices,
    monkeypatch: pytest.MonkeyPatch,
    engaged: list[str],
    kwargs: dict[str, Any],
    ago_s: float,
) -> None:
    server_a, server_b, _host, _port = pair
    _session(pair, monkeypatch, ago_s=ago_s)

    moved = _offload(server_a, server_b, monkeypatch, **kwargs)

    assert moved["ok"] is True, moved
    assert "resume_requested" not in moved, moved


def test_the_cli_says_the_requested_resume_on_an_offload_receipt() -> None:
    """The CLI reuses its ``--engage-on-arrival`` request sentence, no new copy."""
    from local_operator import cli

    result = {
        "ok": True,
        "session_id": SESSION,
        "new_session_id": SESSION,
        "mode": "move",
        "state": "completed",
        "from_device": {"device_id": "d_source", "name": "this-device"},
        "to_device": {"device_id": "d_destination", "name": "dest-device"},
        "phase": "done",
        "phases": [{"phase": "done", "at": 0.0}],
        "resume_requested": {"reason": "active", "session_id": SESSION},
    }
    asked = "dest-device was asked to start it as soon as it lands"

    lines = cli._sessions_move_words(result, session_id=SESSION, to="dest-device")

    assert any(line.startswith(asked) for line in lines), lines
    del result["resume_requested"]
    silent = cli._sessions_move_words(result, session_id=SESSION, to="dest-device")
    assert not any(line.startswith(asked) for line in silent), silent
