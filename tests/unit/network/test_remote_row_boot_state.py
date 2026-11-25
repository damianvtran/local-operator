"""A remote session whose runtime is booting is not a stopped one (PR #4572 QA Q6).

After ``lop network sessions --send`` returned "<session> is still starting on
<peer>; message NOT queued", the sender's ``lop network sessions`` and the
sessions tool's ``info`` showed that session as ``stored``, pid 0, "not
running" — byte-identical to a session stopped on purpose — so a sender could
not tell a boot (an engage was accepted, a runtime is being constructed) from a
stop or a wedge.

The owner now adds an additive ``boot_state``/``boot_age_s`` claim to a cold row
while a live, identity-verified process holds the session's transcript lease and
no run record is published (the same "constructing" signal ``engage_runtime``
reads). ``state`` stays ``stored`` so every liveness branch, and every older
reader, keeps treating the row as cold. These tests pin:

* the owner row carries ``starting`` with a real nonzero age;
* past the shared cold-boot budget it carries ``not_responding``;
* the CLI text and ``--json`` show / carry it;
* stopped and live rows are byte-unchanged.

The lease is held by THIS test process through the product's own
``acquire_session_lease``, so the probe's birth-token identity check is the real
one, not a stub.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import relay

SESSION = "9f3ac1e0b7d2"
PEER_ID = "d_" + "b" * 32
PEER_NAME = "cloud-node-1"


def _server(root: Path) -> relay.RelayServer:
    return relay.RelayServer(
        root=root, settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1")
    )


def _seed(root: Path, session_id: str = SESSION) -> Path:
    """A session the catalogue ranks: one real turn in its transcript."""
    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "transcript.jsonl").write_text(
        json.dumps({"role": "user", "content": "hello"}) + "\n", encoding="utf-8"
    )
    return directory


def _backdate_claim(directory: Path, seconds: float) -> None:
    """Move the lease claim's acquisition time back, instead of sleeping."""
    from local_operator.session_lease import LEASE_NAME

    stamp = time.time() - seconds
    os.utime(directory / LEASE_NAME, (stamp, stamp))


def _row(server: relay.RelayServer, session_id: str = SESSION) -> dict[str, Any]:
    rows = [row for row in server.local_session_rows() if row["session_id"] == session_id]
    assert len(rows) == 1, rows
    return rows[0]


@pytest.fixture()
def held(root: Path) -> Iterator[Path]:
    """A seeded session whose transcript lease THIS process holds, unpublished."""
    from local_operator.session_lease import acquire_session_lease

    directory = _seed(root)
    lease = acquire_session_lease(directory)
    try:
        yield directory
    finally:
        lease.release()


def _as_remote(row: dict[str, Any]) -> dict[str, Any]:
    """The row as the VIEWER's relay re-addresses it (``_fan_out_catalog``)."""
    item = dict(row)
    item["locality"] = "remote"
    item["peer"] = {
        "device_id": PEER_ID,
        "name": PEER_NAME,
        "network_id": "n_test",
        "reachable": True,
        "age_s": 0.0,
    }
    return item


def _listing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    rows: list[dict[str, Any]],
    *,
    as_json: bool,
) -> str:
    """``lop network sessions --peer <peer>`` over these rows, as printed."""
    from local_operator.network import cli as net_cli

    answer = {
        "ok": True,
        "sessions": [_as_remote(row) for row in rows],
        "peers": {PEER_ID: {"device_id": PEER_ID, "name": PEER_NAME, "reachable": True}},
    }
    monkeypatch.setattr(
        net_cli, "_relay_answer", lambda op, **fields: json.loads(json.dumps(answer))
    )
    capsys.readouterr()
    rc = net_cli._cmd_sessions(  # noqa: SLF001 - the command's own handler
        argparse.Namespace(peer=PEER_NAME, all_peers=False, json=as_json)
    )
    assert rc == 0
    return capsys.readouterr().out


def test_a_session_whose_runtime_is_booting_lists_as_starting_with_its_age(
    root: Path, held: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _backdate_claim(held, 7.0)
    row = _row(_server(root))
    assert row["boot_state"] == "starting", row
    assert 7.0 <= row["boot_age_s"] < 30.0, row
    # Still cold for every liveness branch: nothing to dial, no gate, not published.
    assert row["state"] == "stored" and row["pid"] == 0 and row["pending"] is None, row

    text = _listing(monkeypatch, capsys, [row], as_json=False)
    line = next(item for item in text.splitlines() if item.startswith(SESSION))
    assert "starting · 7s" in line, line
    assert "not running" not in line, line

    payload = json.loads(_listing(monkeypatch, capsys, [row], as_json=True))
    (listed,) = payload["sessions"]
    assert listed["boot_state"] == "starting", listed
    assert listed["boot_age_s"] >= 7.0, listed
    assert listed["state"] == "stored", listed


def test_a_boot_past_the_cold_boot_budget_reads_as_not_answering(
    root: Path, held: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from local_operator.network import cli as net_cli
    from local_operator.network import types as net_types

    # ONE budget for both ends of the wait: the sender's is the owner's.
    assert net_cli.PILOT_COLD_BOOT_BUDGET_S == net_types.REMOTE_COLD_BOOT_BUDGET_S

    _backdate_claim(held, 45.0)
    row = _row(_server(root))
    assert row["boot_state"] == "not_responding", row
    assert row["boot_age_s"] >= 45.0, row
    line = next(
        item
        for item in _listing(monkeypatch, capsys, [row], as_json=False).splitlines()
        if item.startswith(SESSION)
    )
    assert "not answering · starting for 45s" in line, line

    # The budget is read at call time, and the edge is the budget itself.
    _backdate_claim(held, 3.0)
    monkeypatch.setattr(net_types, "REMOTE_COLD_BOOT_BUDGET_S", 2.0)
    assert _row(_server(root))["boot_state"] == "not_responding"
    monkeypatch.setattr(net_types, "REMOTE_COLD_BOOT_BUDGET_S", 60.0)
    assert _row(_server(root))["boot_state"] == "starting"


def test_a_stopped_session_row_is_byte_unchanged(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """No lease holder, no record: the ``stored`` row it always was, key for key."""
    _seed(root)
    row = _row(_server(root))
    assert "boot_state" not in row and "boot_age_s" not in row, row
    assert row["state"] == "stored" and row["pid"] == 0 and row["age_s"] == 0.0
    line = next(
        item
        for item in _listing(monkeypatch, capsys, [row], as_json=False).splitlines()
        if item.startswith(SESSION)
    )
    assert "not running" in line and "starting" not in line, line


def test_a_dead_lease_holder_is_stopped_not_starting(root: Path) -> None:
    """A claim naming a gone pid is no boot: the probe's liveness test decides."""
    from local_operator.session_lease import LEASE_NAME

    directory = _seed(root)
    (directory / LEASE_NAME).write_text(
        json.dumps({"schema": 1, "session_id": SESSION, "generation": "g", "pid": 2**22 + 7}),
        encoding="utf-8",
    )
    assert "boot_state" not in _row(_server(root))


def test_a_live_runtime_row_is_unchanged_while_it_holds_its_lease(root: Path, held: Path) -> None:
    """A published runtime holds its lease for life; it keeps its registry state."""
    from local_operator.session.runtime import registry
    from local_operator.session.runtime.types import SessionRecord

    registry.publish(
        SessionRecord(
            pid=os.getpid(),
            kind="tui",
            session_id=SESSION,
            conversation_name="live one",
            cwd=str(root),
            model_label="test/model",
            control_port=0,
            control_key="",
        ),
        root,
    )
    row = _row(_server(root))
    assert row["state"] == "live", row
    assert "boot_state" not in row and "boot_age_s" not in row, row
    assert row["pid"] == os.getpid()


def test_older_and_liveness_readers_still_read_a_booting_row_as_cold(
    root: Path, held: Path
) -> None:
    """Mixed-version contract: the token stays ``stored``, so no reader misfiles it.

    An older viewer ignores the additive keys; the readers that branch on
    liveness (needs claim, the sidebar's live-state, the dedupe preference)
    see exactly the stored row they saw before.
    """
    from local_operator.network.projection import PeerRow
    from local_operator.network.types import row_needs_claim
    from local_operator.session.peer_rows import _live_state

    row = _row(_server(root))
    assert row["boot_state"] == "starting"
    assert row_needs_claim(state=row["state"], pending="approval") is None
    claim = PeerRow.from_json(_as_remote(row), device_id=PEER_ID)
    assert claim.state == "stored" and claim.pid == 0
    assert _live_state(claim) == "", "a booting row must not be filed as running"


def test_the_sessions_tool_names_a_booting_remote_session(root: Path, held: Path) -> None:
    """``sessions info``/``list`` on a peer said "not running" for a boot; now it says so."""
    from local_operator.tools.builtin import _sessions_peer_status

    _backdate_claim(held, 12.0)
    assert _sessions_peer_status(_as_remote(_row(_server(root)))) == "starting · 12s"
    _backdate_claim(held, 90.0)
    assert (
        _sessions_peer_status(_as_remote(_row(_server(root)))) == "not answering · starting for 1m"
    )


def test_row_words_ignore_a_boot_claim_on_a_live_row_or_an_unreadable_age() -> None:
    """The claim is honoured only beside ``stored``, and only with a usable age."""
    from local_operator.resume import english_status_render, session_row_state_words

    def words(**row: Any) -> str:
        return session_row_state_words(row, render=english_status_render)

    assert words(state="stored") == "not running"
    assert words(state="live", boot_state="starting", boot_age_s=3) == "live"
    assert words(state="stored", boot_state="starting", boot_age_s=True) == "not running"
    assert words(state="stored", boot_state="starting", boot_age_s=float("nan")) == "not running"
    assert words(state="stored", boot_state="starting", boot_age_s=-1) == "not running"
    assert words(state="stored", boot_state="bogus", boot_age_s=3) == "not running"
    assert words(state="stored", boot_state="starting", boot_age_s=3.9) == "starting · 3s"


def test_lease_claimed_at_reads_the_claim_only_while_a_verified_holder_lives(root: Path) -> None:
    """The relay's age source (PR #4588): the claim's mtime for a live holder, else None.

    Lives in launch so the relay never imports ``session_lease`` beyond
    ``lease_holder`` (``test_the_relay_never_becomes_a_session_owner``).
    """
    from local_operator.session.runtime.launch import lease_claimed_at
    from local_operator.session_lease import LEASE_NAME, acquire_session_lease

    directory = _seed(root)
    assert lease_claimed_at(root, SESSION) is None  # no claim at all

    lease = acquire_session_lease(directory)
    try:
        _backdate_claim(directory, 42.0)
        claimed = lease_claimed_at(root, SESSION)
        assert claimed == (directory / LEASE_NAME).stat().st_mtime
        assert 41.0 < time.time() - claimed < 60.0
    finally:
        lease.release()
    assert lease_claimed_at(root, SESSION) is None  # released

    (directory / LEASE_NAME).write_text(
        json.dumps({"schema": 1, "session_id": SESSION, "generation": "g", "pid": 2**22 + 7}),
        encoding="utf-8",
    )
    assert lease_claimed_at(root, SESSION) is None  # a dead pid is no holder
