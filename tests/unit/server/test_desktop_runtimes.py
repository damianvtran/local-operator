"""``GET /v1/desktop/runtimes``: the roster, over HTTP and inside its budget.

The route is the thin half — auth, the query surface, the response model — and
the properties worth pinning are the ones an operator would notice if they broke:

* it is behind the desktop plane like every other ``/v1/desktop/`` route;
* a runtime with NO discovery record still appears, named, with the port a client
  would dial (this is the measured gap: 34 of 57 runtimes were in that state from
  this store's point of view);
* a dead pid reads ``gone`` and is never dialled;
* ``probe=false`` returns the inventory without opening a socket;
* the composition runs OFF the event loop, so a wedged runtime cannot stall every
  other request this daemon serves.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.server.routes import desktop_runtimes
from local_operator.session.runtime import reclaim, roster
from local_operator.session.runtime.reclaim import RuntimeProcess, SocketEvidence
from local_operator.session.runtime.types import SessionRecord

TOKEN = "desktop-runtimes-token"
MINE = os.getpid()
GONE_PID = 999_999


def _record(pid: int, *, session_id: str, port: int, heartbeat_at: float | None = None):
    return SessionRecord.from_json(
        {
            "pid": pid,
            "kind": "daemon",
            "session_id": session_id,
            "conversation_name": "synthetic",
            "cwd": "/tmp/synthetic",
            "model_label": "synthetic-model",
            "control_port": port,
            "control_key": "k",
            "heartbeat_at": time.time() if heartbeat_at is None else heartbeat_at,
            "version": "0.56.11",
        }
    )


def _write(root: Path, record: SessionRecord) -> None:
    directory = root / "run" / "mobile"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{record.pid}.json").write_text(json.dumps(record.to_json()))


def _census(*pids: int) -> list[RuntimeProcess]:
    return [
        RuntimeProcess(
            pid=pid,
            parent_pid=1,
            age_s=3600.0,
            cpu_s=0.5,
            command=f"/usr/bin/python3 -P -m {reclaim.RUNTIME_MODULE}",
        )
        for pid in pids
    ]


@pytest_asyncio.fixture
async def desktop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    # The ancestor walk is covered by its own unit test; here it would refuse the
    # test process itself (pid MINE is an ancestor of nothing, but pid 1 is), and
    # every row under test is a synthetic one.
    monkeypatch.setattr(reclaim, "ancestor_pids", lambda *args, **kwargs: frozenset())
    # LIVENESS IS DECLARED, NOT GUESSED. ``MINE + 1`` is a synthetic runtime's pid,
    # and whether the kernel happens to have handed it to some other process is not a
    # property of this route — it made the "reachable" case pass or fail with the
    # machine's pid churn. The rule under test is what the roster does with a live and
    # a dead row, so the answers are pinned here.
    real_alive = roster.registry.pid_alive
    monkeypatch.setattr(
        roster.registry,
        "pid_alive",
        lambda pid, **kwargs: (
            False
            if pid == GONE_PID
            else True if pid in {MINE, MINE + 1} else real_alive(pid, **kwargs)
        ),
    )
    app = FastAPI()
    app.include_router(desktop_runtimes.router)
    app.state.config_manager = ConfigManager(tmp_path)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost") as bare:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://localhost",
            headers={"Authorization": f"Bearer {TOKEN}"},
        ) as client:
            yield bare, client, tmp_path


@pytest.mark.asyncio
async def test_requires_the_desktop_plane(desktop) -> None:
    bare, _client, _root = desktop
    response = await bare.get("/v1/desktop/runtimes")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_a_recordless_runtime_is_listed_with_its_port_and_a_dead_one_reads_gone(
    desktop, monkeypatch: pytest.MonkeyPatch
) -> None:
    _bare, client, root = desktop
    # Two synthetic runtimes: one recorded (so it has a reader-facing existence) and
    # one the census alone can see — the shape the measured fleet was full of. Plus a
    # record whose process is gone.
    _write(root, _record(MINE, session_id="recorded", port=5000))
    _write(root, _record(GONE_PID, session_id="dead", port=5001))
    monkeypatch.setattr(roster, "runtime_processes", lambda **kw: _census(MINE, MINE + 1))
    monkeypatch.setattr(
        roster,
        "socket_evidence",
        lambda **kw: SocketEvidence(ports={MINE + 1: 6001}, available=True),
    )
    dialled: list[int] = []

    def connect(port: int, timeout: float) -> bool:
        dialled.append(port)
        return True

    monkeypatch.setattr(roster, "_connect_ok", connect)

    response = await client.get("/v1/desktop/runtimes")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == 200
    rows = {row["pid"]: row for row in payload["result"]["runtimes"]}
    assert set(rows) == {MINE, MINE + 1, GONE_PID}
    # The recorded runtime: named from its record, heartbeating, reachable.
    assert rows[MINE]["session_id"] == "recorded"
    assert rows[MINE]["has_record"] is True
    assert rows[MINE]["reachability"] == "live"
    assert rows[MINE]["reclaimable"] is False
    assert rows[MINE]["reclaim_refusal"] == reclaim.REFUSAL_RECORD_PRESENT
    # The record-less runtime: no session id, but a PORT from the socket table, so a
    # client can dial it — which is the whole point of composing from four sources.
    assert rows[MINE + 1]["has_record"] is False
    assert rows[MINE + 1]["port"] == 6001
    assert rows[MINE + 1]["reachability"] == "live"
    # The corpse: reported, and never dialled.
    assert rows[GONE_PID]["reachability"] == "gone"
    assert rows[GONE_PID]["state"] == "stale"
    assert 5001 not in dialled
    assert payload["result"]["socket_table"] is True
    assert "process-table" in payload["result"]["sources"]


@pytest.mark.asyncio
async def test_probe_false_returns_the_inventory_without_opening_a_socket(
    desktop, monkeypatch: pytest.MonkeyPatch
) -> None:
    _bare, client, root = desktop
    _write(root, _record(MINE, session_id="recorded", port=5000))
    monkeypatch.setattr(roster, "runtime_processes", lambda **kw: _census(MINE))
    monkeypatch.setattr(roster, "socket_evidence", lambda **kw: SocketEvidence())

    def connect(port: int, timeout: float) -> bool:  # pragma: no cover - must not run
        raise AssertionError("probe=false must not dial")

    monkeypatch.setattr(roster, "_connect_ok", connect)
    response = await client.get("/v1/desktop/runtimes?probe=false")
    assert response.status_code == 200
    row = response.json()["result"]["runtimes"][0]
    assert row["port"] == 5000
    assert row["reachability"] == "unknown"


@pytest.mark.asyncio
async def test_composition_runs_off_the_event_loop(
    desktop, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A wedged runtime is what makes a connect block, and this daemon serves every
    # other desktop request on the same loop: the composition must not be on it.
    _bare, client, _root = desktop
    threads: list[int] = []
    import threading

    def slow_census(**kwargs):
        threads.append(threading.get_ident())
        return _census(MINE)

    monkeypatch.setattr(roster, "runtime_processes", slow_census)
    monkeypatch.setattr(roster, "socket_evidence", lambda **kw: SocketEvidence())
    response = await client.get("/v1/desktop/runtimes")
    assert response.status_code == 200
    assert threads and threads[0] != threading.get_ident()


@pytest.mark.asyncio
async def test_the_budget_is_bounded_by_the_route(desktop) -> None:
    _bare, client, _root = desktop
    # The ceiling is part of the contract: a caller cannot ask for an unbounded
    # answer, because the endpoint's whole value is that it always returns.
    response = await client.get("/v1/desktop/runtimes?budget_s=600")
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_a_record_whose_heartbeat_went_quiet_reads_wedged(
    desktop, monkeypatch: pytest.MonkeyPatch
) -> None:
    _bare, client, root = desktop
    _write(root, _record(MINE, session_id="stuck", port=5000, heartbeat_at=time.time() - 10_000))
    monkeypatch.setattr(roster, "runtime_processes", lambda **kw: _census(MINE))
    monkeypatch.setattr(roster, "socket_evidence", lambda **kw: SocketEvidence())
    monkeypatch.setattr(roster, "_connect_ok", lambda port, timeout: False)
    response = await client.get("/v1/desktop/runtimes")
    row = response.json()["result"]["runtimes"][0]
    assert row["state"] == "wedged"
    assert row["reachability"] == "unreachable"
    assert row["heartbeat_age_s"] > 9_000


@pytest.mark.asyncio
async def test_a_dead_pid_does_not_make_the_route_wait(
    desktop, monkeypatch: pytest.MonkeyPatch
) -> None:
    # THE HANG THIS ROUTE MUST NOT HAVE. A corpse in the store plus a socket table
    # that names its port: the row is `gone` and the response arrives promptly,
    # because liveness is asked before a connect is ever attempted.
    _bare, client, root = desktop
    _write(root, _record(GONE_PID, session_id="corpse", port=5000))
    monkeypatch.setattr(roster, "runtime_processes", lambda **kw: [])
    monkeypatch.setattr(
        roster,
        "socket_evidence",
        lambda **kw: SocketEvidence(ports={GONE_PID: 5000}, available=True),
    )
    monkeypatch.setattr(
        roster, "_connect_ok", lambda port, timeout: pytest.fail("dialled a corpse")
    )
    started = asyncio.get_event_loop().time()
    response = await client.get("/v1/desktop/runtimes")
    assert response.status_code == 200
    assert response.json()["result"]["runtimes"][0]["reachability"] == "gone"
    assert asyncio.get_event_loop().time() - started < 5.0


@pytest.mark.asyncio
async def test_the_response_does_not_wait_out_the_probe_queue(
    desktop, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A roster bigger than the probe pool must still return inside its budget.

    ``ThreadPoolExecutor``'s context manager joins the QUEUE, so forty rows against
    a 0.1 s budget cost ``ceil(40 / 8) x 0.25 s`` — the budget was a hope, and QA
    measured 1.69 s at this endpoint against a 0.1 s request (review round 1, R1-4).
    The rows here are real records with real ports and a probe that never answers
    inside the budget, which is the shape that used to queue.
    """
    _bare, client, root = desktop
    pids = list(range(MINE + 1, MINE + 41))
    for pid in pids:
        _write(root, _record(pid, session_id=f"s{pid}", port=19000 + pid))
    # Liveness is what gates the dial, and these synthetic pids are not processes:
    # without this the rows would be `gone` and no probe would be queued at all.
    # The census is emptied too, so the roster is exactly these forty rows and not
    # whatever runtimes happen to be live on the machine running the suite.
    monkeypatch.setattr(roster, "runtime_processes", lambda **kw: [])
    monkeypatch.setattr(reclaim.registry, "pid_alive", lambda pid, **kwargs: True)
    monkeypatch.setattr(roster, "socket_evidence", lambda **kw: SocketEvidence())
    # A listener whose accept queue never drains: the connect blocks for the whole
    # probe timeout rather than refusing.
    monkeypatch.setattr(roster, "_connect_ok", lambda port, timeout: time.sleep(timeout) or False)

    started = time.monotonic()
    response = await client.get("/v1/desktop/runtimes?budget_s=0.1")
    elapsed = time.monotonic() - started
    assert response.status_code == 200
    assert elapsed < 1.0, elapsed
    rows = response.json()["result"]["runtimes"]
    assert len(rows) == len(pids)
    # Every row is ANSWERED, and the ones the budget cut short say so rather than
    # claiming the runtime did not answer.
    assert {row["reachability"] for row in rows} == {"unknown"}
