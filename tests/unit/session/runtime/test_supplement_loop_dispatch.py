"""The real threaded control socket must marshal every supplement read/mutation.

Only the provider completion is held; runner persistence, leases, runtime RPC and
lazy reads are real. An in-process handle call cannot detect this wrong-loop bug.
"""

import asyncio
import threading
from pathlib import Path
from typing import Any

import pytest

from local_operator.session.runtime.server import RuntimeServer
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.session_lease import acquire_session_lease
from local_operator.supplements import generator, policy
from tests.unit.session.runtime.test_daemon_serving_plane import (
    _let_the_server_settle,
    _request,
)
from tests.unit.session.test_session import wait_for
from tests.unit.supplements.test_runner_generator import (
    DATASETS,
    MODEL,
    _decision,
    _make_session,
    _rows,
)


@pytest.mark.asyncio
async def test_all_supplement_ops_use_owner_loop_over_real_threaded_rpc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    directory = tmp_path / "sessions" / "aabbcc001122"
    lease = acquire_session_lease(directory)
    session = _make_session(directory, tmp_path)
    handle = ServingSessionHandle(
        session, asyncio.get_running_loop(), install_gates=False, cwd=str(tmp_path)
    )
    owner_thread = threading.get_ident()
    owner_loop = asyncio.get_running_loop()
    requests: list[Any] = []
    cancelled: list[int] = []

    async def complete(request: Any) -> Any:
        assert threading.get_ident() == owner_thread
        assert asyncio.get_running_loop() is owner_loop
        requests.append(request)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append(len(requests))
            raise

    handle._supplements._complete = complete
    handle._supplements._resolve_model = lambda settings: generator.DesignModel(
        MODEL, "session", "test/m"
    )
    runtime = RuntimeServer(handle, kind="daemon")
    runtime.start()
    assert await runtime.wait_until_published()
    anchor = "answer"
    settings = policy.SupplementSettings.from_values(
        {"supplements": {"graphics": True, "timeoutS": 30}}
    )
    handle._supplements._render(
        anchor, _decision(), DATASETS, "Compare latency", "Prepared comparison", settings
    )
    sequence = 0

    async def rpc(op: str, **fields: Any) -> dict[str, Any]:
        nonlocal sequence
        sequence += 1
        reply = await asyncio.to_thread(
            _request,
            runtime.record,
            {"op": op, "req": sequence, **fields},
            auth={"key": runtime.record.control_key, "supplements": True},
        )
        assert reply.get("op") == "ack", reply
        await _let_the_server_settle()
        return reply

    try:
        assert runtime._loop is not owner_loop
        assert runtime._thread is not threading.current_thread()
        await wait_for(lambda: bool(requests) and bool(_rows(session.transcript)))
        job = _rows(session.transcript)[-1]["job"]
        read = await rpc("supplements_for", anchors=[anchor])
        assert read["supplements"][anchor]["job"] == job
        await rpc("supplement_cancel", anchor=anchor, job=job)
        await wait_for(lambda: _rows(session.transcript)[-1]["state"] == "cancelled")
        old_version = _rows(session.transcript)[-1]["version"]
        await rpc("supplement_restart", anchor=anchor, job=job)
        await wait_for(lambda: len(requests) >= 2)
        assert _rows(session.transcript)[-1]["version"] == old_version + 1
        await rpc("supplement_steer", anchor=anchor, job=job, text="Use a table")
        await wait_for(lambda: len(requests) >= 3)
        # Steer journals the cut before allocating its replacement generation.
        assert _rows(session.transcript)[-1]["version"] == old_version + 3
        assert "Use a table" in str(requests[-1].messages)
        await rpc("supplement_dismiss", anchor=anchor)
        final = await rpc("supplements_for", anchors=[anchor])
        assert final["supplements"][anchor]["state"] == "skipped"
        assert final["supplements"][anchor]["dismissed"] is True
        assert len(cancelled) == 3
    finally:
        runtime.close()
        await handle.dispose()
        await session.dispose()
        lease.release()
