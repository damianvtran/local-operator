"""H5 remediation — admission coalescing and the pre-fetch fence (S-2, H5-1,
H5-4, S-9) and the boundary scheduler (S-2/H5-1), on fakes only.

Matrix rules exercised (rows 1/2/3/9): N boundaries share ONE fetch; a settled
runtime admits a fresh boundary fetch; revocation cancels queued work and
discards an awaited stale result; a drifted/disabled runtime performs ZERO
fetches; reads stay distinguishable from mutations under paused/read-only.

The fence is substituted at ``local_operator.team_bridge.authority``'s module
attribute because both call sites import it at CALL time — the real fence is
covered by ``test_h5_remediation_fence.py``.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pytest

from local_operator.team_bridge import authority as fence_mod
from local_operator.team_bridge.host import (
    admission_refusal,
    admit_bridge_runtime,
    admit_soon,
)
from local_operator.team_bridge.tool import BridgeRuntime

pytestmark = pytest.mark.asyncio


class _ScriptedFence:
    """One refusal per call, from a script; the last value repeats."""

    def __init__(self, *values: object) -> None:
        self._values = list(values) or [None]
        self.calls = 0

    def __call__(self, runtime: object) -> object:
        value = self._values[min(self.calls, len(self._values) - 1)]
        self.calls += 1
        return value


_REFUSAL = {"code": "owner_changed", "error": "drifted"}


class _Client:
    def __init__(self, payload: object | None = None, gate: asyncio.Event | None = None) -> None:
        self.calls = 0
        self._payload = payload
        self._gate = gate

    async def status(self) -> object:
        self.calls += 1
        if self._gate is not None:
            await self._gate.wait()
        return self._payload


def _payload(
    tenant_id: str = "ten_1", endpoint_id: str = "ep_1", state: str = "active"
) -> dict[str, Any]:
    return {
        "api_version": "1",
        "tenants": [
            {
                "tenant_id": tenant_id,
                "eligible": True,
                "endpoint": {"id": endpoint_id, "state": state},
            }
        ],
    }


def _runtime(client: object) -> BridgeRuntime:
    return cast(
        BridgeRuntime,
        SimpleNamespace(
            owner=SimpleNamespace(tenant_id="ten_1", endpoint_id="ep_1"),
            config_dir="/tmp/none",
            tenant_id="ten_1",
            session_id="sess-1",
            client=client,
            admission="pending",
            admission_note="",
            _admission_task=None,
        ),
    )


@pytest.fixture(autouse=True)
def _fence_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fence_mod, "authority_fence", lambda runtime: None)


async def test_two_boundaries_share_one_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    gate = asyncio.Event()
    client = _Client(_payload(), gate=gate)
    runtime = _runtime(client)
    first = asyncio.create_task(admit_bridge_runtime(runtime))
    second = asyncio.create_task(admit_bridge_runtime(runtime))
    await asyncio.sleep(0.05)  # let both reach the join
    gate.set()
    outcomes = await asyncio.gather(first, second)
    assert outcomes == ["admitted", "admitted"]
    assert client.calls == 1  # N callers, ONE fetch — the H5-1 discriminator


async def test_settled_boundary_may_fetch_again() -> None:
    client = _Client(_payload())
    runtime = _runtime(client)
    assert await admit_bridge_runtime(runtime) == "admitted"
    assert client.calls == 1
    assert await admit_bridge_runtime(runtime) == "admitted"
    assert client.calls == 2  # the slot cleared after settlement


async def test_drift_refusal_performs_zero_fetches_and_cancels_inflight() -> None:
    starting = asyncio.get_running_loop().create_task(asyncio.sleep(60))
    client = _Client(_payload())
    runtime = _runtime(client)
    fence = _ScriptedFence(_REFUSAL)
    original = fence_mod.authority_fence
    fence_mod.authority_fence = fence
    try:
        runtime._admission_task = starting
        state = await admit_bridge_runtime(runtime)
    finally:
        fence_mod.authority_fence = original
    assert state == "pending"
    assert client.calls == 0
    # ``cancel()`` only REQUESTS cancellation; the loop has to run once before
    # the task reports itself cancelled. One yield, no weakening of the gate.
    await asyncio.sleep(0)
    assert starting.cancelled()
    assert runtime._admission_task is None


async def test_revocation_after_await_discards_the_result() -> None:
    client = _Client(_payload())
    runtime = _runtime(client)
    fence = _ScriptedFence(None, None, _REFUSAL)  # entry / pre-GET / post-await
    original = fence_mod.authority_fence
    fence_mod.authority_fence = fence
    try:
        state = await admit_bridge_runtime(runtime)
    finally:
        fence_mod.authority_fence = original
    assert client.calls == 1
    assert state == "pending"  # the awaited answer was DISCARDED, never promoted
    assert runtime.admission == "pending"


async def test_admit_soon_is_bounded_and_fenced() -> None:
    client = _Client(_payload())
    runtime = _runtime(client)
    admit_soon(runtime)
    admit_soon(runtime)  # second boundary JOINS, never a second fetch
    await asyncio.sleep(0.05)
    assert client.calls == 1
    # A drifted runtime schedules nothing at all.
    blocked_client = _Client(_payload())
    blocked = _runtime(blocked_client)
    original = fence_mod.authority_fence
    fence_mod.authority_fence = _ScriptedFence(_REFUSAL)
    try:
        admit_soon(blocked)
    finally:
        fence_mod.authority_fence = original
    assert blocked._admission_task is None
    # The claim is made on the client FAKE, which is where ``calls`` lives;
    # ``runtime.client`` is the cast seam (see test_h5_remediation_doors).
    assert blocked_client.calls == 0


async def test_reads_pass_paused_and_read_only_but_mutations_refuse() -> None:
    runtime = _runtime(_Client(_payload()))
    runtime.admission = "disabled"
    runtime.admission_note = "read_only"
    assert admission_refusal(runtime, is_read=True) is None
    refusal = admission_refusal(runtime, is_read=False)
    assert refusal is not None
    assert refusal["code"] == "bridge_read_only"
    runtime.admission = "paused"
    runtime.admission_note = ""
    assert admission_refusal(runtime, is_read=True) is None
    refusal = admission_refusal(runtime, is_read=False)
    assert refusal is not None
    assert refusal["code"] == "bridge_paused"
    for state in ("signin", "pending"):
        runtime.admission = state
        assert admission_refusal(runtime, is_read=True) is not None
        assert admission_refusal(runtime, is_read=False) is not None
    runtime.admission = "disabled"
    runtime.admission_note = "opted_out"
    assert admission_refusal(runtime, is_read=True) is not None
