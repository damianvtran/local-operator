"""The desktop code-memory routes, driven through the REAL app and a fake owner.

The panel's whole bug was a route that answered 404 for every session because it
resolved the id through the wrong registry (agent-directory UUIDs), so the state
a client can actually receive is the contract under test here — every state and
every refusal code, over the assembled router with its bearer gate, rather than a
hand-called handler.

The owner is a fake BRIDGE rather than a fake ``variables_op``: the routes reach
the session through ``host(request).session(...)`` → ``bridge.remote``, and the
two claims that matter are properties of that seam:

* a COLD session is read without engaging anything — ``bind_runtime`` is
  recorded, and it must stay unrecorded, because a panel on a chat that has not
  started is not a reason to start a process;
* a mutation on a cold session refuses (``runtime_cold``) rather than spawning,
  for the same reason plus one more: a variable written into a namespace no cell
  has ever run in is a value the panel would show and no cell could use.

The fake records ``bind_runtime`` explicitly so the negative assertion has
something to fail against; asserting ``is_cold`` was reported would pass even if
the route had bound the runtime first.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.server.routes import desktop_lifecycle

TOKEN = "desktop-variables-test-token"
#: A real canonical session id shape (12 hex), which is exactly the id the
#: legacy agent route could never resolve.
SESSION = "8fd6c6a40934"


class FakeRemote:
    """A viewer facade with the two facts the routes read and nothing else."""

    def __init__(self, *, cold: bool = False) -> None:
        self._cold = cold
        self.binds = 0
        self.calls: list[tuple[Any, ...]] = []
        self.answers: dict[str, dict[str, Any]] = {}

    @property
    def is_cold(self) -> bool:
        return self._cold

    async def bind_runtime(self) -> None:
        self.binds += 1

    async def variables_op(self, action: str, key: str = "", value: str = "", value_type: str = ""):
        self.calls.append((action, key, value, value_type))
        return self.answers.get(action, {"ok": True, "state": "ok"})


class FakePool:
    """``DesktopSessions``-shaped: the route's only door to a session."""

    def __init__(self, remote: FakeRemote) -> None:
        self.remote = remote

    @contextlib.asynccontextmanager
    async def session(self, session_id: str):
        if session_id != SESSION:
            raise KeyError("Unknown session")
        yield SimpleNamespace(remote=self.remote)


@pytest_asyncio.fixture
async def desktop(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    # No test here writes, but a redirected HOME/config dir is the house rule for
    # anything that builds the real app: the isolation must not depend on this
    # suite continuing to have no side effects.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    app = FastAPI()
    app.include_router(desktop_lifecycle.router)
    remote = FakeRemote()
    app.state.desktop_sessions = FakePool(remote)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        yield client, remote


def _variables_url(session_id: str = SESSION, key: str | None = None) -> str:
    base = f"/v1/desktop/sessions/{session_id}/variables"
    return f"{base}/{key}" if key is not None else base


# ---------------------------------------------------------------------------
# reads
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_populated_read_reports_what_the_kernel_holds(desktop) -> None:
    client, remote = desktop
    remote.answers["list"] = {
        "ok": True,
        "state": "observed",
        "kernel": "resident",
        "variables": [
            {
                "key": "outstanding",
                "type": "int",
                "value": "3",
                "editable": True,
                "truncated": False,
            }
        ],
        "truncated": False,
    }

    response = await client.get(_variables_url())

    assert response.status_code == 200
    data = response.json()["result"]["data"]
    assert data == {
        "state": "observed",
        "runtime": "running",
        "kernel": "resident",
        "variables": [remote.answers["list"]["variables"][0]],
        "truncated": False,
    }
    assert remote.calls == [("list", "", "", "")]


@pytest.mark.asyncio
async def test_a_cold_session_reads_absent_without_engaging_a_runtime(desktop) -> None:
    """The keystone: reading a panel must not start a process."""
    client, remote = desktop
    remote._cold = True

    response = await client.get(_variables_url())

    assert response.status_code == 200
    assert response.json()["result"]["data"] == {
        "state": "observed",
        "runtime": "absent",
        "kernel": "absent",
        "variables": [],
        "truncated": False,
    }
    assert remote.binds == 0, "a read engaged the runtime"
    assert remote.calls == [], "a cold session has no kernel to ask"


@pytest.mark.asyncio
async def test_an_empty_namespace_with_a_live_kernel_is_observed_and_empty(desktop) -> None:
    client, remote = desktop
    remote.answers["list"] = {
        "ok": True,
        "state": "observed",
        "kernel": "resident",
        "variables": [],
        "truncated": False,
    }

    data = (await client.get(_variables_url())).json()["result"]["data"]

    assert data["state"] == "observed" and data["kernel"] == "resident"
    assert data["variables"] == []


@pytest.mark.asyncio
async def test_a_kernel_released_after_idling_is_absent_with_a_running_runtime(desktop) -> None:
    """Its own sentence: "the interpreter was released", not "this chat is new"."""
    client, remote = desktop
    remote.answers["list"] = {
        "ok": True,
        "state": "observed",
        "kernel": "absent",
        "variables": [],
        "truncated": False,
    }

    data = (await client.get(_variables_url())).json()["result"]["data"]

    assert data["runtime"] == "running" and data["kernel"] == "absent"
    assert data["variables"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["busy", "unsupported"])
async def test_a_state_that_is_not_a_reading_carries_no_variables_key(desktop, state) -> None:
    """The model's whole point: nothing can paint "empty" over "unknown"."""
    client, remote = desktop
    remote.answers["list"] = {"ok": True, "state": state} if state == "busy" else {"state": state}

    response = await client.get(_variables_url())

    assert response.status_code == 200
    data = response.json()["result"]["data"]
    assert data == {"state": state}
    assert "variables" not in data


@pytest.mark.asyncio
async def test_a_lost_kernel_race_reads_as_absent_rather_than_empty_memory(desktop) -> None:
    """The owner can lose its kernel between the cold check and the verb."""
    client, remote = desktop
    remote.answers["list"] = {
        "ok": False,
        "code": "no_kernel",
        "message": "This chat's Python interpreter is not running",
    }

    data = (await client.get(_variables_url())).json()["result"]["data"]

    assert data["runtime"] == "running" and data["kernel"] == "absent"
    assert data["variables"] == []


@pytest.mark.asyncio
async def test_an_unrecognised_read_refusal_is_terminal_not_a_permanent_reading(desktop) -> None:
    """Each cause gets its own state: ``busy`` is a retry, not a catch-all.

    Folding every refusal into ``busy`` would leave the panel on its "reading…"
    affordance forever, because nothing later in the interaction completes it.
    A cause this build cannot report as a reading therefore takes the terminal
    state, where the panel says so and stops asking.
    """
    client, remote = desktop
    remote.answers["list"] = {"ok": False, "code": "invalid_value", "message": "not a reading"}

    data = (await client.get(_variables_url())).json()["result"]["data"]

    assert data == {"state": "unsupported"}


@pytest.mark.asyncio
async def test_a_namespace_that_moved_under_the_read_is_retryable_not_empty(desktop) -> None:
    client, remote = desktop
    remote.answers["list"] = {
        "ok": False,
        "code": "changed_under_read",
        "message": "The namespace changed while it was being read; try again.",
    }

    data = (await client.get(_variables_url())).json()["result"]["data"]

    assert data == {"state": "busy"}


# ---------------------------------------------------------------------------
# writes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_and_update_and_delete_reach_the_owner_with_their_verb(desktop) -> None:
    client, remote = desktop
    remote.answers["set"] = {
        "ok": True,
        "state": "ok",
        "variable": {
            "key": "outstanding",
            "type": "int",
            "value": "3",
            "editable": True,
            "truncated": False,
        },
    }
    remote.answers["update"] = remote.answers["set"]

    created = await client.post(
        _variables_url(), json={"key": "outstanding", "value": "3", "type": "int"}
    )
    updated = await client.patch(
        _variables_url(key="outstanding"), json={"value": "4", "type": "int"}
    )
    deleted = await client.delete(_variables_url(key="outstanding"))

    assert created.status_code == 200
    assert created.json()["result"]["data"]["variable"]["key"] == "outstanding"
    assert updated.status_code == 200
    assert deleted.status_code == 200
    assert deleted.json()["result"]["data"] == {"state": "ok"}
    assert remote.calls == [
        ("set", "outstanding", "3", "int"),
        ("update", "outstanding", "4", "int"),
        ("delete", "outstanding", "", ""),
    ]


@pytest.mark.asyncio
async def test_a_mutation_on_a_cold_session_is_refused_without_engaging(desktop) -> None:
    client, remote = desktop
    remote._cold = True

    response = await client.post(_variables_url(), json={"key": "k", "value": "1", "type": "int"})

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "runtime_cold"
    assert remote.binds == 0, "a mutation engaged the runtime"
    assert remote.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("code", "status"),
    [
        ("no_kernel", 409),
        ("kernel_busy", 409),
        ("reserved_name", 409),
        ("already_exists", 409),
        ("invalid_value", 409),
        ("too_large", 409),
        ("not_found", 404),
    ],
)
async def test_a_write_refusal_reaches_the_renderer_with_its_code(desktop, code, status) -> None:
    """``desktopResult`` lifts ``detail.code``; a bare sentence would be unactionsable."""
    client, remote = desktop
    remote.answers["set"] = {"ok": False, "code": code, "message": "a vetted sentence"}
    remote.answers["update"] = remote.answers["set"]

    response = await client.post(_variables_url(), json={"key": "k", "value": "1", "type": "int"})

    assert response.status_code == status
    assert response.json()["detail"] == {"code": code, "message": "a vetted sentence"}


@pytest.mark.asyncio
async def test_an_unknown_type_is_refused_by_the_body_model(desktop) -> None:
    """``Literal`` from the shared table: 422, and the owner is never asked."""
    client, remote = desktop

    response = await client.post(_variables_url(), json={"key": "k", "value": "1", "type": "tuple"})

    assert response.status_code == 422
    assert remote.calls == []


@pytest.mark.asyncio
async def test_the_owner_decides_the_name_and_size_policy_not_the_route(desktop) -> None:
    """The refusal is decided where the namespace is, and the route relays it.

    Deliberately NOT re-implemented here: the desktop backend is the VIEWER on
    this seam, and a second copy of the key denylist would be the drift the one
    shared table exists to prevent (`session/variable_ops.py`). The KEY still
    crosses verbatim so an owner on a newer build can decide for itself.
    """
    client, remote = desktop
    remote.answers["set"] = {
        "ok": False,
        "code": "reserved_name",
        "message": "That name belongs to the interpreter and cannot be used.",
    }

    response = await client.post(
        _variables_url(), json={"key": "__builtins__", "value": "1", "type": "int"}
    )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "reserved_name"
    assert remote.calls == [("set", "__builtins__", "1", "int")]


# ---------------------------------------------------------------------------
# the gate and the address
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_unknown_or_foreign_session_is_a_404(desktop) -> None:
    client, _remote = desktop

    assert (await client.get(_variables_url("0123456789ab"))).status_code == 404
    # The legacy shape (a UUID 'agent id') is not a session and must not resolve.
    assert (
        await client.get(_variables_url("6f3a1c2e-0000-4000-8000-000000000000"))
    ).status_code == 404


@pytest.mark.asyncio
async def test_the_wrong_bearer_is_refused(desktop) -> None:
    client, _remote = desktop

    response = await client.get(
        _variables_url(), headers={"Authorization": "Bearer not-the-desktop-token"}
    )

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_every_route_requires_the_desktop_bearer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Read, create, update and delete are all behind the same gate."""
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    app = FastAPI()
    app.include_router(desktop_lifecycle.router)
    app.state.desktop_sessions = FakePool(FakeRemote())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost") as client:
        assert (await client.get(_variables_url())).status_code == 401
        assert (
            await client.post(_variables_url(), json={"key": "k", "value": "1", "type": "int"})
        ).status_code == 401
        assert (
            await client.patch(_variables_url(key="k"), json={"value": "1", "type": "int"})
        ).status_code == 401
        assert (await client.delete(_variables_url(key="k"))).status_code == 401
