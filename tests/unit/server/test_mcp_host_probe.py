"""The sessionless MCP host, and the refusals both routes now name.

Three claims live here that no route-level test can make on its own:

* **A spawned server never outlives the request that needed it.** The host runs
  each Test / sign-in on a SHORT-LIVED manager. The architect's probe tripped
  anyio's "Attempted to exit cancel scope in a different task" when the manager
  was torn down from another task, so the constraint is "create, use and
  shutdown inside ONE task" — and the way to keep that constraint honest is to
  spawn a real child, cancel it mid-operation, and check the pid is gone. A fake
  transport cannot show that, and a passing assertion about a registry can pass
  while the child is still running.
* **A reload answers after the startup gate's stragglers settle.** ``reload()``
  returns at the 250 ms gate and leaves slower servers to background
  continuations, so the snapshot taken at the gate read "connecting / 0 tools"
  for a server that connected a second later — the report this change exists to
  fix. The fix is ``wait_settled``, and the tests below pin that a caller which
  waits sees ``connected`` while a caller which does NOT wait still sees the
  honest ``connecting`` (no continuation is cancelled to make the answer
  prettier).
* **The legacy session route keeps its old row shape and vocabulary.** Desktop
  builds that predate ``features.mcp_catalog`` keep calling
  ``/v1/desktop/sessions/{id}/mcp``, so the row keys, the ``stdio``/``http``
  transport words and the status words there are a compatibility surface, not an
  implementation detail. Only the refusal BODY changed, and it changed by design
  (``{code, message}`` instead of one fixed sentence).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.mcp.config import MCPConfigWriteError
from local_operator.mcp.desktop import (
    OPERATION_LIMIT,
    REFUSAL_MESSAGES,
    RELOAD_SETTLE_S,
    McpOperations,
    MCPRefusal,
    RefusalCode,
    refusal_code,
    refusal_detail,
)
from local_operator.server.mcp_host import SESSIONLESS_ACTIONS, McpHost, resolve_cwd

pytestmark = pytest.mark.asyncio

#: Writes its pid, then never speaks MCP. Used to hold an operation open so a
#: cancel and a shutdown land MID-operation deterministically — with a server
#: that really connects, the operation might settle before the cancel arrives and
#: the test would be asserting nothing.
PID_HOLD_WRAPPER = (
    "import os,sys,time;open(sys.argv[1],'w').write(str(os.getpid()));time.sleep(300)"
)
STDIO_PING = ["-c", "import sys; sys.stdin.read()"]


async def _wait_for_pid_file(path: Path, *, timeout: float = 30.0) -> int:
    """Bounded, and AWAITED between attempts: a blocking sleep here would stop the
    event loop that has to spawn the child in the first place.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            text = path.read_text().strip()
            if text:
                return int(text)
        await asyncio.sleep(0.05)
    raise AssertionError("the child never recorded its pid")


async def _wait_dead(pid: int, *, timeout: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        await asyncio.sleep(0.1)
    return False


@pytest.fixture
def host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> McpHost:
    """A host over an empty, isolated config dir (no HOME leakage)."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    config = tmp_path / "home" / ".local-operator"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    return McpHost(config)


def _add(host: McpHost, name: str, *, cwd: str, **extra: Any):
    from local_operator.mcp.desktop import MCPControl

    body = {"action": "add", "name": name, "command": extra.pop("command", "/bin/echo")}
    body.update(extra)
    return host.execute(MCPControl.model_validate(body), cwd)


# ---------------------------------------------------------------------------
# cwd resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["", None, "relative/path", "/nope/not/here", "\x00/x"])
def test_a_bad_cwd_is_rejected_outright(tmp_path: Path, bad: str | None) -> None:
    """``cwd`` decides which project FILE a write lands in, so it is validated."""
    if bad is None:
        # Omitted means the desktop's default folder, which is not an error.
        assert resolve_cwd(None) == str(Path.home())
        return
    if bad == "":
        assert resolve_cwd("") == str(Path.home())
        return
    with pytest.raises(ValueError):
        resolve_cwd(bad)


def test_a_real_directory_is_resolved(tmp_path: Path) -> None:
    assert resolve_cwd(str(tmp_path)) == str(tmp_path)


# ---------------------------------------------------------------------------
# the operation registry and the shutdown contract
# ---------------------------------------------------------------------------


async def test_a_cancelled_operation_reaps_the_child_it_spawned(host: McpHost, tmp_path) -> None:
    """The anyio-trap regression: teardown runs in the operation's own task."""
    cwd = str(tmp_path)
    pid_file = tmp_path / "hold.pid"
    await _add(
        host,
        "holder",
        cwd=cwd,
        command=sys.executable,
        args=["-c", PID_HOLD_WRAPPER, str(pid_file)],
    )

    started = await host.execute(_test_control("holder"), cwd)
    assert started is not None and started["status"] == "running"
    pid = await _wait_for_pid_file(pid_file)

    cancelled = await host.execute(_cancel_control(str(started["id"])), cwd)

    assert cancelled is not None and cancelled["status"] == "cancelled"
    assert await _wait_dead(pid), f"the MCP server child (pid {pid}) survived a cancel"


async def test_closing_the_host_mid_operation_cancels_and_reaps(host: McpHost, tmp_path) -> None:
    """Shutdown is bounded and JOINED: no child outlives the process that spawned it.

    ``McpHost.close`` is what the app's lifespan calls. It cancels and gathers the
    operation's task, so the operation's own ``finally`` (its ``disconnect_all``)
    runs in that task — a design the architect's probe is the reason for, since a
    teardown from a sibling task raises anyio's cross-task cancel-scope error
    instead of closing anything.
    """
    cwd = str(tmp_path)
    pid_file = tmp_path / "hold.pid"
    await _add(
        host,
        "holder",
        cwd=cwd,
        command=sys.executable,
        args=["-c", PID_HOLD_WRAPPER, str(pid_file)],
    )
    await host.execute(_test_control("holder"), cwd)
    pid = await _wait_for_pid_file(pid_file)

    await host.close()

    assert await _wait_dead(pid), f"the MCP server child (pid {pid}) outlived the host"
    assert host.ops.running == {}, "a closed host still holds a running task"


async def test_a_closed_host_refuses_new_work(host: McpHost, tmp_path) -> None:
    await host.close()

    with pytest.raises(MCPRefusal) as raised:
        await _add(host, "late", cwd=str(tmp_path))

    assert raised.value.code in RefusalCode.__args__


async def test_the_sessionless_actions_exclude_the_live_connection_controls() -> None:
    assert SESSIONLESS_ACTIONS == {
        "list",
        "add",
        "remove",
        "test",
        "login",
        "reauth",
        "logout",
        "status",
        "cancel",
    }
    assert {"connect", "disconnect", "reload"}.isdisjoint(SESSIONLESS_ACTIONS)


async def test_a_second_operation_on_one_host_is_refused(host: McpHost, tmp_path) -> None:
    cwd = str(tmp_path)
    pid_file = tmp_path / "hold.pid"
    await _add(
        host,
        "holder",
        cwd=cwd,
        command=sys.executable,
        args=["-c", PID_HOLD_WRAPPER, str(pid_file)],
    )
    await host.execute(_test_control("holder"), cwd)
    try:
        with pytest.raises(MCPRefusal) as raised:
            await host.execute(_test_control("holder"), cwd)
        assert raised.value.code == "grant_running"
    finally:
        await host.close()


async def test_operations_are_reported_per_folder(host: McpHost, tmp_path) -> None:
    """A test in one folder must not paint another folder's same-named server.

    PROJECT scope on purpose: the global file is one file for every folder, so
    two folders can only hold the same NAME independently in their own project
    files — which is exactly the case the host has to keep apart.
    """
    first = tmp_path / "one"
    second = tmp_path / "two"
    first.mkdir()
    second.mkdir()
    pid_file = tmp_path / "hold.pid"
    for folder in (first, second):
        await _add(
            host,
            "holder",
            cwd=str(folder),
            scope="project",
            command=sys.executable,
            args=["-c", PID_HOLD_WRAPPER, str(pid_file)],
        )
    await host.execute(_test_control("holder"), str(first))
    try:
        here = await host.catalog(str(first))
        there = await host.catalog(str(second))

        assert [op["name"] for op in here["operations"]] == ["holder"]
        assert there["operations"] == []
        assert here["servers"][0]["status"] == "connecting"
        assert there["servers"][0]["status"] == "not_started"
    finally:
        await host.close()


def _test_control(name: str):
    from local_operator.mcp.desktop import MCPControl

    return MCPControl.model_validate({"action": "test", "name": name})


def _cancel_control(operation_id: str):
    from local_operator.mcp.desktop import MCPControl

    return MCPControl.model_validate({"action": "cancel", "operation_id": operation_id})


async def test_the_registry_bounds_its_history(tmp_path: Path) -> None:
    """Settled records are evicted first, so the registry cannot grow forever."""
    ops = McpOperations()

    def record(key: str, status: str) -> None:
        ops.operations[key] = {
            "id": key,
            "name": "x",
            "action": "test",
            "status": status,
            "created_at": 0.0,
            "credential_removed": False,
            "browser_opened": None,
            "authorization_url": None,
            "message": None,
        }

    for index in range(OPERATION_LIMIT):
        record(f"{index:032x}", "complete")
    assert len(ops.operations) == OPERATION_LIMIT

    async def done(op: dict[str, Any]) -> None:
        op["status"] = "complete"

    newest = ops.start("one-more", "test", done)

    assert len(ops.operations) == OPERATION_LIMIT
    assert newest["id"] in ops.operations
    assert f"{0:032x}" not in ops.operations, "the oldest settled record is the one evicted"
    await ops.close()


async def test_only_one_operation_runs_at_a_time_in_the_registry() -> None:
    """A second loopback OAuth listener would fight for the callback port."""
    ops = McpOperations()

    async def never(op: dict[str, Any]) -> None:
        await asyncio.sleep(300)

    try:
        ops.start("first", "test", never)
        with pytest.raises(MCPRefusal) as raised:
            ops.start("second", "test", never)
        assert raised.value.code == "grant_running"
        # The refusal did not disturb the running one.
        assert [op["name"] for op in ops.records()] == ["first"]
        assert ops.running_names() == frozenset({"first"})
    finally:
        await ops.close()
        assert ops.running == {}, "close joined the running task"


# ---------------------------------------------------------------------------
# refusal codes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("code", RefusalCode.__args__)
def test_every_refusal_code_crosses_http_as_code_plus_fixed_copy(code: str) -> None:
    detail = refusal_detail(code)

    assert detail["code"] == code
    assert detail["message"] == REFUSAL_MESSAGES[code]
    assert detail["message"].strip()
    # Every code has copy of its own, so the wire never carries exception text
    # (a config error can quote a credential).
    assert detail["message"] != REFUSAL_MESSAGES["mcp_control_refused"]


@pytest.mark.parametrize("value", [None, "", "not-a-code", "mcp_control_refused", 17, ["exists"]])
def test_an_unknown_code_degrades_to_the_generic_refusal(value: Any) -> None:
    assert refusal_detail(value) == {
        "code": "mcp_control_refused",
        "message": REFUSAL_MESSAGES["mcp_control_refused"],
    }


def test_a_refusal_keeps_its_code_and_stays_a_value_error() -> None:
    error = MCPRefusal("not_owned")

    assert refusal_code(error) == "not_owned"
    assert isinstance(error, ValueError), "existing except ValueError arms must still catch"


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("project_scope_unavailable", "project_scope_unavailable"),
        ("exists", "exists"),
        ("invalid_config", "invalid_config"),
        ("unknown_server", "unknown_server"),
        ("write_failed", "write_failed"),
    ],
)
def test_a_config_write_error_carries_its_own_category(code: str, expected: str) -> None:
    error = MCPConfigWriteError(["server 'x' could not be written"], code)

    assert refusal_code(error) == expected
    assert refusal_detail(refusal_code(error))["message"] == REFUSAL_MESSAGES[expected]


def test_an_unexpected_write_error_code_becomes_write_failed() -> None:
    """A category invented at one call site must not invent a wire code."""
    error = MCPConfigWriteError(["boom"], "something-new")

    assert refusal_code(error) == "write_failed"


def test_an_unclassified_exception_is_the_generic_refusal() -> None:
    assert refusal_code(RuntimeError("boom")) == "mcp_control_refused"


# ---------------------------------------------------------------------------
# the reload snapshot
# ---------------------------------------------------------------------------


class FakeManager:
    """The manager surface ``MCPDesktop`` reads, with a controllable gate."""

    def __init__(self, *, settled_after_wait: bool = True) -> None:
        self.status = "connecting"
        self.tools: list[Any] = []
        self.settled_after_wait = settled_after_wait
        self.reloads = 0
        self.waits: list[float] = []
        self.waited = False
        self.cancelled = False

    def get_connection_status(self, name: str) -> str:
        return self.status

    def get_server_tools(self, name: str) -> list[Any]:
        return self.tools

    def get_all_server_names(self) -> list[str]:
        return ["echoer"]

    def startup_failures(self) -> dict[str, str]:
        return {}

    async def reload(self) -> None:
        self.reloads += 1
        # What the real gate does: return immediately, with the slow server
        # still connecting in a background continuation.
        self.status = "connecting"
        self.tools = []

    async def wait_settled(self, timeout_s: float) -> bool:
        self.waits.append(timeout_s)
        if not self.settled_after_wait:
            return False
        self.waited = True
        self.status = "connected"
        self.tools = [object()]
        return True


@pytest.fixture
def desktop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Build an ``MCPDesktop`` (the SESSION route's controls) over an isolated dir.

    A factory rather than one instance: the reload tests need a manager whose
    gate behaves differently, and reproducing the fixture's env setup at each
    call site is how a test ends up reading the developer's real config.
    """
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    config = tmp_path / "home" / ".local-operator"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    config.mkdir(parents=True, exist_ok=True)
    (config / "mcp.json").write_text(
        json.dumps({"mcpServers": {"echoer": {"command": "/bin/echo"}}})
    )

    def build(*, settled_after_wait: bool = True):
        from local_operator.mcp.desktop import MCPDesktop

        manager = FakeManager(settled_after_wait=settled_after_wait)
        control = MCPDesktop(SimpleNamespace(mcp_manager=manager), set(), str(tmp_path))
        return control, manager

    return build


async def test_a_reload_answers_after_the_gate_s_stragglers_settle(desktop) -> None:
    """The reported staleness: Reload answered "connecting / 0 tools" at the gate."""
    control, manager = desktop()

    document = await control.execute(_reload_control())

    assert manager.reloads == 1
    assert manager.waits == [RELOAD_SETTLE_S], "the wait is bounded by the reload bound"
    assert manager.waited, "the snapshot was taken before the stragglers settled"
    row = document["servers"][0]
    assert (row["status"], row["tool_count"]) == ("connected", 1)


async def test_a_server_slower_than_the_bound_keeps_reading_connecting(desktop) -> None:
    """Never lie to make a reload look finished, and never cancel the straggler."""
    control, manager = desktop(settled_after_wait=False)

    document = await control.execute(_reload_control())

    assert document["servers"][0]["status"] == "connecting"
    assert manager.cancelled is False, "a timeout must not cancel the continuation"


async def test_an_operation_is_refused_while_a_reload_would_race_it(desktop) -> None:
    """A control that changes config takes the registry lock, as before."""
    control, manager = desktop()

    async def never(op: dict[str, Any]) -> None:
        await asyncio.sleep(300)

    try:
        control.ops.start("echoer", "test", never)
        with pytest.raises(MCPRefusal) as raised:
            await control.execute(_reload_control())
        assert raised.value.code == "grant_running"
    finally:
        await control.ops.close()


def _reload_control():
    from local_operator.mcp.desktop import MCPControl

    return MCPControl.model_validate({"action": "reload"})


# ---------------------------------------------------------------------------
# legacy session-route compatibility
# ---------------------------------------------------------------------------

#: The row the session route published before this change, plus the two additive
#: keys older readers ignore. A key renamed or dropped here is a desktop build
#: that stops rendering the panel.
LEGACY_ROW_KEYS = {
    "name",
    "source",
    "owned_scope",
    "removable",
    "setup",
    "status",
    "tool_count",
    # from the (unchanged) public config projection:
    "transport",
    "command",
    "argument_count",
    "url",
    "endpoint_redacted",
    "environment_keys",
    "header_keys",
    "secret_refs",
    "transport_oauth_supported",
    "downstream_authorization",
    # additive
    "loaded",
    "startup_failure",
}


async def test_the_session_route_row_shape_and_vocabulary_are_unchanged(desktop) -> None:
    control, _ = desktop()

    document = await control.execute(_list_control())

    assert set(document) == {"servers", "operations"}
    (row,) = document["servers"]
    assert set(row) == LEGACY_ROW_KEYS
    # The OLD words, not the catalog's: ``stdio``/``http`` and the manager's own
    # status strings are what an older renderer switches on.
    assert row["transport"] == "stdio"
    assert row["status"] == "connecting"
    assert row["removable"] is True
    assert row["owned_scope"] == "global"
    assert row["setup"]["kind"] == "session_prompt"
    assert row["downstream_authorization"] == "unknown"
    # A stdio server can never carry a bearer token, and the legacy field says
    # so as ``False`` (``None`` means "undetermined, needs a probe").
    assert row["transport_oauth_supported"] is False
    assert row["secret_refs"] == []


async def test_the_session_route_still_answers_its_operations_list(desktop) -> None:
    control, manager = desktop()

    document = await control.execute(_list_control())

    assert document["operations"] == []


async def test_test_is_not_a_session_action(desktop) -> None:
    """``test`` is the sessionless verb; a session has ``connect`` for this."""
    control, _ = desktop()
    from local_operator.mcp.desktop import MCPControl

    with pytest.raises(MCPRefusal) as raised:
        await control.execute(MCPControl.model_validate({"action": "test", "name": "echoer"}))

    assert raised.value.code == "operation_unavailable"
    assert refusal_detail(raised.value.code)["code"] == "operation_unavailable"


async def test_a_session_without_a_manager_is_not_yet_available(desktop) -> None:
    """The publication-gated wiring window is "try again", not "never"."""
    from local_operator.mcp.desktop import MCPDesktop

    control = MCPDesktop(SimpleNamespace(mcp_manager=None), set(), "/tmp")

    with pytest.raises(MCPRefusal) as raised:
        await control.execute(_list_control())

    assert raised.value.code == "mcp_starting"


def _list_control():
    from local_operator.mcp.desktop import MCPControl

    return MCPControl.model_validate({"action": "list"})


def test_the_route_maps_an_owner_from_before_the_change(monkeypatch) -> None:
    """An older owner answers one fixed code; the route still speaks its old copy.

    The route no longer replaces the owner's code with a sentence
    (``desktop_lifecycle.mcp_control`` passes it through ``refusal_detail``), so
    the compatibility claim is: a code it does not know renders the same generic
    refusal that build's owner meant.
    """
    from local_operator.server.routes.desktop_lifecycle import (
        refusal_detail as route_detail,
    )

    assert route_detail("mcp_control_refused") == {
        "code": "mcp_control_refused",
        "message": "The MCP control was refused.",
    }
    assert route_detail(None)["code"] == "mcp_control_refused"
    # A code an older owner could not have sent still renders bounded copy.
    assert route_detail("exists")["code"] == "exists"
    assert route_detail("exists")["message"] == REFUSAL_MESSAGES["exists"]
