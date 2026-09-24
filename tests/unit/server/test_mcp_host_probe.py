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
from pydantic import SecretStr

from local_operator.mcp.catalog import PROBE_TTL_S, ProbeResult
from local_operator.mcp.config import MCPConfigWriteError
from local_operator.mcp.credentials import MCPCredentials
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

#: A literal header value for the configs that already send their own key. It is
#: never read as a credential: the short, obviously-fake shape is the point.
LITERAL_KEY = "fixture-header-value"

#: The URL ``_add_remote`` writes by default: the key the challenge ledger uses.
DEFAULT_REMOTE_URL = "https://mcp.example.com/sse"

#: The repo's one real stdio MCP peer (one tool, ``fixture_echo``), reached by
#: PATH rather than copied, for the tests whose subject is a COMPLETED operation
#: (a probe written through the real path, and the pruning it does on write).
FIXTURE_SERVER = Path(__file__).resolve().parents[2] / "e2e" / "desktop_mcp_fixture.py"

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


def _add_remote(host: McpHost, name: str, *, cwd: str, **extra: Any):
    """A URL server, which is the shape the grant and key actions apply to."""
    from local_operator.mcp.desktop import MCPControl

    body = {"action": "add", "name": name, "url": "https://mcp.example.com/sse"}
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


def test_the_default_folder_shorthand_is_expanded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``~`` is what the desktop SENDS for the default conversation's folder.

    It is not an absolute path, so a literal ``isabs`` check answered the page's
    own default folder with 422 ``invalid_cwd`` — the install this whole change
    exists to serve. Every other route that takes this folder already accepts
    it, and the MCP config layer expands it too.

    ``HOME`` is redirected: the folder has to EXIST for the check to pass, so a
    test against the real home would write into it.
    """
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    assert resolve_cwd("~") == str(home)
    nested = home / "nested"
    nested.mkdir()
    assert resolve_cwd("~/nested") == str(nested)


@pytest.mark.parametrize(
    "bad", ["~nosuchuser", "~nosuchuser/project", "~root", "~root/", "~daemon/x"]
)
def test_only_the_callers_own_home_shorthand_is_expanded(bad: str) -> None:
    """``~user`` is refused whether or not the account exists (R2-m3).

    ``os.path.expanduser`` resolves ``~root`` to a real directory on macOS and
    Linux, so "no such home to expand" was never the guard it was described as.
    The desktop sends ``~`` or an absolute path; another account's home in
    shorthand is not a case it has, so the shorthand is refused outright.
    """
    with pytest.raises(ValueError):
        resolve_cwd(bad)


def test_another_accounts_home_is_still_reachable_as_an_absolute_path() -> None:
    """The refusal is of the SHORTHAND, not the folder: ``/`` always exists."""
    assert resolve_cwd(os.path.abspath(os.sep)) == os.path.abspath(os.sep)


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


# ---------------------------------------------------------------------------
# what a probe measured, and what invalidates it
# ---------------------------------------------------------------------------


def _seed_probe(
    host: McpHost, cwd: str, name: str, *, status: str = "connected", tools: int | None = 3
) -> None:
    """Record a probe exactly as a completed Test / sign-in records one.

    The digest is the CONFIG's own, so the catalog trusts the entry — every test
    below asserts the row reads it BEFORE it invalidates anything, which is what
    makes the entry a live answer rather than a dead one. Writing the store
    directly is the point: it IS the cache under test, and the alternative (a
    real Test) would need a reachable server to produce the same row.
    """
    from local_operator.mcp.config import load_all_mcp_configs
    from local_operator.mcp.tool_cache import config_digest

    configs, _ = load_all_mcp_configs(cwd)
    host.probes[(cwd, name)] = ProbeResult(
        status=status,  # type: ignore[arg-type]
        reason=None,
        tool_count=tools,
        observed_at=time.time(),
        digest=config_digest(configs[name]),
    )


async def _first_row(host: McpHost, cwd: str) -> dict[str, Any]:
    return (await host.catalog(cwd))["servers"][0]


async def _settle(host: McpHost, operation_id: str, *, timeout: float = 60.0) -> dict[str, Any]:
    """Wait for the operation to leave the running set; return its record."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = next((op for op in host.ops.records() if op["id"] == operation_id), None)
        if record is not None and operation_id not in host.ops.running:
            return record
        await asyncio.sleep(0.05)
    raise AssertionError(f"the operation {operation_id} never settled")


async def test_a_sign_out_forgets_the_probe_it_invalidates(host: McpHost, tmp_path: Path) -> None:
    """A probe that outlives the fact it measured is a lie about that fact.

    A completed sign-out leaves the row reading "Connected, 3 tools" beside a
    Sign in action for up to ``PROBE_TTL_S``. Nothing else can remove this
    entry: ``_record_grant`` is skipped for ``logout`` altogether, and ``_start``
    creates the operation's task without yielding to it, so the assertion right
    after ``execute`` is the invalidation and not a race the operation won.
    """
    cwd = str(tmp_path)
    await _add(host, "echoer", cwd=cwd, command=sys.executable, args=STDIO_PING)
    _seed_probe(host, cwd, "echoer")
    assert (await _first_row(host, cwd))["status"] == "connected", "the seeded probe is live"

    operation = await host.execute(_logout_control("echoer"), cwd)

    assert operation is not None
    assert (cwd, "echoer") not in host.probes
    settled = await _settle(host, str(operation["id"]))
    assert settled["status"] in {"complete", "failed"}, settled
    after = await _first_row(host, cwd)
    assert after["status"] != "connected", after
    assert after["status_basis"] == "stored", after


async def test_a_credentials_write_forgets_the_probe_it_invalidates(
    host: McpHost, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The row told the user to set the key they had just set.

    A Test records ``needs_sign_in`` when a required ``${KEY}`` is missing, and
    that answer stayed the row's after a real write — next to the same row's
    ``auth.signed_in: true``. The write changes the durable fact a probe
    measured, so the answer is dropped and the row is recomputed inside the same
    response: ``routes/desktop_mcp.py`` repaints from a catalog built after the
    operation.
    """
    # Real store, no persistent broker: the same seam the credentials tests use.
    monkeypatch.setattr("local_operator.secrets.client.ensure_broker", lambda *a, **kw: False)
    cwd = str(tmp_path)
    await _add_remote(host, "remote", cwd=cwd, headers={"Authorization": "${PROBEKEY}"})
    _seed_probe(host, cwd, "remote", status="needs_sign_in", tools=None)
    before = await _first_row(host, cwd)
    assert (before["status"], before["auth"]["signed_in"]) == ("needs_sign_in", False)

    result = await host.store_credentials(
        MCPCredentials(name="remote", values={"PROBEKEY": SecretStr("padlock")}), cwd
    )

    assert result["code"] == "saved", result
    assert (cwd, "remote") not in host.probes
    after = await _first_row(host, cwd)
    assert (after["status"], after["status_basis"]) == ("not_started", "stored"), after
    assert after["auth"]["signed_in"] is True, after


@pytest.mark.parametrize("action", ["login", "reauth"])
async def test_a_grant_action_drops_the_probe_before_it_starts(
    host: McpHost, tmp_path: Path, action: str
) -> None:
    """The paths that never COMPLETE are why this happens at start.

    ``reauth`` deletes the stored credential before it reconnects, so a cancel
    in that window leaves the server with no credential — and ``_record_grant``
    never runs, so nothing would have dropped the old answer. Asserted on the
    store right after ``execute`` returns, which is before the operation's task
    has had a chance to run at all.
    """
    cwd = str(tmp_path)
    await _add_remote(host, "remote", cwd=cwd)
    _seed_probe(host, cwd, "remote")
    assert (await _first_row(host, cwd))["status"] == "connected", "the seeded probe is live"

    operation = await host.execute(_grant_control(action, "remote"), cwd)

    assert operation is not None
    assert (cwd, "remote") not in host.probes
    try:
        await host.execute(_cancel_control(str(operation["id"])), cwd)
    finally:
        await host.close()
    assert (await _first_row(host, cwd))["status"] != "connected"


# -- R2-M1: the facts a probe measured are keyed by URL and by secret id --------
#
# Each test below reproduces one contradiction the round-2 review measured on the
# previous head, where invalidation dropped only ``probes[(cwd, name)]``. They use
# TWO folders (the default one, which is the config dir's parent, and a project
# folder) and TWO servers sharing one ``${ID}``, because the stale answer lived
# exactly one folder switch or one sibling server away from the row acted on.


@pytest.fixture
def folders(tmp_path: Path, host: McpHost) -> tuple[str, str]:
    """``(home, project)``: the desktop default folder and a second folder."""
    project = tmp_path / "project"
    project.mkdir()
    return str(tmp_path / "home"), str(project)


@pytest.fixture
def real_store(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real encrypted store, with no persistent broker process."""
    monkeypatch.setattr("local_operator.secrets.client.ensure_broker", lambda *a, **kw: False)


async def test_a_sign_out_in_one_folder_forgets_the_probe_recorded_in_another(
    host: McpHost, folders: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contradiction 1: "Connected, 3 tools" beside Sign in, after a sign-out.

    The server is global, so it is the same server — and the same URL-keyed
    grant — in both folders. The grant is REAL (``auth.db`` under the isolated
    config dir) and so is the logout: the assertion is on what the user reads
    in the folder they did NOT act in.
    """
    from local_operator.mcp.auth import McpTokenStorage, server_has_stored_grant
    from local_operator.mcp.desktop import MCPControl

    # The logout records the URL in the per-process challenge ledger; keep that
    # write inside this test.
    monkeypatch.setattr("local_operator.mcp.auth.OAUTH_CHALLENGES", {})
    home, project = folders
    url = "https://mcp.example.com/sse"
    await host.execute(
        MCPControl.model_validate({"action": "add", "name": "remote", "url": url, "oauth": True}),
        home,
    )
    McpTokenStorage(url)._write({"tokens": {"access_token": "placeholder"}})
    assert server_has_stored_grant(url)
    _seed_probe(host, home, "remote")
    assert (await _first_row(host, home))["status"] == "connected", "the seeded probe is live"

    operation = await host.execute(_logout_control("remote"), project)
    assert operation is not None
    settled = await _settle(host, str(operation["id"]))

    assert settled["status"] == "complete", settled
    assert not server_has_stored_grant(url)
    row = await _first_row(host, home)
    assert (row["status"], row["status_basis"]) == ("needs_sign_in", "stored"), row
    assert row["auth"]["signed_in"] is False
    assert "sign_in" in row["actions"]


async def test_a_key_written_through_one_server_forgets_its_siblings_probe(
    host: McpHost, folders: tuple[str, str], real_store: None
) -> None:
    """Contradiction 2: ``needs_sign_in`` from a probe beside ``signed_in: true``.

    ``a`` and ``b`` both read ``${SHARED}``; the key is written through ``a``,
    and ``b`` is the row that kept the stale answer.
    """
    home, _ = folders
    for name in ("a", "b"):
        await _add_remote(
            host,
            name,
            cwd=home,
            url=f"https://{name}.example.com/mcp",
            headers={"Authorization": "${SHARED}"},
        )
    _seed_probe(host, home, "b", status="needs_sign_in", tools=None)
    rows = {row["name"]: row for row in (await host.catalog(home))["servers"]}
    assert (rows["b"]["status"], rows["b"]["status_basis"]) == ("needs_sign_in", "probe")

    result = await host.store_credentials(
        MCPCredentials(name="a", values={"SHARED": SecretStr("padlock")}), home
    )

    assert result["code"] == "saved", result
    rows = {row["name"]: row for row in (await host.catalog(home))["servers"]}
    for name in ("a", "b"):
        assert rows[name]["auth"]["signed_in"] is True, rows[name]
        assert (rows[name]["status"], rows[name]["status_basis"]) == ("not_started", "stored")


async def test_a_key_written_from_another_folder_forgets_this_folders_probe(
    host: McpHost, folders: tuple[str, str], real_store: None
) -> None:
    """Contradiction 3: the same global server's key, written from elsewhere."""
    home, project = folders
    await _add_remote(host, "remote", cwd=home, headers={"Authorization": "${PROBEKEY}"})
    _seed_probe(host, home, "remote", status="needs_sign_in", tools=None)
    assert (await _first_row(host, home))["status_basis"] == "probe"

    result = await host.store_credentials(
        MCPCredentials(name="remote", values={"PROBEKEY": SecretStr("padlock")}), project
    )

    assert result["code"] == "saved", result
    row = await _first_row(host, home)
    assert row["auth"]["signed_in"] is True, row
    assert (row["status"], row["status_basis"]) == ("not_started", "stored"), row


async def test_a_refused_key_write_forgets_nothing(
    host: McpHost, folders: tuple[str, str], real_store: None
) -> None:
    """A write the store refused changed no fact, so every probe still holds."""
    home, _ = folders
    await _add_remote(host, "remote", cwd=home, headers={"Authorization": "${PROBEKEY}"})
    first = MCPCredentials(name="remote", values={"PROBEKEY": SecretStr("padlock")})
    assert (await host.store_credentials(first, home))["code"] == "saved"
    _seed_probe(host, home, "remote")

    again = MCPCredentials(name="remote", values={"PROBEKEY": SecretStr("deadbolt")})
    result = await host.store_credentials(again, home)

    assert result["code"] == "replace_confirmation_required", result
    assert (home, "remote") in host.probes
    assert (await _first_row(host, home))["status_basis"] == "probe"


#: Waits for ``argv[2]`` to exist, THEN serves the real fixture server. The pid
#: file says the child is up — so the manager has already resolved its env from
#: the store — and the gate file lets the test write a new key before the Test
#: can finish, which is the in-flight window R2-m1 is about.
GATED_SERVER = (
    "import os,sys,time,runpy;"
    "open(sys.argv[1],'w').write(str(os.getpid()));"
    "[time.sleep(0.05) for _ in iter(lambda: os.path.exists(sys.argv[2]), True)];"
    "sys.argv=[sys.argv[3]];runpy.run_path(sys.argv[0],run_name='__main__')"
)


async def test_a_test_in_flight_across_a_key_replace_records_no_probe(
    host: McpHost, tmp_path: Path, folders: tuple[str, str], real_store: None
) -> None:
    """R2-m1: the Test resolved the OLD value, so its answer is not the row's.

    The operation itself still settles and reports what it saw (``complete``);
    only the cached probe is withheld, so the row recomputes from the stores.
    """
    home, _ = folders
    pid_file, gate = tmp_path / "gated.pid", tmp_path / "gate"
    await _add(
        host,
        "gated",
        cwd=home,
        command=sys.executable,
        args=["-c", GATED_SERVER, str(pid_file), str(gate), str(FIXTURE_SERVER)],
        env={"FIXTURE_KEY": "${FIXTURE_KEY}"},
    )
    first = MCPCredentials(name="gated", values={"FIXTURE_KEY": SecretStr("padlock")})
    assert (await host.store_credentials(first, home))["code"] == "saved"

    operation = await host.execute(_test_control("gated"), home)
    assert operation is not None
    await _wait_for_pid_file(pid_file)
    replace = MCPCredentials(
        name="gated",
        values={"FIXTURE_KEY": SecretStr("deadbolt")},
        confirmed_replace=["FIXTURE_KEY"],
    )
    assert (await host.store_credentials(replace, home))["code"] == "saved"
    gate.touch()
    settled = await _settle(host, str(operation["id"]))

    assert settled["status"] == "complete", settled
    assert (home, "gated") not in host.probes, "a probe measured with the old key was kept"
    row = await _first_row(host, home)
    assert (row["status"], row["status_basis"]) == ("not_started", "stored"), row


# -- add_key: binding a header for a server that declares no reference -------------


async def test_add_key_binds_a_header_reference_and_stores_the_value(
    host: McpHost, folders: tuple[str, str], real_store: None, ledger: dict[str, bool]
) -> None:
    """The row goes from ``add_key`` to ``set_key`` with its key held."""
    home, _ = folders
    await _add_remote(host, "acme", cwd=home)
    ledger[DEFAULT_REMOTE_URL] = False

    result = await host.store_credentials(
        MCPCredentials(name="acme", values={"ACME_KEY": SecretStr("padlock")}),
        home,
        header="X-Api-Key",
    )

    assert result["code"] == "saved", result
    doc = json.loads((Path(home) / ".local-operator" / "mcp.json").read_text())
    assert doc["mcpServers"]["acme"]["headers"] == {"X-Api-Key": "${ACME_KEY}"}
    row = await _first_row(host, home)
    assert row["auth"]["secret_refs"] == [{"id": "ACME_KEY", "state": "encrypted"}]
    assert "set_key" in row["actions"] and "add_key" not in row["actions"]


@pytest.mark.parametrize(
    ("header", "secret_id"),
    [
        ("Authorization", "ACME_KEY"),  # the server already sets it
        ("Content-Type", "ACME_KEY"),  # the transport owns it
        ("X-Api-Key\r\nX-Evil", "ACME_KEY"),  # not a header token
        ("X-Api-Key", "not-a-reference"),  # the resolver would never publish it
    ],
)
async def test_add_key_refuses_and_writes_nothing(
    host: McpHost, folders: tuple[str, str], real_store: None, header: str, secret_id: str
) -> None:
    home, _ = folders
    await _add_remote(host, "acme", cwd=home, headers={"Authorization": "${OTHER}"})
    path = Path(home) / ".local-operator" / "mcp.json"
    before = path.read_text()

    result = await host.store_credentials(
        MCPCredentials(name="acme", values={secret_id: SecretStr("padlock")}),
        home,
        header=header,
    )

    assert (result["code"], result["saved_ids"]) == ("invalid_target", []), result
    assert path.read_text() == before


async def test_add_key_rolls_the_binding_back_when_the_store_refuses(
    host: McpHost, folders: tuple[str, str], real_store: None, ledger: dict[str, bool]
) -> None:
    """The id is already held and the replace was not confirmed: config unchanged."""
    home, _ = folders
    await _add_remote(host, "keeper", cwd=home, headers={"Authorization": "${ACME_KEY}"})
    held = MCPCredentials(name="keeper", values={"ACME_KEY": SecretStr("padlock")})
    assert (await host.store_credentials(held, home))["code"] == "saved"
    await _add_remote(host, "acme", cwd=home, url="https://acme.example.com/mcp")
    ledger["https://acme.example.com/mcp"] = False
    path = Path(home) / ".local-operator" / "mcp.json"
    before = path.read_text()

    result = await host.store_credentials(
        MCPCredentials(name="acme", values={"ACME_KEY": SecretStr("deadbolt")}),
        home,
        header="X-Api-Key",
    )

    assert result["code"] == "replace_confirmation_required", result
    assert json.loads(path.read_text()) == json.loads(before)


@pytest.fixture
def ledger(monkeypatch: pytest.MonkeyPatch) -> dict[str, bool]:
    """The per-process 401 ledger, isolated around one test.

    A row is offered ``add_key`` either because its config says ``auth.type:
    apikey`` or because a Test watched it answer 401/403 with no OAuth
    discovery. Every ``add_key`` test below has to state WHICH of those it is
    exercising, because the write path now refuses any row the catalog would not
    offer the action on (review round 3, R3-m1).
    """
    from local_operator.mcp import auth

    fresh: dict[str, bool] = {}
    monkeypatch.setattr(auth, "OAUTH_CHALLENGES", fresh)
    return fresh


async def test_a_server_that_already_sends_a_key_is_offered_no_add_key(
    host: McpHost, folders: tuple[str, str], ledger: dict[str, bool]
) -> None:
    """R3-M1: a literal header keeps the row at ``not_started`` + a real key.

    The row was ``needs_sign_in`` / ``signed_in: false`` / ``add_key`` while the
    server was in fact sending its key and working. Written raw because the
    desktop's own writer refuses a literal header — this is the hand-edited or
    foreign-imported shape, which is the one the loader passes through to the
    transport untouched.
    """
    home, _ = folders
    url = DEFAULT_REMOTE_URL
    path = Path(home) / ".local-operator" / "mcp.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "literal": {
                        "type": "http",
                        "url": url,
                        "headers": {"X-Api-Key": LITERAL_KEY},
                        "auth": {"type": "apikey"},
                    }
                }
            }
        )
    )
    ledger[url] = False

    row = await _first_row(host, home)

    assert row["auth"] == {"kind": "api_key", "signed_in": True, "secret_refs": []}, row
    assert (row["status"], row["status_basis"]) == ("not_started", "stored"), row
    assert row["actions"] == ["test", "remove"], row


async def test_a_header_write_is_refused_for_a_row_that_does_not_offer_add_key(
    host: McpHost, folders: tuple[str, str], real_store: None, ledger: dict[str, bool]
) -> None:
    """R3-m1: the write path enforces exactly what the row offers.

    Measured on the previous head: an ``auth.type: oauth`` server returned
    ``saved`` and got a second ``Authorization`` bound beside the OAuth
    provider's own. Only a client that ignored ``actions`` reaches this, which is
    the point — the server is the enforcement point, not the page.
    """
    home, _ = folders
    url = DEFAULT_REMOTE_URL
    await _add_remote(host, "oauth", cwd=home, url=url, oauth=True)
    await _add_remote(host, "plain", cwd=home)
    path = Path(home) / ".local-operator" / "mcp.json"
    before = path.read_bytes()

    for name in ("oauth", "plain"):
        result = await host.store_credentials(
            MCPCredentials(name=name, values={"ACME_KEY": SecretStr("padlock")}),
            home,
            header="Authorization",
        )
        assert (result["code"], result["saved_ids"]) == ("invalid_target", []), result

    assert path.read_bytes() == before, "a refused write touched the config"
    rows = {row["name"]: row for row in (await host.catalog(home))["servers"]}
    assert "add_key" not in rows["oauth"]["actions"], rows["oauth"]
    assert "add_key" not in rows["plain"]["actions"], rows["plain"]


async def test_add_key_refuses_more_than_one_id(
    host: McpHost, folders: tuple[str, str], real_store: None, ledger: dict[str, bool]
) -> None:
    """R3-m3: the one-id rule the ``add_key`` form states.

    The header can name exactly one ``${ID}``, so a body with two is refused
    rather than silently binding whichever came first out of a dict.
    """
    home, _ = folders
    await _add_remote(host, "acme", cwd=home)
    ledger[DEFAULT_REMOTE_URL] = False
    path = Path(home) / ".local-operator" / "mcp.json"
    before = path.read_bytes()

    result = await host.store_credentials(
        MCPCredentials(
            name="acme", values={"ACME_KEY": SecretStr("padlock"), "OTHER_KEY": SecretStr("bar")}
        ),
        home,
        header="X-Api-Key",
    )

    assert (result["code"], result["saved_ids"]) == ("invalid_target", []), result
    assert path.read_bytes() == before


async def test_a_refused_add_key_leaves_the_file_byte_identical(
    host: McpHost, folders: tuple[str, str], real_store: None, ledger: dict[str, bool]
) -> None:
    """QA Q-2: the rollback restores the file's BYTES, not a re-serialisation.

    The store refusing after the bind used to leave a hand-formatted file
    reindented (4 spaces to 2) with a final newline it never had. The bytes are
    the user's file, so nothing short of them is "as it was".
    """
    home, _ = folders
    await _add_remote(host, "keeper", cwd=home, headers={"Authorization": "${ACME_KEY}"})
    held = MCPCredentials(name="keeper", values={"ACME_KEY": SecretStr("padlock")})
    assert (await host.store_credentials(held, home))["code"] == "saved"
    await _add_remote(host, "acme", cwd=home, url=DEFAULT_REMOTE_URL)
    ledger[DEFAULT_REMOTE_URL] = False
    path = Path(home) / ".local-operator" / "mcp.json"
    # Hand-formatted: 4-space indent and no final newline, which is what the
    # writer does NOT produce and therefore what a re-serialisation loses.
    doc = json.loads(path.read_text())
    hand = json.dumps(doc, indent=4, ensure_ascii=False)
    path.write_text(hand)

    result = await host.store_credentials(
        MCPCredentials(name="acme", values={"ACME_KEY": SecretStr("deadbolt")}),
        home,
        header="X-Api-Key",
    )

    assert result["code"] == "replace_confirmation_required", result
    assert path.read_text() == hand, "the refused write changed the file's bytes"
    assert "headers" not in json.loads(path.read_text())["mcpServers"]["acme"]


async def test_a_key_write_takes_the_registry_lock(
    host: McpHost, folders: tuple[str, str], real_store: None, ledger: dict[str, bool]
) -> None:
    """R3-m3: the bind is a CONFIG write, so it serialises with add/remove.

    Asserted by HOLDING the lock and showing the write cannot proceed, which is
    the property the lock exists for: two read-modify-write passes over one
    mcp.json lose whichever landed first.
    """
    home, _ = folders
    await _add_remote(host, "acme", cwd=home)
    ledger[DEFAULT_REMOTE_URL] = False

    async with host.ops.lock:
        task = asyncio.create_task(
            host.store_credentials(
                MCPCredentials(name="acme", values={"ACME_KEY": SecretStr("padlock")}),
                home,
                header="X-Api-Key",
            )
        )
        await asyncio.sleep(0.25)
        assert not task.done(), "the key write ran without the registry lock"

    result = await asyncio.wait_for(task, timeout=30)
    assert result["code"] == "saved", result


async def test_a_login_across_a_key_write_records_no_probe(
    host: McpHost, folders: tuple[str, str], real_store: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R3-m3: the ``_record_grant`` epoch gate.

    A sign-in resolves the values stored when it STARTED, so when a key write
    lands mid-grant its "connected" is about a credential that is gone. The
    grant's task is held open here between ``_record_grant``'s two observations,
    which is exactly the window the epoch closes.
    """
    home, _ = folders
    await _add_remote(host, "remote", cwd=home, headers={"Authorization": "${PROBEKEY}"})
    started, release = asyncio.Event(), asyncio.Event()

    async def fake_grant(manager, action, name, op, cfg):
        op["status"] = "complete"
        started.set()
        await release.wait()
        return 3

    monkeypatch.setattr("local_operator.server.mcp_host.grant_operation", fake_grant)
    operation = await host.execute(_grant_control("login", "remote"), home)
    assert operation is not None
    await asyncio.wait_for(started.wait(), timeout=30)
    replace = MCPCredentials(
        name="remote", values={"PROBEKEY": SecretStr("deadbolt")}, confirmed_replace=["PROBEKEY"]
    )
    assert (await host.store_credentials(replace, home))["code"] == "saved"
    release.set()
    settled = await _settle(host, str(operation["id"]))

    assert settled["status"] == "complete", settled
    assert (home, "remote") not in host.probes, "a grant recorded the pre-write facts"
    row = await _first_row(host, home)
    assert (row["status"], row["status_basis"]) == ("not_started", "stored"), row


async def test_a_new_probe_prunes_the_expired_ones(host: McpHost, tmp_path: Path) -> None:
    """``PROBE_TTL_S`` is read at read time, so nothing else ever collected one.

    A long-lived daemon kept one entry per folder × server ever tested, each
    holding its sanitized reason string. The real path is exercised: a Test that
    reaches a real MCP server writes its probe through the same helper.
    """
    cwd = str(tmp_path)
    await _add(host, "fixture", cwd=cwd, command=sys.executable, args=[str(FIXTURE_SERVER)])
    stale = ProbeResult(
        status="connected",
        reason="a server that is long gone",
        tool_count=1,
        observed_at=time.time() - PROBE_TTL_S - 1.0,
        digest="0" * 64,
    )
    host.probes[(str(tmp_path / "other"), "gone")] = stale

    operation = await host.execute(_test_control("fixture"), cwd)
    assert operation is not None
    settled = await _settle(host, str(operation["id"]))

    assert settled["status"] == "complete", settled
    assert (str(tmp_path / "other"), "gone") not in host.probes, "an expired probe survived"
    assert (cwd, "fixture") in host.probes


def _cancel_control(operation_id: str):
    from local_operator.mcp.desktop import MCPControl

    return MCPControl.model_validate({"action": "cancel", "operation_id": operation_id})


def _logout_control(name: str):
    from local_operator.mcp.desktop import MCPControl

    return MCPControl.model_validate({"action": "logout", "name": name, "confirmed": True})


def _grant_control(action: str, name: str):
    from local_operator.mcp.desktop import MCPControl

    return MCPControl.model_validate({"action": action, "name": name, "confirmed": True})


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
