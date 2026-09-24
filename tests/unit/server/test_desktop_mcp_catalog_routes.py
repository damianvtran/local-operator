"""``/v1/desktop/mcp`` — MCP management with no session and no model.

The bug this surface exists for: every MCP read and write used to go through
``/v1/desktop/sessions/{id}/mcp``, which needs a conversation, which needs a
configured model. On a fresh install the Settings > Integrations page could not
list anything, and on a configured one, editing a JSON file started a whole
runtime and spawned every server in it.

So the state under test here is the one the old route could not serve at all: an
EMPTY config dir, NO model configured, NO session, and every listed action still
working. The owner-related behaviours are driven through a fake pool shaped like
``DesktopSessions`` (``host(request).session(...)`` → ``bridge.remote``), because
the two claims that matter about the overlay are properties of that seam: a read
must never BIND a runtime, and only an already-warm session in the SAME folder
may overlay live statuses.

The ``test`` action spawns a real child process (the repo's canonical stdio MCP
peer, ``tests/e2e/desktop_mcp_fixture.py``), wrapped so the child records its own
pid: "the operation tore its manager down" is only evidence if the thing that was
spawned is actually gone, and a fake transport cannot show that.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.server.retire import RETIRING_STATE_ATTR
from local_operator.server.routes import desktop_mcp

TOKEN = "desktop-mcp-route-test-token"
#: The repo's one real stdio MCP peer (one tool, ``fixture_echo``), shared with
#: the e2e suite rather than copied: a second fixture server would drift.
FIXTURE_SERVER = Path(__file__).resolve().parents[2] / "e2e" / "desktop_mcp_fixture.py"
#: Writes the child's pid, then ``execv``-s the real server — so the pid the test
#: watches is the server's own, and stdout/stdin stay the MCP pipes.
PID_WRAPPER = (
    "import os,sys;open(sys.argv[2],'w').write(str(os.getpid()));"
    "os.execv(sys.executable,[sys.executable,sys.argv[1]])"
)

pytestmark = pytest.mark.asyncio


class FakeRemote:
    """A viewer facade with the facts ``_live_overlay`` reads and nothing else."""

    def __init__(self, cwd: str, *, cold: bool = False, rows: list[dict[str, Any]] | None = None):
        self.frontend_state = SimpleNamespace(cwd=cwd)
        self._cold = cold
        self._rows = rows if rows is not None else []
        self.binds = 0
        self.slashes: list[tuple[str, str]] = []

    @property
    def is_cold(self) -> bool:
        return self._cold

    async def bind_runtime(self) -> None:
        self.binds += 1

    async def route_shared_slash(self, command: str, args: str) -> dict[str, Any]:
        self.slashes.append((command, args))
        return {"kind": "block", "data": {"servers": self._rows, "operations": []}}


class FakePool:
    """``DesktopSessions``-shaped: the route's only door to a session."""

    def __init__(self, remotes: dict[str, FakeRemote]) -> None:
        self.remotes = remotes

    @contextlib.asynccontextmanager
    async def session(self, session_id: str, *, read: bool = False):
        del read
        remote = self.remotes.get(session_id)
        if remote is None:
            raise KeyError("Unknown session")
        yield SimpleNamespace(remote=remote)


@pytest_asyncio.fixture
async def app_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A daemon with an EMPTY config dir, no model, and a fake session pool.

    ``HOME`` and the config dir both move, because a "fresh install" is a claim
    about both: the loader reads the user-scope MCP files under ``HOME``, and the
    global ``mcp.json`` and the secret store live under the config dir.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    # The real install's shape: ``~/.local-operator``. That is what makes the
    # DEFAULT folder the colliding one, which is the scope bug this route has to
    # answer for; a config dir somewhere else would quietly test the easy case.
    config = home / ".local-operator"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    app = FastAPI()
    app.include_router(desktop_mcp.router)
    app.state.config_manager = SimpleNamespace(config_dir=config)
    app.state.desktop_sessions = FakePool({})
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        yield SimpleNamespace(client=client, app=app, home=home, config=config, tmp=tmp_path)


def _data(response) -> dict[str, Any]:
    body = response.json()
    assert "result" in body, body
    return body["result"]["data"]


def _row(document: dict[str, Any], name: str) -> dict[str, Any]:
    for row in document["servers"]:
        if row["name"] == name:
            return row
    raise AssertionError(f"no row for {name}: {[r['name'] for r in document['servers']]}")


def _add_body(
    folder: str, *, name: str = "echoer", scope: str = "global", **extra: Any
) -> dict[str, Any]:
    body = {
        "action": "add",
        "name": name,
        "scope": scope,
        "command": sys.executable,
        "args": ["-c", "import sys; sys.stdin.read()"],
        "cwd": folder,
    }
    body.update(extra)
    return body


# ---------------------------------------------------------------------------
# the fresh install
# ---------------------------------------------------------------------------


async def test_a_fresh_install_lists_nothing_and_needs_no_session(app_env) -> None:
    """No model configured, no conversation, and the page still answers."""
    response = await app_env.client.get("/v1/desktop/mcp")

    assert response.status_code == 200
    document = _data(response)
    assert document["cwd"] == str(Path.home())
    assert document["servers"] == []
    assert document["operations"] == []
    assert document["status_source"] == "config"
    assert document["session_id"] is None
    assert document["global_path"] == str(app_env.config / "mcp.json")
    # The default folder IS the config dir's parent here, so its "project" file
    # would be the global one — the scope the UI must not offer.
    assert document["project_scope_available"] is False
    assert document["project_path"] is None


async def test_a_folder_with_its_own_file_gets_a_project_scope(app_env) -> None:
    project = app_env.tmp / "project"
    project.mkdir()

    document = _data(await app_env.client.get("/v1/desktop/mcp", params={"cwd": str(project)}))

    assert document["project_scope_available"] is True
    assert document["project_path"] == str(project / ".local-operator" / "mcp.json")
    assert document["cwd"] == str(project)


@pytest.mark.parametrize("bad", ["relative/path", "/nonexistent/folder/for/tests"])
async def test_a_cwd_that_is_not_an_absolute_existing_directory_is_422(app_env, bad: str) -> None:
    response = await app_env.client.get("/v1/desktop/mcp", params={"cwd": bad})

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_cwd"


async def test_a_post_cwd_is_validated_the_same_way(app_env) -> None:
    response = await app_env.client.post(
        "/v1/desktop/mcp", json=_add_body("/nonexistent/folder/for/tests")
    )

    assert response.status_code == 422


# ---------------------------------------------------------------------------
# add / remove
# ---------------------------------------------------------------------------


async def test_add_writes_the_config_and_answers_the_whole_catalog(app_env) -> None:
    response = await app_env.client.post("/v1/desktop/mcp", json=_add_body(str(Path.home())))

    assert response.status_code == 200
    document = _data(response)
    assert document["operation"] is None
    row = _row(document, "echoer")
    assert row["scope"] == "global"
    assert row["transport"] == "local_command"
    assert row["status"] == "not_started"
    assert row["source"]["editable"] is True
    assert row["actions"] == ["test", "remove"]
    written = json.loads((app_env.config / "mcp.json").read_text())
    assert written["mcpServers"]["echoer"]["command"] == sys.executable


async def test_add_into_a_project_scope_writes_the_project_file(app_env) -> None:
    project = app_env.tmp / "project"
    project.mkdir()

    response = await app_env.client.post(
        "/v1/desktop/mcp", json=_add_body(str(project), scope="project")
    )

    assert response.status_code == 200
    row = _row(_data(response), "echoer")
    assert row["scope"] == "project"
    assert row["project_cwd"] == str(project)
    assert (project / ".local-operator" / "mcp.json").exists()
    assert not (app_env.config / "mcp.json").exists()


async def test_a_project_add_where_the_files_collide_is_refused(app_env) -> None:
    """The default folder: "only this folder" means the global file, so refuse."""
    response = await app_env.client.post(
        "/v1/desktop/mcp", json=_add_body(str(Path.home()), scope="project")
    )

    assert response.status_code == 409
    assert response.json()["detail"] == {
        "code": "project_scope_unavailable",
        "message": (
            "This folder has no separate project scope; its project file is the global one."
        ),
    }
    assert not (app_env.config / "mcp.json").exists(), "a refused write touched the file"


async def test_add_twice_is_refused_with_exists(app_env) -> None:
    first = await app_env.client.post("/v1/desktop/mcp", json=_add_body(str(Path.home())))
    assert first.status_code == 200

    second = await app_env.client.post("/v1/desktop/mcp", json=_add_body(str(Path.home())))

    assert second.status_code == 409
    assert second.json()["detail"]["code"] == "exists"


async def test_remove_needs_the_owning_scope_and_is_then_unknown(app_env) -> None:
    await app_env.client.post("/v1/desktop/mcp", json=_add_body(str(Path.home())))

    removed = await app_env.client.post(
        "/v1/desktop/mcp",
        json={"action": "remove", "name": "echoer", "scope": "global", "confirmed": True},
    )

    assert removed.status_code == 200
    assert _data(removed)["servers"] == []
    again = await app_env.client.post(
        "/v1/desktop/mcp",
        json={"action": "remove", "name": "echoer", "scope": "global", "confirmed": True},
    )
    assert again.status_code == 409
    assert again.json()["detail"]["code"] == "unknown_server"


async def test_removing_an_imported_server_is_not_owned(app_env) -> None:
    """``<cwd>/.mcp.json`` is read, never written: the remove must say so."""
    project = app_env.tmp / "project"
    project.mkdir()
    (project / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"borrowed": {"command": sys.executable}}})
    )

    response = await app_env.client.post(
        "/v1/desktop/mcp",
        json={
            "action": "remove",
            "name": "borrowed",
            "scope": "project",
            "confirmed": True,
            "cwd": str(project),
        },
    )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "not_owned"
    assert (project / ".mcp.json").exists(), "a refusal rewrote a foreign file"


async def test_an_unconfirmed_remove_is_rejected_by_the_schema(app_env) -> None:
    """``MCPControl``'s closed schema still guards the sessionless route."""
    response = await app_env.client.post(
        "/v1/desktop/mcp", json={"action": "remove", "name": "echoer", "scope": "global"}
    )

    assert response.status_code == 422


# ---------------------------------------------------------------------------
# operations
# ---------------------------------------------------------------------------


async def _until_settled(
    client: AsyncClient, folder: str, *, timeout: float = 60.0
) -> dict[str, Any]:
    """Poll the catalog until no operation is running AND no row is connecting.

    Both halves matter, and they settle a moment apart on purpose: an operation
    records its outcome (``complete``) and only THEN tears its manager down, so
    during that teardown the row still reads ``connecting``. Waiting only on the
    operation status would sample the row in that window; waiting on the row is
    what a user is actually waiting for, and it is what the UI's poll condition
    (a connecting row OR a running operation) converges on.
    """
    deadline = time.monotonic() + timeout
    document: dict[str, Any] = {}
    while time.monotonic() < deadline:
        document = _data(await client.get("/v1/desktop/mcp", params={"cwd": folder}))
        busy = any(op["status"] == "running" for op in document["operations"])
        connecting = any(row["status"] == "connecting" for row in document["servers"])
        if not busy and not connecting:
            return document
        await asyncio.sleep(0.2)
    raise AssertionError(f"an operation never settled: {document}")


def _wait_for_pid_file(path: Path, *, timeout: float = 30.0) -> int:
    """Blocking on purpose: this polls for a file a LONG-LIVED child writes, from a
    synchronous helper used after an async settle — see the awaited sibling in
    ``test_mcp_host_probe.py``, where the child is still being spawned and
    blocking the loop would stop it from ever starting.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            text = path.read_text().strip()
            if text:
                return int(text)
        time.sleep(0.05)
    raise AssertionError("the child never recorded its pid")


def _wait_dead(pid: int, *, timeout: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.1)
    return False


async def test_test_connects_a_real_child_and_leaves_nothing_behind(app_env) -> None:
    """The whole op, end to end: start, settle connected, child gone."""
    pid_file = app_env.tmp / "child.pid"
    body = _add_body(str(Path.home()), name="fixture")
    body["args"] = ["-c", PID_WRAPPER, str(FIXTURE_SERVER), str(pid_file)]
    assert (await app_env.client.post("/v1/desktop/mcp", json=body)).status_code == 200

    started = await app_env.client.post(
        "/v1/desktop/mcp", json={"action": "test", "name": "fixture", "cwd": str(Path.home())}
    )

    assert started.status_code == 200
    document = _data(started)
    operation = document["operation"]
    assert operation["action"] == "test"
    assert operation["status"] == "running"
    assert operation["browser_opened"] is None, "a test must never open a browser"
    # While it runs, the row is connecting with the OPERATION basis (the
    # contract): no probe has answered yet, and the pair is what lets a client
    # tell "a Test is running" from "a Test ended like this".
    row = _row(document, "fixture")
    assert (row["status"], row["status_basis"]) == ("connecting", "operation")
    assert row["status_observed_at"] is None

    settled = await _until_settled(app_env.client, str(Path.home()))
    row = _row(settled, "fixture")
    assert row["status"] == "connected", row
    assert row["tool_count"] == 1, "the fixture exposes exactly one tool"
    # ``probe``, not ``live``: this answer came from an explicit Test on the
    # sessionless host, and a probe is never allowed to masquerade as a runtime.
    assert row["tool_count_basis"] == "probe"
    assert row["status_basis"] == "probe"
    assert row["status_observed_at"] is not None
    assert settled["operations"][0]["status"] == "complete"

    pid = _wait_for_pid_file(pid_file)
    assert _wait_dead(pid), f"the MCP server child (pid {pid}) outlived its operation"


async def test_a_failed_test_is_a_failed_operation_with_a_reason(app_env) -> None:
    body = _add_body(str(Path.home()), name="doomed", args=["-c", "raise SystemExit(3)"])
    assert (await app_env.client.post("/v1/desktop/mcp", json=body)).status_code == 200

    await app_env.client.post(
        "/v1/desktop/mcp", json={"action": "test", "name": "doomed", "cwd": str(Path.home())}
    )
    settled = await _until_settled(app_env.client, str(Path.home()))

    row = _row(settled, "doomed")
    assert row["status"] == "error"
    assert row["status_reason"], "an error row must say why"
    assert settled["operations"][0]["status"] == "failed"
    assert settled["operations"][0]["message"] == row["status_reason"]


async def test_a_second_operation_while_one_runs_is_refused(app_env) -> None:
    """One operation at a time: a second loopback listener would fight for the port.

    The first operation is a server that WILL NOT finish (a child that never
    speaks MCP) rather than the fixture: with a real, fast connect the second
    press could land after the first op had already settled, which is a race the
    assertion is not about.
    """
    body = _add_body(str(Path.home()), name="slow", args=["-c", "import time; time.sleep(60)"])
    await app_env.client.post("/v1/desktop/mcp", json=body)

    first = await app_env.client.post(
        "/v1/desktop/mcp", json={"action": "test", "name": "slow", "cwd": str(Path.home())}
    )
    assert first.status_code == 200
    try:
        second = await app_env.client.post(
            "/v1/desktop/mcp",
            json={"action": "test", "name": "slow", "cwd": str(Path.home())},
        )
        assert second.status_code == 409
        assert second.json()["detail"]["code"] == "grant_running"
        # A second attempt to START is refused, but reading the running one works.
        listing = await app_env.client.get("/v1/desktop/mcp", params={"cwd": str(Path.home())})
        assert _row(_data(listing), "slow")["status"] == "connecting"
    finally:
        operation_id = _data(first)["operation"]["id"]
        await app_env.client.post(
            "/v1/desktop/mcp",
            json={
                "action": "cancel",
                "operation_id": operation_id,
                "cwd": str(Path.home()),
            },
        )
        await _until_settled(app_env.client, str(Path.home()))


async def test_status_and_cancel_name_one_operation(app_env) -> None:
    body = _add_body(str(Path.home()), name="slow", args=["-c", "import time; time.sleep(60)"])
    await app_env.client.post("/v1/desktop/mcp", json=body)
    started = await app_env.client.post(
        "/v1/desktop/mcp", json={"action": "test", "name": "slow", "cwd": str(Path.home())}
    )
    operation_id = _data(started)["operation"]["id"]

    status = await app_env.client.post(
        "/v1/desktop/mcp",
        json={"action": "status", "operation_id": operation_id, "cwd": str(Path.home())},
    )
    assert status.status_code == 200
    assert _data(status)["operation"]["id"] == operation_id

    cancelled = await app_env.client.post(
        "/v1/desktop/mcp",
        json={"action": "cancel", "operation_id": operation_id, "cwd": str(Path.home())},
    )
    assert cancelled.status_code == 200
    settled = await _until_settled(app_env.client, str(Path.home()))
    assert settled["operations"][0]["status"] == "cancelled"
    assert _row(settled, "slow")["status"] == "not_started", "a cancelled probe claims nothing"


async def test_an_unknown_operation_id_is_refused_with_a_code(app_env) -> None:
    response = await app_env.client.post(
        "/v1/desktop/mcp",
        json={
            "action": "status",
            "operation_id": "0" * 32,
            "cwd": str(Path.home()),
        },
    )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "operation_unavailable"


@pytest.mark.parametrize("action", ["connect", "disconnect", "reload"])
async def test_live_connection_controls_stay_on_the_session_route(app_env, action: str) -> None:
    response = await app_env.client.post(
        "/v1/desktop/mcp",
        json={"action": action, "name": "echoer", "confirmed": True, "cwd": str(Path.home())},
    )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "operation_unavailable"


async def test_a_login_on_a_local_command_is_refused_as_oauth_unsupported(app_env) -> None:
    await app_env.client.post("/v1/desktop/mcp", json=_add_body(str(Path.home())))

    response = await app_env.client.post(
        "/v1/desktop/mcp", json={"action": "login", "name": "echoer", "cwd": str(Path.home())}
    )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "oauth_unsupported"


# ---------------------------------------------------------------------------
# credentials
# ---------------------------------------------------------------------------


async def test_a_key_is_stored_without_a_session(app_env) -> None:
    body = _add_body(str(Path.home()), name="keyed", env={"API_TOKEN": "${API_TOKEN}"})
    await app_env.client.post("/v1/desktop/mcp", json=body)

    response = await app_env.client.post(
        "/v1/desktop/mcp/credentials",
        json={
            "name": "keyed",
            "values": {"API_TOKEN": "placeholder-value"},
            "cwd": str(Path.home()),
        },
    )

    assert response.status_code == 200
    data = _data(response)
    assert data["name"] == "keyed"
    assert data["saved_ids"] == ["API_TOKEN"]
    assert data["failed_ids"] == []
    assert data["code"] == "saved"
    assert _row(data["catalog"], "keyed")["auth"]["secret_refs"] == [
        {"id": "API_TOKEN", "state": "encrypted"}
    ]
    assert _row(data["catalog"], "keyed")["status"] == "not_started"


async def test_a_key_for_an_unknown_server_stores_nothing(app_env) -> None:
    response = await app_env.client.post(
        "/v1/desktop/mcp/credentials",
        json={"name": "absent", "values": {"API_TOKEN": "placeholder-value"}},
    )

    assert response.status_code == 200
    data = _data(response)
    assert data["saved_ids"] == []
    assert data["code"] == "invalid_target"
    assert data["catalog"]["servers"] == []


# ---------------------------------------------------------------------------
# the live overlay
# ---------------------------------------------------------------------------


async def test_a_cold_session_is_not_read_and_binds_nothing(app_env) -> None:
    """The overlay is an enrichment: no runtime is started to enrich a list."""
    project = app_env.tmp / "project"
    project.mkdir()
    remote = FakeRemote(str(project), cold=True)
    app_env.app.state.desktop_sessions = FakePool({"8fd6c6a40934": remote})

    document = _data(
        await app_env.client.get(
            "/v1/desktop/mcp",
            params={"cwd": str(project), "session_id": "8fd6c6a40934"},
        )
    )

    assert document["status_source"] == "config"
    assert document["session_id"] is None
    assert remote.binds == 0, "a list bound a runtime"
    assert remote.slashes == [], "a cold session has no runtime to ask"


async def test_a_warm_session_in_the_same_folder_overlays_its_statuses(app_env) -> None:
    project = app_env.tmp / "project"
    project.mkdir()
    await app_env.client.post(
        "/v1/desktop/mcp",
        json={
            "action": "add",
            "name": "echoer",
            "scope": "project",
            "cwd": str(project),
            "command": sys.executable,
            "args": [],
        },
    )
    remote = FakeRemote(
        str(project),
        rows=[
            {
                "name": "echoer",
                "status": "connected",
                "tool_count": 4,
                "loaded": True,
                "startup_failure": None,
            }
        ],
    )
    app_env.app.state.desktop_sessions = FakePool({"8fd6c6a40934": remote})

    document = _data(
        await app_env.client.get(
            "/v1/desktop/mcp", params={"cwd": str(project), "session_id": "8fd6c6a40934"}
        )
    )

    assert document["status_source"] == "live"
    assert document["session_id"] == "8fd6c6a40934"
    row = _row(document, "echoer")
    assert (row["status"], row["status_basis"]) == ("connected", "live")
    assert (row["tool_count"], row["tool_count_basis"]) == (4, "live")
    assert row["actions"] == ["test", "remove", "disconnect"]
    assert remote.binds == 0, "the overlay must not bind either"


async def test_a_warm_session_with_no_loaded_rows_is_not_a_live_document(app_env) -> None:
    """An overlay that reached no row must not stamp the document ``live``.

    A warm runtime that has loaded none of this folder's servers (they were
    added after the conversation started) contributes nothing a row can report,
    so every status in the answer comes from config. Reporting ``live`` there —
    and echoing the id — would tell a client its statuses are a runtime's, and
    set ``session_id`` on a document with no live fact in it.
    """
    project = app_env.tmp / "project"
    project.mkdir()
    await app_env.client.post(
        "/v1/desktop/mcp",
        json={
            "action": "add",
            "name": "echoer",
            "scope": "project",
            "cwd": str(project),
            "command": sys.executable,
            "args": [],
        },
    )
    # The runtime knows the folder but has loaded no server (`loaded: false` is
    # what the legacy snapshot says for one added after the conversation began).
    remote = FakeRemote(
        str(project),
        rows=[{"name": "echoer", "status": "cold", "tool_count": 0, "loaded": False}],
    )
    app_env.app.state.desktop_sessions = FakePool({"8fd6c6a40934": remote})

    document = _data(
        await app_env.client.get(
            "/v1/desktop/mcp", params={"cwd": str(project), "session_id": "8fd6c6a40934"}
        )
    )

    assert document["status_source"] == "config"
    assert document["session_id"] is None
    row = _row(document, "echoer")
    assert (row["status"], row["status_basis"]) == ("not_started", "stored")


async def test_a_warm_session_in_another_folder_is_not_an_overlay(app_env) -> None:
    """Its "connected" is about a different config set, so it must not be shown."""
    project = app_env.tmp / "project"
    project.mkdir()
    elsewhere = app_env.tmp / "elsewhere"
    elsewhere.mkdir()
    remote = FakeRemote(str(elsewhere), rows=[{"name": "x", "status": "connected"}])
    app_env.app.state.desktop_sessions = FakePool({"8fd6c6a40934": remote})

    document = _data(
        await app_env.client.get(
            "/v1/desktop/mcp", params={"cwd": str(project), "session_id": "8fd6c6a40934"}
        )
    )

    assert document["status_source"] == "config"
    assert document["session_id"] is None


async def test_an_unknown_session_degrades_to_the_config_answer(app_env) -> None:
    response = await app_env.client.get("/v1/desktop/mcp", params={"session_id": "8fd6c6a40934"})

    assert response.status_code == 200
    assert _data(response)["status_source"] == "config"


# ---------------------------------------------------------------------------
# the retiring daemon
# ---------------------------------------------------------------------------


async def test_a_latched_daemon_refuses_new_work_but_keeps_answering_the_list(
    app_env,
) -> None:
    setattr(app_env.app.state, RETIRING_STATE_ATTR, True)

    refused = await app_env.client.post("/v1/desktop/mcp", json=_add_body(str(Path.home())))
    assert refused.status_code == 503
    assert refused.json()["detail"]["code"] == "daemon-retiring"

    listed = await app_env.client.get("/v1/desktop/mcp")
    assert listed.status_code == 200
    assert _data(listed)["servers"] == []
