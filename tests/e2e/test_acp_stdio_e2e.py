"""``lop acp`` as a real subprocess, driven by the ACP SDK's own client.

The assembled path: the installed console script, ``claim_stdio``, the CLI
dispatch, the SDK session on ``--hosting test`` (the mock provider), the
approval bridge and the event mapping, all over real pipes. The mock's
markers drive it (``providers/clients.py`` ``MockClient``): plain text gives
two deltas and ``end_turn``; ``[bash:N]`` makes the model call the REAL
``bash`` tool with ``sleep N``, which is exec-tier and therefore raises a
permission request.

Isolation: its own ``HOME`` and config dir, every ``CMUX_*``/``LOP_*``
variable stripped (an inherited ``CMUX_WORKSPACE_ID`` once let a headless rig
rename the operator's real workspaces), notifications off through
``harness_child_env``.
"""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

acp = pytest.importorskip("acp")

from acp.schema import AllowedOutcome, RequestPermissionResponse  # noqa: E402

pytestmark = pytest.mark.e2e

#: An upper bound on an awaited event, never a sleep. Generous: the child pays
#: the CLI's whole cold import and a session build, 10-30 s on a loaded host.
GUARD_S = 180.0


def _lop_command() -> list[str]:
    """The installed console script beside this interpreter, else ``-m``."""
    script = Path(sys.executable).with_name("lop")
    if script.exists():
        return [str(script), "acp"]
    return [sys.executable, "-m", "local_operator.cli", "acp"]


def _child_env(home: Path) -> dict[str, str]:
    from local_operator.agent_shell import harness_child_env

    env = {
        key: value
        for key, value in harness_child_env().items()
        if not key.startswith(("CMUX_", "LOP_")) and key != "NO_COLOR"
    }
    env["HOME"] = str(home)
    env["LOCAL_OPERATOR_CONFIG_DIR"] = str(home / ".local-operator")
    env["LOCAL_OPERATOR_HOME"] = str(home / "local-operator-home")
    env["TERM"] = "xterm-256color"
    return env


def _configure(config_dir: Path) -> None:
    from local_operator.config import ConfigManager

    config_dir.mkdir(parents=True, exist_ok=True)
    # The classification probe is off for the same reason the exec e2e turns
    # it off: it would add model calls the assertions below do not expect.
    ConfigManager(config_dir=config_dir).update_config(
        {"hosting": "test", "model_name": "mock", "classification": {"auto": False}}
    )


class Client:
    def __init__(self) -> None:
        self.updates: list[Any] = []
        self.permissions: list[Any] = []
        self.reply = "allow_once"
        self.asked = asyncio.Event()

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        self.updates.append(update)

    async def request_permission(
        self, session_id: str, tool_call: Any, options: list[Any], **kwargs: Any
    ) -> RequestPermissionResponse:
        self.permissions.append(tool_call)
        self.asked.set()
        return RequestPermissionResponse(
            outcome=AllowedOutcome(outcome="selected", option_id=self.reply)
        )


@asynccontextmanager
async def agent(tmp_path: Path) -> AsyncIterator[tuple[Any, Client, Any, Path]]:
    home = tmp_path / "home"
    work = home / "work"
    work.mkdir(parents=True)
    _configure(home / ".local-operator")
    client = Client()
    command, *args = _lop_command()
    async with acp.spawn_agent_process(client, command, *args, env=_child_env(home), cwd=work) as (
        conn,
        process,
    ):
        yield conn, client, process, work


async def _session(conn: Any, work: Path) -> str:
    await asyncio.wait_for(conn.initialize(protocol_version=acp.PROTOCOL_VERSION), GUARD_S)
    response = await asyncio.wait_for(conn.new_session(cwd=str(work), mcp_servers=[]), GUARD_S)
    return response.session_id


@pytest.mark.asyncio
async def test_initialize_advertises_http_mcp_and_a_turn_streams(tmp_path: Path) -> None:
    async with agent(tmp_path) as (conn, client, process, work):
        init = await asyncio.wait_for(
            conn.initialize(protocol_version=acp.PROTOCOL_VERSION), GUARD_S
        )
        # D8 (revised): the live Gateway refuses a backend without this.
        assert init.agent_capabilities.mcp_capabilities.http is True
        assert init.agent_capabilities.load_session is False
        session_id = (
            await asyncio.wait_for(conn.new_session(cwd=str(work), mcp_servers=[]), GUARD_S)
        ).session_id
        sessions_dir = tmp_path / "home" / ".local-operator" / "sessions"
        assert (sessions_dir / session_id).is_dir(), "the ACP id IS the lop session id"
        response = await asyncio.wait_for(
            conn.prompt(session_id=session_id, prompt=[acp.text_block("hi")]), GUARD_S
        )
        assert response.stop_reason == "end_turn"
        text = "".join(
            u.content.text for u in client.updates if u.session_update == "agent_message_chunk"
        )
        assert text == "Hello from the mock provider!"
    # The context exit closed stdin; the agent must exit on its own, cleanly.
    assert process.returncode == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reply", "status"), [("allow_once", "completed"), ("reject_once", "failed")]
)
async def test_bash_raises_a_permission_request(tmp_path: Path, reply: str, status: str) -> None:
    async with agent(tmp_path) as (conn, client, process, work):
        client.reply = reply
        session_id = await _session(conn, work)
        response = await asyncio.wait_for(
            conn.prompt(session_id=session_id, prompt=[acp.text_block("go [bash:1]")]),
            GUARD_S,
        )
        assert response.stop_reason == "end_turn"
        [asked] = client.permissions
        starts = [u for u in client.updates if u.session_update == "tool_call"]
        assert [s.tool_call_id for s in starts] == [asked.tool_call_id]
        assert starts[0].kind == "execute"
        ends = [
            u
            for u in client.updates
            if u.session_update == "tool_call_update" and u.status is not None
        ]
        assert ends[-1].tool_call_id == asked.tool_call_id
        assert ends[-1].status == status
    assert process.returncode == 0


@pytest.mark.asyncio
async def test_cancel_during_a_running_tool_returns_cancelled(tmp_path: Path) -> None:
    async with agent(tmp_path) as (conn, client, process, work):
        session_id = await _session(conn, work)
        turn = asyncio.ensure_future(
            conn.prompt(session_id=session_id, prompt=[acp.text_block("wait [bash:30]")])
        )
        await asyncio.wait_for(client.asked.wait(), GUARD_S)
        # The allow is on the wire; give the sleep a moment to start, then stop.
        await asyncio.sleep(0.5)
        await conn.cancel(session_id=session_id)
        response = await asyncio.wait_for(turn, 60)
        assert response.stop_reason == "cancelled"
    assert process.returncode == 0


def test_the_child_environment_carries_no_cmux_or_lop_variables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rig's own isolation claim, checked rather than assumed."""
    monkeypatch.setenv("CMUX_WORKSPACE_ID", "synthetic-workspace-0000")
    monkeypatch.setenv("LOP_MOBILE_CHILD_SESSION", "synthetic")
    env = _child_env(tmp_path)
    assert not [k for k in env if k.startswith(("CMUX_", "LOP_"))]
    assert env["HOME"] == str(tmp_path) and os.environ["HOME"] != env["HOME"]


@pytest.mark.parametrize("flags", [["--yolo"], ["--resume", "synthetic-0000"]])
def test_flags_acp_cannot_honour_are_refused(tmp_path: Path, flags: list[str]) -> None:
    """QA Q3 through the real binary: refused at the parse, stdout untouched."""
    import subprocess

    home = tmp_path / "home"
    (home / "work").mkdir(parents=True)
    _configure(home / ".local-operator")
    result = subprocess.run(
        [*_lop_command(), *flags],
        env=_child_env(home),
        cwd=home / "work",
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=GUARD_S,
    )
    assert result.returncode == 2
    assert result.stdout == ""
    assert f"lop acp does not take {flags[0]}" in result.stderr
