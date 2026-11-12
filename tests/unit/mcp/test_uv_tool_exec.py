"""Bare ``uvx <tool>`` resolves to the installed tool binary; everything else keeps uvx.

The fake ``uv`` here is a shell script that answers ``uv tool dir`` with a
temp directory, so no real uv, cache or lock is involved.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from local_operator.mcp import uv_tool_exec
from local_operator.mcp.uv_tool_exec import resolve_uv_tool_command

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX tool layout")


def _exe(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(0o755)
    return path


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(uv_tool_exec.OPT_OUT_ENV, raising=False)
    uv_tool_exec.clear_tool_dir_cache()
    yield
    uv_tool_exec.clear_tool_dir_cache()


@pytest.fixture
def rig(tmp_path: Path):
    tools = tmp_path / "tools"
    bindir = tmp_path / "bin"
    _exe(
        bindir / "uv", f'#!/bin/sh\n[ "$1 $2" = "tool dir" ] && echo "{tools}" && exit 0\nexit 2\n'
    )
    _exe(bindir / "uvx", "#!/bin/sh\nexit 3\n")
    binary = _exe(tools / "workspace-mcp" / "bin" / "workspace-mcp", "#!/bin/sh\n")
    env = {"PATH": str(bindir), "HOME": str(tmp_path)}
    return {"tools": tools, "bindir": bindir, "binary": binary, "env": env}


def test_bare_uvx_name_resolves_to_installed_binary(rig):
    got = resolve_uv_tool_command(
        "uvx", ["workspace-mcp", "--tool-tier", "core"], env=rig["env"], server_env={}
    )
    assert got == (str(rig["binary"]), ["--tool-tier", "core"])


def test_absolute_uvx_path_resolves(rig):
    got = resolve_uv_tool_command(
        str(rig["bindir"] / "uvx"), ["workspace-mcp"], env=rig["env"], server_env={}
    )
    assert got == (str(rig["binary"]), [])


@pytest.mark.parametrize("sub", ["run", "uvx"])
def test_uv_tool_run_and_uvx_forms(rig, sub):
    got = resolve_uv_tool_command(
        "uv", ["tool", sub, "workspace-mcp", "-x"], env=rig["env"], server_env={}
    )
    assert got == (str(rig["binary"]), ["-x"])


@pytest.mark.parametrize(
    "command,args",
    [
        ("uvx", ["workspace-mcp==1.2.3"]),
        ("uvx", ["workspace-mcp@latest"]),
        ("uvx", ["workspace-mcp>=1"]),
        ("uvx", ["workspace-mcp[extra]"]),
        ("uvx", ["--from", "workspace-mcp", "workspace-mcp"]),
        ("uvx", ["--with", "httpx", "workspace-mcp"]),
        ("uvx", ["--python", "3.12", "workspace-mcp"]),
        ("uvx", ["-q", "workspace-mcp"]),
        ("uvx", ["Workspace_MCP"]),
        ("uvx", ["./workspace-mcp"]),
        ("uvx", []),
        ("uv", ["run", "workspace-mcp"]),
        ("uv", ["--quiet", "tool", "run", "workspace-mcp"]),
        ("uv", ["tool", "run", "--from", "x", "workspace-mcp"]),
        ("npx", ["workspace-mcp"]),
        ("workspace-mcp", []),
    ],
)
def test_non_bare_or_non_uvx_falls_back(rig, command, args):
    assert resolve_uv_tool_command(command, args, env=rig["env"], server_env={}) is None


def test_missing_binary_falls_back(rig):
    assert resolve_uv_tool_command("uvx", ["other-mcp"], env=rig["env"], server_env={}) is None


def test_non_executable_binary_falls_back(rig):
    rig["binary"].chmod(0o644)
    assert resolve_uv_tool_command("uvx", ["workspace-mcp"], env=rig["env"], server_env={}) is None


def test_server_uv_env_override_falls_back(rig):
    got = resolve_uv_tool_command(
        "uvx", ["workspace-mcp"], env=rig["env"], server_env={"UV_PYTHON": "3.11"}
    )
    assert got is None


def test_uv_tool_dir_failure_falls_back(rig):
    _exe(rig["bindir"] / "uv", "#!/bin/sh\necho boom >&2\nexit 1\n")
    assert resolve_uv_tool_command("uvx", ["workspace-mcp"], env=rig["env"], server_env={}) is None


def test_uv_tool_dir_timeout_falls_back(rig, monkeypatch):
    def _timeout(*a, **k):
        raise subprocess.TimeoutExpired(cmd="uv tool dir", timeout=k.get("timeout", 0))

    monkeypatch.setattr(uv_tool_exec.subprocess, "run", _timeout)
    assert resolve_uv_tool_command("uvx", ["workspace-mcp"], env=rig["env"], server_env={}) is None


def test_uv_missing_falls_back(rig, tmp_path):
    env = dict(rig["env"], PATH=str(tmp_path / "empty"))
    assert resolve_uv_tool_command("uvx", ["workspace-mcp"], env=env, server_env={}) is None


def test_opt_out_env_keeps_uvx(rig, monkeypatch):
    monkeypatch.setenv(uv_tool_exec.OPT_OUT_ENV, "0")
    assert resolve_uv_tool_command("uvx", ["workspace-mcp"], env=rig["env"], server_env={}) is None


def test_tool_dir_is_queried_once_then_cached(rig, tmp_path):
    counter = tmp_path / "calls"
    _exe(
        rig["bindir"] / "uv",
        f'#!/bin/sh\necho x >> "{counter}"\necho "{rig["tools"]}"\n',
    )
    for _ in range(3):
        assert resolve_uv_tool_command("uvx", ["workspace-mcp"], env=rig["env"], server_env={})
    assert counter.read_text().count("x") == 1


def test_stdio_transport_spawns_resolved_binary(rig, monkeypatch):
    """The real spawn site hands ``open_process`` the binary, args unchanged."""
    import anyio

    from local_operator.mcp import manager
    from local_operator.mcp.config import MCPStdioServerConfig

    seen: dict[str, object] = {}

    async def _fake_open_process(target, **kwargs):
        seen["target"] = target
        seen["cwd"] = kwargs.get("cwd")
        seen["env_keys"] = set(kwargs["env"])
        raise RuntimeError("stop after spawn")

    monkeypatch.setattr(anyio, "open_process", _fake_open_process)
    monkeypatch.setenv("PATH", rig["env"]["PATH"])
    cfg = MCPStdioServerConfig(
        command="uvx", args=["workspace-mcp", "--a"], env={"FOO": "1"}, cwd=str(rig["tools"])
    )

    async def _go():
        stderr_log = manager.McpServerStderr("srv")
        async with manager._stdio_transport(cfg, lambda: None, stderr_log):
            pass

    with pytest.raises(RuntimeError, match="stop after spawn"):
        anyio.run(_go)
    assert seen["target"] == [str(rig["binary"]), "--a"]
    assert seen["cwd"] == str(rig["tools"])
    assert "FOO" in seen["env_keys"]  # type: ignore[operator]
    assert os.access(rig["binary"], os.X_OK)
