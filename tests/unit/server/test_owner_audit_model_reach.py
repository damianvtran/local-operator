"""Model-run code cannot reach the human-only owner-audit routes (#3022 QA r1, Q1).

QA's reproduction: in the default ``shell_environment.mode=inherit`` the
daemon's ``LOCAL_OPERATOR_DESKTOP_TOKEN`` was in every session runtime's
environment, so a model-authored ``curl -H "Authorization: Bearer
$LOCAL_OPERATOR_DESKTOP_TOKEN"`` against ``/owner-audit/.../messages`` got 200
with bodies the agent's screening REJECTED. These tests drive the REAL ``bash``
executor and the REAL ``eval`` kernel against the PR's own loopback desktop
server and stub hub, with the daemon's launch environment carrying the token,
and pin the three places it is now withheld:

1. every model child's environment (``tools.shell_env``), in BOTH modes;
2. the session runtime's launch environment, which a child reads back with
   ``ps eww $PPID`` (``session.runtime.launch``);
3. the daemon's own environment. That half is ``server.desktop.adopt_spawn_bearer``
   and its own suite (``tests/unit/server/test_desktop_bearer_channel.py``)
   covers it, including the re-exec onto a pipe that keeps the value out of the
   exec-time block ``ps eww <pid>`` reads.

Each refusal test has a positive control: the same request WITH the bearer
answers 200, so a 401 here means the bearer was missing, not that the rig broke.
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path

import pytest
import pytest_asyncio

from local_operator import agent_shell
from local_operator.harness.types import AbortSignal, ToolContext
from local_operator.server import desktop
from local_operator.tools import builtin
from local_operator.tools import eval as eval_tool
from local_operator.tools import shell_env
from tests.unit.server.test_desktop_team_bridge_owner_audit import BASE, CID, _q
from tests.unit.server.test_desktop_team_bridge_owner_audit import (  # noqa: F401
    proxy as owner_audit_proxy,
)

DESKTOP_TOKEN = "desktop-token"


@pytest.fixture
def proxy(request):
    """The owner-audit route rig (real loopback desktop server + stub #221 hub)."""

    return request.getfixturevalue(
        "owner_audit_proxy"
    )  # the proxy fixture's LOCAL_OPERATOR_DESKTOP_TOKEN


@pytest.fixture(autouse=True)
def _inherit_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default mode, where Q1 was measured, resolved without a config file."""

    monkeypatch.setattr(shell_env, "load_policy", lambda: shell_env.ShellEnvironmentPolicy())


@pytest_asyncio.fixture
async def clean_kernels():
    """Reap the eval workers this test spawned.

    NOT autouse: an async generator fixture on a test that has no event loop of its
    own (the sync tests in this module) still owes a finalizer, and CI failed the
    shard with ``Event loop is closed`` for every test in the file -- teardown only,
    all six passed (run 40623539454, shard 0). Only the kernel test asks for it.
    """

    yield
    for kernel in list(eval_tool._KERNELS.values()):
        await eval_tool._close_kernel(kernel)
    eval_tool._KERNELS.clear()


def _curl(url: str, token_expr: str) -> str:
    return (
        f'curl -s -o /dev/null -w "%{{http_code}}" '
        f'-H "Authorization: Bearer {token_expr}" {shlex.quote(url)}'
    )


@pytest.mark.asyncio
async def test_the_bash_tool_cannot_use_the_desktop_bearer(proxy, tmp_path: Path) -> None:
    client, calls, _state, _ = proxy
    assert os.environ[desktop.TOKEN_ENV] == DESKTOP_TOKEN  # the daemon's launch env
    url = str(client.base_url) + f"{BASE}/{CID}/messages" + _q()
    context = ToolContext(cwd=str(tmp_path))

    probe = await builtin.execute_bash(
        "q1",
        {
            "command": f"{_curl(url, '$LOCAL_OPERATOR_DESKTOP_TOKEN')}; "
            'echo; echo "NAMES=$(env | grep -c DESKTOP_TOKEN)"'
        },
        AbortSignal(),
        None,
        context,
    )
    assert "401" in probe.text, probe.text  # was 200 before the fix
    assert "NAMES=0" in probe.text, probe.text  # no such name in the child
    assert calls == [], "the hub was reached with the bearer the child should not hold"

    # Positive control: the route and rig are live; only the bearer was missing.
    control = await builtin.execute_bash(
        "q1-control", {"command": _curl(url, DESKTOP_TOKEN)}, AbortSignal(), None, context
    )
    assert "200" in control.text, control.text


@pytest.mark.asyncio
async def test_the_eval_kernel_cannot_use_the_desktop_bearer(
    proxy, tmp_path: Path, clean_kernels: None
) -> None:
    client, calls, _state, _ = proxy
    url = str(client.base_url) + f"{BASE}/{CID}/messages" + _q()
    code = (
        "import os, urllib.request, urllib.error\n"
        f"req = urllib.request.Request({url!r}, headers={{'Authorization': "
        "'Bearer ' + os.environ.get('LOCAL_OPERATOR_DESKTOP_TOKEN', '')})\n"
        "try:\n    print('STATUS', urllib.request.urlopen(req).status)\n"
        "except urllib.error.HTTPError as e:\n    print('STATUS', e.code)\n"
        "print('HAS_TOKEN', 'LOCAL_OPERATOR_DESKTOP_TOKEN' in os.environ)\n"
    )
    tool = eval_tool.build_eval_tool()
    result = await tool.execute(  # type: ignore[operator]
        "q1-eval", {"code": code}, None, None, ToolContext(cwd=str(tmp_path), session_id="q1")
    )
    assert "STATUS 401" in result.text, result.text
    assert "HAS_TOKEN False" in result.text, result.text
    assert calls == []


def test_the_withheld_names_match_the_daemon() -> None:
    assert desktop.TOKEN_ENV in agent_shell.DAEMON_BEARER_ENVS


@pytest.mark.parametrize("mode", [shell_env.MODE_INHERIT, shell_env.MODE_ALLOWLIST])
def test_no_policy_or_injection_hands_a_child_the_bearer(mode: str) -> None:
    policy = shell_env.ShellEnvironmentPolicy(mode=mode, inherit=(desktop.TOKEN_ENV,))
    env = shell_env.child_environment(
        policy,
        parent={"PATH": "/usr/bin", desktop.TOKEN_ENV: "t"},
        injections={desktop.TOKEN_ENV: "t"},
    )
    assert desktop.TOKEN_ENV not in env and env["PATH"] == "/usr/bin"


def test_the_session_runtime_launch_environment_omits_the_bearer(monkeypatch, tmp_path) -> None:
    """The runtime is the model children's PARENT: `ps eww $PPID` reads this."""

    from local_operator.session.runtime import launch

    captured: dict[str, dict[str, str]] = {}

    def spawn(*args, **kwargs):
        captured["env"] = dict(kwargs["env"])
        raise RuntimeError("stop after capturing the environment")

    from local_operator.session.runtime import standby

    monkeypatch.setenv(desktop.TOKEN_ENV, DESKTOP_TOKEN)
    monkeypatch.setattr(standby, "adoption_possible", lambda: False)
    monkeypatch.setattr(launch, "spawn_disclaimed", spawn)
    with pytest.raises(Exception):
        launch._spawn_runtime("0123456789ab", str(tmp_path), defer_materialise=False)
    assert "env" in captured, "the spawn site was not reached"
    assert desktop.TOKEN_ENV not in captured["env"]
    assert captured["env"].get("LOP_MOBILE_CHILD_RESUME") == "0123456789ab"
