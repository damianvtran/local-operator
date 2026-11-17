"""The guard's own ``gh`` lookup authenticates as the session's shell does.

Field report 2026-11-17 (lop-remote-node-2, guard 0.106.6, session
d5188ee174d4): every guard lookup read "PR state unknown (gh lookup on
damianvtran/local-operator failed or timed out)" while ``gh pr list --head
<branch>`` in the agent's shell answered at once (#3963 MERGED). node-2 borrows
the operator's github login over the mesh; ``execute_bash`` injects it (and the
session credential store's ``credential_env()``) into the bash child only, and
the guard's ``gh`` ran on the bare harness environment, unauthenticated.

These drive the real ``_gh_prs`` against a fake ``gh`` on PATH that answers only
with the session's ``GH_CONFIG_DIR``/``GH_TOKEN``, the real thread-pool
propagation and ``check_tool_call`` wiring, and the real provider
``bash_guard_forge_auth`` over a fake store and a stubbed mesh borrow. Every
token is synthetic.
"""

from __future__ import annotations

import json
import os
import stat
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.guardrails import check as guard_check
from local_operator.guardrails import classify
from local_operator.guardrails import provenance as prov

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell stub")

TOKEN = "synthetic-session-token"  # not a credential: the stub compares it
HEAD = "a" * 40

_GH = """#!/bin/sh
# Fake gh: answers only when the session's auth reached it, like a real gh
# with no login of its own (the node-2 shape).
# exec: the timeout kill must reach the sleeper, or it outlives the test.
if [ -n "$FAKE_GH_SLEEP" ]; then exec sleep "$FAKE_GH_SLEEP"; fi
if [ "$GH_TOKEN" != "{token}" ] || [ "$GH_CONFIG_DIR" != "{config}" ]; then
  echo "To use GitHub CLI in automation, set the GH_TOKEN environment variable." >&2
  exit 4
fi
if [ "$GH_PROMPT_DISABLED" != "1" ]; then echo "prompt not disabled" >&2; exit 5; fi
printf '%s\\n' '{rows}'
"""


@pytest.fixture
def fake_gh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    config = str(tmp_path / "session-gh-config")
    rows = json.dumps(
        [
            {
                "number": 3963,
                "state": "MERGED",
                "mergedAt": "2026-11-17T01:02:03Z",
                "closedAt": "2026-11-17T01:02:03Z",
                "updatedAt": "2026-11-17T01:02:03Z",
                "headRefOid": HEAD,
            }
        ]
    )
    gh = bin_dir / "gh"
    gh.write_text(_GH.format(token=TOKEN, config=config, rows=rows))
    gh.chmod(gh.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    # The harness environment: no gh auth at all (node-2).
    for name in ("GH_TOKEN", "GITHUB_TOKEN", "GH_CONFIG_DIR", "FAKE_GH_SLEEP"):
        monkeypatch.delenv(name, raising=False)
    return {"GH_TOKEN": TOKEN, "GH_CONFIG_DIR": config}


def _with_auth(provider: Any, fn: Any) -> Any:
    token = prov.FORGE_AUTH.set(provider)
    try:
        return fn()
    finally:
        prov.FORGE_AUTH.reset(token)


def test_without_the_session_auth_gh_reads_unknown(fake_gh: dict[str, str]) -> None:
    """The field failure, reproduced: bare harness env -> ``None`` -> unknown."""
    assert prov._gh_prs("damianvtran/local-operator", "fix/x") is None


def test_with_the_session_auth_gh_answers(fake_gh: dict[str, str]) -> None:
    got = _with_auth(lambda: fake_gh, lambda: prov._gh_prs("damianvtran/local-operator", "fix/x"))
    assert got == [prov.ForgeChange(3963, "merged", "2026-11-17", HEAD)]
    assert _with_auth(
        lambda: fake_gh, lambda: prov._gh_open_prs("damianvtran/local-operator", "fix/x")
    ) == [3963]
    # The harness environment itself is left untouched.
    assert "GH_TOKEN" not in os.environ


def test_gh_failure_timeout_or_broken_provider_stays_unknown(
    fake_gh: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    wrong = dict(fake_gh, GH_TOKEN="not-the-session-token")
    assert _with_auth(lambda: wrong, lambda: prov._gh_prs("o/r", "b")) is None

    def boom() -> dict[str, str]:
        raise RuntimeError("broker down")

    assert _with_auth(boom, lambda: prov._gh_prs("o/r", "b")) is None
    monkeypatch.setattr(prov, "_GH_TIMEOUT_S", 0.3)
    monkeypatch.setenv("FAKE_GH_SLEEP", "3")
    started = time.monotonic()
    assert _with_auth(lambda: fake_gh, lambda: prov._gh_prs("o/r", "b")) is None
    assert time.monotonic() - started < 2.5


def test_the_auth_reaches_the_reach_lookup_pool(fake_gh: dict[str, str]) -> None:
    """``_bounded_lookups`` runs ``worktree_activity`` on a ThreadPoolExecutor,
    which does not copy context variables."""
    jobs = {"prs": lambda: prov._gh_prs("o/r", "b")}
    got = _with_auth(
        lambda: fake_gh, lambda: classify._bounded_lookups(jobs, time.monotonic() + 10)
    )
    assert got["prs"] == [prov.ForgeChange(3963, "merged", "2026-11-17", HEAD)]
    assert prov.FORGE_AUTH.get() is None


@pytest.mark.asyncio
async def test_check_tool_call_installs_the_context_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[Any] = []

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen.append(prov.FORGE_AUTH.get())
        raise RuntimeError("stop here")  # check_tool_call fails closed; fine for the spy

    def provider() -> dict[str, str]:
        return {"GH_TOKEN": TOKEN}

    monkeypatch.setattr(guard_check, "classify_bash", spy)
    ctx = guard_check.CallContext(cwd="/", env={}, scratchpad=None, forge_auth=provider)
    await guard_check.check_tool_call(
        "bash", {"command": "rm -rf /tmp/x"}, ctx, judge=None, settings=guard_check.GuardSettings()
    )
    assert seen == [provider]
    assert prov.FORGE_AUTH.get() is None


class _Store:
    def __init__(self) -> None:
        self.registered: list[str] = []

    def credential_env(self) -> dict[str, str]:
        return {"GH_CONFIG_DIR": "/s/gh", "GH_TOKEN": "store-token", "OPENAI_API_KEY": "nope"}

    def register_redaction(self, value: str) -> None:
        self.registered.append(value)


def test_bash_guard_forge_auth_merges_what_bash_injects(monkeypatch: pytest.MonkeyPatch) -> None:
    from local_operator.network.credentials import github
    from local_operator.tools.builtin import bash_guard_forge_auth

    calls: list[dict[str, Any]] = []

    def lent(**kwargs: Any) -> tuple[dict[str, str], str]:
        calls.append(kwargs)
        kwargs["register"]("lent-token")  # the contract: registered before returned
        return {"GH_TOKEN": "lent-token", "GIT_CONFIG_COUNT": "1"}, "lent-token"

    monkeypatch.setattr(github, "lent_github_env", lent)
    store = _Store()
    context = SimpleNamespace(session_id="sid0000001", variables=store)
    provider = bash_guard_forge_auth(context)  # type: ignore[arg-type]
    assert provider is not None
    assert calls == []  # lazy: no borrow until a forge lookup asks
    # Same order as execute_bash (store first, then the borrow); forge names only.
    assert provider() == {"GH_CONFIG_DIR": "/s/gh", "GH_TOKEN": "lent-token"}
    assert provider() == {"GH_CONFIG_DIR": "/s/gh", "GH_TOKEN": "lent-token"}
    assert len(calls) == 1 and calls[0]["session_id"] == "sid0000001"
    assert store.registered == ["lent-token"]
    assert bash_guard_forge_auth(None) is None
