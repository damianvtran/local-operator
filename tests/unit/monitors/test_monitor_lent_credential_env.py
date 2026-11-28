"""A monitor's ``gh`` tick reaches the mesh-lent github login, and never leaks it.

THE INCIDENT. On a mesh node that borrows its github login from the owner
device, a monitor watching ``gh pr view <n> --json state`` failed with "gh: To
use GitHub CLI in automation, set the GH_TOKEN environment variable." while the
same command through the session's ``bash`` worked. Both run ``execute_bash``;
the difference was the context. The parked-session probe
(``monitors/parked.py``, the subprocess the wake supervisor runs so a parked
session stays parked on quiet ticks) built its ``ToolContext`` with no
``variables`` store, and bash hands a child the lent ``GH_TOKEN`` only when the
context carries a ledger that can scrub it — so every parked tick ran ``gh``
unauthenticated, failed, and engaged the session it was meant to leave parked.

The lent env now has ONE door, ``github.lent_github_env``, taken by the bash
spawn path (hence every monitor tick) and the code-request resolver alike.
These cells pin both monitor paths end to end with the real bash tool and a
fake ``gh`` on PATH (the real borrow is stubbed at ``borrowed_git_env``, the
seam below the shared door), plus the door's own contract.

``gh auth status`` itself is not watchable (the read-only gate refuses ``gh
auth``), so the fake answers the watchable ``gh pr view`` the incident ran,
reporting auth the way ``gh auth status`` would.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import (
    AbortSignal,
    ModelSpec,
    StreamEndEvent,
    ToolContext,
)
from local_operator.monitors import parked
from local_operator.monitors import state as monitor_state
from local_operator.monitors import store as monitor_store
from local_operator.monitors.spec import MonitorSpec
from local_operator.network.credentials import github as github_mod
from local_operator.session.session import Session, _monitor_comparable_text
from local_operator.session.transcript import Transcript
from local_operator.tools.builtin import build_bash_tool
from local_operator.variables import VariableStore
from tests.unit.monitors.support import drain_checks, seed_session_baseline

SID = "lentenv00001"
NOW = 1_800_000_000_000
COMMAND = "gh pr view 3951 --json state"
LOGGED_IN = "Logged in to github.com (GH_TOKEN)"
GH_ERROR = "gh: To use GitHub CLI in automation, set the GH_TOKEN environment variable."
MODEL = ModelSpec(provider="test", model_id="m", context_window=100_000)

#: The fake ``gh``: the real one's answer with and without a token in its env.
#: It ALSO prints the token (as ``gh auth status --show-token`` would), so the
#: no-leak assertions exercise the masking rather than a value never printed.
_FAKE_GH = f"""#!/bin/sh
if [ -n "$GH_TOKEN" ]; then
  echo "{LOGGED_IN}"
  echo "token: $GH_TOKEN"
  echo '{{"state":"OPEN"}}'
else
  echo "{GH_ERROR}" >&2
  exit 4
fi
"""


@pytest.fixture
def lent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """A stubbed lent token, a fake ``gh`` first on PATH, no ambient token."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    fake = bindir / "gh"
    fake.write_text(_FAKE_GH, encoding="utf-8")
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}")
    # A CI runner (or this very harness) may carry a real token: the child must
    # see only what the monitor path injects, or the test proves nothing.
    for name in ("GH_TOKEN", "GITHUB_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    token = f"ghs_lentmonitor_{os.urandom(6).hex()}"
    injected = github_mod.git_env_for_token(token)
    monkeypatch.setattr(github_mod, "borrowed_git_env", lambda **_kwargs: (dict(injected), token))
    config = tmp_path / "cfg"
    config.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    return token


def _tree_text(root: Path) -> str:
    """Every byte the monitor layer persisted under ``root``, as text."""
    return "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in sorted(root.rglob("*"))
        if path.is_file()
    )


@pytest.mark.asyncio
async def test_a_parked_tick_reaches_the_lent_login_and_stays_quiet(
    tmp_path: Path, lent: str
) -> None:
    """The incident's path: the parked probe's ``gh`` tick is authenticated.

    Baseline = the logged-in answer, hashed through the same pipeline the
    session uses, so a probe whose ``gh`` ran unauthenticated reads "check
    failed" and ENGAGES (the pre-fix behaviour) while the fixed probe reads the
    unchanged answer and stays QUIET.
    """
    root = tmp_path / "cfg"
    row: dict[str, Any] = {
        "id": "m1",
        "name": "pr-state",
        "tool": "bash",
        "arguments": {"command": COMMAND},
        "every_ms": 60_000,
        "until_at": None,
        "cwd": str(tmp_path),
        "created_at": 1,
        "next_due_at": NOW - 1,
        "disabled": False,
    }
    spec = MonitorSpec.model_validate({k: row[k] for k in MonitorSpec.model_fields if k in row})
    # The baseline as the in-session scheduler would have stored it: a context
    # WITH a ledger, the shape every real session's context has.
    reference = await build_bash_tool(None).execute(
        "baseline",
        {"command": COMMAND},
        AbortSignal(),
        None,
        ToolContext(cwd=str(tmp_path), session_id=SID, variables=VariableStore(cwd=str(tmp_path))),
    )
    assert LOGGED_IN in reference.text, reference.text
    assert lent not in reference.text, "the baseline text leaked the lent token"
    comparable, _blind, _handle = _monitor_comparable_text(reference.text, reference.details)
    monitor_store.write_entry(
        root, SID, cwd=str(tmp_path), monitors=[row], preserve={"parked_at": NOW - 10}
    )
    # The (hash, blob) pair the session's scheduler commits for this baseline.
    seed_session_baseline(
        root,
        SID,
        spec,
        str(comparable),
        {"schema": monitor_state.STATE_SCHEMA, "checks": 1, "next_due_at": NOW - 1},
    )

    outcome = await parked.probe_session(root, SID, now_ms=NOW)

    assert outcome == {"verdict": "quiet", "why": "unchanged", "checked": 1}, outcome
    counters = monitor_state.read_counters(root, SID, "m1") or {}
    assert counters.get("last_check_at") == NOW, counters
    assert lent not in _tree_text(root), "the parked tick persisted the lent token"


@pytest.mark.asyncio
async def test_a_session_monitor_reports_logged_in_and_never_stores_the_token(
    tmp_path: Path, lent: str
) -> None:
    """The in-session path: the armed baseline is the logged-in answer, masked."""
    root = tmp_path / "cfg"
    session = Session(
        model=MODEL,
        stream_fn=lambda request, signal: _end(),
        tools=[build_bash_tool()],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: [],
        cwd=str(tmp_path),
        # The ledger every real session carries (``session_factory``); without
        # one bash withholds the lent env by design.
        variables=VariableStore(cwd=str(tmp_path)),
    )
    clock = [NOW]
    session._monitors._now = lambda: clock[0]  # type: ignore[method-assign]
    try:
        created = await session._monitors.create(
            {"name": "pr-state", "tool": "bash", "arguments": {"command": COMMAND}, "every": "60s"},
            cwd=str(tmp_path),
        )
        assert created.get("created"), created
        entry = next(iter(session._monitors._entries.values()))
        clock[0] = int(entry.counters["next_due_at"]) + 1
        await session._monitors.pump(now_ms=clock[0] + 1)
        await drain_checks(session._monitors, session)

        counters = next(iter(session._monitors._entries.values())).counters
        assert not counters.get("last_error"), counters
        assert not counters.get("failing"), counters
        snapshot = monitor_state.read_snapshot(root, session._session_id, "m1") or {}
        stored = str(snapshot.get("snapshot") or "")
        assert LOGGED_IN in stored, stored
        assert GH_ERROR not in stored, stored
        assert lent not in _tree_text(root), "the session monitor persisted the lent token"
    finally:
        await session.dispose()


async def _end():  # noqa: ANN202 — the stream_fn's async generator
    yield StreamEndEvent(stop_reason="stop")
