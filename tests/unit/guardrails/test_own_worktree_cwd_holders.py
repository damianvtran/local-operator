"""A session's OWN processes inside its own worktree do not make it "busy".

Field case (2026-11-14, lop-remote-node-2, Linux, lop 0.102.5): session
99ffc68d3d05 ran ``git worktree remove --force .worktrees/attr3880-ci`` on a
detached tree it had created. The ledger matched (``e1``) but the guard read
the tree ``busy`` — some process of this uid had its cwd inside — and the
judge refused: "the worktree is busy and has substantial uncommitted
changes". ``busy`` was decided by ``provenance._live_cwd_inside``: ANY process
of the uid with its cwd in the tree, the deleting session's own eval kernel
and bash children included, and the judge saw only the bare word.

Operator directive (verbatim): "the safety guards should never refuse being
able to clean up uncommitted files that look like scratch, safe to delete,
etc. ... this sort of ask should never come up again."

Pinned here, with REAL processes (``sleep`` with its cwd in the tree):

- (a) the field shape with the session's own child inside → LOW for
  ``git worktree remove --force`` and ``rm -rf``;
- (b) the same tree with a foreign process inside (an orphan with no session
  record, or another session's harness child) → HIGH, holder named by pid,
  command and session in the reason AND the judge's facts;
- an unreadable process inventory → HIGH, named as ``busy=unknown`` with why;
- a cached inventory hit whose holder has exited is re-read, not believed.

Every process started here is reaped by exact pid; nothing touches a live
session, the operator's registry or the operator's ledger.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from local_operator.guardrails import provenance as prov
from local_operator.guardrails.check import CallContext, GuardSettings, check_tool_call
from local_operator.guardrails.classify import Risk
from tests.unit.guardrails import test_own_detached_worktree_scratch as field
from tests.unit.guardrails.test_own_detached_worktree_scratch import (
    S2,
    S,
    _attr3880,
    _env,
    _rank,
)

# The field file's fixtures (tmp HOME with an idle forge; the ``attr3880-ci``
# repository), shared by assignment so pytest collects them here too.
home = field.home
repo = field.repo

#: The real implementation, captured before the shared ``home`` fixture swaps
#: it for ``lambda _: False`` (that fixture keeps its synthetic trees idle).
_REAL_LIVE_CWD_INSIDE = prov._live_cwd_inside

FIELD = [
    "cd ~/local-operator && git worktree remove --force .worktrees/attr3880-ci",
    "rm -rf ~/local-operator/.worktrees/attr3880-ci",
]


@pytest.fixture
def live(monkeypatch: pytest.MonkeyPatch, home: Path) -> Iterator[list[int]]:
    """Real cwd inventory, fresh cache; every pid appended here is killed after."""
    monkeypatch.setattr(prov, "_live_cwd_inside", _REAL_LIVE_CWD_INSIDE)
    monkeypatch.setattr(prov, "_cwd_inventory_cache", None)
    pids: list[int] = []
    yield pids
    for pid in pids:
        # A group leader (``start_new_session``) takes its children with it;
        # any other pid has no group of that id (ESRCH) and is killed alone.
        try:
            os.killpg(pid, signal.SIGKILL)
            continue
        except (ProcessLookupError, PermissionError):
            pass
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _own_child_in(tree: Path, pids: list[int]) -> subprocess.Popen[bytes]:
    """A child of THIS process (the harness, in production) with its cwd inside."""
    child = subprocess.Popen(["sleep", "120"], cwd=tree)
    pids.append(child.pid)
    return child


def _orphan_in(tree: Path, pids: list[int]) -> int:
    """A process with its cwd inside whose parent exited: reparented, no record."""
    out = subprocess.run(
        ["bash", "-c", "nohup sleep 120 >/dev/null 2>&1 & echo $!"],
        cwd=tree,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    pid = int(out)
    pids.append(pid)
    return pid


def _wait_seen(tree: Path, pid: int) -> None:
    """Block until the inventory shows a process inside ``tree`` (fork-exec race).

    Only the pre-#3987-remediation surface (``_live_cwd_inside`` and the cache)
    is used, so this file also runs against e9c27eb to prove test (a) red there.
    """
    real = os.path.realpath(tree)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        prov._cwd_inventory_cache = None
        if _REAL_LIVE_CWD_INSIDE(real):
            prov._cwd_inventory_cache = None
            return
        time.sleep(0.05)
    raise AssertionError(f"pid {pid} never showed a cwd inside {tree}")


@pytest.mark.parametrize("command", FIELD)
def test_a_field_shape_with_own_process_inside_is_low(
    home: Path, repo: Path, live: list[int], command: str
) -> None:
    wt = _attr3880(home, repo)
    child = _own_child_in(wt, live)
    _wait_seen(wt, child.pid)
    # The tree really is held, and an anonymous caller still reads it busy.
    assert prov.worktree_state(str(wt)) == "busy"
    risk, reason = _rank(home, command)
    assert risk is Risk.LOW, reason


def test_a_holder_is_named_as_this_sessions(home: Path, repo: Path, live: list[int]) -> None:
    wt = _attr3880(home, repo)
    child = _own_child_in(wt, live)
    _wait_seen(wt, child.pid)
    holders = prov.cwd_holders(os.path.realpath(wt), S)
    assert holders is not None
    mine = [h for h in holders if h.pid == child.pid]
    assert mine and mine[0].family and mine[0].owner == "this session"
    assert mine[0].command == "sleep"


@pytest.mark.parametrize("command", FIELD)
def test_b_foreign_orphan_inside_stays_high_and_is_named(
    home: Path, repo: Path, live: list[int], command: str
) -> None:
    wt = _attr3880(home, repo)
    pid = _orphan_in(wt, live)
    _wait_seen(wt, pid)
    risk, reason = _rank(home, command)
    assert risk is Risk.HIGH, reason
    assert f"pid {pid} sleep (no session record; unknown process)" in reason, reason
    assert "busy: cwd held by" in reason


def test_b_own_and_foreign_together_stay_high(home: Path, repo: Path, live: list[int]) -> None:
    """One family holder does not launder a foreign one."""
    wt = _attr3880(home, repo)
    child = _own_child_in(wt, live)
    pid = _orphan_in(wt, live)
    _wait_seen(wt, child.pid)
    _wait_seen(wt, pid)
    risk, reason = _rank(home, FIELD[0])
    assert risk is Risk.HIGH, reason
    assert f"pid {child.pid} sleep (this session)" in reason
    assert f"pid {pid} sleep (no session record; unknown process)" in reason


def test_b_another_sessions_harness_child_is_named_by_session(
    home: Path, repo: Path, live: list[int]
) -> None:
    """Attribution comes from the session registry's pid, not from text."""
    from local_operator.session.runtime import registry
    from local_operator.session.runtime.types import SessionRecord

    wt = _attr3880(home, repo)
    # A stand-in "harness" for S2 (registry record by pid) with a child inside.
    harness = subprocess.Popen(["bash", "-c", "sleep 120 & wait"], cwd=wt, start_new_session=True)
    live.append(harness.pid)
    registry.publish(
        SessionRecord(
            pid=harness.pid,
            kind="exec",
            session_id=S2,
            conversation_name="other",
            cwd=str(home),
            model_label="m",
            control_port=1,
            control_key="k",
        )
    )
    try:
        _wait_seen(wt, harness.pid)
        risk, reason = _rank(home, FIELD[0])
        assert risk is Risk.HIGH, reason
        assert f"pid {harness.pid} bash (session {S2}, another session)" in reason, reason
    finally:
        registry.unpublish(harness.pid)


@pytest.mark.asyncio
async def test_b_judge_facts_name_the_foreign_holder(
    home: Path, repo: Path, live: list[int]
) -> None:
    wt = _attr3880(home, repo)
    pid = _orphan_in(wt, live)
    _wait_seen(wt, pid)
    prompts: list[str] = []

    async def judge(system: str, prompt: str) -> str:
        prompts.append(prompt)
        return "BLOCK: someone is in it"

    ctx = CallContext(
        cwd=str(home), env=_env(home), scratchpad=None, session_id=S, intent="clean up"
    )
    outcome = await check_tool_call(
        "bash",
        {"command": FIELD[0]},
        ctx,
        judge=None,
        settings=GuardSettings(),
        forked_judge=judge,
    )
    assert outcome is not None and outcome.blocked
    assert prompts, "a foreign holder must reach the judge"
    assert f"git: busy: cwd held by pid {pid} sleep (no session record" in prompts[0]


def test_unreadable_inventory_is_busy_unknown_and_named(
    home: Path, repo: Path, live: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    wt = _attr3880(home, repo)

    def unreadable(*, fresh: bool = False) -> None:
        prov._cwd_inventory_error = "cwd of pid 4242 unreadable (Permission denied)"
        return None

    monkeypatch.setattr(prov, "_cwd_entries", unreadable)
    assert prov.worktree_state(str(wt), own=True, session_id=S) == "busy"
    risk, reason = _rank(home, FIELD[0])
    assert risk is Risk.HIGH, reason
    assert (
        "busy=unknown: process inventory unreadable "
        "(cwd of pid 4242 unreadable (Permission denied))"
    ) in reason


def test_unreadable_process_table_keeps_own_holder_busy(
    home: Path, repo: Path, live: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The holder is visible but its ancestry is not: ownership unknown, fail closed."""
    wt = _attr3880(home, repo)
    child = _own_child_in(wt, live)
    _wait_seen(wt, child.pid)
    monkeypatch.setattr(prov, "_process_table", lambda: None)
    assert prov.worktree_state(str(wt), own=True, session_id=S) == "busy"
    assert _rank(home, FIELD[0])[0] is Risk.HIGH


def test_stale_cached_hit_is_reread(home: Path, repo: Path, live: list[int]) -> None:
    """A holder that exited inside the 30 s TTL must not keep the tree busy."""
    wt = _attr3880(home, repo)
    real = os.path.realpath(wt)
    prov._cwd_inventory_cache = (time.monotonic(), ((999_999_999, real),))
    assert prov._live_cwd_inside(real) is False
    assert prov.worktree_state(str(wt)) == "dirty"


def test_without_session_id_own_holder_still_busy(home: Path, repo: Path, live: list[int]) -> None:
    """The waiver needs the deleting session's identity; anonymous callers keep busy."""
    wt = _attr3880(home, repo)
    child = _own_child_in(wt, live)
    _wait_seen(wt, child.pid)
    assert prov.worktree_state(str(wt), own=True) == "busy"
    assert prov.worktree_state(str(wt), own=True, session_id=S) == "dirty"
