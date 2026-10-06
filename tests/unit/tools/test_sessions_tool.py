"""The in-session ``sessions`` tool against real substrates.

Modeled on ``test_send_tool.py``: the tool's own logic is never mocked. The
resolver, the registry, the session store and — for the spawn/resume tests —
the REAL nested CLI are exercised, because the tool's whole job is to hand the
CLI the right argv/environment and to report what the world then says. The few
doubles that exist are at process boundaries the product cannot be asked to
provide in a unit test (a launcher subprocess that must be timed, a stop ladder
whose waits are seconds long), and each one is named where it is used.

ISOLATION: every test monkeypatches ``HOME`` and ``LOCAL_OPERATOR_CONFIG_DIR``
to a ``tmp_path`` root, so nothing here can read or touch the operator's live
store. The spawn tests additionally unset every inherited ``CMUX_*``/``LOP_*``
name (the gate brief's rule: an inherited ``CMUX_WORKSPACE_ID`` once let a
headless test rename real cmux workspaces) and reap any worker they started,
scoped to the pid the run itself published — never a bare ``pgrep``.

Where a test says "real CLI", it means ``python -P -m local_operator.cli exec
…`` resolved from this worktree's interpreter, configured with
``hosting: test`` / ``model_name: mock`` so the run needs no network and ends
by itself.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import pytest

from local_operator import session_lease
from local_operator.agent_shell import AGENT_SHELL_ENV, MAY_DELEGATE_ENV
from local_operator.harness.types import FAULT_INVALID_ARGUMENTS, FAULT_KEY, ToolContext
from local_operator.resume import (
    ORIGIN_AGENT_SHELL,
    ORIGIN_AGENT_WORKSTREAM,
    ORIGIN_SUBAGENT,
    is_user_session,
    mark_session_origin,
    write_session_title,
)
from local_operator.scratchpad import SCRATCHPAD_PATH_ENV
from local_operator.session import bulk_resume
from local_operator.session.archived import set_archived
from local_operator.session.attention import AttentionStore
from local_operator.session.bulk_resume import ResumeOutcome, ResumeSelection
from local_operator.session.runtime import control, registry
from local_operator.session.runtime.types import SessionRecord
from local_operator.tools.builtin import (
    _SESSIONS_OP_FIELDS,
    _SESSIONS_TOOL_DESCRIPTION,
    SessionsParams,
    _describe_sessions_approval,
    _sessions_marker_extras,
    _sessions_open_argv,
    _sessions_open_body,
    _sessions_open_env,
    _sessions_published_pid,
    _sessions_tier,
    _sessions_validation_error,
    build_sessions_tool,
    execute_sessions,
    render_sessions_reference,
)
from local_operator.tools.registry import create_tools

REQUESTER_ID = "req000000001"
REQUESTER_NAME = "requester"


@pytest.fixture(autouse=True)
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A root of this test's own, and none of the harness's runtime identity.

    ``HOME`` moves too: the child CLI derives its cache and agent home from the
    real home when only the config override is set, and a test run must not
    leave either behind. The ``CMUX_*``/``LOP_*`` strip is the fleet rule for
    anything that boots product code — exec boots no TUI, but the rule is
    cheap and an inherited workspace id has cost this fleet before.
    """
    home = tmp_path / "home"
    home.mkdir()
    root = tmp_path / "store"
    root.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    monkeypatch.delenv(SCRATCHPAD_PATH_ENV, raising=False)
    for name in list(os.environ):
        if name.startswith(("CMUX_", "LOP_")):
            monkeypatch.delenv(name, raising=False)
    return root


def _write_config(root: Path) -> None:
    """The mock hosting, so a spawned run needs no provider and ends by itself."""
    (root / "config.yml").write_text(
        "version: 0.0.0\nvalues:\n  hosting: test\n  model_name: mock\n", encoding="utf-8"
    )


def _session(root: Path, session_id: str, name: str) -> Path:
    """A real conversation directory: transcript, birth stamp and stored title.

    Written through the readers the listings use (``write_session_title`` rather
    than a hand-rolled sidecar) — the same rule ``test_agent_workstream.py``
    sets: a row that appears here appears because the product can name it.
    """
    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "created_at.json").write_text("1700000000.0", encoding="utf-8")
    (directory / "transcript.jsonl").write_text(
        '{"id":"e1","ts":1,"type":"message",'
        '"payload":{"kind":"message","role":"user","content":[{"text":"go"}]}}\n',
        encoding="utf-8",
    )
    write_session_title(directory, name, user_set=False, past_names=[])
    return directory


def _requester(root: Path) -> Path:
    """The requesting session as a fan-out presents itself: a delegated child.

    Its own marker carries ``label``/``agent`` (what ``harness.subagent``
    writes) and its scratchpad path is what the spawned child reads to name
    the opener — the only source the stamp has for that identity.
    """
    requester = _session(root, REQUESTER_ID, REQUESTER_NAME)
    mark_session_origin(requester, ORIGIN_SUBAGENT, label="1428 review", agent="coder")
    (requester / "scratchpad").mkdir()
    return requester


def _context(root: Path, *, may_delegate: bool = True, **overrides: Any) -> ToolContext:
    fields: dict[str, Any] = {
        "cwd": str(root),
        "session_id": REQUESTER_ID,
        "scratchpad_dir": str(root / "sessions" / REQUESTER_ID / "scratchpad"),
        "subagent_launcher": lambda label, prompt, *, agent="task", effort=None: "job-x",
        "may_delegate": may_delegate,
    }
    fields.update(overrides)
    return ToolContext(**fields)


def _publish_record(
    root: Path, session_id: str, name: str, *, pid: int | None = None, **overrides: Any
) -> SessionRecord:
    """Publish a real discovery record for the running test process.

    The test process's own pid is used so the record classifies as ``live`` —
    the same trick ``test_send_tool.py`` uses, and the reason its store is a
    ``tmp_path``: a record naming this pid must never be visible to anything
    outside the test's isolated root. ``pid`` overrides it for the one test
    that needs TWO live records at once (one pid, one record file — a second
    publish at the same pid would overwrite the first, not add to it).
    """
    record = SessionRecord(
        pid=os.getpid() if pid is None else pid,
        kind="tui",
        session_id=session_id,
        conversation_name=name,
        cwd=str(root),
        model_label="test/mock",
        control_port=1,
        control_key="k" * 32,
        started=True,
        **overrides,
    )
    registry.publish(record, root)
    return record


def _live_pid() -> "subprocess.Popen[bytes]":
    """A separate real process, so a record can name a live pid that is not
    this process's — two of these make two live records for one scan."""
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])


async def _reap_worker(root: Path, session_id: str) -> None:
    """Stop a spawned worker this test started, if it is still alive.

    Scoped to the pid the run itself published for THIS session id in THIS
    root — the fleet rule that only pids we created may be signalled. A run
    that already finished has no record and needs nothing.
    """
    for rec, state in registry.scan(root):
        if rec.session_id != session_id or state not in ("live", "wedged"):
            continue
        try:
            os.kill(rec.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        for _ in range(50):
            try:
                os.kill(rec.pid, 0)
            except ProcessLookupError:
                return
            await asyncio.sleep(0.1)
        try:
            os.kill(rec.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return


# --- 1. the createIf gate, observed on the array the provider receives ------


def test_builder_returns_none_without_the_delegation_surface() -> None:
    assert build_sessions_tool(ToolContext(cwd=".")) is None


def test_a_launcher_context_builds_it() -> None:
    tool = build_sessions_tool(_context(Path(".")))
    assert tool is not None
    assert tool.name == "sessions"
    assert tool.approval_tier == "exec"
    assert tool.call_approval_tier is _sessions_tier
    assert tool.describe_approval is _describe_sessions_approval
    assert tool.concurrency == "exclusive"
    assert tool.interruptible is False


def test_default_set_drops_sessions_without_a_launcher_and_appends_it_with_one() -> None:
    without = [tool.name for tool in create_tools(ToolContext(cwd="."))]
    assert "sessions" not in without

    with_launcher = [
        tool.name
        for tool in create_tools(
            ToolContext(
                cwd=".",
                subagent_launcher=lambda label, prompt, *, agent="task", effort=None: "j",
            )
        )
    ]
    # Appended at the END on purpose: appending never shifts the provider-visible
    # array prefix the prompt cache keys on.
    assert with_launcher[-1] == "sessions"


# --- 2/3. spawn: argv + environment, asserted at the executor seam ----------


def _spawn_params(**overrides: Any) -> SessionsParams:
    fields: dict[str, Any] = {"op": "spawn", "prompt": "say hello", "name": "night-audit"}
    fields.update(overrides)
    return SessionsParams(**fields)


def test_spawn_argv_passes_workstream_by_default() -> None:
    argv = _sessions_open_argv(_spawn_params())
    assert argv[0] == "exec"
    assert "--background" in argv
    assert "--workstream" in argv
    assert argv[-2] == "--"
    assert argv[-1] == "say hello"


def test_spawn_argv_ephemeral_is_the_explicit_opt_out() -> None:
    argv = _sessions_open_argv(_spawn_params(visibility="ephemeral"))
    assert "--workstream" not in argv


def test_spawn_argv_maps_the_optional_fields_and_shields_a_dash_prompt() -> None:
    argv = _sessions_open_argv(
        _spawn_params(
            prompt="-- verify everything",
            name="audit",
            team="release",
            profile="reviewer",
            model="anthropic/claude-sonnet-5",
        )
    )
    assert "--name" in argv and argv[argv.index("--name") + 1] == "audit"
    assert argv[argv.index("--team") + 1] == "release"
    assert argv[argv.index("--profile") + 1] == "reviewer"
    assert argv[argv.index("--model") + 1] == "anthropic/claude-sonnet-5"
    # The separator is what keeps a leading-dash prompt a prompt (argparse would
    # otherwise read it as the next option); checked against the real parser.
    assert argv[-2:] == ["--", "-- verify everything"]


def test_resume_argv_never_passes_workstream() -> None:
    argv = _sessions_open_argv(
        SessionsParams(op="resume", session="abcdef123456", prompt="continue"),
        resume_id="abcdef123456",
    )
    assert "--workstream" not in argv
    assert argv[argv.index("--resume") + 1] == "abcdef123456"
    assert argv[-2:] == ["--", "continue"]


def test_child_env_signs_the_marker_the_guard_reads(root: Path, monkeypatch: Any) -> None:
    ctx = _context(root)
    env = _sessions_open_env(ctx)
    assert env[AGENT_SHELL_ENV] == "1"
    assert env[MAY_DELEGATE_ENV] == "1"
    assert env[SCRATCHPAD_PATH_ENV] == str(root / "sessions" / REQUESTER_ID / "scratchpad")


def test_child_env_strips_inherited_caller_prefixes_but_keeps_injections(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The caller's session markers must never steer the spawned CLI.

    A live desktop-engaged runtime carries ``LOP_RUNTIME_ADOPT_SESSION``,
    ``LOP_RUNTIME_DEFER_MATERIALISE`` and ``LOP_MOBILE_CHILD_RESUME=<its own
    id>`` (measured on this host); inherited, they relax the child's resume
    rules and make it claim the CALLER's session to anything reading a
    process's environment. ``sdk._scoped_process_env`` strips the same two
    prefixes for its children and ``standby.CONTRACT_KEYS`` pins the warm-pool
    half of the rule — this test pins the third child-spawn path to it.
    """
    monkeypatch.setenv("LOP_RUNTIME_ADOPT_SESSION", "1")
    monkeypatch.setenv("LOP_RUNTIME_DEFER_MATERIALISE", "1")
    monkeypatch.setenv("LOP_MOBILE_CHILD_RESUME", REQUESTER_ID)
    monkeypatch.setenv("LOP_MOBILE_CHILD_PROVIDER", "deepseek")
    monkeypatch.setenv("CMUX_WORKSPACE_ID", "ws-caller")

    env = _sessions_open_env(_context(root))

    assert [name for name in env if name.startswith(("CMUX_", "LOP_"))] == []
    # The deliberate grants are untouched — including the names the child
    # contract is built from (LOCAL_OPERATOR_* is a different prefix).
    assert env[AGENT_SHELL_ENV] == "1"
    assert env[MAY_DELEGATE_ENV] == "1"
    assert env[SCRATCHPAD_PATH_ENV] == str(root / "sessions" / REQUESTER_ID / "scratchpad")
    assert "PATH" in env


def test_the_allowance_is_three_armed(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # may_delegate False, name NOT inherited -> not written at all: the name is
    # the mechanism, and a session that never had the allowance is not handed
    # its spelling (test_agent_shell_guard.py pins the same arms for bash).
    monkeypatch.delenv(MAY_DELEGATE_ENV, raising=False)
    env = _sessions_open_env(_context(root, may_delegate=False))
    assert MAY_DELEGATE_ENV not in env

    # may_delegate False, name inherited -> cleared, or the child would be
    # admitted on an allowance nobody granted it.
    monkeypatch.setenv(MAY_DELEGATE_ENV, "1")
    env = _sessions_open_env(_context(root, may_delegate=False))
    assert env[MAY_DELEGATE_ENV] == ""

    # A context that cannot answer (the loop with no host) fails closed —
    # and takes the same two arms as a False context, because the test
    # process's OWN environment decides which one applies (a run under a
    # delegating shell inherits the name; the runner must not smuggle it in).
    monkeypatch.delenv(MAY_DELEGATE_ENV, raising=False)
    env = _sessions_open_env(None)
    assert MAY_DELEGATE_ENV not in env


# --- the real nested CLI: spawn visible-by-default, ephemeral opt-out --------


@pytest.mark.asyncio
async def test_spawn_default_opens_a_listed_workstream(root: Path) -> None:
    """The incident fix, end to end: default spawn is listed with its opener.

    Real CLI, real store, test hosting. The receipt's job id and session id
    must round-trip against the durable ledger, and the run must be listed the
    moment it exists — that is the property the raw-CLI incident broke.
    """
    from local_operator.exec_mode import job_status

    _write_config(root)
    _requester(root)
    ctx = _context(root)

    result = await execute_sessions(
        "t", {"op": "spawn", "prompt": "say hello", "name": "visible-by-default"}, None, None, ctx
    )
    try:
        assert not result.is_error, result.text
        details = result.details or {}
        assert details["op"] == "spawn"
        assert details["origin"] == ORIGIN_AGENT_WORKSTREAM
        assert details["sidebar_visibility"] == "listed"
        session_id = str(details["session_id"])
        assert session_id
        assert details["job_id"]

        directory = root / "sessions" / session_id
        assert directory.is_dir()
        origin = json.loads((directory / "origin.json").read_text(encoding="utf-8"))
        assert origin["origin"] == ORIGIN_AGENT_WORKSTREAM
        assert origin["opened_by"]["session"] == REQUESTER_ID
        assert origin["opened_by"]["agent"] == "coder"
        assert is_user_session(directory)

        # The receipt is the ledger's own answer, not a second composition.
        ledger = job_status(str(details["job_id"]))
        assert ledger["session_id"] == session_id
    finally:
        await _reap_worker(root, str((result.details or {}).get("session_id") or ""))


@pytest.mark.asyncio
async def test_spawn_ephemeral_is_hidden_and_says_how_to_reach_it(root: Path) -> None:
    _write_config(root)
    _requester(root)
    ctx = _context(root)

    result = await execute_sessions(
        "t",
        {"op": "spawn", "prompt": "quick", "name": "throwaway", "visibility": "ephemeral"},
        None,
        None,
        ctx,
    )
    try:
        assert not result.is_error, result.text
        details = result.details or {}
        assert details["origin"] == ORIGIN_AGENT_SHELL
        assert details["sidebar_visibility"] == "hidden"
        directory = root / "sessions" / str(details["session_id"])
        assert json.loads((directory / "origin.json").read_text(encoding="utf-8"))["origin"] == (
            ORIGIN_AGENT_SHELL
        )
        assert not is_user_session(directory)
        # The receipt carries the way back for both sides.
        assert "lop exec --resume" in result.text
        assert "lop sessions" in result.text
    finally:
        await _reap_worker(root, str((result.details or {}).get("session_id") or ""))


@pytest.mark.asyncio
async def test_spawn_refused_without_the_allowance_creates_nothing(root: Path) -> None:
    """Guard passthrough: a session that may not delegate gets the CLI's own
    refusal, and the store is untouched — nothing was half-created."""
    _write_config(root)
    _requester(root)
    ctx = _context(root, may_delegate=False)
    # ``.``-filtered: a fresh store's config migration writes its
    # ``sessions/.local-operator-store`` marker before any guard runs, and the
    # claim under test is that no SESSION directory appears.
    before = sorted(p.name for p in (root / "sessions").iterdir() if not p.name.startswith("."))

    result = await execute_sessions(
        "t", {"op": "spawn", "prompt": "denied", "name": "never"}, None, None, ctx
    )

    assert result.is_error
    # The refusal arrives with the child's own logging in front of it (stderr
    # carries warnings), so it is contained in the text rather than leading it —
    # and it is the CLI's sentence, verbatim: the tool adds no predicate and
    # rewrites nothing.
    assert "exec failed:" in result.text
    assert "cannot open one" in result.text
    after = sorted(p.name for p in (root / "sessions").iterdir() if not p.name.startswith("."))
    assert after == before


# --- list / info ------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_reports_live_rows_and_details_carry_the_published_row(root: Path) -> None:
    from local_operator.info.collect import session_rows

    _session(root, "aaaa11112222", "stored-one")
    record = _publish_record(root, "bbbb33334444", "live-one")
    try:
        result = await execute_sessions("t", {"op": "list"}, None, None, _context(root))
        assert not result.is_error, result.text
        details = result.details or {}
        assert details["op"] == "list"
        assert details["count"] == 1 and details["total"] == 1
        assert "[live] live-one" in result.text
        assert "stored-one" not in result.text  # include_stored defaults off

        published = session_rows(root)
        mine = next(row for row in published if row["session_id"] == record.session_id)
        # Live-measurement fields (RSS, uptime, heartbeat age) move between two
        # scans of the SAME process; the listing's contract is the row SHAPE and
        # every stable field, so the volatile four are compared by presence and
        # everything else by value.
        volatile = {"rss_bytes", "footprint_bytes", "uptime_s", "heartbeat_age_s"}
        assert list(details["rows"][0]) == list(mine)  # pinned key order
        stable = {k: v for k, v in details["rows"][0].items() if k not in volatile}
        expected = {k: v for k, v in mine.items() if k not in volatile}
        assert stable == expected
    finally:
        registry.unpublish(record.pid, root)


@pytest.mark.asyncio
async def test_list_include_stored_appends_stored_rows_as_stored(root: Path) -> None:
    _session(root, "aaaa11112222", "stored-one")
    result = await execute_sessions(
        "t", {"op": "list", "include_stored": True}, None, None, _context(root)
    )
    assert not result.is_error, result.text
    assert "[stored] stored-one" in result.text
    row = (result.details or {})["rows"][0]
    assert row["state"] == "stored"
    assert row["session_id"] == "aaaa11112222"
    # A stored row must not be dressed as a running one.
    assert "up " not in result.text


@pytest.mark.asyncio
async def test_info_reports_origin_visibility_and_opener(root: Path) -> None:
    directory = _session(root, "cccc55556666", "stated")
    mark_session_origin(
        directory,
        ORIGIN_AGENT_WORKSTREAM,
        opened_by={"agent": "coder", "label": "1428 review", "session": REQUESTER_ID},
    )
    result = await execute_sessions(
        "t", {"op": "info", "target": "stated"}, None, None, _context(root)
    )
    assert not result.is_error, result.text
    details = result.details or {}
    assert details["session_id"] == "cccc55556666"
    assert details["session_origin"] == ORIGIN_AGENT_WORKSTREAM
    assert details["sidebar_visibility"] == "listed"
    assert details["opened_by"] == {
        "agent": "coder",
        "label": "1428 review",
        "session": REQUESTER_ID,
    }
    assert details["session_dir"] == str(directory)
    assert details["transcript_path"] == str(directory / "transcript.jsonl")
    assert "origin: agent-workstream" in result.text


@pytest.mark.asyncio
async def test_info_describes_a_stored_hidden_session_from_disk(root: Path) -> None:
    """QA round 1, Q1: an ``agent-shell`` session is excluded from every
    listing scan at ANY limit, so ``info`` must compose from disk (marker
    extras + dir/transcript) instead of claiming a newest-first window it
    was never in. Addressed by exact id, the only address a hidden session
    has."""
    directory = _session(root, "hidden000001", "sneaky")
    mark_session_origin(directory, ORIGIN_AGENT_SHELL)
    result = await execute_sessions(
        "t", {"op": "info", "session": "hidden000001"}, None, None, _context(root)
    )
    assert not result.is_error, result.text
    assert "hidden from every listing scan by its origin marker" in result.text
    assert "window" not in result.text
    details = result.details or {}
    assert details["sidebar_visibility"] == "hidden"
    assert details["session_origin"] == ORIGIN_AGENT_SHELL
    assert details["session_dir"] == str(directory)
    assert details["transcript_path"] == str(directory / "transcript.jsonl")
    assert details["listed"] is False


@pytest.mark.asyncio
async def test_info_names_the_archived_arm_instead_of_a_window(root: Path) -> None:
    """R-5: an archived id is dropped before the activity read, at ANY
    limit, so "past the newest-first window" is false for it. The sentence
    must name the archive arm — and keep the resume-by-id hint, which the
    archive docstring guarantees (an archived session still resolves by
    explicit id)."""
    _session(root, "archived0001", "filed away")
    assert set_archived(root, "archived0001", True)
    result = await execute_sessions(
        "t", {"op": "info", "session": "archived0001"}, None, None, _context(root)
    )
    assert not result.is_error, result.text
    assert "archived — not offered in listings" in result.text
    assert "window" not in result.text
    assert "resumed by its exact session id" in result.text


@pytest.mark.asyncio
async def test_info_names_the_no_activity_arm_instead_of_a_window(root: Path) -> None:
    """R-5: a directory holding only created_at.json has no activity clock
    (the clock is the transcript and the mail spool), is never a listing row,
    and is not a resumable session — the sentence says that, and offers no
    resume hint, because there is nothing to reopen."""
    directory = root / "sessions" / "noactiv00001"
    directory.mkdir(parents=True)
    (directory / "created_at.json").write_text("{}\n", encoding="utf-8")
    result = await execute_sessions(
        "t", {"op": "info", "session": "noactiv00001"}, None, None, _context(root)
    )
    assert not result.is_error, result.text
    assert "no recorded activity to list or resume it by" in result.text
    assert "window" not in result.text
    assert "resumed by its exact session id" not in result.text


@pytest.mark.asyncio
async def test_ambiguous_address_returns_the_shared_candidate_lines(root: Path) -> None:
    sleeper_a = _live_pid()
    sleeper_b = _live_pid()
    record_a = _publish_record(root, "dddd77778888", "twin one", pid=sleeper_a.pid)
    record_b = _publish_record(root, "eeee99990000", "twin two", pid=sleeper_b.pid)
    try:
        result = await execute_sessions(
            "t", {"op": "info", "target": "twin"}, None, None, _context(root)
        )
        assert result.is_error
        assert "2 sessions match; drop `target` and retry with pid=<n> instead" in result.text
        assert f"pid={record_a.pid}" in result.text and f"pid={record_b.pid}" in result.text
    finally:
        registry.unpublish(record_a.pid, root)
        registry.unpublish(record_b.pid, root)
        for sleeper in (sleeper_a, sleeper_b):
            sleeper.terminate()
            sleeper.wait(timeout=10)

    # The stored half disambiguates in its own grammar, with session ids.
    _session(root, "ffff11112222", "store twin A")
    _session(root, "ffff33334444", "store twin B")
    result = await execute_sessions(
        "t", {"op": "info", "target": "store twin"}, None, None, _context(root)
    )
    assert result.is_error
    assert "2 stored sessions match" in result.text
    assert "session=ffff11112222" in result.text and "session=ffff33334444" in result.text


@pytest.mark.asyncio
async def test_a_miss_names_both_searches(root: Path) -> None:
    result = await execute_sessions(
        "t", {"op": "info", "target": "nothing-here"}, None, None, _context(root)
    )
    assert result.is_error
    assert result.text == "no session matches 'nothing-here' (searched live and stored sessions)"


@pytest.mark.asyncio
async def test_list_query_uses_the_store_search(root: Path) -> None:
    _session(root, "aaaa11112222", "the shard report")
    result = await execute_sessions(
        "t", {"op": "list", "query": "shard"}, None, None, _context(root)
    )
    assert not result.is_error, result.text
    assert "the shard report" in result.text
    assert "session aaaa11112222" in result.text
    assert "locates sessions, not positions" in result.text


# --- stop -------------------------------------------------------------------


def test_tier_table_is_per_op() -> None:
    assert _sessions_tier({"op": "list"}) == "read"
    assert _sessions_tier({"op": "info"}) == "read"
    assert _sessions_tier({"op": "peek"}) == "read"
    # ``help`` renders a fixed string and touches nothing; it must not prompt.
    assert _sessions_tier({"op": "help"}) == "read"
    assert _sessions_tier({"op": "spawn"}) == "write"
    assert _sessions_tier({"op": "resume"}) == "write"
    assert _sessions_tier({"op": "stop"}) == "exec"
    # An unknown op must never fold to a read.
    assert _sessions_tier({}) == "exec"


def test_describe_approval_sentences_are_pinned() -> None:
    assert _describe_sessions_approval(
        {"op": "spawn", "name": "night-audit", "prompt": "go"}, "."
    ) == ('open "night-audit" as a listed workstream: go')
    assert (
        _describe_sessions_approval(
            {"op": "spawn", "name": "x", "prompt": "go", "visibility": "ephemeral"}, "."
        )
        == 'open "x" as an ephemeral session: go'
    )
    assert (
        _describe_sessions_approval(
            {"op": "spawn", "prompt": "go", "team": "release", "profile": "coder"}, "."
        )
        == "open as a listed workstream (team release, profile coder): go"
    )
    assert _describe_sessions_approval({"op": "stop", "pid": 48213}, ".") == (
        "stop pid 48213: ends its current run and releases the session lease"
    )
    assert _describe_sessions_approval({"op": "resume", "session": "abcdef123456"}, ".") == (
        "resume session abcdef123456: reopens its transcript headlessly"
    )
    assert _describe_sessions_approval({"op": "list"}, ".") == "list"


@pytest.mark.asyncio
async def test_stop_runs_the_graceful_ladder_only(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The target's pid must NOT be this process's: the tool runs inside the
    # caller's runtime, and a target naming our own pid is a self-stop, which
    # the tool refuses before the ladder (review round 1, R-2). A real live
    # subprocess makes the record "live" without being self.
    worker = _live_pid()
    record = _publish_record(root, "abcdabcdabcd", "stoppable", pid=worker.pid)
    captured: dict[str, Any] = {}

    class _Outcome:
        pid = record.pid
        session_id = record.session_id
        name = record.conversation_name
        method = "socket"
        wakes_dormant = 1
        monitors_dormant = 0
        line = 'stopped "stoppable" — 1 wake dormant until you reopen it'

    async def _fake_stop(target: Any, **kwargs: Any) -> Any:
        captured["target"] = target
        captured.update(kwargs)
        return _Outcome()

    monkeypatch.setattr(control, "stop_session", _fake_stop)
    try:
        result = await execute_sessions(
            "t", {"op": "stop", "session": record.session_id}, None, None, _context(root)
        )
        assert not result.is_error, result.text
        assert result.text == _Outcome.line
        assert captured["target"] == record
        # v1 is the graceful ladder only: no SIGKILL from the tool, and the
        # marker records which front end asked.
        assert captured["force"] is False
        assert captured["timeout_s"] == control.DEFAULT_TIMEOUT_S
        assert captured["_command"] == "sessions tool"
        assert (result.details or {})["method"] == "socket"
    finally:
        registry.unpublish(record.pid, root)
        worker.terminate()
        worker.wait(timeout=10)


@pytest.mark.asyncio
async def test_stop_refuses_a_self_target_and_never_reaches_the_ladder(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R-2: a stop resolving to THIS process is refused before the ladder.

    ``_publish_record`` defaults to this process's pid, so the record is
    exactly the shape a session's own registration takes — and the stop would
    socket-op, then signal, the very run making the call. "Nothing killed" is
    asserted as the ladder never being entered.
    """
    record = _publish_record(root, "self00000000", "self target")
    reached: list[Any] = []

    async def _must_not_run(target: Any, **kwargs: Any) -> Any:  # pragma: no cover
        reached.append(target)
        raise AssertionError("the stop ladder must not be reached for a self target")

    monkeypatch.setattr(control, "stop_session", _must_not_run)
    try:
        result = await execute_sessions(
            "t", {"op": "stop", "session": record.session_id}, None, None, _context(root)
        )
        assert result.is_error
        assert "cannot stop itself" in result.text
        assert reached == []
    finally:
        registry.unpublish(record.pid, root)


@pytest.mark.asyncio
async def test_stop_refuses_a_stored_session(root: Path) -> None:
    _session(root, "aaaa11112222", "sleeping")
    result = await execute_sessions(
        "t", {"op": "stop", "session": "aaaa11112222"}, None, None, _context(root)
    )
    assert result.is_error
    assert "is not running" in result.text


# --- resume -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_resume_of_a_hidden_session_keeps_its_stamp_and_says_so(root: Path) -> None:
    """``origin.json`` is immutable: a resume reports, never re-stamps."""
    _write_config(root)
    _requester(root)
    directory = _session(root, "aaaa11112222", "was hidden")
    mark_session_origin(directory, ORIGIN_AGENT_SHELL)
    before = (directory / "origin.json").read_bytes()

    result = await execute_sessions(
        "t",
        {"op": "resume", "session": "aaaa11112222", "prompt": "continue"},
        None,
        None,
        _context(root),
    )
    try:
        assert not result.is_error, result.text
        details = result.details or {}
        assert details["origin"] == ORIGIN_AGENT_SHELL
        assert details["sidebar_visibility"] == "hidden"
        assert details["visibility_changed"] is False
        assert "stays hidden" in result.text
        assert (directory / "origin.json").read_bytes() == before
    finally:
        await _reap_worker(root, str((result.details or {}).get("session_id") or "aaaa11112222"))


@pytest.mark.asyncio
async def test_resume_refuses_a_live_runtime_with_the_cli_sentence(root: Path) -> None:
    _write_config(root)
    _requester(root)
    directory = _session(root, "aaaa11112222", "held")
    lease = session_lease.acquire_session_lease(directory)
    before = sorted(p.name for p in (root / "sessions").iterdir() if not p.name.startswith("."))
    try:
        result = await execute_sessions(
            "t",
            {"op": "resume", "session": "aaaa11112222", "prompt": "continue"},
            None,
            None,
            _context(root),
        )
        assert result.is_error
        assert "is already open in another process" in result.text
        assert f"(pid {os.getpid()})" in result.text
        after = sorted(p.name for p in (root / "sessions").iterdir() if not p.name.startswith("."))
        assert after == before
    finally:
        lease.release()


@pytest.mark.asyncio
async def test_resume_refuses_a_self_target(root: Path) -> None:
    """R-2's resume arm: a live record naming this process is this session.

    ``resume`` reopens a stored/stopped conversation; one that is already
    live here has nothing to reopen, and the CLI never gets the dial.
    """
    record = _publish_record(root, "self11111111", "self target")
    try:
        result = await execute_sessions(
            "t",
            {"op": "resume", "session": record.session_id, "prompt": "continue"},
            None,
            None,
            _context(root),
        )
        assert result.is_error
        assert "cannot resume itself" in result.text
    finally:
        registry.unpublish(record.pid, root)


# --- receipts and schema budget ---------------------------------------------


def test_marker_extras_say_unknown_when_the_directory_is_missing(root: Path) -> None:
    """R-4: the unreadable arm must not assert a visibility it could not read."""
    extras = _sessions_marker_extras("000000000000")
    assert extras["session_dir"] is None
    assert extras["sidebar_visibility"] == "unknown"


def test_resume_receipt_spaces_the_id_once() -> None:
    """Q2: ``{label}`` ended with a space and ``{where}`` began with one, so
    the resume receipt shipped a double space ('reopened "held"  (session …)').

    Both branches are pinned — the fix shares one ``named``/``lead`` pair, and
    a regression in either spelling shows up here.
    """
    resume_text = _sessions_open_body(
        SessionsParams(op="resume", session="a1", prompt="go"),
        {
            "op": "resume",
            "name": "held",
            "session_id": "a1b2c3d4e5f6",
            "job_id": "j1",
            "origin": ORIGIN_AGENT_SHELL,
            "sidebar_visibility": "hidden",
        },
    )
    assert 'reopened "held" (session a1b2c3d4e5f6, job j1)' in resume_text
    assert '"held"  (' not in resume_text
    assert "  " not in resume_text
    unnamed_resume = _sessions_open_body(
        SessionsParams(op="resume", session="a1", prompt="go"),
        {
            "op": "resume",
            "session_id": "a1b2c3d4e5f6",
            "origin": ORIGIN_AGENT_SHELL,
            "sidebar_visibility": "hidden",
        },
    )
    assert "reopened (session a1b2c3d4e5f6)" in unnamed_resume
    assert "reopened  (" not in unnamed_resume
    assert "  " not in unnamed_resume
    spawn_text = _sessions_open_body(
        SessionsParams(op="spawn", prompt="go"),
        {
            "op": "spawn",
            "name": "w",
            "session_id": "s1",
            "sidebar_visibility": "listed",
            "origin": ORIGIN_AGENT_WORKSTREAM,
        },
    )
    assert 'opened "w" as a listed workstream (session s1)' in spawn_text
    unnamed = _sessions_open_body(
        SessionsParams(op="spawn", prompt="go"),
        {"op": "spawn", "session_id": "s1", "sidebar_visibility": "listed"},
    )
    assert "opened as a listed workstream" in unnamed
    assert "opened  as" not in unnamed


def test_published_pid_requires_the_record_to_name_our_session(root: Path) -> None:
    record = _publish_record(root, "abcdabcdabcd", "pidful")
    path = registry.record_path(record.pid, root)
    assert (
        _sessions_published_pid({"runtime_path": str(path), "session_id": record.session_id})
        == record.pid
    )
    # A recycled pid's leftover file must not be reported as this run's.
    assert (
        _sessions_published_pid({"runtime_path": str(path), "session_id": "othereid00000"}) is None
    )
    assert _sessions_published_pid({"session_id": record.session_id}) is None
    registry.unpublish(record.pid, root)


def test_validation_refusals_are_legible_and_per_op() -> None:
    assert _sessions_validation_error(SessionsParams(op="spawn")) == (
        "spawn needs `prompt`: the message the opened run executes. A headless exec "
        "refuses a prompt-less run the same way."
    )
    refusal = _sessions_validation_error(SessionsParams(op="spawn", prompt="go", background=False))
    assert refusal is not None and "not supported in v1" in refusal
    refusal = _sessions_validation_error(
        SessionsParams(op="resume", session="a", prompt="p", visibility="workstream")
    )
    assert refusal is not None and "never re-stamped" in refusal
    refusal = _sessions_validation_error(SessionsParams(op="stop", prompt="p"))
    assert refusal is not None
    assert refusal.startswith("`prompt` applies to spawn/resume only. ")
    assert "`stop` takes: session|target|pid, peer." in refusal
    assert refusal.endswith("Call op='help' for the full per-op reference.")
    refusal = _sessions_validation_error(SessionsParams(op="list", session="a"))
    assert refusal is not None
    assert refusal.startswith("`session` does not apply to op='list' — it takes no address. ")
    assert "`list` takes: peer, scope, include_stored, limit, query." in refusal
    refusal = _sessions_validation_error(SessionsParams(op="spawn", prompt="go", target="x"))
    assert refusal is not None and "creates a new session" in refusal
    refusal = _sessions_validation_error(SessionsParams(op="info"))
    assert refusal is not None and "exactly one of" in refusal


def test_schema_budget_is_measured_with_the_repos_own_ruler() -> None:
    """The design's §3.4 budget, re-measured for the peek surface (PR B).

    Ruler: ``compaction/tokens.count_text_tokens`` (cl100k_base via tiktoken,
    the repo's estimator and the one §3.4 was measured with); subject: the
    exact JSON the provider sees as the tool's ``parameters``.

    §14.6 locked the lifecycle tool at ≤700 tokens (shipped at 699 in PR A).
    The six peek window fields cannot fit under it: with their descriptions
    written lean the tool measures 945, and the 21-field floor with EVERY
    field description stripped is still 644 — holding 700 would mean shipping
    an undocumented schema. The ceiling below pins the measured figure with
    the same knife-edge headroom the 700 had; tighten it when the schema is
    trimmed, never widen it to admit a verbose field. (The CI-side cost of the
    same growth is the ratchet entry for PR B in
    ``scripts/bench_context_budget.py``.)

    RAISED 950 -> 1085 and 200 -> 231 for the bulk-resume SET form
    (``feat/sessions-bulk-resume-1001``), stated with the arithmetic because
    this guard exists to make copy growth an explicit decision. The change
    adds four boolean fields to the schema — ``paused``/``failed``/``all``
    (the selectors) and ``dry_run`` — plus the ``limit`` description's
    resume clause; measured on the merged tree with this test's own ruler:
    schema 949 -> 1,084 (+135: the five fields' names, titles, types,
    defaults and lean one-clause descriptions; a first draft with fuller
    descriptions measured 1,112 and was trimmed, which is the "never widen
    for verbosity" rule working), description 200 -> 230 (+30: the four new
    accepted inputs in the derived per-op summary, ~18, and the one prose
    clause naming the set form, ~12). The ceilings are the measured figures
    + 1, the knife-edge headroom the prior pin kept. The full vocabulary
    (what `paused` covers, the default cap, the dry-run preview) lives in
    the on-demand ``op='help'`` reference and the CLI's `--help`, which is
    where the budget wants it.

    RAISED 1085 -> 1099 for the role-word fix (``fix/role-word-recipient``).
    ``SessionsParams.target`` now states the two new rules the resolver applies
    — an exact name/id beats a substring, and a team role word is refused —
    which grew the derived ``parameters`` JSON by 14 tokens (1,084 -> 1,098
    measured with this test's own ruler; the ceiling is +1 over the measurement,
    as before). The tool DESCRIPTION is unchanged (230), because the statement
    belongs on the field it governs, not in the per-op summary. The full
    vocabulary stays in ``op='help'``.

    RAISED 1099 -> 1213 and 231 -> 273 for the mesh fields
    (``feat/sessions-remote-tools-b``): ``peer`` and ``scope`` add two
    properties to the schema (1,212 measured with this test's own ruler; their
    descriptions are the design §4 strings verbatim, drafted lean there), and
    the description gains §4's one mesh clause (272 measured — the derived
    per-op summary grows across five ops and the clause itself is one
    sentence). The ceilings are the measured figures + 1, the same knife-edge
    headroom as every pin above.
    """
    from local_operator.compaction.tokens import count_text_tokens
    from local_operator.tools.builtin import _SESSIONS_TOOL_DESCRIPTION

    params = json.dumps(SessionsParams.model_json_schema(), ensure_ascii=False)
    assert count_text_tokens(params) <= 1213
    assert count_text_tokens(_SESSIONS_TOOL_DESCRIPTION) <= 273


# --- peek (PR B): bounded transcript inspection ------------------------------


def _synth_row(
    i: int,
    *,
    role: str | None = None,
    text: str | None = None,
    pad: int = 0,
) -> str:
    """One message row shaped like a real journal's, ~``pad`` bytes of filler."""
    role = role or ("user", "assistant", "tool")[i % 3]
    payload: dict[str, Any] = {
        "kind": "message",
        "role": role,
        "content": [{"text": text if text is not None else f"row {i} " + "x" * pad}],
    }
    if role == "assistant":
        payload["tool_calls"] = [{"name": "bash", "arguments": {"command": f"cmd {i}"}}]
    if role == "tool":
        payload["tool_name"] = "bash"
    return json.dumps({"id": f"{i:032x}", "ts": i, "type": "message", "payload": payload})


def _write_journal(root: Path, session_id: str, lines: list[str], *, title: str = "") -> Path:
    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "created_at.json").write_text("1700000000.0", encoding="utf-8")
    (directory / "transcript.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if title:
        # A stored title keeps ``session_name`` off the transcript, so the
        # byte-counting instrument below sees only the peek read it measures.
        write_session_title(directory, title, user_set=False, past_names=[])
    return directory


def _big_journal(
    root: Path,
    session_id: str,
    *,
    rows: int,
    pad: int,
    custom_every: int = 0,
    needle: str = "",
    needle_row: int = -1,
) -> tuple[Path, list[str], list[str]]:
    """A large synthetic journal: (directory, row ids, raw lines).

    Rows are ~``pad`` bytes of filler so a whole-file parse is a byte count
    away from being detectable, and every ``custom_every``-th row appends a
    bookkeeping row — the shape that renders as NO step, which is what the
    step walks exist to tolerate. ``needle`` lands in row ``needle_row``.
    """
    lines: list[str] = []
    ids: list[str] = []
    for i in range(1, rows + 1):
        text = f"row {i} {needle} " + "x" * pad if needle and i == needle_row else None
        lines.append(_synth_row(i, text=text, pad=pad))
        ids.append(f"{i:032x}")
        if custom_every and i % custom_every == 0:
            lines.append(
                json.dumps(
                    {
                        "id": f"c{i:031x}",
                        "ts": i,
                        "type": "custom",
                        "payload": {
                            "kind": "custom",
                            "custom_type": "todo_snapshot",
                            "details": {"text": "t"},
                        },
                    }
                )
            )
    directory = _write_journal(root, session_id, lines, title=f"journal {session_id[:4]}")
    return directory, ids, lines


async def _peek(root: Path, case: dict[str, Any]) -> Any:
    return await execute_sessions("t", case, None, None, _context(root))


def _rows_in(text: str) -> set[int]:
    """The ``row N`` markers a rendered window actually shows.

    Bodies arrive through ``_clip``, which STRIPS, so "row 148 " loses its
    trailing space and a substring check on that space would be quietly wrong;
    the word-boundary digit run is the spelling that survives the strip.
    """
    return {int(match) for match in re.findall(r"row (\d+)\b", text)}


@pytest.mark.asyncio
async def test_peek_tail_read_cost_is_the_window_not_the_journal(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§11.8's structural byte bound, pointed at the default peek read.

    The instrument is ``test_transcript.py``'s own ``_counted_reads`` — bytes
    handed OUT of ``Path.open`` for this one file, which no machine load can
    move — reused rather than re-written so the peek and the reader it rides
    are held to one ruler. A whole-file parse on this fixture reads ~7.7 MB;
    the bound below is the page plus two chunks. Can fail: make
    ``_peek_page_rows`` return ``10**9`` (the reader then walks the journal
    back to its start) and this test goes red on the byte count.
    """
    from local_operator.session import transcript as transcript_module
    from local_operator.tools.builtin import _peek_page_rows
    from tests.unit.session.test_transcript import _counted_reads

    directory, ids, lines = _big_journal(root, "aaaa11112222", rows=8000, pad=900, custom_every=10)
    path = directory / "transcript.jsonl"
    size = path.stat().st_size

    with _counted_reads(monkeypatch, path) as counted:
        result = await _peek(root, {"op": "peek", "session": "aaaa11112222"})
    assert not result.is_error, result.text
    details = result.details or {}
    assert details["mode"] == "tail" and details["steps_shown"] == 12
    assert 8000 in _rows_in(result.text)  # the newest message row is the newest step
    assert details["has_older"] is True

    page_bytes = sum(len(line.encode("utf-8")) + 1 for line in lines[-_peek_page_rows(12) :])
    assert counted[0] <= page_bytes + 2 * transcript_module._BACKWARD_CHUNK_BYTES
    # ... and never anything like the journal, however long the journal is.
    assert counted[0] * 4 < size


@pytest.mark.asyncio
async def test_peek_head_read_cost_is_the_window_not_the_journal(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The head read's bound, same instrument, bounded the other direction.

    The forward walker exists as a generator precisely so the caller's stop is
    the bound: this consumes the first rows and never walks the journal. Can
    fail: drop the stop condition in ``_peek_head`` (let the walker run to
    EOF) and the byte count grows to the journal's size.
    """
    from local_operator.session import transcript as transcript_module
    from tests.unit.session.test_transcript import _counted_reads

    directory, ids, lines = _big_journal(root, "bbbb22223333", rows=8000, pad=900, custom_every=10)
    path = directory / "transcript.jsonl"
    size = path.stat().st_size

    with _counted_reads(monkeypatch, path) as counted:
        result = await _peek(root, {"op": "peek", "session": "bbbb22223333", "head": 3})
    assert not result.is_error, result.text
    details = result.details or {}
    assert details["mode"] == "head" and details["steps_shown"] == 3
    # The window is rows 1..3, oldest first. Asserting on "row 4" would be
    # WRONG to expect absent: it is the one-row lookahead consumed to answer
    # ``has_newer`` — consumed, but never rendered, so the shown rows are 1-3.
    assert _rows_in(result.text) == {1, 2, 3}

    head_bytes = sum(len(line.encode("utf-8")) + 1 for line in lines[:6])
    assert counted[0] <= head_bytes + 2 * transcript_module._BACKWARD_CHUNK_BYTES
    assert counted[0] * 4 < size


@pytest.mark.asyncio
async def test_peek_cursor_pages_round_trip_without_gaps_or_repeats(root: Path) -> None:
    """Two ``before_id`` pages chained through the footer's own cursor.

    The cursor must be the oldest SHOWN step, never the oldest row the walk
    read past while hunting for steps: a cursor into a row the caller never
    saw would make the next page skip everything between (the first draft did
    exactly that and skipped 36 rows on a dense page). The assertions below
    are the rows themselves, so a regression is named, not inferred.
    """
    directory, ids, _ = _big_journal(root, "cccc33334444", rows=200, pad=40)
    first = await _peek(root, {"op": "peek", "session": "cccc33334444", "before_id": ids[159]})
    assert not first.is_error, first.text
    # The 12 steps immediately before row 160, oldest→newest: rows 148..159.
    assert _rows_in(first.text) == set(range(148, 160))
    cursor = re.search(r"before_id=([0-9a-f]{32}) for earlier", first.text)
    assert cursor is not None, first.text
    assert cursor.group(1) == ids[147]  # the OLDEST SHOWN step, not ~ids[111]

    second = await _peek(
        root, {"op": "peek", "session": "cccc33334444", "before_id": cursor.group(1)}
    )
    assert not second.is_error, second.text
    assert _rows_in(second.text) == set(range(136, 148))


@pytest.mark.asyncio
async def test_peek_around_window_reports_both_edges(root: Path) -> None:
    directory, ids, _ = _big_journal(root, "dddd77778888", rows=100, pad=40)
    result = await _peek(root, {"op": "peek", "session": "dddd77778888", "around_id": ids[49]})
    assert not result.is_error, result.text
    details = result.details or {}
    assert details["mode"] == "around" and details["steps_shown"] == 12
    assert details["has_older"] is True and details["has_newer"] is True
    assert 50 in _rows_in(result.text)  # the anchor itself is a step
    assert "before_id=" in result.text and "around_id=" in result.text


@pytest.mark.asyncio
async def test_peek_search_finds_a_deep_needle_within_the_budget(root: Path) -> None:
    """A needle ~3 MB deep is found, windowed, and its depth reported."""
    from local_operator.tools.builtin import _PEEK_SCAN_BYTES

    directory, ids, _ = _big_journal(
        root, "eeee44445555", rows=12000, pad=300, needle="needle-alpha", needle_row=6000
    )
    result = await _peek(root, {"op": "peek", "session": "eeee44445555", "query": "needle-alpha"})
    assert not result.is_error, result.text
    details = result.details or {}
    assert details["mode"] == "search" and details["match_id"] == ids[5999]
    assert "needle-alpha" in result.text and "← match" in result.text
    # The depth is the point: a scan that stopped near EOF would report a few
    # KB; this one had to walk the ~2.8 MB above the needle.
    assert 2_000_000 <= details["scanned_bytes"] <= _PEEK_SCAN_BYTES


@pytest.mark.asyncio
async def test_peek_search_budget_is_a_depth_and_moves_with_the_needle(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The prove-can-fail pair for the search budget (§11.9): same 64 KiB
    budget, needle at ~5 KB depth (hit, ``scanned_bytes`` under the cap) and
    the SAME fixture shape with the needle at ~330 KB depth (honest miss) —
    the miss message names the bound and a pointer to widen."""
    from local_operator.tools import builtin as builtin_module

    monkeypatch.setattr(builtin_module, "_PEEK_SCAN_BYTES", 64 * 1024)

    _, shallow_ids, _ = _big_journal(
        root, "ffff66667777", rows=400, pad=900, needle="needle-shallow", needle_row=395
    )
    hit = await _peek(root, {"op": "peek", "session": "ffff66667777", "query": "needle-shallow"})
    assert not hit.is_error, hit.text
    assert (hit.details or {})["match_id"] == shallow_ids[394]
    assert (hit.details or {})["scanned_bytes"] < 64 * 1024

    _big_journal(root, "aaaa66667777", rows=400, pad=900, needle="needle-deep", needle_row=50)
    miss = await _peek(root, {"op": "peek", "session": "aaaa66667777", "query": "needle-deep"})
    assert not miss.is_error  # an honest miss is an ANSWER, not an error
    assert miss.text.startswith("no match for 'needle-deep'")
    assert "bounded" in miss.text and "around_id" in miss.text
    details = miss.details or {}
    assert details["steps_shown"] == 0
    assert 64 * 1024 <= details["scanned_bytes"] <= 64 * 1024 + 2000


@pytest.mark.asyncio
async def test_peek_digest_shape_and_counts(root: Path) -> None:
    """§8.3: counts per kind, the newest ask/reply, the tool tail, ≤10 lines."""
    rows = [
        ("user", "add a retry budget to the flaky shard test"),
        ("assistant", "reading the shard report first"),
        ("tool", "exit code: 0"),
        ("assistant", "three retries in; the shard is green"),
        ("tool", "ran pytest -q"),
        ("user", "ship it"),
        ("tool", "wrote the file"),
        ("tool", "no output"),
    ]
    lines = [_synth_row(i, role=role, text=text) for i, (role, text) in enumerate(rows, 1)]
    _write_journal(root, "aaaa99990000", lines, title="digest case")

    result = await _peek(root, {"op": "peek", "session": "aaaa99990000", "digest": True})
    assert not result.is_error, result.text
    fold = result.text.splitlines()
    assert fold[0] == (
        "digest: 8 steps seen (user 2, assistant 2, tool 4) — stored; whole transcript"
    )
    assert fold[1] == "ask: ship it"
    assert fold[2] == "assistant: three retries in; the shard is green"
    assert fold[3:] == [
        "tool: bash · exit code: 0",
        "tool: bash · ran pytest -q",
        "tool: bash · wrote the file",
        "tool: bash · no output",
    ]
    assert len(fold) <= 10
    assert result.details == {
        "op": "peek",
        "mode": "digest",
        "session_id": "aaaa99990000",
        "steps_seen": 8,
        "tool_calls": 4,
        "has_older": False,
    }


@pytest.mark.asyncio
async def test_peek_digest_empty_and_one_turn_cases(root: Path) -> None:
    _write_journal(root, "bbbb99990000", [], title="empty")
    result = await _peek(root, {"op": "peek", "session": "bbbb99990000", "digest": True})
    assert not result.is_error, result.text
    assert result.text.startswith("digest: 0 steps seen")
    assert (result.details or {})["steps_seen"] == 0

    _write_journal(root, "cccc99990000", [_synth_row(1, role="user", text="hello")], title="one")
    result = await _peek(root, {"op": "peek", "session": "cccc99990000", "digest": True})
    assert not result.is_error, result.text
    assert result.text.splitlines() == [
        "digest: 1 steps seen (user 1, assistant 0, tool 0) — stored; whole transcript",
        "ask: hello",
    ]


@pytest.mark.asyncio
async def test_peek_output_budgets_are_guarded_by_char_proxy_and_tokens(root: Path) -> None:
    """§8.4's per-op output budgets, measured on each op's DEFAULT invocation.

    Ruler: the repo's own ``count_text_tokens`` (cl100k_base when tiktoken is
    installed, the chars/4 proxy otherwise — the fallback discipline
    ``compaction/tokens.py`` documents) plus the character proxy (4 chars per
    token). The peek fixture is the WORST default case — every step body at
    the 600-char clip cap — so the ceilings pin the maximum the default
    invocation can emit, not a friendly average. §8.4's figures are
    bodies-only approximations ("12 steps × ≤600 chars" is 7,200 chars ≈ 1,800
    tokens at the proxy, with no room for headings or the footer); the numbers
    below are this implementation's measured worst case (peek 8,385 chars /
    2,153 cl100k tokens; digest 817 / 241) with a small margin, and a
    regression that stops clipping or drops the digest's line cap goes red.
    ``stop`` is not measured: its body is the kill-switch ladder's own
    receipt painted verbatim, and the tool adds only a bounded ``waited`` list.
    """
    from local_operator.compaction.tokens import count_text_tokens
    from local_operator.tools.builtin import _sessions_open_body

    long_body = (
        "The frobnicator test failed on shard 3. Running pytest -q tests/unit/tools "
        "-x gave an exit code of 1 after 42 seconds.\n"
    ) * 30
    long_body = (long_body * 4)[:2000]
    rows = [_synth_row(i, text=long_body) for i in range(1, 31)]
    _write_journal(root, "dddd99990000", rows, title="budget")

    result = await _peek(root, {"op": "peek", "session": "dddd99990000"})
    assert not result.is_error, result.text
    assert "chars elided" in result.text  # the clip is real, not assumed
    assert len(result.text) <= 8_600
    assert count_text_tokens(result.text) <= 2_400

    result = await _peek(root, {"op": "peek", "session": "dddd99990000", "digest": True})
    assert not result.is_error, result.text
    assert len(result.text.splitlines()) <= 10
    assert len(result.text) <= 900
    assert count_text_tokens(result.text) <= 300

    # The lifecycle ops' rows of the same table, measured the same way: a real
    # list over one live + one stored row, one info read, and the two receipts
    # through their own body builder.
    _publish_record(root, "dddd99990000", "budget", pid=os.getpid())
    _write_journal(
        root, "eeee99990000", [_synth_row(1, role="user", text="stored row")], title="stored"
    )
    result = await _peek(root, {"op": "list", "include_stored": True, "limit": 5})
    assert not result.is_error, result.text
    assert len(result.text) <= 2_400 and count_text_tokens(result.text) <= 600

    result = await _peek(root, {"op": "info", "session": "dddd99990000"})
    assert not result.is_error, result.text
    assert len(result.text) <= 800 and count_text_tokens(result.text) <= 200

    receipt = _sessions_open_body(
        _spawn_params(),
        {
            "session_id": "a1b2c3d4e5f6",
            "job_id": "3f9c2b17",
            "name": "night-audit",
            "state": "starting",
            "origin": "agent-workstream",
            "sidebar_visibility": "listed",
            "opener": {"agent": "coder", "label": "sessions-PR-B", "session": "0b39"},
        },
    )
    assert len(receipt) <= 480 and count_text_tokens(receipt) <= 120


@pytest.mark.asyncio
async def test_peek_spill_fit_on_an_oversize_window(root: Path) -> None:
    """§14.7, verified: spill_truncate applies cleanly to peek bodies at these
    sizes. UNCLAMPED bodies (≤600 chars — no clamp marker) fit inline for the
    default 12-step window; the moment any body needs the marker, the max-fill
    default can cross the 8 KiB tool limit and then spills — the DESIGNED
    handling (§8.4's budget vs the tool limit), not a bug, and both directions
    are pinned below. A 50-step window is over the limit by construction; its
    handle's byte count is the FULL body, never the raw ~32 KB."""
    rows = [_synth_row(i, text="s" * 600) for i in range(1, 61)]
    _write_journal(root, "ffff99990011", rows, title="spill case")

    small = await _peek(root, {"op": "peek", "session": "ffff99990011"})
    assert not small.is_error, small.text
    assert "spill" not in (small.details or {})
    assert len(small.text) < 8 * 1024

    # The same default window over bodies that need the clamp marker — the
    # budget test's own multiline fixture (its line structure is what pushes
    # the raw body past 8 KiB: clipped head + marker + tail, indented per
    # line). The result arrives elided around a handle: the claim above, locked.
    long_body = (
        "The frobnicator test failed on shard 3. Running pytest -q tests/unit/tools "
        "-x gave an exit code of 1 after 42 seconds.\n"
    ) * 30
    long_body = (long_body * 4)[:2000]
    clipped_rows = [_synth_row(i, text=long_body) for i in range(1, 61)]
    _write_journal(root, "ffff99990012", clipped_rows, title="spill clip case")
    clipped = await _peek(root, {"op": "peek", "session": "ffff99990012"})
    assert not clipped.is_error, clipped.text
    assert "spill" in (clipped.details or {})
    assert len(clipped.text) < 9_000

    big = await _peek(root, {"op": "peek", "session": "ffff99990011", "steps": 50})
    assert not big.is_error, big.text
    spill = (big.details or {}).get("spill")
    assert spill is not None and spill["handle"].startswith("spill://")
    assert spill["bytes"] > 20_000  # the body it replaced is the raw 50-step window
    assert len(big.text) < 9_000  # the elided body, never the raw ~32k


@pytest.mark.asyncio
async def test_peek_empty_transcript_reports_no_steps(root: Path) -> None:
    _write_journal(root, "ffff99990000", [], title="empty")
    result = await _peek(root, {"op": "peek", "session": "ffff99990000"})
    assert not result.is_error, result.text
    assert "no steps yet" in result.text
    assert (result.details or {})["steps_shown"] == 0


@pytest.mark.asyncio
async def test_peek_refuses_unknown_ids_and_a_missing_transcript(root: Path) -> None:
    _write_journal(root, "aaaa77778888", [_synth_row(1, role="user", text="hi")], title="t")

    result = await _peek(root, {"op": "peek", "session": "aaaa77778888", "before_id": "f" * 32})
    assert result.is_error and "no transcript entry with id" in result.text

    result = await _peek(root, {"op": "peek", "session": "aaaa77778888", "around_id": "f" * 32})
    assert result.is_error and "no transcript entry with id" in result.text

    result = await _peek(
        root, {"op": "peek", "session": "aaaa77778888", "query": "([", "regex": True}
    )
    # R-5: malformed arguments carry the fault marker, the shape `read`'s
    # spill-search and `grep` use for the same class ("invalid regex '<p>': …").
    from local_operator.harness.types import FAULT_INVALID_ARGUMENTS, FAULT_KEY

    assert result.is_error is True
    assert result.text.startswith("invalid regex '([':")
    assert (result.details or {})[FAULT_KEY] == FAULT_INVALID_ARGUMENTS

    (root / "sessions" / "bbbb77778888").mkdir(parents=True)
    result = await _peek(root, {"op": "peek", "session": "bbbb77778888"})
    assert result.is_error and "has no transcript to peek at" in result.text


@pytest.mark.asyncio
async def test_peek_regex_query_is_compiled_stripped_like_the_needle(root: Path) -> None:
    """R-6: the executor's pre-compiled pattern is the SAME stripped spelling
    the needle path (and validation, and the footer) uses. Pre-R-5 the search
    compiled the stripped needle itself, so a padded regex query used to match
    what literal mode matches; the move to an executor-side compile must not
    change that. Padded and unpadded regex stay in agreement, and literal mode
    is the control they both answer to."""
    rows = [_synth_row(i, text=f"row {i} " + "x" * 30) for i in range(1, 61)]
    rows[29] = _synth_row(30, text="row 30 the flaky shard needs a retry budget")
    _write_journal(root, "aaaa88889999", rows, title="padded regex")

    padded = await _peek(
        root,
        {
            "op": "peek",
            "session": "aaaa88889999",
            "query": "  flaky .*budget  ",
            "regex": True,
        },
    )
    assert not padded.is_error, padded.text
    assert (padded.details or {})["match_id"] == f"{30:032x}"

    control = await _peek(
        root, {"op": "peek", "session": "aaaa88889999", "query": "flaky .*budget", "regex": True}
    )
    assert not control.is_error, control.text
    assert (control.details or {})["match_id"] == f"{30:032x}"

    literal = await _peek(
        root, {"op": "peek", "session": "aaaa88889999", "query": "  flaky shard needs  "}
    )
    assert not literal.is_error, literal.text
    assert (literal.details or {})["match_id"] == f"{30:032x}"


def test_peek_scan_budget_matches_the_readers_cursor_window() -> None:
    """§8.1's "the same budget": the search's cap and the reader's own
    16 MiB cursor window are one number, pinned so they cannot drift apart."""
    from local_operator.session import transcript as transcript_module
    from local_operator.tools.builtin import _PEEK_SCAN_BYTES

    assert _PEEK_SCAN_BYTES == transcript_module._PAGE_LOCATE_WINDOW_BYTES


def test_peek_validation_refusals_are_legible() -> None:
    refusal = _sessions_validation_error(SessionsParams(op="peek", target="x", steps=5, head=3))
    assert refusal is not None and "one window at a time" in refusal
    refusal = _sessions_validation_error(
        SessionsParams(op="peek", target="x", steps=5, before_id="a")
    )
    assert refusal is not None and "one window at a time" in refusal
    refusal = _sessions_validation_error(SessionsParams(op="peek", target="x", regex=True))
    assert refusal == "`regex` needs `query`: it selects how the query matches."
    # R-1: the regex-without-query refusal is hoisted above the digest branch,
    # so the one shape that used to slip through — a digest folds no query —
    # is refused like every other regex-without-query call.
    refusal = _sessions_validation_error(
        SessionsParams(op="peek", target="x", digest=True, regex=True)
    )
    assert refusal == "`regex` needs `query`: it selects how the query matches."
    refusal = _sessions_validation_error(
        SessionsParams(op="peek", target="x", digest=True, steps=4)
    )
    assert refusal is not None and "drop `steps`" in refusal
    refusal = _sessions_validation_error(
        SessionsParams(op="peek", target="x", digest=True, query="q")
    )
    assert refusal is not None and "different reads" in refusal
    refusal = _sessions_validation_error(SessionsParams(op="peek", target="x", query="q", head=3))
    assert refusal is not None and "`steps=N` around the match" in refusal
    refusal = _sessions_validation_error(SessionsParams(op="peek", target="x", steps=0))
    assert refusal == "`steps` needs a count of at least 1."
    refusal = _sessions_validation_error(SessionsParams(op="peek", target="x", head=51))
    assert refusal is not None and "too many for one peek (max 50)" in refusal
    refusal = _sessions_validation_error(SessionsParams(op="peek", target="x", before_id="  "))
    assert refusal == "`before_id` needs the entry id an earlier peek returned."
    # The window fields are peek-only, and the refusal says so rather than
    # misdirecting the caller at another op.
    refusal = _sessions_validation_error(SessionsParams(op="info", target="x", steps=4))
    assert refusal is not None
    assert refusal.startswith("`steps` applies to op='peek' only. ")
    assert "`info` takes: session|target|pid, peer." in refusal


# ---------------------------------------------------------------------------
# The incident's contracts: the advertisement, the refusals, the receipt
# ---------------------------------------------------------------------------


def test_the_description_advertises_every_ops_accepted_set() -> None:
    """Text-vs-table drift guard, in BOTH directions.

    The per-op summary is DERIVED from ``_SESSIONS_OP_FIELDS``; this test
    re-derives the advertised set from the rendered text and demands it EQUAL
    the table's — a field added to one without the other, a name in the text
    no op takes, or an op missing from either, fails here instead of in a
    caller's retry loop (the incident was thirteen of those).
    """
    from typing import get_args

    literal_ops = tuple(get_args(SessionsParams.model_fields["op"].annotation))
    assert set(literal_ops) == set(_SESSIONS_OP_FIELDS)
    for op, fields in _SESSIONS_OP_FIELDS.items():
        match = re.search(
            rf"(?:^|; |— ){re.escape(op)}: ([^;.]+?)(?:;|\.)", _SESSIONS_TOOL_DESCRIPTION
        )
        assert match is not None, f"{op} is missing from the description's per-op summary"
        listed = {part for part in re.split(r"[|, ]+", match.group(1)) if part}
        expected = fields - {"op"}
        assert listed == expected or (not expected and listed == {"none"})


def test_the_reference_is_deterministic_bounded_and_never_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The renderer contract the audit lane's ``read tool://sessions`` serves:
    same bytes on every call, bounded, and a fallback instead of a raise."""

    first = render_sessions_reference()
    assert first == render_sessions_reference()
    assert len(first) <= 8 * 1024, "the shared reference must stay under its 8 KiB cap"
    for op in _SESSIONS_OP_FIELDS:
        assert f"`{op}` —" in first

    import local_operator.tools.builtin as builtin

    def boom() -> str:
        raise RuntimeError("boom")

    monkeypatch.setattr(builtin, "_sessions_reference_body", boom)
    fallback = render_sessions_reference()
    assert "reference unavailable" in fallback
    assert "sessions" in fallback


def test_the_reference_derives_the_peek_bound_from_the_constant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """M3 (agent review round 1): the one enumerable bound is generated.

    The renderer reads ``comms.PEEK_MAX_STEPS`` at call time, so a changed
    constant changes the help text — the drift this replaces was a hand copy
    that would have kept saying 50.
    """
    import local_operator.harness.comms as comms

    assert f"max {comms.PEEK_MAX_STEPS} steps" in render_sessions_reference()
    monkeypatch.setattr(comms, "PEEK_MAX_STEPS", 41)
    assert "max 41 steps" in render_sessions_reference()


@pytest.mark.asyncio
async def test_help_returns_the_reference_and_refuses_an_address() -> None:
    """``help``: read-tier, no address, and the same bytes as the renderer."""
    result = await execute_sessions("t", {"op": "help"}, None, None, None)
    assert not result.is_error, result.text
    assert result.text == render_sessions_reference()
    # Round 1 (NIT): the reference's refusal line must match behavior —
    # `include_stored` widens only the LOCAL half (refused beside
    # `scope='remote'` alone), while `query` is refused beside `peer` too.
    assert (
        "`include_stored` widens only the local half and is refused beside `scope='remote'`."
        in result.text
    )

    refusal = await execute_sessions("t", {"op": "help", "session": "a1"}, None, None, None)
    assert refusal.is_error
    assert refusal.text.startswith("`session` does not apply to op='help' — it takes no address. ")
    assert "`help` takes no extra params." in refusal.text
    assert refusal.text.endswith("Call op='help' for the full per-op reference.")


@pytest.mark.asyncio
async def test_a_stray_parameter_refusal_names_the_ops_accepted_set() -> None:
    """The incident's exact call: ``timeout_ms`` on ``resume``.

    The refusal must name the op's accepted set and the help route, in the
    message the model reads — not only the field that was wrong.
    """
    result = await execute_sessions(
        "t",
        {"op": "resume", "session": "x", "prompt": "y", "timeout_ms": 1},
        None,
        None,
        None,
    )
    assert result.is_error
    assert result.text == (
        "`timeout_ms` is not a sessions parameter. `resume` takes: "
        "session|target|pid, peer, prompt, background, paused, failed, all, dry_run, limit. "
        "Call op='help' for the full per-op reference."
    )
    assert (result.details or {}).get(FAULT_KEY) == FAULT_INVALID_ARGUMENTS

    # When the op itself is unusable the clause degrades to the op list rather
    # than guessing one op's set, and the other errors keep the generic lines.
    mixed = await execute_sessions("t", {"op": "bogus", "timeout_ms": 1}, None, None, None)
    assert mixed.is_error
    assert mixed.text.startswith("invalid arguments:")
    assert "The ops are list, info, spawn, resume, stop, peek, help." in mixed.text


@pytest.mark.asyncio
async def test_resume_receipt_claims_live_only_when_the_job_went_live(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A running ledger means the reopen receipt; the id alone does not."""
    _requester(root)
    _session(root, "aaaa11112222", "stopped once")

    async def fake_launch(argv: list[str], env: dict[str, str], cwd: str | None):
        return 0, "Background job cafebabe0001: running (execution receipt)\n"

    def fake_status(job_id: str, *, reconcile: bool = True) -> dict[str, Any]:
        return {
            "id": job_id,
            "status": "running",
            "session_id": "aaaa11112222",
            "log": str(root / "logs" / "exec-live.log"),
        }

    monkeypatch.setattr("local_operator.tools.builtin._sessions_launch", fake_launch)
    monkeypatch.setattr("local_operator.exec_mode.job_status", fake_status)
    monkeypatch.setattr("local_operator.tools.builtin.SESSIONS_READY_GRACE_S", 0.05)

    result = await execute_sessions(
        "t",
        {"op": "resume", "session": "aaaa11112222", "prompt": "continue"},
        None,
        None,
        _context(root),
    )

    assert not result.is_error, result.text
    assert result.text.startswith("reopened")
    assert (result.details or {}).get("readiness") == "live"


@pytest.mark.asyncio
async def test_a_resume_job_that_died_before_going_live_is_a_loud_error(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The silent no-op class, refused loudly: job dead, session never live.

    The answer must carry the worker's own last log line, the log path, and
    the CLI fallback for the exact session — not a "reopened" receipt.
    """
    _requester(root)
    _session(root, "aaaa11112222", "stopped once")
    log = root / "logs" / "exec-dead.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        "# local-operator exec background job\nRuntimeError: provider exploded\n",
        encoding="utf-8",
    )

    async def fake_launch(argv: list[str], env: dict[str, str], cwd: str | None):
        return 0, "Background job deadbeef0001: failed (execution receipt)\n"

    def fake_status(job_id: str, *, reconcile: bool = True) -> dict[str, Any]:
        return {"id": job_id, "status": "failed", "log": str(log)}

    monkeypatch.setattr("local_operator.tools.builtin._sessions_launch", fake_launch)
    monkeypatch.setattr("local_operator.exec_mode.job_status", fake_status)
    monkeypatch.setattr("local_operator.tools.builtin.SESSIONS_READY_GRACE_S", 0.05)

    result = await execute_sessions(
        "t",
        {"op": "resume", "session": "aaaa11112222", "prompt": "continue"},
        None,
        None,
        _context(root),
    )

    assert result.is_error
    assert "did not become a live session (failed)" in result.text
    assert "RuntimeError: provider exploded" in result.text
    assert str(log) in result.text
    assert "`lop exec --resume aaaa11112222 --background`" in result.text


@pytest.mark.asyncio
async def test_a_still_starting_resume_reports_starting_not_reopened(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No published session within the grace: the honest starting receipt."""
    _requester(root)
    _session(root, "aaaa11112222", "stopped once")

    async def fake_launch(argv: list[str], env: dict[str, str], cwd: str | None):
        return 0, "Background job feedface0001: starting (execution receipt)\n"

    def fake_status(job_id: str, *, reconcile: bool = True) -> dict[str, Any]:
        return {"id": job_id, "status": "starting"}

    monkeypatch.setattr("local_operator.tools.builtin._sessions_launch", fake_launch)
    monkeypatch.setattr("local_operator.exec_mode.job_status", fake_status)
    monkeypatch.setattr("local_operator.tools.builtin.SESSIONS_READY_GRACE_S", 0.05)

    result = await execute_sessions(
        "t",
        {"op": "resume", "session": "aaaa11112222", "prompt": "continue"},
        None,
        None,
        _context(root),
    )

    assert not result.is_error, result.text
    assert result.text.startswith("reopen requested")
    assert "still starting" in result.text
    assert "`lop exec --status feedface0001`" in result.text
    assert (result.details or {}).get("readiness") == "starting"


@pytest.mark.parametrize(
    ("status", "op", "expected_route"),
    [
        ("cancelled", "resume", "Retry with `lop exec --resume aaaa11112222 --background`"),
        ("interrupted", "resume", "Retry with `lop exec --resume aaaa11112222 --background`"),
        ("", "resume", "Retry with `lop exec --resume aaaa11112222 --background`"),
        ("failed", "spawn", "Follow up with `lop exec --status feedface0001`"),
        ("cancelled", "spawn", "Follow up with `lop exec --status feedface0001`"),
        ("", "spawn", "Follow up with `lop exec --status feedface0001`"),
    ],
)
@pytest.mark.asyncio
async def test_every_dead_or_missing_ledger_arm_is_loud_with_its_own_route(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    op: str,
    expected_route: str,
) -> None:
    """M2 (agent review round 1): every terminal/missing arm, both ops, pinned.

    ``failed``, live and starting have their own tests above; this table covers
    what they did NOT pin: ``cancelled``/``interrupted`` and the
    no-ledger-record wording, plus the spawn path's route — a status read is
    not a retry, so its verb differs (N1).
    """
    _requester(root)
    if op == "resume":
        _session(root, "aaaa11112222", "stopped once")
    log = root / "logs" / "exec-dead.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("ValueError: worker exploded\n", encoding="utf-8")

    async def fake_launch(argv: list[str], env: dict[str, str], cwd: str | None):
        return 0, "Background job feedface0001: starting (execution receipt)\n"

    def fake_status(job_id: str, *, reconcile: bool = True) -> dict[str, Any]:
        return {"id": job_id, "status": status, "log": str(log)}

    monkeypatch.setattr("local_operator.tools.builtin._sessions_launch", fake_launch)
    monkeypatch.setattr("local_operator.exec_mode.job_status", fake_status)
    monkeypatch.setattr("local_operator.tools.builtin.SESSIONS_READY_GRACE_S", 0.05)

    params: dict[str, Any] = {"op": op, "prompt": "continue"}
    if op == "resume":
        params["session"] = "aaaa11112222"
    result = await execute_sessions("t", params, None, None, _context(root))

    expected_outcome = status or "no ledger record"
    assert result.is_error
    assert f"did not become a live session ({expected_outcome})" in result.text
    assert "ValueError: worker exploded" in result.text
    assert expected_route in result.text
    assert "`lop sessions`" in result.text


@pytest.mark.asyncio
async def test_resume_of_a_stopped_session_really_relaunches(root: Path) -> None:
    """The before/after proof against the REAL CLI and ledger, in-process.

    "Stopped" maps onto this codebase's own vocabulary (no runtime record; the
    stored row's ``state: "stored"``): the run must go live on the ledger and
    the transcript must GROW — a receipt alone is not a relaunch.
    """
    from local_operator.exec_mode import job_status

    _write_config(root)
    _requester(root)
    directory = _session(root, "resumed00001", "stopped once")
    before = (directory / "transcript.jsonl").read_text(encoding="utf-8").count("\n")
    try:
        result = await execute_sessions(
            "t",
            {"op": "resume", "session": "resumed00001", "prompt": "continue"},
            None,
            None,
            _context(root),
        )
        assert not result.is_error, result.text
        job = str((result.details or {})["job_id"])

        deadline = time.monotonic() + 120.0
        grew = False
        state: dict[str, Any] = {}
        while time.monotonic() < deadline:
            state = job_status(job)
            after = (directory / "transcript.jsonl").read_text(encoding="utf-8").count("\n")
            grew = grew or after > before
            status = str(state.get("status") or "")
            if status in ("failed", "cancelled", "interrupted"):
                break
            if grew and status in ("running", "succeeded", "completed"):
                break
            await asyncio.sleep(0.5)

        assert grew, "the resumed run wrote nothing to the transcript; ledger: " + repr(state)
        assert str(state.get("status")) in ("running", "succeeded", "completed"), state
    finally:
        await _reap_worker(root, "resumed00001")


@pytest.mark.asyncio
async def test_resume_resolves_an_archived_session_by_exact_id(root: Path) -> None:
    """Archive is the nearest real state to the operator's "disposed": hidden
    from LISTINGS, and the archive contract keeps exact-id resolution — resume
    through the tool must honor that, and the receipt must not claim more."""
    _write_config(root)
    _requester(root)
    _session(root, "archived0009", "filed away")
    assert set_archived(root, "archived0009", True)
    try:
        result = await execute_sessions(
            "t",
            {"op": "resume", "session": "archived0009", "prompt": "continue"},
            None,
            None,
            _context(root),
        )
        assert not result.is_error, result.text
        assert "reopen" in result.text
    finally:
        await _reap_worker(root, "archived0009")


# --- bulk resume: the SET form of `resume` (2026-10-01) ----------------------


@pytest.mark.asyncio
async def test_a_set_selection_refuses_an_address(root: Path) -> None:
    """A set and an address are two different requests; composing is a typo."""
    result = await execute_sessions(
        "t",
        {"op": "resume", "paused": True, "session": "aaaa11112222"},
        None,
        None,
        _context(root),
    )
    assert result.is_error
    assert "a set selection takes no `session`" in result.text


@pytest.mark.asyncio
async def test_set_fields_are_refused_on_other_ops(root: Path) -> None:
    result = await execute_sessions("t", {"op": "list", "paused": True}, None, None, _context(root))
    assert result.is_error
    assert "`paused` applies to op='resume' only." in result.text


@pytest.mark.asyncio
async def test_dry_run_needs_a_set_selection(root: Path) -> None:
    result = await execute_sessions(
        "t",
        {"op": "resume", "session": "aaaa11112222", "prompt": "go", "dry_run": True},
        None,
        None,
        _context(root),
    )
    assert result.is_error
    assert "`dry_run` needs a set selection" in result.text


@pytest.mark.asyncio
async def test_a_set_selection_needs_no_prompt_but_rejects_a_blank_one(root: Path) -> None:
    # A blank prompt IS refused — an explicit empty message is a misspelled
    # intent, not "use the default".
    blank = await execute_sessions(
        "t",
        {"op": "resume", "paused": True, "prompt": "   "},
        None,
        None,
        _context(root),
    )
    assert blank.is_error
    assert "must be a non-empty message when given" in blank.text
    # With no prompt at all it reaches execution; an empty store answers
    # "nothing started" (not an error) — the mock-free half of the contract.
    empty = await execute_sessions(
        "t", {"op": "resume", "paused": True}, None, None, _context(root)
    )
    assert not empty.is_error
    assert "no stored sessions match" in empty.text


def test_the_batch_approval_description_names_the_sets_and_the_commitment(
    root: Path,
) -> None:
    described = _describe_sessions_approval(
        {"op": "resume", "paused": True, "failed": True, "limit": 10}, str(root)
    )
    assert described == (
        "resume the paused + failed session set, cap 10: starts one headless "
        "run per selected session"
    )
    dry = _describe_sessions_approval({"op": "resume", "all": True, "dry_run": True}, str(root))
    assert dry == "review the all stored session set: resumes nothing (dry run)"
    # All wins (review round 1, m3): the approval names the selection the
    # selector actually makes, not the combination that was passed.
    combined = _describe_sessions_approval({"op": "resume", "paused": True, "all": True}, str(root))
    assert combined == (
        "resume the all stored session set: starts one headless run per selected session"
    )
    # The single form's sentence is untouched.
    single = _describe_sessions_approval(
        {"op": "resume", "session": "aaaa11112222", "prompt": "go"}, str(root)
    )
    assert single == "resume session aaaa11112222: reopens its transcript headlessly"


@pytest.mark.asyncio
async def test_the_batch_summary_renders_per_session_outcomes(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The set form renders one line per session plus the counts and details."""

    def fake_select(*args: Any, **kwargs: Any) -> ResumeSelection:
        return ResumeSelection(
            sessions=(("sess-ok", 1.0), ("sess-bad", 2.0)), matched=5, kinds=frozenset()
        )

    async def fake_run(rows: Any, **kwargs: Any) -> list[ResumeOutcome]:
        assert [r[0] for r in rows] == ["sess-ok", "sess-bad"]
        return [
            ResumeOutcome(
                session_id="sess-ok", name="ok one", ok=True, status="running", job_id="j1"
            ),
            ResumeOutcome(
                session_id="sess-bad",
                name="bad one",
                ok=False,
                status="failed",
                job_id="j2",
                detail="worker died: model exploded",
            ),
        ]

    monkeypatch.setattr(bulk_resume, "select_resume_candidates", fake_select)
    monkeypatch.setattr(bulk_resume, "live_session_ids", lambda root: set())
    monkeypatch.setattr(bulk_resume, "resume_sessions", fake_run)

    result = await execute_sessions(
        "t", {"op": "resume", "paused": True, "limit": 3}, None, None, _context(root)
    )
    assert not result.is_error, result.text
    assert "2 session(s) — 1 ok, 1 failed, 0 unresolved" in result.text
    assert '- ok    sess-ok  "ok one"  (running, job j1)' in result.text
    assert '- FAIL  sess-bad  "bad one"  — worker died: model exploded' in result.text
    assert "newest 2 of 5 matched" in result.text
    details = result.details or {}
    assert details["batch"] is True and details["count"] == 2
    assert details["sessions"][1]["session_id"] == "sess-bad"


@pytest.mark.asyncio
async def test_a_dry_run_beside_all_names_the_all_selection(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """m3 (review round 1): the tool's dry-run text follows the selection."""

    def fake_select(*args: Any, **kwargs: Any) -> ResumeSelection:
        return ResumeSelection(sessions=(("sess-x", 1.0),), matched=1, kinds=frozenset())

    monkeypatch.setattr(bulk_resume, "select_resume_candidates", fake_select)
    monkeypatch.setattr(bulk_resume, "live_session_ids", lambda root: set())

    result = await execute_sessions(
        "t",
        {"op": "resume", "failed": True, "all": True, "dry_run": True},
        None,
        None,
        _context(root),
    )
    assert not result.is_error, result.text
    assert "resume set (all)" in result.text
    assert "failed+all" not in result.text


@pytest.mark.asyncio
async def test_resume_set_form_really_reopens_each_session(root: Path) -> None:
    """The REAL path: two paused sessions, one `resume(paused=True)` call.

    Real nested CLI children on mock hosting, like the single-resume tests:
    the tool's job is to hand each child the right argv and environment, and
    only a real run proves the whole batch wiring.
    """
    _write_config(root)
    _requester(root)
    for session_id, name in (("bulk00000001", "bulk one"), ("bulk00000002", "bulk two")):
        _session(root, session_id, name)
    store = AttentionStore(root / "attention.db")
    for session_id in ("bulk00000001", "bulk00000002"):
        store.publish(
            f"session/{session_id}",
            str(uuid.uuid4()),
            f"e-{session_id}",
            "interrupted",
            reason="seed",
        )
    try:
        result = await execute_sessions(
            "t", {"op": "resume", "paused": True}, None, None, _context(root)
        )
        assert not result.is_error, result.text
        details = result.details or {}
        assert details.get("batch") is True
        assert details.get("count") == 2, result.text
        assert details.get("ok") == 2, result.text
    finally:
        for session_id in ("bulk00000001", "bulk00000002"):
            await _reap_worker(root, session_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("op", ["info", "resume", "stop", "peek"])
async def test_a_role_word_address_is_refused_for_every_target_op(
    op: str, root: Path, tmp_path: Path
) -> None:
    """All four address-bearing ops inherit the guard.

    ``stop`` is the one that makes this a SAFETY pin rather than a nicety: the
    substring tier would have ended the unrelated session whose title merely
    contains the word. ``info``/``peek`` would have described it as the target;
    ``resume`` would have reopened it.
    """
    from local_operator.teams import TeamEditFields, TeamRegistry

    teams = TeamRegistry(tmp_path)
    teams.create_team(TeamEditFields(name="lopdev", manager="manager"))
    record = _publish_record(root, "dddd77778888", "Article-search campaigns: manager")
    try:
        args: dict[str, Any] = {"op": op, "target": "manager"}
        if op == "resume":
            args["prompt"] = "go"
        result = await execute_sessions("t", args, None, None, _context(root, team_registry=teams))
        assert result.is_error
        assert "is a team role (roles on: lopdev), not a session address" in result.text
        # Not the no-match form: that phrasing would send the caller to the
        # stored half, where a namesake could be acted on instead.
        assert "no live session matches" not in result.text
        assert "no session matches" not in result.text
    finally:
        registry.unpublish(record.pid, root)


# ---------------------------------------------------------------------------
# PR-B: the mesh — another device's sessions (design §3C, §5, §9)
# ---------------------------------------------------------------------------

PEER = "cloud-node-1"


def _fed_doc(**overrides: Any) -> dict[str, Any]:
    """A family ``--json`` catalogue document, in the CLI's own shape.

    The local-locality row is the half the tool MUST drop (it is this device's
    own catalogue, already in hand); the remote rows carry the ``locality`` and
    ``peer`` markers the federated listing pins.
    """
    block = {"device_id": "d_b", "name": PEER, "reachable": True, "reason": ""}
    doc: dict[str, Any] = {
        "ok": True,
        "sessions": [
            {
                "session_id": "dddd77778888",
                "conversation_name": "fed-local",
                "state": "live",
                "locality": "local",
                "peer": None,
            },
            {
                "session_id": "aaaa11112222",
                "conversation_name": "shared-one",
                "state": "live",
                "started": 10.0,
                "locality": "remote",
                "peer": block,
            },
            {
                "session_id": "bbbb33334444",
                "conversation_name": "second",
                "state": "stored",
                "started": 5.0,
                "locality": "remote",
                "peer": block,
            },
        ],
        "peers": {"d_b": block},
        "device_name": "this-device",
    }
    doc.update(overrides)
    return doc


def _fake_mesh(
    monkeypatch: pytest.MonkeyPatch,
    payload: Any,
    *,
    rc: int = 0,
    calls: list[tuple[list[str], float]] | None = None,
) -> None:
    """Replace the ONE process boundary a mesh op has: the child call.

    ``_mesh_cli_json`` is where the subprocess ends and the parsed document
    begins, so a double here exercises every line of the tool's own logic —
    argv construction included, since the fake receives what the tool spelled.
    """

    async def fake(argv: list[str], timeout: float, context: object | None = None):
        if calls is not None:
            calls.append((argv, timeout))
        return rc, payload, ""

    monkeypatch.setattr("local_operator.tools.builtin._mesh_cli_json", fake)


@pytest.mark.asyncio
async def test_list_is_a_union_of_local_and_remote_rows_with_the_device_on_each(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Design §3C: `list` defaults to local+remote, drops the federated
    document's own-locality rows, and marks every remote row with the device
    that holds it."""
    _session(root, "cccc55556666", "local-stored")
    calls: list[tuple[list[str], float]] = []
    _fake_mesh(monkeypatch, _fed_doc(), calls=calls)
    result = await execute_sessions(
        "t", {"op": "list", "include_stored": True, "limit": 10}, None, None, _context(root)
    )
    assert not result.is_error, result.text
    assert "[stored] local-stored" in result.text
    assert "[live] shared-one" in result.text and "on cloud-node-1" in result.text
    # The document's own-locality row is this device's catalogue, already in
    # hand — merging it would list local sessions twice.
    assert "fed-local" not in result.text
    details = result.details or {}
    assert details["scope"] == "all"
    assert [row["session_id"] for row in details["rows"]] == [
        "cccc55556666",
        "aaaa11112222",
        "bbbb33334444",
    ]
    assert calls == [(["network", "sessions", "--json", "--all-peers"], 30.0)]


@pytest.mark.asyncio
async def test_list_scope_remote_and_peer_narrow_the_fetch(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _session(root, "cccc55556666", "local-stored")
    calls: list[tuple[list[str], float]] = []
    _fake_mesh(monkeypatch, _fed_doc(), calls=calls)
    result = await execute_sessions(
        "t",
        {"op": "list", "scope": "remote", "peer": PEER, "limit": 10},
        None,
        None,
        _context(root),
    )
    assert not result.is_error, result.text
    assert "local-stored" not in result.text
    assert "shared-one" in result.text
    assert calls == [(["network", "sessions", "--json", "--peer", PEER], 30.0)]


@pytest.mark.asyncio
async def test_list_with_no_mesh_degrades_to_local_rows_and_one_note(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Design §3C: a relay that cannot be asked is ONE note on what WAS
    readable, never an empty answer dressed as a complete one — and never an
    error, because the local half is a real result."""
    _session(root, "cccc55556666", "local-stored")
    _fake_mesh(
        monkeypatch,
        {
            "ok": False,
            "code": "relay_unavailable",
            "message": "the relay is not running; start it with `lop network start`",
        },
        rc=1,
    )
    result = await execute_sessions(
        "t", {"op": "list", "include_stored": True}, None, None, _context(root)
    )
    assert not result.is_error, result.text
    assert "local-stored" in result.text
    assert result.text.count("no remote rows") == 1
    assert "lop network start" in result.text
    # scope='remote' has no local half, so the note IS the answer — still one.
    result = await execute_sessions(
        "t", {"op": "list", "scope": "remote"}, None, None, _context(root)
    )
    assert not result.is_error, result.text
    assert "no remote rows" in result.text
    assert "shared-one" not in result.text


@pytest.mark.asyncio
async def test_list_names_unreachable_peers_and_caps_the_merged_union(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mesh half may not unbounded-grow the listing (the cap applies
    POST-merge), and a device that did not answer is named rather than
    silently dropped."""
    doc = _fed_doc()
    doc["peers"]["d_c"] = {
        "device_id": "d_c",
        "name": "old-box",
        "reachable": False,
        "reason": "connect_failed:ConnectionRefusedError",
    }
    _fake_mesh(monkeypatch, doc)
    result = await execute_sessions(
        "t", {"op": "list", "scope": "remote", "limit": 1}, None, None, _context(root)
    )
    assert not result.is_error, result.text
    assert "old-box: unreachable" in result.text
    assert "2 rows available; 1 shown" in result.text
    assert len((result.details or {})["rows"]) == 1


@pytest.mark.asyncio
async def test_remote_info_resolves_a_name_or_id_and_refuses_ambiguity(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§3C: info selects via the shared resolver; an ambiguous name lists the
    candidate rows rather than guessing, and a miss says what IS held."""
    _fake_mesh(monkeypatch, _fed_doc())
    result = await execute_sessions(
        "t", {"op": "info", "peer": PEER, "target": "shared"}, None, None, _context(root)
    )
    assert not result.is_error, result.text
    assert "shared-one" in result.text and "held by cloud-node-1" in result.text
    assert (result.details or {})["session_id"] == "aaaa11112222"

    result = await execute_sessions(
        "t", {"op": "info", "peer": PEER, "session": "bbbb33334444"}, None, None, _context(root)
    )
    assert not result.is_error, result.text
    assert "second" in result.text

    doc = _fed_doc()
    doc["sessions"].append(
        {
            "session_id": "eeee99990000",
            "conversation_name": "shared-two",
            "state": "live",
            "started": 1.0,
            "locality": "remote",
            "peer": {"device_id": "d_b", "name": PEER, "reachable": True, "reason": ""},
        }
    )
    _fake_mesh(monkeypatch, doc)
    result = await execute_sessions(
        "t", {"op": "info", "peer": PEER, "target": "shared"}, None, None, _context(root)
    )
    assert result.is_error
    assert "2 sessions matching" in result.text
    assert "aaaa11112222" in result.text and "eeee99990000" in result.text

    result = await execute_sessions(
        "t", {"op": "info", "peer": PEER, "target": "nope"}, None, None, _context(root)
    )
    assert result.is_error and "does not hold" in result.text


@pytest.mark.asyncio
async def test_remote_peek_reads_a_tail_and_relays_the_stored_gate(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§3C with §8.3: the tail window where the session LIVES; a stored
    session's refusal (raised by the CLI BEFORE any bind) arrives verbatim,
    already naming the warm-up."""
    calls: list[tuple[list[str], float]] = []
    _fake_mesh(
        monkeypatch,
        {
            "ok": True,
            "session_id": "aaaa11112222",
            "peer": PEER,
            "verb": "peek",
            "steps": 2,
            "rows": [
                {"role": "user", "text": "hi"},
                {"role": "assistant", "text": "there"},
            ],
            "has_older": True,
        },
        calls=calls,
    )
    result = await execute_sessions(
        "t",
        {"op": "peek", "peer": PEER, "session": "aaaa11112222", "steps": 2},
        None,
        None,
        _context(root),
    )
    assert not result.is_error, result.text
    assert "1. user: hi" in result.text and "2. assistant: there" in result.text
    assert calls == [
        (
            [
                "network",
                "sessions",
                "--json",
                "--peer",
                PEER,
                "--peek",
                "aaaa11112222",
                "--steps",
                "2",
            ],
            net_pilot_bound(),
        )
    ]

    _fake_mesh(
        monkeypatch,
        {
            "ok": False,
            "code": "session_stored",
            "message": (
                "aaaa11112222 is stored on cloud-node-1, so there is nothing running "
                "to read — a peek never starts a runtime there. Warm it first: `lop "
                "network sessions --peer cloud-node-1 --engage aaaa11112222` (an "
                "agent resumes with `sessions` op='resume'), then peek."
            ),
        },
        rc=1,
    )
    result = await execute_sessions(
        "t", {"op": "peek", "peer": PEER, "session": "aaaa11112222"}, None, None, _context(root)
    )
    assert result.is_error
    assert "--engage" in result.text and "op='resume'" in result.text


def net_pilot_bound() -> float:
    """The pilot act's bound, read from the surface under test (one home)."""
    from local_operator.tools import builtin

    return builtin._mesh_pilot_timeout_s()


@pytest.mark.asyncio
async def test_remote_ops_map_to_the_pilot_argv_and_bounds(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """send/engage/stop/create — each op's exact spelling and §3A bound,
    captured at the process boundary the tool owns."""
    calls: list[tuple[list[str], float]] = []

    async def fake(argv: list[str], timeout: float, context: object | None = None):
        calls.append((argv, timeout))
        if "--send" in argv:
            return (
                0,
                {
                    "session_id": "s1",
                    "peer": PEER,
                    "verb": "send",
                    "ok": True,
                    "outcome": "finished",
                    "reply": "r",
                },
                "",
            )
        if "--engage" in argv:
            return 0, {"ok": True, "engaged": True, "detail": "runtime joining"}, ""
        if "--stop" in argv:
            return 0, {"ok": True, "outcome": "stopped", "detail": "stopped (pid 4)"}, ""
        return 0, {"ok": True, "session_id": "s_new", "admitted": True}, ""

    monkeypatch.setattr("local_operator.tools.builtin._mesh_cli_json", fake)
    await execute_sessions(
        "t",
        {"op": "resume", "peer": PEER, "session": "s1", "prompt": "go"},
        None,
        None,
        _context(root),
    )
    await execute_sessions(
        "t", {"op": "resume", "peer": PEER, "session": "s1"}, None, None, _context(root)
    )
    await execute_sessions(
        "t", {"op": "stop", "peer": PEER, "session": "s1"}, None, None, _context(root)
    )
    await execute_sessions(
        "t",
        {"op": "spawn", "peer": PEER, "prompt": "go", "model": "anthropic/claude-sonnet-5-5"},
        None,
        None,
        _context(root),
    )
    assert calls == [
        (["network", "sessions", "--json", "--peer", PEER, "--send", "s1", "--", "go"], 600.0),
        (["network", "sessions", "--json", "--peer", PEER, "--engage", "s1"], 180.0),
        (["network", "sessions", "--json", "--peer", PEER, "--stop", "s1"], 300.0),
        (
            [
                "network",
                "sessions",
                "--json",
                "--peer",
                PEER,
                "--create",
                "--prompt",
                "go",
                "--hosting",
                "anthropic",
                "--model",
                "claude-sonnet-5-5",
            ],
            180.0,
        ),
    ]
    # A bare model id cannot be split; refused BEFORE any child exists.
    result = await execute_sessions(
        "t",
        {"op": "spawn", "peer": PEER, "prompt": "go", "model": "bare"},
        None,
        None,
        _context(root),
    )
    assert result.is_error and "<provider>/<model-id>" in result.text
    assert len(calls) == 4


@pytest.mark.asyncio
async def test_remote_stop_names_the_cli_force_escalation_on_skipped(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§8.4: no force in the tool — the one outcome whose remedy is a flag
    names the CLI escalation instead."""
    _fake_mesh(
        monkeypatch,
        {"ok": False, "outcome": "skipped", "detail": "A turn is in flight there."},
        rc=1,
    )
    result = await execute_sessions(
        "t", {"op": "stop", "peer": PEER, "session": "s1"}, None, None, _context(root)
    )
    assert result.is_error  # the ladder did not end it
    assert "lop network sessions --peer cloud-node-1 --stop s1 --force" in result.text


def test_mesh_field_refusals_name_the_local_route() -> None:
    """The remote/local composition refusals: each names the field that does
    not apply and the route that does, rather than dropping it silently."""
    refusal = _sessions_validation_error(
        SessionsParams(op="peek", peer="p", session="s", query="q")
    )
    assert refusal is not None
    assert (
        "reads THIS device's transcript" in refusal and "drop `peer` for the local read" in refusal
    )
    refusal = _sessions_validation_error(SessionsParams(op="list", peer="p", scope="local"))
    assert refusal is not None and "drop one" in refusal
    refusal = _sessions_validation_error(SessionsParams(op="list", scope="remote", query="q"))
    assert refusal is not None and "list with `scope='local'`" in refusal
    refusal = _sessions_validation_error(SessionsParams(op="resume", peer="p", all=True))
    assert refusal is not None and "set form" in refusal
    refusal = _sessions_validation_error(
        SessionsParams(op="spawn", peer="p", prompt="go", visibility="workstream")
    )
    assert refusal is not None and "minted on that device" in refusal
    refusal = _sessions_validation_error(SessionsParams(op="info", peer="p", session="s", pid=3))
    assert refusal is not None and "`pid` names a process on THIS machine" in refusal


# ---------------------------------------------------------------------------
# the peer hint — a bare remote id names the peer that holds it
# ---------------------------------------------------------------------------
#
# The same fix-forward as the send tool's (see tests/unit/tools/test_send_tool.py
# for the full arm matrix — warm/cold reads, no mesh, unreadable relay, unknown
# id). info/peek/stop/resume resolve through the ONE `_sessions_target`, so all
# four inherit the hint at one call site; these cells pin the inheritance and
# the guards the ops side owns: the warm cache answers with no read, a NAME
# miss stays a local question, and the local success path pays nothing.


def _hint_peer_row(session_id: str = "ffff12345678", *, device_name: str = "cloud-node-1"):
    from local_operator.resume import SessionRow

    return SessionRow(
        session_id,
        1.0,
        "remote work",
        locality="remote",
        owner_device="d_cloud_node_1",
        owner_device_name=device_name,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("op", ["info", "peek", "stop", "resume"])
async def test_a_bare_remote_id_names_the_peer_on_every_addressed_op(
    root: Path, monkeypatch: pytest.MonkeyPatch, op: str
) -> None:
    """The four ops that take an address and an optional `peer` all refuse
    through `_sessions_target`; the exact-id miss now names the device that
    holds the id, on every one of them. The warm cache must answer — the read
    is poisoned, matching the send tool's zero-read guard."""
    from local_operator.paths import config_dir
    from local_operator.session import peer_rows as peer_rows_mod
    from local_operator.session.peer_rows import clear_cache, seed_peer_row

    clear_cache()
    seed_peer_row(config_dir(), _hint_peer_row())

    def _forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("a warm cache hit paid a listing read")

    monkeypatch.setattr(peer_rows_mod, "peer_session_rows", _forbidden)

    args: dict[str, Any] = {"op": op, "session": "ffff12345678"}
    if op == "resume":
        # A local resume requires a prompt by validation; the miss must still be
        # resolved (and named) BEFORE anything would be launched.
        args["prompt"] = "go"
    result = await execute_sessions("t", args, None, None, _context(root))
    assert result.is_error
    assert result.text == "`ffff12345678` is held by cloud-node-1 — pass `peer=cloud-node-1`"


@pytest.mark.asyncio
async def test_an_unknown_exact_id_keeps_the_resolver_sentence(root: Path) -> None:
    """No relay on this device, so no catalogue is asked and nothing moves: the
    resolver's own sentence stands — the exact sentence the defect reported."""
    from local_operator.session.peer_rows import clear_cache

    clear_cache()
    result = await execute_sessions(
        "t", {"op": "info", "session": "ffff99998888"}, None, None, _context(root)
    )
    assert result.is_error
    assert result.text == "no session found with session id 'ffff99998888'"


@pytest.mark.asyncio
async def test_a_locally_resolved_info_pays_no_peer_read(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ops half of the zero-read guard: a session THIS device holds resolves
    from the registry with both peer-catalogue entry points poisoned."""
    from local_operator.session import peer_rows as peer_rows_mod

    _publish_record(root, "aaaa11112222", "local work")

    def _forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("a local resolution paid a peer read")

    monkeypatch.setattr(peer_rows_mod, "peer_session_row", _forbidden)
    monkeypatch.setattr(peer_rows_mod, "peer_session_rows", _forbidden)
    result = await execute_sessions(
        "t", {"op": "info", "session": "aaaa11112222"}, None, None, _context(root)
    )
    assert not result.is_error, result.text
    assert "local work" in result.text


@pytest.mark.asyncio
async def test_a_name_miss_stays_a_local_question(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exact FULL-ID matching only: a name that matches nothing locally is not
    searched against the peer catalogue — a name can match sessions on two
    devices, and a diagnostic sentence must not pick between them. Both
    catalogue entry points are poisoned, so any lookup fails the cell."""
    from local_operator.paths import config_dir
    from local_operator.session import peer_rows as peer_rows_mod
    from local_operator.session.peer_rows import clear_cache, seed_peer_row

    clear_cache()
    seed_peer_row(config_dir(), _hint_peer_row(session_id="aaaa11112222"))

    def _forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("a name miss asked the peer catalogue")

    monkeypatch.setattr(peer_rows_mod, "peer_session_row", _forbidden)
    monkeypatch.setattr(peer_rows_mod, "peer_session_rows", _forbidden)

    result = await execute_sessions(
        "t", {"op": "info", "target": "remote work"}, None, None, _context(root)
    )
    assert result.is_error
    assert result.text == "no session matches 'remote work' (searched live and stored sessions)"


@pytest.mark.asyncio
@pytest.mark.parametrize("op", ["info", "peek", "stop", "resume"])
async def test_a_full_id_target_names_the_peer_on_every_addressed_op(
    root: Path, monkeypatch: pytest.MonkeyPatch, op: str
) -> None:
    """ROUND-1 EXTEND: the `target=<full id>` spelling fires on all four ops
    exactly as `session=` does — the same shared resolver, the same warm-cache
    zero-read property (the read is poisoned)."""
    from local_operator.paths import config_dir
    from local_operator.session import peer_rows as peer_rows_mod
    from local_operator.session.peer_rows import clear_cache, seed_peer_row

    clear_cache()
    seed_peer_row(config_dir(), _hint_peer_row())

    def _forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("a warm cache hit paid a listing read")

    monkeypatch.setattr(peer_rows_mod, "peer_session_rows", _forbidden)

    args: dict[str, Any] = {"op": op, "target": "ffff12345678"}
    if op == "resume":
        # A local resume requires a prompt by validation; the miss must still
        # be resolved (and named) BEFORE anything would be launched.
        args["prompt"] = "go"
    result = await execute_sessions("t", args, None, None, _context(root))
    assert result.is_error
    assert result.text == "`ffff12345678` is held by cloud-node-1 — pass `peer=cloud-node-1`"
