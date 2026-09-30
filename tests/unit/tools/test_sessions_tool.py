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
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from local_operator import session_lease
from local_operator.agent_shell import AGENT_SHELL_ENV, MAY_DELEGATE_ENV
from local_operator.harness.types import ToolContext
from local_operator.resume import (
    ORIGIN_AGENT_SHELL,
    ORIGIN_AGENT_WORKSTREAM,
    ORIGIN_SUBAGENT,
    is_user_session,
    mark_session_origin,
    write_session_title,
)
from local_operator.scratchpad import SCRATCHPAD_PATH_ENV
from local_operator.session.archived import set_archived
from local_operator.session.runtime import control, registry
from local_operator.session.runtime.types import SessionRecord
from local_operator.tools.builtin import (
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
    assert refusal == "`prompt` applies to spawn/resume only."
    refusal = _sessions_validation_error(SessionsParams(op="list", session="a"))
    assert refusal == "`session` does not apply to op='list' — it takes no address."
    refusal = _sessions_validation_error(SessionsParams(op="spawn", prompt="go", target="x"))
    assert refusal is not None and "creates a new session" in refusal
    refusal = _sessions_validation_error(SessionsParams(op="info"))
    assert refusal is not None and "exactly one of" in refusal


def test_schema_budget_is_measured_with_the_repos_own_ruler() -> None:
    """The design note's §3.4 budget, guarded so a later field cannot drift it.

    Ruler: ``compaction/tokens.count_text_tokens`` (cl100k_base via tiktoken,
    the repo's estimator — the same ruler the design measured the draft with);
    subject: the exact JSON the provider sees as the tool's ``parameters``.
    """
    from local_operator.compaction.tokens import count_text_tokens
    from local_operator.tools.builtin import _SESSIONS_TOOL_DESCRIPTION

    params = json.dumps(SessionsParams.model_json_schema(), ensure_ascii=False)
    assert count_text_tokens(params) <= 700
    assert count_text_tokens(_SESSIONS_TOOL_DESCRIPTION) <= 200
