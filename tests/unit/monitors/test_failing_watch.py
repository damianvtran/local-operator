"""A watch whose call FAILS must say so — never diff two identical errors.

THE INCIDENT THIS FILE PINS. A bash monitor watched a PR's CI with
``gh pr view <n> --json statusCheckRollup,mergeStateStatus`` every 5 minutes.
The ticks ran in the arming session's directory, which was not a git
repository, so every tick failed with ``failed to run git: fatal: not a git
repository`` — 18 of them over 90 minutes, while the PR went from 9 pending to
40/40 green including the merge gate. Nothing was ever delivered, and the list
footer said "nothing has changed (confirm the call observes what you expect)".

WHY IT WAS SILENT, and therefore what these tests must prove:
``execute_bash`` reports a NON-ZERO EXIT as an ORDINARY SUCCESS result (its text
starts ``exit code: 1`` and ``is_error`` is False, because the model's contract
depends on that). So ``Session._run_monitor_check`` returned an ordinary
``{"text": "exit code: 1\\nfailed to run git: ..."}``, and the scheduler diffed
two identical error texts: no change, no delivery, no strike, no notice. The
fix keeps the bash contract and closes the silence at the monitor seam:

- the tool publishes the exit status as STRUCTURED data (``exit_code``,
  ``stderr_head``, ``no_output``), so the outcome is a field read, never a parse
  of the framed body;
- a non-zero exit WITH a stderr complaint is a ``failing`` episode: one notice
  per episode, on the FIRST failed check if the watch never worked (it is
  broken, not blipping) and on the SECOND consecutive one if it has;
- a cwd the watch needs can be NAMED at arm (``cwd``), and arming probes the
  call once so a watch that cannot work is announced in the receipt;
- the list footer says the watch is failing instead of "nothing has changed".

THE COUNTER-ATTACK these tests also pin: a WORKING watch must stay silent. One
transient failure does not page, an empty result is not a failure, and a quiet
tick produces nothing at all.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import ModelSpec, StreamEndEvent, ToolContext
from local_operator.monitors import store as monitor_store
from local_operator.monitors.delivery import MonitorNotice, format_monitor_notice_text
from local_operator.monitors.scheduler import (
    EMPTY_CHECKS_BEFORE_NOTICE,
    FAILING_CHECKS_AFTER_BASELINE,
    MonitorScheduler,
)
from local_operator.monitors.settings import MonitorSettings
from local_operator.monitors.spec import MonitorSpec
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.tools import builtin
from local_operator.tools.builtin import build_bash_tool
from tests.unit.monitors.support import drain_checks

MODEL = ModelSpec(provider="test", model_id="m", context_window=100_000)
NOW = 1_756_000_000_000
MINUTE = 60_000
#: The incident's own failure text, asserted by shape rather than by exact
#: wording: git's phrase is stable across versions and platforms.
NOT_A_REPO = "not a git repository"


def _stream(request: Any, signal: Any):  # noqa: ANN202
    async def gen():
        yield StreamEndEvent(stop_reason="stop")

    return gen()


@pytest.fixture
def config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "cfg"
    root.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    return root


def _git_env() -> dict[str, str]:
    """A git env that cannot reach a credential store.

    Two reasons, both measured elsewhere on this host: a test that runs under an
    isolated ``HOME`` must not make any process ask macOS for a login keychain,
    and a commit-creating test must not depend on the operator's git identity.
    """
    env = {key: value for key, value in os.environ.items() if not key.startswith(("CMUX_", "LOP_"))}
    env.update(
        {
            "GIT_CONFIG_SYSTEM": "/dev/null",
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_AUTHOR_NAME": "monitor test",
            "GIT_AUTHOR_EMAIL": "monitor@test.invalid",
            "GIT_COMMITTER_NAME": "monitor test",
            "GIT_COMMITTER_EMAIL": "monitor@test.invalid",
        }
    )
    return env


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        timeout=60,
        env=_git_env(),
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    return completed.stdout.strip()


def _git_repo(root: Path) -> Path:
    """A real git repository with one commit."""
    repo = root / "repo"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main")
    (repo / "file.txt").write_text("one\n", encoding="utf-8")
    _git(repo, "add", "file.txt")
    _git(repo, "commit", "-q", "-m", "first")
    return repo


def _not_a_repo(root: Path) -> Path:
    """A directory that is provably not inside a git repository.

    Self-validating: if the host's temp root ever sits inside a repository the
    fixture fails here instead of the test silently passing a watch that
    succeeded.
    """
    outside = root / "not-a-repo"
    outside.mkdir()
    probe = subprocess.run(
        ["git", "-C", str(outside), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        timeout=60,
        env=_git_env(),
    )
    assert probe.returncode != 0, f"{outside} is inside a git repository: {probe.stdout!r}"
    assert (
        NOT_A_REPO in probe.stderr or "not a git repository" in probe.stderr.lower()
    ), probe.stderr
    return outside


class _Sinks:
    """The scheduler's two outbound sinks, recorded instead of sent."""

    def __init__(self) -> None:
        self.notices: list[Any] = []
        self.deliveries: list[Any] = []


def _session(root: Path, cwd: Path) -> tuple[Session, _Sinks]:
    """A real Session owning the real bash tool, with its sinks recorded."""
    session = Session(
        model=MODEL,
        stream_fn=_stream,
        tools=[build_bash_tool()],
        transcript=Transcript(root / "sess"),
        system_blocks_provider=lambda: [],
        cwd=str(cwd),
    )
    sinks = _Sinks()

    async def announce(notice: Any) -> None:
        sinks.notices.append(notice)

    async def deliver(delivery: Any) -> None:
        sinks.deliveries.append(delivery)

    session._monitors._announce = announce  # type: ignore[method-assign]
    session._monitors._deliver = deliver  # type: ignore[method-assign]
    return session, sinks


async def _tick(session: Session, clock: list[int], *, advance_ms: int = 5_000) -> None:
    """One real scheduled tick: pump the scheduler, then await its work.

    The clock moves to the row's OWN ``next_due_at`` rather than by a fixed
    step: a check's due instant carries the randomised jitter ``_advance`` adds
    and the failure ladder's backoff, and a fixed step silently skipped ticks
    (the first version of this file advanced exactly one interval and the
    delivery test read as "no delivery" when no check had run at all).
    """
    due = next(
        (
            int(entry.counters["next_due_at"])
            for entry in session._monitors._entries.values()
            if isinstance(entry.counters.get("next_due_at"), int)
        ),
        None,
    )
    clock[0] = due + 1 if due is not None and due > clock[0] else clock[0] + advance_ms
    await session._monitors.pump(now_ms=clock[0] + 1)
    await drain_checks(session._monitors, session)


def _row(session: Session) -> dict[str, Any]:
    rows = session._monitors.index_rows()
    assert rows, "the monitor must be published in the index"
    return rows[0]


# ---------------------------------------------------------------------------
# (a) the incident: a git/gh watch armed outside a repository
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_git_watch_outside_a_repo_reports_the_failure_on_its_first_tick(
    tmp_path: Path, config_dir: Path
) -> None:
    """The repro. Same failure class as the incident: exit 128, fatal on stderr.

    ``git rev-parse --show-toplevel`` fails exactly the way the reporter's
    ``gh pr view`` did — same root cause (the tick's directory is not a
    repository), same stderr shape — without needing a network or a real PR.
    """
    outside = _not_a_repo(tmp_path)
    session, sinks = _session(tmp_path, outside)
    clock = [NOW]
    session._monitors._now = lambda: clock[0]  # type: ignore[method-assign]
    try:
        outcome = await session._monitors.create(
            {
                "name": "pr-ci",
                "tool": "bash",
                "arguments": {"command": "git rev-parse --show-toplevel"},
                "every": "60s",
            },
            cwd=str(outside),
        )
        assert outcome.get("created"), outcome
        # The arm did not probe (no "probe" in the request), so the watch's
        # FIRST tick is what must speak — the incident's own first tick failed.
        assert sinks.notices == []
        await _tick(session, clock)

        assert len(sinks.notices) == 1, sinks.notices
        notice = sinks.notices[0]
        assert notice.kind == "failing", notice
        assert notice.no_baseline is True, "a watch that never worked is not blipping"
        assert NOT_A_REPO in notice.detail, notice.detail
        text = format_monitor_notice_text(notice)
        assert text.startswith("(monitor)"), text
        assert "watch is failing:" in text, text
        assert NOT_A_REPO in text, text
        assert "pr-ci" in text and "m1" in text, text
        # A notice is not a delivery, and nothing was diffed into one.
        assert sinks.deliveries == []

        # THE LIST SURFACE: the row must carry the failure, not the neutral
        # sentence the reporter was reading for 90 minutes.
        row = _row(session)
        assert row["failing"] is True, row
        hint = monitor_store.health_hint(row, None)
        assert hint is not None and hint.startswith("WATCH IS FAILING"), hint
        assert NOT_A_REPO in hint, hint
        row_text = builtin._monitor_row_text(row, clock[0])
        assert "WATCH IS FAILING" in row_text, row_text
        assert "nothing has changed" not in row_text, row_text
        assert "confirm the call observes what you expect" not in row_text, row_text
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_the_failing_notice_is_latched_once_per_episode(
    tmp_path: Path, config_dir: Path
) -> None:
    """One notice, not one per tick: the incident was 18 identical failures."""
    outside = _not_a_repo(tmp_path)
    session, sinks = _session(tmp_path, outside)
    clock = [NOW]
    session._monitors._now = lambda: clock[0]  # type: ignore[method-assign]
    try:
        await session._monitors.create(
            {
                "name": "pr-ci",
                "tool": "bash",
                "arguments": {"command": "git rev-parse --show-toplevel"},
                "every": "60s",
            },
            cwd=str(outside),
        )
        for _ in range(3):
            await _tick(session, clock, advance_ms=5 * MINUTE)
        assert [notice.kind for notice in sinks.notices] == ["failing"], sinks.notices
    finally:
        await session.dispose()


# ---------------------------------------------------------------------------
# (b) the positive path: a cwd the watch can work in
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_watch_with_a_named_cwd_delivers_when_the_repo_changes(
    tmp_path: Path, config_dir: Path
) -> None:
    """The whole point of ``cwd``: the same call, in a directory that works.

    The watch runs ``git rev-parse --short HEAD`` in a real repository while the
    arming session's own directory is NOT one — which is the incident's shape
    with the one thing it lacked (a directory the call can run in).
    """
    outside = _not_a_repo(tmp_path)
    repo = _git_repo(tmp_path)
    session, sinks = _session(tmp_path, outside)
    clock = [NOW]
    session._monitors._now = lambda: clock[0]  # type: ignore[method-assign]
    try:
        outcome = await session._monitors.create(
            {
                "name": "head-sha",
                "tool": "bash",
                "arguments": {"command": "git rev-parse --short HEAD"},
                "cwd": str(repo),
                "every": "60s",
            },
            cwd=str(outside),
        )
        assert outcome.get("created"), outcome
        assert outcome["spec"].cwd == str(repo), outcome["spec"].cwd

        # First tick is the baseline: nothing to compare yet, nothing delivered,
        # and no notice (a working watch says nothing).
        await _tick(session, clock)
        assert sinks.deliveries == []
        assert sinks.notices == []

        before = _git(repo, "rev-parse", "--short", "HEAD")
        _git(repo, "commit", "-q", "--allow-empty", "-m", "second")
        after = _git(repo, "rev-parse", "--short", "HEAD")
        assert after != before

        await _tick(session, clock, advance_ms=MINUTE)
        assert len(sinks.deliveries) == 1, sinks.deliveries
        delivered = sinks.deliveries[0]
        assert after in delivered.delta_text, delivered.delta_text
        assert after != before
        assert sinks.notices == [], sinks.notices

        # AND THE SAME CALL WITHOUT A cwd, in the non-repo session, cannot work:
        # that is the half the cwd argument exists for.
        bare, bare_sinks = _session(tmp_path / "bare", outside)
        bare_clock = [NOW]
        bare._monitors._now = lambda: bare_clock[0]  # type: ignore[method-assign]
        try:
            await bare._monitors.create(
                {
                    "name": "head-sha",
                    "tool": "bash",
                    "arguments": {"command": "git rev-parse --short HEAD"},
                    "every": "60s",
                },
                cwd=str(outside),
            )
            await _tick(bare, bare_clock)
            assert [notice.kind for notice in bare_sinks.notices] == ["failing"]
            assert NOT_A_REPO in bare_sinks.notices[0].detail
            assert bare_sinks.deliveries == []
        finally:
            await bare.dispose()
    finally:
        await session.dispose()


# ---------------------------------------------------------------------------
# (c) the counter-attack: a WORKING watch stays silent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_quiet_working_watch_produces_no_notices_and_no_deliveries(
    tmp_path: Path, config_dir: Path
) -> None:
    """Identical output every tick: nothing delivered, nothing announced.

    This is the reviewer's named attack on the fix (make a working watch noisy),
    so it is pinned with the production arm path — a probe at arm — over several
    ticks.
    """
    watched = tmp_path / "watched.txt"
    watched.write_text("stable\n", encoding="utf-8")
    session, sinks = _session(tmp_path, tmp_path)
    clock = [NOW]
    session._monitors._now = lambda: clock[0]  # type: ignore[method-assign]
    try:
        outcome = await session._monitors.create(
            {
                "name": "quiet",
                "tool": "bash",
                "arguments": {"command": f"cat {watched}"},
                "every": "60s",
                "probe": True,
            },
            cwd=str(tmp_path),
        )
        assert outcome.get("created"), outcome
        assert outcome.get("baseline_adopted") is True, outcome
        assert "baseline_error" not in outcome, outcome
        for _ in range(4):
            await _tick(session, clock, advance_ms=MINUTE)
        assert sinks.deliveries == [], sinks.deliveries
        assert sinks.notices == [], sinks.notices
        # Not the failing/empty vocabulary: a working watch must never be
        # accused of being broken (the pre-existing neutral "N checks, 0
        # deliveries" hint is a different sentence and stays as it was).
        hint = monitor_store.health_hint(_row(session), None) or ""
        assert "FAILING" not in hint and "returned nothing" not in hint, hint
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_single_blip_then_recovery_pages_nobody(tmp_path: Path, config_dir: Path) -> None:
    """N=2 after a baseline: one transient failure is not news.

    A rate limit, a network hiccup or a restarting server looks exactly like
    this, so a single failed check of a watch that has worked must stay silent —
    and the recovery must deliver nothing, because the content never changed.
    """
    watched = tmp_path / "watched.txt"
    watched.write_text("stable\n", encoding="utf-8")
    session, sinks = _session(tmp_path, tmp_path)
    clock = [NOW]
    session._monitors._now = lambda: clock[0]  # type: ignore[method-assign]
    try:
        await session._monitors.create(
            {
                "name": "blip",
                "tool": "bash",
                "arguments": {"command": f"cat {watched}"},
                "every": "60s",
            },
            cwd=str(tmp_path),
        )
        await _tick(session, clock)  # baseline
        watched.unlink()  # the blip: the call fails once
        await _tick(session, clock, advance_ms=MINUTE)
        watched.write_text("stable\n", encoding="utf-8")  # same content again
        await _tick(session, clock, advance_ms=5 * MINUTE)

        assert sinks.notices == [], sinks.notices
        assert sinks.deliveries == [], sinks.deliveries
        counters = session._monitors.index_rows()[0]
        assert counters["consecutive_failures"] == 0, counters
    finally:
        await session.dispose()


# ---------------------------------------------------------------------------
# (d) arm-time validation, the cwd refusal, and the empty-output rule
# ---------------------------------------------------------------------------


async def _arm_via_tool(session: Session, session_cwd: str, **params: Any) -> Any:
    """Arm through the real ``monitor`` tool (the production path)."""
    context = ToolContext(cwd=session_cwd, session_id="s", monitor_scheduler=session._monitors)
    tool = builtin.build_monitor_tool(context)
    assert tool is not None
    request = {"op": "create", "tool": "bash", "every": "60s", **params}
    return await tool.execute("call-1", request, None, None, context)


@pytest.mark.asyncio
async def test_arming_a_broken_watch_warns_in_the_receipt_and_still_arms(
    tmp_path: Path, config_dir: Path
) -> None:
    """D6: the probe names the failure at ARM; the watch is not refused.

    A watch on a resource that does not exist yet is legitimate, so the arm
    succeeds — but its receipt leads with the failure instead of reading like a
    healthy arm beside a call that can never work.
    """
    outside = _not_a_repo(tmp_path)
    session, sinks = _session(tmp_path, outside)
    clock = [NOW]
    session._monitors._now = lambda: clock[0]  # type: ignore[method-assign]
    try:
        result = await _arm_via_tool(
            session,
            str(outside),
            name="pr-ci",
            arguments={"command": "git rev-parse --show-toplevel"},
        )
        assert result.is_error is False, result.text
        assert "Armed monitor" in result.text, result.text
        assert "WATCH IS FAILING" in result.text, result.text
        assert NOT_A_REPO in result.text, result.text
        assert result.details is not None
        assert NOT_A_REPO in str(result.details["baseline_error"])
        assert result.details["baseline_adopted"] is False
        # Armed anyway, and the first tick announces the episode.
        assert len(session._monitors.monitors) == 1
        assert sinks.notices == []
        await _tick(session, clock)
        assert [notice.kind for notice in sinks.notices] == ["failing"]
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_succeeding_probe_adopts_the_baseline_with_one_call(
    tmp_path: Path, config_dir: Path
) -> None:
    """D6's other half: no second call, and the receipt says so."""
    watched = tmp_path / "watched.txt"
    watched.write_text("one\n", encoding="utf-8")
    session, sinks = _session(tmp_path, tmp_path)
    clock = [NOW]
    session._monitors._now = lambda: clock[0]  # type: ignore[method-assign]
    calls: list[str] = []
    real_runner = session._monitors._check_runner

    async def counting_runner(spec: MonitorSpec) -> Any:
        calls.append(spec.id)
        return await real_runner(spec)

    session._monitors._check_runner = counting_runner  # type: ignore[method-assign]
    try:
        result = await _arm_via_tool(
            session,
            str(tmp_path),
            name="adopt",
            arguments={"command": f"cat {watched}"},
        )
        assert "Armed monitor" in result.text, result.text
        assert "WATCH IS FAILING" not in result.text, result.text
        assert "baseline was captured just now" in result.text, result.text
        assert result.details is not None and result.details["baseline_adopted"] is True
        assert len(calls) == 1, calls  # the arm spent exactly one call

        # The adopted baseline is real: a change is delivered on the next tick.
        watched.write_text("two\n", encoding="utf-8")
        await _tick(session, clock, advance_ms=MINUTE)
        assert len(calls) == 2, calls
        assert len(sinks.deliveries) == 1, sinks.deliveries
        assert "two" in sinks.deliveries[0].delta_text
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_named_cwd_that_is_not_a_directory_is_refused(
    tmp_path: Path, config_dir: Path
) -> None:
    """A bad cwd is a sentence at arm, never a watch that fails every tick."""
    session, sinks = _session(tmp_path, tmp_path)
    try:
        result = await _arm_via_tool(
            session,
            str(tmp_path),
            name="bad-cwd",
            arguments={"command": "ls"},
            cwd=str(tmp_path / "does-not-exist"),
        )
        assert result.is_error is True, result.text
        assert "not an existing directory" in result.text, result.text
        assert list(session._monitors.monitors) == []
        assert sinks.deliveries == []
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_an_empty_result_is_not_a_failure_but_is_announced_once(
    tmp_path: Path, config_dir: Path
) -> None:
    """D4: a call that has NEVER produced output says so, once — quietly.

    ``grep`` with no match is a legitimate quiet watch, so empty output is never
    a failure and never pages on the first tick; but a watch that has produced
    nothing for three consecutive checks can never detect a change, and that is
    worth one sentence.
    """
    # An EMPTY FILE, not `true`: the monitor allow-list refuses a command with
    # no read (``true`` is not read-only-provable), and `cat` on an empty file
    # is the same shape — exit 0, nothing on either stream.
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("", encoding="utf-8")
    session, sinks = _session(tmp_path, tmp_path)
    clock = [NOW]
    session._monitors._now = lambda: clock[0]  # type: ignore[method-assign]
    try:
        outcome = await session._monitors.create(
            {
                "name": "silent",
                "tool": "bash",
                "arguments": {"command": f"cat {empty_file}"},
                "every": "60s",
            },
            cwd=str(tmp_path),
        )
        assert outcome.get("created"), outcome
        for _ in range(EMPTY_CHECKS_BEFORE_NOTICE - 1):
            await _tick(session, clock, advance_ms=MINUTE)
        assert sinks.notices == [], "an empty result is not a failure"

        await _tick(session, clock, advance_ms=MINUTE)
        assert [notice.kind for notice in sinks.notices] == ["empty"], sinks.notices
        text = format_monitor_notice_text(sinks.notices[0])
        assert "returns nothing" in text, text
        assert str(EMPTY_CHECKS_BEFORE_NOTICE) in text, text

        # Latched: the fourth and fifth ticks add nothing.
        for _ in range(2):
            await _tick(session, clock, advance_ms=MINUTE)
        assert len(sinks.notices) == 1, sinks.notices
        assert sinks.deliveries == []
        # And the row stops claiming "nothing has changed".
        row = _row(session)
        hint = monitor_store.health_hint(row, None)
        assert hint is not None and "returned nothing" in hint, hint
    finally:
        await session.dispose()


# ---------------------------------------------------------------------------
# The seams underneath: structured exit facts and the gh -R allowlist
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bash_publishes_its_exit_status_and_stderr_as_structured_details() -> None:
    """D5: the exit status rides ``details``, so no consumer text-parses.

    This is the field the whole fix reads. It must be present on the ordinary
    (``is_error`` False) return, because that is the path a failing bash call
    takes.
    """
    tool = build_bash_tool()
    failed = await tool.execute("c1", {"command": "ls /definitely-not-a-real-dir-xyz"}, None)
    assert failed.is_error is False, "the bash contract: a non-zero exit is not an error"
    assert failed.text.startswith("exit code: ")
    assert failed.details is not None
    assert failed.details["exit_code"] not in (0, None)
    assert "definitely-not-a-real-dir-xyz" in failed.details["stderr_head"]
    assert failed.details["no_output"] is False

    quiet = await tool.execute("c2", {"command": "true"}, None)
    assert quiet.details is not None
    assert quiet.details["exit_code"] == 0
    assert quiet.details["no_output"] is True
    # A SUCCESSFUL command's details stay minimal: no stderr field to carry.
    assert "stderr_head" not in quiet.details

    chatty = await tool.execute("c3", {"command": "printf hi"}, None)
    assert chatty.details is not None and chatty.details["no_output"] is False


def test_gh_repo_naming_flag_is_allowed_on_read_only_verbs_only() -> None:
    """D2: ``gh pr view/checks -R owner/name`` is read-only; writes stay refused.

    ``--repo`` was already permitted — it names WHICH repo to read — so the short
    spelling was refused for spelling alone, with no workaround the reporter
    could use (``gh pr checks`` has no ``-R``-free alternative and ``gh api`` is
    refused as method-flippable).
    """
    from local_operator.monitors import readonly

    assert readonly._bash_verdict("gh pr view 1 -R owner/name") is None
    assert readonly._bash_verdict("gh pr checks 1 -R owner/name") is None
    assert readonly._bash_verdict("gh pr view 1 --repo owner/name") is None
    # The verb gate is untouched: a repo-naming flag never unlocks a write.
    refused = readonly._bash_verdict("gh pr merge 1 -R owner/name")
    assert refused is not None and "not read-only" in refused
    # Scoped to gh: glab's short-flag space is per verb, so it keeps --repo only.
    assert readonly._bash_verdict("glab mr view 1 -R owner/name") is not None
    assert readonly._bash_verdict("glab mr view 1 --repo owner/name") is None


# ---------------------------------------------------------------------------
# The policy unit level: thresholds, latching and recovery on a driven harness
# ---------------------------------------------------------------------------


class _Harness:
    """A scheduler with a scripted check runner — the ``test_scheduler`` shape.

    Used for the cases the session rig cannot drive deterministically (an exact
    sequence of failing/empty/succeeding outcomes) and for the threshold
    arithmetic itself.
    """

    def __init__(self, tmp_path: Path) -> None:
        self.now_ms = NOW
        self.results: list[dict[str, Any]] = []
        self.notices: list[Any] = []
        self.deliveries: list[Any] = []
        self.scheduler = MonitorScheduler(
            now=lambda: self.now_ms,
            config_dir=tmp_path / "cfg",
            session_id="sess",
            settings=MonitorSettings(),
            validate=lambda tool, args: None,
            run_check=self._run_check,
            deliver=self._deliver,
            persist=lambda monitors: None,
            announce=self._announce,
            uniform=lambda low, high: low,
        )

    async def _run_check(self, monitor: MonitorSpec) -> dict[str, Any]:
        if self.results:
            return self.results.pop(0)
        return {"text": "same", "error": None}

    async def _deliver(self, delivery: Any) -> None:
        self.deliveries.append(delivery)

    async def _announce(self, notice: Any) -> None:
        self.notices.append(notice)

    async def arm(self, **extra: Any) -> None:
        outcome = await self.scheduler.create(
            {"name": "watch", "tool": "bash", "arguments": {"command": "ls"}, "every": "60s"},
            cwd="/tmp",
        )
        assert outcome.get("created"), outcome

    async def tick(self) -> None:
        # Due-driven for the same reason as ``_tick``: the failure ladder doubles
        # ``next_due_at`` after each failed check, so a fixed step would skip the
        # very checks these tests count.
        due = next(
            (
                int(entry.counters["next_due_at"])
                for entry in self.scheduler._entries.values()
                if isinstance(entry.counters.get("next_due_at"), int)
            ),
            None,
        )
        self.now_ms = due + 1 if due is not None and due > self.now_ms else self.now_ms + MINUTE
        await self.scheduler.pump(now_ms=self.now_ms)
        await drain_checks(self.scheduler)


@pytest.mark.asyncio
async def test_the_failing_threshold_is_one_without_a_baseline_and_two_with_one(
    tmp_path: Path,
) -> None:
    """D3's N, asserted as the two numbers rather than as prose."""
    harness = _Harness(tmp_path)
    try:
        await harness.arm()
        # No baseline yet: the FIRST failure announces (it has never worked).
        harness.results.append({"error": "boom: no such repo", "kind": "failing"})
        await harness.tick()
        assert [notice.kind for notice in harness.notices] == ["failing"]
        assert harness.notices[0].no_baseline is True

        # A success establishes a baseline, and the episode closes with the
        # matching recovery.
        harness.results.append({"text": "A", "error": None})
        await harness.tick()
        assert [notice.kind for notice in harness.notices] == ["failing", "restored"]

        # Now one failure is a blip: silent.
        harness.results.append({"error": "boom: rate limited", "kind": "failing"})
        await harness.tick()
        assert len(harness.notices) == 2, harness.notices

        # The second consecutive one is news.
        harness.results.append({"error": "boom: still failing", "kind": "failing"})
        await harness.tick()
        assert [notice.kind for notice in harness.notices] == [
            "failing",
            "restored",
            "failing",
        ]
        assert harness.notices[-1].failures == FAILING_CHECKS_AFTER_BASELINE
        assert harness.notices[-1].no_baseline is False
        assert harness.deliveries == []
    finally:
        harness.scheduler.dispose()


@pytest.mark.asyncio
async def test_an_ordinary_tool_error_keeps_todays_ladder(tmp_path: Path) -> None:
    """Every non-``failing`` error path is unchanged: strikes, then a disable."""
    harness = _Harness(tmp_path)
    try:
        await harness.arm()
        for _ in range(MonitorSettings().max_consecutive_failures):
            harness.results.append({"error": "the tool returned an error"})
            await harness.tick()
        kinds = [notice.kind for notice in harness.notices]
        assert kinds == ["disabled"], kinds
        assert harness.scheduler.monitors[0].id == "m1"
        counters = harness.scheduler._entries["m1"].counters
        assert counters["disabled"] is True
    finally:
        harness.scheduler.dispose()


@pytest.mark.asyncio
async def test_a_success_after_a_told_failing_episode_announces_recovery(
    tmp_path: Path,
) -> None:
    """The recovery edge, so the failing notice is never the last word."""
    harness = _Harness(tmp_path)
    try:
        await harness.arm()
        harness.results.append(
            {"error": "failed to run git: fatal: not a git repository", "kind": "failing"}
        )
        await harness.tick()
        assert [notice.kind for notice in harness.notices] == ["failing"]
        harness.results.append({"text": "recovered", "error": None})
        await harness.tick()
        assert [notice.kind for notice in harness.notices] == ["failing", "restored"]
        counters = harness.scheduler._entries["m1"].counters
        assert counters["failing"] is False
        assert counters["failing_notified"] is False
    finally:
        harness.scheduler.dispose()


# ---------------------------------------------------------------------------
# The receipt card: a failing watch must not render with a DELTA's chrome
# ---------------------------------------------------------------------------


def test_a_failing_notice_is_recognised_as_a_notice_by_the_receipt_card() -> None:
    """``monitor_notice_kind`` reads the first line's phrase, so it needs one.

    Without a marker the card treated a failing notice as a DELTA: the envelope
    polish for a change, no warning ink, and a headline that was the whole
    block. That is the same "it looks healthy" shape the incident had, one layer
    up — so the registration is pinned, and so is the ink tier.
    """
    from local_operator.harness.rows import (
        monitor_notice_kind,
        monitor_receipt_headline,
    )
    from local_operator.tui.widgets.transcript import MonitorDeltaBlock

    text = format_monitor_notice_text(
        MonitorNotice(
            monitor_id="m1",
            name="pr-ci",
            tool="bash",
            kind="failing",
            at_ms=NOW,
            failures=1,
            detail="failed to run git: fatal: not a git repository",
            no_baseline=True,
        )
    )
    assert monitor_notice_kind(text) == "failing"
    assert "failing" in MonitorDeltaBlock.WARNING_NOTICE_KINDS
    headline = monitor_receipt_headline(text)
    assert "watch is failing:" in headline, headline
    assert "(via bash)" not in headline, headline
    assert "not a git repository" in headline, headline

    empty_text = format_monitor_notice_text(
        MonitorNotice(
            monitor_id="m1", name="quiet", tool="bash", kind="empty", at_ms=NOW, failures=3
        )
    )
    assert monitor_notice_kind(empty_text) == "empty"
    # A call that answers with nothing is a fact, not a fault: neutral ink.
    assert "empty" not in MonitorDeltaBlock.WARNING_NOTICE_KINDS
    assert "returns nothing" in monitor_receipt_headline(empty_text)
