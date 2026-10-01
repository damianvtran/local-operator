"""The soft query budget: what counts as a query, and what happens at the mark.

Three properties matter more than coverage here:

1. **The classifier's false NEGATIVES are the expensive ones.** A query it misses
   is a seven-minute walk (the shape this exists for); one it over-includes is a
   trivial command that never reaches 10 s. So the matrix pins the measured shape
   AND the ordinary shell a model writes every day — pipes, builds, a quoted
   mention.
2. **The thresholds are pure.** ``Budget.advisory_due`` / ``stop_due`` take an
   elapsed float and return a bool, with no process and no clock, so the policy is
   a unit target and the integration case below only has to prove the wiring.
3. **The stop is a RESULT, not a log line.** An interception nobody reads is the
   silent multi-minute query, so the last test drives the real ``execute_bash``
   with a tiny configured budget and asserts the line and the flag.

Per AGENTS.md "Timing, flakes": nothing here asserts a wall time. The integration
case is sized structurally — the command does ~3 s of work against a 2 s budget —
and it asserts the STOP HAPPENED and that the non-query twin finished, never how
long either took.
"""

from __future__ import annotations

import pytest

from local_operator.harness.types import ToolContext
from local_operator.tools import builtin
from local_operator.tools import query_budget as qb

#: The shape that cost a live session seven minutes, verbatim. Every piece of it
#: is locally reasonable and the static guard correctly lets it through.
MEASURED_LOOP = (
    "for f in $(find ~/.local-operator/sessions -maxdepth 2 -name transcript.jsonl "
    '-mmin -720 | head -80); do grep -ql PATTERN "$f"; done'
)

#: Commands that ARE a filesystem query. The block list is where a miss is
#: expensive, so it includes a loop body, a substitution, a bare walk, `du`, and
#: the wrappers an agent reaches for when it already suspects the search is slow.
QUERY_SHAPED = [
    MEASURED_LOOP,
    "grep -rn p .",
    "find . -name '*.ts' -maxdepth 3",
    "du -sh ~/Downloads",
    "rg -n p src/",
    'for f in *.ts; do du -sh "$f"; done',
    'while read f; do grep -rn x "$f"; done',
    'echo "$(find . -maxdepth 1)"',
    "FOO=$(du -sh .) echo hi",
    "git commit -m 'fix' && grep -rn p src/",
    # Wrappers (review m2): each carries a real walk and must be seen through.
    "timeout 300 find / -name x",
    "timeout -s KILL 300 find / -name x",
    "sudo find / -name x",
    "env X=1 grep -rn p .",
    "time grep -rn p .",
    "nice -n 5 find / -name x",
    "command rg -n p",
    "nohup du -sh .",
    # `bash -c` is parsed, not skipped (review Q3).
    "bash -c 'grep -rn x .'",
    "sh -c 'find / -name x'",
    # A piped grep that CARRIES a recursion flag still walks.
    "cat x | grep -rn p .",
    # A wrapper around a recursed grep is still a query: this classifier judges
    # the command's SHAPE (the static guard owns the root), and `timeout 300`
    # changes nothing about what the grep does.
    "timeout 300 grep -rn p src/",
]

#: Commands that are NOT. Each one is ordinary shell the model writes daily, and
#: a false positive here would put a 60 s budget on a build or a test run — which
#: is the expensive direction (review M2).
NOT_QUERY_SHAPED = [
    # The measured M2 twins: a long-runner ANYWHERE vetoes the class, because the
    # elapsed second the budget measures belongs to the suite, not the grep.
    "python3 -c 'time.sleep(4)'; grep -c x f.txt",
    "python3 -c 'import time; time.sleep(4)' && echo built && find . -maxdepth 1 -name f.txt",
    "until grep -q ready log.txt; do sleep 1; done",
    "make -j8 && find build -name '*.so'",
    "pytest tests -x; grep -c ok log",
    "npm run build && grep -q ok out.txt",
    "cargo build; du -sh target",
    # …and the plain shapes: a grep with no recursion flag reads files it names,
    # a piped stage reads stdin, and a build is a build.
    "pytest | grep fail",
    "pytest -q tests/unit",
    "make build",
    "npm install",
    "python -m pip install -e .",
    "echo grep",
    "echo 'grep -rn p .'",
    'echo "grep -rn p ."',
    "echo '$(find .)'",
    "ls -la | head",
    'while read f; do cat "$f"; done',
    'while read f; do grep -q x "$f"; done',
    "if grep -q x f; then echo y; fi",
    "git grep -n x origin/main",
    "sleep 30",
    "true",
    "",
    "   ",
]


@pytest.mark.parametrize("command", QUERY_SHAPED)
def test_query_shaped_commands_are_classified(command: str) -> None:
    assert qb.is_query_shaped(command), command
    assert qb.query_class(command) is not None


@pytest.mark.parametrize("command", NOT_QUERY_SHAPED)
def test_ordinary_commands_are_not_queries(command: str) -> None:
    assert not qb.is_query_shaped(command), command
    assert qb.query_class(command) is None


def test_the_measured_loop_is_classified_as_a_substitution() -> None:
    """The reason NAMES what made it a query, because that phrase is what the
    model reads in the advisory and the stop."""
    cls = qb.query_class(MEASURED_LOOP)
    assert cls is not None
    assert "find" in cls


def test_a_piped_stage_reading_stdin_is_not_a_query() -> None:
    """`pytest | grep fail` reads the previous stage's stdout: no walk, no
    budget. The exemption is per-program and per-segment, like the static
    guard's stream-filter rule."""
    assert not qb.is_query_shaped("pytest -q | grep -c fail")
    # …but a walk in a SUBSTITUTION inside a piped stage still counts.
    assert qb.is_query_shaped("echo x | head $(find . -maxdepth 1 | wc -l)")


def test_the_grant_is_never_read_from_the_process_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(qb.ALLOW_ENV, "1")
    assert not qb.allow_slow_query(MEASURED_LOOP), "the process env must not grant"
    assert not qb.allow_slow_query("du -sh .")


def test_the_grant_works_on_the_search_itself() -> None:
    """The forms a SIMPLE command can carry: a leading assignment, or an `env`
    prefix (review Q1)."""
    assert qb.allow_slow_query(f"{qb.ALLOW_ENV}=1 grep -rn p .")
    assert qb.allow_slow_query(f"{qb.ALLOW_ENV}=true find . -maxdepth 1")
    assert qb.allow_slow_query(f"env {qb.ALLOW_ENV}=1 find . -maxdepth 1")
    assert not qb.allow_slow_query(f"{qb.ALLOW_ENV}=0 find . -maxdepth 1")


def test_the_grant_works_on_a_compound_command() -> None:
    """The forms a LOOP can carry (review Q1).

    `ALLOW=1 for f in …` is a bash SYNTAX ERROR — bash rejects an assignment in
    front of a `for` — so the inline prefix cannot be the only advertised form.
    A standalone statement, an `export`, and a `bash -c` wrapper all work, and all
    three are what the stop message now names.
    """
    assert qb.allow_slow_query(f"{qb.ALLOW_ENV}=1; {MEASURED_LOOP}")
    assert qb.allow_slow_query(f"export {qb.ALLOW_ENV}=1; {MEASURED_LOOP}")
    assert qb.allow_slow_query(f"{qb.ALLOW_ENV}=1 bash -c '{MEASURED_LOOP}'")
    assert not qb.allow_slow_query(MEASURED_LOOP)


def test_a_grant_on_an_unrelated_segment_does_not_disarm_the_guard() -> None:
    """review Q2: the grant must sit on a WALK-ISH segment, on a standalone
    statement, or on an `env` prefix of a walk — otherwise `X=1 echo hi` beside a
    loop would waive the budget for the loop."""
    assert not qb.allow_slow_query(f"{qb.ALLOW_ENV}=1 echo hi; {MEASURED_LOOP}")
    assert not qb.allow_slow_query(f"{qb.ALLOW_ENV}=1 echo hi")
    # …while the standalone statement, which is a deliberate act, does.
    assert qb.allow_slow_query(f"{qb.ALLOW_ENV}=1; echo hi; {MEASURED_LOOP}")


# ---------------------------------------------------------------------------
# The verdicts, as pure functions of an elapsed float. No clock, no process.
# ---------------------------------------------------------------------------


def test_the_advisory_fires_at_the_soft_mark_and_the_stop_at_the_budget() -> None:
    budget = qb.Budget(seconds=60)
    assert not budget.advisory_due(9.99)
    assert budget.advisory_due(10.0)  # inclusive: the mark is the mark
    assert not budget.stop_due(59.99)
    assert budget.stop_due(60.0)
    assert budget.stop_due(600.0)


def test_the_grant_skips_the_stop_and_never_the_advisory() -> None:
    """Both halves are load-bearing: the grant must not silence the advisory (a
    justified slow query is still a slow query), and it must not be a no-op."""
    budget = qb.Budget(seconds=60)
    assert budget.advisory_due(120.0)
    assert budget.stop_due(120.0)
    assert not budget.stop_due(120.0, allowed=True)


def test_disabled_silences_both_and_stop_false_is_warn_only() -> None:
    off = qb.Budget(enabled=False)
    assert not off.advisory_due(1e9)
    assert not off.stop_due(1e9)
    warn_only = qb.Budget(stop=False)
    assert warn_only.advisory_due(1e9)
    assert not warn_only.stop_due(1e9)


def test_the_defaults_are_the_protective_ones() -> None:
    budget = qb.Budget()
    assert budget.enabled is True
    assert budget.stop is True
    assert budget.seconds == qb.QUERY_BUDGET_SECONDS_DEFAULT == 60
    assert qb.ADVISORY_AFTER_S == 10.0


def test_the_stop_message_states_the_stop_and_every_way_out() -> None:
    msg = qb.stop_message(63.4, "`find`")
    assert msg.startswith("STOPPED AT SOFT QUERY BUDGET (63s)")
    assert "`find`" in msg  # the class reason travels into the message
    for hint in ("maxdepth", "max-count", "sessions", "/resume", "/settings"):
        assert hint in msg, hint
    assert qb.ALLOW_ENV in msg
    # review m1: the WORKING forms, not a generic "prefix" — the prefix does not
    # parse in front of a `for` loop, so the message names the statement form too.
    assert f"{qb.ALLOW_ENV}=1;" in msg
    assert "for f in" in msg
    # …and it says who this budget does NOT apply to, so a stopped model does not
    # conclude that builds are budgeted too.
    assert "installs" in msg and "test runs" in msg


def test_the_advisory_names_the_elapsed_time_and_the_budget() -> None:
    msg = qb.advisory_message(12.5, "`grep`", 60)
    assert "12s" in msg
    assert "60s" in msg
    assert qb.ALLOW_ENV in msg


# ---------------------------------------------------------------------------
# The reader: LIVE scope, so an edit lands on the NEXT command.
# ---------------------------------------------------------------------------


def _write_config(config_dir: object, rows: str) -> None:
    from pathlib import Path

    root = Path(str(config_dir))
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.yml").write_text(f"values:\n  bash:\n    query_budget:\n{rows}")


def test_a_config_edit_is_read_on_the_next_command(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pathlib import Path

    config_dir = Path(str(tmp_path)) / "config"
    _write_config(config_dir, "      seconds: 17\n")
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))
    assert builtin._query_budget_config().seconds == 17

    # …and the edit really is per call, not per process.
    _write_config(config_dir, "      seconds: 23\n")
    assert builtin._query_budget_config().seconds == 23


def test_a_bad_or_impossible_budget_degrades_to_a_sane_guard(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hand-edited config.yml can hold a string where a number belongs, and a
    budget of 0 would stop every query instantly. Neither must raise inside the
    guard that protects the command, and neither may leave the guard toothless:
    the type falls back to the default and the value to a 1 s floor."""
    from pathlib import Path

    config_dir = Path(str(tmp_path)) / "config"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))

    _write_config(config_dir, '      seconds: "soon"\n')
    assert builtin._query_budget_config().seconds == qb.QUERY_BUDGET_SECONDS_DEFAULT

    _write_config(config_dir, "      seconds: 0\n")
    resolved = builtin._query_budget_config()
    assert resolved.seconds == 1
    assert resolved.enabled is True


# ---------------------------------------------------------------------------
# The wiring: the real tool, a tiny budget, and a query that outlasts it.
# ---------------------------------------------------------------------------


def _loop_over(work_dir: str, files: int, per_file: float) -> str:
    """A query-shaped command sized to run ~``files * per_file`` seconds.

    Written as a substitution (`$(find …)`) so it is a query on the measured
    pattern rather than on a bare command word, and so the static root guard
    accepts it — the shape the budget exists to catch is exactly the one the
    static layer must let through.
    """
    return f'for f in $(find {work_dir} -maxdepth 1 -name "*.txt"); ' f"do sleep {per_file}; done"


#: Printed by a run that reaches the end. A granted run asserts on this rather
#: than on the absence of a message: "no stop line" is also what a command that
#: never started produces, which would let a broken grant pass (review Q1).
_SENTINEL = "GRANTED-RUN-COMPLETED"


@pytest.mark.slow
@pytest.mark.asyncio
async def test_a_query_is_stopped_at_the_budget_while_its_twins_are_not(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole feature, through the real path.

    Three commands, one configured budget of 2 s: the query is stopped (and says
    so, at the head, with the flag set), an ordinary long command runs to
    completion, and a query carrying the inline grant runs to completion with the
    advisory still shown. No wall time is asserted anywhere — the query does ~3 s
    of work against a 2 s budget and the twins are checked for the ABSENCE of a
    stop, so the case cannot flake on a loaded host.
    """
    from pathlib import Path

    config_dir = Path(str(tmp_path)) / "config"
    _write_config(config_dir, "      seconds: 2\n")
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))

    work = Path(str(tmp_path)) / "work"
    work.mkdir()
    for index in range(6):
        (work / f"f{index}.txt").write_text("x")
    context = ToolContext(cwd=str(work))

    # 1. The query: 6 x 0.5 s of work under a 2 s budget.
    command = _loop_over(str(work), 6, 0.5)
    assert qb.is_query_shaped(command), "the integration shape must classify as a query"
    result = await builtin.execute_bash("qb-1", {"command": command}, None, None, context)
    assert result.is_error is True
    first = result.text.splitlines()[0]
    assert first.startswith("STOPPED AT SOFT QUERY BUDGET"), result.text
    assert result.details and result.details.get("query_budget_stopped") is True
    assert result.details.get("query_budget_seconds") == 2

    # 2. The ordinary twin: same duration, not a query, so it finishes.
    ordinary = await builtin.execute_bash(
        "qb-2", {"command": "python3 -c 'import time; time.sleep(3)'"}, None, None, context
    )
    assert ordinary.is_error is False, ordinary.text
    assert "STOPPED AT SOFT QUERY BUDGET" not in ordinary.text
    assert not (ordinary.details or {}).get("query_budget_stopped")

    # 3. The granted twin — NON-VACUOUS (review Q1): the granted run must prove
    # it RAN TO COMPLETION past a budget shorter than its own duration, or a
    # missing grant mechanism would pass this test. The grant is the statement
    # form, which is the one a loop can actually carry.
    sentinel = f"echo {_SENTINEL}"
    granted = await builtin.execute_bash(
        "qb-3",
        {"command": f"{qb.ALLOW_ENV}=1; {command}; {sentinel}"},
        None,
        None,
        context,
    )
    assert "STOPPED AT SOFT QUERY BUDGET" not in granted.text
    assert not (granted.details or {}).get("query_budget_stopped")
    assert _SENTINEL in granted.text, "the granted run did not finish: " + granted.text
    # …and the SAME command WITHOUT the grant is stopped, so deleting the grant
    # mechanism fails here rather than silently passing.
    ungranted = await builtin.execute_bash(
        "qb-3b", {"command": f"{command}; {sentinel}"}, None, None, context
    )
    assert ungranted.is_error is True, ungranted.text
    assert _SENTINEL not in ungranted.text


@pytest.mark.slow
@pytest.mark.asyncio
async def test_a_query_behind_bash_c_is_classified_and_stoppable(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """review Q3: `bash -c '<loop>'` is the natural way to write a compound
    command AND the only way to hang a grant on one, so it must be parsed rather
    than skipped — and the grant on the wrapper must carry into the inner one."""
    from pathlib import Path

    config_dir = Path(str(tmp_path)) / "config"
    _write_config(config_dir, "      seconds: 2\n")
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))

    work = Path(str(tmp_path)) / "work"
    work.mkdir()
    for index in range(6):
        (work / f"f{index}.txt").write_text("x")
    context = ToolContext(cwd=str(work))

    inner = _loop_over(str(work), 6, 0.5)
    wrapped = f"bash -c '{inner}'"
    assert qb.is_query_shaped(wrapped), "the wrapper must not hide the loop"

    stopped = await builtin.execute_bash("qb-5", {"command": wrapped}, None, None, context)
    assert stopped.is_error is True, stopped.text
    assert stopped.text.splitlines()[0].startswith("STOPPED AT SOFT QUERY BUDGET")

    granted = await builtin.execute_bash(
        "qb-6",
        {"command": f"{qb.ALLOW_ENV}=1 {wrapped}; echo {_SENTINEL}"},
        None,
        None,
        context,
    )
    assert _SENTINEL in granted.text, granted.text
    assert not (granted.details or {}).get("query_budget_stopped")


@pytest.mark.slow
@pytest.mark.asyncio
async def test_the_advisory_reaches_the_live_card_and_the_result(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """At 10 s the model gets a line it can act on — on the live card while the
    command runs, and in the result afterwards.

    The budget is 120 s rather than just above the run (review m5): the command
    needs 24 x 0.5 s of work to cross the 10 s mark, and on a loaded host the loop
    overhead could otherwise push it past a tight budget and turn the case into a
    stop — a false fail. It costs nothing, because the command ends by itself.
    """
    from pathlib import Path

    config_dir = Path(str(tmp_path)) / "config"
    _write_config(config_dir, "      seconds: 120\n")
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))

    work = Path(str(tmp_path)) / "work"
    work.mkdir()
    for index in range(24):
        (work / f"f{index}.txt").write_text("x")
    context = ToolContext(cwd=str(work))
    command = _loop_over(str(work), 24, 0.5)  # ~12 s, so it crosses the 10 s mark

    seen: list[str] = []

    def on_update(update: object) -> None:
        details = getattr(update, "details", None) or {}
        advisory = details.get("query_budget_advisory")
        if advisory:
            seen.append(advisory)

    result = await builtin.execute_bash("qb-4", {"command": command}, None, on_update, context)
    assert seen, "the advisory never rode the live-update channel"
    assert "QUERY BUDGET" in seen[0]
    assert qb.ALLOW_ENV in seen[0]
    assert "QUERY BUDGET" in result.text, "the advisory was not carried into the result"
    # The run finished INSIDE its budget, so the receipt is the advisory and not a
    # stop — the two must never be confused in the result.
    assert result.is_error is False, result.text
    assert "STOPPED AT SOFT QUERY BUDGET" not in result.text
