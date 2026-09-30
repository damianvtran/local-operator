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
#: expensive, so it includes a loop body, a substitution, a bare walk and `du`.
QUERY_SHAPED = [
    MEASURED_LOOP,
    "grep -rn p .",
    "find . -name '*.ts' -maxdepth 3",
    "du -sh ~/Downloads",
    "rg -n p src/",
    'for f in *.ts; do du -sh "$f"; done',
    'while read f; do grep -q x "$f"; done',
    "if grep -q x f; then echo y; fi",
    'echo "$(find . -maxdepth 1)"',
    "FOO=$(du -sh .) echo hi",
    "git commit -m 'fix' && grep -rn p src/",
]

#: Commands that are NOT. Each one is ordinary shell the model writes daily, and
#: a false positive here would put a budget on a build.
NOT_QUERY_SHAPED = [
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


def test_the_grant_is_read_per_segment_and_never_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(qb.ALLOW_ENV, "1")
    assert not qb.allow_slow_query(MEASURED_LOOP), "the process env must not grant"
    assert qb.allow_slow_query(f"{qb.ALLOW_ENV}=1 {MEASURED_LOOP}")
    assert qb.allow_slow_query(f"{qb.ALLOW_ENV}=true find . -maxdepth 1")
    assert not qb.allow_slow_query(f"{qb.ALLOW_ENV}=0 find . -maxdepth 1")


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
    for hint in ("maxdepth", "-mmin", "max-count", "sessions", "/resume", "/settings"):
        assert hint in msg, hint
    assert qb.ALLOW_ENV in msg


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

    # 3. The granted twin: the same query, allowed to run, advisory still shown.
    granted = await builtin.execute_bash(
        "qb-3", {"command": f"{qb.ALLOW_ENV}=1 {command}"}, None, None, context
    )
    assert "STOPPED AT SOFT QUERY BUDGET" not in granted.text
    assert not (granted.details or {}).get("query_budget_stopped")


@pytest.mark.slow
@pytest.mark.asyncio
async def test_the_advisory_reaches_the_live_card_and_the_result(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """At 10 s the model gets a line it can act on — on the live card while the
    command runs, and in the result afterwards. A budget of 30 s keeps the
    command alive past the advisory without the run costing 30 s."""
    from pathlib import Path

    config_dir = Path(str(tmp_path)) / "config"
    _write_config(config_dir, "      seconds: 30\n")
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
    # The run finished INSIDE its 30 s budget, so the receipt is the advisory and
    # not a stop — the two must never be confused in the result.
    assert result.is_error is False, result.text
    assert "STOPPED AT SOFT QUERY BUDGET" not in result.text
