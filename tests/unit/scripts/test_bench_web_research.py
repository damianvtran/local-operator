"""Correctness grading of the web-research A/B benchmark.

A benchmark whose grader can call a wrong answer correct, or average a crashed
run into wall time, produces numbers that argue for the wrong change. These
pin the two review-round-1 findings: contradictory answers must not pass, and
invalid runs must be excluded from every comparison figure.
"""

from __future__ import annotations

import scripts.bench_web_research as bench
from scripts.bench_web_research import (
    TASKS,
    Arm,
    RunRecord,
    is_valid,
    summarise,
    validate_answer,
)


def _task(slug: str) -> bench.Task:
    return next(t for t in TASKS if t.slug == slug)


def test_correct_answer_passes_and_contradiction_fails() -> None:
    task = _task("pg-max-connections")
    assert validate_answer(task, "The default `max_connections` is **100**.")
    # R1 negative control, verbatim in substance: right value present but the
    # answer actually states a different one.
    assert not validate_answer(task, "The default max_connections is 50, not 100.")
    assert not validate_answer(task, "It is not 100.")
    assert not validate_answer(task, "There is no 100 default documented.")
    assert not validate_answer(task, "The default is 1000.")


def test_expected_value_does_not_match_inside_a_longer_number() -> None:
    task = _task("pg-max-connections")
    assert not validate_answer(task, "The default is 1000 connections.")
    assert not validate_answer(task, "The default is 10000.")
    # A thousands-separated expected value still matches its own spelling.
    h2 = _task("h2-initial-window")
    assert validate_answer(h2, "The initial value is 65,535 octets (216-1).")
    assert validate_answer(h2, "65535 octets.")
    assert not validate_answer(h2, "65536 octets.")


def test_contradictory_year_and_code_fail() -> None:
    git = _task("git-first-release")
    assert validate_answer(git, "Git was created in 2005.")
    assert not validate_answer(git, "Git was created in 2005, not 2006.")
    misdirected = _task("rfc9110-misdirected")
    assert validate_answer(misdirected, "The status code is 421 (Misdirected Request).")
    assert not validate_answer(misdirected, "It is 422, not 421.")


def test_empty_answer_and_missing_key_fail() -> None:
    assert not validate_answer(_task("pg-max-connections"), "")
    assert not validate_answer(_task("pg-max-connections"), "I could not find out.")


def test_invalid_run_is_never_correct() -> None:
    task = _task("pg-max-connections")
    record = RunRecord(arm="a", task=task.slug, rep=1, exit=1, answer="100")
    assert not is_valid(record)
    # parse_capture takes the Task OBJECT (Q1): a slug or preview regex can no
    # longer be passed by mistake. Empty stream -> no answer -> not correct.
    record.exit = 0
    bench.parse_capture("", record, task)
    assert record.correct is False, "no message_end events means no answer"
    record2 = RunRecord(arm="a", task=task.slug, rep=1, exit=1, answer="100")
    assert not is_valid(record2)


def test_summarise_excludes_invalid_runs_from_means_and_ratios() -> None:
    arms = [
        Arm("a", bench.Path("/a"), bench.Path("/a/python"), "a"),
        Arm("b", bench.Path("/b"), bench.Path("/b/python"), "b"),
    ]
    task = _task("pg-max-connections")
    good = RunRecord(
        arm="a", task=task.slug, rep=1, exit=0, wall_s=20.0, answer="100", correct=True
    )
    # A crashed run that is FAST and whose answer happens to match: it must not
    # lower arm a's mean wall time nor count as correct.
    crashed = RunRecord(
        arm="a", task=task.slug, rep=2, exit=1, wall_s=1.0, answer="100", correct=True
    )
    other = RunRecord(
        arm="b", task=task.slug, rep=1, exit=0, wall_s=40.0, answer="100", correct=True
    )
    summary = summarise([good, crashed, other], arms, [task])
    key = f"{task.slug}|a"
    assert summary["per"][key]["n"] == 1
    assert summary["per"][key]["attempted"] == 2
    assert summary["per"][key]["wall_mean"] == 20.0
    assert summary["per"][key]["correct"] == 1
    assert summary["totals"]["a"]["invalid"] == 1
    assert summary["totals"]["a"]["runs"] == 1
    assert summary["invalid_runs"][0]["exit"] == 1
    # The paired ratio uses only eligible runs: 40/20, not 40/((20+1)/2).
    assert summary["ratios"]["b"][task.slug] == 2.0


def _capture_with_answer(text: str) -> str:
    import json

    return json.dumps(
        {
            "type": "message_end",
            "message": {
                "id": "m1",
                "role": "assistant",
                "content": [{"type": "text", "text": text}],
            },
        }
    )


def test_parse_capture_grades_through_the_task_object() -> None:
    """Q1 wiring: drive parse_capture exactly as run_live does (Task object)."""
    task = _task("pg-max-connections")
    record = RunRecord(arm="a", task=task.slug, rep=1, exit=0)
    bench.parse_capture(_capture_with_answer("The default max_connections is 100."), record, task)
    assert record.correct is True
    bad = RunRecord(arm="a", task=task.slug, rep=1, exit=0)
    bench.parse_capture(
        _capture_with_answer("The default max_connections is 50, not 100."), bad, task
    )
    assert bad.correct is False


def test_adjacent_documented_values_are_not_wrong_answers() -> None:
    """Q2: correct answers that also state an adjacent documented value pass."""
    sqlite = _task("sqlite-max-column")
    answer = (
        "SQLite's default `SQLITE_MAX_COLUMN` is **2000**.\n"
        "- You can raise it at compile time, up to a maximum of 32767.\n"
        "- You can lower it at run time with `sqlite3_limit()`."
    )
    assert validate_answer(sqlite, answer)
    pep8 = _task("pep8-docstring-length")
    answer = (
        "PEP 8 limits docstrings and comments to **72 characters** per line.\n"
        "Code lines are limited to 79 characters."
    )
    assert validate_answer(pep8, answer)


def test_true_contradictions_still_fail_at_clause_level() -> None:
    """The Q2 relaxation must not re-admit the R1 false positives."""
    pg = _task("pg-max-connections")
    assert not validate_answer(pg, "The default max_connections is 50, not 100.")
    assert not validate_answer(pg, "The default is 1000.")
    assert not validate_answer(pg, "It is not 100.")
    git = _task("git-first-release")
    assert not validate_answer(git, "Git was created in 2005, not 2006.")
    sqlite = _task("sqlite-max-column")
    # A distractor stated AS the answer is still wrong even with 2000 nearby.
    assert not validate_answer(sqlite, "The default SQLITE_MAX_COLUMN is 32767, not 2000.")
