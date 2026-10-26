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


# (slug, answer, expected). Correct answers, contradictions (R1/R3/Q4),
# adjacent documented values (Q2/Q5), negated distractors (M1), numeric
# prefixes (M2) and additive idioms (N1) — the union of every review round's
# cases, so a relaxation on one axis cannot silently reopen another.
CASES: list[tuple[str, str, bool]] = [
    # plain correct answers
    ("pg-max-connections", "The default max_connections is 100.", True),
    ("git-first-release", "Git was created in 2005.", True),
    ("h2-initial-window", "The initial value is 65,535 octets (216-1).", True),
    ("h2-initial-window", "65535 octets.", True),
    ("rfc9110-misdirected", "The status code is 421 (Misdirected Request).", True),
    # R1 / R3 / Q4 contradictions — same clause and cross-clause
    ("pg-max-connections", "The default max_connections is 50, not 100.", False),
    ("pg-max-connections", "Many guides say 100, but the actual default is 50.", False),
    ("pg-max-connections", "Not 100 — the default is 50.", False),
    ("pg-max-connections", "The default is 50. Not 100.", False),
    ("pg-max-connections", "No. 100 is not the default; it is 50.", False),
    ("pg-max-connections", "100 is not the default.", False),
    ("pg-max-connections", "It is not 100.", False),
    ("pg-max-connections", "There is no 100 default documented.", False),
    ("pg-max-connections", "The default is 1000.", False),
    ("git-first-release", "Never 2005 — it was 2006.", False),
    # M1: negated distractor is a correct answer
    ("git-first-release", "Git was created in 2005, not 2006.", True),
    (
        "git-first-release",
        "Git was first released in 2006. Linus Torvalds began work on it in 2005.",
        False,
    ),
    (
        "pg-max-connections",
        "The default max_connections is 50.\n\nNote: older releases defaulted to 100.",
        False,
    ),
    (
        "h2-initial-window",
        "The initial value is 65536 octets. (… some docs round it to 65,535.)",
        False,
    ),
    (
        "https-default-port",
        "HTTPS uses port 8443 by default; 443 is reserved for the TLS handshake.",
        False,
    ),
    ("rfc9110-misdirected", "Misdirected Request is 422 — 421 is Too Early.", False),
    ("sqlite-max-column", "The default is 32767, while older builds used 2000.", False),
    ("sqlite-max-column", "The default SQLITE_MAX_COLUMN is 32767, not 2000.", False),
    # M2: expected value must not match a longer/decimal/thousands number
    ("pg-max-connections", "The default is 10000.", False),
    ("pg-max-connections", "The default is 100,000.", False),
    ("pg-max-connections", "The default is 100.5.", False),
    ("h2-initial-window", "65536 octets.", False),
    # M1: a negated distractor is a correct answer, not a contradiction
    ("pg-max-connections", "The default is 100, not 50.", True),
    ("pg-max-connections", "The default is 100 (not 1000).", True),
    # Q2 / Q5: adjacent documented values in genuine context
    (
        "sqlite-max-column",
        "SQLite's default SQLITE_MAX_COLUMN is **2000**.\n"
        "- You can raise it at compile time, up to a maximum of 32767.\n"
        "- You can lower it at run time with `sqlite3_limit()`.",
        True,
    ),
    (
        "pep8-docstring-length",
        "PEP 8 limits docstrings and comments to **72 characters** per line.\n"
        "Code lines are limited to 79 characters.",
        True,
    ),
    ("pep8-docstring-length", "72 characters for docstrings/comments (79 for code).", True),
    ("pep8-docstring-length", "| docstrings/comments | 72 | code | 79 |", True),
    ("pep8-docstring-length", "Docstrings are 72, code is 79.", True),
    (
        "pep8-docstring-length",
        'Docstrings/comments are 72. PEP 8 says "Limit all lines to a maximum of 79 characters."',
        True,
    ),
    ("pep8-docstring-length", "Docstrings are 72. Other lines are limited to 79.", True),
    # a distractor stated AS the asked value still fails even with context declared
    ("pep8-docstring-length", "Docstrings are limited to 79 characters.", False),
    ("pep8-docstring-length", "The docstring limit is 79, not 72.", False),
    # N1: additive idiom is not a negation
    ("python-creator", "Guido van Rossum is not only Python's creator but also its BDFL.", True),
]


def test_grader_matrix() -> None:
    mismatches = [
        (slug, answer, want, validate_answer(_task(slug), answer))
        for slug, answer, want in CASES
        if validate_answer(_task(slug), answer) != want
    ]
    assert not mismatches, "\n".join(str(m) for m in mismatches)


def test_every_review_contradiction_is_covered() -> None:
    """Guard the guard: the matrix must contain both accept and reject cases
    for the tasks whose graders have exemptions, so a future relaxation trips
    a reject case instead of only passes."""
    for slug in ("pg-max-connections", "pep8-docstring-length", "sqlite-max-column"):
        wants = {want for s, _a, want in CASES if s == slug}
        assert wants == {True, False}, slug


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
