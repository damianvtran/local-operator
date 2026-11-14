"""``scripts/audit_time_assertions.py``: pin the bucket rules on synthetic source.

WHY. The audit is only useful if its "derived vs hard-coded" answer is right,
and PR #3627's round-1 QA (Q1) and review (R4, R7) found three ways it was not:
the word "noise" anywhere in a function body (docstring, comment, an unrelated
variable) filed a hard-coded ceiling as calibrated; a tolerance nested in
arithmetic (``time.time() + 1``) was read as "0 s"; and a clock on the LARGE
side of ``<=`` — which load only makes truer — was counted as a crossable
ceiling. Each case below is one of those shapes, written as a tiny test module
in ``tmp_path`` and swept exactly as the real tree is, so a regression in the
classifier fails here rather than silently re-bucketing the generated doc.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from scripts.audit_time_assertions import Finding, findings


def _sweep(tmp_path: Path, source: str) -> dict[str, Finding]:
    """Sweep one synthetic module; rows keyed by the function that holds them."""
    root = tmp_path / "tests"
    root.mkdir()
    (root / "test_synthetic.py").write_text(textwrap.dedent(source), encoding="utf-8")
    rows = findings(root)
    by_fn = {row["fn"]: row for row in rows}
    assert len(by_fn) == len(rows), "one assertion per synthetic function"
    return by_fn


def test_noise_in_prose_or_unrelated_names_does_not_make_a_bound_calibrated(
    tmp_path: Path,
) -> None:
    """Q1: the three misfiled shapes — a variable named ``noise``, a comment and
    a docstring mentioning noise — all keep their hard-coded ceiling in B."""
    rows = _sweep(
        tmp_path,
        '''
        import time

        def test_variable_named_noise():
            noise = "x" * 10
            start = time.perf_counter()
            len(noise)
            elapsed = time.perf_counter() - start
            assert elapsed < 2.0

        def test_comment_mentions_noise():
            started = time.perf_counter()
            # tens of ms of scheduling noise
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            assert elapsed_ms < 500.0

        def test_docstring_mentions_noise():
            """Far above clock noise."""
            started = time.perf_counter()
            per_leg_ms = (time.perf_counter() - started) * 1000.0 / 1000.0
            assert per_leg_ms < 2.0
        ''',
    )
    for name in (
        "test_variable_named_noise",
        "test_comment_mentions_noise",
        "test_docstring_mentions_noise",
    ):
        assert rows[name]["calibrated"] is False, name
        assert rows[name]["bucket"] == "B", (name, rows[name]["why"])
    # A millisecond-named measurement is judged in seconds: 500 ms is a 0.5 s
    # ceiling (B), not a 500 s hang detector (D).
    assert "0.5 s" in rows["test_comment_mentions_noise"]["why"]


def test_a_bound_that_names_a_measured_floor_is_calibrated(tmp_path: Path) -> None:
    """The positive control for Q1: the bound side itself references a value
    measured in the run (a clock-derived name, or one spelled as a floor). The
    ``0`` lower bound is what admits each row: an operand that is not wholly
    literal never does."""
    rows = _sweep(
        tmp_path,
        """
        import time

        def test_measured_baseline():
            t0 = time.perf_counter()
            baseline = time.perf_counter() - t0
            t1 = time.perf_counter()
            elapsed = time.perf_counter() - t1
            assert 0.0 < elapsed < baseline * 5

        def test_named_floor(noise_floor):
            t1 = time.perf_counter()
            elapsed = time.perf_counter() - t1
            assert 0 <= elapsed < noise_floor * 3
        """,
    )
    for name in ("test_measured_baseline", "test_named_floor"):
        assert rows[name]["calibrated"] is True, name
        assert rows[name]["bucket"] == "A", (name, rows[name]["why"])


def test_a_clock_on_the_large_side_is_a_lower_bound(tmp_path: Path) -> None:
    """R4/Q4: ``0.0 < created <= time.time() + 1`` — a stall only grows the
    right-hand clock read, so the assertion gets truer. The row exists because
    of ``0.0``; the ``+ 1`` beside a clock read is judged by direction, not
    lifted out as a "0 s" bound."""
    rows = _sweep(
        tmp_path,
        """
        import time

        def test_created_stamp(previews):
            created = previews.created_at("x")
            assert 0.0 < created <= time.time() + 1
        """,
    )
    row = rows["test_created_stamp"]
    assert row["ops"] == ["<", "<="]
    assert row["consts"] == [0.0]
    assert row["ceiling"] is False
    assert row["bucket"] == "A", row["why"]


def test_stamp_checks_and_literal_arithmetic(tmp_path: Path) -> None:
    """A clock on the SMALL side is the crossable stamp-vs-now shape. An
    operand made only of literals is evaluated (``0.4 * 1.8`` is a 0.72 s
    ceiling); one mixing a literal with a name (``budget + 0.3``) is not a
    stated number, so it does not admit a row with its slack as the bound."""
    rows = _sweep(
        tmp_path,
        """
        import time

        def test_now_minus_stamp(stamp):
            assert time.time() - stamp < 5

        def test_literal_arithmetic():
            start = time.perf_counter()
            elapsed = time.perf_counter() - start
            assert elapsed < 0.4 * 1.8

        def test_slack_on_a_name(budget):
            start = time.perf_counter()
            elapsed = time.perf_counter() - start
            assert elapsed <= budget + 0.3

        def test_wide_tolerance(stamp):
            assert time.time() - stamp < 60
        """,
    )
    assert rows["test_now_minus_stamp"]["stamp"] is True
    assert rows["test_now_minus_stamp"]["bucket"] == "B"
    assert "5 s stamp-vs-now" in rows["test_now_minus_stamp"]["why"]
    assert rows["test_wide_tolerance"]["bucket"] == "A"
    assert rows["test_literal_arithmetic"]["bucket"] == "B"
    assert "0.72 s" in rows["test_literal_arithmetic"]["why"]
    assert "test_slack_on_a_name" not in rows


def test_operators_come_from_the_ast_not_the_source_text(tmp_path: Path) -> None:
    """R4: a ``<`` inside a string literal is not a comparison operator. Before,
    ops were regexed out of the unparsed expression, so an equality check whose
    key merely spells ``elapsed <`` was filed as a 0.5 s ceiling in B."""
    rows = _sweep(
        tmp_path,
        """
        import time

        def test_string_holds_an_operator(labels):
            start = time.perf_counter()
            elapsed = time.perf_counter() - start
            assert labels.get("elapsed <") == 0.5, elapsed

        def test_lower_bound():
            start = time.perf_counter()
            elapsed = time.perf_counter() - start
            assert elapsed >= 0.1
        """,
    )
    assert "test_string_holds_an_operator" not in rows
    assert rows["test_lower_bound"]["ops"] == [">="]
    assert rows["test_lower_bound"]["bucket"] == "A"


def test_cpu_ceilings_are_labelled_cpu_and_hangs_are_d(tmp_path: Path) -> None:
    """R7: a CPU-time ceiling says CPU, not wall clock; and the 3 s line still
    splits B from D. A subscript index is not a bound (``timings[0]``)."""
    rows = _sweep(
        tmp_path,
        """
        import time

        def test_cpu_ceiling():
            c0 = time.process_time()
            cpu = time.process_time() - c0
            assert cpu < 0.05

        def test_hang_detector():
            start = time.monotonic()
            elapsed = time.monotonic() - start
            assert elapsed < 10.0

        def test_subscript_is_not_a_bound():
            durations = [time.perf_counter(), time.perf_counter()]
            assert durations[0] <= durations[1]
        """,
    )
    assert rows["test_cpu_ceiling"]["bucket"] == "B"
    assert "CPU-time ceiling" in rows["test_cpu_ceiling"]["why"]
    assert "wall-clock" not in rows["test_cpu_ceiling"]["why"]
    assert rows["test_hang_detector"]["bucket"] == "D"
    assert "test_subscript_is_not_a_bound" not in rows
