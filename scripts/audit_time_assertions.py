#!/usr/bin/env python3
"""Audit every assertion in ``tests/`` that encodes a time or CPU quantity.

WHY. A test that asserts a wall-clock ceiling on work whose duration the host
controls is a future red shard: the code can be unchanged and the runner busier.
This script exists so the inventory is generated rather than recalled — the
question "which of these are derived and which are hard-coded?" is answered by
re-running it, not by grepping and hoping the list is still current.

WHAT IT DOES. Parses every test module, keeps the ``assert`` statements whose
comparison mixes a time/CPU quantity with a numeric literal, resolves whether
the compared name came from a clock read in the same function (so a policy
constant check is not mistaken for a latency measurement), and buckets each hit:

``A``  derived / load-independent — no clock at all (code constants, policy
       values), a lower bound (load only inflates elapsed), a bound owned by a
       code constant, a calibrated noise floor, or a wide timestamp tolerance.
``B``  hard-coded wall-clock ceiling a loaded runner can cross, under 3 s.
``C``  sleep-then-assert: a fixed ``pilot.pause()``/``sleep`` immediately before
       an assertion. Order-dependent, no property behind it.
``D``  bounded wait: a ceiling of 3 s or more over a blocking path, which only a
       real hang trips.

The 3 s line between B and D is stated rather than tuned: a runner can add
hundreds of milliseconds to any call (CI measured 653 ms of loop CPU against a
530 ms calibrated ceiling, and a 0.99 s sync call against a 0.5 s ceiling), so a
sub-second and low-single-digit-second bound is inside scheduling noise, while a
3 s-plus ceiling is already the failure mode "this hung".

USAGE::

    .venv/bin/python scripts/audit_time_assertions.py --out docs/TEST_TIME_ASSERTIONS.md
    .venv/bin/python scripts/audit_time_assertions.py --bucket B    # just the risk
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import re
import sys
from typing import TypedDict


class Finding(TypedDict):
    """One time/CPU assertion. ``bucket``/``why`` start empty and are filled by
    ``classify`` once the row exists, because classification reads the other
    fields; keeping them required (not ``NotRequired``) lets the renderers index
    them without a presence check."""

    file: str
    line: int
    fn: str
    expr: str
    op: str
    consts: list[int | float]
    measured: bool
    body: str
    bucket: str
    why: str


class SettleNode(TypedDict):
    """A fixed wait whose very next statement is an assertion (bucket C)."""

    file: str
    line: int
    fn: str
    wait: str


#: What a time/CPU quantity is spelled like in this suite.
TIME_STRICT = re.compile(
    r"(perf_counter|time\.monotonic|time\.time|monotonic\("
    r"|elapsed|\bduration|_duration|\btook\b|\blatency"
    r"|\bms\b|_ms\b|millis|seconds|secs|\bsec\b|_seconds|\bsecs\b"
    r"|timeout|deadline|budget|ceiling|\bcpu\b|usec|micro|nanosec|\bns\b)",
    re.I,
)
CLOCKCALL = re.compile(r"perf_counter|time\.monotonic|monotonic\(|time\.time\(")
CODE_CONST = re.compile(r"\b[A-Z][A-Z0-9_]{3,}\b")
CALIBRATED = re.compile(r"(calibrat|noise)", re.I)
FUNC = (ast.FunctionDef, ast.AsyncFunctionDef)
SETTLE = re.compile(r"(\.pause\(\)|asyncio\.sleep\(|time\.sleep\()")


class Src:
    """Source text with a line-range helper, and no character offsets.

    COL_OFFSET IS A UTF-8 BYTE OFFSET, not a character index, so slicing a
    ``str`` by it silently misaligns on any line carrying a non-ASCII character
    — and this repository's comments are full of em dashes. An earlier version
    sliced that way, fell back to ``ast.get_source_segment`` on every misaligned
    node, and turned a 35 s sweep into one that did not finish in five minutes.
    Line numbers ARE reliable, so bodies are taken by line range, and single
    expressions are rendered with ``ast.unparse`` instead of sliced at all.
    """

    def __init__(self, text: str) -> None:
        self.text = text
        self.lines = text.splitlines(keepends=True)

    def body(self, node: ast.AST) -> str:
        """The source of ``node``, by line range (1-based, inclusive)."""
        try:
            first = node.lineno - 1  # type: ignore[attr-defined]
            last = node.end_lineno  # type: ignore[attr-defined]
            return "".join(self.lines[first:last])
        except Exception:  # noqa: BLE001 — a range failure must not drop a finding
            return ""

    def seg(self, node: ast.AST) -> str:
        """A single expression, normalised. ``unparse`` is offset-independent."""
        try:
            return ast.unparse(node)
        except Exception:  # noqa: BLE001 — an unparsable node must not drop a finding
            return ""


def clock_names(fn: ast.AST) -> set[str]:
    """Names in ``fn`` whose value came from a clock read.

    Walked rather than text-matched: unparsing every assignment to find the word
    "perf_counter" made the sweep quadratic-ish and slow enough to be unusable,
    and the AST already says exactly which calls are clocks.
    """

    def is_clock(node: ast.AST) -> bool:
        if not isinstance(node, ast.Call):
            return False
        target = node.func
        parts: list[str] = []
        while isinstance(target, ast.Attribute):
            parts.append(target.attr)
            target = target.value
        if isinstance(target, ast.Name):
            parts.append(target.id)
        dotted = ".".join(reversed(parts))
        return any(
            marker in dotted
            for marker in ("perf_counter", "monotonic", "time.time", "process_time", "loop.time")
        )

    names: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            value, targets = node.value, node.targets
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            value, targets = node.value, [node.target]
        else:
            continue
        if not any(is_clock(sub) for sub in ast.walk(value)):
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                names.add(target.id)
            elif isinstance(target, ast.Tuple):
                names.update(e.id for e in target.elts if isinstance(e, ast.Name))
    return names


def classify(row: Finding) -> tuple[str, str]:
    """Bucket one finding. Rules are ordered; the first match wins."""
    if not row["measured"]:
        return "A", "no clock read — code constants / policy values; load cannot move it"
    ops = re.findall(r"<=|>=|<|>", row["expr"])
    if not any(op in ("<", "<=") for op in ops):
        return "A", "lower bound on elapsed — load only inflates it"
    literal = max(abs(value) for value in row["consts"])
    if CODE_CONST.search(row["expr"]):
        return "A", "derived — the bound is a code constant the implementation owns"
    if "time.time()" in row["expr"]:
        if literal >= 30:
            return "A", (
                f"{literal:g} s timestamp tolerance — load does not move wall-clock arithmetic"
            )
        return "B", (
            f"{literal:g} s stamp-vs-now tolerance — a stall between write and read crosses it"
        )
    if CALIBRATED.search(row["body"]):
        return "A", "derived — calibrated against a measured noise floor"
    if literal >= 3.0:
        return "D", f"{literal:g} s ceiling over a blocking path — only a real hang trips it"
    return "B", f"hard-coded {literal:g} s wall-clock ceiling — a loaded runner crosses it"


def findings(root: pathlib.Path) -> list[Finding]:
    rows: list[Finding] = []
    seen: set[tuple[str, int, str]] = set()
    for path in sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts):
        try:
            src_text = path.read_text(encoding="utf-8")
            tree = ast.parse(src_text)
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue
        src = Src(src_text)
        for fn in (node for node in ast.walk(tree) if isinstance(node, FUNC)):
            names: set[str] | None = None
            for substatement in ast.walk(fn):
                if not isinstance(substatement, ast.Assert):
                    continue
                for compare in ast.walk(substatement.test):
                    if not isinstance(compare, ast.Compare):
                        continue
                    expr = src.seg(compare)
                    if not TIME_STRICT.search(expr):
                        continue
                    ops = re.findall(r"<=|>=|<|>", expr)
                    if not ops:
                        continue
                    consts = [
                        operand.value
                        for operand in [compare.left, *compare.comparators]
                        if isinstance(operand, ast.Constant)
                        and isinstance(operand.value, (int, float))
                        and not isinstance(operand.value, bool)
                    ]
                    if not consts:
                        continue
                    key = (str(path), compare.lineno, expr)
                    if key in seen:
                        continue
                    seen.add(key)
                    if names is None:
                        names = clock_names(fn)
                    rows.append(
                        {
                            "file": str(path.relative_to(root.parent)),
                            "line": compare.lineno,
                            "fn": fn.name,
                            "expr": " ".join(expr.split()),
                            "op": ops[0],
                            "consts": consts,
                            "measured": bool(CLOCKCALL.search(expr))
                            or any(re.search(rf"\b{re.escape(n)}\b", expr) for n in names),
                            "body": src.body(fn)[:2600],
                            "bucket": "",
                            "why": "",
                        }
                    )
    for row in rows:
        row["bucket"], row["why"] = classify(row)
    return rows


def settle_nodes(root: pathlib.Path) -> list[SettleNode]:
    """Fixed waits whose very next statement is an assertion (bucket C)."""
    out: list[SettleNode] = []
    for path in sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts):
        try:
            src_text = path.read_text(encoding="utf-8")
            tree = ast.parse(src_text)
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue
        src = Src(src_text)
        for fn in (node for node in ast.walk(tree) if isinstance(node, FUNC)):
            if not fn.name.startswith("test"):
                continue
            for index, stmt in enumerate(fn.body):
                if not SETTLE.search(src.body(stmt)):
                    continue
                following = fn.body[index + 1] if index + 1 < len(fn.body) else None
                if isinstance(following, ast.Assert):
                    out.append(
                        {
                            "file": str(path.relative_to(root.parent)),
                            "line": stmt.lineno,
                            "fn": fn.name,
                            "wait": (
                                "pilot.pause()" if ".pause()" in src.body(stmt) else "sleep(...)"
                            ),
                        }
                    )
    return out


def render(rows: list[Finding], settles: list[SettleNode], base: str) -> str:
    import collections

    counts = collections.Counter(row["bucket"] for row in rows)
    lines: list[str] = [
        "# Time and CPU assertions in the test suite",
        "",
        "GENERATED by `scripts/audit_time_assertions.py` against "
        f"`{base}`. Re-run it rather than editing this file by hand.",
        "",
        "Every assertion in `tests/` that mixes a time or CPU quantity with a",
        "numeric literal, bucketed by whether it is DERIVED from something real",
        "or HARD-CODED to a number the host controls:",
        "",
        "| Bucket | Count | Meaning |",
        "| --- | --- | --- |",
        f"| A | {counts['A']} | derived or load-independent — safe by construction |",
        f"| B | {counts['B']} | hard-coded wall-clock ceiling under 3 s — a future red shard |",
        f"| C | {len(settles)} | fixed `pilot.pause()`/`sleep` before an assertion — "
        "order-dependent |",
        f"| D | {counts['D']} | bounded wait of 3 s or more — fires only on a real hang |",
        f"| **total** | **{sum(counts.values())} time/CPU asserts + {len(settles)} settle "
        "nodes** | |",
        "",
        "The 3 s line between B and D is stated rather than tuned: CI has measured a",
        "loaded runner adding hundreds of milliseconds to a call the product did not",
        "slow down (653 ms of loop CPU against a 530 ms calibrated ceiling; a 0.99 s",
        "sync call against a 0.5 s ceiling). Sub-second and low-single-digit-second",
        "bounds are inside that noise; a 3 s-plus ceiling is already the failure mode",
        '"this hung".',
        "",
        "Bucket C is an aggregate rather than a list because each node is one",
        "`pilot.pause()`, and the useful number is where they cluster.",
        "",
        "The counts describe the tree AS COMMITTED WITH THIS DOCUMENT. The sweep that",
        "opened the work found 314 comparisons with 78 in bucket B; six became derived",
        "in the same PR (below), which is the whole difference.",
        "",
        "## Converted in this PR",
        "",
        "Six bucket-B ceilings were replaced by the property behind them. None had its",
        "number raised: a wider constant moves the failure rather than removing it.",
        "",
        "| node | was | now asserts |",
        "| --- | --- | --- |",
        "| `tests/unit/tui/test_provider_recovery_hint.py:113` "
        "(`test_the_sync_arm_is_cache_only_and_never_blocks`) | `elapsed < 0.5` | the sync arm "
        "called the cache-only producer exactly once, and never reached the probing "
        "producer, a probe twin, the network or `time.sleep` |",
        "| `tests/unit/tui/test_provider_recovery_hint.py:155` "
        "(`test_a_deliberately_slow_probe_does_not_delay_the_sync_arm`) | `elapsed < 0.5` | "
        'the same provenance check with 5 s probes armed — "does not delay" derived '
        'from "the only slow things on the path were never entered" |',
        "| `tests/unit/session/test_index_prewarm.py:216` | `elapsed < 0.05` | the refresh had "
        "not begun when the starter returned (its gate was still shut), so startup "
        "scheduled the warm rather than running it |",
        "| `tests/unit/tools/test_builtin_tools.py:8262` | `elapsed < 0.05` | the signature "
        "regex is never ENTERED for phrase-free stderr, and entered exactly once for "
        "input that carries the phrase (the control) |",
        "| `tests/unit/team_bridge/test_inline_data.py:165` | `elapsed < 0.05` | one marker "
        "search for a body carrying 50,000 markers, which no per-URI parser can satisfy |",
        "| `tests/unit/tui/test_subagent_stats.py:1704` | `elapsed < 0.2` | the repaint "
        "resolved no model on its own thread (a 10 s listing there would be a blocked "
        "paint) |",
        "",
        "Each conversion carries a negative control in its PR: reverting the product",
        "behaviour fails the new assertion deterministically, fast host or loaded, with",
        "no timing involved.",
        "",
        "## Still to convert",
        "",
        "Bucket B is 72 entries after these six, tightest first in the table above.",
        "The next five by shard cost are `test_mcp/test_manager.py:1860` (0.09 s, the",
        "teardown-was-not-a-polling-tick claim), `test_oauth_401_detached_persist.py:330`",
        "(0.1 s), `test_saver/test_round1_remediation.py:803` (0.1 s),",
        "`test_session_factory_classification.py:1605` (0.2 s) and",
        "`test_desktop_sessions.py:8352` (0.3 s). None is hard to convert; they are the",
        "remaining calls on the plan rather than a design problem.",
        "",
    ]

    lines += [
        "## Bucket B — hard-coded ceilings (the future red shards)",
        "",
        "Sorted by the ceiling, tightest first: the tighter the number, the less",
        "the host has to stall for the assertion to fail.",
        "",
        "| file:line | assertion | bound | why it can cross |",
        "| --- | --- | --- | --- |",
    ]
    for row in sorted(rows, key=lambda r: (max(abs(v) for v in r["consts"]), r["file"], r["line"])):
        if row["bucket"] != "B":
            continue
        bound = max(abs(value) for value in row["consts"])
        expr = row["expr"].replace("|", "\\|")
        lines.append(f"| `{row['file']}:{row['line']}` | `{expr}` | {bound:g} s | {row['why']} |")

    lines += [
        "",
        "## Bucket D — hang detectors",
        "",
        "A ceiling of 3 s or more, where a real regression is a HANG rather than",
        "a slowdown, so the number only has to be safely larger than the work.",
        "",
        "| file:line | assertion | bound | why it is D and not B |",
        "| --- | --- | --- | --- |",
    ]
    for row in sorted(rows, key=lambda r: (r["file"], r["line"])):
        if row["bucket"] != "D":
            continue
        bound = max(abs(value) for value in row["consts"])
        expr = row["expr"].replace("|", "\\|")
        lines.append(f"| `{row['file']}:{row['line']}` | `{expr}` | {bound:g} s | {row['why']} |")

    lines += [
        "",
        "## Bucket A — derived or load-independent",
        "",
        "| file:line | assertion | why it is safe |",
        "| --- | --- | --- |",
    ]
    for row in sorted(rows, key=lambda r: (r["file"], r["line"])):
        if row["bucket"] != "A":
            continue
        expr = row["expr"].replace("|", "\\|")
        lines.append(f"| `{row['file']}:{row['line']}` | `{expr}` | {row['why']} |")

    lines += [
        "",
        "## Bucket C — sleep-then-assert",
        "",
        f"{len(settles)} spots where a fixed wait is the very next statement before an",
        "assertion. They cluster hard, which is what makes them one shared helper's",
        "job rather than a hundred edits:",
        "",
        "| file | count |",
        "| --- | --- |",
    ]
    per_file = collections.Counter(node["file"] for node in settles)
    for path, count in per_file.most_common(25):
        lines.append(f"| `{path}` | {count} |")
    waits = collections.Counter(node["wait"] for node in settles)
    lines += ["", "| wait | count |", "| --- | --- |"]
    for wait, count in waits.most_common():
        lines.append(f"| `{wait}` | {count} |")
    lines += [
        "",
        "`tests/unit/tui/_settle.py` is the shared helper: it pumps a bounded COUNT of",
        "event-loop turns until the product's own condition holds. A turn count survives",
        "contention where a wall-clock budget does not — a loaded host inflates how long",
        "each turn takes, not how many turns the app needs.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="tests", help="test tree to sweep (default: tests)")
    parser.add_argument("--out", default="", help="write the markdown here instead of stdout")
    parser.add_argument("--base", default="origin/main", help="revision the sweep ran against")
    parser.add_argument("--bucket", default="", help="print only one bucket (A/B/C/D)")
    args = parser.parse_args()

    root = pathlib.Path(args.root).resolve()
    rows = findings(root)
    settles = settle_nodes(root)

    if args.bucket:
        wanted = args.bucket.upper()
        if wanted == "C":
            for node in settles:
                print(f"{node['file']}:{node['line']}  {node['fn']}  {node['wait']}")
        else:
            for row in sorted(rows, key=lambda r: (r["file"], r["line"])):
                if row["bucket"] == wanted:
                    print(f"{row['file']}:{row['line']}  {row['expr']}")
        return 0

    text = render(rows, settles, args.base)
    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"{len(rows)} time/CPU asserts, {len(settles)} settle nodes -> {out}")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
