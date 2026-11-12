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
    #: The comparison operators, read from ``ast.Compare.ops`` — never regexed
    #: out of source text, where a ``<`` inside a string literal would count.
    ops: list[str]
    #: The value of every operand built only from numeric literals (``0.5``,
    #: ``-1``, ``0.4 * 1.8``). Gates whether the row exists at all.
    consts: list[int | float]
    #: Any operand is a clock read or a name assigned from one.
    measured: bool
    #: Some ``small <= large`` pair puts a measured quantity on the small side
    #: with no fresh clock read on the large side: the only shape a stall can
    #: push past its bound. ``False`` means every pair is a lower bound.
    ceiling: bool
    #: The ceiling pairs' literal bounds IN SECONDS (``elapsed_ms < 500`` -> 0.5).
    bound_consts: list[int | float]
    #: Source of the bound side(s), for the code-constant check.
    bound_expr: str
    #: The bound side names a calibrated / measured-floor value (see
    #: ``CALIBRATED_NAME``) — read from identifiers in the compared expression,
    #: never from the docstring, comments or unrelated variables in the body.
    calibrated: bool
    #: A ``time.time()`` read sits on the measured side: a stamp-vs-now check.
    stamp: bool
    #: The measured side is CPU time, not wall clock.
    cpu: bool
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
#: An identifier on the BOUND side whose name says it was measured in-run (a
#: calibrated noise floor, a baseline). Matched against ``ast.Name``/attribute
#: identifiers of the compared expression only: a QA pass found the earlier
#: whole-body regex filing hard-coded ceilings as calibrated because the word
#: "noise" appeared in a docstring, a comment or an unrelated variable.
CALIBRATED_NAME = re.compile(r"(calibrat|floor|baseline|noise)", re.I)
CPU_NAME = re.compile(r"(\bcpu|cpu_|_cpu|process_time|thread_time)", re.I)
#: ``ast`` comparison operator -> spelling, for the ordering ops this audit reads.
ORDER_OPS: dict[type[ast.cmpop], str] = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">="}
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
    if not row["ceiling"]:
        return "A", "lower bound on elapsed — load only inflates it"
    if CODE_CONST.search(row["bound_expr"]):
        return "A", "derived — the bound is a code constant the implementation owns"
    if row["calibrated"]:
        return "A", "derived — the bound names a value measured in the same run"
    if not row["bound_consts"]:
        # The literal that admitted the row sits on a lower-bound pair, and the
        # ceiling is an expression: neither provably derived nor a stated number.
        return "B", f"bound `{row['bound_expr']}` is a local value — hard-coded unless derived"
    literal = max(abs(value) for value in row["bound_consts"])
    if row["stamp"]:
        if literal >= 30:
            return "A", (
                f"{literal:g} s timestamp tolerance — load does not move wall-clock arithmetic"
            )
        return "B", (
            f"{literal:g} s stamp-vs-now tolerance — a stall between write and read crosses it"
        )
    clock = "CPU-time" if row["cpu"] else "wall-clock"
    if literal >= 3.0:
        return "D", (
            f"{literal:g} s {clock} ceiling over a blocking path — only a real hang trips it"
        )
    return "B", f"hard-coded {literal:g} s {clock} ceiling — a loaded runner crosses it"


def _is_clock_call(node: ast.AST) -> bool:
    """A call to a clock, spelled any way this suite spells one."""
    if not isinstance(node, ast.Call):
        return False
    try:
        dotted = ast.unparse(node.func)
    except Exception:  # noqa: BLE001 — an unparsable callee is simply not a clock
        return False
    return bool(CLOCKCALL.search(dotted + "(")) or any(
        marker in dotted for marker in ("process_time", "thread_time", "loop.time")
    )


def _const_value(node: ast.AST) -> float | None:
    """The value of an operand built ONLY from numeric literals, else ``None``.

    ``0.5``, ``-1`` and ``0.4 * 1.8`` are stated numbers, so they are bounds.
    An operand that mixes a literal with anything else — a name, a call, a
    subscript — is not: ``budget + 0.3`` is judged by what ``budget`` is, not
    by its slack, and ``timings[0]`` is not a "0 s ceiling". Keeping admission
    to whole-literal operands also keeps the sweep's population what the doc
    has always described: comparisons against a NUMBER.
    """
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        inner = _const_value(node.operand)
        if inner is None:
            return None
        return -inner if isinstance(node.op, ast.USub) else inner
    if isinstance(node, ast.BinOp):
        left, right = _const_value(node.left), _const_value(node.right)
        if left is None or right is None:
            return None
        try:
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
        except ZeroDivisionError:
            return None
    return None


#: Unit suffixes on the MEASURED side, so ``elapsed_ms < 500.0`` is judged as
#: the 0.5 s it is rather than as a 500 s hang detector.
UNIT_SCALE: tuple[tuple[re.Pattern[str], float], ...] = (
    (re.compile(r"(_ms\b|\bms\b|millis)", re.I), 1e-3),
    (re.compile(r"(_us\b|\bus\b|usec|micros)", re.I), 1e-6),
    (re.compile(r"(_ns\b|\bns\b|nanosec)", re.I), 1e-9),
)


def _unit_scale(measured: ast.AST) -> float:
    """Seconds per unit of the measured side: from its name, or ``* 1000``."""
    for sub in ast.walk(measured):
        if (
            isinstance(sub, ast.BinOp)
            and isinstance(sub.op, ast.Mult)
            and any(
                isinstance(side, ast.Constant) and side.value in (1000, 1000.0)
                for side in (sub.left, sub.right)
            )
        ):
            return 1e-3
    try:
        spelled = ast.unparse(measured)
    except Exception:  # noqa: BLE001 — no text means no unit hint; seconds
        return 1.0
    for pattern, scale in UNIT_SCALE:
        if pattern.search(spelled):
            return scale
    return 1.0


def _identifiers(node: ast.AST) -> list[str]:
    """Names and attribute names in ``node``, excluding called functions."""
    callees = {id(sub.func) for sub in ast.walk(node) if isinstance(sub, ast.Call)}
    out: list[str] = []
    for sub in ast.walk(node):
        if id(sub) in callees:
            continue
        if isinstance(sub, ast.Name):
            out.append(sub.id)
        elif isinstance(sub, ast.Attribute):
            out.append(sub.attr)
    return out


class CompareFacts(TypedDict):
    """What :func:`describe_compare` reads off one ``ast.Compare``; see :class:`Finding`."""

    ops: list[str]
    consts: list[int | float]
    measured: bool
    ceiling: bool
    bound_consts: list[int | float]
    bound_expr: str
    calibrated: bool
    stamp: bool
    cpu: bool


def describe_compare(compare: ast.Compare, names: set[str]) -> CompareFacts | None:
    """The AST facts ``classify`` reads, or ``None`` when ``compare`` is no bound.

    Each adjacent pair of a chained comparison is normalised to ``small OP
    large``. A pair is a CEILING when the small side is measured (a clock read,
    or a name assigned from one) and the large side holds no fresh clock read:
    then a stall grows the small side toward the bound. A clock on the large
    side (``created <= time.time() + 1``) only grows under load, so that pair
    gets truer, not falser — a lower bound on elapsed.

    The bound is CALIBRATED only when the large side itself names a value
    measured in the run: an identifier assigned from a clock read in the same
    function, or one spelled like a floor/baseline (``CALIBRATED_NAME``).
    """
    operands = [compare.left, *compare.comparators]

    def measured(node: ast.AST) -> bool:
        return any(_is_clock_call(sub) for sub in ast.walk(node)) or any(
            ident in names for ident in _identifiers(node)
        )

    ops: list[str] = []
    ceilings: list[tuple[ast.AST, ast.AST]] = []
    for index, op in enumerate(compare.ops):
        spelled = ORDER_OPS.get(type(op))
        if spelled is None:
            continue
        ops.append(spelled)
        left, right = operands[index], operands[index + 1]
        small, large = (left, right) if spelled in ("<", "<=") else (right, left)
        if measured(small) and not any(_is_clock_call(sub) for sub in ast.walk(large)):
            ceilings.append((small, large))
    if not ops:
        return None
    consts = [
        value for value in (_const_value(operand) for operand in operands) if value is not None
    ]
    if not consts:
        return None

    def text(node: ast.AST) -> str:
        try:
            return ast.unparse(node)
        except Exception:  # noqa: BLE001 — an unparsable node must not drop a finding
            return ""

    smalls = " ".join(text(small) for small, _large in ceilings)
    stamp = "time.time()" in smalls
    bound_consts: list[int | float] = []
    for small, large in ceilings:
        value = _const_value(large)
        if value is not None:
            # A stamp check is in seconds by construction; elsewhere the
            # measured side's unit decides (``elapsed_ms < 500`` is 0.5 s).
            bound_consts.append(value * (1.0 if stamp else _unit_scale(small)))
    return {
        "ops": ops,
        "consts": consts,
        "measured": any(measured(operand) for operand in operands),
        "ceiling": bool(ceilings),
        "bound_consts": bound_consts,
        "bound_expr": " ".join(text(large) for _small, large in ceilings),
        "calibrated": any(
            ident in names or bool(CALIBRATED_NAME.search(ident))
            for _small, large in ceilings
            for ident in _identifiers(large)
        ),
        "stamp": stamp,
        "cpu": bool(CPU_NAME.search(smalls)),
    }


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
                    key = (str(path), compare.lineno, expr)
                    if key in seen:
                        continue
                    if names is None:
                        names = clock_names(fn)
                    facts = describe_compare(compare, names)
                    if facts is None:
                        continue
                    seen.add(key)
                    rows.append(
                        {
                            "file": str(path.relative_to(root.parent)),
                            "line": compare.lineno,
                            "fn": fn.name,
                            "expr": " ".join(expr.split()),
                            "ops": facts["ops"],
                            "consts": facts["consts"],
                            "measured": facts["measured"],
                            "ceiling": facts["ceiling"],
                            "bound_consts": facts["bound_consts"],
                            "bound_expr": facts["bound_expr"],
                            "calibrated": facts["calibrated"],
                            "stamp": facts["stamp"],
                            "cpu": facts["cpu"],
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


#: The ceilings this PR replaced with the property behind them, by pytest node
#: id: (node, the removed assertion, what it asserts now). Hand-written because
#: it is history, not something a sweep can recover — and keyed by node id so
#: it cannot drift onto the wrong line as the files change.
CONVERTED: tuple[tuple[str, str, str], ...] = (
    (
        "tests/unit/tui/test_provider_recovery_hint.py"
        "::test_the_sync_arm_is_cache_only_and_never_blocks",
        "elapsed < 0.5",
        "the sync arm called the cache-only producer exactly once, and never reached the "
        "probing producer, a probe twin, the network or `time.sleep`",
    ),
    (
        "tests/unit/tui/test_provider_recovery_hint.py"
        "::test_a_deliberately_slow_probe_does_not_delay_the_sync_arm",
        "elapsed < 0.5",
        'the same provenance check with 5 s probes armed — "does not delay" derived from '
        '"the only slow things on the path were never entered"',
    ),
    (
        "tests/unit/session/test_index_prewarm.py"
        "::test_starting_the_warm_never_blocks_and_never_raises",
        "elapsed < 0.05",
        "no journal had reached `start_refresh` when the synchronous starter returned, so "
        "startup scheduled the warm rather than running it",
    ),
    (
        "tests/unit/tools/test_builtin_tools.py"
        "::test_a_stderr_with_no_signature_phrase_is_never_scanned",
        "elapsed < 0.05",
        "the signature regex is never ENTERED for phrase-free stderr, and entered exactly "
        "once for input that carries the phrase (the control)",
    ),
    (
        "tests/unit/team_bridge/test_inline_data.py"
        "::test_a_one_megabyte_marker_dense_body_stays_fast",
        "elapsed < 0.05",
        "one marker search for a body carrying 50,000 markers, which no per-URI parser "
        "can satisfy",
    ),
    (
        "tests/unit/tui/test_subagent_stats.py::test_pricing_a_child_never_blocks_the_paint",
        "elapsed < 0.2",
        "the repaint resolved no model on its own thread (a 10 s listing there would be "
        "a blocked paint)",
    ),
)


def _bound(row: Finding) -> float:
    """The number a row is judged by, in seconds: its ceiling literal, else any literal."""
    return max(abs(value) for value in (row["bound_consts"] or row["consts"]))


def render(rows: list[Finding], settles: list[SettleNode], label: str) -> str:
    import collections

    counts = collections.Counter(row["bucket"] for row in rows)
    lines: list[str] = [
        "# Time and CPU assertions in the test suite",
        "",
        "GENERATED by `scripts/audit_time_assertions.py`, which sweeps `tests/` in"
        " the working tree it runs in"
        + (f" (labelled `{label}`)" if label else "")
        + ". Re-run it rather than editing this file by hand.",
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
        "The counts describe the tree this document is committed with; every",
        "count in this file is computed by the sweep, none is typed in.",
        "",
        "## Converted in this PR",
        "",
        f"{len(CONVERTED)} bucket-B ceilings were replaced by the property behind them.",
        "None had its number raised: a wider constant moves the failure rather than",
        "removing it. Nodes are cited by pytest node id, which does not drift the way",
        "a line number does.",
        "",
        "| node | was | now asserts |",
        "| --- | --- | --- |",
    ]
    for node_id, was, now in CONVERTED:
        lines.append(f"| `{node_id}` | `{was}` | {now} |")
    b_rows = sorted(
        (row for row in rows if row["bucket"] == "B"),
        key=lambda r: (_bound(r), r["file"], r["line"]),
    )
    converted_ids = {node_id for node_id, _was, _now in CONVERTED}
    still_b = sorted({f"{r['file']}::{r['fn']}" for r in b_rows} & converted_ids)
    lines += [
        "",
        "Each conversion carries a negative control in its PR: reverting the product",
        "behaviour fails the new assertion deterministically, fast host or loaded, with",
        "no timing involved. "
        + (
            "None of these nodes still holds a bucket-B row."
            if not still_b
            else f"{len(still_b)} still hold a bucket-B row: "
            + ", ".join(f"`{n}`" for n in still_b)
            + "."
        ),
        "",
        "## Still to convert",
        "",
        f"Bucket B has {len(b_rows)} entries, tightest first in the table below. The",
        "five tightest:",
        "",
    ]
    for row in b_rows[:5]:
        lines.append(f"- `{row['file']}:{row['line']}` ({_bound(row):g} s) — `{row['fn']}`")
    lines.append("")

    lines += [
        "## Bucket B — hard-coded ceilings (the future red shards)",
        "",
        "Sorted by the ceiling, tightest first: the tighter the number, the less",
        "the host has to stall for the assertion to fail.",
        "",
        "| file:line | assertion | bound | why it can cross |",
        "| --- | --- | --- | --- |",
    ]
    for row in b_rows:
        bound = _bound(row)
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
        bound = _bound(row)
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
    # A LABEL, not a checkout: the sweep always reads the working tree under
    # --root, so the header must not claim a revision it never read.
    parser.add_argument("--label", default="", help="optional label for the swept tree")
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

    text = render(rows, settles, args.label)
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
