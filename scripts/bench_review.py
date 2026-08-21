#!/usr/bin/env python3
"""A/B benchmark for reviewer-subagent model choice.

The question this answers is the one the agent seeds had to guess at: does a
review round on a CHEAPER tier model finish faster without going blind, and
does a STRONGER session-class model pay for itself in coverage — or just in
latency and unrelated noise?

``bench_task_cost.py`` measures the harness doing implementation work and
verifies outcomes itself. This does the same for REVIEW: it scaffolds a small
repo as a git repository, plants a diff with a known set of defects, runs the
``reviewer`` role against it headlessly, and then scores the findings from
this process. A reviewer cannot inflate its own coverage number because the
ground truth (which defects exist, which files are off-limits) never enters
its prompt.

Two signals, both decided script-side:

- **Coverage** — how many of the planted defects appear in the review, matched
  by keyword per defect. A planted defect the review never mentions is a miss
  no matter how thorough the prose sounds.
- **Discipline** — findings rooted OUTSIDE the diff (``unrelated``), matched
  by file reference. The scaffold ships two decoy files with smelly but
  untouched code; citing them is the "unrelated findings" failure priced
  against whatever the model saved.

Reported per arm: turns, tool calls, tokens (fresh/cached/output), warm cache
rate, cost at live prices, wall time, coverage (found/planted), unrelated
findings, and severity-miscounts (a planted defect reported below MAJOR).

Run:
    OPENROUTER_API_KEY=... .venv/bin/python scripts/bench_review.py \
        --arm lo=deepseek/deepseek-v4-flash-0731 \
        --arm hi=anthropic/claude-sonnet-4-5 --runs 2
    .venv/bin/python scripts/bench_review.py --arm only=MODEL --runs 1 --keep
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DEFAULT_HOSTING = "openrouter"
DEFAULT_TIMEOUT_S = 900


# --------------------------------------------------------------------------
# The scaffold: base tree, planted diff, ground truth
# --------------------------------------------------------------------------
#
# The base tree is a small "inventory service". The diff adds a discount
# feature and refactors the pricing path, planting defects that are each
# invisible to a skim and cheap to confirm with the tests present in the tree
# — the same shape the reviewer guidance tells the role to chase (run what is
# cheap to run). Every keyword matcher below is quoted in the GROUND TRUTH
# table the report prints, so a matched finding is auditable back to the
# review text.

BASE_FILES: dict[str, str] = {
    "inventory/__init__.py": "",
    "inventory/models.py": '''"""Domain records for the inventory service."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Product:
    sku: str
    name: str
    unit_price_cents: int
    stock: int
''',
    "inventory/pricing.py": '''"""Pricing math. All money is integer cents; callers never see floats."""

from inventory.models import Product


def line_total(product: Product, quantity: int) -> int:
    """Gross line total in cents."""
    if quantity < 0:
        raise ValueError("quantity must be >= 0")
    return product.unit_price_cents * quantity


def order_total(lines: list[tuple[Product, int]]) -> int:
    """Gross order total in cents."""
    return sum(line_total(product, qty) for product, qty in lines)
''',
    "inventory/service.py": '''"""Order placement over the pricing module."""

from inventory.models import Product
from inventory.pricing import order_total


def place_order(lines: list[tuple[Product, int]]) -> dict:
    total = order_total(lines)
    return {"status": "ok", "total_cents": total}
''',
    "tests/test_pricing.py": '''import pytest

from inventory.models import Product
from inventory.pricing import line_total, order_total
from inventory.service import place_order


def make(price=1000, stock=10):
    return Product(sku="SKU-1", name="widget", unit_price_cents=price, stock=stock)


def test_line_total():
    assert line_total(make(), 3) == 3000


def test_line_total_rejects_negative_quantity():
    with pytest.raises(ValueError):
        line_total(make(), -1)


def test_order_total_sums_lines():
    assert order_total([(make(), 2), (make(500), 4)]) == 4000


def test_place_order_reports_total():
    assert place_order([(make(), 1)])["total_cents"] == 1000
''',
    # DECOY 1: smelly but UNTOUCHED by the diff. A reviewer citing this file
    # wandered off the diff — exactly the unrelated-finding failure.
    "legacy/importer.py": '''"""CSV importer scheduled for deletion. Known-ugly; do not touch."""


def load_rows(path):
    import csv  # local import, mutable default below, broad except: legacy

    cache = []
    with open(path) as handle:  # never closed on some forks, encoding unset
        for row in csv.reader(handle):
            try:
                cache.append((row[0], int(row[1])))
            except Exception:
                pass  # swallows everything, on purpose, since 2019
    return cache
''',
    # DECOY 2: a second off-diff target, so "unrelated" is not a one-file
    # accident a model can luck out of.
    "tools/reindex.py": '''"""One-shot backfill kept for reference. Known-ugly; do not touch."""

import subprocess


def reindex(table):
    # Shell string built by concatenation; dead code kept for the audit trail.
    subprocess.run("reindexdb --table " + table, shell=True, check=False)
''',
}

# Files the diff touches. Findings rooted anywhere else are `unrelated`.
DIFF_FILES = ("inventory/pricing.py", "inventory/service.py", "tests/test_pricing.py")

# The planted defects. Scoring a defect as FOUND takes BOTH of:
#
# 1. ``topic`` -- the defect's identifying vocabulary (what it is about), and
# 2. ``assertion`` -- defect-asserting vocabulary (that something is WRONG:
#    missing, removed, wrong, never, broken, ...).
#
# Keyword-only matching cannot tell a finding from an exoneration: "the
# ValueError guard is preserved correctly" contains every keyword of the
# dropped-guard defect. Review rounds 2 and 3 each reproduced the failure on
# different matchers -- the fix is structural, not a tighter keyword list: a
# bare ``stock`` or ``guard`` or ``10% off`` can never score on its own, and
# review quotes of the code ("dividing by 10000 is correct") carry topic but
# no assertion.
PLANTED: list[tuple[str, str, str, str]] = [
    (
        "percentage-off-by-100x",
        "MAJOR",
        # pct=10 means 10%; the diff divides by 10000 and prices at 0.1%.
        r"10000|percent|pct\b|discount|divisor",
        r"wrong|incorrect|instead of|should (be|have been)|100\s?[x\u00d7]|factor of 100|two orders of magnitude|0\.1\s?%|not 900|bug|broken|off by",
    ),
    (
        "negative-discount-inflates-total",
        "MAJOR",
        # No validation on pct: pct=-50 CHARGES 150%.
        r"negative|validat|pct|discount",
        r"missing|no validation|unvalidated|(without|lacks?).{0,20}validat|inflat|nonsense|silent|wrong|accepts any|not (validated|checked|clamped)",
    ),
    (
        "rounding-float-cents",
        "MINOR",
        # / returns float; the module's own contract says integer cents.
        r"float|round|integer cents|int\(\)|//|true division",
        r"break|violat|wrong|invariant|instead|should|bug|defect|returns a float|not an? int",
    ),
    (
        "stock-never-decremented",
        "MAJOR",
        # place_order still ignores Product.stock; overselling stays possible.
        r"stock|inventory|oversell",
        r"never|not|missing|without|ignored|untrack|decrement|reduc|updat|oversell",
    ),
    (
        "lost-negative-quantity-guard",
        "BLOCKER",
        # The refactor dropped `quantity < 0 -> ValueError`; negative
        # quantities now produce negative (refund) totals silently.
        r"negative (quantity|quantities)|ValueError|guard|refund",
        r"drop|remov|lost|gone|missing|no longer|silent|regress|delet",
    ),
]


DIFF_FILES_CONTENT: dict[str, str] = {
    "inventory/pricing.py": '''"""Pricing math. All money is integer cents; callers never see floats."""

from inventory.models import Product


def line_total(product: Product, quantity: int) -> int:
    """Gross line total in cents."""
    return product.unit_price_cents * quantity


def discounted_total(product: Product, quantity: int, pct: int) -> int:
    """Line total with a percentage discount, e.g. pct=10 takes 10% off."""
    gross = line_total(product, quantity)
    return gross - (gross * pct) / 10000


def order_total(lines: list[tuple[Product, int]]) -> int:
    """Gross order total in cents."""
    return sum(line_total(product, qty) for product, qty in lines)
''',
    "inventory/service.py": '''"""Order placement over the pricing module."""

from inventory.models import Product
from inventory.pricing import discounted_total, order_total


def place_order(lines: list[tuple[Product, int]], discount_pct: int = 0) -> dict:
    if discount_pct:
        total = sum(discounted_total(p, q, discount_pct) for p, q in lines)
    else:
        total = order_total(lines)
    return {"status": "ok", "total_cents": total}
''',
    "tests/test_pricing.py": '''import pytest

from inventory.models import Product
from inventory.pricing import discounted_total, line_total, order_total
from inventory.service import place_order


def make(price=1000, stock=10):
    return Product(sku="SKU-1", name="widget", unit_price_cents=price, stock=stock)


def test_line_total():
    assert line_total(make(), 3) == 3000


def test_order_total_sums_lines():
    assert order_total([(make(), 2), (make(500), 4)]) == 4000


def test_place_order_reports_total():
    assert place_order([(make(), 1)])["total_cents"] == 1000


def test_discounted_total_takes_ten_percent():
    assert discounted_total(make(), 1, 10) == 900
''',
}

REVIEW_PROMPT = """Review the changes on this branch. The base is the `main` branch; the work is on the current HEAD.

Use `git diff main..HEAD` to see the change. The review is the diff; the tree is context.

There is a test suite (`pytest`) you can run to check your claims cheaply.

Report findings classified BLOCKER / MAJOR / MINOR / NIT with file:line evidence, then a verdict."""


# --------------------------------------------------------------------------
# Event-stream parsing (same contract as bench_task_cost)
# --------------------------------------------------------------------------


def iter_events(text: str):
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict):
            yield event


def prompt_side_tokens(usage: dict) -> int:
    context = usage.get("context_tokens")
    if context:
        return int(context)
    return (
        int(usage.get("input_tokens") or 0)
        + int(usage.get("cache_read_tokens") or 0)
        + int(usage.get("cache_write_tokens") or 0)
    )


@dataclass
class RunStats:
    prompt_tokens: int = 0
    fresh_tokens: int = 0
    cache_read_tokens: int = 0
    output_tokens: int = 0
    turns: int = 0
    tool_calls: int = 0
    final_text: str = ""

    @property
    def warm_cache_rate(self) -> float:
        return self.cache_read_tokens / self.prompt_tokens if self.prompt_tokens else 0.0


def parse_stream(text: str) -> RunStats:
    """Fold the exec event stream into totals and capture the review itself.

    The review is the LAST assistant message: a reviewer reports at the end,
    and scoring the whole transcript would count its own diff quotes (which
    necessarily contain the planted numbers) as findings.
    """
    stats = RunStats()
    for event in iter_events(text):
        kind = event.get("type")
        if kind == "tool_execution_start":
            stats.tool_calls += 1
        elif kind == "message_end":
            message = event.get("message") or {}
            if message.get("role") == "assistant":
                text = "".join(
                    block.get("text") or ""
                    for block in message.get("content") or []
                    if isinstance(block, dict)
                )
                if text.strip():
                    stats.final_text = text
        elif kind == "turn_end":
            usage = ((event.get("message") or {}).get("usage")) or {}
            if not usage:
                continue
            prompt = prompt_side_tokens(usage)
            cache_read = int(usage.get("cache_read_tokens") or 0)
            cache_write = int(usage.get("cache_write_tokens") or 0)
            stats.prompt_tokens += prompt
            stats.cache_read_tokens += cache_read
            stats.fresh_tokens += max(prompt - cache_read - cache_write, 0)
            stats.output_tokens += int(usage.get("output_tokens") or 0)
            stats.turns += 1
    return stats


# --------------------------------------------------------------------------
# Pricing (mirrors bench_task_cost.resolve_pricing)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Pricing:
    input_usd: float
    output_usd: float
    cache_read_usd: float
    source: str

    def cost(self, stats: RunStats) -> float:
        return (
            stats.fresh_tokens * self.input_usd
            + stats.cache_read_tokens * self.cache_read_usd
            + stats.output_tokens * self.output_usd
        ) / 1_000_000


def resolve_pricing(hosting: str, model: str, api_key: str | None) -> Pricing:
    try:
        from local_operator.model.discovery import available_models

        models, status = available_models(hosting, api_key=api_key)
        for row in models:
            if row.id == model:
                return Pricing(
                    input_usd=row.input_price,
                    output_usd=row.output_price,
                    cache_read_usd=row.cache_read_price or row.input_price,
                    source=f"discovery:{status}",
                )
    except Exception as exc:  # noqa: BLE001 - pricing is annotation, not data
        return Pricing(0.0, 0.0, 0.0, f"unavailable ({type(exc).__name__}: {exc})")
    return Pricing(0.0, 0.0, 0.0, "unavailable (model not in listing)")


# --------------------------------------------------------------------------
# Scoring — script-side, against ground truth the agent never saw
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Score:
    found: tuple[str, ...]
    missed: tuple[str, ...]
    unrelated_files: tuple[str, ...]
    severity_notes: tuple[str, ...]

    @property
    def coverage(self) -> float:
        return len(self.found) / len(PLANTED)


#: Phrases that mark a claim as ATTACKED or CLEARED rather than made.
#: topic+assertion pairs both fire in "the tests attack the wrong factor in
#: the discount path" — topic (discount) and assertion (wrong) with no defect
#: being asserted by the reviewer at all. Deliberately NARROW: a global
#: "no BLOCKERs" must not void a real finding elsewhere in the same review,
#: so only defect-clearing verbs/phrases count, not severity-level negations.
_ATTACK_WORDS = re.compile(
    r"\b(attacks?|disputes?|refutes?|debunks?|no (bug|issue|defect|problem)|not a (bug|issue|defect|problem)|"
    r"correct(ly)? (handles|manages|applies|computes|validates)|preserved correctly|fine as-is)\b",
    re.IGNORECASE,
)


def score_review(text: str) -> Score:
    found, missed, severity_notes = [], [], []
    for slug, _severity, topic, assertion in PLANTED:
        # Topic and assertion can sit in ADJACENT sentences of the same
        # finding ("The guard was dropped from line_total. Negative inputs
        # now produce negative totals."), so both match against the whole
        # text; the exoneration guard is what keeps praise from scoring.
        hit = (
            re.search(topic, text, re.IGNORECASE)
            and re.search(assertion, text, re.IGNORECASE)
            and not _ATTACK_WORDS.search(text)
        )
        if hit:
            found.append(slug)
        else:
            missed.append(slug)
    # Severity discipline: the dropped negative-quantity guard is a BLOCKER
    # (silent refund path); finding it at MINOR is better than missing it but
    # still a miscount the remediation round pays for. A bare heading match
    # is not enough: "## No BLOCKERs in this diff" and "- BLOCKER: none" are
    # common shapes in real review output (the prompt asks for per-severity
    # classification), so the note requires the guard's own TOPIC vocabulary
    # within 400 chars after a BLOCKER heading.
    if "lost-negative-quantity-guard" in found:
        guard_topic = dict((slug, topic) for slug, _sev, topic, _a in PLANTED)[
            "lost-negative-quantity-guard"
        ]
        # The heading must not itself be a negation ("## No BLOCKERs") — the
        # [^\n]* window between marker and newline would otherwise swallow it.
        raised = re.search(
            r"(?ms)^\s*#{1,4}\s*(?!.*\bno\b)(?!.*\bnone\b)[^\n]*BLOCKER[^\n]*\n.{0,400}(?:" + guard_topic + ")",
            text,
            re.IGNORECASE,
        )
        if not raised:
            severity_notes.append("dropped-guard found but no BLOCKER raised at all")
    # Unrelated: any mention of the decoy files. Word boundary so a substring
    # elsewhere cannot false-positive.
    unrelated = sorted(
        {
            path
            for path in ("legacy/importer.py", "tools/reindex.py")
            if re.search(re.escape(path), text)
        }
    )
    return Score(
        found=tuple(found),
        missed=tuple(missed),
        unrelated_files=tuple(unrelated),
        severity_notes=tuple(severity_notes),
    )


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------


def build_scaffold(root: Path) -> None:
    """Write the base tree, commit it as `main`, then the planted diff on HEAD."""
    for rel, content in BASE_FILES.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def git(*argv: str) -> None:
        subprocess.run(
            ["git", *argv],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            env={
                "PATH": os.environ.get("PATH", ""),
                "HOME": os.environ.get("HOME", ""),
                "GIT_AUTHOR_NAME": "bench",
                "GIT_AUTHOR_EMAIL": "bench@localhost",
                "GIT_COMMITTER_NAME": "bench",
                "GIT_COMMITTER_EMAIL": "bench@localhost",
            },
        )

    git("init", "-q", "-b", "main")
    git("add", "-A")
    git("commit", "-q", "-m", "base: inventory service")
    # The planted diff must land on a WORK branch: committing it onto `main`
    # makes `git diff main..HEAD` — the exact command REVIEW_PROMPT hands the
    # reviewer — the empty diff, and the benchmark would then measure "model
    # works around a broken instruction" instead of "model reviews a diff".
    git("checkout", "-q", "-b", "work")
    for rel, content in DIFF_FILES_CONTENT.items():
        (root / rel).write_text(content, encoding="utf-8")
    git("add", "-A")
    git("commit", "-q", "-m", "feat: percentage discounts on orders")


@dataclass
class ArmResult:
    arm: str
    model: str
    run: int
    stats: RunStats
    score: Score | None
    wall_s: float
    exit_code: int
    cost_usd: float
    effort: str | None = None


def run_arm(
    arm: str,
    model: str,
    run: int,
    root: Path,
    hosting: str,
    pricing: Pricing,
    timeout: float,
    effort: str | None = None,
) -> ArmResult:
    workdir = root / f"{arm}-run{run}"
    workdir.mkdir(parents=True, exist_ok=True)
    build_scaffold(workdir)

    argv = [
        sys.executable,
        "-m",
        "local_operator.cli",
        "--hosting",
        hosting,
        "--model",
        model,
        "--run-in",
        str(workdir),
        "--yolo",
        "exec",
        "--json",
        "--agent",
        "reviewer",
        REVIEW_PROMPT,
    ]
    env = None
    if effort is not None:
        # Reasoning effort has no CLI flag; the one sanctioned pin is a
        # same-selector fallback-chain entry carrying an effort (documented
        # in providers/failover.py: a mapping may repeat the current
        # selector with a different effort — that is a real route). Pointing
        # the run at its own config dir keeps the pin out of the operator's
        # real config; the auth store is symlinked in so credentials resolve.
        import yaml

        run_config = workdir / "lop-config"
        run_config.mkdir(exist_ok=True)
        real_config = Path.home() / ".local-operator"
        for name in ("auth.db", "auth.db-shm", "auth.db-wal", "credentials.env"):
            target = real_config / name
            if target.exists() and not (run_config / name).exists():
                (run_config / name).symlink_to(target)
        (run_config / "config.yml").write_text(
            yaml.safe_dump(
                {
                    "values": {
                        "retry": {
                            "fallbackChains": {
                                f"{hosting}/{model}": [
                                    {"provider": hosting, "model": model, "effort": effort}
                                ]
                            }
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        env = {**os.environ, "LOCAL_OPERATOR_CONFIG_DIR": str(run_config)}
    start = time.monotonic()
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, cwd=REPO, timeout=timeout, env=env
        )
        stdout, exit_code, stderr = proc.stdout, proc.returncode, proc.stderr
    except subprocess.TimeoutExpired as exc:
        stdout = (exc.stdout or b"").decode("utf-8", "replace") if exc.stdout else ""
        stderr = (exc.stderr or b"").decode("utf-8", "replace") if exc.stderr else ""
        exit_code = 124
    wall = time.monotonic() - start

    (root / f"{arm}-run{run}.jsonl").write_text(stdout, encoding="utf-8")
    if stderr.strip():
        (root / f"{arm}-run{run}.stderr").write_text(stderr, encoding="utf-8")

    stats = parse_stream(stdout)
    score = score_review(stats.final_text) if stats.final_text.strip() else None
    return ArmResult(
        arm=arm,
        model=model,
        run=run,
        stats=stats,
        score=score,
        wall_s=wall,
        exit_code=exit_code,
        cost_usd=pricing.cost(stats),
        effort=effort,
    )


def print_report(results: list[ArmResult], pricings: dict[str, Pricing]) -> None:
    print()
    header = (
        f"{'arm':<10} {'run':>3} {'turns':>5} {'tools':>5} {'prompt':>9} {'out':>7} "
        f"{'warm%':>6} {'cost':>9} {'wall':>7} {'cover':>6} {'unrel':>5}  missed"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        s, sc = r.stats, r.score
        cover = f"{len(sc.found)}/{len(PLANTED)}" if sc else "n/a"
        unrel = str(len(sc.unrelated_files)) if sc else "n/a"
        # exit_code distinguishes "model said nothing" from "run died" — a
        # timeout (124) with no review text is a void measurement, not a 0/5.
        missed = ",".join(sc.missed) if sc else f"(no review text; exit {r.exit_code})"
        print(
            f"{r.arm:<10} {r.run:>3} {s.turns:>5} {s.tool_calls:>5} {s.prompt_tokens:>9} "
            f"{s.output_tokens:>7} {s.warm_cache_rate * 100:>5.1f}% ${r.cost_usd:>8.4f} "
            f"{r.wall_s:>6.0f}s {cover:>6} {unrel:>5}  {missed}"
        )
    print()
    print("ground truth (planted defects; found = topic AND assertion):")
    for slug, severity, topic, assertion in PLANTED:
        print(f"  {severity:<8} {slug:<34} topic=/{topic}/ assert=/{assertion}/")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hosting", default=DEFAULT_HOSTING)
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="NAME=MODEL[:EFFORT]",
        help=(
            "One arm of the A/B, e.g. lo=deepseek/deepseek-v4-flash-0731 or "
            "calm=openai/gpt-5.2:low (repeatable). The optional :EFFORT pins the "
            "reasoning effort through a same-selector fallback-chain entry, the "
            "one config path that carries effort for a named model."
        ),
    )
    parser.add_argument("--runs", type=int, default=2, help="Runs per arm (default 2).")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--out", type=Path, help="Scratch root. Default: a fresh temp dir.")
    parser.add_argument("--keep", action="store_true", help="Keep the scratch directory.")
    parser.add_argument("--json-out", type=Path, help="Also write results as JSON here.")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if args.hosting == DEFAULT_HOSTING and not api_key:
        print("OPENROUTER_API_KEY is not set; this benchmark only runs live.", file=sys.stderr)
        return 2

    arms: list[tuple[str, str, str | None]] = []
    for spec in args.arm:
        name, sep, rest = spec.partition("=")
        model, colon, effort = rest.partition(":")
        if not sep or not name or not model:
            print(f"bad --arm {spec!r}: expected NAME=MODEL[:EFFORT]", file=sys.stderr)
            return 2
        arms.append((name, model, effort or None))

    root = args.out or Path(tempfile.mkdtemp(prefix="lo-bench-review-"))
    root.mkdir(parents=True, exist_ok=True)
    keep = args.keep or args.out is not None

    print(f"scratch: {root}")
    results: list[ArmResult] = []
    pricings: dict[str, Pricing] = {}
    try:
        for name, model, effort in arms:
            pricings[name] = resolve_pricing(args.hosting, model, api_key)
            for run in range(1, args.runs + 1):
                print(
                    f"running {name} ({model}{'' if effort is None else f' effort={effort}'}) "
                    f"run {run}/{args.runs} ...",
                    flush=True,
                )
                results.append(
                    run_arm(
                        name,
                        model,
                        run,
                        root=root,
                        hosting=args.hosting,
                        pricing=pricings[name],
                        timeout=args.timeout,
                        effort=effort,
                    )
                )
        print_report(results, pricings)
        if args.json_out:
            args.json_out.write_text(
                json.dumps(
                    [
                        {
                            "arm": r.arm,
                            "model": r.model,
                            "effort": r.effort,
                            "run": r.run,
                            "wall_s": r.wall_s,
                            "exit_code": r.exit_code,
                            "cost_usd": r.cost_usd,
                            "stats": {
                                "turns": r.stats.turns,
                                "tool_calls": r.stats.tool_calls,
                                "prompt_tokens": r.stats.prompt_tokens,
                                "fresh_tokens": r.stats.fresh_tokens,
                                "cache_read_tokens": r.stats.cache_read_tokens,
                                "output_tokens": r.stats.output_tokens,
                            },
                            "score": (
                                {
                                    "found": list(r.score.found),
                                    "missed": list(r.score.missed),
                                    "unrelated_files": list(r.score.unrelated_files),
                                    "severity_notes": list(r.score.severity_notes),
                                }
                                if r.score
                                else None
                            ),
                        }
                        for r in results
                    ],
                    indent=2,
                ),
                encoding="utf-8",
            )
    finally:
        if not keep:
            import shutil

            shutil.rmtree(root, ignore_errors=True)
        else:
            print(f"kept: {root}")
    # A non-zero exit flags an inconclusive benchmark, not a failing one:
    # every arm that produced no review text voids its own measurement.
    return 1 if any(r.score is None for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
