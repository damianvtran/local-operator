#!/usr/bin/env python3
"""Per-task cost benchmark on genuinely complicated work.

``bench_cache_rate.py`` answers "does the request prefix stay byte-stable"
and ``bench_context_budget.py`` answers "how big is the start context". Both
measure the harness at rest. This script measures the harness doing a job:
it scaffolds a real codebase into a scratch directory, hands the agent a task
against it through ``local_operator.cli exec --json``, and then *verifies the
outcome itself*.

Why the verification is script-side and not agent-side: an agent that says
"all tests pass" is reporting an intention, not a result. Every fixture here
restores its contract test from a pristine copy the agent cannot reach and
adds a hold-out test the agent never saw, then runs pytest (or the produced
CLI) from this process. A task the script cannot decide is reported as
``unverified`` rather than quietly counted as a pass.

Reported per task: turns, tool calls, prompt-side tokens split into fresh vs
cache-read, output tokens, warm cache rate, cost in USD at live provider
prices, wall time, and the pass/fail of the objective check. The per-turn
cache trajectory is reported too, because the question the prompt layout
exists to answer is whether the cache rate *holds* as tool results pile up or
decays over a long trajectory -- a single averaged number hides that.

Run:
    OPENROUTER_API_KEY=... .venv/bin/python scripts/bench_task_cost.py
    .venv/bin/python scripts/bench_task_cost.py --task refactor --keep
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DEFAULT_HOSTING = "openrouter"
DEFAULT_MODEL = "deepseek/deepseek-v4-flash-0731"
#: Per-task ceiling. Long enough for a real multi-turn build, short enough
#: that one stuck task cannot hold the whole benchmark hostage.
DEFAULT_TIMEOUT_S = 900


# --------------------------------------------------------------------------
# Event-stream parsing
# --------------------------------------------------------------------------


def iter_events(text: str) -> Iterator[dict[str, Any]]:
    """Yield the JSON objects on ``exec --json``'s stdout.

    Undecodable lines are skipped rather than fatal. stdout is contractually
    the event stream (the CLI routes its own notices to stderr), but a
    third-party library printing one stray line must not void a paid-for
    measurement -- the tokens were already spent.
    """
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


def prompt_side_tokens(usage: dict[str, Any]) -> int:
    """Total tokens the provider read to produce this turn.

    The two provider families disagree on what ``input_tokens`` means:
    OpenAI-compatible gateways report ``prompt_tokens`` *including* the cached
    prefix, Anthropic reports it *excluding* it. ``context_tokens`` is the
    field the provider adapters normalise to mean "the whole prompt side", so
    it is the only denominator that gives the same cache rate on both. The sum
    is the fallback for a provider that omitted it.
    """
    context = usage.get("context_tokens")
    if context:
        return int(context)
    return (
        int(usage.get("input_tokens") or 0)
        + int(usage.get("cache_read_tokens") or 0)
        + int(usage.get("cache_write_tokens") or 0)
    )


@dataclass
class TurnUsage:
    """One assistant turn's token accounting, provider-normalised."""

    prompt: int
    fresh: int
    cache_read: int
    cache_write: int
    output: int

    @property
    def cache_rate(self) -> float:
        return self.cache_read / self.prompt if self.prompt else 0.0


@dataclass
class RunStats:
    """Everything the event stream says about one agent run."""

    turns: list[TurnUsage] = field(default_factory=list)
    tools: Counter[str] = field(default_factory=Counter)
    aborted: bool = False
    error: str | None = None
    compactions: int = 0
    retries: int = 0
    # Characters, not tokens, and deliberately so: the tokeniser is optional
    # in this install, and the question these three answer is a ratio, which
    # a byte count answers just as well as a token count would.
    tool_result_chars: int = 0
    tool_arg_chars: int = 0
    assistant_text_chars: int = 0

    @property
    def transcript_chars(self) -> int:
        """Everything the agent appended to its own context after turn 1."""
        return self.tool_result_chars + self.tool_arg_chars + self.assistant_text_chars

    @property
    def tool_traffic_share(self) -> float:
        """Fraction of the appended context that is tool traffic, not prose.

        Tool ARGUMENTS count as tool traffic: for ``write`` and ``edit`` the
        argument *is* the file body, and it is billed twice -- once as output
        when the model emits it, then on every later turn as prompt.
        """
        total = self.transcript_chars
        return (self.tool_result_chars + self.tool_arg_chars) / total if total else 0.0

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    @property
    def tool_calls(self) -> int:
        return sum(self.tools.values())

    @property
    def prompt_tokens(self) -> int:
        return sum(t.prompt for t in self.turns)

    @property
    def fresh_tokens(self) -> int:
        return sum(t.fresh for t in self.turns)

    @property
    def cache_read_tokens(self) -> int:
        return sum(t.cache_read for t in self.turns)

    @property
    def cache_write_tokens(self) -> int:
        return sum(t.cache_write for t in self.turns)

    @property
    def output_tokens(self) -> int:
        return sum(t.output for t in self.turns)

    @property
    def cache_rate(self) -> float:
        return self.cache_read_tokens / self.prompt_tokens if self.prompt_tokens else 0.0

    @property
    def warm_cache_rate(self) -> float:
        """Cache rate over every turn but the first.

        Turn 1 is the only turn that can legitimately miss: nothing in the
        conversation has been sent yet. Including it drags the average down by
        an amount that depends purely on how many turns the task happened to
        take, which makes short and long tasks incomparable. The warm rate is
        the number that says whether the layout keeps working.
        """
        return _rate(self.turns[1:])

    def window_rate(self, start: int, end: int) -> float:
        return _rate(self.turns[start:end])


def _rate(turns: Sequence[TurnUsage]) -> float:
    prompt = sum(t.prompt for t in turns)
    return sum(t.cache_read for t in turns) / prompt if prompt else 0.0


def parse_stream(text: str) -> RunStats:
    """Fold the event stream into per-turn usage and a tool histogram.

    Usage is taken from ``turn_end`` rather than the ``agent_end`` summary:
    ``agent_end`` carries the same numbers but flattened across the whole run,
    and the per-turn sequence is exactly what the cache-decay question needs.
    """
    stats = RunStats()
    for event in iter_events(text):
        kind = event.get("type")
        if kind == "tool_execution_start":
            stats.tools[str(event.get("tool_name") or "?")] += 1
            stats.tool_arg_chars += len(json.dumps(event.get("args") or {}))
        elif kind == "tool_execution_end":
            stats.tool_result_chars += len(json.dumps((event.get("result") or {}).get("content")))
        elif kind == "message_end":
            message = event.get("message") or {}
            if message.get("role") == "assistant":
                stats.assistant_text_chars += sum(
                    len(block.get("text") or "")
                    for block in message.get("content") or []
                    if isinstance(block, dict)
                )
        elif kind == "compaction_start":
            stats.compactions += 1
        elif kind == "retry_start":
            stats.retries += 1
        elif kind == "turn_end":
            usage = ((event.get("message") or {}).get("usage")) or {}
            if not usage:
                # A turn can end without usage when the provider dropped the
                # trailing usage chunk. Counting it as a zero-token turn would
                # understate the cache rate's denominator, so skip it.
                continue
            prompt = prompt_side_tokens(usage)
            cache_read = int(usage.get("cache_read_tokens") or 0)
            cache_write = int(usage.get("cache_write_tokens") or 0)
            stats.turns.append(
                TurnUsage(
                    prompt=prompt,
                    # Clamped: a provider that over-reports cached tokens
                    # relative to the prompt total would otherwise produce a
                    # negative billable count and a negative cost.
                    fresh=max(prompt - cache_read - cache_write, 0),
                    cache_read=cache_read,
                    cache_write=cache_write,
                    output=int(usage.get("output_tokens") or 0),
                )
            )
        elif kind == "agent_end":
            stats.aborted = bool(event.get("aborted"))
            stats.error = event.get("error")
    return stats


# --------------------------------------------------------------------------
# Pricing
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Pricing:
    """Per-million-token prices, as the provider quotes them today."""

    input_usd: float
    output_usd: float
    cache_read_usd: float
    source: str

    def cost(self, stats: RunStats) -> float:
        return (
            stats.fresh_tokens * self.input_usd
            + stats.cache_read_tokens * self.cache_read_usd
            # Cache writes are billed at the plain input rate unless the
            # provider quotes a separate one; none of the listings this script
            # reads do, and guessing a premium would inflate the total.
            + stats.cache_write_tokens * self.input_usd
            + stats.output_tokens * self.output_usd
        ) / 1_000_000

    def prompt_cost(self, stats: RunStats) -> float:
        """Everything billed on the way IN, cached and uncached alike."""
        return (
            stats.fresh_tokens * self.input_usd
            + stats.cache_read_tokens * self.cache_read_usd
            + stats.cache_write_tokens * self.input_usd
        ) / 1_000_000

    def output_cost(self, stats: RunStats) -> float:
        return stats.output_tokens * self.output_usd / 1_000_000

    def cost_uncached(self, stats: RunStats) -> float:
        """What the same trajectory would cost with caching switched off.

        This is the honest way to price the cache: the saving is not
        ``cache_read * input_price``, it is the difference between two whole
        runs, and quoting it makes a low cache rate visibly expensive.
        """
        return (
            stats.prompt_tokens * self.input_usd + stats.output_tokens * self.output_usd
        ) / 1_000_000


def resolve_pricing(hosting: str, model: str, api_key: str | None) -> Pricing:
    """Live prices for ``model``, falling back to zeroes rather than failing.

    A pricing lookup failure must not void the token measurement, which is the
    part that took real time and money to collect. ``source`` records which
    path answered so a $0.0000 row is legible as "unpriced" instead of "free".
    """
    try:
        from local_operator.model.discovery import available_models

        models, status = available_models(hosting, api_key=api_key)
        for row in models:
            if row.id == model:
                return Pricing(
                    input_usd=row.input_price,
                    output_usd=row.output_price,
                    # A model with no quoted cache-read leg still gets charged
                    # for those tokens; falling back to the input price keeps
                    # the total right instead of silently zeroing them.
                    cache_read_usd=row.cache_read_price or row.input_price,
                    source=f"discovery:{status}",
                )
    except Exception as exc:  # noqa: BLE001 - pricing is annotation, not data
        return Pricing(0.0, 0.0, 0.0, f"unavailable ({type(exc).__name__}: {exc})")
    return Pricing(0.0, 0.0, 0.0, "unavailable (model not in listing)")


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Check:
    """The script's own verdict on a task.

    ``ok is None`` means the script could not decide -- reported as
    ``unverified``, never folded into the pass count.
    """

    ok: bool | None
    detail: str


@dataclass(frozen=True)
class Fixture:
    slug: str
    title: str
    why: str
    prompt: str
    #: Files written before the run. Declarative so the whole scaffold is
    #: reproducible from a clean checkout with no fixture data on disk.
    scaffold: dict[str, str]
    verify: Callable[[Path], Check]


def _write_tree(root: Path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _run_pytest(root: Path, targets: Sequence[str]) -> tuple[bool, str]:
    """Run pytest in ``root`` from *this* process, not the agent's shell."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header", *targets],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=300,
    )
    tail = (proc.stdout or proc.stderr).strip().splitlines()
    return proc.returncode == 0, tail[-1] if tail else f"exit {proc.returncode}"


# The scaffolds below deliberately contain no "BUG"/"FIXME" markers: a comment
# pointing at the defect turns a debugging task into a reading task.

SKU_REGEX = r"[A-Z]{3}-\d{4}"

_DUPLICATED_SKU_BLOCK = f"""import re

SKU_PATTERN = re.compile(r"^{SKU_REGEX}$")


def _normalize_sku(raw):
    if not isinstance(raw, str):
        raise TypeError("sku must be a string")
    candidate = raw.strip().upper().replace("_", "-")
    if not SKU_PATTERN.match(candidate):
        raise ValueError("malformed sku: {{!r}}".format(raw))
    return candidate
"""

REFACTOR_FILES = {
    "conftest.py": "",
    "inventory/__init__.py": "",
    "inventory/products.py": _DUPLICATED_SKU_BLOCK
    + '''

def register_product(sku, name):
    """Return a product record with a normalized sku."""
    if not name or not name.strip():
        raise ValueError("name is required")
    return {"sku": _normalize_sku(sku), "name": name.strip()}
''',
    "inventory/orders.py": _DUPLICATED_SKU_BLOCK
    + '''

def place_order(sku, quantity):
    """Return an order record with a normalized sku."""
    if not isinstance(quantity, int) or quantity < 1:
        raise ValueError("quantity must be a positive integer")
    return {"sku": _normalize_sku(sku), "quantity": quantity}
''',
    "inventory/customers.py": _DUPLICATED_SKU_BLOCK
    + '''

def assign_favourite(customer_id, sku):
    """Attach a favourite sku to a customer."""
    if not isinstance(customer_id, int):
        raise TypeError("customer_id must be an int")
    return {"customer_id": customer_id, "favourite": _normalize_sku(sku)}
''',
    "tests/test_inventory.py": """import pytest

from inventory.customers import assign_favourite
from inventory.orders import place_order
from inventory.products import register_product


def test_product_normalizes_sku():
    assert register_product(" abc-1234 ", "Widget")["sku"] == "ABC-1234"


def test_product_rejects_blank_name():
    with pytest.raises(ValueError):
        register_product("ABC-1234", "  ")


def test_order_normalizes_underscore_form():
    assert place_order("abc_1234", 3)["sku"] == "ABC-1234"


def test_order_rejects_zero_quantity():
    with pytest.raises(ValueError):
        place_order("ABC-1234", 0)


def test_customer_normalizes_sku():
    assert assign_favourite(7, "xyz-9999")["favourite"] == "XYZ-9999"


def test_malformed_sku_raises_value_error():
    with pytest.raises(ValueError):
        register_product("AB-1234", "Widget")


def test_non_string_sku_raises_type_error():
    with pytest.raises(TypeError):
        place_order(1234, 1)
""",
}

REFACTOR_HOLDOUT = '''"""Hold-out checks written after the run; the agent never saw these."""

import pytest

from inventory.customers import assign_favourite
from inventory.orders import place_order
from inventory.products import register_product


@pytest.mark.parametrize("fn", [
    lambda s: register_product(s, "n")["sku"],
    lambda s: place_order(s, 1)["sku"],
    lambda s: assign_favourite(1, s)["favourite"],
])
@pytest.mark.parametrize("raw,expected", [
    ("qrs_0001", "QRS-0001"),
    ("  tuv-4321\\t", "TUV-4321"),
    ("MNO-0000", "MNO-0000"),
])
def test_all_three_modules_normalize_identically(fn, raw, expected):
    assert fn(raw) == expected


@pytest.mark.parametrize("fn", [
    lambda s: register_product(s, "n"),
    lambda s: place_order(s, 1),
    lambda s: assign_favourite(1, s),
])
@pytest.mark.parametrize("bad", ["ABCD-1234", "AB-1234", "ABC-123", "ABC-12345", ""])
def test_all_three_modules_reject_identically(fn, bad):
    with pytest.raises(ValueError):
        fn(bad)
'''


def verify_refactor(root: Path) -> Check:
    """Contract test restored, hold-out added, and the duplication actually gone.

    Three separate failure modes have to be excluded, and passing tests alone
    excludes none of them: the agent editing the suite to match a regression,
    the agent special-casing the exact inputs the suite uses, and the agent
    declaring victory without removing the duplication the task was about.
    """
    _write_tree(root, {"tests/test_inventory.py": REFACTOR_FILES["tests/test_inventory.py"]})
    _write_tree(root, {"tests/test_refactor_holdout.py": REFACTOR_HOLDOUT})
    passed, summary = _run_pytest(
        root, ["tests/test_inventory.py", "tests/test_refactor_holdout.py"]
    )

    # Two independent duplication signals, both thresholded at "at most one".
    # `== 1` would fail a legitimate refactor that rewrote the regex or the
    # string ops on its way into the shared module; the failure this check
    # exists to catch is duplication SURVIVING, which is always >= 2. The
    # hold-out suite above is what proves the logic still exists at all.
    sources = [
        p
        for p in sorted(root.rglob("*.py"))
        if "tests" not in p.relative_to(root).parts and p.name != "conftest.py"
    ]
    texts = {p: p.read_text(encoding="utf-8", errors="replace") for p in sources}
    pattern_hits = [p for p, t in texts.items() if SKU_REGEX in t]
    # The three characteristic operations of the normalisation body. A module
    # that no longer performs them is no longer carrying a copy of it.
    body_hits = [
        p for p, t in texts.items() if all(op in t for op in (".strip()", ".upper()", ".replace("))
    ]
    deduped = len(pattern_hits) <= 1 and len(body_hits) <= 1

    def _names(paths: list[Path]) -> str:
        return ", ".join(p.relative_to(root).as_posix() for p in paths) or "none"

    return Check(
        ok=passed and deduped,
        detail=(
            f"pytest={'pass' if passed else 'FAIL'} ({summary}); "
            f"sku pattern in {len(pattern_hits)} file(s) [{_names(pattern_hits)}]; "
            f"normalisation body in {len(body_hits)} file(s) [{_names(body_hits)}]; "
            "both must be <=1"
        ),
    )


DEBUG_FILES = {
    "conftest.py": "",
    "pricing/__init__.py": "",
    "pricing/tiers.py": '''"""Volume quoting across price tiers.

Prices are in cents per unit and apply to the units that fall inside each
band, cheapest band last.
"""

TIERS = {
    "standard": [(10, 500), (50, 450), (None, 400)],
    "partner": [(10, 400), (50, 340), (None, 300)],
}

_QUOTE_CACHE = {}


def _bands(tier):
    if tier not in TIERS:
        raise KeyError("unknown tier: {!r}".format(tier))
    return TIERS[tier]


def quote(units, tier="standard"):
    """Total price in cents for ``units`` at ``tier``."""
    if not isinstance(units, int) or units < 0:
        raise ValueError("units must be a non-negative integer")
    cache_key = units
    if cache_key in _QUOTE_CACHE:
        return _QUOTE_CACHE[cache_key]

    total = 0
    consumed = 0
    for limit, price in _bands(tier):
        available = units - consumed if limit is None else min(limit, units) - consumed
        if available <= 0:
            continue
        total += available * price
        consumed += available
    _QUOTE_CACHE[cache_key] = total
    return total


def clear_cache():
    _QUOTE_CACHE.clear()
''',
    "tests/test_pricing.py": """from pricing.tiers import quote


def test_standard_spans_two_bands():
    # 10 units @ 500 + 10 units @ 450
    assert quote(20, "standard") == 9500


def test_partner_is_cheaper_than_standard_for_the_same_volume():
    assert quote(20, "partner") == 7400
    assert quote(20, "partner") < quote(20, "standard")
""",
}

DEBUG_HOLDOUT = '''"""Hold-out checks written after the run; the agent never saw these."""

import pytest

from pricing.tiers import quote


@pytest.mark.parametrize("units,tier,expected", [
    (0, "standard", 0),
    (5, "standard", 2500),
    (10, "standard", 5000),
    (60, "standard", 27000),
    (5, "partner", 2000),
    (60, "partner", 20600),
    (100, "partner", 32600),
])
def test_quotes_are_correct_in_any_call_order(units, tier, expected):
    assert quote(units, tier) == expected


def test_repeated_calls_are_stable_and_tier_sensitive():
    # Deliberately interleaved: a cache keyed on units alone returns the
    # first tier's answer for the second one.
    first = quote(35, "partner")
    other = quote(35, "standard")
    assert first == quote(35, "partner")
    assert other == quote(35, "standard")
    assert first < other


def test_negative_units_rejected():
    with pytest.raises(ValueError):
        quote(-1, "standard")
'''


def verify_debug(root: Path) -> Check:
    _write_tree(root, {"tests/test_pricing.py": DEBUG_FILES["tests/test_pricing.py"]})
    _write_tree(root, {"tests/test_pricing_holdout.py": DEBUG_HOLDOUT})
    passed, summary = _run_pytest(root, ["tests/test_pricing.py", "tests/test_pricing_holdout.py"])
    return Check(ok=passed, detail=f"pytest={'pass' if passed else 'FAIL'} ({summary})")


# The build-from-scratch fixture ships no scaffold at all beyond a README of
# the spec: the point is to measure what producing working code from nothing
# costs, and a starting skeleton would pay part of that bill up front.
BUILD_SPEC = """\
Write `iniround.py` at the root of this directory: a CLI that parses a small
INI dialect and round-trips it. Use only the Python standard library, and do
not use `configparser` (its dialect differs from the one specified here).

Subcommands:

  python iniround.py parse FILE
      Print a JSON object to stdout mapping section name -> object of
      key -> value (all values are strings). Keys appearing before any
      section header go under the empty-string section "". The empty-string
      section is present only if such keys exist. Preserve file order for
      both sections and keys.

  python iniround.py emit FILE
      Read the JSON produced by `parse` from FILE and print INI text to
      stdout. The output must satisfy: parse(emit(parse(x))) == parse(x).

Dialect:
  - A section header is a line whose stripped form is `[name]`; `name` is the
    header contents stripped of surrounding whitespace.
  - A comment is a line whose stripped form starts with `;` or `#`. Ignore it.
  - A blank (whitespace-only) line is ignored.
  - Any other line is `key = value`, split on the FIRST `=` only, so a value
    may itself contain `=`. Strip whitespace around both key and value. An
    empty value is legal and parses to "".
  - Any other non-blank, non-comment line (no `=`, not a section header) is a
    parse error: write a message to stderr, print nothing to stdout, and exit
    with status 2.

Exit 0 on success. Verify your own work before finishing.
"""

BUILD_INPUT = """\
; leading comment
preamble = yes
  spaced   =   value with spaces

[server]
# a hash comment
host=localhost
port = 8080
motd = welcome = home

[empty]

[flags]
debug =
verbose = 1
"""

BUILD_EXPECTED = {
    "": {"preamble": "yes", "spaced": "value with spaces"},
    "server": {"host": "localhost", "port": "8080", "motd": "welcome = home"},
    "empty": {},
    "flags": {"debug": "", "verbose": "1"},
}

BUILD_MALFORMED = "[server]\nhost localhost\n"


def verify_build(root: Path) -> Check:
    """Exercise the produced CLI with inputs it was never shown."""
    script = root / "iniround.py"
    if not script.is_file():
        return Check(ok=False, detail="iniround.py was not created")

    def run(*argv: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(script), *argv],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=60,
        )

    probe = root / "_probe.ini"
    probe.write_text(BUILD_INPUT, encoding="utf-8")
    failures: list[str] = []

    parsed = run("parse", str(probe))
    if parsed.returncode != 0:
        return Check(
            ok=False, detail=f"parse exited {parsed.returncode}: {parsed.stderr.strip()[:160]}"
        )
    try:
        first = json.loads(parsed.stdout)
    except ValueError as exc:
        return Check(ok=False, detail=f"parse stdout is not JSON ({exc})")
    if first != BUILD_EXPECTED:
        failures.append(f"parse mismatch: got {json.dumps(first)[:200]}")

    # Round trip through the agent's own emitter, then back through its own
    # parser: this is the property the spec actually promised, and it fails
    # loudly for an emitter that drops empty sections or mangles a value
    # containing '='.
    probe_json = root / "_probe.json"
    probe_json.write_text(json.dumps(first), encoding="utf-8")
    emitted = run("emit", str(probe_json))
    if emitted.returncode != 0:
        failures.append(f"emit exited {emitted.returncode}: {emitted.stderr.strip()[:120]}")
    else:
        round_trip_ini = root / "_probe_round.ini"
        round_trip_ini.write_text(emitted.stdout, encoding="utf-8")
        again = run("parse", str(round_trip_ini))
        if again.returncode != 0:
            failures.append(f"re-parse of emitted text exited {again.returncode}")
        else:
            try:
                second = json.loads(again.stdout)
            except ValueError:
                second = None
            if second != first:
                failures.append("round trip is not stable")

    bad = root / "_bad.ini"
    bad.write_text(BUILD_MALFORMED, encoding="utf-8")
    err = run("parse", str(bad))
    if err.returncode != 2:
        failures.append(f"malformed input exited {err.returncode}, expected 2")
    if err.stdout.strip():
        failures.append("malformed input still wrote to stdout")

    return Check(
        ok=not failures, detail="; ".join(failures) if failures else "all CLI probes passed"
    )


LONGHAUL_SPEC = """\
Write `sheetcalc.py` at the root of this directory: a spreadsheet formula
evaluator. Standard library only.

  python sheetcalc.py FILE

FILE is a text file, one `CELL=CONTENT` assignment per line, e.g. `A1=3`.
Print to stdout a JSON object mapping every assigned cell name to its value.

Rules:
  - Content that is a number is that number (int stays int, e.g. 3 not 3.0).
  - Content starting with `=` is a formula over: cell references (`A1`),
    integer/decimal literals, `+ - * /`, parentheses, and the functions
    SUM, AVG, MIN, MAX which each take one range argument `A1:A3`
    (ranges are rectangular and may span rows and columns).
  - Normal arithmetic precedence: `*` and `/` bind tighter than `+` and `-`.
  - An empty or unassigned cell referenced by a formula counts as 0.
  - Division by zero makes that cell the string "#DIV/0!".
  - A reference cycle makes every cell in the cycle the string "#CYCLE!".
  - Anything else non-numeric is a literal string value.
  - Results that are whole numbers must print as integers, not 4.0.

Exit 0 on success. Verify your own work before finishing.
"""

LONGHAUL_INPUT = """\
A1=3
A2=4
A3=5
B1==A1+A2*2
B2==(A1+A2)*2
B3==SUM(A1:A3)
C1==AVG(A1:A3)
C2==MAX(A1:B2)
C3==MIN(A1:A3)
D1==A1/0
D2==B3-A3
D3==Z9+A1
E1==E2
E2==E1
F1=hello
"""

LONGHAUL_EXPECTED = {
    "A1": 3,
    "A2": 4,
    "A3": 5,
    "B1": 11,
    "B2": 14,
    "B3": 12,
    "C1": 4,
    "C2": 14,
    "C3": 3,
    "D1": "#DIV/0!",
    "D2": 7,
    "D3": 3,
    "E1": "#CYCLE!",
    "E2": "#CYCLE!",
    "F1": "hello",
}


def verify_longhaul(root: Path) -> Check:
    script = root / "sheetcalc.py"
    if not script.is_file():
        return Check(ok=False, detail="sheetcalc.py was not created")
    probe = root / "_sheet.txt"
    probe.write_text(LONGHAUL_INPUT, encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(script), str(probe)],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0:
        return Check(ok=False, detail=f"exited {proc.returncode}: {proc.stderr.strip()[:160]}")
    try:
        got = json.loads(proc.stdout)
    except ValueError as exc:
        return Check(ok=False, detail=f"stdout is not JSON ({exc})")
    wrong = {
        cell: (got.get(cell, "<missing>"), expected)
        for cell, expected in LONGHAUL_EXPECTED.items()
        if got.get(cell, "<missing>") != expected
    }
    if wrong:
        shown = ", ".join(f"{c}: got {g!r} want {w!r}" for c, (g, w) in sorted(wrong.items())[:6])
        return Check(
            ok=False, detail=f"{len(wrong)}/{len(LONGHAUL_EXPECTED)} cells wrong -- {shown}"
        )
    return Check(ok=True, detail=f"all {len(LONGHAUL_EXPECTED)} cells correct")


FIXTURES: list[Fixture] = [
    Fixture(
        slug="refactor",
        title="Multi-file refactor behind a fixed test suite",
        why=(
            "The most common shape of real maintenance work: identical logic "
            "copy-pasted across modules, a test suite that must not move, and "
            "no new behaviour to invent. It forces the agent to read several "
            "files before writing any."
        ),
        prompt=(
            "This package duplicates its SKU normalisation and validation logic "
            "verbatim in inventory/products.py, inventory/orders.py and "
            "inventory/customers.py. Refactor so that logic is defined exactly "
            "once and imported by all three modules; the duplicate definitions "
            "must be gone, not merely unused. Public behaviour must not change. "
            "tests/test_inventory.py is the contract: keep it passing and do not "
            "modify it. Run the suite with: "
            f"{sys.executable} -m pytest -q"
        ),
        scaffold=REFACTOR_FILES,
        verify=verify_refactor,
    ),
    Fixture(
        slug="debug",
        title="Diagnose and fix a seeded defect",
        why=(
            "A memoisation key that omits one of its inputs: wrong only for "
            "certain call sequences, invisible in a single-case test, and not "
            "findable by reading the failing assertion alone. This is the "
            "debugging loop -- reproduce, localise, fix -- not a typo hunt."
        ),
        prompt=(
            "tests/test_pricing.py fails. Find the defect in the pricing package "
            "and fix it. Do not modify anything under tests/. Run the suite with: "
            f"{sys.executable} -m pytest -q"
        ),
        scaffold=DEBUG_FILES,
        verify=verify_debug,
    ),
    Fixture(
        slug="build",
        title="Build a CLI to a written spec, from nothing",
        why=(
            "Greenfield work against a precise interface contract, including "
            "the parts implementations usually get wrong: an empty section, a "
            "value containing the delimiter, and a specified non-zero exit "
            "code. Round-tripping is a property the script can check, so the "
            "result does not depend on the agent's own tests."
        ),
        prompt=BUILD_SPEC,
        scaffold={"SPEC.md": BUILD_SPEC},
        verify=verify_build,
    ),
    Fixture(
        slug="longhaul",
        title="Long-horizon build: a formula evaluator",
        why=(
            "Deliberately the longest trajectory in the set -- a tokeniser, a "
            "precedence-aware parser, range functions, cycle detection and "
            "error sentinels. It exists to answer whether the warm cache rate "
            "holds once tool results have accumulated over many turns, which a "
            "short task cannot show."
        ),
        prompt=LONGHAUL_SPEC,
        scaffold={"SPEC.md": LONGHAUL_SPEC},
        verify=verify_longhaul,
    ),
]


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


@dataclass
class TaskResult:
    fixture: Fixture
    stats: RunStats
    check: Check
    wall_s: float
    exit_code: int
    cost_usd: float
    cost_uncached_usd: float
    workdir: Path


def run_fixture(
    fixture: Fixture,
    *,
    root: Path,
    hosting: str,
    model: str,
    pricing: Pricing,
    timeout: float,
) -> TaskResult:
    workdir = root / fixture.slug
    workdir.mkdir(parents=True, exist_ok=True)
    _write_tree(workdir, fixture.scaffold)

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
        fixture.prompt,
    ]
    start = time.monotonic()
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            cwd=REPO,
            timeout=timeout,
        )
        stdout, exit_code = proc.stdout, proc.returncode
        stderr = proc.stderr
    except subprocess.TimeoutExpired as exc:
        # A timed-out run still burned tokens; keep whatever the stream got so
        # the cost of the failure is reported instead of lost.
        stdout = (exc.stdout or b"").decode("utf-8", "replace") if exc.stdout else ""
        stderr = (exc.stderr or b"").decode("utf-8", "replace") if exc.stderr else ""
        exit_code = 124
    wall = time.monotonic() - start

    (root / f"{fixture.slug}.jsonl").write_text(stdout, encoding="utf-8")
    if stderr.strip():
        (root / f"{fixture.slug}.stderr").write_text(stderr, encoding="utf-8")

    stats = parse_stream(stdout)
    if not stats.turns:
        check = Check(
            ok=None,
            detail=f"no turns in the event stream (exit {exit_code}); see {fixture.slug}.stderr",
        )
    else:
        try:
            check = fixture.verify(workdir)
        except Exception as exc:  # noqa: BLE001 - an undecidable check is a result
            check = Check(ok=None, detail=f"verifier raised {type(exc).__name__}: {exc}")
    return TaskResult(
        fixture=fixture,
        stats=stats,
        check=check,
        wall_s=wall,
        exit_code=exit_code,
        cost_usd=pricing.cost(stats),
        cost_uncached_usd=pricing.cost_uncached(stats),
        workdir=workdir,
    )


def _verdict(check: Check) -> str:
    if check.ok is None:
        return "unverified"
    return "PASS" if check.ok else "FAIL"


def print_report(results: list[TaskResult], pricing: Pricing, model: str) -> None:
    print()
    print(
        f"model: {model}   pricing: {pricing.source} "
        f"(in ${pricing.input_usd:.4f}/M, out ${pricing.output_usd:.4f}/M, "
        f"cache-read ${pricing.cache_read_usd:.4f}/M)"
    )
    print()
    header = (
        f"{'task':<10} {'turns':>5} {'tools':>5} {'prompt':>9} {'fresh':>8} "
        f"{'cached':>9} {'out':>7} {'warm$':>7} {'cost':>9} {'wall':>7}  outcome"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        s = r.stats
        print(
            f"{r.fixture.slug:<10} {s.turn_count:>5} {s.tool_calls:>5} "
            f"{s.prompt_tokens:>9,} {s.fresh_tokens:>8,} {s.cache_read_tokens:>9,} "
            f"{s.output_tokens:>7,} {s.warm_cache_rate:>6.1%} "
            f"${r.cost_usd:>8.4f} {r.wall_s:>6.1f}s  {_verdict(r.check)}"
        )
    total_cost = sum(r.cost_usd for r in results)
    total_uncached = sum(r.cost_uncached_usd for r in results)
    # The column is the WARM rate, so the total has to drop every task's
    # first turn too. Pooling all turns instead silently re-admits the cold
    # ones and reports a number the header does not describe.
    warm_turns = [t for r in results for t in r.stats.turns[1:]]
    print("-" * len(header))
    print(
        f"{'TOTAL':<10} {sum(r.stats.turn_count for r in results):>5} "
        f"{sum(r.stats.tool_calls for r in results):>5} "
        f"{sum(r.stats.prompt_tokens for r in results):>9,} "
        f"{sum(r.stats.fresh_tokens for r in results):>8,} "
        f"{sum(r.stats.cache_read_tokens for r in results):>9,} "
        f"{sum(r.stats.output_tokens for r in results):>7,} "
        f"{_rate(warm_turns):>6.1%} ${total_cost:>8.4f} "
        f"{sum(r.wall_s for r in results):>6.1f}s"
    )
    if total_uncached:
        print(
            f"\nsame trajectories with caching off: ${total_uncached:.4f} "
            f"({1 - total_cost / total_uncached:.1%} saved)"
        )

    print("\nverification (run by this script, not by the agent):")
    for r in results:
        print(f"  {r.fixture.slug:<10} {_verdict(r.check):<10} {r.check.detail}")

    print("\nwhere the prompt-side tokens come from:")
    print(
        f"  {'task':<10} {'start ctx':>9} {'final ctx':>9} {'resent base':>12} "
        f"{'accumulated':>12}  top tools"
    )
    for r in results:
        s = r.stats
        if not s.turns:
            continue
        # Turn 1's prompt IS the start context (system blocks + tool schemas +
        # the task): it is re-sent on every subsequent turn, so `base * turns`
        # is the share of all prompt tokens the harness itself is responsible
        # for. The remainder is conversation the agent generated.
        base = s.turns[0].prompt
        resent = base * s.turn_count
        accumulated = s.prompt_tokens - resent
        share = resent / s.prompt_tokens if s.prompt_tokens else 0.0
        print(
            f"  {r.fixture.slug:<10} {base:>9,} {s.turns[-1].prompt:>9,} "
            f"{resent:>11,} ({share:.0%}) {accumulated:>11,}  "
            f"{', '.join(f'{n}x{c}' for n, c in s.tools.most_common(4)) or '-'}"
        )

    print("\nwhat the accumulated context is made of, and what it costs:")
    print(
        f"  {'task':<10} {'tool results':>12} {'tool args':>10} {'asst prose':>11} "
        f"{'tool share':>10} | {'prompt $':>9} {'output $':>9}"
    )
    for r in results:
        s = r.stats
        if not s.transcript_chars:
            continue
        total = r.cost_usd or 1.0
        print(
            f"  {r.fixture.slug:<10} {s.tool_result_chars:>11,}c {s.tool_arg_chars:>9,}c "
            f"{s.assistant_text_chars:>10,}c {s.tool_traffic_share:>10.0%} | "
            f"{pricing.prompt_cost(s) / total:>8.0%} {pricing.output_cost(s) / total:>8.0%}"
        )

    print("\ncache trajectory (per-turn read/prompt; does the warm rate hold?):")
    for r in results:
        s = r.stats
        if s.turn_count < 2:
            print(f"  {r.fixture.slug:<10} too few turns to say")
            continue
        mid = 1 + (s.turn_count - 1) // 2
        print(
            f"  {r.fixture.slug:<10} turn1 {s.turns[0].cache_rate:>6.1%} | "
            f"early(2..{mid}) {s.window_rate(1, mid):>6.1%} | "
            f"late({mid + 1}..{s.turn_count}) {s.window_rate(mid, None):>6.1%} | "
            f"warm {s.warm_cache_rate:>6.1%}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hosting", default=DEFAULT_HOSTING)
    parser.add_argument("--model", default=os.environ.get("LO_BENCH_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--task",
        action="append",
        choices=[f.slug for f in FIXTURES],
        help="Run only this fixture (repeatable). Default: all.",
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument(
        "--out",
        type=Path,
        help="Scratch root for the fixtures. Default: a fresh temp directory.",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep the scratch directory (implied by --out).",
    )
    parser.add_argument("--json-out", type=Path, help="Also write the results as JSON here.")
    args = parser.parse_args()

    selected = [f for f in FIXTURES if not args.task or f.slug in args.task]
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if args.hosting == DEFAULT_HOSTING and not api_key:
        print("OPENROUTER_API_KEY is not set; this benchmark only runs live.", file=sys.stderr)
        return 2

    root = args.out or Path(tempfile.mkdtemp(prefix="lo-bench-cost-"))
    root.mkdir(parents=True, exist_ok=True)
    keep = args.keep or args.out is not None
    pricing = resolve_pricing(args.hosting, args.model, api_key)

    print(f"scratch: {root}")
    results: list[TaskResult] = []
    try:
        for fixture in selected:
            print(f"running {fixture.slug} ...", flush=True)
            results.append(
                run_fixture(
                    fixture,
                    root=root,
                    hosting=args.hosting,
                    model=args.model,
                    pricing=pricing,
                    timeout=args.timeout,
                )
            )
        print_report(results, pricing, args.model)
        if args.json_out:
            args.json_out.write_text(
                json.dumps(
                    {
                        "model": args.model,
                        "hosting": args.hosting,
                        "pricing": {
                            "source": pricing.source,
                            "input_usd_per_m": pricing.input_usd,
                            "output_usd_per_m": pricing.output_usd,
                            "cache_read_usd_per_m": pricing.cache_read_usd,
                        },
                        "tasks": [
                            {
                                "slug": r.fixture.slug,
                                "turns": r.stats.turn_count,
                                "tool_calls": r.stats.tool_calls,
                                "tools": dict(r.stats.tools),
                                "prompt_tokens": r.stats.prompt_tokens,
                                "fresh_tokens": r.stats.fresh_tokens,
                                "cache_read_tokens": r.stats.cache_read_tokens,
                                "output_tokens": r.stats.output_tokens,
                                "cache_rate": r.stats.cache_rate,
                                "warm_cache_rate": r.stats.warm_cache_rate,
                                "per_turn": [
                                    {"prompt": t.prompt, "cache_read": t.cache_read}
                                    for t in r.stats.turns
                                ],
                                "tool_result_chars": r.stats.tool_result_chars,
                                "tool_arg_chars": r.stats.tool_arg_chars,
                                "assistant_text_chars": r.stats.assistant_text_chars,
                                "prompt_cost_usd": pricing.prompt_cost(r.stats),
                                "output_cost_usd": pricing.output_cost(r.stats),
                                "compactions": r.stats.compactions,
                                "retries": r.stats.retries,
                                "cost_usd": r.cost_usd,
                                "cost_uncached_usd": r.cost_uncached_usd,
                                "wall_s": r.wall_s,
                                "exit_code": r.exit_code,
                                "verified": r.check.ok,
                                "verdict": _verdict(r.check),
                                "detail": r.check.detail,
                            }
                            for r in results
                        ],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"\nwrote {args.json_out}")
    finally:
        if keep:
            print(f"\nscratch kept at {root}")
        else:
            shutil.rmtree(root, ignore_errors=True)

    # An unverified or failed task is a legitimate finding, not a crash: the
    # numbers are still valid and the report already says which. The exit code
    # exists so CI can notice a regression, so it tracks the verdicts.
    return 0 if all(r.check.ok for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
