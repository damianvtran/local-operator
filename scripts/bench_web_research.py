#!/usr/bin/env python3
"""Live A/B benchmark for web-research turns (``web_search`` + ``web_fetch``).

Why this exists
---------------
Research turns are dominated by model time (~89%), and each extra round trip
costs ~30 s. The cheapest speedup is a round trip that never happens — e.g. a
``read spill://…?q=`` re-read after an oversized ``web_fetch`` preview hid the
answer in the elided middle. Whether a change actually removes round trips at
the same or lower token cost is a property of real models on real pages, so it
can only be measured live, A/B, on the same questions.

Two modes
---------
- ``--offline`` (deterministic, no tokens): fetches each long-page task's URL
  once through EACH arm's own ``run_fetch`` and reports whether the expected
  answer is in the preview — head/tail (no query) vs query passages (when the
  arm supports ``query``). This is the mechanism evidence.
- live (default): runs every task × rep × arm through the real CLI
  (``lop exec --json``), interleaving arms, and reports wall time, round trips,
  tool calls, spill re-reads, tokens, cost and a regex correctness grade.

Isolation (constraints, all deliberate — see AGENTS.md "Isolating a run")
--------------------------------------------------------------------------
- Every run gets a FRESH root: ``HOME`` and ``LOCAL_OPERATOR_CONFIG_DIR`` both
  point inside it and the cwd is inside it, so no cache, project MCP config or
  spill/fetch cache leaks between runs or from the operator's real home.
- ``CMUX_*``, ``LOP_*`` and ``LOCAL_OPERATOR_*`` are stripped (a ``lop`` parent
  exports names the child product reads), then the bench sets what it means.
- ``config.yml`` is written by EXPLICIT path inside the root — never through an
  environment fallback, which in an agent shell names the operator's live
  config — and the absolute path is printed.
- ``web_search`` providers are pinned identically for every arm (an ordered
  list plus every other provider excluded), so arms differ only by code.
- Only ``web_search,web_fetch,read`` are exposed: no bash, hence no git/gh
  keychain lookups from a scratch ``HOME``.
- The provider key comes ONLY from the caller's environment
  (``OPENROUTER_API_KEY``, optionally ``TAVILY_API_KEY``). It is passed to the
  child's environment and never printed or written to disk.
- Each arm's interpreter must import ITS OWN worktree (``local_operator.__file__``)
  — a symlinked or stale venv silently benchmarks someone else's code.

Usage
-----
    # mechanism check, no tokens
    .venv/bin/python scripts/bench_web_research.py --offline \\
        --arm base=~/lo-wt/base --arm head=. --out "$LOCAL_OPERATOR_SCRATCHPAD/wr"

    # live A/B (key exported inline, never echoed)
    OPENROUTER_API_KEY="$(lop secret get OPENROUTER_API_KEY_DEV)" \\
        .venv/bin/python scripts/bench_web_research.py \\
        --arm base=~/lo-wt/base --arm head=. --reps 3 --out "$LOCAL_OPERATOR_SCRATCHPAD/wr"

Output (JSON + a markdown table) goes under ``--out``: scratch, never the repo.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from local_operator.agent_shell import harness_child_env  # noqa: E402

#: The operator's default model and effort, as routed through OpenRouter
#: (id confirmed against https://openrouter.ai/api/v1/models on 2026-10-26).
DEFAULT_MODEL = "anthropic/claude-sonnet-5.5"
DEFAULT_EFFORT = "high"
HOSTING = "openrouter"
TOOLS = "web_search,web_fetch,read"
#: Env prefixes a ``lop`` parent exports that the CHILD product reads.
STRIPPED_PREFIXES = ("CMUX_", "LOP_", "LOCAL_OPERATOR_")
#: macOS hands an agent shell an XPC_FLAGS value under which a child's DNS
#: resolution can fail; a bench whose every fetch fails measures nothing.
STRIPPED_NAMES = ("XPC_FLAGS",)
PASSTHROUGH_KEYS = ("OPENROUTER_API_KEY", "TAVILY_API_KEY")


@dataclass(frozen=True)
class Task:
    """One question a user would type, with how to grade the answer.

    Every expected answer was checked against the primary source named in
    ``url`` on 2026-10-26 (the page text fetched through ``web_fetch``), not
    recalled: a wrong key grades a right answer as a failure.
    """

    slug: str
    kind: str  # "long": the answer sits deep in a long page; "short": snippet-answerable
    prompt: str
    grade: str  # preview regex; NOT the correctness grader (see validate_answer)
    url: str = ""  # primary source; long tasks are fetched by --offline
    query: str = ""  # the fixed, realistic query --offline uses
    preview: str = ""  # regex proving the ANSWER (not just the topic) is in a preview
    # Correctness is deliberately NOT the bare ``grade`` regex: an unanchored
    # match calls "the default is 50, not 100" correct. ``expected`` holds the
    # accepted literal answers (any one suffices); ``distractors`` holds wrong
    # values for the ASKED question. ``validate_answer`` requires an expected
    # value that is not negated, and rejects a distractor only when the answer
    # PRESENTS IT AS the asked value — an adjacent documented value mentioned
    # for context (PEP 8's 72 for docstrings vs 79 for code) is not a wrong
    # answer. Both are matched as whole tokens, so "100" never fires inside
    # "1000" or "10000".
    expected: tuple[str, ...] = ()
    distractors: tuple[str, ...] = ()
    # Regexes for phrases where a distractor is legitimate ADJACENT context,
    # not the asked answer (e.g. SQLite's compile-time maximum 32767 next to
    # the default 2000). Each match is scrubbed before the distractor scan, so
    # a wrong value is only ever forgiven in a context the task author named.
    # Declared per task rather than inferred: an inferred exemption is how the
    # clause-level grader re-admitted "50, but the default is 100".
    allowed_context: tuple[str, ...] = ()


TASKS: tuple[Task, ...] = (
    Task(
        "pg-max-connections",
        "long",
        "What's the default max_connections in PostgreSQL? Check the official docs.",
        r"\b100\b",
        "https://www.postgresql.org/docs/current/runtime-config-connection.html",
        "max_connections default",
        r"default is typically 100",
        expected=("100",),
        distractors=("1000", "10000", "50", "200", "500"),
    ),
    Task(
        "sqlite-max-column",
        "long",
        "What is SQLite's default SQLITE_MAX_COLUMN limit? Check sqlite.org.",
        r"\b2,?000\b",
        "https://www.sqlite.org/limits.html",
        "SQLITE_MAX_COLUMN default",
        r"default setting for SQLITE\\?_MAX\\?_COLUMN is 2000",
        expected=("2000", "2,000"),
        distractors=("32767", "1000", "4000", "200"),
        # 32767 is the compile-time MAXIMUM, legitimately mentioned next to the
        # default. Forgive it only in a raise/max/up-to/at-most/upper-bound
        # phrase; "the default is 32767" is not scrubbed and still fails.
        allowed_context=(
            # \b on every keyword: without it "max" matched the MAX inside
            # SQLITE_MAX_COLUMN and the phrase swallowed "_COLUMN is 32767",
            # scrubbing a wrong answer's own value (round-3 R4).
            r"\b(?:max(?:imum)?|raise[sd]?|raising|up to|at most|increas\w*|"
            r"upper (?:bound|limit)|compile[- ]time (?:max(?:imum)?|limit)|larger than)\b"
            r"[^.\n]{0,60}?32,?767",
            r"32,?767[^.\n]{0,30}?\b(?:max(?:imum)?|at most|upper (?:bound|limit))\b",
        ),
    ),
    Task(
        "h2-initial-window",
        "long",
        "In HTTP/2 (RFC 9113), what's the initial value of SETTINGS_INITIAL_WINDOW_SIZE?",
        r"65,?535",
        "https://www.rfc-editor.org/rfc/rfc9113.html",
        "SETTINGS_INITIAL_WINDOW_SIZE initial value",
        r"initial value is 2\^?16-1 \(65,535\)|initial value is 216-1 \(65,535\)",
        expected=("65535", "65,535"),
        distractors=("65536", "65534", "655350"),
    ),
    Task(
        "git-gc-auto",
        "long",
        "What's the default value of git's gc.auto setting? Check the git docs.",
        r"\b6,?700\b",
        "https://git-scm.com/docs/git-gc",
        "gc.auto default value",
        r"default value is 6700",
        expected=("6700", "6,700"),
        distractors=("670", "67000", "256"),
    ),
    Task(
        "pep8-docstring-length",
        "long",
        "Per PEP 8, what line length limit applies to docstrings and comments?",
        r"\b72\b",
        "https://peps.python.org/pep-0008/",
        "docstrings comments line length limit",
        r"limited to 72",
        expected=("72",),
        distractors=("79", "80", "88", "120"),
        # 79 is the CODE line limit; 72 is the docstring/comment one. Forgive
        # 79 only where it is attached to code (either order, incl. table rows
        # and parentheticals), never as the docstring/comment answer itself.
        allowed_context=(
            # \b again: "code" matched inside unicode/encode/decode.
            r"\b(?:code|program|source)\b[^\n]{0,40}?\b79\b",
            r"\b79\b[^\n]{0,40}?\b(?:code|program|source)\b",
            # The document's own words for the general line limit, which a
            # correct answer quotes alongside the 72 docstring limit: "Limit
            # all lines to a maximum of 79 characters", "other lines … 79".
            r"\b(?:all lines|other lines|every line|lines in general)\b[^\n]{0,50}?\b79\b",
            r"\b79\b[^\n]{0,30}?\b(?:for all lines|for other lines|all lines)\b",
        ),
    ),
    Task(
        "rfc9110-misdirected",
        "long",
        "Which HTTP status code is 'Misdirected Request' in RFC 9110, and when is it sent?",
        r"\b421\b",
        "https://www.rfc-editor.org/rfc/rfc9110.html",
        "Misdirected Request status code",
        r"The 421 \(Misdirected Request\) status code indicates",
        expected=("421",),
        distractors=("420", "422", "400", "404", "429"),
    ),
    # Snippet-answerable: a regression guard — the change must not slow these down.
    Task(
        "pg-default-port",
        "short",
        "What port does PostgreSQL listen on by default?",
        r"\b5432\b",
        "https://www.postgresql.org/docs/current/runtime-config-connection.html",
        expected=("5432",),
        distractors=("5433", "54321", "54320"),
    ),
    Task(
        "python-creator",
        "short",
        "Who created the Python programming language?",
        r"guido",
        "https://docs.python.org/3/faq/general.html",
        expected=("guido van rossum", "guido"),
        distractors=(),
    ),
    Task(
        "git-first-release",
        "short",
        "What year was Git first created?",
        r"\b2005\b",
        "https://git-scm.com/book/en/v2/Getting-Started-A-Short-History-of-Git",
        expected=("2005",),
        distractors=("2006", "2007", "2010", "2004"),
    ),
    Task(
        "https-default-port",
        "short",
        "What's the default TCP port for HTTPS?",
        r"\b443\b",
        "https://www.rfc-editor.org/rfc/rfc9110.html",
        expected=("443",),
        distractors=("444", "80", "8443", "4433"),
        # 80 is HTTP's port, legitimately mentioned beside 443. Declared so a
        # correct answer is not rejected for naming it (round-3 M5).
        allowed_context=(
            r"\bHTTP\b[^.\n]{0,30}?\b80\b",
            r"\b80\b[^.\n]{0,30}?\bHTTP\b",
        ),
    ),
)


@dataclass
class Arm:
    name: str
    path: Path
    python: Path
    resolved: str = ""


@dataclass
class RunRecord:
    arm: str
    task: str
    rep: int
    exit: int = 0
    wall_s: float = 0.0
    round_trips: int = 0
    tool_calls: dict[str, int] = field(default_factory=dict)
    fetch_with_query: int = 0
    fetch_without_query: int = 0
    spill_reads: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float | None = None
    answer: str = ""
    correct: bool = False
    root: str = ""
    error: str = ""


# -- correctness -------------------------------------------------------------

#: Negations that turn a value into a NON-answer. Case-insensitive: a
#: sentence-initial "Not"/"Never"/"No" negates just as much as a lower-case one
#: (round-2 R3: the case-sensitive pattern let "Not 100" through).
_NEGATION_RE = re.compile(
    # ``no`` NOT followed by punctuation: "no 100 default" negates the value,
    # while "No, the default is 100." is an interjection (round-3 M5).
    r"(?:\bnot\b|\bnever\b|\bno\b(?!\s*[,;:.!?])"
    r"|\bisn't\b|\bwasn't\b|\baren't\b|\bweren't\b"
    r"|\bdon't\b|\bdoesn't\b|\bdidn't\b"
    r"|\binstead of\b|\brather than\b)",
    re.IGNORECASE,
)

#: How far around a value a negation is looked for. Both directions: "100 is
#: not the default" negates AFTER the value, "not 100" before it.
_NEGATION_WINDOW = 24


def _literal_pattern(value: str) -> re.Pattern[str]:
    """Whole-value literal match: "100" must not match inside "1000", "10000",
    "100,000" or "100.5" (M2: the digit-only lookahead allowed the last two).
    """
    if any(ch.isdigit() for ch in value):
        # N2: also reject a unit suffix ("100k"), which is not the plain value.
        return re.compile(
            rf"(?<![\d.,]){re.escape(value)}(?![\d]|[.,]\d|[kKmMgG]\b)", re.IGNORECASE
        )
    return re.compile(rf"\b{re.escape(value)}\b", re.IGNORECASE)


#: Forward negation needs a COPULA: "100 is not the default" negates the value,
#: but "100, not 50" does not (the "not" belongs to the OTHER value). Only the
#: copula form is looked for after the value; a bare "not" after it is someone
#: else's negation (M1).
_FORWARD_NEGATION_RE = re.compile(
    r"^\s*(?:is|was|are|were|be|being|been|do|does|did)\s+(?:not|never)\b"
    r"|^\s*(?:isn't|wasn't|aren't|weren't|don't|doesn't|didn't)\b",
    re.IGNORECASE,
)


def _is_negated(text: str, match: re.Match[str]) -> bool:
    """Whether this value occurrence is negated, looking BOTH ways.

    Backwards: any negation word in the window ("not 100", "Never 2005").
    Forwards: only a copula negation ("100 is not the default"). Case-insensitive
    throughout (R3: sentence-initial Not/Never/No must count).
    """
    before = text[max(0, match.start() - _NEGATION_WINDOW) : match.start()]
    after = text[match.end() : match.end() + _NEGATION_WINDOW]
    # "not only / not just / no doubt" are additive idioms, not negations:
    # "Guido van Rossum is not only Python's creator but also …" is a correct
    # answer, and scanning the raw window would negate his name (round-2 N1).
    before = re.sub(r"not (?:only|just)|no doubt", "", before, flags=re.IGNORECASE)
    # An admonition is not a negation either: "Don't change it: 100 is the
    # default." states the answer (round-3 M5). Narrow on purpose — dropping
    # don't/doesn't/didn't wholesale would accept "doesn't equal 100".
    before = re.sub(
        r"\b(?:don't|do not)\s+(?:change|touch|move|worry|forget)\b",
        "",
        before,
        flags=re.IGNORECASE,
    )
    return bool(_NEGATION_RE.search(before)) or bool(_FORWARD_NEGATION_RE.match(after))


def _scrub_allowed_context(task: Task, answer: str) -> str:
    """Blank the DISTRACTOR LITERALS that sit inside a declared context phrase.

    Only the wrong value is blanked, never the whole phrase: blanking the span
    would erase an expected value inside it too ("SQLITE_MAX_COLUMN defaults to
    2000 and can be raised to 32767" lost its 2000), and a wrong value stated AS
    the answer must stay visible so the run fails (round-3 R4).

    Offsets are preserved (the replacement is spaces of the same length), so
    every later match keeps its position in the original answer.
    """
    chars: list[str] | None = None
    for pattern in task.allowed_context:
        for context in re.finditer(pattern, answer, re.IGNORECASE):
            phrase = context.group(0)
            for wrong in task.distractors:
                for match in _literal_pattern(wrong).finditer(phrase):
                    if chars is None:
                        chars = list(answer)
                    begin = context.start() + match.start()
                    for index in range(begin, begin + (match.end() - match.start())):
                        chars[index] = " "
    return "".join(chars) if chars is not None else answer


def validate_answer(task: Task, answer: str) -> bool:
    """Whether ``answer`` states the task's answer, without contradicting it.

    Document-level, with narrow declared exemptions (round-2 R3/Q4 design):
    - any distractor anywhere fails the run, EXCEPT inside a task-declared
      ``allowed_context`` phrase (scrubbed first). A wrong value in a different
      clause/sentence/bullet from the right one is still a wrong answer — the
      clause-level grader missed exactly that mirror case;
    - the expected value must appear at least once UNNEGATED (negation checked
      on both sides of the value, case-insensitively).

    Adjacent documented values are handled by naming their context, not by
    relaxing the distractor rule: SQLite's "up to 32767" and PEP 8's "79 for
    code" are declared on those tasks, so those correct answers pass while
    "the default is 50, not 100" and "50. Not 100." still fail.
    """
    if not answer.strip() or not task.expected:
        return False
    scrubbed = _scrub_allowed_context(task, answer)
    for wrong in task.distractors:
        for match in _literal_pattern(wrong).finditer(scrubbed):
            # A NEGATED distractor is not a wrong answer — "100, not 50" and
            # "2005, not 2006" state the right value and reject the wrong one
            # (M1). Only an unnegated distractor fails the run.
            if not _is_negated(scrubbed, match):
                return False
    for value in task.expected:
        for match in _literal_pattern(value).finditer(scrubbed):
            if not _is_negated(scrubbed, match):
                return True
    return False


def is_valid(record: RunRecord) -> bool:
    """A run eligible for correctness AND performance comparison.

    A failed/crashed/timed-out run (exit != 0) or one with no final answer is
    not a data point: it is usually fast, so including it would flatter wall
    time, and its answer (often empty or an apology) says nothing about
    correctness. Such runs are still recorded and reported, separately.
    """
    return record.exit == 0 and not record.error and bool(record.answer.strip())


# -- isolation ---------------------------------------------------------------


def clean_env(home: Path, config_dir: Path) -> dict[str, str]:
    """The child's environment: stripped of parent session names, isolated."""
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(STRIPPED_PREFIXES) and k not in STRIPPED_NAMES
    }
    env["HOME"] = str(home)
    env["LOCAL_OPERATOR_CONFIG_DIR"] = str(config_dir)
    # No bash tool is exposed, but lop itself may shell out to git at startup;
    # the system gitconfig names the osxkeychain helper, so neutralise it.
    env["GIT_CONFIG_SYSTEM"] = "/dev/null"
    env["GIT_TERMINAL_PROMPT"] = "0"
    return harness_child_env(env)


def write_config(config_dir: Path, providers: list[str]) -> Path:
    """Write ``config.yml`` at an EXPLICIT path and return it.

    Pins web_search identically for every arm: ``ordered`` walks the list as
    given, and every provider not listed is excluded so the resolver cannot
    append whatever this machine happens to have credentials for.
    """
    from local_operator.web_search.models import PROVIDER_IDS

    config_dir.mkdir(parents=True, exist_ok=True)
    path = (config_dir / "config.yml").resolve()
    excluded = [p for p in PROVIDER_IDS if p not in providers]
    values = {
        "web_search": {
            "enabled": True,
            "strategy": "ordered",
            "providers": providers,
            "excluded_providers": excluded,
        },
    }
    # Notifications are silenced by ``harness_child_env`` (env switch), not
    # here. JSON is valid YAML, and keeps this script free of a yaml import;
    # ``version``/``metadata`` are left for the product to stamp.
    path.write_text(json.dumps({"values": values}, indent=2) + "\n")
    return path


def fresh_root(out: Path, label: str) -> tuple[Path, Path, Path]:
    root = Path(tempfile.mkdtemp(prefix=f"{label}-", dir=out / "runs")).resolve()
    home = root / "home"
    config_dir = home / ".local-operator"
    work = root / "work"
    for d in (home, config_dir, work):
        d.mkdir(parents=True, exist_ok=True)
    return root, config_dir, work


def resolve_arm(spec: str) -> Arm:
    if "=" not in spec:
        raise SystemExit(f"--arm wants NAME=<worktree path>, got {spec!r}")
    name, raw = spec.split("=", 1)
    path = Path(raw).expanduser().resolve()
    python = path / ".venv" / "bin" / "python"
    if not python.exists():
        raise SystemExit(f"arm {name}: no venv interpreter at {python}")
    # cwd=/ so the import is decided by the venv's editable finder, not by a
    # cwd that happens to contain a local_operator package.
    probe = subprocess.run(
        [str(python), "-c", "import local_operator; print(local_operator.__file__)"],
        capture_output=True,
        text=True,
        cwd="/",
        env=clean_env(Path(tempfile.gettempdir()), Path(tempfile.gettempdir()) / "lo-probe"),
        timeout=120,
    )
    resolved = probe.stdout.strip()
    expected = str(path / "local_operator" / "__init__.py")
    if resolved != expected:
        raise SystemExit(
            f"arm {name}: venv imports {resolved or probe.stderr.strip()!r}, not {expected!r}"
        )
    return Arm(name, path, python, resolved)


# -- offline mode --------------------------------------------------------------

#: Run INSIDE each arm's interpreter, so the preview is exactly what that
#: arm's code produces. Reads its inputs from argv/stdin; prints one JSON line.
_OFFLINE_SNIPPET = r"""
import asyncio, inspect, json, re, sys
from local_operator.harness.types import ToolContext
from local_operator.web_fetch.tool import run_fetch
spec = json.loads(sys.stdin.read())
async def main():
    ctx = ToolContext(cwd=".", session_id="bench-offline")
    out = {"supports_query": "query" in inspect.signature(run_fetch).parameters}
    plain, details, err = await run_fetch(spec["url"], tool_name="web_fetch", context=ctx)
    out.update(status=details.get("status"), lines=details.get("lines"),
               oversized="spill" in details, error=bool(err))
    out["headtail_chars"] = len(plain)
    out["headtail_hit"] = bool(re.search(spec["preview"], plain))
    out["headtail_text"] = plain
    if out["supports_query"]:
        q, qd, _ = await run_fetch(spec["url"], tool_name="web_fetch", context=ctx,
                                   query=spec["query"])
        out["query_chars"] = len(q)
        out["query_hit"] = bool(re.search(spec["preview"], q))
        out["query_passages"] = qd.get("query_passages")
        out["query_text"] = q
    print(json.dumps(out))
asyncio.run(main())
"""


def run_offline(arms: list[Arm], tasks: list[Task], out: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in tasks:
        if task.kind != "long":
            continue
        for arm in arms:
            root, config_dir, work = fresh_root(out, f"offline-{arm.name}-{task.slug}")
            spec = {"url": task.url, "query": task.query, "preview": task.preview}
            proc = subprocess.run(
                [str(arm.python), "-c", _OFFLINE_SNIPPET],
                input=json.dumps(spec),
                capture_output=True,
                text=True,
                cwd=work,
                env=clean_env(root / "home", config_dir),
                timeout=180,
            )
            last = (proc.stdout.strip().splitlines() or ["{}"])[-1]
            try:
                row = json.loads(last)
            except ValueError:
                row = {"error": proc.stderr.strip()[-300:]}
            row.update(arm=arm.name, task=task.slug, query=task.query)
            # The preview texts are the evidence a reviewer reads; they go to
            # their own files so the JSON summary stays scannable.
            previews = out / "previews"
            previews.mkdir(exist_ok=True)
            for kind in ("headtail", "query"):
                text = row.pop(f"{kind}_text", None)
                if text is not None:
                    (previews / f"{task.slug}.{arm.name}.{kind}.txt").write_text(text)
            rows.append(row)
            print(
                f"  {task.slug:24} {arm.name:8} head/tail="
                f"{'HIT' if row.get('headtail_hit') else 'miss'}"
                + (
                    f"  query={'HIT' if row.get('query_hit') else 'miss'}"
                    f" ({row.get('query_chars')} chars)"
                    if row.get("supports_query")
                    else "  query=n/a"
                ),
                flush=True,
            )
            shutil.rmtree(root, ignore_errors=True)
    return rows


def offline_markdown(rows: list[dict[str, Any]], arms: list[Arm]) -> str:
    head = "| task | query | " + " | ".join(f"{a.name} head/tail | {a.name} query" for a in arms)
    lines = [head + " |", "|" + " --- |" * (2 + 2 * len(arms))]
    by = {(r["task"], r["arm"]): r for r in rows}
    for task in dict.fromkeys(r["task"] for r in rows):
        cells = [task, f"`{by[(task, arms[0].name)].get('query', '')}`"]
        for arm in arms:
            r = by.get((task, arm.name), {})
            cells.append("✅" if r.get("headtail_hit") else "❌")
            cells.append(
                ("✅" if r.get("query_hit") else "❌") + f" ({r.get('query_chars')} ch)"
                if r.get("supports_query")
                else "n/a"
            )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


# -- live mode ---------------------------------------------------------------


def parse_capture(stdout: str, record: RunRecord, task: Task) -> None:
    """Fill ``record`` from an ``exec --json`` event stream.

    ``message_end`` and ``turn_end`` carry the same final Message, so one
    family is chosen and messages are deduplicated by id — the same rule
    ``bench_complex_tasks.tally_cost`` uses — or every count doubles.
    """
    events = []
    for line in stdout.splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            continue
    kind = "message_end" if any(e.get("type") == "message_end" for e in events) else "turn_end"
    seen: set[str] = set()
    cost = 0.0
    priced = False
    answer = ""
    for event in events:
        if event.get("type") != kind:
            continue
        message = event.get("message") or {}
        if message.get("role") != "assistant":
            continue
        ident = message.get("id")
        if ident and ident in seen:
            continue
        if ident:
            seen.add(ident)
        record.round_trips += 1
        usage = message.get("usage") or {}
        record.input_tokens += int(usage.get("input_tokens") or 0)
        record.output_tokens += int(usage.get("output_tokens") or 0)
        record.cache_read_tokens += int(usage.get("cache_read_tokens") or 0)
        record.cache_write_tokens += int(usage.get("cache_write_tokens") or 0)
        # Provider-reported cost first; the estimate only when none was billed.
        # Unknown pricing stays None rather than a fabricated free run.
        for key in ("usd_cost", "estimated_usd_cost"):
            value = usage.get(key)
            if isinstance(value, (int, float)) and value:
                cost += float(value)
                priced = True
                break
        for call in message.get("tool_calls") or []:
            name = str(call.get("name") or "")
            args = call.get("arguments") or {}
            record.tool_calls[name] = record.tool_calls.get(name, 0) + 1
            if name == "web_fetch":
                if str(args.get("query") or "").strip():
                    record.fetch_with_query += 1
                else:
                    record.fetch_without_query += 1
            if name == "read" and str(args.get("path") or "").startswith("spill://"):
                record.spill_reads += 1
        texts = [
            c.get("text", "")
            for c in message.get("content") or []
            if isinstance(c, dict) and c.get("type") == "text"
        ]
        if any(t.strip() for t in texts):
            answer = "\n".join(texts)
    record.cost_usd = round(cost, 5) if priced else None
    record.answer = answer.strip()
    # Validity first: an exit != 0 / empty-answer run is never "correct".
    # The Task OBJECT is passed (not its slug or preview regex): taking the
    # object makes the Q1 wiring mix-up — a preview regex where a slug was
    # expected, silently grading every run False — unrepresentable.
    record.correct = bool(is_valid(record) and validate_answer(task, record.answer))


def run_live(arm: Arm, task: Task, rep: int, args: argparse.Namespace) -> RunRecord:
    record = RunRecord(arm=arm.name, task=task.slug, rep=rep)
    root, config_dir, work = fresh_root(args.out, f"{arm.name}-{task.slug}-{rep}")
    record.root = str(root)
    config_path = write_config(config_dir, args.search_providers)
    print(f"  [{arm.name}] {task.slug} rep {rep}: config {config_path}", flush=True)
    cmd = [
        str(arm.python),
        "-m",
        "local_operator.cli",
        "--hosting",
        HOSTING,
        "--model",
        args.model,
        "--run-in",
        str(work),
        "--yolo",
        "exec",
        "--json",
        "--tools",
        TOOLS,
        "--effort",
        args.effort,
        task.prompt,
    ]
    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=work,
            env=clean_env(root / "home", config_dir),
            timeout=args.timeout,
        )
        record.exit = proc.returncode
        stdout, stderr = proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        record.exit = -1
        record.error = f"timeout after {args.timeout}s"
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = ""
    record.wall_s = round(time.monotonic() - start, 2)
    (root / "capture.jsonl").write_text(stdout)
    if record.exit not in (0, -1):
        # The tail only: stderr carries log lines, never the key (it is read
        # from the environment and never logged by the product).
        record.error = stderr.strip()[-400:]
    parse_capture(stdout, record, task)
    return record


def _mean(xs: list[float]) -> float | None:
    return round(statistics.fmean(xs), 3) if xs else None


def _median(xs: list[float]) -> float | None:
    return round(statistics.median(xs), 3) if xs else None


def summarise(records: list[RunRecord], arms: list[Arm], tasks: list[Task]) -> dict[str, Any]:
    # Invalid runs are reported separately and NEVER averaged into wall time,
    # correctness or the paired ratios (a crashed run is fast and would flatter
    # whichever arm crashed more).
    invalid = [r for r in records if not is_valid(r)]
    eligible = [r for r in records if is_valid(r)]
    per: dict[str, dict[str, Any]] = {}
    for task in tasks:
        for arm in arms:
            rs = [r for r in eligible if r.task == task.slug and r.arm == arm.name]
            if not rs:
                continue
            costs = [r.cost_usd for r in rs if r.cost_usd is not None]
            per[f"{task.slug}|{arm.name}"] = {
                "task": task.slug,
                "kind": task.kind,
                "arm": arm.name,
                "n": len(rs),
                "attempted": sum(1 for r in records if r.task == task.slug and r.arm == arm.name),
                "wall_mean": _mean([r.wall_s for r in rs]),
                "wall_median": _median([r.wall_s for r in rs]),
                "correct": sum(r.correct for r in rs),
                "round_trips_mean": _mean([r.round_trips for r in rs]),
                "spill_reads_mean": _mean([r.spill_reads for r in rs]),
                "fetch_query_share": _mean(
                    [
                        r.fetch_with_query / (r.fetch_with_query + r.fetch_without_query)
                        for r in rs
                        if r.fetch_with_query + r.fetch_without_query
                    ]
                ),
                "input_tokens_mean": _mean([r.input_tokens for r in rs]),
                "output_tokens_mean": _mean([r.output_tokens for r in rs]),
                "cost_mean": _mean(costs),
            }
    # Paired per-task ratio of medians against the FIRST arm (the control): a
    # ratio per task, so one slow page cannot dominate the comparison.
    control = arms[0].name
    ratios: dict[str, dict[str, float]] = {}
    for arm in arms[1:]:
        for task in tasks:
            a = per.get(f"{task.slug}|{control}")
            b = per.get(f"{task.slug}|{arm.name}")
            if a and b and a["wall_median"] and b["wall_median"]:
                ratios.setdefault(arm.name, {})[task.slug] = round(
                    b["wall_median"] / a["wall_median"], 3
                )
    geo = {
        name: round(statistics.geometric_mean(list(rs.values())), 3)
        for name, rs in ratios.items()
        if rs
    }
    totals = {
        arm.name: {
            "runs": sum(1 for r in eligible if r.arm == arm.name),
            "attempted": sum(1 for r in records if r.arm == arm.name),
            "invalid": sum(1 for r in invalid if r.arm == arm.name),
            "correct": sum(r.correct for r in eligible if r.arm == arm.name),
            "cost": round(sum(r.cost_usd or 0.0 for r in records if r.arm == arm.name), 4),
        }
        for arm in arms
    }
    invalid_report = [
        {"arm": r.arm, "task": r.task, "rep": r.rep, "exit": r.exit, "error": r.error}
        for r in invalid
    ]
    return {
        "per": per,
        "ratios": ratios,
        "geomean_ratio": geo,
        "totals": totals,
        "invalid_runs": invalid_report,
    }


def live_markdown(summary: dict[str, Any], arms: list[Arm]) -> str:
    lines = [
        "| task | kind | arm | n | wall mean | wall median | correct | round trips |"
        " spill re-reads | fetch w/ query | in tok | out tok | cost $ |",
        "|" + " --- |" * 13,
    ]
    for row in summary["per"].values():
        lines.append(
            "| {task} | {kind} | {arm} | {n} | {wall_mean} | {wall_median} | {correct}/{n} |"
            " {round_trips_mean} | {spill_reads_mean} | {fetch_query_share} |"
            " {input_tokens_mean} | {output_tokens_mean} | {cost_mean} |".format(**row)
        )
    lines.append("")
    if summary.get("invalid_runs"):
        lines.append(
            "Invalid runs (excluded from every mean/ratio/correctness figure): "
            + "; ".join(
                f"{r['arm']}/{r['task']} rep {r['rep']} exit {r['exit']}"
                for r in summary["invalid_runs"]
            )
        )
        lines.append("")
    for name, rs in summary["ratios"].items():
        lines.append(
            f"Paired wall ratio {name}/{arms[0].name} (median per task): "
            + ", ".join(f"{k} {v}" for k, v in rs.items())
            + f" — geomean {summary['geomean_ratio'].get(name)}"
        )
    for name, t in summary["totals"].items():
        lines.append(f"{name}: {t['correct']}/{t['runs']} correct, ${t['cost']}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--arm", action="append", required=True, help="NAME=<worktree path>")
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--tasks", default="", help="comma-separated task slugs (default: all)")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True, help="scratch dir (never the repo)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--effort", default=DEFAULT_EFFORT)
    parser.add_argument("--timeout", type=int, default=900, help="per-run seconds")
    parser.add_argument(
        "--search-providers",
        type=lambda s: [p for p in s.split(",") if p],
        default=["duckduckgo", "tavily"],
        help="ordered web_search providers, pinned identically for every arm",
    )
    parser.add_argument(
        "--max-usd",
        type=float,
        default=None,
        help="stop scheduling runs once this much has been spent",
    )
    parser.add_argument("--offline", action="store_true", help="preview check only, no tokens")
    parser.add_argument("--keep-roots", action="store_true", help="keep per-run roots")
    args = parser.parse_args()

    args.out = args.out.expanduser().resolve()
    repo = Path(__file__).resolve().parent.parent
    if args.out == repo or repo in args.out.parents:
        raise SystemExit("--out must be outside the repository (scratch only)")
    (args.out / "runs").mkdir(parents=True, exist_ok=True)
    if len(args.arm) < 2:
        raise SystemExit("at least two --arm values are needed for an A/B")
    arms = [resolve_arm(spec) for spec in args.arm]
    for arm in arms:
        print(f"arm {arm.name}: {arm.python} imports {arm.resolved}")
    wanted = {s for s in args.tasks.split(",") if s}
    unknown = wanted - {t.slug for t in TASKS}
    if unknown:
        raise SystemExit(f"unknown task(s): {', '.join(sorted(unknown))}")
    tasks = [t for t in TASKS if not wanted or t.slug in wanted]

    if args.offline:
        rows = run_offline(arms, tasks, args.out)
        (args.out / "offline.json").write_text(json.dumps(rows, indent=2))
        table = offline_markdown(rows, arms)
        (args.out / "offline.md").write_text(table)
        print("\n" + table + f"\nwrote {args.out / 'offline.json'} and offline.md")
        return 0

    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY must be exported by the caller for a live run")

    # Interleave arms per task/rep, rotating which arm goes first so neither
    # arm systematically meets a cold provider cache or a warm one.
    jobs: list[tuple[Arm, Task, int]] = []
    for rep in range(1, args.reps + 1):
        for index, task in enumerate(tasks):
            shift = (rep + index) % len(arms)
            for arm in arms[shift:] + arms[:shift]:
                jobs.append((arm, task, rep))

    records: list[RunRecord] = []
    lock = threading.Lock()
    spent = {"usd": 0.0}

    def work(job: tuple[Arm, Task, int]) -> RunRecord | None:
        arm, task, rep = job
        with lock:
            if args.max_usd is not None and spent["usd"] >= args.max_usd:
                return None
        record = run_live(arm, task, rep, args)
        with lock:
            spent["usd"] += record.cost_usd or 0.0
            records.append(record)
        print(
            f"  [{arm.name}] {task.slug} rep {rep}: exit {record.exit} {record.wall_s}s "
            f"rt={record.round_trips} tools={record.tool_calls} spill={record.spill_reads} "
            f"correct={record.correct} cost={record.cost_usd} (spent ${spent['usd']:.3f})",
            flush=True,
        )
        if not args.keep_roots:
            shutil.rmtree(record.root, ignore_errors=True)
        return record

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        list(pool.map(work, jobs))

    summary = summarise(records, arms, tasks)
    payload = {
        "model": args.model,
        "effort": args.effort,
        "search_providers": args.search_providers,
        "arms": {a.name: str(a.path) for a in arms},
        "records": [r.__dict__ for r in records],
        "summary": summary,
    }
    (args.out / "results.json").write_text(json.dumps(payload, indent=2))
    table = live_markdown(summary, arms)
    (args.out / "report.md").write_text(table)
    print("\n" + table + f"\nwrote {args.out / 'results.json'} and report.md")
    return 0 if records else 1


if __name__ == "__main__":
    raise SystemExit(main())
