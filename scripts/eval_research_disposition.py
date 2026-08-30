#!/usr/bin/env python3
"""A/B eval: does the agent reach for the web when its own knowledge runs out?

The change under test widens the harness's definition of verification. Before
it, ``system.md``'s first working principle enumerated verification as "run the
command, read the file, search the workspace" — a complete-sounding list whose
members are all LOCAL. An agent following that principle faithfully never
searches the web, because by the definition it was handed it has already
verified. The fix adds "look it up on the web" to that clause and one bullet
naming the trigger (you notice you are guessing about something outside this
machine).

Why an eval and not a unit test: the behaviour is probabilistic, so the artifact
is a recorded rate, not an assertion. This lives in ``scripts/`` and never runs
in CI — it costs real provider tokens and needs network.

WHAT IT MEASURES, and why both halves are load-bearing:

- SHOULD-trigger scenarios: the agent is asked something whose answer is not on
  this machine (a dependency error from a version it cannot know, a published
  advisory, current ecosystem practice). Passing means it searched.
- SHOULD-NOT-trigger scenarios: pure local refactors, questions about lop
  itself, arithmetic. Passing means it did NOT search. This half is the
  guardrail on the whole change: prompting an agent to search is trivial, and
  the failure it buys is a reflexive search on every trivial task, which is a
  latency and token regression the user pays on work that never needed it.
  S13 ("how do I change my custom instructions in lop") is the highest-signal
  negative in the set — the correct behaviour is ``read guide://configuration``
  per system.md's guide rule, so a web search there is a REGRESSION AGAINST AN
  EXISTING RULE, not merely a wasted call.

Each arm runs against a git worktree, so "before" is the real unmodified
prompt as shipped, not a reconstruction. The scenarios are held identical
across arms and the model is pinned; only the checkout differs.

Both arms run against an ISOLATED ``LOCAL_OPERATOR_CONFIG_DIR`` seeded only
with the operator's credentials, so what is measured is the PACKAGED prompt.
Running in the real config dir would mix in the operator's
``system_prompt.md``, MCP servers and skills, none of which ship to users, and
any of which could produce the search behaviour on its own and be mistaken for
the change working.

The variable is ``LOCAL_OPERATOR_CONFIG_DIR`` (``paths.py:26``), NOT
``LOCAL_OPERATOR_HOME`` (``paths.py:46``) — the latter relocates the agent's
workspace, not its config, so setting it leaves transcripts in the real config
dir where this script cannot find them. That misread is silent and costly: the
run completes, no transcript is located, every scenario reports zero tool calls,
and both arms score 0% as though the agent had chosen not to search.

Usage:
    .venv/bin/python scripts/eval_research_disposition.py \
        --before /tmp/lop-before --after /tmp/lop-websearch --runs 3
"""

from __future__ import annotations

import argparse
import json
import os
import hashlib
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

WEB_TOOLS = {"web_search", "web_fetch"}

#: An agent can reach the network WITHOUT the web tools, by curling from `bash`
#: or opening a URL from `eval`. Counting only ``web_search``/``web_fetch``
#: would score those runs as "did not research" and understate BOTH arms —
#: observed on the first real run, where both arms queried the OSV advisory API
#: through curl. That is still research, so it is scored as research; the two
#: are reported separately because moving a run from improvised curl to the
#: purpose-built tool is a real improvement (bounded output, spill handles,
#: caching, provider fallback) even though the intent was already there.
_URL_RE = re.compile(r"https?://", re.IGNORECASE)
_NETWORK_CMD_RE = re.compile(r"\b(curl|wget|urllib|httpx|requests\.get|nc)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Scenario:
    """One prompt plus the behaviour we expect from it.

    ``should_search`` is the graded expectation. ``rationale`` is recorded in
    the output so a later reader can tell a scenario that was mis-specified
    from a model that genuinely misbehaved.
    """

    sid: str
    prompt: str
    should_search: bool
    rationale: str


# Ten SHOULD-trigger scenarios spanning the failure modes the user named:
# third-party error messages, unfamiliar/post-cutoff APIs, version-specific
# behaviour, security advisories, and design/UX practice research.
SHOULD_TRIGGER: list[Scenario] = [
    Scenario(
        "S1",
        "I'm getting this error from a dependency and I don't understand it:\n"
        "  pydantic.errors.PydanticUserError: `@root_validator` with pre=False "
        "must specify `skip_on_failure=True`.\n"
        "Why is this happening and what's the current fix? Don't change any "
        "files, just explain.",
        True,
        "Third-party error message with version-specific semantics.",
    ),
    Scenario(
        "S2",
        "Does httpx.AsyncClient still accept the `proxies=` argument in the "
        "latest release, or was it renamed? Answer only; change nothing.",
        True,
        "Version-specific API behaviour that moved after training cutoffs.",
    ),
    Scenario(
        "S3",
        "Are there any published security advisories affecting requests 2.31.0? "
        "Just tell me, don't edit anything.",
        True,
        "Published advisory: recall is unsafe, the CVE list is authoritative.",
    ),
    Scenario(
        "S4",
        "What's the current recommended way to build a Python wheel for PyPI - "
        "`uv build`, `python -m build`, or something else? What does the "
        "ecosystem actually do now? Answer only.",
        True,
        "Current ecosystem practice; the answer changes year to year.",
    ),
    Scenario(
        "S5",
        "I want to redesign our terminal session picker to feel more like a "
        "modern command palette. What patterns do good command palettes use? "
        "Just give me the design thinking, don't write code.",
        True,
        "Design/UX inspiration - the user's named second case.",
    ),
    Scenario(
        "S6",
        "What's the current best practice for placing prompt-cache breakpoints "
        "with Anthropic's API? Answer only, change nothing.",
        True,
        "Live provider documentation, not a stable fact.",
    ),
    Scenario(
        "S7",
        "Our CI fails with: `error: externally-managed-environment` when pip "
        "installs. What causes this and what's the accepted fix? Explain only.",
        True,
        "Ecosystem-level packaging change (PEP 668) with an evolving fix.",
    ),
    Scenario(
        "S8",
        "Is Textual's `App.save_screenshot` still the supported way to export a "
        "frame, or is there a newer API? Answer only.",
        True,
        "Library API currency for a dependency this repo actually uses.",
    ),
    Scenario(
        "S9",
        "What are the accessibility contrast requirements we should hold a "
        "terminal UI colour theme to? Answer only.",
        True,
        "External standard (WCAG); precise thresholds should be checked.",
    ),
    Scenario(
        "S10",
        "A provider returned HTTP 529 to our client. What is that status and "
        "how should a client handle it? Explain only, change nothing.",
        True,
        "Non-standard status code; guessing produces a plausible wrong answer.",
    ),
]

# Four SHOULD-NOT scenarios. These are the guardrail: the change must not buy
# its search rate by making the agent search reflexively.
SHOULD_NOT_TRIGGER: list[Scenario] = [
    Scenario(
        "S11",
        "In this repo, what does the function `_render_tool_inventory` return? "
        "Answer from the code.",
        False,
        "Purely local: the answer is in prompts_api.py.",
    ),
    Scenario(
        "S12",
        "How do I change my custom instructions in lop?",
        False,
        "Answer is guide://configuration. A web search is a regression "
        "against system.md's existing guide rule - highest-signal negative.",
    ),
    Scenario(
        "S13",
        "If a session costs 9865 always-on tokens and we add 154, what is the "
        "percentage increase? Just the number.",
        False,
        "Arithmetic. No external fact exists to look up.",
    ),
    Scenario(
        "S14",
        "List the Python files in the local_operator/web_search directory and "
        "tell me which one defines the tool.",
        False,
        "Pure filesystem inspection.",
    ),
]

ALL_SCENARIOS = SHOULD_TRIGGER + SHOULD_NOT_TRIGGER


def _newest_session_dir(sessions_root: Path, since: float) -> Path | None:
    """The session directory created after ``since``.

    exec mode does not print its session id, so the transcript is located by
    mtime. The ``since`` floor keeps a concurrent session from being adopted.
    """

    best: tuple[float, Path] | None = None
    if not sessions_root.is_dir():
        return None
    for child in sessions_root.iterdir():
        transcript = child / "transcript.jsonl"
        if not transcript.is_file():
            continue
        mtime = transcript.stat().st_mtime
        if mtime < since:
            continue
        if best is None or mtime > best[0]:
            best = (mtime, child)
    return best[1] if best else None


def _tool_calls(transcript: Path) -> list[tuple[str, str]]:
    """Every tool name the assistant invoked, in order.

    Tool invocations live on the assistant message's ``tool_calls`` array, not
    in a distinct record type — verified against a real transcript before this
    was written, because scanning for a ``tool_call`` record type silently
    returns zero for every run and reads as "the agent never searched".
    """

    calls: list[tuple[str, str]] = []
    if not transcript.is_file():
        return calls
    for line in transcript.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line).get("payload") or {}
        except json.JSONDecodeError:
            continue
        for call in payload.get("tool_calls") or []:
            name = call.get("name")
            if isinstance(name, str):
                calls.append((name, json.dumps(call.get("arguments") or {})))
    return calls


def _improvised_network(calls: list[tuple[str, str]]) -> list[str]:
    """Calls that reached the network without using a web tool.

    Matches a URL AND a network verb in the arguments of a general-purpose
    tool. Requiring both keeps a shell command that merely mentions a URL in a
    comment, or a `pip install` against the local index, from scoring as
    research.
    """

    hits: list[str] = []
    for name, arguments in calls:
        if name in WEB_TOOLS or name not in {"bash", "eval"}:
            continue
        if _URL_RE.search(arguments) and _NETWORK_CMD_RE.search(arguments):
            hits.append(name)
    return hits


def run_one(repo: Path, scenario: Scenario, timeout: int, home: Path) -> dict[str, Any]:
    """Run one scenario in one arm and report which tools it used.

    Runs in a scratch cwd so a scenario cannot accidentally edit the checkout,
    with ``--yolo`` so an approval prompt cannot stall a headless run.
    """

    sessions_root = home / "sessions"
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="lo-eval-") as scratch:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(repo)
        env["LOCAL_OPERATOR_CONFIG_DIR"] = str(home)
        # Empty value disables ecosystem skill scanning (``skills/api.py:64-67``).
        # Without it the operator's ~/.agents and ~/.omp skills load into both
        # arms; a skill that mentions research would then supply the behaviour
        # under test and the eval would credit it to the prompt change.
        env["LOCAL_OPERATOR_SKILL_EXTRA_ROOTS"] = ""
        # ``--yolo`` and ``--run-in`` are GLOBAL flags and must precede the
        # subcommand; placed after ``exec`` argparse rejects them and the run
        # exits 2 having done nothing, which reads as "the agent chose not to
        # search" rather than as the harness bug it is.
        cmd = [
            str(repo / ".venv" / "bin" / "python"),
            "-c",
            "import sys; from local_operator.cli import main; sys.exit(main())",
            "--yolo",
            "--run-in",
            scratch,
            "exec",
            scenario.prompt,
        ]
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(repo),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            status = "ok" if proc.returncode == 0 else f"exit {proc.returncode}"
            stderr_tail = proc.stderr[-400:]
        except subprocess.TimeoutExpired:
            status, stderr_tail = "timeout", ""

    # Give the transcript writer a moment to flush before it is read.
    time.sleep(1.0)
    session_dir = _newest_session_dir(sessions_root, started)
    if session_dir is None:
        # Loudly, not silently. "No transcript" and "the agent made no tool
        # calls" are indistinguishable in the output table, and the first is a
        # harness fault that would otherwise be published as a behavioural
        # result. This exact confusion cost a debugging round already.
        status = f"{status}/NO-TRANSCRIPT"
    calls = _tool_calls(session_dir / "transcript.jsonl") if session_dir else []
    used_web = [name for name, _ in calls if name in WEB_TOOLS]
    improvised = _improvised_network(calls)
    return {
        "scenario": scenario.sid,
        "should_search": scenario.should_search,
        "status": status,
        # "researched" is the graded behaviour: did it go to the network at
        # all. "used_web_tool" is the narrower, better-quality path.
        "researched": bool(used_web or improvised),
        "used_web_tool": bool(used_web),
        "web_calls": used_web,
        "improvised_network": improvised,
        "all_calls": [name for name, _ in calls],
        "session": session_dir.name if session_dir else None,
        "stderr_tail": stderr_tail,
    }


def _arm_fingerprint(repo: Path) -> str:
    """Hash of the prompt text an arm will actually send.

    A run takes over an hour, and both arms read their prompt from the working
    tree on every scenario — so editing a prompt file mid-run silently splits
    the results between two different texts and averages them into one number.
    Recording the fingerprint at the start and re-checking it at the end turns
    that from an invisible contamination into a reported failure.
    """

    digest = hashlib.sha256()
    for rel in (
        "local_operator/prompts_md/system.md",
        "local_operator/agent_seeds/coder.md",
        "local_operator/agent_seeds/designer.md",
    ):
        path = repo / rel
        digest.update(path.read_bytes() if path.is_file() else b"")
    return digest.hexdigest()[:12]


def _prepare_home(home: Path, hosting: str, model: str) -> Path:
    """Seed a clean config dir carrying credentials and nothing else.

    Only ``credentials.env`` and the auth database are copied across: the eval
    needs to reach a provider, and must NOT inherit the operator's standing
    instructions, MCP servers, skills or agent registry. Copying the whole home
    would measure this machine's configuration rather than what ships.
    """

    home.mkdir(parents=True, exist_ok=True)
    real = Path.home() / ".local-operator"
    for name in ("credentials.env", "auth.db", "auth.db-shm", "auth.db-wal"):
        source = real / name
        if source.is_file() and not (home / name).exists():
            shutil.copy2(source, home / name)
    # An explicit empty MCP registry: absent, discovery may fall back to a
    # default path and re-introduce the operator's servers.
    mcp = home / "mcp.json"
    if not mcp.exists():
        mcp.write_text(json.dumps({"mcpServers": {}}), encoding="utf-8")
    # Minimal config, built through the real ConfigManager rather than
    # hand-written YAML: the loader requires metadata keys (``created_at``)
    # that are easy to omit, and a partial file fails at session construction
    # with a bare KeyError. Only hosting/model and the web-search settings are
    # set, so the operator's compaction, retry and fallback tuning cannot
    # influence either arm and the settings under test stay at their defaults.
    config = home / "config.yml"
    if not config.exists():
        # Imported from the repo root explicitly rather than relying on the
        # caller's PYTHONPATH: this script is routinely run from a COPIED tree
        # (a pinned snapshot of an arm), where the package is not importable
        # by default and the failure lands after argument parsing.
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from local_operator.config import ConfigManager

        manager = ConfigManager(home)
        manager.update_config(
            {
                "hosting": hosting,
                "model_name": model,
                "auto_save_conversation": False,
                "web_search": {
                    "enabled": True,
                    "providers": ["duckduckgo"],
                    "strategy": "round_robin",
                    "timeout_seconds": 20.0,
                },
            }
        )
    return home


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True, help="worktree of the base ref")
    parser.add_argument("--after", required=True, help="worktree with the change")
    parser.add_argument("--runs", type=int, default=3, help="repeats per scenario")
    parser.add_argument("--timeout", type=int, default=420)
    parser.add_argument("--save", default="", help="write results JSON here")
    parser.add_argument("--only", default="", help="comma-separated scenario ids")
    parser.add_argument(
        "--home",
        default="/tmp/lo-eval-home",
        help="isolated LOCAL_OPERATOR_CONFIG_DIR for both arms",
    )
    parser.add_argument("--hosting", default="anthropic")
    parser.add_argument("--model", default="claude-opus-5")
    args = parser.parse_args()

    home = _prepare_home(Path(args.home), args.hosting, args.model)
    print(f"isolated config dir: {home} (hosting={args.hosting} model={args.model})")

    wanted = {s.strip() for s in args.only.split(",") if s.strip()}
    scenarios = [s for s in ALL_SCENARIOS if not wanted or s.sid in wanted]

    arms = {"before": Path(args.before), "after": Path(args.after)}
    fingerprints = {arm: _arm_fingerprint(repo) for arm, repo in arms.items()}
    for arm, fingerprint in fingerprints.items():
        print(f"  {arm:6} prompt fingerprint {fingerprint}")
    if fingerprints["before"] == fingerprints["after"]:
        print("ERROR: both arms have identical prompt text; nothing to compare.")
        return 2

    results: list[dict[str, Any]] = []
    for arm, repo in arms.items():
        for scenario in scenarios:
            for run_index in range(args.runs):
                record = run_one(repo, scenario, args.timeout, home)
                record["arm"] = arm
                record["run"] = run_index
                results.append(record)
                if record["used_web_tool"]:
                    flag = "WEB"
                elif record["researched"]:
                    flag = "net"
                else:
                    flag = "   "
                print(
                    f"[{arm:6}] {scenario.sid:4} run{run_index} {flag} "
                    f"({record['status']}) calls={len(record['all_calls'])}",
                    flush=True,
                )

    print("\n=== per scenario: researched (of which via web tool) ===")
    print(f"{'id':5} {'expect':7} {'before':>12} {'after':>12}")
    summary: dict[str, Any] = {}
    for scenario in scenarios:
        row: dict[str, Any] = {"should_search": scenario.should_search}
        for arm in arms:
            runs = [r for r in results if r["arm"] == arm and r["scenario"] == scenario.sid]
            hits = sum(1 for r in runs if r["researched"])
            tool_hits = sum(1 for r in runs if r["used_web_tool"])
            row[arm] = f"{hits}/{len(runs)} ({tool_hits}w)" if runs else "-"
        summary[scenario.sid] = row
        expect = "search" if scenario.should_search else "NO"
        print(f"{scenario.sid:5} {expect:7} {row['before']:>12} {row['after']:>12}")

    drifted = {arm: _arm_fingerprint(repo) for arm, repo in arms.items()}
    if drifted != fingerprints:
        changed = [a for a in arms if drifted[a] != fingerprints[a]]
        print(
            f"\nERROR: prompt text changed mid-run in arm(s) {changed}. "
            "These results mix two different prompts and must be discarded."
        )
        return 2

    broken = [r for r in results if "NO-TRANSCRIPT" in r["status"] or r["status"] == "timeout"]
    if broken:
        print(
            f"\nWARNING: {len(broken)} run(s) produced no usable transcript; "
            "those rows measure the harness, not the agent."
        )

    for label, subset in (("SHOULD trigger", SHOULD_TRIGGER), ("SHOULD NOT", SHOULD_NOT_TRIGGER)):
        ids = {s.sid for s in subset} & {s.sid for s in scenarios}
        if not ids:
            continue
        print(f"\n{label}:")
        for arm in arms:
            runs = [r for r in results if r["arm"] == arm and r["scenario"] in ids]
            hits = sum(1 for r in runs if r["researched"])
            tool_hits = sum(1 for r in runs if r["used_web_tool"])
            pct = (100.0 * hits / len(runs)) if runs else 0.0
            tool_pct = (100.0 * tool_hits / len(runs)) if runs else 0.0
            print(
                f"  {arm:6} researched {hits}/{len(runs)} ({pct:.0f}%), "
                f"via web tool {tool_hits}/{len(runs)} ({tool_pct:.0f}%)"
            )

    if args.save:
        Path(args.save).write_text(
            json.dumps({"results": results, "summary": summary}, indent=2),
            encoding="utf-8",
        )
        print(f"\nsaved {args.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
