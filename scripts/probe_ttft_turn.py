"""Instrumented REAL turn: where does the wall clock go between submit and first paint?

WHY THIS EXISTS
===============
``scripts/bench_ttft.py`` stubs the provider so the local cost is measurable. This
script does the opposite: it drives the REAL provider through the REAL local stack
(``create_session`` — the same factory the TUI uses) and stamps every stage, which
is the only way to see the split the operator is complaining about:

    submit ──▶ stream fn entered ──▶ first reasoning delta ──▶ first TEXT delta
              (all local pre-work)   (the user sees NOTHING here)   (what they wait for)

That middle stage is the finding: ``providers/clients.py`` yields
``StreamReasoningDelta`` and the runtime's stream dispatch DROPS it, so a turn that
reasons — 86.5% of this machine's deepseek-flash turns — shows nothing for the whole
reasoning phase. Against real deepseek this script measured that invisible interval
at 455 ms on a 102-token prompt and 1,750 ms on a 142.5k-token cached one, and a
re-run on a clean checkout reproduced it at 461 ms p50 (first reasoning 1167 ms,
first text 1628 ms, 2/2 turns reasoned, load ~190).

MARKS, all monotonic ms from the submit:

* ``prep_ms``            — submit -> the stream fn was entered (all local work first)
* ``first_event_ms``     — first provider event of any kind (``StreamStartEvent``)
* ``first_reasoning_ms`` — first reasoning delta the RUNTIME produced
* ``first_text_ms``      — first text delta at the runtime
* ``visible_text_ms``    — first text delta the FRONT END was handed, via the same
  ``Session.subscribe`` seam the TUI's controller and the headless renderer use.
  The gap between this and ``first_text_ms`` is the runtime→front-end hop.
* ``total_ms``           — the turn finished

``--concurrency`` runs N sessions in one process, all submitting at once, which is
the load-sensitivity arm: local prep grows with it and the provider barely moves.

HOW TO RUN IT (NOT IN CI — IT SPENDS MONEY)
===========================================
    .venv/bin/python scripts/probe_ttft_turn.py --runs 7 --concurrency 1
    .venv/bin/python scripts/probe_ttft_turn.py --runs 2 --concurrency 8
    .venv/bin/python scripts/probe_ttft_turn.py --turns 1 --prompt "Reply with: ok"

It REFUSES to run under GitHub Actions (a paid provider), and pytest does not
collect it — two layers, because a script that quietly joined a pipeline would burn
money per push.

ISOLATION AND CREDENTIALS
=========================
A fresh ``HOME``, ``LOCAL_OPERATOR_CONFIG_DIR`` and ``TMPDIR`` per invocation; the
operator's live sessions are never touched and a measured child can never attach to
one.

The API key is resolved FIRST, in-process, from the operator's own store with
``AuthStore().get_api_key(<provider>, read_only=True)`` — before anything is
isolated, because an isolated home has no store to read. It is then handed to the
isolated run through the provider's OWN environment variable
(``DEEPSEEK_API_KEY``), which is the leg the credential cascade already reads, so:
no key is printed, logged, written to a file, or passed on a command line, and no
copy of the operator's credential database is created.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.ttft.isolate import kill_registered_children, make_run  # noqa: E402


#: Provider → the environment variable its own registry declares for its key. Read
#: from the registry rather than hard-coded, so a renamed variable cannot silently
#: stop resolving and turn this probe into a credential mystery.
def provider_env_key(provider: str) -> str | None:
    from local_operator.providers.registry import get_provider_definition

    definition = get_provider_definition(provider)
    env_keys = getattr(definition, "env_keys", None) if definition else None
    if isinstance(env_keys, str):
        return env_keys
    if isinstance(env_keys, (tuple, list)) and env_keys:
        return str(env_keys[0])
    return None


async def resolve_key(provider: str) -> str:
    """The provider's API key, from the operator's store, read-only and never printed.

    ``async`` because the caller is already inside an event loop: ``asyncio.run``
    here raised ``RuntimeError: asyncio.run() cannot be called from a running event
    loop`` on the first real invocation, and a silently un-awaited coroutine would
    have been the alternative.
    """
    from local_operator.providers.auth_store import AuthStore

    store = AuthStore()
    try:
        key = await store.get_api_key(provider, read_only=True)
    finally:
        store.close()
    if not key:
        raise SystemExit(
            f"no {provider} API key in the credential store; run `lop login {provider}` first"
        )
    return key


class TimedStream:
    """Wraps the session's stream fn and stamps the runtime's own first-of-kind marks.

    This sits BELOW the front end on purpose: it is where a reasoning delta is still
    visible, and it is the only place that can tell "the model thought and said
    nothing" from "the runtime threw away what it said".
    """

    def __init__(self, wrapped: Any, marks: dict[str, Any], base: float) -> None:
        self._wrapped = wrapped
        self._marks = marks
        self._base = base

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

    def __call__(self, request: Any, signal: Any = None) -> Any:
        marks = self._marks
        marks.setdefault("prep_ms", (time.monotonic() - self._base) * 1000)
        marks.setdefault("context_tokens_hint", getattr(request, "context_tokens_hint", None))
        source = self._wrapped(request, signal)

        async def relay() -> Any:
            async for event in source:
                now = (time.monotonic() - self._base) * 1000
                kind = str(getattr(event, "type", ""))
                marks.setdefault("first_event_ms", now)
                marks.setdefault("first_event_type", kind)
                if kind == "reasoning_delta":
                    marks.setdefault("first_reasoning_ms", now)
                    marks["reasoning_events"] = marks.get("reasoning_events", 0) + 1
                elif kind == "text_delta":
                    marks.setdefault("first_text_ms", now)
                    marks["text_events"] = marks.get("text_events", 0) + 1
                elif kind == "tool_call_delta":
                    marks.setdefault("first_tool_ms", now)
                yield event

        return relay()


def front_end_marks(session: Any, marks: dict[str, Any], base: float) -> None:
    """Subscribe the way a front end does, to date what the USER is handed."""
    from local_operator.harness.types import MessageUpdateEvent

    def handler(event: Any) -> None:
        if isinstance(event, MessageUpdateEvent) and getattr(event, "delta", ""):
            marks.setdefault("visible_text_ms", (time.monotonic() - base) * 1000)

    session.subscribe(handler)


def _build_args(hosting: str, model: str) -> Any:
    args = argparse.Namespace()
    for key, value in {
        "hosting": hosting,
        "model": model,
        "agent_name": None,
        "agent_id": None,
        "yolo": True,
        "train": False,
        "resume": None,
        "agent": None,
    }.items():
        setattr(args, key, value)
    return args


async def one_session(
    index: int, run: Any, *, hosting: str, model: str, prompt: str, turns: int
) -> dict[str, Any]:
    """One real session, ``turns`` sequential prompts, stamps on each."""
    from local_operator.agents import AgentRegistry
    from local_operator.config import ConfigManager
    from local_operator.credentials import CredentialManager
    from local_operator.session_factory import create_session, warm_session_imports

    marks: dict[str, Any] = {"session": index, "turns": []}
    session: Any = None
    try:
        await asyncio.to_thread(warm_session_imports)
        build_start = time.monotonic()
        session = await create_session(
            _build_args(hosting, model),
            ConfigManager(run.config_dir),
            CredentialManager(run.config_dir),
            AgentRegistry(run.config_dir),
            cwd=str(run.cwd),
        )
        marks["session_build_ms"] = (time.monotonic() - build_start) * 1000
        inner = cast(Any, session)._stream_fn
        for turn in range(1, turns + 1):
            turn_marks: dict[str, Any] = {}
            base = time.monotonic()
            cast(Any, session)._stream_fn = TimedStream(inner, turn_marks, base)
            front_end_marks(session, turn_marks, base)
            await session.prompt(prompt)
            turn_marks["total_ms"] = (time.monotonic() - base) * 1000
            turn_marks["turn"] = turn
            marks["turns"].append(turn_marks)
    except Exception as exc:  # noqa: BLE001 — a provider error is a result
        marks["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if session is not None:
            try:
                await session.dispose()
            except Exception:  # noqa: BLE001 — teardown is best-effort
                pass
    return marks


SUMMARY_KEYS = (
    "session_build_ms",
    "prep_ms",
    "first_event_ms",
    "first_reasoning_ms",
    "first_text_ms",
    "visible_text_ms",
    "total_ms",
    "context_tokens_hint",
)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=7, help="waves of concurrent sessions")
    parser.add_argument("--concurrency", type=int, default=1, help="sessions submitting at once")
    parser.add_argument("--turns", type=int, default=2, help="prompts per session")
    parser.add_argument("--provider", type=str, default="deepseek")
    parser.add_argument("--model", type=str, default="deepseek-flash")
    # A prompt that INVITES reasoning is the default, deliberately: with a neutral
    # one-liner deepseek-flash answers without thinking (measured 0/1 turns reasoned
    # on "Reply with the single word: ok"), which would leave the very phase this
    # probe exists for unmeasured and read as "reasoning never happens".
    parser.add_argument(
        "--prompt",
        type=str,
        default="Think it through step by step first, then reply with the single word: ok",
    )
    parser.add_argument("--json", type=str, default="", help="write every session here")
    args = parser.parse_args()

    if os.environ.get("GITHUB_ACTIONS"):
        print(
            "refusing to run under GitHub Actions: this probe calls a PAID provider. "
            "See the module docstring for how it is meant to be invoked.",
            file=sys.stderr,
        )
        return 2

    env_key = provider_env_key(args.provider)
    if not env_key:
        raise SystemExit(f"provider {args.provider!r} declares no environment variable for its key")
    # Resolved BEFORE isolation: an isolated home has no credential store to read.
    os.environ[env_key] = await resolve_key(args.provider)

    run = make_run(prefix="lop-ttft-turn-")
    previous_home = os.environ.get("HOME")
    try:
        run.activate()
        run.seed(hosting=args.provider, model=args.model)
        # The environment the sessions read: this process's, isolated, with the
        # provider key present (it is the leg the credential cascade reads).
        os.environ[env_key] = os.environ[env_key]
        print(
            f"probe: provider={args.provider}/{args.model} runs={args.runs} "
            f"concurrency={args.concurrency} turns={args.turns} "
            f"load={os.getloadavg()} platform={platform.platform()}"
        )
        out: list[dict[str, Any]] = []
        try:
            for wave in range(args.runs):
                sessions = await asyncio.gather(
                    *(
                        one_session(
                            wave * args.concurrency + index,
                            run,
                            hosting=args.provider,
                            model=args.model,
                            prompt=args.prompt,
                            turns=args.turns,
                        )
                        for index in range(args.concurrency)
                    )
                )
                for session_marks in sessions:
                    session_marks["wave"] = wave
                    out.append(session_marks)
                    print(json.dumps(session_marks), flush=True)
        finally:
            kill_registered_children(run.config_dir)
    finally:
        run.teardown()
        if previous_home is not None:
            os.environ["HOME"] = previous_home
        shutil.rmtree(run.root, ignore_errors=True)

    turns = [turn for row in out for turn in row.get("turns", [])]
    print(f"\n=== SUMMARY (ms) runs={args.runs} concurrency={args.concurrency} ===")
    print(f"turns measured: {len(turns)}")
    for key in SUMMARY_KEYS:
        values = [turn[key] for turn in turns if isinstance(turn.get(key), (int, float))]
        if not values:
            continue
        print(
            f"{key:20s} n={len(values):3d} min={min(values):8.0f} "
            f"p50={statistics.median(values):8.0f} max={max(values):8.0f}"
        )
    reasoned = [turn for turn in turns if "first_reasoning_ms" in turn]
    gaps = [
        turn["first_text_ms"] - turn["first_reasoning_ms"]
        for turn in reasoned
        if "first_text_ms" in turn
    ]
    print(
        f"turns that reasoned: {len(reasoned)}/{len(turns)}"
        + (f"; invisible gap p50 {statistics.median(gaps):.0f} ms" if gaps else "")
    )
    errors = [row["error"] for row in out if "error" in row]
    if errors:
        print(f"errors: {len(errors)} e.g. {errors[0][:200]}")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
