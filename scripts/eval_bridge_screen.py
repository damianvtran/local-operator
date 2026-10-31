#!/usr/bin/env python3
"""Screen-only bench for the Team Bridge inbound screener (bypass / false-block rates).

WHAT THIS IS. The command-line surface for
``local_operator/evaluation/team_bridge_moderation/screenbench.py``: every corpus
message is screened N times per (model, template) through the PRODUCTION
``ScreenModelRunner`` and ``parse_verdict``, with no agent turn, and the result is
reported as a classifier: bypass rate (adversarial delivered) and false-block rate
(legitimate not delivered), with cluster-bootstrap CIs. The pre-registered protocol
is ``evals/team_bridge_moderation/screen/PROTOCOL.md``.

USAGE

    # Plan only: corpus and template digests, call count. No model call.
    python scripts/eval_bridge_screen.py --corpus adv.jsonl --corpus legit.jsonl \\
        --template bridge-screen-v2 --template cand.txt \\
        --model anthropic/claude-opus-5-5 --repeat 5 --split heldout --dry-run

    # Run (writes OUT/trials.jsonl incrementally, then OUT/summary.{json,md}).
    python scripts/eval_bridge_screen.py ... --max-calls 4000 --out /tmp/tb-screen

    # Paired comparison over an existing run's trials (no model call).
    python scripts/eval_bridge_screen.py --trials /tmp/tb-screen/trials.jsonl \\
        --compare bridge-screen-v2 cand --out /tmp/tb-screen

WHAT IT TOUCHES. Never ``~/.local-operator/team_bridge`` (no store, budget or
receipts). Provider credentials are read from the operator's config dir through
the same plain ``AuthStore`` the production screening connector builds
(``team_bridge/connector/__main__.py``): ``auth.db`` under the config root, with
that store's ordinary open side effects (schema check, WAL, an OAuth refresh
persisting a rotated token). Resolves are ``read_only=True``, so no credential is
blocked and no session stickiness moves. Each call is
recorded in the shared analytics ledger under the session id ``tb-screen-bench``.
Everything else is written under ``--out``.

EXIT CODES
    0  the run (or comparison) completed; every planned trial ran and none is invalid
    1  the plan is invalid (bad corpus, refused template, bad flags); nothing ran
    3  preflight refused (no provider credential for a model)
    5  PARTIAL: --max-calls stopped some trials, or some trials are invalid
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

EXIT_OK = 0
EXIT_PLAN = 1
EXIT_PREFLIGHT = 3
EXIT_PARTIAL = 5

#: Same reason as ``eval_bridge_moderation.py``: an inherited parent-session variable
#: can decide a child runtime's provider or model behind the bench's back.
_STRIP_PREFIXES = ("LOP_MOBILE_", "LOP_RUNTIME_", "CMUX_")
#: The analytics session id every bench call is attributed to (see module docstring).
BENCH_SESSION_ID = "tb-screen-bench"
#: The fixture world's organisation, so screens see the same ``{{organization}}`` the
#: agent-turn harness renders; a corpus line may override it per message.
DEFAULT_ORGANIZATION = "Minerva"


def _parse_route(value: str) -> tuple[str, str]:
    text = (value or "").strip()
    if "/" not in text:
        raise argparse.ArgumentTypeError("--model must be provider/model-id")
    provider, model_id = text.split("/", 1)
    if not provider.strip() or not model_id.strip():
        raise argparse.ArgumentTypeError("--model must name both a provider and a model id")
    return provider.strip(), model_id.strip()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Screen-only Team Bridge bench: bypass and false-block rates per model."
    )
    parser.add_argument("--corpus", action="append", type=Path, default=[], help="JSONL corpus")
    parser.add_argument(
        "--template",
        action="append",
        default=[],
        help="an embedded version (bridge-screen-v2) or a candidate template file; repeatable",
    )
    parser.add_argument(
        "--model", action="append", type=_parse_route, default=[], help="provider/model-id"
    )
    parser.add_argument(
        "--repeat", type=int, default=5, help="trials per (message, model, template)"
    )
    parser.add_argument("--split", choices=("dev", "heldout", "all"), default="all")
    parser.add_argument("--workers", type=int, default=4, help="concurrent screens")
    parser.add_argument("--out", type=Path, help="output directory")
    parser.add_argument(
        "--max-calls", type=int, help="hard cap on physical screen calls, retries included"
    )
    parser.add_argument("--organization", default=DEFAULT_ORGANIZATION)
    parser.add_argument("--dry-run", action="store_true", help="print the plan; make no call")
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("A", "B"),
        help="paired comparison of template B against baseline A (label or sha prefix)",
    )
    parser.add_argument(
        "--trials",
        action="append",
        type=Path,
        default=[],
        help="existing trials.jsonl to summarize/compare instead of running",
    )
    return parser


def _strip_inherited_prefixes() -> list[str]:
    removed = [name for name in os.environ if name.startswith(_STRIP_PREFIXES)]
    for name in removed:
        os.environ.pop(name, None)
    return removed


def _write(out: Path, stem: str, payload: object, markdown: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{stem}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (out / f"{stem}.md").write_text(markdown, encoding="utf-8")


def _preflight(models, auth) -> str | None:
    """Say which model has no credential, without naming any value."""

    for provider, model_id in models:
        try:
            key = asyncio.run(auth.get_api_key(provider, read_only=True, model_id=model_id))
        except Exception as error:  # noqa: BLE001 - an unusable store is a preflight answer
            return f"credential lookup failed for {provider!r} ({type(error).__name__})"
        if not key:
            return f"no provider credential resolved for {provider!r}"
    return None


def _open_auth_store(root: Path | None = None):
    """The operator's inference credential store, built as production builds it.

    WHY NOT ``AuthStore.open_read_only()``. That constructor is METADATA-ONLY (the
    paired-phone seam): it never assigns the writable connection, and the key
    cascade (``get_api_key`` -> ``_usable_key_rows`` -> ``is_blocked``) reads
    ``self._conn``, so every real run died in preflight with ``AttributeError``.
    The screening connector opens a plain store with an explicit ``db_path`` (the
    ``config_dir`` argument alone only feeds the env-override tier), and the bench
    measures that connector's screen, so it opens the same thing. Nothing here
    names ``team_bridge/``; an audit-hook run of a real call confirmed no path
    under it is opened.
    """

    from local_operator.paths import config_dir
    from local_operator.providers.auth_store import AuthStore

    root = Path(root) if root is not None else config_dir()
    return AuthStore(db_path=root / "auth.db", config_dir=root)


def _leg_factory(auth, models, organization: str):
    """One shared production stream function per model; a fresh ScreenLeg per attempt."""

    from local_operator.evaluation.team_bridge_moderation.gates import ScreenLeg
    from local_operator.model.configure import create_stream_fn
    from local_operator.paths import config_dir

    streams = {
        model: create_stream_fn(auth, settings=None, session_id=BENCH_SESSION_ID)
        for model in models
    }

    def factory(model, template):
        return ScreenLeg(
            model_route=model,
            config_dir=config_dir(),
            screen_version=template.label,
            template_text=template.text,
            stream_fn=streams[model],
        )

    return factory, streams


async def _close_streams(streams) -> None:
    for stream in streams.values():
        closer = getattr(stream, "close", None)
        if closer is not None:
            result = closer()
            if hasattr(result, "__await__"):
                await result


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    stripped = _strip_inherited_prefixes()

    from local_operator.evaluation.team_bridge_moderation import screenbench as sb

    if args.trials:
        trials = sb.load_trials(args.trials)
        if args.out is None:
            print("--out is required", file=sys.stderr)
            return EXIT_PLAN
        return _report(sb, trials, args)

    try:
        plan = sb.build_plan(
            corpus=args.corpus,
            templates=args.template,
            models=args.model,
            repeat=args.repeat,
            split=args.split,
            workers=args.workers,
            max_calls=args.max_calls,
            organization=args.organization,
        )
    except sb.PlanError as error:
        print(f"plan refused: {error}", file=sys.stderr)
        return EXIT_PLAN

    described = sb.describe_plan(plan)
    if args.dry_run:
        if stripped:
            print(f"stripped inherited env: {', '.join(sorted(stripped))}")
        print(json.dumps(described, indent=2))
        print(
            f"dry run: {plan.trial_count} trial(s), up to {plan.worst_case_calls} call(s) "
            f"with retries; no model call made"
        )
        return EXIT_OK
    if args.out is None:
        print("--out is required for a run", file=sys.stderr)
        return EXIT_PLAN
    if args.max_calls is None:
        print("--max-calls is required for a run (a hard spend cap)", file=sys.stderr)
        return EXIT_PLAN

    auth = _open_auth_store()
    try:
        refusal = _preflight(plan.models, auth)
        if refusal:
            print(f"preflight refused: {refusal}", file=sys.stderr)
            return EXIT_PREFLIGHT
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "plan.json").write_text(json.dumps(described, indent=2) + "\n", "utf-8")
        factory, streams = _leg_factory(auth, plan.models, plan.organization)

        def progress(done: int, total: int) -> None:
            if done == total or done % 25 == 0:
                print(f"  {done}/{total} trials", flush=True)

        async def run():
            try:
                return await sb.run_plan(
                    plan, out_dir=args.out, leg_factory=factory, progress=progress
                )
            finally:
                await _close_streams(streams)

        trials = asyncio.run(run())
    finally:
        auth.close()
    return _report(sb, trials, args)


def _report(sb, trials, args) -> int:
    summary = sb.summarize(trials)
    _write(args.out, "summary", summary, sb.markdown_summary(summary))
    print(sb.markdown_summary(summary))
    if args.compare:
        try:
            result = sb.compare(trials, args.compare[0], args.compare[1])
        except sb.PlanError as error:
            print(f"compare refused: {error}", file=sys.stderr)
            return EXIT_PLAN
        _write(args.out, "compare", result, sb.markdown_compare(result))
        print(sb.markdown_compare(result))
    partial = any(t.status != "valid" for t in trials)
    return EXIT_PARTIAL if partial else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
