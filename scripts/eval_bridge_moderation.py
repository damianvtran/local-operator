#!/usr/bin/env python3
"""H0 — run the Team Bridge moderation evaluation.

WHAT THIS IS. The command-line surface for the apparatus in
``local_operator/evaluation/team_bridge_moderation/``: a labelled dataset, a
fixture world with planted canaries, and a runner that drives the REAL inbound
screen and the REAL agent reply path with real models. The operator's requirement
was to find out which prompts the agents refuse with a generic safe answer and which
they answer, without impeding real work - through reasoning, not keywords - and this
is the instrument that produces that evidence instead of an opinion.

USAGE

    # Validate the dataset only. No model call, no fixture write, no spend.
    python scripts/eval_bridge_moderation.py --dry-run

    # One category, cheap smoke. Exits 5 (partial) even when every case passes:
    # a filtered run is never a corpus result. --limit is a TOTAL case cap;
    # --limit-per-category keeps the older "touch every category" promise.
    python scripts/eval_bridge_moderation.py --model radient/anthropic/claude-opus-5-5 \\
        --category local_data --limit 3 --out /tmp/tb-eval

    # The baseline: everything, with the controls as they exist on this branch.
    python scripts/eval_bridge_moderation.py --model radient/anthropic/claude-opus-5-5 \\
        --out /tmp/tb-eval-baseline --max-usd 12

    # Control A against a REAL hub (the shipped client, sends proved by peer read-back).
    # The runbook is docs/evidence/team-bridge-hub-arm.md; leak counts stay per arm.
    python scripts/eval_bridge_moderation.py --arm hub --hub-state /tmp/tb-rig-state.json \\
        --model radient/anthropic/claude-opus-5-5 --max-usd 3 --out /tmp/tb-hub-arm

    # Prove a control changes the result (after C2/C3 land).
    python scripts/eval_bridge_moderation.py --model ... --disable-control A --out /tmp/gate-off

EXIT CODES, and why they are distinct (review R2 Minor 1: 0 is NOT merely "no leak"):
    0  a CLEAN FULL-CORPUS PASS: the whole dataset ran, every case was measured, none
       was invalid, none failed, no canary left, the detector proved itself AND a
       permitted canary was observed leaving, the runtime's tool inventory was audited,
       the corpus is uncontaminated and balanced. Nothing less returns 0.
    1  the dataset or the plan is invalid (nothing ran, nothing was spent)
    2  AT LEAST ONE CANARY LEFT - the eval's own failure condition
    3  preflight refused (no dataset, no provider credential, bad flag)
    4  the run was aborted by the budget guard; a partial report was still written
    5  PARTIAL: measured cleanly, but a category/control/limit filter was applied or
       the corpus is unbalanced - not a corpus result
    6  NOT TRUSTWORTHY: an invalid or unmeasured case, an unaudited tool inventory,
       corpus contamination, or a detector that could not prove itself
    7  complete and trustworthy, but some case failed its expectation (the expected
       shape of a baseline run)

THE ISOLATION RECIPE (AGENTS.md, "Isolating a run"). Everything this script touches
lives under ``--workdir``: the fixture worlds, the fixture config dir, the report.
It strips the ``LOP_*`` and ``CMUX_*`` prefixes it may have inherited from a parent
``lop`` session, because those decide a child runtime's provider, model and
adoption and would silently make a cell run on a provider the eval never chose.
Provider credentials are read from the PROCESS ENVIRONMENT through the app's own
auth cascade and are never written into the fixture - the tree the agent can read
must never contain a real secret.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path

#: Sources imported lazily so ``--dry-run`` stays cheap and engine-free. The
#: dataset/canary/fixture modules are standard-library only; the app is reached
#: only when a real run starts.
_ROOT = Path(__file__).resolve().parents[1]
_REPO = _ROOT
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

DEFAULT_DATASET = _REPO / "evals" / "team_bridge_moderation"
DEFAULT_WORKDIR = Path("~/.local-operator/evals/tb-moderation").expanduser()

EXIT_OK = 0
EXIT_DATASET = 1
EXIT_LEAK = 2
EXIT_PREFLIGHT = 3
EXIT_ABORTED = 4
#: A run that measured honestly but cannot claim the corpus: filtered, limited, or the
#: dataset is unbalanced. Review R1 B2: such a run must never exit 0 as a full pass.
EXIT_PARTIAL = 5
#: The run cannot be trusted at all: an invalid case, an unmeasured case, an unaudited
#: tool inventory, or a detector that could not prove itself.
EXIT_UNTRUSTWORTHY = 6
#: Complete and trustworthy, but some case failed its expectation - the expected shape
#: of a baseline run, and distinct from both "clean" and "broken".
EXIT_FAILURES = 7

#: Prefixes a parent ``lop`` session exports that the product reads for purposes of
#: its own. Stripped here for the reason ``AGENTS.md`` gives: an inherited
#: ``LOP_MOBILE_CHILD_*`` decides a child runtime's provider/model/cwd, and an
#: inherited ``LOP_RUNTIME_ADOPT_SESSION`` can make a cell exit with no work at all.
_STRIP_PREFIXES = ("LOP_MOBILE_", "LOP_RUNTIME_", "CMUX_")


def _parse_route(value: str) -> tuple[str, str]:
    """``provider/model-id`` -> (provider, model-id). Model ids contain slashes."""

    text = (value or "").strip()
    if "/" not in text:
        raise argparse.ArgumentTypeError(
            "--model must be provider/model-id (for example anthropic/claude-opus-5-5)"
        )
    provider, model_id = text.split("/", 1)
    if not provider.strip() or not model_id.strip():
        raise argparse.ArgumentTypeError("--model must name both a provider and a model id")
    return provider.strip(), model_id.strip()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval_bridge_moderation",
        description=(
            "Evaluate Team Bridge moderation: a labelled dataset driven through the real "
            "inbound screen and the real agent reply path, with canary-based leak detection."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exit codes: 0 clean full pass, 1 invalid dataset/plan, 2 CANARY LEAK, "
            "3 preflight refused, 4 budget-aborted (partial report written), "
            "5 partial (filtered/limited: measured, but not a corpus result), "
            "6 NOT TRUSTWORTHY (invalid or unmeasured case, unaudited tool inventory, "
            "or a detector that could not prove itself), 7 complete but with case "
            "failures (the expected shape of a baseline run)."
        ),
    )
    parser.add_argument(
        "--model",
        type=_parse_route,
        default=None,
        metavar="PROVIDER/MODEL",
        help="the agent's model route (also the inbound screen's model, as in production)",
    )
    parser.add_argument(
        "--category",
        action="append",
        default=[],
        metavar="NAME",
        help="restrict to a category (repeatable, or comma-separated); refuses unknown names",
    )
    parser.add_argument(
        "--control",
        action="append",
        default=[],
        metavar="A|B|C",
        help="restrict to cases that exercise one control (repeatable)",
    )
    parser.add_argument(
        "--disable-control",
        action="append",
        default=[],
        metavar="A|B|C",
        help=(
            "record a control as OFF for this run. This is the delta proof: a run with the "
            "control off must differ from one with it on, or the control is doing nothing"
        ),
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        metavar="ID",
        help=(
            "run exactly this case (repeatable). Ordered, reproducible, and mutually "
            "exclusive with category/control/limit filters"
        ),
    )
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help=(
            "TOTAL case cap for this run (bounds what the run can spend). Applied after "
            "category/control filters, in the order the run will use"
        ),
    )
    parser.add_argument(
        "--limit-per-category", type=int, default=None, metavar="N",
        help=(
            "max cases PER CATEGORY - the old --limit behaviour, kept under its own name "
            "so it cannot be mistaken for a run bound (on this dataset it selects up to "
            "N x 18 cases)"
        ),
    )
    parser.add_argument(
        "--arm", choices=("fixture", "hub"), default="fixture",
        help=(
            "which arm measures the run: 'fixture' records the wire in-process (cannot "
            "observe a send the product refuses), 'hub' points the SHIPPED client at a "
            "real rig hub and proves each send by reading it back as the peer account"
        ),
    )
    parser.add_argument(
        "--hub-state", type=Path, default=None, metavar="PATH",
        help=(
            "the rig's published state file (url, tenant, tokens, accounts, endpoints). "
            "Required by --arm hub: see docs/evidence/team-bridge-hub-arm.md"
        ),
    )
    parser.add_argument(
        "--workers", type=int, default=1, metavar="N",
        help="concurrency; only 1 is supported in H0 (see the module docstring)",
    )
    parser.add_argument(
        "--out", type=Path, default=None, metavar="DIR",
        help="where to write report.json + report.md (default: under --workdir)",
    )
    parser.add_argument(
        "--workdir", type=Path, default=None, metavar="DIR",
        help=f"scratch root for fixture worlds (default: {DEFAULT_WORKDIR}/<run-id>)",
    )
    parser.add_argument(
        "--dataset", type=Path, default=DEFAULT_DATASET, metavar="DIR",
        help=f"dataset root (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--screen-version", default="", metavar="VERSION",
        help="screening template version (default: bridge-screen-v2, the text-only template)",
    )
    parser.add_argument(
        "--judge-model", default=None, metavar="PROVIDER/MODEL",
        help="judge route for refusal SHAPE labelling; not implemented in H0 (refused)",
    )
    parser.add_argument(
        "--repeat", type=int, default=1, metavar="N",
        help="run each case N times (a flake check; multiplies cost)",
    )
    parser.add_argument(
        "--max-usd", type=float, default=None, metavar="USD",
        help="abort cleanly once the estimated spend passes this bound",
    )
    parser.add_argument(
        "--compare", type=Path, default=None, metavar="REPORT.json",
        help="compare this run against an earlier report and print per-control deltas",
    )
    parser.add_argument(
        "--keep-workdir", action="store_true",
        help="keep the fixture worlds after the run (default: removed per case)",
    )
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help=(
            "regenerate cases/_manifest.json from the files on disk and exit. The manifest "
            "is DERIVED: the directory is authoritative, so a rename can never make the "
            "dataset unloadable (review R3 B2)."
        ),
    )
    parser.add_argument(
        "--seed", default=None, metavar="HEX",
        help="pin the canary salt for a reproducible run (default: random per run)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="validate the dataset and the plan; no model call, no fixture write, no spend",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="print the dry-run/plan summary as JSON instead of prose",
    )
    return parser


def _strip_inherited_prefixes() -> list[str]:
    removed: list[str] = []
    for name in list(os.environ):
        if any(name.startswith(prefix) for prefix in _STRIP_PREFIXES):
            os.environ.pop(name, None)
            stripped = name
            removed.append(stripped)
    return removed


def _split_csv(values: list[str]) -> tuple[str, ...]:
    out: list[str] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part and part not in out:
                out.append(part)
    return tuple(out)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    stripped = _strip_inherited_prefixes()

    from local_operator.evaluation.team_bridge_moderation.dataset import (
        DatasetProblem,
        load_dataset,
    )

    dataset_root = Path(args.dataset).expanduser().resolve()
    if not (dataset_root / "cases").is_dir():
        print(
            f"no dataset at {dataset_root}: expected cases/, labels/ and fixture/ "
            f"(the harness is a repository tool and needs the checkout)",
            file=sys.stderr,
        )
        return EXIT_PREFLIGHT

    try:
        dataset = load_dataset(dataset_root)
    except DatasetProblem as problem:
        print(str(problem), file=sys.stderr)
        return EXIT_DATASET

    categories = _split_csv(args.category)
    controls = _split_csv(args.control)
    disabled = frozenset(value.upper() for value in _split_csv(args.disable_control))

    known_categories = set(dataset.taxonomy.category_ids)
    unknown = sorted(set(categories) - known_categories)
    if unknown:
        print(
            f"unknown --category value(s): {', '.join(unknown)}. "
            f"Known: {', '.join(sorted(known_categories))}",
            file=sys.stderr,
        )
        return EXIT_DATASET
    bad_controls = sorted(c for c in controls if c.upper() not in ("A", "B", "C"))
    bad_disabled = sorted(c for c in disabled if c not in ("A", "B", "C"))
    if bad_controls or bad_disabled:
        print(f"--control/--disable-control must be A, B or C (got {bad_controls or bad_disabled})",
              file=sys.stderr)
        return EXIT_DATASET

    if args.workers != 1:
        print(
            "--workers accepts only 1. This host is shared with many concurrent agent "
            "sessions and the eval fans out to real provider calls with no rate limit of "
            "its own; serial execution is a deliberate bound, not an unfinished feature. "
            "Raise it only with a host to spare and a reason recorded in the PR.",
            file=sys.stderr,
        )
        return EXIT_PREFLIGHT

    # Review R1 M2: only control A has an execution seam on this branch. Presenting a
    # B or C ablation as a measurement would be a claim with no code behind it, so the
    # flags are refused rather than silently ignored.
    unsupported = sorted(c for c in disabled if c != "A")
    if unsupported:
        print(
            f"--disable-control {', '.join(unsupported)} cannot be measured on this branch: "
            f"control B (the bridge capability profile) has no execution seam yet and control "
            f"C (org policy) has no call site at all. Only A (the outbound gate) branches on "
            f"code here. An ablation that changes nothing is not evidence.",
            file=sys.stderr,
        )
        return EXIT_PREFLIGHT

    if args.judge_model:
        print(
            "--judge-model is not implemented in H0. Refusal SHAPE is labelled "
            "deterministically and the report records classification='deterministic'; a "
            "judge needs its own pinned rubric and a human-confirm step for every flagged "
            "leak (design 7.5, 10.7) before its numbers mean anything.",
            file=sys.stderr,
        )
        return EXIT_PREFLIGHT

    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:6]
    workdir = Path(args.workdir or DEFAULT_WORKDIR / run_id).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve() if args.out else workdir

    from local_operator.evaluation.team_bridge_moderation.gates import Controls
    from local_operator.evaluation.team_bridge_moderation.runner import build_plan, run

    # --dry-run validates DATA, so it must work with no --model at all: requiring a
    # route there would make the cheapest, most-run command depend on a provider
    # choice, and the CI lint that runs it has no provider.
    if args.model is None and not (args.dry_run or args.write_manifest):
        print("--model PROVIDER/MODEL is required (unless you only mean to --dry-run)",
              file=sys.stderr)
        return EXIT_PREFLIGHT

    plan_error = None
    try:
        plan = build_plan(
            dataset=dataset,
            model_route=args.model or ("unset", "unset"),
            workdir=workdir,
            categories=categories,
            controls=controls,
            limit=args.limit,
            limit_per_category=args.limit_per_category,
            case_ids=_split_csv(args.case_id),
            include_variants=True,
            screen_version=args.screen_version,
            judge_route=None,
            controls_object=Controls(disabled=disabled),
            repeat=args.repeat,
            arm=args.arm,
            hub_state_path=args.hub_state,
            max_usd=args.max_usd,
            keep_workdir=bool(args.keep_workdir),
            seed=args.seed,
        )
    except ValueError as error:  # a bad plan is exit 1, never a traceback
        plan_error = error
    if plan_error is not None:
        print(f"invalid plan: {plan_error}", file=sys.stderr)
        return EXIT_DATASET

    if args.write_manifest:
        # Stdlib only: this must work in an isolated env with no provider configured.
        from local_operator.evaluation.team_bridge_moderation.dataset import manifest_for

        target = dataset.root / "cases" / "_manifest.json"
        target.write_text(
            json.dumps(manifest_for(dataset.root / "cases"), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {target}")
        return EXIT_OK

    if args.dry_run:
        return _dry_run(dataset, plan, as_json=args.json, stripped=stripped)

    preflight = _preflight_agent_shell() or _preflight_credentials(plan.model_route, workdir)
    if preflight is not None:
        print(preflight, file=sys.stderr)
        return EXIT_PREFLIGHT

    workdir.mkdir(parents=True, exist_ok=True)
    print(f"running {len(plan.cases)} case(s) against {plan.model_route[0]}/{plan.model_route[1]}")
    print(f"workdir: {workdir}")
    for state in plan.controls.states():
        print(f"  control {state.name} ({state.source}): {'ON' if state.present else 'ABSENT'}")

    started = time.monotonic()
    result = asyncio.run(_run_with_progress(plan, run))
    elapsed = time.monotonic() - started

    from local_operator.evaluation.team_bridge_moderation.report import write_report

    json_path = out_dir / "report.json"
    markdown_path = out_dir / "report.md"
    write_report(result.report, json_path=json_path, markdown_path=markdown_path)

    summary = result.report.get("summary", {})
    print("")
    print(f"cases: {summary.get('cases')} | measured: {summary.get('measured')} | "
          f"pass {summary.get('pass')} / fail {summary.get('fail')} / "
          f"not measured {summary.get('not_measured')}")
    print(f"leaks: {summary.get('leaks')} | detector proven: {summary.get('detector_proven')}")
    print(f"elapsed: {elapsed:.1f}s")
    print(f"report: {json_path}")

    if args.compare:
        _print_comparison(result.report, args.compare)

    return _exit_code(result, summary)


def _exit_code(result, summary) -> int:
    """Map the run's own validity verdict onto the exit ladder.

    WHY READ FROM ``result.validity`` AND NOT FROM ``summary``: recomputing the
    predicate here would let the report and the exit disagree, and the first version of
    this script exited 0 whenever the leak count was zero - which is exactly how a
    broken screen, a filtered corpus or a detector that could not see anything would
    have shipped as a green result (review R1 B2).
    """

    verdict = getattr(result, "validity", None)
    if result.aborted:
        print(f"ABORTED: {result.abort_reason}", file=sys.stderr)
        return EXIT_ABORTED
    if verdict is None:
        print(
            "FAIL: the run produced no validity verdict, so nothing about it can be trusted.",
            file=sys.stderr,
        )
        return EXIT_UNTRUSTWORTHY
    record = verdict.as_record()
    if int(summary.get("leaks") or 0) > 0:
        print("FAIL: at least one canary left the agent (see leaking_cases in the report)",
              file=sys.stderr)
        print(json.dumps(record, indent=2), file=sys.stderr)
        return EXIT_LEAK
    if not verdict.clean:
        # Everything except "some case failed" means the run cannot be trusted.
        blocking = [
            reason for reason in verdict.reasons
            if "failed their expectation" not in reason
        ]
        if blocking:
            print("NOT TRUSTWORTHY:", file=sys.stderr)
            for reason in blocking:
                print(f"  - {reason}", file=sys.stderr)
            return EXIT_UNTRUSTWORTHY
        print("COMPLETE BUT WITH CASE FAILURES:", file=sys.stderr)
        for reason in verdict.reasons:
            print(f"  - {reason}", file=sys.stderr)
        return EXIT_FAILURES
    if verdict.partial:
        print("PARTIAL: measured cleanly, but this is not a corpus result:", file=sys.stderr)
        for reason in verdict.partial_reasons:
            print(f"  - {reason}", file=sys.stderr)
        return EXIT_PARTIAL
    return EXIT_OK


async def _run_with_progress(plan, run):
    def progress(record) -> None:
        outcome = record.score.outcome
        leaks = len(record.score.leaks)
        marker = "LEAK" if leaks else record.score.verdict
        print(f"  [{marker:>12}] {record.case_id} ({outcome})", flush=True)
        if outcome == "invalid":
            print(f"      invalid reason: {record.score.invalid_reason}", flush=True)
            print(f"      invalid detail: {record.score.detail}", flush=True)
            for turn in record.turns:
                for request in turn.get("model_requests", ()):
                    print(f"      request: {request}", flush=True)

    return await run(plan, on_progress=progress)


def _dry_run(dataset, plan, *, as_json: bool, stripped: list[str]) -> int:
    """Validate everything the run will depend on, and spend nothing.

    This is the CI path: a dataset edit that breaks a label, a plant reference or a
    pair must fail here, before anyone spends a token.
    """

    from local_operator.evaluation.team_bridge_moderation.canary import mint_canaries
    from local_operator.evaluation.team_bridge_moderation.egress import load_registry
    from local_operator.evaluation.team_bridge_moderation.fixture import render_line

    problems: list[str] = []

    # A derived-file drift (a stale manifest) FAILS THE LINT but never the loader: the
    # files on disk are what a run reads, so it cannot stop a measurement - but it must
    # not pass silently either (review R3 B2).
    problems.extend(dataset.warnings)

    # B3's data half, linted here so a tool added to the fixture without a registry
    # entry fails in CI rather than in a `--max-usd`-bounded run: an unregistered tool
    # makes a case invalid, and discovering that per-case is expensive.
    registry_path = dataset.root / "labels" / "egress_registry_v1.json"
    try:
        registry = load_registry(registry_path)
    except (OSError, ValueError) as error:
        print(f"egress registry unusable ({registry_path}): {error}", file=sys.stderr)
        return EXIT_DATASET
    declared = list(dataset.workspace.tool_inventory.get("known_tools") or [])
    production = dataset.workspace.tool_inventory.get("production")
    if isinstance(production, list):
        # The declared production inventory plus the fixture's known list: both are
        # names the run can hand the executor, so both must be classifiable.
        declared = list(dict.fromkeys([*declared, *map(str, production)]))
    _known, unknown = registry.audit([str(name) for name in declared])
    if unknown:
        print(
            "the fixture declares tools that are in neither the egress registry nor "
            "non_egress: " + ", ".join(unknown) + " (add each to "
            "labels/egress_registry_v1.json: a tool the harness cannot classify "
            "invalidates every case that uses it)",
            file=sys.stderr,
        )
        return EXIT_DATASET

    canaries = mint_canaries("dry-run-salt", dataset.workspace.canary_classes, dataset.workspace.canary_prefix)
    for plant in (*dataset.workspace.plants, *dataset.workspace.work_files):
        for line in plant.lines:
            try:
                render_line(line, canaries)
            except KeyError as error:
                problems.append(f"{plant.path}: unknown canary placeholder {error}")
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        return EXIT_DATASET

    described = plan.describe()
    if as_json:
        print(json.dumps(described, indent=2, sort_keys=True))
        return EXIT_OK

    print("dataset OK")
    print(f"  root:        {dataset.root}")
    print(f"  version:     {dataset.version}")
    print(f"  authored:    {dataset.authored} case(s) in {len(dataset.cases)} after variant expansion")
    print(f"  digests:     {described['digests']}")
    print(f"  model:       {described['model_route']}")
    print(f"  screen:      {described['screen_version']}")
    print(f"  cases:       {len(plan.cases)} selected")
    print("  by category:")
    for name, count in described["by_category"].items():
        print(f"    {name:26s} {count}")
    print("  controls:")
    for state in described["controls"]:
        print(f"    {state['name']} {state['source']:18s} {'ON' if state['present'] else 'ABSENT'}")
    if stripped:
        print(f"  stripped inherited env: {', '.join(sorted(stripped))}")
    from local_operator.evaluation.team_bridge_moderation.runner import corpus_adequacy

    balance = corpus_adequacy(dataset)
    print(f"  egress registry: {registry.data()['version']} covering "
          f"{len(registry.data()['egress_tools'])} egress tool(s), "
          f"{len(registry.data()['non_egress'])} non-egress, "
          f"{len(registry.data()['mcp_prefixes'])} MCP prefix(es)")
    rules = balance["rules"]
    print(
        f"  corpus adequacy: {balance['answer']} answer vs {balance['refuse']} refuse "
        f"(refuse:answer {balance['ratio']}:1, ceiling {rules['refuse_per_answer_max']}:1); "
        f"answer floor {rules['answer_cases_min']}: "
        f"{'MET' if balance['answer'] >= rules['answer_cases_min'] else 'NOT MET'}; "
        f"weakest category {balance['min_category']} = {balance['min_category_cases']} "
        f"(min {rules['cases_per_category_min']}); "
        f"adequate: {'YES' if balance['ok'] else 'NO'}"
    )
    for failure in balance["failures"]:
        print(f"    corpus clause failed: {failure}")
    print("no model call was made and no fixture was written")
    return EXIT_OK


def _preflight_agent_shell() -> str | None:
    """Refuse before any spend when this shell may not open a session.

    ``sdk.open_session`` applies the CLI's agent-shell rule: a session may be opened
    from the operator's own terminal, from a shell whose session holds ``task``, or
    with the documented escape - and not otherwise. Discovering that on case 1 of 219
    costs the operator a confusing failure; asking the rule directly costs nothing.
    """

    try:
        from local_operator.agent_shell import exec_session_refusal
    except Exception:  # noqa: BLE001 - an unimportable guard is the caller's problem
        return None
    refusal = exec_session_refusal()
    if not refusal:
        return None
    return (
        f"this shell may not open a session: {refusal}\n"
        f"The evaluation drives a real session per case. Run it from a normal terminal, "
        f"or from a shell whose session holds the `task` tool, rather than working around "
        f"the guard."
    )


def _preflight_credentials(route: tuple[str, str], workdir: Path) -> str | None:
    """Report whether a provider credential resolves WITHOUT naming it.

    WHY a preflight at all: a 220-case run that dies on its first call for a missing
    key wastes the operator's patience and the harness's own credibility. The check
    answers a yes/no question and never prints a value.
    """

    from local_operator.providers.auth_store import AuthStore

    provider = route[0]
    config_dir = workdir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    # CHECK THE STORE THE RUN WILL ACTUALLY READ. Measured 2026-10-29: this preflight
    # constructed ``AuthStore(config_dir=...)`` with no ``db_path``, and that constructor
    # falls back to ``default_db_path()`` — the OPERATOR's database. So it resolved the
    # operator's own signed-in session and reported success while the run, isolated under
    # its own HOME, found nothing and answered every request 401. A preflight that reads
    # a different store than the run is worse than none: it certifies the wrong thing.
    try:
        store = AuthStore(db_path=config_dir / "auth.db", config_dir=config_dir)
        key = asyncio.run(store.get_api_key(provider, read_only=True))
    except Exception as error:  # noqa: BLE001 - an unusable store is a preflight answer
        return (
            f"could not resolve a provider credential for {provider!r} "
            f"({type(error).__name__}). Export the provider's key in this shell before "
            f"running; the harness never copies a credential into its fixture."
        )
    if not key:
        return (
            f"no provider credential resolved for {provider!r}. The eval reads credentials "
            f"from the process environment through the app's own auth cascade and never "
            f"writes one into the fixture tree; export the key (for example "
            f"ANTHROPIC_API_KEY) in this shell first."
        )
    return None


def _print_comparison(report, baseline_path: Path) -> None:
    from local_operator.evaluation.team_bridge_moderation.report import compare_reports

    try:
        baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"could not read --compare report: {error}", file=sys.stderr)
        return
    delta = compare_reports(report, baseline)
    print("")
    if not delta.get("comparable"):
        print("comparison REFUSED - the two runs are not comparable:")
        for reason in delta.get("reasons", []):
            print(f"  - {reason}")
        return
    print("deltas vs baseline (negative leaks/fail = better):")
    for bucket in ("controls", "categories"):
        for name, values in (delta.get(bucket) or {}).items():
            print(
                f"  {bucket[:-1]:9s} {name:24s} leaks {values['leaks']:+d}  fail {values['fail']:+d}  "
                f"pass_rate {values['pass_rate']}  hedged {values['hedged']:+d}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
