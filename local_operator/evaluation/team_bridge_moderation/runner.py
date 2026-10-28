"""The runner: plan, execute one case at a time, score, report, exit.

WHY ``--workers 1`` IS THE ONLY SUPPORTED VALUE (and why the flag exists anyway).
This machine is shared with roughly twenty-five concurrent agent sessions, and a
run fans out to real provider calls with no rate limit this harness controls. The
flag exists so a future host with headroom can raise it deliberately; today a value
above one is refused with that reason, rather than silently serialising and letting
a reader believe a run was parallel.

THE RUN ORDER IS DELIBERATE: cases are grouped by category and interleaved by
control, so an aborted run has exercised a spread of classes rather than the whole
of the first file. A budget guard checks spend before each case and aborts cleanly
with a partial report - a truncated run with honest numbers beats a complete run
nobody could afford.

WHAT THIS MODULE DOES NOT DO: it never writes into the repository, never touches
the operator's config or home, and never leaves a fixture world behind unless
``--keep-workdir`` says so.
"""

from __future__ import annotations

import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .agent import AgentLeg
from .dataset import Dataset, ModerationCase
from .gates import Controls, ScreenLeg, Usage
from .report import CaseRecord, ReportHeader, build_report
from .scoring import score_case

__all__ = [
    "RunPlan",
    "RunResult",
    "build_plan",
    "run",
]

#: What the report says this apparatus cannot measure. Stated ONCE here and copied
#: into every report header, because a limitation that lives only in a docstring is
#: a limitation a reviewer never sees.
LIMITATIONS: tuple[str, ...] = (
    "the wire is recorded, not reached: a RecordingClient stands in for the hub, so "
    "availability and server-side refusal are not measured, and 'the message left' means "
    "this harness recorded it leaving",
    "the bridge runtime is installed admitted, because a fixture has no hub to settle "
    "admission; an unadmitted run would refuse every bridge op and report a false clean",
    "no capture tool is in the inventory: the screenshot and clipboard vectors are measured "
    "through the fixture's stand-in files and through whether an out-of-profile tool was "
    "attempted, never by taking a real screenshot of this machine",
    "control A (the outbound gate) and control B (the bridge capability profile) do not exist "
    "on this branch; their absence is recorded per run and is what a later delta run proves",
    "refusal SHAPE is labelled deterministically unless --judge-model is given; the label is "
    "a measurement and never a gate",
)


@dataclass(frozen=True)
class RunPlan:
    """What a run will do, decided before anything expensive happens."""

    dataset: Dataset
    cases: tuple[ModerationCase, ...]
    model_route: tuple[str, str]
    screen_version: str
    judge_route: tuple[str, str] | None
    workdir: Path
    keep_workdir: bool
    controls: Controls
    repeat: int
    max_usd: float | None

    def describe(self) -> Mapping[str, Any]:
        by_category: dict[str, int] = {}
        for case in self.cases:
            by_category[case.category] = by_category.get(case.category, 0) + 1
        return {
            "model_route": f"{self.model_route[0]}/{self.model_route[1]}",
            "screen_version": self.screen_version,
            "judge_route": (f"{self.judge_route[0]}/{self.judge_route[1]}" if self.judge_route else ""),
            "cases": len(self.cases),
            "authored": self.dataset.authored,
            "by_category": dict(sorted(by_category.items())),
            "repeat": self.repeat,
            "workdir": str(self.workdir),
            "controls": [dict(state.as_record()) for state in self.controls.states()],
            "dataset_version": self.dataset.version,
            "digests": dict(self.dataset.digests),
        }


@dataclass
class RunResult:
    """The completed (or aborted) run."""

    report: Mapping[str, Any]
    records: tuple[CaseRecord, ...]
    aborted: bool = False
    abort_reason: str = ""


def build_plan(
    *,
    dataset: Dataset,
    model_route: tuple[str, str],
    workdir: Path,
    categories: Sequence[str] = (),
    controls: Sequence[str] = (),
    limit: int | None = None,
    include_variants: bool = True,
    screen_version: str = "",
    judge_route: tuple[str, str] | None = None,
    controls_object: Controls | None = None,
    repeat: int = 1,
    max_usd: float | None = None,
    keep_workdir: bool = False,
) -> RunPlan:
    """Resolve the filter set into a concrete plan BEFORE any spend.

    ``--limit`` is per category (see ``Dataset.select``), so a cheap smoke run still
    covers every category rather than the alphabetically first one.
    """

    from .gates import DEFAULT_SCREEN_VERSION

    chosen = dataset.select(
        categories=categories,
        controls=controls,
        limit=limit,
        include_variants=include_variants,
    )
    if not chosen:
        raise ValueError("the filter selected no cases; nothing would be measured")
    return RunPlan(
        dataset=dataset,
        cases=chosen,
        model_route=model_route,
        screen_version=screen_version or DEFAULT_SCREEN_VERSION,
        judge_route=judge_route,
        workdir=Path(workdir),
        keep_workdir=bool(keep_workdir),
        controls=controls_object or Controls(),
        repeat=max(1, int(repeat)),
        max_usd=max_usd,
    )


async def run(plan: RunPlan, *, on_progress: Any | None = None) -> RunResult:
    """Execute the plan and return the report. One case at a time, always."""

    settings = _settings_for(plan.workdir)
    screen = ScreenLeg(
        model_route=plan.model_route,
        config_dir=plan.workdir / "config",
        settings=settings,
        screen_version=plan.screen_version,
    )
    records: list[CaseRecord] = []
    latencies: list[float] = []
    input_tokens = 0
    output_tokens = 0
    started = time.time()
    aborted = False
    abort_reason = ""

    run_salt = secrets.token_hex(8)
    for repeat_index in range(plan.repeat):
        for case in plan.cases:
            if plan.max_usd is not None and _estimated_usd(input_tokens, output_tokens) > plan.max_usd:
                aborted = True
                abort_reason = (
                    f"budget guard: estimated spend "
                    f"${_estimated_usd(input_tokens, output_tokens):.2f} exceeded --max-usd "
                    f"${plan.max_usd:.2f}; stopped before case {case.id}"
                )
                break
            record, usage = await _run_one(
                case=case,
                plan=plan,
                screen=screen,
                run_salt=run_salt,
                repeat_index=repeat_index,
            )
            records.append(record)
            latencies.append(sum(turn.get("latency_s", 0.0) for turn in record.turns) or 0.0)
            input_tokens += usage.input_tokens
            output_tokens += usage.output_tokens
            if on_progress is not None:
                on_progress(record)
        if aborted:
            break

    header = ReportHeader(
        model_route=f"{plan.model_route[0]}/{plan.model_route[1]}",
        screen_version=plan.screen_version,
        judge_route=(f"{plan.judge_route[0]}/{plan.judge_route[1]}" if plan.judge_route else ""),
        repo_head=_repo_head(),
        branch=_git_branch(),
        dataset_version=plan.dataset.version,
        dataset_digest=str(plan.dataset.digests.get("cases_sha256", "")),
        taxonomy_version=plan.dataset.taxonomy.version,
        workspace_digest=str(plan.dataset.digests.get("workspace_sha256", "")),
        controls=tuple(state.as_record() for state in plan.controls.states()),
        canary_classes=plan.dataset.workspace.canary_classes,
        repeat=plan.repeat,
        started_at=started,
        limitations=LIMITATIONS,
    )
    report = build_report(
        header=header,
        records=records,
        latency_s=latencies,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        abort_reason=abort_reason,
    )
    return RunResult(
        report=report,
        records=tuple(records),
        aborted=aborted,
        abort_reason=abort_reason,
    )


async def _run_one(
    *,
    case: ModerationCase,
    plan: RunPlan,
    screen: ScreenLeg,
    run_salt: str,
    repeat_index: int,
) -> tuple[CaseRecord, Usage]:
    """One case, start to finish, with its own world and its own canaries.

    The world is built and torn down INSIDE this function on purpose: a case's
    fixture must not outlive the case. If a later case could read an earlier case's
    Downloads folder, a canary found in its answer would be unattributable - the one
    property the whole measurement rests on.
    """

    from .fixture import build_fixture, teardown_fixture, verify_fixture

    world_root = plan.workdir / "worlds" / f"{repeat_index}-{case.id}"
    world = build_fixture(
        root=world_root,
        workspace=plan.dataset.workspace,
        salt=f"{run_salt}:{case.id}:{repeat_index}",
        case_id=case.id,
    )
    problems = verify_fixture(world)
    if problems:
        teardown_fixture(world)
        raise RuntimeError(
            "fixture world failed verification (a canary that did not render makes its "
            "leak class undetectable): " + "; ".join(problems)
        )

    inventory = plan.dataset.workspace.tool_inventory.get("production")
    leg = AgentLeg(
        world=world,
        provider=plan.model_route[0],
        model_id=plan.model_route[1],
        controls=plan.controls,
        inventory=tuple(inventory) if isinstance(inventory, list) else None,
    )
    usage = Usage()
    try:
        run = await leg.run_case(case, screen=screen.screen)
        for turn in run.turns:
            usage = usage + Usage(
                input_tokens=turn.inbound.usage.input_tokens,
                output_tokens=turn.inbound.usage.output_tokens,
            )
        score = score_case(
            case=case,
            run=run,
            dataset=plan.dataset,
            canaries=world.canaries,
            judge=None,
        )
    finally:
        if not plan.keep_workdir:
            teardown_fixture(world)

    record = CaseRecord(
        case_id=case.id,
        label=case.label,
        category=case.category,
        expected_disposition=case.expected_disposition,
        control=case.control,
        form=case.form,
        score=score,
        turns=tuple(turn.as_record() for turn in run.turns),
        variant_of=case.variant_of,
        pair_id=case.pair_id,
        d10_item=case.d10_item,
        laundering=case.laundering,
    )
    return record, usage


def _settings_for(workdir: Path) -> Mapping[str, Any] | None:
    """The config values a provider call needs, or ``None`` for defaults.

    WHY the fixture config dir and not the operator's: a run must not read, migrate
    or write the operator's config. The fixture config directory is empty by design,
    so the app's shipped defaults apply and nothing of the operator's is touched.
    """

    return None


def _estimated_usd(input_tokens: int, output_tokens: int) -> float:
    """A deliberately crude upper-bound cost estimate for the budget guard.

    WHY crude and why an upper bound: the guard exists to stop a runaway run, not to
    report spend. It assumes an expensive model's list price for both directions, so
    it fires early rather than late; the report carries real token counts alongside.
    """

    return (input_tokens / 1_000_000) * 15.0 + (output_tokens / 1_000_000) * 75.0


def _repo_head() -> str:
    try:
        return _git("rev-parse", "HEAD")
    except Exception:  # noqa: BLE001 - a missing git is not a run failure
        return ""


def _git_branch() -> str:
    try:
        return _git("rev-parse", "--abbrev-ref", "HEAD")
    except Exception:  # noqa: BLE001
        return ""


def _git(*args: str) -> str:
    """Read git state WITHOUT spawning a shell or touching credentials.

    ``GIT_CONFIG_SYSTEM=/dev/null`` and ``GIT_TERMINAL_PROMPT=0`` are set on the
    child: the system gitconfig on macOS installs the osxkeychain helper, and a
    keychain lookup from a run whose HOME has been redirected is how a rig raises a
    "keychain not found" prompt on the operator's screen. Nothing here needs a
    credential, so the helper is disabled rather than trusted to be idle.
    """

    import subprocess

    env = {
        **os.environ,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_ASKPASS": "/bin/false",
    }
    completed = subprocess.run(
        ("git", *args),
        capture_output=True,
        text=True,
        timeout=10,
        env=env,
        cwd=str(Path(__file__).resolve().parents[3]),
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""
