"""Recreate a moved working set's dependency trees ON THIS DEVICE (design §9, phase 2).

WHY THIS EXISTS. A session move carries worktrees, unpushed commits and dirty files,
but never ``node_modules`` or ``.venv``: they are platform-specific, GB-scale and
rebuilt from the lockfile in minutes (§9). PR B (#4245) listed the recreation recipe
for each restored worktree and ran nothing, so the operator still arrived to a tree
that could not build until they typed each command. This module RUNS those recipes,
and only when the operator at the destination asks (``lop sessions setup-deps``, or
``--setup-deps`` on a recall, where this device is the destination).

THE SECURITY MODEL, in the order a reviewer should check it:

* **Destination-local authority only.** Nothing here is reachable from a peer frame.
  The person whose home the installs write into (and whose shell runs the package
  manager's lifecycle scripts) is the one who typed the command. A ``--to <peer>``
  move never runs setup there; it names the command to run on that device instead.
* **No peer-provided text runs.** The recipes are re-derived HERE from the restored
  tree's own lockfiles (``workset_transfer.recipes_for``), each is re-checked against
  the closed table ``workset.trusted_recipe`` accepts, and then mapped to a fixed
  argv by :data:`SETUP_ARGV` — a shell is never invoked, so ``&&``/``;``/``$()`` in a
  recipe can never mean anything. The pnpm commons recipe's ``rm -rf`` is never run:
  it becomes a single ``symlink`` into an absent (or empty) path.
* **Destination-contained cwd.** The working directory comes from THIS device's
  ``pending.json`` (written by its own restore), and is re-checked against the same
  rules the restore used: ``contained_path`` (no symlinked/aliased component) and
  ``primary_path_refusal`` under ``mesh.move.working_set_roots`` (no dot/system
  directory, not inside the config dir, not on PATH).
* **Never destroys.** An existing ``node_modules``/``.venv`` (a recall updating a tree
  in place still has the old one) is ``present`` and left untouched; only an absent
  path or an EMPTY real directory is used. No source file, no dependency tree and no
  pre-existing target is removed. A failure, timeout or Ctrl-C stops setup and leaves
  the restored working set exactly as the move left it.
* **No toolchain install.** A missing ``pnpm``/``uv``/... is reported with the exact
  recipe to run once the operator has installed it.
* **Bounded.** Each step runs through ``bounded_contract.supervise_direct`` (its own
  process group, SIGTERM then SIGKILL, memory guard) with the limits the
  package-manager build steps of ``mobile/install.py`` already use — not new numbers.

The receipt (``setup.json`` beside ``pending.json``) is written before and after every
step, so ``lop sessions setup-deps <sid> --status`` reads what happened even after a
crash or an interrupt.
"""

from __future__ import annotations

import functools
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from local_operator.network import workset

#: The receipt file, beside the restore's own ``pending.json``.
SETUP_RECEIPT: Final = "setup.json"
#: Audit event, one row per setup run (kinds, states and exit codes only).
AUDIT_WORKSET_SETUP: Final = "session.workset.setup"

#: ``Recipe.run`` (exactly as ``workset.RECIPE_RUNS`` spells it) → the argv steps that
#: run INSTEAD of a shell. Keyed by the run string, so a recipe outside the closed set
#: has no argv at all; ``test_setup_argv_covers_every_recipe_run`` pins that every
#: entry of ``RECIPE_RUNS`` is here and that the argv joins back to the same text.
SETUP_ARGV: Final[dict[str, tuple[tuple[str, ...], ...]]] = {
    "pnpm install --frozen-lockfile": (("pnpm", "install", "--frozen-lockfile"),),
    "npm ci": (("npm", "ci"),),
    "yarn install --immutable": (("yarn", "install", "--immutable"),),
    "yarn install --frozen-lockfile": (("yarn", "install", "--frozen-lockfile"),),
    "uv venv && uv sync --frozen": (("uv", "venv"), ("uv", "sync", "--frozen")),
}
#: The commons recipe (§9 row 1). Never executed as text: see :func:`_share_primary`.
SHARE_PRIMARY_KIND: Final = "pnpm_share_primary"
#: ``uv_pip`` carries the requirements file NAME (the only variable part); the same
#: shell-inert grammar ``workset.trusted_recipe`` accepts, with the name captured.
_UV_PIP_RUN: Final = re.compile(r"uv venv && uv pip install -r (requirements[A-Za-z0-9._-]*\.txt)")
#: Which directory each kind creates — the path whose prior existence means "leave it".
TARGET_DIR: Final[dict[str, str]] = {
    "pnpm": "node_modules",
    SHARE_PRIMARY_KIND: "node_modules",
    "npm": "node_modules",
    "yarn": "node_modules",
    "uv": ".venv",
    "uv_pip": ".venv",
}
#: Kinds printed but never run by this module: fetching LFS content is the user's call
#: after reading ``.lfsconfig`` (S10), and ``none`` has nothing to run.
MANUAL_KINDS: Final = frozenset({"git_lfs"})
SKIP_KINDS: Final = frozenset({"none"})

#: Environment that would point an install at a tree OTHER than the worktree's own:
#: an activated venv (``uv sync``/``uv pip`` honour ``VIRTUAL_ENV``/``UV_PROJECT_
#: ENVIRONMENT`` and would rewrite the operator's active environment, the opposite of
#: "never touch a pre-existing tree") and a conda prefix. Removed from the child env.
_FOREIGN_ENV: Final = ("VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT", "CONDA_PREFIX")

#: Row states. ``done``/``present``/``skipped`` need nothing more; the rest leave the
#: setup INCOMPLETE and say what to do (the restored working set is intact either way).
OK_STATES: Final = frozenset({"done", "present", "skipped"})
ROW_STATES: Final = OK_STATES | frozenset(
    {"failed", "timeout", "cancelled", "missing_toolchain", "refused", "manual", "not_run"}
)


@dataclass
class SetupRow:
    """One recipe in one restored worktree, and what happened to it."""

    cwd: str  # home-relative destination worktree
    kind: str
    run: str
    target: str = ""  # home-relative node_modules/.venv this recipe creates
    state: str = "not_run"
    reason: str = ""
    missing: str = ""  # the absent executable, for ``missing_toolchain``
    steps: list[dict[str, Any]] = field(default_factory=list)
    log: str = ""


@dataclass
class SetupPlan:
    """What :func:`run_setup` would do. ``refused`` rows never run."""

    session_id: str
    rows: list[SetupRow]
    not_restored: list[str]  # source paths whose worktree is not restored here (yet)


# ----------------------------------------------------------------------- limits


def _limits() -> tuple[float, float, Any]:
    """(timeout s, grace s, memory budget) for ONE package-manager step.

    Reused from ``mobile/install.py``'s bundle build steps — the same commands
    (``pnpm install --frozen-lockfile``) under the same hazard (a runaway install
    group) — so this module introduces no number of its own. See the rationale on
    ``_BUILD_STEP_TIMEOUT``/``_BUILD_KILL_GRACE``/``_STEP_MEMORY_FLOOR_MB`` there.
    """
    from local_operator.mobile import install

    return install._BUILD_STEP_TIMEOUT, install._BUILD_KILL_GRACE, install._step_memory_budget()


# ---------------------------------------------------------------- validation


def steps_for(kind: str, run: str) -> tuple[tuple[str, ...], ...] | None:
    """The fixed argv steps for a recipe, or ``None`` when it is outside the closed set.

    Two gates, both required: ``workset.trusted_recipe`` (the table the consent card
    already trusts) and :data:`SETUP_ARGV` (the argv this module will run). A run string
    that is not byte-for-byte one of ours has no argv, whatever it says.
    """
    if workset.trusted_recipe({"kind": kind, "run": run, "cwd": ""}) is None:
        return None
    if kind == "uv_pip":
        match = _UV_PIP_RUN.fullmatch(run)
        if match is None:
            return None
        return (("uv", "venv"), ("uv", "pip", "install", "-r", match.group(1)))
    steps = SETUP_ARGV.get(run)
    if steps is None or kind not in TARGET_DIR:
        return None
    return steps


def cwd_refusal(home: Path, config: Path, home_rel: str, roots: Sequence[str]) -> str:
    """``""`` when setup may run in ``home_rel`` on this device, else why not.

    The restore's own placement rules, re-applied at execution time (the tree could
    have been replaced by a symlink since): the worktree, or for
    ``<primary>/.worktrees/<name>`` its primary, must pass ``primary_path_refusal``,
    and the worktree itself must resolve through ``contained_path`` to a real
    directory.
    """
    from local_operator.network import workset_transfer as engine

    match = re.fullmatch(r"(~/.+)/\.worktrees/([^/]+)", home_rel or "")
    if match is not None and match.group(2).startswith("."):
        return "hidden worktree name"
    primary_rel = match.group(1) if match else home_rel
    refusal = engine.primary_path_refusal(home, primary_rel, roots=roots, forbid=[config])
    if refusal:
        return refusal
    try:
        path = engine.contained_path(home, home_rel)
    except engine.ContainmentError as exc:
        return exc.detail
    if path.is_symlink() or not path.is_dir():
        return "not a directory"
    return ""


def _occupied(path: Path) -> bool:
    """Anything at ``path`` except an empty REAL directory (a symlink always counts)."""
    if not os.path.lexists(path):
        return False
    if path.is_symlink() or not path.is_dir():
        return True
    return any(True for _ in os.scandir(path))


# ---------------------------------------------------------------------- planning


def _receipt_dir(config: Path, session_id: str) -> Path:
    from local_operator.network import workset_move

    return workset_move.arrived_dir(config, session_id)


def _valid_session_id(session_id: str) -> bool:
    """The move's own id rule (``workset_move._sid``): an id naming ``..`` never joins a path."""
    from local_operator.network import workset_move

    return bool(workset_move._SESSION_ID.match(str(session_id)))


def _restored_rows(config: Path, session_id: str) -> tuple[list[dict[str, Any]], list[str]]:
    """THIS device's restore record: (restored rows, source paths not restored)."""
    from local_operator.network import workset_move

    pending = workset_move._read_json(_receipt_dir(config, session_id) / "pending.json")
    if pending is None:
        return [], []
    restored = pending.get("restored") or {}
    rows: list[dict[str, Any]] = []
    waiting: list[str] = []
    for plan in pending.get("plans") or []:
        row = restored.get(str(plan.get("n"))) if isinstance(restored, Mapping) else None
        if isinstance(row, Mapping) and row.get("state") == "restored" and row.get("dest_path"):
            rows.append(dict(row))
        else:
            waiting.append(str(plan.get("path") or ""))
    return rows, waiting


def plan_setup(config: Path, home: Path, session_id: str) -> SetupPlan | None:
    """The recipes for every worktree of ``session_id`` restored on this device.

    ``None`` when no working set arrived here. Recipes are derived NOW from each
    restored tree (``recipes_for``), never read from the peer's offer.
    """
    from local_operator.network import workset_transfer as engine

    if not _valid_session_id(session_id):
        return None
    rows, waiting = _restored_rows(config, session_id)
    if not rows and not waiting:
        return None
    roots = workset.restore_roots(config)
    out: list[SetupRow] = []
    for raw in rows:
        dest_rel = str(raw.get("dest_path") or "")
        refusal = cwd_refusal(home, config, dest_rel, roots)
        if refusal:
            out.append(SetupRow(dest_rel, "", "", state="refused", reason=refusal))
            continue
        dest = engine.contained_path(home, dest_rel)
        primary = dest.parent.parent if dest.parent.name == ".worktrees" else dest
        for recipe in engine.recipes_for(dest, primary, home=home):
            name = TARGET_DIR.get(recipe.kind, "")
            row = SetupRow(
                dest_rel,
                recipe.kind,
                recipe.run,
                target=f"{dest_rel}/{name}" if name else "",
            )
            if recipe.kind in SKIP_KINDS:
                row.state = "skipped"
            elif recipe.kind in MANUAL_KINDS:
                row.state, row.reason = "manual", "user_decides"
            elif recipe.kind != SHARE_PRIMARY_KIND and steps_for(recipe.kind, recipe.run) is None:
                row.state, row.reason = "refused", "unknown_recipe"
            elif recipe.kind == SHARE_PRIMARY_KIND and (
                workset.trusted_recipe(recipe.to_dict()) is None
            ):
                row.state, row.reason = "refused", "unknown_recipe"
            elif name and _occupied(dest / name):
                row.state, row.reason = "present", "target_exists"
            out.append(row)
    return SetupPlan(session_id, out, waiting)


# --------------------------------------------------------------------- execution


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_receipt(directory: Path, payload: Mapping[str, Any]) -> None:
    from local_operator.network import workset_move

    workset_move._write_json(directory / SETUP_RECEIPT, payload)


def read_receipt(config: Path, session_id: str) -> dict[str, Any] | None:
    """The last setup run's receipt for ``session_id`` on this device, or ``None``."""
    from local_operator.network import workset_move

    if not _valid_session_id(session_id):
        return None
    return workset_move._read_json(_receipt_dir(config, session_id) / SETUP_RECEIPT)


def _child_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in _FOREIGN_ENV}
    # Strip the multiplexer's identity so a lifecycle script cannot address the
    # operator's terminal panes (AGENTS.md: rigs strip inherited CMUX_*).
    return {k: v for k, v in env.items() if not k.startswith("CMUX_")}


@contextmanager
def _interrupt_latch(stop: dict[str, int]) -> Iterator[None]:
    """Turn SIGINT/SIGTERM into ``stop`` for the bounded runner (main thread only).

    ``supervise_direct`` reads ``stop["signum"]`` each tick and signals the step's own
    group; latching here means Ctrl-C ends the install and is recorded as
    ``cancelled`` rather than a traceback that leaves the receipt saying ``running``.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def latch(signum: int, _frame: Any) -> None:
        stop["signum"] = signum

    previous = {sig: signal.signal(sig, latch) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def _share_primary(dest: Path) -> str:
    """The commons recipe without its ``rm -rf``: link only into an absent/empty path.

    Re-checks the §9 condition now (lockfile byte-identical to the primary's, the
    primary has a real ``node_modules``), so a tree that drifted since the restore
    falls back to a refusal rather than sharing a tree built from another lockfile.
    Returns ``""`` on success, else a reason code.
    """
    primary = dest.parent.parent
    try:
        same = (dest / "pnpm-lock.yaml").read_bytes() == (primary / "pnpm-lock.yaml").read_bytes()
    except OSError:
        same = False
    shared = primary / "node_modules"
    if not same or shared.is_symlink() or not shared.is_dir():
        return "lockfile_differs"
    target = dest / "node_modules"
    if _occupied(target):
        return "target_exists"
    if os.path.lexists(target):
        target.rmdir()  # empty real directory only (``_occupied`` said so); rmdir refuses otherwise
    os.symlink("../../node_modules", target)
    return ""


def run_setup(
    config: Path,
    home: Path,
    plan: SetupPlan,
    *,
    which: Callable[[str], str | None] = shutil.which,
    supervise: Callable[..., None] | None = None,
    limits: tuple[float, float, Any] | None = None,
    progress: Callable[[SetupRow], None] | None = None,
    starting: Callable[[SetupRow], None] | None = None,
    stop: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Run ``plan``'s runnable rows, in order, recording each step; return the receipt.

    A failed, timed-out or cancelled step stops THAT row; a cancellation also stops
    every later row (``not_run``). Nothing already restored is touched on any path.
    ``which``/``supervise``/``limits`` are injectable for tests; production uses PATH,
    ``bounded_contract.supervise_direct`` and :func:`_limits`.
    """
    from local_operator import bounded_contract
    from local_operator.network import workset_transfer as engine

    directory = _receipt_dir(config, plan.session_id)
    supervise = supervise or bounded_contract.supervise_direct
    timeout, grace, budget = limits or _limits()
    stop = stop if stop is not None else {"signum": 0}
    roots = workset.restore_roots(config)
    receipt: dict[str, Any] = {
        "session_id": plan.session_id,
        "started_at": _now(),
        "state": "running",
        "not_restored": list(plan.not_restored),
        "limits": {"timeout_s": timeout, "grace_s": grace},
        "rows": [asdict(r) for r in plan.rows],
    }
    _write_receipt(directory, receipt)
    env = _child_env()
    with _interrupt_latch(stop):
        for index, row in enumerate(plan.rows):
            if row.state != "not_run":
                continue
            if stop["signum"]:
                break
            # Re-check at the moment of execution: the plan may be minutes old.
            refusal = cwd_refusal(home, config, row.cwd, roots)
            if refusal:
                row.state, row.reason = "refused", refusal
            else:
                dest = engine.contained_path(home, row.cwd)
                name = TARGET_DIR.get(row.kind, "")
                if name and _occupied(dest / name):
                    row.state, row.reason = "present", "target_exists"
                elif row.kind == SHARE_PRIMARY_KIND:
                    reason = _share_primary(dest)
                    row.state, row.reason = ("done", "") if not reason else ("refused", reason)
                    row.steps.append({"action": "symlink", "to": "../../node_modules"})
                elif _missing_tool(row, which):
                    pass  # ``_missing_tool`` recorded it; nothing was started
                else:
                    if starting is not None:
                        starting(row)
                    _run_row(
                        row,
                        dest,
                        directory,
                        index,
                        env,
                        which,
                        supervise,
                        stop,
                        timeout,
                        grace,
                        budget,
                    )
            receipt["rows"][index] = asdict(row)
            _write_receipt(directory, receipt)
            if progress is not None:
                progress(row)
    for index, row in enumerate(plan.rows):
        if row.state == "not_run" and stop["signum"]:
            row.state, row.reason = "not_run", "cancelled"
            receipt["rows"][index] = asdict(row)
    receipt["finished_at"] = _now()
    receipt["state"] = summary_state(plan.rows, cancelled=bool(stop["signum"]))
    _write_receipt(directory, receipt)
    return receipt


def _missing_tool(row: SetupRow, which: Callable[[str], str | None]) -> bool:
    """Record ``missing_toolchain`` when any step's executable is not on PATH.

    Checked for EVERY step before the first runs, so ``uv venv`` never creates a
    venv that a missing second tool would leave half-built. Never installs one.
    """
    for argv in steps_for(row.kind, row.run) or ():
        if which(argv[0]) is None:
            row.state, row.reason, row.missing = "missing_toolchain", "not_on_path", argv[0]
            return True
    return False


def _run_row(
    row: SetupRow,
    dest: Path,
    directory: Path,
    index: int,
    env: Mapping[str, str],
    which: Callable[[str], str | None],
    supervise: Callable[..., None],
    stop: dict[str, int],
    timeout: float,
    grace: float,
    budget: Any,
) -> None:
    from local_operator import bounded_contract

    steps = steps_for(row.kind, row.run) or ()
    log_path = directory / f"setup-{index}.log"
    row.log = str(log_path)
    with open(log_path, "ab") as raw, open(log_path, "a", encoding="utf-8") as text:
        for argv in steps:
            outcome = bounded_contract.Outcome()
            started = time.monotonic()
            text.write(f"$ {' '.join(argv)}\n")
            text.flush()
            spawn = functools.partial(
                subprocess.Popen,
                cwd=str(dest),
                stdin=subprocess.DEVNULL,
                stdout=raw,
                stderr=subprocess.STDOUT,
            )
            supervise(
                list(argv),
                timeout=timeout,
                grace=grace,
                env=dict(env),
                outcome=outcome,
                stop=stop,
                budget=budget,
                stream=text,
                spawn=spawn,
            )
            row.steps.append(
                {
                    "argv": list(argv),
                    "exit_code": outcome.command_exit_code,
                    "cause": outcome.termination_cause,
                    "elapsed_s": round(time.monotonic() - started, 3),
                }
            )
            cause = outcome.termination_cause
            if cause == "interruption" or stop["signum"]:
                row.state, row.reason = "cancelled", "interrupted"
                return
            if cause == "deadline":
                row.state, row.reason = "timeout", "deadline"
                return
            if outcome.command_exit_code != 0 or cause != "child-exit":
                row.state, row.reason = "failed", cause
                return
    row.state = "done"


def summary_state(rows: Sequence[SetupRow], *, cancelled: bool = False) -> str:
    """``complete`` when every row needs nothing more, else ``incomplete``/``cancelled``."""
    if cancelled:
        return "cancelled"
    return "complete" if all(r.state in OK_STATES for r in rows) else "incomplete"


def audit(config: Path, session_id: str, receipt: Mapping[str, Any]) -> None:
    """One ``session.workset.setup`` row: kinds, states, exit codes — no paths, no output.

    Written from the CLI process (setup is never a relay action), so it opens THIS
    device's audit log directly, the way ``network/cli.py`` records a local grant
    change when the relay is not running. A record is not a gate: a failure to write
    it never changes the setup's outcome.
    """
    try:
        from local_operator.network.audit import AuditEvent, AuditLog

        log = AuditLog.from_config(config)
        try:
            log.record(
                AuditEvent(
                    event=AUDIT_WORKSET_SETUP,
                    actor="self",
                    session_id=session_id,
                    subject=session_id,
                    outcome="ok" if receipt.get("state") == "complete" else "failed",
                    detail={
                        "state": str(receipt.get("state") or ""),
                        # Flat ``kind:state:codes`` strings: the audit writer keeps
                        # scalars and lists of scalars, and a nested dict would be
                        # flattened to its repr.
                        "rows": [
                            "{}:{}:{}".format(
                                r.get("kind") or "",
                                r.get("state") or "",
                                ",".join(str(s.get("exit_code")) for s in r.get("steps") or []),
                            )
                            for r in receipt.get("rows") or []
                        ][:50],
                    },
                )
            )
        finally:
            log.close()
    except Exception:  # noqa: BLE001 — a record is not a gate
        pass


def plan_json(plan: SetupPlan) -> dict[str, Any]:
    return {
        "session_id": plan.session_id,
        "rows": [asdict(r) for r in plan.rows],
        "not_restored": list(plan.not_restored),
    }


def receipt_json(receipt: Mapping[str, Any]) -> str:
    return json.dumps(dict(receipt), indent=2, sort_keys=True, default=str)


__all__ = [
    "AUDIT_WORKSET_SETUP",
    "SETUP_ARGV",
    "SETUP_RECEIPT",
    "SetupPlan",
    "SetupRow",
    "audit",
    "cwd_refusal",
    "plan_json",
    "plan_setup",
    "read_receipt",
    "receipt_json",
    "run_setup",
    "steps_for",
    "summary_state",
]
