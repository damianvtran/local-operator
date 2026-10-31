"""``lop disk reclaim``: argument registration and the operator-facing command.

WHY A SEPARATE MODULE. ``local_operator/cli.py`` builds its parser on every
``lop`` invocation, so registration here is stdlib-only (same rule as
``network.cli``): the planner (:mod:`local_operator.disk_pressure`) is imported by
the handler, never at parser-build time.

THE SHAPE OF THE COMMAND
    ``lop disk reclaim`` is a DRY RUN. It prints what would be reclaimed (with
    allocated-byte MEASUREMENTS, not promises) and what is KEPT, grouped by
    reason so the owner can act on a whole class. ``--apply`` re-prints that
    preview, then asks for a typed ``yes`` (``--yes`` skips the prompt for
    scripts), then removes ONLY the previewed items, re-validating each one at
    removal time. ``--json`` emits the receipt document for either mode.

    Categories are opt-in and never defaulted: ``--repo PATH`` (repeatable; the
    operator names every repository whose linked worktrees may be considered),
    ``--go-build-cache`` and ``--scratchpads``. Naming none is a usage error,
    because a command that guessed a scope under disk pressure is the incident
    this command exists to prevent.

EXIT CODES, in precedence order: 2 usage error, a refused category (a dry run
that REFUSES a requested category also exits 2, so a script cannot mistake a
half-answered preview for a full one), or the confirmation declined/unavailable;
else 1 some previewed item failed or was skipped at removal time, or a receipt
update failed; else 0 (including a clean dry run and "nothing to reclaim").
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

#: Kept words for the operator-facing rows. Output is operator CLI text,
#: English-only, like the sessions-cleanup section (docs/I18N.md).
_VERBS = {"dry-run": "would reclaim", "apply": "will reclaim"}


def add_parser(subparsers: Any, parent_parser: Any = None) -> None:
    """Register ``lop disk`` and its ``reclaim`` action (stdlib-only)."""
    parents = [parent_parser] if parent_parser is not None else []
    disk = subparsers.add_parser(
        "disk",
        help="Disk-pressure tools: measure first, reclaim only what is provably safe",
        parents=parents,
    )
    actions = disk.add_subparsers(dest="disk_command")
    reclaim = actions.add_parser(
        "reclaim",
        help="Preview (default) or apply a bounded, fail-closed disk reclaim",
        description=(
            "Preview what can be reclaimed, and why everything else is kept. Nothing is "
            "removed unless --apply is given AND confirmed. Removal goes only through the "
            "guarded removers: `git worktree remove` (no --force), `go clean -cache`, and "
            "the scratchpad content pass `lop sessions cleanup --scratchpads-only` uses. "
            "Sizes are allocated-byte measurements, not promised reclaim; the receipt's "
            "disk_usage before/after readings are the only figure of what was freed."
        ),
        parents=parents,
    )
    reclaim.add_argument(
        "--repo",
        action="append",
        default=None,
        metavar="PATH",
        help=(
            "a repository whose LINKED worktrees may be reclaimed (repeatable; never "
            "defaulted or discovered; its primary checkout is never a candidate)"
        ),
    )
    reclaim.add_argument(
        "--go-build-cache",
        action="store_true",
        help="include the Go build cache (cache-only; removed with `go clean -cache`)",
    )
    reclaim.add_argument(
        "--scratchpads",
        action="store_true",
        help=(
            "include delegated-session scratchpad build/cache content, under exactly the "
            "policy, window and guards of `lop sessions cleanup --scratchpads-only`"
        ),
    )
    reclaim.add_argument(
        "--min-idle-minutes",
        type=int,
        default=None,
        metavar="N",
        help="worktree idle window in minutes (default and minimum 60; cannot be lowered)",
    )
    reclaim.add_argument(
        "--apply",
        action="store_true",
        help="remove the previewed items after a typed confirmation (default: preview only)",
    )
    reclaim.add_argument(
        "--yes",
        action="store_true",
        help="with --apply, skip the typed confirmation (for scripts)",
    )
    reclaim.add_argument(
        "--force",
        action="store_true",
        help=(
            "scratchpads only: run once even if the delegated/scratchpad config switches "
            "are off (the same override as `lop sessions cleanup --force`); no guard is skipped"
        ),
    )
    reclaim.add_argument("--json", action="store_true", help="print the receipt as JSON")


def main(args: argparse.Namespace) -> int:
    """Dispatch ``lop disk <action>``."""
    action = getattr(args, "disk_command", None)
    if action != "reclaim":
        print(  # i18n: ignore operator CLI output, English-only; disk-reclaim section
            "usage: lop disk reclaim [--repo PATH]... [--go-build-cache] [--scratchpads] "
            "[--apply [--yes]] [--json]",
            file=sys.stderr,
        )
        return 2
    return reclaim_command(args)


def _fmt_bytes(value: int | None, truncated: bool) -> str:
    """Human size; ``>= `` prefixes a truncated (lower-bound) measurement."""
    if value is None:
        return "-"
    prefix = ">= " if truncated else ""
    if value < 1024:
        return f"{prefix}{value} B"
    size = float(value)
    for unit in ("KiB", "MiB", "GiB"):
        size /= 1024
        if size < 1024:
            return f"{prefix}{size:.1f} {unit}"
    return f"{prefix}{size / 1024:.1f} TiB"


def _print_preview(plan_: Any, *, mode: str) -> None:
    """The grouped, human preview. Reclaim rows first, then KEPT grouped by reason code."""
    from local_operator import disk_pressure as dp

    out = _say
    verb = _VERBS[mode]
    titles = (
        (dp.CATEGORY_WORKTREE, "Linked worktrees"),
        (dp.CATEGORY_GO_CACHE, "Go build cache"),
        (dp.CATEGORY_SCRATCHPAD, "Session scratchpad content"),
    )
    for category, title in titles:
        rows = [i for i in plan_.items if i.category == category]
        if not rows and not any(r["category"] == category for r in plan_.refusals):
            continue
        out(f"== {title} ==")
        for refusal in plan_.refusals:
            if refusal["category"] == category:
                where = f" ({refusal['repo']})" if refusal.get("repo") else ""
                out(f"  REFUSED{where}: {refusal['reason']}")
        for item in (r for r in rows if r.action == dp.ACTION_RECLAIM):
            size = _fmt_bytes(item.allocated_bytes, item.bytes_truncated)
            out(f"  {verb:<14} {size:>12}  {item.path}")
            out(f"  {'':<14} {'':>12}  ({item.reason})")
        kept = [r for r in rows if r.action == dp.ACTION_KEEP and r.code != "refused"]
        by_code: dict[str, list[Any]] = {}
        for item in kept:
            by_code.setdefault(item.code, []).append(item)
        for code in sorted(by_code):
            group = by_code[code]
            hint = group[0].hint
            out(f"  KEPT [{code}] x{len(group)}" + (f" -- {hint}" if hint else ""))
            for item in group:
                out(f"      {item.path or item.id}: {item.reason}")
        out("")
    summary = dp.summarize(plan_)
    out(
        f"{summary['reclaimable']} item(s) reclaimable, {summary['kept']} kept; "
        "measured allocated "
        + _fmt_bytes(summary["measured_allocated_bytes"], summary["measured_is_lower_bound"])
    )
    out(f"note: {dp.MEASUREMENT_NOTE}")


def _say(text: str = "") -> None:
    """Operator-facing line on stdout (the preview)."""
    print(text)  # i18n: ignore operator CLI output, English-only; disk-reclaim section


def _err(text: str, **_ignored: Any) -> None:
    """Operator-facing line on STDERR, so ``--json`` stdout stays one document."""
    # i18n: ignore operator CLI output, English-only; disk-reclaim section
    print(text, file=sys.stderr)


def _emit_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2))  # i18n: ignore machine-readable receipt


def reclaim_command(args: argparse.Namespace) -> int:
    """Plan, optionally confirm, optionally apply. See the module docstring."""
    from local_operator import disk_pressure as dp
    from local_operator.paths import config_dir

    json_mode = bool(getattr(args, "json", False))
    repos = tuple(getattr(args, "repo", None) or ())
    go_cache = bool(getattr(args, "go_build_cache", False))
    scratch = bool(getattr(args, "scratchpads", False))
    apply_mode = bool(getattr(args, "apply", False))
    yes = bool(getattr(args, "yes", False))
    force = bool(getattr(args, "force", False))
    min_idle = getattr(args, "min_idle_minutes", None)
    min_idle = dp.DEFAULT_MIN_IDLE_MINUTES if min_idle is None else int(min_idle)

    def refuse(message: str) -> int:
        if json_mode:
            _emit_json({"kind": "disk-reclaim-receipt", "status": "refused", "error": message})
        else:
            _err(message)
        return 2

    problem = dp.validate_min_idle(min_idle)
    if problem:
        return refuse(problem)
    if yes and not apply_mode:
        return refuse("--yes only applies with --apply")
    if force and not scratch:
        return refuse("--force only applies to --scratchpads")
    if not (repos or go_cache or scratch):
        return refuse(
            "nothing selected: name at least one of --repo PATH, --go-build-cache, "
            "--scratchpads (nothing is discovered or defaulted)"
        )

    root = config_dir()
    selection = dp.Selection(
        repos=repos,
        go_build_cache=go_cache,
        scratchpads=scratch,
        min_idle_minutes=min_idle,
        force=force,
    )
    plan_ = dp.plan(selection, root)

    if not apply_mode:
        if json_mode:
            _emit_json(dp.preview_payload(plan_))
        else:
            _print_preview(plan_, mode="dry-run")
            _err("nothing was removed (dry run); add --apply to remove the items above")
        return 2 if plan_.refusals else 0

    if not json_mode:
        _print_preview(plan_, mode="apply")
    if not plan_.reclaimable:
        if json_mode:
            payload = dp.preview_payload(plan_)
            payload["status"] = "nothing-to-do"
            payload["mode"] = "apply"
            _emit_json(payload)
        else:
            _err("nothing to reclaim")
        return 2 if plan_.refusals else 0

    if not yes:
        if not sys.stdin.isatty():
            return refuse(
                "refusing: not a terminal and --yes was not given, so nothing was removed"
            )
        prompt = (
            f"Remove {len(plan_.reclaimable)} previewed item(s)? Only these will be "
            "touched, each re-checked at removal time. Type 'yes' to confirm: "
        )
        # The prompt goes to stderr in --json mode so stdout stays one document.
        if json_mode:
            print(prompt, end="", file=sys.stderr)
            try:
                answer = sys.stdin.readline()
            except (EOFError, KeyboardInterrupt):
                answer = ""
        else:
            try:
                answer = input(prompt)
            except (EOFError, KeyboardInterrupt):
                answer = ""
        if answer.strip().lower() != "yes":
            if json_mode:
                _emit_json(
                    {"kind": "disk-reclaim-receipt", "status": "declined", "confirmed": False}
                )
            else:
                _err("not confirmed; nothing was removed")
            return 2

    def progress(item: Any) -> None:
        if not json_mode:
            tail = f" -- {item.detail}" if item.detail else ""
            _err(f"  {item.outcome:<8} {item.path or item.id}{tail}")

    result = dp.apply(
        plan_,
        root,
        # Stamped by the receipt writer itself, so the durable file and stdout
        # agree instead of the file omitting what the JSON claimed.
        confirmation=dp.CONFIRMATION_YES if yes else dp.CONFIRMATION_TYPED,
        on_progress=progress,
    )
    payload = result.payload
    # Even JSON callers get a visible warning on stderr; stdout remains one
    # receipt document, and the warning also survives in that document.
    for warning in payload.get("receipt_warnings") or []:
        _err(f"warning: {warning}")
    if json_mode:
        _emit_json(payload)
    else:
        _err(f"status: {payload['status']}")
        readings = zip(
            payload.get("disk_usage_before") or [], payload.get("disk_usage_after") or []
        )
        for before, after in readings:
            delta = after["free"] - before["free"]
            _err(
                f"  free space on {before['path']}: {before['free']} -> {after['free']} bytes "
                f"({'+' if delta >= 0 else ''}{delta}); allocated/logical sizes above "
                "are not reclaimed bytes"
            )
        if result.receipt_path:
            _err(f"receipt: {result.receipt_path}")
        if payload.get("error"):
            _err(payload["error"])
    return result.exit_code
