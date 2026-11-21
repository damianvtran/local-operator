"""Read-only dependency STATUS for a moved working set on THIS device (design §9).

WHY THIS EXISTS. A session move carries worktrees, unpushed commits and dirty files,
but never ``node_modules`` or ``.venv``: they are platform-specific, GB-scale and
rebuilt from the lockfile (§9). After a move the person (and the moved agent) needs
to know, per restored worktree, whether a dependency tree is still to be created,
already there, or could not be checked — otherwise "the work moved" is believed
while nothing builds.

WHAT THIS MODULE IS NOT. It never creates, links, removes or installs anything and
never starts a process: there is no runner, no argv table and no receipt here. The
security ruling on #4378 (owner 60334) is that a peer frame, the moved agent's
shell, stdin, a CLI flag or a file transfer is not the destination operator's
authority to run a package manager's lifecycle scripts. Until an authenticated,
move-bound operator opt-in exists (the next lane; ``docs/design/move-working-set.md``
§9 "Phase 2 exit criteria"), this slice reports and nothing more. Do not grow an
executor in here behind a flag: that lane starts from its own reviewed design.

WHAT IT READS, and why each read is safe to do unattended:

* THIS device's restore record (``workset-in/<sid>/pending.json``), written by its own
  restore — never the peer's offer;
* each restored worktree's lockfiles, through the same ``recipes_for`` the restore
  used, to name the dependency KIND (``pnpm``, ``uv``, ...);
* the placement rules the restore applied (``cwd_refusal``), so a record pointing at a
  dot directory or through a symlink is reported ``refused`` rather than described;
* ``lstat``/one ``scandir`` entry of the target, never following a symlink;
* ``shutil.which`` for the package manager (a PATH lookup; nothing is executed).

TRUTHFULNESS RULES the states encode (review findings S7/U2/R6 on #4378):

* an existing target is ``present_unverified`` — this module cannot tell a complete
  tree from one an interrupted install left half-built, so it never says "ready";
* anything that cannot be read (permissions, I/O) is ``unknown``, never "present"
  and never "pending";
* worktrees not restored here yet stay listed (``not_restored``); refused and
  manual rows are listed too, so no row disappears behind a summary.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

from local_operator.network import workset

#: Version of the ``--status --json`` document. ``executes: false`` is part of the
#: contract: a script reading it learns that nothing ran and nothing will.
STATUS_SCHEMA: Final = 1

#: The commons recipe kind (§9 row 1: link the primary's ``node_modules``).
SHARE_PRIMARY_KIND: Final = "pnpm_share_primary"
#: Which directory each kind would create — the path whose state is reported.
TARGET_DIR: Final[dict[str, str]] = {
    "pnpm": "node_modules",
    SHARE_PRIMARY_KIND: "node_modules",
    "npm": "node_modules",
    "yarn": "node_modules",
    "uv": ".venv",
    "uv_pip": ".venv",
}
#: The executable each kind needs, for the informational "not installed" hint. The
#: share kind needs none (it would be a link). Looked up on PATH, never run.
TOOL_FOR: Final[dict[str, str]] = {
    "pnpm": "pnpm",
    "npm": "npm",
    "yarn": "yarn",
    "uv": "uv",
    "uv_pip": "uv",
}
#: Kinds that stay the user's call by design: fetching LFS content follows reading
#: ``.lfsconfig`` (S10 of the move design).
MANUAL_KINDS: Final = frozenset({"git_lfs"})
#: Kinds with no dependency tree at all (Cargo/Go fetch on first build).
NOT_NEEDED_KINDS: Final = frozenset({"none"})

#: Row states (stable codes for ``--json``; human labels are keyed per locale).
ROW_STATES: Final = frozenset(
    {"pending", "present_unverified", "unknown", "refused", "manual", "not_needed"}
)
#: Overall states, most severe first: ``unknown`` wins over ``pending`` so a partial
#: read is never summarised as a plan; ``present_unverified`` only when every row is
#: present or needs nothing and every worktree is restored.
OVERALL_STATES: Final = ("unknown", "pending", "present_unverified", "nothing_needed")


@dataclass
class StatusRow:
    """One dependency kind in one restored worktree, as observed here, now."""

    cwd: str  # home-relative destination worktree
    kind: str
    target: str = ""  # home-relative node_modules/.venv the kind would create
    state: str = "pending"
    reason: str = ""
    missing_tool: str = ""  # the package manager absent from PATH, informational


@dataclass
class DependencyStatus:
    """What this device can say about ``session_id``'s dependency trees. Never acts."""

    session_id: str
    rows: list[StatusRow]
    not_restored: list[str]  # source paths whose worktree is not restored here (yet)

    @property
    def state(self) -> str:
        states = {row.state for row in self.rows}
        if "unknown" in states:
            return "unknown"
        if self.not_restored or states & {"pending", "refused", "manual"}:
            return "pending"
        if "present_unverified" in states:
            return "present_unverified"
        return "nothing_needed"

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": STATUS_SCHEMA,
            "executes": False,
            "ran": False,
            "session_id": self.session_id,
            "state": self.state,
            "rows": [asdict(row) for row in self.rows],
            "not_restored": list(self.not_restored),
        }


# ---------------------------------------------------------------- validation


def cwd_refusal(home: Path, config: Path, home_rel: str, roots: Sequence[str]) -> str:
    """``""`` when ``home_rel`` passes the restore's own placement rules, else why not.

    The worktree, or for ``<primary>/.worktrees/<name>`` its primary, must pass
    ``primary_path_refusal`` and resolve through ``contained_path`` to a real
    directory (it could have been replaced by a symlink since the restore). A refused
    record is reported, never described from whatever it points at.
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


def target_state(path: Path) -> tuple[str, str]:
    """``(state, reason)`` for a dependency target, read without following links.

    Absent or an EMPTY real directory is ``pending``. Anything else that exists —
    a non-empty directory, a file, a symlink (dangling or not) — is
    ``present_unverified``: it may be the operator's own tree or one an interrupted
    install left half-built, and nothing here can tell which. An ``OSError`` (an
    unreadable directory, an I/O error) is ``unknown``: the status must not guess.
    """
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return "pending", "absent"
    except OSError as exc:
        return "unknown", f"unreadable ({type(exc).__name__})"
    if not stat.S_ISDIR(info.st_mode):
        return "present_unverified", "symlink" if stat.S_ISLNK(info.st_mode) else "not_directory"
    try:
        with os.scandir(path) as entries:
            empty = next(entries, None) is None
    except OSError as exc:
        return "unknown", f"unreadable ({type(exc).__name__})"
    return ("pending", "empty") if empty else ("present_unverified", "exists")


# ---------------------------------------------------------------------- status


def _arrived_dir(config: Path, session_id: str) -> Path:
    from local_operator.network import workset_move

    return workset_move.arrived_dir(config, session_id)


def _valid_session_id(session_id: str) -> bool:
    """The move's own id rule (``workset_move._sid``): an id naming ``..`` never joins a path."""
    from local_operator.network import workset_move

    return bool(workset_move._SESSION_ID.match(str(session_id)))


def _restored_rows(config: Path, session_id: str) -> tuple[list[dict[str, Any]], list[str]]:
    """THIS device's restore record: (restored rows, source paths not restored)."""
    from local_operator.network import workset_move

    pending = workset_move._read_json(_arrived_dir(config, session_id) / "pending.json")
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


def _worktree_rows(
    home: Path, config: Path, dest_rel: str, roots: Sequence[str]
) -> list[StatusRow]:
    from local_operator.network import workset_transfer as engine

    refusal = cwd_refusal(home, config, dest_rel, roots)
    if refusal:
        return [StatusRow(dest_rel, "", state="refused", reason=refusal)]
    dest = engine.contained_path(home, dest_rel)
    primary = dest.parent.parent if dest.parent.name == ".worktrees" else dest
    out: list[StatusRow] = []
    for recipe in engine.recipes_for(dest, primary, home=home):
        name = TARGET_DIR.get(recipe.kind, "")
        row = StatusRow(dest_rel, recipe.kind, target=f"{dest_rel}/{name}" if name else "")
        if recipe.kind in NOT_NEEDED_KINDS:
            row.state = "not_needed"
        elif recipe.kind in MANUAL_KINDS:
            row.state, row.reason = "manual", "user_decides"
        elif not name or workset.trusted_recipe(recipe.to_dict()) is None:
            # A kind outside the closed recipe table is named, never described further.
            row.state, row.reason = "refused", "unknown_recipe"
        else:
            row.state, row.reason = target_state(dest / name)
        if row.state == "pending" and recipe.kind in TOOL_FOR:
            tool = TOOL_FOR[recipe.kind]
            if shutil.which(tool) is None:
                row.missing_tool = tool
        out.append(row)
    return out


def dependency_status(config: Path, home: Path, session_id: str) -> DependencyStatus | None:
    """The dependency status of every worktree of ``session_id`` restored on this device.

    ``None`` when no working set arrived here (or the id is not a session id).
    Read-only: see the module docstring. A worktree whose lockfiles or placement
    cannot be read yields one ``unknown`` row for that worktree, so one unreadable
    tree never hides the others nor turns into a claim about them.
    """
    if not _valid_session_id(session_id):
        return None
    rows, waiting = _restored_rows(config, session_id)
    if not rows and not waiting:
        return None
    roots = workset.restore_roots(config)
    out: list[StatusRow] = []
    for raw in rows:
        dest_rel = str(raw.get("dest_path") or "")
        try:
            out.extend(_worktree_rows(home, config, dest_rel, roots))
        except (OSError, ValueError) as exc:
            out.append(
                StatusRow(
                    dest_rel, "", state="unknown", reason=f"unreadable ({type(exc).__name__})"
                )
            )
    return DependencyStatus(session_id, out, waiting)


__all__ = [
    "MANUAL_KINDS",
    "NOT_NEEDED_KINDS",
    "OVERALL_STATES",
    "ROW_STATES",
    "SHARE_PRIMARY_KIND",
    "STATUS_SCHEMA",
    "TARGET_DIR",
    "TOOL_FOR",
    "DependencyStatus",
    "StatusRow",
    "cwd_refusal",
    "dependency_status",
    "target_state",
]
