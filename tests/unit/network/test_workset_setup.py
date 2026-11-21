"""Read-only dependency status for a moved working set (move-working-set.md §9, #4378).

``workset_setup`` REPORTS what the destination's restored worktrees need; it never
runs, installs, links or removes anything (security ruling on #4378, owner 60334).
What is pinned here:

* the module has no execution surface at all (no runner, no argv table, no process
  or filesystem-mutation API), and computing status with recording package managers
  on PATH launches none of them;
* a target's state is read without following links and without guessing: absent or
  empty is ``pending``; anything existing is ``present_unverified`` (never "ready");
  an unreadable entry is ``unknown``; nothing is modified;
* refused, manual and not-restored rows stay listed, and the overall state never
  hides them;
* the model-facing arrival note carries rows as DATA (path, kind, state) with no
  recipe shell text, names only ``--status``, and keeps its guard when the status
  could not be computed.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import workset, workset_setup

SID = "a1b2c3d4"

_ROOT_SKIPS_PERMISSIONS = hasattr(os, "geteuid") and os.geteuid() == 0


# ------------------------------------------------------------------ fixtures


@pytest.fixture()
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    home = tmp_path / "home"
    config = tmp_path / "config"
    bindir = tmp_path / "bin"
    for d in (home, config, bindir):
        d.mkdir()
    # Recording package managers are the ONLY ones on PATH (plus `sh`'s helpers): a
    # launch of any of them, by any code path, leaves ``launched``.
    sysbin = tmp_path / "sysbin"
    sysbin.mkdir()
    for util in ("mkdir", "sh"):
        found = shutil.which(util)
        assert found, util
        (sysbin / util).symlink_to(found)
    launched = tmp_path / "launched"
    for tool in ("pnpm", "npm", "yarn", "uv"):
        script = bindir / tool
        script.write_text(f'#!/bin/sh\necho "{tool} $*" >> "{launched}"\nmkdir -p node_modules\n')
        script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}:{sysbin}")
    return {"home": home, "config": config, "bin": bindir, "launched": launched}


def _restored(
    config: Path, rows: list[str], *, waiting: Sequence[str] = (), epoch: str = "e1"
) -> Path:
    """THIS device's restore record, as ``workset_move.restore_arrived`` writes it.

    ``epoch`` is the move that staged it (``lease_epoch`` in the record); the arrival
    note binds to it, so ``e1`` matches ``_spool``'s move unless a test says otherwise.
    """
    directory = config / "network" / "workset-in" / SID
    directory.mkdir(parents=True)
    plans, restored = [], {}
    for n, dest in enumerate(rows):
        plans.append({"n": n, "path": dest})
        restored[str(n)] = {"state": "restored", "path": dest, "dest_path": dest}
    for n, path in enumerate(waiting, start=len(rows)):
        plans.append({"n": n, "path": path})
        restored[str(n)] = {"state": "pending_restore", "path": path, "reason": "x"}
    (directory / "pending.json").write_text(
        json.dumps({"lease_epoch": epoch, "plans": plans, "restored": restored}),
        encoding="utf-8",
    )
    return directory


def _worktree(home: Path, rel: str, files: dict[str, str]) -> Path:
    path = home / rel
    path.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (path / name).write_text(text, encoding="utf-8")
    return path


def _status(env: dict[str, Path]) -> workset_setup.DependencyStatus:
    status = workset_setup.dependency_status(env["config"], env["home"], SID)
    assert status is not None
    return status


def _snapshot(root: Path) -> dict[str, tuple[int, int]]:
    """``(mode, size)`` per path under ``root`` without following links."""
    out = {}
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            path = Path(dirpath) / name
            info = os.lstat(path)
            out[str(path.relative_to(root))] = (info.st_mode, info.st_size)
    return out


# ---------------------------------------------------- no execution surface


def test_the_module_has_no_execution_or_mutation_surface() -> None:
    """Read-only by construction, not by convention (#4378 M1).

    No runner, argv table, receipt writer or audit hook exists to be reached by a
    flag, a peer frame or a later caller; and the module imports nothing that could
    start a process.
    """
    for gone in ("run_setup", "SETUP_ARGV", "steps_for", "read_receipt", "audit", "plan_setup"):
        assert not hasattr(workset_setup, gone), gone
    tree = ast.parse(Path(workset_setup.__file__).read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in (node.names if isinstance(node, ast.Import) else [])
    } | {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert not imported & {"subprocess", "signal", "threading", "asyncio", "multiprocessing"}
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not calls & {
        "symlink",
        "rmdir",
        "unlink",
        "remove",
        "rmtree",
        "mkdir",
        "write_text",
        "Popen",
        "run",
        "system",
        "execv",
        "execvp",
        "spawn",
        "replace",
        "rename",
    }


def test_status_over_every_state_launches_nothing_and_changes_nothing(
    env: dict[str, Path],
) -> None:
    home = env["home"]
    _worktree(home, "src/app", {"package-lock.json": "{}"})
    web = _worktree(home, "src/web", {"pnpm-lock.yaml": "lockfileVersion: 9\n"})
    (web / "node_modules").mkdir()
    (web / "node_modules" / "mine.js").write_text("keep\n")
    _worktree(home, "src/py", {"uv.lock": "version = 1\n"})
    _restored(env["config"], ["~/src/app", "~/src/web", "~/src/py"], waiting=["~/src/later"])
    before = _snapshot(home)

    status = _status(env)

    assert [(r.cwd, r.kind, r.state) for r in status.rows] == [
        ("~/src/app", "npm", "pending"),
        ("~/src/web", "pnpm", "present_unverified"),
        ("~/src/py", "uv", "pending"),
    ]
    assert status.not_restored == ["~/src/later"]
    assert status.state == "pending"
    assert not env["launched"].exists()
    assert _snapshot(home) == before
    assert not (env["config"] / "network" / "workset-in" / SID / "setup.json").exists()


def test_the_json_document_declares_it_executes_nothing(env: dict[str, Path]) -> None:
    _worktree(env["home"], "src/app", {"package-lock.json": "{}"})
    _restored(env["config"], ["~/src/app"])

    document = _status(env).to_json()

    assert document["executes"] is False and document["ran"] is False
    assert document["schema"] == workset_setup.STATUS_SCHEMA
    assert "run" not in document["rows"][0]  # no recipe shell text in the machine form


# ------------------------------------------------- truthful target state


def test_an_empty_dir_is_pending_and_left_in_place(tmp_path: Path) -> None:
    (tmp_path / "node_modules").mkdir()

    assert workset_setup.target_state(tmp_path / "node_modules") == ("pending", "empty")
    assert workset_setup.target_state(tmp_path / "absent") == ("pending", "absent")
    assert (tmp_path / "node_modules").is_dir()


def test_anything_existing_is_present_unverified_never_ready(tmp_path: Path) -> None:
    """A half-built tree from an interrupted install looks like any other: unverified."""
    partial = tmp_path / "node_modules"
    partial.mkdir()
    (partial / ".partial").write_text("")
    (tmp_path / "file").write_text("x")
    os.symlink(tmp_path / "nowhere", tmp_path / "dangling")

    assert workset_setup.target_state(partial) == ("present_unverified", "exists")
    assert workset_setup.target_state(tmp_path / "file") == ("present_unverified", "not_directory")
    assert workset_setup.target_state(tmp_path / "dangling") == ("present_unverified", "symlink")
    assert os.path.islink(tmp_path / "dangling") and not (tmp_path / "nowhere").exists()


@pytest.mark.skipif(_ROOT_SKIPS_PERMISSIONS, reason="root reads a 000 directory")
def test_an_unreadable_target_is_unknown_not_present(env: dict[str, Path]) -> None:
    """R2/R6: a ``PermissionError`` used to escape (or read as "present")."""
    wt = _worktree(env["home"], "src/app", {"package-lock.json": "{}"})
    locked = wt / "node_modules"
    locked.mkdir()
    (locked / "x").write_text("1")
    locked.chmod(0)
    try:
        assert workset_setup.target_state(locked)[0] == "unknown"
        _restored(env["config"], ["~/src/app"])
        status = _status(env)
    finally:
        locked.chmod(0o755)

    assert [(r.state, r.kind) for r in status.rows] == [("unknown", "npm")]
    assert status.state == "unknown"
    assert (locked / "x").read_text() == "1"


@pytest.mark.skipif(_ROOT_SKIPS_PERMISSIONS, reason="root reads a 000 directory")
def test_an_unreadable_worktree_is_one_unknown_row_and_the_rest_still_report(
    env: dict[str, Path],
) -> None:
    locked = _worktree(env["home"], "src/locked", {"package-lock.json": "{}"})
    _worktree(env["home"], "src/app", {"package-lock.json": "{}"})
    _restored(env["config"], ["~/src/locked", "~/src/app"])
    locked.chmod(0)
    try:
        status = _status(env)
    finally:
        locked.chmod(0o755)

    assert [(r.cwd, r.state) for r in status.rows] == [
        ("~/src/locked", "unknown"),
        ("~/src/app", "pending"),
    ]
    assert status.state == "unknown"


def test_a_refused_placement_is_listed_not_described(env: dict[str, Path]) -> None:
    _worktree(env["home"], ".config/evil", {"package-lock.json": "{}"})
    _restored(env["config"], ["~/.config/evil"])

    status = _status(env)

    assert [(r.state, r.kind) for r in status.rows] == [("refused", "")]
    assert status.state == "pending"  # a refused row is never summarised as done


def test_cwd_is_rechecked_against_the_restore_placement_rules(env: dict[str, Path]) -> None:
    home, config = env["home"], env["config"]
    roots = workset.DEFAULT_RESTORE_ROOTS
    _worktree(home, "src/repo", {})
    _worktree(home, "src/repo/.worktrees/feat", {})
    _worktree(home, ".config/evil", {})
    (home / "elsewhere").mkdir()
    os.symlink(home / "elsewhere", home / "src" / "link")

    assert workset_setup.cwd_refusal(home, config, "~/src/repo", roots) == ""
    assert workset_setup.cwd_refusal(home, config, "~/src/repo/.worktrees/feat", roots) == ""
    for bad in (
        "~/.config/evil",
        "~/src/../.ssh",
        "/etc",
        "~/src/link",
        "~/src/missing",
        "~/src/repo/.worktrees/.hidden",
    ):
        assert workset_setup.cwd_refusal(home, config, bad, roots), bad


def test_manual_and_not_needed_rows_and_overall_states(env: dict[str, Path]) -> None:
    _worktree(env["home"], "src/go", {"go.mod": "module x\n"})
    _restored(env["config"], ["~/src/go"])

    status = _status(env)

    assert {r.state for r in status.rows} <= {"not_needed", "manual"}
    assert status.state in {"nothing_needed", "pending"}
    assert workset_setup.dependency_status(env["config"], env["home"], "../escape") is None


def test_no_arrival_is_none(env: dict[str, Path]) -> None:
    assert workset_setup.dependency_status(env["config"], env["home"], SID) is None


def test_a_missing_package_manager_is_named_by_path_lookup_only(env: dict[str, Path]) -> None:
    (env["bin"] / "npm").unlink()
    _worktree(env["home"], "src/app", {"package-lock.json": "{}"})
    _restored(env["config"], ["~/src/app"])

    (row,) = _status(env).rows

    assert (row.state, row.missing_tool) == ("pending", "npm")
    assert not env["launched"].exists()


# ------------------------------------------------- the arrival note (receiver boundary)


_SETUP_CMD = re.compile(r"lop sessions setup-deps\b[^\n]*")
_SHELLISH = ("rm -rf", "npm ci", "pnpm install", "uv sync", "uv venv", "&&", "ln -s", "--yes")


def _assert_status_only(text: str) -> None:
    """The receiver boundary (#4378): data rows, only ``--status``, an explicit guard."""
    for token in _SHELLISH:
        assert token not in text, token
    commands = _SETUP_CMD.findall(text)
    assert commands, text
    assert all(c.rstrip(". ").endswith(f"setup-deps {SID} --status") for c in commands), commands
    assert "Do not set up dependencies or run a package manager" in text


def test_the_arrival_note_lists_rows_as_data_without_recipe_text() -> None:
    from local_operator.network import workset_probe

    block = {
        "left_behind": [
            {
                "path": "~/src/web/.worktrees/feat/node_modules",
                "recipe": {
                    "kind": "pnpm_share_primary",
                    "run": "rm -rf node_modules && ln -s ../../node_modules node_modules",
                },
            }
        ]
    }
    text = workset_probe.arrival_note(
        block,
        session_id=SID,
        source="laptop",
        destination="desk",
        dependencies={
            "state": "pending",
            "rows": [
                {"cwd": "~/src/app", "kind": "npm", "state": "pending"},
                {"cwd": "~/src/web", "kind": "pnpm", "state": "present_unverified"},
                {"cwd": "~/src/x", "kind": "", "state": "refused"},
                {"cwd": "~/src/assets", "kind": "git_lfs", "state": "manual"},
            ],
            "not_restored": ["~/src/later"],
        },
    )

    assert "- ~/src/app: npm, not set up" in text
    assert "- ~/src/web: pnpm, a folder exists but was not checked" in text
    assert "- ~/src/x: dependencies, not eligible" in text
    assert "- ~/src/assets: git_lfs, left to the user" in text
    assert "- ~/src/later: not restored on this device yet" in text
    assert "kind pnpm_share_primary; not moved" in text
    _assert_status_only(text)


def test_a_failed_status_keeps_the_guard_and_says_unknown() -> None:
    """R3: a failure used to drop the whole paragraph ("nothing pending")."""
    from local_operator.network import workset_probe

    text = workset_probe.arrival_note(
        None,
        session_id=SID,
        source="laptop",
        destination="desk",
        dependencies={"state": "unknown", "rows": [], "not_restored": []},
    )

    assert "Status unknown" in text
    _assert_status_only(text)


def test_no_working_set_here_says_nothing_about_dependencies() -> None:
    from local_operator.network import workset_probe

    text = workset_probe.arrival_note(None, session_id=SID, source="laptop", destination="desk")

    assert "setup-deps" not in text and "Dependency trees" not in text


def _spool(env: dict[str, Path], epoch: str = "e1") -> tuple[str, Any]:
    from types import SimpleNamespace

    from local_operator.network import mobility
    from local_operator.session.runtime.inbox import peek_inbox

    (env["config"] / "sessions" / SID).mkdir(parents=True, exist_ok=True)
    # The note path reads only ``root``, ``workset_home`` and ``_own_label``.
    server: Any = SimpleNamespace(
        root=env["config"], workset_home=str(env["home"]), _own_label=lambda: "desk"
    )
    state = mobility._arrival_resume_note(
        server, SID, lease_epoch=epoch, block=None, source="laptop"
    )
    again = mobility._arrival_resume_note(
        server, SID, lease_epoch=epoch, block=None, source="laptop"
    )
    assert again == "already_delivered"
    return state, peek_inbox(env["config"] / "sessions" / SID)


def test_the_spooled_note_carries_this_devices_status_once_and_launches_nothing(
    env: dict[str, Path],
) -> None:
    wt = _worktree(env["home"], "src/app", {"package-lock.json": "{}"})
    present = _worktree(env["home"], "src/done", {"package-lock.json": "{}"})
    (present / "node_modules").mkdir()
    (present / "node_modules" / "x.js").write_text("1\n")
    _restored(env["config"], ["~/src/app", "~/src/done"])

    state, rows = _spool(env)

    assert state == "delivered"
    (row,) = rows
    assert "- ~/src/app: npm, not set up" in row.text
    assert "- ~/src/done: npm, a folder exists but was not checked" in row.text
    _assert_status_only(row.text)
    assert not env["launched"].exists() and not (wt / "node_modules").exists()


@pytest.mark.skipif(_ROOT_SKIPS_PERMISSIONS, reason="root reads a 000 directory")
def test_an_unreadable_tree_at_arrival_still_spools_the_guard(env: dict[str, Path]) -> None:
    """R3 probe B, unmocked: an unreadable dependency tree at arrival time."""
    wt = _worktree(env["home"], "src/app", {"package-lock.json": "{}"})
    (wt / "node_modules").mkdir()
    (wt / "node_modules").chmod(0)
    _restored(env["config"], ["~/src/app"])
    try:
        state, rows = _spool(env)
    finally:
        (wt / "node_modules").chmod(0o755)

    assert state == "delivered"
    (row,) = rows
    assert "- ~/src/app: npm, status unknown" in row.text
    _assert_status_only(row.text)
    assert not env["launched"].exists()


# ------------------------------------------- S2-R2: the note binds to THIS move


def test_an_earlier_moves_record_never_speaks_for_this_arrival(
    env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """S2-R2: a return that carried no working set leaves the earlier move's record in
    ``workset-in/<sid>``. The note for THIS move (epoch e2) must not present that
    record's checkout as this arrival's — and must not even inspect its paths."""
    _worktree(env["home"], "src/old-checkout", {"package-lock.json": "{}"})
    _restored(env["config"], ["~/src/old-checkout"], waiting=["~/src/old-later"], epoch="e1")

    def no_path_inspection(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("an earlier move's worktree was inspected for this arrival")

    monkeypatch.setattr(workset_setup, "_worktree_rows", no_path_inspection)

    state, rows = _spool(env, epoch="e2")

    assert state == "delivered"
    (row,) = rows
    assert "old-checkout" not in row.text and "old-later" not in row.text
    assert "Dependency trees" not in row.text and "setup-deps" not in row.text
    assert "No working files were recorded" in row.text
    assert not env["launched"].exists()


@pytest.mark.parametrize("current", ["", "e2"], ids=["no-epoch", "other-epoch"])
def test_the_bound_reader_returns_nothing_for_another_or_no_epoch(
    env: dict[str, Path], current: str
) -> None:
    """An empty epoch never matches (no manufactured binding); the record is untouched."""
    _worktree(env["home"], "src/app", {"package-lock.json": "{}"})
    directory = _restored(env["config"], ["~/src/app"], epoch="e1")
    before = (directory / "pending.json").read_bytes()

    got = workset_setup.dependency_status(env["config"], env["home"], SID, lease_epoch=current)

    assert got is None
    assert (directory / "pending.json").read_bytes() == before


def test_the_same_moves_record_is_reported_and_the_standalone_query_names_its_epoch(
    env: dict[str, Path],
) -> None:
    _worktree(env["home"], "src/app", {"package-lock.json": "{}"})
    _restored(env["config"], ["~/src/app"], epoch="e1")

    bound = workset_setup.dependency_status(env["config"], env["home"], SID, lease_epoch="e1")
    standalone = workset_setup.dependency_status(env["config"], env["home"], SID)

    assert bound is not None and standalone is not None
    assert [(r.cwd, r.state) for r in bound.rows] == [("~/src/app", "pending")]
    assert standalone.to_json()["restore_record"] == {"state": "read", "lease_epoch": "e1"}


_DAMAGED: dict[str, bytes] = {
    "bad-json": b"{bad",
    "bad-utf8": b"\xff\xfe{}",
    "not-an-object": b"[]",
    "plans-not-a-list": b'{"lease_epoch": "e1", "plans": "x", "restored": {}}',
    "plans-missing": b'{"lease_epoch": "e1", "restored": {}}',
    "plan-not-a-mapping": b'{"lease_epoch": "e1", "plans": ["x"], "restored": {}}',
    "restored-not-a-mapping": b'{"lease_epoch": "e1", "plans": [{"n": 0}], "restored": []}',
    "epoch-not-a-string": b'{"lease_epoch": 5, "plans": [{"n": 0}], "restored": {}}',
}


@pytest.mark.parametrize("raw", list(_DAMAGED.values()), ids=list(_DAMAGED))
@pytest.mark.parametrize("bound", [None, "e1"], ids=["standalone", "arrival"])
def test_a_damaged_record_is_unknown_never_absent_and_never_a_traceback(
    env: dict[str, Path], raw: bytes, bound: str | None
) -> None:
    """S7-R2 / R2-1 / U10 / D13: unreadable is not "no working set arrived"."""
    directory = env["config"] / "network" / "workset-in" / SID
    directory.mkdir(parents=True)
    (directory / "pending.json").write_bytes(raw)

    # The standalone call keeps the pre-S2-R2 signature, so it discriminates on 71cbd4d.
    kwargs = {} if bound is None else {"lease_epoch": bound}
    got = workset_setup.dependency_status(env["config"], env["home"], SID, **kwargs)

    assert got is not None
    assert (got.state, got.rows, got.not_restored) == ("unknown", [], [])
    assert got.to_json()["restore_record"]["state"] == "unreadable"
    assert (directory / "pending.json").read_bytes() == raw


@pytest.mark.skipif(_ROOT_SKIPS_PERMISSIONS, reason="root reads a 000 file")
def test_an_unreadable_record_file_is_unknown(env: dict[str, Path]) -> None:
    directory = _restored(env["config"], ["~/src/app"])
    (directory / "pending.json").chmod(0)
    try:
        got = workset_setup.dependency_status(env["config"], env["home"], SID)
    finally:
        (directory / "pending.json").chmod(0o600)

    assert got is not None and got.state == "unknown" and got.rows == []


def test_a_damaged_record_at_arrival_keeps_the_guard_and_names_no_path(
    env: dict[str, Path],
) -> None:
    directory = env["config"] / "network" / "workset-in" / SID
    directory.mkdir(parents=True)
    (directory / "pending.json").write_text('{"plans": ["~/src/old"]', encoding="utf-8")

    state, rows = _spool(env, epoch="e2")

    assert state == "delivered"
    (row,) = rows
    assert "Status unknown" in row.text and "~/src/old" not in row.text
    _assert_status_only(row.text)
    assert not env["launched"].exists()
