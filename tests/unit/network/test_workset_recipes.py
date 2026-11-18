"""Dependency trees are left behind with a printed recreation recipe (design §9).

Recipes are listed, never run (lifecycle scripts execute on install), and they
follow the dependency-commons rule: a ``.worktrees/<name>`` whose pnpm lockfile
matches the primary's shares the primary's tree instead of installing another;
a Python venv is recreated, never symlinked.
"""

from __future__ import annotations

from pathlib import Path

from local_operator.network import workset_transfer as wt


def _tree(tmp_path: Path, files: dict[str, str]) -> tuple[Path, Path]:
    primary = tmp_path / "proj"
    worktree = primary / ".worktrees" / "feat"
    worktree.mkdir(parents=True)
    for name, body in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    return worktree, primary


def test_pnpm_worktree_matching_primary_shares_its_tree(tmp_path: Path) -> None:
    wt_dir, primary = _tree(
        tmp_path, {"proj/pnpm-lock.yaml": "L", "proj/.worktrees/feat/pnpm-lock.yaml": "L"}
    )
    (primary / "node_modules").mkdir()
    (recipe,) = wt.recipes_for(wt_dir, primary, home=tmp_path)
    assert recipe == {
        "kind": "pnpm_share_primary",
        "run": "rm -rf node_modules && ln -s ../../node_modules node_modules",
        "cwd": "~/proj/.worktrees/feat",
    }


def test_pnpm_with_a_different_lockfile_installs_frozen(tmp_path: Path) -> None:
    wt_dir, primary = _tree(
        tmp_path, {"proj/pnpm-lock.yaml": "L1", "proj/.worktrees/feat/pnpm-lock.yaml": "L2"}
    )
    (primary / "node_modules").mkdir()
    assert [r["run"] for r in wt.recipes_for(wt_dir, primary)] == ["pnpm install --frozen-lockfile"]


def test_npm_yarn_uv_and_requirements(tmp_path: Path) -> None:
    wt_dir, primary = _tree(
        tmp_path,
        {
            "proj/.worktrees/feat/package-lock.json": "{}",
            "proj/.worktrees/feat/yarn.lock": "",
            "proj/.worktrees/feat/.yarnrc.yml": "",
            "proj/.worktrees/feat/uv.lock": "",
            "proj/.worktrees/feat/requirements-dev.txt": "",
        },
    )
    runs = {r["kind"]: r["run"] for r in wt.recipes_for(wt_dir, primary)}
    assert runs == {
        "npm": "npm ci",
        "yarn": "yarn install --immutable",
        "uv": "uv venv && uv sync --frozen",
    }
    (wt_dir / "uv.lock").unlink()
    runs = {r["kind"]: r["run"] for r in wt.recipes_for(wt_dir, primary)}
    assert runs["uv_pip"] == "uv venv && uv pip install -r requirements-dev.txt"


def test_npm_is_never_suggested_in_a_pnpm_repo_and_cargo_needs_nothing(tmp_path: Path) -> None:
    wt_dir, primary = _tree(
        tmp_path,
        {
            "proj/.worktrees/feat/pnpm-lock.yaml": "L",
            "proj/.worktrees/feat/package-lock.json": "{}",
        },
    )
    assert [r["kind"] for r in wt.recipes_for(wt_dir, primary)] == ["pnpm"]
    other, primary2 = _tree(tmp_path / "rust", {"proj/.worktrees/feat/Cargo.lock": ""})
    assert [(r["kind"], r["run"]) for r in wt.recipes_for(other, primary2)] == [("none", "")]
