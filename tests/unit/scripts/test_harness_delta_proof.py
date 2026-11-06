"""``scripts/harness_delta_proof.py``: HARNESS-ONLY only when no byte the
shipping pipeline consumes differs; every other delta opens a review round.

The operator's non-negotiable is the negative control
(``test_negative_control_one_product_byte_under_harness_churn``): an exemption
that cannot be shown to refuse a real product change is worse than the round
inflation it removes (local-operator-mobile PR #156: 40 rounds, product done by
round 3). Every case builds a tiny scratch repository and drives the script as a
subprocess, exactly as its Run line documents, asserting the exit code AND the
named reason — a right exit code for the wrong reason is a coincidence, not a
proof. Git runs with ``GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_SYSTEM=/dev/null
GIT_TERMINAL_PROMPT=0`` (AGENTS.md keychain rule; see test_rebase_proof.py).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "harness_delta_proof.py"

#: A mobile-shaped manifest: the trap PR #156 exposed is that release.yml runs
#: scripts under the harness directory, so a directory rule alone is unsafe.
MANIFEST = """\
# test manifest
harness: scripts/ci/**
harness: src/testing/**
harness: e2e/**
harness: .github/workflows/**
ship: .github/workflows/release.yml
ship: app/**
ship: src/**
ship: package.json
"""


def _env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("GIT_", "CMUX_"))}
    env.update(
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_SYSTEM="/dev/null",
        GIT_CONFIG_GLOBAL="/dev/null",
        GIT_TERMINAL_PROMPT="0",
        GIT_AUTHOR_NAME="t",
        GIT_AUTHOR_EMAIL="t@example.invalid",
        GIT_COMMITTER_NAME="t",
        GIT_COMMITTER_EMAIL="t@example.invalid",
    )
    return env


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], env=_env(), check=True, capture_output=True, text=True
    ).stdout.strip()


def _commit(repo: Path, files: dict[str, str | None], message: str) -> str:
    """Write (or delete, for ``None``) files and commit; returns the new SHA."""
    for name, body in files.items():
        path = repo / name
        if body is None:
            path.unlink()
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "--allow-empty", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _setup(tmp_path: Path, manifest: str | None = MANIFEST) -> tuple[Path, str]:
    """main with product, ship and harness files; a ``pr`` branch whose first
    commit is the reviewed (clean-round) head R. Returns (repo, R)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    files: dict[str, str | None] = {
        "app/main.ts": "import { x } from '../src/lib';\nexport const app = x;\n",
        "src/lib.ts": "export const x = 1;\n",
        "src/testing/helper.ts": "export const h = 1;\n",
        "scripts/ci/check.ts": "console.log('evidence only');\n",
        "scripts/ci/version.ts": "console.log('1.0.0');\n",
        ".github/workflows/release.yml": "steps:\n  - run: bun scripts/ci/version.ts\n",
        ".github/workflows/native.yml": "steps:\n  - run: bun scripts/ci/check.ts\n",
        "package.json": '{"name": "m"}\n',
        "README.md": "readme\n",
    }
    if manifest is not None:
        files[".github/harness-paths"] = manifest
    _commit(repo, files, "base")
    _git(repo, "checkout", "-q", "-b", "pr")
    reviewed = _commit(repo, {"src/lib.ts": "export const x = 2;\n"}, "feat: product change")
    return repo, reviewed


def _run(repo: Path, reviewed: str, head: str, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            reviewed,
            head,
            "--base",
            "main",
            "--repo",
            str(repo),
            *extra,
        ],
        env=_env(),
        capture_output=True,
        text=True,
        check=False,
    )


def test_new_harness_file_only_is_harness_only(tmp_path: Path) -> None:
    repo, reviewed = _setup(tmp_path)
    head = _commit(repo, {"e2e/new-probe.ts": "probe\n"}, "test(e2e): add probe")
    result = _run(repo, reviewed, head)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "HARNESS-ONLY" in result.stdout
    assert "`e2e/new-probe.ts` — harness" in result.stdout


@pytest.mark.parametrize(
    ("product", "kind"),
    [("src/lib.ts", "ship"), ("lib/core.ts", "product")],
    ids=["declared-ship", "undeclared-product"],
)
def test_negative_control_one_product_byte_under_harness_churn(
    tmp_path: Path, product: str, kind: str
) -> None:
    """The operator's non-negotiable: 300 harness lines must not launder one
    product byte. Run over a ``ship:``-declared file AND a file no glob names,
    so the product-path check is shown to hold on its own, not only behind the
    ship-closure check (a mutation that disables either one fails a case)."""
    repo, _ = _setup(tmp_path)
    reviewed = _commit(repo, {"lib/core.ts": "export const y = 1;\n"}, "feat: core")
    churn = "".join(f"// harness line {i}\n" for i in range(300))
    before = (repo / product).read_bytes()
    after = before.replace(b"= 1;", b"= 3;") if b"= 1;" in before else before.replace(b"2", b"3")
    assert len(before) == len(after) and sum(a != b for a, b in zip(before, after)) == 1
    head = _commit(
        repo,
        {"e2e/churn.ts": churn, product: after.decode()},
        "test(e2e): churn plus one product byte",
    )
    numstat = _git(repo, "diff", "--numstat", reviewed, head).splitlines()
    assert sorted(line.split("\t")[2] for line in numstat) == sorted(["e2e/churn.ts", product])
    result = _run(repo, reviewed, head)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "ROUND REQUIRED" in result.stdout
    assert f"product path changed: `{product}` ({kind})" in result.stdout
    assert "e2e/churn.ts" not in result.stdout.split("Verdict")[1]  # names the product file only


def test_harness_file_named_by_ship_workflow_is_ship_reachable(tmp_path: Path) -> None:
    """release.yml runs scripts/ci/version.ts: it is under a harness glob yet
    alters the shipped artifact — the PR #156 trap."""
    repo, reviewed = _setup(tmp_path)
    head = _commit(repo, {"scripts/ci/version.ts": "console.log('2.0.0');\n"}, "ci: bump")
    result = _run(repo, reviewed, head)
    assert result.returncode == 1, result.stdout
    assert "ship-reachable: `scripts/ci/version.ts`" in result.stdout
    assert "named by `.github/workflows/release.yml`" in result.stdout


def test_evidence_only_harness_script_stays_harness_only(tmp_path: Path) -> None:
    """Counterpart to the trap: check.ts is run only by an evidence workflow."""
    repo, reviewed = _setup(tmp_path)
    head = _commit(repo, {"scripts/ci/check.ts": "console.log('more');\n"}, "ci: probe")
    result = _run(repo, reviewed, head)
    assert result.returncode == 0, result.stdout


def test_transitive_import_chain_is_ship_reachable(tmp_path: Path) -> None:
    repo, _ = _setup(tmp_path)
    reviewed = _commit(
        repo,
        {
            "src/boot.ts": "import { a } from './testing/a';\nexport const boot = a;\n",
            "src/testing/a.ts": "import { b } from './b';\nexport const a = b;\n",
            "src/testing/b.ts": "export const b = 1;\n",
        },
        "feat: boot imports testing helper",
    )
    head = _commit(repo, {"src/testing/b.ts": "export const b = 2;\n"}, "test: tweak b")
    result = _run(repo, reviewed, head)
    assert result.returncode == 1, result.stdout
    assert "ship-reachable: `src/testing/b.ts`" in result.stdout
    assert "named by `src/testing/a.ts`" in result.stdout


def test_dynamic_template_literal_reference_is_ship_reachable(tmp_path: Path) -> None:
    repo, _ = _setup(tmp_path)
    reviewed = _commit(
        repo,
        {
            "app/fetcher.ts": "const m = await import(`${root}testing/cookie-jar-fetch.ts`);\n",
            "src/testing/cookie-jar-fetch.ts": "export const f = 1;\n",
        },
        "feat: dynamic import",
    )
    head = _commit(repo, {"src/testing/cookie-jar-fetch.ts": "export const f = 2;\n"}, "t")
    result = _run(repo, reviewed, head)
    assert result.returncode == 1, result.stdout
    assert "ship-reachable: `src/testing/cookie-jar-fetch.ts`" in result.stdout


def test_manifest_change_opens_a_round(tmp_path: Path) -> None:
    """The manifest is product: widening the exemption set needs review."""
    repo, reviewed = _setup(tmp_path)
    head = _commit(
        repo, {".github/harness-paths": MANIFEST + "harness: src/**\n"}, "chore: widen harness"
    )
    result = _run(repo, reviewed, head)
    assert result.returncode == 1, result.stdout
    assert "manifest changed" in result.stdout


def test_base_moved_by_merging_main_opens_a_round(tmp_path: Path) -> None:
    repo, reviewed = _setup(tmp_path)
    _git(repo, "checkout", "-q", "main")
    _commit(repo, {"README.md": "readme 2\n"}, "docs: upstream")
    _git(repo, "checkout", "-q", "pr")
    _git(repo, "merge", "-q", "--no-edit", "main")
    head = _commit(repo, {"e2e/probe.ts": "p\n"}, "test: probe")
    result = _run(repo, reviewed, head)
    assert result.returncode == 1, result.stdout
    assert "base moved" in result.stdout


def test_missing_manifest_is_exit_two(tmp_path: Path) -> None:
    repo, reviewed = _setup(tmp_path, manifest=None)
    head = _commit(repo, {"e2e/probe.ts": "p\n"}, "test: probe")
    result = _run(repo, reviewed, head)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "no manifest: every delta is a round" in result.stderr


def test_unknown_ref_is_exit_two(tmp_path: Path) -> None:
    repo, reviewed = _setup(tmp_path)
    result = _run(repo, reviewed, "deadbeef" * 5)
    assert result.returncode == 2
    assert "could not be computed" in result.stderr


def test_product_deletion_opens_a_round(tmp_path: Path) -> None:
    repo, reviewed = _setup(tmp_path)
    head = _commit(repo, {"README.md": None}, "docs: drop readme")
    result = _run(repo, reviewed, head)
    assert result.returncode == 1, result.stdout
    assert "product path changed: `README.md`" in result.stdout


def test_rename_product_to_harness_counts_both_sides(tmp_path: Path) -> None:
    repo, reviewed = _setup(tmp_path)
    (repo / "e2e").mkdir()
    _git(repo, "mv", "README.md", "e2e/README.md")
    _git(repo, "commit", "-q", "-m", "move")
    head = _git(repo, "rev-parse", "HEAD")
    result = _run(repo, reviewed, head)
    assert result.returncode == 1, result.stdout
    assert "product path changed: `README.md`" in result.stdout


def test_mode_only_change_on_product_file_opens_a_round(tmp_path: Path) -> None:
    repo, reviewed = _setup(tmp_path)
    _git(repo, "update-index", "--chmod=+x", "README.md")
    _git(repo, "commit", "-q", "-m", "chmod")
    head = _git(repo, "rev-parse", "HEAD")
    assert _git(repo, "diff", "--name-only", reviewed, head) == "README.md"
    result = _run(repo, reviewed, head)
    assert result.returncode == 1, result.stdout
    assert "product path changed: `README.md`" in result.stdout


def test_external_manifest_replay_is_labelled(tmp_path: Path) -> None:
    repo, reviewed = _setup(tmp_path, manifest=None)
    head = _commit(repo, {"e2e/probe.ts": "p\n"}, "test: probe")
    external = tmp_path / "manifest"
    external.write_text(MANIFEST)
    result = _run(repo, reviewed, head, "--manifest", str(external))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "external (replay)" in result.stdout


def test_split_numbers_on_known_fixture(tmp_path: Path) -> None:
    """vs merge-base: product = src/lib.ts (1 added) + app/x.ts (2);
    harness = e2e/a.ts (4) → 4 > 3 → SPLIT."""
    repo, reviewed = _setup(tmp_path)
    head = _commit(repo, {"app/x.ts": "a\nb\n", "e2e/a.ts": "1\n2\n3\n4\n"}, "feat: both")
    result = _run(repo, reviewed, head, "--split")
    assert "Added lines vs merge-base: product 3, harness 4" in result.stdout
    assert "SPLIT" in result.stdout and "NO SPLIT" not in result.stdout
