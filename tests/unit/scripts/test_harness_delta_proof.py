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

from scripts.harness_delta_proof import Manifest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "harness_delta_proof.py"

#: A mobile-shaped manifest: the trap PR #156 exposed is that release.yml runs
#: scripts under the harness directory, so a directory rule alone is unsafe.
MANIFEST = """\
# test manifest
harness: scripts/ci/**
harness: src/testing/**
harness: e2e/**
harness: .github/workflows/native.yml
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


# --- round-1 findings: each attack reproduced first, then refused -------------
# Every case below is a constructor the proof must not read as "harness only".
# They fail closed by design: an input the proof cannot classify with certainty
# is a round, and an input it cannot read at all is exit 2 (never a pass).

GITHUB_EXACT = MANIFEST.replace(
    "harness: .github/workflows/native.yml", "harness: .github/workflows/native.yml"
)


def test_gitlink_bump_under_a_ship_path_opens_a_round(tmp_path: Path, monkeypatch) -> None:
    """R1: `.gitmodules ignore = all` used to hide the whole diff entry."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    sub = "1111111111111111111111111111111111111111"
    _commit(
        repo,
        {
            ".github/harness-paths": MANIFEST,
            "src/lib.ts": "export const x = 1;\n",
            ".gitmodules": '[submodule "src/vendor"]\n\tpath = src/vendor\n\tignore = all\n',
        },
        "base",
    )
    _git(repo, "update-index", "--add", "--cacheinfo", f"160000,{sub},src/vendor")
    _git(repo, "commit", "-q", "-m", "add gitlink")
    reviewed = _git(repo, "rev-parse", "HEAD")
    bumped = "2222222222222222222222222222222222222222"
    _git(repo, "update-index", "--add", "--cacheinfo", f"160000,{bumped},src/vendor")
    _git(repo, "commit", "-q", "-m", "bump gitlink")
    head = _git(repo, "rev-parse", "HEAD")
    assert _git(repo, "diff", "--name-only", reviewed, head) == ""  # the trap
    result = _run(repo, reviewed, head)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "src/vendor" in result.stdout and "gitlink" in result.stdout


def test_newline_in_a_ship_path_is_exit_two(tmp_path: Path) -> None:
    """R2: `**` compiles to `.*`, so without DOTALL a newline path was invisible.

    It classified `product`, never seeded the closure, and the payload it ran left
    the closure — a fail-open one layer above the batch protocol. With DOTALL the
    glob matches, and a newline/CR path anywhere the proof must classify is exit 2
    rather than a verdict it cannot defend.
    """
    manifest = Manifest.parse(MANIFEST, "unit")
    assert manifest.classify("src/a\nb.py") == "ship"

    repo, reviewed = _setup(tmp_path)
    head = _commit(
        repo,
        {
            "src/a\nb.py": "open('scripts/ci/payload.sh').read()\n",
            "scripts/ci/payload.sh": "echo 1\n",
        },
        "feat: newline path",
    )
    result = _run(repo, reviewed, head)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "newline" in result.stderr


def test_newline_ship_path_at_head_is_exit_two_even_when_unchanged(tmp_path: Path) -> None:
    """The ship SET is checked too, not only paths the delta touches."""
    repo, _ = _setup(tmp_path)
    reviewed = _commit(repo, {"src/a\nb.py": "print(1)\n"}, "feat: newline path")
    head = _commit(repo, {"scripts/ci/payload.sh": "echo 2\n"}, "ci: payload")
    result = _run(repo, reviewed, head)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "newline" in result.stderr


def test_ship_symlink_to_a_harness_directory_pulls_its_contents(tmp_path: Path) -> None:
    """R3: a ship symlink into a HARNESS directory hid every file under it.

    The payload sits under `src/testing/**` (harness) and the symlink under
    `app/**` (ship): with the payload under a product path the test passed for the
    wrong reason, which round 2 called out.
    """
    repo, _ = _setup(tmp_path)
    reviewed = _commit(
        repo,
        {"src/testing/zzpay/PAYLOAD.md": "p1\n", "app/main.ts": "export const a = 1;\n"},
        "feat: fixture",
    )
    (repo / "app").mkdir(exist_ok=True)
    (repo / "app" / "zzpay").symlink_to("../src/testing/zzpay")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "feat: ship symlink")
    reviewed = _git(repo, "rev-parse", "HEAD")
    head = _commit(repo, {"src/testing/zzpay/PAYLOAD.md": "p2\n"}, "chore: payload")
    result = _run(repo, reviewed, head)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "src/testing/zzpay/PAYLOAD.md" in result.stdout
    assert "harness" in result.stdout.split("src/testing/zzpay/PAYLOAD.md")[1][:40]


def test_ship_symlink_that_escapes_the_repo_opens_a_round(tmp_path: Path) -> None:
    repo, _ = _setup(tmp_path)
    (repo / "src" / "escape").symlink_to("/etc/hosts")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "feat: escaping symlink")
    reviewed = _git(repo, "rev-parse", "HEAD")
    head = _commit(repo, {"e2e/probe.ts": "p\n"}, "test: probe")
    result = _run(repo, reviewed, head)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "symlink" in result.stdout


def test_wildcard_harness_glob_under_github_is_rejected(tmp_path: Path) -> None:
    """Q1/M3: a new workflow must not be exemptable by a wildcard."""
    repo, reviewed = _setup(
        tmp_path,
        manifest=MANIFEST.replace(
            "harness: .github/workflows/native.yml", "harness: .github/workflows/**"
        ),
    )
    head = _commit(repo, {"e2e/probe.ts": "p\n"}, "test: probe")
    result = _run(repo, reviewed, head)
    assert result.returncode == 2, result.stdout + result.stderr
    assert ".github" in result.stderr and "exact" in result.stderr


def test_added_workflow_is_product_with_exact_filenames(tmp_path: Path) -> None:
    repo, reviewed = _setup(tmp_path)
    head = _commit(
        repo,
        {".github/workflows/new-release.yml": "on: release\npermissions:\n  contents: write\n"},
        "ci: new workflow",
    )
    result = _run(repo, reviewed, head)
    assert result.returncode == 1, result.stdout + result.stderr
    assert ".github/workflows/new-release.yml" in result.stdout


def test_harness_test_file_guarding_ship_code_is_product(tmp_path: Path) -> None:
    """Q-M1: a test under a harness glob that asserts on shipped code is product."""
    repo, reviewed = _setup(tmp_path)
    head = _commit(
        repo,
        {"e2e/lib.spec.ts": "import { x } from '../src/lib';\nexpect(x).toBe(1);\n"},
        "test(e2e): assert ship code",
    )
    result = _run(repo, reviewed, head)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "e2e/lib.spec.ts" in result.stdout
    # A harness test that names nothing shipped stays exempt. Branch from the
    # reviewed head, not from `head`: a delta measured from `reviewed` would
    # still carry `lib.spec.ts`, which is product by the rule under test.
    _git(repo, "checkout", "-q", "--detach", reviewed)
    clean = _commit(repo, {"e2e/plain.spec.ts": "expect(1).toBe(1);\n"}, "test(e2e): plain")
    assert _run(repo, reviewed, clean).returncode == 0


def test_tsconfig_alias_reference_is_ship_reachable(tmp_path: Path) -> None:
    """M1/Q3: `@/testing/x` resolves through tsconfig paths, not the token."""
    repo, _ = _setup(tmp_path)
    reviewed = _commit(
        repo,
        {
            "tsconfig.json": '{"compilerOptions": {"baseUrl": ".", "paths": {"@/*": ["src/*"]}}}\n',
            "src/boot.ts": "import { h } from '@/testing/helper';\nexport const b = h;\n",
        },
        "feat: alias import",
    )
    head = _commit(repo, {"src/testing/helper.ts": "export const h = 2;\n"}, "test: tweak")
    result = _run(repo, reviewed, head)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "src/testing/helper.ts" in result.stdout


def test_alias_specifier_matches_by_suffix_without_tsconfig(tmp_path: Path) -> None:
    repo, _ = _setup(tmp_path)
    reviewed = _commit(
        repo,
        {
            "src/boot2.ts": "const m = await import('@/testing/helper');\nexport default m;\n",
            "src/testing/helper.ts": "export const h = 1;\n",
        },
        "feat: suffix import",
    )
    head = _commit(repo, {"src/testing/helper.ts": "export const h = 2;\n"}, "test: tweak")
    result = _run(repo, reviewed, head)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "src/testing/helper.ts" in result.stdout


def test_python_absolute_import_of_harness_module_is_ship_reachable(tmp_path: Path) -> None:
    """M2: `import helper` reaches a harness `helper.py` conservatively."""
    repo, _ = _setup(tmp_path)
    reviewed = _commit(
        repo,
        {"src/app.py": "import helper\n\nprint(helper)\n", "src/testing/helper.py": "v = 1\n"},
        "feat: absolute import",
    )
    head = _commit(repo, {"src/testing/helper.py": "v = 2\n"}, "chore: tweak")
    result = _run(repo, reviewed, head)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "src/testing/helper.py" in result.stdout


def test_dot_directory_reference_and_directory_spelling_are_matched(tmp_path: Path) -> None:
    """Q2/R4: `raw.strip('./')` turned `.ci/probe.sh` into `ci/probe.sh`."""
    repo, _ = _setup(
        tmp_path,
        manifest=MANIFEST + "harness: .ci/**\nship: .ci/probe.sh\n",
    )
    reviewed = _commit(
        repo,
        {"src/runner.py": "CMD = '.ci/probe.sh'\n", ".ci/probe.sh": "echo 1\n"},
        "feat: dot dir",
    )
    head = _commit(repo, {".ci/probe.sh": "echo 2\n"}, "chore: dot dir probe")
    result = _run(repo, reviewed, head)
    assert result.returncode == 1, result.stdout + result.stderr
    assert ".ci/probe.sh" in result.stdout

    (tmp_path / "b").mkdir()
    repo2, _ = _setup(tmp_path / "b", manifest=MANIFEST + "harness: .ci/**\n")
    reviewed2 = _commit(
        repo2,
        {"src/runner2.py": "CMD = 'bash .ci/'\n", ".ci/probe.sh": "echo 1\n"},
        "feat: dot dir",
    )
    head2 = _commit(repo2, {".ci/probe.sh": "echo 2\n"}, "chore: dot dir probe")
    result2 = _run(repo2, reviewed2, head2)
    assert result2.returncode == 1, result2.stdout + result2.stderr
    assert ".ci/probe.sh" in result2.stdout


def test_gitlab_manifest_path_is_found(tmp_path: Path) -> None:
    repo, reviewed = _setup(tmp_path, manifest=None)
    _commit(repo, {".gitlab/harness-paths": MANIFEST}, "chore: gitlab manifest")
    reviewed = _commit(repo, {"src/lib.ts": "export const x = 4;\n"}, "feat: product")
    head = _commit(repo, {"e2e/probe.ts": "p\n"}, "test: probe")
    result = _run(repo, reviewed, head)
    assert result.returncode == 0, result.stdout + result.stderr
    assert ".gitlab/harness-paths" in result.stdout


def test_output_states_the_invariant_and_the_fail_closed_rules(tmp_path: Path) -> None:
    repo, reviewed = _setup(tmp_path)
    head = _commit(repo, {"e2e/probe.ts": "p\n"}, "test: probe")
    out = _run(repo, reviewed, head).stdout
    assert "no byte the shipping pipeline consumes differs" in out
    assert "fail-closed" in out.lower()


def test_deleted_harness_test_is_judged_not_crashed(tmp_path: Path) -> None:
    """A deletion carries the all-zero object id on the head side.

    Reading that id made the proof exit 2 ("object 000… is missing") instead of
    returning a verdict — found replaying PR #156 rounds 27 and 36. A deleted
    test that guarded shipped code is product; one that guarded nothing is
    harness.
    """
    repo, _ = _setup(tmp_path)
    reviewed = _commit(
        repo,
        {
            "e2e/guarded.spec.ts": "import { x } from '../src/lib';\nexpect(x).toBe(1);\n",
            "e2e/plain.spec.ts": "expect(1).toBe(1);\n",
        },
        "test(e2e): two specs",
    )
    head = _commit(repo, {"e2e/guarded.spec.ts": None}, "test(e2e): drop the guard")
    result = _run(repo, reviewed, head)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "guards ship file" in result.stdout

    (tmp_path / "b").mkdir()
    repo2, reviewed2 = _setup(tmp_path / "b")
    reviewed2 = _commit(repo2, {"e2e/plain.spec.ts": "expect(1).toBe(1);\n"}, "test(e2e): plain")
    head2 = _commit(repo2, {"e2e/plain.spec.ts": None}, "test(e2e): drop plain")
    result2 = _run(repo2, reviewed2, head2)
    assert result2.returncode == 0, result2.stdout + result2.stderr


def test_manifest_unknown_keyword_is_exit_two(tmp_path: Path) -> None:
    """`gate:` was removed (round-2 decision): a manifest carrying it is exit 2."""
    repo, reviewed = _setup(tmp_path, manifest=MANIFEST + "gate: Makefile\n")
    head = _commit(repo, {"e2e/probe.ts": "p\n"}, "test: probe")
    result = _run(repo, reviewed, head)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "unknown keyword" in result.stderr


def test_bare_directory_word_is_not_a_path(tmp_path: Path) -> None:
    """A word is not a path: only a directory SPELLED as one pulls its contents.

    publish.yml's prose `scripts` pulled all 327 files under scripts/ into the
    closure and made every probe ship-reachable (measured round 2). The narrowing
    is independent of the removed gate tier, so it stays.
    """
    repo, _ = _setup(tmp_path)
    reviewed = _commit(
        repo,
        {"src/notes.py": "# the scripts below are evidence only\n", "scripts/ci/probe.ts": "x\n"},
        "feat: prose mention",
    )
    head = _commit(repo, {"scripts/ci/probe.ts": "y\n"}, "ci: probe")
    result = _run(repo, reviewed, head)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "HARNESS-ONLY" in result.stdout

    # The same mention spelled as a directory does pull the contents.
    (tmp_path / "b").mkdir()
    repo2, _ = _setup(tmp_path / "b")
    reviewed2 = _commit(
        repo2,
        {"src/runner3.py": "CMD = 'bash scripts/ci/'\n", "scripts/ci/probe.ts": "x\n"},
        "feat: directory spelling",
    )
    head2 = _commit(repo2, {"scripts/ci/probe.ts": "y\n"}, "ci: probe")
    result2 = _run(repo2, reviewed2, head2)
    assert result2.returncode == 1, result2.stdout + result2.stderr
    assert "scripts/ci/probe.ts" in result2.stdout


# --- shared helpers for the #3540 cases -----------------------------------------


def _reach(tmp_path: Path, ship: dict[str, str | None], harness_path: str = "scripts/ci/check.ts"):
    """Commit ``ship`` files on top of the reviewed head, then change one harness
    file; return the proof run for that harness-only delta."""
    repo, _ = _setup(tmp_path)
    reviewed = _commit(repo, ship, "feat: ship files")
    head = _commit(repo, {harness_path: "console.log('changed');\n"}, "ci: harness edit")
    return _run(repo, reviewed, head)


def _exempt(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 0, result.stdout + result.stderr
    assert "HARNESS-ONLY" in result.stdout


def _reached(result: subprocess.CompletedProcess[str], path: str = "scripts/ci/check.ts") -> None:
    assert result.returncode == 1, result.stdout + result.stderr
    assert f"ship-reachable: `{path}`" in result.stdout


# --- `uses: ./<dir>` -----------------------------------------------------------------
# A local action is a DIRECTORY holding action.yml, and the step names it without a
# trailing slash (`uses: ./scripts/ci`), so the spelled-directory rule never fired
# and every file under it — the action.yml it runs and the scripts that runs — was
# exempt-able while release.yml executed it (found writing #3540's tests first).


def test_ship_workflow_local_uses_directory_reaches_its_contents(tmp_path: Path) -> None:
    wf = "jobs:\n  r:\n    steps:\n      - uses: ./scripts/ci\n"
    _reached(_reach(tmp_path, {".github/workflows/release.yml": wf}))


def test_remote_uses_and_bare_directory_word_do_not_reach(tmp_path: Path) -> None:
    """Negative: a marketplace action is not a repo directory, and prose naming
    the directory without `uses: ./` stays a word, as on the base."""
    wf = (
        "jobs:\n  r:\n    steps:\n      - uses: scripts/ci@v1\n"
        "      - name: see scripts/ci for probes\n        run: echo ok\n"
    )
    _exempt(_reach(tmp_path, {".github/workflows/release.yml": wf}))


# --- workflow/action files never take the carve-out (#3540 round 3) -----------------
# Rounds 1-3 tried to exempt a changed harness workflow unless it could ship
# (trigger, secret, write permission, publishing step), read by three successive
# YAML readers; reviewers broke each one. A changed workflow/action is now ALWAYS
# a round. Every attack those rounds found is kept below as a regression: each
# changes a harness workflow, so each must exit 1 — now by the structural rule.

_SAFE_NATIVE = (
    "on:\n  pull_request:\n  workflow_dispatch:\n"
    "permissions:\n  contents: read\n"
    "jobs:\n  n:\n    runs-on: ubuntu-latest\n    steps:\n"
    "      - run: bun scripts/ci/check.ts\n"
    "        env:\n          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}\n"
    "      - uses: actions/upload-artifact@v4\n        with:\n          name: evidence\n"
)


def _native_delta(tmp_path: Path, head_text: str) -> subprocess.CompletedProcess[str]:
    repo, _ = _setup(tmp_path)
    reviewed = _commit(repo, {".github/workflows/native.yml": _SAFE_NATIVE}, "ci: native")
    head = _commit(repo, {".github/workflows/native.yml": head_text}, "ci: native edit")
    return _run(repo, reviewed, head)


def _workflow_round(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 1, result.stdout + result.stderr
    assert "workflow/action file changed: `.github/workflows/native.yml`" in result.stdout
    assert "never take the carve-out" in result.stdout


def test_harness_workflow_edit_without_new_powers_opens_a_round(tmp_path: Path) -> None:
    """Same triggers, GITHUB_TOKEN only, read permission, an upload step the
    reviewed head already had: exempt in rounds 1-2, a round now — whether a
    workflow edit is harmless is not decidable without parsing YAML as GitHub."""
    edited = _SAFE_NATIVE.replace("bun scripts/ci/check.ts", "bun scripts/ci/check.ts --fast")
    _workflow_round(_native_delta(tmp_path, edited))


def test_harness_workflow_comment_naming_powers_opens_a_round(tmp_path: Path) -> None:
    """QA case (g): a comment that only names powers — a round."""
    edited = "# was on: push with secrets.NPM_TOKEN and contents: write\n" + _SAFE_NATIVE
    _workflow_round(_native_delta(tmp_path, edited))


def test_harness_action_and_uses_named_yaml_open_a_round(tmp_path: Path) -> None:
    """Outside `.github/`: an `action.yml` anywhere, and any YAML a `uses: ./…`
    names (here a reusable workflow under scripts/ci), are workflow files too."""
    repo, _ = _setup(tmp_path)
    reviewed = _commit(
        repo,
        {
            "scripts/ci/action.yml": "runs:\n  using: composite\n  steps: []\n",
            "scripts/ci/reuse.yaml": "on: workflow_call\njobs: {}\n",
            ".github/workflows/native.yml": "jobs:\n  r:\n    uses: './scripts/ci/reuse.yaml'\n",
        },
        "ci: local action and reusable workflow",
    )
    head = _commit(
        repo,
        {
            "scripts/ci/action.yml": "runs:\n  using: composite\n  steps: [{run: x}]\n",
            "scripts/ci/reuse.yaml": "on: workflow_call\njobs: {a: {}}\n",
        },
        "ci: edit both",
    )
    result = _run(repo, reviewed, head)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "workflow/action file changed: `scripts/ci/action.yml`" in result.stdout
    assert "workflow/action file changed: `scripts/ci/reuse.yaml`" in result.stdout


def test_plain_harness_yaml_not_named_by_uses_stays_harness_only(tmp_path: Path) -> None:
    """Negative: a harness YAML data file no `uses:` names is not a workflow file."""
    repo, reviewed = _setup(tmp_path)
    head = _commit(repo, {"e2e/fixture.yml": "cells: [a, b]\n"}, "test(e2e): fixture")
    _exempt(_run(repo, reviewed, head))


_ATTACK_TAIL = (
    "permissions:\n  contents: read\njobs:\n  n:\n    runs-on: ubuntu-latest\n    steps:\n"
    "      - run: bun scripts/ci/check.ts\n      - uses: actions/upload-artifact@v4\n"
)
#: Spelled in two halves only so the literal reads as YAML here, not as prose.
_INHERIT = "inh" + "erit"


@pytest.mark.parametrize(
    ("head_text", "reason"),
    [
        # Review R1 / QA Q1: comment stripping cut real shell and block scalars.
        (_SAFE_NATIVE + '      - run: |\n          N="a #"; gh release create v9\n', "gh release"),
        (
            _SAFE_NATIVE + '      - run: |\n          N="a #"; echo ${{ secrets.NPM_TOKEN }}\n',
            "NPM_TOKEN",
        ),
        (_SAFE_NATIVE + "      - run: |\n          # ${{ secrets.NPM_TOKEN }}\n", "NPM_TOKEN"),
        (
            _SAFE_NATIVE
            + "      - run: >-\n          echo hi\n          #${{ secrets.NPM_TOKEN }}\n",
            "NPM_TOKEN",
        ),
        # Review R2: whole-context, quoted and indexed forms.
        (_SAFE_NATIVE + "      - run: echo '${{ toJSON(secrets) }}'\n", "secret"),
        (
            _SAFE_NATIVE.replace("    runs-on:", f'    secrets: "{_INHERIT}"\n    runs-on:', 1),
            "secret",
        ),
        (_SAFE_NATIVE + "      - run: echo ${{ secrets['NPM'] }}\n", "secret"),
        # QA Q2: quoted permission key/value.
        (_SAFE_NATIVE.replace("contents: read", '"contents": "write"'), "write permission"),
        (_SAFE_NATIVE.replace("contents: read", "'contents': 'write'"), "write permission"),
        (_SAFE_NATIVE.replace("  contents: read", "  write-all"), "write permission"),
        # QA Q3 / review R4: event shapes a line parser missed.
        ('on:\n  pull_request:\n  "push":\n' + _ATTACK_TAIL, "trigger `push`"),
        ("on:\n- pull_request\n- push\n" + _ATTACK_TAIL, "trigger `push`"),
        ("on:\n  - pull_request\n  - 'push'\n" + _ATTACK_TAIL, "trigger `push`"),
        ("on: [pull_request,\n  push]\n" + _ATTACK_TAIL, "trigger `push`"),
        ('"on": {pull_request: {}, workflow_call: {}}\n' + _ATTACK_TAIL, "trigger `workflow_call`"),
        ("on: release\n" + _ATTACK_TAIL, "trigger `release`"),
    ],
    ids=[
        "hash-in-shell-release",
        "hash-in-shell-secret",
        "block-scalar-comment-secret",
        "folded-scalar-secret",
        "tojson-secrets",
        "quoted-inherit",
        "indexed-secret",
        "quoted-permission-dq",
        "quoted-permission-sq",
        "write-all",
        "quoted-event-key",
        "seq-at-key-indent",
        "quoted-seq-item",
        "multiline-flow-seq",
        "flow-map-workflow-call",
        "scalar-event",
    ],
)
def test_harness_workflow_power_attacks_open_a_round(
    tmp_path: Path, head_text: str, reason: str
) -> None:
    """Every round-1 reviewer/QA attack on #3540 that exited 0, as a regression."""
    # `reason` names the power the attack aimed at; the structural rule catches
    # it without reading the power at all.
    del reason
    _workflow_round(_native_delta(tmp_path, head_text))


@pytest.mark.parametrize(
    "workflow",
    [
        "jobs:\n  r:\n    steps:\n      - {uses: ./scripts/ci}\n",
        "jobs:\n  r:\n    steps:\n      - { uses: ./scripts/ci }\n",
        'jobs:\n  r:\n    steps:\n      - "uses": ./scripts/ci\n',
        "jobs:\n  r:\n    steps: [{name: a, uses: ./scripts/ci}]\n",
        "jobs:\n  r:\n    steps:\n      - uses: './scripts/ci'\n",
    ],
    ids=["flow-map", "flow-map-spaced", "quoted-key", "flow-seq", "quoted-value"],
)
def test_local_uses_in_any_yaml_shape_reaches(tmp_path: Path, workflow: str) -> None:
    """Review R3 / QA Q4: `uses: ./<dir>` in flow or quoted form is still a step."""
    _reached(_reach(tmp_path, {".github/workflows/release.yml": workflow}))


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        (("  workflow_dispatch:\n", "  workflow_dispatch:\n  push:\n"), "trigger `push`"),
        (("on:\n  pull_request:\n", "on: [pull_request, schedule]\nx:\n"), "trigger `schedule`"),
        (("secrets.GITHUB_TOKEN", "secrets.NPM_TOKEN"), "secret `NPM_TOKEN`"),
        (("contents: read", "contents: write"), "write permission"),
        (("contents: read", "id-token: write"), "write permission"),
        (
            ("      - run: bun", "      - uses: softprops/action-gh-release@v2\n      - run: bun"),
            "publishing step `softprops/action-gh-release`",
        ),
        (("      - run: bun", "      - run: npm publish\n      - run: bun"), "`npm publish`"),
    ],
    ids=["push", "schedule-flow", "secret", "contents-write", "id-token-write", "release", "npm"],
)
def test_harness_workflow_gaining_release_powers_opens_a_round(
    tmp_path: Path, change: tuple[str, str], reason: str
) -> None:
    old, new = change
    assert old in _SAFE_NATIVE
    del reason  # the power aimed at; the structural rule needs no reading of it
    _workflow_round(_native_delta(tmp_path, _SAFE_NATIVE.replace(old, new, 1)))


# --- #3540 rounds 2-3: attacks on the YAML scanners, kept as regressions -----------
# Comments after events, CRLF, double-quoted escapes and line joins, anchors,
# tags, complex keys (round 2), and plain scalars that desynced the scanner's
# quote or block-scalar tracking (round 3: review R3-M1, QA Q9). All change a
# harness workflow, so all exit 1 by the structural rule. Ship files are still
# scanned raw AND with double-quoted escapes decoded (last test).

_S = _SAFE_NATIVE
_BS = "\\"


@pytest.mark.parametrize(
    ("head_text", "reason"),
    [
        # Review R2-m1 / QA Q6: a trailing comment or CRLF after an event.
        (
            _S.replace("on:\n  pull_request:\n  workflow_dispatch:\n", "on: push # x\n"),
            "trigger `push`",
        ),
        (
            _S.replace(
                "on:\n  pull_request:\n  workflow_dispatch:\n",
                "on:\n  - pull_request\n  - push  # also main\n",
            ),
            "trigger `push`",
        ),
        (
            _S.replace(
                "on:\n  pull_request:\n  workflow_dispatch:\n",
                "on:\n  - pull_request\n  - workflow_dispatch\n  - 'push' # x\n",
            ),
            "trigger `push`",
        ),
        (
            _S.replace(
                "on:\n  pull_request:\n  workflow_dispatch:\n",
                "on:\n  - pull_request\n  - push\n",
            ).replace("\n", "\r\n"),
            "trigger `push`",
        ),
        # Review R2-M1 / QA Q7: double-quoted escapes and line joins.
        (_S + f'      - run: "echo ${{{{ {_BS}x73ecrets.NPM_TOKEN }}}} | npm publish"\n', "escape"),
        (_S + f'      - run: "echo ${{{{ se{_BS}\n          crets.NPM_TOKEN }}}}"\n', "escape"),
        (_S.replace("contents: read", f'contents: "wr{_BS}x69te"'), "escape"),
        (_S.replace("contents: read", f'contents: "{_BS}x77rite"'), "escape"),
        (_S.replace("contents: read", f'contents: read\n  id-token: "wr{_BS}u0069te"'), "escape"),
        (_S.replace("permissions:", f'"perm{_BS}x69ssions":'), "escape"),
        (
            _S.replace(
                "on:\n  pull_request:\n  workflow_dispatch:\n",
                f'on:\n  - pull_request\n  - workflow_dispatch\n  - "{_BS}x70ush"\n',
            ),
            "escape",
        ),
        # QA Q8: anchors and aliases; plus tags and complex keys (same class).
        (
            _S.replace(
                "on:\n  pull_request:\n  workflow_dispatch:\n",
                "env:\n  EV: &ev push\non:\n  - *ev\n  - workflow_dispatch\n",
            ),
            "anchor/alias",
        ),
        (_S.replace("contents: read", "contents: !!str read"), "tag"),
        (
            _S.replace("  workflow_dispatch:\n", "  workflow_dispatch:\n  ? pull_request\n"),
            "complex key",
        ),
        (
            _S.replace("  workflow_dispatch:\n", '  workflow_dispatch:\n  "unterminated\n'),
            "unterminated",
        ),
        # Review R3-M1: ` - '` in a plain scalar desynced quote tracking.
        (
            "name: native - 'evidence\n"
            + _S.replace("contents: read", f'contents: "wr{_BS}x69te"')
            + "      - run: echo 'done'\n",
            "escape",
        ),
        (
            'name: native - "evidence\n'
            + _S.replace("contents: read", f'contents: "wr{_BS}x69te"'),
            "escape",
        ),
        # QA Q9: ` - |` inside a flow collection read as a block-scalar header.
        (
            "on: [pull_request, workflow_dispatch]\n"
            "jobs: {e2e: {name: x - |\n  , runs-on: ubuntu-latest, permissions: "
            f'{{contents: "{_BS}x77rite"}}, '
            f'steps: [{{run: "echo ${{{{ {_BS}x73ecrets.NPM_TOKEN }}}}"}}]}}}}\n',
            "escape",
        ),
        (
            "on: {pull_request: {branches: [x - |\n  ]}, workflow_dispatch: null, "
            f'"{_BS}x70ush": null}}\n' + _ATTACK_TAIL,
            "escape",
        ),
    ],
    ids=[
        "scalar-event-comment",
        "seq-item-comment",
        "quoted-seq-item-comment",
        "crlf",
        "dq-escape-secret",
        "dq-linejoin-secret",
        "dq-escape-write",
        "qa-n4-write",
        "dq-unicode-id-token",
        "dq-escape-permissions-key",
        "qa-n3-escaped-event",
        "qa-n6-anchor-alias",
        "tag",
        "complex-key",
        "unterminated-dq",
        "r3-name-single-quote",
        "r3-name-double-quote",
        "q9-fj-flow-block-header",
        "q9-fo-flow-escaped-event",
    ],
)
def test_harness_workflow_scanner_attacks_open_a_round(
    tmp_path: Path, head_text: str, reason: str
) -> None:
    # `reason` names the power the attack aimed at; the structural rule catches
    # it without reading the power at all.
    del reason
    _workflow_round(_native_delta(tmp_path, head_text))


@pytest.mark.parametrize(
    "workflow",
    [
        f'jobs:\n  r:\n    steps:\n      - uses: "{_BS}x2e/scripts/ci"\n',
        f'jobs:\n  r:\n    steps:\n      - uses: "{_BS}u002e/scripts/ci"\n',
        "jobs:\n  r:\n    steps:\n      - uses: ./scripts/ci\r\n",
    ],
    ids=["dq-escaped-dot", "dq-unicode-dot", "crlf"],
)
def test_ship_reference_through_yaml_escapes_reaches(tmp_path: Path, workflow: str) -> None:
    """Review R2-M1: a ship file is scanned raw AND with double-quoted escapes
    decoded, so `"\\x2e/scripts/ci"` still names the local action directory."""
    _reached(_reach(tmp_path, {".github/workflows/release.yml": workflow}))
