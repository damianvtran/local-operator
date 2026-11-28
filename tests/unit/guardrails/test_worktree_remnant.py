"""A failed ``git worktree remove`` remnant ranks as an ordinary directory.

Field case 2026-11-24 (disk recovery, guard 0.112.8): Finder wrote a
``.DS_Store`` while Git was deleting a clean, merged tree, so ``git worktree
remove`` unregistered the tree, deleted its admin directory and then failed
with "Directory not empty". The leftover folder still held its ``.git`` pointer
file and ranked as "a git repository root" (sent to the judge), while an
identical leftover without the pointer was an ordinary path. See
``classify._removed_worktree_remnant``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from local_operator.guardrails.classify import _removed_worktree_remnant

ENV = {**os.environ, "GIT_CONFIG_SYSTEM": "/dev/null", "GIT_TERMINAL_PROMPT": "0"}


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, env=ENV)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git("init", "-q", "-b", "main", cwd=root)
    _git(
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@t",
        "commit",
        "-q",
        "--allow-empty",
        "-m",
        "i",
        cwd=root,
    )
    return root


def _tree(repo: Path, name: str) -> Path:
    wt = repo / ".worktrees" / name
    _git("worktree", "add", "-q", "-b", f"fix/{name}", str(wt), cwd=repo)
    (wt / "kept.txt").write_text("tracked leftover\n")
    return wt


def test_a_live_linked_worktree_is_not_a_remnant(repo: Path) -> None:
    assert not _removed_worktree_remnant(str(_tree(repo, "live")))


def test_admin_dir_gone_with_committed_files_and_ds_store_is_a_remnant(repo: Path) -> None:
    wt = _committed_tree(repo, "half-removed")
    (wt / ".DS_Store").write_bytes(b"\x00\x00\x00\x01Bud1")  # Finder's mid-delete write
    # What a failed ``git worktree remove`` leaves: admin dir gone, folder kept.
    shutil.rmtree(repo / ".git" / "worktrees" / "half-removed")
    assert _removed_worktree_remnant(str(wt))


@pytest.mark.parametrize("extra", ["untracked", "symlink", "ds-store-dir", "cr-twin", "lf-name"])
def test_any_content_git_does_not_hold_keeps_the_rank(repo: Path, extra: str) -> None:
    wt = _committed_tree(repo, f"extra-{extra}")
    if extra == "untracked":
        (wt / "notes.md").write_text("never committed\n")
    elif extra == "symlink":
        (wt / "link").symlink_to(wt / "kept.txt")
    elif extra == "cr-twin":
        # #4536 R2-M1: Git strips the CR, so this would hash as ``kept.txt``.
        (wt / "kept.txt\r").write_text("never committed\n")
    elif extra == "lf-name":
        (wt / "a\nb").write_text("never committed\n")
    else:
        (wt / ".DS_Store").mkdir()
        (wt / ".DS_Store" / "data").write_text("not Finder's\n")
    shutil.rmtree(repo / ".git" / "worktrees" / f"extra-{extra}")
    assert not _removed_worktree_remnant(str(wt))


def _committed_tree(repo: Path, name: str) -> Path:
    """A tree whose only file is committed: a true remnant loses nothing."""
    wt = repo / ".worktrees" / name
    _git("worktree", "add", "-q", "-b", f"fix/{name}", str(wt), cwd=repo)
    (wt / "kept.txt").write_text(f"tracked {name}\n")
    _git("add", "kept.txt", cwd=wt)
    _git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", name, cwd=wt)
    return wt


@pytest.mark.parametrize("dirty", [False, True], ids=["committed", "uncommitted"])
def test_moved_then_pruned_tree_is_a_remnant_only_without_new_content(
    repo: Path, tmp_path: Path, dirty: bool
) -> None:
    """#4536 R1-B1: a hand-moved tree (same basename) that ``git worktree
    prune`` unregistered passes every pointer check; only its content decides."""
    wt = _committed_tree(repo, "feat")
    if dirty:
        (wt / "kept.txt").write_text("uncommitted work\n")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    moved = elsewhere / "feat"
    wt.rename(moved)
    _git("worktree", "prune", cwd=repo)
    assert not (repo / ".git" / "worktrees" / "feat").exists()
    assert _removed_worktree_remnant(str(moved)) is (not dirty)


@pytest.mark.parametrize("shape", ["git-dir", "two-lines", "relative-outside", "admin-present"])
def test_any_other_shape_keeps_the_repository_rank(repo: Path, shape: str) -> None:
    # Committed content, so ONLY the pointer shape can make it fail.
    wt = _committed_tree(repo, f"shape-{shape}")
    marker = wt / ".git"
    pointer = marker.read_text()
    (
        shutil.rmtree(repo / ".git" / "worktrees" / f"shape-{shape}")
        if shape != "admin-present"
        else None
    )
    if shape == "git-dir":
        marker.unlink()
        marker.mkdir()
    elif shape == "two-lines":
        marker.write_text(pointer + "extra\n")
    elif shape == "relative-outside":
        marker.write_text(f"gitdir: ../../elsewhere/.git/worktrees/shape-{shape}\n")
    assert not _removed_worktree_remnant(str(wt))


def test_rm_rf_rank_follows_the_remnant_rule(repo: Path) -> None:
    """Classification level: a live linked tree is still held (HIGH, judged),
    the remnant is an ordinary path and is never 'a git repository root'."""
    from local_operator.guardrails.classify import Risk, classify_bash

    live = _tree(repo, "still-live")
    gone = _committed_tree(repo, "remnant")
    shutil.rmtree(repo / ".git" / "worktrees" / "remnant")
    env = {"HOME": str(repo.parent)}

    def rank(path: Path):
        return classify_bash(
            f"rm -rf {path}", cwd=str(repo), env=env, scratchpad=None, session_id="aaaaaaaaaaa1"
        )

    assert rank(live).risk is Risk.HIGH
    remnant = rank(gone)
    reasons = " ".join([t.reason for t in remnant.targets] + list(remnant.reasons))
    assert "a git repository root" not in reasons and "linked worktree" not in reasons, reasons


@pytest.mark.parametrize("payload", ["pyc-only", "py-in-pycache"])
def test_pointer_remnant_sheds_bytecode_by_content(repo: Path, payload: str) -> None:
    """Pass-3 leftovers (2026-11-24) held only for ``__pycache__``: Git's
    aborted walk never reached them, and bytecode is never committed."""
    wt = _committed_tree(repo, f"pyc-{payload}")
    pyc = wt / "pkg" / "__pycache__"
    pyc.mkdir(parents=True)
    (pyc / "mod.cpython-313.pyc").write_bytes(b"\x00bytecode")
    if payload == "py-in-pycache":
        (pyc / "notes.py").write_text("hand-written\n")
    shutil.rmtree(repo / ".git" / "worktrees" / f"pyc-{payload}")
    assert _removed_worktree_remnant(str(wt)) is (payload == "pyc-only")


@pytest.mark.parametrize("registered", [False, True], ids=["dropped", "still-registered"])
def test_pointerless_folder_under_worktrees_needs_git_to_have_dropped_it(
    repo: Path, registered: bool
) -> None:
    """Pass-3 (2026-11-24): Git deleted the pointer before failing. A folder at
    ``<repo>/.worktrees/<name>`` with no ``.git`` is a remnant only when Git no
    longer lists it and its content is already committed."""
    wt = _committed_tree(repo, f"nopointer-{registered}")
    (wt / ".git").unlink()
    if not registered:
        shutil.rmtree(repo / ".git" / "worktrees" / f"nopointer-{registered}")
    assert _removed_worktree_remnant(str(wt)) is (not registered)


def test_pointerless_folder_outside_worktrees_is_not_a_remnant(repo: Path, tmp_path: Path) -> None:
    wt = _committed_tree(repo, "elsewhere")
    (wt / ".git").unlink()
    shutil.rmtree(repo / ".git" / "worktrees" / "elsewhere")
    moved = tmp_path / "loose"
    wt.rename(moved)
    assert not _removed_worktree_remnant(str(moved))


def test_remnant_rank_carries_the_lossless_reason(repo: Path) -> None:
    from local_operator.guardrails.classify import REMNANT_REASON, Risk, classify_bash

    gone = _committed_tree(repo, "lossless")
    shutil.rmtree(repo / ".git" / "worktrees" / "lossless")
    v = classify_bash(
        f"rm -rf {gone}",
        cwd=str(repo),
        env={"HOME": str(repo.parent)},
        scratchpad=None,
        session_id="aaaaaaaaaaa1",
    )
    assert v.risk is Risk.LOW and v.targets[0].reason == REMNANT_REASON, v.reasons


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("lossless", "status", "held"),
    [
        (True, "timed_out", False),
        (False, "timed_out", True),
        (True, "declined", True),
        (True, "open", True),
    ],
    ids=["remnant", "uncommitted", "declined-remnant", "open-remnant"],
)
async def test_timed_out_ask_does_not_hold_a_lossless_remnant(
    repo: Path, monkeypatch: pytest.MonkeyPatch, lossless: bool, status: str, held: bool
) -> None:
    """Field case 2026-11-24 (disk recovery, a-58df): the session that ASKED
    about two leftovers could not delete them after the ask timed out, though
    any other session could. An unanswered ask protects the operator's data;
    a remnant whose every byte Git holds has none. An uncommitted file in the
    same folder keeps the ordinary ask hold."""
    from types import SimpleNamespace

    from local_operator.guardrails import provenance as prov
    from local_operator.guardrails.check import (
        CallContext,
        GuardSettings,
        check_tool_call,
    )

    name = f"asked-{lossless}-{status}"
    gone = _committed_tree(repo, name)
    if not lossless:
        (gone / "notes.md").write_text("never committed\n")
    shutil.rmtree(repo / ".git" / "worktrees" / name)
    real = os.path.realpath(gone)
    asked = SimpleNamespace(
        ask_id="a-58df",
        session_id="aaaaaaaaaaa1",
        # A decline is an answer that approves nothing (#4545 R1-M1).
        status=status,
        label={"timed_out": "TIMED OUT with no answer", "open": "still open, not yet answered"}.get(
            status, "declined"
        ),
        text=f"May I delete {real}?",
        approved="",
        created_at=0,
        never_answered=status != "declined",
        answer=None,
        untrusted_yes=False,
        answered_at=0,
        has_answer=status == "declined",
        expires_at=0,
    )
    monkeypatch.setattr(prov, "unanswered_asks", lambda sid, *a, **k: [asked])
    calls: list[str] = []

    async def judge(system: str, prompt: str) -> str:
        calls.append(prompt)
        return "BLOCK: judged"

    ctx = CallContext(
        cwd=str(repo),
        env={"HOME": str(repo.parent), "PATH": os.environ.get("PATH", "")},
        scratchpad=None,
        session_id="aaaaaaaaaaa1",
    )
    out = await check_tool_call(
        "bash", {"command": f"rm -rf {real}"}, ctx, settings=GuardSettings(), judge=judge
    )
    if held:
        assert out is not None and out.blocked, out
    else:
        assert out is None or not out.blocked, out
        assert calls == []


@pytest.mark.parametrize(
    ("payload", "remnant"),
    [
        ('{"version":"7.0.2","root":[1],"fileNames":["a.ts"]}', True),
        ('{"version":"5.4.5","program":{"fileNames":[]}}', True),
        ("my notes, not tsc state\n", False),
        ('{"version":"5.4.5"}', False),
        ('["version", "root"]', False),
    ],
    ids=["tsc7-root", "tsc5-program", "text", "json-no-program", "json-list"],
)
def test_tsbuildinfo_is_shed_only_by_content(repo: Path, payload: str, remnant: bool) -> None:
    """Field case 2026-11-24 (UI leftovers, guard 0.112.10): a remnant holding
    only committed blobs plus ``.cache/typecheck/node.tsbuildinfo``. The file
    is lossless only when its CONTENT is tsc incremental state."""
    from local_operator.guardrails.classify import _removed_worktree_remnant

    wt = _committed_tree(repo, f"tsb-{remnant}-{abs(hash(payload))}")
    (wt / ".cache" / "typecheck").mkdir(parents=True)
    (wt / ".cache" / "typecheck" / "node.tsbuildinfo").write_text(payload)
    shutil.rmtree(repo / ".git" / "worktrees" / wt.name)
    assert _removed_worktree_remnant(str(wt)) is remnant


def test_remnant_memo_skips_only_unchanged_hashing(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#4545 R1-m2: a large lossless leftover is hashed once per checkpoint,
    and ANY change to its files (new, edited) forces a fresh hash."""
    import subprocess as sp

    from local_operator.guardrails import classify as c

    monkeypatch.setattr(c, "_REMNANT_MEMO", {})
    gone = _committed_tree(repo, "memo")
    shutil.rmtree(repo / ".git" / "worktrees" / "memo")
    hashes: list[int] = []
    real_run = sp.run

    def counted(cmd, *a, **k):
        if "hash-object" in cmd:
            hashes.append(1)
        return real_run(cmd, *a, **k)

    monkeypatch.setattr(c.subprocess, "run", counted)
    assert c._removed_worktree_remnant(str(gone)) is True
    assert c._removed_worktree_remnant(str(gone)) is True
    assert len(hashes) == 1  # unchanged: reused
    (gone / "draft.md").write_text("uncommitted\n")
    assert c._removed_worktree_remnant(str(gone)) is False
    assert len(hashes) == 2  # new file: re-hashed, and held
    (gone / "draft.md").unlink()
    assert c._removed_worktree_remnant(str(gone)) is True
    monkeypatch.setattr(c, "_REMNANT_MEMO_TTL_S", 0.0)
    assert c._removed_worktree_remnant(str(gone)) is True
    assert len(hashes) == 4  # expired: re-hashed


def test_remnant_review_minors(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """#4550 R1-m1: a same-size rewrite with its mtime restored is re-hashed
    (ctime moves). R1-m2: deeply nested JSON under a tsbuildinfo name holds
    instead of raising RecursionError out of the guard."""
    from local_operator.guardrails import classify as c

    monkeypatch.setattr(c, "_REMNANT_MEMO", {})
    gone = _committed_tree(repo, "minors")
    shutil.rmtree(repo / ".git" / "worktrees" / "minors")
    victim = next(p for p in gone.rglob("*") if p.is_file() and p.name != ".git")
    assert c._removed_worktree_remnant(str(gone)) is True
    st = victim.stat()
    body = victim.read_bytes()
    victim.write_bytes(bytes(b ^ 1 for b in body) if body else b"")
    os.utime(victim, ns=(st.st_atime_ns, st.st_mtime_ns))
    if body:
        assert c._removed_worktree_remnant(str(gone)) is False
    victim.write_bytes(body)
    deep = gone / "deep.tsbuildinfo"
    deep.write_text("[" * 200_000 + "]" * 200_000)
    assert c._removed_worktree_remnant(str(gone)) is False
