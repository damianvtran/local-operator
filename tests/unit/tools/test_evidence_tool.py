"""``evidence``: a reader child's create-only, content-only, worktree-bound receipt writer.

Why these tests exist: the tool is a sanctioned WRITE handed to roles that
otherwise cannot write, so every refusal rule is a boundary. Each test pins one
rule against a real git checkout on disk (no mocks of the filesystem), and the
child-assembly tests drive the real ``_build_child_session`` so the grant is
proven on the path a delegated reviewer actually takes.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from local_operator.harness.types import TextContent, ToolContext, ToolResult
from local_operator.tools import evidence_tool
from local_operator.tools.evidence_tool import (
    MAX_EVIDENCE_BYTES,
    EvidenceRefused,
    build_evidence_tool,
    verified_worktree_root,
    write_evidence,
)

#: The tool reads nothing from its context (its root is bound at build).
_CTX = ToolContext(cwd="/", session_id="evidence-test")


def _text_of(result: ToolResult) -> str:
    return "".join(b.text for b in result.content if isinstance(b, TextContent))


_GIT_ENV = {"GIT_CONFIG_SYSTEM": "/dev/null", "GIT_CONFIG_GLOBAL": "/dev/null"}


@pytest.fixture(autouse=True)
def _git_isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in _GIT_ENV.items():
        monkeypatch.setenv(key, value)


def _repo(path: Path, *dirs: str) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True, env={**os.environ, **_GIT_ENV})
    for rel in dirs:
        (path / rel).mkdir(parents=True, exist_ok=True)
    return Path(os.path.realpath(path))


@pytest.fixture
def root(tmp_path: Path) -> str:
    return str(_repo(tmp_path / "wt", "i18n/receipts", "docs/evidence"))


def _refused(root: str, path: str, content: str = "x") -> str:
    with pytest.raises(EvidenceRefused) as info:
        write_evidence(root, path, content)
    return str(info.value)


def test_creates_a_new_receipt_with_mode_0644(root: str) -> None:
    os.umask(0o077)  # a tight umask must not decide the receipt's mode
    try:
        written, size, digest = write_evidence(root, "i18n/receipts/r1/fr.json", '{"ok": 1}\n')
    finally:
        os.umask(0o022)
    target = Path(root, "i18n/receipts/r1/fr.json")
    assert target.read_text() == '{"ok": 1}\n'
    assert (written, size) == ("i18n/receipts/r1/fr.json", 10)
    assert digest == hashlib.sha256(b'{"ok": 1}\n').hexdigest()
    assert target.stat().st_mode & 0o777 == 0o644


def test_refuses_an_existing_file_and_leaves_it_untouched(root: str) -> None:
    existing = Path(root, "i18n/receipts/old.json")
    existing.write_text("theirs")
    assert "create-only" in _refused(root, "i18n/receipts/old.json", "mine")
    assert existing.read_text() == "theirs"


@pytest.mark.parametrize(
    ("path", "rule"),
    [
        ("i18n/receipts/../../x", "`..`"),
        ("../i18n/receipts/x", "`..`"),
        ("/etc/x", "absolute"),
        ("~/i18n/receipts/x", "`~`"),
        ("i18n//receipts/x", "empty or `.`"),
        ("src/app.py", "outside the evidence directories"),
        ("i18n/receipts", "outside the evidence directories"),
        ("i18n/audits/evidence/x.md", "does not exist in this worktree"),
    ],
)
def test_refuses_each_path_rule_by_name(root: str, path: str, rule: str) -> None:
    assert rule in _refused(root, path)


def test_refuses_a_symlinked_directory_component(root: str, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    Path(root, "i18n/receipts/link").symlink_to(outside, target_is_directory=True)
    assert "symlink" in _refused(root, "i18n/receipts/link/x.json")
    assert list(outside.iterdir()) == []


def test_refuses_a_symlinked_allowed_directory(tmp_path: Path) -> None:
    root = _repo(tmp_path / "wt2", "real")
    (root / "docs").mkdir()
    (root / "docs" / "evidence").symlink_to(root / "real", target_is_directory=True)
    assert "symlink" in _refused(str(root), "docs/evidence/x.json")


def test_refuses_an_oversize_payload_before_touching_disk(root: str) -> None:
    message = _refused(root, "i18n/receipts/new/big.txt", "a" * (MAX_EVIDENCE_BYTES + 1))
    assert "cap" in message
    assert not Path(root, "i18n/receipts/new").exists()


def test_refuses_a_guard_evidence_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A checkout sitting INSIDE the guard's own ledger directory is the one way
    # an allow-listed relative path can resolve onto guard evidence.
    config = tmp_path / "cfg"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    root = _repo(config / "guardrails", "i18n/receipts")
    message = _refused(str(root), "i18n/receipts/x.json")
    assert "the guardrail's own records" in message
    assert not (root / "i18n/receipts/x.json").exists()


def test_the_checkpoint_ranks_evidence_like_write() -> None:
    from local_operator.guardrails.check import FILE_DESTINATION_TOOLS

    assert FILE_DESTINATION_TOOLS["evidence"] == "path"


def test_worktree_binding_accepts_only_a_checkout_top_level(tmp_path: Path) -> None:
    root = _repo(tmp_path / "wt3", "docs/evidence")
    plain = tmp_path / "plain"
    plain.mkdir()
    assert verified_worktree_root(str(root)) == str(root)
    assert verified_worktree_root(str(root / "docs")) is None  # a subdirectory
    assert verified_worktree_root(str(plain)) is None  # not a checkout
    assert verified_worktree_root(str(tmp_path / "missing")) is None
    assert verified_worktree_root("relative/path") is None


def test_worktree_binding_accepts_a_linked_worktree(tmp_path: Path) -> None:
    # The briefed PR worktree is usually ``<repo>/.worktrees/<name>``, a LINKED
    # worktree whose ``.git`` is a file: it must bind like the primary does.
    primary = _repo(tmp_path / "primary")
    env = {**os.environ, **_GIT_ENV}
    ident = ["-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    subprocess.run(
        ["git", "-C", str(primary), *ident, "commit", "-q", "--allow-empty", "-m", "init"],
        check=True,
        env=env,
    )
    linked = primary / ".worktrees" / "pr"
    subprocess.run(
        ["git", "-C", str(primary), "worktree", "add", "-q", str(linked)], check=True, env=env
    )
    assert verified_worktree_root(str(linked)) == os.path.realpath(linked)


@pytest.mark.asyncio
async def test_the_tool_reports_hash_then_refuses_the_overwrite(root: str) -> None:
    tool = build_evidence_tool(root)
    args = {"path": "docs/evidence/r1.md", "content": "receipt\n"}
    first = await tool.execute("c1", args, None, None, _CTX)
    assert not first.is_error, first
    assert hashlib.sha256(b"receipt\n").hexdigest() in _text_of(first)
    second = await tool.execute("c2", {**args, "content": "forged\n"}, None, None, _CTX)
    assert second.is_error
    assert "create-only" in _text_of(second)
    assert Path(root, "docs/evidence/r1.md").read_text() == "receipt\n"
    # Content only: a source path is not a parameter.
    third = await tool.execute(
        "c3", {"path": "docs/evidence/r2.md", "source": "/etc/hosts"}, None, None, _CTX
    )
    assert third.is_error


# --- Child assembly -----------------------------------------------------------


@pytest.fixture
def tmp_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    config_dir = tmp_path / ".local-operator"
    config_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    return config_dir


async def _child_tool_names(agent: str, prompt: str) -> set[str]:
    import argparse

    from local_operator.agents import AgentRegistry
    from local_operator.config import ConfigManager
    from local_operator.harness.subagent import _build_child_session, _resolve_role
    from local_operator.session.session import Session
    from local_operator.session_factory import create_session

    config = Path(os.environ["LOCAL_OPERATOR_CONFIG_DIR"])
    args = argparse.Namespace(
        hosting="test", model="test", agent_name=None, agent_id=None, yolo=True, train=False
    )
    parent = await create_session(args, ConfigManager(config), AgentRegistry(config))
    assert isinstance(parent, Session)
    try:
        child = await _build_child_session(
            label=f"{agent}-probe",
            prompt=prompt,
            parent_session=parent,
            model_spec=None,
            job_id=f"job-{agent}",
            agent=agent,
            profile=_resolve_role(agent, parent),
        )
        try:
            return {tool.name for tool in child._tools}
        finally:
            await child.dispose()
    finally:
        await parent.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("agent", ["reviewer", "qa-tester"])
async def test_a_reader_with_a_worktree_gets_the_tool_and_one_without_does_not(
    agent: str, tmp_config_dir: Path, tmp_path: Path
) -> None:
    root = _repo(tmp_path / "pr-wt", "docs/evidence")
    # A template mention first: binding takes the first VALID field.
    brief = f"Brief: `Worktree: <abs path> @ <sha>`.\nWorktree: {root} @ abc123 · Round: 1"
    assert "evidence" in await _child_tool_names(agent, brief)
    assert "evidence" not in await _child_tool_names(agent, "Review the diff.")
    assert "evidence" not in await _child_tool_names(agent, f"Worktree: {tmp_path} @ abc")


@pytest.mark.asyncio
@pytest.mark.parametrize("agent", ["coder", "manager", "task"])
async def test_writers_and_managers_never_get_the_tool(
    agent: str, tmp_config_dir: Path, tmp_path: Path
) -> None:
    root = _repo(tmp_path / "pr-wt", "docs/evidence")
    assert "evidence" not in await _child_tool_names(agent, f"Worktree: {root} @ abc")


def test_no_registry_or_seed_row_names_the_tool() -> None:
    from local_operator.tools.registry import DEFAULT_TOOL_NAMES, TOOL_BUILDERS

    assert "evidence" not in DEFAULT_TOOL_NAMES
    assert "evidence" not in TOOL_BUILDERS
    manifest = Path(evidence_tool.__file__).parents[1] / "agent_seeds" / "manifest.json"
    rows = json.loads(manifest.read_text())
    rows = rows["seeds"]
    for row in rows:
        assert "evidence" not in (row.get("tools") or []), row.get("name")
