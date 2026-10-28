"""Explicit checkout roots use the sanctioned reader, not a changed session cwd.

All copy here is synthetic and committed to isolated miniature repositories.
No test invokes the reader against operator/product catalogue content.
"""

import json
import subprocess
from pathlib import Path

import pytest

from local_operator.tools.public_copy import (
    ReadPublicCopyParams,
    execute_read_public_copy,
)
from tests.unit.secrets.test_public_copy_boundaries import (
    UI_CATALOGUE,
    _context,
    _git,
    _public_repo,
    _ui_repo,
)


@pytest.mark.asyncio
async def test_explicit_root_reads_committed_copy_without_moving_session(tmp_path):
    root = _ui_repo(tmp_path)
    context = _context(tmp_path, tmp_path)
    params = ReadPublicCopyParams(path=UI_CATALOGUE, repo_root=str(root))
    result = await execute_read_public_copy("read1", params.model_dump(), context=context)
    assert not result.is_error, result.text
    assert context.cwd == str(tmp_path)
    assert result.details["repository_root"] == str(root.resolve())
    assert len(result.details["commit"]) == 40
    audit = json.loads(Path(result.details["audit_log"]).read_text().splitlines()[-1])
    assert audit["repository_root"] == str(root.resolve())
    assert audit["commit"] == result.details["commit"]
    assert audit["repository"] == "damianvtran/local-operator-ui"


@pytest.mark.asyncio
async def test_core_and_linked_worktree_from_home_context(tmp_path):
    root = _public_repo(tmp_path)
    _git(root, "remote", "add", "origin", "https://github.com/damianvtran/local-operator.git")
    linked = tmp_path / "linked"
    _git(root, "worktree", "add", "--detach", str(linked), "HEAD")
    assert (linked / ".git").is_file()
    context = _context(tmp_path, tmp_path)
    for selected in ("repo", "linked"):
        result = await execute_read_public_copy(
            "core",
            {
                "path": "local_operator/i18n/catalogues/en/wire.slash.json",
                "repo_root": selected,
            },
            context=context,
        )
        assert not result.is_error, result.text
        assert result.details["repository"] == "damianvtran/local-operator"
        assert context.cwd == str(tmp_path)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "origin",
    [
        None,
        "https://github.com/other/repo.git",
        "https://example.test/damianvtran/local-operator-ui.git",
        "/local/damianvtran/local-operator-ui.git",
        "git@other.test:damianvtran/local-operator-ui.git",
        "https://github.com/extra/damianvtran/local-operator-ui.git",
        "https://github.com/damianvtran/local-operator.git",
    ],
)
async def test_explicit_root_refuses_wrong_identity_or_family(tmp_path, origin):
    root = _ui_repo(tmp_path, origin=origin)
    result = await execute_read_public_copy(
        "no",
        {
            "path": UI_CATALOGUE,
            "repo_root": str(root),
        },
        context=_context(tmp_path, tmp_path),
    )
    assert result.is_error
    assert "public_copy_repository_not_allowed" in result.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path", ["README.md", "../outside.json", "/absolute.json", "src/i18n/other.json"]
)
async def test_explicit_root_does_not_expand_file_allowlist(tmp_path, path):
    root = _ui_repo(tmp_path)
    result = await execute_read_public_copy(
        "no", {"path": path, "repo_root": str(root)}, context=_context(tmp_path, tmp_path)
    )
    assert result.is_error


@pytest.mark.asyncio
async def test_omitting_root_still_uses_home_context(tmp_path):
    _ui_repo(tmp_path)
    context = _context(tmp_path, tmp_path)
    result = await execute_read_public_copy("no", {"path": UI_CATALOGUE}, context=context)
    assert result.is_error
    assert "not a git checkout" in result.text


@pytest.mark.asyncio
async def test_confined_root_refuses_before_git(tmp_path, monkeypatch):
    from local_operator.tools import public_copy

    root = _ui_repo(tmp_path)
    jail = tmp_path / "jail"
    jail.mkdir()
    context = _context(tmp_path, tmp_path).model_copy(update={"confinement_root": str(jail)})
    monkeypatch.setattr(
        public_copy.subprocess, "run", lambda *a, **k: pytest.fail("Git ran outside confinement")
    )
    result = await execute_read_public_copy(
        "no", {"path": UI_CATALOGUE, "repo_root": str(root)}, context=context
    )
    assert result.is_error
    assert "confined" in result.text


@pytest.mark.asyncio
async def test_missing_kernel_confinement_never_runs_git(tmp_path, monkeypatch):
    from local_operator.tools import public_copy
    from local_operator.tools.confinement import ToolConfinement

    root = _ui_repo(tmp_path)
    context = _context(root, tmp_path).model_copy(update={"confinement_root": str(root)})
    monkeypatch.setattr(ToolConfinement, "wrap", lambda self, argv: None)
    monkeypatch.setattr(
        public_copy.subprocess, "run", lambda *a, **k: pytest.fail("unconfined Git ran")
    )
    result = await execute_read_public_copy(
        "no", {"path": UI_CATALOGUE, "repo_root": str(root)}, context=context
    )
    assert result.is_error


@pytest.mark.asyncio
async def test_linked_metadata_outside_confinement_is_not_read(tmp_path):
    root = _ui_repo(tmp_path)
    linked = tmp_path / "jail"
    _git(root, "worktree", "add", "--detach", str(linked), "HEAD")
    context = _context(linked, tmp_path).model_copy(update={"confinement_root": str(linked)})
    result = await execute_read_public_copy(
        "no", {"path": UI_CATALOGUE, "repo_root": str(linked)}, context=context
    )
    assert result.is_error, "linked metadata outside confinement was accepted"


@pytest.mark.asyncio
async def test_explicit_root_refuses_dirty_staged_and_untracked_copy(tmp_path):
    root = _ui_repo(tmp_path)
    context = _context(tmp_path, tmp_path)
    file = root / UI_CATALOGUE
    file.write_text(json.dumps({"wire.slash.credential": "changed synthetic copy"}))
    for stage in (False, True):
        if stage:
            _git(root, "add", UI_CATALOGUE)
        result = await execute_read_public_copy(
            "no", {"path": UI_CATALOGUE, "repo_root": str(root)}, context=context
        )
        assert result.is_error
    untracked = "src/i18n/catalogues/en/untracked.json"
    (root / untracked).write_text('{"key":"synthetic"}')
    result = await execute_read_public_copy(
        "no", {"path": untracked, "repo_root": str(root)}, context=context
    )
    assert result.is_error


@pytest.mark.asyncio
async def test_head_race_keeps_content_and_audit_on_pinned_commit(tmp_path, monkeypatch):
    from local_operator.tools import public_copy

    root = _ui_repo(tmp_path)

    def head():
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()

    first = head()
    (root / UI_CATALOGUE).write_text('{"wire.slash.credential":"second synthetic version"}')
    _git(root, "add", UI_CATALOGUE)
    _git(root, "commit", "-qm", "second fixture")
    second = head()
    _git(root, "checkout", "--detach", first)
    run = public_copy.subprocess.run
    moved = False

    def racing_run(argv, **kwargs):
        nonlocal moved
        result = run(argv, **kwargs)
        if "HEAD^{commit}" in argv and not moved:
            moved = True
            # Move just the ref after the tool captured it. Index/worktree
            # remain at the first commit; reads must not follow the new HEAD.
            run(
                ["git", "-C", str(root), "update-ref", "HEAD", second],
                check=True,
                capture_output=True,
            )
        return result

    monkeypatch.setattr(public_copy.subprocess, "run", racing_run)
    result = await execute_read_public_copy(
        "race", {"path": UI_CATALOGUE, "repo_root": str(root)}, context=_context(tmp_path, tmp_path)
    )
    assert not result.is_error, result.text
    assert "second synthetic version" not in result.text
    assert result.details["commit"] == first
    audit = json.loads(Path(result.details["audit_log"]).read_text().splitlines()[-1])
    assert audit["commit"] == first


@pytest.mark.asyncio
@pytest.mark.parametrize("available", [True, False])
async def test_registered_tool_loop_honors_inventory_and_records_audit(tmp_path, available):
    from local_operator.harness.loop import AgentLoop, LoopContext
    from local_operator.harness.types import (
        Message,
        StreamEndEvent,
        StreamTextDelta,
        ToolExecutionEndEvent,
    )
    from local_operator.tools.registry import create_tools
    from tests.unit.harness.test_loop import (
        ScriptedStream,
        make_config,
        tool_call_delta,
    )

    root = _ui_repo(tmp_path)
    context = _context(tmp_path, tmp_path)
    tools = create_tools(context, enabled=["read_public_copy"])
    tool = next(item for item in tools if item.name == "read_public_copy")
    assert tool.approval_tier == "read"
    stream = ScriptedStream(
        [
            [
                tool_call_delta(
                    0,
                    id="registered",
                    name="read_public_copy",
                    args=json.dumps(
                        {
                            "path": UI_CATALOGUE,
                            "repo_root": str(root),
                            "key": "wire.slash.credential",
                        }
                    ),
                ),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamTextDelta(delta="done"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    events = [
        event
        async for event in AgentLoop().run(
            [Message.user("read the synthetic committed key")],
            LoopContext(tools=[tool] if available else [], tool_context=context),
            make_config(stream),
            None,
        )
    ]
    if available:
        result = next(event.result for event in events if isinstance(event, ToolExecutionEndEvent))
        assert not result.is_error
        audit = json.loads(Path(result.details["audit_log"]).read_text().splitlines()[-1])
        assert audit["repository_root"] == str(root.resolve())
        assert audit["session"] == "abc123"
        assert audit["sha256"] == result.details["sha256"]
        assert audit["commit"] == result.details["commit"]
        assert json.loads(result.text)["wire.slash.credential"].startswith("Type or paste")
    else:
        from local_operator.paths import config_dir

        failure = next(
            message
            for message in events[-1].messages
            if isinstance(message, Message) and message.role == "tool"
        )
        assert failure.is_error and "Tool not found" in failure.text
        audit_path = config_dir() / "audit" / "public-copy-reads.jsonl"
        records = audit_path.read_text().splitlines() if audit_path.exists() else []
        assert not any(
            json.loads(line).get("repository_root") == str(root.resolve()) for line in records
        )
    assert context.cwd == str(tmp_path)


@pytest.mark.asyncio
async def test_explicit_symlink_escape_refuses_before_git(tmp_path, monkeypatch):
    from local_operator.tools import public_copy

    root = _ui_repo(tmp_path)
    target = root / UI_CATALOGUE
    target.unlink()
    outside = tmp_path / "outside.json"
    outside.write_text('{"synthetic":"must not return"}')
    target.symlink_to(outside)
    monkeypatch.setattr(
        public_copy.subprocess, "run", lambda *a, **k: pytest.fail("Git read escaped path")
    )
    result = await execute_read_public_copy(
        "no", {"path": UI_CATALOGUE, "repo_root": str(root)}, context=_context(tmp_path, tmp_path)
    )
    assert result.is_error
    assert "must not return" not in result.text


@pytest.mark.asyncio
async def test_explicit_subdirectory_is_not_checkout_root(tmp_path):
    root = _ui_repo(tmp_path)
    result = await execute_read_public_copy(
        "no",
        {"path": UI_CATALOGUE, "repo_root": str(root / "src")},
        context=_context(tmp_path, tmp_path),
    )
    assert result.is_error
    assert "not the checkout root" in result.text


@pytest.mark.asyncio
async def test_explicit_root_audit_failure_returns_no_copy(tmp_path, monkeypatch):
    from local_operator.tools import public_copy

    root = _ui_repo(tmp_path)

    def fail(*args):
        raise PermissionError("synthetic audit failure")

    monkeypatch.setattr(public_copy, "_open_audit_log", fail)
    result = await execute_read_public_copy(
        "no", {"path": UI_CATALOGUE, "repo_root": str(root)}, context=_context(tmp_path, tmp_path)
    )
    assert result.is_error
    assert "Type or paste" not in result.text


@pytest.mark.asyncio
async def test_git_environment_cannot_redirect_explicit_root(tmp_path, monkeypatch):
    root = _ui_repo(tmp_path)
    unrelated = _public_repo(tmp_path)
    monkeypatch.setenv("GIT_DIR", str(unrelated / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(unrelated))
    monkeypatch.setenv("GIT_INDEX_FILE", str(unrelated / ".git/index"))
    result = await execute_read_public_copy(
        "yes", {"path": UI_CATALOGUE, "repo_root": str(root)}, context=_context(tmp_path, tmp_path)
    )
    assert not result.is_error, result.text
    assert result.details["repository"] == "damianvtran/local-operator-ui"


@pytest.mark.asyncio
@pytest.mark.parametrize("driver", ["clean", "process"])
@pytest.mark.parametrize("filter_name", ["side-effect", "unspecified", "unset"])
async def test_cleanliness_check_does_not_execute_git_clean_filter(tmp_path, driver, filter_name):
    import os
    import shlex

    root = _ui_repo(tmp_path)
    (root / ".gitattributes").write_text(f"*.json filter={filter_name}\n")
    _git(root, "add", ".gitattributes")
    _git(root, "commit", "-qm", "synthetic attribute")
    marker = tmp_path / "filter-ran"
    _git(
        root,
        "config",
        f"filter.{filter_name}.{driver}",
        f"sh -c 'echo ran > {shlex.quote(str(marker))}; cat'",
    )
    target = root / UI_CATALOGUE
    # Break Git's cached stat match without changing the committed content.
    stat = target.stat()
    os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000000))
    result = await execute_read_public_copy(
        "filter",
        {"path": UI_CATALOGUE, "repo_root": str(root)},
        context=_context(tmp_path, tmp_path),
    )
    assert result.is_error and "public_copy_filter_not_supported" in result.text
    assert not marker.exists(), "read-tier Git diff executed a configured clean filter"


@pytest.mark.asyncio
async def test_cleanliness_disables_fsmonitor_external_diff_and_textconv(tmp_path):
    import os

    root = _ui_repo(tmp_path)
    marker = tmp_path / "git-helper-ran"
    helper = tmp_path / "git-helper.sh"
    helper.write_text(f"#!/bin/sh\necho ran > '{marker}'\ncat\n")
    helper.chmod(0o700)
    (root / ".gitattributes").write_text("*.json diff=side-effect\n")
    _git(root, "add", ".gitattributes")
    _git(root, "commit", "-qm", "synthetic diff attribute")
    _git(root, "config", "core.fsmonitor", str(helper))
    _git(root, "config", "diff.external", str(helper))
    _git(root, "config", "diff.side-effect.textconv", str(helper))
    file = root / UI_CATALOGUE
    stat = file.stat()
    os.utime(file, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000000))
    result = await execute_read_public_copy(
        "helpers",
        {"path": UI_CATALOGUE, "repo_root": str(root)},
        context=_context(tmp_path, tmp_path),
    )
    assert not result.is_error, result.text
    assert not marker.exists()


@pytest.mark.asyncio
async def test_every_git_probe_disables_transports_and_lazy_fetch(tmp_path, monkeypatch):
    from local_operator.tools import public_copy

    root = _ui_repo(tmp_path)
    run = public_copy.subprocess.run
    observed = []

    def constrained(argv, **kwargs):
        observed.append(kwargs["env"])
        assert kwargs["env"]["GIT_ALLOW_PROTOCOL"] == ""
        assert kwargs["env"]["GIT_NO_LAZY_FETCH"] == "1"
        return run(argv, **kwargs)

    monkeypatch.setenv("GIT_ALLOW_PROTOCOL", "https:ssh:file")
    monkeypatch.setattr(public_copy.subprocess, "run", constrained)
    result = await execute_read_public_copy(
        "transport",
        {"path": UI_CATALOGUE, "repo_root": str(root)},
        context=_context(tmp_path, tmp_path),
    )
    assert not result.is_error, result.text
    assert observed
