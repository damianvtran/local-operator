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
    UI_ORIGIN_SSH,
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


def _per_user_temp() -> Path:
    """The directory the macOS sandbox lets a confined child READ.

    ``ToolConfinement`` allows reads of ``/private/var/folders`` (its documented
    residual), so a repository there is the discriminating placement. It is
    named explicitly rather than inherited from ``tmp_path``: pytest's base
    follows ``TMPDIR``, so the original test passed or failed with the rig and
    code review called the reader clean on a rig that hid the gap.
    """
    import sys

    from local_operator.tools.confinement import sandbox_unavailable_reason

    if sys.platform != "darwin" or sandbox_unavailable_reason() is not None:
        pytest.skip("the per-user-temp read allowance is a macOS seatbelt property")
    # ``getconf`` rather than ``os.confstr``: the Darwin name is not in Python's
    # table. It is the directory the sandbox profile names (/private/var/folders).
    reported = subprocess.run(
        ["getconf", "DARWIN_USER_TEMP_DIR"], capture_output=True, text=True, check=True
    ).stdout.strip()
    return Path(reported).resolve()


@pytest.fixture
def user_temp_dir():
    import shutil
    import uuid

    base = _per_user_temp() / f"pcr-{uuid.uuid4().hex}"
    base.mkdir()
    try:
        yield base
    finally:
        shutil.rmtree(base, ignore_errors=True)


@pytest.mark.asyncio
async def test_linked_metadata_in_per_user_temp_is_refused(user_temp_dir, tmp_path):
    """The sandbox lets Git read per-user temp, so the READER must refuse it."""
    import subprocess

    from local_operator.tools.confinement import ToolConfinement

    root = _ui_repo(user_temp_dir)
    jail = tmp_path / "jail"
    jail.mkdir()
    linked = jail / "wt"
    _git(root, "worktree", "add", "--detach", str(linked), "HEAD")
    assert (linked / ".git").is_file()
    common = (root / ".git").resolve()
    # CONTROL: the sandbox alone would have served this. Without it a green
    # result could be an unrelated refusal, which is how the gap was missed.
    probe = subprocess.run(
        ToolConfinement.at(jail).wrap(["/bin/cat", str(common / "HEAD")]),
        capture_output=True,
        check=False,
    )
    assert probe.returncode == 0, "the kernel sandbox no longer allows this read; revisit the test"

    context = _context(linked, tmp_path).model_copy(update={"confinement_root": str(jail)})
    result = await execute_read_public_copy(
        "no", {"path": UI_CATALOGUE, "repo_root": str(linked)}, context=context
    )
    assert result.is_error
    assert result.details["refusal"]["code"] == "public_copy_git_metadata_outside_confinement"
    assert result.details["refusal"]["reason"] == "git_dir_outside_root"
    assert "Type or paste" not in result.text


@pytest.mark.asyncio
async def test_alternates_outside_confinement_are_refused(user_temp_dir, tmp_path):
    """Object alternates are followed by Git just like the common dir."""
    donor = _ui_repo(user_temp_dir)
    clone = tmp_path / "clone"
    _git(donor, "clone", "-q", "--shared", str(donor), str(clone))
    _git(clone, "remote", "set-url", "origin", UI_ORIGIN_SSH)
    assert (clone / ".git/objects/info/alternates").is_file()
    context = _context(clone, tmp_path).model_copy(update={"confinement_root": str(clone)})
    result = await execute_read_public_copy(
        "no", {"path": UI_CATALOGUE, "repo_root": str(clone)}, context=context
    )
    assert result.is_error
    assert result.details["refusal"]["reason"] == "alternates_outside_root"


@pytest.mark.asyncio
async def test_a_self_contained_confined_checkout_still_reads(tmp_path):
    root = _ui_repo(tmp_path)
    context = _context(root, tmp_path).model_copy(update={"confinement_root": str(root)})
    result = await execute_read_public_copy(
        "ok", {"path": UI_CATALOGUE, "repo_root": str(root)}, context=context
    )
    assert not result.is_error, result.text


@pytest.mark.asyncio
async def test_unconfined_linked_worktree_is_unaffected(user_temp_dir, tmp_path):
    root = _ui_repo(user_temp_dir)
    linked = tmp_path / "wt"
    _git(root, "worktree", "add", "--detach", str(linked), "HEAD")
    result = await execute_read_public_copy(
        "ok",
        {"path": UI_CATALOGUE, "repo_root": str(linked)},
        context=_context(tmp_path, tmp_path),
    )
    assert not result.is_error, result.text


def _refusal_body(result):
    """The structured repair fields a refusal returns: ``details`` AND the text line."""
    assert result.is_error
    body = result.details["refusal"]
    first, _, rest = result.text.partition("\n")
    assert first.startswith("refused: ")
    # providers never see ``details``: the same object must be in the text.
    assert json.loads(rest) == body
    return body


@pytest.mark.asyncio
async def test_wrong_checkout_for_path_names_the_repository_to_use(tmp_path):
    """U1/J: core path against the UI checkout is 'use the other checkout', not 'never'."""
    root = _ui_repo(tmp_path)
    result = await execute_read_public_copy(
        "x",
        {"path": "local_operator/i18n/catalogues/en/wire.slash.json", "repo_root": str(root)},
        context=_context(tmp_path, tmp_path),
    )
    body = _refusal_body(result)
    assert body["code"] == "public_copy_repository_not_allowed"
    assert body["reason"] == "wrong_checkout_for_path"
    assert body["checkout_is"] == "damianvtran/local-operator-ui"
    assert body["path_needs"] == "damianvtran/local-operator"
    assert body["accepted_repositories"] == [
        "damianvtran/local-operator",
        "damianvtran/local-operator-ui",
    ]
    assert set(body["path_families"]) == set(body["accepted_repositories"])


@pytest.mark.asyncio
async def test_untrusted_origin_is_told_apart_from_a_wrong_checkout(tmp_path):
    """U1/L: a lookalike origin is 'never allowed', and its URL is not echoed."""
    secret_host = "https://token123:hunter2@example.invalid/damianvtran/local-operator-ui.git"
    root = _ui_repo(tmp_path, origin=secret_host)
    result = await execute_read_public_copy(
        "x", {"path": UI_CATALOGUE, "repo_root": str(root)}, context=_context(tmp_path, tmp_path)
    )
    body = _refusal_body(result)
    assert body["reason"] == "origin_not_accepted"
    assert body["checkout_is"] is None
    assert "token123" not in result.text and "example.invalid" not in result.text


@pytest.mark.asyncio
async def test_typo_and_blank_root_are_distinguishable(tmp_path):
    ctx = _context(tmp_path, tmp_path)
    typo = await execute_read_public_copy(
        "x", {"path": UI_CATALOGUE, "repo_root": str(tmp_path / "nope")}, context=ctx
    )
    blank = await execute_read_public_copy(
        "x", {"path": UI_CATALOGUE, "repo_root": "   "}, context=ctx
    )
    assert _refusal_body(typo)["reason"] == "repo_root_does_not_exist"
    assert _refusal_body(typo)["repo_root"] == str(tmp_path / "nope")
    assert _refusal_body(blank)["reason"] == "repo_root_blank_or_not_a_string"


@pytest.mark.asyncio
async def test_subdirectory_root_says_it_is_not_the_checkout_root(tmp_path):
    """U3: a subdirectory must not be reported as a wrong path."""
    root = _ui_repo(tmp_path)
    result = await execute_read_public_copy(
        "x",
        {"path": "i18n/catalogues/en/wire.slash.json", "repo_root": str(root / "src")},
        context=_context(tmp_path, tmp_path),
    )
    body = _refusal_body(result)
    assert body["code"] == "public_copy_path_not_allowed"
    assert body["reason"] == "repo_root_is_not_the_checkout_root"
    assert "src/i18n" not in result.text.split("\n", 1)[0]


@pytest.mark.asyncio
async def test_declared_filter_names_the_path_and_driver(tmp_path):
    root = _ui_repo(tmp_path)
    (root / ".gitattributes").write_text("*.json filter=custom-driver\n")
    _git(root, "add", ".gitattributes")
    _git(root, "commit", "-qm", "attrs")
    result = await execute_read_public_copy(
        "x", {"path": UI_CATALOGUE, "repo_root": str(root)}, context=_context(tmp_path, tmp_path)
    )
    body = _refusal_body(result)
    assert body["code"] == "public_copy_filter_not_supported"
    assert body["reason"] == "path_declares_git_filter"
    assert body["path"] == UI_CATALOGUE and body["filter"] == "custom-driver"


@pytest.mark.asyncio
async def test_home_root_session_is_told_about_repo_root(tmp_path):
    """U2: the first failure points at the option that exists for it."""
    result = await execute_read_public_copy(
        "x", {"path": UI_CATALOGUE}, context=_context(tmp_path, tmp_path)
    )
    body = _refusal_body(result)
    assert result.text.startswith("refused: the workspace is not a git checkout")
    assert body["fix"] == "pass repo_root"
    # an explicit root that is not a checkout keeps the plain legacy sentence
    plain = tmp_path / "plain"
    plain.mkdir()
    again = await execute_read_public_copy(
        "x", {"path": UI_CATALOGUE, "repo_root": str(plain)}, context=_context(tmp_path, tmp_path)
    )
    assert again.is_error and again.details is None


@pytest.mark.asyncio
async def test_absolute_path_with_a_root_is_not_called_empty(tmp_path):
    root = _ui_repo(tmp_path)
    result = await execute_read_public_copy(
        "x",
        {"path": str(root / UI_CATALOGUE), "repo_root": str(root)},
        context=_context(tmp_path, tmp_path),
    )
    assert _refusal_body(result)["code"] == "public_copy_path_must_be_relative"


@pytest.mark.asyncio
async def test_loose_default_origin_is_not_recorded_as_verified(tmp_path):
    """D1/D4: the default path keeps its loose match but never claims a verified identity."""
    root = _ui_repo(tmp_path, origin="/local/mirror/damianvtran/local-operator-ui.git")
    result = await execute_read_public_copy(
        "x", {"path": UI_CATALOGUE}, context=_context(root, tmp_path)
    )
    assert not result.is_error, result.text
    assert result.details["repository"] is None
    assert result.details["repository_verified"] is False
    audit = json.loads(Path(result.details["audit_log"]).read_text().splitlines()[-1])
    assert audit["repository"] is None and audit["repository_verified"] is False


@pytest.mark.asyncio
async def test_no_origin_is_null_not_an_empty_string(tmp_path):
    """D4: a core checkout with no origin reads on the default root, with ``None`` identity."""
    core = _public_repo(tmp_path)
    result = await execute_read_public_copy(
        "x",
        {"path": "local_operator/i18n/catalogues/en/wire.slash.json"},
        context=_context(core, tmp_path),
    )
    assert not result.is_error, result.text
    assert result.details["repository"] is None
    assert result.details["repository_verified"] is False


@pytest.mark.asyncio
async def test_verified_explicit_root_records_the_identity(tmp_path):
    root = _ui_repo(tmp_path)
    result = await execute_read_public_copy(
        "x", {"path": UI_CATALOGUE, "repo_root": str(root)}, context=_context(tmp_path, tmp_path)
    )
    assert result.details["repository"] == "damianvtran/local-operator-ui"
    assert result.details["repository_verified"] is True


@pytest.mark.asyncio
async def test_missing_promisor_blob_is_not_called_uncommitted(tmp_path):
    """Q1: the file IS committed; only its object is unavailable."""
    root = _ui_repo(tmp_path)
    blob = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", f"HEAD:{UI_CATALOGUE}"], text=True
    ).strip()
    loose = root / ".git" / "objects" / blob[:2] / blob[2:]
    assert loose.is_file(), "fixture precondition: the blob is a loose object"
    moved = tmp_path / "moved-object"
    loose.rename(moved)
    result = await execute_read_public_copy(
        "x", {"path": UI_CATALOGUE, "repo_root": str(root)}, context=_context(tmp_path, tmp_path)
    )
    body = _refusal_body(result)
    assert body["code"] == "public_copy_blob_unavailable"
    assert body["reason"] == "object_missing_locally"
    assert "not committed" not in result.text


@pytest.mark.asyncio
async def test_home_root_call_without_repo_root_points_at_the_option(tmp_path):
    """U2: the first place a home-root session fails names the option that exists for it."""
    result = await execute_read_public_copy(
        "x", {"path": UI_CATALOGUE}, context=_context(tmp_path, tmp_path)
    )
    body = _refusal_body(result)
    assert result.text.startswith("refused: the workspace is not a git checkout")
    assert body["code"] == "public_copy_workspace_not_a_checkout"
    assert body["fix"] == "pass repo_root"
    assert body["accepted_repositories"]
    # an explicit root that is not a checkout keeps the plain sentence (no hint loop)
    plain = await execute_read_public_copy(
        "x",
        {"path": UI_CATALOGUE, "repo_root": str(tmp_path)},
        context=_context(tmp_path, tmp_path),
    )
    assert plain.text.startswith("refused: the workspace is not a git checkout")
    assert "\n" not in plain.text
