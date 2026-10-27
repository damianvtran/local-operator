"""The configuration run's ``skill`` tool (``tools/skill_tool.py``).

Real folders, the real storage (``ManagedSkillStorage.author``, the production
``apply`` for hub links) and the real link check: every assertion is about the
bytes on disk, so nothing in the write path is mocked. The tool is built through
``create_tools`` exactly as a session builds it. Where a guard is the run's OWN
addition over the Skills tab's route (the patch overlay, the shadowing refusal),
a negative control shows the unguarded path really does the harm the guard
prevents -- a guard whose absence changes nothing is not tested by its presence.

The factory-built run (real declaration, real merge) is pinned in
``tests/unit/test_session_factory.py``; the declaration itself in
``tests/unit/server/test_agents_config_run.py``.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.harness.types import AgentTool, ToolContext
from local_operator.hub_sync import check as chk
from local_operator.hub_sync import provenance as prov
from local_operator.hub_sync.identity import LinkError, LinkIdentity
from local_operator.hub_sync.provenance import make_record
from local_operator.hub_sync.skill_document import SkillDocument
from local_operator.hub_sync.skill_storage import ManagedSkillStorage
from local_operator.server.utils import desktop_skill_writes as writes
from local_operator.skills.api import PACKAGED_SKILL_ROOT
from local_operator.tools.registry import create_tools

pytestmark = pytest.mark.asyncio

HUB = "https://hub.example.test/api"


def _skill_md(name: str, description: str = "A skill.", body: str = "body\n") -> str:
    return f"---\nname: {name}\ndescription: {description}\n---\n{body}"


def _builtin_name() -> str:
    return sorted(
        child.name
        for child in PACKAGED_SKILL_ROOT.iterdir()
        if child.is_dir() and (child / "SKILL.md").is_file()
    )[0]


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """An isolated HOME + profile, and the tool built as a session builds it."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    config_dir = home / ".local-operator"
    config_dir.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))
    monkeypatch.delenv("LOCAL_OPERATOR_SKILL_EXTRA_ROOTS", raising=False)
    context = ToolContext(cwd=str(config_dir), memory_config_dir=str(config_dir))
    tools = create_tools(context, ["skill"])
    assert [tool.name for tool in tools] == ["skill"]
    return SimpleNamespace(home=home, config_dir=config_dir, context=context, tool=tools[0])


async def _call(env: SimpleNamespace, **args: Any) -> tuple[bool, str]:
    tool: AgentTool = env.tool
    result = await tool.execute("call", args, None, None, env.context)
    return bool(result.is_error), result.content[0].text  # type: ignore[union-attr]


def _fingerprint_line(text: str) -> str:
    return next(line.split()[1] for line in text.splitlines() if line.startswith("fingerprint:"))


def _make_local(env: SimpleNamespace, name: str = "demo") -> Path:
    writes.create(
        env.config_dir,
        name,
        files=[
            {"path": "SKILL.md", "content": _skill_md(name)},
            {"path": "references/notes.md", "content": "keep me\n"},
        ],
    )
    return env.config_dir / "skills" / name


def _tree(folder: Path) -> dict[str, bytes]:
    return {
        path.relative_to(folder).as_posix(): path.read_bytes()
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


def _link(config_dir: Path, name: str = "linked-skill") -> SkillDocument:
    """A pulled org skill: folder and baseline written by the production ``apply``."""
    document = SkillDocument.from_json(
        {
            "document_version": 1,
            "name": name,
            "description": f"hub {name}",
            "files": [
                {"path": "SKILL.md", "content": _skill_md(name, f"hub {name}", "hub body\n")},
                {"path": "scripts/doctor.sh", "content": "#!/bin/sh\necho ok\n"},
            ],
        }
    )
    identity = LinkIdentity("skill", f"id-{name}", HUB, "org", "tenant-1", f"sk-{name}")
    baseline = replace(
        make_record(
            "skill", identity.local_id, identity.hub_id, "tenant-1", document.to_json(), "pull"
        ),
        identity=identity,
        remote_revision='"r1"',
        lineage="clean",
        local_alias=name,
    )
    ManagedSkillStorage(config_dir).apply(
        identity, name, document, baseline, expected_fingerprint=None
    )
    return document


# -- the tool's shape ----------------------------------------------------------------


async def test_the_tool_mirrors_agent_and_team_and_needs_a_profile_root() -> None:
    tool = create_tools(ToolContext(memory_config_dir="/p"), ["skill"])[0]
    assert (tool.approval_tier, tool.concurrency, tool.interruptible) == (
        "read",
        "exclusive",
        False,
    )
    assert tool.parameters["properties"]["op"]["enum"] == ["list", "show", "create", "update"]
    # createIf: no profile root, no tool.
    assert create_tools(ToolContext(), ["skill"]) == []


# -- list / show ----------------------------------------------------------------------


async def test_list_marks_what_is_editable_and_why_the_rest_is_not(env) -> None:
    _make_local(env)
    error, text = await _call(env, op="list")
    assert not error, text
    assert "- demo [local, editable]" in text
    builtin = _builtin_name()
    assert f"- {builtin} [builtin, read-only (built-in)]" in text
    assert "Copy to my skills" in text


async def test_show_returns_skill_md_and_the_file_listing_and_one_file_by_path(env) -> None:
    folder = _make_local(env)
    error, text = await _call(env, op="show", name="demo")
    assert not error, text
    assert _fingerprint_line(text) == SkillDocument.from_directory(folder).fingerprint
    assert "description: A skill." in text and "- references/notes.md (8 bytes)" in text
    assert "keep me" not in text, "the other files are listed, not inlined"
    error, text = await _call(env, op="show", name="demo", path="references/notes.md")
    assert not error and text.endswith("--- references/notes.md ---\nkeep me\n"), text


async def test_show_of_a_builtin_is_readable_but_says_read_only(env) -> None:
    error, text = await _call(env, op="show", name=_builtin_name())
    assert not error, text
    assert "editable: no (built-in)" in text and "Copy to my skills" in text
    assert "fingerprint:" not in text, "no fingerprint to update with on a read-only skill"


# -- create ---------------------------------------------------------------------------


async def test_create_lands_only_in_the_managed_root(env) -> None:
    error, text = await _call(
        env, op="create", name="fresh", files=[{"path": "SKILL.md", "content": _skill_md("fresh")}]
    )
    assert not error, text
    assert (env.config_dir / "skills" / "fresh" / "SKILL.md").is_file()
    assert sorted(p.name for p in (env.config_dir / "skills").iterdir()) == ["fresh"]


@pytest.mark.parametrize("where", ["builtin", "external"])
async def test_create_over_a_read_only_name_is_refused(env, where: str) -> None:
    """A same-named local copy would shadow the original for EVERY session."""
    if where == "builtin":
        name = _builtin_name()
    else:
        name = "eco-skill"
        folder = env.home / ".claude" / "skills" / name
        folder.mkdir(parents=True)
        (folder / "SKILL.md").write_text(_skill_md(name))
    error, text = await _call(
        env, op="create", name=name, files=[{"path": "SKILL.md", "content": _skill_md(name)}]
    )
    assert error and text.startswith("skill-not-editable:"), text
    assert "Copy to my skills" in text
    assert not (env.config_dir / "skills" / name).exists()


async def test_negative_control_the_unguarded_create_does_shadow_a_builtin(env) -> None:
    """Why ``require_unshadowed`` exists: the route's ``create`` (Copy to my skills
    is a create) happily makes the shadowing copy."""
    name = _builtin_name()
    writes.create(env.config_dir, name, files=[{"path": "SKILL.md", "content": _skill_md(name)}])
    assert (env.config_dir / "skills" / name / "SKILL.md").is_file()


async def test_create_over_a_linked_alias_is_alias_occupied(env) -> None:
    _link(env.config_dir)
    error, text = await _call(
        env,
        op="create",
        name="linked-skill",
        files=[{"path": "SKILL.md", "content": _skill_md("linked-skill")}],
    )
    assert error and text.startswith("alias-occupied:"), text


# -- update ---------------------------------------------------------------------------


async def test_a_one_file_update_keeps_the_reference_files(env) -> None:
    folder = _make_local(env)
    fingerprint = SkillDocument.from_directory(folder).fingerprint
    error, text = await _call(
        env,
        op="update",
        name="demo",
        expected_fingerprint=fingerprint,
        files=[{"path": "SKILL.md", "content": _skill_md("demo", body="## Troubleshooting\n")}],
    )
    assert not error, text
    tree = _tree(folder)
    assert tree["references/notes.md"] == b"keep me\n"
    assert b"## Troubleshooting" in tree["SKILL.md"]
    assert _fingerprint_line(text) == SkillDocument.from_directory(folder).fingerprint
    # The previous bytes are kept (the storage's own backup).
    backups = list((env.config_dir / "hub" / "backups").glob("skill-*"))
    assert any(
        (backup / "SKILL.md").is_file()
        and b"Troubleshooting" not in (backup / "SKILL.md").read_bytes()
        for backup in backups
    ), backups


async def test_negative_control_a_whole_set_update_drops_the_references(env) -> None:
    """Why ``update`` is a patch: the route's whole-set replace deletes what a
    one-file edit did not resend."""
    folder = _make_local(env)
    fingerprint = SkillDocument.from_directory(folder).fingerprint
    writes.update(
        env.config_dir,
        "demo",
        [{"path": "SKILL.md", "content": _skill_md("demo", body="x\n")}],
        fingerprint,
    )
    assert list(_tree(folder)) == ["SKILL.md"]


async def test_remove_paths_deletes_and_nothing_else(env) -> None:
    folder = _make_local(env)
    fingerprint = SkillDocument.from_directory(folder).fingerprint
    error, text = await _call(
        env,
        op="update",
        name="demo",
        expected_fingerprint=fingerprint,
        remove_paths=["references/notes.md"],
    )
    assert not error, text
    assert list(_tree(folder)) == ["SKILL.md"]
    error, text = await _call(
        env,
        op="update",
        name="demo",
        expected_fingerprint=SkillDocument.from_directory(folder).fingerprint,
        remove_paths=["references/gone.md"],
    )
    assert error and "no file to remove" in text, text


async def test_a_stale_fingerprint_is_refused_and_the_bytes_are_unchanged(env) -> None:
    folder = _make_local(env)
    stale = SkillDocument.from_directory(folder).fingerprint
    # Someone else edits first (the Skills tab, another run).
    writes.patch(
        env.config_dir,
        "demo",
        [{"path": "SKILL.md", "content": _skill_md("demo", body="theirs\n")}],
        (),
        stale,
    )
    before = _tree(folder)
    error, text = await _call(
        env,
        op="update",
        name="demo",
        expected_fingerprint=stale,
        files=[{"path": "SKILL.md", "content": _skill_md("demo", body="mine\n")}],
    )
    assert error and text.startswith("local-concurrent-edit:"), text
    assert "Show it again" in text
    assert _tree(folder) == before


async def test_a_stale_fingerprint_is_refused_even_for_an_edit_that_changes_nothing(env) -> None:
    """``author`` answers a no-op update with success WITHOUT checking the CAS (a
    retry after a lost response must not 409 against its own first effect). The
    run has no such retry, so ``patch`` checks first: a model holding a stale
    fingerprint must re-read, not be told its edit "landed"."""
    folder = _make_local(env)
    stale = SkillDocument.from_directory(folder).fingerprint
    theirs = _skill_md("demo", body="theirs\n")
    writes.patch(env.config_dir, "demo", [{"path": "SKILL.md", "content": theirs}], (), stale)
    error, text = await _call(
        env,
        op="update",
        name="demo",
        expected_fingerprint=stale,
        files=[{"path": "SKILL.md", "content": theirs}],
    )
    assert error and text.startswith("local-concurrent-edit:"), text


async def test_the_cas_is_rechecked_after_the_overlay_is_built(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The overlay is built from one read; a write that lands between that read
    and the swap must still be refused -- the caller's fingerprint, not one read
    later, is what reaches ``author``'s check under the lock."""
    folder = _make_local(env)
    fingerprint = SkillDocument.from_directory(folder).fingerprint
    real_build = writes.document_from_files

    def racing_build(name: str, files: Any) -> SkillDocument:
        # After patch's read, before the write: another writer commits.
        (folder / "references" / "notes.md").write_text("raced\n")
        return real_build(name, files)

    monkeypatch.setattr(writes, "document_from_files", racing_build)
    with pytest.raises(LinkError) as caught:
        writes.patch(
            env.config_dir,
            "demo",
            [{"path": "SKILL.md", "content": _skill_md("demo", body="mine\n")}],
            (),
            fingerprint,
        )
    assert caught.value.code == "local-concurrent-edit"
    assert (folder / "references" / "notes.md").read_text() == "raced\n"
    assert b"mine" not in (folder / "SKILL.md").read_bytes()


@pytest.mark.parametrize("where", ["builtin", "external", "project"])
async def test_read_only_sources_are_not_editable(env, where: str, monkeypatch) -> None:
    if where == "builtin":
        name = _builtin_name()
        original = None
    else:
        name = f"{where}-skill"
        root = (
            env.home / ".claude" / "skills"
            if where == "external"
            else env.home / "repo" / ".local-operator" / "skills"
        )
        original = root / name
        original.mkdir(parents=True)
        (original / "SKILL.md").write_text(_skill_md(name))
        if where == "project":
            # A project skill reaches discovery only from a cwd under its repo;
            # the tool's root set is home-derived, so make the walk find it.
            monkeypatch.setattr(Path, "home", lambda: env.home / "repo")
    error, text = await _call(
        env,
        op="update",
        name=name,
        expected_fingerprint="a" * 64,
        files=[{"path": "SKILL.md", "content": _skill_md(name, body="x\n")}],
    )
    assert error and text.startswith("skill-not-editable:"), text
    assert "Copy to my skills" in text
    assert not (env.config_dir / "skills" / name).exists()
    if original is not None:
        assert (original / "SKILL.md").read_text() == _skill_md(name)


async def test_a_bundle_link_is_not_editable(env) -> None:
    from local_operator.hub_sync.skill_bundle import build_bundle
    from local_operator.hub_sync.skill_bundle_storage import ManagedBundleStorage

    source = env.home / "bundle-src"
    source.mkdir()
    (source / "SKILL.md").write_text(_skill_md("pdf-forms"))
    (source / "references").mkdir()
    (source / "references" / "guide.md").write_text("alpha\n")
    # A binary asset is what makes a skill a bundle in practice, and it is what
    # makes the refusal's ORDER matter: read as a text document first, this folder
    # answers "invalid skill" instead of "not editable here".
    (source / "assets").mkdir()
    (source / "assets" / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\xff\xfe")
    built = build_bundle(source)
    archive = env.home / "bundle.zip"
    archive.write_bytes(built.data)
    identity = LinkIdentity("skill", "bundle-local", HUB, "public", None, "remote-bundle")
    baseline = replace(
        prov.make_record(
            "skill", identity.local_id, identity.hub_id, None, built.manifest.to_json(), "pull"
        ),
        identity=identity,
        remote_revision='"r1"',
        lineage="clean",
        local_alias="pdf-forms",
    )
    ManagedBundleStorage(env.config_dir).apply(
        identity, "pdf-forms", archive, built.manifest, baseline, expected_fingerprint=None
    )
    folder = env.config_dir / "skills" / "pdf-forms"
    before = _tree(folder)
    error, text = await _call(
        env,
        op="update",
        name="pdf-forms",
        expected_fingerprint="a" * 64,
        files=[{"path": "SKILL.md", "content": _skill_md("pdf-forms", body="x\n")}],
    )
    assert error and text.startswith("skill-not-editable:"), text
    assert _tree(folder) == before


@pytest.mark.parametrize(
    "path", ["../evil.md", "scripts/../../x.md", "/abs.md", "notes.md", ".hidden/x.md"]
)
async def test_hostile_paths_are_refused_by_storage(env, path: str) -> None:
    folder = _make_local(env)
    before = _tree(folder)
    error, text = await _call(
        env,
        op="update",
        name="demo",
        expected_fingerprint=SkillDocument.from_directory(folder).fingerprint,
        files=[{"path": path, "content": "x"}],
    )
    assert error and text.startswith("skill-validation:"), text
    assert _tree(folder) == before
    assert not (env.config_dir / "evil.md").exists()


async def test_an_oversize_file_is_refused_by_storage(env) -> None:
    folder = _make_local(env)
    before = _tree(folder)
    error, text = await _call(
        env,
        op="update",
        name="demo",
        expected_fingerprint=SkillDocument.from_directory(folder).fingerprint,
        files=[{"path": "references/big.md", "content": "x" * (200 * 1024 + 1)}],
    )
    assert error and text.startswith("skill-too-large:"), text
    assert _tree(folder) == before


async def test_editing_a_linked_skill_keeps_the_baseline_and_the_check_says_local_only(env) -> None:
    document = _link(env.config_dir)
    baseline_file = prov.baseline_path(env.config_dir, "skill", "id-linked-skill")
    baseline_bytes = baseline_file.read_bytes()
    error, text = await _call(
        env,
        op="update",
        name="linked-skill",
        expected_fingerprint=document.fingerprint,
        files=[
            {
                "path": "SKILL.md",
                "content": _skill_md("linked-skill", "hub linked-skill", "EDITED\n"),
            }
        ],
    )
    assert not error, text
    assert "was not published" in text
    assert baseline_file.read_bytes() == baseline_bytes
    folder = env.config_dir / "skills" / "linked-skill"
    assert (folder / "scripts" / "doctor.sh").read_text() == "#!/bin/sh\necho ok\n"

    class Remote:
        api_key = "k"

    def fetch(_client: Any, _skill_id: str, **_kw: Any) -> dict[str, Any]:
        return {**document.to_json(), "revision": '"r1"'}

    checks = chk.check_hub_skills(
        config_dir=env.config_dir, client_for_tenant=lambda _t: Remote(), fetch=fetch
    )
    assert [c.classification.state for c in checks if c.classification] == ["local-only"]


# -- argument rules -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("args", "needle"),
    [
        ({"op": "show"}, "needs 'name'"),
        ({"op": "update", "name": "demo", "files": []}, "expected_fingerprint"),
        ({"op": "create", "name": "demo"}, "needs 'files'"),
        ({"op": "list", "path": "SKILL.md"}, "'path' is for op='show' only"),
        ({"op": "delete", "name": "demo"}, "invalid arguments"),
    ],
)
async def test_malformed_calls_are_refused_before_any_read(env, args, needle) -> None:
    error, text = await _call(env, **args)
    assert error and needle in text, text
