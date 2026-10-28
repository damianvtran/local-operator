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
    assert tool.parameters["properties"]["op"]["enum"] == [
        "list",
        "show",
        "create",
        "update",
        "edit",
    ]
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
    assert "description: A skill." in text and "- references/notes.md (8 chars)" in text
    assert "keep me" not in text, "the other files are listed, not inlined"
    error, text = await _call(env, op="show", name="demo", path="references/notes.md")
    assert (
        not error and "--- references/notes.md: lines 1-1 of 1, 8 chars ---\nkeep me\n" in text
    ), text
    assert "MORE" not in text


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
    """Under the schema's CHARACTER bound but over storage's BYTE bound: the
    storage layer is still the authority (two-byte characters)."""
    folder = _make_local(env)
    before = _tree(folder)
    error, text = await _call(
        env,
        op="update",
        name="demo",
        expected_fingerprint=SkillDocument.from_directory(folder).fingerprint,
        files=[{"path": "references/big.md", "content": "é" * (110 * 1024)}],
    )
    assert error and text.startswith("skill-too-large:"), text
    assert _tree(folder) == before


async def test_the_schema_refuses_a_file_over_the_storage_limit_before_parsing(env) -> None:
    """Review R5: the per-file bound is the storage limit, not 1 MiB."""
    error, text = await _call(
        env,
        op="create",
        name="fresh",
        files=[{"path": "SKILL.md", "content": "x" * (200 * 1024 + 1)}],
    )
    assert error and "at most 204800 characters" in text, text
    assert not (env.config_dir / "skills" / "fresh").exists()


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


# -- through the REAL loop: what the model receives, not what the tool returned --------
#
# Review R1: the loop cuts every tool result to the replay bound before the next
# request, so a test of ``tool.execute`` alone cannot see what the model reads. These
# drive the real ``AgentLoop`` with a scripted provider and assert on the REQUESTS it
# sent, which is the only place "what the model received" exists.


def _big_skill_md(name: str = "big", lines: int = 1000) -> str:
    body = "".join(
        f"line {index:04d}: step {index} of the long procedure\n" for index in range(lines)
    )
    return _skill_md(name, body=body)


class _ScriptedModel:
    """A provider that answers each request with the next scripted tool call.

    ``plan(previous_tool_text) -> args | None`` decides the next call from the text
    the model was SENT for the previous result (after the loop's replay bound), so
    a test can only act on what actually reached the model.
    """

    def __init__(self, plan: Any) -> None:
        self.plan = plan
        self.requests: list[Any] = []

    def __call__(self, request: Any, signal: Any) -> Any:
        import json

        from local_operator.harness.types import (
            Message,
            StreamEndEvent,
            StreamTextDelta,
            StreamToolCallDelta,
            TextContent,
        )

        self.requests.append(request)
        tool_rows = [m for m in request.messages if isinstance(m, Message) and m.role == "tool"]
        seen = None
        if tool_rows:
            block = tool_rows[-1].content[0]
            assert isinstance(block, TextContent)
            seen = block.text
        args = self.plan(seen)

        async def gen() -> Any:
            if args is None:
                yield StreamTextDelta(delta="done")
                yield StreamEndEvent(stop_reason="stop")
                return
            yield StreamToolCallDelta(
                index=0, id=f"c{len(self.requests)}", name="skill", argument_delta=json.dumps(args)
            )
            yield StreamEndEvent(stop_reason="tool_calls")

        return gen()


async def _drive(env: SimpleNamespace, plan: Any, *, masked: bool = False) -> _ScriptedModel:
    """Run the real loop. ``masked=True`` wires the SESSION's redaction exactly as a
    session does (review R6): a real ``Session`` over a ``VariableStore`` supplies
    both the loop's ``redact_tool_result`` and the tool's ``context.variables``, so
    the model reads what a real run's model reads and the tool judges by the same
    mask."""
    from local_operator.harness.loop import AgentLoop, LoopContext
    from local_operator.harness.types import LoopConfig, Message, ModelSpec

    model = _ScriptedModel(plan)
    redact = None
    tool_context = env.context
    if masked:
        from local_operator.session.session import Session
        from local_operator.session.transcript import Transcript
        from local_operator.variables import VariableStore

        session = Session(
            model=ModelSpec(provider="test", model_id="test-model", context_window=200_000),
            stream_fn=model,
            tools=[env.tool],
            transcript=Transcript(env.home / "masked-session"),
            system_blocks_provider=lambda: ["sys"],
            variables=VariableStore(cwd=str(env.home), env={}),
        )
        redact = session._redact_tool_result_text
        tool_context = session._build_tool_context()
        assert tool_context.memory_config_dir == str(env.config_dir)
    context = LoopContext(system_blocks=["sys"], tools=[env.tool], tool_context=tool_context)
    config = LoopConfig(
        model=ModelSpec(provider="test", model_id="test-model", context_window=200_000),
        convert_to_llm=lambda messages: [m for m in messages if isinstance(m, Message)],
        stream_fn=model,
        redact_tool_result=redact,
    )
    await AgentLoop().run_to_end([Message.user("revise the big skill")], context, config, None)
    return model


def _page_body(seen: str) -> str:
    """The file text of one ``show`` page as the model received it."""
    start = seen.index(
        "---\n", seen.index("--- SKILL.md: lines") if "--- SKILL.md: lines" in seen else 0
    )
    body = seen[start + 4 :]
    cut = min(
        (body.find(marker) for marker in ("\n[", "\n\nother files") if marker in body), default=-1
    )
    return body[: cut + 1] if cut >= 0 else body


async def test_paging_through_the_loop_delivers_every_line(env) -> None:
    import re

    content = _big_skill_md()
    assert len(content) > 32 * 1024
    writes.create(env.config_dir, "big", files=[{"path": "SKILL.md", "content": content}])
    received: list[str] = []

    def plan(seen: str | None) -> dict[str, Any] | None:
        if seen is None:
            return {"op": "show", "name": "big"}
        assert "elided on replay" not in seen, "a page was cut on replay"
        received.append(_page_body(seen))
        more = re.search(r"\[MORE: op='show' path='SKILL.md' offset=(\d+)\]", seen)
        return {"op": "show", "name": "big", "offset": int(more.group(1))} if more else None

    model = await _drive(env, plan)
    assert len(model.requests) > 5, "a 32 KB file needs several pages"
    assert "".join(received) == content, "every line, in order, exactly once"


async def test_negative_control_one_whole_show_loses_the_middle_on_replay(env) -> None:
    """Why paging is needed: a result over the bound DOES lose its middle in the
    loop. Bypasses the pager by returning the whole file from a stand-in tool."""
    from local_operator.harness.types import AgentTool, TextContent, ToolResult

    content = _big_skill_md()

    async def whole(tool_call_id: str, args: dict[str, Any], *_rest: Any) -> ToolResult:
        return ToolResult(
            tool_call_id=tool_call_id, tool_name="skill", content=[TextContent(text=content)]
        )

    env.tool = AgentTool(
        name="skill", label="x", description="x", parameters={"type": "object"}, execute=whole
    )
    seen_texts: list[str] = []

    def plan(seen: str | None) -> dict[str, Any] | None:
        if seen is None:
            return {"op": "show", "name": "big"}
        seen_texts.append(seen)
        return None

    await _drive(env, plan)
    assert "elided on replay" in seen_texts[0]
    assert "line 0500" not in seen_texts[0]


async def test_a_whole_file_update_of_a_long_file_is_refused_through_the_loop(env) -> None:
    content = _big_skill_md()
    folder = env.config_dir / "skills" / "big"
    writes.create(env.config_dir, "big", files=[{"path": "SKILL.md", "content": content}])
    before = _tree(folder)
    results: list[str] = []

    def plan(seen: str | None) -> dict[str, Any] | None:
        if seen is None:
            return {"op": "show", "name": "big"}
        if not results:
            results.append(seen)
            # What a model would compose from a partial view: the head it saw.
            return {
                "op": "update",
                "name": "big",
                "expected_fingerprint": _fingerprint_line(seen),
                "files": [{"path": "SKILL.md", "content": _page_body(seen)}],
            }
        results.append(seen)
        return None

    await _drive(env, plan)
    assert results[1].startswith("skill-file-too-long-to-replace:"), results[1]
    assert "cannot be removed here" not in results[1], "N1: a replace gets no removal hint"
    assert "op='edit'" in results[1]
    assert _tree(folder) == before


async def test_an_edit_through_the_loop_changes_only_its_text(env) -> None:
    content = _big_skill_md()
    folder = env.config_dir / "skills" / "big"
    writes.create(
        env.config_dir,
        "big",
        files=[
            {"path": "SKILL.md", "content": content},
            {"path": "references/notes.md", "content": "keep me\n"},
        ],
    )
    results: list[str] = []

    def plan(seen: str | None) -> dict[str, Any] | None:
        if seen is None:
            return {"op": "show", "name": "big"}
        results.append(seen)
        if len(results) == 1:
            line = "line 0007: step 7 of the long procedure\n"
            assert line in seen, "the edit's old_text is text the model received"
            return {
                "op": "edit",
                "name": "big",
                "expected_fingerprint": _fingerprint_line(seen),
                "edits": [
                    {
                        "path": "SKILL.md",
                        "old_text": line,
                        "new_text": line + "## Troubleshooting\n",
                    }
                ],
            }
        return None

    await _drive(env, plan)
    assert results[1].startswith("updated skill"), results[1]
    after = (folder / "SKILL.md").read_text()
    head, _, tail = content.partition("line 0008:")
    assert after == head + "## Troubleshooting\n" + "line 0008:" + tail
    assert (folder / "references" / "notes.md").read_text() == "keep me\n"


async def test_every_result_fits_the_replay_bound(env) -> None:
    """The structural invariant behind R1/R2: no op answers past ``RESULT_MAX_CHARS``."""
    from local_operator.harness.replay_bound import DEFAULT_REPLAY_BOUND_CHARS
    from local_operator.tools.skill_tool import RESULT_MAX_CHARS

    assert RESULT_MAX_CHARS < DEFAULT_REPLAY_BOUND_CHARS
    many = [{"path": "SKILL.md", "content": _big_skill_md("wide", 300)}] + [
        {"path": f"references/{'n' * 60}-{index:02d}.md", "content": "x\n"} for index in range(60)
    ]
    writes.create(env.config_dir, "wide", files=many)
    # A long single line, too: the pager must cut it rather than overflow.
    writes.create(
        env.config_dir,
        "oneline",
        files=[{"path": "SKILL.md", "content": _skill_md("oneline", body="y" * 20_000 + "\n")}],
    )
    for index in range(80):
        writes.create(
            env.config_dir,
            f"skill-{index:02d}",
            files=[{"path": "SKILL.md", "content": _skill_md(f"skill-{index:02d}", "d" * 300)}],
        )
    calls = [
        {"op": "list"},
        {"op": "show", "name": "wide"},
        {"op": "show", "name": "wide", "offset": 200},
        {"op": "show", "name": "oneline"},
        {"op": "show", "name": _builtin_name()},
    ]
    for args in calls:
        error, text = await _call(env, **args)
        assert not error, (args, text)
        assert len(text) <= RESULT_MAX_CHARS, (args, len(text))
    _error, text = await _call(env, op="show", name="wide")
    assert "more, not listed to fit this page" in text


async def test_list_pages_and_names_every_skill(env) -> None:
    """Review R2: following MORE reaches every skill; none is silently dropped."""
    import re

    for index in range(80):
        writes.create(
            env.config_dir,
            f"skill-{index:02d}",
            files=[{"path": "SKILL.md", "content": _skill_md(f"skill-{index:02d}", "d" * 300)}],
        )
    names: list[str] = []
    offset = 1
    pages = 0
    while True:
        error, text = await _call(env, op="list", offset=offset)
        assert not error, text
        pages += 1
        names += re.findall(r"^- (skill-\d\d) \[", text, flags=re.M)
        more = re.search(r"MORE: op='list' offset=(\d+)", text)
        if more is None:
            break
        offset = int(more.group(1))
    assert pages > 1
    assert names == [f"skill-{index:02d}" for index in range(80)]


async def test_edit_refuses_a_missing_or_ambiguous_match_and_writes_nothing(env) -> None:
    folder = env.config_dir / "skills" / "rep"
    writes.create(
        env.config_dir,
        "rep",
        files=[{"path": "SKILL.md", "content": _skill_md("rep", body="same\nsame\nonce\n")}],
    )
    fingerprint = SkillDocument.from_directory(folder).fingerprint
    before = _tree(folder)
    for old_text, count in (("same\n", 2), ("absent\n", 0)):
        error, text = await _call(
            env,
            op="edit",
            name="rep",
            expected_fingerprint=fingerprint,
            edits=[
                {"path": "SKILL.md", "old_text": "once\n", "new_text": "ONCE\n"},
                {"path": "SKILL.md", "old_text": old_text, "new_text": "x\n"},
            ],
        )
        assert error and text.startswith("skill-edit-no-unique-match:"), text
        assert f"occurs {count} times" in text
        assert _tree(folder) == before, "all edits land in one write, or none do"


async def test_edit_with_a_stale_fingerprint_is_refused(env) -> None:
    folder = _make_local(env)
    stale = SkillDocument.from_directory(folder).fingerprint
    writes.patch(
        env.config_dir, "demo", [{"path": "references/notes.md", "content": "theirs\n"}], (), stale
    )
    error, text = await _call(
        env,
        op="edit",
        name="demo",
        expected_fingerprint=stale,
        edits=[{"path": "SKILL.md", "old_text": "body\n", "new_text": "mine\n"}],
    )
    assert error and text.startswith("local-concurrent-edit:"), text
    assert b"mine" not in (folder / "SKILL.md").read_bytes()


async def test_duplicate_paths_in_an_update_are_refused(env) -> None:
    """Review R4: two versions of one file must not silently become one."""
    folder = _make_local(env)
    before = _tree(folder)
    error, text = await _call(
        env,
        op="update",
        name="demo",
        expected_fingerprint=SkillDocument.from_directory(folder).fingerprint,
        files=[
            {"path": "references/notes.md", "content": "first\n"},
            {"path": "references/notes.md", "content": "second\n"},
        ],
    )
    assert error and "names references/notes.md twice" in text, text
    assert _tree(folder) == before


async def test_list_and_create_agree_on_a_moved_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review R3: with the profile moved off HOME, a managed copy of a built-in is
    shown as THIS profile's editable skill, and a create over it is answered the
    way that row implies (``alias-occupied``: it exists and you may edit it), not
    "read-only, use Copy to my skills"."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    config_dir = tmp_path / "moved-profile"
    config_dir.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))
    monkeypatch.delenv("LOCAL_OPERATOR_SKILL_EXTRA_ROOTS", raising=False)
    context = ToolContext(cwd=str(config_dir), memory_config_dir=str(config_dir))
    moved = SimpleNamespace(
        home=home, config_dir=config_dir, context=context, tool=create_tools(context, ["skill"])[0]
    )
    builtin = _builtin_name()
    writes.create(config_dir, builtin, copy_from=builtin)
    error, text = await _call(moved, op="list")
    assert not error and f"- {builtin} [local, editable]" in text, text
    error, text = await _call(
        moved,
        op="create",
        name=builtin,
        files=[{"path": "SKILL.md", "content": _skill_md(builtin)}],
    )
    assert error and text.startswith("alias-occupied:"), text


# -- review round 2: the credential mask is a second lossy view (R6), R7, R8 -----------

_SECRET_MD = (
    "---\nname: db\ndescription: Database access.\n---\n"
    "Connect with these settings:\n"
    "password: changeme123\n"
    "Then run the migration.\n"
)


async def test_a_masked_file_cannot_be_replaced_whole_through_the_loop(env) -> None:
    """R6 (a): the model reads ``[redacted]`` where the file holds a value, while the
    fingerprint passes; a whole-file rewrite from that view is refused, bytes kept."""
    writes.create(env.config_dir, "db", files=[{"path": "SKILL.md", "content": _SECRET_MD}])
    folder = env.config_dir / "skills" / "db"
    before = _tree(folder)
    results: list[str] = []

    def plan(seen: str | None) -> dict[str, Any] | None:
        if seen is None:
            return {"op": "show", "name": "db"}
        results.append(seen)
        if len(results) == 1:
            page = _page_body(seen)
            assert "changeme123" not in page and "[redacted]" in page, "the model saw the mask"
            return {
                "op": "update",
                "name": "db",
                "expected_fingerprint": _fingerprint_line(seen),
                "files": [{"path": "SKILL.md", "content": page.replace("Then", "Next,")}],
            }
        return None

    await _drive(env, plan, masked=True)
    assert results[1].startswith("skill-file-masked:"), results[1]
    assert "op='edit'" in results[1]
    assert _tree(folder) == before


async def test_edit_fails_closed_on_masked_text_through_the_loop(env) -> None:
    """R6: ``edit`` cannot quote a masked line (0 matches) and cannot re-type one
    (adds a marker); an edit of unmasked text lands and carries the value over."""
    writes.create(env.config_dir, "db", files=[{"path": "SKILL.md", "content": _SECRET_MD}])
    folder = env.config_dir / "skills" / "db"
    before = _tree(folder)
    results: list[str] = []

    def plan(seen: str | None) -> dict[str, Any] | None:
        if seen is None:
            return {"op": "show", "name": "db"}
        results.append(seen)
        fingerprint = _fingerprint_line(results[0])
        steps = [
            {"path": "SKILL.md", "old_text": "password: [redacted]\n", "new_text": "x\n"},
            {
                "path": "SKILL.md",
                "old_text": "Then run the migration.\n",
                "new_text": "password: [redacted]\nThen run the migration.\n",
            },
            {
                "path": "SKILL.md",
                "old_text": "Then run the migration.\n",
                "new_text": "Then run the migration twice.\n",
            },
        ]
        if len(results) <= len(steps):
            return {
                "op": "edit",
                "name": "db",
                "expected_fingerprint": fingerprint,
                "edits": [steps[len(results) - 1]],
            }
        return None

    await _drive(env, plan, masked=True)
    assert results[1].startswith("skill-edit-no-unique-match:"), results[1]
    assert results[2].startswith("skill-marker-introduced:"), results[2]
    assert "old_text/new_text" in results[2]
    assert results[3].startswith("updated skill"), results[3]
    after = (folder / "SKILL.md").read_text()
    assert "password: changeme123\n" in after and "migration twice" in after
    assert "[redacted]" not in after
    assert before != _tree(folder)


async def test_a_write_that_adds_the_marker_is_refused_but_an_existing_one_is_kept(env) -> None:
    """R6 (b), the count rule: introducing the marker is refused on create and
    update; a file that already holds one (a doc about the mask) stays editable."""
    error, text = await _call(
        env,
        op="create",
        name="fresh",
        files=[{"path": "SKILL.md", "content": _skill_md("fresh", body="key: [redacted]\n")}],
    )
    assert error and text.startswith("skill-marker-introduced:"), text
    assert "old_text" not in text, "N2: create has no old_text"
    assert not (env.config_dir / "skills" / "fresh").exists()
    folder = _make_local(env)
    error, text = await _call(
        env,
        op="update",
        name="demo",
        expected_fingerprint=SkillDocument.from_directory(folder).fingerprint,
        files=[{"path": "references/new.md", "content": "token [redacted]\n"}],
    )
    assert error and text.startswith("skill-marker-introduced:"), text
    assert "old_text" not in text, "N2: update has no old_text"
    doc = _skill_md("about", body="The mask reads [redacted].\n")
    writes.create(env.config_dir, "about", files=[{"path": "SKILL.md", "content": doc}])
    about = env.config_dir / "skills" / "about"
    error, text = await _call(
        env,
        op="edit",
        name="about",
        expected_fingerprint=SkillDocument.from_directory(about).fingerprint,
        edits=[{"path": "SKILL.md", "old_text": "The mask", "new_text": "The harness mask"}],
    )
    assert not error, text


async def test_an_overlapping_old_text_is_not_unique(env) -> None:
    """R7: ``- x\\n- x`` matches ``- x\\n- x\\n- x\\n`` at two offsets; ``str.count``
    said one."""
    body = "- x\n- x\n- x\n"
    writes.create(
        env.config_dir, "lst", files=[{"path": "SKILL.md", "content": _skill_md("lst", body=body)}]
    )
    folder = env.config_dir / "skills" / "lst"
    before = _tree(folder)
    error, text = await _call(
        env,
        op="edit",
        name="lst",
        expected_fingerprint=SkillDocument.from_directory(folder).fingerprint,
        edits=[{"path": "SKILL.md", "old_text": "- x\n- x", "new_text": "- y\n- y"}],
    )
    assert error and "occurs 2 times" in text, text
    assert _tree(folder) == before


async def test_a_long_file_cannot_be_removed_so_remove_then_readd_cannot_replace_it(env) -> None:
    """R8: the whole-file bound covers removal too, so the two-call
    remove-then-re-add the reviewer probed stops at its first step."""
    long_text = "".join(f"reference line {index}\n" for index in range(800))
    assert len(long_text) > 4096
    writes.create(
        env.config_dir,
        "refs",
        files=[
            {"path": "SKILL.md", "content": _skill_md("refs")},
            {"path": "references/long.md", "content": long_text},
            {"path": "references/short.md", "content": "short\n"},
        ],
    )
    folder = env.config_dir / "skills" / "refs"
    before = _tree(folder)
    error, text = await _call(
        env,
        op="update",
        name="refs",
        expected_fingerprint=SkillDocument.from_directory(folder).fingerprint,
        remove_paths=["references/long.md"],
    )
    assert error and text.startswith("skill-file-too-long-to-replace:"), text
    assert "the operator can remove it on the Skills tab" in text
    assert "cannot be removed whole here" in text
    assert _tree(folder) == before
    error, text = await _call(
        env,
        op="update",
        name="refs",
        expected_fingerprint=SkillDocument.from_directory(folder).fingerprint,
        remove_paths=["references/short.md"],
    )
    assert not error, text
