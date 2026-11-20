"""``write``/``edit`` refuse a path rooted at an unexpanded shell variable.

File tools expand ``~`` and nothing else, so ``$LOCAL_OPERATOR_SCRATCHPAD/x.md``
used to resolve to ``<cwd>/$LOCAL_OPERATOR_SCRATCHPAD/x.md`` and ``write`` created a
directory literally named ``$LOCAL_OPERATOR_SCRATCHPAD`` — under the home, it had
collected 36 entries from many sessions by the 2026-11-20 audit. These tests drive
the REAL tool executors (``create_tools`` -> ``execute_write``/``execute_edit``)
with ``HOME`` pointed into ``tmp_path``, so a ``~/`` spelling cannot touch the
operator's home, and assert on the filesystem, not just the error text.

The negative controls matter as much as the refusals: a ``$`` that is part of a
real file name (``price$5.md``) must keep working, which is why the check is
narrowed to a bare variable reference as the FIRST component with a path under it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator.harness.types import FAULT_KEY, AgentTool, ToolContext
from local_operator.tools.registry import create_tools


def _setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ToolContext, Path, Path, Path, dict[str, AgentTool]]:
    """A cwd, an isolated HOME and a scratchpad root, all inside ``tmp_path``."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    workspace = tmp_path / "ws"
    workspace.mkdir()
    pad = tmp_path / "sessions" / "sess-x" / "scratchpad"
    context = ToolContext(
        cwd=str(workspace),
        session_id="env-var-path-test",
        scratchpad_dir=str(pad),
        request_approval=_approve,
    )
    return context, workspace, home, pad, {tool.name: tool for tool in create_tools(context)}


async def _approve(tier: str, description: str) -> bool:
    return True


def _dollar_entries(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if "$" in p.name]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("$LOCAL_OPERATOR_SCRATCHPAD/x.md", 'write(path="scratchpad://x.md")'),
        ("${LOCAL_OPERATOR_SCRATCHPAD}/x.md", 'write(path="scratchpad://x.md")'),
        ("~/$LOCAL_OPERATOR_SCRATCHPAD/x.md", 'write(path="scratchpad://x.md")'),
        ("  $LOCAL_OPERATOR_SCRATCHPAD//runs/deep.csv", 'write(path="scratchpad://runs/deep.csv")'),
        ("$LOCAL_OPERATOR_SCRATCHPAD/", 'write(path="scratchpad://<name>")'),
        ("$HOME/notes/x.md", 'write(path="~/notes/x.md")'),
        ("$FOO/x.md", "Pass the absolute path instead"),
        ("~/${FOO}/x.md", "Pass the absolute path instead"),
    ],
)
async def test_write_refuses_a_variable_root_and_creates_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path: str, expected: str
) -> None:
    context, workspace, home, pad, tools = _setup(tmp_path, monkeypatch)

    result = await tools["write"].execute(
        "c", {"path": path, "content": "hi\n"}, None, None, context
    )

    assert result.is_error is True
    assert "write does not expand environment variables" in result.text
    assert "nothing was written" in result.text
    assert expected in result.text
    assert result.details is not None and result.details[FAULT_KEY] == "invalid_arguments"
    where = "your home directory" if path.strip().startswith("~") else "the working directory"
    assert f"under {where}" in result.text
    # The point of the fix: no literal ``$NAME`` directory anywhere — not in the
    # cwd, not in the (isolated) home — and the scratchpad was not created either.
    assert _dollar_entries(tmp_path) == []
    assert not pad.exists()


@pytest.mark.asyncio
async def test_edit_refuses_a_variable_root_and_leaves_existing_litter_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A literal ``$LOCAL_OPERATOR_SCRATCHPAD`` directory already exists (other
    sessions hold receipts naming it), so ``edit`` WOULD find the file. It must
    still refuse, and neither the refusal nor a refused ``write`` may touch it."""
    context, workspace, home, pad, tools = _setup(tmp_path, monkeypatch)
    litter = workspace / "$LOCAL_OPERATOR_SCRATCHPAD"
    litter.mkdir()
    existing = litter / "x.md"
    existing.write_bytes(b"alpha\nbeta\n")
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))

    for spelling in ("$LOCAL_OPERATOR_SCRATCHPAD/x.md", "${LOCAL_OPERATOR_SCRATCHPAD}/new.md"):
        edit = await tools["edit"].execute(
            "c", {"path": spelling, "old_text": "alpha", "new_text": "gamma"}, None, None, context
        )
        assert edit.is_error is True
        assert "edit does not expand environment variables" in edit.text
        assert "nothing was changed" in edit.text
        assert 'edit(path="scratchpad://' in edit.text
        assert edit.details is not None and edit.details[FAULT_KEY] == "invalid_arguments"

    write = await tools["write"].execute(
        "c",
        {"path": "$LOCAL_OPERATOR_SCRATCHPAD/x.md", "content": "clobber\n"},
        None,
        None,
        context,
    )
    assert write.is_error is True

    assert existing.read_bytes() == b"alpha\nbeta\n"
    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*")) == before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    [
        "a$b.md",
        "notes/price$5.md",
        "notes/$draft/x.md",  # not the first component
        "$FOO.bak/x.md",  # the component only STARTS with a reference
        "${FOO}bar/x.md",
        "$5/x.md",  # not an identifier: not a variable a shell would expand
        "$FOO",  # a bare file name, not a root anything is created under
    ],
)
async def test_a_dollar_that_is_part_of_a_real_name_still_writes_and_edits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    context, workspace, home, pad, tools = _setup(tmp_path, monkeypatch)

    write = await tools["write"].execute(
        "c", {"path": path, "content": "one\n"}, None, None, context
    )
    assert write.is_error is False, write.text
    target = workspace / path
    assert target.read_text(encoding="utf-8") == "one\n"

    edit = await tools["edit"].execute(
        "c", {"path": path, "old_text": "one", "new_text": "two"}, None, None, context
    )
    assert edit.is_error is False, edit.text
    assert target.read_text(encoding="utf-8") == "two\n"


@pytest.mark.asyncio
async def test_the_scratchpad_scheme_the_refusal_names_still_works(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The remedy is only actionable if the spelling it names succeeds."""
    context, workspace, home, pad, tools = _setup(tmp_path, monkeypatch)

    write = await tools["write"].execute(
        "c", {"path": "scratchpad://x.md", "content": "one\n"}, None, None, context
    )
    assert write.is_error is False, write.text
    edit = await tools["edit"].execute(
        "c",
        {"path": "scratchpad://x.md", "old_text": "one", "new_text": "two"},
        None,
        None,
        context,
    )
    assert edit.is_error is False, edit.text
    assert (pad / "x.md").read_text(encoding="utf-8") == "two\n"
    assert _dollar_entries(tmp_path) == []
