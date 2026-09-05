"""The legacy HTTP edit route reads only its workspace or the supplied buffer."""

import os
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from local_operator.agents import AgentEditFields
from local_operator.server.routes import chat


@pytest.fixture
def edit_case(test_app_client, dummy_registry, tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "note.txt"
    target.write_text("workspace original")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside private sentinel")
    agent = dummy_registry.create_agent(
        AgentEditFields.model_validate(
            {"name": "Editor", "current_working_directory": str(workspace)}
        )
    )
    executor = MagicMock()
    executor.invoke_model = AsyncMock(
        return_value=SimpleNamespace(
            content=(
                "<replacements>\n<<<<<<< SEARCH\nworkspace original\n=======\n"
                "workspace edited\n>>>>>>> REPLACE\n</replacements>"
            )
        )
    )
    factory = MagicMock(return_value=SimpleNamespace(executor=executor))
    monkeypatch.setattr(chat, "create_operator", factory)
    return SimpleNamespace(
        client=test_app_client,
        registry=dummy_registry,
        workspace=workspace,
        target=target,
        outside=outside,
        agent=agent,
        executor=executor,
        factory=factory,
        url=f"/v1/chat/agents/{agent.id}/edit",
    )


async def _post(case, path, **extra):
    return await case.client.post(
        case.url,
        json={
            "file_path": str(path),
            "edit_prompt": "Edit this",
            "hosting": "test",
            "model": "test-model",
            **extra,
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("relative", [True, False])
async def test_workspace_file_reaches_model_without_writes(edit_case, relative):
    case = edit_case
    response = await _post(case, "note.txt" if relative else case.target)
    assert response.status_code == 200, response.text
    assert response.json()["result"]["file_path"] == str(case.target.resolve())
    assert response.json()["result"]["edit_diffs"] == [
        {"find": "workspace original", "replace": "workspace edited"}
    ]
    prompt = case.executor.append_to_history.call_args.args[0].content
    assert "workspace original" in prompt
    assert "outside private sentinel" not in prompt
    assert case.target.read_text() == "workspace original"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["absolute", "traversal", "prefix-sibling", "symlink"])
async def test_outside_file_is_denied_before_model_creation(edit_case, mode):
    case = edit_case
    target = case.outside
    if mode == "traversal":
        target = "../outside.txt"
    elif mode == "prefix-sibling":
        sibling = case.workspace.with_name("workspace-other")
        sibling.mkdir()
        target = sibling / "note.txt"
        target.write_text("outside private sentinel")
    elif mode == "symlink":
        target = case.workspace / "linked.txt"
        target.symlink_to(case.outside)
    response = await _post(case, target)
    assert response.status_code == 403
    assert response.json()["detail"] == "File is outside the agent workspace"
    case.factory.assert_not_called()
    assert case.outside.read_text() == "outside private sentinel"


@pytest.mark.asyncio
async def test_in_workspace_symlink_is_allowed(edit_case):
    case = edit_case
    link = case.workspace / "linked.txt"
    link.symlink_to(case.target)
    response = await _post(case, link)
    assert response.status_code == 200, response.text
    assert response.json()["result"]["file_path"] == str(case.target.resolve())


@pytest.mark.asyncio
@pytest.mark.parametrize("workspace", [None, "", " ", "relative", "missing", "file"])
async def test_unavailable_workspace_fails_closed(edit_case, workspace):
    case = edit_case
    if workspace == "missing":
        workspace = str(case.workspace / "missing")
    elif workspace == "file":
        workspace = str(case.target)
    # Include an absent setting without relying on model coercion/defaults.
    case.registry._agents[case.agent.id].current_working_directory = workspace
    response = await _post(case, case.target)
    assert response.status_code == 403
    assert response.json()["detail"] == "Agent workspace is unavailable"
    case.factory.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["", "unsaved editor buffer"])
async def test_supplied_buffer_has_no_path_or_workspace_io(edit_case, content, monkeypatch):
    case = edit_case
    case.registry._agents[case.agent.id].current_working_directory = None

    def forbidden(*args, **kwargs):
        raise AssertionError("buffer mode must not resolve or read a host path")

    monkeypatch.setattr(chat, "_read_edit_workspace_file", forbidden)
    monkeypatch.setattr(chat, "FilePath", forbidden)
    case.executor.invoke_model.return_value = SimpleNamespace(
        content=(
            f"<replacements>\n<<<<<<< SEARCH\n{content}\n=======\n"
            "buffer edited\n>>>>>>> REPLACE\n</replacements>"
        )
    )
    response = await _post(case, "~/not-a-server-file/../display.txt", file_content=content)
    assert response.status_code == 200, response.text
    assert response.json()["result"]["file_path"] == "~/not-a-server-file/../display.txt"
    prompt = case.executor.append_to_history.call_args.args[0].content
    assert f"<file_content>\n{content}\n</file_content>" in prompt
    assert "outside private sentinel" not in prompt


@pytest.mark.asyncio
async def test_explicit_null_still_enforces_workspace(edit_case):
    response = await _post(edit_case, edit_case.outside, file_content=None)
    assert response.status_code == 403
    edit_case.factory.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind, status", [("missing", 404), ("directory", 400), ("invalid", 400), ("binary", 400)]
)
async def test_invalid_workspace_targets_are_client_errors(edit_case, kind, status):
    case = edit_case
    target = case.workspace / "missing"
    if kind == "directory":
        target = case.workspace
    elif kind == "invalid":
        target = "bad\x00path"
    elif kind == "binary":
        target = case.workspace / "binary.txt"
        target.write_bytes(b"\xff\xfe")
    response = await _post(case, target)
    assert response.status_code == status
    case.factory.assert_not_called()


@pytest.mark.asyncio
async def test_unknown_agent_remains_not_found(edit_case):
    edit_case.url = "/v1/chat/agents/missing/edit"
    response = await _post(edit_case, edit_case.target, file_content="buffer")
    assert response.status_code == 404
    edit_case.factory.assert_not_called()


@pytest.mark.asyncio
async def test_explicit_filesystem_root_is_a_trusted_workspace(edit_case):
    case = edit_case
    case.registry._agents[case.agent.id].current_working_directory = case.target.anchor
    response = await _post(case, case.target)
    assert response.status_code == 200, response.text


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "nt", reason="Native Windows path semantics")
async def test_windows_other_drive_is_denied(edit_case):
    case = edit_case
    other_drive = "Z:" if case.workspace.drive.upper() != "Z:" else "Y:"
    response = await _post(case, other_drive + "\\outside.txt")
    assert response.status_code == 403
    case.factory.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "nt", reason="Native Windows junction semantics")
async def test_windows_junction_escape_is_denied(edit_case):
    case = edit_case
    link = case.workspace / "junction"
    subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(case.outside.parent)],
        check=True,
        capture_output=True,
    )
    response = await _post(case, link / case.outside.name)
    assert response.status_code == 403
    case.factory.assert_not_called()
