"""Dry exec selection must match the composition root before any key check."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from local_operator import exec_mode
from local_operator.config import ConfigManager
from local_operator.exec_mode import ExecArgs
from local_operator.paths import CONFIG_DIR_ENV
from tests.unit.session.test_model_selection import _journal, _selection_row

SID = "abc123def456"


@pytest.fixture
def resume_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "config"
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv(CONFIG_DIR_ENV, str(root))
    ConfigManager(root).update_config({"hosting": "radient", "model_name": "global-model"})
    _journal(root / "sessions" / SID, [_selection_row("deepseek/deepseek-chat")])
    return root


def test_dry_resume_uses_saved_selection_before_global_key_preflight(resume_root: Path) -> None:
    assert exec_mode.resolve_hosting_model_dry(ExecArgs(resume=SID)) == (
        "deepseek",
        "deepseek-chat",
    )


@pytest.mark.parametrize("background", [False, True])
def test_dry_resume_explicit_flags_still_override_saved_selection(
    resume_root: Path, background: bool
) -> None:
    assert exec_mode.resolve_hosting_model_dry(
        ExecArgs(resume=SID, background=background, hosting="test", model="override")
    ) == ("test", "override")
    assert exec_mode.resolve_hosting_model_dry(
        ExecArgs(resume=SID, background=background, model="saved-host-model")
    ) == ("deepseek", "saved-host-model")


def test_dry_resume_saved_selection_beats_team_suggestion(resume_root: Path) -> None:
    from local_operator.teams import ModelSuggestion, TeamEditFields, TeamRegistry

    TeamRegistry(resume_root).create_team(
        TeamEditFields(
            name="resume-team",
            manager="manager",
            instructions="synthetic",
            model_suggestion=ModelSuggestion(hosting="openrouter", model="team-model"),
        )
    )
    assert exec_mode.resolve_hosting_model_dry(ExecArgs(resume=SID, team="resume-team")) == (
        "deepseek",
        "deepseek-chat",
    )


def test_dry_resume_saved_selection_beats_later_agent_model_edit(resume_root: Path) -> None:
    from local_operator.agents import AgentEditFields, AgentRegistry

    agent = AgentRegistry(resume_root).create_agent(
        AgentEditFields(
            name="resume-agent",
            security_prompt=None,
            hosting="anthropic",
            model="agent-model",
            description=None,
            last_message=None,
            temperature=None,
            tags=[],
            categories=[],
            top_p=None,
            top_k=None,
            max_tokens=None,
            stop=None,
            frequency_penalty=None,
            presence_penalty=None,
            seed=None,
            current_working_directory=None,
        )
    )
    assert exec_mode.resolve_hosting_model_dry(ExecArgs(resume=SID, agent_id=agent.id)) == (
        "deepseek",
        "deepseek-chat",
    )


@pytest.mark.parametrize("saved", [None, "unknown-vendor/not-a-model"])
def test_dry_resume_legacy_or_invalid_selection_preserves_birth_precedence(
    resume_root: Path, saved: str | None
) -> None:
    _journal(resume_root / "sessions" / SID, [] if saved is None else [_selection_row(saved)])
    assert exec_mode.resolve_hosting_model_dry(ExecArgs(resume=SID)) == ("radient", "global-model")
    assert exec_mode.resolve_hosting_model_dry(ExecArgs(resume="missing-history")) == (
        "radient",
        "global-model",
    )


def test_background_spawn_preflights_saved_selection_without_model_override(
    resume_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pairs = []
    resolve = exec_mode.resolve_hosting_model_dry

    def observe(args):
        pair = resolve(args)
        pairs.append(pair)
        return pair

    monkeypatch.setattr(exec_mode, "resolve_hosting_model_dry", observe)
    from local_operator import macos_disclaim

    class StopBeforeDetach(Exception):
        pass

    # macOS uses native posix_spawn, not Popen. Stop at the actual shared
    # boundary: this selector test must neither detach nor fabricate a worker.
    process = MagicMock(side_effect=StopBeforeDetach)
    forbidden_spawn = MagicMock(side_effect=AssertionError("unexpected spawn boundary"))
    monkeypatch.setattr(macos_disclaim, "spawn_disclaimed", forbidden_spawn)
    monkeypatch.setattr(exec_mode.subprocess, "Popen", forbidden_spawn)
    monkeypatch.setattr(exec_mode, "spawn_disclaimed", process)
    with pytest.raises(StopBeforeDetach):
        exec_mode._spawn_background("synthetic request", ExecArgs(resume=SID, background=True))
    process.assert_called_once()
    forbidden_spawn.assert_not_called()
    assert pairs == [("deepseek", "deepseek-chat")]
    argv = process.call_args.args[0]
    assert any(SID in str(arg) for arg in argv)
    assert not any(str(arg).startswith("--hosting") for arg in argv)
