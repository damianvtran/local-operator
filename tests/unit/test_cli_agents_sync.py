"""``lop agents sync``: the CLI surface over the shared sync coordinator.

The parser is pinned here (a new subcommand is additive to the golden legacy
surface, so only a dedicated test notices if it disappears) and the handler is
driven directly, the way ``test_radient_hub_base_resolution`` drives
``agents_delete_command``: the CLI's job is to resolve the hub client, call the
one coordinator, and print its report — anything smarter belongs in
``agent_sync`` and is tested there.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from local_operator.agent_profiles import install_seed
from local_operator.agents import AgentRegistry
from local_operator.cli import agents_sync_command, build_cli_parser


@pytest.fixture()
def parser() -> argparse.ArgumentParser:
    return build_cli_parser()


@pytest.fixture()
def isolated_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A config dir no run of these tests can escape from.

    The handler constructs real ``ConfigManager``/``AgentRegistry`` instances
    from the directory it is handed and resolves a real credential, so
    ``Path.home()`` is redirected as well and ``RADIENT_API_KEY`` is cleared —
    otherwise the operator's own login would turn the degradation test into a
    live hub call.
    """

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("RADIENT_API_KEY", raising=False)
    return tmp_path / ".local-operator"


def test_the_parser_accepts_the_documented_flags(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args(["agents", "sync", "--name", "reviewer", "--force"])
    assert args.agents_command == "sync"
    assert args.name == "reviewer"
    assert args.force is True
    assert args.all is False

    plain = parser.parse_args(["agents", "sync"])
    assert plain.name is None and plain.force is False

    explicit_all = parser.parse_args(["agents", "sync", "--all"])
    assert explicit_all.all is True


def test_name_and_all_are_mutually_exclusive(parser: argparse.ArgumentParser) -> None:
    with pytest.raises(SystemExit):
        parser.parse_args(["agents", "sync", "--name", "reviewer", "--all"])


def test_the_command_reports_an_installed_starter_as_current(
    isolated_config_dir: Path, capsys
) -> None:
    registry = AgentRegistry(isolated_config_dir)
    assert install_seed("reviewer", registry=registry) is not None

    args = build_cli_parser().parse_args(["agents", "sync"])
    rc = agents_sync_command(args, registry, isolated_config_dir)

    out = capsys.readouterr().out
    assert rc == 0
    assert "reviewer: up-to-date" in out
    assert "1 up-to-date." in out


def test_the_command_syncs_only_the_named_profile(isolated_config_dir: Path, capsys) -> None:
    registry = AgentRegistry(isolated_config_dir)
    assert install_seed("reviewer", registry=registry) is not None
    assert install_seed("coder", registry=registry) is not None

    args = build_cli_parser().parse_args(["agents", "sync", "--name", "coder"])
    rc = agents_sync_command(args, registry, isolated_config_dir)

    out = capsys.readouterr().out
    assert rc == 0
    assert "coder: up-to-date" in out
    assert "reviewer" not in out


def test_the_command_answers_a_name_that_is_not_installed(
    isolated_config_dir: Path, capsys
) -> None:
    registry = AgentRegistry(isolated_config_dir)

    args = build_cli_parser().parse_args(["agents", "sync", "--name", "reviewer"])
    rc = agents_sync_command(args, registry, isolated_config_dir)

    out = capsys.readouterr().out
    assert rc == 0
    assert "reviewer: not installed" in out
    assert "op='install'" in out
