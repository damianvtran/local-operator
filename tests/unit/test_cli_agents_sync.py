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
import shutil
from pathlib import Path

import pytest

from local_operator.agent_profiles import install_seed
from local_operator.agents import AgentRegistry
from local_operator.cli import (
    _seed_sync_command,
    _startup_surface,
    agents_sync_command,
    build_cli_parser,
)
from tests.unit.test_agent_profiles import _tree_bytes, publish_seed


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


def test_the_all_flag_names_every_installed_profile(isolated_config_dir: Path, capsys) -> None:
    """``--all`` is read by the handler, not just parsed (agent review round 1, n1).

    The flag is the explicit spelling of the set the absence of ``--name``
    selects — the two inputs are INTENTIONALLY equivalent, which is exactly why
    the handler must consult ``args.all`` rather than ignore it: an unread flag
    is a lie in ``--help``. This pins the advertised meaning end to end; the
    mutual-exclusivity contract stays covered above.
    """

    registry = AgentRegistry(isolated_config_dir)
    assert install_seed("reviewer", registry=registry) is not None
    assert install_seed("coder", registry=registry) is not None

    args = build_cli_parser().parse_args(["agents", "sync", "--all"])
    rc = agents_sync_command(args, registry, isolated_config_dir)

    out = capsys.readouterr().out
    assert rc == 0
    assert "reviewer: up-to-date" in out
    assert "coder: up-to-date" in out


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


def _scratch_seeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A scratch copy of the packaged seeds (ledger included), installed as the seed dir."""

    import local_operator.agent_profiles as agent_profiles

    seeds = tmp_path / "agent_seeds"
    shutil.copytree(Path(agent_profiles.SEEDS_DIR), seeds)
    monkeypatch.setattr(agent_profiles, "SEEDS_DIR", seeds)
    return seeds


def test_check_reports_a_behind_starter_and_writes_nothing(
    isolated_config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """#2060 defect 2: ``--check`` used to skip the seed arm entirely.

    A behind, provably-clean starter answered "not installed" under --name and
    "nothing to sync" without one. It must now be REPORTED - and writing
    anything would have broken the promise the flag makes, so every byte under
    the config dir is asserted unchanged, byte level.
    """

    seeds = _scratch_seeds(tmp_path, monkeypatch)
    registry = AgentRegistry(isolated_config_dir)
    assert install_seed("reviewer", registry=registry) is not None
    publish_seed(seeds, "reviewer", version="2.0.0", body="REVIEWER v2 GUIDANCE")
    before = _tree_bytes(isolated_config_dir / "agents")

    args = build_cli_parser().parse_args(["agents", "sync", "--check"])
    rc = agents_sync_command(args, registry, isolated_config_dir)

    out = capsys.readouterr().out
    assert rc == 0
    assert "update available" in out
    assert "not installed" not in out
    # The seed-owned surface is byte-identical, and no notice was queued. (The
    # hub arm still refreshes its own status store under --check - documented
    # behaviour, and outside this promise.)
    assert _tree_bytes(isolated_config_dir / "agents") == before
    assert not (isolated_config_dir / ".seed-notices.json").exists()


def test_dry_run_writes_nothing_for_the_seed_arm(
    isolated_config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """The F2 trap: ``--dry-run`` ("Show what would change; write nothing") WROTE.

    With a clean+behind row the seed arm updated in place under ``--dry-run`` -
    seed_version moved on disk as the command printed "1 updated". The arm is
    now classified read-only for this flag and the bytes are asserted.
    """

    seeds = _scratch_seeds(tmp_path, monkeypatch)
    registry = AgentRegistry(isolated_config_dir)
    assert install_seed("reviewer", registry=registry) is not None
    publish_seed(seeds, "reviewer", version="2.0.0", body="REVIEWER v2 GUIDANCE")
    before = _tree_bytes(isolated_config_dir / "agents")

    args = build_cli_parser().parse_args(["agents", "sync", "--dry-run"])
    rc = agents_sync_command(args, registry, isolated_config_dir)

    out = capsys.readouterr().out
    assert rc == 0
    assert "update available" in out
    assert "updated to the packaged starter" not in out
    assert _tree_bytes(isolated_config_dir / "agents") == before
    assert not (isolated_config_dir / ".seed-notices.json").exists()


def test_an_unconfirmed_replace_never_forces_the_seed_arm(
    isolated_config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """D1: ``--replace`` without ``--yes`` must still change nothing.

    The seed arm runs BEFORE ``_hub_sync_run`` validates the pair, so a force
    derived from ``replace`` alone would have overwritten an edited row moments
    before the hub arm refused the invocation. Only ``replace AND yes`` (or the
    deprecated ``--force``) grants it; the confirmation form right after proves
    the force path still works.
    """

    seeds = _scratch_seeds(tmp_path, monkeypatch)
    registry = AgentRegistry(isolated_config_dir)
    assert install_seed("reviewer", registry=registry) is not None
    installed = registry.get_agent_by_name("reviewer")
    assert installed is not None
    registry.set_agent_system_prompt(installed.id, "MY EDITED PROMPT")
    publish_seed(seeds, "reviewer", version="2.0.0", body="REVIEWER v2 GUIDANCE")

    refused = build_cli_parser().parse_args(["agents", "sync", "--replace"])
    rc = agents_sync_command(refused, registry, isolated_config_dir)
    assert rc == 1
    assert "confirm with --yes" in capsys.readouterr().out
    assert registry.get_agent_system_prompt(installed.id) == "MY EDITED PROMPT"

    confirmed = build_cli_parser().parse_args(["agents", "sync", "--replace", "--yes"])
    rc = agents_sync_command(confirmed, registry, isolated_config_dir)
    assert rc == 0
    assert registry.get_agent_system_prompt(installed.id).strip() == "REVIEWER v2 GUIDANCE"


def test_replace_yes_applies_seeds_even_when_the_hub_arm_then_errors(
    isolated_config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """D1's accepted trade, asserted: the pair IS the user's discard instruction.

    The seed and hub families are independent; a hub refusal after the seed arm
    has applied (HubBusy here) must not retroactively make the seed update
    wrong - the invocation asked for the packaged text and got it.
    """

    from local_operator.hub_sync import service as svc

    def busy(*args: object, **kwargs: object) -> object:
        raise svc.HubBusy("another writer holds the hub store")

    monkeypatch.setattr("local_operator.hub_sync.service.apply_items", busy)
    seeds = _scratch_seeds(tmp_path, monkeypatch)
    registry = AgentRegistry(isolated_config_dir)
    assert install_seed("reviewer", registry=registry) is not None
    installed = registry.get_agent_by_name("reviewer")
    assert installed is not None
    registry.set_agent_system_prompt(installed.id, "MY EDITED PROMPT")
    publish_seed(seeds, "reviewer", version="2.0.0", body="REVIEWER v2 GUIDANCE")

    args = build_cli_parser().parse_args(["agents", "sync", "--replace", "--yes"])
    rc = agents_sync_command(args, registry, isolated_config_dir)

    assert rc == 1  # the hub arm reported HubBusy
    assert "hub store" in capsys.readouterr().out


def test_an_unconfirmed_replace_changes_nothing_on_a_clean_row(
    isolated_config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """R1-5/Q2, byte level: the refusal runs BEFORE the seed arm can write.

    The pre-fix order applied every clean-but-behind starter and THEN printed
    "confirm with --yes" - the refusal text and the addendum's "changes
    nothing" promise both lied. The flag validation is now hoisted ahead of
    the seed arm, so a clean, behind row on the refusing invocation is
    untouched down to the byte.
    """

    seeds = _scratch_seeds(tmp_path, monkeypatch)
    registry = AgentRegistry(isolated_config_dir)
    assert install_seed("reviewer", registry=registry) is not None
    publish_seed(seeds, "reviewer", version="2.0.0", body="REVIEWER v2 GUIDANCE")
    installed = registry.get_agent_by_name("reviewer")
    assert installed is not None
    version_before = next(tag for tag in installed.tags if tag.startswith("seed_version:"))
    prompt_before = registry.get_agent_system_prompt(installed.id)
    before = _tree_bytes(isolated_config_dir)

    args = build_cli_parser().parse_args(["agents", "sync", "--replace"])
    rc = agents_sync_command(args, registry, isolated_config_dir)

    out = capsys.readouterr().out
    assert rc == 1
    assert "confirm with --yes" in out
    assert _tree_bytes(isolated_config_dir) == before
    refreshed = AgentRegistry(isolated_config_dir).get_agent_by_name("reviewer")
    assert refreshed is not None
    assert next(tag for tag in refreshed.tags if tag.startswith("seed_version:")) == version_before
    assert registry.get_agent_system_prompt(refreshed.id) == prompt_before


def test_check_with_merge_flags_refuses_before_any_seed_write(
    isolated_config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """``--check --replace --yes`` is refused before the seed arm, byte level."""

    seeds = _scratch_seeds(tmp_path, monkeypatch)
    registry = AgentRegistry(isolated_config_dir)
    assert install_seed("reviewer", registry=registry) is not None
    publish_seed(seeds, "reviewer", version="2.0.0", body="REVIEWER v2 GUIDANCE")
    before = _tree_bytes(isolated_config_dir)

    args = build_cli_parser().parse_args(["agents", "sync", "--check", "--replace", "--yes"])
    rc = agents_sync_command(args, registry, isolated_config_dir)

    out = capsys.readouterr().out
    assert rc == 1
    assert "drop the other merge flags" in out
    # The refusal now runs BEFORE the seed arm: pre-fix this output also
    # carried the arm's "update available" report first (the ordering bug).
    assert "update available" not in out
    assert _tree_bytes(isolated_config_dir) == before


def test_force_prints_exactly_one_deprecation_warning(
    isolated_config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """Hoisting the validation must not double the ``--force`` warning.

    ``_prepare_sync_flags`` runs twice for ``agents sync`` (once before the
    seed arm, once in ``_hub_sync_run``); normalisation is idempotent, so the
    warning fires once and the flag still applies like ``--replace --yes``.
    """

    seeds = _scratch_seeds(tmp_path, monkeypatch)
    registry = AgentRegistry(isolated_config_dir)
    assert install_seed("reviewer", registry=registry) is not None
    publish_seed(seeds, "reviewer", version="2.0.0", body="REVIEWER v2 GUIDANCE")

    args = build_cli_parser().parse_args(["agents", "sync", "--force"])
    rc = agents_sync_command(args, registry, isolated_config_dir)

    captured = capsys.readouterr()
    assert rc == 0
    assert captured.err.count("--force is deprecated") == 1
    row = registry.get_agent_by_name("reviewer")
    assert row is not None
    assert registry.get_agent_system_prompt(row.id).strip() == "REVIEWER v2 GUIDANCE"


def test_startup_surface_separates_daemons_from_human_surfaces() -> None:
    """``lop serve`` and each foreground ``serve`` form are daemons (R1-1/D3).

    A notice shown by a process whose stderr is a log nobody reads is a notice
    nobody saw, so those launches are REPORT-ONLY. Every other terminal
    command - ``agents list`` and ``mobile start`` (a control command a person
    types) included - keeps the ``cli`` slot; a bare launch keeps the TUI
    gate. The supervised LaunchAgents never reach the seam at all (they run
    ``-m local_operator.<unit>``), so these spellings are the whole set.
    """

    parser = build_cli_parser()
    for argv in (
        ["serve"],
        ["wake", "serve"],
        ["mobile", "serve"],
        ["browser", "serve"],
        ["tunnel", "serve"],
        ["network", "serve"],
    ):
        assert _startup_surface(parser.parse_args(argv)) == "daemon", argv
    for argv in (
        ["wake", "status"],
        ["mobile", "status"],
        ["mobile", "start"],
        ["tunnel", "status"],
        ["agents", "list"],
        ["agents", "sync"],
        ["--no-tui"],
    ):
        assert _startup_surface(parser.parse_args(argv)) == "cli", argv
    assert _startup_surface(parser.parse_args(["--tui"])) == "tui"


def test_seed_sync_command_carves_out_the_switch_edit() -> None:
    """The second no-write carve-out: editing the pass's own switch (U6b)."""

    parser = build_cli_parser()
    switch_edit = parser.parse_args(["config", "edit", "agents.auto_update.seeds", "false"])
    assert _seed_sync_command(switch_edit) == "config edit agents.auto_update.seeds"
    assert _seed_sync_command(parser.parse_args(["config", "edit", "model_name", "x"])) is None
    assert _seed_sync_command(parser.parse_args(["agents", "sync"])) == "agents sync"
    assert _seed_sync_command(parser.parse_args(["agents", "list"])) is None
