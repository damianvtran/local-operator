import json
import logging
import tempfile
from argparse import Namespace
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

from local_operator.config import (
    DEFAULT_CONFIG,
    Config,
    ConfigManager,
    ConfigWriteRefused,
)


@pytest.fixture
def temp_config_dir():
    """Create a temporary directory for config files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


def test_config_initialization():
    """Test Config class initialization with dictionary."""
    config_dict = {
        "version": "1.0.0",
        "metadata": {
            "created_at": "",
            "last_modified": "",
            "description": "Local Operator configuration file",
        },
        "values": {
            "conversation_length": 5,
            "detail_length": 3,
            "hosting": "test_host",
            "model_name": "test_model",
        },
    }
    config = Config(config_dict)

    assert config.version == "1.0.0"
    assert config.metadata["description"] == "Local Operator configuration file"
    assert config.get_value("conversation_length") == 5
    assert config.get_value("detail_length") == 3
    assert config.get_value("hosting") == "test_host"
    assert config.get_value("model_name") == "test_model"


@patch("local_operator.config.version")
def test_config_initialization_with_default_version(mock_version):
    """Test Config class initialization with default version."""
    mock_version.return_value = "2.0.0"
    config = Config({})
    assert config.version == "2.0.0"


def test_config_manager_initialization(temp_config_dir):
    """Test ConfigManager initialization creates config file if not exists."""
    config_manager = ConfigManager(temp_config_dir)

    assert config_manager.config.version is not None
    assert isinstance(config_manager.config, Config)
    assert config_manager.get_config_value("conversation_length") == DEFAULT_CONFIG.get_value(
        "conversation_length"
    )
    assert config_manager.get_config_value("providers") == DEFAULT_CONFIG.get_value("providers")


def test_config_managers_do_not_share_nested_defaults(temp_config_dir):
    """A provider setup in one fresh config must not alter another."""
    first = ConfigManager(temp_config_dir / "first")
    first_search = dict(first.get_config_value("web_search"))
    first_search["providers"] = ["searxng"]
    first_search["searxng_endpoint"] = "https://search.example.test"
    first.config.set_value("web_search", first_search)

    second = ConfigManager(temp_config_dir / "second")

    assert second.get_config_value("web_search") == DEFAULT_CONFIG.get_value("web_search")


@patch("local_operator.config.version")
def test_config_manager_version_warning(mock_version, temp_config_dir, capsys):
    """Test ConfigManager warns about old config versions."""
    mock_version.return_value = "1.0.0"

    # Create config file with old version
    test_config = {
        "version": "2.0.0",
        "metadata": {
            "created_at": "",
            "last_modified": "",
            "description": "Local Operator configuration file",
        },
        "values": {},
    }
    config_file = temp_config_dir / "config.yml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(test_config, f)

    ConfigManager(temp_config_dir)
    captured = capsys.readouterr()
    # stderr: ConfigManager is constructed on the `exec --json` path, so a
    # warning on stdout is a non-JSON line in the middle of the event stream.
    assert (
        "Warning: Your config file version (2.0.0) is newer than the current version (1.0.0)"
        in captured.err
    )


def test_config_manager_load_existing(temp_config_dir):
    """Test ConfigManager loads existing config file."""
    test_config = {
        "version": "1.0.0",
        "metadata": {
            "created_at": "",
            "last_modified": "",
            "description": "Local Operator configuration file",
        },
        "values": {
            "conversation_length": 20,
            "detail_length": 15,
            "hosting": "custom_host",
            "model_name": "custom_model",
        },
    }

    config_file = temp_config_dir / "config.yml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(test_config, f)

    config_manager = ConfigManager(temp_config_dir)
    assert config_manager.get_config_value("conversation_length") == 20
    assert config_manager.get_config_value("hosting") == "custom_host"


def test_config_manager_load_missing_file(temp_config_dir):
    """Test ConfigManager loads default config when file doesn't exist."""
    config_file = temp_config_dir / "nonexistent.yml"

    config_manager = ConfigManager(config_file)

    # Should create file with default values
    assert config_manager.config.version == DEFAULT_CONFIG.version
    assert config_manager.get_config_value("conversation_length") == DEFAULT_CONFIG.get_value(
        "conversation_length"
    )
    assert config_manager.get_config_value("detail_length") == DEFAULT_CONFIG.get_value(
        "detail_length"
    )
    assert config_manager.get_config_value("hosting") == DEFAULT_CONFIG.get_value("hosting")
    assert config_manager.get_config_value("model_name") == DEFAULT_CONFIG.get_value("model_name")


def test_config_manager_load_empty_file(temp_config_dir):
    """Test ConfigManager loads default config when file is empty."""
    config_file = temp_config_dir / "config.yml"
    config_file.touch()  # Create empty file

    config_manager = ConfigManager(temp_config_dir)

    # Should load default values
    assert config_manager.config.version == DEFAULT_CONFIG.version
    assert config_manager.get_config_value("conversation_length") == DEFAULT_CONFIG.get_value(
        "conversation_length"
    )
    assert config_manager.get_config_value("detail_length") == DEFAULT_CONFIG.get_value(
        "detail_length"
    )
    assert config_manager.get_config_value("hosting") == DEFAULT_CONFIG.get_value("hosting")
    assert config_manager.get_config_value("model_name") == DEFAULT_CONFIG.get_value("model_name")


def test_config_manager_load_partial_values(temp_config_dir):
    """Test ConfigManager loads default values for missing fields."""
    test_config = {
        "version": "1.0.0",
        "metadata": {"created_at": "", "last_modified": "", "description": "Test config"},
        "values": {
            "conversation_length": 50,  # Only specify some values
            "hosting": "custom_host",
            # detail_length and model_name intentionally omitted
        },
    }

    config_file = temp_config_dir / "config.yml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(test_config, f)

    config_manager = ConfigManager(temp_config_dir)

    # Specified values should match test config
    assert config_manager.get_config_value("conversation_length") == 50
    assert config_manager.get_config_value("hosting") == "custom_host"

    # Missing values should use defaults
    assert config_manager.get_config_value("detail_length") == DEFAULT_CONFIG.get_value(
        "detail_length"
    )
    assert config_manager.get_config_value("model_name") == DEFAULT_CONFIG.get_value("model_name")
    assert config_manager.get_config_value("providers") == DEFAULT_CONFIG.get_value("providers")


def test_config_manager_update_config(temp_config_dir):
    """Test updating configuration values."""
    config_manager = ConfigManager(temp_config_dir)

    updates = {"conversation_length": 25, "hosting": "new_host"}
    config_manager.update_config(updates)

    # Verify updates in memory
    assert config_manager.get_config_value("conversation_length") == 25
    assert config_manager.get_config_value("hosting") == "new_host"

    # Verify updates persisted to file
    with open(config_manager.config_file, "r", encoding="utf-8") as f:
        saved_config = yaml.safe_load(f)
    assert saved_config["values"]["conversation_length"] == 25
    assert saved_config["values"]["hosting"] == "new_host"


def test_config_manager_reset_defaults(temp_config_dir):
    """Test resetting configuration to defaults."""
    config_manager = ConfigManager(temp_config_dir)

    # First modify some values
    config_manager.update_config({"conversation_length": 30})

    # Then reset to defaults
    config_manager.reset_to_defaults()

    assert config_manager.config.version == DEFAULT_CONFIG.version
    assert config_manager.get_config_value("conversation_length") == DEFAULT_CONFIG.get_value(
        "conversation_length"
    )
    assert config_manager.get_config_value("hosting") == DEFAULT_CONFIG.get_value("hosting")


def test_config_manager_get_set(temp_config_dir):
    """Test getting and setting individual config values."""
    config_manager = ConfigManager(temp_config_dir)

    # Test get with default
    assert config_manager.get_config_value("nonexistent", "default") == "default"

    # Test set and get
    config_manager.set_config_value("hosting", "test_host")
    assert config_manager.get_config_value("hosting") == "test_host"

    # Verify persistence
    with open(config_manager.config_file, "r", encoding="utf-8") as f:
        saved_config = yaml.safe_load(f)
    assert saved_config["values"]["hosting"] == "test_host"


def test_config_manager_update_from_args(temp_config_dir):
    """Test updating config from command line arguments."""
    config_manager = ConfigManager(temp_config_dir)

    args = Namespace(hosting="cli_host", model="cli_model")
    config_manager.update_config_from_args(args)

    assert config_manager.get_config_value("hosting") == "cli_host"
    assert config_manager.get_config_value("model_name") == "cli_model"


# --- version ordering -------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1.2.3", (1, 2, 3)),
        ("0.15.10", (0, 15, 10)),
        # Only LEADING digits per segment: collecting every digit made
        # "1.2.3rc1" parse as (1, 2, 31), i.e. newer than its own release.
        ("1.2.3rc1", (1, 2, 3)),
        ("1.2.3.dev4", (1, 2, 3)),
        ("1.2.3-beta.1", (1, 2, 3)),
        # Malformed input compares as zero instead of raising — an advisory
        # warning must never stop the CLI from starting.
        ("", (0,)),
        ("   ", (0,)),
        ("x.y.z", (0,)),
    ],
)
def test_version_tuple_parsing(raw, expected) -> None:
    from local_operator.config import _version_tuple

    assert _version_tuple(raw) == expected


@pytest.mark.parametrize(
    "left,right,newer",
    [
        # The bug the tuple parse exists to fix: string compare said
        # "1.10.0" > "1.9.0" was False, so the warning fired on the wrong set.
        ("1.10.0", "1.9.0", True),
        ("1.9.0", "1.10.0", False),
        ("2.0.0", "1.0.0", True),
        ("1.2.3", "1.2.3", False),
        # A pre-release must NOT read as newer than its release.
        ("1.2.3rc1", "1.2.3", False),
        ("", "1.0.0", False),
    ],
)
def test_version_ordering(left, right, newer) -> None:
    from local_operator.config import _version_tuple

    assert (_version_tuple(left) > _version_tuple(right)) is newer


# --- Malformed config handling (item 6) -------------------------------------


def test_config_manager_malformed_yaml_backs_up_and_defaults(temp_config_dir, capsys):
    """A YAML syntax error backs the file up to config.yml.bad and starts with
    defaults instead of a raw traceback (item 6)."""
    config_file = temp_config_dir / "config.yml"
    config_file.write_text(": : not valid yaml [")
    manager = ConfigManager(temp_config_dir)
    # Degraded to defaults rather than crashing.
    assert manager.get_config_value("hosting") == ""
    # Backup name is `config.yml.bad.<timestamp>` so a second bad edit cannot
    # clobber the first (round-1 CR-MINOR-3), hence the glob rather than an
    # exact-name check.
    assert list(temp_config_dir.glob("config.yml.bad.*"))
    err = capsys.readouterr().err
    assert "could not parse" in err


def test_config_manager_non_mapping_top_level_backs_up(temp_config_dir, capsys):
    """A config whose top level is a list/scalar is rejected the same way as a
    parse error (item 6)."""
    config_file = temp_config_dir / "config.yml"
    config_file.write_text("- just\n- a\n- list\n")
    manager = ConfigManager(temp_config_dir)
    assert manager.get_config_value("hosting") == ""
    assert list(temp_config_dir.glob("config.yml.bad.*"))
    err = capsys.readouterr().err
    assert "not a valid configuration mapping" in err


def test_config_dir_created_0700(tmp_path):
    """A config dir the manager CREATES is 0700 (item 17) — never chmod an
    existing one."""
    import os

    if os.name != "posix":
        pytest.skip("permission test is Unix-only")
    fresh = tmp_path / "made"
    manager = ConfigManager(fresh)
    manager._write_config(vars(manager.config))
    assert fresh.stat().st_mode & 0o077 == 0


# --- the one-time session-cleanup migration ----------------------------------


def _write_config(config_dir: Path, values: dict[str, object]) -> Path:
    path = config_dir / "config.yml"
    metadata = {"created_at": "x", "last_modified": "x", "description": "d"}
    path.write_text(yaml.safe_dump({"version": "0.1.0", "metadata": metadata, "values": values}))
    return path


_RETIRED = {
    "session_retention_max_sessions": 200,
    "session_retention_max_bytes": 0,
    "session_retention_max_age_days": 0,
    "session.reap_unused": True,
    "session": {"reap_unused": True},
}


def test_loading_a_config_is_read_only(tmp_path):
    """PR #645 round 5: the migration used to run from ``_load_config``, so
    constructing a ConfigManager on the operator's real dir REWROTE his
    config and dropped the store marker into his real store — from an
    un-isolated probe script, while the change was under review. A load is
    a read. Every retired key present, nothing may change: not the file, not
    the store, no backup, no stamp."""
    from local_operator.config_migrations import LEGACY_STAMP_NAME
    from local_operator.session.cleanup import STORE_MARKER_NAME

    (tmp_path / "sessions" / "abc").mkdir(parents=True)
    path = _write_config(tmp_path, {"hosting": "anthropic", **_RETIRED})
    before = path.read_bytes()
    manager = ConfigManager(tmp_path)
    manager.get_config()
    manager.get_config_value("session.reap_unused")
    manager.get_nested_value(("session", "cleanup", "enabled"))
    assert path.read_bytes() == before, "loading rewrote config.yml"
    assert not (tmp_path / "sessions" / STORE_MARKER_NAME).exists(), "loading marked the store"
    assert not list(tmp_path.glob("config.yml.pre-cleanup-migration.*"))
    assert not (tmp_path / LEGACY_STAMP_NAME).exists()


def test_migration_pins_the_old_reapers_off_in_both_spellings(tmp_path, caplog, monkeypatch):
    """The explicit migration WRITES ``session.reap_unused: false`` in the
    flat spelling the #576 reaper read AND the nested one ``/settings``
    wrote — it never removes them. An older runtime that can still start on
    this machine (the window between migrating and every process being on
    the new version) must read its opt-out as False; removing the key is
    what let the installed reaper fire during this PR's review."""
    from local_operator.config_migrations import migrate_session_cleanup

    # The WARNING level belongs to the REAL-home shape (a human's install);
    # under pytest HOME is redirected, which the migration deliberately
    # DEMOTES to DEBUG (see the redirected-arm test below). Pin the input the
    # same way every home-sensitive test does, rather than depending on the
    # runner's HOME.
    monkeypatch.setattr("local_operator.supervisors.real_home", lambda: Path.home().resolve())
    _write_config(tmp_path, {"hosting": "anthropic", **_RETIRED})
    with caplog.at_level(logging.WARNING, logger="local_operator.config_migrations"):
        changes = migrate_session_cleanup(tmp_path)
    assert changes, "nothing migrated"

    stored = yaml.safe_load((tmp_path / "config.yml").read_text())["values"]
    assert stored["session.reap_unused"] is False, "the flat key the old reaper reads"
    assert stored["session"]["reap_unused"] is False, "the nested key /settings wrote"
    for gone in (
        "session_retention_max_sessions",
        "session_retention_max_bytes",
        "session_retention_max_age_days",
    ):
        assert gone not in stored, gone
    assert stored["session"]["cleanup"]["enabled"] is False
    assert stored["hosting"] == "anthropic", "unrelated keys survive"

    # THE OLD ACCESSOR. This is exactly what ``retention.sweep_from_config``
    # on 0.45–0.47 evaluates before reaping; it must say False.
    manager = ConfigManager(tmp_path)
    assert manager.get_config_value("session.reap_unused", True) is False
    assert manager.get_nested_value(("session", "cleanup", "enabled")) is False

    backups = sorted(tmp_path.glob("config.yml.pre-cleanup-migration.*"))
    assert len(backups) == 1
    original = yaml.safe_load(backups[0].read_text())["values"]
    assert original["session.reap_unused"] is True
    assert original["session_retention_max_sessions"] == 200

    messages = [r.message for r in caplog.records]
    assert any("config migration" in m and "reap_unused" in m for m in messages), messages


def test_migration_is_idempotent_and_a_no_op_once_migrated(tmp_path):
    from local_operator.config_migrations import migrate_session_cleanup

    _write_config(tmp_path, {"session": {"reap_unused": True}})
    assert migrate_session_cleanup(tmp_path)
    after = (tmp_path / "config.yml").read_bytes()
    assert migrate_session_cleanup(tmp_path) == []
    assert (tmp_path / "config.yml").read_bytes() == after
    assert len(list(tmp_path.glob("config.yml.pre-cleanup-migration.*"))) == 1


def test_migration_is_a_no_op_on_a_final_shape_config(tmp_path):
    from local_operator.config_migrations import migrate_session_cleanup

    path = _write_config(
        tmp_path,
        {
            "hosting": "anthropic",
            "session.reap_unused": False,
            "session": {"reap_unused": False, "cleanup": {"enabled": False}},
        },
    )
    before = path.read_bytes()
    assert migrate_session_cleanup(tmp_path) == []
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("config.yml.pre-cleanup-migration.*"))
    assert {entry.name for entry in tmp_path.iterdir()} == {"config.yml"}


def test_absent_startup_migration_creates_no_directory_or_mutex(tmp_path: Path) -> None:
    from local_operator.config_migrations import migrate_session_cleanup

    root = tmp_path / "absent-profile"
    assert migrate_session_cleanup(root) == []
    assert not root.exists()


def test_migration_locked_recheck_preserves_intervening_unrelated_setting(
    tmp_path: Path, monkeypatch
) -> None:
    from local_operator import config_migrations

    _write_config(tmp_path, {"session": {"reap_unused": True}})
    original = config_migrations._migrate_session_cleanup_locked

    def intervene(root: Path):
        # The eligibility read is not the mutation base: a sibling setting
        # changed after it must survive the actual locked fresh-read/write.
        ConfigManager(root).set_config_value("language", "fr")
        return original(root)

    monkeypatch.setattr(config_migrations, "_migrate_session_cleanup_locked", intervene)
    assert config_migrations.migrate_session_cleanup(tmp_path)
    assert ConfigManager(tmp_path).get_config_value("language") == "fr"


def test_migration_is_debug_only_under_a_redirected_home(tmp_path, caplog, monkeypatch):
    """T7: the per-run home keeps the WORK and drops only the noise.

    ``lop exec`` and agent-runtime-svc rewrite a run config per Execute, so
    this migration fires on EVERY run; the WARNING was one stderr line per run
    that the adapter persists as an audit event. The safety properties are
    untouched — keys WRITTEN in both spellings, backup taken, unrelated keys
    (the ``providers`` ZDR pin) byte-identical — and only the level moves.
    Mutation: demote unconditionally (the real-home arm above goes red);
    stop writing a key or skip the backup (this cell goes red).
    """
    from local_operator.config_migrations import migrate_session_cleanup

    monkeypatch.setattr(
        "local_operator.supervisors.real_home", lambda: Path("/nonexistent-foreign-home")
    )
    _write_config(
        tmp_path,
        {"hosting": "anthropic", "providers": {"radient": {"zdr": True}}, **_RETIRED},
    )
    with caplog.at_level(logging.DEBUG, logger="local_operator.config_migrations"):
        changes = migrate_session_cleanup(tmp_path)
    assert changes, "the migration still runs under a redirected HOME"

    stored = yaml.safe_load((tmp_path / "config.yml").read_text())["values"]
    assert stored["session.reap_unused"] is False
    assert stored["session"]["reap_unused"] is False
    assert stored["providers"] == {"radient": {"zdr": True}}, "the ZDR pin moved"
    assert (
        len(list(tmp_path.glob("config.yml.pre-cleanup-migration.*"))) == 1
    ), "the backup is a safety property, not a nicety"

    levels = {
        record.levelno
        for record in caplog.records
        if "config migration" in record.message and "reap_unused" in record.message
    }
    assert levels == {logging.DEBUG}, "an automation run must not warn per run"


def test_startup_seam_is_gated_by_the_config_not_a_stamp(tmp_path, monkeypatch):
    """Round 5 R5-4: a config restored from its backup (retired keys back,
    opt-out gone) must be migrated AGAIN. There is no stamp to say "done";
    the migration's own no-op path is the gate."""
    from local_operator import config_migrations

    _write_config(tmp_path, {"session": {"reap_unused": True}})
    config_migrations.run_startup_migrations(tmp_path)
    assert not (tmp_path / config_migrations.LEGACY_STAMP_NAME).exists()
    after = (tmp_path / "config.yml").read_bytes()
    config_migrations.run_startup_migrations(tmp_path)
    assert (tmp_path / "config.yml").read_bytes() == after, "second launch rewrote"
    # Restore the backup by hand: the belt must be fastened again next launch.
    backup = next(tmp_path.glob("config.yml.pre-cleanup-migration.*"))
    (tmp_path / "config.yml").write_bytes(backup.read_bytes())
    config_migrations.run_startup_migrations(tmp_path)
    stored = yaml.safe_load((tmp_path / "config.yml").read_text())["values"]
    assert stored["session.reap_unused"] is False and stored["session"]["reap_unused"] is False


def test_a_corrupt_legacy_stamp_cannot_stop_lop(tmp_path):
    """Round 5 R5-2: the round-5 candidate read a ``.migrations`` stamp and a
    non-UTF-8 one raised on the start path. Nothing reads it now; it must be
    inert whatever its bytes, and the migration must still run."""
    from local_operator import config_migrations

    _write_config(tmp_path, {"session": {"reap_unused": True}})
    (tmp_path / config_migrations.LEGACY_STAMP_NAME).write_bytes(b"\xff\xfe\x00garbage")
    config_migrations.run_startup_migrations(tmp_path)  # must not raise
    stored = yaml.safe_load((tmp_path / "config.yml").read_text())["values"]
    assert stored["session.reap_unused"] is False


def test_a_failed_backup_leaves_the_config_alone_and_retries(tmp_path, monkeypatch, caplog):
    """Round 5 R5-3: a backup that cannot be written must not rewrite the
    file AND must not be recorded as done — the next launch retries and,
    once the backup succeeds, fastens the belt."""
    from local_operator import config_migrations

    path = _write_config(tmp_path, {"session": {"reap_unused": True}})
    before = path.read_bytes()
    real_write_bytes = Path.write_bytes

    def refuse(self, data):  # noqa: ANN001, ANN202
        if ".pre-cleanup-migration." in self.name:
            raise PermissionError("read-only")
        return real_write_bytes(self, data)

    monkeypatch.setattr(Path, "write_bytes", refuse)
    with caplog.at_level(logging.WARNING, logger="local_operator.config_migrations"):
        config_migrations.run_startup_migrations(tmp_path)
    assert path.read_bytes() == before, "rewrote without a backup"
    assert any("could not back up" in r.message for r in caplog.records)
    monkeypatch.undo()
    config_migrations.run_startup_migrations(tmp_path)
    stored = yaml.safe_load(path.read_text())["values"]
    assert stored["session.reap_unused"] is False, "the retry did not fasten the belt"
    assert len(list(tmp_path.glob("config.yml.pre-cleanup-migration.*"))) == 1


def test_startup_seam_never_stops_lop_from_starting(tmp_path, monkeypatch):
    from local_operator import config_migrations

    _write_config(tmp_path, {"session": {"reap_unused": True}})

    def boom(_config_dir):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(config_migrations, "migrate_session_cleanup", boom)
    config_migrations.run_startup_migrations(tmp_path)  # must not raise


def test_a_corrupt_config_cannot_stop_the_seam(tmp_path):
    """Whatever ``ConfigManager`` does with an unreadable config, the seam
    must return: the start path handles that file the same way moments
    later, and a traceback here would be a second, earlier failure."""
    from local_operator import config_migrations

    (tmp_path / "config.yml").write_bytes(b"\xff\xfe: [not yaml")
    config_migrations.run_startup_migrations(tmp_path)  # must not raise


def test_migration_merges_into_an_existing_cleanup_block(tmp_path):
    """A user who already set cleanup limits keeps them; ``enabled`` is
    pinned to false only if it was absent."""
    from local_operator.config_migrations import migrate_session_cleanup

    _write_config(
        tmp_path,
        {"session": {"reap_unused": False, "cleanup": {"enabled": True, "max_sessions": 50}}},
    )
    migrate_session_cleanup(tmp_path)
    stored = yaml.safe_load((tmp_path / "config.yml").read_text())["values"]
    assert stored["session"]["cleanup"] == {"enabled": True, "max_sessions": 50}
    assert stored["session"]["reap_unused"] is False and stored["session.reap_unused"] is False


def test_migration_marks_an_existing_store_and_only_the_migration_does(tmp_path):
    from local_operator.config_migrations import migrate_session_cleanup
    from local_operator.session.cleanup import STORE_MARKER_NAME

    (tmp_path / "sessions" / "abc").mkdir(parents=True)
    _write_config(tmp_path, {"session.reap_unused": True})
    ConfigManager(tmp_path)
    assert not (tmp_path / "sessions" / STORE_MARKER_NAME).exists()
    migrate_session_cleanup(tmp_path)
    assert (tmp_path / "sessions" / STORE_MARKER_NAME).is_file()
    assert (tmp_path / "sessions" / "abc").is_dir()


def test_default_config_has_cleanup_disabled():
    cleanup = DEFAULT_CONFIG.values["session"]["cleanup"]
    assert cleanup == {
        "enabled": False,
        "max_sessions": 0,
        "max_inactive_days": 0,
        "max_total_bytes": 0,
        "remove_empty": False,
    }
    assert "session_retention_max_sessions" not in DEFAULT_CONFIG.values


def test_get_nested_value_walks_and_falls_back(tmp_path):
    manager = ConfigManager(tmp_path)
    manager.update_config({"session": {"cleanup": {"max_sessions": 3}}, "flat": "yes"})
    assert manager.get_nested_value(("session", "cleanup", "max_sessions")) == 3
    assert manager.get_nested_value(("session", "cleanup", "nope"), "d") == "d"
    assert manager.get_nested_value(("flat", "deeper"), "d") == "d"
    assert manager.get_nested_value(("flat",)) == "yes"


def test_the_migration_has_exactly_one_caller_and_marking_has_two():
    """Round 5: the migration ran from ``_load_config`` and a mere
    ``ConfigManager()`` rewrote the operator's live config. The seam is
    ``cli.main`` and NOTHING else may call the migration; the store marker
    is written only by session construction (its own store), by the
    migration (after it succeeds), and by mesh ADOPTION — never by
    ConfigManager, never by cleanup's read path.

    THE THIRD MARKER SITE, and why it is enumerated here rather than spelled
    elsewhere: a session moved onto this device arrives as a directory in
    ``sessions/``, and ``remove_session_dir`` refuses an unmarked store — so a
    destination that had never created a session locally (a freshly paired
    second machine, which is the case the mesh exists for) could not delete
    anything it adopted. Measured on the two-device rig 2026-09-26: the recall
    committed, wrote its tombstone, called ``remove_session_dir``, logged the
    refusal and LEFT THE COPY — one id on two devices, which is INV-1. This list
    is the inventory of every place that may write the marker, so a new one is a
    reviewed act; ``local_operator/network/mobility.py`` is that act.

    THE FOURTH SITE is ``session/cleanup.hand_off_session_dir``: the mesh move's
    commit point renames a session directory OUT of ``sessions/``, and it marks
    the store first for the same authority ``_remove_handed_away`` already used —
    a device that only ever ADOPTED a session over the mesh may never have created
    one, so its store carries no marker and the handoff's own guards would refuse
    the very directory the move just decided to retire. It is the product path
    that decided where the conversation lives, not a reader guessing.
    """
    import ast

    package = Path(__file__).resolve().parents[2] / "local_operator"
    migration_callers: set[str] = set()
    mark_callers: set[str] = set()
    for path in sorted(package.rglob("*.py")):
        rel = path.relative_to(package.parent).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name in ("run_startup_migrations", "migrate_session_cleanup"):
                migration_callers.add(rel)
            if name == "mark_store":
                mark_callers.add(rel)
    assert migration_callers == {
        "local_operator/cli.py",
        "local_operator/config_migrations.py",
    }, migration_callers
    assert mark_callers == {
        "local_operator/session_factory.py",
        "local_operator/config_migrations.py",
        "local_operator/network/mobility.py",
        "local_operator/session/cleanup.py",
    }, mark_callers


# ---------------------------------------------------------------------------
# The key that was dropped on load and erased on save
# ---------------------------------------------------------------------------


def _write_config_with_extra(config_dir: Path, extra: Mapping[str, Any]) -> Path:
    """``config.yml`` with ``extra`` at the TOP LEVEL, beside ``values``.

    This is the spelling a person writes for a setting the docs name in dots
    (``network.advertise_hosts``), and it is the exact file the mesh bug was
    reported with: an operator declaring their public address so a peer could dial
    it.
    """
    path = config_dir / "config.yml"
    document = {
        "version": "0.1.0",
        "metadata": {"created_at": "x", "last_modified": "x", "description": "d"},
        "values": {"conversation_length": 100},
        **extra,
    }
    path.write_text(yaml.safe_dump(document))
    return path


@pytest.fixture
def warned_fresh(monkeypatch: pytest.MonkeyPatch) -> None:
    """The once-per-process warning memo, emptied so each test sees its own first load.

    ``config._UNMODELLED_WARNED`` is process-global on purpose (see there), which
    makes a test that asserts on the warning order-dependent unless it starts from
    an empty memo.
    """
    from local_operator import config as config_mod

    monkeypatch.setattr(config_mod, "_UNMODELLED_WARNED", set())


def test_a_top_level_key_the_store_does_not_model_is_never_erased(tmp_path: Path) -> None:
    """THE ERASE half. A write used to delete every top-level key but the three it models.

    ``_write_config`` serialises ``vars(self.config)``, which holds only
    ``version``/``metadata``/``values``, so a hand-written ``network:`` block was gone
    after the next write — and writes happen on ordinary paths: this is what the
    startup cleanup migration does (it leaves a ``.pre-cleanup-migration`` backup
    beside the file for that reason), and ``/settings`` writes on every Enter. An
    operator's declared address therefore vanished with no error, the failure the
    mesh lane reported as "any ``lop`` run rewrites the file without it".
    """
    declared = {"network": {"advertise_hosts": ["203.0.113.7:4097"]}}
    path = _write_config_with_extra(tmp_path, declared)
    manager = ConfigManager(tmp_path)

    # The write the migration and the settings page both perform.
    manager._write_config(vars(manager.config))

    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert document["network"] == declared["network"], document.keys()
    # ...and the write still did its OWN job: the modelled keys are all there.
    assert document["values"]["conversation_length"] == 100
    assert document["metadata"]["last_modified"]


def test_the_startup_migration_keeps_a_key_it_does_not_model(tmp_path: Path) -> None:
    """The same erase, through the REAL writer: the migration that rewrites the file.

    Run end to end rather than by calling ``_write_config`` directly, because the
    migration is the path the operator actually hits — one ``lop`` verb on a fresh
    install is enough to trigger it and it is what left the backup behind.
    """
    from local_operator.config_migrations import migrate_session_cleanup

    declared = ["203.0.113.7:4097"]
    path = _write_config_with_extra(tmp_path, {"network": {"advertise_hosts": declared}})

    changes = migrate_session_cleanup(tmp_path)

    assert changes, "the migration must have done its own work for this to prove anything"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert document["network"]["advertise_hosts"] == declared
    # Its own job, on the same file: the retired reapers' opt-out is pinned.
    assert document["values"]["session"]["reap_unused"] is False


def test_a_top_level_key_is_reported_rather_than_silently_ignored(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, warned_fresh: None
) -> None:
    """THE DROP half. It is still not READ — and now it is not quiet either.

    Every setting is read from ``values``, so this key does nothing for as long as
    it sits there. That is a defensible schema; being silent about it is not, because
    the file gives the operator no sign and the docs spell the path in dots. The
    warning has to name BOTH the key and where a setting lives, or it is just noise.
    """
    _write_config_with_extra(tmp_path, {"network": {"advertise_hosts": ["203.0.113.7:4097"]}})

    with caplog.at_level(logging.WARNING):
        manager = ConfigManager(tmp_path)

    assert manager.get_nested_value(("network", "advertise_hosts"), "DEFAULT") == "DEFAULT"
    messages = [record.getMessage() for record in caplog.records]
    reported = [message for message in messages if "top-level" in message]
    assert len(reported) == 1, messages
    assert "network" in reported[0]
    assert "values.network" in reported[0]


def test_a_modelled_config_reports_nothing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, warned_fresh: None
) -> None:
    """The mirror, so the warning is not simply always on: a normal file is silent."""
    _write_config(tmp_path, {"conversation_length": 7})

    with caplog.at_level(logging.WARNING):
        ConfigManager(tmp_path)

    assert not [r for r in caplog.records if "top-level" in r.getMessage()]


def test_the_registry_writes_the_key_where_the_mesh_reads_it(tmp_path: Path) -> None:
    """THE SANCTIONED ROUTE, end to end: /settings → the file → ``advertise_endpoints``.

    The bug was not only that the natural spelling did nothing; there was no working
    route at all for the three keys ``NetworkSettings.from_config`` reads, so no
    surface could show or write them and the mesh docs' own remediation ("put that
    hostname in `network.advertise_hosts`") pointed at a key with no writer. This
    pins the path the fix adds: the registry's tuple is the reader's tuple, the
    value lands where ``get_nested_value`` walks, and the relay publishes it.
    """
    from local_operator import settings_io
    from local_operator.network import relay

    setting = settings_io.BY_KEY["network.advertise_hosts"]
    assert setting.path == ("network", "advertise_hosts")

    manager = ConfigManager(tmp_path)
    settings_io.write_setting(manager, setting, ["198.51.100.9:4100"])

    reread = ConfigManager(tmp_path)
    assert reread.get_nested_value(("network", "advertise_hosts")) == ["198.51.100.9:4100"]
    settings = relay.NetworkSettings.from_config(tmp_path)
    assert settings.advertise_hosts == ("198.51.100.9:4100",)
    # And it is the FIRST candidate a peer is told to dial, before the detected
    # addresses: only the operator knows about a tunnel or a public address.
    assert relay.advertise_endpoints(settings)[0] == "198.51.100.9:4100"


def test_the_port_and_listen_address_have_rows_too(tmp_path: Path) -> None:
    """The other two keys ``from_config`` reads, which had no writer for the same reason.

    An advertised ``host:port`` needs the port, and whether the relay accepts at all
    is ``listen_address``: a page that offered the endpoints and hid these two would
    be offering half of one setting's contract.
    """
    from local_operator import settings_io
    from local_operator.network import relay

    manager = ConfigManager(tmp_path)
    for key, value in (("network.port", 4123), ("network.listen_address", "127.0.0.1")):
        settings_io.write_setting(manager, settings_io.BY_KEY[key], value)

    settings = relay.NetworkSettings.from_config(tmp_path)
    assert settings.port == 4123
    assert settings.listen_address == "127.0.0.1"
    # The dial-only answer, which is what this pair of values means together.
    assert relay.advertise_endpoints(settings) == ["127.0.0.1:4123"]


def test_a_non_string_top_level_key_cannot_take_the_store_down(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, warned_fresh: None
) -> None:
    """The report must SURVIVE the file it reports on, whatever YAML made of the keys.

    YAML 1.1 resolves bare scalars by its own rules, so a hand-written ``config.yml`` can
    carry top-level keys that are not strings: ``2024:`` is an int, ``on:``/``yes:`` a
    bool, and dates, floats and null exist besides. Sorting a mix of those against a
    string raised ``TypeError: '<' not supported between instances of 'int' and 'str'``
    inside ``ConfigManager.__init__`` — so every ``lop`` verb died with a stack-trace
    panel on exactly the file class this mechanism exists to make visible, and
    ``lop config edit network.port 4098`` wrote nothing at all. The base commit loads that
    file (it has no report to run), which is what made this a regression rather than a
    pre-existing limit (review round 1, B1).

    The decision, asserted here: a key this store cannot SPELL — it has no ``values.``
    home to be moved to — is passed over in silence, and a string key beside it is still
    reported.
    """
    (tmp_path / "config.yml").write_text(
        "version: 0.1.0\n"
        "metadata:\n  created_at: x\n  last_modified: x\n  description: d\n"
        "values:\n  conversation_length: 100\n"
        "network:\n  advertise_hosts:\n    - 203.0.113.7:4097\n"
        "2024:\n  archived: true\n",
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING):
        manager = ConfigManager(tmp_path)  # TypeError here before the fix

    reported = [r.getMessage() for r in caplog.records if "top-level" in r.getMessage()]
    assert len(reported) == 1, reported
    assert "network" in reported[0]
    assert "values.network" in reported[0]
    # The unnameable key is REPORTED but never given a `values.` home: naming one would
    # send the operator to a path that cannot exist (review round 1, optional nit).
    assert "2024" in reported[0]
    assert "values.2024" not in reported[0], reported[0]
    assert "cannot be a settings path at all" in reported[0]

    # AND THE VERB THAT DIED NOW ROUND-TRIPS, with both keys still in the file.
    manager._write_config(vars(manager.config))

    document = yaml.safe_load((tmp_path / "config.yml").read_text(encoding="utf-8"))
    assert document["network"]["advertise_hosts"] == ["203.0.113.7:4097"]
    assert 2024 in document, sorted(map(repr, document))


# --- #1920: which writes `set_config_value` accepts, and where they land -----


def test_a_dotted_nested_key_is_refused_before_anything_is_mutated(tmp_path: Path) -> None:
    """The reported defect: success reported, nothing a reader looks at changed.

    ``Config.set_value`` is a plain ``dict.__setitem__``, so
    ``set_config_value("subagents.models.hi", …)`` stored the whole dotted name as a
    top-level key and returned normally — while ``read_effort_tier_selectors`` and
    ``get_nested_value`` walk ``values.subagents.models`` and saw nothing. Both halves
    are asserted here because only both together are the bug: not merely that the call
    fails, but that it fails BEFORE the file or the manager is touched, so a caller
    that catches the error cannot be left holding a half-applied write.
    """
    manager = ConfigManager(tmp_path)
    manager.set_config_value("hosting", "seed")

    config_file = manager.config_file
    before_bytes = config_file.read_bytes()
    before_values = dict(manager.get_config().values)

    with pytest.raises(ValueError) as raised:
        manager.set_config_value("subagents.models.hi", "openai/gpt-5-mini")

    message = str(raised.value)
    assert "subagents.models.hi" in message
    # Actionable: it names the route that does work, not just the refusal.
    assert "lop config edit subagents.models.hi" in message
    assert "settings_io.write_setting" in message

    assert config_file.read_bytes() == before_bytes
    assert dict(manager.get_config().values) == before_values
    assert "subagents.models.hi" not in manager.get_config().values
    # And nothing appeared at the NESTED path either — the write never happened at
    # all, rather than having been redirected to its correct home.
    assert manager.get_nested_value(("subagents", "models", "hi")) is None


def test_a_dotted_key_naming_nothing_is_refused_too(tmp_path: Path) -> None:
    """Undeclared dotted keys are the same trap with no registry entry to name.

    A near-miss spelling (``subagent.models.hi``) is exactly how this arises in
    practice, and it is the case the registry cannot describe: there is no ``path``
    to point at, so the refusal can only send the caller to ``lop config list``.
    """
    manager = ConfigManager(tmp_path)

    with pytest.raises(ValueError) as raised:
        manager.set_config_value("subagent.models.hi", "openai/gpt-5-mini")

    message = str(raised.value)
    assert "subagent.models.hi" in message
    assert "lop config" in message
    assert "subagent.models.hi" not in manager.get_config().values


def test_every_flat_dotted_key_still_round_trips_as_a_literal_top_level_key(
    tmp_path: Path,
) -> None:
    """The half that must NOT change: ``display.shimmer``'s dot is literal.

    ``tui/settings.py`` reads ``values["display.shimmer"]`` verbatim, so the repair
    for the defect above must not be "split every dotted key": that would write a
    ``display:`` mapping nothing reads — one silent failure traded for another
    (``settings_io``'s "THE ``display.*`` FLAT-KEY TRAP"). Driven off the REGISTRY
    rather than a hard-coded list, so the sixth display flag is covered on the day it
    is declared instead of the day someone remembers this test exists.
    """
    from local_operator import settings_io

    keys = settings_io.flat_dotted_keys()
    assert "display.shimmer" in keys and "keymap.new_session" in keys, keys

    manager = ConfigManager(tmp_path)
    for key in keys:
        default = settings_io.BY_KEY[key].default
        probe = (not default) if isinstance(default, bool) else "probe-value"
        manager.set_config_value(key, probe)

        # Through the VERBATIM top-level reader the consumers use — not the call's
        # own return, and not a nested walk, which is the whole distinction.
        got = ConfigManager(tmp_path).get_config_value(key, "<absent>")
        assert got == probe, f"{key}: wrote {probe!r}, read back {got!r}"

    document = yaml.safe_load((tmp_path / "config.yml").read_text(encoding="utf-8"))
    stored = document["values"]
    for key in keys:
        assert key in stored, f"{key} is not a literal top-level key"
        # And it was NOT split into a nesting level on the way out.
        assert not isinstance(stored.get(key.split(".")[0]), dict), key


def test_the_sanctioned_route_reaches_the_map_the_tier_reader_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``settings_io.write_setting`` → the file → ``read_effort_tier_selectors``.

    The refusal above is only half a fix unless the route it names actually works, and
    the reader is the harness's own (``harness/subagent.py``), which resolves its root
    from ``config_dir()`` rather than taking a directory — so the env var is what makes
    this read the file the write went to. Asserted against the READER's output rather
    than the file's bytes: that is the difference between "the value is somewhere" and
    "the runtime sees it".
    """
    from local_operator import settings_io
    from local_operator.harness.subagent import read_effort_tier_selectors

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))

    setting = settings_io.BY_KEY["subagents.models.hi"]
    assert setting.path == ("subagents", "models", "hi")
    assert not setting.is_flat_dotted

    # Resolved by NAME, spelled exactly as the refusal message tells the caller to
    # spell it (`settings_io.write_setting(manager, settings_io.resolve_key(key),
    # value)`). Design round 1, D2: the message used to name a form that raises
    # AttributeError when taken literally, so the named route is pinned here
    # rather than assumed callable.
    #
    # `resolve_key` is typed `Setting | None` legitimately — it answers "not a
    # declared key" with None, which is what the undeclared branch of the refusal
    # relies on — so the narrow is an assert rather than a cast: it states the
    # invariant this test depends on and fails loudly if the registry loses the key.
    resolved = settings_io.resolve_key("subagents.models.hi")
    assert resolved is not None
    assert resolved == setting
    settings_io.write_setting(ConfigManager(tmp_path), resolved, "openai/gpt-5-mini")

    assert read_effort_tier_selectors() == {"hi": "openai/gpt-5-mini"}
    assert ConfigManager(tmp_path).get_nested_value(("subagents", "models", "hi")) == (
        "openai/gpt-5-mini"
    )


def test_a_stale_manager_cannot_revert_a_sibling_write(tmp_path: Path) -> None:
    """The vanished-sibling half: the whole-snapshot re-dump.

    ``set_config_value`` writes ``vars(self.config)`` — the WHOLE in-memory mapping,
    not the one key. A manager constructed before another writer's change therefore
    reverted it, which is how a field silently disappears between consecutive writes
    (#1920's second half; ``settings_io._reload_before_write`` records the same
    mechanism one layer up). Three consecutive writes through separate managers, with
    the third holding the oldest snapshot, is the shape that reproduced it.
    """
    ConfigManager(tmp_path).set_config_value("hosting", "prime")

    stale = ConfigManager(tmp_path)  # snapshot taken BEFORE the two writes below

    ConfigManager(tmp_path).set_config_value("hosting", "write-1")
    ConfigManager(tmp_path).set_config_value("web_search", {"enabled": False})

    stale.set_config_value("model_name", "write-3")

    after = ConfigManager(tmp_path).get_config().values
    assert after["model_name"] == "write-3"
    assert after["hosting"] == "write-1", "write 1 was reverted by the stale snapshot"
    assert after["web_search"]["enabled"] is False, "write 2 was reverted by the stale snapshot"


def test_update_config_refuses_a_dotted_key_before_anything_is_mutated(tmp_path: Path) -> None:
    """The second whole-snapshot writer carried the same two halves (#1920).

    ``update_config`` had ``set_config_value``'s old body: a plain ``set_value``
    then a whole-file dump, so a dotted key was stored inertly and reported as a
    success. Refused here for the same reason, and refused BEFORE the loop that
    applies the updates — the mixed dict below is what makes that observable: a
    guard placed after the first ``set_value`` would leave ``hosting`` mutated in
    memory while the write never happened.
    """
    manager = ConfigManager(tmp_path)
    manager.set_config_value("hosting", "seed")
    before_bytes = manager.config_file.read_bytes()

    with pytest.raises(ValueError) as raised:
        manager.update_config({"session.cleanup.enabled": True, "hosting": "other"})

    assert "session.cleanup.enabled" in str(raised.value)
    assert manager.config_file.read_bytes() == before_bytes
    assert manager.get_config_value("hosting") == "seed"
    # The nested home is the shipped default, not the True that was asked for.
    assert manager.get_nested_value(("session", "cleanup", "enabled")) is False


def test_update_config_does_not_revert_a_sibling_write(tmp_path: Path) -> None:
    """The same stale-snapshot revert as ``set_config_value``, on the writer the
    server actually reaches: ``app.state.config_manager`` is built once at
    startup and reused for every request, so one ``PATCH /v1/config`` used to
    re-dump the whole startup snapshot over anything written since.
    """
    ConfigManager(tmp_path).set_config_value("hosting", "prime")

    stale = ConfigManager(tmp_path)  # snapshot taken BEFORE the other write
    ConfigManager(tmp_path).set_config_value("hosting", "write-1")

    stale.update_config({"model_name": "write-2"})

    after = ConfigManager(tmp_path).get_config().values
    assert after["model_name"] == "write-2"
    assert after["hosting"] == "write-1", "the concurrent write was reverted"


def test_update_config_without_updates_is_a_flush_not_a_merge(tmp_path: Path) -> None:
    """The exception that keeps ``reset_setting`` working, pinned rather than
    left implicit in ``update_config``'s body.

    ``settings_io._delete``'s top-level branch removes the key from the LIVE
    mapping and then persists it with ``update_config({}, write=True)``. That call
    must write the in-memory state as it stands; had the reload been applied to it
    too, the delete would be read back off disk and written again, silently
    undoing every ``reset_setting`` on a flat-dotted key.

    Since deep preservation, the flush also has to DECLARE the key it removed:
    an undeclared gap in the payload is a setting the write did not model, and
    ``_write_config`` keeps those on purpose. Both halves are asserted here — the
    declaration lands the delete, the omission alone does not.
    """
    manager = ConfigManager(tmp_path)
    manager.set_config_value("display.shimmer", False)

    del manager.get_config().values["display.shimmer"]  # the shape `_delete` uses
    manager.update_config({}, write=True)  # undeclared: preservation keeps it

    assert ConfigManager(tmp_path).get_config().values["display.shimmer"] is False

    del manager.get_config().values["display.shimmer"]
    manager.update_config({}, write=True, drop={("display.shimmer",): "reset by /settings"})

    assert "display.shimmer" not in ConfigManager(tmp_path).get_config().values


def test_a_non_string_key_is_not_a_dotted_key(tmp_path: Path) -> None:
    """A non-``str`` key must not turn the guard into a ``TypeError``.

    A top-level key the store does not model is a shape this repo meets on
    purpose — an int key survives a load and is reported to the user by
    ``_report_unmodelled_top_level``, which
    ``test_a_non_string_top_level_key_cannot_take_the_store_down`` pins. A bare
    ``"." not in key`` would raise ``TypeError: argument of type 'int' is not
    iterable`` from a guard whose whole job is to explain a refusal, naming
    neither the key nor the refusal.
    """
    manager = ConfigManager(tmp_path)
    # Two DELIBERATE `arg-type` violations, and they are the point of the test:
    # `key` is annotated `str` while the guard under test exists precisely because
    # a Python caller can hand it something else. The repo's shape for a
    # deliberate mismatch is the scoped ignore with the reason on it, as in
    # `tests/unit/classification/support.py:63`.
    manager.set_config_value(2024, "x")  # type: ignore[arg-type]  # the shape under test

    assert ConfigManager(tmp_path).get_config().values[2024] == "x"  # type: ignore[arg-type]


def test_a_config_that_goes_bad_under_a_live_manager_aborts_the_write(tmp_path: Path) -> None:
    """The rule that makes the reload safe instead of destructive.

    ``_load_config`` does not raise on a malformed file: it prints, renames the file
    to ``.bad.<stamp>`` and returns fresh DEFAULTS. So reloading as the base of a write
    would dump those defaults over the user's config, leaving only the broken two-line
    edit recoverable from the backup — the last good config gone. ``settings_io``
    refuses instead, and this asserts ``set_config_value`` inherits that rather than
    re-deriving its own weaker version.

    Constructed over a VALID file on purpose: constructing over a broken one already
    renames it at load time, so the case worth pinning is a file that goes bad while
    the manager is alive — a hand-edit in another window, a truncated write.
    """
    from local_operator.settings_io import ConfigUnreadableError

    manager = ConfigManager(tmp_path)
    manager.set_config_value("hosting", "openai")
    config_file = manager.config_file

    broken = "values:\n\thosting: anthropic\n"  # a tab, which YAML rejects
    config_file.write_text(broken, encoding="utf-8")

    with pytest.raises(ConfigUnreadableError):
        manager.set_config_value("model_name", "gpt-5")

    assert config_file.read_text(encoding="utf-8") == broken
    # The write did not rename it either: the `.bad` backup is `_load_config`'s move,
    # and taking it would already have destroyed the file the user is mid-edit on.
    assert not list(tmp_path.glob("*.bad.*"))
    assert manager.get_config_value("hosting", "<absent>") == "openai"


def test_a_file_whose_only_stray_keys_are_unnameable_is_still_reported(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, warned_fresh: None
) -> None:
    """Every unmodelled key non-string is not the same as nothing to say.

    The silence decision covers keys this store cannot SPELL — but a file whose ONLY stray
    keys are those still has keys nobody reads, and reporting nothing there is the
    original bug in miniature: the operator sees no sign at all (review round 1, optional
    nit). They are named by ``repr`` with the advice that applies (delete, or re-spell as
    a string), because there is no ``values.`` path to offer.
    """
    (tmp_path / "config.yml").write_text(
        "version: 0.1.0\n"
        "metadata:\n  created_at: x\n  last_modified: x\n  description: d\n"
        "values:\n  conversation_length: 100\n"
        "2024:\n  archived: true\n",
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING):
        ConfigManager(tmp_path)

    reported = [r.getMessage() for r in caplog.records if "top-level" in r.getMessage()]
    assert len(reported) == 1, reported
    assert "2024" in reported[0]
    assert "cannot be a settings path at all" in reported[0]


def test_a_tier_mapping_round_trips_through_the_settings_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#2428: ``{model, effort}`` is REGISTERED on the existing tier rows, so
    ``lop config edit`` / the page can write it, the watcher's per-key diff sees
    it, and the reader returns the selector the launch path resolves."""
    from local_operator import settings_io
    from local_operator.harness.subagent import (
        read_effort_tier_selectors,
        read_effort_tier_targets,
    )

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    setting = settings_io.resolve_key("subagents.models.hi")
    assert setting is not None

    typed = '{"model": "anthropic/claude-opus-5-5", "effort": "xhigh"}'
    value = settings_io.coerce(setting, typed)
    assert value == {"model": "anthropic/claude-opus-5-5", "effort": "xhigh"}
    settings_io.write_setting(ConfigManager(tmp_path), setting, value)

    assert read_effort_tier_targets() == {"hi": ("anthropic/claude-opus-5-5", "xhigh")}
    assert read_effort_tier_selectors() == {"hi": "anthropic/claude-opus-5-5"}
    # Shown back as JSON the TEXT editor can round-trip.
    shown = settings_io.read_setting(ConfigManager(tmp_path), setting)
    assert json.loads(shown) == value
    assert settings_io.coerce(setting, shown) == value
    # A bare selector is still plain text, and a bad mapping is refused at write.
    assert settings_io.coerce(setting, "openai/gpt-5-mini") == "openai/gpt-5-mini"
    with pytest.raises(ValueError):
        settings_io.write_setting(
            ConfigManager(tmp_path), setting, {"model": "a/b", "effort": "extreme"}
        )
    with pytest.raises(ValueError, match="expected JSON"):
        settings_io.write_setting(ConfigManager(tmp_path), setting, '{"model": "a/b",')


def test_a_tier_mapping_survives_writers_that_skip_coerce(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#2428 R1-1: ``PATCH /v1/settings`` and the agent ``setting`` tool call
    ``write_setting`` WITHOUT ``coerce``. ``read_setting`` serves a mapping as
    JSON text, so that text must be stored as the mapping again (the desktop
    round trip), stored normalised, and refused when it is not JSON."""
    from local_operator import settings_io

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    setting = settings_io.resolve_key("subagents.models.hi")
    assert setting is not None

    def stored() -> object:
        return ConfigManager(tmp_path).get_config_value("subagents", None)["models"]["hi"]

    settings_io.write_setting(
        ConfigManager(tmp_path), setting, '{"model": "anthropic/claude-opus-5-5", "effort": "high"}'
    )
    assert stored() == {"model": "anthropic/claude-opus-5-5", "effort": "high"}

    served = settings_io.read_setting(ConfigManager(tmp_path), setting)
    settings_io.write_setting(ConfigManager(tmp_path), setting, served)
    assert stored() == {"model": "anthropic/claude-opus-5-5", "effort": "high"}
    assert settings_io.read_setting(ConfigManager(tmp_path), setting) == served

    # QA N1: stored as the launch path will act on it, not as typed.
    settings_io.write_setting(
        ConfigManager(tmp_path), setting, {"model": " a/b ", "effort": " MAX "}
    )
    assert stored() == {"model": "a/b", "effort": "max"}
    # R1-3: ``auto`` is "no override", stored as the omission it means.
    settings_io.write_setting(
        ConfigManager(tmp_path), setting, '{"model": "a/b", "effort": "auto"}'
    )
    assert stored() == {"model": "a/b"}

    config_file = ConfigManager(tmp_path).config_file
    before = config_file.read_bytes()
    with pytest.raises(ValueError, match="expected JSON"):
        settings_io.write_setting(ConfigManager(tmp_path), setting, '{"model": "a/b",')
    assert config_file.read_bytes() == before
    # A bare selector is unchanged by the normalisation.
    settings_io.write_setting(ConfigManager(tmp_path), setting, "openai/gpt-5-mini")
    assert stored() == "openai/gpt-5-mini"


# ---------------------------------------------------------------------------
# A write must not model LESS than the file holds
# ---------------------------------------------------------------------------

#: The operator's live config at the 2026-10-23 incident, cut to the keys the loss
#: was measured on: the top-level ``values`` keys no schema models (``language``,
#: ``keymap.quick_send``, ``team_bridge``, ``tui``, ``compaction``, ``fork``,
#: ``hub``, ``network``, ``supplements``), the nested user values a defaults-shaped
#: write erased (``retry.fallbackChains``/``pinnedFallback``, ``subagents.*``),
#: and two value changes (``model_effort``, ``web_search.providers``).
_INCIDENT_USER_VALUES: dict[str, Any] = {
    "language": "de",
    "keymap.quick_send": "ctrl+j",
    "team_bridge": {"session": "f007224f570c"},
    "tui": {"theme": "gruvbox"},
    "compaction": {"threshold": 0.7},
    "fork": {"enabled": True},
    "hub": {"url": "https://hub.example"},
    "network": {"advertise_hosts": ["203.0.113.7:4097"]},
    "supplements": {"enabled": False},
    "retry": {
        "maxRetries": 4,
        "usageAwareFallback": True,
        "pinnedFallback": "openai/gpt-5",
        "fallbackChains": {"default": ["openai/gpt-5", "openrouter/deepseek/deepseek-chat"]},
    },
    "subagents": {
        "model_choice": "med",
        "models": {"hi": "openai/gpt-5", "med": "openrouter/x", "lo": "local/y"},
    },
    "web_search": {"providers": ["exa"], "enabled": True},
    "model_effort": "high",
}

#: The exact 34 bytes a QA rig wrote over the live config (2026-10-23 04:21:42).
#: Byte-identical to ``config.yml.pre-cleanup-migration.20261023-043540``.
_INCIDENT_EXTERNAL_WRITE = b"values:\n  tool_approval_mode: ask\n"


def _write_incident_config(config_dir: Path) -> Path:
    """The 26-key-shaped config the external write replaced, on disk."""
    path = config_dir / "config.yml"
    values = {**deepcopy(DEFAULT_CONFIG.values), **_INCIDENT_USER_VALUES}
    values["session"] = {**values["session"], "reap_unused": False}
    document = {
        "version": "0.1.0",
        "metadata": {"created_at": "x", "last_modified": "x", "description": "d"},
        "values": values,
    }
    path.write_text(yaml.safe_dump(document))
    return path


def _disk_values(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["values"]


@pytest.fixture
def preserved_fresh(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty the once-per-process preserved-path memo (see ``config._PRESERVED_WARNED``)."""
    from local_operator import config as config_mod

    monkeypatch.setattr(config_mod, "_PRESERVED_WARNED", set())


def test_unknown_values_keys_survive_every_writer_in_this_build(tmp_path: Path) -> None:
    """The keys a schema does not model are carried through, not erased.

    The incident's nine top-level keys live INSIDE ``values`` (``language``,
    ``keymap.quick_send``, ``team_bridge``, …), which is a different shape from
    the unmodelled TOP-LEVEL carry-through pinned above. The external QA write is
    what removed them on 2026-10-23; this pins the writers in this build, so a
    later change to any of them cannot quietly introduce the same loss.
    """
    from local_operator import settings_io
    from local_operator.cli import config_create_command
    from local_operator.config_migrations import migrate_session_cleanup

    path = _write_incident_config(tmp_path)
    expected = {
        "language": "de",
        "keymap.quick_send": "ctrl+j",
        "team_bridge": {"session": "f007224f570c"},
    }

    manager = ConfigManager(tmp_path)
    settings_io.write_setting(manager, settings_io.BY_KEY["model_name"], "m-1")
    assert {key: _disk_values(path)[key] for key in expected} == expected

    ConfigManager(tmp_path)._write_config(vars(ConfigManager(tmp_path).config))
    assert {key: _disk_values(path)[key] for key in expected} == expected

    assert migrate_session_cleanup(tmp_path), "the migration must do its own work here"
    assert {key: _disk_values(path)[key] for key in expected} == expected

    # `lop config create` on an existing readable file preserves it.
    monkeypatch_env = {"LOCAL_OPERATOR_CONFIG_DIR": str(tmp_path)}
    with patch.dict("os.environ", monkeypatch_env):
        with patch("pathlib.Path.home", lambda: tmp_path):
            assert config_create_command() == 0
    assert {key: _disk_values(path)[key] for key in expected} == expected


def test_a_default_shaped_subtree_write_keeps_nested_user_values(tmp_path: Path) -> None:
    """A writer whose dict is DEFAULT-SHAPED must not flatten the nested user keys.

    This is the September 2026 class (a run collapsed ``subagents.models`` to
    ``{}`` and killed the tier it fed) and it is what ``web_search/service.py``
    and ``web_fetch/service.py`` do on every save: they hand ``set_config_value``
    the pydantic model's own field set, not the user's mapping. The writer's own
    value still lands — preservation is for the keys it never mentioned.
    """
    path = _write_incident_config(tmp_path)
    manager = ConfigManager(tmp_path)

    retry = dict(DEFAULT_CONFIG.values["retry"])
    retry["maxRetries"] = 3  # the writer's own change, which must land
    manager.set_config_value("retry", retry)

    subagents = dict(DEFAULT_CONFIG.values["subagents"])  # {} — the collapsing shape
    manager.set_config_value("subagents", subagents)

    values = _disk_values(path)
    assert values["retry"]["maxRetries"] == 3
    assert values["retry"]["pinnedFallback"] == "openai/gpt-5"
    assert values["retry"]["fallbackChains"] == {
        "default": ["openai/gpt-5", "openrouter/deepseek/deepseek-chat"]
    }
    assert values["subagents"]["model_choice"] == "med"
    assert values["subagents"]["models"]["hi"] == "openai/gpt-5"

    # And the same through the migration, which rewrites the whole file.
    from local_operator.config_migrations import migrate_session_cleanup

    assert migrate_session_cleanup(tmp_path), "the migration must rewrite the file here"
    values = _disk_values(path)
    assert values["retry"]["fallbackChains"]["default"][0] == "openai/gpt-5"
    assert values["subagents"]["models"]["hi"] == "openai/gpt-5"
    assert values["web_search"]["providers"] == ["exa"]


def test_a_manager_that_never_read_the_file_cannot_write_defaults_over_it(
    tmp_path: Path,
) -> None:
    """``_fresh_default_config`` must never be the base of a write.

    A manager constructed while the config dir was empty holds defaults and knows
    it never read the file. If a populated ``config.yml`` appears before it
    writes (another process, a restore, an editor), dumping those defaults over
    it is the same 2026-10-23 loss with a different author. It refuses instead —
    naming the settings it would have removed — and leaves the bytes untouched.
    """
    manager = ConfigManager(tmp_path)
    path = _write_incident_config(tmp_path)
    before = path.read_bytes()

    with pytest.raises(ConfigWriteRefused) as refused:
        manager._write_config(vars(manager.config))

    assert refused.value.code == "config-write-refused"
    assert "language" in str(refused.value)
    assert "subagents.models.hi" in str(refused.value)
    assert path.read_bytes() == before

    # The flush path too: `update_config({})` writes the in-memory snapshot.
    with pytest.raises(ConfigWriteRefused):
        manager.update_config({}, write=True)
    assert path.read_bytes() == before


def test_a_document_with_only_metadata_counts_as_never_read(tmp_path: Path) -> None:
    """A truthy document is not a document that carried SETTINGS.

    ``version:``/``metadata:`` with no ``values:`` parses to a non-empty mapping, so
    a truthiness test would let this manager believe it read the operator's settings
    — while its snapshot is the shipped defaults. Populate the file afterwards and
    the write would restore every default over them with no refusal: the same class
    as the missing-file case, one shape over.
    """
    path = tmp_path / "config.yml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": "0.1.0",
                "metadata": {"created_at": "x", "last_modified": "x", "description": "d"},
            }
        )
    )
    manager = ConfigManager(tmp_path)  # no `values:` to read

    _write_incident_config(tmp_path)  # ...and something populates it afterwards
    before = path.read_bytes()

    with pytest.raises(ConfigWriteRefused) as refused:
        manager._write_config(vars(manager.config))

    assert "language" in str(refused.value)
    assert path.read_bytes() == before


def test_an_undeclared_missing_setting_is_preserved_and_reported(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    preserved_fresh: None,
) -> None:
    """Preserved keys are reported; a declared drop is the ONLY way to remove one.

    Preservation without accounting would make every removal look like a silent
    failure. Each write says which paths it kept because it did not model them,
    and a caller that means to remove one declares it — with a reason, which is
    what turns "the key is gone" into an audit trail.
    """
    path = _write_incident_config(tmp_path)
    manager = ConfigManager(tmp_path)
    retry = dict(manager.get_config_value("retry"))
    retry.pop("fallbackChains")
    retry.pop("pinnedFallback")

    with caplog.at_level(logging.WARNING):
        manager.set_config_value("retry", retry)

    after = _disk_values(path)["retry"]
    assert after["fallbackChains"]["default"][0] == "openai/gpt-5"
    assert after["pinnedFallback"] == "openai/gpt-5"
    preserved = [r.getMessage() for r in caplog.records if "did not model" in r.getMessage()]
    assert preserved, "the preserved keys must be named"
    assert "retry.fallbackChains" in preserved[0]
    assert "retry.pinnedFallback" in preserved[0]

    caplog.clear()
    # A fresh snapshot, because the first write committed the preserved keys into
    # the manager's live mapping (and into the dict it was handed — see
    # ``_write_config``: preservation is applied to the payload in place).
    remaining = dict(ConfigManager(tmp_path).get_config_value("retry"))
    remaining.pop("fallbackChains")
    remaining.pop("pinnedFallback")
    with caplog.at_level(logging.WARNING):
        manager.set_config_value(
            "retry",
            remaining,
            drop={("retry", "fallbackChains"): "chains are managed by the cascade page"},
        )
    assert "fallbackChains" not in _disk_values(path)["retry"]
    dropped = [
        r.getMessage()
        for r in caplog.records
        if "chains are managed by the cascade page" in r.getMessage()
    ]
    assert dropped and "retry.fallbackChains" in dropped[0]

    before = path.read_bytes()
    with pytest.raises(ValueError, match="reason"):
        manager.set_config_value("retry", remaining, drop={("retry", "pinnedFallback"): " "})
    # A drop path that is also IN the payload is a contradiction, not a removal.
    with pytest.raises(ValueError, match="still present"):
        manager.set_config_value(
            "retry",
            {**retry, "fallbackChains": {}},
            drop={("retry", "fallbackChains"): "x"},
        )
    assert path.read_bytes() == before


def test_the_startup_report_names_the_settings_an_external_write_removed(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The 2026-10-23 incident, end to end.

    A baseline inventory exists; an external writer drops the file to the exact
    34 bytes the QA rig wrote; the next launch names what went missing, on
    stderr, before the cleanup migration's back-filled defaults can mask it. A
    daemon launch must NOT consume the notice: it logs at DEBUG and leaves the
    inventory as it was, so the next human surface reports it.
    """
    from local_operator.config_migrations import run_startup_migrations

    path = _write_incident_config(tmp_path)
    inventory = tmp_path / ".config-keys.json"

    # First launch records the baseline (silently), the second settles it after
    # the cleanup migration's own additive keys.
    run_startup_migrations(tmp_path, surface="cli")
    run_startup_migrations(tmp_path, surface="cli")
    assert inventory.exists()
    recorded = json.loads(inventory.read_text(encoding="utf-8"))
    assert recorded["schema"] == 1
    assert "subagents.models.hi" in recorded["keys"]

    path.write_bytes(_INCIDENT_EXTERNAL_WRITE)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        run_startup_migrations(tmp_path, surface="cli")

    err = capsys.readouterr().err
    assert "lost" in err
    for named in (
        "language",
        "keymap.quick_send",
        "team_bridge.session",
        "subagents.models.hi",
        "retry.fallbackChains.default",
    ):
        assert named in err, named
    warned = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("lost" in message for message in warned)
    # It must NOT re-baseline over the loss, or the next launch would be silent.
    assert json.loads(inventory.read_text(encoding="utf-8")) == recorded

    # A daemon launch leaves the notice for the next human surface.
    path.write_bytes(_INCIDENT_EXTERNAL_WRITE)
    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        run_startup_migrations(tmp_path, surface="daemon")
    assert "lost" not in capsys.readouterr().err
    assert not [
        r for r in caplog.records if r.levelno >= logging.WARNING and "lost" in r.getMessage()
    ]
    assert [r for r in caplog.records if r.levelno == logging.DEBUG and "lost" in r.getMessage()]
    assert json.loads(inventory.read_text(encoding="utf-8")) == recorded
    assert recorded["keys"] and recorded["recorded_at"]

    # ...and the next CLI launch gets it.
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        run_startup_migrations(tmp_path, surface="cli")
    assert "language" in capsys.readouterr().err


def test_the_key_inventory_is_not_rewritten_when_nothing_changed(tmp_path: Path) -> None:
    """An unchanged launch stays WRITE-FREE once a baseline exists.

    ``lop config list``, ``--version`` and the TUI all run the startup seam, so an
    unconditional rewrite would put a file write (and a new ``recorded_at``) on
    every read-only command — the same "reading is writing" defect the migration
    seam was built to end.
    """
    from local_operator.config_migrations import run_startup_migrations

    _write_incident_config(tmp_path)
    # The cleanup migration ADDS keys after the first launch's inventory is
    # recorded (its own belt), so the first two launches legitimately re-baseline.
    # That is the "set differs" path this test is not about; run until it settles.
    for _ in range(3):
        run_startup_migrations(tmp_path, surface="cli")
    inventory = tmp_path / ".config-keys.json"
    before = inventory.read_bytes()
    before_mtime = inventory.stat().st_mtime_ns

    run_startup_migrations(tmp_path, surface="cli")  # a read-only launch

    assert inventory.read_bytes() == before
    assert inventory.stat().st_mtime_ns == before_mtime


def test_a_zero_byte_config_is_not_a_shrink_and_leaves_the_inventory_alone(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Truncation is unreadable-as-absent, not "every key was lost".

    An interrupted write (the pre-atomic-write window) or a ``> config.yml``
    leaves 0 bytes. Reporting 25 lost settings would be wrong, and re-baselining
    on it would erase the one record that says what the file used to hold.
    """
    from local_operator.config_migrations import run_startup_migrations

    path = _write_incident_config(tmp_path)
    run_startup_migrations(tmp_path, surface="cli")
    run_startup_migrations(tmp_path, surface="cli")
    inventory = tmp_path / ".config-keys.json"
    before = inventory.read_bytes()
    before_mtime = inventory.stat().st_mtime_ns

    path.write_bytes(b"")
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        run_startup_migrations(tmp_path, surface="cli")

    assert "lost" not in capsys.readouterr().err
    assert not [
        r for r in caplog.records if r.levelno >= logging.WARNING and "lost" in r.getMessage()
    ]
    assert inventory.read_bytes() == before
    assert inventory.stat().st_mtime_ns == before_mtime


def test_the_migration_retires_only_its_own_keys(tmp_path: Path) -> None:
    """The migration's own job still happens, and it declares what it removes.

    Its deletion of the retired ceilings is now an explicit ``drop``: without the
    declaration, deep preservation would copy the three keys straight back from
    disk and the retirement would become a no-op that looks like it worked.
    """
    from local_operator.config_migrations import (
        _RETIRED_CEILINGS,
        migrate_session_cleanup,
    )

    path = _write_incident_config(tmp_path)
    values = _disk_values(path)
    for key in _RETIRED_CEILINGS:
        values[key] = 7
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["values"] = values
    path.write_text(yaml.safe_dump(document))

    changes = migrate_session_cleanup(tmp_path)

    after = _disk_values(path)
    assert [f"removed {key}" for key in _RETIRED_CEILINGS if f"removed {key}" in changes]
    assert not set(_RETIRED_CEILINGS) & set(after)
    assert after["session"]["reap_unused"] is False
    assert after["session.reap_unused"] is False
    assert after["session"]["cleanup"]["enabled"] is False
    assert after["language"] == "de"
    assert after["retry"]["fallbackChains"]["default"][0] == "openai/gpt-5"
    assert after["subagents"]["models"]["hi"] == "openai/gpt-5"
