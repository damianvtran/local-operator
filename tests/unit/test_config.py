import logging
import tempfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from local_operator.config import DEFAULT_CONFIG, Config, ConfigManager


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
    assert config_manager.get_config_value("providers") == {
        "openai": {"api": "responses", "use_max_context_window": True},
        "anthropic": {"cache_ttl_1h_min_context_tokens": 150_000},
    }


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
    assert config_manager.get_config_value("providers") == {
        "openai": {"api": "responses", "use_max_context_window": True},
        "anthropic": {"cache_ttl_1h_min_context_tokens": 150_000},
    }


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


def test_migration_pins_the_old_reapers_off_in_both_spellings(tmp_path, caplog):
    """The explicit migration WRITES ``session.reap_unused: false`` in the
    flat spelling the #576 reaper read AND the nested one ``/settings``
    wrote — it never removes them. An older runtime that can still start on
    this machine (the window between migrating and every process being on
    the new version) must read its opt-out as False; removing the key is
    what let the installed reaper fire during this PR's review."""
    from local_operator.config_migrations import migrate_session_cleanup

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
    is written only by session construction (its own store) and by the
    migration (after it succeeds) — never by ConfigManager, never by
    cleanup's read path."""
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
    }, mark_callers
