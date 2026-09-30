from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from local_operator.hub_sync.settings import DEFAULT_CHECK_INTERVAL_MIN, HubSyncSettings


def _cm(values: dict[str, Any]) -> Any:
    def nested(path: tuple[str, ...], default: Any = None) -> Any:
        cur: Any = values
        for part in path:
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    return SimpleNamespace(get_nested_value=nested)


def test_defaults_are_on_and_hourly() -> None:
    s = HubSyncSettings.from_config(_cm({}))
    assert (s.auto_agents, s.auto_teams, s.interval_min, s.merge_model) == (True, True, 60, "")


def test_it_reads_the_nested_path_the_registry_writes() -> None:
    s = HubSyncSettings.from_config(
        _cm({"hub": {"auto_update": {"agents": False, "teams": False}, "check_interval_min": 15}})
    )
    assert (s.auto_agents, s.auto_teams, s.interval_min) == (False, False, 15)


def test_a_dotted_literal_key_is_not_read() -> None:
    # The flat accessor bug: a literal "hub.auto_update.agents" key is NOT the setting.
    s = HubSyncSettings.from_config(_cm({"hub.auto_update.agents": False}))
    assert s.auto_agents is True


def test_bad_types_fall_back_to_the_default_and_the_interval_is_clamped() -> None:
    s = HubSyncSettings.from_config(
        _cm({"hub": {"auto_update": {"agents": "false"}, "check_interval_min": "soon"}})
    )
    assert s.auto_agents is True and s.interval_min == DEFAULT_CHECK_INTERVAL_MIN
    assert HubSyncSettings.from_config(_cm({"hub": {"check_interval_min": 1}})).interval_min == 5
    assert (
        HubSyncSettings.from_config(_cm({"hub": {"check_interval_min": 10**6}})).interval_min
        == 1440
    )
    assert (
        HubSyncSettings.from_config(_cm({"hub": {"check_interval_min": True}})).interval_min == 60
    )


def test_read_fresh_sees_a_write_from_another_manager(tmp_path) -> None:
    """A long-lived manager holds what it loaded at boot; the LIVE section must not."""

    from local_operator.config import ConfigManager

    held = ConfigManager(tmp_path)
    assert HubSyncSettings.from_config(held).auto_agents is True
    ConfigManager(tmp_path).set_config_value("hub", {"auto_update": {"agents": False}})
    assert HubSyncSettings.from_config(held).auto_agents is True  # stale by construction
    assert HubSyncSettings.read_fresh(held).auto_agents is False


def test_read_fresh_sees_merge_model_and_never_moves_a_torn_config(tmp_path, monkeypatch) -> None:
    """R2-1 / R2-2: merge_model is live too, and a periodic read never touches the file."""

    from local_operator.config import ConfigManager
    from local_operator.hub_sync.resolver import resolve_merge_model

    held = ConfigManager(tmp_path)
    ConfigManager(tmp_path).set_config_value("hub", {"merge_model": "openai/gpt-4o"})
    assert HubSyncSettings.read_fresh(held).merge_model == "openai/gpt-4o"
    assert HubSyncSettings.from_config(held).merge_model == ""  # the held manager IS stale
    # ...and the resolver, which the daemon builds from that held manager, follows the disk.
    monkeypatch.setattr(
        "local_operator.model.configure.build_model_spec",
        lambda hosting, model, info=None: SimpleNamespace(),
    )
    assert resolve_merge_model(held).label == "openai/gpt-4o"

    cfg = tmp_path / "config.yml"
    cfg.write_text("values: [unterminated\n  : :", encoding="utf-8")
    before = cfg.read_bytes()
    fallback = HubSyncSettings.read_fresh(held)  # torn read -> the held manager's view
    assert fallback == HubSyncSettings.from_config(held)
    assert cfg.read_bytes() == before and sorted(p.name for p in tmp_path.iterdir()) == [
        "config.yml"
    ]
