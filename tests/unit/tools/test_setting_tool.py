"""The ``setting`` tool (wave 4 slice A): the sanctioned agent write path.

The contract this file pins:
* READS go through the registry — `get` returns the row's current value and
  whether it is the default, for any key `/settings` lists.
* WRITES go through ``settings_io.write_setting`` — THE one funnel — so the
  registry's own validation answers (its message verbatim, never a tool-local
  paraphrase), the sub-mapping merge rule holds, and the config-watcher
  notification fires exactly as it does for the TUI page and the HTTP route.
* The tool never raises into the model: unknown keys, refused values and an
  unreadable config all come back as error RESULTS.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.tools import builtin


@pytest.fixture
def config_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An isolated config root the tool's resolver will actually read."""
    home = tmp_path / "config"
    home.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(home))
    monkeypatch.delenv("LOP_LANG", raising=False)
    return home


def _run(args: dict[str, Any]) -> Any:
    tool = builtin.build_setting_tool()
    return asyncio.run(tool.execute("t-set", args, None, None, None))  # type: ignore[operator]


async def _execute(args: dict[str, Any]) -> Any:
    tool = builtin.build_setting_tool()
    return await tool.execute("t-set", args, None, None, None)  # type: ignore[operator]


def _payload(result: Any) -> dict[str, Any]:
    return json.loads(result.content[0].text)


def test_get_reads_a_registry_row(config_home: Path) -> None:
    result = _run({"op": "get", "key": "language"})
    assert result.is_error is False
    assert _payload(result) == {"key": "language", "value": "auto", "is_default": True}


def test_set_writes_through_the_funnel_and_reads_back(config_home: Path) -> None:
    result = _run({"op": "set", "key": "language", "value": "en"})
    assert result.is_error is False
    assert _payload(result) == {"key": "language", "value": "en", "is_default": False}
    # The write is the SAME one every other surface makes: values.language in
    # config.yml, through ConfigManager.set_config_value.
    import yaml

    written = yaml.safe_load((config_home / "config.yml").read_text(encoding="utf-8"))
    assert written["values"]["language"] == "en"


def test_set_refuses_an_out_of_choices_value_with_the_registrys_message(
    config_home: Path,
) -> None:
    """The registry is the only validator: today's language row is shipped-gated
    (`auto`, `en` until a translation wave ships), and the tool surfaces exactly
    the message /settings shows for the same value — never a paraphrase."""
    from local_operator import settings_io

    setting = settings_io.resolve_key("language")
    assert setting is not None
    from local_operator.config import ConfigManager

    with pytest.raises(ValueError) as refusal:
        settings_io.write_setting(ConfigManager(config_home), setting, "xx")
    result = _run({"op": "set", "key": "language", "value": "xx"})
    assert result.is_error is True
    assert result.content[0].text == str(refusal.value)


def test_an_unknown_key_names_how_to_find_keys(config_home: Path) -> None:
    result = _run({"op": "get", "key": "does.not.exist"})
    assert result.is_error is True
    assert "does.not.exist" in result.content[0].text
    assert "lop config list" in result.content[0].text


def test_set_with_an_empty_value_unsets_a_row_that_supports_it(config_home: Path) -> None:
    """The route/CLI clear gesture, mirrored: an empty string DELETES the key
    (`empty_unsets` rows), so it reads as its default again — storing a literal
    "" would report a deliberate choice the user never made."""
    assert _run({"op": "set", "key": "hosting", "value": "example-host"}).is_error is False
    result = _run({"op": "set", "key": "hosting", "value": ""})
    assert result.is_error is False
    assert _payload(result)["is_default"] is True
    import yaml

    written = yaml.safe_load((config_home / "config.yml").read_text(encoding="utf-8"))
    assert "hosting" not in (written.get("values") or {})


def test_a_cascade_key_is_refused_with_the_snapshot_contract_named(config_home: Path) -> None:
    """Cascade keys carry a base-snapshot edit contract the tool has no field
    for: refused EXPLICITLY (with the alternative named) rather than written
    raw, which would accept shapes `_normalize_chains` drops on read — the
    silent divergence the shared funnel exists to prevent. The refusal must not
    touch the file."""
    _run({"op": "set", "key": "language", "value": "en"})
    before = (config_home / "config.yml").read_bytes()
    result = _run(
        {"op": "set", "key": "retry.fallbackChains", "value": {"default": ["zai/glm-5.3"]}}
    )
    assert result.is_error is True
    text = result.content[0].text
    assert "snapshot" in text and "base" in text
    assert (config_home / "config.yml").read_bytes() == before


def test_a_bad_op_is_an_invalid_arguments_error(config_home: Path) -> None:
    result = _run({"op": "reset", "key": "language"})
    assert result.is_error is True


@pytest.mark.asyncio
async def test_the_tool_write_is_unattributed_and_converges_on_disk(config_home: Path) -> None:
    """The approval boundary (CI round 3): the TOOL's write must not reach an
    in-process listener — ``source="local"`` means "the operator", and this
    tool can run in a process that holds a live approval gate — while the file
    still converges for every observer, arriving at the writer's own watcher as
    a DISK change (tighten-only for any live gate)."""
    from local_operator.config_watch import process_watcher

    watcher = process_watcher(config_home)
    watcher.start()
    seen: list[Any] = []
    watcher.subscribe(seen.append)
    try:
        result = await _execute({"op": "set", "key": "language", "value": "en"})
        assert result.is_error is False
        assert seen == []  # no local attribution
        change = watcher.poll_now()
        assert change is not None
        assert change.source == "disk"
        assert change.changed_keys == {"language"}
    finally:
        await watcher.stop()


def test_a_broken_config_is_backed_up_and_the_write_lands_on_defaults(
    config_home: Path,
) -> None:
    """Fresh-process reality, the same one ``lop config edit`` has: constructing
    the manager is what degrades a broken file — it is MOVED to a timestamped
    ``.bad.<ts>`` backup (never deleted) with the loud stderr banner — and the
    write then lands on the fresh defaults. The route's refusal path (409)
    applies to its long-lived manager; the tool mirrors the CLI, and the backup
    is what keeps the user's bytes recoverable."""
    broken = "values: [not, a, mapping\n"
    (config_home / "config.yml").write_text(broken, encoding="utf-8")
    result = _run({"op": "set", "key": "language", "value": "en"})
    assert result.is_error is False
    backups = list(config_home.glob("config.yml.bad.*"))
    assert len(backups) == 1, backups
    assert backups[0].read_text(encoding="utf-8") == broken
    import yaml

    written = yaml.safe_load((config_home / "config.yml").read_text(encoding="utf-8"))
    assert written["values"]["language"] == "en"
