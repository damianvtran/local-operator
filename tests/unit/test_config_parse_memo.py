"""``config.yml`` is parsed once per version of the file, and never served stale.

WHY THESE TESTS EXIST. A runtime child builds five ``ConfigManager``s before it
can publish its record, and each one used to re-parse ``config.yml`` with the
pure-Python YAML scanner and re-read the package METADATA twice: 185 + ~80 ms of
the ~440 ms CPU a session construction costs, on a host where one CPU
millisecond is 10-17 ms of wall time (see ``config._parse_config_stream``).

The memo is only acceptable with its invalidation story, so most of this file
pins the story rather than the saving: every way the file can change on disk —
the product's own atomic write, an in-place rewrite of the same size, a
deletion — must be seen by the very next manager. The saving is pinned
structurally (a parse COUNT), never by a timer (AGENTS.md "Timing, flakes").
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

import local_operator.config as config_mod
from local_operator.config import ConfigManager


@pytest.fixture(autouse=True)
def _fresh_memo(monkeypatch: pytest.MonkeyPatch) -> None:
    # Per-test memo: a key from another test's temp file must never answer here.
    monkeypatch.setattr(config_mod, "_PARSED", {}, raising=False)


@pytest.fixture
def parses(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Count real YAML parses of any stream, through the module's own ``yaml``."""
    seen: list[str] = []
    real = yaml.load

    def counting(stream, Loader):  # type: ignore[no-untyped-def]
        seen.append(getattr(stream, "name", "?"))
        return real(stream, Loader=Loader)

    monkeypatch.setattr(config_mod.yaml, "load", counting)
    return seen


def _seed(directory: Path, **values: object) -> ConfigManager:
    manager = ConfigManager(directory)
    manager.update_config(dict(values))
    return manager


def test_repeat_managers_parse_the_file_once(tmp_path: Path, parses: list[str]) -> None:
    _seed(tmp_path, hosting="test", model_name="m1")
    parses.clear()
    for _ in range(5):
        assert ConfigManager(tmp_path).get_config_value("model_name") == "m1"
    # Five managers, one parse: the construction path's five managers now cost
    # one scan instead of five. On origin/main this is 5.
    assert len(parses) == 1


def test_an_atomic_write_by_another_manager_is_seen(tmp_path: Path, parses: list[str]) -> None:
    _seed(tmp_path, hosting="test", model_name="m1")
    assert ConfigManager(tmp_path).get_config_value("model_name") == "m1"
    # The product's own writer: temp file + os.replace -> a new inode.
    ConfigManager(tmp_path).set_config_value("model_name", "m2")
    assert ConfigManager(tmp_path).get_config_value("model_name") == "m2"


def test_an_in_place_same_size_rewrite_is_seen(tmp_path: Path) -> None:
    """The case an mtime-second or size-only key would miss: same inode, same
    length, rewritten in place. ``mtime_ns``/``ctime_ns`` must carry it."""
    _seed(tmp_path, hosting="test", model_name="aaaa")
    path = tmp_path / "config.yml"
    assert ConfigManager(tmp_path).get_config_value("model_name") == "aaaa"
    before = os.stat(path)
    text = path.read_text(encoding="utf-8")
    assert text.count("aaaa") == 1
    with open(path, "r+", encoding="utf-8") as handle:  # same inode, same size
        handle.write(text.replace("aaaa", "bbbb"))
    after = os.stat(path)
    assert (after.st_ino, after.st_size) == (before.st_ino, before.st_size)
    assert ConfigManager(tmp_path).get_config_value("model_name") == "bbbb"


def test_a_rewrite_with_mtime_forged_back_is_still_seen(tmp_path: Path) -> None:
    """``os.utime`` can put mtime back; nothing can put ctime back. The key holds
    both, so even a writer that restores the old mtime is not served stale."""
    _seed(tmp_path, hosting="test", model_name="aaaa")
    path = tmp_path / "config.yml"
    assert ConfigManager(tmp_path).get_config_value("model_name") == "aaaa"
    before = os.stat(path)
    text = path.read_text(encoding="utf-8")
    with open(path, "r+", encoding="utf-8") as handle:
        handle.write(text.replace("aaaa", "cccc"))
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert os.stat(path).st_mtime_ns == before.st_mtime_ns
    assert ConfigManager(tmp_path).get_config_value("model_name") == "cccc"


def test_managers_never_share_the_cached_dict(tmp_path: Path) -> None:
    _seed(tmp_path, hosting="test", model_name="m1")
    first = ConfigManager(tmp_path)
    first.config.set_value("model_name", "mutated-in-memory")
    first.config.values.setdefault("providers", {})["x"] = 1
    second = ConfigManager(tmp_path)
    assert second.get_config_value("model_name") == "m1"
    assert "x" not in (second.get_config_value("providers") or {})


def test_a_bad_file_is_not_cached_and_its_replacement_is_read(tmp_path: Path) -> None:
    path = tmp_path / "config.yml"
    path.write_text("values: [unclosed\n", encoding="utf-8")
    ConfigManager(tmp_path)  # moves it aside, falls back to defaults
    assert not path.exists()
    _seed(tmp_path, hosting="test", model_name="after-bad")
    assert ConfigManager(tmp_path).get_config_value("model_name") == "after-bad"


def test_a_deleted_file_reads_as_defaults(tmp_path: Path) -> None:
    _seed(tmp_path, hosting="test", model_name="m1")
    assert ConfigManager(tmp_path).get_config_value("model_name") == "m1"
    (tmp_path / "config.yml").unlink()
    assert ConfigManager(tmp_path).get_config_value("model_name") != "m1"


def test_the_package_version_is_resolved_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake(name: str) -> str:
        calls.append(name)
        return "9.9.9"

    monkeypatch.setattr(config_mod, "version", fake)
    monkeypatch.setattr(config_mod, "_PACKAGE_VERSION", None)
    assert [config_mod._package_version() for _ in range(4)] == ["9.9.9"] * 4
    assert calls == ["local-operator"]


def test_a_patched_resolver_is_honoured_over_an_earlier_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests patch ``local_operator.config.version``; the cache must not answer
    them with a value some earlier caller resolved through the real one."""
    config_mod._package_version()  # prime with the real resolver
    monkeypatch.setattr(config_mod, "version", lambda _name: "0.0.1-patched")
    assert config_mod._package_version() == "0.0.1-patched"
