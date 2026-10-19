"""``asks.lapsed_visible_for``'s one reader (decision D8)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator.asks import store, visibility
from local_operator.config_watch import process_watcher


@pytest.mark.parametrize(
    ("value", "ms"),
    [
        ("at_expiry", 0),
        ("1h", 3_600_000),
        ("24h", 86_400_000),
        ("7d", store.LATE_WINDOW_S * 1000),
    ],
)
def test_every_choice_reads_as_its_window(value: str, ms: int) -> None:
    assert visibility.lapsed_visible_ms_from_values({"asks": {"lapsed_visible_for": value}}) == ms


@pytest.mark.parametrize(
    "values",
    [
        {},
        {"asks": None},
        {"asks": "7d"},
        {"asks": {}},
        {"asks": {"lapsed_visible_for": "2h"}},
        {"asks": {"lapsed_visible_for": 3600}},
        {"asks": {"lapsed_visible_for": None}},
        {"asks": {"lapsed_visible_for": ["1h"]}},
    ],
)
def test_garbage_reads_as_the_seven_day_default(values: dict[str, Any]) -> None:
    assert visibility.lapsed_visible_ms_from_values(values) == store.LATE_WINDOW_S * 1000


def test_the_reader_follows_a_config_file_written_after_start(tmp_path: Path) -> None:
    from local_operator import settings_io
    from local_operator.config import ConfigManager

    assert visibility.lapsed_visible_ms(tmp_path) == store.DEFAULT_LAPSED_VISIBLE_MS
    settings_io._store(ConfigManager(tmp_path), ("asks", "lapsed_visible_for"), "at_expiry")
    # ``poll_now`` is the desktop server's spelling (no started watcher there).
    assert visibility.lapsed_visible_ms(tmp_path, poll_now=True) == 0
    # And a process whose watcher already polled reads the snapshot as is.
    process_watcher(tmp_path).poll_now()
    assert visibility.lapsed_visible_ms(tmp_path) == 0
