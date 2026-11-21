"""The heartbeat age inside a live row label renders through catalogue keys.

WHY THIS FILE EXISTS. PR #4334 keyed the live rungs of ``CatalogEntry.status``
but the ``wedged_heartbeat`` sentence still received ``info.model``'s ASCII
``4m`` / ``3h 12m``, so a translated "Not answering" row carried English unit
letters. :func:`catalog.status_duration` renders the four
``wire.session_status.duration_*`` arms instead. Pinned here:

* English is BYTE-IDENTICAL to ``info.model.format_duration`` on every arm and
  boundary (clients, feed dedupe and other tests carry those spellings),
  including no grouping separator at 1000+ days;
* each arm DISPATCHES through its own key, in the row-label locale
  (``_label_locale``), not the TUI's active-locale seam;
* a locale whose catalogue lacks the keys falls back to English source.
"""

from __future__ import annotations

import json

import pytest

from local_operator.i18n import catalogues, resolve, text
from local_operator.info.model import format_duration as ascii_duration
from local_operator.resume import SessionRow
from local_operator.session import catalog

ROW = SessionRow("synthetic001", 1.0, "Synthetic session")

#: Every arm and both sides of every threshold, plus clamps and a fraction.
BOUNDARIES = [
    -5.0,  # clamps to 0
    0.0,
    0.9,  # truncates, never rounds up
    1.0,
    59.0,
    59.99,
    60.0,
    61.0,
    3599.0,
    3600.0,
    3660.0,
    86399.0,
    86400.0,
    86400.0 * 2 + 3600 * 4,
    86400.0 * 999 + 3600 * 23,
    86400.0 * 1000,  # first value a number argument would group
    86400.0 * 12345 + 3600 * 7,
]


@pytest.fixture
def english(monkeypatch):
    monkeypatch.setattr(catalog, "_label_locale", lambda: "en")


@pytest.mark.usefixtures("english")
@pytest.mark.parametrize("seconds", BOUNDARIES)
def test_english_matches_the_ascii_formatter_byte_for_byte(seconds):
    assert catalog.status_duration(seconds) == ascii_duration(seconds)


@pytest.mark.usefixtures("english")
@pytest.mark.parametrize(
    ("seconds", "label"),
    [
        (42.0, "42s"),
        (240.0, "4m"),
        (3600.0 * 3 + 60 * 12, "3h 12m"),
        (3600.0, "1h 0m"),
        (86400.0 * 2 + 3600 * 4, "2d 4h"),
        (86400.0 * 1000, "1000d 0h"),
    ],
)
def test_english_source_spellings(seconds, label):
    assert catalog.status_duration(seconds) == label


@pytest.mark.usefixtures("english")
def test_english_wedged_row_is_byte_identical():
    row = ROW._replace(live_state="wedged", heartbeat_age_s=3600.0 * 3 + 60 * 12)
    assert (
        catalog.CatalogEntry(row).status
        == "Not answering · process alive (last heartbeat 3h 12m ago)"
    )


FIXTURE = {
    "duration_seconds": "fx:s[{count, number}]",
    "duration_minutes": "fx:m[{count, number}]",
    "duration_hours_minutes": "fx:hm[{hours, number}|{minutes, number}]",
    "duration_days_hours": "fx:dh[{days}|{hours, number}]",
    "wedged_heartbeat": "fx:wedged_heartbeat[{duration}]",
}


def _install(tmp_path, monkeypatch, fr_overrides):
    """Isolated catalogue root: real English plus an opaque ``fr`` fixture."""
    source = catalogues._CATALOGUES / "en" / "wire.session_status.json"
    messages = json.loads(source.read_text())
    prefix = "wire.session_status."
    assert {prefix + name for name in FIXTURE} <= set(messages)
    localized = {k: v for k, v in messages.items() if not k.startswith(prefix + "duration_")}
    localized.update({prefix + name: value for name, value in fr_overrides.items()})
    for locale, payload in (("en", messages), ("fr", localized)):
        (tmp_path / locale).mkdir()
        (tmp_path / locale / "wire.session_status.json").write_text(json.dumps(payload))
    monkeypatch.setattr(catalogues, "_CATALOGUES", tmp_path)
    # Eligibility is the fixture's too: never alter the real shipped set.
    monkeypatch.setattr(resolve, "shipped_locales", lambda: ("en", "fr"))


@pytest.fixture
def fixture_fr(tmp_path, monkeypatch):
    _install(tmp_path, monkeypatch, FIXTURE)
    monkeypatch.setattr(catalog, "_label_locale", lambda: "fr")


@pytest.mark.usefixtures("fixture_fr")
@pytest.mark.parametrize(
    ("seconds", "label"),
    [
        (0.0, "fx:s[0]"),
        (59.0, "fx:s[59]"),
        (60.0, "fx:m[1]"),
        (3599.0, "fx:m[59]"),
        (3600.0, "fx:hm[1|0]"),
        (86399.0, "fx:hm[23|59]"),
        (86400.0, "fx:dh[1|0]"),
        # The string argument is why: a `{days, number}` would read `1 000` in fr.
        (86400.0 * 1000 + 3600 * 5, "fx:dh[1000|5]"),
    ],
)
def test_every_arm_dispatches_through_its_key(seconds, label):
    assert catalog.status_duration(seconds) == label


@pytest.mark.usefixtures("fixture_fr")
def test_wedged_row_embeds_the_localized_duration():
    row = ROW._replace(live_state="wedged", heartbeat_age_s=240.0)
    assert catalog.CatalogEntry(row).status == "fx:wedged_heartbeat[fx:m[4]]"


@pytest.mark.usefixtures("fixture_fr")
def test_label_locale_outranks_the_active_locale_seam():
    # The TUI's active locale must not leak into a server-built row label.
    previous = text.active_locale()
    text.set_active_locale("en")
    try:
        assert catalog.status_duration(240.0) == "fx:m[4]"
    finally:
        text.set_active_locale(previous)


def test_missing_duration_keys_fall_back_to_english_source(tmp_path, monkeypatch):
    # fr carries the sentence but none of the duration keys: the arm renders
    # the English source (and English formatting) inside the French sentence.
    _install(tmp_path, monkeypatch, {"wedged_heartbeat": "fx:wedged_heartbeat[{duration}]"})
    monkeypatch.setattr(catalog, "_label_locale", lambda: "fr")
    assert catalog.status_duration(86400.0 * 1000) == "1000d 0h"
    row = ROW._replace(live_state="wedged", heartbeat_age_s=3600.0 * 3 + 60 * 12)
    assert catalog.CatalogEntry(row).status == "fx:wedged_heartbeat[3h 12m]"


def test_unshipped_locale_renders_english(tmp_path, monkeypatch):
    _install(tmp_path, monkeypatch, FIXTURE)
    monkeypatch.setattr(resolve, "shipped_locales", lambda: ("en",))
    monkeypatch.setattr(catalog, "_label_locale", lambda: "fr")
    assert catalog.status_duration(3600.0 * 3 + 60 * 12) == "3h 12m"
