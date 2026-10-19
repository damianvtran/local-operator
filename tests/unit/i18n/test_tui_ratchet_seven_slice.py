"""Ratchet slice 7: settings_view (tui.small) and usage_panel (tui.usage).

Two namespaces, because the two files do not share one: `settings_view` is a
widget and extends the series' widget namespace `tui.small`, while
`usage_panel` already owned `tui.usage` (five `paid.*` keys) and extends it —
forcing its copy into `tui.small` would have split one file's strings across
two catalogues.

Every assertion is DIFFERENTIAL: the left side is the generated key rendering
through the real resolver, the right side is the expression the site carried
before this change, written out literally. Equal means no bytes moved. Two
sites `press a key…` and the `fetching…`/`refreshing…` pair are written with
`\\u2026` escapes in the source; the catalogue carries the decoded character,
and these tests compare the DECODED strings, which is what a user sees.

One extraction is deliberately NOT one of the scanner's findings: the
`{n} window(s)` plural in `usage_panel` sits inside a list literal in an
assignment, and the scanner reads only call arguments — so its sibling
counters were listed and it was not. It is extracted anyway, because leaving
one count word untranslated beside four translated ones is exactly the
incoherence this programme exists to remove. That asymmetry (argument
position visible, assignment invisible) is the documented remaining blind
spot pinned in `test_check.py::test_a_name_reference_stays_invisible`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from local_operator.i18n import catalogues, resolve, text
from local_operator.i18n.keys import tui_small as _s
from local_operator.i18n.keys import tui_usage as _u
from local_operator.i18n.text import tr
from local_operator.tui.widgets import settings_view

REPO = Path(__file__).resolve().parents[3]
_small = tr("tui.small")
_usage = tr("tui.usage")

#: The two files this slice activated, exact paths (never a directory prefix:
#: sibling widgets still carry literals, so a prefix would red-CI the tree).
ACTIVATED = (
    "local_operator/tui/widgets/settings_view.py",
    "local_operator/tui/widgets/usage_panel.py",
)

#: Slice 6's activated files, pinned so this slice's baseline edit cannot walk
#: them back.
RETAINED = (
    "local_operator/tui/widgets/goal_panel.py",
    "local_operator/tui/widgets/network_panel.py",
)

#: The keys this slice minted, one per extracted literal (the windows plural
#: is the extra: see the module docstring).
NEW_SMALL = (
    "tui.small.settings_view.default_marker",
    "tui.small.settings_view.hop_count",
    "tui.small.settings_view.reorder_hint",
    "tui.small.settings_view.press_key",
    "tui.small.settings_view.hop_row_hint",
    "tui.small.settings_view.chain_row_hint",
    "tui.small.settings_view.chain_add_hint",
    "tui.small.settings_view.hop_add_hint",
    "tui.small.settings_view.read_only_pane",
    "tui.small.settings_view.overflow_more",
    "tui.small.settings_view.read_only_label",
    "tui.small.settings_view.title",
    "tui.small.settings_view.bool_on",
    "tui.small.settings_view.bool_off",
)
NEW_USAGE = (
    "tui.usage.windows_count",
    "tui.usage.exhausted",
    "tui.usage.near_limit",
    "tui.usage.reported_balance",
    "tui.usage.not_reported",
    "tui.usage.unavailable",
    "tui.usage.resets_in",
    "tui.usage.no_windows",
    "tui.usage.fetching",
    "tui.usage.title",
    "tui.usage.sign_in_expired",
    "tui.usage.stale",
    "tui.usage.refreshing",
    "tui.usage.refresh_failed",
    "tui.usage.showing_prefix",
    "tui.usage.of_separator",
)


def test_every_new_key_exists_in_its_namespace_with_a_full_sidecar() -> None:
    for namespace, keys in (("tui.small", NEW_SMALL), ("tui.usage", NEW_USAGE)):
        live = catalogues.load_catalogue("en", namespace)
        sidecar = catalogues.context("en", namespace)
        for key in keys:
            assert key in live, key
            entry = sidecar[key]
            assert all(entry.get(field) for field in ("role", "tone", "note")), key


def test_settings_view_rows_are_byte_identical_to_the_pre_slice_copy() -> None:
    # Was: "  (default)"
    assert _small(_s.settings_view_default_marker()) == "  (default)"
    assert _small(_s.settings_view_reorder_hint()) == "  shift+↑↓ reorder · d delete"
    # Was: "press a key\u2026" (escape in source, decoded here)
    assert _small(_s.settings_view_press_key()) == "press a key…"
    assert (
        _small(_s.settings_view_hop_row_hint())
        == "shift+↑↓ reorder · d delete · enter edits this hop"
    )
    assert _small(_s.settings_view_chain_row_hint()) == "enter opens the chain · d deletes it"
    assert (
        _small(_s.settings_view_chain_add_hint())
        == "enter, then type: <chain-key> <provider>/<model>"
    )
    assert _small(_s.settings_view_hop_add_hint()) == "enter, then type: <provider>/<model>"
    # Two read-only words: the pane's has NO indent, the label's has two spaces.
    assert _small(_s.settings_view_read_only_pane()) == "read-only"
    assert _small(_s.settings_view_read_only_label()) == "  read-only"
    assert _small(_s.settings_view_title()) == "settings"
    assert _small(_s.settings_view_bool_on()) == "on"
    assert _small(_s.settings_view_bool_off()) == "off"


@pytest.mark.parametrize(
    ("hops", "expected"), [(1, "  1 hop"), (2, "  2 hops"), (1200, "  1200 hops")]
)
def test_settings_view_hop_count_follows_the_number(hops: int, expected: str) -> None:
    # Was: f"  {len(hops)} hop{'' if len(hops) == 1 else 's'}" — indent, raw digits.
    rendered = _small(_s.settings_view_hop_count(count=hops, shown=str(hops)))
    assert rendered == expected
    assert "," not in rendered


def test_settings_view_overflow_keeps_its_indent_and_raw_digit() -> None:
    # Was: f"  … {spilled} more"
    rendered = _small(_s.settings_view_overflow_more(spilled="1200"))
    assert rendered == "  … 1200 more"


def test_usage_panel_counters_are_byte_identical_to_the_pre_slice_copy() -> None:
    # Was: f"{self.exhausted} exhausted" and its three siblings.
    assert _usage(_u.exhausted(value="3")) == "3 exhausted"
    assert _usage(_u.near_limit(value="2")) == "2 near limit"
    assert _usage(_u.not_reported(value="7")) == "7 not reported"
    assert _usage(_u.sign_in_expired(value="1")) == "1 sign-in expired"
    assert _usage(_u.stale(value="4")) == "4 stale"


@pytest.mark.parametrize(
    ("count", "expected"), [(1, "1 window"), (2, "2 windows"), (1200, "1200 windows")]
)
def test_usage_window_count_follows_the_number(count: int, expected: str) -> None:
    # Was: f"{self.windows} window{'s' if self.windows != 1 else ''}" — the
    # unscanned sibling; raw digits.
    rendered = _usage(_u.windows_count(count=count, shown=str(count)))
    assert rendered == expected
    assert "," not in rendered


@pytest.mark.parametrize(
    ("count", "expected"),
    [(1, "1 reported balance"), (2, "2 reported balances"), (10000, "10000 reported balances")],
)
def test_usage_reported_balance_follows_the_number(count: int, expected: str) -> None:
    rendered = _usage(_u.reported_balance(count=count, shown=str(count)))
    assert rendered == expected
    assert "," not in rendered


def test_usage_panel_chrome_is_byte_identical_to_the_pre_slice_copy() -> None:
    assert _usage(_u.unavailable()) == "  ·  usage unavailable"
    # Was: f", resets in {countdown}" — the value is an already-humanised string.
    assert _usage(_u.resets_in(value="5m")) == ", resets in 5m"
    assert _usage(_u.no_windows()) == "  no windows reported"
    assert _usage(_u.fetching()) == "fetching…"
    assert _usage(_u.refreshing()) == "refreshing…"
    assert _usage(_u.title()) == "Usage"
    assert _usage(_u.refresh_failed()) == "refresh failed — showing last known numbers"
    # The scroll counter's two pieces: the trailing/leading spaces are layout.
    assert _usage(_u.showing_prefix()) == "showing "
    assert _usage(_u.of_separator()) == " of "


def test_the_scroll_counter_reassembles_to_the_original_row() -> None:
    # Was: "showing " + str(end) + " of " + str(total)
    row = _usage(_u.showing_prefix()) + "12" + _usage(_u.of_separator()) + "40"
    assert row == "showing 12 of 40"


def test_the_live_catalogue_namespaces_still_contain_their_older_keys() -> None:
    # tui.usage's paid.* keys predate this slice, and the consolidation of the
    # function-local import into the module-level resolver rides on them being
    # untouched — so ALL FIVE are pinned, not a sample. Their source call sites
    # are asserted to resolve through the hoisted resolver in the same test.
    live = catalogues.load_catalogue("en", "tui.usage")
    assert live["tui.usage.paid.continue"] == "included quota used; credits will continue"
    assert live["tui.usage.paid.family_caps"] == "model-family caps still apply"
    assert live["tui.usage.paid.blocked"] == "Spend unavailable"
    assert live["tui.usage.paid.eligibility_unknown"] == "Spend eligibility unknown"
    assert live["tui.usage.paid.unlimited"] == "Unlimited paid allowance"
    # And the five source sites now read the hoisted `_x` alias rather than the
    # removed function-local module name.
    source = (REPO / "local_operator/tui/widgets/usage_panel.py").read_text(encoding="utf-8")
    assert "tui_usage." not in source
    assert source.count("_x.paid_") >= 5


@pytest.fixture
def fr_corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Only this fixture may make a synthetic locale eligible, never production."""
    vals = catalogues.load_catalogue("en", "tui.small")
    for locale, values in (("en", vals), ("fr", {k: "FR " + v for k, v in vals.items()})):
        folder = tmp_path / locale
        folder.mkdir()
        (folder / "tui.small.json").write_text(json.dumps(values), encoding="utf-8")
    monkeypatch.setattr(catalogues, "_CATALOGUES", tmp_path)
    monkeypatch.setattr(text, "_active_locale", "en")
    monkeypatch.setattr(text, "_refreshers", [settings_view._reload_copy])
    monkeypatch.setattr(resolve, "shipped_locales", lambda: ("en", "fr"))
    monkeypatch.delenv("LOP_I18N", raising=False)
    text._sources.cache_clear()
    yield
    # Restore materialised constants and the real catalogue root, so fixture
    # translations never escape this test.
    text.set_active_locale("en")
    settings_view._reload_copy()
    text._sources.cache_clear()


def test_the_bool_choices_follow_a_live_language_change(fr_corpus) -> None:
    """Q1: the import-time constant must not pin the language it was built in.

    Before this module registered a refresher, a locale move changed what the
    resolver returned per call while ``_BOOL_CHOICES`` kept the import-time
    language — a half-translated settings row that only shows up after a live
    switch, which is why the reproduction (not the reading) is the evidence.
    """
    assert [c.label for c in settings_view._BOOL_CHOICES] == ["on", "off"]
    assert _small(_s.settings_view_bool_on()) == "on"

    text.set_active_locale("fr")

    # Both halves must move together: the per-call resolution AND the constant.
    assert _small(_s.settings_view_bool_on()) == "FR on"
    assert [c.label for c in settings_view._BOOL_CHOICES] == ["FR on", "FR off"]


def test_every_activated_file_lost_its_ceiling() -> None:
    """The toothlessness guard: enforced WITHOUT a ceiling, exact path only."""
    baseline = json.loads((REPO / "i18n" / "baseline.json").read_text(encoding="utf-8"))
    for path in ACTIVATED:
        assert path in baseline["enforced"], f"{path} is not activated"
        assert path not in baseline["files"], f"{path} kept its ceiling; the gate is toothless"
    for path in RETAINED:
        assert path in baseline["enforced"], f"{path} lost its activation"
        assert path not in baseline["files"], f"{path} regained a ceiling"
    assert not [e for e in baseline["enforced"] if e.endswith("/") and "tui" in e]


@pytest.mark.parametrize("path", ACTIVATED)
def test_a_planted_literal_fails_the_activated_checker(
    path: str, tmp_path, monkeypatch, i18n_check
) -> None:
    """Prove the activation BITES: the real checker entrypoint, isolated copy."""
    source = (REPO / path).read_text(encoding="utf-8")
    target = tmp_path / path
    target.parent.mkdir(parents=True)
    target.write_text(source, encoding="utf-8")
    baseline_dir = tmp_path / "i18n"
    baseline_dir.mkdir()
    baseline = baseline_dir / "baseline.json"
    baseline.write_text(json.dumps({"note": "slice fixture", "enforced": [path], "files": {}}))
    allowlist = baseline_dir / "allowlist.toml"
    allowlist.write_text("entries = []\n")
    monkeypatch.setattr(i18n_check, "REPO", tmp_path)
    monkeypatch.setattr(i18n_check, "BASELINE", baseline)
    monkeypatch.setattr(i18n_check, "ALLOWLIST", allowlist)
    monkeypatch.setattr(i18n_check, "check_catalogues", lambda: [])
    assert i18n_check.main([]) == 0
    target.write_text(source + '\nprint("planted slice prose")\n', encoding="utf-8")
    assert i18n_check.main([]) == 1
    target.write_text(source, encoding="utf-8")
    assert i18n_check.main([]) == 0
