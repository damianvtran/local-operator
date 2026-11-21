"""Whole Info source contract; marker fixtures are NOT locale translations.

The English golden is derived from the manager-pinned base, not from the new
implementation. It protects full/short/terse semantics and clipboard bytes while
independent translators/auditors own the seven real target catalogues.
"""
from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from local_operator.i18n import catalogues
from local_operator.i18n.keys import tui_info
from local_operator.i18n.text import set_active_locale
from local_operator.info.copy import InfoRow, info_plural, info_text, paint_info_row
from local_operator.info.model import InfoSnapshot, SessionsInfo
from local_operator.info.render import build_export, plural
from local_operator.tui.widgets.info_panel import _Body, build_info_report
from tests.unit.tui.test_info_panel import _fleet_sessions, _live, _quiet_owner_sessions, _snapshot

FIXTURE = Path(__file__).with_name("fixtures") / "info_readout_en_golden.json"
ROOT = Path(__file__).resolve().parents[3]


def source_cases() -> dict[str, InfoSnapshot | None]:
    normal = replace(_snapshot(), captured_at=0)
    return {
        "loading": None,
        "populated": normal,
        "empty": InfoSnapshot(),
        "registry_unavailable": replace(normal, sessions=SessionsInfo(available=False)),
        "partial_fleet": replace(normal, sessions=_fleet_sessions(subagents_unreported=1)),
        "queued_only": replace(
                           normal,
                           sessions=_fleet_sessions(
                                        fleet_session_trajectories=0,
                                        fleet_subagents_running=0,
                                        fleet_subagents_queued=3,
                                        fleet_trajectories=0,
                                    ),
                       ),
        "not_answering": replace(normal, sessions=_quiet_owner_sessions()),
        "shadowed": replace(
                        normal,
                        install=replace(
                                    normal.install,
                                    import_path='/opt/checkout/local_operator',
                                    import_path_foreign=True,
                                ),
                    ),
        "latest_unknown": replace(normal, install=replace(normal.install, latest_known=None)),
        "installed_unreadable": replace(normal, install=replace(normal.install, version="")),
        "degraded": replace(normal, degraded=(("env.credentials", "probe unavailable"),)),
    }


@pytest.fixture(autouse=True)
def restore_locale():
    set_active_locale("en")
    yield
    set_active_locale("en")


def test_english_matches_pinned_source_bytes() -> None:
    golden = json.loads(FIXTURE.read_text())
    assert golden["base"] == "7e5da6775c41d867365347f380e2d01fbbeeab29"
    for name, snapshot in source_cases().items():
        for width in (38, 51, 65, 88, 126):
            text = build_info_report(snapshot, _live(), width).plain
            actual = hashlib.sha256(text.encode()).hexdigest()
            assert actual == golden["panels"][name][str(width)], (
                name,
                width,
                text,
            )
        if snapshot is not None:
            actual = hashlib.sha256(build_export(snapshot).encode()).hexdigest()
            assert actual == golden["exports"][name], name


def test_every_source_key_has_valid_translator_context() -> None:
    source = catalogues.load_catalogue("en", "tui.info")
    contexts = catalogues.context("en", "tui.info")
    assert len(source) == 277  # Frozen source inventory for the translation handoff.
    assert set(contexts) == set(source)
    for key, context in contexts.items():
        assert catalogues.context_entry_problem(context) is None, key
        assert context["role"] and context["note"] and context["tone"]


def test_missing_namespace_and_key_use_english_source_plural_rules(monkeypatch, tmp_path) -> None:
    root = tmp_path / "catalogues"
    (root / "en").mkdir(parents=True)
    (root / "fr").mkdir()
    (root / "en" / "tui.info.json").write_text(json.dumps({
        "tui.info.count.session": "{count, plural, one {# session} other {# sessions}}",
    }))
    monkeypatch.setattr(catalogues, "_CATALOGUES", root)
    set_active_locale("fr")
    assert info_plural(0, "session") == "0 sessions"
    assert info_plural(1, "session") == "1 session"
    assert info_plural(2, "session") == "2 sessions"
    # A carried namespace missing this key follows the identical source route.
    (root / "fr" / "tui.info.json").write_text('{}')
    from local_operator.i18n.text import _sources
    _sources.cache_clear()
    assert info_plural(0, "session") == "0 sessions"


def test_marker_locale_reaches_body_and_clipboard_without_mutating_snapshot(
    monkeypatch, tmp_path
) -> None:
    root = tmp_path / "catalogues"
    (root / "en").mkdir(parents=True)
    (root / "fr").mkdir()
    source = catalogues.load_catalogue("en", "tui.info")
    (root / "en" / "tui.info.json").write_text(json.dumps(source))
    markers = dict(source)
    markers["tui.info.heading.install"] = "MARKER-INSTALL"
    markers["tui.info.label.version"] = "MARKER-VERSION"
    markers["tui.info.export.heading_install"] = "## MARKER-EXPORT"
    (root / "fr" / "tui.info.json").write_text(json.dumps(markers))
    monkeypatch.setattr(catalogues, "_CATALOGUES", root)
    snapshot = source_cases()["populated"]
    before = asdict(snapshot)
    set_active_locale("fr")
    assert "MARKER-INSTALL" in build_info_report(snapshot, _live()).plain
    assert "MARKER-VERSION" in build_info_report(snapshot, _live()).plain
    exported = build_export(snapshot)
    assert "## MARKER-EXPORT" in exported
    assert exported.startswith("local-operator 0.51.6 · uv-tool")
    assert asdict(snapshot) == before
    assert not any(control in exported for control in "\u2066\u2067\u2069")
    # The shared CLI API is intentionally NOT switched to the Info namespace.
    assert plural(2, "wake") == "2 wakes"


def test_terminal_fit_uses_cells_and_preserves_source_before_projection() -> None:
    body = _Body(51)
    body.kv("界" * 12, "model/id", notes=("界" * 20, "fits"))
    row = body.lines[0]
    assert row.plain.startswith("  " + "界" * 12 + "  model/id")
    assert row.cell_len <= 51
    # Projection follows truncation; English fallback must not acquire RLI.
    row = InfoRow()
    row.append(info_text(tui_info.label_version()))
    row.truncate(4)
    assert paint_info_row(row).plain == "Vers"


def test_rtl_projection_is_source_aware_post_fit_and_not_exported(monkeypatch, tmp_path) -> None:
    root = tmp_path / "catalogues"
    (root / "en").mkdir(parents=True)
    (root / "ur").mkdir()
    source = catalogues.load_catalogue("en", "tui.info")
    (root / "en" / "tui.info.json").write_text(json.dumps(source))
    # Synthetic glyph probes, NOT translations and NOT a native-layout claim.
    markers = {"tui.info.label.version": "\u0627" * 5}
    markers["tui.info.not_answering_clause.remedy_one"] = "\u0627 " + source[
        "tui.info.not_answering_clause.remedy_one"
    ]
    (root / "ur" / "tui.info.json").write_text(json.dumps(markers))
    monkeypatch.setattr(catalogues, "_CATALOGUES", root)
    set_active_locale("ur")
    row = InfoRow()
    row.append(info_text(tui_info.label_version()))
    row.truncate(4)
    assert paint_info_row(row).plain == "\u2067" + "\u0627" * 4 + "\u2069"
    fallback = InfoRow()
    fallback.append(info_text(tui_info.label_model()))
    assert paint_info_row(fallback).plain == "Model"
    command = InfoRow()
    command.append(info_text(tui_info.not_answering_clause_remedy_one(pid="4243")))
    painted = paint_info_row(command).plain
    assert "\u2066'lop stop --pid 4243 --force'\u2069" in painted
    exported = build_export(source_cases()["not_answering"])
    assert not any(control in exported for control in "\u2066\u2067\u2069")


def _unkeyed_custom_labels(source: str) -> list[str]:
    """Owned custom sink control, independent of the broad ratchet's blind spot.

    This intentionally checks labels/headings, not raw model/path/data values.
    #4414 owns the general scanner patch; this source contract does not import
    that old scanner commit or widen unrelated renderer sinks.
    """
    found = []
    for node in ast.walk(ast.parse(source)):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "body"
                and node.func.attr in {"kv", "header"} and node.args
                and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
            found.append(node.args[0].value)
    return found


def test_custom_label_guard_positive_negative_and_technical_controls() -> None:
    assert _unkeyed_custom_labels('body.kv("Unkeyed label", model_id)') == ["Unkeyed label"]
    assert _unkeyed_custom_labels('body.header("Unkeyed heading")') == ["Unkeyed heading"]
    assert _unkeyed_custom_labels('body.kv(_i(keys.label_model()), "provider/model")') == []
    assert _unkeyed_custom_labels('body.kv(label, "/technical/path")') == []
    source = (ROOT / "local_operator/tui/widgets/info_panel.py").read_text()
    assert _unkeyed_custom_labels(source) == []
