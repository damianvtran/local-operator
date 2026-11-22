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
from local_operator.info.copy import (
    InfoRow,
    info_duration,
    info_plural,
    info_text,
    paint_info_row,
)
from local_operator.info.model import (
    InfoSnapshot,
    InstallInfo,
    SessionLine,
    SessionsInfo,
    format_duration,
)
from local_operator.info.render import build_export, not_answering_clause, plural
from local_operator.tui.widgets.info_panel import (
    _Body,
    _install_section,
    build_info_report,
)
from tests.unit.tui.test_info_panel import (
    _fleet_sessions,
    _live,
    _quiet_owner_sessions,
    _snapshot,
)

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
                import_path="/opt/checkout/local_operator",
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
    (root / "en" / "tui.info.json").write_text(
        json.dumps(
            {
                "tui.info.count.session": "{count, plural, one {# session} other {# sessions}}",
            }
        )
    )
    monkeypatch.setattr(catalogues, "_CATALOGUES", root)
    set_active_locale("fr")
    assert info_plural(0, "session") == "0 sessions"
    assert info_plural(1, "session") == "1 session"
    assert info_plural(2, "session") == "2 sessions"
    # A carried namespace missing this key follows the identical source route.
    (root / "fr" / "tui.info.json").write_text("{}")
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
    assert snapshot is not None
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
    markers["tui.info.not_answering_clause.remedy_one"] = (
        "\u0627 " + source["tui.info.not_answering_clause.remedy_one"]
    )
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
    snapshot = source_cases()["not_answering"]
    assert snapshot is not None
    exported = build_export(snapshot)
    assert not any(control in exported for control in "\u2066\u2067\u2069")


@pytest.mark.parametrize("seconds", [-1, 0, 59, 60, 3599, 3600, 86399, 86400, 86400000])
def test_human_duration_keeps_english_thresholds_and_ungrouped_days(seconds) -> None:
    assert info_duration(seconds) == format_duration(seconds)


def test_duration_uses_audited_units_in_info_locale() -> None:
    from local_operator.i18n.keys import wire_session_status
    from local_operator.i18n.text import tr_record

    set_active_locale("fr")
    expected = tr_record("wire.session_status")(
        wire_session_status.duration_hours_minutes(hours=1, minutes=1)
    )
    duration = info_duration(3660)
    assert duration == expected.text != "1h 1m"
    assert all(part.source == expected.source_locale for part in duration.parts)
    outer = info_text(tui_info.install_build_age(build_age_s=duration))
    assert any(part is duration.parts[0] for part in outer.parts)


def test_missing_duration_is_whole_field_select_not_zero_or_latest_unknown(
    monkeypatch, tmp_path
) -> None:
    from local_operator.i18n.runtime import render_message
    from scripts.i18n.generate import _param_types

    source = catalogues.load_catalogue("en", "tui.info")
    root = tmp_path / "catalogues"
    for locale in ("en", "fr"):
        (root / locale).mkdir(parents=True)
    (root / "en" / "tui.info.json").write_text(json.dumps(source))
    markers = dict(source)
    for key, arg, label in (
        ("tui.info.export.build_age", "build_age_s", "  build age         "),
        ("tui.info.export.uptime", "uptime_s", "  uptime            "),
    ):
        # A select plus same-name interpolation retains the generated str API.
        assert _param_types(source[key]) == [(arg, "str")]
        assert render_message(source[key], {arg: "unmeasured"}, "en") == label + "unknown"
        assert render_message(source[key], {arg: "0s"}, "en") == label + "0s"
        markers[key] = (
            "{"
            + arg
            + ", select, unmeasured {MARKER-MISSING-"
            + arg
            + "} other {MARKER-MEASURED {"
            + arg
            + "}}}"
        )
    (root / "fr" / "tui.info.json").write_text(json.dumps(markers))
    monkeypatch.setattr(catalogues, "_CATALOGUES", root)
    set_active_locale("fr")
    exported = build_export(InfoSnapshot())
    assert "MARKER-MISSING-build_age_s" in exported
    assert "MARKER-MISSING-uptime_s" in exported
    assert "MARKER-MEASURED" not in exported
    assert "unmeasured" not in exported
    assert exported.startswith("local-operator unknown · unknown install")


@pytest.mark.parametrize("width", [51, 126, 1000])
def test_nested_actual_pid_remedy_survives_production_wrap(width) -> None:
    set_active_locale("ur")
    snapshot = InfoSnapshot(
        sessions=SessionsInfo(
            total=1,
            wedged=1,
            lines=(SessionLine(pid=4243, state="wedged", heartbeat_age_s=65),),
        )
    )
    clause = not_answering_clause(snapshot.sessions)
    body = _Body(width)
    body.note(clause)
    painted = body.to_text().plain
    for command in ("'lop stop --pid 4243'", "'/stop 4243'", "'lop stop --pid 4243 --force'"):
        assert "\u2066" + command + "\u2069" in painted
    assert all(row.cell_len <= width for row in body.lines)
    assert not any(c in build_export(snapshot) for c in "\u2066\u2067\u2069")


def test_nested_message_preserves_actual_fallback_source(monkeypatch, tmp_path) -> None:
    root = tmp_path / "catalogues"
    source = catalogues.load_catalogue("en", "tui.info")
    for locale in ("en", "ur"):
        (root / locale).mkdir(parents=True)
    (root / "en" / "tui.info.json").write_text(json.dumps(source))
    # Synthetic outer glyphs with an English-only leaf, not native translation.
    (root / "ur" / "tui.info.json").write_text(
        json.dumps(
            {
                "tui.info.not_answering_clause.full": "ا {remedy} ا {remedy}",
            }
        )
    )
    monkeypatch.setattr(catalogues, "_CATALOGUES", root)
    set_active_locale("ur")
    leaf = info_text(tui_info.not_answering_clause_remedy_one(pid="4243"))
    outer = info_text(
        tui_info.not_answering_clause_full(count=1, age="65s", pids="4243", remedy=leaf)
    )
    assert sum(part is leaf.parts[0] for part in outer.parts) == 2
    body = _Body(1000)
    body.note(outer)
    assert "\u2066'lop stop" not in body.to_text().plain
    assert all(part.source == "en" for part in outer.parts if "lop stop" in part.text)


@pytest.mark.parametrize("locale", ["en", "fr", "es", "zh-CN", "vi", "hi", "ru", "ur"])
def test_source_identity_survives_same_narrow_input(locale) -> None:
    set_active_locale(locale)
    body = _Body(51)
    _install_section(
        body,
        InfoSnapshot(
            install=InstallInfo(
                version="0.90.0",
                is_git_snapshot=True,
                source_ref="a1b2c3d4e5f67890",
            )
        ),
        False,
    )
    assert "a1b2c3d4e5f6" in body.to_text().plain
    assert all(row.cell_len <= 51 for row in body.lines)


def test_semantic_context_counterexamples_follow_actual_model_facts(monkeypatch, tmp_path) -> None:
    from types import SimpleNamespace

    from local_operator.info.collect import build_subagent_tree
    from local_operator.info.copy import info_state
    from local_operator.session.runtime import registry, stall_watchdog
    from local_operator.session.runtime.types import SessionRecord

    # Dead PID plus fresh heartbeat is stale; a live PID plus old heartbeat is
    # not. These facts, not the English word's everyday meaning, own translation.
    # Use the registry's actual record contract, with the same PID/beat facts;
    # default live-state fields must not turn a liveness probe into a mock API.
    record = SessionRecord(
        pid=4243,
        heartbeat_at=100.0,
        kind="tui",
        session_id="synthetic-info",
        conversation_name="",
        cwd=str(tmp_path),
        model_label="",
        control_port=0,
        control_key="",
    )
    monkeypatch.setattr(registry, "pid_alive", lambda *a, **kw: False)
    dead = registry.classify(record, now=100.0, check_zombie=False)
    assert dead.state == "stale" and dead.heartbeat_age_s == 0
    assert info_state(dead.state) == info_text(tui_info.state_stale())
    monkeypatch.setattr(registry, "pid_alive", lambda *a, **kw: True)
    assert registry.classify(record, now=1000.0, check_zombie=False).state == "wedged"

    # A held label may be conservative/unproven, not a diagnosed hung process.
    text = (
        stall_watchdog.FIRED_MARKER + " synthetic\n" + stall_watchdog.HELD_MARKER + "old format\n"
    )
    assert (
        stall_watchdog.held_reading(tmp_path / "synthetic.dump", text)
        == stall_watchdog.HELD_UNPROVEN
    )

    # Five hidden siblings occupy one hidden level: deeper counts nodes.
    nodes = [SimpleNamespace(job_id="root", parent_job_id=None)]
    nodes += [SimpleNamespace(job_id=f"child-{n}", parent_job_id="root") for n in range(5)]
    visible, deepest, hidden = build_subagent_tree(nodes, max_depth=1)
    assert len(visible) == 1 and deepest == 1 and hidden == 5

    # A known runtime census can coexist with incomplete trajectory reporting.
    snapshot = InfoSnapshot(
        sessions=SessionsInfo(
            live=4,
            total=4,
            subagents_unreported=1,
            fleet_trajectories=7,
        )
    )
    rendered = build_info_report(snapshot, _live(), 126).plain
    assert str(info_text(tui_info.agents_summary_full(runtimes=4, bound="≥", total=7))) in rendered
    assert "≥4" not in rendered


def _unkeyed_custom_labels(source: str) -> list[str]:
    """Owned custom sink control, independent of the broad ratchet's blind spot.

    This intentionally checks labels/headings, not raw model/path/data values.
    #4414 owns the general scanner patch; this source contract does not import
    that old scanner commit or widen unrelated renderer sinks.
    """
    found = []
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "body"
            and node.func.attr in {"kv", "header"}
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            found.append(node.args[0].value)
    return found


def test_custom_label_guard_positive_negative_and_technical_controls() -> None:
    assert _unkeyed_custom_labels('body.kv("Unkeyed label", model_id)') == ["Unkeyed label"]
    assert _unkeyed_custom_labels('body.header("Unkeyed heading")') == ["Unkeyed heading"]
    assert _unkeyed_custom_labels('body.kv(_i(keys.label_model()), "provider/model")') == []
    assert _unkeyed_custom_labels('body.kv(label, "/technical/path")') == []
    source = (ROOT / "local_operator/tui/widgets/info_panel.py").read_text()
    assert _unkeyed_custom_labels(source) == []
