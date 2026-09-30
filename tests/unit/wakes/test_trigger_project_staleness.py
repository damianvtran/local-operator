"""The project-staleness source: the rule, the threshold, the payload.

THE RULE UNDER TEST IS A MIRROR, deliberately: the store's
``projects.progress_is_stale`` is the authority, this source is the supervisor's
stdlib copy, and the two must agree about the WINDOW even though only one of
them can import the other. So these tests pin the source against a fabricated
projects store and the published settings snapshot — and one test pins the
mirrored constant to ``projects.PROJECT_PROGRESS_STALE_S`` so the default can
never silently fork.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from local_operator.wakes import triggers
from local_operator.wakes.triggers import TriggerContext
from local_operator.wakes.triggers.sources import project_staleness

NOW_S = 1_700_000_000.0
NOW_MS = int(NOW_S * 1000)


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "config"
    root.mkdir()
    return root


def _row(root: Path, project_id: str, **fields: Any) -> None:
    row = {
        "id": project_id,
        "name": project_id,
        "status": "active",
        "progress": "did a thing",
        "progress_updated_at": NOW_S - 6 * 3600,
        "sessions": [],
    }
    row.update(fields)
    (root / "projects").mkdir(exist_ok=True)
    (root / "projects" / f"{project_id}.json").write_text(json.dumps(row))


def _ctx(root: Path, values: dict[str, Any] | None = None) -> TriggerContext:
    return TriggerContext(config_dir=root, now_ms=NOW_MS, values=dict(values or triggers.DEFAULTS))


def _evaluate(root: Path, values: dict[str, Any] | None = None) -> list[triggers.TriggerInstance]:
    return list(project_staleness.SOURCE.evaluate(_ctx(root, values)))


# -- the rule --------------------------------------------------------------------


def test_only_live_statuses_with_an_old_report_are_stale(tmp_path: Path) -> None:
    root = _root(tmp_path)
    _row(root, "planning", status="planning")
    _row(root, "active", status="active")
    _row(root, "qa", status="qa")
    _row(root, "validation", status="validation")
    _row(root, "paused", status="paused", progress_updated_at=None)
    _row(root, "done", status="done", progress_updated_at=None)
    _row(root, "archived", status="archived", progress_updated_at=None)
    _row(root, "fresh", status="active", progress_updated_at=NOW_S - 60)

    keys = sorted(i.key for i in _evaluate(root))
    assert keys == ["active", "planning", "qa", "validation"]


def test_no_progress_is_stale_by_construction(tmp_path: Path) -> None:
    root = _root(tmp_path)
    _row(root, "blank", progress="", progress_updated_at=NOW_S - 60)
    _row(root, "missing", progress=None, progress_updated_at=None)
    _row(root, "unstamped", progress="words", progress_updated_at=None)

    keys = sorted(i.key for i in _evaluate(root))
    assert keys == ["blank", "missing", "unstamped"]


def test_the_threshold_comes_from_the_snapshot(tmp_path: Path) -> None:
    root = _root(tmp_path)
    # 3 h 59 m old: stale at a 2 h window, fresh at the 4 h default.
    _row(root, "border", progress_updated_at=NOW_S - (4 * 3600 - 60))

    assert [i.key for i in _evaluate(root)] == []  # default 4 h: fresh
    values = dict(triggers.DEFAULTS)
    values["projects.stale_after_hours"] = 2
    assert [i.key for i in _evaluate(root, values)] == ["border"]


def test_the_default_matches_the_store_constant() -> None:
    """The mirror's default is pinned to the authority it mirrors."""
    from local_operator.projects import PROJECT_PROGRESS_STALE_S

    assert project_staleness._DEFAULT_STALE_S == PROJECT_PROGRESS_STALE_S


def test_a_malformed_snapshot_value_falls_back_to_the_default() -> None:
    assert project_staleness.stale_after_s({"projects.stale_after_hours": "soon"}) == 14400.0
    assert project_staleness.stale_after_s({"projects.stale_after_hours": 0}) == 14400.0
    assert project_staleness.stale_after_s({}) == 14400.0
    assert project_staleness.stale_after_s({"projects.stale_after_hours": 2}) == 7200.0


def test_an_unreadable_row_costs_one_candidate(tmp_path: Path) -> None:
    root = _root(tmp_path)
    _row(root, "good")
    (root / "projects" / "torn.json").write_text("{not json")
    (root / "projects" / ".hidden").write_text("{}")

    assert [i.key for i in _evaluate(root)] == ["good"]


def test_an_absent_store_is_zero_candidates(tmp_path: Path) -> None:
    root = _root(tmp_path)
    assert _evaluate(root) == []


# -- fingerprint and payload ------------------------------------------------------


def test_the_fingerprint_is_the_completion_checks_identity(tmp_path: Path) -> None:
    root = _root(tmp_path)
    stamp = NOW_S - 6 * 3600
    _row(root, "atlas", status="qa", progress_updated_at=stamp)

    instance = _evaluate(root)[0]
    assert instance.source == "project_staleness"
    assert instance.key == "atlas"
    assert instance.fingerprint == ("atlas", "qa", int(stamp))


def test_the_payload_carries_display_name_age_and_session_liveness(tmp_path: Path) -> None:
    root = _root(tmp_path)
    # A linked session that exists with a transcript and a spool.
    session_dir = root / "sessions" / "ab12cd34ef56"
    session_dir.mkdir(parents=True)
    (session_dir / "transcript.jsonl").write_text("")
    (session_dir / "inbox.jsonl").write_text("")
    _row(
        root,
        "atlas",
        name="atlas-migration",
        title="Atlas migration",
        sessions=["ab12cd34ef56", "ffffeeeedddd"],
    )

    payload = _evaluate(root)[0].payload
    assert payload["display_name"] == "Atlas migration"
    assert payload["status"] == "active"
    assert payload["progress_age_s"] == 6 * 3600
    live = {s["id"]: s["live"] for s in payload["sessions"]}
    assert live["ab12cd34ef56"] == "cold"
    assert live["ffffeeeedddd"] == "missing"
    assert payload["sessions"][0]["last_activity_age_s"] is not None


def test_liveness_reads_the_runtime_registry_when_it_answers(tmp_path: Path, monkeypatch) -> None:
    root = _root(tmp_path)
    for sid in ("ab12cd34ef56", "deadbeef0001"):
        directory = root / "sessions" / sid
        directory.mkdir(parents=True)
        (directory / "transcript.jsonl").write_text("")
    _row(root, "atlas", sessions=["ab12cd34ef56", "deadbeef0001"])

    class _Record:
        def __init__(self, sid: str) -> None:
            self.session_id = sid

    import local_operator.session.runtime.registry as registry_mod

    monkeypatch.setattr(
        registry_mod,
        "scan",
        lambda config_dir: [(_Record("ab12cd34ef56"), "live"), (_Record("deadbeef0001"), "wedged")],
    )

    payload = _evaluate(root)[0].payload
    live = {s["id"]: s["live"] for s in payload["sessions"]}
    assert live == {"ab12cd34ef56": "live", "deadbeef0001": "wedged"}


def test_a_scan_failure_reads_cold_not_live(tmp_path: Path, monkeypatch) -> None:
    root = _root(tmp_path)
    directory = root / "sessions" / "ab12cd34ef56"
    directory.mkdir(parents=True)
    (directory / "transcript.jsonl").write_text("")
    _row(root, "atlas", sessions=["ab12cd34ef56"])

    import local_operator.session.runtime.registry as registry_mod

    def _explode(config_dir):  # noqa: ANN001
        raise RuntimeError("registry down")

    monkeypatch.setattr(registry_mod, "scan", _explode)
    payload = _evaluate(root)[0].payload
    assert payload["sessions"][0]["live"] == "cold"


def test_the_source_enabled_switch_reads_the_snapshot(tmp_path: Path) -> None:
    assert project_staleness.SOURCE.enabled({}) is True
    assert (
        project_staleness.SOURCE.enabled({"wakes.triggers.project_staleness.enabled": False})
        is False
    )
    # Fail-open on a malformed enabled value only means "watch" when the value
    # is not a recognised false — the registry writes real bools, so a junk
    # value should read the default, not flip the source off by accident.
    assert project_staleness.SOURCE.enabled({"wakes.triggers.project_staleness.enabled": 1}) is True
