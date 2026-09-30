"""The generic wake-trigger layer: registry, records, dedupe, bounds, gates.

THE PROPERTY UNDER TEST is that the layer is a WITNESS, not a second wake
mechanism: an evaluation pass turns conditions into ONE pending record per
target, with dedupe/bounds/suppression enforced at creation, and every write
to the target's schedule list stays with the target's own engine. The
supervisor-side integration (due sets, fireability, the ``--once`` pass) is
pinned in ``test_supervisor.py``; the source's own rule and payload in
``test_trigger_project_staleness.py``; the consume/settle protocol in
``tests/unit/aida/test_aida_triggers.py``.

Fail-proof note: the dedupe and budget tests are exercised in a mutated state
(fingerprint moved / window rolled) so a layer that ignored either would red
them; the zero-footprint test asserts the SET OF FILES, so any record write on
a disabled install fails it by name.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from local_operator.wakes import triggers

NOW_MS = 1_700_000_000_000


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "config"
    root.mkdir()
    return root


def _aida(root: Path, session_id: str = "aida00000001", *, attachment: str = "aida") -> Path:
    """An enabled install: her state file + a session directory."""
    (root / "aida").mkdir(exist_ok=True)
    (root / "aida" / "state.json").write_text(
        json.dumps({"schema_version": 1, "session_id": session_id})
    )
    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "transcript.jsonl").write_text("")
    (directory / "attachment.json").write_text(
        json.dumps({"team": "", "agent": attachment, "goal": ""})
    )
    return directory


def _project(root: Path, project_id: str, **fields: Any) -> None:
    row = {
        "id": project_id,
        "name": project_id,
        "status": "active",
        "progress": "did a thing",
        "progress_updated_at": (NOW_MS / 1000.0) - 6 * 3600,
        "sessions": [],
    }
    row.update(fields)
    (root / "projects").mkdir(exist_ok=True)
    (root / "projects" / f"{project_id}.json").write_text(json.dumps(row))


def _instance(
    key: str = "atlas",
    *,
    source: str = "project_staleness",
    fingerprint: tuple[Any, ...] = ("atlas", "active", 1),
    age_s: float = 21600.0,
) -> triggers.TriggerInstance:
    return triggers.TriggerInstance(
        source=source, key=key, fingerprint=fingerprint, payload={"display_name": key}, age_s=age_s
    )


class _FakeSource:
    def __init__(self, name: str, *, enabled: bool = True, instances: Sequence[Any] = ()) -> None:
        self.name = name
        self._enabled = enabled
        self._instances = list(instances)
        self.evaluations = 0

    def enabled(self, values: Mapping[str, Any]) -> bool:
        return self._enabled

    def evaluate(self, ctx: triggers.TriggerContext) -> Sequence[Any]:
        self.evaluations += 1
        return list(self._instances)


@pytest.fixture(autouse=True)
def _clean_registry():
    triggers._reset_registry()
    yield
    triggers._reset_registry()


# -- the registry ---------------------------------------------------------------


def test_sources_evaluate_in_name_order_and_respect_enabled(tmp_path: Path) -> None:
    beta = _FakeSource("beta", instances=[_instance(key="b")])
    alpha = _FakeSource("alpha", instances=[_instance(key="a")])
    off = _FakeSource("gamma", enabled=False, instances=[_instance(key="g")])
    triggers.register(beta)
    triggers.register(alpha)
    triggers.register(off)

    out = triggers.evaluate_all(_root(tmp_path), NOW_MS, {})

    assert [i.key for i in out] == ["a", "b"]
    assert off.evaluations == 0
    assert beta.evaluations == 1


def test_a_failing_source_costs_only_its_own_pass(tmp_path: Path) -> None:
    class _Exploding(_FakeSource):
        def evaluate(self, ctx: triggers.TriggerContext) -> Sequence[Any]:
            raise RuntimeError("boom")

    triggers.register(_Exploding("bad"))
    triggers.register(_FakeSource("good", instances=[_instance(key="ok")]))
    assert [i.key for i in triggers.evaluate_all(_root(tmp_path), NOW_MS, {})] == ["ok"]


# -- the suppression matrix ------------------------------------------------------


def test_no_aida_environment_switch_writes_nothing(tmp_path: Path, monkeypatch) -> None:
    root = _root(tmp_path)
    _aida(root)
    _project(root, "atlas")
    monkeypatch.setenv("LOCAL_OPERATOR_NO_AIDA", "1")
    before = sorted(str(p.relative_to(root)) for p in root.rglob("*"))
    assert triggers.sweep(root, now_ms=NOW_MS) == []
    assert sorted(str(p.relative_to(root)) for p in root.rglob("*")) == before


def test_zero_footprint_without_her_state_file(tmp_path: Path) -> None:
    """A never-enabled install pays READS only — no dir, no record, no writes."""
    root = _root(tmp_path)
    _project(root, "atlas")
    before = sorted(str(p.relative_to(root)) for p in root.rglob("*"))
    assert triggers.sweep(root, now_ms=NOW_MS) == []
    assert sorted(str(p.relative_to(root)) for p in root.rglob("*")) == before


def test_the_master_switch_and_the_source_switch_suppress(tmp_path: Path) -> None:
    root = _root(tmp_path)
    _aida(root)
    _project(root, "atlas")
    (root / "wakes" / "triggers").mkdir(parents=True)

    (root / "wakes" / "triggers" / "settings.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "values": {"wakes.triggers.enabled": False},
            }
        )
    )
    assert triggers.sweep(root, now_ms=NOW_MS) == []

    (root / "wakes" / "triggers" / "settings.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "values": {"wakes.triggers.project_staleness.enabled": False},
            }
        )
    )
    assert triggers.sweep(root, now_ms=NOW_MS) == []

    (root / "wakes" / "triggers" / "settings.json").unlink()
    assert triggers.sweep(root, now_ms=NOW_MS) == ["aida00000001"]


def test_a_paused_target_writes_no_record(tmp_path: Path) -> None:
    root = _root(tmp_path)
    _aida(root)
    _project(root, "atlas")
    # ``/aida pause`` stamps ``held_at`` on her wake-index entry (the entry
    # carries the index schema, or the reader skips it as unknown).
    (root / "wakes").mkdir(exist_ok=True)
    entry = root / "wakes" / "aida00000001.json"
    entry.write_text(
        json.dumps({"schema": 1, "session_id": "aida00000001", "held_at": NOW_MS, "schedules": []})
    )
    assert triggers.sweep(root, now_ms=NOW_MS) == []
    # Resume clears it; the still-stale project re-candidates.
    entry.write_text(json.dumps({"schema": 1, "session_id": "aida00000001", "schedules": []}))
    assert triggers.sweep(root, now_ms=NOW_MS) == ["aida00000001"]


def test_a_settings_pause_and_a_disable_decline_at_creation(tmp_path: Path) -> None:
    """Both decline states reach the pass through the published snapshot.

    Round 1 (R1): the creation gate read only the entry's ``held_at``, so a
    pause written through the settings row (which stamped nothing) and
    ``aida.enabled=false`` both kept sweeping records. The gate now asks the
    snapshot, which both surfaces write.
    """
    root = _root(tmp_path)
    _aida(root)
    _project(root, "atlas")
    settings = root / "wakes" / "triggers" / "settings.json"
    settings.parent.mkdir(parents=True, exist_ok=True)

    settings.write_text(json.dumps({"schema_version": 1, "values": {"aida.cadence.paused": True}}))
    assert triggers.declines(root, "aida00000001") == "paused"
    assert triggers.sweep(root, now_ms=NOW_MS) == []

    settings.write_text(json.dumps({"schema_version": 1, "values": {"aida.enabled": False}}))
    assert triggers.declines(root, "aida00000001") == "disabled"
    assert triggers.sweep(root, now_ms=NOW_MS) == []

    settings.unlink()
    assert triggers.declines(root, "aida00000001") is None
    assert triggers.sweep(root, now_ms=NOW_MS) == ["aida00000001"]


def test_a_burst_under_a_lever_creates_nothing(tmp_path: Path) -> None:
    """QA round 1: a disabled/paused target must not get a record from a burst
    either — creation is gated before evaluation, and the file set proves
    nothing was written."""
    root = _root(tmp_path)
    _aida(root)
    for index in range(5):
        _project(root, f"p-{index}")
    settings = root / "wakes" / "triggers" / "settings.json"
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text(json.dumps({"schema_version": 1, "values": {"aida.enabled": False}}))
    before = sorted(str(p.relative_to(root)) for p in root.rglob("*"))
    assert triggers.sweep(root, now_ms=NOW_MS) == []
    assert sorted(str(p.relative_to(root)) for p in root.rglob("*")) == before


# -- the class mirror ------------------------------------------------------------


def test_the_seed_resolves_proactive_and_an_empty_attachment_reads_reactive(tmp_path: Path) -> None:
    root = _root(tmp_path)
    _aida(root)
    assert triggers._class_reactive(root, "aida00000001") is False

    _aida(root, "aida00000002", attachment="")
    # No attachment at all: never positively proactive ⇒ fail closed.
    assert triggers._class_reactive(root, "aida00000002") is True


def test_a_registry_row_shadows_the_seed(tmp_path: Path) -> None:
    root = _root(tmp_path)
    _aida(root)
    row = root / "agents" / "aid1"
    row.mkdir(parents=True)
    (row / "agent.yml").write_text("name: aida\ntags:\n- role\n- class:reactive\n")
    assert triggers._class_reactive(root, "aida00000001") is True

    (row / "agent.yml").write_text("name: aida\ntags:\n- role\n- class:proactive\n")
    assert triggers._class_reactive(root, "aida00000001") is False


def test_a_role_row_without_a_class_tag_reads_reactive(tmp_path: Path) -> None:
    root = _root(tmp_path)
    _aida(root)
    row = root / "agents" / "aid1"
    row.mkdir(parents=True)
    # Every pre-class row reads reactive (the app's own rule) — this mirror
    # must not silently upgrade it because the seed says proactive.
    (row / "agent.yml").write_text("name: aida\ntags:\n- role\n")
    assert triggers._class_reactive(root, "aida00000001") is True


def test_an_unreadable_agent_row_reads_reactive(tmp_path: Path) -> None:
    root = _root(tmp_path)
    _aida(root)
    row = root / "agents" / "aid1"
    row.mkdir(parents=True)
    (row / "agent.yml").write_text("name: aida\ntags:\n- role\n- class:proactive\n")
    (row / "agent.yml").chmod(0o000)
    try:
        assert triggers._class_reactive(root, "aida00000001") is True
    finally:
        (row / "agent.yml").chmod(0o644)


def test_a_role_row_reads_flow_tags_and_a_commented_name(tmp_path: Path) -> None:
    """PyYAML shapes the line reader used to miss: a flow-style tag list and a
    trailing comment on the ``name:`` line. Both made the row invisible to the
    mirror, which then read the SEED's class while the engine read the ROW's
    (review round 1, R3)."""
    root = _root(tmp_path)
    _aida(root)
    row = root / "agents" / "aid1"
    row.mkdir(parents=True)
    (row / "agent.yml").write_text(
        "name: aida  # the operator's assistant\ntags: [role, class:reactive]\n"
    )
    assert triggers._class_reactive(root, "aida00000001") is True

    (row / "agent.yml").write_text('name: "aida" # note\ntags: ["role", "class:proactive"]\n')
    assert triggers._class_reactive(root, "aida00000001") is False


def test_an_unreadable_tags_shape_reads_reactive(tmp_path: Path) -> None:
    """A ``tags:`` value this reader cannot parse is DOUBT, not "no role tag"
    (R3): the mirror fails closed so the seed cannot silently win a class the
    engine would read differently."""
    root = _root(tmp_path)
    _aida(root)
    row = root / "agents" / "aid1"
    row.mkdir(parents=True)
    (row / "agent.yml").write_text("name: aida\ntags: [role, class:proactive\n")  # unbalanced
    assert triggers._class_reactive(root, "aida00000001") is True


# -- records, dedupe, bounds -----------------------------------------------------


def test_a_sweep_writes_one_record_and_dedupes_on_an_unchanged_fingerprint(
    tmp_path: Path, monkeypatch
) -> None:
    root = _root(tmp_path)
    _aida(root)
    instance = _instance()
    triggers.register(_FakeSource("project_staleness", instances=[instance]))

    assert triggers.sweep(root, now_ms=NOW_MS) == ["aida00000001"]
    record = triggers.read_pending_record(root, "aida00000001")
    assert record is not None
    assert record["instances"][0]["key"] == "atlas"
    assert record["instances"][0]["fingerprint"] == ["atlas", "active", 1]

    # Unchanged fingerprint: silent (the no-spam rule).
    assert triggers.sweep(root, now_ms=NOW_MS + 1000) == []

    # Fingerprint moves and the minimum gap has cleared: a new episode.
    moved = _instance(fingerprint=("atlas", "active", 2))
    triggers._reset_registry()
    triggers.register(_FakeSource("project_staleness", instances=[moved]))
    assert triggers.sweep(root, now_ms=NOW_MS + 61 * 60 * 1000) == ["aida00000001"]


def test_the_minimum_gap_blocks_and_then_clears(tmp_path: Path) -> None:
    root = _root(tmp_path)
    _aida(root)
    triggers.register(_FakeSource("project_staleness", instances=[_instance()]))
    assert triggers.sweep(root, now_ms=NOW_MS) == ["aida00000001"]
    triggers.settle(
        root, "aida00000001", [("project_staleness", "atlas")], expected_updated_at_ms=None
    )

    moved = _instance(fingerprint=("atlas", "active", 9))
    triggers._reset_registry()
    triggers.register(_FakeSource("project_staleness", instances=[moved]))
    # Still inside the 60-minute default gap: blocked, and NOT marked notified.
    assert triggers.sweep(root, now_ms=NOW_MS + 30 * 60 * 1000) == []
    # The gap clears: the same candidate fires.
    assert triggers.sweep(root, now_ms=NOW_MS + 61 * 60 * 1000) == ["aida00000001"]


def test_the_per_day_budget_blocks_and_rolls_over(tmp_path: Path) -> None:
    root = _root(tmp_path)
    _aida(root)
    settings = root / "wakes" / "triggers"
    settings.mkdir(parents=True)
    (settings / "settings.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "values": {"wakes.triggers.max_per_day": 1, "wakes.triggers.min_gap_minutes": 10},
            }
        )
    )

    triggers.register(
        _FakeSource("project_staleness", instances=[_instance(fingerprint=("a", "active", 1))])
    )
    assert triggers.sweep(root, now_ms=NOW_MS) == ["aida00000001"]
    triggers.settle(
        root, "aida00000001", [("project_staleness", "atlas")], expected_updated_at_ms=None
    )

    # A different episode inside the rolling window: budget-blocked.
    triggers._reset_registry()
    triggers.register(
        _FakeSource("project_staleness", instances=[_instance(fingerprint=("a", "active", 2))])
    )
    assert triggers.sweep(root, now_ms=NOW_MS + 30 * 60 * 1000) == []
    # 24 h after the first fire the window rolls: allowed again.
    assert triggers.sweep(root, now_ms=NOW_MS + 24 * 3600 * 1000 + 1) == ["aida00000001"]


def test_zero_disables_the_budget(tmp_path: Path) -> None:
    root = _root(tmp_path)
    _aida(root)
    settings = root / "wakes" / "triggers"
    settings.mkdir(parents=True)
    (settings / "settings.json").write_text(
        json.dumps({"schema_version": 1, "values": {"wakes.triggers.max_per_day": 0}})
    )
    triggers.register(_FakeSource("project_staleness", instances=[_instance()]))
    assert triggers.sweep(root, now_ms=NOW_MS) == []


def test_instances_merge_into_one_record_and_cap_older_first(tmp_path: Path) -> None:
    root = _root(tmp_path)
    _aida(root)
    instances = [
        _instance(key=f"p{i:02d}", fingerprint=(f"p{i:02d}", "active", i), age_s=float(1000 - i))
        for i in range(25)
    ]
    triggers.register(_FakeSource("project_staleness", instances=instances))
    assert triggers.sweep(root, now_ms=NOW_MS) == ["aida00000001"]

    record = triggers.read_pending_record(root, "aida00000001")
    assert record is not None
    assert len(record["instances"]) == triggers.INSTANCE_CAP
    assert record["overflow"] == 25 - triggers.INSTANCE_CAP
    # Older-first: the oldest age (highest age_s) wins the cap.
    assert record["instances"][0]["key"] == "p00"


def test_commit_marks_notified_only_when_a_record_was_written(tmp_path: Path) -> None:
    root = _root(tmp_path)
    _aida(root)
    triggers.register(_FakeSource("project_staleness", instances=[_instance()]))

    # Budget-blocked (max 0): nothing written, nothing marked notified.
    settings = root / "wakes" / "triggers"
    settings.mkdir(parents=True)
    (settings / "settings.json").write_text(
        json.dumps({"schema_version": 1, "values": {"wakes.triggers.max_per_day": 0}})
    )
    assert triggers.sweep(root, now_ms=NOW_MS) == []
    # Blocked means NOTHING was written — not even the dedupe/budget state.
    assert not (root / "wakes" / "triggers" / "state.json").exists()

    # Budget clears: the SAME candidate fires (it was not lost).
    (settings / "settings.json").write_text(
        json.dumps({"schema_version": 1, "values": {"wakes.triggers.max_per_day": 6}})
    )
    assert triggers.sweep(root, now_ms=NOW_MS) == ["aida00000001"]
    state = json.loads((root / "wakes" / "triggers" / "state.json").read_text())
    assert state["instances"]["project_staleness::atlas"] == ["atlas", "active", 1]


# -- settle and attempts ---------------------------------------------------------


def test_settle_is_compare_and_delete(tmp_path: Path) -> None:
    root = _root(tmp_path)
    _aida(root)
    triggers.register(_FakeSource("project_staleness", instances=[_instance()]))
    triggers.sweep(root, now_ms=NOW_MS)
    record = triggers.read_pending_record(root, "aida00000001")
    assert record is not None

    # A stale expected version leaves the record alone (a merge won the race).
    assert (
        triggers.settle(
            root,
            "aida00000001",
            [("project_staleness", "atlas")],
            expected_updated_at_ms=record["updated_at_ms"] - 1,
        )
        is False
    )
    assert triggers.read_pending_record(root, "aida00000001") is not None

    assert (
        triggers.settle(
            root,
            "aida00000001",
            [("project_staleness", "atlas")],
            expected_updated_at_ms=record["updated_at_ms"],
        )
        is True
    )
    assert triggers.read_pending_record(root, "aida00000001") is None


def test_settle_keeps_instances_it_was_not_told_about(tmp_path: Path) -> None:
    root = _root(tmp_path)
    _aida(root)
    triggers.register(
        _FakeSource(
            "project_staleness",
            instances=[
                _instance(key="atlas"),
                _instance(key="billing", fingerprint=("billing", "active", 1)),
            ],
        )
    )
    triggers.sweep(root, now_ms=NOW_MS)

    assert triggers.settle(root, "aida00000001", [("project_staleness", "atlas")]) is True
    record = triggers.read_pending_record(root, "aida00000001")
    assert record is not None
    assert [i["key"] for i in record["instances"]] == ["billing"]


def test_attempts_walk_only_on_walk_reasons(tmp_path: Path) -> None:
    root = _root(tmp_path)
    _aida(root)
    triggers.register(_FakeSource("project_staleness", instances=[_instance()]))
    triggers.sweep(root, now_ms=NOW_MS)

    triggers.note_attempt(root, "aida00000001", reason="started")
    record = triggers.read_pending_record(root, "aida00000001")
    assert record is not None and record["attempts"] == 0

    triggers.note_attempt(root, "aida00000001", reason="failed")
    record = triggers.read_pending_record(root, "aida00000001")
    assert record is not None
    assert record["attempts"] == 1
    assert record["next_attempt_ms"] == record["last_attempt_ms"] + int(
        triggers.backoff_s(1) * 1000
    )
    assert triggers.backoff_s(1) == triggers.RETRY_BASE_S
    assert triggers.backoff_s(20) == triggers.RETRY_CAP_S

    # No record: a no-op, not an error.
    triggers.settle(
        root, "aida00000001", [("project_staleness", "atlas")], expected_updated_at_ms=None
    )
    triggers.note_attempt(root, "aida00000001", reason="failed")


# -- reconcile -------------------------------------------------------------------


def test_reconcile_drops_expired_records(tmp_path: Path) -> None:
    root = _root(tmp_path)
    _aida(root)
    triggers.register(_FakeSource("project_staleness", instances=[_instance()]))
    triggers.sweep(root, now_ms=NOW_MS)
    record = triggers.read_pending_record(root, "aida00000001")
    assert record is not None
    record["updated_at_ms"] = NOW_MS - int((triggers.RECORD_TTL_S + 60) * 1000)
    (root / "wakes" / "triggers" / "pending" / "aida00000001.json").write_text(json.dumps(record))

    pending = triggers.read_pending(root)
    triggers.reconcile(root, pending)
    assert pending == {}
    assert triggers.read_pending_record(root, "aida00000001") is None


def test_reconcile_drops_ghost_targets(tmp_path: Path) -> None:
    root = _root(tmp_path)
    ghost = "ghost0000001"
    pending = {
        ghost: {
            "schema_version": 1,
            "target": ghost,
            "updated_at_ms": NOW_MS,
            "noted_at_ms": NOW_MS,
            "instances": [{"source": "project_staleness", "key": "x", "fingerprint": ["x"]}],
        }
    }
    (root / "wakes" / "triggers" / "pending").mkdir(parents=True)
    (root / "wakes" / "triggers" / "pending" / f"{ghost}.json").write_text(
        json.dumps(pending[ghost])
    )

    # No session directory: a ghost. It is dropped, file and mapping alike.
    triggers.reconcile(root, pending)
    assert pending == {}
    assert not (root / "wakes" / "triggers" / "pending" / f"{ghost}.json").exists()


# -- settings snapshot -----------------------------------------------------------


def test_publish_and_read_settings_round_trip(tmp_path: Path, monkeypatch) -> None:
    root = _root(tmp_path)
    (root / "config.yml").write_text(
        "values:\n"
        "  projects:\n    stale_after_hours: 2\n"
        "  wakes:\n    triggers:\n      max_per_day: 3\n"
    )
    # Isolate the manager the publisher builds for itself.
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))

    assert triggers.publish_settings(root) is True
    values = triggers.read_settings(root)
    assert values["projects.stale_after_hours"] == 2
    assert values["wakes.triggers.max_per_day"] == 3
    # Untouched keys carry the registry defaults.
    assert values["wakes.triggers.min_gap_minutes"] == 60


def test_read_settings_falls_back_to_defaults(tmp_path: Path) -> None:
    root = _root(tmp_path)
    assert triggers.read_settings(root) == dict(triggers.DEFAULTS)
    (root / "wakes" / "triggers").mkdir(parents=True)
    (root / "wakes" / "triggers" / "settings.json").write_text("{not json")
    assert triggers.read_settings(root) == dict(triggers.DEFAULTS)


def test_the_snapshot_defaults_pin_the_aida_levers() -> None:
    """The two ``aida.*`` snapshot defaults are the engine's own constants:
    an absent snapshot must not flip either lever."""
    from local_operator.aida import proactive

    assert triggers.DEFAULTS["aida.enabled"] == proactive.DEFAULT_ENABLED
    assert triggers.DEFAULTS["aida.cadence.paused"] == proactive.DEFAULT_PAUSED


def test_a_settings_pause_stamps_and_clears_the_hold_marker(tmp_path: Path, monkeypatch) -> None:
    """A pause written through the settings row now stamps exactly like
    ``/aida pause`` does (round 1, R1): the facade hook routes the write
    through the pause writer's marker helpers.

    A ROWLESS entry is deliberately untouched — ``store.write_entry`` treats an
    empty schedule list as "remove the entry" — and is covered instead by the
    snapshot gate above.
    """
    from local_operator import settings_io
    from local_operator.config import ConfigManager

    root = _root(tmp_path)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    _aida(root)
    entry = root / "wakes" / "aida00000001.json"
    entry.parent.mkdir(exist_ok=True)
    entry.write_text(
        json.dumps(
            {"schema": 1, "session_id": "aida00000001", "schedules": [{"id": "aida-cadence"}]}
        )
    )
    manager = ConfigManager(config_dir=root)
    setting = settings_io.resolve_key("aida.cadence.paused")
    assert setting is not None

    settings_io.write_setting(manager, setting, True)
    assert "held_at" in json.loads(entry.read_text())
    assert triggers.read_settings(root)["aida.cadence.paused"] is True

    settings_io.reset_setting(manager, setting)
    assert "held_at" not in json.loads(entry.read_text())
    assert triggers.read_settings(root)["aida.cadence.paused"] is False


def test_a_settings_write_republishes_the_snapshot(tmp_path: Path, monkeypatch) -> None:
    """The facade hook: a TUI/UI/CLI edit lands in the snapshot immediately.

    This is the write shape every real editor takes (``write_setting``/
    ``reset_setting``), and it is what makes the Wake triggers section's LIVE
    label true: the consumer — the supervisor's evaluation pass — never reads
    ``config.yml``, so the snapshot must move on the edit itself, not on a
    restart and not on the next daily reconcile (``tests/unit/session/
    test_config_live.py`` cites this cell for the host-owned exemption).
    """
    from local_operator import settings_io
    from local_operator.config import ConfigManager

    root = _root(tmp_path)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    manager = ConfigManager(config_dir=root)
    setting = settings_io.resolve_key("wakes.triggers.max_per_day")
    assert setting is not None

    settings_io.write_setting(manager, setting, 3)
    assert triggers.read_settings(root)["wakes.triggers.max_per_day"] == 3

    settings_io.reset_setting(manager, setting)
    assert triggers.read_settings(root)["wakes.triggers.max_per_day"] == 6

    # The Aida levers ride the same snapshot (round 1, R1): a pause/enable edit
    # through the facade republishes for the supervisor's record plumbing.
    pause = settings_io.resolve_key("aida.cadence.paused")
    assert pause is not None
    settings_io.write_setting(manager, pause, True)
    assert triggers.read_settings(root)["aida.cadence.paused"] is True
    settings_io.reset_setting(manager, pause)
    assert triggers.read_settings(root)["aida.cadence.paused"] is False

    # An edit of an UNRELATED key must not create the trigger store at all
    # (the gate in ``settings_io._publish_trigger_settings``).
    (tmp_path / "other").mkdir()
    others = _root(tmp_path / "other")
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(others))
    other_manager = ConfigManager(config_dir=others)
    retry = settings_io.resolve_key("retry.maxRetries")
    assert retry is not None
    settings_io.write_setting(other_manager, retry, 3)
    assert not (others / "wakes").exists()


def test_the_row_id_is_deterministic_over_the_fingerprints() -> None:
    a = [{"fingerprint": ["atlas", "active", 2]}, {"fingerprint": ["billing", "qa", 3]}]
    b = list(reversed(a))
    assert triggers.trigger_row_id(a) == triggers.trigger_row_id(b)
    assert triggers.trigger_row_id(a).startswith("aida-trigger-")
    assert len(triggers.trigger_row_id(a)) == len("aida-trigger-") + 8
    moved = [{"fingerprint": ["atlas", "active", 9]}, {"fingerprint": ["billing", "qa", 3]}]
    assert triggers.trigger_row_id(moved) != triggers.trigger_row_id(a)


def test_next_attempt_is_defensive() -> None:
    assert triggers.next_attempt_at_ms({"next_attempt_ms": "soon"}) == 0
    assert triggers.next_attempt_at_ms({"next_attempt_ms": True}) == 0
    assert triggers.next_attempt_at_ms({"next_attempt_ms": 123}) == 123
