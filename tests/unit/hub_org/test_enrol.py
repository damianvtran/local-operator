"""Auto-enrol: new organization items reach this machine without being asked.

Pins the contract in ``hub_org.enrol``: gated by the Radient Team record,
single-flight, a 60-minute per-org listing floor, only unseen items handed to
``pull_all`` with ``on_conflict="adopt-link"`` and no auto-push, ``failed``
items retried, the setting's off switch, the adopted report recorded for
``lop hub org``, and the sign-in hook firing exactly once per login. Slice A's
``list_catalogue``/``pull_all`` are stubbed through the guarded seam.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from local_operator.hub_org import eligibility as el
from local_operator.hub_org import enrol

BASE = "https://hub.example.test/v1"
ORG = el.OrgRef("t-1", "Acme", "member")


@dataclass
class Item:
    kind: str
    hub_id: str
    name: str
    local: str = "absent"


@dataclass
class Report:
    items: list[Any] = field(default_factory=list)
    adopted: list[dict[str, Any]] = field(default_factory=list)


class Hub:
    """Stub catalogue + pull recording every call."""

    def __init__(self, items: list[Item]) -> None:
        self.items = items
        self.lists = 0
        self.pulls: list[dict[str, Any]] = []
        self.failed: set[str] = set()
        self.adopted: list[dict[str, Any]] = []

    def list_catalogue(self, _client, tenant, kinds):
        assert tenant == ORG.tenant_id and tuple(kinds) == enrol.KINDS
        self.lists += 1
        return list(self.items)

    def pull_all(self, _cm, tenant, **kwargs):
        self.pulls.append({"tenant": tenant, **kwargs})
        outcomes = [
            {
                "kind": i.kind,
                "hub_id": i.hub_id,
                "name": i.name,
                "outcome": "failed" if i.hub_id in self.failed else "pulled",
            }
            for i in self.items
            if i.hub_id in kwargs["names"]
        ]
        return Report(items=outcomes, adopted=list(self.adopted))


@pytest.fixture
def cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    el.reset_memo()
    monkeypatch.setattr(el, "_configured_base_url", lambda _c: BASE)
    monkeypatch.setattr(el, "person_client", lambda *_a, **_k: (object(), "acct"))
    el.record_path(tmp_path).parent.mkdir(parents=True)
    el.record_path(tmp_path).write_text(
        json.dumps(
            {
                "schema": el.SCHEMA,
                "account_id": "acct",
                "base_url": BASE,
                "checked_at": time.time(),
                "orgs": [{"tenant_id": "t-1", "tenant_name": "Acme", "role": "member"}],
            }
        )
    )
    return tmp_path


def _wire(monkeypatch: pytest.MonkeyPatch, hub: Hub) -> None:
    monkeypatch.setattr(enrol, "_slice_a", lambda: (hub.list_catalogue, hub.pull_all))


def test_first_enrol_pulls_everything_with_adopt_link(cfg: Path, monkeypatch) -> None:
    hub = Hub([Item("agent", "a1", "coder"), Item("skill", "s1", "deploy", local="conflict")])
    hub.adopted = [
        {"name": "deploy", "kind": "skill", "hub_id": "s1", "result": "identical", "backup": "/b"}
    ]
    _wire(monkeypatch, hub)
    enrol.sync_new_items(cfg)
    (call,) = hub.pulls
    assert call["names"] == ["a1", "s1"]
    assert call["on_conflict"] == "adopt-link"
    assert call["subscribe"] is True and call["auto_push"] is False
    record = _record(cfg)
    assert record["seen"]["t-1"] == {"agent": ["a1"], "skill": ["s1"]}
    last = enrol.last_enrol(cfg)["t-1"]
    assert last["adopted"] == hub.adopted and last["pulled"] == 1


def test_linked_items_are_left_to_the_runner(cfg: Path, monkeypatch) -> None:
    hub = Hub([Item("agent", "a1", "coder", local="linked")])
    _wire(monkeypatch, hub)
    enrol.sync_new_items(cfg)
    assert hub.pulls == []
    assert _record(cfg)["seen"]["t-1"] == {"agent": ["a1"]}


def test_floor_skips_a_second_listing_within_the_hour(cfg: Path, monkeypatch) -> None:
    hub = Hub([Item("agent", "a1", "coder")])
    _wire(monkeypatch, hub)
    enrol.sync_new_items(cfg)
    enrol.sync_new_items(cfg)
    assert hub.lists == 1
    due = _record(cfg)["next_catalogue_at"]["t-1"]
    assert due >= time.time() + enrol.CATALOGUE_FLOOR_SECONDS - 60


def test_new_id_after_the_floor_is_the_only_one_pulled(cfg: Path, monkeypatch) -> None:
    hub = Hub([Item("agent", "a1", "coder")])
    _wire(monkeypatch, hub)
    enrol.sync_new_items(cfg)
    hub.items.append(Item("team", "t9", "lopdev"))

    def _elapse(data):
        data["next_catalogue_at"]["t-1"] = time.time() - 1

    el.update_record(cfg, _elapse)
    enrol.sync_new_items(cfg)
    assert [p["names"] for p in hub.pulls] == [["a1"], ["t9"]]


def test_failed_item_is_retried_next_floor(cfg: Path, monkeypatch) -> None:
    hub = Hub([Item("agent", "a1", "coder"), Item("agent", "a2", "qa")])
    hub.failed = {"a2"}
    _wire(monkeypatch, hub)
    enrol.sync_new_items(cfg)
    assert _record(cfg)["seen"]["t-1"] == {"agent": ["a1"]}


def test_lock_loser_skips(cfg: Path, monkeypatch) -> None:
    hub = Hub([Item("agent", "a1", "coder")])
    _wire(monkeypatch, hub)
    with el.refresh_lock(cfg):
        enrol.sync_new_items(cfg)
    assert hub.lists == 0 and hub.pulls == []


def test_ineligible_does_nothing(tmp_path: Path, monkeypatch) -> None:
    el.reset_memo()
    hub = Hub([Item("agent", "a1", "coder")])
    _wire(monkeypatch, hub)
    enrol.sync_new_items(tmp_path)
    assert hub.lists == 0


def test_setting_off_means_nothing_is_pulled(cfg: Path, monkeypatch) -> None:
    hub = Hub([Item("agent", "a1", "coder")])
    _wire(monkeypatch, hub)
    monkeypatch.setattr(el, "_fetch_memberships", lambda *_: pytest.fail("ttl fresh"))
    (cfg / "config.yml").write_text("values:\n  hub:\n    org:\n      auto_enrol: false\n")
    el.reset_memo()
    enrol.run_boundary(cfg, "timer")
    assert hub.lists == 0 and hub.pulls == []


def test_setting_default_on(cfg: Path, monkeypatch) -> None:
    hub = Hub([Item("agent", "a1", "coder")])
    _wire(monkeypatch, hub)
    enrol.run_boundary(cfg, "timer")
    assert len(hub.pulls) == 1


def test_missing_slice_a_is_a_no_op(cfg: Path, monkeypatch) -> None:
    monkeypatch.setattr(enrol, "_slice_a", lambda: None)
    enrol.sync_new_items(cfg)  # no raise
    assert "t-1" not in (_record(cfg).get("seen") or {})


def test_team_plan_required_closes_the_gate(cfg: Path, monkeypatch) -> None:
    class Refused(Exception):
        status_code = 403
        code = "team_plan_required"

    def refuse(*_a, **_k):
        raise Refused()

    hub = Hub([])
    hub.list_catalogue = refuse  # type: ignore[assignment]
    _wire(monkeypatch, hub)
    monkeypatch.setattr(el, "_fetch_memberships", lambda *_: ("acct", []))
    enrol.sync_new_items(cfg)
    assert el.is_eligible(cfg) is False


def test_boundary_never_raises(cfg: Path, monkeypatch) -> None:
    monkeypatch.setattr(el, "refresh", lambda *a, **k: (_ for _ in ()).throw(RuntimeError()))
    enrol.run_boundary(cfg, "sign-in")


def test_background_boundary_runs_on_a_thread(cfg: Path, monkeypatch) -> None:
    seen: list[str] = []
    monkeypatch.setattr(enrol, "run_boundary", lambda c, b, **k: seen.append(b))
    thread = enrol.on_boundary(cfg, "daemon-start")
    assert thread is not None
    thread.join(5)
    assert seen == ["daemon-start"]


def test_radient_sign_in_fires_the_boundary_once_and_logout_clears(
    tmp_path: Path, monkeypatch
) -> None:
    """The REAL store's login completion and logout hooks."""
    from local_operator.providers.auth_store import AuthStore

    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(enrol, "on_boundary", lambda c, b, **k: calls.append((c, b, k)))
    store = AuthStore(db_path=tmp_path / "auth.db")
    try:
        store._org_hub_boundary("sign-in")
        assert calls == [(tmp_path, "sign-in", {"force": True})]
        el.record_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
        el.record_path(tmp_path).write_text("{}")
        store.upsert_credential("radient", {"api_key": "synthetic", "type": "api_key"})
        store.delete_credentials_for_provider("radient")
        assert not el.record_path(tmp_path).exists()
    finally:
        store.close()


def _record(config_dir: Path) -> dict[str, Any]:
    record = el.read_record(config_dir)
    assert record is not None
    return record
