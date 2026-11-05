"""The Radient Team gate: who may see any organization Agent Hub surface.

The matrix pins the server's entitlement predicate (membership ``active`` AND
plan ``active``/``past_due``) and every fail-closed read: a wrong presence is
the up-sell the operator forbade, so each ambiguous record must read closed.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Iterator

import pytest

from local_operator.hub_org import eligibility as el

BASE = "https://hub.example.test/v1"


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    el.reset_memo()
    monkeypatch.setattr(el, "_configured_base_url", lambda _cfg: BASE)
    yield
    el.reset_memo()


def _membership(
    status: str = "active", plan: str | None = "active", role: str = "member"
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "tenant_id": "t-1",
        "tenant_name": "Acme",
        "role": role,
        "status": status,
    }
    if plan is not None:
        entry["plan"] = {"status": plan, "seats": 5}
    return entry


@pytest.mark.parametrize(
    ("membership_status", "plan_status", "expected"),
    [
        ("active", None, False),
        ("active", "canceled", False),
        ("active", "active", True),
        ("active", "past_due", True),
        ("pending", "active", False),
        ("pending", "past_due", False),
        ("pending", None, False),
    ],
)
def test_entitlement_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, membership_status, plan_status, expected
) -> None:
    monkeypatch.setattr(
        el,
        "_fetch_memberships",
        lambda _cfg, _base: ("acct-1", [_membership(membership_status, plan_status)]),
    )
    orgs = el.refresh(tmp_path, boundary="test", force=True)
    assert bool(orgs) is expected
    assert el.is_eligible(tmp_path) is expected
    if expected:
        assert orgs == [el.OrgRef("t-1", "Acme", "member")]


def test_owner_without_plan_is_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        el, "_fetch_memberships", lambda *_: ("a", [_membership(plan=None, role="owner")])
    )
    assert el.refresh(tmp_path, boundary="test", force=True) == []
    assert el.is_eligible(tmp_path) is False


def test_signed_out_writes_closed_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(el, "_fetch_memberships", lambda *_: None)
    assert el.refresh(tmp_path, boundary="test", force=True) == []
    record = json.loads(el.record_path(tmp_path).read_text())
    assert record["orgs"] == [] and record["base_url"] == BASE
    assert oct(os.stat(el.record_path(tmp_path)).st_mode & 0o777) == "0o600"


def _seed(tmp_path: Path, **overrides) -> None:
    record = {
        "schema": el.SCHEMA,
        "account_id": "a",
        "base_url": BASE,
        "checked_at": time.time(),
        "orgs": [{"tenant_id": "t-1", "tenant_name": "Acme", "role": "admin"}],
        "seen": {},
        "next_catalogue_at": {},
    }
    record.update(overrides)
    el.record_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    el.record_path(tmp_path).write_text(json.dumps(record))


def test_no_record_is_closed(tmp_path: Path) -> None:
    assert el.eligible_orgs(tmp_path) == []


def test_fresh_record_is_open_without_network(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(el, "_fetch_memberships", lambda *_: pytest.fail("network on read"))
    _seed(tmp_path)
    assert el.eligible_orgs(tmp_path) == [el.OrgRef("t-1", "Acme", "admin")]


def test_base_mismatch_is_closed(tmp_path: Path) -> None:
    _seed(tmp_path, base_url="https://other.example.test/v1")
    assert el.is_eligible(tmp_path) is False


def test_stale_record_is_closed(tmp_path: Path) -> None:
    _seed(tmp_path, checked_at=time.time() - el.MAX_AGE_SECONDS - 5)
    assert el.is_eligible(tmp_path) is False


@pytest.mark.parametrize("body", ["{not json", "[]", json.dumps({"schema": 99, "orgs": []})])
def test_corrupt_record_is_closed(tmp_path: Path, body: str) -> None:
    el.record_path(tmp_path).parent.mkdir(parents=True)
    el.record_path(tmp_path).write_text(body)
    assert el.is_eligible(tmp_path) is False


def test_malformed_org_entry_is_closed(tmp_path: Path) -> None:
    _seed(tmp_path, orgs=[{"tenant_name": "no id"}])
    assert el.is_eligible(tmp_path) is False


def test_clear_closes(tmp_path: Path) -> None:
    _seed(tmp_path)
    assert el.is_eligible(tmp_path)
    el.clear(tmp_path)
    assert el.is_eligible(tmp_path) is False


def test_ttl_skips_network_unless_forced(tmp_path: Path, monkeypatch) -> None:
    calls: list[int] = []

    def fetch(*_):
        calls.append(1)
        return ("a", [])

    monkeypatch.setattr(el, "_fetch_memberships", fetch)
    _seed(tmp_path)
    assert el.refresh(tmp_path, boundary="timer") == [el.OrgRef("t-1", "Acme", "admin")]
    assert calls == []
    assert el.refresh(tmp_path, boundary="explicit", force=True) == []
    assert calls == [1]


def test_transient_failure_keeps_orgs_and_backs_off(tmp_path: Path, monkeypatch) -> None:
    _seed(tmp_path, checked_at=time.time() - el.REFRESH_TTL_SECONDS - 5)
    calls: list[int] = []

    def boom(*_):
        calls.append(1)
        raise ConnectionError("offline")

    monkeypatch.setattr(el, "_fetch_memberships", boom)
    assert el.refresh(tmp_path, boundary="timer") == [el.OrgRef("t-1", "Acme", "admin")]
    assert isinstance(_record(tmp_path)["last_error_at"], float)
    el.refresh(tmp_path, boundary="timer")
    assert calls == [1]  # inside the 30 min backoff


def test_team_plan_required_closes(tmp_path: Path, monkeypatch) -> None:
    _seed(tmp_path)

    def refused(*_):
        raise RefusedError(403, "team_plan_required")

    monkeypatch.setattr(el, "_fetch_memberships", refused)
    assert el.refresh(tmp_path, boundary="team-plan-required", force=True) == []
    assert el.is_eligible(tmp_path) is False


class RefusedError(Exception):
    def __init__(self, status_code: int, code: str) -> None:
        super().__init__(code)
        self.status_code = status_code
        self.code = code


def test_account_switch_resets_enrol_bookkeeping(tmp_path: Path, monkeypatch) -> None:
    _seed(tmp_path, seen={"t-1": {"agent": ["x"]}}, next_catalogue_at={"t-1": 1.0})
    monkeypatch.setattr(el, "_fetch_memberships", lambda *_: ("b", [_membership()]))
    el.refresh(tmp_path, boundary="sign-in", force=True)
    record = _record(tmp_path)
    assert record["seen"] == {} and record["next_catalogue_at"] == {}


def test_same_account_keeps_enrol_bookkeeping(tmp_path: Path, monkeypatch) -> None:
    _seed(tmp_path, seen={"t-1": {"agent": ["x"]}})
    monkeypatch.setattr(el, "_fetch_memberships", lambda *_: ("a", [_membership()]))
    el.refresh(tmp_path, boundary="sign-in", force=True)
    assert _record(tmp_path)["seen"] == {"t-1": {"agent": ["x"]}}


def test_lock_loser_skips_refresh(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(el, "_fetch_memberships", lambda *_: pytest.fail("loser fetched"))
    with el.refresh_lock(tmp_path) as held:
        assert held
        assert el.refresh(tmp_path, boundary="test", force=True) == []


def test_never_raises_on_unexpected_error(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(el, "_refresh", lambda *a, **k: (_ for _ in ()).throw(RuntimeError()))
    assert el.refresh(tmp_path, boundary="test") == []


def test_record_never_holds_a_token(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(el, "_fetch_memberships", lambda *_: ("a", [_membership()]))
    el.refresh(tmp_path, boundary="test", force=True)
    assert set(json.loads(el.record_path(tmp_path).read_text())) <= {
        "schema",
        "account_id",
        "base_url",
        "checked_at",
        "orgs",
        "seen",
        "next_catalogue_at",
        "last_error_at",
        "last_enrol",
    }


def _record(config_dir: Path) -> dict[str, Any]:
    record = el.read_record(config_dir)
    assert record is not None
    return record
