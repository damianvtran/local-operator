"""The gate: every conjunct must hold, any failure reads closed; refresh classifies route 1."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from local_operator.hub_org import eligibility as hub
from local_operator.vault_agent import eligibility

from .conftest import ENDPOINT, TENANT, Transport, make_client

BASE = "https://api.example.test/v1"
OWNER = SimpleNamespace(tenant_id=TENANT, endpoint_id=ENDPOINT)


@pytest.fixture
def world(config_dir, monkeypatch):
    """Hub record open for TENANT, an owner on ENDPOINT, an open Vault record."""

    state = {
        "orgs": [hub.OrgRef(tenant_id=TENANT, tenant_name="Acme", role="member")],
        "owner": OWNER,
    }
    monkeypatch.setattr(hub, "eligible_orgs", lambda _root: list(state["orgs"]))
    monkeypatch.setattr(eligibility, "_owner", lambda _root: state["owner"])
    monkeypatch.setattr(eligibility, "_configured_base_url", lambda _root: BASE)
    eligibility._write_json(
        eligibility.vault_dir(config_dir) / eligibility.ELIGIBILITY_NAME,
        {
            "base_url": BASE,
            "checked_at": time.time(),
            "tenant_id": TENANT,
            "endpoint_id": ENDPOINT,
            "delegations_available": True,
            "eligible": True,
            "tenant_state": "enabled",
            "principal_state": "approved",
        },
    )
    return state


def _rewrite(config_dir, **changes):
    path = eligibility.vault_dir(config_dir) / eligibility.ELIGIBILITY_NAME
    data = eligibility._read_json(path)
    assert data is not None
    data.update(changes)
    eligibility._write_json(path, data)


def test_open_when_every_conjunct_holds(config_dir, world) -> None:
    gate = eligibility.gate(config_dir)
    assert gate == eligibility.Gate(tenant_id=TENANT, endpoint_id=ENDPOINT, tenant_name="Acme")


def test_record_is_private(config_dir, world) -> None:
    path = eligibility.vault_dir(config_dir) / eligibility.ELIGIBILITY_NAME
    assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize(
    "change",
    [
        {"delegations_available": False},  # the dark gate: production today
        {"eligible": False},
        {"tenant_state": "frozen"},
        {"principal_state": "pending"},
        {"tenant_id": "tnt_other"},
        {"endpoint_id": "bep_other"},
        {"base_url": "https://elsewhere.test/v1"},
        {"checked_at": time.time() - eligibility.MAX_AGE_SECONDS - 5},
    ],
)
def test_closed_when_the_vault_record_disagrees(config_dir, world, change) -> None:
    _rewrite(config_dir, **change)
    assert eligibility.gate(config_dir) is None


def test_closed_without_the_team_record(config_dir, world) -> None:
    world["orgs"] = []
    assert eligibility.gate(config_dir) is None


def test_closed_without_a_team_bridge_owner(config_dir, world) -> None:
    world["owner"] = None
    assert eligibility.gate(config_dir) is None


def test_closed_when_the_owner_tenant_is_not_a_team_org(config_dir, world) -> None:
    world["owner"] = SimpleNamespace(tenant_id="tnt_personal", endpoint_id=ENDPOINT)
    assert eligibility.gate(config_dir) is None


def test_closed_on_a_corrupt_record(config_dir, world) -> None:
    (eligibility.vault_dir(config_dir) / eligibility.ELIGIBILITY_NAME).write_text("{")
    assert eligibility.gate(config_dir) is None


def test_closed_when_the_gate_read_raises(config_dir, world, monkeypatch) -> None:
    def boom(_root):
        raise RuntimeError("sqlite is busy")

    monkeypatch.setattr(eligibility, "_owner", boom)
    assert eligibility.gate(config_dir) is None


# -- refresh --------------------------------------------------------------------


def _status_body(**org):
    entry = {
        "tenant_id": TENANT,
        "eligible": True,
        "tenant_state": "enabled",
        "vault_principal": {"state": "approved"},
    }
    entry.update(org)
    return {
        "api_version": "radient-vault-api-v1",
        "delegations_available": True,
        "organizations": [entry],
    }


@pytest.fixture
def refreshing(config_dir, world, monkeypatch):
    transport = Transport()
    person = SimpleNamespace()
    monkeypatch.setattr(hub, "person_client", lambda _root, _base=None: (person, "acct"))
    monkeypatch.setattr(
        "local_operator.vault_agent.client.VaultClient.for_person",
        classmethod(lambda cls, _p: make_client(transport)),
    )
    (eligibility.vault_dir(config_dir) / eligibility.ELIGIBILITY_NAME).unlink()
    return transport


def test_refresh_opens_on_an_approved_member_with_delegations(config_dir, refreshing) -> None:
    refreshing.answers.append((200, _status_body()))
    observed = eligibility.refresh(config_dir, boundary="explicit", force=True)
    assert observed.state == "open"
    assert refreshing.calls[0]["params"] == {"tenant_id": TENANT}
    assert eligibility.gate(config_dir) is not None


@pytest.mark.parametrize(
    ("body", "state"),
    [
        ({**_status_body(), "delegations_available": False}, "unavailable"),
        (_status_body(eligible=False, tenant_state="not_available"), "not_eligible"),
        (_status_body(eligible=False, tenant_state="frozen"), "frozen"),
        (_status_body(vault_principal={"state": "pending"}), "not_approved"),
    ],
)
def test_refresh_keeps_the_gate_closed_and_says_why(config_dir, refreshing, body, state) -> None:
    refreshing.answers.append((200, body))
    assert eligibility.refresh(config_dir, boundary="explicit", force=True).state == state
    assert eligibility.gate(config_dir) is None


def test_refresh_reads_an_unregistered_route_as_the_dark_gate(config_dir, refreshing) -> None:
    refreshing.answers.append((404, {"code": "not_found"}))
    assert eligibility.refresh(config_dir, boundary="explicit", force=True).state == "unavailable"
    assert eligibility.gate(config_dir) is None


def test_refresh_signed_out_writes_closed_without_a_call(
    config_dir, refreshing, monkeypatch
) -> None:
    monkeypatch.setattr(hub, "person_client", lambda _root, _base=None: None)
    assert eligibility.refresh(config_dir, boundary="explicit", force=True).state == "signed_out"
    assert refreshing.calls == []
    assert eligibility.gate(config_dir) is None


def test_refresh_without_force_honours_the_ttl(config_dir, refreshing) -> None:
    refreshing.answers.append((200, _status_body()))
    eligibility.refresh(config_dir, boundary="explicit", force=True)
    assert eligibility.refresh(config_dir, boundary="timer").state == "open"
    assert len(refreshing.calls) == 1


@pytest.mark.parametrize(
    "orgs", [[], [hub.OrgRef(tenant_id=TENANT, tenant_name="Acme", role="member")]]
)
def test_hub_boundaries_refresh_the_vault_gate_only_for_a_team_account(
    config_dir, monkeypatch, orgs
) -> None:
    from local_operator.hub_org import enrol

    calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(hub, "refresh", lambda _root, boundary, force=False: orgs)
    monkeypatch.setattr(enrol, "auto_enrol_enabled", lambda _root: False)
    monkeypatch.setattr(
        eligibility, "refresh", lambda _root, boundary, force=False: calls.append((boundary, force))
    )
    enrol.run_boundary(config_dir, "sign-in", force=True)
    assert calls == ([("sign-in", True)] if orgs else [])
