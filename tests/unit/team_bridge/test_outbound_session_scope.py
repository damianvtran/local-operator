"""Outbound writer authority must never become inbound/ask/body authority.

Use a real persisted enrollment and config; substitute only status transport
and explicit damaged-state reads. Admission alone cannot authorize inbound
consumption, even after another writer has successfully settled as admitted.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from local_operator import settings_io
from local_operator.config import ConfigManager
from local_operator.team_bridge.authority import (
    authority_fence,
    outbound_authority_fence,
)
from local_operator.team_bridge.host import (
    admit_bridge_runtime,
    admit_soon,
    bridge_runtime_for_session,
    outbound_tool_fence,
    owner_fence_refusal,
)
from local_operator.team_bridge.store import OwnerReceipt, OwnerSnapshot, Store
from local_operator.team_bridge.tool import BridgeRuntime

OWNER = OwnerSnapshot(
    credential_id=7,
    account_id="acc_a",
    tenant_id="tnt_1",
    endpoint_id="bep_1",
    canonical_base_url="https://hub.example/v1",
    enrollment_generation=1,
)


@pytest.fixture
def enrolled(tmp_path: Path) -> ConfigManager:
    manager = ConfigManager(tmp_path)
    settings_io.write_setting(manager, settings_io.BY_KEY["team_bridge.enabled"], True)
    Store(tmp_path).commit_enrollment(
        OWNER, OwnerReceipt("bridge_consent_v1", "ref-1"), cos_session_id="sess_cos"
    )
    return manager


def _writer(manager: ConfigManager, session_id: str = "article-search-manager") -> BridgeRuntime:
    runtime = bridge_runtime_for_session(config_manager=manager, session_id=session_id)
    assert runtime is not None
    return runtime


@pytest.mark.parametrize("session_id", ["sess_cos", "sess_other", "article-search-manager"])
def test_writer_runtime_does_not_grant_inbound_authority(
    enrolled: ConfigManager, session_id: str
) -> None:
    runtime = _writer(enrolled, session_id)
    assert runtime.admission == "pending"
    assert runtime.owner == OWNER
    assert outbound_authority_fence(runtime) is None
    assert outbound_tool_fence(runtime) is None
    for fence in (authority_fence, owner_fence_refusal):
        refusal = fence(runtime)
        if session_id == "sess_cos":
            assert refusal is None
        else:
            assert refusal is not None and refusal["code"] == "owner_changed"
    assert Store(enrolled.config_dir).designated_session_for(OWNER) == "sess_cos"


def test_empty_session_never_gets_outbound_authority(enrolled: ConfigManager) -> None:
    assert bridge_runtime_for_session(config_manager=enrolled, session_id="") is None
    runtime = _writer(enrolled)
    runtime.session_id = ""
    refusal = outbound_tool_fence(runtime)
    assert refusal is not None and refusal["code"] == "owner_changed"


@pytest.mark.parametrize("designation", [None, "another_cos"])
def test_designation_changes_only_inbound_authority(
    enrolled: ConfigManager, monkeypatch: pytest.MonkeyPatch, designation: str | None
) -> None:
    runtime = _writer(enrolled, "sess_cos")
    monkeypatch.setattr(Store, "designated_session_for", lambda self, owner: designation)
    assert _writer(enrolled).session_id == "article-search-manager"
    assert outbound_tool_fence(runtime) is None
    assert owner_fence_refusal(runtime) is not None


@pytest.mark.parametrize(
    "field,value",
    [
        ("canonical_base_url", "https://other.example/v1"),
        ("account_id", "acc_other"),
        ("tenant_id", "tnt_other"),
        ("endpoint_id", "bep_other"),
        ("credential_id", 8),
        ("enrollment_generation", 2),
    ],
)
def test_outbound_preserves_every_owner_binding_check(
    enrolled: ConfigManager, field: str, value: Any
) -> None:
    runtime = _writer(enrolled)
    runtime.owner = replace(OWNER, **{field: value})
    refusal = outbound_tool_fence(runtime)
    assert refusal is not None and refusal["code"] == "owner_changed"


def test_outbound_preserves_runtime_tenant_check(enrolled: ConfigManager) -> None:
    runtime = _writer(enrolled)
    runtime.tenant_id = "tnt_other"
    refusal = outbound_tool_fence(runtime)
    assert refusal is not None and refusal["code"] == "owner_changed"


@pytest.mark.parametrize(
    "method,result,code",
    [
        ("active_owner", None, "owner_changed"),
        ("current_generation_is_live", False, "owner_changed"),
        ("owner_consent_receipt", None, "consent_revoked"),
    ],
)
def test_outbound_preserves_persisted_authority_checks(
    enrolled: ConfigManager,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    result: Any,
    code: str,
) -> None:
    runtime = _writer(enrolled)
    monkeypatch.setattr(Store, method, lambda *args: result)
    refusal = outbound_tool_fence(runtime)
    assert refusal is not None and refusal["code"] == code
    assert (
        bridge_runtime_for_session(config_manager=enrolled, session_id=runtime.session_id) is None
    )


def test_outbound_rejects_empty_committed_generation(
    enrolled: ConfigManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _writer(enrolled)
    owner = replace(OWNER, enrollment_generation=0)
    runtime.owner = owner
    monkeypatch.setattr(Store, "active_owner", lambda self: owner)
    refusal = outbound_tool_fence(runtime)
    assert refusal is not None and refusal["code"] == "owner_changed"


@pytest.mark.parametrize("values", [None, {"team_bridge": {"enabled": False}}])
def test_outbound_rechecks_config_fail_closed(
    enrolled: ConfigManager, monkeypatch: pytest.MonkeyPatch, values: Any
) -> None:
    runtime = _writer(enrolled)
    monkeypatch.setattr("local_operator.config.read_config_values", lambda config_dir: values)
    refusal = outbound_tool_fence(runtime)
    assert refusal is not None and refusal["code"] == "feature_off"


@pytest.mark.asyncio
@pytest.mark.parametrize("scheduled", [False, True])
async def test_regular_writer_admission_still_refuses_inbound(
    enrolled: ConfigManager, monkeypatch: pytest.MonkeyPatch, scheduled: bool
) -> None:
    runtime = _writer(enrolled)
    calls: list[str] = []

    async def status() -> dict[str, Any]:
        calls.append("status")
        return {
            "api_version": "1",
            "tenants": [
                {
                    "tenant_id": OWNER.tenant_id,
                    "eligible": True,
                    "endpoint": {"id": OWNER.endpoint_id, "state": "active"},
                }
            ],
        }

    monkeypatch.setattr(runtime.client, "status", status)
    if scheduled:
        admit_soon(runtime)
        assert runtime._admission_task is not None
        await runtime._admission_task
    else:
        assert await admit_bridge_runtime(runtime) == "admitted"
    assert runtime.admission == "admitted"
    assert calls == ["status"]
    assert outbound_tool_fence(runtime) is None
    assert owner_fence_refusal(runtime) is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("revoke_during_fetch", [False, True])
async def test_regular_writer_admission_cannot_outlive_consent(
    enrolled: ConfigManager, monkeypatch: pytest.MonkeyPatch, revoke_during_fetch: bool
) -> None:
    runtime = _writer(enrolled)
    calls: list[str] = []

    def revoke() -> None:
        monkeypatch.setattr(Store, "owner_consent_receipt", lambda self, owner: None)

    async def status() -> dict[str, Any]:
        calls.append("status")
        revoke()
        return {
            "api_version": "1",
            "tenants": [
                {
                    "tenant_id": OWNER.tenant_id,
                    "eligible": True,
                    "endpoint": {"id": OWNER.endpoint_id, "state": "active"},
                }
            ],
        }

    monkeypatch.setattr(runtime.client, "status", status)
    if not revoke_during_fetch:
        revoke()
    assert await admit_bridge_runtime(runtime) == "pending"
    assert calls == (["status"] if revoke_during_fetch else [])
    refusal = outbound_tool_fence(runtime)
    assert refusal is not None and refusal["code"] == "consent_revoked"
