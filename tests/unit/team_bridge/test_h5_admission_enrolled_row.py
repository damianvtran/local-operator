"""H5 remediation — admission reads the ENROLLED tenant row directly (S-4,
H5-3). The default selection (``client.select_tenant``: first eligible row) is
the onboarding choice and must never decide an existing binding's admission —
with another eligible org earlier in the list the old path misread a valid
enrolled row as ``owner_changed`` or lost the bound read-only posture.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from local_operator.team_bridge.host import _admission_from_payload

pytestmark = pytest.mark.asyncio


def _owner() -> SimpleNamespace:
    return SimpleNamespace(tenant_id="ten_B", endpoint_id="ep_B")


def _row(tenant_id: str, *, eligible: bool = True, state: str = "active", endpoint: str = "ep") -> dict:
    return {
        "tenant_id": tenant_id,
        "eligible": eligible,
        "endpoint": {"id": f"{endpoint}", "state": state} if endpoint else None,
    }


def _payload(*rows: dict, api_version: object = "1") -> dict:
    return {"api_version": api_version, "tenants": list(rows)}


async def test_enrolled_row_second_in_payload_still_admits() -> None:
    payload = _payload(
        _row("ten_A", endpoint="ep_A"),  # an OTHER eligible org, earlier
        _row("ten_B", endpoint="ep_B"),
    )
    assert _admission_from_payload(payload, _owner()) == ("admitted", "")


async def test_enrolled_row_read_only_posture_is_read_not_owner_changed() -> None:
    payload = _payload(
        _row("ten_A", endpoint="ep_A"),
        _row("ten_B", eligible=False, endpoint="ep_B"),  # F: retained, read-only
    )
    assert _admission_from_payload(payload, _owner()) == ("disabled", "read_only")


async def test_enrolled_row_paused_is_paused() -> None:
    payload = _payload(_row("ten_B", state="paused", endpoint="ep_B"))
    assert _admission_from_payload(payload, _owner()) == ("paused", "")


async def test_missing_enrolled_row_is_typed_not_enabled() -> None:
    payload = _payload(_row("ten_A", endpoint="ep_A"))
    assert _admission_from_payload(payload, _owner()) == ("disabled", "not_enabled")


async def test_duplicate_enrolled_rows_are_ambiguous() -> None:
    payload = _payload(_row("ten_B", endpoint="ep_B"), _row("ten_B", endpoint="ep_B"))
    assert _admission_from_payload(payload, _owner()) == ("disabled", "ambiguous_tenant_row")


async def test_enrolled_row_wrong_endpoint_is_owner_changed() -> None:
    payload = _payload(_row("ten_B", endpoint="ep_OTHER"))
    assert _admission_from_payload(payload, _owner()) == ("disabled", "owner_changed")


async def test_unsupported_api_version_disables_without_row_reads() -> None:
    payload = _payload(_row("ten_B", endpoint="ep_B"), api_version="999")
    assert _admission_from_payload(payload, _owner()) == ("disabled", "update_required")


async def test_endpoint_absent_is_not_enabled() -> None:
    payload = _payload(_row("ten_B", endpoint=""))
    assert _admission_from_payload(payload, _owner()) == ("disabled", "not_enabled")
