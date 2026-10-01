"""The desktop's approval surface: the badge read, the two answers, the refusals.

WHAT IS PINNED HERE: the frozen list shape (§3.5) as the route publishes it; that
``approve`` runs the SAME presence-gated signing call the CLI verb runs and the
host with no key gets the setup sentence as a 409; that a deny is ordinary and
``write-once`` (a second decision is 409); that an unknown record is 404; that
the router is bearer-gated; and that the frozen THREE route paths are exactly
what this backend serves.

Everything runs against the REAL store on a real temporary filesystem, through
the real ``errors()`` ladder and response models — no fake, because the point of
the surface is that it maps the store's own refusals verbatim. The signing half
uses a REAL file-only operator key (write-once answer is verified by
``operator.verify``, not by a double).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.network import approvals as A
from local_operator.server.routes import desktop_approvals
from local_operator.server.utils.desktop_sessions import DesktopSessions
from tests.unit.network.test_approvals_store import _device_request, _make_key

DESKTOP_TOKEN = "synthetic-desktop-token"


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated config root (used directly as the config dir) + no installed anchor."""
    from local_operator.operator import trust

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", DESKTOP_TOKEN)
    for name in list(os.environ):
        if name.startswith(("CMUX_", "LOP_")):
            monkeypatch.delenv(name, raising=False)

    def absent(uid: int | str | None = None) -> Any:
        return trust.AnchorLoad(
            anchor=None,
            path=trust.anchor_path(uid),
            root_owned=False,
            reason="pinned absent by the test",
            exists=False,
        )

    monkeypatch.setattr(trust, "load_anchor", absent)
    return tmp_path


@pytest_asyncio.fixture
async def api(root: Path):
    app = FastAPI()
    app.state.config_manager = ConfigManager(root)
    app.state.desktop_sessions = DesktopSessions(root)
    app.include_router(desktop_approvals.router)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {DESKTOP_TOKEN}"},
    ) as client:
        yield client, app


def _file_device_record(root: Path, *, with_anchor: bool = False) -> dict[str, Any]:
    _make_key(root)
    what: dict[str, Any] = {
        "install": True,
        "connect": True,
        "unattended": True,
        "grant": ["approve"],
    }
    if with_anchor:
        trio = A.local_anchor_trio(root)
        assert trio is not None
        what["anchor"] = {key: trio[key] for key in ("key_id", "spki_fp", "statement_digest")}
    return A.create_request(**_device_request(A.new_request_id(), what=what), root=root)


# ---------------------------------------------------------------------------
# The read
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_badge_reads_the_frozen_shape(api: Any) -> None:
    client, _app = api
    record = _file_device_record(Path(_app_root()))
    response = await client.get("/v1/desktop/approvals")
    assert response.status_code == 200, response.text
    body = response.json()
    rows = body["result"]["approvals"]
    assert [row["approval_id"] for row in rows] == [record["approval_id"]]
    row = rows[0]
    # The frozen keys, exactly (§3.5): the where-block rides under `device`.
    assert set(row) == {
        "approval_id",
        "state",
        "what",
        "requested_by",
        "expires_at",
        "device",
        "machine",
    }
    assert row["state"] == "requested"
    assert row["device"]["host"] == "99.79.190.164"
    assert row["machine"] is None


# ---------------------------------------------------------------------------
# The answers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_approve_signs_with_the_local_key_and_answers_the_decision_shape(api: Any) -> None:
    client, _app = api
    record = _file_device_record(Path(_app_root()), with_anchor=True)
    response = await client.post(f"/v1/desktop/approvals/{record['approval_id']}/approve")
    assert response.status_code == 200, response.text
    body = response.json()["result"]
    assert set(body) == {"approval_id", "state", "signature"}
    assert body["state"] == "approved"
    assert body["signature"]["key_id"], "the verified key_id must be published"

    # And the store agrees, which is what makes the route's answer evidence.
    stored = A.load_record(record["approval_id"])
    assert stored["state"] == "approved"
    assert stored["signature"]["key_id"] == body["signature"]["key_id"]


@pytest.mark.asyncio
async def test_approve_without_a_signing_surface_is_409_with_the_setup_sentence(
    api: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _app = api
    record = _file_device_record(Path(_app_root()), with_anchor=True)
    import local_operator.operator.sign as sign_mod

    monkeypatch.setattr(sign_mod, "load_signer", lambda **kwargs: None)
    response = await client.post(f"/v1/desktop/approvals/{record['approval_id']}/approve")
    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "approval_signing_unavailable"
    assert "set it up" in detail["message"]
    assert "`" not in detail["message"], "no terminal command may be named (§2.9)"
    assert A.load_record(record["approval_id"])["state"] == "requested"


@pytest.mark.asyncio
async def test_deny_is_ordinary_and_a_second_decision_refuses(api: Any) -> None:
    client, _app = api
    record = _file_device_record(Path(_app_root()))
    denied = await client.post(f"/v1/desktop/approvals/{record['approval_id']}/deny")
    assert denied.status_code == 200, denied.text
    assert denied.json()["result"]["state"] == "denied"
    assert denied.json()["result"]["signature"] == {"key_id": ""}

    again = await client.post(f"/v1/desktop/approvals/{record['approval_id']}/deny")
    assert again.status_code == 409, again.text
    assert again.json()["detail"]["code"] == "approval_decision_conflict"

    approved = await client.post(f"/v1/desktop/approvals/{record['approval_id']}/approve")
    assert approved.status_code == 409
    assert approved.json()["detail"]["code"] == "approval_decision_conflict"


@pytest.mark.asyncio
async def test_an_unknown_record_is_404(api: Any) -> None:
    client, _app = api
    response = await client.post("/v1/desktop/approvals/ap_missing/deny")
    assert response.status_code == 404, response.text
    assert response.json()["detail"]["code"] == "unknown_approval"


# ---------------------------------------------------------------------------
# The boundary + the frozen paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_router_is_bearer_gated(api: Any) -> None:
    _client, app = api
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost") as naked:
        response = await naked.get("/v1/desktop/approvals")
    assert response.status_code in (401, 403), response.text


def test_the_three_frozen_paths_are_the_whole_surface() -> None:
    """Exactly the frozen three routes — and ``features.approvals`` advertises them."""
    from local_operator.server.features import feature_flags

    routes = {
        (route.path, tuple(sorted(route.methods)))  # type: ignore[attr-defined]
        for route in desktop_approvals.router.routes
    }
    assert routes == {
        ("/v1/desktop/approvals", ("GET",)),
        ("/v1/desktop/approvals/{approval_id}/approve", ("POST",)),
        ("/v1/desktop/approvals/{approval_id}/deny", ("POST",)),
    }
    assert feature_flags()["approvals"] == 1


def _app_root() -> str:
    return str(os.environ["LOCAL_OPERATOR_CONFIG_DIR"])
