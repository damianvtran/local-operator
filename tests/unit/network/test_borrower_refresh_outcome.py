"""The borrower READS the owner's answer to its 401 report.

Operator ask (2026-11-08): "also check how auth failures on remote are handled, if
something expires on a remote device is it able to refresh from local". #3560 made
the owner force a real refresh on a borrower's 401 report and answer what it did
(``refreshed`` / ``unchanged`` / ``refresh_failed`` with a ``code`` and maybe a
``retry_after_ms``). The borrower discarded that answer, so when the owner could not
refresh, the person on the borrowing device saw the provider's 401 again on every
retry, with no remedy.

What is real here: the borrower's ``MeshCredentialClient``, the mesh-aware store
over a real ``AuthStore``, the failover driver's ``_no_credential_error``. What is
replaced is only the wire — the relay's control socket (``control_request``) for
the report, and ``grant_async`` for the borrow — because what is under test is what
the BORROWER does with the owner's answer.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.network.credentials import client as client_mod
from local_operator.network.credentials import placement as placement_mod
from local_operator.network.credentials import store as mesh_store
from local_operator.network.credentials.types import BrokerError, CredentialRef, Grant

OWNER = "d_00000000000000000000000000000001"
OWNER_NAME = "studio-mac"
NETWORK = "n_refresh_outcome_test"
#: Assembled, not spelled: not a secret, and a token-shaped literal is the shape
#: every reviewer has to stop and check.
DEAD_BEARER = "-".join(("dead", "bearer"))
NEW_BEARER = "-".join(("new", "bearer"))


class _Unauthorized(Exception):
    """A provider 401 — classified ``unauthorized`` by ``report_kind_for``."""


def _grant(token: str) -> Grant:
    return Grant(
        access_token=token,
        kind="bearer",
        token_expires_at_ms=int((time.time() + 3600) * 1000),
        grant_expires_at_ms=int((time.time() + 900) * 1000),
        credential_ref=CredentialRef(
            owner_device=OWNER,
            owner_device_name=OWNER_NAME,
            provider="openai",
            kind="oauth",
            credential_id=7,
        ),
        served_by=OWNER,
        grant_id="g_test",
    )


class _Rig:
    """A borrower wired to a scripted owner: report answers and grants are queued."""

    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch, reply: dict[str, Any]):
        from local_operator.network.identity import mint
        from local_operator.providers.auth_store import AuthStore

        monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
        identity = mint(root, name="borrower")
        document = placement_mod.PlacementDocument(NETWORK, root=root, written_by=OWNER)
        document.declare(
            "openai", owner_device=OWNER, owner_device_name=OWNER_NAME, provider="openai", by=OWNER
        )
        document.grant("openai", identity.device_id, scope="session", by=OWNER)
        document.save()
        self.client = client_mod.MeshCredentialClient(
            root=root, self_device=identity.device_id, network_id=NETWORK, placement=document
        )
        self.reply = reply
        self.reports: list[dict[str, Any]] = []
        self.dials: list[str] = []
        self.next_grants: list[Grant] = [_grant(DEAD_BEARER)]

        def control(_record: Any, op: str, **fields: Any) -> dict[str, Any]:
            self.reports.append({"op": op, **fields})
            return {"detail": dict(self.reply)}

        async def grant_async(key: str, **_: Any) -> Grant:
            self.dials.append(key)
            return self.next_grants[0] if len(self.next_grants) == 1 else self.next_grants.pop(0)

        monkeypatch.setattr(
            "local_operator.network.store.find_own_relay", lambda root=None: object()
        )
        monkeypatch.setattr("local_operator.network.relay.control_request", control)
        monkeypatch.setattr(self.client, "grant_async", grant_async)
        self.store = mesh_store.MeshAwareAuthStore(
            AuthStore(config_dir=root), mesh=self.client, config_dir=root
        )

    def turn_fails_with_401(self) -> None:
        """One borrowed request refused by the provider, as the driver sees it."""
        assert asyncio.run(self.store.get_api_key("openai", "sess-1")) == DEAD_BEARER
        assert self.store.rotate_sibling("openai", "sess-1", _Unauthorized(), DEAD_BEARER) is False

    def close(self) -> None:
        self.store.close()


@pytest.fixture(autouse=True)
def _a_401_is_unauthorized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mesh_store, "report_kind_for", lambda _error: "unauthorized")


def _owner_reply(action: str, **fields: Any) -> dict[str, Any]:
    return {
        "kind": "ack",
        "key": "openai",
        "owner_device": OWNER,
        "owner_device_name": OWNER_NAME,
        "action": action,
        **fields,
    }


def test_refresh_failed_is_surfaced_with_the_remedy_to_run_on_the_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The turn's error names the owner, the command to run THERE and the retry."""
    from local_operator.providers.failover import _no_credential_error

    rig = _Rig(
        tmp_path,
        monkeypatch,
        _owner_reply("refresh_failed", code="grant_invalid", remedy="/login openai"),
    )
    try:
        rig.turn_fails_with_401()
        # The driver's re-resolve after the rotation: answered by the hold, no dial.
        rig.dials.clear()
        assert asyncio.run(rig.store.get_api_key("openai", "sess-1")) is None
        assert rig.dials == [], "the borrower re-borrowed the bearer the owner cannot refresh"
        text = str(_no_credential_error(rig.store, "openai"))
        assert OWNER_NAME in text, text
        assert "'/login openai' there" in text, text
        assert "has expired" in text, text
        # No ``retry_after`` from the owner: the hold is the owner's report window.
        assert "again in 5 min" in text, text
        assert DEAD_BEARER not in text
    finally:
        rig.close()


def test_retry_after_is_honoured_for_reports_and_borrows_then_released(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inside ``retry_after`` nothing is sent to the owner; after it, both resume."""
    from local_operator.providers.failover import _no_credential_error

    clock = [1000.0]
    monkeypatch.setattr(client_mod.time, "monotonic", lambda: clock[0])
    rig = _Rig(
        tmp_path,
        monkeypatch,
        _owner_reply("refresh_failed", code="refresh_failed", retry_after_ms=30_000),
    )
    try:
        rig.turn_fails_with_401()
        assert len(rig.reports) == 1
        text = str(_no_credential_error(rig.store, "openai"))
        assert "could not refresh it just now" in text and "again in 30 s" in text, text

        # 20 s later: another 401 report is answered from the hold, not dialled, and
        # the borrow is refused locally with the countdown re-rendered.
        clock[0] += 20.0
        rig.client.grants.put("openai", "sess-1", _grant(DEAD_BEARER))
        held = rig.client.report_sync("openai", kind="unauthorized", session_id="sess-1")
        assert isinstance(held, BrokerError) and held.code == "refresh_failed"
        assert len(rig.reports) == 1, "a report inside retry_after reached the owner"
        rig.dials.clear()
        assert asyncio.run(rig.store.get_api_key("openai", "sess-1")) is None
        assert rig.dials == []
        assert "again in 10 s" in str(_no_credential_error(rig.store, "openai"))

        # Past retry_after: the borrow dials again and the owner's new token serves.
        clock[0] += 11.0
        rig.next_grants = [_grant(NEW_BEARER)]
        assert asyncio.run(rig.store.get_api_key("openai", "sess-1")) == NEW_BEARER
        assert rig.dials == ["openai"]
    finally:
        rig.close()


def test_a_lifecycle_change_on_the_owner_ends_the_hold_early(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The operator running the remedy on the owner must not wait out the timer."""
    fence = [frozenset({("session", (1, 1), "")})]
    monkeypatch.setattr(
        client_mod.MeshCredentialClient, "_lifecycle_fence", lambda self, key: fence[0]
    )
    rig = _Rig(tmp_path, monkeypatch, _owner_reply("unchanged"))
    try:
        rig.turn_fails_with_401()
        assert isinstance(rig.client.refresh_hold("openai"), BrokerError)
        fence[0] = frozenset({("session", (2, 1), "")})  # the owner signed in again
        assert rig.client.refresh_hold("openai") is None
    finally:
        rig.close()


def test_refreshed_stays_silent_and_the_retry_is_served_the_new_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The owner fixed it: no refusal, no hold, and the re-resolve gets the new bearer."""
    rig = _Rig(tmp_path, monkeypatch, _owner_reply("refreshed"))
    try:
        rig.turn_fails_with_401()
        assert rig.store.no_credential_reason("openai") is None
        assert rig.client.refresh_hold("openai") is None
        rig.next_grants = [_grant(NEW_BEARER)]
        rig.dials.clear()
        assert asyncio.run(rig.store.get_api_key("openai", "sess-1")) == NEW_BEARER
        assert rig.dials == ["openai"]
    finally:
        rig.close()


def test_the_sentence_never_echoes_the_peers_remedy_and_names_the_owner_command() -> None:
    """The command is composed here from the key; a peer cannot supply words."""
    from local_operator.network.credentials.messages import render_owner_refresh_outcome

    text = render_owner_refresh_outcome("openai", OWNER_NAME, code="unchanged", retry_after_ms=0)
    assert text == (
        f"The provider refused the 'openai' sign-in borrowed from {OWNER_NAME}, and "
        f"{OWNER_NAME} could not get a new token for it. Sign in again on {OWNER_NAME}: "
        "run '/login openai' there."
    )
