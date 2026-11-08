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
replaced is only the wire — the relay's control socket (``control_request``), for the
report and the grant ask — and the grant's adopt CAS, because what is under test is what
the BORROWER does with the owner's answer. The cells over the real owner broker and
the real failover driver live in test_borrower_refresh_outcome_driver.py.
"""

from __future__ import annotations

import asyncio
import dataclasses
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


def _grant(token: str, credential_id: int = 7) -> Grant:
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
            credential_id=credential_id,
        ),
        served_by=OWNER,
        grant_id=f"g_{credential_id}",
    )


class _Rig:
    """A borrower wired to a scripted owner with logins ``rows`` (row id -> bearer).

    The borrower's REAL ``request_grant_sync`` builds every grant ask, so the
    ``exclude_credential_ids`` a hold adds is what the scripted owner reads; only
    the adopt step (membership/policy CAS, which needs a joined network) is stood
    in for, because what is under test is what the borrower ASKS and SAYS.
    """

    def __init__(
        self,
        root: Path,
        monkeypatch: pytest.MonkeyPatch,
        reply: dict[str, Any],
        rows: dict[int, str] | None = None,
    ):
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
        #: The owner's logins, in its selection order. ``tombstoned`` are the rows
        #: its cascade refuses on its own (a dead grant).
        self.rows: dict[int, str] = dict(rows or {7: DEAD_BEARER})
        self.tombstoned: set[int] = set()
        self.reports: list[dict[str, Any]] = []
        self.asks: list[list[int]] = []

        def control(_record: Any, op: str, **fields: Any) -> dict[str, Any]:
            if op == "credential_grant":
                return {"detail": self._lend(list(fields.get("exclude_credential_ids") or []))}
            self.reports.append({"op": op, **fields})
            return {"detail": dict(self.reply)}

        monkeypatch.setattr(
            "local_operator.network.store.find_own_relay", lambda root=None: object()
        )
        monkeypatch.setattr("local_operator.network.relay.control_request", control)

        def adopt(grant: Grant, key: str, session_id: str, *_a: Any, **_k: Any) -> Grant:
            # The cache write is the part of adopting the report path reads: the
            # lent row's id travels with the report from the cached grant.
            self.client.grants.put(key, session_id, grant)
            return grant

        monkeypatch.setattr(self.client, "_adopt_online", adopt)
        self.store = mesh_store.MeshAwareAuthStore(
            AuthStore(config_dir=root), mesh=self.client, config_dir=root
        )

    def _lend(self, excluded: list[int]) -> dict[str, Any]:
        self.asks.append(excluded)
        for row_id, bearer in self.rows.items():
            if row_id not in excluded and row_id not in self.tombstoned:
                return _grant(bearer, row_id).to_detail()
        if excluded:
            return BrokerError(
                code="quota_blocked", key="openai", owner_device=OWNER, reason="holder_excluded"
            ).to_detail()
        return BrokerError(
            code="no_local_credential",
            key="openai",
            owner_device=OWNER,
            message=f"{OWNER_NAME} holds no usable credential for 'openai'",
        ).to_detail()

    def borrow(self) -> str | None:
        return asyncio.run(self.store.get_api_key("openai", "sess-1"))

    def provider_refuses(self, bearer: str) -> None:
        """The provider's 401 on ``bearer``, as the failover driver hands it over."""
        assert self.store.rotate_sibling("openai", "sess-1", _Unauthorized(), bearer) is False

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


def test_a_dead_grant_with_nothing_else_to_lend_says_sign_in_on_the_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The turn's error names the owner and the command to run there, once each."""
    from local_operator.providers.failover import _borrowed_refresh_refusal

    rig = _Rig(
        tmp_path,
        monkeypatch,
        _owner_reply("refresh_failed", code="grant_invalid", remedy="/login evil"),
    )
    try:
        assert rig.borrow() == DEAD_BEARER
        rig.tombstoned.add(7)  # the owner's IdP refused the grant; its cascade skips it
        rig.provider_refuses(DEAD_BEARER)
        assert rig.borrow() is None
        expected = (
            f"The 'openai' sign-in on {OWNER_NAME} has expired. "
            f"Run 'lop login openai' on {OWNER_NAME}, then try again."
        )
        assert rig.store.no_credential_reason("openai") == expected
        # The text the FIRST failing turn ends on (#3616 QA Q2), not the bare 401.
        refused = _borrowed_refresh_refusal(rig.store, "openai")
        assert refused is not None and refused.message == expected
        assert "evil" not in expected
        # A ``grant_invalid`` row is NOT hidden from the ask: the owner's own cascade
        # refuses it, and a re-sign-in there must serve on the very next ask (Q4).
        assert rig.asks[-1] == []
        rig.tombstoned.clear()
        rig.rows = {7: NEW_BEARER}
        assert rig.borrow() == NEW_BEARER
        assert rig.store.no_credential_reason("openai") is None
        # The owner lent login 7 again: the stale ``grant_invalid`` hold is gone, so
        # a 401 on the NEW bearer is reported rather than answered from it.
        assert rig.client.refresh_hold("openai", 7) is None
    finally:
        rig.close()


def test_a_dead_login_never_blocks_a_working_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#3616 M1/Q1: login 7 cannot be refreshed, login 8 works — 8 serves, no advice."""
    from local_operator.providers.failover import _borrowed_refresh_refusal

    rig = _Rig(
        tmp_path,
        monkeypatch,
        _owner_reply("refresh_failed", code="refresh_failed", retry_after_ms=30_000),
        rows={7: DEAD_BEARER, 8: NEW_BEARER},
    )
    try:
        assert rig.borrow() == DEAD_BEARER
        rig.provider_refuses(DEAD_BEARER)
        # The owner would hand the same dead bearer back for 7, so 7 is hidden...
        assert rig.borrow() == NEW_BEARER
        assert rig.asks[-1] == [7]
        # ...and nothing tells the operator to sign in while 8 serves.
        assert rig.store.no_credential_reason("openai") is None
        assert _borrowed_refresh_refusal(rig.store, "openai") is None
        # A 401 on the SIBLING is still reported: the hold is login 7's alone.
        rig.provider_refuses(NEW_BEARER)
        assert len(rig.reports) == 2, rig.reports
        assert [r.get("credential_id") for r in rig.reports] == [7, 8]
    finally:
        rig.close()


def test_temporary_failure_leads_with_the_wait_and_retry_after_is_honoured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """IdP down: wait first, sign-in only if it persists; no repeat report inside the wait."""
    clock = [1000.0]
    monkeypatch.setattr(client_mod.time, "monotonic", lambda: clock[0])
    rig = _Rig(
        tmp_path,
        monkeypatch,
        _owner_reply("refresh_failed", code="refresh_failed", retry_after_ms=30_000),
    )
    try:
        assert rig.borrow() == DEAD_BEARER
        rig.provider_refuses(DEAD_BEARER)
        assert len(rig.reports) == 1
        assert rig.borrow() is None
        assert rig.store.no_credential_reason("openai") == (
            f"{OWNER_NAME} could not refresh the 'openai' sign-in just now. Try again in "
            f"about 30 s; if it keeps failing, run 'lop login openai' on {OWNER_NAME}."
        )

        # 20 s later a repeat 401 on that login is answered from the hold, not sent,
        # and the countdown is re-rendered.
        clock[0] += 20.0
        rig.client.grants.put("openai", "sess-1", _grant(DEAD_BEARER))
        held = rig.client.report_sync("openai", kind="unauthorized", session_id="sess-1")
        assert isinstance(held, BrokerError) and held.code == "refresh_failed"
        assert len(rig.reports) == 1, "a report inside retry_after reached the owner"
        assert rig.borrow() is None
        assert "about 10 s" in str(rig.store.no_credential_reason("openai"))

        # Past retry_after: login 7 is asked for again, and the owner's new token serves.
        clock[0] += 11.0
        rig.rows = {7: NEW_BEARER}
        assert rig.borrow() == NEW_BEARER
        assert rig.asks[-1] == []
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
        assert rig.borrow() == DEAD_BEARER
        rig.provider_refuses(DEAD_BEARER)
        assert isinstance(rig.client.refresh_hold("openai"), BrokerError)
        fence[0] = frozenset({("session", (2, 1), "")})  # the owner signed in again
        assert rig.client.refresh_hold("openai") is None
    finally:
        rig.close()


def test_refreshed_clears_an_existing_hold_and_the_retry_gets_the_new_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#3616 m3: a hold is set first, then the owner's ``refreshed`` clears it."""
    rig = _Rig(tmp_path, monkeypatch, _owner_reply("unchanged"))
    try:
        assert rig.borrow() == DEAD_BEARER
        rig.provider_refuses(DEAD_BEARER)
        assert rig.client.refresh_hold("openai", 7) is not None

        # The hold's time passes, the next 401 reaches the owner, which now refreshes.
        hold = rig.client._refresh_holds[("openai", 7)]  # noqa: SLF001
        expired = dataclasses.replace(hold, until=0.0)
        rig.client._refresh_holds[("openai", 7)] = expired  # noqa: SLF001
        rig.reply = _owner_reply("refreshed")
        rig.client.grants.put("openai", "sess-1", _grant(DEAD_BEARER))
        assert rig.client.report_sync("openai", kind="unauthorized", session_id="sess-1") is None
        assert rig.client.refresh_hold("openai") is None

        # A hold that is still LIVE is cleared by ``refreshed`` too.
        rig.reply = _owner_reply("unchanged")
        rig.client.grants.put("openai", "sess-1", _grant(DEAD_BEARER))
        rig.client.report_sync("openai", kind="unauthorized", session_id="sess-1")
        assert rig.client.refresh_hold("openai", 7) is not None
        rig.client._note_refresh_outcome("openai", _owner_reply("refreshed"), 7)  # noqa: SLF001
        assert rig.client.refresh_hold("openai") is None

        rig.rows = {7: NEW_BEARER}
        assert rig.borrow() == NEW_BEARER
        assert rig.store.no_credential_reason("openai") is None
    finally:
        rig.close()


@pytest.mark.parametrize(
    "stated", [10**12, "1e400", float("inf"), -5, True, "x", None, 0, [1], {"a": 1}]
)
def test_a_hostile_retry_after_is_capped_or_defaulted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stated: Any
) -> None:
    """#3616 m2: whatever the owner sends, the hold is positive and at most 300 s."""
    clock = [1000.0]
    monkeypatch.setattr(client_mod.time, "monotonic", lambda: clock[0])
    rig = _Rig(
        tmp_path,
        monkeypatch,
        _owner_reply("refresh_failed", code="refresh_failed", retry_after_ms=stated),
    )
    try:
        assert rig.borrow() == DEAD_BEARER
        rig.provider_refuses(DEAD_BEARER)
        hold = rig.client._refresh_holds[("openai", 7)]  # noqa: SLF001
        assert 0 < hold.until - clock[0] <= 300.0, (stated, hold.until - clock[0])
    finally:
        rig.close()


def test_a_hostile_owner_name_stays_one_safe_line() -> None:
    """#3616 m1/Q5: an owner-authored name cannot repaint or forge a line."""
    from local_operator.network.credentials.messages import render_owner_refresh_outcome

    hostile = "evil\x1b[31mRED\x1b[0m\nSecond line\r\u202eright-to-left\x07" + "x" * 200
    for code in ("grant_invalid", "refresh_failed", "unchanged"):
        text = render_owner_refresh_outcome("openai", hostile, code=code, retry_after_ms=30_000)
        assert "\n" not in text and "\r" not in text and "\x1b" not in text, repr(text)
        assert "\x07" not in text and "\u202e" not in text, repr(text)
        assert text.isprintable(), repr(text)
        assert "x" * 100 not in text, "the name was not bounded"


def test_the_sentences_are_short_and_never_echo_the_peers_remedy() -> None:
    """D2's measured targets, and the command composed here from the key (D3)."""
    from local_operator.network.credentials.messages import render_owner_refresh_outcome

    dead = render_owner_refresh_outcome("openai", "studio-mac-mini", code="grant_invalid")
    assert dead == (
        "The 'openai' sign-in on studio-mac-mini has expired. "
        "Run 'lop login openai' on studio-mac-mini, then try again."
    )
    assert len(dead) <= 133, len(dead)
    longest = render_owner_refresh_outcome(
        "alibaba-token-plan", "n" * 46, code="refresh_failed", retry_after_ms=300_000
    )
    assert len(longest) <= 275, (len(longest), longest)
    assert longest.count("n" * 46) == 2, longest
    mcp = render_owner_refresh_outcome("mcp:https://x.test/mcp", "box", code="grant_invalid")
    assert "'/mcp login <name>'" in mcp and "https://" not in mcp.split("'")[-2], mcp


def test_a_held_login_never_relabels_an_unrelated_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#3616 review m2: only a refresh refusal is restated by the failover hook."""
    from local_operator.providers.failover import _borrowed_refresh_refusal

    rig = _Rig(tmp_path, monkeypatch, _owner_reply("unchanged"), rows={7: DEAD_BEARER})
    try:
        assert rig.borrow() == DEAD_BEARER
        rig.provider_refuses(DEAD_BEARER)
        assert rig.client.refresh_hold("openai", 7) is not None
        rig.store._refusals["openai"] = BrokerError(  # noqa: SLF001
            code="owner_offline", key="openai", message="studio-mac is offline"
        )
        assert rig.store.borrowed_refresh_refusal("openai") is None
        assert _borrowed_refresh_refusal(rig.store, "openai") is None
    finally:
        rig.close()


def test_a_resting_siblings_return_time_wins_over_the_hold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#3616 review m1: login 7 held, sibling 8 rate-limited — the 429's wait is shown."""
    rig = _Rig(
        tmp_path,
        monkeypatch,
        _owner_reply("refresh_failed", code="refresh_failed", retry_after_ms=300_000),
        rows={7: DEAD_BEARER, 8: NEW_BEARER},
    )
    try:
        assert rig.borrow() == DEAD_BEARER
        rig.provider_refuses(DEAD_BEARER)
        assert rig.borrow() == NEW_BEARER
        # Sibling 8 answers a usage limit: the real cooldown path rests it.
        rig.client.report_sync("openai", kind="quota", session_id="sess-1", retry_after_ms=120_000)
        assert rig.borrow() is None
        assert sorted(rig.asks[-1]) == [7, 8]
        reason = str(rig.store.no_credential_reason("openai"))
        assert "rate-limited" in reason and "could not refresh" not in reason, reason
    finally:
        rig.close()


def test_an_unnamed_owner_does_not_open_the_sentence_in_lowercase() -> None:
    """#3616 design D7 / review n1."""
    from local_operator.network.credentials.messages import render_owner_refresh_outcome

    for code in ("refresh_failed", "unchanged"):
        text = render_owner_refresh_outcome("openai", "", code=code, retry_after_ms=30_000)
        assert text.startswith("The owner device "), text
    named = render_owner_refresh_outcome("openai", "damians-mbp", code="refresh_failed")
    assert named.startswith("damians-mbp could not"), named
