"""A BUSY lender is bridged by the held grant exactly like an absent one.

THE INCIDENT (2026-11-17 04:05-04:45Z, the owner Mac's own audit): 34
``credential.grant_refused`` rows, all ``owner_busy``, provider github, for one
borrower, in bursts matching that node's tool reporting "set GH_TOKEN". The Mac was
swapping and its GitHub source read overran its deadline. The borrower held an
admitted grant whose 24 h hold had hours left, but only ``owner_offline`` /
``owner_unreachable`` were bridged, so a slow lender ended access a sleeping one
would not have.

THE OPERATOR'S REQUEST (verbatim): "check how auth failures on remote are handled,
if something expires on a remote device is it able to refresh from local ... so that
remote sessions can use those to refresh on their own while valid".

What is pinned here, all on the existing loopback owner rig (a real broker whose gh
source read is slowed past its deadline, synthetic ids and tokens only):

* ``owner_busy`` after the lease expired serves the held grant through the tool door
  (``lent_github_env``), writes ``held_served`` and never a deny;
* the owner relay's slot-full refusal (``rate_limited``) does the same, and its
  "did not finish within N s" timer reads ``owner_unreachable`` — no drop, no deny;
* the invariants: never past ``hold_end``; never a revoked, panicked, policy-denied,
  stale-generation or membership-lost grant; a forced refresh is never answered from
  memory;
* the borrower writes its own coalesced ``credential.borrow_failed`` row.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any

import pytest

from local_operator.network.credentials import github as github_mod
from tests.unit.network.test_github_owner_source_isolation import (  # noqa: F401 — fixtures
    app_keypair,
    borrower,
    devices,
    env,
    github_api,
)

SESSION = "busy-hold-session"


def _audit_rows(root: Any, event: str) -> list[dict[str, Any]]:
    from local_operator.network.audit import audit_path

    path = audit_path(root)
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return [row for row in rows if row.get("event") == event]


def _deny_code(client: Any) -> str:
    """The durable held-policy code recorded for (owner, github), "" when none."""
    policy = client._policy()
    with policy.transaction() as document:
        return str(policy.code(document, client.owner_of("github"), "github"))


def _tool_token(client: Any) -> str:
    """The tool's redaction-gated door, not the inner broker call."""
    tool_env, _ = github_mod.lent_github_env(
        session_id=SESSION, register=lambda _t: None, client=client
    )
    return str(tool_env.get("GH_TOKEN") or "")


def _expire_lease(client: Any) -> Any:
    held = client.grants.peek("github", SESSION)
    assert held is not None and held.hold_until_ms > time.time() * 1000 + 60_000
    held.grant_expires_at_ms = int(time.time() * 1000) - 1
    return held


@pytest.fixture
def slow_source(borrower: Any, monkeypatch: pytest.MonkeyPatch) -> Any:  # noqa: F811
    """The owner's gh read now overruns its deadline: the owner answers owner_busy."""
    from local_operator.network.credentials import github_source
    from tests.unit.network.test_github_source_ladder import GHO

    b = borrower
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(b.rig.env.root))
    assert _tool_token(b.client), "the initial borrow failed"
    release = threading.Event()

    def stuck() -> str:
        release.wait(5)
        return GHO

    b.rig.broker._github_source.invalidate()
    monkeypatch.setattr(github_mod, "read_gh_token", stuck)
    monkeypatch.setattr(github_source, "SOURCE_TIMEOUT_S", 0.04)
    try:
        yield b
    finally:
        release.set()
        thread = getattr(b.rig.broker._github_source, "_thread", None)
        if thread is not None:
            thread.join(2)


def test_owner_busy_after_expiry_keeps_gh_token_through_the_tool_door(slow_source: Any) -> None:
    b = slow_source
    from tests.unit.network.test_github_source_ladder import _ask

    held = _expire_lease(b.client)
    # The REAL owner answers owner_busy (its source read overran), not a stub.
    wire = _ask(b.rig, for_session=SESSION)
    assert wire["code"] == "rate_limited" and wire["reason"] == "owner_busy"
    asks = len(b.calls)

    assert _tool_token(b.client), "a busy owner ended a hold with hours left"
    assert len(b.calls) == asks + 1, "served from the hold: no retry storm"
    served = b.client.grants.peek("github", SESSION)
    assert served is held, "the admitted grant is kept, not replaced or dropped"
    rows = _audit_rows(b.rig.env.root, "credential.held_served")
    assert rows and rows[-1]["cause"] == "busy"
    assert rows[-1]["detail"]["hold_until"] <= held.hold_until_ms
    assert _deny_code(b.client) == "", "busy is not a revocation: no deny"


@pytest.mark.parametrize(
    "sentence, code",
    [
        (
            "this device is already running {n} long operations for its peers; "
            "net_broker was not started — try again in a moment",
            "rate_limited",
        ),
        (
            "net_broker did not finish within 75 s on this device; it may still "
            "complete, so ask for its status before retrying",
            "owner_unreachable",
        ),
    ],
    ids=["slot-full", "did-not-finish"],
)
def test_owner_relay_load_refusals_serve_the_hold_and_write_no_deny(
    borrower: Any, monkeypatch: pytest.MonkeyPatch, sentence: str, code: str  # noqa: F811
) -> None:
    """The relay's OWN sentences, classified by the relay-side client as on a node."""
    from local_operator.network.relay import SLOW_OP_MAX_PENDING

    b = borrower
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(b.rig.env.root))
    assert _tool_token(b.client)
    held = _expire_lease(b.client)
    reply = {"op": "error", "message": sentence.format(n=SLOW_OP_MAX_PENDING)}
    detail = b.client._detail_from_owner(reply, "github", "github")
    assert detail["code"] == code
    b.response[0] = detail

    assert _tool_token(b.client), f"the relay's {code} ended the hold"
    assert b.client.grants.peek("github", SESSION) is held
    assert _deny_code(b.client) == ""
    assert _audit_rows(b.rig.env.root, "credential.held_served")


def test_rate_limited_on_a_current_lease_is_the_plain_lease_without_a_retry(
    borrower: Any,  # noqa: F811
) -> None:
    b = borrower
    original = b.client.request_grant_sync("github", session_id="current")
    original.grant_expires_at_ms = int(time.time() * 1000) + 60_000
    b.response[0] = {"kind": "error", "code": "rate_limited", "key": "github"}
    served = b.client.request_grant_sync("github", session_id="current")
    assert served is original and not served.held_offline
    assert len(b.calls) == 2, "a slot-full relay is not re-asked 250 ms later"


# -- invariants: (a) hold_end, (b) revoked/denied/stale/panic/membership, (c) deny --


def test_busy_never_serves_past_hold_end(slow_source: Any) -> None:
    b = slow_source
    held = _expire_lease(b.client)
    held.hold_until_ms = held.grant_expires_at_ms  # the hold ended with the lease
    assert not _tool_token(b.client)
    assert b.client.grants.peek("github", SESSION) is None


def _deny(client: Any, *, network: bool) -> None:
    policy = client._policy()
    with policy.transaction() as document:
        if network:
            # Bumps the NETWORK generation every admitted grant was fenced with, so
            # each one is stale, whatever its own row says.
            policy.deny_network(document)
        else:
            policy.deny(document, client.owner_of("github"), "github", "consent_withdrawn")


@pytest.mark.parametrize(
    "ending", ["policy-deny", "network-generation", "panic", "membership", "revoke"]
)
def test_busy_never_serves_a_fenced_grant(slow_source: Any, ending: str) -> None:
    from local_operator.network import store
    from tests.unit.network.test_credentials_github import BORROWER_DEVICE, OWNER_DEVICE

    b = slow_source
    _expire_lease(b.client)
    if ending in ("policy-deny", "network-generation"):
        _deny(b.client, network=ending == "network-generation")
    elif ending == "panic":
        with store.mutate("n_gh", b.rig.env.root) as record:
            record.trust = "untrusted"
            store.save(record, b.rig.env.root)
    elif ending == "membership":
        with store.mutate("n_gh", b.rig.env.root) as record:
            record.members = [m for m in record.members if m.device_id != BORROWER_DEVICE]
            store.save(record, b.rig.env.root)
    else:
        b.rig.document.revoke("github", BORROWER_DEVICE, by=OWNER_DEVICE)
        b.rig.document.save()
        # The owner is still busy (it has not answered the revoke by name): only
        # the borrower's own fences can stop the hold here.
        b.response[0] = {"kind": "error", "code": "owner_busy", "key": "github"}
    served = b.client.request_grant_sync("github", session_id=SESSION)
    assert not hasattr(served, "access_token"), f"{ending}: a fenced grant was served"
    assert not _tool_token(b.client)
    if ending == "policy-deny":
        assert served.code == "consent_withdrawn" and _deny_code(b.client) == "consent_withdrawn"


def test_forced_refresh_is_never_answered_from_the_hold(slow_source: Any) -> None:
    from local_operator.network.credentials.types import BrokerError

    b = slow_source
    _expire_lease(b.client)
    refused = b.client.request_grant_sync("github", session_id=SESSION, force_refresh=True)
    assert isinstance(refused, BrokerError) and refused.code == "owner_busy"


def test_borrow_failed_is_written_by_the_borrower_and_coalesced(slow_source: Any) -> None:
    from local_operator.network.redial import RefusalCoalescer

    b = slow_source
    now = [1000.0]
    b.client._borrow_failed_rows = RefusalCoalescer(clock=lambda: now[0])
    held = _expire_lease(b.client)
    for _ in range(3):
        assert _tool_token(b.client)
    rows = _audit_rows(b.rig.env.root, "credential.borrow_failed")
    assert len(rows) == 1, rows
    detail = rows[0]["detail"]
    assert detail["code"] == "owner_busy" and detail["held"] is True
    assert detail["provider"] == "github" and detail["act"] == held.credential_ref.owner_device
    assert rows[0]["cause"] == "busy" and "repeats" not in detail
    now[0] += 301.0
    assert _tool_token(b.client)
    rows = _audit_rows(b.rig.env.root, "credential.borrow_failed")
    assert len(rows) == 2 and rows[1]["detail"]["repeats"] == 2
    from local_operator.network.audit import audit_path

    text = audit_path(b.rig.env.root).read_text(encoding="utf-8")
    assert held.access_token not in text, "token material in the audit log"
