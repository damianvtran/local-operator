"""Name -> member resolution for ``send``/``reply``: the owner's reachable set.

WHY A SECOND FILE, and why a different scope. ``channel_post`` resolves a name
inside the room the message lands in, because it carries a conversation id.
``send``/``reply`` carry **no conversation id at all** — they write to the peer
inbox — so the conversation rule cannot apply to them, and the ratified rule for
this path is the owner's REACHABLE set: the bridge directory intersected with the
owner's own ``send`` allowlist. Narrower than the directory on purpose: a name may
only resolve to somebody this owner has explicitly permitted.

Three guards hold together here, and each has its own test below:

1. **exactly one match** — the shared resolver, unchanged;
2. **bridge-enabled at resolution time** — presence in the allowlist is not the
   same fact as being able to receive, and a message to a switched-off bridge is
   a message that goes nowhere;
3. **a specific refusal** — a name that matches nobody says BOTH reasons it
   might: not on this team, or their bridge is not enabled. A model that cannot
   tell those apart retries uselessly instead of telling its human something true.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from local_operator.team_bridge import client as tb_client
from local_operator.team_bridge import host as host_mod
from local_operator.team_bridge.client import TeamBridgeClient
from local_operator.team_bridge.recipient_resolution import (
    members_from_allowlist,
    reachable_targets,
    resolution_audit_path,
)
from local_operator.team_bridge.tool import execute_bridge
from local_operator.team_bridge.types import MessageRef
from tests.unit.team_bridge.test_channel_tool import _runtime
from tests.unit.team_bridge.test_client import FakeStore, _ok

TENANT = "ten_1"

#: Two colleagues in the ORGANIZATION directory.
ADA = {"account_id": "acc_ada", "display_label": "Ada Lovelace", "bridge_enabled": True}
BO = {"account_id": "acc_bo", "display_label": "Bo Chen", "bridge_enabled": True}
#: A third who is in the directory with a switched-OFF bridge.
CY = {"account_id": "acc_cy", "display_label": "Cy Off", "bridge_enabled": False}


def _allow(account_id: str, direction: str = "send", effect: str = "allow") -> dict[str, Any]:
    return {"direction": direction, "effect": effect, "target": {"account_id": account_id}}


def _directory(*rows: dict[str, Any]) -> dict[str, Any]:
    return {"members": list(rows)}


def _allowlist(*entries: dict[str, Any]) -> dict[str, Any]:
    return {"allowlist": {"tenant_id": TENANT, "entries": list(entries)}}


def _client(
    directory: Any, allowlist: Any, sent: Any = None
) -> tuple[TeamBridgeClient, list[httpx.Request]]:
    """A real client over a mock transport; the requests it saw are returned."""

    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/directory"):
            return _ok(directory)
        if request.url.path.endswith("/allowlist"):
            return _ok(allowlist)
        return _ok(sent or {"message": {"id": "bmsg_out", "thread_id": "bth_1"}})

    client = TeamBridgeClient(
        None,
        base_url="https://hub.example/v1",
        store=FakeStore(),
        transport=httpx.MockTransport(handler),
    )
    return client, requests


def _posted(requests: list[httpx.Request]) -> list[dict[str, Any]]:
    return [json.loads(r.content) for r in requests if r.method == "POST"]


def _audit_rows(config_dir: Path) -> list[dict[str, Any]]:
    path = resolution_audit_path(config_dir)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


@pytest.fixture(autouse=True)
def _destination_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tb_client, "org_oauth_destination_allowed", lambda base: True)


@pytest.fixture(autouse=True)
def _gates_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(host_mod, "owner_fence_refusal", lambda runtime: None)
    monkeypatch.setattr(host_mod, "admission_refusal", lambda runtime, *, is_read=False: None)


# ---------------------------------------------------------------------------
# The reachable set itself: the intersection, read once
# ---------------------------------------------------------------------------


def test_the_reachable_set_is_the_send_allowlist_minus_denies() -> None:
    """Only ``send`` + ``allow`` targets are reachable; a deny wins over an allow."""
    payload = _allowlist(
        _allow("acc_ada"),
        _allow("acc_bo"),
        _allow("acc_bo", effect="deny"),
        _allow("acc_cy", direction="receive"),
    )
    assert reachable_targets(payload) == frozenset({("account_id", "acc_ada")})


def test_an_empty_allowlist_reaches_nobody_rather_than_the_directory() -> None:
    """The least-privilege consequence, asserted rather than assumed.

    A resolver that fell back to the organization directory when the owner had
    listed nobody would be guessing on the owner's behalf. Sending to an explicit
    account id is unaffected — that path is not a guess.
    """
    assert reachable_targets(_allowlist()) == frozenset()
    assert members_from_allowlist(_directory(ADA, BO), _allowlist()) == ()


def test_a_member_must_be_in_both_reads() -> None:
    """Directory-only members and allowlist-only ids are both not candidates."""
    candidates = members_from_allowlist(
        _directory(ADA, BO), _allowlist(_allow("acc_ada"), _allow("acc_missing"))
    )
    assert [member.account_id for member in candidates] == ["acc_ada"]


def test_unreadable_payloads_yield_no_candidates() -> None:
    assert members_from_allowlist(None, _allowlist(_allow("acc_ada"))) == ()
    assert members_from_allowlist(_directory(ADA), {"allowlist": "garbage"}) == ()
    assert members_from_allowlist({"members": "garbage"}, _allowlist(_allow("acc_ada"))) == ()


# ---------------------------------------------------------------------------
# A unique, enabled, allow-listed name delivers — with the id on the wire
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_unique_name_sends_to_the_resolved_account(tmp_path) -> None:
    client, requests = _client(_directory(ADA, BO), _allowlist(_allow("acc_ada"), _allow("acc_bo")))
    outcome = await execute_bridge(
        _runtime(client, config_dir=tmp_path),
        {"op": "send", "to": {"name": "ada lovelace"}, "text": "the QA round is ready"},
    )
    assert outcome["ok"] is True, outcome
    body = _posted(requests)[-1]
    assert body["to"] == {"account_id": "acc_ada"}
    assert body["type"] == "note"
    # The name never reached the wire; only the id did.
    assert "Ada Lovelace" not in json.dumps(body)
    # The audit row names the SCOPE that resolved it, and the tenant it read.
    row = _audit_rows(tmp_path)[-1]
    assert row["scope"] == "allowlist"
    assert row["scope_id"] == TENANT
    assert row["resolved_account_id"] == "acc_ada"
    assert row["matched_count"] == 1
    assert row["outcome"] == "resolved"


# ---------------------------------------------------------------------------
# The three refusals that make a wrong delivery impossible
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_name_in_the_org_but_not_in_the_allowlist_does_not_resolve(tmp_path) -> None:
    """LEAST PRIVILEGE for this path: the directory alone must never resolve."""
    client, requests = _client(_directory(ADA, BO), _allowlist(_allow("acc_bo")))
    outcome = await execute_bridge(
        _runtime(client, config_dir=tmp_path),
        {"op": "send", "to": {"name": "Ada Lovelace"}, "text": "hi"},
    )
    assert outcome["ok"] is False and outcome["code"] == "invalid_request"
    assert "no active member by that name in your bridge allowlist" in outcome["error"]
    # BOTH reasons, because one without the other sends the model guessing.
    assert "not on this team" in outcome["error"]
    assert "bridge is not enabled" in outcome["error"]
    assert _posted(requests) == []
    row = _audit_rows(tmp_path)[-1]
    assert row["outcome"] == "no_match"
    assert row["resolved_account_id"] is None
    assert row["scope"] == "allowlist"


@pytest.mark.asyncio
async def test_a_name_in_the_allowlist_with_a_switched_off_bridge_is_its_own_refusal(
    tmp_path,
) -> None:
    """Present but unreachable is NOT "nobody by that name", and must not read as it."""
    client, requests = _client(_directory(CY), _allowlist(_allow("acc_cy")))
    outcome = await execute_bridge(
        _runtime(client, config_dir=tmp_path),
        {"op": "send", "to": {"name": "Cy Off"}, "text": "hi"},
    )
    assert outcome["ok"] is False and outcome["code"] == "invalid_request"
    assert "bridge is not enabled" in outcome["error"]
    assert "this is not a name problem" in outcome["error"]
    # It is not the ambiguity or the zero-match sentence.
    assert "no active member" not in outcome["error"]
    assert _posted(requests) == []
    row = _audit_rows(tmp_path)[-1]
    assert row["outcome"] == "bridge_disabled"
    assert row["resolved_account_id"] == "acc_cy"
    assert row["matched_count"] == 1


@pytest.mark.asyncio
async def test_two_members_with_one_label_are_ambiguity_not_a_preference(tmp_path) -> None:
    """No first-row fallback, no human-over-agent preference: the candidates are listed."""
    first = {"account_id": "acc_1", "display_label": "Member", "bridge_enabled": True}
    second = {"account_id": "acc_2", "display_label": "Member", "bridge_enabled": True}
    client, requests = _client(
        _directory(first, second), _allowlist(_allow("acc_1"), _allow("acc_2"))
    )
    outcome = await execute_bridge(
        _runtime(client, config_dir=tmp_path),
        {"op": "send", "to": {"name": "Member"}, "text": "hi"},
    )
    assert outcome["ok"] is False and outcome["code"] == "invalid_request"
    assert "matches 2 members in your bridge allowlist" in outcome["error"]
    assert "acc_1" in outcome["error"] and "acc_2" in outcome["error"]
    assert _posted(requests) == []
    row = _audit_rows(tmp_path)[-1]
    assert row["outcome"] == "ambiguous"
    assert row["matched_count"] == 2


@pytest.mark.asyncio
async def test_a_denied_colleague_does_not_resolve(tmp_path) -> None:
    """A deny is the narrower rule: it wins even where an allow also names them."""
    client, requests = _client(
        _directory(ADA), _allowlist(_allow("acc_ada"), _allow("acc_ada", effect="deny"))
    )
    outcome = await execute_bridge(
        _runtime(client, config_dir=tmp_path),
        {"op": "send", "to": {"name": "Ada Lovelace"}, "text": "hi"},
    )
    assert outcome["ok"] is False and outcome["code"] == "invalid_request"
    assert _posted(requests) == []


# ---------------------------------------------------------------------------
# reply: the same resolver, then the pair lane's own guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_reply_name_resolves_and_the_answer_addresses_that_account(
    tmp_path, monkeypatch
) -> None:
    """A name on ``reply`` resolves AFTER admission, then the sender check applies.

    The guard is relocated, never dropped: the resolved pair must be the sender of
    the message being answered, exactly as an explicit ``account_id`` must be.
    """
    from local_operator.team_bridge.store import OwnerReceipt, OwnerSnapshot, Store

    owner = OwnerSnapshot(
        credential_id=1,
        account_id="acc_owner",
        tenant_id=TENANT,
        endpoint_id="bep_owner",
        canonical_base_url="https://hub.example",
        enrollment_generation=1,
    )
    store = Store(tmp_path)
    # A delivery row is only readable for a LIVE enrollment, so the owner is
    # committed first — the same setup the ack/reply tests use.
    store.commit_enrollment(owner, OwnerReceipt(consent_version="cv-1", consent_ref="cr-1"))
    store.merge_page(
        owner,
        [
            MessageRef(
                message_id="bmsg_ada",
                thread_id="bth_ada",
                tenant_id=TENANT,
                message_type="note",
                body_hash="b" * 64,
                seq=1,
                origin="member",
                delivery_state="unread",
                state_at="2026-10-20T00:00:00Z",
                sender_account_id="acc_ada",
                sender_endpoint_id="bep_ada",
                recipient_account_id="acc_owner",
                recipient_endpoint_id="bep_owner",
                request_state=None,
                request_expires_at=None,
                request_ref=None,
            ).validated()
        ],
        None,
    )
    client, requests = _client(
        _directory(ADA), _allowlist(_allow("acc_ada")), sent={"message": {"id": "bmsg_out"}}
    )
    runtime = _runtime(client, config_dir=tmp_path, endpoint_id="bep_owner")
    runtime.owner = owner
    outcome = await execute_bridge(
        runtime,
        {"op": "reply", "message_id": "bmsg_ada", "to": {"name": "Ada Lovelace"}, "text": "ack"},
    )
    assert outcome["ok"] is True, outcome
    body = _posted(requests)[-1]
    assert body["to"] == {"account_id": "acc_ada"}
    assert body["thread_id"] == "bth_ada"
    assert _audit_rows(tmp_path)[-1]["scope"] == "allowlist"


# ---------------------------------------------------------------------------
# The reads: a failure is recorded, never silent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_failed_directory_read_is_recorded_and_refuses(tmp_path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            503,
            json={"error": "unavailable"},
        )

    client = TeamBridgeClient(
        None,
        base_url="https://hub.example/v1",
        store=FakeStore(),
        transport=httpx.MockTransport(handler),
    )
    outcome = await execute_bridge(
        _runtime(client, config_dir=tmp_path),
        {"op": "send", "to": {"name": "Ada Lovelace"}, "text": "hi"},
    )
    assert outcome["ok"] is False
    assert outcome["code"] in ("directory_failed", "unavailable")
    rows = _audit_rows(tmp_path)
    assert rows and rows[-1]["outcome"] == "unavailable"
    assert rows[-1]["resolved_account_id"] is None


# ---------------------------------------------------------------------------
# Shape: the closed one-of still holds, and the op is named in the schema
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_name_plus_a_second_key_still_refuses_before_any_read(tmp_path) -> None:
    client, requests = _client(_directory(ADA), _allowlist(_allow("acc_ada")))
    outcome = await execute_bridge(
        _runtime(client, config_dir=tmp_path),
        {"op": "send", "to": {"name": "Ada Lovelace", "account_id": "acc_ada"}, "text": "hi"},
    )
    assert outcome["ok"] is False and outcome["code"] == "invalid_request"
    assert requests == []


def test_the_schema_says_which_scope_each_op_resolves_in() -> None:
    from local_operator.team_bridge.types import (
        SEND_RECIPIENT_DESCRIPTION as description,
    )

    assert "channel_post" in description
    assert "active members" in description
    assert "send/reply" in description
    assert "allowlist" in description
