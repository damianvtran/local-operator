"""A remote session is load-balanced the way a local one is (P1b).

THE GAPS (architect report on 05613b5ebb5a, reproduced by these cells on base):

* G1 — no stickiness. The owner's lend runs ``get_oauth_access(read_only=True)``
  outside any selection context, so no success receipt can anchor it: a remote
  session moved account whenever the owner's usage evidence moved, rewriting its
  whole prompt-cache prefix, while a local session stayed on its warm account.
* G2 — the owner's ``retry.accountSelectionPolicy`` never reached a lend; it ran
  ``reset-soonest`` whatever the operator chose.
* G3 — a 529 on a borrowed row was not failed over. The borrower's rotation has
  no local sibling, the owner's report arm changes nothing for ``unavailable``,
  and the re-ask was lent the same row: the turn died on bearer-a.

THE FIX is inputs, not a second selector: the borrower keeps the receipt and the
5xx demotion for its own sessions (keyed by the OWNER's row id) and sends them as
optional ``affinity`` / ``demote_credential_ids``; the owner applies them to that
one resolve through ``lend_selection_context``. Nothing on the owner is written
on a peer's word (F3), which several cells below check directly.

The cells drive the REAL owner broker and the REAL borrower client/store through
``stream_with_failover`` with a real ``OpenAICompatClient`` (#3066's ``borrowed``
rig); only the relay hop is in-process. The public OpenAI base URL is used so
the client sets its real ``openai:chat`` wire hint — that hint is what makes a
success a receipt at all, locally and remotely alike.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from local_operator.harness.types import ChatRequest, Message, ModelSpec, StreamEndEvent
from local_operator.network.credentials.types import peer_affinity
from local_operator.providers.auth_store import AuthStore
from local_operator.providers.clients import OpenAICompatClient
from local_operator.providers.failover import stream_with_failover
from local_operator.providers.usage import UsageAmount, UsageLimit, UsageReport
from local_operator.providers.usage_cache import UsageCacheStore, provider_cache_key
from tests.unit.network.test_credential_limit_skip import (  # noqa: F401 — fixtures by import
    borrowed,
    devices,
    idp,
    mesh,
    owner,
    stub_provider,
)
from tests.unit.network.test_credentials_owner import STUB_PROVIDER

_FIXTURES = (borrowed, owner, mesh, devices, idp, stub_provider)

MODEL = "claude-opus-5-5"
HOUR_MS = 3_600_000
SETTINGS = {"retry": {"baseDelayMs": 1, "maxRetries": 2, "fallbackChains": {}}}
EMAIL_A = "owner@example.test"
EMAIL_B = "second@example.test"


class _Stub:
    """OpenAI-compatible endpoint: 200 for every bearer except those told to 529."""

    def __init__(self, overloaded: tuple[str, ...] = ()) -> None:
        self.overloaded = set(overloaded)
        self.seen: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        bearer = request.headers.get("authorization", "").removeprefix("Bearer ")
        self.seen.append(bearer)
        if bearer in self.overloaded:
            return httpx.Response(
                529, json={"type": "error", "error": {"type": "overloaded_error"}}
            )
        chunk = {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}
        return httpx.Response(
            200,
            content=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode(),
            headers={"content-type": "text/event-stream"},
        )


def _request(session_id: str) -> ChatRequest:
    # ``supports_prompt_cache`` + ``prompt_cache_key`` on the public base URL is
    # what makes the real client set its ``openai:chat`` wire hint.
    return ChatRequest(
        model=ModelSpec(provider=STUB_PROVIDER, model_id=MODEL, supports_prompt_cache=True),
        messages=[Message.user("synthetic question")],
        prompt_cache_key=f"cache-{session_id}",
    )


async def _turn(auth: Any, stub: _Stub, session_id: str) -> list[Any]:
    request = _request(session_id)
    async with httpx.AsyncClient(transport=httpx.MockTransport(stub)) as http:
        client = OpenAICompatClient(
            "https://api.openai.com/v1", http_client=http, openai_api="chat_completions"
        )
        return [
            event
            async for event in stream_with_failover(
                request, auth, SETTINGS, lambda _spec: client, session_id=session_id
            )
        ]


def _ended(events: list[Any]) -> bool:
    return any(isinstance(event, StreamEndEvent) for event in events)


def _usage(root: Path, resets_ms: dict[str, int]) -> None:
    """The owner's warmer payload: one 5h window per account, resetting as given."""
    now = int(time.time() * 1000)
    cache = UsageCacheStore(root / "usage_cache.db")
    try:
        cache.set(
            provider_cache_key(STUB_PROVIDER, "fp"),
            STUB_PROVIDER,
            [
                UsageReport(
                    provider=STUB_PROVIDER,
                    fetched_at=now,
                    identity=email,
                    limits=[
                        UsageLimit(
                            id="five_hour",
                            label="5h",
                            amount=UsageAmount(used_fraction=0.3),
                            window="5 hour",
                            resets_at_ms=now + offset,
                            shared=True,
                        ),
                        UsageLimit(
                            id="seven_day",
                            label="7d",
                            amount=UsageAmount(used_fraction=0.3),
                            window="7 day",
                            resets_at_ms=now + 100 * HOUR_MS,
                            shared=True,
                        ),
                    ],
                )
                for email, offset in resets_ms.items()
            ],
            expires_at_ms=now + HOUR_MS,
        )
    finally:
        cache.close()


def _grants(rig: Any) -> list[dict[str, Any]]:
    return [
        row["detail"]
        for row in rig.owner.audit.tail(200, network_id="n_owner")
        if row.get("event") == "credential.grant"
    ]


def _owner_untouched(rig: Any) -> None:
    """F3: a peer's anchor or demotion never became owner routing state."""
    assert not rig.owner.auth._sticky, rig.owner.auth._sticky  # noqa: SLF001
    assert not rig.owner.auth._deprioritized, rig.owner.auth._deprioritized  # noqa: SLF001


# ---------------------------------------------------------------------------
# G1 — stickiness across turns
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_remote_session_keeps_its_account_when_the_usage_evidence_moves(
    borrowed: Any,
) -> None:
    rig = borrowed
    _usage(rig.owner.root, {EMAIL_A: 1 * HOUR_MS, EMAIL_B: 4 * HOUR_MS})
    stub = _Stub()

    assert _ended(await _turn(rig.store, stub, "sess-1"))
    assert stub.seen == ["bearer-a"], "first pick is the reset-soonest account"
    # Nothing to say on the first ask: the frame is the one an older owner reads.
    assert "affinity" not in rig.relay.grant_frames[0]
    assert "demote_credential_ids" not in rig.relay.grant_frames[0]

    # The evidence moves: B now resets soonest, so a fresh pick would be B.
    _usage(rig.owner.root, {EMAIL_A: 4 * HOUR_MS, EMAIL_B: 1 * HOUR_MS})
    for _ in range(2):
        assert _ended(await _turn(rig.store, stub, "sess-1"))
    assert stub.seen == ["bearer-a"] * 3, f"the warm session moved account: {stub.seen}"
    assert rig.relay.grant_frames[-1]["affinity"] == {
        "account_id": rig.row_a.id,
        "account_created_at": rig.row_a.created_at,
    }
    rows = _grants(rig)
    assert [r["credential_id"] for r in rows] == [rig.row_a.id] * 3
    assert [r["anchored"] for r in rows] == [False, True, True]
    assert {r["policy"] for r in rows} == {"reset-soonest"}

    # The anchor is per session: a new one takes today's reset-soonest pick.
    stub.seen.clear()
    assert _ended(await _turn(rig.store, stub, "sess-2"))
    assert stub.seen == ["bearer-b"]
    _owner_untouched(rig)


# ---------------------------------------------------------------------------
# G3 — a 529 fails over, as it does locally
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_529_on_a_borrowed_row_ends_the_turn_on_the_sibling(borrowed: Any) -> None:
    rig = borrowed
    _usage(rig.owner.root, {EMAIL_A: 1 * HOUR_MS, EMAIL_B: 4 * HOUR_MS})
    stub = _Stub(overloaded=("bearer-a",))

    events = await _turn(rig.store, stub, "sess-529")
    assert _ended(events), f"the turn died on the overloaded row: {stub.seen}"
    assert stub.seen[0] == "bearer-a" and stub.seen[-1] == "bearer-b", stub.seen
    assert rig.relay.grant_frames[-1]["demote_credential_ids"] == [rig.row_a.id]
    # The provider failed, not the account: no cooldown, no owner block, no mark.
    assert not rig.client.state.cooldowns(
        STUB_PROVIDER, MODEL, owner_device=rig.client.owner_of(STUB_PROVIDER)
    )
    assert not rig.owner.auth.is_blocked(rig.row_a.id, STUB_PROVIDER)
    _owner_untouched(rig)

    # The success on B is the session's new lease; the next turn stays there.
    stub.seen.clear()
    assert _ended(await _turn(rig.store, stub, "sess-529"))
    assert stub.seen == ["bearer-b"], stub.seen
    assert _grants(rig)[-1]["anchored"] is True


@pytest.mark.asyncio
async def test_local_control_a_529_moves_an_owner_session_to_the_sibling(borrowed: Any) -> None:
    """The same stub and evidence on the owner's own store: the behaviour matched."""
    rig = borrowed
    _usage(rig.owner.root, {EMAIL_A: 1 * HOUR_MS, EMAIL_B: 4 * HOUR_MS})
    stub = _Stub(overloaded=("bearer-a",))
    local = AuthStore(config_dir=rig.owner.root)
    try:
        assert _ended(await _turn(local, stub, "sess-local"))
    finally:
        local.close()
    assert stub.seen[0] == "bearer-a" and stub.seen[-1] == "bearer-b", stub.seen


@pytest.mark.asyncio
async def test_first_pick_matches_a_local_session_at_the_same_moment(borrowed: Any) -> None:
    rig = borrowed
    _usage(rig.owner.root, {EMAIL_A: 4 * HOUR_MS, EMAIL_B: 1 * HOUR_MS})
    remote, local_stub = _Stub(), _Stub()
    local = AuthStore(config_dir=rig.owner.root)
    try:
        assert _ended(await _turn(rig.store, remote, "sess-first"))
        assert _ended(await _turn(local, local_stub, "sess-first"))
    finally:
        local.close()
    assert remote.seen == local_stub.seen == ["bearer-b"]


# ---------------------------------------------------------------------------
# G2 — the owner's selection policy
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_lend_follows_the_owners_selection_policy(borrowed: Any) -> None:
    rig = borrowed
    _usage(rig.owner.root, {EMAIL_A: 1 * HOUR_MS, EMAIL_B: 4 * HOUR_MS})
    # The layout ``/settings`` writes: every setting lives under ``values``.
    (rig.owner.root / "config.yml").write_text(
        "values:\n  retry:\n    accountSelectionPolicy: fair\n"
    )
    expected = AuthStore(config_dir=rig.owner.root)
    expected.configure_account_selection_policy("fair")
    try:
        # A session whose ``fair`` pick differs from reset-soonest's (A), so the
        # cell can only pass if the policy reached the lend.
        session = None
        for index in range(64):
            candidate = f"sess-fair-{index}"
            access = await expected.get_oauth_access(
                STUB_PROVIDER, candidate, read_only=True, model_id=MODEL
            )
            if access is not None and access.credential_id == rig.row_b.id:
                session = candidate
                break
        assert session is not None, "no session id hashes to B under fair"
    finally:
        expected.close()
    stub = _Stub()
    assert _ended(await _turn(rig.store, stub, session))
    assert stub.seen == ["bearer-b"], f"the lend ignored the owner's policy: {stub.seen}"
    assert _grants(rig)[-1]["policy"] == "fair"


# ---------------------------------------------------------------------------
# A peer's anchor is a preference among rows it may already be lent, never more
# ---------------------------------------------------------------------------


def _lend(rig: Any, session: str, **fields: Any) -> dict[str, Any]:
    return rig.client.serve_grant(
        STUB_PROVIDER, provider=STUB_PROVIDER, session_id=session, model_id=MODEL, **fields
    )


def test_a_forged_or_stale_anchor_is_ignored(borrowed: Any) -> None:
    rig = borrowed
    _usage(rig.owner.root, {EMAIL_A: 1 * HOUR_MS, EMAIL_B: 4 * HOUR_MS})
    a = (rig.row_a.id, rig.row_a.created_at)
    b = (rig.row_b.id, rig.row_b.created_at)

    def lent(detail: dict[str, Any]) -> int:
        return int(detail["credential_ref"]["credential_id"])

    # Honoured: B anchors over reset-soonest's A.
    assert lent(_lend(rig, "s", affinity=b)) == rig.row_b.id
    assert _grants(rig)[-1]["anchored"] is True
    # Wrong created_at (a re-created row under a reused id): ignored.
    assert lent(_lend(rig, "s", affinity=(b[0], b[1] + 1))) == rig.row_a.id
    # An excluded row (the borrower's own cooldown) cannot be reached by anchoring.
    assert lent(_lend(rig, "s", affinity=b, exclude_credential_ids=[rig.row_b.id])) == rig.row_a.id
    # An owner-blocked row cannot be reached by anchoring.
    rig.owner.auth.block_credential(rig.row_b.id, STUB_PROVIDER, block_ms=60_000)
    assert lent(_lend(rig, "s", affinity=b)) == rig.row_a.id
    rig.owner.auth.clear_blocks_for_model(rig.row_b.id, STUB_PROVIDER, MODEL)
    # A row the owner does not have anchors nothing.
    assert lent(_lend(rig, "s", affinity=(9_999, a[1]))) == rig.row_a.id
    assert [r["anchored"] for r in _grants(rig)[-4:]] == [False] * 4
    # A peer demotion reorders this lend only, and an anchor still beats it (the
    # demotion exemption a local receipt has).
    assert lent(_lend(rig, "s", demote_credential_ids=[rig.row_a.id])) == rig.row_b.id
    assert lent(_lend(rig, "s", affinity=a, demote_credential_ids=[rig.row_a.id])) == rig.row_a.id
    # Every row demoted still serves (the store's existing second pass).
    assert lent(_lend(rig, "s", demote_credential_ids=[rig.row_a.id, rig.row_b.id])) in {
        rig.row_a.id,
        rig.row_b.id,
    }
    _owner_untouched(rig)


def test_the_anchor_validator_is_strict() -> None:
    assert peer_affinity({"account_id": 3, "account_created_at": 1_700_000_000_000}) == (
        3,
        1_700_000_000_000,
    )
    for junk in (
        None,
        [3, 1],
        {"account_id": "3", "account_created_at": 1},
        {"account_id": 3.0, "account_created_at": 1},
        {"account_id": True, "account_created_at": 1},
        {"account_id": 3, "account_created_at": 0},
        {"account_id": -3, "account_created_at": 1},
        {"account_id": 3, "account_created_at": 2**60},
        {"account_id": 3},
    ):
        assert peer_affinity(junk) is None, junk


@pytest.mark.asyncio
async def test_expired_receipt_reselects_and_old_owner_records_none(borrowed: Any) -> None:
    from local_operator.providers.account_affinity import request_compatibility

    rig = borrowed
    _usage(rig.owner.root, {EMAIL_A: HOUR_MS, EMAIL_B: 4 * HOUR_MS})
    stub = _Stub()
    await _turn(rig.store, stub, "expiry")
    receipts = rig.store._receipt_store()
    key = rig.store._receipt_key(STUB_PROVIDER, rig.client.owner_of(STUB_PROVIDER))
    receipt = receipts.get("expiry", key, MODEL, request_compatibility(_request("expiry")))
    assert receipt is not None and receipt.account_id == rig.row_a.id
    # The monotonic deadline is authoritative even before the wall clock catches up.
    receipts._deadlines[("expiry", key)] = time.monotonic() - 1
    _usage(rig.owner.root, {EMAIL_A: 4 * HOUR_MS, EMAIL_B: HOUR_MS})
    await _turn(rig.store, stub, "expiry")
    assert stub.seen == ["bearer-a", "bearer-b"]
    assert "affinity" not in rig.relay.grant_frames[-1]

    from local_operator.network.credentials.types import Grant

    detail = _lend(rig, "legacy")
    detail["credential_ref"].pop("account_created_at")
    grant = Grant.from_detail(detail)
    assert grant.credential_ref.account_created_at == 0
    access = rig.store._access_from_grant(grant)
    from local_operator.providers.account_affinity import CacheAffinityHint

    req = _request("legacy")
    rig.store.record_account_success(
        STUB_PROVIDER,
        "legacy",
        access,
        req,
        CacheAffinityHint("openai:chat", request_compatibility(req), 300_000, "bounded-heuristic"),
        int(time.time() * 1000),
        time.monotonic(),
    )
    assert receipts.get("legacy", key, MODEL, request_compatibility(req)) is None


@pytest.mark.asyncio
async def test_an_old_completion_keeps_its_own_owner_row_identity(borrowed: Any) -> None:
    from local_operator.providers.account_affinity import (
        CacheAffinityHint,
        request_compatibility,
    )

    rig = borrowed
    _usage(rig.owner.root, {EMAIL_A: HOUR_MS, EMAIL_B: 4 * HOUR_MS})
    first = await rig.store.get_oauth_access(STUB_PROVIDER, "concurrent", model_id=MODEL)
    _usage(rig.owner.root, {EMAIL_A: 4 * HOUR_MS, EMAIL_B: HOUR_MS})
    second = await rig.store.get_oauth_access(STUB_PROVIDER, "concurrent", model_id=MODEL)
    assert first.access_token != second.access_token
    req = _request("concurrent")
    hint = CacheAffinityHint(
        "openai:chat", request_compatibility(req), 300_000, "bounded-heuristic"
    )
    sent = int(time.time() * 1000)
    # Completion of the first request AFTER a second resolve must still name A.
    rig.store.record_account_success(
        STUB_PROVIDER, "concurrent", first, req, hint, sent, time.monotonic()
    )
    key = rig.store._receipt_key(STUB_PROVIDER, rig.client.owner_of(STUB_PROVIDER))
    receipt = rig.store._receipt_store().get("concurrent", key, MODEL, hint.compatibility)
    assert receipt is not None and receipt.account_id == rig.row_a.id


def test_a_depleted_anchor_is_ineligible(borrowed: Any) -> None:
    rig = borrowed
    _usage(rig.owner.root, {EMAIL_A: HOUR_MS, EMAIL_B: 4 * HOUR_MS})
    cache = UsageCacheStore(rig.owner.root / "usage_cache.db")
    now = int(time.time() * 1000) + 1
    cache.set(
        provider_cache_key(STUB_PROVIDER, "depleted"),
        STUB_PROVIDER,
        [
            UsageReport(
                provider=STUB_PROVIDER,
                fetched_at=now,
                identity=EMAIL_A,
                limits=[
                    UsageLimit(
                        id="five_hour",
                        label="5h",
                        shared=True,
                        amount=UsageAmount(used_fraction=1.0),
                        window="5 hour",
                        resets_at_ms=now + HOUR_MS,
                    )
                ],
            )
        ],
        expires_at_ms=now + HOUR_MS,
    )
    cache.close()
    detail = _lend(rig, "depleted", affinity=(rig.row_a.id, rig.row_a.created_at))
    assert detail["credential_ref"]["credential_id"] == rig.row_b.id
    assert _grants(rig)[-1]["anchored"] is False
    _owner_untouched(rig)


def test_lend_context_is_scoped_and_does_not_mutate_owner(borrowed: Any) -> None:
    from local_operator.providers.account_affinity import (
        current_lend_anchor,
        current_lend_demotions,
        current_selection_policy,
        lend_selection_context,
    )

    rig = borrowed
    with pytest.raises(RuntimeError):
        with lend_selection_context(
            "fair", (rig.row_a.id, rig.row_a.created_at), frozenset({rig.row_b.id})
        ):
            assert current_selection_policy() == "fair"
            assert current_lend_anchor() == (rig.row_a.id, rig.row_a.created_at)
            assert rig.owner.auth._active_demotions(STUB_PROVIDER) == {rig.row_b.id}
            raise RuntimeError("synthetic cancellation-like unwind")
    assert current_selection_policy() is None
    assert current_lend_anchor() is None
    assert not current_lend_demotions()
    assert not rig.owner.auth._active_demotions(STUB_PROVIDER)
    _owner_untouched(rig)
