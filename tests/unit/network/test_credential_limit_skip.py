"""F-ANTH-429: a credential that answered a long rate limit sits out its window.

THE INCIDENT (remote session ce43d8ddddad, lop-remote-node-2): a borrowed
Anthropic login answered ``429 … retry in 24h``. The borrower reported it, the
owner — by design (``owner.REMOTE_QUOTA_BLOCK_MAX_MS``, F3) — wrote at most a
60 s family block on a peer's word, and the very next grant re-served the same
spent row. The turn ended on the 429 although the owner held other logins.

A LOCAL row in the same position is blocked for the advertised window by
``AuthStore.rotate_sibling`` and the cascade moves on. These cells pin that ONE
rule for both paths, end to end through ``stream_with_failover`` with a real
``OpenAICompatClient`` talking to an httpx stub provider:

* the request after the 429 is served by credential B, and
* credential A is not asked again inside its window (a second request in the
  same session goes straight to B; A's bearer never reaches the wire again).

"Its window" is the local rule's: the advertised wait capped at
``MAX_CREDENTIAL_BLOCK_MS`` (an hour), after which the row is re-probed once.

The borrowed cell drives the REAL owner broker (``MeshCredentialBroker`` from
``test_credentials_owner``'s ``owner`` fixture, two rows on the owner's store)
and the REAL ``MeshCredentialClient``/``MeshAwareAuthStore`` on the borrower;
only the relay hop is short-circuited, by handing the client's leg-1 frame to
the leg-1 op and the leg-2 frame to the owner's ``on_broker`` in-process.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from local_operator.harness.types import ChatRequest, ModelSpec, StreamEndEvent
from local_operator.network.credentials import client as client_mod
from local_operator.network.credentials import placement as placement_mod
from local_operator.network.credentials.client import MeshCredentialClient
from local_operator.network.credentials.state import PlacementState
from local_operator.network.credentials.store import MeshAwareAuthStore
from local_operator.network.credentials.types import BrokerError
from local_operator.providers.auth_store import MAX_CREDENTIAL_BLOCK_MS, AuthStore
from local_operator.providers.clients import OpenAICompatClient
from local_operator.providers.failover import stream_with_failover
from tests.unit.network.test_credentials_real_link import mesh  # noqa: F401 — fixture by import
from tests.unit.network.test_relay_e2e import devices  # noqa: F401 — fixture by import
from tests.unit.network.test_credentials_owner import (  # noqa: F401 — fixtures by import
    BORROWER_DEVICE,
    OWNER_DEVICE,
    STUB_PROVIDER,
    _Link,
    idp,
    owner,
    stub_provider,
)

# ``owner`` and ``mesh`` are fixtures registered by the imports above and reached
# with ``getfixturevalue``; the names are kept bound so the registration holds.
_FIXTURES = (owner, mesh, devices, idp, stub_provider)

DAY_S = 24 * 60 * 60
SETTINGS = {"retry": {"baseDelayMs": 1, "maxRetries": 2, "fallbackChains": {}}}


class StubProvider:
    """An OpenAI-compatible endpoint: 429 retry-in-24h for one bearer, 200 otherwise."""

    def __init__(self, spent_bearer: str) -> None:
        self.spent = spent_bearer
        self.seen: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        bearer = request.headers.get("authorization", "").removeprefix("Bearer ")
        self.seen.append(bearer)
        if bearer == self.spent:
            return httpx.Response(
                429,
                headers={"retry-after": str(DAY_S)},
                json={"error": {"message": "rate limit or quota exceeded", "type": "rate_limit"}},
            )
        chunk = {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}
        return httpx.Response(
            200,
            content=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode(),
            headers={"content-type": "text/event-stream"},
        )


async def _turn(auth: Any, stub: StubProvider, provider: str, session_id: str) -> list[Any]:
    request = ChatRequest(model=ModelSpec(provider=provider, model_id="claude-opus-5-5"))
    async with httpx.AsyncClient(transport=httpx.MockTransport(stub)) as http:

        async def client_for(spec: ModelSpec) -> Any:
            return OpenAICompatClient("https://stub.invalid/v1", http_client=http)

        return [
            event
            async for event in stream_with_failover(
                request, auth, SETTINGS, client_for, session_id=session_id
            )
        ]


def _ended(events: list[Any]) -> bool:
    return any(isinstance(event, StreamEndEvent) for event in events)


# ---------------------------------------------------------------------------
# The shared rule
# ---------------------------------------------------------------------------


def test_the_window_is_the_advertised_wait_floored_and_family_scoped() -> None:
    """One rule, read by the local block AND the borrower's cooldown."""
    from local_operator.providers.auth_store import failure_block_window

    # A day advertised: the window is the local ceiling every block already had.
    scope, window = failure_block_window(
        DAY_S * 1000, usage_limited=True, model_id="claude-opus-5-5"
    )
    assert (scope, window) == ("model:opus", MAX_CREDENTIAL_BLOCK_MS)
    # Inside the ceiling the advertised wait is honoured exactly.
    assert failure_block_window(120_000, usage_limited=True)[1] == 120_000
    # No advertised wait: the floor, as before.
    assert failure_block_window(None, usage_limited=True)[1] == 60_000
    # Not a usage limit: account-wide, never family-scoped.
    assert failure_block_window(5, usage_limited=False, model_id="claude-opus-5-5")[0] == ""


# ---------------------------------------------------------------------------
# Local: the control the borrowed path must match
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_local_a_24h_429_moves_to_the_sibling_and_rests_the_row(tmp_path: Path) -> None:
    store = AuthStore(db_path=tmp_path / "auth.db")
    try:
        row_a = store.upsert_credential("openai", {"type": "api_key", "key": "key-a"})
        row_b = store.upsert_credential("openai", {"type": "api_key", "key": "key-b"})
        store.pin_session_credential("openai", "sess-local", row_a.id)
        stub = StubProvider("key-a")

        first = await _turn(store, stub, "openai", "sess-local")
        assert _ended(first), first
        assert stub.seen == ["key-a", "key-b"], stub.seen

        remaining = store.blocked_remaining_ms(row_a.id, "openai", "claude-opus-5-5")
        # The local rule: the advertised day, capped at the store's ceiling.
        assert remaining > MAX_CREDENTIAL_BLOCK_MS - 60_000, remaining
        assert not store.is_blocked_for_model(row_b.id, "openai", "claude-opus-5-5")

        stub.seen.clear()
        second = await _turn(store, stub, "openai", "sess-local")
        assert _ended(second), second
        assert stub.seen == ["key-b"], "the spent row was asked again inside its window"
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Borrowed: the same rule through the broker
# ---------------------------------------------------------------------------


class _InProcessRelay:
    """Leg 1 (control op) → leg 2 (owner's ``on_broker``), with no sockets.

    Every frame the real client builds goes through the real leg-1 op and the
    real owner handler; only the transport is replaced, which is exactly the part
    these cells are not about.
    """

    def __init__(self, borrower: MeshCredentialClient, broker: Any) -> None:
        from local_operator.network.credentials import _LocalOps

        self.borrower = borrower
        self.broker = broker
        self.ops = _LocalOps(server=None)
        self.ops._client = borrower  # noqa: SLF001 — the op's resolved client, in-process
        self.grant_frames: list[dict[str, Any]] = []
        self.req = 0

    def control(self, _record: Any, op: str, **frame: Any) -> dict[str, Any]:
        frame.pop("timeout", None)
        if op == "credential_grant":
            return {"op": "ack", "detail": self.ops.grant(frame)}
        if op == "credential_report":
            return {"op": "ack", "detail": self.ops.report(frame)}
        raise AssertionError(f"unexpected op {op}")

    def ask_owner(self, _owner: str, frame: dict[str, Any]) -> dict[str, Any]:
        if frame.get("kind") == "grant":
            self.grant_frames.append(dict(frame))
        # ``req`` is what the peer link stamps on every request so the answer can
        # be matched; the owner refuses a frame without one.
        self.req += 1
        return self.broker.on_broker(_Link(BORROWER_DEVICE), {**frame, "req": self.req})


def _borrower_root(tmp_path: Path, owner_root: Path) -> Path:
    """The borrower's root: its own copy of the owner's placement, nothing else."""
    root = tmp_path / "borrower"
    (root / "network" / "credentials" / "n_owner").mkdir(parents=True)
    source = placement_mod.placement_path("n_owner", owner_root)
    placement_mod.placement_path("n_owner", root).write_bytes(source.read_bytes())
    return root


class _NoLocalLogins:
    """The borrower has no login of its own: every serve is a borrow."""

    async def get_api_key(self, *_a: Any, **_k: Any) -> None:
        return None

    async def get_oauth_access(self, *_a: Any, **_k: Any) -> None:
        return None

    def list_credentials(self, *_a: Any, **_k: Any) -> list[Any]:
        return []

    def session_credential_id(self, *_a: Any, **_k: Any) -> None:
        return None

    def rotate_sibling(self, *_a: Any, **_k: Any) -> bool:
        return False

    def release_session_credential(self, *_a: Any, **_k: Any) -> None:
        return None


@pytest.fixture()
def borrowed(request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Two live rows on the owner (A spent, B healthy); a borrower with none.

    The owner fixture is reached by NAME (``getfixturevalue``), the convention the
    two-root cells use, so the import that registers it is not shadowed.
    """
    owner_rig = request.getfixturevalue("owner")
    now_ms = int(time.time() * 1000)
    owner_rig.auth.upsert_credential(
        STUB_PROVIDER,
        {
            "type": "oauth",
            "access": "bearer-a",
            "expires": now_ms + 3_600_000,
            "refresh": "",
            "email": "owner@example.test",
        },
    )
    row_b = owner_rig.auth.upsert_credential(
        STUB_PROVIDER,
        {
            "type": "oauth",
            "access": "bearer-b",
            "expires": now_ms + 3_600_000,
            "refresh": "",
            "email": "second@example.test",
        },
    )
    # The first upsert carries the fixture row's email, so it UPDATES that row in
    # place (same identity key): the owner has exactly two rows, A and B.
    rows = owner_rig.auth.list_credentials(STUB_PROVIDER)
    assert len(rows) == 2, rows
    row_a = next(row for row in rows if row.id != row_b.id)
    root = _borrower_root(tmp_path, owner_rig.root)
    client = MeshCredentialClient(
        root=root,
        self_device=BORROWER_DEVICE,
        network_id="n_owner",
        placement=placement_mod.PlacementDocument.load(
            "n_owner", root, self_device=BORROWER_DEVICE
        ),
        state=PlacementState("n_owner", root=root),
    )
    relay = _InProcessRelay(client, owner_rig.broker)
    from local_operator.network import relay as relay_mod
    from local_operator.network import store as net_store

    monkeypatch.setattr(net_store, "find_own_relay", lambda _root: object())
    monkeypatch.setattr(relay_mod, "control_request", relay.control)
    monkeypatch.setattr(
        client_mod.MeshCredentialClient,
        "_ask_owner_directly",
        lambda _self, owner_device, frame: relay.ask_owner(owner_device, frame),
    )
    # The relay-side twin the leg-1 op reaches is the SAME client object here, so
    # its ``owner_of`` and ``should_borrow`` read the same document.
    store = MeshAwareAuthStore(_NoLocalLogins(), mesh=client, config_dir=root)
    return type(
        "Borrowed",
        (),
        {
            "store": store,
            "client": client,
            "relay": relay,
            "row_a": row_a,
            "row_b": row_b,
            "owner": owner_rig,
            "root": root,
        },
    )


@pytest.mark.asyncio
async def test_borrowed_a_24h_429_moves_to_the_next_row_and_rests_the_spent_one(
    borrowed: Any,
) -> None:
    # The owner's order puts A first for this session: demote B so the FIRST
    # lend is the spent row, which is the incident's shape.
    borrowed.owner.auth.deprioritize_credential(STUB_PROVIDER, borrowed.row_b.id)
    stub = StubProvider("bearer-a")

    first = await _turn(borrowed.store, stub, STUB_PROVIDER, "sess-borrow")
    assert _ended(first), f"the turn ended on the 429 instead of moving on: {first}"
    assert stub.seen[0] == "bearer-a", stub.seen
    assert stub.seen[-1] == "bearer-b", stub.seen
    assert stub.seen.count("bearer-a") == 1, f"the spent row was re-served: {stub.seen}"

    # The cooldown is the LOCAL rule's window, named by the account's label.
    resting = borrowed.client.state.cooldowns(STUB_PROVIDER, "claude-opus-5-5")
    assert [row["credential_id"] for row in resting] == [borrowed.row_a.id], resting
    assert resting[0]["label"] == "owner@example.test", resting
    left_s = resting[0]["until"] - time.time()
    assert MAX_CREDENTIAL_BLOCK_MS / 1000 - 60 < left_s <= MAX_CREDENTIAL_BLOCK_MS / 1000, resting
    # ...and it survives the process: a fresh state read sees it.
    reloaded = PlacementState.load("n_owner", borrowed.root)
    assert [row["credential_id"] for row in reloaded.cooldowns(STUB_PROVIDER, "claude-opus-5-5")] == [
        borrowed.row_a.id
    ]

    # Same session, next request: A is not asked inside its window.
    stub.seen.clear()
    second = await _turn(borrowed.store, stub, STUB_PROVIDER, "sess-borrow")
    assert _ended(second), second
    assert stub.seen == ["bearer-b"], f"the spent row was asked again: {stub.seen}"
    assert borrowed.relay.grant_frames[-1].get("exclude_credential_ids") == [borrowed.row_a.id]

    # F3 still holds: nothing the borrower did blocks the owner's row beyond
    # the owner's own bounded 60 s family block.
    assert borrowed.owner.auth.blocked_remaining_ms(
        borrowed.row_a.id, STUB_PROVIDER, "claude-opus-5-5"
    ) <= 60_000


def test_every_row_resting_reads_as_a_quota_rest_naming_the_accounts(borrowed: Any) -> None:
    """All lendable rows cooling: ``quota_blocked`` with the labels, never "sign in"."""
    until = time.time() + 3_600
    # A family-scoped rest on A and an account-wide one on B: the two shapes the
    # local rule writes, so the scope read is exercised as well as the refusal.
    for row, label, scope in (
        (borrowed.row_a, "owner@example.test", "model:opus"),
        (borrowed.row_b, "second@example.test", ""),
    ):
        borrowed.client.state.cool_down(
            STUB_PROVIDER,
            owner_device=OWNER_DEVICE,
            credential_id=row.id,
            block_scope=scope,
            until=until,
            label=label,
        )
    outcome = borrowed.client.request_grant_sync(
        STUB_PROVIDER, session_id="sess-all", model_id="claude-opus-5-5", provider=STUB_PROVIDER
    )
    assert isinstance(outcome, BrokerError), outcome
    assert outcome.code == "quota_blocked", outcome
    assert "owner@example.test" in outcome.message and "second@example.test" in outcome.message
    assert "sign" not in outcome.message.lower(), outcome.message
    # The incident text the failover driver reads (``no_credential_reason``) is
    # this sentence: the turn's error names the resting accounts by label only.
    asyncio.run(borrowed.store.get_api_key(STUB_PROVIDER, "sess-all", model_id="claude-opus-5-5"))
    reason = borrowed.store.no_credential_reason(STUB_PROVIDER)
    assert reason and "owner@example.test" in reason and "second@example.test" in reason, reason
    assert "bearer-" not in reason, reason
    # A model of another family is gated only by the account-wide rest.
    other = borrowed.client.state.cooldowns(STUB_PROVIDER, "claude-haiku-5")
    assert [row["credential_id"] for row in other] == [borrowed.row_b.id]


def test_a_peer_exclusion_never_blocks_the_owners_row(borrowed: Any) -> None:
    """The exclusion narrows ONE resolve; the owner's own routing is untouched."""
    from tests.unit.network.test_credentials_owner import _detail, _grant_frame

    before = borrowed.owner.auth._conn.execute(  # noqa: SLF001 — the block table is the fact
        "SELECT COUNT(*) FROM auth_credential_blocks"
    ).fetchone()[0]
    frame = _grant_frame(BORROWER_DEVICE, session="sess-x", model_id="claude-opus-5-5")
    frame["exclude_credential_ids"] = [borrowed.row_b.id, "junk", -4]
    detail = _detail(borrowed.owner.broker.on_broker(_Link(BORROWER_DEVICE), frame))
    assert detail["kind"] == "grant", detail
    assert detail["credential_ref"]["credential_id"] != borrowed.row_b.id, detail
    after = borrowed.owner.auth._conn.execute(  # noqa: SLF001
        "SELECT COUNT(*) FROM auth_credential_blocks"
    ).fetchone()[0]
    assert after == before


# ---------------------------------------------------------------------------
# Borrowed, over REAL relays: control socket, peer link, owner broker
# ---------------------------------------------------------------------------


class _FirstBearerSpent(StubProvider):
    """429 retry-in-24h for whichever bearer reaches the wire FIRST.

    Over the real rig the owner's first row refreshes against the stub IdP, so its
    bearer is minted at run time; spending "the first one lent" is the incident's
    shape without the test having to predict the minted value.
    """

    def __init__(self) -> None:
        super().__init__(spent_bearer="")

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if not self.spent:
            self.spent = request.headers.get("authorization", "").removeprefix("Bearer ")
        return super().__call__(request)


@pytest.mark.asyncio
async def test_over_real_relays_the_spent_row_rests_and_the_next_row_serves(
    request: pytest.FixtureRequest,
) -> None:
    from tests.unit.network.test_credentials_real_link import _pull, _share

    pair = request.getfixturevalue("mesh")
    _share(pair)
    _pull(pair)
    owner_auth = AuthStore(config_dir=pair.a.root)
    try:
        second = owner_auth.upsert_credential(
            STUB_PROVIDER,
            {
                "type": "oauth",
                "access": "bearer-second",
                "expires": int(time.time() * 1000) + 3_600_000,
                "refresh": "",
                "email": "second@example.test",
            },
        )
        client = MeshCredentialClient.for_this_device(pair.b.root)
        assert client is not None, "the share never reached the borrower"
        store = MeshAwareAuthStore(_NoLocalLogins(), mesh=client, config_dir=pair.b.root)
        stub = _FirstBearerSpent()

        first = await _turn(store, stub, STUB_PROVIDER, "sess-real")
        assert _ended(first), f"the turn ended on the 429: {first}"
        assert len(stub.seen) == 2 and stub.seen[0] != stub.seen[1], stub.seen
        resting = client.state.cooldowns(STUB_PROVIDER, "claude-opus-5-5")
        assert len(resting) == 1 and resting[0]["credential_id"] != second.id, resting

        stub.seen.clear()
        again = await _turn(store, stub, STUB_PROVIDER, "sess-real")
        assert _ended(again), again
        assert stub.spent not in stub.seen, f"the spent row was asked again: {stub.seen}"
    finally:
        owner_auth.close()
