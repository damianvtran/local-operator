"""The turn-side BODY consumer — the per-call invariant, against a real store.

Every case here drives the REAL ``Store`` (sqlite on disk) through the module
the Session calls, because the invariant is about durable state transitions:
a green double would not prove a claim was released or a row stayed pending.

The aside rule is asserted structurally at the end: the aside builder never
consults the pending turn, so a frame cannot enter a compaction request.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.team_bridge.pending_turn import FetchedBody, PendingTurn
from local_operator.team_bridge.request_body import BodyBudget
from local_operator.team_bridge.store import (
    OwnerReceipt,
    OwnerSnapshot,
    Store,
    StoreRefused,
    state_db_path,
)

CONVERSATION_ID = "bcv_abcdefgh"
MESSAGE_ID = "bcmsg_abcdef01"
DIGEST = "9f2a" * 16
FINGERPRINT = "aa11" * 16
SESSION = "sess_cos"
BODY_TEXT = "canary body from the channel"


def _owner() -> OwnerSnapshot:
    return OwnerSnapshot(
        credential_id=7,
        account_id="acc_1",
        tenant_id="3f2a1111-2222-4333-8444-555566667777",
        endpoint_id="ep_1",
        canonical_base_url="https://hub.example",
        enrollment_generation=1,
    )


def _store(tmp_path: Path) -> tuple[Store, int]:
    store = Store(tmp_path)
    store.commit_enrollment(
        _owner(), OwnerReceipt("bridge_consent_v1", "ref-1"), cos_session_id=SESSION
    )
    # Use the store's OWN trusted publisher rather than raw SQL: the policy
    # row's shape is the store's business, and a hand-built row would let this
    # test pass against a schema the real path cannot write.
    epoch = store.publish_screening_policy(
        _owner(),
        version="v1",
        template_sha256="ab" * 32,
        default_model="qa-default",
        revision="rev-1",
    )
    return store, epoch


def _reserve(store: Store, epoch: int) -> bool:
    return store.record_body_reservation(
        _owner(),
        conversation_id=CONVERSATION_ID,
        message_id=MESSAGE_ID,
        conversation_seq=3,
        mention=True,
        body_sha256=DIGEST,
        screening_fingerprint=FINGERPRINT,
        policy_epoch=epoch,
        consent_version="bridge_consent_v1",
        designated_session=SESSION,
        now=time.time(),
    )


def _raw_row(tmp_path: Path) -> dict[str, Any]:
    """The row WITHOUT the live-generation fence.

    Needed after a re-login: every fenced reader correctly answers None for the
    dead generation, so the orphan's own state can only be inspected directly.
    """

    with sqlite3.connect(state_db_path(tmp_path)) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM body_reservations WHERE message_id = ?", (MESSAGE_ID,)
        ).fetchone()
    assert row is not None
    return dict(row)


def _row(store: Store) -> dict[str, Any]:
    row = store.body_reservation(_owner(), CONVERSATION_ID, MESSAGE_ID)
    assert row is not None
    return row


async def _fetch(conversation_id: str, message_id: str) -> FetchedBody | None:
    return FetchedBody(
        body=BODY_TEXT,
        body_sha256=DIGEST,
        conversation_label="#qa",
        sender_label="Ada",
    )


def _budget() -> BodyBudget:
    return BodyBudget(available_bytes=100_000, measured_from="test budget")


def _turn(tmp_path: Path) -> PendingTurn:
    return PendingTurn(_owner(), SESSION, config_dir=tmp_path)


def test_live_host_binding_is_rechecked_before_each_body_injection(tmp_path) -> None:
    store, epoch = _store(tmp_path)
    assert _reserve(store, epoch)
    binding = {"current": True}
    turn = PendingTurn(
        _owner(),
        SESSION,
        config_dir=tmp_path,
        binding_guard=lambda: binding["current"],
    )
    assert asyncio.run(turn.open(budget=_budget(), fetch=_fetch))
    assert turn.for_call() is not None
    binding["current"] = False
    assert store.current_generation_is_live(_owner()), "captured owner still valid in SQLite"
    assert turn.for_call() is None
    row = _row(store)
    assert row["state"] == "pending" and row["claim_token"] is None
    assert row["last_error"] == "admission_drift"


def test_selected_neighbor_does_not_reclaim_failed_oldest_or_change_its_reason(tmp_path) -> None:
    store, epoch = _store(tmp_path)
    assert _reserve(store, epoch)
    second_id = "bcmsg_abcdef02"
    assert store.record_body_reservation(
        _owner(),
        conversation_id=CONVERSATION_ID,
        message_id=second_id,
        conversation_seq=4,
        mention=False,
        body_sha256=DIGEST,
        screening_fingerprint=FINGERPRINT,
        policy_epoch=epoch,
        consent_version="bridge_consent_v1",
        designated_session=SESSION,
        now=time.time(),
    )
    first = _turn(tmp_path)
    assert not asyncio.run(
        first.open(
            budget=BodyBudget(available_bytes=1, measured_from="bounded capacity control"),
            fetch=_fetch,
        )
    )
    failed = _row(store)
    assert failed["state"] == "pending" and failed["last_error"] == "body_not_fitted"
    candidate = store.pending_body_reservation_for(
        _owner(),
        SESSION,
        exclude=frozenset({(CONVERSATION_ID, MESSAGE_ID)}),
        admitted_before=time.time(),
    )
    assert candidate is not None and candidate["message_id"] == second_id
    second = _turn(tmp_path)
    assert asyncio.run(
        second.open(
            budget=_budget(),
            fetch=_fetch,
            selected=(CONVERSATION_ID, second_id),
        )
    )
    assert _row(store) == failed, "failed A was not rewritten to make B selectable"
    claimed = store.body_reservation(_owner(), CONVERSATION_ID, second_id)
    assert claimed is not None and claimed["state"] == "claimed"
    second.settle(entry_id=None, error=None, stop_reason=None, aborted=False)


def test_cancelled_hydration_releases_its_claim_without_waiting_for_expiry(tmp_path) -> None:
    store, epoch = _store(tmp_path)
    assert _reserve(store, epoch)
    turn = _turn(tmp_path)

    async def exercise() -> None:
        entered = asyncio.Event()

        async def suspended_fetch(conversation_id, message_id):
            entered.set()
            await asyncio.Event().wait()

        task = asyncio.create_task(turn.open(budget=_budget(), fetch=suspended_fetch))
        await asyncio.wait_for(entered.wait(), 2)
        assert _row(store)["state"] == "claimed"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())
    row = _row(store)
    assert row["state"] == "pending"
    assert row["claim_token"] is None
    assert turn.has_claim is False


def test_a_pending_row_is_claimed_and_its_frame_cached(tmp_path) -> None:
    store, epoch = _store(tmp_path)
    assert _reserve(store, epoch)
    turn = _turn(tmp_path)

    assert asyncio.run(turn.open(budget=_budget(), fetch=_fetch)) is True
    assert _row(store)["state"] == "claimed"

    text = turn.for_call()
    assert text is not None and "END UNTRUSTED DATA OBJECT" in text
    assert BODY_TEXT in text
    # The SAME bytes on the next physical call of this turn (S-1): re-rendering
    # would be a second frame rather than a re-send of the admitted one.
    assert turn.for_call() == text
    assert turn.appended_calls == 2


def test_settlement_refuses_to_complete_a_turn_that_never_appended(tmp_path) -> None:
    """S-16a: the silent-loss guard. A successful verdict is not enough."""

    store, epoch = _store(tmp_path)
    assert _reserve(store, epoch)
    turn = _turn(tmp_path)
    assert asyncio.run(turn.open(budget=_budget(), fetch=_fetch)) is True

    outcome = turn.settle(entry_id="entry_1", error=None, stop_reason="stop", aborted=False)

    assert outcome == "never_appended"
    row = _row(store)
    assert row["state"] == "pending" and row["last_error"] == "never_appended"


def test_drifted_consent_omits_the_frame_and_leaves_the_row_pending(tmp_path) -> None:
    """S-21: a withdrawal after the first call must stop the very next one."""

    store, epoch = _store(tmp_path)
    assert _reserve(store, epoch)
    turn = _turn(tmp_path)
    assert asyncio.run(turn.open(budget=_budget(), fetch=_fetch)) is True
    assert turn.for_call() is not None

    with sqlite3.connect(state_db_path(tmp_path)) as conn:
        conn.execute("UPDATE enrollments SET consent_version = 'bridge_consent_v2'")

    assert turn.for_call() is None, "a drifted consent must not carry the body"
    outcome = turn.settle(entry_id="entry_1", error=None, stop_reason="stop", aborted=False)
    assert outcome == "drifted"
    row = _row(store)
    assert row["state"] == "pending" and row["last_error"] == "admission_drift"


def test_a_lost_lease_stops_injection_and_reports_claim_lost(tmp_path) -> None:
    """S-12: a SECOND consumer that reclaims an expired row takes the lease.

    Expiry alone does not steal a live claim — only a competing claimant does,
    which is why the test drives a real second PendingTurn rather than editing
    the expiry column and hoping.
    """

    store, epoch = _store(tmp_path)
    assert _reserve(store, epoch)
    first = _turn(tmp_path)
    assert asyncio.run(first.open(budget=_budget(), fetch=_fetch)) is True

    with sqlite3.connect(state_db_path(tmp_path)) as conn:
        conn.execute("UPDATE body_reservations SET claim_expires_at = ?", (time.time() - 1,))

    second = _turn(tmp_path)
    assert asyncio.run(second.open(budget=_budget(), fetch=_fetch)) is True

    assert first.for_call() is None, "a turn that lost its lease must send without"
    assert second.for_call() is not None
    assert (
        first.settle(entry_id="entry_1", error=None, stop_reason="stop", aborted=False)
        == "claim_lost"
    )


def test_a_successful_turn_completes_and_clears_the_claim_tokens(tmp_path) -> None:
    store, epoch = _store(tmp_path)
    assert _reserve(store, epoch)
    turn = _turn(tmp_path)
    assert asyncio.run(turn.open(budget=_budget(), fetch=_fetch)) is True
    assert turn.for_call() is not None

    outcome = turn.settle(entry_id="entry_1", error=None, stop_reason="stop", aborted=False)

    assert outcome == "completed"
    row = _row(store)
    assert row["state"] == "completed"
    assert row["completed_entry_id"] == "entry_1"
    assert row["claim_token"] is None and row["turn_token"] is None


def test_an_aborted_or_failed_turn_leaves_the_row_retryable(tmp_path) -> None:
    store, epoch = _store(tmp_path)
    assert _reserve(store, epoch)
    turn = _turn(tmp_path)
    assert asyncio.run(turn.open(budget=_budget(), fetch=_fetch)) is True
    assert turn.for_call() is not None

    assert (
        turn.settle(entry_id="entry_1", error="boom", stop_reason="error", aborted=False)
        == "unverified"
    )
    assert _row(store)["state"] == "pending"


def test_a_real_relogin_does_not_serve_and_retires_the_old_generation(tmp_path) -> None:
    """B1: a REAL second enrollment, not a synthetic owner.

    The first version of this test built an owner that was never enrolled, so
    ``_live_generation`` returned None before the row query and the test passed
    against the pre-fix reader. Here the store actually commits credential 8 at
    generation 2, so the only thing that can exclude the old row is the binding
    predicate — and the orphan is retired rather than parked.
    """

    store, epoch = _store(tmp_path)
    assert _reserve(store, epoch)

    relogged = OwnerSnapshot(
        credential_id=8,
        account_id="acc_1",
        tenant_id="3f2a1111-2222-4333-8444-555566667777",
        endpoint_id="ep_1",
        canonical_base_url="https://hub.example",
        enrollment_generation=2,
    )
    store.commit_enrollment(
        relogged, OwnerReceipt("bridge_consent_v1", "ref-2"), cos_session_id=SESSION
    )

    # The orphan is terminal, not pending: nothing can ever claim it.
    orphan = _raw_row(tmp_path)
    assert orphan["state"] == "retired"
    assert orphan["terminal_code"] == "binding_mismatch"
    assert store.pending_body_reservation_for(relogged, SESSION) is None

    turn = PendingTurn(relogged, SESSION, config_dir=tmp_path)
    assert asyncio.run(turn.open(budget=_budget(), fetch=_fetch)) is False


def test_the_tool_boundary_renewal_keeps_the_lease(tmp_path) -> None:
    """M2: the second renewal site — the tool batch — actually extends it.

    Without this, a long tool run lets the lease expire, a second turn claims
    the row and the same body is served twice (S-12).
    """

    store, epoch = _store(tmp_path)
    assert _reserve(store, epoch)
    turn = _turn(tmp_path)
    assert asyncio.run(turn.open(budget=_budget(), fetch=_fetch)) is True

    with sqlite3.connect(state_db_path(tmp_path)) as conn:
        conn.execute("UPDATE body_reservations SET claim_expires_at = ?", (time.time() + 5,))

    assert turn.renew() is True
    row = _row(store)
    assert row["state"] == "claimed"
    assert float(row["claim_expires_at"]) > time.time() + 60, "the lease must extend"


def test_the_lease_ttl_is_floored_and_capped(tmp_path) -> None:
    """M2: a caller may extend the lease, never shrink or pin it forever."""

    floored = PendingTurn(_owner(), SESSION, config_dir=tmp_path, lease_ttl_seconds=60)
    assert floored._ttl == 8 * 60 * 60
    capped = PendingTurn(_owner(), SESSION, config_dir=tmp_path, lease_ttl_seconds=10**9)
    assert capped._ttl == 24 * 60 * 60


def test_a_store_refusal_on_renew_is_drift_not_a_stolen_lease(tmp_path, monkeypatch) -> None:
    """m2: a busy store must not be reported as somebody else's claim."""

    store, epoch = _store(tmp_path)
    assert _reserve(store, epoch)
    turn = _turn(tmp_path)
    assert asyncio.run(turn.open(budget=_budget(), fetch=_fetch)) is True

    def busy(*_args: Any, **_kwargs: Any) -> bool:
        raise StoreRefused("state_unreadable", "busy")

    monkeypatch.setattr(Store, "renew_body_claim", busy)
    assert turn.for_call() is None
    assert (
        turn.settle(entry_id="entry_1", error=None, stop_reason="stop", aborted=False) == "drifted"
    )
    assert _row(store)["last_error"] == "admission_drift"


def test_a_hydrated_frame_carries_the_closed_sender_kind(tmp_path, monkeypatch) -> None:
    """M1: the kind the hub reported reaches the frame's trusted section."""

    store, epoch = _store(tmp_path)
    assert _reserve(store, epoch)

    async def agent_fetch(conversation_id: str, message_id: str) -> FetchedBody | None:
        return FetchedBody(
            body=BODY_TEXT,
            body_sha256=DIGEST,
            conversation_label="#qa",
            sender_label="Ada",
            sender_kind="agent",
        )

    turn = _turn(tmp_path)
    assert asyncio.run(turn.open(budget=_budget(), fetch=agent_fetch)) is True
    text = turn.for_call()
    assert text is not None
    assert "sender_kind: agent" in text


def test_the_aside_builder_never_consults_the_pending_turn() -> None:
    """The privacy boundary is structural: the aside path has no frame input.

    ``_read_only_prompt`` is the only builder for asides and compaction, and it
    reads ``_wire_legal_snapshot`` — not the turn's claim. If a future change
    routes a frame into an aside, this assertion is what fails first.
    """

    import inspect

    from local_operator.session import session as session_mod

    source = inspect.getsource(session_mod.Session._read_only_prompt)
    assert "_pending_body_turn" not in source
    assert "transform_context" not in source


# -- a routed reservation: the session that asked reads it ---------------------


def _reserve_routed(store: Store, epoch: int, routed: str) -> bool:
    return store.record_body_reservation(
        _owner(),
        conversation_id=CONVERSATION_ID,
        message_id=MESSAGE_ID,
        conversation_seq=3,
        mention=False,
        body_sha256=DIGEST,
        screening_fingerprint=FINGERPRINT,
        policy_epoch=epoch,
        consent_version="bridge_consent_v1",
        designated_session=SESSION,
        now=time.time(),
        routed_session=routed,
    )


def test_the_routed_session_claims_serves_and_completes_the_row(tmp_path) -> None:
    store, epoch = _store(tmp_path)
    assert _reserve_routed(store, epoch, "sess_task")
    designated = _turn(tmp_path)
    routed = PendingTurn(_owner(), "sess_task", config_dir=tmp_path)

    # The designated session is blind to a row routed elsewhere.
    assert asyncio.run(designated.open(budget=_budget(), fetch=_fetch)) is False
    assert _row(store)["state"] == "pending"

    assert asyncio.run(routed.open(budget=_budget(), fetch=_fetch)) is True
    text = routed.for_call()
    assert text is not None and BODY_TEXT in text
    assert (
        routed.settle(entry_id="e1", error=None, stop_reason="stop", aborted=False) == "completed"
    )
    assert _row(store)["state"] == "completed"


def test_a_designation_move_makes_the_routed_turn_drift_and_release(tmp_path) -> None:
    store, epoch = _store(tmp_path)
    assert _reserve_routed(store, epoch, "sess_task")
    routed = PendingTurn(_owner(), "sess_task", config_dir=tmp_path)
    assert asyncio.run(routed.open(budget=_budget(), fetch=_fetch)) is True

    store.update_designated_session(_owner(), "sess_new")

    assert routed.for_call() is None  # the per-call fence no longer holds


def test_pending_turn_anchors_frame_index_across_mid_turn_tool_calls(tmp_path) -> None:
    """REGRESSION (cache): Call 1 pins the ephemeral frame index right after
    pre-turn history; Call 2+ (after assistant tool_use and tool_result are
    appended) keeps the frame at that anchored index instead of teleporting it
    to ``messages[-1]`` and busting the prefix cache."""
    from local_operator.harness.render import _default_convert_to_llm
    from local_operator.harness.types import Message, TextContent, ToolCall
    from local_operator.team_bridge.card_projection import presented_bridge_ref

    store, epoch = _store(tmp_path)
    assert _reserve(store, epoch)
    turn = _turn(tmp_path)
    assert asyncio.run(turn.open(budget=_budget(), fetch=_fetch)) is True

    card = Message.user("wake card")
    call1_input = [card]
    call1 = turn.transform(call1_input)
    assert len(call1_input) == 1, "transform must not mutate its input list"
    assert len(call1) == 2
    assert call1[0] is card
    frame1 = call1[1]
    assert getattr(frame1, "_ephemeral", False) is True
    assert "_ephemeral" not in frame1.model_dump()
    assert BODY_TEXT in frame1.details["text"]

    # Mid-turn: assistant calls a tool and receives a result.
    asst = Message(
        role="assistant",
        tool_calls=[ToolCall(id="c1", name="noop", arguments={})],
    )
    tool = Message(
        role="tool",
        tool_call_id="c1",
        tool_name="noop",
        content=[TextContent(text="ok")],
    )
    call2_input = [card, asst, tool]
    call2 = turn.transform(call2_input)
    assert len(call2_input) == 3, "transform must not mutate its input list"
    assert len(call2) == 4
    # Frame stays anchored at index 1 — BEFORE the assistant/tool messages.
    assert call2[0] is card
    frame2 = call2[1]
    assert call2[2] is asst
    assert call2[3] is tool
    assert frame2.id == frame1.id
    assert frame2.details == frame1.details
    assert getattr(frame2, "_ephemeral", False) is True

    # Rendered Messages propagate _ephemeral and preserve admission proof.
    rendered1 = _default_convert_to_llm(call1)
    rendered2 = _default_convert_to_llm(call2)
    assert getattr(rendered1[1], "_ephemeral", False) is True
    assert getattr(rendered2[1], "_ephemeral", False) is True
    assert "_ephemeral" not in rendered1[1].model_dump()
    assert [m.model_dump() for m in rendered2[:2]] == [m.model_dump() for m in rendered1]
    assert presented_bridge_ref(rendered1) is not None
    assert presented_bridge_ref(rendered2) == presented_bridge_ref(rendered1)

    # Mid-turn history shrink (e.g., compaction) clamps cleanly without IndexError.
    shrunk = turn.transform([])
    assert len(shrunk) == 1 and BODY_TEXT in shrunk[0].details["text"]


# -- anchor survives mid-turn compaction (PR #4370 round 1: R1 / Q1) ----------


def _asst_calls(*ids: str) -> Any:
    from local_operator.harness.types import Message, ToolCall

    return Message(
        role="assistant", tool_calls=[ToolCall(id=i, name="noop", arguments={}) for i in ids]
    )


def _tool_reply(call_id: str) -> Any:
    from local_operator.harness.types import Message, TextContent

    return Message(
        role="tool", tool_call_id=call_id, tool_name="noop", content=[TextContent(text="ok")]
    )


def _frame_positions(shaped: list[Any]) -> list[int]:
    return [i for i, m in enumerate(shaped) if getattr(m, "custom_type", None)]


def _wire_pairing_failures(shaped: list[Any]) -> list[str]:
    """Build BOTH real wire bodies and check every tool call is answered next.

    Anthropic: an assistant ``tool_use`` must be followed by a user message
    whose FIRST block is its ``tool_result``. OpenAI-compat: assistant
    ``tool_calls`` must be followed by ``tool`` messages. Either violation is a
    provider 400, which is the failure mode the index anchor produced.
    """
    from local_operator.harness.render import _default_convert_to_llm
    from local_operator.harness.types import ChatRequest, ModelSpec
    from local_operator.providers.clients import AnthropicClient, OpenAICompatClient

    rendered = _default_convert_to_llm(shaped)
    anthropic = AnthropicClient()._build_body(
        ChatRequest(
            model=ModelSpec(provider="anthropic", model_id="claude-opus-5", context_window=200_000),
            system_blocks=["s"],
            messages=rendered,
        )
    )
    openai = OpenAICompatClient("https://api.radienthq.com/v1")._build_body(
        ChatRequest(
            model=ModelSpec(
                provider="radient",
                model_id="anthropic/claude-sonnet-5.5",
                context_window=200_000,
                supports_prompt_cache=True,
            ),
            system_blocks=["s"],
            messages=rendered,
        )
    )
    failures: list[str] = []
    msgs = anthropic["messages"]
    for i, m in enumerate(msgs):
        if m["role"] == "assistant" and any(b.get("type") == "tool_use" for b in m["content"]):
            nxt = msgs[i + 1] if i + 1 < len(msgs) else {}
            first = (nxt.get("content") or [{}])[0]
            if nxt.get("role") != "user" or first.get("type") != "tool_result":
                failures.append(f"anthropic[{i}]")
    msgs = openai["messages"]
    for i, m in enumerate(msgs):
        if m["role"] == "assistant" and m.get("tool_calls"):
            nxt = msgs[i + 1] if i + 1 < len(msgs) else {}
            if nxt.get("role") != "tool":
                failures.append(f"openai_compat[{i}]")
    return failures


def _open_turn(tmp_path) -> PendingTurn:
    store, epoch = _store(tmp_path)
    assert _reserve(store, epoch)
    turn = _turn(tmp_path)
    assert asyncio.run(turn.open(budget=_budget(), fetch=_fetch)) is True
    return turn


def test_compaction_swap_never_splits_a_tool_pair(tmp_path) -> None:
    """REGRESSION (R1/Q1): Call 1 sees ``[prior, wake]`` (old anchor index 2);
    mid-turn compaction then hands a NEW, longer list whose index 2 sits
    between ``tool_use c5`` and its result. The frame must not land there:
    its predecessor (the wake card) is gone, so it goes last."""
    from local_operator.harness.types import Message

    turn = _open_turn(tmp_path)
    call1 = turn.transform([Message.user("prior"), Message.user("wake card")])
    assert _frame_positions(call1) == [2]

    compacted = [
        Message.user("[compaction summary]"),
        _asst_calls("c5"),
        _tool_reply("c5"),
        _asst_calls("c6"),
        _tool_reply("c6"),
    ]
    shaped = turn.transform(compacted)
    assert _frame_positions(shaped) == [5]
    assert _wire_pairing_failures(shaped) == []
    # The QA shape: the compacted list KEEPS a wake card, but a different one.
    shaped = turn.transform([Message.user("[summary]"), Message.user("wake card"), *compacted[1:]])
    assert _frame_positions(shaped) == [6]
    assert _wire_pairing_failures(shaped) == []


def test_anchor_follows_its_predecessor_when_compaction_keeps_it(tmp_path) -> None:
    """A compaction that keeps the wake card keeps the frame right after it,
    at its NEW position, and by ``id`` even when the caller re-copied it."""
    from local_operator.harness.types import Message

    turn = _open_turn(tmp_path)
    wake = Message.user("wake card")
    turn.transform([Message.user("prior"), Message.user("older"), wake])

    kept = [Message.user("[summary]"), wake.model_copy(), _asst_calls("c5"), _tool_reply("c5")]
    shaped = turn.transform(kept)
    assert _frame_positions(shaped) == [2]
    assert shaped[1] is kept[1]
    assert _wire_pairing_failures(shaped) == []


def test_anchor_never_lands_inside_a_parallel_tool_batch(tmp_path) -> None:
    """Predecessor followed by a tool result (a parallel batch answered in two
    rows) or predecessor carrying tool calls: either insert would split a
    pair, so the frame goes last instead."""
    from local_operator.harness.types import Message

    turn = _open_turn(tmp_path)
    first_reply = _tool_reply("a")
    turn.transform([Message.user("u"), _asst_calls("a", "b"), first_reply])
    batch = [Message.user("u"), _asst_calls("a", "b"), first_reply, _tool_reply("b")]
    shaped = turn.transform(batch)
    assert _frame_positions(shaped) == [4]
    assert _wire_pairing_failures(shaped) == []

    turn2 = _open_turn(tmp_path / "second")
    caller = _asst_calls("x")
    turn2.transform([Message.user("u"), caller])
    shaped = turn2.transform([Message.user("u"), caller, _tool_reply("x")])
    assert _frame_positions(shaped) == [3]


def test_anchor_holds_a_stable_prefix_across_ordinary_tool_steps(tmp_path) -> None:
    """No compaction: every tool step keeps the frame at the same position and
    the request prefix through the frame byte-identical (the cache saving)."""
    from local_operator.harness.types import Message

    turn = _open_turn(tmp_path)
    history = [Message.user("prior"), Message.user("wake card")]
    shapes = [turn.transform(list(history))]
    for step in range(3):
        history += [_asst_calls(f"c{step}"), _tool_reply(f"c{step}")]
        shapes.append(turn.transform(list(history)))
    assert {tuple(_frame_positions(s)) for s in shapes} == {(2,)}
    prefix = [m.model_dump() for m in shapes[0][:3]]
    assert all([m.model_dump() for m in s[:3]] == prefix for s in shapes)
    assert all(_wire_pairing_failures(s) == [] for s in shapes)


@pytest.mark.parametrize(
    ("kind", "carries_rule"),
    [("channel", True), ("group", True), ("dm", False), (None, False), ("bogus", False)],
)
def test_a_hydrated_frame_carries_the_participation_rule_only_for_channels(
    tmp_path, kind, carries_rule
) -> None:
    """The fetched body's conversation kind reaches the frame (operator
    2026-11-20: every agent answered every channel post). ``sender_kind="agent"``
    also exercises ``assemble_injection``'s source rebuild, which must keep the
    kind; an unexpected kind degrades to "did not say", never a refusal."""

    from local_operator.harness.message_types import BRIDGE_CHANNEL_PARTICIPATION_RULE

    store, epoch = _store(tmp_path)
    assert _reserve(store, epoch)

    async def kind_fetch(conversation_id: str, message_id: str) -> FetchedBody | None:
        return FetchedBody(
            body=BODY_TEXT,
            body_sha256=DIGEST,
            conversation_label="#qa",
            sender_label="Ada",
            sender_kind="agent",
            conversation_kind=kind,
        )

    turn = _turn(tmp_path)
    assert asyncio.run(turn.open(budget=_budget(), fetch=kind_fetch)) is True
    text = turn.for_call()
    assert text is not None
    assert "sender_kind: agent" in text
    assert (BRIDGE_CHANNEL_PARTICIPATION_RULE in text) is carries_rule
