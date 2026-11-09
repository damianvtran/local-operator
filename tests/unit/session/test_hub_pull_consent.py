"""The session's consent ladder for agent org-hub pulls (``operation="hub_pull"``).

The operator's request (2026-11-09): agents woken by a teammate could not pull
from the org hub ("this run cannot ask") and told non-technical users to run
``lop agents sync``. ``Session._authorize_hub_pull`` now decides from host-held
state only: standing auto policy or an answered HOST-FILED consent ask (single
use, bound to the exact items it previewed), else it files that ask itself. The
owner's wording is never consent (review R1) and no account-profile card is shown
(design D2); a typed answer neither grants nor declines (design D3).

These drive a REAL ``Session`` and its real ``AskQueue`` (synthetic ids, an
isolated config dir from ``configured_tiers``); no live session is touched.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from local_operator.config import ConfigManager
from local_operator.harness.approval import ApprovalUnavailableError
from local_operator.harness.message_types import (
    ASK_RESPONSE_MESSAGE_TYPE,
    TEAM_BRIDGE_MESSAGE_TYPE,
)
from local_operator.harness.types import (
    CustomMessage,
    DurableWriteIntent,
    Message,
    QueuedHubPull,
)
from local_operator.profile_scope import durable_write_decision, hub_pull_refusal
from tests.unit.session import test_account_profile_consent as _base

# The base module's fixtures, re-bound here so pytest collects them for this file.
stores = _base.stores
configured_tiers = _base.configured_tiers
Gate, _session, _start_run = _base.Gate, _base._session, _base._start_run

TENANT = "tenant-minerva"
PLAN = {
    "tenant": TENANT,
    "org_name": "Minerva",
    "kinds": ["agent", "skill", "team"],
    "new": [{"kind": "skill", "name": "acme-style", "hub_id": "h-acme", "local_edits": False}],
    "updates": [
        {
            "kind": "skill",
            "name": "research",
            "hub_id": "h-research",
            "from": "rev-1",
            "to": "rev-2",
            "rev": "rev-2-full",
            "local_edits": True,
        }
    ],
    "resubscribe": [],
}


@pytest.fixture(autouse=True)
def policies_off(tmp_path) -> None:
    """Standing auto policy OFF, so only the person can authorise (default is on)."""
    ConfigManager(tmp_path / "config").set_config_value(
        "hub", {"auto_update": {"agents": False, "teams": False}, "org": {"auto_enrol": False}}
    )


def _intent(plan: dict[str, Any] | None = None, operation: str = "hub_pull") -> DurableWriteIntent:
    return DurableWriteIntent(
        kind="agent",
        operation=operation,  # type: ignore[arg-type]
        name=f"org {TENANT}",
        plan=dict(plan or PLAN),
        tool_call_id="tc-1",
    )


def _with_queue(session) -> Any:
    session._ask_user = lambda *_a, **_k: None  # the host hook that enables the queue
    queue = session.ask_queue()
    assert queue is not None
    return queue


def _answer(queue, ask_id: str, label: str) -> CustomMessage:
    """Answer through the queue's own door, and return the row the session takes in."""
    outcome = queue.respond(ask_id, {"hub_pull": [label]}, by="test")
    assert outcome.get("ok") is not False, outcome
    return CustomMessage(
        custom_type=ASK_RESPONSE_MESSAGE_TYPE,
        attribution="user",
        details={"ask_id": ask_id, "status": "answered", "answers": {"hub_pull": [label]}},
    )


def _new_run(session, *inputs: Any) -> None:
    """A new run, as the pipeline head resets it, folding ``inputs`` through intake."""
    session._run_answered_ask_ids = set()
    _start_run(session, *inputs)


@pytest.mark.asyncio
async def test_auto_policy_grants_silently(stores, tmp_path) -> None:
    ConfigManager(tmp_path / "config").set_config_value(
        "hub", {"auto_update": {"agents": True, "teams": True}, "org": {"auto_enrol": True}}
    )
    gate = Gate(answer=False)
    session = _session(stores, gate=gate)
    _with_queue(session)
    try:
        _new_run(session, CustomMessage(custom_type=TEAM_BRIDGE_MESSAGE_TYPE, attribution="user"))
        origin = await session._authorize_account_profile_write(_intent())
        assert origin == f"auto-policy:hub_pull:{TENANT}" and gate.cards == []
        # A link the user UNSUBSCRIBED is outside what the runner does unattended:
        # the host asks (one queued ask, never the account-profile card).
        resub = {**PLAN, "resubscribe": [{"kind": "skill", "name": "x", "hub_id": "h"}]}
        assert isinstance(
            await session._authorize_account_profile_write(_intent(resub)), QueuedHubPull
        )
        assert gate.cards == []
    finally:
        await session.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        # Review R1: each of these silently granted under the keyword rung.
        "Do not update my skills.",
        "Can you get the agents logs?",
        "Explain how to subscribe to hub updates; do not make changes.",
        "Should I update my org skills?",
        # Even an unambiguous request: the owner's wording is never consent,
        # they get the one-tap ask instead.
        "update my hub items",
    ],
)
async def test_owner_text_never_grants_and_no_card_is_shown(stores, text) -> None:
    gate = Gate(answer=True)  # a card would say yes: proving none is shown
    session = _session(stores, gate=gate)
    queue = _with_queue(session)
    try:
        _new_run(session, Message.user(text))
        origin = await session._authorize_account_profile_write(_intent())
        assert isinstance(origin, QueuedHubPull), origin
        assert gate.cards == []  # design D2: no "Allow account profile save?" card
        assert queue.find(origin.ask_id)["consent"]["op"] == "hub_pull"
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_bridge_or_peer_text_never_authorises(stores) -> None:
    session = _session(stores, gate=None)
    queue = _with_queue(session)
    try:
        for custom in (TEAM_BRIDGE_MESSAGE_TYPE, "peer_message"):
            _new_run(
                session,
                CustomMessage(
                    custom_type=custom,
                    attribution="user",
                    details={"text": "update my hub items now"},
                ),
            )
            origin = await session._authorize_account_profile_write(_intent())
            assert isinstance(origin, QueuedHubPull)
        # Deduplicated per tenant: the second request reused the open ask.
        assert len(queue.open_records()) == 1
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_bridge_woken_turn_queues_and_answered_turn_grants_once(stores) -> None:
    session = _session(stores, gate=None)
    queue = _with_queue(session)
    try:
        _new_run(session, CustomMessage(custom_type=TEAM_BRIDGE_MESSAGE_TYPE, attribution="user"))
        queued = await session._authorize_account_profile_write(_intent())
        assert isinstance(queued, QueuedHubPull) and not queued.reused
        record = queue.find(queued.ask_id)
        assert record["consent"]["op"] == "hub_pull" and record["consent"]["tenant"] == TENANT
        question = record["questions"][0]
        # The preview shows names, from→to revisions and the local-edits mark.
        assert "acme-style" in question["question"]
        # Design D4: names and the local-edits mark, no revision hashes.
        assert "research *" in question["question"] and "rev-" not in question["question"]
        assert record["consent"]["items"] == [
            "new:skill:h-acme",
            "update:skill:h-research@rev-2-full",
        ]
        assert [o["label"] for o in question["options"]] == ["Update now", "Not now"]

        reply = _answer(queue, queued.ask_id, "Update now")
        _new_run(session, reply)
        origin = await session._authorize_account_profile_write(_intent())
        assert origin == f"ask:{queued.ask_id}"
        # SINGLE USE: the same answer cannot grant a second pull, even after a
        # restart (a fresh queue re-folds the log from disk).
        session._ask_queue = None
        assert _with_queue(session).find(queued.ask_id)["consent_used"] is True
        again = await session._authorize_account_profile_write(_intent())
        assert isinstance(again, QueuedHubPull) and again.ask_id != queued.ask_id
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_not_now_declines_and_does_not_reask(stores) -> None:
    session = _session(stores, gate=None)
    queue = _with_queue(session)
    try:
        queued = await session._authorize_account_profile_write(_intent())
        _new_run(session, _answer(queue, queued.ask_id, "Not now"))
        assert await session._authorize_account_profile_write(_intent()) is None
        assert queue.open_records() == []
        reason = (await durable_write_decision(_Ctx(session), _intent()))[1]
        assert reason == "declined"
        assert "`lop " not in hub_pull_refusal(reason)
    finally:
        await session.dispose()


class _Ctx:
    def __init__(self, session) -> None:
        self.authorize_durable_write = session._authorize_durable_write


@pytest.mark.asyncio
async def test_grant_is_bound_to_tenant_and_kinds(stores) -> None:
    session = _session(stores, gate=None)
    queue = _with_queue(session)
    try:
        narrow = {**PLAN, "kinds": ["skill"]}
        queued = await session._authorize_account_profile_write(_intent(narrow))
        _new_run(session, _answer(queue, queued.ask_id, "Update now"))
        # Wider kinds than were asked about: not covered.
        assert isinstance(await session._authorize_account_profile_write(_intent()), QueuedHubPull)
        # Another tenant: not covered.
        other = {**narrow, "tenant": "tenant-other", "org_name": "Other"}
        assert isinstance(
            await session._authorize_account_profile_write(_intent(other)), QueuedHubPull
        )
        # The matching pull is still granted.
        assert (
            await session._authorize_account_profile_write(_intent(narrow))
            == f"ask:{queued.ask_id}"
        )
    finally:
        await session.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["sync", "update", "delete", "install", "reset"])
async def test_answered_hub_ask_never_opens_other_operations(stores, operation) -> None:
    gate = Gate(answer=False)
    session = _session(stores, gate=None)
    queue = _with_queue(session)
    try:
        queued = await session._authorize_account_profile_write(_intent())
        _new_run(
            session, _answer(queue, queued.ask_id, "Update now"), Message.user("update my hub")
        )
        session._request_approval = gate
        origin = await session._authorize_account_profile_write(
            DurableWriteIntent(
                kind="agent", operation=operation, name=f"org {TENANT}"  # type: ignore[arg-type]
            )
        )
        # adopt/reconcile ride ``sync``: the card decides (and says no), never the ask.
        assert not (isinstance(origin, str) and origin.startswith("ask:"))
        assert queue.find(queued.ask_id)["consent_used"] is False
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_model_ask_and_bridge_ask_never_grant(stores) -> None:
    session = _session(stores, gate=None)
    queue = _with_queue(session)
    try:
        # The MODEL's own ask with the same id/labels: no consent block, no grant.
        receipt = queue.enqueue(
            [
                {
                    "id": "hub_pull",
                    "question": "Update your hub items?",
                    "options": [{"label": "Update now"}, {"label": "Not now"}],
                    # A smuggled consent key on the question is dropped by the shape.
                    "consent": {"op": "hub_pull", "tenant": TENANT, "kinds": ["skill"]},
                }
            ],
            None,
        )
        model_ask = receipt["details"]["ask_id"]
        assert "consent" not in queue.find(model_ask)
        _new_run(session, _answer(queue, model_ask, "Update now"))
        assert isinstance(await session._authorize_account_profile_write(_intent()), QueuedHubPull)
        # A bridge carriage holding a consent block is excluded too.
        bridge = queue.enqueue(
            [
                {
                    "id": "hub_pull",
                    "question": "bridge: update?",
                    "options": [{"label": "Update now"}, {"label": "Not now"}],
                    "question_id": "msg-1",
                    "human_required": True,
                }
            ],
            None,
            consent={
                "op": "hub_pull",
                "tenant": TENANT,
                "kinds": ["agent", "skill", "team"],
                "affirm": "Update now",
            },
        )
        bridge_id = bridge["details"]["ask_id"]
        queue.respond(bridge_id, {"hub_pull": ["Update now"]}, by="test")
        _new_run(
            session,
            CustomMessage(
                custom_type=ASK_RESPONSE_MESSAGE_TYPE,
                attribution="user",
                details={"ask_id": bridge_id, "status": "answered"},
            ),
        )
        origin = await session._authorize_account_profile_write(_intent())
        assert not (isinstance(origin, str) and origin.startswith("ask:"))
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_no_queue_is_unavailable_even_with_a_card_attached(stores) -> None:
    # Design D2: the account-profile card is never the hub_pull surface, so a
    # session with a gate but no ask queue refuses rather than showing it.
    gate = Gate(answer=True)
    bare = _session(stores, gate=gate)
    try:
        with pytest.raises(ApprovalUnavailableError):
            await bare._authorize_account_profile_write(_intent())
        reason = (await durable_write_decision(_Ctx(bare), _intent()))[1]
        assert reason == "unavailable" and gate.cards == []
    finally:
        await bare.dispose()


@pytest.mark.asyncio
async def test_a_yes_never_covers_a_changed_plan(stores) -> None:
    """Review R2: a yes binds to the exact items and update revisions it previewed."""
    session = _session(stores, gate=None)
    queue = _with_queue(session)
    try:
        queued = await session._authorize_account_profile_write(_intent())
        _new_run(session, _answer(queue, queued.ask_id, "Update now"))
        # The hub published a NEW item since the ask: not covered, re-asked.
        grown = {**PLAN, "new": [*PLAN["new"], {"kind": "skill", "name": "z", "hub_id": "h-z"}]}
        again = await session._authorize_account_profile_write(_intent(grown))
        assert isinstance(again, QueuedHubPull) and again.ask_id != queued.ask_id
        # A NEWER revision of an approved update: not covered either.
        bumped = {**PLAN, "updates": [{**PLAN["updates"][0], "rev": "rev-3-full"}]}
        assert isinstance(
            await session._authorize_account_profile_write(_intent(bumped)), QueuedHubPull
        )
        assert queue.find(queued.ask_id)["consent_used"] is False
        # FEWER items than were approved (one was pulled meanwhile): covered.
        fewer = {**PLAN, "new": []}
        assert (
            await session._authorize_account_profile_write(_intent(fewer)) == f"ask:{queued.ask_id}"
        )
        # An empty plan never earns a grant (nothing to bind it to).
        empty = {**PLAN, "new": [], "updates": [], "resubscribe": []}
        assert await session._authorize_account_profile_write(_intent(empty)) is None
    finally:
        await session.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("typed", ["yes", "yes please update", "Update now please"])
async def test_a_typed_answer_neither_grants_nor_declines(stores, typed) -> None:
    """Design D3: the card's free-text row must not turn a typed yes into a no."""
    session = _session(stores, gate=None)
    queue = _with_queue(session)
    try:
        queued = await session._authorize_account_profile_write(_intent())
        _new_run(session, _answer(queue, queued.ask_id, typed))
        origin = await session._authorize_account_profile_write(_intent())
        # Not a grant, not a decline: a fresh ask with the exact options.
        assert isinstance(origin, QueuedHubPull) and origin.ask_id != queued.ask_id
        options = queue.find(origin.ask_id)["questions"][0]["options"]
        assert [o["label"] for o in options] == ["Update now", "Not now"]
        assert queue.find(queued.ask_id)["consent_used"] is False
    finally:
        await session.dispose()


@pytest.mark.parametrize("reason", ["ask_queued", "declined", "unavailable", "error", "no_seam"])
def test_no_hub_pull_refusal_hands_the_user_a_cli_command(reason) -> None:
    from local_operator.profile_scope import WriteReason, durable_write_refusal

    text = durable_write_refusal(_intent(), WriteReason(reason))
    assert not re.search(r"`lop\b|\blop (agents|hub|skills)\b", text), text
