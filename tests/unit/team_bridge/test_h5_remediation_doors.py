"""H5 remediation — the explicit doors matrix (security rows 6/9/10) and the
settlement interleavings (S-5), source-pinned:

- EVERY transport door refuses an empty/differing GENERATION before any POST
  (attempt / retry; recovery's scope+gen refusal lives in
  ``test_h5_intents_provenance.py``) — no unchecked lower-level helper exists;
- the agent tool carries NO decide/retry vocabulary and a retry op never
  validates into one; a ``take_up`` read passes the gates without implicitly
  mutating server state or minting a decide;
- duplicate consumption in EVERY state returns ``created=False`` and schedules
  nothing, and settlement interleavings (success vs transient/signin/refusal)
  keep ``sent``/``refused`` terminal with the ORIGINAL receipt byte-identical.

Fakes: sqlite real (tmp_path); fences/transport substituted at their module
attributes (both are imported at CALL time).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import cast

import pytest

from local_operator.team_bridge import authority as fence_mod
from local_operator.team_bridge import decide as decide_mod
from local_operator.team_bridge import host as host_mod
from local_operator.team_bridge.decide_intents import (
    DecideIntents,
    DecideIntentsRefused,
    attempt_decide_intent,
    retry_from_human_action,
)
from local_operator.team_bridge.tool import BridgeRuntime, body_digest, execute_bridge

pytestmark = pytest.mark.asyncio

OWNER_KEY = json.dumps(["https://hub.example", "acc_1", "ten_1"])


def _runtime(tmp_path) -> BridgeRuntime:
    return cast(
        BridgeRuntime,
        SimpleNamespace(
            owner=SimpleNamespace(
                canonical_base_url="https://hub.example",
                account_id="acc_1",
                tenant_id="ten_1",
                endpoint_id="ep_1",
                credential_id=7,
                enrollment_generation=7,
            ),
            config_dir=tmp_path,
            tenant_id="ten_1",
            session_id="sess-1",
            client=object(),
        ),
    )


@pytest.fixture(autouse=True)
def _fence_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fence_mod, "authority_fence", lambda runtime: None)


def _record(intents: DecideIntents, runtime, *, generation: str | None = None, ask: str = "a-1"):
    return intents.record(
        OWNER_KEY,
        ask_id=ask,
        question_id=f"bmsg_{ask}",
        word="approve",
        generation=str(runtime.owner.enrollment_generation) if generation is None else generation,
        scope_json=json.dumps(fence_mod.scope_from_runtime(runtime)),
        consumption_row_id=f"row_{ask}",
    )


_SETTLE = {
    # The FIRST state an intent can hold is already ``pending`` (the row is
    # written before any transport), so the pending arm is a no-op — a table
    # entry, not a missing key.
    "pending": lambda intents, intent: None,
    "sent": lambda intents, intent: intents.mark_sent(intent.intent_id, {"id": "req_original"}),
    "transient": lambda intents, intent: intents.mark_transient(
        intent.intent_id, "bridge_unreachable"
    ),
    "signin": lambda intents, intent: intents.mark_signin(intent.intent_id),
    "refused": lambda intents, intent: intents.mark_refused(intent.intent_id, "bridge_disabled"),
}


async def test_attempt_and_retry_refuse_empty_generation_before_any_post(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    posted: list[object] = []

    async def _fake_decide(client, decision, **kwargs):
        posted.append(decision)
        return {"request": {}}

    monkeypatch.setattr(decide_mod, "decide_from_human_answer", _fake_decide)
    intents = DecideIntents(tmp_path)
    runtime = _runtime(tmp_path)
    empty, _ = _record(intents, runtime, generation="", ask="a-empty")
    with pytest.raises(DecideIntentsRefused) as exc:
        await attempt_decide_intent(runtime.client, intents, empty, runtime=runtime)
    assert exc.value.code == "owner_changed"
    with pytest.raises(DecideIntentsRefused) as exc2:
        await retry_from_human_action(runtime.client, intents, empty.intent_id, runtime=runtime)
    assert exc2.value.code == "owner_changed"
    assert posted == []  # neither door reached the wire


@pytest.mark.parametrize("state", ["sent", "refused"])
async def test_attempt_and_retry_refuse_settled_states_with_zero_posts(
    tmp_path, monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    posted: list[object] = []

    async def _fake_decide(client, decision, **kwargs):
        posted.append(decision)
        return {"request": {}}

    monkeypatch.setattr(decide_mod, "decide_from_human_answer", _fake_decide)
    intents = DecideIntents(tmp_path)
    runtime = _runtime(tmp_path)
    intent, _ = _record(intents, runtime, ask=f"a-{state}")
    _SETTLE[state](intents, intent)
    with pytest.raises(DecideIntentsRefused) as exc:
        await attempt_decide_intent(runtime.client, intents, intent, runtime=runtime)
    assert exc.value.code == "not_retryable"  # terminal through the raw door too
    with pytest.raises(DecideIntentsRefused) as exc2:
        await retry_from_human_action(runtime.client, intents, intent.intent_id, runtime=runtime)
    assert exc2.value.code == "not_retryable"
    assert posted == []


@pytest.mark.parametrize("state", ["pending", "signin", "sent", "refused"])
async def test_duplicate_consumption_in_every_state_is_not_created(tmp_path, state: str) -> None:
    intents = DecideIntents(tmp_path)
    runtime = _runtime(tmp_path)
    intent, created = _record(intents, runtime)
    assert created is True
    if state != "pending":
        _SETTLE[state](intents, intent)
    again, created_again = _record(intents, runtime)
    assert created_again is False  # the dispatch guard's exact input (S-5)
    assert again.intent_id == intent.intent_id
    assert again.state == state


@pytest.mark.parametrize(
    "first,second,expected,keeps_receipt",
    [
        ("sent", "transient", "sent", True),
        ("sent", "signin", "sent", True),
        ("sent", "refused", "sent", True),
        ("transient", "sent", "sent", True),
        ("signin", "sent", "sent", True),
        ("refused", "sent", "refused", False),
        ("refused", "transient", "refused", False),
        ("pending", "transient", "pending", False),
        ("pending", "signin", "signin", False),
    ],
)
async def test_settlement_interleavings_are_terminal_safe(
    tmp_path, first: str, second: str, expected: str, keeps_receipt: bool
) -> None:
    intents = DecideIntents(tmp_path)
    runtime = _runtime(tmp_path)
    intent, _ = _record(intents, runtime)
    _SETTLE[first](intents, intent)
    settled = intents.get(intent.intent_id)
    _SETTLE[second](intents, intent)
    after = intents.get(intent.intent_id)
    assert after.state == expected
    if keeps_receipt:
        # m8: the recorded outcome's receipt is authoritative and byte-stable
        # through any later transient/signin/refusal — never cleared, never
        # rewritten. A transient first leaves state ``pending`` with NO receipt
        # (that is the settled shape of a transport failure), so the receipt
        # that must survive is the one the SUCCESS settle recorded.
        assert json.loads(after.receipt_json)["id"] == "req_original"
        if settled.receipt_json is not None:
            assert after.receipt_json == settled.receipt_json


async def test_tool_has_no_decide_or_retry_authority(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(host_mod, "owner_fence_refusal", lambda runtime: None)
    monkeypatch.setattr(host_mod, "admission_refusal", lambda runtime, *, is_read=False: None)
    runtime = BridgeRuntime(
        client=object(),
        config_dir=tmp_path,
        tenant_id="ten_1",
        owner=SimpleNamespace(account_id="acc_1", tenant_id="ten_1"),
        session_id="sess-1",
        admission="admitted",
    )
    decided = await execute_bridge(runtime, {"op": "decide", "message_id": "bmsg_1"})
    assert decided["ok"] is False and decided["code"] == "not_permitted"
    retried = await execute_bridge(runtime, {"op": "retry", "message_id": "bmsg_1"})
    assert retried["ok"] is False and retried["code"] == "invalid_request"


async def test_take_up_read_passes_gates_without_minting_a_decide(tmp_path, monkeypatch) -> None:
    posted: list[object] = []

    async def _fake_decide(client, decision, **kwargs):
        posted.append(decision)
        return {"request": {}}

    monkeypatch.setattr(decide_mod, "decide_from_human_answer", _fake_decide)
    monkeypatch.setattr(host_mod, "owner_fence_refusal", lambda runtime: None)
    monkeypatch.setattr(host_mod, "admission_refusal", lambda runtime, *, is_read=False: None)
    text = "the screened body"

    class _Client:
        async def message(self, message_id: str, tenant_id: str | None = None):
            return {"message": {"body": {"text": text}}}

    runtime = BridgeRuntime(
        client=_Client(),
        config_dir=tmp_path,
        tenant_id="ten_1",
        owner=SimpleNamespace(account_id="acc_1", tenant_id="ten_1"),
        session_id="sess-1",
        admission="admitted",
    )
    outcome = await execute_bridge(
        runtime,
        {"op": "take_up", "message_id": "bmsg_1", "body_sha256": body_digest(text)},
    )
    assert outcome["ok"] is True
    assert outcome["text"] == text
    assert posted == []  # a fetch never drives a decide
