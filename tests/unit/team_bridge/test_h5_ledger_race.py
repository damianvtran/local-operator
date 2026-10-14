"""H5 — the two-task settlement race on the REAL ledger (SOURCE).

``attempt_decide_intent`` is driven by TWO concurrent tasks that BOTH enter the
transport before either settles (an ``asyncio.Event`` barrier in the patched
``decide.decide_from_human_answer``). The contract: the recorded ORIGINAL
receipt is byte-stable — a success followed by a late transient, and a
transient followed by a later success, both leave ONE canonical receipt; the
transient never downgrades a settled ``sent`` row (terminal, m8) and the late
success settles the same canonical record. The fixtures are the provenance
suite's own (``_record``/``_runtime``), so both files bind the same real
ledger construction.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from local_operator.team_bridge import authority as fence_mod
from local_operator.team_bridge import decide as decide_mod
from local_operator.team_bridge.client import BridgeSignInRequired, BridgeUnreachable
from local_operator.team_bridge.decide_intents import DecideIntents, attempt_decide_intent
from tests.unit.team_bridge.test_h5_intents_provenance import _record, _runtime

LATE_ERRORS = [BridgeUnreachable, BridgeSignInRequired]


@pytest.fixture(autouse=True)
def _fence_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same boundary substitution as the provenance suite (whose ``_record`` and
    ``_runtime`` this file borrows): ``attempt_decide_intent`` consults the
    shared offline fence at CALL time, and a bare tmp_path has no
    enrollment/consent/feature flag, so the real fence answers ``feature_off``
    and the race never reaches the ledger it exists to exercise. Autouse
    fixtures do not travel with an imported helper, hence the local copy."""

    monkeypatch.setattr(fence_mod, "authority_fence", lambda runtime: None)


@pytest.mark.parametrize("late_error", LATE_ERRORS, ids=["transient", "signin"])
@pytest.mark.asyncio
async def test_a_late_transient_or_signin_cannot_downgrade_a_settled_receipt(
    tmp_path, monkeypatch: pytest.MonkeyPatch, late_error
) -> None:
    intents = DecideIntents(tmp_path)
    runtime = _runtime(tmp_path)
    intent, _ = _record(intents, runtime)
    both_entered = asyncio.Event()
    entered = {"n": 0}

    async def barred(client, decision, **kwargs):
        entered["n"] += 1
        index = entered["n"]
        if index >= 2:
            both_entered.set()
        # BOTH transports must be INSIDE the call before ANY settles: the
        # barrier — not the arrival of a first return — is the overlap
        # evidence Root requires.
        await asyncio.wait_for(both_entered.wait(), 2.0)
        if index == 2:
            return {"request": {"receipt": "orig-1"}}  # success after both entered
        # The late TRANSIENT/SIGNIN arm releases only after the success
        # DUPLICABLY settled — the downgrade shape, not a sequential replay.
        while str(getattr(intents.get(intent.intent_id), "state", "")) != "sent":
            await asyncio.sleep(0.01)
        raise late_error("late settlement")

    monkeypatch.setattr(decide_mod, "decide_from_human_answer", barred)
    first = asyncio.create_task(attempt_decide_intent(runtime.client, intents, intent, runtime=runtime))
    second = asyncio.create_task(attempt_decide_intent(runtime.client, intents, intent, runtime=runtime))
    await asyncio.wait_for(asyncio.gather(first, second), 5.0)
    assert entered["n"] == 2  # exactly two transports entered the call
    row = intents.get(intent.intent_id)
    assert row is not None and row.state == "sent"
    raw = getattr(row, "receipt_json", None)
    assert raw and "orig-1" in str(raw)
    # Durability: a FRESH ledger instance reads the SAME raw receipt_json.
    fresh = DecideIntents(tmp_path).get(intent.intent_id)
    assert fresh is not None and fresh.receipt_json == raw


@pytest.mark.parametrize("late", ["transient", "signin"])
@pytest.mark.asyncio
async def test_a_late_failure_then_a_late_success_settles_the_same_canonical_receipt(
    tmp_path, monkeypatch: pytest.MonkeyPatch, late: str
) -> None:
    intents = DecideIntents(tmp_path)
    runtime = _runtime(tmp_path)
    intent, _ = _record(intents, runtime)
    both_entered = asyncio.Event()
    entered = {"n": 0}

    def _settled_error() -> str:
        row = intents.get(intent.intent_id)
        return str(getattr(row, "error_code", "") or "")

    async def barred(client, decision, **kwargs):
        entered["n"] += 1
        index = entered["n"]
        if index >= 2:
            # FALSE PREDICATE CONTROL: with BOTH transports still blocked
            # nothing has settled. The row's INITIAL state is already
            # "pending", so "state is pending/signin" is TRUE before any
            # settle and can never be a completion marker; the typed error is
            # still empty here.
            assert _settled_error() == ""
            both_entered.set()
        await asyncio.wait_for(both_entered.wait(), 2.0)
        if index == 2:
            # The late arm settles ITSELF first (attempt_decide_intent
            # records the typed outcome before returning): transient →
            # pending + error_code "bridge_unreachable"; signin → state
            # "signin" + "signin_required".
            if late == "transient":
                raise BridgeUnreachable()
            raise BridgeSignInRequired()
        # The late SUCCESS releases only after the failure durably SETTLED —
        # its typed error recorded, not merely its transport returned.
        while _settled_error() == "":
            await asyncio.sleep(0.01)
        assert _settled_error() == (
            "bridge_unreachable" if late == "transient" else "signin_required"
        )
        return {"request": {"receipt": "orig-1"}}

    monkeypatch.setattr(decide_mod, "decide_from_human_answer", barred)
    first = asyncio.create_task(attempt_decide_intent(runtime.client, intents, intent, runtime=runtime))
    second = asyncio.create_task(attempt_decide_intent(runtime.client, intents, intent, runtime=runtime))
    await asyncio.wait_for(asyncio.gather(first, second), 5.0)
    assert entered["n"] == 2
    row = intents.get(intent.intent_id)
    assert row is not None and row.state == "sent"
    raw = getattr(row, "receipt_json", None)
    assert raw and "orig-1" in str(raw)
    fresh = DecideIntents(tmp_path).get(intent.intent_id)
    assert fresh is not None and fresh.receipt_json == raw
    # And now the row is TERMINAL: neither transport may re-POST it.
    with pytest.raises(Exception) as refusal:
        await attempt_decide_intent(runtime.client, intents, intent, runtime=runtime)
    assert "not_retryable" in str(getattr(refusal.value, "code", "")) or "terminal" in str(
        refusal.value
    )
