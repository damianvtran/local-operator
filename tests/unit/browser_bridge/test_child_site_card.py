"""A ``task`` subagent raises the signed site card through its ancestor's queue.

Security review 60334 is the acceptance list these cells pin:

* the ancestor takes TYPED facts only and rebuilds the canonical card itself —
  byte-identical to a top-level session's card for the same facts;
* the card is bound to the actual child (live lineage), the node and the live
  request (pending in the child, unexpired); anything else is refused;
* withdraw is scoped to that requester's own cards; the lapse bound is counted
  once per family (review m2);
* a finished subagent's cards are withdrawn and its entries cancelled behind a
  fence that stops new dispatch first and accounts for dispatch in flight, and
  a decision racing that teardown signs nothing (S60334-1, design correction (2));
* ONE commit protocol per card with an explicit ``unknown`` outcome: a lost
  reply, a timeout or a cancelled task is never read as "nothing happened",
  and a lapse while deciding still records the acknowledged answer
  (R2-1, design correction (1));
* one open card per origin and NO sharing: a second requester is refused
  ``card_busy`` with the open card's ids and an explicit ``await_access``
  next step, its entry left to the extension's TTL (manager 79, design
  correction (2)); a top-level card and a child card in either arrival order
  never claim or withdraw each other (design correction (3));
* the operator-signed approve, the sticky deny and expiry resolve the child's
  ``await_access`` exactly as for a top-level session (it polls the extension);
* with no ancestor that can queue, the child fails closed and says why.

Every session gets its OWN id (review m3/n1): ``make_session`` names them all
"sess" from the transcript directory, which would hide an id-keyed mix-up.
The family's bindings are held by object identity anyway.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.browser_bridge import site_approval
from local_operator.browser_bridge.site_approval import SiteAccessFacts
from local_operator.harness.approval import (
    mint_bridge_decide_grant,
    reset_bridge_decide_grants_for_tests,
)
from tests.unit.browser_bridge.test_site_approval import VECTOR

NODE_NAME = "node-a"
URL = "http://127.0.0.1:8765/x"


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    reset_bridge_decide_grants_for_tests()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "local_operator.operator.trust.load_anchor",
        lambda *a, **k: SimpleNamespace(anchor=None, usable=False),
    )
    monkeypatch.setattr("local_operator.tools.builtin._node_display_name", lambda: NODE_NAME)
    monkeypatch.setattr("local_operator.browser_bridge.backend._on_headless_node", lambda: True)
    from local_operator.session.session import Session

    monkeypatch.setattr(Session, "_this_node_device_id", lambda self: VECTOR["node_device_id"])
    yield
    reset_bridge_decide_grants_for_tests()


def _child_of(tmp_path: Path, parent: Any, name: str):
    """A ``task`` child of ``parent`` with its own, distinct session id."""
    from tests.unit.session.test_session import ScriptedStream, make_session

    child = make_session(
        tmp_path / name,
        ScriptedStream([[]]),
        parent_session=parent,
        parent_session_id=str(parent.session_id),
        session_id=f"{name}-sess",
    )
    assert child.ask_queue() is None, "a task child has no queue of its own"
    return child


def _family(tmp_path: Path, *, root_queue: bool = True):
    """A root (with an ask surface unless ``root_queue`` is off) and its child."""
    from tests.unit.session.test_session import ScriptedStream, make_session

    root = make_session(tmp_path / "root", ScriptedStream([[]]), session_id="root-sess")
    if root_queue:

        async def _never(_questions):
            return None  # pragma: no cover

        root.set_ask_handler(_never)
    return root, _child_of(tmp_path, root, "child")


def _facts(entry_id: str = VECTOR["entry_id"], **over: Any) -> SiteAccessFacts:
    fields: dict[str, Any] = {
        "origin": VECTOR["origin"],
        "node_device_id": VECTOR["node_device_id"],
        "entry_id": entry_id,
        "expires_at_ms": int(time.time() * 1000) + 600_000,
    }
    fields.update(over)
    return SiteAccessFacts(**fields)


class _Extension:
    """The node's browser extension, as far as the tool sees it.

    One entry per REQUESTER (``approval-store.ts`` ``findPending`` is keyed by
    origin and requester): the first requester's entry is the vector's, later
    ones get their own ids. A signed approve settles every pending entry for
    the origin (``decideAccess``: ``policyCovers``) and stores a persistent
    grant; a signed deny decides only the selected entry and is remembered
    for the origin (``node-site-decisions.ts``). ``before_decide`` lets a cell
    run something (a teardown) while a decision is in flight, before the
    extension applied it; ``after_apply`` after it applied but before the
    reply reaches the host (R2-1); ``cancel_gate`` holds every
    ``cancel_access`` until it is set, so a cell can park a teardown mid-way.
    ``refuse`` makes the next submit come back unapplied with that reason
    while the entry stays pending (a refusal the requester side cannot see).
    """

    def __init__(self, entry_id: str = VECTOR["entry_id"]) -> None:
        self.first_entry = entry_id
        self.entries: dict[str, dict[str, str]] = {}
        self.allowed = False
        self.denied = False
        self.before_decide: Any = None
        self.after_apply: Any = None
        self.cancel_gate: Any = None
        self.refuse: str | None = None
        self.submitted: list[dict[str, Any]] = []
        self.calls: list[tuple[str, dict[str, Any]]] = []

    @property
    def state(self) -> str:
        """The first requester's entry state (single-requester cells)."""
        return next(iter(self.entries.values()))["state"] if self.entries else "none"

    @state.setter
    def state(self, value: str) -> None:
        for entry in self.entries.values():
            entry["state"] = value

    def entry_of(self, requester: str) -> str:
        return self.entries[requester]["entry_id"]

    async def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((method, params))
        requester = str(params.get("requester") or "")
        entry = self.entries.get(requester)
        if method == "request_access":
            if self.denied:
                return {"origin": VECTOR["origin"], "state": "denied", "approval": "signed"}
            if self.allowed:
                # The signed approve stored a PERSISTENT exact-origin grant
                # (``grantExactOriginLocked``), answered before anything queues.
                return {"origin": VECTOR["origin"], "state": "allowed"}
            if entry is None:
                entry_id = self.first_entry if not self.entries else f"entry-{requester}"
                entry = self.entries[requester] = {"entry_id": entry_id, "state": "pending"}
            entry["state"] = "pending"
            return {
                "origin": VECTOR["origin"],
                "state": "pending",
                "entryId": entry["entry_id"],
                "approval": "signed",
                "scope": "origin",
                "node_device_id": VECTOR["node_device_id"],
                "expires_at": int(time.time() * 1000) + 600_000,
            }
        if method == "cancel_access":
            if self.cancel_gate is not None:
                await self.cancel_gate.wait()
            if entry is None or entry["state"] != "pending":
                return {"origin": VECTOR["origin"], "state": "none"}
            entry["state"] = "none"
            return {"origin": VECTOR["origin"], "state": "cancelled"}
        return {"origin": VECTOR["origin"], "state": entry["state"] if entry else "none"}

    def _entry_for(self, question_id: str) -> dict[str, str] | None:
        for entry in self.entries.values():
            if entry["state"] == "pending" and question_id == site_approval.site_question_id(
                node_device_id=VECTOR["node_device_id"],
                entry_id=entry["entry_id"],
                origin=VECTOR["origin"],
                scope="origin",
            ):
                return entry
        return None

    async def submit(self, params: dict[str, Any]) -> dict[str, Any]:
        reply = await self._apply(params)
        if reply.get("applied"):
            self.submitted.append(params)
            if self.after_apply is not None:
                await self.after_apply()
        return reply

    async def _apply(self, params: dict[str, Any]) -> dict[str, Any]:
        if self.before_decide is not None:
            await self.before_decide()
        if self.refuse is not None:
            return {"applied": False, "reason": self.refuse}
        selected = self._entry_for(params["question_id"])
        if selected is None:
            return {"applied": False, "reason": "no_live_request"}
        if params["word"] == "approve":
            self.allowed = True
            for entry in self.entries.values():
                if entry["state"] == "pending":
                    entry["state"] = "allowed"
            return {"applied": True, "state": "allowed", "origin": VECTOR["origin"]}
        self.denied = True
        selected["state"] = "denied"
        return {"applied": True, "state": "denied", "origin": VECTOR["origin"]}


async def _browser(action: str, context: Any, ext: _Extension, **extra: Any) -> Any:
    from local_operator.tools import builtin

    params = builtin.BrowserParams(action=action, url=URL, **extra)
    return await builtin._bridge_access("call-1", action, params, context, client=ext)


def _text_of(result: Any) -> str:
    return " ".join("".join(getattr(block, "text", "") for block in result.content).split())


def _open(root) -> list[dict[str, Any]]:
    return root.ask_queue().open_records()


# --- the child door: card in the ancestor, byte-identical ---------------------------


@pytest.mark.asyncio
async def test_a_child_card_lands_in_the_ancestor_byte_identical(tmp_path: Path) -> None:
    root, child = _family(tmp_path)
    context = child._build_tool_context()
    assert context.enqueue_ask is None and callable(context.enqueue_site_card)
    result = await _browser("request_access", context, _Extension())
    assert not result.is_error, _text_of(result)
    [record] = _open(root)
    [question] = record["questions"]
    # What a top-level session queues for the same facts, through the same
    # queue projection.
    top = site_approval.site_ask_question(
        origin=VECTOR["origin"],
        node_name=NODE_NAME,
        node_device_id=VECTOR["node_device_id"],
        entry_id=VECTOR["entry_id"],
    )
    assert question["question_id"] == VECTOR["question_id"]
    assert result.details["ask_id"] == record["ask_id"]
    binding = root._site_cards[record["ask_id"]]
    assert binding.owner() is child and binding.entry_id == VECTOR["entry_id"]
    # The control: the SAME queue takes the top-level card once the child's
    # identical card is withdrawn (an open identical text is refused).
    assert child.withdraw_site_asks(VECTOR["origin"]) == [record["ask_id"]]
    receipt = root._enqueue_ask([top], 600)
    assert receipt["ok"], receipt
    [control] = _open(root)
    assert question == control["questions"][0]
    assert repr(question) == repr(control["questions"][0])


@pytest.mark.asyncio
@pytest.mark.parametrize("word,state", [("approve", "allowed"), ("deny", "denied")])
async def test_the_signed_decision_resolves_the_childs_await(
    tmp_path: Path, word: str, state: str
) -> None:
    root, child = _family(tmp_path)
    ext = _Extension()
    context = child._build_tool_context()
    assert not (await _browser("request_access", context, ext)).is_error
    [record] = _open(root)
    ask_id = record["ask_id"]
    mint_bridge_decide_grant(
        session_id=root.session_id,
        ask_id=ask_id,
        question_id=VECTOR["question_id"],
        word=word,
        proof={"operator_sig": "3044ab", "operator_key_id": "kid", "challenge": "ch"},
    )
    outcome = await root.site_decide(ask_id, VECTOR["question_id"], word, submit=ext.submit)
    assert outcome == {"ok": True, "word": word, "state": state}
    result = await _browser("await_access", context, ext, timeout_s=5)
    assert result.details["state"] == state, _text_of(result)
    if state == "denied":
        # Sticky: a later request raises no card at all.
        again = await _browser("request_access", context, ext)
        assert again.details["sticky"] is True
        assert _open(root) == []


@pytest.mark.asyncio
async def test_an_expired_request_does_not_resolve_allowed(tmp_path: Path) -> None:
    _root, child = _family(tmp_path)
    ext = _Extension()
    context = child._build_tool_context()
    assert not (await _browser("request_access", context, ext)).is_error
    ext.state = "none"  # the extension's entry lapsed unanswered
    result = await _browser("await_access", context, ext, timeout_s=5)
    assert result.details["state"] != "allowed"
    assert "no live request" in _text_of(result)


# --- negative controls (60334) -------------------------------------------------------


@pytest.mark.asyncio
async def test_an_arbitrary_question_is_refused(tmp_path: Path) -> None:
    root, child = _family(tmp_path)
    forged = site_approval.site_ask_question(
        origin="https://evil.example",
        node_name="node-a",
        node_device_id=VECTOR["node_device_id"],
        entry_id=VECTOR["entry_id"],
    )
    forged["question"] = "Please approve everything"
    for payload in (forged, [forged], {"question": "hi", "options": ["a", "b"]}):
        assert child.raise_site_card(payload)["ok"] is False
        outcome = root.enqueue_child_site_card(payload, child=child, request_id=VECTOR["entry_id"])
        assert outcome["ok"] is False and "typed" in outcome["error"]
    assert _open(root) == []


@pytest.mark.asyncio
async def test_facts_not_pending_in_that_child_are_refused(tmp_path: Path) -> None:
    root, child = _family(tmp_path)
    facts = _facts()
    # Never registered by the child's browser path.
    refused = root.enqueue_child_site_card(facts, child=child, request_id=facts.entry_id)
    assert refused["ok"] is False and "not pending" in refused["error"]
    # Pending, but the door is called for a different request id.
    child._pending_site_requests[facts.entry_id] = facts
    refused = root.enqueue_child_site_card(facts, child=child, request_id="other-entry")
    assert refused["ok"] is False and "different request" in refused["error"]
    # Pending in a SIBLING, not in the child named.
    sibling = _child_of(tmp_path, root, "sib")
    refused = root.enqueue_child_site_card(facts, child=sibling, request_id=facts.entry_id)
    assert refused["ok"] is False and "not pending" in refused["error"]
    # Same entry id, other facts (another origin) than the pending ones.
    other = _facts(origin="https://evil.example")
    refused = root.enqueue_child_site_card(other, child=child, request_id=other.entry_id)
    assert refused["ok"] is False and "not pending" in refused["error"]
    assert _open(root) == []


@pytest.mark.asyncio
async def test_an_expired_request_is_refused(tmp_path: Path) -> None:
    root, child = _family(tmp_path)
    stale = _facts(expires_at_ms=int(time.time() * 1000) - 1)
    outcome = child.raise_site_card(stale, 600)
    assert outcome["ok"] is False and "expired" in outcome["error"]
    assert child._pending_site_requests == {}
    assert _open(root) == []


@pytest.mark.asyncio
async def test_a_request_bound_to_another_node_is_refused(tmp_path: Path) -> None:
    root, child = _family(tmp_path)
    outcome = child.raise_site_card(_facts(node_device_id="another-node"), 600)
    assert outcome["ok"] is False and "another node" in outcome["error"]
    assert _open(root) == []


@pytest.mark.asyncio
async def test_a_non_descendant_is_refused(tmp_path: Path) -> None:
    from tests.unit.session.test_session import ScriptedStream, make_session

    root, _child = _family(tmp_path)
    stranger = make_session(tmp_path / "stranger", ScriptedStream([[]]), session_id="stranger")
    # Claims the root's id as its parent id but holds no live link to it.
    impostor = make_session(
        tmp_path / "impostor",
        ScriptedStream([[]]),
        parent_session_id=str(root.session_id),
        session_id="impostor",
    )
    facts = _facts()
    for who in (stranger, impostor, root):
        who._pending_site_requests[facts.entry_id] = facts
        outcome = root.enqueue_child_site_card(facts, child=who, request_id=facts.entry_id)
        assert outcome["ok"] is False and "not a live subagent" in outcome["error"]
    assert _open(root) == []


# --- scoped bookkeeping --------------------------------------------------------------


@pytest.mark.asyncio
async def test_withdraw_is_scoped_to_the_requesters_own_cards(tmp_path: Path) -> None:
    root, child = _family(tmp_path)
    child_receipt = child.raise_site_card(_facts(), 600)
    assert child_receipt["ok"], child_receipt
    child_ask = child_receipt["details"]["ask_id"]
    # The root's own withdraw leaves the child's card alone ...
    assert root.withdraw_site_asks(VECTOR["origin"]) == []
    # ... and the child's withdraw reaches only its own.
    assert child.withdraw_site_asks(VECTOR["origin"]) == [child_ask]
    assert child._pending_site_requests == {}
    # Pruned once withdrawn (review N1).
    assert root._site_cards == {}
    # A withdrawn request is no longer pending: its facts raise nothing.
    stale = root.enqueue_child_site_card(_facts(), child=child, request_id=VECTOR["entry_id"])
    assert stale["ok"] is False and "not pending" in stale["error"]


@pytest.mark.asyncio
async def test_the_lapse_bound_is_counted_once_per_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review m2: two lapsed cards for an origin stop EVERY requester in the
    family — N children may not raise 2N unanswered cards. Fails on 59b440bdf,
    where each child (and the root) had a counter of its own."""
    root, child = _family(tmp_path)
    sibling = _child_of(tmp_path, root, "sibling")
    queue = root.ask_queue()
    real_now = queue._now
    ext = _Extension()
    for hour, who in enumerate((child, sibling), start=1):
        raised = await _browser("request_access", who._build_tool_context(), ext)
        assert not raised.is_error and raised.details.get("ask_id"), _text_of(raised)
        # The queue's clock moves on an hour: the card lapses unanswered
        # before the next requester raises its own.
        monkeypatch.setattr(queue, "_now", lambda h=hour: real_now() + h * 3_600_000)
        assert queue.open_records() == []
    for who in (root, child, sibling):
        assert who.site_ask_lapses(VECTOR["origin"]) == 2
    third = _child_of(tmp_path, root, "third")
    result = await _browser("request_access", third._build_tool_context(), ext)
    assert result.is_error and "hasn't answered twice" in _text_of(result)
    assert _open(root) == []


# --- fail closed -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_ancestor_queue_fails_closed_and_says_why(tmp_path: Path) -> None:
    _root, child = _family(tmp_path, root_queue=False)
    ext = _Extension()
    context = child._build_tool_context()
    result = await _browser("request_access", context, ext)
    assert result.is_error
    assert result.details["state"] == "none"
    text = _text_of(result)
    assert "none of the sessions that delegated to it can queue an ask" in text
    # The extension entry nobody can decide is withdrawn, as before.
    assert [m for m, _ in ext.calls] == ["request_access", "cancel_access"]


@pytest.mark.asyncio
async def test_top_level_keeps_mains_enqueue_ask_path(tmp_path: Path) -> None:
    """T16: a session with its own queue has NO site-card door and no fence,
    so its card goes through main's ``enqueue_ask`` unchanged, is bound to no
    requester, and is decided by main's path (any submit failure is
    ``site_unreachable``, never the child-only ``site_outcome_unknown``)."""
    from local_operator.browser_bridge.backend import BridgeUnreachable

    root, _child = _family(tmp_path)
    context = root._build_tool_context()
    assert callable(context.enqueue_ask)
    assert context.enqueue_site_card is None and context.begin_site_request is None
    assert callable(context.withdraw_site_asks) and callable(context.site_ask_lapses)
    result = await _browser("request_access", context, _Extension())
    assert not result.is_error, _text_of(result)
    [record] = _open(root)
    assert root._site_cards == {}
    mint_bridge_decide_grant(
        session_id=root.session_id,
        ask_id=record["ask_id"],
        question_id=VECTOR["question_id"],
        word="approve",
        proof={"operator_sig": "3044ab", "operator_key_id": "kid", "challenge": "ch"},
    )

    async def _lost(_params: dict[str, Any]) -> dict[str, Any]:
        raise BridgeUnreachable("read timeout")

    outcome = await root.site_decide(
        record["ask_id"], VECTOR["question_id"], "approve", submit=_lost
    )
    assert outcome["ok"] is False and outcome["code"] == "site_unreachable"


@pytest.mark.asyncio
async def test_a_grandchild_reaches_the_nearest_ancestor_with_a_queue(tmp_path: Path) -> None:
    root, child = _family(tmp_path)
    grandchild = _child_of(tmp_path, child, "grandchild")
    context = grandchild._build_tool_context()
    result = await _browser("request_access", context, _Extension())
    assert not result.is_error, _text_of(result)
    [record] = _open(root)
    assert root._site_cards[record["ask_id"]].owner() is grandchild


# --- helpers ------------------------------------------------------------------------


async def _decide(root: Any, ext: _Extension, word: str, submit: Any = None) -> dict[str, Any]:
    """The operator's signed answer to the one open card, through ``site_decide``."""
    [record] = _open(root)
    [question] = record["questions"]
    mint_bridge_decide_grant(
        session_id=root.session_id,
        ask_id=record["ask_id"],
        question_id=question["question_id"],
        word=word,
        proof={"operator_sig": "3044ab", "operator_key_id": "kid", "challenge": "ch"},
    )
    return await root.site_decide(
        record["ask_id"], question["question_id"], word, submit=submit or ext.submit
    )


def _cancels_by(ext: _Extension, session: Any) -> int:
    return sum(
        1
        for method, params in ext.calls
        if method == "cancel_access" and params.get("requester") == f"session:{session.session_id}"
    )


def _cancelled_by(ext: _Extension, session: Any) -> bool:
    return _cancels_by(ext, session) > 0


async def _until(predicate: Any) -> None:
    """Yield to the loop until ``predicate()`` holds — an ordering barrier on
    observable state, never a wall-clock sleep (bounded so a broken ordering
    fails rather than hangs)."""
    import asyncio

    for _ in range(500):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("ordering barrier never reached")


def _status(root: Any, ask_id: str) -> str:
    return str(root.ask_queue().find(ask_id)["status"])


class _Counting:
    """A ``submit`` that counts calls and delegates (or raises ``exc``)."""

    def __init__(self, ext: _Extension, exc: BaseException | None = None) -> None:
        self.ext, self.exc, self.calls = ext, exc, 0

    async def __call__(self, params: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        return await self.ext.submit(params)


async def _child_card(tmp_path: Path) -> tuple[Any, Any, _Extension, str]:
    root, child = _family(tmp_path)
    ext = _Extension()
    assert not (await _browser("request_access", child._build_tool_context(), ext)).is_error
    [record] = _open(root)
    return root, child, ext, record["ask_id"]


# --- the commit protocol: unknown is explicit (R2-1, 60334 design correction (1)) ----


@pytest.mark.asyncio
async def test_t1_a_reply_lost_after_apply_is_unknown_not_unreachable(tmp_path: Path) -> None:
    """T1. Fails on 1c29d7c22, where any submit exception read "nothing was
    decided" (``site_unreachable``) and left the card open for a resubmit
    that would then be misreported."""
    from local_operator.browser_bridge.backend import BridgeUnreachable

    root, child, ext, ask_id = await _child_card(tmp_path)

    async def _lose_reply() -> None:
        raise BridgeUnreachable("read timeout after delivery")

    ext.after_apply = _lose_reply
    outcome = await _decide(root, ext, "approve")
    assert outcome["ok"] is False and outcome["code"] == "site_outcome_unknown", outcome
    assert "may or may not" in outcome["error"]
    assert root._site_cards[ask_id].state == "unknown"
    assert _status(root, ask_id) == "open", "unknown is neither answered nor withdrawn"
    # A second tap is refused WITHOUT resubmitting.
    again = _Counting(ext)
    outcome = await _decide(root, ext, "approve", submit=again)
    assert outcome["code"] == "site_outcome_unknown" and again.calls == 0
    # The child reads the truth from the extension.
    result = await _browser("await_access", child._build_tool_context(), ext, timeout_s=5)
    assert result.details["state"] == "allowed", _text_of(result)


@pytest.mark.asyncio
async def test_t2_a_submit_timeout_is_unknown(tmp_path: Path) -> None:
    """T2. The bridge call's own timeout (``BridgeClient`` ``timeout_for``)
    surfaces as an exception after dispatch: unknown. No new cap is added on
    top of the bridge's (delivery guide: no cap without a measured bound)."""
    root, _child, ext, ask_id = await _child_card(tmp_path)
    outcome = await _decide(root, ext, "approve", submit=_Counting(ext, TimeoutError()))
    assert outcome["code"] == "site_outcome_unknown"
    assert root._site_cards[ask_id].state == "unknown" and _status(root, ask_id) == "open"


@pytest.mark.asyncio
async def test_t2c_a_cancelled_decision_is_unknown_and_propagates(tmp_path: Path) -> None:
    """60334 design correction (1): task cancellation mid-submit cannot prove
    the decision never reached the extension, so the card is ``unknown`` (not
    reopened), the cancellation propagates, and the local decision grant is
    withdrawn — which revokes no browser grant."""
    import asyncio

    from local_operator.harness.approval import bridge_decide_grant_proof

    root, _child, ext, ask_id = await _child_card(tmp_path)
    entered, never = asyncio.Event(), asyncio.Event()

    async def _parked(_params: dict[str, Any]) -> dict[str, Any]:
        entered.set()
        await never.wait()
        raise AssertionError("unreachable")  # pragma: no cover

    task = asyncio.ensure_future(_decide(root, ext, "approve", submit=_parked))
    await entered.wait()
    assert root._site_cards[ask_id].state == "deciding"
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    card = root._site_cards[ask_id]
    assert card.state == "unknown" and card.settled.is_set()
    assert _status(root, ask_id) == "open"
    assert not bridge_decide_grant_proof(
        session_id=root.session_id, ask_id=ask_id, question_id=VECTOR["question_id"], word="approve"
    ), "the local decision grant is withdrawn"
    again = _Counting(ext)
    outcome = await _decide(root, ext, "approve", submit=again)
    assert outcome["code"] == "site_outcome_unknown" and again.calls == 0


@pytest.mark.asyncio
async def test_t3_an_undelivered_decision_reopens_the_card(tmp_path: Path) -> None:
    """T3. ``HostPreDeliveryFailure`` proves nothing was dispatched: the card
    is open again and the next tap submits (main's behaviour)."""
    from local_operator.browser_bridge.backend import HostPreDeliveryFailure

    root, _child, ext, ask_id = await _child_card(tmp_path)
    outcome = await _decide(root, ext, "approve", submit=_Counting(ext, HostPreDeliveryFailure()))
    assert outcome["code"] == "site_unreachable"
    assert root._site_cards[ask_id].state == "open"
    again = _Counting(ext)
    outcome = await _decide(root, ext, "approve", submit=again)
    assert outcome["ok"] is True and again.calls == 1
    assert _status(root, ask_id) == "answered" and ask_id not in root._site_cards


@pytest.mark.asyncio
@pytest.mark.parametrize("word,state", [("approve", "allowed"), ("deny", "denied")])
async def test_t4_the_ack_is_recorded_before_any_await(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, word: str, state: str
) -> None:
    """T4 (R2-1). The submit is parked; a teardown starts and waits on the
    card; the extension applies. The answer is recorded in the same
    synchronous run as the acknowledgement (a ``call_soon`` scheduled at the
    acknowledgement runs only AFTER ``respond``), and the teardown issues no
    withdraw and no cancel for the card."""
    import asyncio

    root, child, ext, ask_id = await _child_card(tmp_path)
    order: list[str] = []
    queue = root.ask_queue()
    real_respond = queue.respond

    def _respond(*args: Any, **kwargs: Any) -> Any:
        order.append("respond")
        return real_respond(*args, **kwargs)

    monkeypatch.setattr(queue, "respond", _respond)
    release = asyncio.Event()
    teardown: list[Any] = []

    async def _park() -> None:
        teardown.append(asyncio.ensure_future(child.dispose()))
        await _until(lambda: child._site_requests_closed)
        await release.wait()

    async def _acked() -> None:
        asyncio.get_running_loop().call_soon(order.append, "loop-turn")

    ext.before_decide = _park
    ext.after_apply = _acked
    decision = asyncio.ensure_future(_decide(root, ext, word))
    await _until(lambda: bool(teardown))
    assert _status(root, ask_id) == "open", "a deciding card is not withdrawn"
    release.set()
    outcome = await decision
    await teardown[0]
    assert outcome == {"ok": True, "word": word, "state": state}
    assert order[:2] == ["respond", "loop-turn"]
    assert _status(root, ask_id) == "answered"
    assert _cancels_by(ext, child) == 0


@pytest.mark.asyncio
async def test_t5_a_teardown_that_ran_first_refuses_the_decision_unsubmitted(
    tmp_path: Path,
) -> None:
    """T5. Teardown THEN answer: the card is withdrawn, the child's entry is
    cancelled exactly once, and the answer submits nothing. Fails on
    59b440bdf, where the approve still granted the origin."""
    root, child, ext, ask_id = await _child_card(tmp_path)
    await child.dispose()
    assert _open(root) == [] and root._site_cards == {}
    assert _cancels_by(ext, child) == 1 and ext.state == "none"
    mint_bridge_decide_grant(
        session_id=root.session_id,
        ask_id=ask_id,
        question_id=VECTOR["question_id"],
        word="approve",
        proof={"operator_sig": "3044ab", "operator_key_id": "kid", "challenge": "ch"},
    )
    counting = _Counting(ext)
    outcome = await root.site_decide(ask_id, VECTOR["question_id"], "approve", submit=counting)
    assert outcome["ok"] is False and counting.calls == 0 and ext.allowed is False


@pytest.mark.asyncio
async def test_t5b_a_decision_for_a_requester_that_is_gone_is_refused_before_signing(
    tmp_path: Path,
) -> None:
    """The decision's own guard, card still open: a closed requester is not
    decided for (defence in depth beside the extension's entry binding)."""
    root, child, ext, _ask_id = await _child_card(tmp_path)
    child._site_requests_closed = True
    counting = _Counting(ext)
    outcome = await _decide(root, ext, "approve", submit=counting)
    assert outcome["ok"] is False and outcome["code"] == "requester_gone"
    assert counting.calls == 0 and ext.allowed is False


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["applied", "refused", "lost"])
async def test_t6_a_teardown_whose_wait_ran_out_leaves_the_card_to_the_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ending: str
) -> None:
    """T6a/b/c. The submit outlives the teardown's bounded wait. The teardown
    neither withdraws the deciding card nor cancels its entry; the decision
    records the truth: answered / withdrawn (the extension said not applied
    and the owner has ended) / unknown (record left open)."""
    import asyncio

    from local_operator.browser_bridge.backend import BridgeUnreachable
    from local_operator.session.session import Session

    monkeypatch.setattr(Session, "_SITE_TEARDOWN_DECIDE_WAIT_S", 0.01)
    root, child, ext, ask_id = await _child_card(tmp_path)
    released = asyncio.Event()

    async def _teardown_times_out() -> None:
        await child.dispose()  # returns: its wait ran out
        assert _status(root, ask_id) == "open"
        assert _cancels_by(ext, child) == 0
        if ending == "refused":
            ext.refuse = "no_live_request"
        released.set()

    async def _lost() -> None:
        raise BridgeUnreachable("reply lost")

    ext.before_decide = _teardown_times_out
    if ending == "lost":
        ext.after_apply = _lost
    outcome = await _decide(root, ext, "approve")
    assert released.is_set()
    assert _cancels_by(ext, child) == 0, "a decided-or-unknown entry is never cancelled"
    if ending == "applied":
        assert outcome["ok"] is True and _status(root, ask_id) == "answered"
    elif ending == "refused":
        assert outcome["ok"] is False and _status(root, ask_id) == "withdrawn"
        assert ask_id not in root._site_cards
    else:
        assert outcome["code"] == "site_outcome_unknown"
        assert _status(root, ask_id) == "open"
        assert root._site_cards[ask_id].state == "unknown"


@pytest.mark.asyncio
async def test_t7_a_refused_decision_for_a_live_owner_reopens_the_card(tmp_path: Path) -> None:
    """T7 (R3-1 analogue). The extension refuses; the owner is still waiting:
    the card stays on the operator's screen and nothing is cancelled."""
    root, child, ext, ask_id = await _child_card(tmp_path)
    ext.refuse = "no_live_request"
    outcome = await _decide(root, ext, "approve")
    assert outcome["ok"] is False and outcome["code"] == "no_live_request"
    assert root._site_cards[ask_id].state == "open" and _status(root, ask_id) == "open"
    assert _cancels_by(ext, child) == 0
    ext.refuse = None
    assert (await _decide(root, ext, "approve"))["ok"] is True


@pytest.mark.asyncio
async def test_t8_the_cancel_boundary_skips_an_unsettled_card_and_in_flight_dispatch(
    tmp_path: Path,
) -> None:
    """T8 (R2-2). At the mutation boundary, an origin whose card of this
    requester is ``unknown`` is not cancelled — nor is an in-flight dispatch's
    entry, whose origin the extension normalises — because the decision may
    have applied. Zero ``cancel_access`` from the requester."""
    from local_operator.browser_bridge.backend import BridgeUnreachable

    root, child, ext, ask_id = await _child_card(tmp_path)
    await _decide(root, ext, "approve", submit=_Counting(ext, BridgeUnreachable("lost")))
    assert root._site_cards[ask_id].state == "unknown"
    import asyncio

    dispatch_cancels: list[str] = []

    async def _cancel_dispatch() -> None:
        dispatch_cancels.append("cancel")

    finish = child.begin_site_request(_cancel_dispatch)  # a dispatch in flight
    assert callable(finish)
    teardown = asyncio.ensure_future(child.dispose())
    await _until(lambda: child._site_requests_closed)
    assert child._site_dispatches == {}, "the in-flight dispatch was snapshotted"
    finish()  # its reply arrives after the fence closed
    await teardown
    assert dispatch_cancels == [], "skipped: the requester has an unknown card"
    assert _cancels_by(ext, child) == 0
    assert _status(root, ask_id) == "open"


@pytest.mark.asyncio
async def test_t9_a_closed_requester_raises_nothing(tmp_path: Path) -> None:
    """T9. After teardown the door refuses: no card, no host queue write."""
    root, child = _family(tmp_path)
    await child.dispose()
    facts = _facts()
    child._pending_site_requests[facts.entry_id] = facts
    outcome = child.raise_site_card(facts, 600)
    assert outcome["ok"] is False and "finished" in outcome["error"]
    assert root.ask_queue().records() == []


# --- the fence before dispatch (60334 design correction (2)) ---------------------------


@pytest.mark.asyncio
async def test_f1_a_finished_requester_dispatches_no_request(tmp_path: Path) -> None:
    """The fence stops ISSUANCE: after teardown began, the browser tool sends
    no ``request_access`` at all. Fails on 1c29d7c22, where the request was
    dispatched (creating an extension entry nobody would cancel) and only the
    later card enqueue was refused."""
    root, child = _family(tmp_path)
    context = child._build_tool_context()
    await child.dispose()
    ext = _Extension()
    result = await _browser("request_access", context, ext)
    assert result.is_error and result.details["code"] == "requester_closed"
    assert result.details["approval_request_sent"] is False
    assert ext.calls == []
    assert root.ask_queue().records() == []


@pytest.mark.asyncio
async def test_f2_a_dispatch_in_flight_at_teardown_is_cancelled(tmp_path: Path) -> None:
    """In-flight accounting: ``request_access`` was sent before teardown and
    its reply lands after. The teardown waits for it, raises nothing for it,
    and its entry ends cancelled."""
    import asyncio

    root, child = _family(tmp_path)
    ext = _Extension()
    gate, entered = asyncio.Event(), asyncio.Event()
    real_call = ext.call

    async def _slow_call(method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "request_access":
            entered.set()
            await gate.wait()
        return await real_call(method, params)

    ext.call = _slow_call  # type: ignore[method-assign]
    request = asyncio.ensure_future(_browser("request_access", child._build_tool_context(), ext))
    await entered.wait()
    assert len(child._site_dispatches) == 1
    teardown = asyncio.ensure_future(child.dispose())
    await _until(lambda: child._site_requests_closed)
    gate.set()
    result = await request
    await teardown
    assert result.is_error, _text_of(result)
    assert root.ask_queue().records() == [], "no card for a finished requester"
    assert ext.state == "none", "the entry that crossed the fence is cancelled"
    assert _cancelled_by(ext, child)


# --- lapse (design §3, 60334 design correction (1)) ------------------------------------


def _lapse(monkeypatch: pytest.MonkeyPatch, root: Any) -> None:
    """Move the queue's clock past every card's deadline (inside the late window)."""
    queue = root.ask_queue()
    real_now = queue._now
    monkeypatch.setattr(queue, "_now", lambda: real_now() + 3_600_000)


@pytest.mark.asyncio
async def test_t10_a_lapsed_card_makes_no_bridge_call_and_still_checks_the_requester(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T10. A card that timed out unanswered makes no extension call when
    pruned. The queue still takes a late answer for it (folded ``late``), so
    its binding is kept and a late tap for a requester that has gone is
    refused before signing."""
    root, child, ext, ask_id = await _child_card(tmp_path)
    calls_before = list(ext.calls)
    _lapse(monkeypatch, root)
    assert _status(root, ask_id) == "timed_out"
    root._prune_site_cards()
    assert ext.calls == calls_before, "lapse makes no bridge call"
    assert ask_id in root._site_cards
    child._site_requests_closed = True
    mint_bridge_decide_grant(
        session_id=root.session_id,
        ask_id=ask_id,
        question_id=VECTOR["question_id"],
        word="approve",
        proof={"operator_sig": "3044ab", "operator_key_id": "kid", "challenge": "ch"},
    )
    counting = _Counting(ext)
    outcome = await root.site_decide(ask_id, VECTOR["question_id"], "approve", submit=counting)
    assert outcome["code"] == "requester_gone" and counting.calls == 0


@pytest.mark.asyncio
async def test_t11_a_lapse_while_deciding_keeps_the_acknowledged_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T11, 60334 design correction (1). The card's ask times out while the
    signed decision is at the extension; the extension then applies it. The
    acknowledged outcome is RECORDED (folded ``late`` with the operator's
    word), never discarded as ``recorded: False``, and the prune left the
    deciding card alone."""
    root, _child, ext, ask_id = await _child_card(tmp_path)

    async def _lapse_mid_decision() -> None:
        _lapse(monkeypatch, root)
        root._prune_site_cards()
        assert root._site_cards[ask_id].state == "deciding"

    ext.before_decide = _lapse_mid_decision
    outcome = await _decide(root, ext, "deny")
    assert outcome == {"ok": True, "word": "deny", "state": "denied"}
    record = root.ask_queue().find(ask_id)
    assert record["status"] == "late"
    assert list(record["answers"].values()) == [["deny"]]


# --- one open card per origin: card_busy (manager 79, design correction (2)) ----------


async def _busy_pair(tmp_path: Path) -> tuple[Any, Any, Any, _Extension, str, Any]:
    root, first = _family(tmp_path)
    second = _child_of(tmp_path, root, "second")
    ext = _Extension()
    assert not (await _browser("request_access", first._build_tool_context(), ext)).is_error
    [record] = _open(root)
    busy = await _browser("request_access", second._build_tool_context(), ext)
    return root, first, second, ext, record["ask_id"], busy


@pytest.mark.asyncio
async def test_t12_a_second_requester_is_refused_card_busy_with_the_cards_id(
    tmp_path: Path,
) -> None:
    """T12. Machine-actionable (manager 79): the stable code, the open card's
    ask and question ids, and ONE named next step. The second requester's own
    entry is left to the extension's TTL — no asynchronous cancel keyed by
    origin and requester (60334 design correction (2)) — and the first card is
    neither claimed nor touched."""
    root, first, second, ext, ask_id, busy = await _busy_pair(tmp_path)
    assert busy.is_error
    details = busy.details
    assert details["code"] == "card_busy" and details["approval"] == "card_busy"
    assert details["card_ask_id"] == ask_id
    assert details["card_question_id"] == VECTOR["question_id"]
    assert details["next_action"] == "await_access" and details["state"] == "pending"
    text = _text_of(busy)
    assert "card_busy" in text and ask_id in text and "action='await_access'" in text
    assert "Do not call request_access while that card is open" in text
    assert _cancels_by(ext, second) == 0 and _cancels_by(ext, first) == 0
    assert ext.entries[f"session:{second.session_id}"]["state"] == "pending"
    [record] = _open(root)
    assert record["ask_id"] == ask_id
    assert root._site_cards[ask_id].owner() is first
    assert second._pending_site_requests == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("word,state", [("approve", "allowed"), ("deny", "denied")])
async def test_t13_after_the_answer_the_busy_requester_reaches_the_truth_without_a_card(
    tmp_path: Path, word: str, state: str
) -> None:
    """T13. Approve: a node-wide grant settles the busy requester's pending
    entry, so its ``await_access`` reports allowed. Deny: remembered, so its
    retry answers denied. Neither raises a new card."""
    root, _first, second, ext, ask_id, _busy = await _busy_pair(tmp_path)
    assert (await _decide(root, ext, word))["ok"] is True
    context = second._build_tool_context()
    if word == "approve":
        waited = await _browser("await_access", context, ext, timeout_s=5)
        assert waited.details["state"] == "allowed", _text_of(waited)
    retry = await _browser("request_access", context, ext)
    assert retry.details["state"] == state, _text_of(retry)
    assert _open(root) == []
    assert [r["ask_id"] for r in root.ask_queue().records()] == [ask_id]


@pytest.mark.asyncio
async def test_t14_the_busy_requesters_teardown_never_touches_the_open_card(
    tmp_path: Path,
) -> None:
    """T14. Owner-only teardown: the second requester ending leaves the first
    requester's card open and issues no cancel for the first requester."""
    root, first, second, ext, ask_id, _busy = await _busy_pair(tmp_path)
    await second.dispose()
    [record] = _open(root)
    assert record["ask_id"] == ask_id and root._site_cards[ask_id].state == "open"
    assert _cancels_by(ext, first) == 0
    assert ext.entries[f"session:{first.session_id}"]["state"] == "pending"


@pytest.mark.asyncio
async def test_t15a_a_child_after_a_top_level_card_is_busy(tmp_path: Path) -> None:
    """T15, 60334 design correction (3), top-level card first: the child is
    refused ``card_busy`` naming the host's card, and the host's card is
    unchanged and still unbound."""
    root, child = _family(tmp_path)
    ext = _Extension()
    assert not (await _browser("request_access", root._build_tool_context(), ext)).is_error
    [record] = _open(root)
    busy = await _browser("request_access", child._build_tool_context(), ext)
    assert busy.details["code"] == "card_busy"
    assert busy.details["card_ask_id"] == record["ask_id"]
    assert [r["ask_id"] for r in _open(root)] == [record["ask_id"]]
    assert root._site_cards == {}
    assert _cancels_by(ext, root) == 0 and _cancels_by(ext, child) == 0


@pytest.mark.asyncio
async def test_t15b_a_top_level_request_after_a_child_card_neither_claims_nor_withdraws_it(
    tmp_path: Path,
) -> None:
    """T15, 60334 design correction (3), child card first: main's unchanged
    top-level path may come back busy (the queue refuses the identical text),
    but its receipt names no ask id — it does not claim the child's card — and
    its stale-card withdraw leaves the child's card and binding alone."""
    root, child, ext, ask_id = await _child_card(tmp_path)
    result = await _browser("request_access", root._build_tool_context(), ext)
    assert "ask_id" not in result.details, result.details
    assert [r["ask_id"] for r in _open(root)] == [ask_id]
    card = root._site_cards[ask_id]
    assert card.owner() is child and card.state == "open"
    assert _cancels_by(ext, child) == 0
