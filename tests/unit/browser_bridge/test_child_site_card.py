"""A ``task`` subagent raises the signed site card through its ancestor's queue.

Security review 60334 is the acceptance list these cells pin:

* the ancestor takes TYPED facts only and rebuilds the canonical card itself —
  byte-identical to a top-level session's card for the same facts;
* the card is bound to the actual child (live lineage), the node and the live
  request (pending in the child, unexpired); anything else is refused;
* withdraw is scoped to that requester's own cards; the lapse bound is counted
  once per family (review m2);
* a finished subagent's cards are withdrawn and its entries cancelled, and a
  decision racing that teardown signs nothing (S60334-1);
* one card per origin, bound per requester: a later requester is a waiter on
  it, resolved by approve AND deny in every arrival order (S60334-2);
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
async def test_a_top_level_session_raises_its_own_card_through_the_same_door(
    tmp_path: Path,
) -> None:
    root, _child = _family(tmp_path)
    context = root._build_tool_context()
    assert callable(context.enqueue_ask) and callable(context.enqueue_site_card)
    result = await _browser("request_access", context, _Extension())
    assert not result.is_error, _text_of(result)
    [record] = _open(root)
    assert root._site_cards[record["ask_id"]].owner() is root
    # A context with only the plain door keeps the plain path.
    plain = context.model_copy(update={"enqueue_site_card": None})
    assert root.withdraw_site_asks(VECTOR["origin"]) == [record["ask_id"]]
    result = await _browser("request_access", plain, _Extension())
    assert not result.is_error, _text_of(result)
    [record] = _open(root)
    assert record["ask_id"] not in root._site_cards


@pytest.mark.asyncio
async def test_a_grandchild_reaches_the_nearest_ancestor_with_a_queue(tmp_path: Path) -> None:
    root, child = _family(tmp_path)
    grandchild = _child_of(tmp_path, child, "grandchild")
    context = grandchild._build_tool_context()
    result = await _browser("request_access", context, _Extension())
    assert not result.is_error, _text_of(result)
    [record] = _open(root)
    assert root._site_cards[record["ask_id"]].owner() is grandchild


@pytest.mark.asyncio
async def test_one_approval_covers_later_requests_for_that_origin(tmp_path: Path) -> None:
    """Manager scope note: after ONE signed approval, a later request for the
    same origin from the child raises no further card. The signed card's only
    scope is the exact origin (``SITE_SCOPE``), granted persistently by the
    extension; there is no per-session or whole-loopback-host signed scope."""
    root, child = _family(tmp_path)
    ext = _Extension()
    context = child._build_tool_context()
    assert not (await _browser("request_access", context, ext)).is_error
    [record] = _open(root)
    mint_bridge_decide_grant(
        session_id=root.session_id,
        ask_id=record["ask_id"],
        question_id=VECTOR["question_id"],
        word="approve",
        proof={"operator_sig": "3044ab", "operator_key_id": "kid", "challenge": "ch"},
    )
    await root.site_decide(record["ask_id"], VECTOR["question_id"], "approve", submit=ext.submit)
    for _ in range(2):
        again = await _browser("request_access", context, ext)
        assert not again.is_error and "allowed" in _text_of(again), _text_of(again)
    assert _open(root) == []
    assert [r["ask_id"] for r in root.ask_queue().records()] == [record["ask_id"]]


# --- teardown (S60334-1) ---------------------------------------------------------------


async def _decide(root: Any, ext: _Extension, word: str) -> dict[str, Any]:
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
        record["ask_id"], question["question_id"], word, submit=ext.submit
    )


def _cancelled_by(ext: _Extension, session: Any) -> bool:
    return any(
        method == "cancel_access" and params.get("requester") == f"session:{session.session_id}"
        for method, params in ext.calls
    )


@pytest.mark.asyncio
async def test_teardown_withdraws_the_card_and_a_later_answer_grants_nothing(
    tmp_path: Path,
) -> None:
    """Teardown THEN answer. Fails on 59b440bdf: the card stayed open after the
    child ended and the operator's approve still granted the origin."""
    root, child = _family(tmp_path)
    ext = _Extension()
    assert not (await _browser("request_access", child._build_tool_context(), ext)).is_error
    [record] = _open(root)
    await child.dispose()
    assert _open(root) == [], "the finished child's card is withdrawn"
    assert root._site_cards == {}
    assert _cancelled_by(ext, child), "the finished child's extension entry is cancelled"
    assert ext.state == "none"
    # The operator answers the (already withdrawn) card from a stale surface.
    mint_bridge_decide_grant(
        session_id=root.session_id,
        ask_id=record["ask_id"],
        question_id=VECTOR["question_id"],
        word="approve",
        proof={"operator_sig": "3044ab", "operator_key_id": "kid", "challenge": "ch"},
    )
    submitted: list[Any] = []

    async def _submit(params: dict[str, Any]) -> dict[str, Any]:
        submitted.append(params)
        return await ext.submit(params)

    outcome = await root.site_decide(
        record["ask_id"], VECTOR["question_id"], "approve", submit=_submit
    )
    assert outcome["ok"] is False
    assert submitted == [] and ext.allowed is False


@pytest.mark.asyncio
async def test_a_decision_for_a_requester_that_is_gone_is_refused_before_signing(
    tmp_path: Path,
) -> None:
    """The decision path's own guard, with the card still open: a requester
    whose request is no longer pending is not decided for (defence in depth
    beside the extension's entry binding)."""
    root, child = _family(tmp_path)
    ext = _Extension()
    assert not (await _browser("request_access", child._build_tool_context(), ext)).is_error
    child._site_requests_closed = True  # finished, card not yet withdrawn
    submitted: list[Any] = []

    async def _submit(params: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover
        submitted.append(params)
        return await ext.submit(params)

    [record] = _open(root)
    mint_bridge_decide_grant(
        session_id=root.session_id,
        ask_id=record["ask_id"],
        question_id=VECTOR["question_id"],
        word="approve",
        proof={"operator_sig": "3044ab", "operator_key_id": "kid", "challenge": "ch"},
    )
    outcome = await root.site_decide(
        record["ask_id"], VECTOR["question_id"], "approve", submit=_submit
    )
    assert outcome["ok"] is False and outcome["code"] == "requester_gone"
    assert submitted == [] and ext.allowed is False


async def _until(predicate: Any) -> None:
    """Yield to the loop until ``predicate()`` holds — an ordering barrier on
    observable state, never a wall-clock sleep (bounded so a broken ordering
    fails rather than hangs)."""
    import asyncio

    for _ in range(200):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("ordering barrier never reached")


def _status(root: Any, ask_id: str) -> str:
    return str(root.ask_queue().find(ask_id)["status"])


@pytest.mark.asyncio
@pytest.mark.parametrize("word,state", [("approve", "allowed"), ("deny", "denied")])
async def test_a_decision_the_extension_took_before_teardown_is_recorded_answered(
    tmp_path: Path, word: str, state: str
) -> None:
    """R2-1, interleaving "submit first": the signed decision is APPLIED at the
    extension, and the child's teardown lands before the reply reaches the
    host. The record must say what the extension did. Fails on f9afe5418: the
    teardown withdrew the card, so the ask said ``withdrawn`` while the
    extension had stored the decision (reviewer m2)."""
    import asyncio

    root, child = _family(tmp_path)
    ext = _Extension()
    assert not (await _browser("request_access", child._build_tool_context(), ext)).is_error
    [record] = _open(root)
    teardown: list[Any] = []

    async def _teardown_lands() -> None:
        teardown.append(asyncio.ensure_future(child.dispose()))
        await _until(lambda: child._site_requests_closed)

    ext.after_apply = _teardown_lands
    outcome = await _decide(root, ext, word)
    await teardown[0]
    assert len(ext.submitted) == 1 and ext.submitted[0]["word"] == word
    assert outcome["ok"] is True and outcome.get("recorded", True) is True, outcome
    assert _status(root, record["ask_id"]) == "answered"
    assert ext.state == state, "the entry the extension decided was not cancelled after"
    assert root._site_cards == {}


@pytest.mark.asyncio
async def test_a_teardown_that_won_the_lock_refuses_the_decision_unsubmitted(
    tmp_path: Path,
) -> None:
    """R2-1, interleaving "teardown first": the teardown has started (parked
    mid-way on its entry cancel) when the signed decision arrives. The
    decision refuses ``requester_gone`` and NOTHING reaches the extension;
    the card's record is ``withdrawn``, matching the extension (no grant)."""
    import asyncio

    root, child = _family(tmp_path)
    ext = _Extension()
    assert not (await _browser("request_access", child._build_tool_context(), ext)).is_error
    [record] = _open(root)
    ext.cancel_gate = asyncio.Event()
    teardown = asyncio.ensure_future(child.dispose())
    await _until(lambda: child._site_requests_closed)
    mint_bridge_decide_grant(
        session_id=root.session_id,
        ask_id=record["ask_id"],
        question_id=VECTOR["question_id"],
        word="approve",
        proof={"operator_sig": "3044ab", "operator_key_id": "kid", "challenge": "ch"},
    )
    outcome = await root.site_decide(
        record["ask_id"], VECTOR["question_id"], "approve", submit=ext.submit
    )
    ext.cancel_gate.set()
    await teardown
    # Refused before any submit: by the closed requester / closed lock, or by
    # the withdrawn record when the teardown's withdraw already landed.
    assert outcome["ok"] is False, outcome
    assert outcome.get("code", "requester_gone") == "requester_gone", outcome
    assert ext.submitted == [] and ext.allowed is False
    assert _status(root, record["ask_id"]) == "withdrawn"
    assert _cancelled_by(ext, child) and ext.state == "none"


@pytest.mark.asyncio
async def test_a_card_whose_lock_a_teardown_holds_is_not_decided(tmp_path: Path) -> None:
    """R2-1, the lock itself: a teardown moved the card ``open -> closed``
    and its withdraw has not landed yet. The decision refuses
    ``requester_gone`` and submits nothing, whatever the requester flags say."""
    root, child = _family(tmp_path)
    ext = _Extension()
    assert not (await _browser("request_access", child._build_tool_context(), ext)).is_error
    [record] = _open(root)
    root._site_cards[record["ask_id"]].state = "closed"
    outcome = await _decide(root, ext, "approve")
    assert outcome["ok"] is False and outcome["code"] == "requester_gone", outcome
    assert ext.submitted == [] and ext.allowed is False


@pytest.mark.asyncio
async def test_a_refused_decision_racing_teardown_is_recorded_withdrawn(tmp_path: Path) -> None:
    """R2-1, the decision is in flight when the teardown lands, and the
    extension REFUSES it (the entry expired there). The teardown waited for
    that outcome and then withdrew the card: ``withdrawn`` exactly because
    nothing was applied."""
    import asyncio

    root, child = _family(tmp_path)
    ext = _Extension()
    assert not (await _browser("request_access", child._build_tool_context(), ext)).is_error
    [record] = _open(root)
    teardown: list[Any] = []

    async def _teardown_then_expire() -> None:
        teardown.append(asyncio.ensure_future(child.dispose()))
        await _until(lambda: child._site_requests_closed)
        # The teardown is parked on the deciding card, so the card is still open.
        assert _open(root) != []
        ext.state = "expired"

    ext.before_decide = _teardown_then_expire
    outcome = await _decide(root, ext, "approve")
    await teardown[0]
    assert outcome["ok"] is False and ext.submitted == [] and ext.allowed is False
    assert _status(root, record["ask_id"]) == "withdrawn"
    assert root._site_cards == {}


@pytest.mark.asyncio
async def test_a_teardown_whose_wait_ran_out_leaves_the_record_to_the_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R2-1, bounded wait: the submit outlives the teardown's wait. The
    teardown does not withdraw a card whose decision is still at the
    extension; the decision records the truth when it returns."""
    from local_operator.session.session import Session

    monkeypatch.setattr(Session, "_SITE_TEARDOWN_DECIDE_WAIT_S", 0)
    root, child = _family(tmp_path)
    ext = _Extension()
    assert not (await _browser("request_access", child._build_tool_context(), ext)).is_error
    [record] = _open(root)

    async def _teardown_times_out() -> None:
        await child.dispose()  # completes: its wait ran out
        assert _open(root) != [], "a deciding card is not withdrawn by a timed-out teardown"

    ext.after_apply = _teardown_times_out
    outcome = await _decide(root, ext, "approve")
    assert outcome["ok"] is True and ext.allowed is True
    assert _status(root, record["ask_id"]) == "answered"


@pytest.mark.asyncio
async def test_a_waiters_timed_out_teardown_leaves_a_live_owners_card_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """60334 R3-1. Fails on 77972952d: a WAITER's teardown counted the shared
    card as in flight, its wait ran out, and it handed the card to the
    decision; the extension refused, and the decision withdrew a card whose
    OWNER was still waiting on it."""
    from local_operator.session.session import Session

    monkeypatch.setattr(Session, "_SITE_TEARDOWN_DECIDE_WAIT_S", 0)
    root, owner = _family(tmp_path)
    waiter = _child_of(tmp_path, root, "sibling")
    ext = _Extension()
    assert not (await _browser("request_access", owner._build_tool_context(), ext)).is_error
    assert not (await _browser("request_access", waiter._build_tool_context(), ext)).is_error
    [record] = _open(root)
    binding = root._site_cards[record["ask_id"]]
    assert binding.owner() is owner

    async def _waiter_ends_then_refuse() -> None:
        await waiter.dispose()  # completes: its wait on the deciding card ran out
        ext.refuse = "no_live_request"

    ext.before_decide = _waiter_ends_then_refuse
    outcome = await _decide(root, ext, "approve")
    assert outcome["ok"] is False and ext.allowed is False, outcome
    assert _status(root, record["ask_id"]) != "withdrawn"
    assert [open_record["ask_id"] for open_record in _open(root)] == [record["ask_id"]]
    assert root._site_cards.get(record["ask_id"]) is binding and binding.state == "open"
    assert ext.entry_of(f"session:{owner.session_id}") in owner._pending_site_requests


@pytest.mark.asyncio
async def test_an_owners_timed_out_teardown_cancels_its_entry_when_the_decision_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """60334 R3-2 / Q3-1. Fails on 77972952d: the owner's teardown skipped its
    entry cancel because the deciding card still bound the entry, and the
    decision's later withdraw never cancelled it, so the dead owner's entry
    stayed pending at the extension."""
    from local_operator.session.session import Session

    monkeypatch.setattr(Session, "_SITE_TEARDOWN_DECIDE_WAIT_S", 0)
    root, child = _family(tmp_path)
    ext = _Extension()
    assert not (await _browser("request_access", child._build_tool_context(), ext)).is_error
    [record] = _open(root)

    async def _owner_ends_then_refuse() -> None:
        await child.dispose()  # completes: its wait ran out
        assert ext.state == "pending", "the bound entry is left to the decision"
        ext.refuse = "no_live_request"

    ext.before_decide = _owner_ends_then_refuse
    outcome = await _decide(root, ext, "approve")
    assert outcome["ok"] is False and ext.allowed is False, outcome
    assert _status(root, record["ask_id"]) == "withdrawn"
    assert root._site_cards == {}
    assert _cancelled_by(ext, child) and ext.state == "none"


# --- one card per origin, bound per requester (S60334-2) ---------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("order", ["child-then-root", "root-then-child", "sibling-then-sibling"])
@pytest.mark.parametrize("word", ["approve", "deny"])
async def test_a_shared_card_resolves_every_requester(
    tmp_path: Path, order: str, word: str
) -> None:
    """Fails on 59b440bdf: the second requester was told "pending" on a card
    bound to the FIRST requester's entry (or refused outright for a sibling),
    so a deny left it waiting on an entry nobody would decide."""
    root, child = _family(tmp_path)
    sibling = _child_of(tmp_path, root, "sibling")
    first, second = {
        "child-then-root": (child, root),
        "root-then-child": (root, child),
        "sibling-then-sibling": (child, sibling),
    }[order]
    ext = _Extension()
    raised = await _browser("request_access", first._build_tool_context(), ext)
    assert not raised.is_error and raised.details.get("shared") is None, _text_of(raised)
    joined = await _browser("request_access", second._build_tool_context(), ext)
    assert not joined.is_error, _text_of(joined)
    assert joined.details["shared"] is True and joined.details["state"] == "pending"
    assert "already has an open card" in _text_of(joined)
    [record] = _open(root)
    assert joined.details["ask_id"] == record["ask_id"] == raised.details["ask_id"]
    binding = root._site_cards[record["ask_id"]]
    assert binding.owner() is first
    assert [ref() for ref, _facts in binding.waiters.values()] == [second]

    outcome = await _decide(root, ext, word)
    assert outcome["ok"] is True, outcome
    assert root._site_cards == {} and _open(root) == []
    first_await = await _browser("await_access", first._build_tool_context(), ext, timeout_s=5)
    second_await = await _browser("await_access", second._build_tool_context(), ext, timeout_s=5)
    if word == "approve":
        assert first_await.details["state"] == "allowed"
        assert second_await.details["state"] == "allowed"
    else:
        assert first_await.details["state"] == "denied"
        # Not left pending: the waiter's entry was withdrawn at the deny ...
        assert _cancelled_by(ext, second)
        assert second_await.details["state"] == "none", _text_of(second_await)
        assert "no live request" in _text_of(second_await)
        # ... and its next request answers the operator's sticky deny.
        again = await _browser("request_access", second._build_tool_context(), ext)
        assert again.details["state"] == "denied" and _open(root) == []
    for who in (first, second):
        assert who._pending_site_requests == {}


@pytest.mark.asyncio
async def test_a_withdrawn_owner_hands_the_card_to_its_waiter(tmp_path: Path) -> None:
    """The owner cancels (or ends) while a sibling waits on its card: the card
    passes to the waiter, bound to the waiter's own entry, and its answer
    decides the waiter's request."""
    root, child = _family(tmp_path)
    sibling = _child_of(tmp_path, root, "sibling")
    ext = _Extension()
    assert not (await _browser("request_access", child._build_tool_context(), ext)).is_error
    joined = await _browser("request_access", sibling._build_tool_context(), ext)
    assert joined.details["shared"] is True
    await child.dispose()
    [record] = _open(root)
    binding = root._site_cards[record["ask_id"]]
    assert binding.owner() is sibling
    assert binding.entry_id == ext.entry_of(f"session:{sibling.session_id}")
    outcome = await _decide(root, ext, "deny")
    assert outcome["ok"] is True, outcome
    result = await _browser("await_access", sibling._build_tool_context(), ext, timeout_s=5)
    assert result.details["state"] == "denied"


@pytest.mark.asyncio
async def test_a_lapsed_shared_card_releases_its_waiters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The card lapses unanswered: its binding is pruned on the family's next
    call (review N1) and the waiter's entry is withdrawn, so its await reports
    "no live request" (the family's lapse bound then governs a re-raise)."""
    import asyncio

    root, child = _family(tmp_path)
    sibling = _child_of(tmp_path, root, "sibling")
    ext = _Extension()
    assert not (await _browser("request_access", child._build_tool_context(), ext)).is_error
    joined = await _browser("request_access", sibling._build_tool_context(), ext)
    assert joined.details["shared"] is True
    queue = root.ask_queue()
    real_now = queue._now
    monkeypatch.setattr(queue, "_now", lambda: real_now() + 3_600_000)
    assert root.site_ask_lapses(VECTOR["origin"]) == 1
    root._prune_site_cards()
    for _ in range(3):
        await asyncio.sleep(0)
    assert root._site_cards == {}
    assert sibling._pending_site_requests == {} and _cancelled_by(ext, sibling)


# --- cleanup vs rebind (60334 R2-2) ------------------------------------------------------


@pytest.mark.asyncio
async def test_a_late_lapse_cleanup_spares_the_entry_a_replacement_card_binds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R2-2: the shared card lapses and its cleanup (cancel the waiter's
    entry) is scheduled; before it runs, the waiter requests again and the
    extension hands back the same still-pending entry, so a REPLACEMENT card
    binds it. The late cleanup must not cancel that entry. Fails on
    f9afe5418: the old card's cleanup cancelled it and the approve of the
    replacement card granted nothing."""
    root, child = _family(tmp_path)
    sibling = _child_of(tmp_path, root, "sibling")
    ext = _Extension()
    assert not (await _browser("request_access", child._build_tool_context(), ext)).is_error
    assert (await _browser("request_access", sibling._build_tool_context(), ext)).details["shared"]
    entry = ext.entry_of(f"session:{sibling.session_id}")
    queue = root.ask_queue()
    real_now = queue._now
    monkeypatch.setattr(queue, "_now", lambda: real_now() + 3_600_000)
    before = set(root._background_tasks)
    root._prune_site_cards()  # the lapse cleanup is scheduled, not yet run
    cleanup = set(root._background_tasks) - before
    assert cleanup, "the lapse scheduled the waiter's entry cancel"
    again = await _browser("request_access", sibling._build_tool_context(), ext)
    assert not again.is_error, _text_of(again)
    [replacement] = _open(root)
    assert root._site_cards[replacement["ask_id"]].entry_id == entry
    # Now the old card's scheduled cleanup runs (late).
    await _until(lambda: all(task.done() for task in cleanup))
    assert ext.entries[f"session:{sibling.session_id}"]["state"] == "pending"
    outcome = await _decide(root, ext, "approve")
    assert outcome["ok"] is True, outcome
    result = await _browser("await_access", sibling._build_tool_context(), ext, timeout_s=5)
    assert result.details["state"] == "allowed"


@pytest.mark.asyncio
async def test_an_old_cards_cleanup_after_a_rebind_spares_the_heirs_entry(tmp_path: Path) -> None:
    """R2-2, rebind: the owner withdraws, its card is re-raised bound to the
    waiter's entry, and a cleanup keyed by the OLD card fires late. Cleanup is
    keyed by (card, entry): the old card's cleanup finds the entry bound to the
    live replacement and skips the cancel, so the heir's approve still grants."""
    import asyncio

    root, child = _family(tmp_path)
    sibling = _child_of(tmp_path, root, "sibling")
    ext = _Extension()
    assert not (await _browser("request_access", child._build_tool_context(), ext)).is_error
    assert (await _browser("request_access", sibling._build_tool_context(), ext)).details["shared"]
    [old] = _open(root)
    entry = ext.entry_of(f"session:{sibling.session_id}")
    cancel = sibling._site_request_cancels[entry]
    late = root._guarded_site_cancel(
        cancel, entry_id=entry, stale_ask_id=old["ask_id"], requester=sibling
    )
    await child.dispose()  # owner withdraws: the card passes to the sibling
    [replacement] = _open(root)
    assert replacement["ask_id"] != old["ask_id"]
    assert root._site_cards[replacement["ask_id"]].owner() is sibling
    await late()
    await asyncio.sleep(0)
    assert ext.entries[f"session:{sibling.session_id}"]["state"] == "pending"
    outcome = await _decide(root, ext, "approve")
    assert outcome["ok"] is True, outcome
    result = await _browser("await_access", sibling._build_tool_context(), ext, timeout_s=5)
    assert result.details["state"] == "allowed"
