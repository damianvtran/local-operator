"""A ``task`` subagent raises the signed site card through its ancestor's queue.

Security review 60334 is the acceptance list these cells pin:

* the ancestor takes TYPED facts only and rebuilds the canonical card itself —
  byte-identical to a top-level session's card for the same facts;
* the card is bound to the actual child (live lineage), the node and the live
  request (pending in the child, unexpired); anything else is refused;
* withdraw/lapse bookkeeping is scoped to that child's own cards;
* the operator-signed approve, the sticky deny and expiry resolve the child's
  ``await_access`` exactly as for a top-level session (it polls the extension);
* with no ancestor that can queue, the child fails closed and says why.
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


def _family(tmp_path: Path, *, root_queue: bool = True):
    """A root (with an ask surface unless ``root_queue`` is off) and its child."""
    from tests.unit.session.test_session import ScriptedStream, make_session

    root = make_session(tmp_path / "root", ScriptedStream([[]]))
    if root_queue:

        async def _never(_questions):
            return None  # pragma: no cover

        root.set_ask_handler(_never)
    child = make_session(
        tmp_path / "child",
        ScriptedStream([[]]),
        parent_session=root,
        parent_session_id=str(root.session_id),
    )
    assert child.ask_queue() is None, "a task child has no queue of its own"
    return root, child


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
    """The node's browser extension, as far as the tool sees it: one entry per
    request, decided only by ``submit_signed_decision`` (via ``site_decide``)."""

    def __init__(self, entry_id: str = VECTOR["entry_id"]) -> None:
        self.entry_id = entry_id
        self.state = "pending"
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((method, params))
        if method == "request_access":
            if self.state == "denied":
                return {"origin": VECTOR["origin"], "state": "denied", "approval": "signed"}
            if self.state == "allowed":
                # The signed approve's ``site`` decision stored a PERSISTENT
                # exact-origin grant (``approval-store.decideAccess`` ->
                # ``grantExactOriginLocked``), so ``callerAlreadyAllowed``
                # answers before anything is queued.
                return {"origin": VECTOR["origin"], "state": "allowed"}
            return {
                "origin": VECTOR["origin"],
                "state": "pending",
                "entryId": self.entry_id,
                "approval": "signed",
                "scope": "origin",
                "node_device_id": VECTOR["node_device_id"],
                "expires_at": int(time.time() * 1000) + 600_000,
            }
        if method == "cancel_access":
            self.state = "none"
            return {"origin": VECTOR["origin"], "state": "cancelled"}
        return {"origin": VECTOR["origin"], "state": self.state}

    async def submit(self, params: dict[str, Any]) -> dict[str, Any]:
        word = params["word"]
        self.state = "allowed" if word == "approve" else "denied"
        return {"applied": True, "state": self.state, "origin": VECTOR["origin"]}


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
    assert root._child_site_asks[record["ask_id"]] == (
        child.session_id,
        VECTOR["entry_id"],
        VECTOR["origin"],
    )
    # The control: the SAME queue takes the top-level card once the child's
    # identical card is withdrawn (an open identical text is refused).
    assert child.withdraw_child_site_asks(VECTOR["origin"]) == [record["ask_id"]]
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
        assert child.raise_site_card_via_ancestor(payload)["ok"] is False
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
    from tests.unit.session.test_session import ScriptedStream, make_session

    sibling = make_session(
        tmp_path / "sib",
        ScriptedStream([[]]),
        parent_session=root,
        parent_session_id=str(root.session_id),
    )
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
    outcome = child.raise_site_card_via_ancestor(stale, 600)
    assert outcome["ok"] is False and "expired" in outcome["error"]
    assert child._pending_site_requests == {}
    assert _open(root) == []


@pytest.mark.asyncio
async def test_a_request_bound_to_another_node_is_refused(tmp_path: Path) -> None:
    root, child = _family(tmp_path)
    outcome = child.raise_site_card_via_ancestor(_facts(node_device_id="another-node"), 600)
    assert outcome["ok"] is False and "another node" in outcome["error"]
    assert _open(root) == []


@pytest.mark.asyncio
async def test_a_non_descendant_is_refused(tmp_path: Path) -> None:
    from tests.unit.session.test_session import ScriptedStream, make_session

    root, _child = _family(tmp_path)
    stranger = make_session(tmp_path / "stranger", ScriptedStream([[]]))
    # Claims the root's id as its parent id but holds no live link to it.
    impostor = make_session(
        tmp_path / "impostor", ScriptedStream([[]]), parent_session_id=str(root.session_id)
    )
    facts = _facts()
    for who in (stranger, impostor, root):
        who._pending_site_requests[facts.entry_id] = facts
        outcome = root.enqueue_child_site_card(facts, child=who, request_id=facts.entry_id)
        assert outcome["ok"] is False and "not a live subagent" in outcome["error"]
    assert _open(root) == []


# --- scoped bookkeeping --------------------------------------------------------------


@pytest.mark.asyncio
async def test_withdraw_and_lapses_are_scoped_to_the_childs_own_cards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, child = _family(tmp_path)
    # The root's OWN card for another entry of the same origin.
    own = _facts(entry_id="root-entry").card(node_name=NODE_NAME)
    own_receipt = root._enqueue_ask([own], 600)
    assert own_receipt["ok"]
    own_id = own_receipt["details"]["ask_id"]
    root.withdraw_site_asks(VECTOR["origin"], keep_question_id=own["question_id"])
    # The same text is open for the root's entry: a child must not be told
    # its request is pending on the root's card.
    clash = child.raise_site_card_via_ancestor(_facts(), 600)
    assert clash["ok"] is False and "another session in this family" in clash["error"]
    assert root.withdraw_site_asks(VECTOR["origin"]) == [own_id]

    child_receipt = child.raise_site_card_via_ancestor(_facts(), 600)
    assert child_receipt["ok"], child_receipt
    child_ask = child_receipt["details"]["ask_id"]
    # The root's own withdraw leaves the child's card alone ...
    assert root.withdraw_site_asks(VECTOR["origin"]) == []
    # ... and the child's withdraw reaches only its own.
    assert child.withdraw_child_site_asks(VECTOR["origin"]) == [child_ask]
    assert child._pending_site_requests == {}
    # A withdrawn request is no longer pending: its facts raise nothing.
    stale = root.enqueue_child_site_card(_facts(), child=child, request_id=VECTOR["entry_id"])
    assert stale["ok"] is False and "not pending" in stale["error"]

    # Lapses: a child's lapsed card counts for the child, not the root.
    lapsing = child.raise_site_card_via_ancestor(_facts(entry_id="entry-3"), 600)
    assert lapsing["ok"], lapsing
    queue = root.ask_queue()
    real_now = queue._now
    monkeypatch.setattr(queue, "_now", lambda: real_now() + 3_600_000)
    assert child.child_site_ask_lapses(VECTOR["origin"]) == 1
    assert root.site_ask_lapses(VECTOR["origin"]) == 0


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
async def test_a_top_level_session_keeps_its_own_door(tmp_path: Path) -> None:
    root, _child = _family(tmp_path)
    context = root._build_tool_context()
    assert callable(context.enqueue_ask) and context.enqueue_site_card is None
    result = await _browser("request_access", context, _Extension())
    assert not result.is_error, _text_of(result)
    [record] = _open(root)
    assert record["ask_id"] not in root._child_site_asks


@pytest.mark.asyncio
async def test_a_grandchild_reaches_the_nearest_ancestor_with_a_queue(tmp_path: Path) -> None:
    from tests.unit.session.test_session import ScriptedStream, make_session

    root, child = _family(tmp_path)
    grandchild = make_session(
        tmp_path / "grandchild",
        ScriptedStream([[]]),
        parent_session=child,
        parent_session_id=str(child.session_id),
    )
    context = grandchild._build_tool_context()
    result = await _browser("request_access", context, _Extension())
    assert not result.is_error, _text_of(result)
    [record] = _open(root)
    assert root._child_site_asks[record["ask_id"]][0] == grandchild.session_id


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
