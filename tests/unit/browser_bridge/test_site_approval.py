"""Signed per-site browser approvals on a headless mesh node (PR-B).

What these cells pin, and why each matters:

* the golden vector — SHARED with ``extension/tests/signed-decision.test.mjs`` —
  so the question id and the signed ``decide`` bytes cannot drift between the
  runtime (Python) and the extension (TypeScript) that re-verifies them;
* the runtime keeps the operator's signed proof beside the one-shot grant ONLY
  on a True signature verdict (a capability admission carries none, so the
  extension refuses it as ``unsigned_decision``);
* the ``site:v1:`` door decides on a session with NO Team Bridge runtime, calls
  the browser FIRST and records the answer only when the browser applied it;
* ``request_access`` on a node raises a code-built, ``human_required`` card, or
  says ``no_approver`` instead of waiting on a popup nobody can click.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.browser_bridge import site_approval
from local_operator.harness import approval
from local_operator.harness.approval import (
    bridge_decide_grant_committed,
    bridge_decide_grant_proof,
    bridge_decide_grant_seen,
    mint_bridge_decide_grant,
    reset_bridge_decide_grants_for_tests,
)

VECTOR_PATH = (
    Path(__file__).resolve().parents[3]
    / "extension"
    / "tests"
    / "fixtures"
    / "site-approval-vector.json"
)
VECTOR: dict[str, Any] = json.loads(VECTOR_PATH.read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def _reset_grants():
    reset_bridge_decide_grants_for_tests()
    yield
    reset_bridge_decide_grants_for_tests()


# --- the shared bytes ------------------------------------------------------------


def test_golden_vector_question_id_message_and_signature() -> None:
    from local_operator.operator.verify import (
        key_id_for,
        signed_message,
        verify_signature,
    )

    question_id = site_approval.site_question_id(
        node_device_id=VECTOR["node_device_id"],
        entry_id=VECTOR["entry_id"],
        origin=VECTOR["origin"],
        scope=VECTOR["scope"],
    )
    assert question_id == VECTOR["question_id"]
    assert site_approval.is_site_question_id(question_id)
    message = signed_message(
        action="decide",
        session_id=VECTOR["session_id"],
        request_id=f"{VECTOR['ask_id']}#{question_id}#{VECTOR['word']}",
        challenge=VECTOR["challenge"],
    )
    assert message.hex() == VECTOR["message_hex"]
    point = bytes.fromhex(VECTOR["anchor_point"])
    assert key_id_for(point) == VECTOR["anchor_key_id"]
    assert verify_signature(spki=point, message=message, signature=bytes.fromhex(VECTOR["sig_der"]))
    other = bytes.fromhex(VECTOR["other_point"])
    assert not verify_signature(
        spki=point, message=message, signature=bytes.fromhex(VECTOR["other_sig_der"])
    )
    assert verify_signature(
        spki=other, message=message, signature=bytes.fromhex(VECTOR["other_sig_der"])
    )


def test_every_bound_field_changes_the_question_id() -> None:
    base = dict(node_device_id="d_1", entry_id="e", origin="https://a.example", scope="origin")
    reference = site_approval.site_question_id(**base)
    for field, value in [
        ("node_device_id", "d_2"),
        ("entry_id", "f"),
        ("origin", "https://b.example"),
        ("scope", "domain"),
    ]:
        assert site_approval.site_question_id(**{**base, field: value}) != reference, field
    # Length prefixes: moving a byte between adjacent fields is a different id.
    assert site_approval.site_question_id(
        node_device_id="d_1e", entry_id="", origin="https://a.example"
    ) != site_approval.site_question_id(
        node_device_id="d_1", entry_id="e", origin="https://a.example"
    )


def test_the_card_question_is_code_built_and_human_required() -> None:
    question = site_approval.site_ask_question(
        origin="http://127.0.0.1:8765", node_name="node-a", node_device_id="d_1", entry_id="e"
    )
    assert question["human_required"] is True
    assert question["question_id"] == site_approval.site_question_id(
        node_device_id="d_1", entry_id="e", origin="http://127.0.0.1:8765"
    )
    assert [option["id"] for option in question["options"]] == ["approve", "deny"]
    assert "http://127.0.0.1:8765" in question["question"]
    assert "node-a" in question["question"]
    assert question["secret"] is False and question["multi"] is False


def test_policy_fields_carry_the_anchor_only_when_there_is_one() -> None:
    assert site_approval.policy_fields(None, "", "") == {"headlessNode": True}
    point = bytes.fromhex(VECTOR["anchor_point"])
    assert site_approval.policy_fields(point, VECTOR["anchor_key_id"], "d_1") == {
        "headlessNode": True,
        "operatorAnchorPoint": VECTOR["anchor_point"],
        "operatorKeyId": VECTOR["anchor_key_id"],
        "nodeDeviceId": "d_1",
    }


def test_anchor_key_held_locally_reads_the_file_backend(tmp_path: Path) -> None:
    from local_operator.operator.keychain import FileKeyBackend, default_file_path

    assert site_approval.anchor_key_held_locally(tmp_path, VECTOR["anchor_key_id"]) is False
    handle = FileKeyBackend(default_file_path(tmp_path)).create()
    assert site_approval.anchor_key_held_locally(tmp_path, handle.key_id) is True
    assert site_approval.anchor_key_held_locally(tmp_path, VECTOR["anchor_key_id"]) is False
    assert site_approval.anchor_key_held_locally(tmp_path, "") is False


# --- the proof stash ---------------------------------------------------------------


def test_a_grant_keeps_its_signed_proof_until_spent() -> None:
    target: dict[str, Any] = dict(session_id="s", ask_id="a", question_id="site:v1:q")
    proof = {"operator_sig": "30", "operator_key_id": "k", "challenge": "c"}
    mint_bridge_decide_grant(**target, word="approve", proof=proof)
    assert bridge_decide_grant_proof(**target, word="approve") == proof
    assert bridge_decide_grant_proof(**target, word="deny") is None
    approval.consume_bridge_decide_grant(**target, word="approve")
    assert bridge_decide_grant_proof(**target, word="approve") is None
    mint_bridge_decide_grant(**target, word="approve")
    assert bridge_decide_grant_proof(**target, word="approve") is None


def _seam(
    verdict: bool | None, challenge: str, *, capability: bool = False, cert: Any = None
) -> Any:
    """``_authority_admitted`` on a minimal stand-in for the runtime server."""
    from local_operator.session.runtime import server as server_mod

    fake = SimpleNamespace(
        _operator_cap="cap",
        _record=SimpleNamespace(session_id="sess-seam"),
        _operator_signature_check=lambda frame, conn: (verdict, challenge),
    )
    conn = SimpleNamespace(operator_nonce="n", operator_salt="s")
    frame = {
        "op": "bridge_decide",
        "ask_id": "ask-1",
        "question_id": "site:v1:q",
        "word": "approve",
        "operator_sig": "3044",
        "operator_key_id": "kid",
    }
    if cert is not None:
        frame["operator_cert"] = cert
    original = server_mod.request_proof_ok
    server_mod.request_proof_ok = lambda **_: capability  # type: ignore[assignment]
    try:
        admit: Any = server_mod.RuntimeServer._authority_admitted
        return admit(fake, frame, conn)
    finally:
        server_mod.request_proof_ok = original  # type: ignore[assignment]


@pytest.mark.parametrize(
    ("verdict", "capability", "admitted", "proof"),
    [
        (True, False, True, {"operator_sig": "3044", "operator_key_id": "kid", "challenge": "ch"}),
        (False, False, False, None),
        (None, True, True, None),
        (False, True, True, None),
    ],
)
def test_the_seam_stashes_proof_only_on_a_true_signature_verdict(
    verdict: bool | None, capability: bool, admitted: bool, proof: dict[str, str] | None
) -> None:
    assert _seam(verdict, "ch", capability=capability) is admitted
    target = dict(session_id="sess-seam", ask_id="ask-1", question_id="site:v1:q")
    assert bridge_decide_grant_seen(**target, word="approve") is admitted
    assert bridge_decide_grant_proof(**target, word="approve") == proof


def test_a_device_certificate_answer_is_admitted_without_proof() -> None:
    """R1-4: a paired device's signature (verdict True through its certificate)
    still admits the decision on the runtime, but no proof is stashed — the
    node's extension knows only the anchor point, so it would have called the
    signature "bad"; without proof it refuses as unsigned, whose sentence says
    to answer from the operator's own computer."""
    assert _seam(True, "ch", cert={"device": "phone"}) is True
    target = dict(session_id="sess-seam", ask_id="ask-1", question_id="site:v1:q")
    assert bridge_decide_grant_seen(**target, word="approve") is True
    assert bridge_decide_grant_proof(**target, word="approve") is None


def test_anchor_policy_fields_omits_an_anchor_whose_key_this_node_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1-2: the extension itself reports ``no_approver`` on a node that holds
    the anchor's private key — the agent could otherwise sign its own decision
    straight into ``submit_signed_decision``."""
    from local_operator.network import identity
    from local_operator.operator import trust
    from local_operator.operator.keychain import FileKeyBackend, default_file_path

    held = FileKeyBackend(default_file_path(tmp_path / "held")).create()
    point = bytes.fromhex(VECTOR["anchor_point"])

    def anchor_for(key_id: str) -> Any:
        return SimpleNamespace(usable=True, anchor=SimpleNamespace(spki=point, key_id=key_id))

    monkeypatch.setattr(identity, "load", lambda: SimpleNamespace(device_id="d_node"))
    monkeypatch.setattr(trust, "load_anchor", lambda: anchor_for(held.key_id))
    assert site_approval.anchor_policy_fields(tmp_path / "held") == {"headlessNode": True}
    monkeypatch.setattr(trust, "load_anchor", lambda: anchor_for(VECTOR["anchor_key_id"]))
    assert site_approval.anchor_policy_fields(tmp_path / "held")["operatorKeyId"] == (
        VECTOR["anchor_key_id"]
    )


# --- the site door on a real session -------------------------------------------------


async def _session_with_site_ask(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tests.unit.session.test_session import ScriptedStream, make_session

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    session = make_session(tmp_path, ScriptedStream([[]]))

    async def _never(_questions):
        return None  # pragma: no cover

    session.set_ask_handler(_never)
    assert session.bridge_runtime is None, "the site door must not need Team Bridge"
    question_id = VECTOR["question_id"]
    question = site_approval.site_ask_question(
        origin=VECTOR["origin"],
        node_name="node-a",
        node_device_id=VECTOR["node_device_id"],
        entry_id=VECTOR["entry_id"],
    )
    assert question["question_id"] == question_id
    receipt = session._enqueue_ask([question], 600)
    assert receipt["ok"], receipt
    # The ``site`` block survives the queue's projection: it is what the
    # signer on the operator's machine rebuilds the card from (R1-1).
    stored = _record(session, receipt["details"]["ask_id"])["questions"][0]
    assert stored["site"] == question["site"]
    return session, receipt["details"]["ask_id"], question_id


def _record(session, ask_id: str) -> dict[str, Any]:
    return [r for r in session.ask_queue().records() if r.get("ask_id") == ask_id][-1]


@pytest.mark.asyncio
async def test_a_signed_site_decision_reaches_the_browser_then_the_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, ask_id, question_id = await _session_with_site_ask(tmp_path, monkeypatch)
    proof = {"operator_sig": "3044ab", "operator_key_id": "kid", "challenge": "ch"}
    mint_bridge_decide_grant(
        session_id=session.session_id,
        ask_id=ask_id,
        question_id=question_id,
        word="approve",
        proof=proof,
    )
    sent: list[dict[str, Any]] = []

    async def submit(params: dict[str, Any]) -> dict[str, Any]:
        sent.append(params)
        assert _record(session, ask_id)["status"] == "open", "browser first, record second"
        return {"applied": True, "state": "allowed", "origin": VECTOR["origin"]}

    outcome = await session.site_decide(ask_id, question_id, "Approve", submit=submit)
    assert outcome == {"ok": True, "word": "approve", "state": "allowed"}
    assert sent == [
        {
            "session_id": session.session_id,
            "ask_id": ask_id,
            "question_id": question_id,
            "word": "approve",
            "challenge": "ch",
            "sig": "3044ab",
            "key_id": "kid",
        }
    ]
    record = _record(session, ask_id)
    assert record["status"] == "answered"
    assert record["answers"] == {"site-access": ["approve"]}
    assert bridge_decide_grant_committed(
        session_id=session.session_id, ask_id=ask_id, question_id=question_id, word="approve"
    )
    await session.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason", ["unsigned_decision", "bad_signature", "wrong_key", "no_live_request", "no_approver"]
)
async def test_a_refused_site_decision_records_nothing_and_withdraws_the_grant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reason: str
) -> None:
    session, ask_id, question_id = await _session_with_site_ask(tmp_path, monkeypatch)
    # A CAPABILITY admission (the node's own console): no proof was stashed.
    mint_bridge_decide_grant(
        session_id=session.session_id, ask_id=ask_id, question_id=question_id, word="approve"
    )
    sent: list[dict[str, Any]] = []

    async def submit(params: dict[str, Any]) -> dict[str, Any]:
        sent.append(params)
        return {"applied": False, "reason": reason}

    outcome = await session.site_decide(ask_id, question_id, "approve", submit=submit)
    assert outcome["ok"] is False and outcome["code"] == reason
    assert outcome["error"]
    # Each refusal says ITS cause (R1-4): a node with no anchor in policy is
    # not a "bad signature", and the signature sentences are not reused for it.
    from local_operator.i18n.keys import browser_site_approval as copy
    from local_operator.i18n.messages import render_resolved

    expected = {
        "unsigned_decision": copy.refused_unsigned(),
        "bad_signature": copy.refused_bad_signature(),
        "wrong_key": copy.refused_bad_signature(),
        "no_approver": copy.refused_no_approver(),
    }.get(reason, copy.refused_expired())
    assert outcome["error"] == render_resolved(expected)
    assert sent[0]["sig"] == "" and sent[0]["challenge"] == ""
    assert _record(session, ask_id)["status"] == "open"
    assert not bridge_decide_grant_seen(
        session_id=session.session_id, ask_id=ask_id, question_id=question_id, word="approve"
    )
    await session.dispose()


@pytest.mark.asyncio
async def test_an_unreachable_browser_decides_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, ask_id, question_id = await _session_with_site_ask(tmp_path, monkeypatch)
    mint_bridge_decide_grant(
        session_id=session.session_id, ask_id=ask_id, question_id=question_id, word="deny"
    )

    async def submit(_params: dict[str, Any]) -> dict[str, Any]:
        raise ConnectionError("bridge down")

    outcome = await session.site_decide(ask_id, question_id, "deny", submit=submit)
    assert outcome["code"] == "site_unreachable"
    assert _record(session, ask_id)["status"] == "open"
    await session.dispose()


@pytest.mark.asyncio
async def test_the_plain_answer_door_still_refuses_a_site_ask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unsigned ``ask_respond`` (a same-uid socket caller, the model) can
    never answer the card: the question id makes it a verified-door ask."""
    session, ask_id, _question_id = await _session_with_site_ask(tmp_path, monkeypatch)
    outcome = session.respond_ask(ask_id, {"site-access": ["approve"]}, by="tui")
    assert outcome["ok"] is False
    assert _record(session, ask_id)["status"] == "open"
    await session.dispose()


# --- the tool on a node ------------------------------------------------------------


class _Client:
    host = "extension"

    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((method, params))
        if method == "request_access":
            return self.result
        return {"state": "cancelled"}


def _context(enqueued: list[Any], receipt: dict[str, Any] | None = None) -> Any:
    from local_operator.harness.types import ToolContext

    def enqueue(questions, timeout=None, tool_call_id=""):
        enqueued.append((questions, timeout, tool_call_id))
        return receipt or {"ok": True, "text": "queued", "details": {"ask_id": "ask_node_1"}}

    return ToolContext(enqueue_ask=enqueue)


async def _request(client: _Client, context: Any) -> Any:
    from local_operator.tools import builtin

    params = builtin.BrowserParams(action="request_access", url="http://127.0.0.1:8765/x")
    return await builtin._bridge_access("call-1", "request_access", params, context, client=client)


def _text_of(result: Any) -> str:
    # Receipts are hard-wrapped for the card; read them as prose.
    return " ".join("".join(getattr(block, "text", "") for block in result.content).split())


@pytest.mark.asyncio
async def test_request_access_on_a_node_queues_the_signed_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "local_operator.operator.trust.load_anchor",
        lambda *a, **k: SimpleNamespace(anchor=None, usable=False),
    )
    client = _Client(
        {
            "origin": VECTOR["origin"],
            "state": "pending",
            "entryId": VECTOR["entry_id"],
            "approval": "signed",
            "scope": "origin",
            "node_device_id": VECTOR["node_device_id"],
        }
    )
    enqueued: list[Any] = []
    result = await _request(client, _context(enqueued))
    assert not result.is_error, _text_of(result)
    [(questions, timeout, call_id)] = enqueued
    [question] = questions
    assert question["question_id"] == VECTOR["question_id"]
    assert question["human_required"] is True
    assert VECTOR["origin"] in question["question"]
    assert 120 <= timeout <= 600 and call_id == "call-1"
    assert result.details["question_id"] == VECTOR["question_id"]
    assert result.details["ask_id"] == "ask_node_1"
    assert "await_access" in _text_of(result)
    assert "extension popup" not in _text_of(result)


@pytest.mark.asyncio
async def test_request_access_on_a_node_without_an_anchor_is_no_approver() -> None:
    client = _Client({"origin": VECTOR["origin"], "state": "none", "approval": "no_approver"})
    enqueued: list[Any] = []
    result = await _request(client, _context(enqueued))
    assert result.is_error
    assert result.details["approval"] == "no_approver"
    assert "Do not retry request_access" in _text_of(result)
    assert enqueued == []


@pytest.mark.asyncio
async def test_a_node_that_holds_the_anchor_key_has_no_approver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The agent could sign its own approval, so nobody independent decides."""
    from local_operator.operator.keychain import FileKeyBackend, default_file_path

    handle = FileKeyBackend(default_file_path(tmp_path)).create()
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: tmp_path)
    monkeypatch.setattr(
        "local_operator.operator.trust.load_anchor",
        lambda *a, **k: SimpleNamespace(anchor=SimpleNamespace(key_id=handle.key_id), usable=True),
    )
    client = _Client(
        {
            "origin": VECTOR["origin"],
            "state": "pending",
            "entryId": VECTOR["entry_id"],
            "approval": "signed",
            "node_device_id": "d_1",
        }
    )
    enqueued: list[Any] = []
    result = await _request(client, _context(enqueued))
    assert result.is_error and result.details["approval"] == "no_approver"
    assert enqueued == []
    assert [method for method, _ in client.calls] == ["request_access", "cancel_access"]


@pytest.mark.asyncio
async def test_a_desktop_extension_keeps_the_popup_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Client({"origin": VECTOR["origin"], "state": "pending", "entryId": "e"})
    enqueued: list[Any] = []
    result = await _request(client, _context(enqueued))
    assert enqueued == []
    assert "approval" not in result.details


def test_the_terminal_labels_a_site_ask_as_its_own_kind() -> None:
    """A site ask rides the bridge DECISION door, but it is not another
    member's request: the TUI's bridge copy would say so, so it gets its own."""
    from local_operator.tui.app import _is_bridge_ask_row, _is_site_ask_row

    site = SimpleNamespace(
        questions=[
            site_approval.site_ask_question(
                origin="https://a.example", node_name="n", node_device_id="d", entry_id="e"
            )
        ]
    )
    bridge = SimpleNamespace(questions=[{"id": "bridge-m", "question_id": "bmsg_1"}])
    assert _is_site_ask_row(site) and _is_bridge_ask_row(site)
    assert not _is_site_ask_row(bridge)
    assert not _is_site_ask_row(SimpleNamespace(questions=[{"id": "q0"}]))


@pytest.mark.asyncio
async def test_the_runtime_op_routes_a_site_question_to_the_site_door(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The runtime's ``bridge_decide`` op takes the ``site:v1:`` door AHEAD of
    the Team Bridge door, which refuses outright on a session with no bridge
    runtime ("agent messages are not active") — a node's usual state."""
    import asyncio

    from local_operator.session.runtime.serving import ServingSessionHandle

    session, ask_id, question_id = await _session_with_site_ask(tmp_path, monkeypatch)
    seen: list[tuple[str, str, str]] = []

    async def site_decide(a: str, q: str, w: str) -> dict[str, Any]:
        seen.append((a, q, w))
        return {"ok": False, "error": "refused by the browser", "code": "bad_signature"}

    monkeypatch.setattr(session, "site_decide", site_decide)
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    with pytest.raises(ValueError, match="refused by the browser"):
        await handle.bridge_decide(ask_id, question_id, "approve")
    assert seen == [(ask_id, question_id, "approve")]
    # A non-site question still takes the Team Bridge door, and its refusal.
    with pytest.raises(ValueError, match="agent messages are not active"):
        await handle.bridge_decide(ask_id, "bmsg_1", "approve")
    await session.dispose()


# --- R1-1: the card the operator reads is the card the signature approves -------
#
# The question text is drawn on the NODE, in a runtime the agent shares a uid
# with, and the signature covers only the opaque ``site:v1:`` digest. So the
# SIGNER (the Mac's TUI and desktop route, through ``asks/decision.py``) rebuilds
# both the digest and the card from the structured ``site`` block before it
# signs. These cards are built by hand, not through ``site_ask_question``, so
# the honest one is byte-for-byte what a node publishes and the forged one is
# what an agent appending to ``asks.jsonl`` could publish.


def _card(*, shown_origin: str, bound_origin: str, node: str = "node-a") -> dict[str, Any]:
    from local_operator.i18n.messages import render

    site = {
        "node_device_id": VECTOR["node_device_id"],
        "entry_id": VECTOR["entry_id"],
        "origin": bound_origin,
        "scope": "origin",
        "node_name": node,
    }
    return {
        "id": "site-access",
        "question": render(
            "browser.site_approval.question", {"origin": shown_origin, "node": node}, locale="en"
        ),
        "options": [
            {
                "id": "approve",
                "label": render("browser.site_approval.option_approve", {}, locale="en"),
            },
            {"id": "deny", "label": render("browser.site_approval.option_deny", {}, locale="en")},
        ],
        "multi": False,
        "secret": False,
        "human_required": True,
        "question_id": site_approval.site_question_id(
            node_device_id=VECTOR["node_device_id"],
            entry_id=VECTOR["entry_id"],
            origin=bound_origin,
        ),
        "site": site,
    }


def _row(question: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(ask_id="a-site", questions=[question])


HONEST = "http://127.0.0.1:8765"
EVIL = "https://evil.example"


def test_the_signer_maps_an_honest_card_to_its_decision() -> None:
    from local_operator.asks.decision import bridge_decision_target

    card = _card(shown_origin=HONEST, bound_origin=HONEST)
    assert bridge_decision_target(_row(card), {"site-access": ["Approve"]}) == (
        "a-site",
        card["question_id"],
        "approve",
    )


def test_the_card_builder_and_the_signer_agree() -> None:
    """The node's builder and the signer's check share one template."""
    from local_operator.asks.decision import bridge_decision_target

    built = site_approval.site_ask_question(
        origin=HONEST,
        node_name="node-a",
        node_device_id=VECTOR["node_device_id"],
        entry_id=VECTOR["entry_id"],
    )
    assert built["question_id"] == VECTOR["question_id"]
    assert built["site"]["origin"] == HONEST
    target = bridge_decision_target(_row(built), {"site-access": ["deny"]})
    assert target is not None and target[2] == "deny"


@pytest.mark.parametrize(
    "mutate",
    [
        # The attack in R1-1: the text says loopback, the digest binds evil.example.
        lambda c: c.update(_card(shown_origin=HONEST, bound_origin=EVIL)),
        # The site block claims loopback, the digest is evil.example's.
        lambda c: c.update(question_id=_card(shown_origin=EVIL, bound_origin=EVIL)["question_id"]),
        # Labels swapped: the operator taps "Deny" and signs approve.
        lambda c: c.update(
            options=[{"id": "approve", "label": "Deny"}, {"id": "deny", "label": "Approve"}]
        ),
        # Free agent prose appended to the code-built sentence.
        lambda c: c.update(question=c["question"] + " (safe: it is only loopback)"),
        # No structured block at all: nothing to check the digest against.
        lambda c: c.pop("site"),
        lambda c: c.update(site={**c["site"], "origin": 7}),
    ],
    ids=["text-a-id-b", "block-a-id-b", "labels-swapped", "extra-prose", "no-block", "bad-block"],
)
def test_the_signer_refuses_a_card_it_cannot_rebuild(mutate) -> None:
    from local_operator.asks.decision import bridge_decision_target

    card = _card(shown_origin=HONEST, bound_origin=HONEST)
    mutate(card)
    with pytest.raises(site_approval.SiteCardMismatch) as caught:
        bridge_decision_target(_row(card), {"site-access": ["Approve"]})
    assert caught.value.code == "site_card_mismatch"
    assert "not signed" in str(caught.value)


def test_a_card_drawn_in_another_shipped_locale_still_verifies() -> None:
    """The node renders in ITS language; the Mac may run another. Any shipped
    locale's template over the SAME bound fields is honest — the template is
    code, only the fields vary, and the fields are what the digest binds."""
    from local_operator.asks.decision import bridge_decision_target
    from local_operator.i18n.messages import render

    card = _card(shown_origin=HONEST, bound_origin=HONEST)
    card["question"] = render(
        "browser.site_approval.question", {"origin": HONEST, "node": "node-a"}, locale="fr"
    )
    card["options"] = [
        {"id": "approve", "label": render("browser.site_approval.option_approve", {}, locale="fr")},
        {"id": "deny", "label": render("browser.site_approval.option_deny", {}, locale="fr")},
    ]
    assert bridge_decision_target(_row(card), {"site-access": ["approve"]}) is not None


@pytest.mark.parametrize("shown", [HONEST, EVIL], ids=["mislabeled", "honest"])
def test_the_terminal_owner_door_signs_only_an_honest_card(shown: str) -> None:
    """The TUI's ``_deliver_owner_answer`` (the attached facade signs on the way
    out) refuses a mislabeled card with the typed sentence and never calls the
    signing door; the honest card reaches it."""
    from local_operator.tui.app import OperatorApp
    from local_operator.tui.widgets.ask_queue import AskRow

    decided: list[tuple[str, str, str]] = []
    session = SimpleNamespace(bridge_decide=lambda *target: decided.append(target) or {"ok": True})
    card = _card(shown_origin=shown, bound_origin=EVIL)
    row = AskRow(
        ask_id="a-site",
        status="pending",
        created_at=0,
        expires_at=0,
        urgent=False,
        questions=(card,),
    )
    # Unbound: the method reads nothing off the app, and a mounted TUI adds
    # nothing to what is under test (the door the row picks).
    deliver: Any = OperatorApp._deliver_owner_answer
    verdict = deliver(SimpleNamespace(), session, row, "a-site", {"site-access": ["Approve"]})
    if shown == EVIL:
        assert decided == [("a-site", card["question_id"], "approve")]
    else:
        assert decided == []
        assert verdict["ok"] is False
        assert verdict["code"] == "site_card_mismatch"
        assert "not signed" in verdict["error"]
