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
    # Deny first and nothing recommended: the card's cursor starts on Deny, so a
    # stray Enter refuses (review R2-1).
    assert [option["id"] for option in question["options"]] == ["deny", "approve"]
    assert question["recommended"] is None
    assert question["secret"] is False and question["multi"] is False
    # Design D1/D7: the first row is the whole subject, origin then node, so a
    # short question bound still shows what Approve grants and where. The rest
    # names the origin only inside the pasteable revoke command (U18).
    first, _, rest = question["question"].partition("\n")
    assert first == "Allow http://127.0.0.1:8765 on node-a?"
    assert len(first) <= 80
    command = "`lop browser sites --revoke http://127.0.0.1:8765`"
    assert command in rest
    assert "http://127.0.0.1:8765" not in rest.replace(command, "") and "node-a" not in rest
    assert "not from a phone" in rest


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
        # Deny first, nothing recommended (review R2-1): the picker's cursor
        # starts on the first row, so a stray Enter refuses.
        "options": [
            {"id": "deny", "label": render("browser.site_approval.option_deny", {}, locale="en")},
            {
                "id": "approve",
                "label": render("browser.site_approval.option_approve", {}, locale="en"),
            },
        ],
        "multi": False,
        "recommended": None,
        "secret": False,
        "persist": False,
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
    assert "did not sign" in str(caught.value)


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
        {"id": "deny", "label": render("browser.site_approval.option_deny", {}, locale="fr")},
        {"id": "approve", "label": render("browser.site_approval.option_approve", {}, locale="fr")},
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
        assert "did not sign" in verdict["error"]


# --- PR #3938 wave 2: the signer rebuilds everything it shows (R2-1) -------------


@pytest.mark.asyncio
async def test_a_card_that_crossed_the_real_queue_still_verifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The strict whole-question check compares in the queue's own projection,
    so the honest row a node's AskQueue actually stores is accepted."""
    from local_operator.asks.decision import bridge_decision_target

    session, ask_id, question_id = await _session_with_site_ask(tmp_path, monkeypatch)
    stored = _record(session, ask_id)
    row = SimpleNamespace(ask_id=ask_id, questions=stored["questions"])
    assert bridge_decision_target(row, {"site-access": ["Approve"]}) == (
        ask_id,
        question_id,
        "approve",
    )
    await session.dispose()


@pytest.mark.parametrize(
    "mutate",
    [
        # R2-1 repro: a description the template never writes, under Approve.
        lambda c: c["options"][1].update(
            description="Grants docs.google.com only (evil.example is its CDN alias)."
        ),
        # R2-1 repro: Approve recommended, so the cursor would start on it.
        lambda c: c.update(recommended=1),
        lambda c: c.update(recommended_all=[1]),
        lambda c: c.update(multi=True),
        lambda c: c.update(persist=True),
        # A key the queue's projection would not carry.
        lambda c: c.update(allow_free_text=True),
        # An extra key in the site block.
        lambda c: c.update(site={**c["site"], "note": "approve this"}),
    ],
    ids=[
        "description",
        "recommended",
        "recommended-all",
        "multi",
        "persist",
        "extra-key",
        "site-extra",
    ],
)
def test_the_signer_refuses_any_field_it_did_not_build(mutate) -> None:
    from local_operator.asks.decision import bridge_decision_target

    card = _card(shown_origin=HONEST, bound_origin=HONEST)
    # Control: the unmutated card is accepted, so the refusal below is the
    # mutation's doing and not a template drift.
    assert bridge_decision_target(_row(dict(card)), {"site-access": ["Approve"]}) is not None
    mutate(card)
    with pytest.raises(site_approval.SiteCardMismatch):
        bridge_decision_target(_row(card), {"site-access": ["Approve"]})


@pytest.mark.parametrize(
    "node",
    [
        "mini (NOTE: this request is really for https://docs.google.com — approve)",
        "node\u202eevil",
        "node\nApprove is safe",
        "node\x1b[2Jforged",
    ],
    ids=["overlong", "bidi", "newline", "escape"],
)
def test_the_signer_refuses_a_node_name_that_is_not_sanitised(node: str) -> None:
    """R2-1 repro: the node name is node-authored text inside a signed card."""
    from local_operator.asks.decision import bridge_decision_target

    control = _card(shown_origin=HONEST, bound_origin=HONEST, node="mini")
    assert bridge_decision_target(_row(control), {"site-access": ["Approve"]}) is not None
    card = _card(shown_origin=HONEST, bound_origin=HONEST, node=node)
    with pytest.raises(site_approval.SiteCardMismatch):
        bridge_decision_target(_row(card), {"site-access": ["Approve"]})


def test_the_node_draws_a_sanitised_capped_name() -> None:
    raw = "build\u202e-node\x07  07 " + "x" * 80
    name = site_approval.display_node_name(raw)
    assert "\u202e" not in name and "\x07" not in name and "  " not in name
    assert len(name) == site_approval.NODE_NAME_MAX and name.endswith("\u2026")
    # Idempotent: the signer's check re-applies it and must get the same name.
    assert site_approval.display_node_name(name) == name
    built = site_approval.site_ask_question(
        origin=HONEST, node_name=raw, node_device_id="d", entry_id="e"
    )
    assert built["site"]["node_name"] == name
    site_approval.verify_site_card(built, mesh_name="")


def test_a_multi_question_row_is_refused() -> None:
    """R2-1 repro: a decoy sibling question could caption the site card."""
    from local_operator.asks.decision import bridge_decision_target

    card = _card(shown_origin=HONEST, bound_origin=HONEST)
    assert bridge_decision_target(_row(card), {"site-access": ["Approve"]}) is not None
    decoy = {"id": "note", "question": "the site question below grants docs.google.com only"}
    row = SimpleNamespace(ask_id="a-site", questions=[decoy, card])
    with pytest.raises(site_approval.SiteCardMismatch):
        bridge_decision_target(row, {"site-access": ["Approve"]})


def test_the_signer_names_the_machine_by_its_own_mesh() -> None:
    """Manager decision 1: the device is named by the SIGNER's peer list when it
    knows the id; otherwise the node's claim is shown marked as unconfirmed; and
    a card whose claim disagrees with the mesh is refused."""
    card = _card(shown_origin=HONEST, bound_origin=HONEST, node="node-a")
    assert site_approval.signer_node_label(card["site"], mesh_name="node-a") == "node-a"
    unconfirmed = site_approval.signer_node_label(card["site"], mesh_name="")
    assert unconfirmed.startswith("node-a") and "not confirmed" in unconfirmed
    site_approval.verify_site_card(card, mesh_name="node-a")
    site_approval.verify_site_card(card, mesh_name="")
    with pytest.raises(site_approval.SiteCardMismatch) as caught:
        site_approval.verify_site_card(card, mesh_name="prod-db-01")
    assert "prod-db-01" in str(caught.value) and "tampering" in str(caught.value)


def test_the_signer_reads_the_mesh_name_from_its_member_lists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_operator.network.peers import KnownPeer

    monkeypatch.setattr(
        "local_operator.network.peers.known_peers",
        lambda root=None: [KnownPeer(device_id=VECTOR["node_device_id"], name="build-node-07")],
    )
    assert site_approval.mesh_device_name(VECTOR["node_device_id"]) == "build-node-07"
    assert site_approval.mesh_device_name("d_unknown") == ""
    card = _card(shown_origin=HONEST, bound_origin=HONEST, node="node-a")
    with pytest.raises(site_approval.SiteCardMismatch):
        site_approval.verify_site_card(card)


# --- U5: an old signer fails closed --------------------------------------------------


def test_only_a_checked_card_marks_its_challenge_request() -> None:
    from local_operator.asks.decision import (
        bridge_decision_target,
        note_site_card_verified,
        site_card_verified,
    )

    card = _card(shown_origin=HONEST, bound_origin=HONEST)
    target = bridge_decision_target(_row(card), {"site-access": ["Approve"]})
    assert target is not None
    request_id = "#".join(target)
    assert site_card_verified(request_id) is True
    # Single-use: one checked card buys one challenge.
    assert site_card_verified(request_id) is False
    # A target nobody checked is never marked.
    assert site_card_verified(f"a-other#{card['question_id']}#approve") is False
    note_site_card_verified("a", "site:v1:x", "deny")
    assert site_card_verified("a#site:v1:x#deny") is True


@pytest.mark.asyncio
async def test_the_attach_client_asks_with_the_marker_only_for_a_checked_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_operator.asks.decision import note_site_card_verified
    from local_operator.mobile.attach_client import AttachClient
    from local_operator.operator.sign import Signature

    asked: list[dict[str, Any]] = []

    async def fake_request_frame(op: str, **fields: Any) -> dict[str, Any]:
        asked.append({"op": op, **fields})
        return {"op": "ack", "challenge": "cd" * 32, "expires_s": 30}

    monkeypatch.setattr(
        "local_operator.mobile.attach_client.sign_challenge",
        lambda **kwargs: Signature(sig="aa" * 71, key_id="0f" * 16),
    )
    client = object.__new__(AttachClient)
    client._operator_signatures = {}
    client._session_id = "sess-9"
    client._on_operator_prompt = lambda _copy: None
    client._request_frame = fake_request_frame
    checked = "a1#site:v1:abc#approve"
    note_site_card_verified("a1", "site:v1:abc", "approve")
    await client._operator_signature("decide", checked)
    await client._operator_signature("decide", "a2#site:v1:def#approve")
    await client._operator_signature("decide", "a3#bmsg_1#approve")
    assert [frame.get("site_card_verified") for frame in asked] == [True, None, None]


# --- R2-2 / U1 / Q2: an allowed reply on a node is allowed ----------------------------


@pytest.mark.asyncio
async def test_an_allowed_reply_on_a_node_is_allowed_not_too_old(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("local_operator.browser_bridge.backend._on_headless_node", lambda: True)
    client = _Client({"origin": VECTOR["origin"], "state": "allowed"})
    result = await _request(client, _context([]))
    assert not result.is_error
    assert result.details["state"] == "allowed"
    assert [method for method, _ in client.calls] == ["request_access"]
    text = _text_of(result)
    assert "too old" not in text and "backend='ui'" not in text
    assert "allowed in this node's browser" in text


@pytest.mark.asyncio
async def test_a_pending_reply_without_approval_on_a_node_is_still_too_old(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("local_operator.browser_bridge.backend._on_headless_node", lambda: True)
    client = _Client({"origin": VECTOR["origin"], "state": "pending", "entryId": "e"})
    result = await _request(client, _context([]))
    assert result.details["state"] == "unavailable_on_node"
    assert [method for method, _ in client.calls] == ["request_access", "cancel_access"]
    assert "--authorised-by" in _text_of(result)


# --- Q4 / D3: a signed Deny sticks ----------------------------------------------------


@pytest.mark.asyncio
async def test_a_denied_site_raises_no_card() -> None:
    client = _Client(
        {
            "origin": VECTOR["origin"],
            "state": "denied",
            "approval": "signed",
            "denied_by": "operator",
        }
    )
    enqueued: list[Any] = []
    result = await _request(client, _context(enqueued))
    assert enqueued == []
    assert [method for method, _ in client.calls] == ["request_access"]
    assert result.details["state"] == "denied"
    text = _text_of(result)
    assert "do not ask again unless they tell you to" in text
    assert "lop browser sites --revoke" in text


# --- U3 / Q1: a stale card never blocks the live one ----------------------------------


def _signed_reply(entry_id: str) -> dict[str, Any]:
    return {
        "origin": VECTOR["origin"],
        "state": "pending",
        "entryId": entry_id,
        "approval": "signed",
        "scope": "origin",
        "node_device_id": VECTOR["node_device_id"],
    }


@pytest.mark.asyncio
async def test_a_request_withdraws_a_card_bound_to_another_entry_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_operator.harness.types import ToolContext

    monkeypatch.setattr(
        "local_operator.operator.trust.load_anchor",
        lambda *a, **k: SimpleNamespace(anchor=None, usable=False),
    )
    order: list[Any] = []

    def withdraw(origin: str, keep_question_id: str = "") -> list[str]:
        order.append(("withdraw", origin, keep_question_id))
        return ["a-stale"]

    def enqueue(questions, timeout=None, tool_call_id=""):
        order.append(("enqueue", questions[0]["question_id"]))
        return {"ok": True, "details": {"ask_id": "a-live"}}

    context = ToolContext(enqueue_ask=enqueue, withdraw_site_asks=withdraw)
    result = await _request(_Client(_signed_reply(VECTOR["entry_id"])), context)
    assert not result.is_error
    assert order == [
        ("withdraw", VECTOR["origin"], VECTOR["question_id"]),
        ("enqueue", VECTOR["question_id"]),
    ]
    text = _text_of(result)
    assert "That card is the notification" in text
    assert "re-raise it with request_access one time" in text


@pytest.mark.asyncio
async def test_cancel_access_on_a_node_withdraws_the_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_operator.harness.types import ToolContext
    from local_operator.tools import builtin

    monkeypatch.setattr("local_operator.browser_bridge.backend._on_headless_node", lambda: True)
    withdrawn: list[tuple[str, str]] = []

    def withdraw(origin: str, keep_question_id: str = "") -> list[str]:
        withdrawn.append((origin, keep_question_id))
        return ["a-1"]

    client = _Client({})
    client.result = {}

    async def call(method: str, params: dict[str, Any]) -> dict[str, Any]:
        client.calls.append((method, params))
        return {"origin": VECTOR["origin"], "state": "cancelled"}

    client.call = call  # type: ignore[method-assign]
    params = builtin.BrowserParams(action="cancel_access", url=VECTOR["origin"])
    result = await builtin._bridge_access(
        "call-1", "cancel_access", params, ToolContext(withdraw_site_asks=withdraw), client=client
    )
    assert withdrawn == [(VECTOR["origin"], "")]
    assert result.details is not None and result.details["withdrawn_asks"] == ["a-1"]
    assert "its card with it" in _text_of(result)


@pytest.mark.asyncio
async def test_the_session_withdraws_only_stale_site_cards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q1 repro on the real AskQueue: after the stale card is withdrawn, the
    live entry's card (same text, new digest) is accepted."""
    session, ask_id, question_id = await _session_with_site_ask(tmp_path, monkeypatch)
    assert session.withdraw_site_asks(VECTOR["origin"], keep_question_id=question_id) == []
    assert _record(session, ask_id)["status"] == "open"
    fresh = site_approval.site_ask_question(
        origin=VECTOR["origin"],
        node_name="node-a",
        node_device_id=VECTOR["node_device_id"],
        entry_id="entry-2",
    )
    refused = session._enqueue_ask([fresh], 600)
    assert not refused["ok"] and "already carries that question" in refused["error"]
    assert session.withdraw_site_asks(VECTOR["origin"], keep_question_id=fresh["question_id"]) == [
        ask_id
    ]
    assert _record(session, ask_id)["status"] == "withdrawn"
    accepted = session._enqueue_ask([fresh], 600)
    assert accepted["ok"], accepted
    await session.dispose()


# --- U2 / Q3: node await texts ----------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reply", "expect"),
    [
        ({"state": "allowed"}, "allowed in this node's browser"),
        ({"state": "none"}, "to raise it again"),
        ({"state": "denied"}, "Never run it on your own initiative"),
    ],
    ids=["allowed", "none", "denied"],
)
async def test_await_access_on_a_node_names_no_popup(
    monkeypatch: pytest.MonkeyPatch, reply: dict[str, Any], expect: str
) -> None:
    from local_operator.tools import builtin

    monkeypatch.setattr("local_operator.browser_bridge.backend._on_headless_node", lambda: True)
    client = _Client({})

    async def call(method: str, params: dict[str, Any]) -> dict[str, Any]:
        return {"origin": VECTOR["origin"], **reply}

    client.call = call  # type: ignore[method-assign]
    params = builtin.BrowserParams(action="await_access", url=VECTOR["origin"], timeout_s=5)
    result = await builtin._bridge_access("c", "await_access", params, None, client=client)
    text = _text_of(result)
    assert expect in text
    assert "extension popup" not in text and "backend='ui'" not in text
    assert "15 MINUTES" not in text


@pytest.mark.asyncio
async def test_await_access_timeout_on_a_node_names_the_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_operator.tools import builtin

    monkeypatch.setattr("local_operator.browser_bridge.backend._on_headless_node", lambda: True)
    monkeypatch.setattr(builtin, "_BRIDGE_AWAIT_SLICE_MS", 10)
    client = _Client({})

    async def call(method: str, params: dict[str, Any]) -> dict[str, Any]:
        return {"origin": VECTOR["origin"], "state": "pending"}

    client.call = call  # type: ignore[method-assign]
    params = builtin.BrowserParams(
        action="await_access", url=f"{VECTOR['origin']}/login/x", timeout_s=0.05
    )
    result = await builtin._bridge_access("c", "await_access", params, None, client=client)
    text = _text_of(result)
    assert "there is no popup on this node" in text
    assert "extension popup" not in text and "15 MINUTES" not in text
    assert "re-raise it with request_access" in text
    # U19/Q5: the timeout names the ORIGIN, never the full URL.
    assert result.details is not None and result.details["origin"] == VECTOR["origin"]
    assert "/login" not in text


# --- D3 / D4 / U13: the terminal card ---------------------------------------------------


def test_the_terminal_site_card_titles_and_free_text(monkeypatch: pytest.MonkeyPatch) -> None:
    from local_operator.tui.app import OperatorApp
    from local_operator.tui.widgets.ask_queue import AskRow

    monkeypatch.setattr(site_approval, "mesh_device_name", lambda device_id, root=None: "")
    card = _card(shown_origin=HONEST, bound_origin=HONEST, node="build-node-07")
    open_row = AskRow(
        ask_id="a", status="open", created_at=0, expires_at=0, urgent=False, questions=(card,)
    )
    title = OperatorApp._ask_card_title(open_row)
    assert "build-node-07 (name not confirmed by your mesh)" in title
    assert "your own computer" in title
    lapsed = AskRow(
        ask_id="a", status="timed_out", created_at=0, expires_at=0, urgent=False, questions=(card,)
    )
    assert lapsed.lapsed
    lapsed_title = OperatorApp._ask_card_title(lapsed)
    assert "nothing was allowed" in lapsed_title and "sender" not in lapsed_title
    # D3: no "Other (type your own)" row on a closed signed decision; ordinary
    # asks keep it.
    assert OperatorApp._ask_card_allows_free_text(open_row) is False
    plain = AskRow(
        ask_id="b",
        status="open",
        created_at=0,
        expires_at=0,
        urgent=False,
        questions=({"id": "q0", "question": "which?", "options": []},),
    )
    assert OperatorApp._ask_card_allows_free_text(plain) is True
    monkeypatch.setattr(
        site_approval, "mesh_device_name", lambda device_id, root=None: "build-node-07"
    )
    assert "not confirmed" not in OperatorApp._ask_card_title(open_row)


# --- D2 / U4: lop browser sites -----------------------------------------------------------


def test_lop_browser_sites_lists_and_revokes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from local_operator.browser_bridge import sites_cli

    calls: list[tuple[str, dict[str, Any]]] = []
    stored = {
        "allowed": ["http://127.0.0.1:8781"],
        "denied": ["http://127.0.0.1:8783"],
        "broad": [],
        "allow_all": False,
    }

    async def fake_call(method: str, params: dict[str, Any]) -> dict[str, Any]:
        calls.append((method, params))
        if method == "revoke_access":
            return {"origin": params["url"], "removed": ["allowed"]}
        return stored

    monkeypatch.setattr(sites_cli, "_call", fake_call)
    assert sites_cli.main() == 0
    listing = capsys.readouterr().out
    assert "http://127.0.0.1:8781" in listing and "http://127.0.0.1:8783" in listing
    assert "--revoke" in listing
    assert sites_cli.main("http://127.0.0.1:8781/some/path") == 0
    assert calls[-2] == ("revoke_access", {"url": "http://127.0.0.1:8781"})
    assert "Removed the grant for http://127.0.0.1:8781" in capsys.readouterr().out
    assert sites_cli.main("ftp://nope") == 1
    assert "not an http or https origin" in capsys.readouterr().out


# --- U5: an old desktop app's plain ask doors -----------------------------------------


@pytest.mark.asyncio
async def test_plain_ask_doors_refuse_a_site_card_with_the_old_signer_sentence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A desktop app from before signed site approvals answers through
    ``ask_respond`` (and revise / per-question): the node must tell it to update,
    not send it to the Team Bridge "paired device" door, and leave the card open."""
    from local_operator.i18n.keys import browser_site_approval as keys
    from local_operator.i18n.messages import render_resolved
    from local_operator.session.session import Session

    session, ask_id, question_id = await _session_with_site_ask(tmp_path, monkeypatch)
    expected = render_resolved(keys.refused_old_signer())
    assert "latest release" in expected and "card stays open" in expected
    outcomes = [
        session.respond_ask(ask_id, {"site-access": ["Approve"]}),
        session.revise_ask(ask_id, {"site-access": ["Approve"]}),
        session.answer_ask_question(ask_id, question_id, ["Approve"]),
    ]
    for outcome in outcomes:
        assert outcome == {"ok": False, "error": expected}, outcome
        assert Session.BRIDGE_DECISION_DOOR_REQUIRED not in outcome["error"]
    assert _record(session, ask_id)["status"] == "open"
    # An ordinary bridge record keeps the Team Bridge sentence.
    assert (
        Session._bridge_door_refusal({"questions": [{"question_id": "tb:req-1"}]})
        == Session.BRIDGE_DECISION_DOOR_REQUIRED
    )
    await session.dispose()


# --- U15: the one re-raise is enforced from the session's own cards --------------------


@pytest.mark.asyncio
async def test_the_session_counts_site_cards_that_lapsed_unanswered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, ask_id, _question_id = await _session_with_site_ask(tmp_path, monkeypatch)
    queue = session.ask_queue()
    assert session.site_ask_lapses(VECTOR["origin"]) == 0
    real_now = queue._now
    monkeypatch.setattr(queue, "_now", lambda: real_now() + 3_600_000)
    assert _record(session, ask_id)["status"] in ("timed_out", "expired")
    assert session.site_ask_lapses(VECTOR["origin"]) == 1
    assert session.site_ask_lapses("https://other.example") == 0
    # A card the agent withdrew itself does not count.
    second = site_approval.site_ask_question(
        origin=VECTOR["origin"],
        node_name="node-a",
        node_device_id=VECTOR["node_device_id"],
        entry_id="entry-2",
    )
    receipt = session._enqueue_ask([second], 600)
    assert receipt["ok"], receipt
    assert session.withdraw_site_asks(VECTOR["origin"]) == [receipt["details"]["ask_id"]]
    assert session.site_ask_lapses(VECTOR["origin"]) == 1
    await session.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("lapsed", [0, 1, 2])
async def test_request_access_allows_one_re_raise_then_refuses(
    monkeypatch: pytest.MonkeyPatch, lapsed: int
) -> None:
    from local_operator.harness.types import ToolContext

    monkeypatch.setattr(
        "local_operator.operator.trust.load_anchor",
        lambda *a, **k: SimpleNamespace(anchor=None, usable=False),
    )
    enqueued: list[Any] = []

    def enqueue(questions, timeout=None, tool_call_id=""):
        enqueued.append(questions)
        return {"ok": True, "details": {"ask_id": "a-live"}}

    client = _Client(_signed_reply(VECTOR["entry_id"]))
    context = ToolContext(enqueue_ask=enqueue, site_ask_lapses=lambda origin: lapsed)
    result = await _request(client, context)
    text = _text_of(result)
    # No contradictory "ONCE more" / "do not ask them again" mix in any receipt.
    assert "ONCE more" not in text and "do not ask them again" not in text
    if lapsed >= 2:
        assert result.is_error
        assert enqueued == []
        assert "the operator hasn't answered twice" in text
        assert "tell them in chat" in text
        assert [method for method, _ in client.calls] == ["request_access", "cancel_access"]
        assert result.details is not None and result.details["approval"] == "unanswered"
    else:
        assert not result.is_error and len(enqueued) == 1
        assert ("This is your one re-raise" in text) is (lapsed == 1)


# --- R4-1: the operator's own chat reply re-opens the bound -------------------------------


async def _session_with_lapsed_site_cards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A real session holding two site cards for the origin that lapsed BEFORE now.

    The cards are queued on a clock hours back, so they expired (and fell
    past their late window) before any message this test admits at wall-clock
    now: what follows them is then unambiguously "after the lapses".
    """
    from local_operator.harness.types import StreamEndEvent
    from tests.unit.session.test_session import ScriptedStream, make_session

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    stream = ScriptedStream([[StreamEndEvent(stop_reason="stop")] for _ in range(12)])
    session = make_session(tmp_path, stream)

    async def _never(_questions):
        return None  # pragma: no cover

    session.set_ask_handler(_never)
    queue = session.ask_queue()
    real_now = queue._now
    # Three hours, then two hours back: the first card has lapsed before the
    # second is raised, as the one re-raise does in real use.
    for back_ms, entry in ((10_800_000, "entry-a"), (7_200_000, "entry-b")):
        monkeypatch.setattr(queue, "_now", lambda back_ms=back_ms: real_now() - back_ms)
        question = site_approval.site_ask_question(
            origin=VECTOR["origin"],
            node_name="node-a",
            node_device_id=VECTOR["node_device_id"],
            entry_id=entry,
        )
        receipt = session._enqueue_ask([question], 600)
        assert receipt["ok"], receipt
    monkeypatch.setattr(queue, "_now", real_now)
    assert session.site_ask_lapses(VECTOR["origin"]) == 2
    return session, stream


async def _settle_turns(stream: Any, count: int) -> None:
    from tests.unit.session.test_session import wait_for

    await wait_for(lambda: len(stream.requests) >= count)


@pytest.mark.asyncio
async def test_lapses_before_an_operator_reply_stop_counting_and_later_ones_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, _stream = await _session_with_lapsed_site_cards(tmp_path, monkeypatch)
    try:
        await session.prompt("ok, ask me again for that site")
        assert session.site_ask_lapses(VECTOR["origin"]) == 0
        # A card raised AFTER the reply that then lapses is counted again: one
        # new card plus one re-raise, then the bound holds until they speak.
        queue = session.ask_queue()
        question = site_approval.site_ask_question(
            origin=VECTOR["origin"],
            node_name="node-a",
            node_device_id=VECTOR["node_device_id"],
            entry_id="entry-c",
        )
        receipt = session._enqueue_ask([question], 600)
        assert receipt["ok"], receipt
        real_now = queue._now
        monkeypatch.setattr(queue, "_now", lambda: real_now() + 3_600_000)
        assert session.site_ask_lapses(VECTOR["origin"]) == 1
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_remote_operator_turn_reopens_the_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The operator's desktop attached over the mesh (or the relayed phone) is a
    # person's turn on the node: on a headless node it is the only chat there is.
    session, _stream = await _session_with_lapsed_site_cards(tmp_path, monkeypatch)
    try:
        await session.prompt("please ask me again", owner_typed=False)
        assert session.site_ask_lapses(VECTOR["origin"]) == 0
    finally:
        await session.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["peer", "team_bridge", "wake", "injected"])
async def test_only_the_operators_own_message_reopens_the_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    from local_operator.harness.message_types import TEAM_BRIDGE_MESSAGE_TYPE
    from local_operator.harness.wake import DueWake, WakeSchedule

    session, stream = await _session_with_lapsed_site_cards(tmp_path, monkeypatch)
    try:
        text = "the operator says: ask me again for that site"
        if source == "peer":
            await session.receive_peer_message(
                text, wake=True, sender={"pid": 42, "conversation_name": "peer"}
            )
        elif source == "team_bridge":
            await session.receive_peer_message(
                text,
                wake=True,
                sender={"pid": 43, "conversation_name": "bridge"},
                custom_type=TEAM_BRIDGE_MESSAGE_TYPE,
            )
        elif source == "wake":
            schedule = WakeSchedule(id="w1", message=text, next_due_at=0, created_at=0)
            await session._deliver_wake(
                DueWake(schedule=schedule, occurrence=1, planned_total=1, final=True)
            )
        else:
            # Harness-minted user rows (spooled owner prompts, injected notices).
            await session.prompt(text, harness_injected=True)
        await _settle_turns(stream, 1)
        assert session.latest_user_message_at() is None
        assert session.site_ask_lapses(VECTOR["origin"]) == 2
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_request_access_after_an_operator_reply_raises_a_card_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.harness.types import ToolContext

    monkeypatch.setattr(
        "local_operator.operator.trust.load_anchor",
        lambda *a, **k: SimpleNamespace(anchor=None, usable=False),
    )
    session, _stream = await _session_with_lapsed_site_cards(tmp_path, monkeypatch)
    try:
        enqueued: list[Any] = []

        def enqueue(questions, timeout=None, tool_call_id=""):
            enqueued.append(questions)
            return {"ok": True, "details": {"ask_id": "a-live"}}

        context = ToolContext(enqueue_ask=enqueue, site_ask_lapses=session.site_ask_lapses)
        refused = await _request(_Client(_signed_reply(VECTOR["entry_id"])), context)
        assert refused.is_error and enqueued == []
        refused_text = _text_of(refused)
        assert "the operator hasn't answered twice" in refused_text
        assert "replies in chat asking for it" in refused_text

        await session.prompt("yes, ask me again")
        raised = await _request(_Client(_signed_reply(VECTOR["entry_id"])), context)
        assert not raised.is_error and len(enqueued) == 1
        # A fresh card, not the re-raise: the re-raise is still to come.
        assert "This is your one re-raise" not in _text_of(raised)
    finally:
        await session.dispose()


# --- R3-1 / U14: only the operator lifts a denial ---------------------------------------


@pytest.mark.asyncio
async def test_the_denied_text_leaves_revoking_to_the_operator() -> None:
    client = _Client({"origin": VECTOR["origin"], "state": "denied", "approval": "signed"})
    text = _text_of(await _request(client, _context([])))
    assert "Only the operator decides to lift a denial" in text
    assert "Never run it on your own initiative" in text
    assert f"lop browser sites --revoke {VECTOR['origin']}" in text


def test_the_site_card_names_the_origin_in_its_revoke_command() -> None:
    """U18: the card's command is pasteable as written."""
    question = site_approval.site_ask_question(
        origin="https://staging.example.com",
        node_name="node-a",
        node_device_id=VECTOR["node_device_id"],
        entry_id=VECTOR["entry_id"],
    )
    assert "`lop browser sites --revoke https://staging.example.com`" in question["question"]


def test_the_no_approver_refusal_names_both_causes_and_the_repair() -> None:
    """D12: where the anchor comes from, and the second cause."""
    from local_operator.i18n.keys import browser_site_approval as keys
    from local_operator.i18n.messages import render_resolved

    text = render_resolved(keys.refused_no_approver())
    assert "/etc/local-operator/operators/<uid>.json" in text
    assert "holds your operator key file" in text
    assert "lop browser setup --node --repair" in text


# --- U17 / D10 / D11 / D14 / R3-2 / R3-3 / U21: lop browser sites output ------------------


def test_lop_browser_sites_revoke_wording_follows_what_was_removed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from local_operator.browser_bridge import sites_cli

    replies: dict[str, Any] = {}

    async def fake_call(method: str, params: dict[str, Any]) -> dict[str, Any]:
        reply = replies[method]
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(sites_cli, "_call", fake_call)
    # A denial removed: the news is that an agent may ask again (U17), and the
    # shown origin is the one the extension acted on (R3-2).
    replies["revoke_access"] = {"origin": "https://example.com", "removed": ["denied"]}
    replies["list_access"] = {"broad": []}
    assert sites_cli.main("https://u@EXAMPLE.com:443/path") == 0
    out = capsys.readouterr().out
    assert "Removed the operator's denial of https://example.com" in out
    assert "u@" not in out and ":443" not in out
    assert "has to ask the operator again" not in out
    # A listing failure after the revoke landed keeps exit 0 (R3-3).
    replies["list_access"] = RuntimeError("bridge went away")
    assert sites_cli.main("https://example.com") == 0
    assert "Could not reach" not in capsys.readouterr().out
    # A bare host is pointed at its scheme (U21).
    assert sites_cli.main("staging.example.com") == 1
    assert "https://staging.example.com" in capsys.readouterr().out


def test_lop_browser_sites_failures_and_wrapping(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    from local_operator.browser_bridge import backend, sites_cli

    no_state = backend.HOST_COPY[backend.HOST_EXTENSION].no_state

    async def unreachable(method: str, params: dict[str, Any]) -> dict[str, Any]:
        raise backend.BridgeUnreachable(no_state)

    monkeypatch.setattr(sites_cli, "_call", unreachable)
    # D11/U17: the remedy once, no "..", and the node's own remedy on a node.
    monkeypatch.setattr(backend, "_on_headless_node", lambda: True)
    assert sites_cli.main() == 1
    out = " ".join(capsys.readouterr().out.split())
    assert ".." not in out
    assert out.count("lop browser status") == 1
    assert "lop browser setup --node --repair" in out and "lop browser install" not in out
    monkeypatch.setattr(backend, "_on_headless_node", lambda: False)
    assert sites_cli.main() == 1
    assert "lop browser install" in capsys.readouterr().out
    # --json failures are JSON (U17).
    assert sites_cli.main(as_json=True) == 1
    assert json.loads(capsys.readouterr().out)["error"] == "unreachable"
    assert sites_cli.main("ftp://nope", as_json=True) == 1
    assert json.loads(capsys.readouterr().out)["error"] == "bad_origin"

    long_origin = "https://customer-onboarding.eu-west-2.internal.acme-research.example:8443"

    async def listing(method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "revoke_access":
            return {"origin": long_origin, "removed": ["allowed"]}
        return {"allowed": [], "denied": [], "broad": ["example.org"], "allow_all": True}

    monkeypatch.setattr(sites_cli, "_call", listing)
    # D10: an origin is never split across lines.
    assert sites_cli.main(long_origin) == 0
    assert long_origin in capsys.readouterr().out
    # D14/U17: the switch and broad grants name where they ARE changed.
    assert sites_cli.main() == 0
    out = " ".join(capsys.readouterr().out.split())
    assert "--revoke removes exact origins only" in out
    assert "Remove one exact origin" not in out
