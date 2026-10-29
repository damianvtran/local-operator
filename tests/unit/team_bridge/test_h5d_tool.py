"""H5d — the bridge tool: hash-verified take-up, local claim/release, sends
through the outbox, and NO decide authority."""

from __future__ import annotations

import json
from typing import Any, cast

import httpx
import pytest

from local_operator.harness.types import TextContent, ToolContext
from local_operator.team_bridge import client as tb_client
from local_operator.team_bridge import host as host_mod
from local_operator.team_bridge.client import TeamBridgeClient
from local_operator.team_bridge.store import OwnerReceipt, OwnerSnapshot, Store
from local_operator.team_bridge.tool import (
    BridgeRuntime,
    body_digest,
    build_bridge_tool,
    execute_bridge,
)
from tests.unit.team_bridge.test_client import FakeStore, _ok

TENANT = "ten_1"


@pytest.fixture(autouse=True)
def _destination_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tb_client, "org_oauth_destination_allowed", lambda base: True)


@pytest.fixture(autouse=True)
def _gate_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """Open BOTH entry gates for the op-level subjects below.

    The tool now calls the LOCAL authority fence FIRST and the admission gate
    second (S-1 remediation), and these tests exercise hash verification,
    claims and sends — not the gates themselves (the fence and the admission
    states, including reads-vs-mutations, are covered by
    ``test_h5_remediation_fence.py`` / ``_coalesce`` / ``_doors``). Both are
    patched at the module attributes the tool imports at CALL time, so the
    pending default cannot shadow the subject here.
    """

    monkeypatch.setattr(host_mod, "owner_fence_refusal", lambda runtime: None)
    monkeypatch.setattr(host_mod, "admission_refusal", lambda runtime, *, is_read=False: None)


def _owner() -> OwnerSnapshot:
    return OwnerSnapshot(
        credential_id=1,
        account_id="acc_1",
        tenant_id=TENANT,
        endpoint_id="bep_1",
        canonical_base_url="https://hub.example",
        enrollment_generation=1,
    )


def _client(handler) -> TeamBridgeClient:
    return TeamBridgeClient(
        None,
        base_url="https://hub.example/v1",
        store=FakeStore(),
        transport=httpx.MockTransport(handler),
    )


def test_the_builder_never_advertises_without_a_runtime() -> None:
    assert build_bridge_tool(ToolContext()) is None
    # The builder reads no client; ``BridgeRuntime`` is typed against the real
    # one, so the stand-in is cast rather than the annotation weakened (see
    # test_h5_remediation_doors).
    runtime = BridgeRuntime(
        client=cast(TeamBridgeClient, object()),
        config_dir=".",
        tenant_id=TENANT,
        owner=_owner(),
        session_id="s1",
    )
    tool = build_bridge_tool(ToolContext(bridge_runtime=runtime))
    assert tool is not None and tool.name == "bridge"
    op_schema = tool.parameters["properties"]["op"]
    assert "decide" not in json.dumps(op_schema)


@pytest.mark.asyncio
async def test_take_up_verifies_the_body_digest_before_handing_it_over(tmp_path) -> None:
    text = "the full screened body, longer than any excerpt"
    runtime = BridgeRuntime(
        client=_client(lambda request: _ok({"message": {"body": {"text": text}}})),
        config_dir=tmp_path,
        tenant_id=TENANT,
        owner=_owner(),
        session_id="s1",
    )
    good = await execute_bridge(
        runtime, {"op": "take_up", "message_id": "bmsg_1", "body_sha256": body_digest(text)}
    )
    assert good["ok"] is True and good["text"] == text

    bad = await execute_bridge(
        runtime, {"op": "take_up", "message_id": "bmsg_1", "body_sha256": "0" * 64}
    )
    assert bad["ok"] is False and bad["code"] == "body_hash_mismatch"


@pytest.mark.asyncio
async def test_the_tool_refuses_the_decide_word_outright(tmp_path) -> None:
    runtime = BridgeRuntime(
        client=_client(lambda request: _ok({})),
        config_dir=tmp_path,
        tenant_id=TENANT,
        owner=_owner(),
        session_id="s1",
    )
    outcome = await execute_bridge(runtime, {"op": "decide", "message_id": "bmsg_1"})
    assert outcome["ok"] is False and outcome["code"] == "not_permitted"
    assert "human" in outcome["error"]


@pytest.mark.asyncio
async def test_ack_working_claims_and_done_releases_with_the_same_token(tmp_path) -> None:
    owner = _owner()
    store = Store(tmp_path)
    store.commit_enrollment(owner, OwnerReceipt(consent_version="cv-1", consent_ref="cr-1"))
    runtime = BridgeRuntime(
        client=_client(lambda request: _ok({})),
        config_dir=tmp_path,
        tenant_id=TENANT,
        owner=owner,
        session_id="s1",
    )
    working = await execute_bridge(
        runtime, {"op": "ack", "message_id": "bmsg_1", "state": "working"}
    )
    assert working["ok"] is True and "bmsg_1" in runtime.claims

    # A second session (different runtime) cannot grab the same message.
    other = BridgeRuntime(
        client=runtime.client, config_dir=tmp_path, tenant_id=TENANT, owner=owner, session_id="s2"
    )
    contended = await execute_bridge(
        other, {"op": "ack", "message_id": "bmsg_1", "state": "working"}
    )
    assert contended["ok"] is False and contended["code"] == "claimed"

    done = await execute_bridge(runtime, {"op": "ack", "message_id": "bmsg_1", "state": "done"})
    assert done["ok"] is True and "bmsg_1" not in runtime.claims
    # 'done' without a live claim by THIS session is refused, not stolen.
    again = await execute_bridge(runtime, {"op": "ack", "message_id": "bmsg_1", "state": "done"})
    assert again["ok"] is False and again["code"] == "not_holder"


@pytest.mark.parametrize(
    "target",
    [
        {"account_id": "acc_2", "kind": "agent"},
        {"endpoint_id": "bep_9", "kind": "agent"},
        {"account_id": "acc_2", "endpoint_id": "bep_9"},
        # A LABEL key is not a recipient: the shape is one of the three keys
        # (endpoint_id, account_id, name) and nothing else, so a caller cannot
        # smuggle a second opinion about who the destination is.
        {"name": "Ada Lovelace", "account_id": "acc_2"},
        {"display_label": "Ada Lovelace"},
        {},
    ],
)
def test_tool_recipient_schema_is_closed(target):
    from pydantic import ValidationError

    from local_operator.team_bridge.tool import BridgeToolParams

    with pytest.raises(ValidationError):
        BridgeToolParams.model_validate({"op": "send", "to": target, "text": "synthetic"})


@pytest.mark.parametrize(
    ("target", "valid"),
    [
        ({"endpoint_id": "bep_9"}, True),
        ({"account_id": "acc_2"}, True),
        ({"account_id": "acc_2", "kind": "agent"}, False),
        ({"endpoint_id": "bep_9", "kind": "agent"}, False),
        ({"account_id": "acc_2", "endpoint_id": "bep_9"}, False),
        ({"account_id": ""}, False),
        ({"account_id": 42}, False),
        # A display name is a legal SHAPE for the one op that can resolve it
        # (channel_post); whether an op may use it is the executor's rule, tested
        # in ``test_recipient_name_resolution.py``. Two keys are still refused.
        ({"name": "Ada Lovelace"}, True),
        ({"name": "Ada Lovelace", "account_id": "acc_2"}, False),
        # The bare-string form a model reaches for: accepted, and only as an id.
        # ``" acc_2"`` is NOT listed here on purpose: padding is refused by the
        # shared validator/executor (asserted below and in the client test), not
        # by a JSON Schema that cannot express "unpadded".
        ("acc_2", True),
        ("", False),
        (42, False),
        ({}, False),
    ],
)
def test_compact_recipient_wire_schema_preserves_closed_one_of(target, valid):
    """Schema compression must not just rely on the executor catching bad input."""
    from jsonschema import Draft202012Validator

    from local_operator.providers.clients import (
        _tools_to_openai,
        _tools_to_openai_responses,
    )
    from local_operator.team_bridge.tool import BridgeToolParams

    runtime = BridgeRuntime(
        client=cast(TeamBridgeClient, object()),
        config_dir=".",
        tenant_id=TENANT,
        owner=_owner(),
        session_id="synthetic-schema",
    )
    tool = build_bridge_tool(ToolContext(bridge_runtime=runtime))
    assert tool is not None
    schema = tool.parameters
    assert "propertyNames" not in json.dumps(schema)
    for published in (
        schema,
        _tools_to_openai([tool])[0]["function"]["parameters"],
        _tools_to_openai_responses([tool])[0]["parameters"],
    ):
        assert published == schema
        Draft202012Validator.check_schema(published)
        errors = list(
            Draft202012Validator(published).iter_errors(
                {"op": "send", "to": target, "text": "synthetic"}
            )
        )
        assert (not errors) is valid
    if valid:
        # The model hands on the CANONICAL pair: a bare id is the account form.
        canonical = {"account_id": target} if isinstance(target, str) else target
        assert (
            BridgeToolParams.model_validate({"op": "send", "to": target}).model_dump()["to"]
            == canonical
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("op", ["send", "reply"])
@pytest.mark.parametrize(
    "target",
    [
        {"account_id": "acc_2", "kind": "agent"},
        {"endpoint_id": "bep_9", "kind": "agent"},
        {"account_id": "acc_2", "typo": "ignored-before"},
        {"account_id": "acc_2", "endpoint_id": ""},
        {"account_id": "acc_2", "endpoint_id": "bep_9"},
        {"account_id": "   "},
        # A DISPLAY NAME is no longer a refused SHAPE for send/reply: it is a
        # RESOLUTION REQUEST that needs the directory and the owner's allowlist,
        # so it refuses after the admission gate instead of here (that path has
        # its own test below). The keys around it still refuse with no I/O.
        {"name": "Ada Lovelace", "kind": "human"},
        # The bare-string form is an id or nothing: empty, padded, and a display
        # NAME written as a bare string (ids never contain whitespace) refuse.
        "",
        "   ",
        " acc_2",
        "acc_2 ",
        "Ada Lovelace",
        {},
    ],
)
async def test_send_recipient_is_closed_before_outbox(tmp_path, monkeypatch, op, target) -> None:
    """A refused target cannot leave a retryable intent or dispatch a request."""
    from local_operator.team_bridge.outbox import Outbox

    records = []
    admissions = []
    original = Outbox.record_send

    def record(self, *args, **kwargs):
        records.append(kwargs)
        return original(self, *args, **kwargs)

    def admission(runtime, *, is_read=False):
        admissions.append(runtime)
        return None

    monkeypatch.setattr(Outbox, "record_send", record)
    monkeypatch.setattr(host_mod, "admission_refusal", admission)
    requests = []
    runtime = BridgeRuntime(
        client=_client(lambda request: (requests.append(request), _ok({}))[1]),
        config_dir=tmp_path,
        tenant_id=TENANT,
        owner=_owner(),
        session_id="synthetic-session",
    )
    result = await execute_bridge(
        runtime, {"op": op, "to": target, "text": "synthetic", "thread_id": "bth_1"}
    )
    assert result["ok"] is False and result["code"] == "invalid_request"
    assert records == []
    assert admissions == []
    assert requests == []


@pytest.mark.asyncio
async def test_a_reply_pairs_thread_and_to_and_the_send_confirms_in_the_outbox(tmp_path) -> None:
    requests: list[httpx.Request] = []
    runtime = BridgeRuntime(
        client=_client(
            lambda request: (
                requests.append(request),
                _ok({"message": {"id": "bmsg_9"}}, status=201),
            )[1]
        ),
        config_dir=tmp_path,
        tenant_id=TENANT,
        owner=_owner(),
        session_id="s1",
    )
    missing_thread = await execute_bridge(
        runtime, {"op": "reply", "to": {"account_id": "acc_2"}, "text": "hi"}
    )
    assert missing_thread["ok"] is False and requests == []

    sent = await execute_bridge(
        runtime,
        {
            "op": "reply",
            "to": {"account_id": "acc_2"},
            "thread_id": "bth_1",
            "text": "hi",
            "type": "note",
        },
    )
    assert sent["ok"] is True and sent["message_id"] == "bmsg_9"
    body = json.loads(requests[0].content)
    assert body["thread_id"] == "bth_1" and body["to"] == {"account_id": "acc_2"}
    assert body["client_ref"]

    from local_operator.team_bridge.outbox import Outbox

    assert Outbox(tmp_path).pending(runtime.key()) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("op", ["send", "reply"])
@pytest.mark.parametrize("status", [200, 201])
@pytest.mark.parametrize("advisory", ["", "synthetic advisory"])
async def test_legacy_send_receipt_exposes_only_server_acceptance(
    tmp_path,
    monkeypatch,
    op,
    status,
    advisory,
) -> None:
    """Both new acceptance and an idempotent server answer keep the old fields.

    Use the real tool adapter so metadata must survive into model-visible
    TextContent as well as details (providers need not serialize details).
    The synthetic response deliberately includes
    body/provenance fields: the additive receipt must never echo them.
    """
    requests: list[httpx.Request] = []
    text = "synthetic private outgoing body"

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _ok(
            {"message": {"id": "bmsg_9", "body": {"text": text}, "origin": "human"}},
            status=status,
        )

    runtime = BridgeRuntime(
        client=_client(handler),
        config_dir=tmp_path,
        tenant_id=TENANT,
        owner=_owner(),
        session_id="synthetic-session",
    )
    monkeypatch.setattr(runtime.client, "guidance_note", lambda: advisory)
    context = ToolContext(bridge_runtime=runtime)
    tool = build_bridge_tool(context)
    assert tool is not None
    args = {"op": op, "to": {"account_id": "acc_2"}, "text": text}
    if op == "reply":
        args["thread_id"] = "bth_1"
    result = await tool.execute("synthetic-call", args, None, None, context)
    assert result.details == {
        "ok": True,
        "message_id": "bmsg_9",
        "state": "sent",
        "destination_type": "peer_inbox",
        "receipt_stage": "server_accepted",
        "conversation_history": False,
        "execution_confirmed": False,
    }
    assert result.is_error is not True
    assert result.details is not None
    assert isinstance(result.content[0], TextContent)
    assert json.loads(result.content[0].text) == result.details
    if advisory:
        assert len(result.content) == 2
        assert isinstance(result.content[1], TextContent)
        assert result.content[1].text == f"[bridge guidance — advisory]\n{advisory}"
    else:
        assert len(result.content) == 1
    assert text not in result.text
    assert text not in json.dumps(result.details)
    assert requests[0].url.path == "/v1/messaging/messages"
    wire = json.loads(requests[0].content)
    assert wire["body"]["text"] == text
    assert wire["to"] == {"account_id": "acc_2"}
    assert wire.get("thread_id") == ("bth_1" if op == "reply" else None)
    metadata = set(result.details).difference({"ok", "message_id", "state"})
    assert not metadata.intersection(wire)


@pytest.mark.asyncio
@pytest.mark.parametrize("op", ["send", "reply"])
async def test_legacy_receipt_on_frozen_pending_intent_replay(tmp_path, monkeypatch, op) -> None:
    """A lost response is not acceptance; replaying the frozen ref can confirm it.

    The test injects the existing pending intent at record_send only on retry.
    This does NOT claim that a second tool call automatically reuses an intent:
    ordinary new tool calls still mint new refs. No outbox schema/dispatcher or
    outgoing-body retention contract is changed by the result metadata.
    """
    from local_operator.team_bridge.outbox import Outbox

    requests: list[httpx.Request] = []
    accepted: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        ref = json.loads(request.content)["client_ref"]
        accepted.setdefault(ref, "bmsg_9")
        if len(requests) == 1:
            raise httpx.ReadError("synthetic lost response", request=request)
        return _ok({"message": {"id": accepted[ref]}}, status=200)

    runtime = BridgeRuntime(
        client=_client(handler),
        config_dir=tmp_path,
        tenant_id=TENANT,
        owner=_owner(),
        session_id="synthetic-session",
    )
    args = {"op": op, "to": {"endpoint_id": "bep_9"}, "text": "synthetic outgoing"}
    if op == "reply":
        args["thread_id"] = "bth_1"
    failed = await execute_bridge(runtime, args)
    assert failed["ok"] is False and failed["code"] == "send_unconfirmed"
    metadata = {
        "destination_type",
        "receipt_stage",
        "conversation_history",
        "execution_confirmed",
    }
    assert not metadata.intersection(failed)
    outbox = Outbox(tmp_path)
    pending = outbox.pending(runtime.key())
    assert len(pending) == 1
    monkeypatch.setattr(Outbox, "record_send", lambda self, owner_key, **kwargs: pending[0])
    confirmed = await execute_bridge(runtime, args)
    assert confirmed == {
        "ok": True,
        "message_id": "bmsg_9",
        "state": "sent",
        "destination_type": "peer_inbox",
        "receipt_stage": "server_accepted",
        "conversation_history": False,
        "execution_confirmed": False,
    }
    assert len(accepted) == 1
    assert [json.loads(request.content)["client_ref"] for request in requests] == [
        pending[0].client_ref,
        pending[0].client_ref,
    ]
    assert requests[0].content == requests[1].content
    assert outbox.pending(runtime.key()) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("op", ["send", "reply"])
async def test_refused_legacy_send_has_no_success_receipt(tmp_path, op) -> None:
    """Definitive refusal remains an error and never acquires success metadata."""
    from local_operator.team_bridge.outbox import Outbox

    runtime = BridgeRuntime(
        client=_client(
            lambda request: httpx.Response(
                403,
                json={"error": {"code": "not_permitted", "message": "synthetic refusal"}},
            )
        ),
        config_dir=tmp_path,
        tenant_id=TENANT,
        owner=_owner(),
        session_id="synthetic-session",
    )
    args = {"op": op, "to": {"account_id": "acc_2"}, "text": "synthetic outgoing"}
    if op == "reply":
        args["thread_id"] = "bth_1"
    result = await execute_bridge(runtime, args)
    assert result["ok"] is False and result["code"] == "send_unconfirmed"
    metadata = {
        "destination_type",
        "receipt_stage",
        "conversation_history",
        "execution_confirmed",
    }
    assert not metadata.intersection(result)
    assert "message_id" not in result and result.get("state") != "sent"
    assert Outbox(tmp_path).pending(runtime.key()) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("op", ["send", "reply"])
@pytest.mark.parametrize(
    "receipt",
    [
        None,
        [],
        "private-response-canary",
        {},
        {"message": None},
        {"message": []},
        {"message": "private-response-canary"},
        {"message": {}},
        {"message": {"id": ""}},
        {"message": {"id": "   "}},
        {"message": {"id": 42}},
        {"message": {"id": "bcmsg_9"}},
        {"message": {"id": "bmsg_bad/path"}},
    ],
)
async def test_malformed_legacy_receipt_stays_pending_until_frozen_replay_confirms(
    tmp_path,
    monkeypatch,
    op,
    receipt,
) -> None:
    """A 2xx envelope is not a send receipt; never retire its retryable intent.

    As above, the replay injects the saved intent, not a fresh manual send.
    The server may have accepted before returning a malformed reply, so both
    the idempotency reference and exact request must survive reconciliation.
    """
    from local_operator.team_bridge.outbox import Outbox

    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return _ok(receipt, status=201)
        return _ok({"message": {"id": "bmsg_9"}}, status=200)

    runtime = BridgeRuntime(
        client=_client(handler),
        config_dir=tmp_path,
        tenant_id=TENANT,
        owner=_owner(),
        session_id="synthetic-session",
    )
    args = {"op": op, "to": {"account_id": "acc_2"}, "text": "synthetic outgoing"}
    if op == "reply":
        args["thread_id"] = "bth_1"
    context = ToolContext(bridge_runtime=runtime)
    tool = build_bridge_tool(context)
    assert tool is not None
    tool_result = await tool.execute("synthetic-invalid", args, None, None, context)
    result = tool_result.details
    assert result is not None
    assert result["ok"] is False and result["code"] == "send_unconfirmed"
    assert result["error"] == "the send did not confirm: send_receipt_invalid"
    assert tool_result.is_error is True
    assert tool_result.text == result["error"]
    assert "server_accepted" not in tool_result.text
    assert "private-response-canary" not in tool_result.text
    assert "private-response-canary" not in json.dumps(result)
    assert "message_id" not in result and result.get("state") != "sent"
    metadata = {
        "destination_type",
        "receipt_stage",
        "conversation_history",
        "execution_confirmed",
    }
    assert not metadata.intersection(result)
    outbox = Outbox(tmp_path)
    pending = outbox.pending(runtime.key())
    assert len(pending) == 1
    assert pending[0].state == "pending" and pending[0].message_id is None
    assert pending[0].error_code is None
    assert pending[0].client_ref == json.loads(requests[0].content)["client_ref"]
    monkeypatch.setattr(Outbox, "record_send", lambda self, owner_key, **kwargs: pending[0])
    replay_result = await tool.execute("synthetic-replay-call", args, None, None, context)
    confirmed = replay_result.details
    assert replay_result.is_error is False
    assert isinstance(replay_result.content[0], TextContent)
    assert json.loads(replay_result.content[0].text) == confirmed
    assert confirmed == {
        "ok": True,
        "message_id": "bmsg_9",
        "state": "sent",
        "destination_type": "peer_inbox",
        "receipt_stage": "server_accepted",
        "conversation_history": False,
        "execution_confirmed": False,
    }
    assert requests[0].content == requests[1].content
    assert outbox.pending(runtime.key()) == []


# --- recipient ergonomics: the shape a model actually writes -----------------


def test_a_bare_string_recipient_is_the_account_id_and_nothing_else() -> None:
    """``{"to": "acct_123"}` is the account form, decided in exactly one place.

    A model reaching for a recipient writes a bare id far more often than the
    nested pair, and one string is one recipient in one field: there is nothing
    to reinterpret and nothing that can be silently dropped, which is why the
    mapping lives in the shared validator both the tool and the wire client
    call — never in either serializer. The object forms (the positive controls)
    are untouched.
    """
    from local_operator.team_bridge.tool import BridgeToolParams
    from local_operator.team_bridge.types import validate_send_recipient

    assert validate_send_recipient("acct_123") == {"account_id": "acct_123"}
    # NO id GRAMMAR is imposed on the bare form: the ids this suite and the
    # measured run actually use must pass. ``is_account_id`` is the WRONG guard
    # here — its ``acc_`` grammar does not admit the ``acct-ada`` spelling, so it
    # would refuse real ids while the server owns the id shape. This client only
    # refuses what cannot be an id at all: empty, padded, internal whitespace.
    for identifier in ("acct-123", "acc_2", "9f8a7b6c-1234-4abc-9def-0123456789ab"):
        assert validate_send_recipient(identifier) == {"account_id": identifier}
    # Pydantic is the tool's first door: it must hand on the canonical pair, or
    # a later serializer would carry the raw string to the wire.
    assert BridgeToolParams.model_validate({"op": "send", "to": "acct_123"}).to == {
        "account_id": "acct_123"
    }
    assert validate_send_recipient({"account_id": "acc_2"}) == {"account_id": "acc_2"}
    assert validate_send_recipient({"endpoint_id": "bep_9"}) == {"endpoint_id": "bep_9"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("acct_123", {"account_id": "acct_123"}),
        ({"account_id": "acct_123"}, {"account_id": "acct_123"}),
        ({"endpoint_id": "bep_9"}, {"endpoint_id": "bep_9"}),
    ],
)
async def test_every_accepted_target_shape_sends_the_same_normalized_pair(
    tmp_path, monkeypatch, target, expected
) -> None:
    """Both shapes are ONE contract: the durable intent and the wire agree.

    The outbox row is what a restart retries, so the normalized pair has to be
    what it records — a raw string surviving into ``to_json`` would replay a
    body the server reads differently from the one that was validated.
    """
    from local_operator.team_bridge.outbox import Outbox

    recorded: list[dict[str, Any]] = []
    original = Outbox.record_send

    def record(self, *args, **kwargs):
        recorded.append(kwargs)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Outbox, "record_send", record)
    requests: list[httpx.Request] = []
    runtime = BridgeRuntime(
        client=_client(
            lambda request: (
                requests.append(request),
                _ok({"message": {"id": "bmsg_9"}}, status=201),
            )[1]
        ),
        config_dir=tmp_path,
        tenant_id=TENANT,
        owner=_owner(),
        session_id="synthetic-session",
    )
    result = await execute_bridge(runtime, {"op": "send", "to": target, "text": "synthetic"})
    assert result["ok"] is True and result["state"] == "sent"
    assert recorded[0]["to"] == expected
    assert json.loads(requests[0].content)["to"] == expected


@pytest.mark.parametrize(
    "target",
    [
        {"name": "Ada Lovelace"},
        {"display_label": "Ada Lovelace"},
        {"account_id": "acc_2", "endpoint_id": "bep_9"},
        # A name written as a BARE string is the same guess through the new
        # door: it reaches the teaching refusal, never the outbox.
        "Ada Lovelace",
    ],
)
def test_a_guessed_name_is_refused_with_the_accepted_shapes_and_the_directory(
    target,
) -> None:
    """The refusal is the correction: a model that guessed a name learns the move.

    This is the PAIR lane, which never resolves a name: it carries no conversation,
    so it has no least-privilege candidate set, and its scope is a separate
    decision. (``channel_post`` does resolve one locally, against that
    conversation's active members — never against the org directory; see
    ``test_recipient_name_resolution.py``.) A bare "unsupported field" taught the
    model nothing, so every refusal names BOTH accepted shapes and where the ids
    come from. It still quotes nothing caller-controlled (no value, no key).
    """
    from local_operator.team_bridge.types import (
        RECIPIENT_ACCOUNT_SHAPE,
        RECIPIENT_ENDPOINT_SHAPE,
        InvalidSendRecipient,
        validate_send_recipient,
    )

    with pytest.raises(InvalidSendRecipient) as refusal:
        validate_send_recipient(target)
    assert refusal.value.code == "invalid_request"
    message = str(refusal.value)
    assert RECIPIENT_ENDPOINT_SHAPE in message
    assert RECIPIENT_ACCOUNT_SHAPE in message
    assert "directory" in message
    for caller_text in ("Ada", "Lovelace", "name", "display_label"):
        assert caller_text not in message


def test_the_published_to_schema_names_both_shapes_and_the_directory() -> None:
    """The schema is the model's FIRST teacher, so it must describe what works.

    Built from the same shape constants the refusals use — one source, so the
    description cannot drift from what the validator accepts, and the bare form
    passes the PUBLISHED schema rather than only Pydantic's runtime check.
    """
    from jsonschema import Draft202012Validator

    from local_operator.providers.clients import (
        _tools_to_openai,
        _tools_to_openai_responses,
    )
    from local_operator.team_bridge.types import (
        RECIPIENT_ACCOUNT_SHAPE,
        RECIPIENT_ENDPOINT_SHAPE,
        SEND_RECIPIENT_DESCRIPTION,
    )

    runtime = BridgeRuntime(
        client=cast(TeamBridgeClient, object()),
        config_dir=".",
        tenant_id=TENANT,
        owner=_owner(),
        session_id="synthetic-schema",
    )
    tool = build_bridge_tool(ToolContext(bridge_runtime=runtime))
    assert tool is not None
    # The DESCRIPTION is the schema's own string: assert it on the parsed schema
    # (json.dumps escapes the quotes the shapes are written with) and on every
    # provider-compacted copy, since that is what a model actually reads.
    description = tool.parameters["properties"]["to"]["description"]
    assert description == SEND_RECIPIENT_DESCRIPTION
    assert RECIPIENT_ACCOUNT_SHAPE in description
    assert RECIPIENT_ENDPOINT_SHAPE in description
    assert "directory" in description
    for compacted in (
        _tools_to_openai([tool])[0]["function"]["parameters"],
        _tools_to_openai_responses([tool])[0]["parameters"],
    ):
        assert compacted["properties"]["to"]["description"] == SEND_RECIPIENT_DESCRIPTION
    for schema in (
        tool.parameters,
        _tools_to_openai([tool])[0]["function"]["parameters"],
        _tools_to_openai_responses([tool])[0]["parameters"],
    ):
        Draft202012Validator.check_schema(schema)
        assert (
            list(
                Draft202012Validator(schema).iter_errors(
                    {"op": "send", "to": "acct_123", "text": "synthetic"}
                )
            )
            == []
        )


@pytest.mark.asyncio
async def test_a_send_name_is_refused_after_the_scope_reads_not_before(
    tmp_path, monkeypatch
) -> None:
    """A name is a RESOLUTION REQUEST, so its refusal costs the scope reads — and no intent.

    The property this file exists for still holds: a refused recipient leaves no
    retryable outbox intent and dispatches no message. What changed with name
    resolution is WHEN it can refuse: the bridge directory and the owner's own
    allowlist are wire reads behind the admission gate, so a name that resolves to
    nobody refuses after them — naming BOTH reasons it might be (not on this team,
    or their bridge is not enabled) — and still leaves nothing behind.
    """
    from local_operator.team_bridge.outbox import Outbox

    records = []
    original = Outbox.record_send

    def record(self, *args, **kwargs):
        records.append(kwargs)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Outbox, "record_send", record)
    requests: list[httpx.Request] = []
    runtime = BridgeRuntime(
        client=_client(
            lambda request: (
                requests.append(request),
                # A directory with one member, and an allowlist naming somebody
                # else: the asked-for name is simply not in the reachable set.
                (
                    _ok(
                        {
                            "members": [
                                {
                                    "account_id": "acc_other",
                                    "display_label": "Bo Chen",
                                    "bridge_enabled": True,
                                }
                            ]
                        }
                    )
                    if request.url.path.endswith("/directory")
                    else _ok(
                        {
                            "allowlist": {
                                "entries": [
                                    {
                                        "direction": "send",
                                        "effect": "allow",
                                        "target": {"account_id": "acc_other"},
                                    }
                                ]
                            }
                        }
                    )
                ),
            )[1],
        ),
        config_dir=tmp_path,
        tenant_id=TENANT,
        owner=_owner(),
        session_id="synthetic-session",
    )
    result = await execute_bridge(
        runtime, {"op": "send", "to": {"name": "Ada Lovelace"}, "text": "synthetic"}
    )
    assert result["ok"] is False and result["code"] == "invalid_request"
    # BOTH reasons, because a model that cannot tell them apart guesses.
    assert "not on this team" in result["error"]
    assert "bridge is not enabled" in result["error"]
    assert records == []
    # The scope reads happened; nothing was sent.
    assert [request.method for request in requests] == ["GET", "GET"]
