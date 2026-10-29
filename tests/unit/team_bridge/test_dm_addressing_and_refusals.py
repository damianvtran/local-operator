"""``dm`` addressing, and what a bridge refusal is allowed to tell the reader.

TWO OPERATOR-VISIBLE DEFECTS, both pinned here because neither had a test that
could have caught it.

1. THE HUB'S FIELD-LEVEL DETAIL WAS DISCARDED. Every messaging 400 from the hub
   carries ``details: {"field": ..., "rule": ...}`` beside a sentence that tells
   the reader to "see details for the field" (agent-server
   ``internal/errors/messages.go`` — the phrase is deliberate; the details ARE
   the actionable half). The tool interpolated only ``str(failure)``, so an
   agent that opened a DM — refused by the hub for
   ``actor_endpoint_id``/``unsupported_kind`` — read "This messaging request is
   not valid. See details for the field." and nothing more, and the cause took
   hours to find. ``_failure_text`` appends the refusal's ``code`` and its
   ``field``/``rule`` pair, and appends NOTHING when a failure carries none, so
   every pre-existing message reads exactly as it did. What may travel is pinned
   to those three keys: a ``details`` payload that also carries a body, a token
   or an account id must not reach the sentence, because it reaches the model,
   the transcript and the screen.

2. ``to`` WAS ADVERTISED AND REFUSED. ``to`` is the shared recipient field — its
   published description is op-agnostic ("Who receives it: …"), ``send`` and
   ``reply`` address through it — and ``dm`` refused it client-side, so the
   documented form never reached the hub. ``dm`` now unfolds ``to`` onto
   ``account_id``/``endpoint_id`` and enters the SAME
   :func:`~local_operator.team_bridge.tool._resolve_dm_target` as the top-level
   fields. These tests pin PARITY, not a second implementation: both spellings
   must open the same DM and produce the same refusals, including the
   exactly-one-recipient rule, the unknown-endpoint answer, the
   endpoint-names-an-agent rule, and the room/pair destinations that remain
   refused.

Nothing here is a live-hub E2E: the client is a recorder and the fences are the
same ones ``test_channel_tool.py`` patches open, so these are TOOL semantics.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from local_operator.team_bridge import host as host_mod
from local_operator.team_bridge.client import (
    BridgeRefusal,
    BridgeUnreachable,
    TeamBridgeClient,
)
from local_operator.team_bridge.store import OwnerSnapshot
from local_operator.team_bridge.tool import BridgeRuntime, _failure_text, execute_bridge

TENANT = "ten_1"
PEER = "acc_peer"
PEER_ENDPOINT = "bep_peer"
DM_ID = "bcv_abc12345"  # bcv_ + 8 lowercase base36 chars: the minted family


@pytest.fixture(autouse=True)
def _gates_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fences have their own files; these tests are about dm and refusals.

    Same shape ``test_channel_tool.py`` uses: a store-less runtime stays readable
    and every op reaches its own branch instead of a fence.
    """

    monkeypatch.setattr(host_mod, "owner_fence_refusal", lambda runtime: None)
    monkeypatch.setattr(host_mod, "outbound_tool_fence", lambda runtime: None)
    monkeypatch.setattr(host_mod, "admission_refusal", lambda runtime, *, is_read=False: None)


def _owner(endpoint_id: str = "bep_1") -> OwnerSnapshot:
    return OwnerSnapshot(
        credential_id=1,
        account_id="acc_owner",
        tenant_id=TENANT,
        endpoint_id=endpoint_id,
        canonical_base_url="https://hub.example",
        enrollment_generation=1,
    )


def _member(
    *,
    account_id: str = PEER,
    label: str = "Robin",
    endpoint_id: str = PEER_ENDPOINT,
    state: str = "active",
    bridge_enabled: bool = True,
) -> dict[str, Any]:
    """One directory row: the hub's own shape, projected by ``_directory_rows``."""

    return {
        "account_id": account_id,
        "display_label": label,
        "role": "member",
        "bridge_enabled": bridge_enabled,
        "bridge_state": "enabled" if bridge_enabled else "disabled",
        "agent": {"endpoint_id": endpoint_id, "name": "Aida", "state": state},
    }


class DmClient:
    """Records the directory read and the DM write; scripts their answers.

    Only the methods a dm touches exist, so a second call shape (a pair send, a
    room read) fails loudly instead of silently passing here.
    """

    def __init__(
        self,
        members: list[dict[str, Any]],
        *,
        create_failure: BaseException | None = None,
        messages_failure: BaseException | None = None,
    ) -> None:
        self.members = members
        self.create_failure = create_failure
        self.messages_failure = messages_failure
        self.calls: list[dict[str, Any]] = []

    async def directory(self, tenant_id: str, *, expected_account_id: str | None = None) -> Any:
        self.calls.append(
            {
                "method": "directory",
                "tenant_id": tenant_id,
                "expected_account_id": expected_account_id,
            }
        )
        return {"members": self.members}

    async def create_conversation(
        self,
        body: Any,
        *,
        actor_endpoint_id: str | None = None,
        origin_session_id: str | None = None,
        expected_account_id: str | None = None,
    ) -> Any:
        self.calls.append(
            {
                "method": "create_conversation",
                "body": dict(body),
                "actor_endpoint_id": actor_endpoint_id,
                "origin_session_id": origin_session_id,
                "expected_account_id": expected_account_id,
            }
        )
        if self.create_failure is not None:
            raise self.create_failure
        return {
            "conversation": {
                "id": DM_ID,
                "kind": "dm",
                "members": [
                    {"kind": "agent", "account_id": "acc_owner", "state": "active"},
                    {"kind": "agent", "account_id": PEER, "state": "active"},
                ],
            }
        }

    async def conversation_send(
        self,
        conversation_id: str,
        *,
        client_ref: str,
        text: str,
        tenant_id: str | None = None,
        agent_label: str | None = None,
        actor_endpoint_id: str | None = None,
        origin_session_id: str | None = None,
        thread_root_id: str | None = None,
        expected_account_id: str | None = None,
    ) -> Any:
        self.calls.append(
            {
                "method": "conversation_send",
                "conversation_id": conversation_id,
                "text": text,
                "thread_root_id": thread_root_id,
            }
        )
        return {"id": "bcmsg_1", "seq": 4}

    async def conversation_messages(
        self,
        conversation_id: str,
        *,
        tenant_id: str | None = None,
        limit: int | None = None,
        before_seq: int | None = None,
        after_seq: int | None = None,
        endpoint_id: str | None = None,
    ) -> Any:
        self.calls.append({"method": "conversation_messages", "conversation_id": conversation_id})
        if self.messages_failure is not None:
            raise self.messages_failure
        return {"messages": []}


def _runtime(client: Any, *, endpoint_id: str = "bep_1") -> BridgeRuntime:
    return BridgeRuntime(
        client=cast(TeamBridgeClient, client),
        config_dir=".",
        tenant_id=TENANT,
        owner=_owner(endpoint_id),
        session_id="sess_1",
    )


def _dm(**extra: Any) -> dict[str, Any]:
    return {"op": "dm", "body": "ping", **extra}


def _created(client: DmClient) -> dict[str, Any]:
    """The one DM-create call this run made, or a failure naming what came instead."""

    calls = [call for call in client.calls if call["method"] == "create_conversation"]
    assert (
        len(calls) == 1
    ), f"expected exactly one DM create, saw {[c['method'] for c in client.calls]}"
    return calls[0]


# -- (2) the documented ``to`` form reaches the ONE resolver ------------------


@pytest.mark.asyncio
async def test_dm_to_endpoint_id_opens_the_same_dm_as_the_top_level_field() -> None:
    """``to={"endpoint_id": …}`` and ``endpoint_id=…`` are ONE destination.

    The DM body is what the hub keys find-or-create on, so byte-equal members is
    the strongest available statement that the alias reaches the same place: not
    "both worked", but "both asked for the same two people".
    """

    by_field = DmClient([_member()])
    by_to = DmClient([_member()])
    field_outcome = await execute_bridge(_runtime(by_field), _dm(endpoint_id=PEER_ENDPOINT))
    to_outcome = await execute_bridge(_runtime(by_to), _dm(to={"endpoint_id": PEER_ENDPOINT}))

    assert field_outcome["ok"] is True, field_outcome
    assert to_outcome["ok"] is True, to_outcome
    assert _created(by_to)["body"] == _created(by_field)["body"]
    assert _created(by_to)["body"]["members"][1] == {
        "account_id": PEER,
        "kind": "agent",
        "endpoint_id": PEER_ENDPOINT,
    }
    # The actor is still this session's own endpoint, whichever spelling came in.
    assert _created(by_to)["actor_endpoint_id"] == "bep_1"
    assert to_outcome["conversation_id"] == field_outcome["conversation_id"] == DM_ID


@pytest.mark.asyncio
@pytest.mark.parametrize("to", [{"account_id": PEER}, PEER])
async def test_dm_to_account_id_and_a_bare_id_reach_the_account_target(to: Any) -> None:
    """The object form and the bare-id form the schema also publishes both work.

    A bare id is one recipient in one field — the shape a model writes first —
    and ``validate_send_recipient`` already reads it as an account id, so the dm
    alias must not narrow it.
    """

    by_field = DmClient([_member()])
    by_to = DmClient([_member()])
    field_outcome = await execute_bridge(_runtime(by_field), _dm(account_id=PEER))
    to_outcome = await execute_bridge(_runtime(by_to), _dm(to=to))

    assert field_outcome["ok"] is True, field_outcome
    assert to_outcome["ok"] is True, to_outcome
    assert _created(by_to)["body"] == _created(by_field)["body"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("args", "expected"),
    [
        # ``to`` naming one key beside the OTHER top-level id: the pair cannot
        # settle two people, and the resolver's own words say so.
        (
            {"to": {"endpoint_id": PEER_ENDPOINT}, "account_id": PEER},
            "dm needs exactly one of account_id or endpoint_id",
        ),
        # Same key, two different ids: never guessed at.
        (
            {"to": {"endpoint_id": PEER_ENDPOINT}, "endpoint_id": "bep_other"},
            "dm to and endpoint_id disagree; name the recipient once",
        ),
        # The top-level form keeps its own refusal, unchanged.
        (
            {"account_id": PEER, "endpoint_id": PEER_ENDPOINT},
            "dm needs exactly one of account_id or endpoint_id",
        ),
        # ``target`` still decides what an endpoint id means: the alias must not
        # become a way around the resolver's human/agent rule.
        (
            {"to": {"endpoint_id": PEER_ENDPOINT}, "target": "human"},
            "endpoint_id addresses an agent; a human needs account_id",
        ),
    ],
)
async def test_dm_to_keeps_the_resolvers_ambiguity_refusals(
    args: dict[str, Any], expected: str
) -> None:
    client = DmClient([_member()])
    outcome = await execute_bridge(_runtime(client), _dm(**args))

    assert outcome["ok"] is False
    assert outcome["code"] == "invalid_request"
    assert outcome["error"] == expected
    assert [c["method"] for c in client.calls] == [], "a refused dm opens no DM"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("key", "value", "reason"),
    [
        ("endpoint_id", "bep_ghost", "unknown_endpoint"),
        ("account_id", "acc_ghost", "not_a_member"),
    ],
)
async def test_dm_to_keeps_the_not_found_refusal_identical(
    key: str, value: str, reason: str
) -> None:
    """A fresh directory is consulted either way, and the answer is the same."""

    by_field = DmClient([_member()])
    by_to = DmClient([_member()])
    field_outcome = await execute_bridge(_runtime(by_field), _dm(**{key: value}))
    to_outcome = await execute_bridge(_runtime(by_to), _dm(to={key: value}))

    assert to_outcome == field_outcome
    assert to_outcome["code"] == "not_reachable"
    assert to_outcome["reason"] == reason
    # Nothing was written for an unknown target, whichever spelling named it.
    assert not [c for c in by_to.calls if c["method"] == "create_conversation"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "args",
    [
        {"thread_id": "thr_1"},
        {"channel_id": DM_ID},
        {"members": [{"account_id": PEER}]},
    ],
)
async def test_dm_still_refuses_room_destinations(args: dict[str, Any]) -> None:
    """``dm`` is not a room op: a thread id, a room id or a member list refuses.

    Only ``to`` changed meaning here; the room fields keep their refusal, and the
    message now names what dm DOES take instead of claiming ``to`` is refused.
    """

    client = DmClient([_member()])
    outcome = await execute_bridge(_runtime(client), _dm(**args))

    assert outcome["ok"] is False
    assert outcome["code"] == "invalid_request"
    assert outcome["error"] == (
        "dm takes account_id, endpoint_id or to; thread_id, channel_id and "
        "members belong to the pair-inbox and room ops"
    )
    assert client.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "to",
    [
        {"channel_id": DM_ID},
        {"account_id": PEER, "endpoint_id": PEER_ENDPOINT},
    ],
)
async def test_dm_to_refuses_a_pair_or_room_destination_at_the_shared_recipient(
    to: dict[str, Any],
) -> None:
    """The closed one-key recipient shape is ``validate_send_recipient``'s, shared.

    A dm must not accept a destination its siblings refuse, and must not answer
    with a dm-specific rule either: the refusal is the same validator's, raised
    before any directory read.
    """

    client = DmClient([_member()])
    outcome = await execute_bridge(_runtime(client), _dm(to=to))

    assert outcome["ok"] is False
    assert outcome["code"] == "invalid_request"
    assert outcome["error"].startswith("bridge args invalid: ")
    assert client.calls == []


# -- (1) the hub's coded detail reaches the reader ---------------------------


@pytest.mark.asyncio
async def test_a_refused_dm_names_the_hubs_code_field_and_rule() -> None:
    """The regression the operator hit: the sentence pointed at dropped detail.

    The hub's answer is reproduced verbatim (``invalid_request`` with
    ``field=actor_endpoint_id``/``rule=unsupported_kind``, the shape
    agent-server returns for an agent-opened DM), so this asserts the reader can
    now see the field and rule the sentence told them to look for.
    """

    refusal = BridgeRefusal(
        "invalid_request",
        "This messaging request is not valid. See details for the field.",
        http_status=400,
        details={"field": "actor_endpoint_id", "rule": "unsupported_kind"},
    )
    outcome = await execute_bridge(
        _runtime(DmClient([_member()], create_failure=refusal)), _dm(endpoint_id=PEER_ENDPOINT)
    )

    assert outcome["ok"] is False, outcome
    assert outcome["code"] == "dm_failed", outcome
    error = outcome["error"]
    assert "This messaging request is not valid. See details for the field." in error
    assert "code=invalid_request" in error
    assert "field=actor_endpoint_id" in error
    assert "rule=unsupported_kind" in error
    # CONTENT-FREE: the recipient ids and the body never enter the sentence.
    assert PEER not in error and PEER_ENDPOINT not in error
    assert "ping" not in error


@pytest.mark.asyncio
async def test_a_failure_with_no_server_answer_reads_exactly_as_before() -> None:
    """No hub answer means no footer — not an empty bracket.

    A transport arm carries a code but never answered, so the sentence must stay
    byte-identical to what shipped: the footer quotes a refusal the hub made,
    never every failure that crosses this path.
    """

    failure = BridgeUnreachable("Radient could not be reached.")
    outcome = await execute_bridge(
        _runtime(DmClient([_member()], create_failure=failure)), _dm(endpoint_id=PEER_ENDPOINT)
    )

    assert outcome["error"] == "the direct message was not opened: Radient could not be reached."
    assert "[" not in outcome["error"]


def test_a_server_answer_with_only_a_code_still_names_it() -> None:
    """The footer quotes an ANSWER: an uncoded 400 body still says WHICH refusal.

    ``_refusal`` re-codes a body with no ``code`` as ``bridge_error``; naming that
    is the difference between "the hub refused this" and "something went wrong".
    """

    failure = BridgeRefusal(
        "bridge_error", "Radient refused the request (HTTP 400).", http_status=400
    )

    assert _failure_text(failure) == (
        "Radient refused the request (HTTP 400). [refusal code=bridge_error]"
    )


def test_the_footer_carries_only_the_code_and_the_field_rule_pair() -> None:
    """What may travel is an allow-list, not a redaction pass over unknown keys."""

    failure = BridgeRefusal(
        "invalid_request",
        "Refused.",
        http_status=400,
        details={
            "field": "actor_endpoint_id",
            "rule": "unsupported_kind",
            "body": "the screened message text",
            "token": "tok_live_123",
            "account_id": "acc_secret",
        },
    )

    assert _failure_text(failure) == (
        "Refused. [refusal code=invalid_request field=actor_endpoint_id rule=unsupported_kind]"
    )


def test_a_smuggled_detail_value_cannot_shape_the_sentence() -> None:
    """A value is a server identifier: flattened and bounded before it lands.

    The sentence is billed into the agent's context and persisted in the
    transcript, so a newline or a paragraph in ``details`` must not be able to
    reformat the result around it.
    """

    failure = BridgeRefusal(
        "invalid_request",
        "Refused.",
        details={"field": "a\nb   c", "rule": "x" * 200},
    )

    text = _failure_text(failure)
    assert "\n" not in text
    assert "field=a b c" in text
    assert f"rule={'x' * 64}" in text
    assert "x" * 65 not in text


@pytest.mark.asyncio
async def test_a_sibling_op_surfaces_the_same_detail() -> None:
    """The sweep is the tool's, not ``dm``'s: a room read reports it too.

    ``channel_history`` wraps a hub failure with the same helper, so the fix does
    not leave the next undiagnosable refusal behind a different op.
    """

    refusal = BridgeRefusal(
        "invalid_request",
        "This messaging request is not valid. See details for the field.",
        http_status=400,
        details={"field": "limit", "rule": "out_of_range"},
    )
    client = DmClient([_member()], messages_failure=refusal)
    outcome = await execute_bridge(_runtime(client), {"op": "channel_history", "channel_id": DM_ID})

    assert outcome["ok"] is False
    assert outcome["code"] == "history_failed"
    assert "code=invalid_request" in outcome["error"]
    assert "field=limit" in outcome["error"]
    assert "rule=out_of_range" in outcome["error"]
