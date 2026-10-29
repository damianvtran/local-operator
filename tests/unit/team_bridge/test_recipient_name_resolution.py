"""Name -> member resolution for ``channel_post``: deliver on exactly one, else refuse.

WHY THIS FILE EXISTS. Measured on this machine: agents tried to message a
colleague with ``to: {"name": "Ada Lovelace"}`` and every call was refused,
because the tool accepted only opaque ids — five allow cases, seven calls, seven
refusals, zero messages on the wire. The feature had never worked for a human.

The fix is a RESOLUTION REQUEST, not a guess. A name is resolved against a
caller-supplied candidate set — for ``channel_post``, THAT conversation's active
members — and delivered only on an exactly-one match:

* one match      -> that member's mention row goes on the wire (ids, never the name);
* no match       -> refuse, saying no active member by that name exists there;
* several        -> refuse with the candidate list (labels and ids), never the first.

What these tests are FOR, in order of how much a wrong implementation would cost:
a name that exists in the ORGANIZATION but not in the conversation must not
resolve (least privilege — the one that catches a directory-wide implementation),
ambiguity must not become a preference between a human and an agent, a prefix or
substring must not match, and every attempt — refusal included — must leave an
audit row, because a later "it went to the wrong person" claim is answered from
that row or not at all.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from local_operator.team_bridge import client as tb_client
from local_operator.team_bridge import host as host_mod
from local_operator.team_bridge.client import TeamBridgeClient
from local_operator.team_bridge.recipient_resolution import (
    RESOLUTION_AUDIT_FILENAME,
    resolution_audit_path,
)
from local_operator.team_bridge.tool import execute_bridge
from local_operator.team_bridge.types import (
    NAMED_CANDIDATES_MAX,
    SCOPE_CONVERSATION,
    MemberCandidate,
    RecipientNameRefused,
    resolve_member_by_name,
)
from tests.unit.team_bridge.test_channel_tool import RecordingClient, _runtime
from tests.unit.team_bridge.test_client import FakeStore, _ok

CHANNEL = "bcv_abcd1234"
TENANT = "ten_1"

#: Ada is a HUMAN member; Bo is an AGENT member. Bo's ``endpoint_id`` is the
#: OPTIMISTIC arm on purpose: an agent member may carry one and the mention row
#: forwards it, but the frozen participant shape does not publish ``endpoint_id``,
#: so production agent mentions are usually ``{kind, account_id}`` alone. Both are
#: legal (``ScheduledMentionIn.endpoint_id`` is optional), and the omitted arm is
#: exercised by every human case here.
ADA = {"account_id": "acc_ada", "display_label": "Ada Lovelace", "kind": "human"}
BO = {"account_id": "acc_bo", "display_label": "Bo Chen", "kind": "agent", "endpoint_id": "bep_9"}

#: A colleague who exists in the ORGANIZATION but is not in this conversation.
#: The least-privilege case: a directory-wide resolver would happily deliver here.
GRACE = {"account_id": "acc_grace", "display_label": "Grace Hopper", "kind": "human"}


def _conversation(*members: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": CHANNEL,
        "kind": "channel",
        "members": list(members),
        "member_count": len(members),
    }


@pytest.fixture(autouse=True)
def _destination_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """The hub-destination guard has its own file; these tests are not about it."""

    monkeypatch.setattr(tb_client, "org_oauth_destination_allowed", lambda base: True)


@pytest.fixture(autouse=True)
def _gates_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same shape ``test_channel_tool`` uses: the fences are not under test here."""

    monkeypatch.setattr(host_mod, "owner_fence_refusal", lambda runtime: None)
    monkeypatch.setattr(host_mod, "admission_refusal", lambda runtime, *, is_read=False: None)


def _audit_rows(config_dir: Path) -> list[dict[str, Any]]:
    path = resolution_audit_path(config_dir)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sends(client: RecordingClient) -> list[dict[str, Any]]:
    return [call for call in client.calls if call["method"] == "conversation_send"]


def _posts_nothing(client: RecordingClient) -> bool:
    return _sends(client) == []


# ---------------------------------------------------------------------------
# The resolver itself: the matching rule, stated once
# ---------------------------------------------------------------------------


def test_the_resolver_matches_a_whole_display_name_case_insensitively() -> None:
    ada = MemberCandidate("acc_ada", "Ada Lovelace", "human")
    assert resolve_member_by_name("Ada Lovelace", [ada]) is ada
    assert resolve_member_by_name("ada lovelace", [ada]) is ada
    assert resolve_member_by_name("  ADA LOVELACE  ", [ada]) is ada


@pytest.mark.parametrize("asked", ["Ada", "Lovelace", "Ada Lov", "ada lovelac"])
def test_a_prefix_or_substring_never_resolves(asked: str) -> None:
    """A prefix is a guess about WHICH Ada, so it is refused, not completed."""
    ada = MemberCandidate("acc_ada", "Ada Lovelace", "human")
    with pytest.raises(RecipientNameRefused) as refusal:
        resolve_member_by_name(asked, [ada])
    assert refusal.value.resolution.matched_count == 0
    assert "no active member by that name in this conversation" in str(refusal.value)


def test_handles_and_labels_are_not_names() -> None:
    """Only ``display_label`` is matched: an endpoint/handle is not a display name."""
    ada = MemberCandidate("acc_ada", "Ada Lovelace", "human", endpoint_id="bep_ada")
    with pytest.raises(RecipientNameRefused):
        resolve_member_by_name("bep_ada", [ada])
    with pytest.raises(RecipientNameRefused):
        resolve_member_by_name("acc_ada", [ada])


def test_zero_matches_and_several_matches_are_both_refusals() -> None:
    one = MemberCandidate("acc_1", "Ada Lovelace", "human")
    two = MemberCandidate("acc_2", "Ada Lovelace", "agent", endpoint_id="bep_2")
    with pytest.raises(RecipientNameRefused) as ambiguous:
        resolve_member_by_name("Ada Lovelace", [one, two])
    assert ambiguous.value.resolution.matched_count == 2
    # The candidates are LISTED so the caller can disambiguate; neither is picked.
    assert "acc_1" in str(ambiguous.value) and "acc_2" in str(ambiguous.value)
    assert ambiguous.value.resolution.resolved is None


# ---------------------------------------------------------------------------
# Through the tool: exactly one delivers, and the WIRE carries the id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_unique_name_delivers_a_mention_of_the_resolved_account(tmp_path) -> None:
    """The name is resolved locally; the body carries the member's ids, not the name."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "GET":
            return _ok(_conversation(ADA, BO))
        return _ok({"message": {"id": "cmsg_1", "seq": 4}})

    runtime = _runtime(
        TeamBridgeClient(
            None,
            base_url="https://hub.example/v1",
            store=FakeStore(),
            transport=httpx.MockTransport(handler),
        ),
        config_dir=tmp_path,
    )
    outcome = await execute_bridge(
        runtime,
        {
            "op": "channel_post",
            "channel_id": CHANNEL,
            "body": "standing by for the QA round",
            "to": {"name": "ada lovelace"},
        },
    )
    assert outcome["ok"] is True, outcome
    assert outcome["resolved_account_id"] == "acc_ada"

    post = requests[-1]
    assert post.method == "POST"
    body = json.loads(body_text := post.content)
    assert body["mentions"] == [{"kind": "human", "account_id": "acc_ada"}]
    # The displayed name never reached the wire; only ids did.
    assert "Ada Lovelace" not in body_text.decode("utf-8")
    # The undirected parts of the wire are unchanged.
    assert body["actor_endpoint_id"] == "bep_1"
    assert body["body"] == "standing by for the QA round"


@pytest.mark.asyncio
async def test_an_undirected_post_stays_byte_identical_and_records_nothing(tmp_path) -> None:
    client = RecordingClient(conversation=_conversation(ADA, BO))
    outcome = await execute_bridge(
        _runtime(client, config_dir=tmp_path),
        {"op": "channel_post", "channel_id": CHANNEL, "body": "hello"},
    )
    assert outcome["ok"] is True
    assert outcome["resolved_account_id"] == ""
    # No conversation read, no mentions key, no audit row: nothing was resolved.
    assert [call["method"] for call in client.calls] == ["conversation_send"]
    assert _sends(client)[0]["mentions"] is None
    assert _audit_rows(tmp_path) == []


# ---------------------------------------------------------------------------
# The refusals: specific, and never a guess
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_name_matching_nobody_refuses_and_names_the_reason(tmp_path) -> None:
    client = RecordingClient(conversation=_conversation(ADA, BO))
    outcome = await execute_bridge(
        _runtime(client, config_dir=tmp_path),
        {"op": "channel_post", "channel_id": CHANNEL, "body": "hi", "to": {"name": "Nobody Here"}},
    )
    assert outcome["ok"] is False and outcome["code"] == "invalid_request"
    assert "no active member by that name in this conversation" in outcome["error"]
    assert _posts_nothing(client)


@pytest.mark.asyncio
async def test_a_name_in_the_org_but_not_in_the_conversation_does_not_resolve(tmp_path) -> None:
    """LEAST PRIVILEGE: the candidate set is THIS conversation's active members.

    Grace is a real colleague in the organization. Resolving her name for a post
    in a conversation she is not in would be a delivery outside the caller's
    scope, so the only correct answer is a refusal — and the org directory must
    never be the thing that resolves it.
    """
    client = RecordingClient(conversation=_conversation(ADA, BO))
    outcome = await execute_bridge(
        _runtime(client, config_dir=tmp_path),
        {
            "op": "channel_post",
            "channel_id": CHANNEL,
            "body": "grace, can you look?",
            "to": {"name": GRACE["display_label"]},
        },
    )
    assert outcome["ok"] is False and outcome["code"] == "invalid_request"
    assert "no active member by that name in this conversation" in outcome["error"]
    assert _posts_nothing(client)
    # Nothing directory-wide was consulted: one conversation read, no other call.
    assert [call["method"] for call in client.calls] == ["conversation"]
    # And the refusal is on the record, as an unresolved attempt.
    assert _audit_rows(tmp_path)[-1]["resolved_account_id"] is None


@pytest.mark.asyncio
async def test_an_agent_and_a_human_with_one_name_are_ambiguity_not_a_preference(tmp_path) -> None:
    """Neither kind wins: preferring one is a guess dressed as a policy."""
    dual = dict(ADA) | {"kind": "agent", "endpoint_id": "bep_dual"}
    client = RecordingClient(conversation=_conversation(ADA, dual, BO))
    outcome = await execute_bridge(
        _runtime(client, config_dir=tmp_path),
        {"op": "channel_post", "channel_id": CHANNEL, "body": "hi", "to": {"name": "Ada Lovelace"}},
    )
    assert outcome["ok"] is False and outcome["code"] == "invalid_request"
    assert "matches 2 members in this conversation" in outcome["error"]
    # Both candidates are named, with the ids that disambiguate them.
    assert "acc_ada" in outcome["error"]
    assert "bep_dual" not in outcome["error"]  # endpoints are not disambiguators
    assert "acc_" in outcome["error"] and outcome["error"].count("acc_") >= 2
    assert _posts_nothing(client)


@pytest.mark.asyncio
async def test_ambiguity_lists_the_candidates_and_asks_for_a_disambiguator(tmp_path) -> None:
    first = {"account_id": "acc_ada1", "display_label": "Ada Lovelace", "kind": "human"}
    second = {"account_id": "acc_ada2", "display_label": "Ada Lovelace", "kind": "human"}
    client = RecordingClient(conversation=_conversation(first, second))
    outcome = await execute_bridge(
        _runtime(client, config_dir=tmp_path),
        {"op": "channel_post", "channel_id": CHANNEL, "body": "hi", "to": {"name": "Ada Lovelace"}},
    )
    assert outcome["ok"] is False
    for account_id in ("acc_ada1", "acc_ada2"):
        assert account_id in outcome["error"]
    assert "ask for" in outcome["error"]
    assert _posts_nothing(client)


@pytest.mark.asyncio
async def test_a_name_plus_a_second_key_refuses_before_any_call(tmp_path) -> None:
    """Exactly one recipient, on every op: two keys is two destinations."""
    client = RecordingClient(conversation=_conversation(ADA))
    for target in (
        {"name": "Ada Lovelace", "account_id": "acc_ada"},
        {"name": "Ada Lovelace", "kind": "human"},
        {"name": "Ada Lovelace", "endpoint_id": "bep_9"},
    ):
        outcome = await execute_bridge(
            _runtime(client, config_dir=tmp_path),
            {"op": "channel_post", "channel_id": CHANNEL, "body": "hi", "to": target},
        )
        assert outcome["ok"] is False and outcome["code"] == "invalid_request"
    assert client.calls == []


@pytest.mark.asyncio
async def test_an_id_recipient_is_not_a_channel_post_destination(tmp_path) -> None:
    """The two id shapes belong to send/reply; the name is this op's form."""
    client = RecordingClient(conversation=_conversation(ADA))
    outcome = await execute_bridge(
        _runtime(client, config_dir=tmp_path),
        {
            "op": "channel_post",
            "channel_id": CHANNEL,
            "body": "hi",
            "to": {"account_id": "acc_ada"},
        },
    )
    assert outcome["ok"] is False and outcome["code"] == "invalid_request"
    assert '{"name": "<display name>"}' in outcome["error"]
    assert client.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["state", "status"])
@pytest.mark.parametrize("value", ["invited", "left", "removed"])
async def test_a_member_who_has_not_joined_is_not_a_candidate(tmp_path, key, value) -> None:
    """An invite that cannot receive must not resolve: that would be a silent non-delivery.

    Parametrised over BOTH spellings the participant shape has worn: ``state`` is
    the frozen contract's, ``status`` is the legacy hub's. A resolver that read
    only one of them would pass a single-key test and still deliver to somebody
    who had left the conversation.
    """
    inactive = dict(ADA) | {key: value}
    client = RecordingClient(conversation=_conversation(inactive, BO))
    outcome = await execute_bridge(
        _runtime(client, config_dir=tmp_path),
        {"op": "channel_post", "channel_id": CHANNEL, "body": "hi", "to": {"name": "Ada Lovelace"}},
    )
    assert outcome["ok"] is False
    assert "no active member by that name" in outcome["error"]
    assert _posts_nothing(client)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra",
    [
        # BOTH orders, because one row only proves the resolver reads a key it
        # would have read anyway: ``{state: invited, status: active}`` also
        # excludes under first-key-wins, so the REVERSE row is the one that
        # fails unless EVERY present state key is consulted.
        {"state": "invited", "status": "active"},
        {"state": "active", "status": "invited"},
        # The third spelling the participant shape has worn, and the one no
        # other case exercises.
        {"membership": "left"},
    ],
)
async def test_any_inactive_spelling_outranks_an_active_one(tmp_path, extra) -> None:
    """Fail closed: a row cannot be joined in one spelling and departed in another.

    ANY present inactive state excludes the member. Reading only the first key
    that happens to be present would let one of these rows through, which is the
    guess this resolver exists not to make.
    """
    mixed = dict(ADA) | extra
    client = RecordingClient(conversation=_conversation(mixed))
    outcome = await execute_bridge(
        _runtime(client, config_dir=tmp_path),
        {"op": "channel_post", "channel_id": CHANNEL, "body": "hi", "to": {"name": "Ada Lovelace"}},
    )
    assert outcome["ok"] is False
    assert "no active member by that name" in outcome["error"]
    assert _posts_nothing(client)


# ---------------------------------------------------------------------------
# The audit record: written before the effect, on refusal as well as success
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_every_attempt_is_recorded_before_the_post_refusal_and_success(tmp_path) -> None:
    client = RecordingClient(conversation=_conversation(ADA, BO))
    runtime = _runtime(client, config_dir=tmp_path)

    delivered = await execute_bridge(
        runtime,
        {"op": "channel_post", "channel_id": CHANNEL, "body": "a", "to": {"name": "Ada Lovelace"}},
    )
    refused = await execute_bridge(
        runtime,
        {"op": "channel_post", "channel_id": CHANNEL, "body": "b", "to": {"name": "Nobody Here"}},
    )
    assert delivered["ok"] is True and refused["ok"] is False

    rows = _audit_rows(tmp_path)
    assert len(rows) == 2
    # The file is JSONL: one row per line, each independently parseable.
    raw = resolution_audit_path(tmp_path).read_text(encoding="utf-8")
    assert len(raw.splitlines()) == 2
    delivered_row, refused_row = rows
    # The success row: what was asked, what it resolved to, how many matched.
    assert delivered_row["requested"] == "Ada Lovelace"
    assert delivered_row["resolved_account_id"] == "acc_ada"
    assert delivered_row["matched_count"] == 1
    assert delivered_row["outcome"] == "resolved"
    assert delivered_row["scope"] == "conversation"
    assert delivered_row["scope_id"] == CHANNEL
    # The refusal row: the SAME shape, with the id null — this is the row a
    # misdelivery claim is answered from, and it exists only if refusal writes it.
    assert refused_row["requested"] == "Nobody Here"
    assert refused_row["resolved_account_id"] is None
    assert refused_row["matched_count"] == 0
    assert refused_row["outcome"] == "no_match"
    assert refused_row["scope_id"] == CHANNEL
    # A body never reaches the audit: the log is about WHO, not what was said.
    raw = resolution_audit_path(tmp_path).read_text(encoding="utf-8")
    assert len(raw.splitlines()) == 2  # the file is JSONL, one row per line
    assert '"body"' not in raw
    assert resolution_audit_path(tmp_path).name == RESOLUTION_AUDIT_FILENAME


@pytest.mark.asyncio
async def test_an_unreadable_candidate_set_is_recorded_as_unavailable(tmp_path) -> None:
    """The one attempt that never classified anything is on the record too.

    ``members_failed`` must not be the silent path: "I asked for Ada and nothing
    happened" has to be reconstructable, and its outcome word must not pretend
    the members were consulted (``no_match`` would say exactly that).
    """

    class UnreadableConversation(RecordingClient):
        async def conversation(self, conversation_id: str, *, tenant_id: Any = None) -> Any:
            self.calls.append({"method": "conversation", "conversation_id": conversation_id})
            raise RuntimeError("hub said no")

    client = UnreadableConversation()
    outcome = await execute_bridge(
        _runtime(client, config_dir=tmp_path),
        {"op": "channel_post", "channel_id": CHANNEL, "body": "hi", "to": {"name": "Ada Lovelace"}},
    )
    assert outcome["ok"] is False and outcome["code"] == "members_failed"
    assert _posts_nothing(client)
    rows = _audit_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["outcome"] == "unavailable"
    assert rows[0]["resolved_account_id"] is None
    assert rows[0]["matched_count"] == 0
    assert rows[0]["requested"] == "Ada Lovelace"
    assert rows[0]["scope_id"] == CHANNEL


@pytest.mark.asyncio
async def test_the_ambiguity_refusal_is_bounded_but_the_count_is_exact(tmp_path) -> None:
    """Many same-label members must not turn one refusal into a page of ids.

    The legacy placeholder label (``Member``) is shared by many participants of
    one conversation, so this is reachable: the sentence names at most
    ``NAMED_CANDIDATES_MAX`` and says how many more there are, while the claim
    about N stays exact.
    """
    crowd = [
        {"account_id": f"acc_{index:03d}", "display_label": "Member", "kind": "human"}
        for index in range(60)
    ]
    client = RecordingClient(conversation=_conversation(*crowd))
    outcome = await execute_bridge(
        _runtime(client, config_dir=tmp_path),
        {"op": "channel_post", "channel_id": CHANNEL, "body": "hi", "to": {"name": "member"}},
    )
    assert outcome["ok"] is False and outcome["code"] == "invalid_request"
    error = outcome["error"]
    assert "matches 60 members" in error
    assert error.count("acc_") == NAMED_CANDIDATES_MAX
    assert f"and {60 - NAMED_CANDIDATES_MAX} more" in error
    assert len(error) < 600
    assert _posts_nothing(client)
    # The AUDIT row is the wider record: it keeps its own bound and the exact count.
    row = _audit_rows(tmp_path)[-1]
    assert row["matched_count"] == 60
    assert len(row["candidates"]) == 20
    assert row["resolved_account_id"] is None


@pytest.mark.asyncio
async def test_the_audit_row_is_written_before_the_post_not_after(tmp_path, monkeypatch) -> None:
    """Ordering is the guarantee: a post that never happened still has its row."""
    client = RecordingClient(conversation=_conversation(ADA, BO))
    seen_at_send: list[int] = []

    original_send = client.conversation_send

    async def send_after_audit(*args: Any, **kwargs: Any) -> dict[str, Any]:
        seen_at_send.append(len(_audit_rows(tmp_path)))
        return await original_send(*args, **kwargs)

    monkeypatch.setattr(client, "conversation_send", send_after_audit)
    outcome = await execute_bridge(
        _runtime(client, config_dir=tmp_path),
        {"op": "channel_post", "channel_id": CHANNEL, "body": "a", "to": {"name": "Ada Lovelace"}},
    )
    assert outcome["ok"] is True
    # The row already existed when the wire call was made.
    assert seen_at_send == [1]


@pytest.mark.asyncio
async def test_an_unwritable_audit_refuses_the_directed_post(tmp_path) -> None:
    """A delivery we cannot account for is not made."""
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    # A FILE where the audit's directory must go: the write cannot succeed.
    (blocked / "team_bridge").write_text("not a directory", encoding="utf-8")
    client = RecordingClient(conversation=_conversation(ADA))
    outcome = await execute_bridge(
        _runtime(client, config_dir=blocked),
        {"op": "channel_post", "channel_id": CHANNEL, "body": "a", "to": {"name": "Ada Lovelace"}},
    )
    assert outcome["ok"] is False and outcome["code"] == "audit_unavailable"
    assert _posts_nothing(client)


# ---------------------------------------------------------------------------
# The serializer: a mention that reached the tool cannot vanish on the wire
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mention",
    [
        {"account_id": "acc_ada"},
        {"kind": "robot", "account_id": "acc_ada"},
        {"kind": "human", "account_id": "acc_ada", "label": "Ada"},
        {"kind": "human", "account_id": " acc_ada"},
        {"kind": "human", "account_id": ""},
        "acc_ada",
    ],
)
async def test_the_client_refuses_a_mention_it_cannot_serialize(mention) -> None:
    """The body is never rebuilt from an unvalidated row (the silent-loss class)."""
    client = TeamBridgeClient(
        None,
        base_url="https://hub.example/v1",
        store=FakeStore(),
        transport=httpx.MockTransport(lambda request: _ok({"message": {"id": "cmsg_1"}})),
    )
    with pytest.raises(ValueError):
        await client.conversation_send(CHANNEL, client_ref="ref-1", text="hi", mentions=[mention])


@pytest.mark.asyncio
async def test_the_client_omits_mentions_entirely_when_there_are_none() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _ok({"message": {"id": "cmsg_1"}})

    client = TeamBridgeClient(
        None,
        base_url="https://hub.example/v1",
        store=FakeStore(),
        transport=httpx.MockTransport(handler),
    )
    await client.conversation_send(CHANNEL, client_ref="ref-1", text="hi", mentions=[])
    body = json.loads(requests[-1].content)
    assert "mentions" not in body


# ---------------------------------------------------------------------------
# The schema: the model can pick the right op by reading, not by failing first
# ---------------------------------------------------------------------------


def test_the_published_schema_states_the_matching_rule_and_the_per_op_scope() -> None:
    from local_operator.team_bridge.tool import BridgeToolParams
    from local_operator.team_bridge.types import SEND_RECIPIENT_DESCRIPTION

    description = BridgeToolParams.model_json_schema()["properties"]["to"]["description"]
    assert description == SEND_RECIPIENT_DESCRIPTION
    # The matching rule, so a model does not have to discover it by refusal.
    assert "case-insensitive" in description
    assert "trimmed" in description
    assert "display name" in description
    assert "substring" in description
    # The per-op scope, which is what decides which op to reach for.
    assert "channel_post" in description
    assert "active members" in description
    assert "send/reply" in description
    # And the shape itself, so the form is discoverable from the schema.
    assert '{"name": "<display name>"}' in description


def test_the_scope_words_are_one_vocabulary_the_audit_can_be_read_by() -> None:
    assert SCOPE_CONVERSATION.name == "conversation"
    assert SCOPE_CONVERSATION.where == "in this conversation"


def test_the_audit_outcome_vocabulary_is_closed_and_enforced(tmp_path) -> None:
    """A sixth outcome word would become a state nobody queries for."""
    from local_operator.team_bridge.recipient_resolution import (
        OUTCOMES,
        record_resolution,
    )
    from local_operator.team_bridge.types import NameResolution, TeamBridgeTypeError

    resolved = NameResolution(
        requested="Ada Lovelace", matches=(MemberCandidate("acc_ada", "Ada Lovelace", "human"),)
    )
    with pytest.raises(TeamBridgeTypeError):
        record_resolution(
            config_dir=tmp_path,
            scope=SCOPE_CONVERSATION,
            scope_id=CHANNEL,
            resolution=resolved,
            outcome="a_word_nobody_queries",
        )
    assert not resolution_audit_path(tmp_path).exists()
    # The documented words all work, and each lands in the row.
    for word in OUTCOMES:
        record_resolution(
            config_dir=tmp_path,
            scope=SCOPE_CONVERSATION,
            scope_id=CHANNEL,
            resolution=resolved,
            outcome=word,
        )
    assert [row["outcome"] for row in _audit_rows(tmp_path)] == list(OUTCOMES)
