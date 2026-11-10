"""bridge-screen-v5: the addressed-aware screening prompt (operator report 2026-11-09, A4).

agent-server serves v5 in an ADDITIVE ``screening_addressed`` status slot beside the
``screening`` slot's v2, and states each conversation delivery's routing on
``events[].ids.addressed`` / ``sender_relation``. This module pins the client half:

* the embedded v5 bytes are the server's (hash pin, byte length);
* v5 is a per-message RENDER choice — only for a text-only message whose routing is
  known, only when the hub offers v5 with matching bytes — and never the policy;
* with v5 absent or mismatched the v2 path is byte-identical to before;
* image/file messages stay on v3/v4;
* routing comes from the hub's event ids (carried on the feed ref), never body text;
* a v5 screen tells the hub so at admit time (the v5 hash).
"""

from __future__ import annotations

import hashlib
import re
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.team_bridge import client as tb_client
from local_operator.team_bridge import screening
from local_operator.team_bridge.client import (
    SCREENING_TEMPLATE_V2,
    SCREENING_TEMPLATE_V2_SHA256,
    SCREENING_TEMPLATE_V5,
    SCREENING_TEMPLATE_V5_PIN,
    SCREENING_VERSION_V2,
    SCREENING_VERSION_V3,
    SCREENING_VERSION_V4,
    SCREENING_VERSION_V5,
)
from local_operator.team_bridge.types import (
    ConversationFeedRef,
    parse_conversation_ref_value,
    project_events_batch,
)
from tests.unit.team_bridge.test_hold_release import (  # noqa: F401 — ``rig`` is a fixture
    BODY,
    CONVERSATION_ID,
    DELIVER,
    MESSAGE_ID,
    Policy,
    Runner,
    _item,
    _screen,
    rig,
)

#: agent-server ``internal/bridge/texts_test.go`` (PR #352, head c9a4a762).
SERVER_V5_SHA256 = "f2e18f6e9a9e45827717776289d030c0ec2153bc25ccc1f39e63cedc53114a2d"


class V5Policy(Policy):
    """The v2 policy of ``test_hold_release`` with the hub's v5 offer accepted."""

    def __init__(self) -> None:
        super().__init__()
        self.addressed_version = SCREENING_VERSION_V5


@pytest.fixture
def prompts(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every prompt the scripted screener received (the rendered bytes under test)."""

    seen: list[str] = []
    original = Runner.__call__

    def recording(self: Runner, prompt: str, *, default_model: str) -> tuple[str, str | None]:
        seen.append(prompt)
        return original(self, prompt, default_model=default_model)

    monkeypatch.setattr(Runner, "__call__", recording)
    return seen


def _routed(addressed: str, *, relation: str | None = None, stated: bool = True) -> dict[str, Any]:
    item = _item()
    item["addressed"] = addressed
    if relation is not None:
        item["relation"] = relation
    if stated:
        item["routing_stated"] = True
    return item


def _nonce(prompt: str) -> str:
    match = re.search(r"<submission-([0-9a-f]+)>", prompt)
    assert match is not None
    return match.group(1)


# --- the embedded bytes ---------------------------------------------------------------


def test_v5_bytes_are_the_servers_pinned_template() -> None:
    raw = SCREENING_TEMPLATE_V5.encode("utf-8")
    assert len(raw) == 2041
    assert hashlib.sha256(raw).hexdigest() == SERVER_V5_SHA256 == SCREENING_TEMPLATE_V5_PIN
    assert tb_client.screening_template_v5_usable()
    assert tb_client.screening_template_sha256(SCREENING_VERSION_V5) == SERVER_V5_SHA256
    assert "Relation: {{relation}}\nAddressed: {{addressed}}\n" in SCREENING_TEMPLATE_V5


def test_v5_is_never_accepted_as_the_policy_template() -> None:
    # v5 needs per-event routing pair rows do not have: a hub serving it in the
    # ``screening`` slot is refused like any unembedded selector.
    assert not tb_client._screening_is_trusted(SCREENING_VERSION_V5, SCREENING_TEMPLATE_V5)
    assert tb_client._screening_is_trusted(SCREENING_VERSION_V2, SCREENING_TEMPLATE_V2)


@pytest.mark.parametrize(
    ("slot", "expected"),
    [
        ({"version": SCREENING_VERSION_V5, "template": SCREENING_TEMPLATE_V5}, "bridge-screen-v5"),
        (None, ""),  # an older hub: no slot
        ({"version": SCREENING_VERSION_V5, "template": SCREENING_TEMPLATE_V5 + " "}, ""),
        ({"version": SCREENING_VERSION_V5, "template": SCREENING_TEMPLATE_V2}, ""),
        ({"version": "bridge-screen-v6", "template": SCREENING_TEMPLATE_V5}, ""),
        ({"version": SCREENING_VERSION_V2, "template": SCREENING_TEMPLATE_V2}, ""),
        ("bridge-screen-v5", ""),
    ],
)
def test_the_offer_requires_the_embedded_version_and_bytes(slot: Any, expected: str) -> None:
    status: dict[str, Any] = {
        "screening": {"version": SCREENING_VERSION_V2, "template": SCREENING_TEMPLATE_V2}
    }
    if slot is not None:
        status["screening_addressed"] = slot
    assert tb_client.screening_addressed_version(status) == expected


def test_drifted_embedded_bytes_disable_the_offer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tb_client, "SCREENING_TEMPLATE_V5_SHA256", "0" * 64)
    status = {
        "screening_addressed": {"version": SCREENING_VERSION_V5, "template": SCREENING_TEMPLATE_V5}
    }
    assert tb_client.screening_addressed_version(status) == ""


def test_the_status_law_records_the_offer_without_ever_refusing_on_it(tmp_path) -> None:
    from local_operator.team_bridge.scheduler import _normalize_accepted_status
    from tests.unit.team_bridge.test_scope_policy import _setup

    store, owner = _setup(tmp_path)
    payload: dict[str, Any] = {
        "api_version": "1",
        "available": True,
        "tenants": [
            {
                "tenant_id": owner.tenant_id,
                "tenant_name": "Org",
                "eligible": True,
                "endpoint": {"id": owner.endpoint_id, "label": "endpoint", "state": "active"},
                "counts": {"unread": 0, "requests_pending": 0, "blocked": 0},
            }
        ],
        "guidance": {"version": "bridge-guidance-v2", "text": "Guidance body."},
        "screening": {"version": SCREENING_VERSION_V2, "template": SCREENING_TEMPLATE_V2},
        "screening_addressed": {"version": SCREENING_VERSION_V5, "template": SCREENING_TEMPLATE_V5},
        "consent": {"version": "bridge_consent_v1", "text": "Consent body."},
    }
    law = _normalize_accepted_status(payload, owner, store)
    assert (law.version, law.template_hash) == (SCREENING_VERSION_V2, SCREENING_TEMPLATE_V2_SHA256)
    assert law.addressed_version == SCREENING_VERSION_V5
    payload["screening_addressed"]["template"] = "tampered"
    law = _normalize_accepted_status(payload, owner, store)
    assert law.version == SCREENING_VERSION_V2 and law.addressed_version == ""


# --- the render choice ------------------------------------------------------------------


def test_v5_offered_renders_the_routing_lines(rig, prompts) -> None:  # noqa: F811
    config_dir, _store, owner = rig
    out = _screen(
        config_dir,
        owner,
        Runner(DELIVER),
        # ``member``: a stated ``owner`` renders only for the owner as a person, who is
        # [redacted] before any prompt (code review round 1 R2, pinned in test_review_held).
        item=_routed("direct", relation="member"),
        policy=V5Policy(),
    )
    assert out.disposition == "deliver"
    assert len(prompts) == 1
    assert "Routing (server facts):\nRelation: member\nAddressed: direct\n" in prompts[0]
    expected = screening.render_prompt(
        body_text=BODY,
        sender="Victor",
        organization="tnt_1",
        msg_type="note",
        nonce=_nonce(prompts[0]),
        version=SCREENING_VERSION_V5,
        addressed="direct",
        relation="member",
    )
    assert prompts[0] == expected.private_text()
    # The outcome names the prompt that judged it; the LOCAL receipt keeps the
    # policy template every store fence compares against.
    assert out.rendered_version == SCREENING_VERSION_V5
    assert out.rendered_template_sha256 == SCREENING_TEMPLATE_V5_PIN
    assert out.receipt is not None
    assert out.receipt.prompt_version == SCREENING_VERSION_V2
    assert out.receipt.template_sha256 == SCREENING_TEMPLATE_V2_SHA256


def test_an_ambient_new_work_ask_renders_addressed_ambient(rig, prompts) -> None:  # noqa: F811
    """The incident: a channel post asking SOMEONE ELSE's aida for new work."""

    config_dir, _store, owner = rig
    _screen(config_dir, owner, Runner(DELIVER), item=_routed("ambient"), policy=V5Policy())
    assert BODY in prompts[0]
    # No relation from the hub is its documented default: member.
    assert "Relation: member\nAddressed: ambient\n" in prompts[0]


def test_v5_absent_keeps_the_v2_prompt_byte_identical(rig, prompts) -> None:  # noqa: F811
    config_dir, _store, owner = rig
    out = _screen(config_dir, owner, Runner(DELIVER), item=_routed("ambient"), policy=Policy())
    expected = screening.render_prompt(
        body_text=BODY,
        sender="Victor",
        organization="tnt_1",
        msg_type="note",
        nonce=_nonce(prompts[0]),
        version=SCREENING_VERSION_V2,
    )
    assert prompts[0] == expected.private_text()
    assert "Addressed:" not in prompts[0] and "Relation:" not in prompts[0]
    assert out.rendered_version == SCREENING_VERSION_V2
    assert out.rendered_template_sha256 == SCREENING_TEMPLATE_V2_SHA256


def test_v5_offered_without_routing_stays_on_v2(rig, prompts) -> None:  # noqa: F811
    # A pair-shaped item (no routing at all) never reaches the v5 slots.
    config_dir, _store, owner = rig
    out = _screen(config_dir, owner, Runner(DELIVER), item=_item(), policy=V5Policy())
    assert "Addressed:" not in prompts[0]
    assert out.rendered_version == SCREENING_VERSION_V2


def test_the_template_choice_keeps_attachments_on_v3_and_v4() -> None:
    snapshot = screening.PolicySnapshot.capture(V5Policy())
    routed: Any = SimpleNamespace(addressed="direct")
    unrouted: Any = SimpleNamespace(addressed="")
    choose = screening.text_screening_version
    assert choose(snapshot, routed, images=object(), files=None) == SCREENING_VERSION_V3
    assert choose(snapshot, routed, images=None, files=object()) == SCREENING_VERSION_V4
    assert choose(snapshot, routed, images=object(), files=object()) == SCREENING_VERSION_V4
    assert choose(snapshot, routed, images=None, files=None) == SCREENING_VERSION_V5
    assert choose(snapshot, unrouted, images=None, files=None) == SCREENING_VERSION_V2
    v2_only = screening.PolicySnapshot.capture(Policy())
    assert choose(v2_only, routed, images=None, files=None) == SCREENING_VERSION_V2


def test_v5_render_refuses_a_value_outside_the_closed_sets() -> None:
    for addressed, relation in (("", "member"), ("mention", "member"), ("direct", "admin")):
        with pytest.raises(ValueError):
            screening.render_prompt(
                body_text="hi",
                sender="S",
                organization="O",
                msg_type="note",
                version=SCREENING_VERSION_V5,
                addressed=addressed,
                relation=relation,
            )


# --- routing provenance and binding --------------------------------------------------


def test_hub_stated_routing_binds_the_envelope_digest(rig) -> None:  # noqa: F811
    _config_dir, _store, owner = rig
    plain, _ = screening.validate_envelope(owner, _item())
    ambient, _ = screening.validate_envelope(owner, _routed("ambient"))
    direct, _ = screening.validate_envelope(owner, _routed("direct"))
    derived, _ = screening.validate_envelope(owner, _routed("direct", stated=False))
    assert plain and ambient and direct and derived
    # A stated routing is a fact about the delivery: the same words asked of this
    # agent and overheard in the room never share a cached verdict...
    assert len({plain.envelope_digest, ambient.envelope_digest, direct.envelope_digest}) == 3
    # ...while a derived one leaves every pre-v5 digest untouched.
    assert derived.envelope_digest == plain.envelope_digest
    assert (direct.addressed, direct.relation, direct.routing_stated) == ("direct", "", True)


def test_feed_events_carry_the_hubs_routing_onto_the_ref() -> None:
    def project(ids: dict[str, Any]) -> ConversationFeedRef:
        _cursor, refs, _resync = project_events_batch(
            {
                "events": [
                    {
                        "type": "message_available",
                        "ids": {
                            "message_id": MESSAGE_ID,
                            "conversation_id": CONVERSATION_ID,
                            "conversation_seq": 3,
                            "sender_kind": "human",
                            **ids,
                        },
                        "at": "2026-11-09T12:00:00Z",
                    }
                ],
                "cursor": "9",
            }
        )
        (value,) = refs
        ref = parse_conversation_ref_value(value.split("=", 1)[1])
        assert ref is not None
        return ref

    stated = project({"addressed": "ambient", "sender_relation": "owner"})
    assert (stated.addressed, stated.sender_relation) == ("ambient", "owner")
    # An older hub (or a value this build does not know) is "not stated", never a
    # refused page; the pre-v5 ref spelling is unchanged.
    old = project({})
    assert (old.addressed, old.sender_relation) == (None, None)
    assert "addressed=" not in old.encode()
    unknown = project({"addressed": "shouted", "sender_relation": ["owner"]})
    assert (unknown.addressed, unknown.sender_relation) == (None, None)


def test_routing_tags_round_trip_and_damaged_tags_refuse() -> None:
    ref = ConversationFeedRef(
        conversation_id=CONVERSATION_ID,
        message_id=MESSAGE_ID,
        conversation_seq=3,
        mention=False,
        sender_kind="agent",
        addressed="everyone",
        sender_relation="member",
    )
    value = ref.encode().split("=", 1)[1]
    assert parse_conversation_ref_value(value) == ref
    assert parse_conversation_ref_value(value.replace("everyone", "everybody")) is None
    assert parse_conversation_ref_value(value.replace("=member", "=boss")) is None


def test_the_connector_item_takes_the_hubs_routing_never_the_body() -> None:
    from local_operator.team_bridge.connector.conversation import screening_routing

    stated = SimpleNamespace(mention=True, addressed="ambient", sender_relation="owner")
    assert screening_routing(stated, {"conversation_kind": "dm"}) == {
        "addressed": "ambient",
        "routing_stated": True,
        "relation": "owner",
    }
    # Derived from the mention/DM/thread facts when the hub said nothing; never stated.
    old = SimpleNamespace(mention=True, addressed=None, sender_relation=None)
    assert screening_routing(old, {}) == {"addressed": "direct"}


def test_a_v5_screen_tells_the_hub_which_prompt_judged_it() -> None:
    from local_operator.team_bridge.connector.conversation import _text_admit_template

    v5 = SimpleNamespace(
        rendered_version=SCREENING_VERSION_V5, rendered_template_sha256=SCREENING_TEMPLATE_V5_PIN
    )
    assert _text_admit_template(v5) == SERVER_V5_SHA256
    # v2 (and a cache hit, which rendered nothing) keep the omitted pre-v5 shape.
    v2 = SimpleNamespace(
        rendered_version=SCREENING_VERSION_V2,
        rendered_template_sha256=SCREENING_TEMPLATE_V2_SHA256,
    )
    assert _text_admit_template(v2) is None
    assert _text_admit_template(SimpleNamespace()) is None


@pytest.mark.asyncio
async def test_a_v5_text_allow_reaches_the_hub_through_the_real_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """QA round 1 Q1: the v5 hash must leave the machine on a text-only admit.

    ``conversation_admit`` used to raise "a grant and template_sha256 travel together"
    for it, so every v5-screened allow was held ``conversation_failed``. The body is
    the shape agent-server #352's ``admitAttachmentGrant`` accepts: no grant, the
    template naming v5.
    """

    import json

    import httpx

    from local_operator.team_bridge.client import SCREENING_TEMPLATE_V3_SHA256
    from tests.unit.team_bridge.test_client import _make_client, _ok

    # The stub hub is not the Radient cloud API; the client's own tests open it the same way.
    monkeypatch.setattr(tb_client, "org_oauth_destination_allowed", lambda base: True)
    seen: list[tuple[str, dict[str, Any]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, json.loads(request.content)))
        return _ok({"verdict": "allow"})

    client = _make_client(handler)
    await client.conversation_admit(
        "bcv_qabusy01",
        "bcmsg_qa6a0001",
        verdict="allow",
        body_hash="0" * 64,
        tenant_id="tnt_qa1",
        endpoint_id="bep_qarecv",
        template_sha256=SCREENING_TEMPLATE_V5_PIN,
    )
    path, body = seen[-1]
    assert path.endswith("/messaging/conversations/bcv_qabusy01/messages/bcmsg_qa6a0001/admit")
    assert body == {
        "verdict": "allow",
        "body_hash": "0" * 64,
        "template_sha256": SERVER_V5_SHA256,
        "tenant_id": "tnt_qa1",
    }
    # Unchanged: the pre-v5 text allow omits the template entirely.
    await client.conversation_admit(
        "bcv_qabusy01", "bcmsg_qa6a0001", verdict="allow", body_hash="h"
    )
    assert "template_sha256" not in seen[-1][1]
    # The grant bindings still hold: a v3/v4 hash names an image/file grant, so it
    # never rides a text-only admit, and the v5 hash never rides a reject.
    with pytest.raises(ValueError):
        await client.conversation_admit(
            "bcv_qabusy01",
            "bcmsg_qa6a0001",
            verdict="allow",
            body_hash="h",
            template_sha256=SCREENING_TEMPLATE_V3_SHA256,
        )
    with pytest.raises(ValueError):
        await client.conversation_admit(
            "bcv_qabusy01",
            "bcmsg_qa6a0001",
            verdict="reject",
            body_hash="h",
            template_sha256=SCREENING_TEMPLATE_V5_PIN,
        )
    assert len(seen) == 2
