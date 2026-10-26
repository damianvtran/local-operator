"""The four recipient notification states, proven as far as the LOCAL half reaches.

WHY this file exists. ``test_conversation_delivery.py`` proves routing with a fake
``deliver_wake`` that only appends to a list, and its own matrix test says so:
"a synthetic contract fixture, not a real server fan-out proof". The operator's
question is what a RECIPIENT actually experiences, so this module drives the
real pieces that test replaces:

* the real ``loop.run_iteration`` long-poll → durable ``feed_refs`` → drain →
  ``plan_feed_refs`` → ``handle_conversation_available`` path (only the wire's
  HTTP verbs are doubled, as the no-hub constraint requires);
* the real admission → reservation → card composition;
* a real ``Session``: the wake is handed to ``Session.receive_peer_message``
  — the method the runtime's socket handler calls — and the test observes the
  provider request and transcript a recipient's agent would actually get.

WHAT IS NOT LOCAL, AND IS THEREFORE NOT CLAIMED AS PROVEN HERE. The fan-out
decision ("does this member get a feed row?") is the HUB's, from the member's
own server-side ``notify`` policy (client.py ``conversation_notify``: "Set the
caller's own notification policy (§10.1) … Another member's policy is never
writable or readable through this call"). No local module contains a mute,
notify or endpoint-disabled decision — ``connector/conversation.py`` starts from
"one ``message_available`` feed row per active agent member whose endpoint
consent allows it", i.e. the row is already the server's verdict. So:

* state 1 (mentioned) is proven end to end: row → card → wake → turn;
* state 2 (unmentioned) is proven in both LOCAL arms — a notify=all row wakes
  with ``mention: no``, and a mention-only member's silence produces no work —
  while the rule that turns the setting into a row is modelled from §10.1 and
  named as the server's, not asserted as local behaviour;
* states 3 (muted) and 4 (endpoint disabled) are proven in the only shape the
  local half can own: the hub emits nothing, and the test fails if any card,
  wake, turn or delivery row appears. The message's survival is asserted on
  the hub's history: the row and its hash are still served to the member, the
  bytes are retained server-side, and no local row consumed either. (An agent
  read carries no body until admission, so that later step is named, not
  claimed.)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.harness.message_types import (
    BRIDGE_UNTRUSTED_CLAUSE,
    TEAM_BRIDGE_MESSAGE_TYPE,
)
from local_operator.harness.types import (
    AbortSignal,
    ChatRequest,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
)
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.team_bridge import pull as pull_mod
from local_operator.team_bridge.client import BridgeRefusal
from local_operator.team_bridge.connector import conversation as conversation_mod
from local_operator.team_bridge.connector import loop as loop_mod
from local_operator.team_bridge.screening import ScreenResult
from local_operator.team_bridge.store import (
    OwnerReceipt,
    OwnerSnapshot,
    Store,
    state_db_path,
)
from local_operator.team_bridge.types import ScreeningReceipt

MODEL = ModelSpec(provider="test", model_id="m", context_window=100_000)
CONVERSATION_ID = "bcv_abcdefgh"
MESSAGE_ID = "bcmsg_abcdef01"
TENANT = "tnt_1"
BODY = "VISIBLE-ONLY-ON-A-READ-7731 the message body that must survive a mute"
HISTORY_AT = "2026-10-20T00:00:00Z"
SENDER = {"account_id": "acc_human", "kind": "human", "display_label": "Ada"}

#: The store tables that record DELIVERY work. ``sync`` is deliberately absent:
#: the poll cursor advancing is ordinary long-poll bookkeeping (the loop parks
#: on the feed), not a message this member was given. Everything here must stay
#: byte-identical across a silent poll.
DELIVERY_TABLES = (
    "feed_refs",
    "deliveries",
    "claims",
    "body_reservations",
    "ack_intents",
    "screening_attempts",
    "screening_receipts",
    "screening_holds",
)

#: How long to let a spawned turn finish: the scripted provider answers at once,
#: so this only bounds a hung turn rather than pacing one.
TURN_SETTLE_TIMEOUT_S = 10.0


@pytest.fixture(autouse=True)
def _isolated_machine(tmp_path, monkeypatch):
    """No ambient config: every global lookup lands in this test's tmp tree.

    The recipient stores and transcripts are per-test paths already; this pins
    the ONE global ``config_dir()`` the receive path may consult (the peer
    sender registry) so nothing can read the developer's real machine.
    """

    root = tmp_path / "machine-config"
    root.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    return root


# --------------------------------------------------------------------------- #
# Real local identity, store and session
# --------------------------------------------------------------------------- #


def _owner(name: str, credential_id: int) -> OwnerSnapshot:
    return OwnerSnapshot(
        credential_id=credential_id,
        account_id=f"acc_{name}",
        tenant_id=TENANT,
        endpoint_id=f"bep_{name}",
        canonical_base_url="https://api.radienthq.com/v1",
        enrollment_generation=1,
    )


def _enroll(config_dir: Path, owner: OwnerSnapshot, session_id: str) -> Store:
    store = Store(config_dir)
    store.commit_enrollment(
        owner, OwnerReceipt("bridge_consent_v1", f"ref-{session_id}"), cos_session_id=session_id
    )
    # The fence re-reads this from disk at every boundary; the production shape
    # is YAML, and JSON is valid YAML, so one literal keeps it exact.
    (config_dir / "config.yml").write_text(
        json.dumps({"values": {"team_bridge": {"enabled": True}}}), encoding="utf-8"
    )
    return store


def _access(owner: OwnerSnapshot) -> Any:
    """The pinned OAuth row the real pull path demands (no cascade exists)."""

    return SimpleNamespace(
        credential_id=owner.credential_id, account_id=owner.account_id, kind="oauth"
    )


class CapturingStream:
    """Answer like a provider, keep every request: the recipient's own view."""

    def __init__(self) -> None:
        self.requests: list[ChatRequest] = []

    def __call__(self, request: ChatRequest, signal: AbortSignal | None):
        self.requests.append(request)

        async def gen():
            yield StreamTextDelta(delta="acknowledged")
            yield StreamEndEvent(stop_reason="stop")

        return gen()


def _request_texts(request: ChatRequest) -> list[str]:
    out: list[str] = []
    for message in request.messages:
        content = getattr(message, "content", None)
        if isinstance(content, str):
            out.append(content)
            continue
        for block in content or []:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                out.append(text)
    return out


@dataclass
class Recipient:
    """One enrolled member on this machine: store, designated session, agent."""

    name: str
    owner: OwnerSnapshot
    config_dir: Path
    session_id: str
    stream: CapturingStream = field(default_factory=CapturingStream)
    session: Session | None = None

    @property
    def endpoint(self) -> str:
        return self.owner.endpoint_id

    @property
    def baseline_requests(self) -> int:
        return self._baseline

    @property
    def live_session(self) -> Session:
        """The engaged session, narrowed for callers that need a real one.

        ``_make_recipient`` always sets it, but the holder is built before the
        engagement exists, so the field is Optional; this property is the one
        place that says so to the type checker.
        """

        assert self.session is not None, "recipient is not engaged yet"
        return self.session

    _baseline: int = 0


async def _make_recipient(recipient: Recipient) -> Recipient:
    """Engage the designated session for real: one user turn, on disk.

    ``engagement_hold`` releases a card for a session with durable history, and
    the unengaged exception is a separate, consent-gated path. A real turn is
    the honest way to make this recipient a session its owner has used.
    """

    recipient.config_dir.mkdir(parents=True, exist_ok=True)
    _enroll(recipient.config_dir, recipient.owner, recipient.session_id)
    directory = recipient.config_dir / "sessions" / recipient.session_id
    directory.mkdir(parents=True)
    recipient.session = Session(
        model=MODEL,
        stream_fn=recipient.stream,
        tools=[],
        transcript=Transcript(directory),
        system_blocks_provider=lambda: ["stable", "env"],
    )
    await recipient.session.prompt("hello from the owner")
    recipient._baseline = len(recipient.stream.requests)
    return recipient


async def _settle(session: Session) -> None:
    """Wait for the wake-spawned turn (and any task it spawned) to finish."""

    async with asyncio.timeout(TURN_SETTLE_TIMEOUT_S):
        while True:
            tasks = [task for task in list(session._background_tasks) if not task.done()]
            if not tasks and not session._is_streaming:
                return
            await asyncio.sleep(0.01)


async def _dispose(recipients: list[Recipient]) -> None:
    for recipient in recipients:
        if recipient.session is not None:
            await recipient.session.dispose()


# --------------------------------------------------------------------------- #
# The hub stand-in: only the wire's verbs are doubled
# --------------------------------------------------------------------------- #


@dataclass
class Hub:
    """One channel message plus the §10.1 fan-out rules that decide its rows.

    ``events`` answers 204 (``None``) when this endpoint gets no row, which is
    what the real long-poll does for a member the server did not address. The
    message itself stays in the channel: a muted member can still read it.
    """

    notify: dict[str, str] = field(default_factory=dict)
    active: dict[str, bool] = field(default_factory=dict)
    mentioned: set[str] = field(default_factory=set)
    events_calls: list[str] = field(default_factory=list)
    body_reads: list[str] = field(default_factory=list)
    history_reads: list[str] = field(default_factory=list)
    admissions: list[dict[str, Any]] = field(default_factory=list)
    admitted: set[str] = field(default_factory=set)
    messages: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {
                "id": MESSAGE_ID,
                "conversation_id": CONVERSATION_ID,
                "seq": 1,
                "sender": dict(SENDER),
                "body": BODY,
                "body_hash": hashlib.sha256(BODY.encode("utf-8")).hexdigest(),
                "origin": "human",
                "created_at": HISTORY_AT,
            }
        ]
    )

    def row_for(self, endpoint: str) -> dict[str, Any] | None:
        """The server's verdict for this member — the half no local code owns."""

        self.events_calls.append(endpoint)
        if not self.active.get(endpoint, False):
            # A disabled endpoint is not addressed even for notify=all or a
            # mention: the server writes no feed row for it.
            return None
        mention = endpoint in self.mentioned
        if not mention and self.notify.get(endpoint) != "all":
            return None
        ids: dict[str, Any] = {
            "message_id": MESSAGE_ID,
            "conversation_id": CONVERSATION_ID,
            "conversation_seq": 1,
        }
        if mention:  # the server's mention flag is ``omitempty``: false is absent
            ids["mention"] = True
        return {"type": "message_available", "ids": ids, "at": HISTORY_AT}

    def events_answer(self, endpoint: str) -> dict[str, Any] | None:
        row = self.row_for(endpoint)
        if row is None:
            return None  # 204: the long poll parked and nothing was addressed
        return {"events": [row], "cursor": "1"}

    def history_page(self, endpoint: str, *, limit: int | None) -> dict[str, Any]:
        self.history_reads.append(endpoint)
        # A disabled endpoint has no read at all.
        if not self.active.get(endpoint, False):
            raise BridgeRefusal("endpoint_disabled", "endpoint is disabled", http_status=403)
        rows: list[dict[str, Any]] = []
        for row in sorted(self.messages, key=lambda row: row["seq"], reverse=True):
            served = dict(row)
            # Agent-member bodies are admitted per endpoint (client.py
            # ``conversation_messages``: "bodies appear only after admit"), so
            # an unadmitted read gets the identity and hash but no bytes. The
            # message itself is still listed: a mute costs the wake, not the row.
            if endpoint not in self.admitted:
                served["body"] = None
            rows.append(served)
        return {"messages": rows[: limit or 50], "has_more": False}


class HubWire:
    """The hub's HTTP verbs, per endpoint, with no socket in sight."""

    def __init__(self, hub: Hub, endpoint: str, base_url: str) -> None:
        self.hub = hub
        self.endpoint = endpoint
        self.base_url = base_url

    async def events(self, **kwargs: Any) -> dict[str, Any] | None:
        return self.hub.events_answer(self.endpoint)

    async def conversation_messages(self, conversation_id: str, **kwargs: Any) -> dict[str, Any]:
        assert conversation_id == CONVERSATION_ID
        return self.hub.history_page(self.endpoint, limit=kwargs.get("limit"))

    async def conversation_body(
        self, conversation_id: str, message_id: str, **kwargs: Any
    ) -> dict[str, Any]:
        assert (conversation_id, message_id) == (CONVERSATION_ID, MESSAGE_ID)
        self.hub.body_reads.append(self.endpoint)
        if not self.hub.active.get(self.endpoint, False):
            raise BridgeRefusal("endpoint_disabled", "endpoint is disabled", http_status=403)
        return {
            "message_id": MESSAGE_ID,
            "body": BODY,
            "body_hash": hashlib.sha256(BODY.encode("utf-8")).hexdigest(),
            "conversation_name": "#qa-coordination",
            "sender_label": SENDER["display_label"],
            "sender_kind": SENDER["kind"],
        }

    async def conversation_admit(
        self, conversation_id: str, message_id: str, *, verdict: str, body_hash: str, **kwargs: Any
    ) -> dict[str, Any]:
        self.hub.admissions.append(
            {"endpoint": self.endpoint, "message_id": message_id, "verdict": verdict}
        )
        self.hub.admitted.add(self.endpoint)
        return {"ok": True}


# --------------------------------------------------------------------------- #
# The wake seam: the runtime's own receive method, not a list append
# --------------------------------------------------------------------------- #


@dataclass
class WakeCall:
    """What the production seam HANDED OVER; the transport is always a mailbox wake.

    ``deliver_wake`` has no mode/wake parameters — every bridge card is a
    mailbox wake — so no mode field is recorded here: asserting one would be
    asserting this test's own default rather than a production decision. The
    wake is observed where it actually lands: the spawned session turn.
    """

    session_id: str
    payload: str
    sender_display: dict[str, str] | None
    bridge_meta: dict[str, Any] | None


class WakeRouter:
    """Stand in for ``deliver_wake``'s transport, keep its contract.

    Production hands the SAME fields to ``deliver_peer_message_outcome`` (or the
    engagement errand), whose live path calls ``Session.receive_peer_message``
    on the designated session. This routes to that method directly: no socket,
    no subprocess, but the real receive path — including the idle+wake turn.
    """

    def __init__(self) -> None:
        self.calls: list[WakeCall] = []
        self.sessions: dict[tuple[Path, str], Session] = {}

    def register(self, config_dir: Path, session_id: str, session: Session) -> None:
        self.sessions[(Path(config_dir), session_id)] = session

    async def __call__(
        self,
        owner: Any,
        payload: str,
        *,
        session_id: str,
        config_dir: Path,
        bridge_meta: dict[str, Any] | None = None,
        sender_display: dict[str, str] | None = None,
    ) -> str:
        self.calls.append(WakeCall(session_id, payload, sender_display, bridge_meta))
        session = self.sessions[(Path(config_dir), session_id)]
        sender: dict[str, Any] = {
            "account_id": owner.account_id,
            "source": TEAM_BRIDGE_MESSAGE_TYPE,
        }
        if sender_display:
            # The same enrichment deliver_wake performs, same normalization.
            for key, field_name in (("name", "sender_name"), ("organization", "organization")):
                value = " ".join(str(sender_display.get(field_name) or "").split())
                if value:
                    sender[key] = value
        detail = await session.receive_peer_message(
            payload,
            mode="mailbox",
            wake=True,
            sender=sender,
            custom_type=TEAM_BRIDGE_MESSAGE_TYPE,
            bridge_meta=bridge_meta,
        )
        await _settle(session)
        return str(detail)


# --------------------------------------------------------------------------- #
# Screening, modelled at its own store boundary (not the subject here)
# --------------------------------------------------------------------------- #


def _receipt(verdict: str, body: str = BODY) -> ScreeningReceipt:
    return ScreeningReceipt(
        verdict=verdict,
        model="qa-default",
        prompt_version="bridge-screen-v1",
        template_sha256="ab" * 32,
        body_sha256=hashlib.sha256(body.encode("utf-8")).hexdigest(),
        categories=(),
    ).validated()


def _persist_receipt(config_dir: Path, receipt: ScreeningReceipt, *, message_id: str) -> None:
    """Model what the real screener writes before it answers ``deliver``.

    The screening PIPELINE is exercised by its own suite; here the verdict is a
    fixture so the recipient states are not gated on a model call. The receipt
    still goes through the store's real table and the connector's real
    ``receipt_for_message`` check — a missing row correctly refuses delivery.
    """

    with sqlite3.connect(state_db_path(config_dir)) as conn:
        row = conn.execute("SELECT owner_key, enrollment_generation FROM enrollments").fetchone()
        assert row is not None
        owner_key, generation = row
        conn.execute(
            "INSERT OR REPLACE INTO screening_receipts (owner_key, message_id, "
            "body_sha256, template_sha256, prompt_version, model, verdict, categories, "
            "envelope_digest, admission_revision, policy_epoch, creator_generation, "
            "created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                owner_key,
                message_id,
                receipt.body_sha256,
                receipt.template_sha256,
                receipt.prompt_version,
                receipt.model,
                receipt.verdict,
                json.dumps(sorted(receipt.categories)),
                "e" * 64,
                "rev-1",
                1,
                generation,
                time.time(),
            ),
        )


def _install_seams(
    monkeypatch: pytest.MonkeyPatch,
    hub: Hub,
    recipients: list[Recipient],
    router: WakeRouter,
) -> None:
    """Double only the wire and the screener; everything else is production."""

    wires = {
        Path(recipient.config_dir): HubWire(
            hub, recipient.endpoint, recipient.owner.canonical_base_url
        )
        for recipient in recipients
    }

    def wire_for(config_dir: Path, client: Any = None) -> Any:
        return wires[Path(config_dir)]

    async def conversation_wire_for(owner: Any, config_dir: Path, auth_store: Any) -> Any:
        return wires[Path(config_dir)]

    async def fake_screen(owner: Any, item: Any, *, config_dir: Path, **kwargs: Any) -> Any:
        assert item["message"]["id"] == MESSAGE_ID
        assert item["message"]["body"]["text"] == BODY
        receipt = _receipt("allow")
        _persist_receipt(config_dir, receipt, message_id=MESSAGE_ID)
        return ScreenResult(disposition="deliver", receipt=receipt)

    monkeypatch.setattr(pull_mod, "_client_for", wire_for)
    monkeypatch.setattr(conversation_mod, "_wire_for", conversation_wire_for)
    monkeypatch.setattr(conversation_mod, "screen_item", fake_screen)
    monkeypatch.setattr(conversation_mod, "deliver_wake", router)


# --------------------------------------------------------------------------- #
# Observation helpers
# --------------------------------------------------------------------------- #


def _delivery_state(config_dir: Path) -> dict[str, list[dict[str, Any]]]:
    with sqlite3.connect(state_db_path(config_dir)) as conn:
        conn.row_factory = sqlite3.Row
        return {
            table: [dict(row) for row in conn.execute(f"SELECT * FROM {table}")]
            for table in DELIVERY_TABLES
        }


def _transcript_rows(recipient: Recipient) -> list[dict[str, Any]]:
    assert recipient.config_dir is not None
    path = recipient.config_dir / "sessions" / recipient.session_id / "transcript.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _card_requests(recipient: Recipient) -> list[str]:
    """The provider requests made AFTER the owner's own engagement turn."""

    return [
        "\n".join(_request_texts(request))
        for request in recipient.stream.requests[recipient.baseline_requests :]
    ]


async def _run(recipient: Recipient) -> dict[str, Any]:
    return await loop_mod.run_iteration(
        recipient.owner,
        config_dir=recipient.config_dir,
        access=_access(recipient.owner),
        auth_store=None,
    )


# --------------------------------------------------------------------------- #
# State 1 — mentioned: row → card → wake → a real turn
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_mentioned_member_is_woken_with_a_card_that_says_mentioned(
    tmp_path, monkeypatch
) -> None:
    recipient = Recipient("a", _owner("a", 1), tmp_path / "a", "sess_a")
    hub = Hub(
        notify={recipient.endpoint: "mute"},  # mention overrides mute
        active={recipient.endpoint: True},
        mentioned={recipient.endpoint},
    )
    router = WakeRouter()
    _install_seams(monkeypatch, hub, [recipient], router)
    await _make_recipient(recipient)
    router.register(recipient.config_dir, recipient.session_id, recipient.live_session)

    before = _delivery_state(recipient.config_dir)
    try:
        outcome = await _run(recipient)
    finally:
        await _settle(recipient.live_session)
        await _dispose([recipient])

    # The real handler delivered the real row, and says why in its own terms.
    assert outcome["handled"][0]["action"] == "delivered", outcome
    assert outcome["handled"][0]["mention"] is True
    assert outcome["handled"][0]["session"] == "sess_a"

    # THE CARD is the wake payload the recipient's agent received.
    assert len(router.calls) == 1
    call = router.calls[0]
    assert call.session_id == "sess_a"
    assert call.sender_display == {"sender_name": "Ada"}
    card = call.payload
    assert "Channel message · #qa-coordination · from Ada (human, same tenant)" in card
    assert BRIDGE_UNTRUSTED_CLAUSE in card
    assert "mention: yes" in card
    assert "conversation_id: bcv_abcdefgh" in card
    assert f"message_id: {MESSAGE_ID}" in card
    assert "guidance: bridge-guidance-v2" in card
    assert BODY not in card  # metadata only; the body is read separately

    # A REAL TURN started from the wake, and the model request carries the card
    # inside the bridge envelope — not as the owner's own words.
    requests = _card_requests(recipient)
    assert len(requests) == 1, "the wake must start exactly one turn"
    assert "mention: yes" in requests[0]
    assert "from Ada (human, same tenant)" in requests[0]
    assert "<team-bridge-message" in requests[0]
    assert BODY not in requests[0]

    # The row is durable in the recipient's own transcript.
    custom = [
        row
        for row in _transcript_rows(recipient)
        if row.get("payload", {}).get("custom_type") == TEAM_BRIDGE_MESSAGE_TYPE
    ]
    assert len(custom) == 1
    assert "mention: yes" in custom[0]["payload"]["details"]["text"]

    # The pending body reservation names the mention the hub reported.
    after = _delivery_state(recipient.config_dir)
    reservations = after["body_reservations"]
    assert len(reservations) == 1
    assert reservations[0]["mention"] == 1
    assert reservations[0]["message_id"] == MESSAGE_ID
    assert before["feed_refs"] == [] and after["feed_refs"] == []  # consumed once


# --------------------------------------------------------------------------- #
# State 2 — unmentioned: the row decides, and only notify=all produces one
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_unmentioned_member_is_woken_exactly_when_the_setting_emits_a_row(
    tmp_path, monkeypatch
) -> None:
    """Both local arms of the §10.1 decision, in one send.

    PRECISELY WHAT IS ASSERTED: a member whose server-side policy is
    ``notify=all`` and who is not mentioned receives one row, one card saying
    ``mention: no``, one wake and one turn; a member whose policy is
    ``notify=mention`` and who is not mentioned receives no row, so the local
    path produces no card, no wake and no turn. The translation from policy to
    row is the HUB's (the stand-in implements §10.1); the local receiver has no
    policy input, so it cannot and does not second-guess the row it is given.
    """

    all_member = Recipient("all", _owner("all", 1), tmp_path / "all", "sess_all")
    mention_policy = Recipient(
        "mentioned_only", _owner("mentioned_only", 2), tmp_path / "mo", "sess_mo"
    )
    hub = Hub(
        notify={all_member.endpoint: "all", mention_policy.endpoint: "mention"},
        active={all_member.endpoint: True, mention_policy.endpoint: True},
        mentioned=set(),
    )
    router = WakeRouter()
    _install_seams(monkeypatch, hub, [all_member, mention_policy], router)
    await _make_recipient(all_member)
    await _make_recipient(mention_policy)
    router.register(all_member.config_dir, all_member.session_id, all_member.live_session)
    router.register(
        mention_policy.config_dir, mention_policy.session_id, mention_policy.live_session
    )

    before = _delivery_state(mention_policy.config_dir)
    try:
        all_outcome = await _run(all_member)
        silent_outcome = await _run(mention_policy)
    finally:
        await _settle(all_member.live_session)
        await _settle(mention_policy.live_session)
        await _dispose([all_member, mention_policy])

    # notify=all, unmentioned: woken, with a card that says so.
    assert all_outcome["handled"][0]["action"] == "delivered", all_outcome
    assert all_outcome["handled"][0]["mention"] is False
    assert len(router.calls) == 1
    card = router.calls[0].payload
    assert "mention: no" in card
    assert "mention: yes" not in card
    # No reply obligation rides the card; reply-vs-quiet is the served
    # guidance's call, not this delivery's. The one reply mention allowed is
    # the conditional ROUTE line (where an answer goes if one is sent), so
    # an answer stays in the conversation instead of the legacy pair inbox.
    reply_lines = [line for line in card.splitlines() if "reply" in line.lower()]
    assert "no_reply" not in card
    assert reply_lines == [
        "if you reply: bridge channel_post with channel_id = this conversation_id; "
        "legacy send/reply never appears in a conversation"
    ]
    assert "guidance: bridge-guidance-v2" in card
    requests = _card_requests(all_member)
    assert len(requests) == 1 and "mention: no" in requests[0]

    # notify=mention, unmentioned: the hub's silence produces NO local work.
    assert silent_outcome["handled"] == []
    assert len(router.calls) == 1, "no second wake"
    assert _card_requests(mention_policy) == []
    assert _delivery_state(mention_policy.config_dir) == before
    assert mention_policy.endpoint in hub.events_calls


# --------------------------------------------------------------------------- #
# State 3 — muted: no wake, no work, and the message is not lost
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_muted_member_is_not_woken_and_the_message_survives_for_a_later_read(
    tmp_path, monkeypatch
) -> None:
    recipient = Recipient("muted", _owner("muted", 1), tmp_path / "muted", "sess_m")
    hub = Hub(
        notify={recipient.endpoint: "mute"},
        active={recipient.endpoint: True},
        mentioned=set(),
    )
    router = WakeRouter()
    _install_seams(monkeypatch, hub, [recipient], router)
    await _make_recipient(recipient)
    router.register(recipient.config_dir, recipient.session_id, recipient.live_session)

    before = _delivery_state(recipient.config_dir)
    baseline_rows = len(_transcript_rows(recipient))
    try:
        outcome = await _run(recipient)
        await _settle(recipient.live_session)
        # Read the transcript BEFORE dispose: teardown appends its own
        # frontend checkpoint, and that row is bookkeeping, not delivery.
        rows_after = _transcript_rows(recipient)
        state_after = _delivery_state(recipient.config_dir)
    finally:
        await _dispose([recipient])

    # NEGATIVE SHAPE, each one a way a mute bug could show up: no handler
    # outcome, no card/wake, no turn, no transcript row, no delivery state.
    assert outcome["handled"] == []
    assert outcome["events"] == 0 and outcome["reissued"] is True  # a 204, not an empty 200
    assert router.calls == []
    assert _card_requests(recipient) == []
    assert len(recipient.stream.requests) == recipient.baseline_requests
    assert len(rows_after) == baseline_rows
    assert state_after == before
    assert hub.events_calls == [recipient.endpoint], "the connector did poll and got silence"
    assert hub.body_reads == [] and hub.admissions == []

    # NOT LOST: the channel still lists the message for this member's read,
    # with its identity and hash intact. The BODY is admitted per endpoint on
    # the hub, so an unadmitted agent read carries no bytes — the later
    # take_up/admit is a separate action, deliberately not exercised here.
    page = await HubWire(
        hub, recipient.endpoint, recipient.owner.canonical_base_url
    ).conversation_messages(CONVERSATION_ID, tenant_id=TENANT, limit=5)
    served = {row["id"]: row for row in page["messages"]}
    assert served[MESSAGE_ID]["seq"] == 1
    assert served[MESSAGE_ID]["sender"] == SENDER
    assert served[MESSAGE_ID]["body"] is None
    assert served[MESSAGE_ID]["body_hash"] == hashlib.sha256(BODY.encode("utf-8")).hexdigest()
    assert hub.messages[0]["body"] == BODY, "the hub kept the bytes; nothing consumed them"
    assert hub.history_reads == [recipient.endpoint]


# --------------------------------------------------------------------------- #
# State 4 — endpoint disabled: nothing at all
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_disabled_endpoint_receives_nothing_at_all(tmp_path, monkeypatch) -> None:
    recipient = Recipient("disabled", _owner("disabled", 1), tmp_path / "disabled", "sess_d")
    hub = Hub(
        notify={recipient.endpoint: "all"},
        active={recipient.endpoint: False},
        # A mention must not override a disabled endpoint either.
        mentioned={recipient.endpoint},
    )
    router = WakeRouter()
    _install_seams(monkeypatch, hub, [recipient], router)
    await _make_recipient(recipient)
    router.register(recipient.config_dir, recipient.session_id, recipient.live_session)

    before = _delivery_state(recipient.config_dir)
    baseline_rows = len(_transcript_rows(recipient))
    baseline_requests = len(recipient.stream.requests)
    try:
        outcome = await _run(recipient)
        await _settle(recipient.live_session)
        rows_after = _transcript_rows(recipient)
        state_after = _delivery_state(recipient.config_dir)
    finally:
        await _dispose([recipient])

    # The server addressed no row; the local half did NOTHING with the silence.
    assert outcome["handled"] == []
    assert outcome["events"] == 0 and outcome["reissued"] is True
    assert router.calls == []
    assert _card_requests(recipient) == []
    assert len(recipient.stream.requests) == baseline_requests
    assert len(rows_after) == baseline_rows
    assert state_after == before
    assert hub.events_calls == [recipient.endpoint]
    # Not even a body/history/admit attempt was made for the disabled member.
    assert hub.body_reads == []
    assert hub.history_reads == []
    assert hub.admissions == []

    # And the hub itself refuses this endpoint's read, so a later read cannot
    # quietly become the delivery the disabled endpoint was denied.
    wire = HubWire(hub, recipient.endpoint, recipient.owner.canonical_base_url)
    with pytest.raises(BridgeRefusal) as refusal:
        await wire.conversation_messages(CONVERSATION_ID, tenant_id=TENANT, limit=5)
    assert refusal.value.http_status == 403
