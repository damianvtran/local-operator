"""Conversation delivery to agent members, end to end at the local seams.

Two agent runtimes (A and B) each own an endpoint, a real native store and a
fake wire. ONE channel message mentions A only. The fake server applies the
documented fan-out rules — mention overrides mute; ``notify=all`` opts in;
a disabled endpoint gets nothing — so the test proves both halves that can be
proven locally: the routing/card/admission/wake path, and that a row the
server would not send produces no card.

The pair inbox reader is spied on and must never see a conversation row.
Both runtimes' fake transcripts are PRINTED at the end (run with ``-s``).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.harness.message_types import BRIDGE_UNTRUSTED_CLAUSE
from local_operator.team_bridge import pull as pull_mod
from local_operator.team_bridge.connector import conversation as conversation_mod
from local_operator.team_bridge.connector import loop as loop_mod
from local_operator.team_bridge.connector.conversation import (
    ConversationCardMetadata,
    build_conversation_card,
)
from local_operator.team_bridge.events import plan_feed_refs
from local_operator.team_bridge.screening import ScreenResult
from local_operator.team_bridge.store import OwnerReceipt, Store
from local_operator.team_bridge.types import OwnerSnapshot, project_events_batch

CONVERSATION_ID = "bcv_abcdefgh"
MESSAGE_ID = "bcmsg_abcdef01"
SENDER = {"account_id": "acc_human", "kind": "human", "display_label": "Ada"}
BODY = "hello from the channel"
HISTORY_AT = "2026-10-20T00:00:00Z"

EXPECTED_CARD_A = f"""Channel message · #qa-coordination · from Ada (human, same tenant)
{BRIDGE_UNTRUSTED_CLAUSE}

mention: yes
message_id: bcmsg_abcdef01
conversation_seq: 1
guidance: bridge-guidance-v2"""


def _owner(name: str, credential_id: int) -> OwnerSnapshot:
    return OwnerSnapshot(
        credential_id=credential_id,
        account_id=f"acc_{name}",
        tenant_id="tnt_1",
        endpoint_id=f"bep_{name}",
        canonical_base_url="https://api.radienthq.com/v1",
        enrollment_generation=1,
    )


def _enroll(config_dir: Path, owner: OwnerSnapshot, session: str) -> Store:
    store = Store(config_dir)
    store.commit_enrollment(
        owner, OwnerReceipt("bridge_consent_v1", f"ref-{session}"), cos_session_id=session
    )
    # S-8a: the fence reads this DISK shape on every boundary.
    (config_dir / "config.yml").write_text(
        json.dumps({"values": {"team_bridge": {"enabled": True}}}), encoding="utf-8"
    )
    return store


def _receipt(verdict: str) -> Any:
    from local_operator.team_bridge.types import ScreeningReceipt

    return ScreeningReceipt(
        verdict=verdict,
        model="qa-default",
        prompt_version="bridge-screen-v1",
        template_sha256="ab" * 32,
        body_sha256=hashlib.sha256(BODY.encode("utf-8")).hexdigest(),
        categories=(),
    ).validated()


@dataclass
class FakeServer:
    """One channel message and the documented per-member delivery rules.

    ``page`` models the server's NEWEST-FIRST list: it filters by the cursors,
    sorts descending and takes ``limit``. That is what makes the target-window
    assertion meaningful (an ``after_seq`` window can hide the target behind
    newer rows; ``before_seq=seq+1, limit=1`` cannot).
    """

    notify: dict[str, str] = field(default_factory=dict)
    active: dict[str, bool] = field(default_factory=dict)
    mentioned: set[str] = field(default_factory=set)
    body: str = BODY
    body_override: str | None = None
    human_member: bool = True
    metadata: bool = True
    conversation_name: str = "#qa-coordination"
    sender: dict[str, Any] = field(default_factory=lambda: dict(SENDER))
    newer_messages: int = 0
    hide_target: bool = False
    include_history_body: bool = False
    body_route_available: bool = True
    view_calls: list[str] = field(default_factory=list)
    history_calls: list[dict[str, Any]] = field(default_factory=list)
    admits: list[dict[str, Any]] = field(default_factory=list)

    def event_for(self, endpoint: str) -> dict[str, Any] | None:
        if not self.active.get(endpoint, False):
            return None
        mention = endpoint in self.mentioned
        if not mention and self.notify.get(endpoint) != "all":
            return None
        ids: dict[str, Any] = {
            "message_id": MESSAGE_ID,
            "conversation_id": CONVERSATION_ID,
            "conversation_seq": 1,
        }
        if mention:  # the server's bool is ``omitempty``: false is ABSENT
            ids["mention"] = True
        if self.metadata:
            ids.update(
                {
                    "conversation_name": self.conversation_name,
                    "sender_label": self.sender["display_label"],
                    "sender_kind": self.sender["kind"],
                }
            )
        return {"type": "message_available", "ids": ids, "at": HISTORY_AT}

    def events_answer(self, endpoint: str) -> dict[str, Any]:
        event = self.event_for(endpoint)
        return {"events": [event] if event else [], "cursor": "1"}

    def view(self, endpoint: str) -> dict[str, Any]:
        self.view_calls.append(endpoint)
        if not self.human_member:
            raise conversation_mod.BridgeRefusal(
                "not_a_member", "person-gated view refused", http_status=403
            )
        return {
            "id": CONVERSATION_ID,
            "kind": "channel",
            "name": self.conversation_name,
            "members": [],
        }

    def _messages(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if not self.hide_target:
            body = self.body if self.include_history_body else None
            rows.append(
                {
                    "id": MESSAGE_ID,
                    "conversation_id": CONVERSATION_ID,
                    "seq": 1,
                    "sender": dict(self.sender),
                    "body": body,
                    "body_hash": hashlib.sha256(self.body.encode("utf-8")).hexdigest(),
                    "origin": "human",
                    "created_at": HISTORY_AT,
                }
            )
        for index in range(self.newer_messages):
            rows.append(
                {
                    "id": f"bcmsg_newer{index:02d}",
                    "conversation_id": CONVERSATION_ID,
                    "seq": index + 2,
                    "sender": dict(self.sender),
                    "body": None,
                    "body_hash": "",
                    "origin": "human",
                    "created_at": HISTORY_AT,
                }
            )
        return rows

    def page(
        self,
        *,
        before_seq: int | None,
        after_seq: int | None,
        limit: int | None,
    ) -> dict[str, Any]:
        self.history_calls.append(
            {"before_seq": before_seq, "after_seq": after_seq, "limit": limit}
        )
        rows = [
            row
            for row in self._messages()
            if (before_seq is None or row["seq"] < before_seq)
            and (after_seq is None or row["seq"] >= after_seq)
        ]
        rows.sort(key=lambda row: row["seq"], reverse=True)
        return {"messages": rows[: limit or 50], "has_more": False}

    def body_response(self) -> dict[str, Any]:
        return {
            "message_id": MESSAGE_ID,
            "body": self.body_override if self.body_override is not None else self.body,
            "body_hash": hashlib.sha256(self.body.encode("utf-8")).hexdigest(),
        }


class FakeWire:
    def __init__(self, server: FakeServer, endpoint: str) -> None:
        self.server = server
        self.endpoint = endpoint

    async def conversation(self, conversation_id: str, **kwargs: Any) -> dict[str, Any]:
        assert conversation_id == CONVERSATION_ID
        return self.server.view(self.endpoint)

    async def conversation_messages(self, conversation_id: str, **kwargs: Any) -> dict[str, Any]:
        assert conversation_id == CONVERSATION_ID
        return self.server.page(
            before_seq=kwargs.get("before_seq"),
            after_seq=kwargs.get("after_seq"),
            limit=kwargs.get("limit"),
        )

    async def conversation_body(self, conversation_id: str, message_id: str, **kwargs: Any):
        assert (conversation_id, message_id) == (CONVERSATION_ID, MESSAGE_ID)
        if not self.server.body_route_available:
            raise conversation_mod.BridgeRefusal(
                "not_found", "classification route unavailable", http_status=404
            )
        return self.server.body_response()

    async def conversation_admit(
        self, conversation_id: str, message_id: str, *, verdict: str, body_hash: str, **kwargs: Any
    ) -> dict[str, Any]:
        self.server.admits.append(
            {"endpoint": self.endpoint, "message_id": message_id, "verdict": verdict}
        )
        return {"ok": True}


def _wire_rig(
    monkeypatch: pytest.MonkeyPatch,
    server: FakeServer,
    runtimes: dict[str, tuple[OwnerSnapshot, Path]],
    *,
    disposition: str = "deliver",
    screen_calls: list[str] | None = None,
    notifications: list[str] | None = None,
    pair_calls: list[str] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[str]]]:
    """Fake every seam; return per-session transcripts and delivery modes."""

    transcripts: dict[str, list[dict[str, Any]]] = {
        session: [] for _, session in ((o, s) for o, _, s in _runtime_rows(runtimes))
    }
    modes: dict[str, list[str]] = {session: [] for session in transcripts}
    screen_calls = screen_calls if screen_calls is not None else []
    notifications = notifications if notifications is not None else []
    pair_calls = pair_calls if pair_calls is not None else []

    refs_by_endpoint: dict[str, list[str]] = {}
    for name, (owner, _config_dir) in runtimes.items():
        _cursor, refs, _resync = project_events_batch(server.events_answer(owner.endpoint_id))
        refs_by_endpoint[owner.endpoint_id] = list(refs)

    async def fake_events(
        owner: Any, *, config_dir: Path, access: Any, after: Any = None, **kw: Any
    ):
        return {"cursor": 1, "refs": list(refs_by_endpoint[owner.endpoint_id]), "resync": False}

    def fake_drain(owner: Any, *, config_dir: Path, upto_seq: int) -> list[str]:
        return list(refs_by_endpoint[owner.endpoint_id])

    async def fake_wire_for(owner: Any, config_dir: Path, auth_store: Any) -> FakeWire:
        return FakeWire(server, owner.endpoint_id)

    async def fake_screen(owner: Any, item: Any, *, config_dir: Path, **kwargs: Any):
        screen_calls.append(owner.endpoint_id)
        if disposition == "hold":
            return ScreenResult(disposition="hold", hold_code="screening_hold")
        if disposition == "bounce":
            return ScreenResult(disposition="bounce", receipt=_receipt("reject"))
        return ScreenResult(disposition="deliver", receipt=_receipt("allow"))

    async def fake_deliver(
        owner: Any,
        payload: str,
        *,
        session_id: str,
        config_dir: Path,
        bridge_meta: Any = None,
        sender_display: Any = None,
        mode: str = "mailbox",
    ) -> str:
        transcripts.setdefault(session_id, []).append(
            {"payload": payload, "mode": mode, "sender_display": sender_display}
        )
        modes.setdefault(session_id, []).append(mode)
        return "delivered and woke the session"

    async def pair_spy(*args: Any, **kwargs: Any):
        pair_calls.append(str(args[1] if len(args) > 1 else "?"))
        raise AssertionError("a conversation row reached the pair inbox reader")

    monkeypatch.setattr(pull_mod, "events", fake_events)
    monkeypatch.setattr(pull_mod, "drain_feed_refs", fake_drain)
    monkeypatch.setattr(conversation_mod, "_wire_for", fake_wire_for)
    monkeypatch.setattr(conversation_mod, "screen_item", fake_screen)
    monkeypatch.setattr(conversation_mod, "deliver_wake", fake_deliver)
    # The rig's sessions are ENGAGED; F8's unused-session gate is tested apart.
    monkeypatch.setattr(conversation_mod, "engagement_hold", lambda designation, config_dir: "")
    monkeypatch.setattr(loop_mod, "handle_message_available", pair_spy)
    monkeypatch.setattr(
        "local_operator.tui.notify.detached_notify",
        lambda *a, **k: notifications.append("detached_notify"),
    )
    monkeypatch.setattr(
        "local_operator.tui.notify._spawn_detached",
        lambda *a, **k: notifications.append("_spawn_detached"),
    )
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: subprocess.CompletedProcess([], 0))
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess([], 0))
    return transcripts, modes


def _runtime_rows(runtimes: dict[str, tuple[OwnerSnapshot, Path]]):
    for name, (owner, config_dir) in runtimes.items():
        session = f"sess_{name}"
        yield owner, config_dir, session


def _run(owner: OwnerSnapshot, config_dir: Path) -> dict[str, Any]:
    return asyncio.run(
        loop_mod.run_iteration(
            owner,
            config_dir=config_dir,
            access=SimpleNamespace(credential_id=owner.credential_id),
            auth_store=None,
        )
    )


def test_card_renders_the_shipped_untrusted_clause_verbatim() -> None:
    """ONE home for the clause: the shipped constant, never a re-worded copy."""

    card = build_conversation_card(
        ConversationCardMetadata(
            conversation_label="#qa-coordination",
            sender_label="Ada",
            sender_kind="human",
            mention=True,
            message_id=MESSAGE_ID,
            conversation_seq=1,
        )
    )
    assert card.splitlines()[1] == BRIDGE_UNTRUSTED_CLAUSE
    # Triage rules stay in the served guidance named by version — the card
    # must not grow a second copy of the §11.2 paragraph.
    assert "§11.2" not in card and "course-correct" not in card
    assert "guidance: bridge-guidance-v2" in card


def test_card_sender_kind_is_dynamic() -> None:
    """The kind comes from the server's sender block, never a literal."""

    card = build_conversation_card(
        ConversationCardMetadata(
            conversation_label="#qa-coordination",
            sender_label="Robo",
            sender_kind="agent",
            mention=False,
            message_id=MESSAGE_ID,
            conversation_seq=1,
        )
    )
    assert "from Robo (agent, same tenant)" in card
    assert "(human, same tenant)" not in card


def _ref(server: FakeServer, endpoint: str = "bep_a") -> Any:
    """The conversation ref the connector would drain for this endpoint."""

    _cursor, refs, _resync = project_events_batch(server.events_answer(endpoint))
    return plan_feed_refs(refs).conversation_deliveries[0]


def test_card_header_collapses_crlf_in_labels() -> None:
    """Server-resolved labels are not newline-free; the header must be one line."""

    card = build_conversation_card(
        ConversationCardMetadata(
            conversation_label="#qa\ncoordination\r",
            sender_label="Ada\r\nEve",
            sender_kind="human",
            mention=True,
            message_id=MESSAGE_ID,
            conversation_seq=1,
        )
    )
    assert card.splitlines()[0] == (
        "Channel message · #qa coordination · from Ada Eve (human, same tenant)"
    )
    assert "\r" not in card


def test_one_channel_message_wakes_mentioned_a_and_notify_all_b(tmp_path, monkeypatch) -> None:
    a = _owner("a", 1)
    b = _owner("b", 2)
    dirs = {"a": tmp_path / "a", "b": tmp_path / "b"}
    for name, owner in (("a", a), ("b", b)):
        dirs[name].mkdir()
        _enroll(dirs[name], owner, f"sess_{name}")

    server = FakeServer(
        notify={"bep_a": "mute", "bep_b": "all"},  # A is muted; the mention overrides it
        active={"bep_a": True, "bep_b": True},
        mentioned={"bep_a"},
    )
    runtimes = {"a": (a, dirs["a"]), "b": (b, dirs["b"])}
    notifications: list[str] = []
    pair_calls: list[str] = []
    transcripts, modes = _wire_rig(
        monkeypatch,
        server,
        runtimes,
        notifications=notifications,
        pair_calls=pair_calls,
    )

    out_a = _run(a, dirs["a"])
    out_b = _run(b, dirs["b"])

    # (a) A: mentioned (overriding mute) -> card says so and the owning session
    # is wake/steered.
    assert out_a["handled"][0]["action"] == "delivered"
    assert out_a["handled"][0]["mention"] is True
    assert out_a["handled"][0]["session"] == "sess_a"
    assert transcripts["sess_a"][0]["payload"] == EXPECTED_CARD_A
    assert "hello from the channel" not in transcripts["sess_a"][0]["payload"]
    assert modes["sess_a"] == ["mailbox"]

    # (b) B: notify=all, mention=false -> card, no reply obligation, same wake.
    assert out_b["handled"][0]["action"] == "delivered"
    assert out_b["handled"][0]["mention"] is False
    assert out_b["handled"][0]["session"] == "sess_b"
    card_b = transcripts["sess_b"][0]["payload"]
    assert "mention: no" in card_b
    # The card is metadata-only for BOTH runtimes: the body reaches an agent
    # through its own conversation read after admission, never through the card.
    assert "hello from the channel" not in card_b
    assert modes["sess_b"] == ["mailbox"]

    # The body route is addressed by message id: neither the person-gated view
    # nor a history window is consulted for the normal path.
    assert server.view_calls == []
    assert server.history_calls == []
    # Admission is per endpoint and identical bytes.
    assert [(row["endpoint"], row["verdict"]) for row in server.admits] == [
        ("bep_a", "allow"),
        ("bep_b", "allow"),
    ]
    # Neither row was treated as a pair message.
    assert pair_calls == []
    # (f) No OS notification by default.
    assert notifications == []

    # (g) Triage copy is NOT duplicated in the card; it lives in the served
    # guidance named by version. Mention state alone distinguishes the cases.
    assert "no_reply" not in card_b
    assert "guidance: bridge-guidance-v2" in card_b
    assert "no_reply" not in transcripts["sess_a"][0]["payload"]

    for session in ("sess_a", "sess_b"):
        print(f"\n--- {session} transcript ---")
        for entry in transcripts[session]:
            print(entry["payload"])


def test_mute_member_gets_no_row_and_disabled_endpoint_gets_no_row(tmp_path, monkeypatch) -> None:
    b = _owner("b", 2)
    config_dir = tmp_path / "b"
    config_dir.mkdir()
    _enroll(config_dir, b, "sess_b")

    # (c) mute (and not mentioned): the server writes no feed row.
    server = FakeServer(notify={"bep_b": "mute"}, active={"bep_b": True}, mentioned=set())
    transcripts, _modes = _wire_rig(monkeypatch, server, {"b": (b, config_dir)})
    out = _run(b, config_dir)
    assert out["handled"] == [] and transcripts["sess_b"] == []

    # (d) endpoint disabled: no row even for notify=all, and a mention does not
    # override the disabled endpoint either.
    server.notify["bep_b"] = "all"
    server.active["bep_b"] = False
    transcripts, _modes = _wire_rig(monkeypatch, server, {"b": (b, config_dir)})
    out = _run(b, config_dir)
    assert out["handled"] == [] and transcripts["sess_b"] == []

    server.mentioned = {"bep_b"}
    transcripts, _modes = _wire_rig(monkeypatch, server, {"b": (b, config_dir)})
    out = _run(b, config_dir)
    assert out["handled"] == [] and transcripts["sess_b"] == []


def test_no_designation_is_badge_only_and_wakes_nothing(tmp_path, monkeypatch) -> None:
    """Consent without a designated session: no card, no wake (the mailbox target)."""

    a = _owner("a", 1)
    config_dir = tmp_path / "a"
    config_dir.mkdir()
    store = Store(config_dir)
    store.commit_enrollment(a, OwnerReceipt("bridge_consent_v1", "ref-a"), cos_session_id=None)
    (config_dir / "config.yml").write_text(
        json.dumps({"values": {"team_bridge": {"enabled": True}}}), encoding="utf-8"
    )

    server = FakeServer(notify={"bep_a": "mute"}, active={"bep_a": True}, mentioned={"bep_a"})
    transcripts, _modes = _wire_rig(monkeypatch, server, {"a": (a, config_dir)})
    out = _run(a, config_dir)

    assert out["handled"][0]["action"] == "badge_only"
    assert transcripts == {"sess_a": []}


def test_agent_only_membership_never_calls_the_person_view(tmp_path, monkeypatch) -> None:
    """BLOCKER proof: the feed labels suffice; the human-gated view is not read."""

    a = _owner("a", 1)
    config_dir = tmp_path / "a"
    config_dir.mkdir()
    _enroll(config_dir, a, "sess_a")

    server = FakeServer(
        notify={"bep_a": "mute"},
        active={"bep_a": True},
        mentioned={"bep_a"},
        human_member=False,  # an account with only an agent membership
    )
    transcripts, _modes = _wire_rig(monkeypatch, server, {"a": (a, config_dir)})
    out = _run(a, config_dir)

    assert out["handled"][0]["action"] == "delivered"
    assert server.view_calls == []
    assert transcripts["sess_a"][0]["payload"] == EXPECTED_CARD_A


def test_body_route_does_not_depend_on_a_newest_window(tmp_path, monkeypatch) -> None:
    """MAJOR proof: 100 newer messages cannot hide the target body."""

    a = _owner("a", 1)
    config_dir = tmp_path / "a"
    config_dir.mkdir()
    _enroll(config_dir, a, "sess_a")

    server = FakeServer(
        notify={"bep_a": "mute"},
        active={"bep_a": True},
        mentioned={"bep_a"},
        newer_messages=100,
    )
    transcripts, _modes = _wire_rig(monkeypatch, server, {"a": (a, config_dir)})
    out = _run(a, config_dir)

    assert out["handled"][0]["action"] == "delivered"
    assert server.history_calls == []
    assert [row["verdict"] for row in server.admits] == ["allow"]
    assert transcripts["sess_a"][0]["payload"] == EXPECTED_CARD_A


def test_body_route_unavailable_falls_back_to_the_target_window(tmp_path, monkeypatch) -> None:
    """A legacy server without the body route still gets the target, not a 404."""

    a = _owner("a", 1)
    config_dir = tmp_path / "a"
    config_dir.mkdir()
    _enroll(config_dir, a, "sess_a")

    server = FakeServer(
        notify={"bep_a": "mute"},
        active={"bep_a": True},
        mentioned={"bep_a"},
        newer_messages=100,
        body_route_available=False,
        include_history_body=True,
    )
    transcripts, _modes = _wire_rig(monkeypatch, server, {"a": (a, config_dir)})
    out = _run(a, config_dir)

    assert out["handled"][0]["action"] == "delivered"
    assert server.history_calls == [{"before_seq": 2, "after_seq": None, "limit": 1}]
    assert [row["verdict"] for row in server.admits] == ["allow"]
    assert transcripts["sess_a"][0]["payload"] == EXPECTED_CARD_A


def test_absent_target_keeps_the_hold(tmp_path, monkeypatch) -> None:
    """A missing target is retryable, not terminal: the hold row survives."""

    a = _owner("a", 1)
    config_dir = tmp_path / "a"
    config_dir.mkdir()
    store = _enroll(config_dir, a, "sess_a")

    server = FakeServer(
        notify={"bep_a": "mute"},
        active={"bep_a": True},
        mentioned={"bep_a"},
        body_route_available=False,
        hide_target=True,
    )
    transcripts, _modes = _wire_rig(monkeypatch, server, {"a": (a, config_dir)})
    out = _run(a, config_dir)

    assert out["handled"][0]["action"] == "hold"
    assert out["handled"][0]["code"] == "body_unavailable"
    assert transcripts["sess_a"] == []
    assert store.screening_hold(a, _ref(server).encode()) is not None


def test_hold_retry_survives_restart(tmp_path, monkeypatch) -> None:
    """A persisted conversation hold is re-driven by its encoded ref."""

    a = _owner("a", 1)
    config_dir = tmp_path / "a"
    config_dir.mkdir()
    store = _enroll(config_dir, a, "sess_a")

    server = FakeServer(notify={"bep_a": "mute"}, active={"bep_a": True}, mentioned={"bep_a"})
    transcripts, _modes = _wire_rig(monkeypatch, server, {"a": (a, config_dir)})
    ref = _ref(server)
    store.persist_hold(a, ref.encode(), "screening_hold", retry_at=0)
    loop_mod._RETRY_ATTEMPTS.clear()  # a fresh process has no in-memory counter

    out = asyncio.run(
        loop_mod.retry_due_holds(
            a,
            config_dir=config_dir,
            access=SimpleNamespace(credential_id=a.credential_id),
            auth_store=None,
        )
    )

    assert out["outcomes"][0]["action"] == "delivered"
    assert transcripts["sess_a"][0]["payload"] == EXPECTED_CARD_A
    assert store.screening_hold(a, ref.encode()) is None
    assert [row["verdict"] for row in server.admits] == ["allow"]


def test_denied_body_stays_metadata_only(tmp_path, monkeypatch) -> None:
    a = _owner("a", 1)
    config_dir = tmp_path / "a"
    config_dir.mkdir()
    _enroll(config_dir, a, "sess_a")

    server = FakeServer(notify={"bep_a": "mute"}, active={"bep_a": True}, mentioned={"bep_a"})
    transcripts, _modes = _wire_rig(
        monkeypatch, server, {"a": (a, config_dir)}, disposition="bounce"
    )
    out = _run(a, config_dir)

    assert out["handled"][0]["action"] == "delivered_unadmitted"
    assert out["handled"][0]["code"] == "screening_bounce"
    card = transcripts["sess_a"][0]["payload"]
    assert "hello from the channel" not in card
    assert card == EXPECTED_CARD_A
    assert [row["verdict"] for row in server.admits] == ["reject"]


def test_mutated_body_is_not_screened_admitted_or_rendered(tmp_path, monkeypatch) -> None:
    a = _owner("a", 1)
    config_dir = tmp_path / "a"
    config_dir.mkdir()
    _enroll(config_dir, a, "sess_a")

    server = FakeServer(
        notify={"bep_a": "mute"},
        active={"bep_a": True},
        mentioned={"bep_a"},
        body_override="tampered bytes",
    )
    screen_calls: list[str] = []
    transcripts, _modes = _wire_rig(
        monkeypatch, server, {"a": (a, config_dir)}, screen_calls=screen_calls
    )
    out = _run(a, config_dir)

    assert out["handled"][0]["action"] == "delivered_unadmitted"
    assert out["handled"][0]["code"] == "body_hash_mismatch"
    assert screen_calls == [] and server.admits == []
    card = transcripts["sess_a"][0]["payload"]
    assert "tampered bytes" not in card
    assert card == EXPECTED_CARD_A


def test_conversation_event_is_never_routed_as_a_pair_delivery() -> None:
    """The pure router: a conversation subject is its own lane, not message_id."""

    _cursor, refs, resync = project_events_batch(
        {
            "events": [
                {
                    "type": "message_available",
                    "ids": {
                        "message_id": MESSAGE_ID,
                        "conversation_id": CONVERSATION_ID,
                        "conversation_seq": 1,
                        "mention": True,
                    },
                    "at": HISTORY_AT,
                }
            ],
            "cursor": "9",
        }
    )
    assert resync is False and len(refs) == 1
    plan = plan_feed_refs(refs)
    assert plan.deliveries == ()
    assert len(plan.conversation_deliveries) == 1
    ref = plan.conversation_deliveries[0]
    assert (ref.conversation_id, ref.message_id, ref.conversation_seq, ref.mention) == (
        CONVERSATION_ID,
        MESSAGE_ID,
        1,
        True,
    )
    # Round-trip: the durable ref spelling parses back to the same record.
    assert plan_feed_refs([ref.encode()]).conversation_deliveries == (ref,)


# -- thread replies (agent-server feat/bridge-threads) -------------------------

ROOT_ID = "bcmsg_root0001"


def test_a_reply_card_names_its_thread_and_how_to_answer_in_it() -> None:
    """A reply's card says it is a thread reply and gives the root id, so the
    agent answers IN the thread; the untrusted clause stays line 2."""

    card = build_conversation_card(
        ConversationCardMetadata(
            conversation_label="#agents",
            sender_label="Bea's agent",
            sender_kind="agent",
            mention=False,
            message_id=MESSAGE_ID,
            conversation_seq=7,
            thread_root_id=ROOT_ID,
        )
    )
    lines = card.splitlines()
    assert lines[0].startswith("Thread reply · #agents · from Bea's agent")
    assert lines[1] == BRIDGE_UNTRUSTED_CLAUSE
    assert f"thread_root_id: {ROOT_ID}" in lines
    assert "channel_post with this thread_root_id" in card
    assert lines[-1] == "guidance: bridge-guidance-v2"


def test_a_channel_level_card_carries_no_thread_lines() -> None:
    card = build_conversation_card(
        ConversationCardMetadata(
            conversation_label="#agents",
            sender_label="Ada",
            sender_kind="human",
            mention=True,
            message_id=MESSAGE_ID,
            conversation_seq=1,
        )
    )
    assert card.startswith("Channel message")
    assert "thread_root_id" not in card


def test_a_reply_event_round_trips_its_root_through_the_persisted_ref() -> None:
    """The event's additive ``thread_root_id`` survives the ONE persisted ref
    spelling (an 8th segment), and the older 4/7-part refs parse unchanged."""

    from local_operator.team_bridge.types import (
        ConversationFeedRef,
        parse_conversation_ref_value,
    )

    batch = {
        "events": [
            {
                "type": "message_available",
                "ids": {
                    "message_id": MESSAGE_ID,
                    "conversation_id": CONVERSATION_ID,
                    "conversation_seq": 7,
                    "conversation_name": "#agents",
                    "sender_label": "Bea's agent",
                    "sender_kind": "agent",
                    "thread_root_id": ROOT_ID,
                },
                "at": HISTORY_AT,
            }
        ],
        "cursor": "1",
    }
    _cursor, refs, _resync = project_events_batch(batch)
    ref = plan_feed_refs(refs).conversation_deliveries[0]
    assert ref.thread_root_id == ROOT_ID
    assert ref.sender_label == "Bea's agent"
    value = ref.encode().split("=", 1)[1]
    assert value.endswith(f"/{ROOT_ID}")
    assert parse_conversation_ref_value(value) == ref

    # A reply without card metadata pads the three label slots empty.
    bare = ConversationFeedRef(CONVERSATION_ID, MESSAGE_ID, 7, False, thread_root_id=ROOT_ID)
    assert parse_conversation_ref_value(bare.encode().split("=", 1)[1]) == bare
    # Pre-thread refs keep their spelling and parse back with no root.
    legacy = ConversationFeedRef(CONVERSATION_ID, MESSAGE_ID, 7, False)
    assert parse_conversation_ref_value(legacy.encode().split("=", 1)[1]) == legacy
    # A root that is not a message id is malformed, never guessed.
    assert parse_conversation_ref_value(f"{CONVERSATION_ID}/{MESSAGE_ID}/7/0////x") is None
