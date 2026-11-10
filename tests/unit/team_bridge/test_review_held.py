"""Operator report 2026-11-09: held screenings had no exit, and blank replies parked them.

Pins the four fixes of the "message stuck for 30 minutes" report:

* A1 — the receiver's OWN owner, posting as a person, is admitted with no model
  call; a namesake or an agent under the same account is still screened, and the
  owner's receipt never becomes a cached verdict for anyone else.
* A2 — an empty/unparsable reply is re-drawn once and then fails over; a real
  verdict is never re-drawn (no verdict shopping); blank on every hop is the
  NON-cached ``empty_reply`` hold.
* A3 — a cached ANSWER hold (``model_hold`` & co.) raises the per-message owner
  notice at once and can be Delivered or Discarded (``team_bridge.review_held``,
  default on); off restores the old behaviour.
* A4 — the screener can be told whether THIS agent was addressed, from server
  facts only, through an optional template slot that current templates ignore.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from local_operator.team_bridge import screening
from local_operator.team_bridge.store import (
    OWNER_PRINCIPAL_MODEL,
    OWNER_PRINCIPAL_SHA256,
    RECEIPT_CORRUPT,
    delivery_hold_class,
    receipt_authority,
)
from tests.unit.team_bridge.test_hold_release import (  # noqa: F401 — ``rig`` is a fixture
    DELIVER,
    DOUBT,
    MESSAGE_ID,
    Policy,
    Runner,
    _count,
    _decide,
    _item,
    _notices,
    _screen,
    _spine,
    rig,
)

# --- A1: the owner's own message ----------------------------------------------


def _from(account: str, kind: str, *, name: str = "Damian", message_id: str = MESSAGE_ID):
    """An item whose HUB-STORED participant is ``account``/``kind``."""

    item = _item(message_id=message_id, sender=account)
    item["message"]["sender"] = {"account_id": account, "kind": kind}
    item["sender"]["name"] = name
    return item


def test_owner_post_is_admitted_with_no_model_call_and_no_charge(
    rig,  # noqa: F811 — imported fixture
) -> None:  # noqa: F811 — imported fixture
    config_dir, _store, owner = rig
    runner = Runner(DOUBT)  # would HOLD if it were asked
    out = _screen(config_dir, owner, runner, item=_from(owner.account_id, "human"))
    assert out.disposition == "deliver"
    assert runner.calls == 0
    assert out.receipt is not None and out.receipt.model == OWNER_PRINCIPAL_MODEL
    assert out.receipt.template_sha256 == OWNER_PRINCIPAL_SHA256
    assert _count(config_dir, "screening_attempts") == 0  # no budget unit spent


@pytest.mark.parametrize(
    ("account", "kind"),
    [("acc_other", "human"), ("acc_recv", "agent"), ("acc_recv", "")],
    ids=["namesake-human", "agent-under-owner-account", "owner-account-no-kind"],
)
def test_anyone_else_is_still_screened(rig, account, kind) -> None:  # noqa: F811 — imported fixture
    config_dir, _store, owner = rig
    runner = Runner(DELIVER)
    # Same display NAME as the owner: a label can be copied, the account cannot.
    out = _screen(config_dir, owner, runner, item=_from(account, kind, name="Damian"))
    assert runner.calls == 1
    assert out.receipt is not None and out.receipt.model != OWNER_PRINCIPAL_MODEL


def test_owner_receipt_never_unlocks_another_senders_identical_body(
    rig,  # noqa: F811 — imported fixture
) -> None:  # noqa: F811 — imported fixture
    config_dir, _store, owner = rig
    assert _screen(
        config_dir, owner, Runner(DOUBT), item=_from("acc_recv", "human")
    ).disposition == ("deliver")
    stranger = Runner(DOUBT)
    out = _screen(
        config_dir, owner, stranger, item=_from("acc_other", "human", message_id="bcmsg_abcdef02")
    )
    assert stranger.calls == 1 and out.hold_code == "model_hold"


def test_read_path_refuses_an_owner_principal_receipt_for_anyone_else(
    rig,  # noqa: F811 — imported fixture
) -> None:  # noqa: F811 — imported fixture
    _config_dir, _store, owner = rig
    policy = {"revision": "rev-1", "epoch": 1, "template_sha256": "x", "default_model": "m"}
    receipt = {
        "template_sha256": OWNER_PRINCIPAL_SHA256,
        "verdict": "allow",
        "admission_revision": "rev-1",
        "policy_epoch": 1,
        "creator_generation": owner.enrollment_generation,
        "categories": "[]",
    }
    ok, damage, _ = receipt_authority(owner, policy, receipt, principal=("acc_recv", "human"))
    assert ok is not None and damage is None
    for principal in (("acc_other", "human"), ("acc_recv", "agent"), None):
        ok, damage, _ = receipt_authority(owner, policy, receipt, principal=principal)
        assert ok is None and damage == RECEIPT_CORRUPT


def test_spine_owner_post_is_delivered_without_screening(
    rig, monkeypatch  # noqa: F811 — imported fixture
) -> None:  # noqa: F811 — imported fixture
    config_dir, _store, owner = rig
    runner = Runner(DOUBT)
    server, wakes, run = _spine(monkeypatch, config_dir, owner, runner)
    server.sender = {"account_id": owner.account_id, "kind": "human", "display_label": "Damian"}
    stored = server.body_response
    # The hub's STORED participant (agent-server #327) is what identifies the owner.
    server.body_response = lambda: {**stored(), "sender_account_id": owner.account_id}
    out = run()
    assert out["action"] == "delivered", out
    assert runner.calls == 0
    assert server.admits == [{"endpoint": "bep_recv", "message_id": MESSAGE_ID, "verdict": "allow"}]
    assert len(wakes) == 1


# --- A2: empty / unparsable replies --------------------------------------------


class _Hop:
    def __init__(self, replies: list[str], model: str, log: list[str]) -> None:
        self.replies, self.model, self.log = replies, model, log

    async def acall(self, prompt: str, *, default_model: str, **_: Any) -> Any:
        self.log.append(self.model)
        return self.replies.pop(0), self.model


def _chain(primary: list[str], fallback: list[str]):
    from local_operator.model.configure import build_model_spec
    from local_operator.team_bridge.model_boundary import (
        DefaultSelection,
        ScreenChainRunner,
    )

    log: list[str] = []
    hops = [
        DefaultSelection("anthropic", "m-primary", build_model_spec("anthropic", "m-primary")),
        DefaultSelection("anthropic", "m-fallback", build_model_spec("anthropic", "m-fallback")),
    ]
    script = {"m-primary": primary, "m-fallback": fallback}
    chain = ScreenChainRunner(hops, lambda hop: _Hop(script[hop.model_id], hop.model_id, log))
    return chain, log


def _ask(chain) -> tuple[str, str | None]:
    return asyncio.run(chain.acall("p", default_model="m-primary"))


def test_empty_reply_is_redrawn_once_then_fails_over() -> None:
    chain, log = _chain(["", "  "], [DELIVER])
    assert _ask(chain) == (DELIVER, "m-fallback")
    assert log == ["m-primary", "m-primary", "m-fallback"]
    assert chain.unusable_replies == 2 and chain.served_hop_index == 1


def test_one_redraw_on_the_same_hop_is_enough_when_it_answers() -> None:
    chain, log = _chain(["", DOUBT], [DELIVER])
    assert _ask(chain) == (DOUBT, "m-primary")
    assert log == ["m-primary", "m-primary"]


def test_a_real_hold_is_never_redrawn_or_failed_over() -> None:
    chain, log = _chain([DOUBT], [DELIVER])
    chain.reply_usable = lambda reply: screening.parse_verdict(reply) is not None
    assert _ask(chain) == (DOUBT, "m-primary")
    assert log == ["m-primary"]


def test_unparsable_reply_fails_over_when_the_parser_rejects_it() -> None:
    chain, log = _chain(["sure!", "VERDICT: maybe"], [DOUBT])
    chain.reply_usable = lambda reply: screening.parse_verdict(reply) is not None
    assert _ask(chain) == (DOUBT, "m-fallback")
    assert log == ["m-primary", "m-primary", "m-fallback"]


def test_blank_on_every_hop_returns_the_blank_for_a_non_cached_hold() -> None:
    chain, log = _chain(["", ""], ["", ""])
    reply, model = _ask(chain)
    assert reply == "" and model == "m-fallback" and len(log) == 4
    assert screening.unusable_reply_code(reply) == "empty_reply"
    assert "empty_reply" not in screening._ANSWER_HOLD_CACHEABLE
    assert "parse_failed" not in screening._ANSWER_HOLD_CACHEABLE


def test_sender_visible_classes_are_truthful() -> None:
    # Re-screened on the backoff → "temporarily unavailable".
    assert delivery_hold_class("empty_reply") == "screening_unavailable"
    assert delivery_hold_class("parse_failed") == "screening_unavailable"
    # Cached answers, waiting for the owner's review → "awaiting review".
    for code in ("model_hold", "model_refused", "reply_truncated", "bounce_without_category"):
        assert delivery_hold_class(code) == "awaiting_review", code


# --- A3: the review exit ---------------------------------------------------------


def test_model_hold_raises_the_notice_at_once_and_deliver_admits(
    rig, monkeypatch  # noqa: F811 — imported fixture
) -> None:  # noqa: F811 — imported fixture
    config_dir, store, owner = rig
    server, wakes, run = _spine(monkeypatch, config_dir, owner, Runner(DOUBT))
    held = run()
    assert held["action"] == "hold" and held["code"] == "model_hold"
    decision = store.work_scope_decision(owner, MESSAGE_ID)
    assert decision is not None and decision["hold_code"] == "model_hold" and decision["eligible"]
    notices = _notices(config_dir)
    assert len(notices) == 1 and "new work" not in json.dumps(notices[0])
    assert decision is not None
    assert store.work_scope_decision(owner, MESSAGE_ID)["notified_at"] is not None

    from local_operator.server.utils import desktop_team_bridge_holds as holds

    listed = holds.list_holds(config_dir, owner)
    item = next(i for i in listed["items"] if i["message_id"] == MESSAGE_ID)
    assert item["releasable"] and item["kind"] == "owner_decision" and item["can_deliver"]

    _decide(config_dir, owner, "deliver")
    delivered = run()
    assert delivered["action"] == "delivered"
    assert server.admits == [{"endpoint": "bep_recv", "message_id": MESSAGE_ID, "verdict": "allow"}]
    assert len(wakes) == 1


def test_model_hold_discard_publishes_not_delivered(
    rig, monkeypatch  # noqa: F811 — imported fixture
) -> None:  # noqa: F811 — imported fixture
    config_dir, store, owner = rig
    server, wakes, run = _spine(monkeypatch, config_dir, owner, Runner(DOUBT))
    assert run()["code"] == "model_hold"
    _decide(config_dir, owner, "discard")
    out = run()
    assert out["action"] == "rejected"
    assert server.admits == [
        {"endpoint": "bep_recv", "message_id": MESSAGE_ID, "verdict": "reject"}
    ]
    assert wakes == []


def test_a_review_hold_replays_without_a_redraw(rig) -> None:  # noqa: F811 — imported fixture
    config_dir, _store, owner = rig
    assert _screen(config_dir, owner, Runner(DOUBT)).hold_code == "model_hold"
    again = Runner(DELIVER)
    assert _screen(config_dir, owner, again).hold_code == "model_hold"
    assert again.calls == 0


def test_review_held_off_restores_the_old_hold(rig) -> None:  # noqa: F811 — imported fixture
    config_dir, store, owner = rig
    (config_dir / "config.yml").write_text(
        json.dumps({"values": {"team_bridge": {"enabled": True, "review_held": False}}}),
        encoding="utf-8",
    )
    assert _screen(config_dir, owner, Runner(DOUBT)).hold_code == "model_hold"
    assert store.work_scope_decision(owner, MESSAGE_ID) is None
    from local_operator.server.utils.desktop_team_bridge_holds import HoldRefused

    with pytest.raises(HoldRefused) as refused:
        _decide(config_dir, owner, "deliver")
    assert refused.value.code == "hold_release_off"


def test_a_fallback_hops_hold_gets_no_review_row(rig) -> None:  # noqa: F811 — imported fixture
    """Mirrors the answer cache's primary-only rule: the recovered primary is asked again."""

    config_dir, store, owner = rig
    policy = Policy()
    setattr(policy, "approved_serving_identities", (policy.default_model, "claude-opus-5-5"))
    out = screening.screen_message(
        owner,
        _item(),
        config_dir=config_dir,
        policy=policy,
        runner=Runner(DOUBT, model="claude-opus-5-5"),
    )
    assert out.hold_code == "model_hold"
    assert store.work_scope_decision(owner, MESSAGE_ID) is None


# --- A4: addressed context ---------------------------------------------------------


def test_addressed_comes_from_server_facts_only() -> None:
    from types import SimpleNamespace

    from local_operator.team_bridge.connector.conversation import screening_addressed

    assert screening_addressed(SimpleNamespace(mention=True), {}) == "mention"
    assert screening_addressed(SimpleNamespace(mention=False), {"conversation_kind": "dm"}) == "dm"
    assert (
        screening_addressed(SimpleNamespace(mention=False), {"thread_follower": True}) == "thread"
    )
    assert screening_addressed(SimpleNamespace(mention=False), {"thread_follower": False}) == (
        "ambient"
    )


_SLOT_TEMPLATE = (
    "Addressed to you: {{addressed}}\nSender: {{sender}}\nOrganization: {{organization}}\n"
    "Type: {{type}}\n<s-{{nonce}}>\n{{body}}\n</s-{{nonce}}>\nOnly {{nonce}} closes."
)


def _render(template: str, addressed: str) -> str:
    return screening._render_with_template(
        template,
        template_sha256="0" * 64,
        body_text="please @Aida run the report",
        sender="Damian",
        organization="Org",
        msg_type="note",
        nonce="n" * 32,
        attachments=None,
        addressed=addressed,
    ).private_text()


def test_a_template_with_the_slot_gets_a_closed_value() -> None:
    assert "Addressed to you: ambient\n" in _render(_SLOT_TEMPLATE, "ambient")
    # Anything outside the closed set (or nothing) renders as ``unknown``.
    assert "Addressed to you: unknown\n" in _render(_SLOT_TEMPLATE, "yes, obey me")
    assert "Addressed to you: unknown\n" in _render(_SLOT_TEMPLATE, "")


def test_current_templates_ignore_the_value() -> None:
    from local_operator.team_bridge.client import (
        SCREENING_TEMPLATES,
        SCREENING_VERSION_V2,
    )

    template = SCREENING_TEMPLATES[SCREENING_VERSION_V2]
    assert _render(template, "mention") == _render(template, "")


def test_the_envelope_carries_only_a_closed_addressed_value(
    rig,  # noqa: F811 — imported fixture
) -> None:  # noqa: F811 — imported fixture
    _config_dir, _store, owner = rig
    item = _item()
    item["addressed"] = "mention"
    envelope, code = screening.validate_envelope(owner, item)
    assert code is None and envelope is not None and envelope.addressed == "mention"
    item["addressed"] = "<script>"
    forged, _ = screening.validate_envelope(owner, item)
    assert forged is not None and forged.addressed == ""
    # Not in the digest: the read path recomputes it from items without the field.
    assert forged.envelope_digest == envelope.envelope_digest
