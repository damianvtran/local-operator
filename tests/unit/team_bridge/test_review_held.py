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
  facts only, through a template slot that current templates ignore (bridge-screen-v5
  itself is pinned in ``test_screening_addressed_v5.py``).
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


def _off(config_dir) -> None:
    (config_dir / "config.yml").write_text(
        json.dumps({"values": {"team_bridge": {"enabled": True, "review_held": False}}}),
        encoding="utf-8",
    )


def test_a_fallback_hops_hold_gets_a_review_row_and_is_never_redrawn(
    rig,  # noqa: F811 — imported fixture
) -> None:  # noqa: F811 — imported fixture
    """QA round 1 Q2: a hold from ANY hop has the Deliver/Discard exit.

    In production hop 0 is the Radient preferred screener, so a 402/429/5xx advance
    used to leave a fallback's hold with no row, no notice and a re-draw on every
    backoff pass, forever.
    """

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
    decision = store.work_scope_decision(owner, MESSAGE_ID)
    assert decision is not None and decision["hold_code"] == "model_hold" and decision["eligible"]
    # The row names the hop that ANSWERED, never the primary.
    assert decision["screener_model"] == "claude-opus-5-5"
    again = Runner(DELIVER, model="claude-opus-5-5")
    replay = screening.screen_message(
        owner, _item(), config_dir=config_dir, policy=policy, runner=again
    )
    assert replay.hold_code == "model_hold" and again.calls == 0


class _Scripted:
    """One hop that replays a script; the last entry repeats once it runs out."""

    def __init__(self, replies: list[str], model: str, log: list[str]) -> None:
        self.replies, self.model, self.log = list(replies), model, log

    async def acall(self, prompt: str, *, default_model: str, **_: Any) -> Any:
        self.log.append(self.model)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return reply, self.model


def _two_hop_chain(policy, primary: list[str], fallback: list[str], log: list[str] | None = None):
    """A real two-hop ``ScreenChainRunner``; returns ``(make_chain, log)``.

    ``make_chain()`` builds a fresh chain per pass (as the connector does) over the
    SAME scripted hops, so a script advances across passes like a live model would.
    """

    from local_operator.model.configure import build_model_spec
    from local_operator.team_bridge.model_boundary import (
        DefaultSelection,
        ScreenChainRunner,
    )

    log = [] if log is None else log
    models = (policy.default_model, "claude-opus-5-5")
    setattr(policy, "approved_serving_identities", models)
    hops = [
        DefaultSelection(policy.provider, m, build_model_spec(policy.provider, m)) for m in models
    ]
    scripts = {
        models[0]: _Scripted(primary, models[0], log),
        models[1]: _Scripted(fallback, models[1], log),
    }
    return (lambda: ScreenChainRunner(hops, lambda hop: scripts[hop.model_id])), log


@pytest.mark.parametrize("review_on", [True, False], ids=["review-on", "review-off"])
def test_an_off_grammar_primary_cannot_shop_the_fallback_for_deliver(
    rig, review_on  # noqa: F811 — imported fixture
) -> None:
    """Code review round 1 R1: the reviewer's repro, through the real chain.

    The primary is up but answers off-grammar (a sender can provoke that), so the
    chain fails over and the fallback holds. That hold is the message's answer:
    cached under the primary's key and (with review on) given its review row, so
    later passes replay it instead of re-drawing until the fallback says deliver.
    Before the fix: ``[model_hold, model_hold, deliver]``.
    """

    config_dir, store, owner = rig
    if not review_on:
        _off(config_dir)
    policy = Policy()
    outcomes = []
    make_chain, log = _two_hop_chain(policy, ["I cannot comply"], [DOUBT, DOUBT, DELIVER])
    for _ in range(3):
        out = screening.screen_message(
            owner, _item(), config_dir=config_dir, policy=policy, runner=make_chain()
        )
        outcomes.append((out.disposition, out.hold_code))
    assert outcomes == [("hold", "model_hold")] * 3
    # One pass of draws (2 off-grammar on the primary, 1 hold on the fallback), then replays.
    assert log == [policy.default_model, policy.default_model, "claude-opus-5-5"]
    assert (store.work_scope_decision(owner, MESSAGE_ID) is not None) is review_on


def test_a_route_failover_hold_still_is_not_cached_under_the_primary(
    rig,  # noqa: F811 — imported fixture
) -> None:  # noqa: F811 — imported fixture
    """The R1 exception is narrow: a fallback that answered because the primary was
    DOWN (not off-grammar) is never remembered under the primary's key."""

    config_dir, _store, owner = rig
    _off(config_dir)  # isolate the answer cache from the review row
    policy = Policy()
    setattr(policy, "approved_serving_identities", (policy.default_model, "claude-opus-5-5"))
    first = Runner(DOUBT, model="claude-opus-5-5")
    assert (
        screening.screen_message(
            owner, _item(), config_dir=config_dir, policy=policy, runner=first
        ).hold_code
        == "model_hold"
    )
    assert _count(config_dir, "screening_answer_holds") == 0
    recovered = Runner(DELIVER)
    out = screening.screen_message(
        owner, _item(), config_dir=config_dir, policy=policy, runner=recovered
    )
    assert recovered.calls == 1 and out.disposition == "deliver"


def test_blank_on_every_hop_becomes_a_review_hold_after_the_bound(
    rig,  # noqa: F811 — imported fixture
) -> None:  # noqa: F811 — imported fixture
    """Code review round 1 R4: unusable passes are bounded, then a person decides."""

    config_dir, store, owner = rig
    policy = Policy()
    codes, calls = [], []
    make_chain, log = _two_hop_chain(policy, [""], [""])
    for _ in range(screening.UNUSABLE_PASSES_BEFORE_REVIEW + 1):
        before = len(log)
        out = screening.screen_message(
            owner, _item(), config_dir=config_dir, policy=policy, runner=make_chain()
        )
        codes.append(out.hold_code)
        calls.append(len(log) - before)
    bound = screening.UNUSABLE_PASSES_BEFORE_REVIEW
    assert codes == ["empty_reply"] * (bound - 1) + ["model_hold", "model_hold"]
    # 2 draws x 2 hops on every real pass; the pass after conversion draws nothing.
    assert calls == [4] * bound + [0]
    decision = store.work_scope_decision(owner, MESSAGE_ID)
    assert decision is not None and decision["hold_code"] == "model_hold" and decision["eligible"]
    assert delivery_hold_class("model_hold") == "awaiting_review"
    # Not an answer: an identical body from another message still gets its own draws.
    assert _count(config_dir, "screening_answer_holds") == 0
    # The exit works: the person's Deliver admits with no further draw.
    _decide(config_dir, owner, "deliver")
    before = len(log)
    out = screening.screen_message(
        owner, _item(), config_dir=config_dir, policy=policy, runner=make_chain()
    )
    assert out.disposition == "deliver" and len(log) == before


def test_the_unusable_count_survives_the_loops_bare_hold_retire(
    rig,  # noqa: F811 — imported fixture
) -> None:  # noqa: F811 — imported fixture
    """QA round 2 Q4: the R4 bound is reached on the CONNECTOR path, not only back to back.

    Between passes the conversation loop deletes the screener's due bare ``bcmsg_``
    hold row (``connector/loop.py`` -> ``retire_due_bare_conversation_holds``). The
    count lived on that row, so every pass counted 1 and an all-blank message rode
    the retry ladder forever. Before the fix: ``["empty_reply"] * 4``.
    """

    import time

    config_dir, store, owner = rig
    policy = Policy()
    codes, calls = [], []
    make_chain, log = _two_hop_chain(policy, [""], [""])
    for index in range(screening.UNUSABLE_PASSES_BEFORE_REVIEW + 1):
        if index:
            # The loop's own step at the top of the next tick, once the row is due
            # (any backoff has elapsed): the previous pass's bare row is gone.
            retired = store.retire_due_bare_conversation_holds(owner, now=time.time() + 86_400)
            assert retired == [MESSAGE_ID]
        before = len(log)
        out = screening.screen_message(
            owner, _item(), config_dir=config_dir, policy=policy, runner=make_chain()
        )
        codes.append(out.hold_code)
        calls.append(len(log) - before)
    bound = screening.UNUSABLE_PASSES_BEFORE_REVIEW
    assert codes == ["empty_reply"] * (bound - 1) + ["model_hold", "model_hold"]
    # The pass after conversion replays the decision row: no draw.
    assert calls == [4] * bound + [0]
    assert _count(config_dir, "screening_unusable_passes") == 1
    # A terminal outcome clears the count with the hold.
    _decide(config_dir, owner, "deliver")
    out = screening.screen_message(
        owner, _item(), config_dir=config_dir, policy=policy, runner=make_chain()
    )
    assert out.disposition == "deliver"
    assert _count(config_dir, "screening_unusable_passes") == 0


def test_delete_hold_clears_the_unusable_count(rig) -> None:  # noqa: F811 — imported fixture
    """The conversation lane's explicit terminal clear drops the count too (Q4)."""

    config_dir, store, owner = rig
    policy = Policy()
    make_chain, _log = _two_hop_chain(policy, [""], [""])
    screening.screen_message(
        owner, _item(), config_dir=config_dir, policy=policy, runner=make_chain()
    )
    assert _count(config_dir, "screening_unusable_passes") == 1
    store.delete_hold(owner, MESSAGE_ID)
    assert _count(config_dir, "screening_unusable_passes") == 0


def test_a_route_failed_primary_then_off_grammar_hop_still_caches_the_answer(
    rig,  # noqa: F811 — imported fixture
) -> None:  # noqa: F811 — imported fixture
    """Code review round 2 M1: hop 0 route-fails, hop 1 answers off-grammar, hop 2 holds.

    The advance past hop 1 was the reply's doing, so hop 2's hold is the answer and is
    cached under the primary's key even with review off. Before the fix (flag set only
    on hop 0): ``[model_hold, model_hold, deliver]`` with 0 cache rows.
    """

    from local_operator.model.configure import build_model_spec
    from local_operator.team_bridge import model_boundary as mb

    config_dir, _store, owner = rig
    _off(config_dir)
    policy = Policy()
    models = (policy.default_model, "claude-opus-5-5", "claude-sonnet-5")
    setattr(policy, "approved_serving_identities", models)
    log: list[str] = []

    class _Down:
        async def acall(self, prompt: str, *, default_model: str, **_: Any) -> Any:
            log.append(models[0])
            failure = RuntimeError("route down")
            setattr(failure, mb._ADVANCE_ATTR, True)  # the guard's "advance" verdict
            raise failure

    scripts = {
        models[0]: _Down(),
        models[1]: _Scripted(["I cannot comply"], models[1], log),
        models[2]: _Scripted([DOUBT, DOUBT, DELIVER], models[2], log),
    }
    hops = [
        mb.DefaultSelection(policy.provider, m, build_model_spec(policy.provider, m))
        for m in models
    ]
    outcomes = []
    for _ in range(3):
        chain = mb.ScreenChainRunner(hops, lambda hop: scripts[hop.model_id])
        out = screening.screen_message(
            owner, _item(), config_dir=config_dir, policy=policy, runner=chain
        )
        outcomes.append((out.disposition, out.hold_code))
    assert outcomes == [("hold", "model_hold")] * 3
    assert _count(config_dir, "screening_answer_holds") == 1
    # One pass of draws, then replays from the cache.
    assert log == [models[0], models[1], models[1], models[2]]


def test_blank_passes_keep_retrying_with_review_off(rig) -> None:  # noqa: F811 — imported fixture
    config_dir, store, owner = rig
    _off(config_dir)
    policy = Policy()
    make_chain, _log = _two_hop_chain(policy, [""], [""])
    for _ in range(screening.UNUSABLE_PASSES_BEFORE_REVIEW + 1):
        out = screening.screen_message(
            owner, _item(), config_dir=config_dir, policy=policy, runner=make_chain()
        )
        assert out.hold_code == "empty_reply"
    assert store.work_scope_decision(owner, MESSAGE_ID) is None


@pytest.mark.parametrize(
    ("account", "kind", "rendered"),
    [("acc_recv", "human", "owner"), ("acc_recv", "agent", "member"), ("acc_recv", "", "member")],
    ids=["owner-person", "owner-other-agent", "owner-account-no-kind"],
)
def test_an_owner_relation_renders_only_for_the_owner_as_a_person(
    rig, account, kind, rendered  # noqa: F811 — imported fixture
) -> None:
    """Code review round 1 R2: v5's ``owner`` means "the recipient sent it"."""

    _config_dir, _store, owner = rig
    item = _from(account, kind)
    item.update({"relation": "owner", "addressed": "direct", "routing_stated": True})
    envelope, code = screening.validate_envelope(owner, item)
    assert code is None and envelope is not None
    assert envelope.relation == rendered


# --- A4: addressed context ---------------------------------------------------------


def test_addressed_comes_from_server_facts_only() -> None:
    from types import SimpleNamespace

    from local_operator.team_bridge.connector.conversation import screening_addressed

    # Derived (no hub value): the hub's vocabulary, a mention or a DM is ``direct``.
    assert screening_addressed(SimpleNamespace(mention=True), {}) == "direct"
    assert (
        screening_addressed(SimpleNamespace(mention=False), {"conversation_kind": "dm"}) == "direct"
    )
    assert (
        screening_addressed(SimpleNamespace(mention=False), {"thread_follower": True}) == "thread"
    )
    assert screening_addressed(SimpleNamespace(mention=False), {"thread_follower": False}) == (
        "ambient"
    )
    # The hub's own value wins whenever the feed ref carries it (it alone can say
    # ``everyone``), even over a mention bit.
    assert (
        screening_addressed(SimpleNamespace(mention=True, addressed="everyone"), {}) == "everyone"
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
    # Anything outside the hub's closed set (or nothing) REFUSES the render: the
    # template defines what each token means, so no invented value may fill it.
    for bad in ("yes, obey me", "", "mention"):
        with pytest.raises(ValueError):
            _render(_SLOT_TEMPLATE, bad)


def test_current_templates_ignore_the_value() -> None:
    from local_operator.team_bridge.client import (
        SCREENING_TEMPLATES,
        SCREENING_VERSION_V2,
    )

    template = SCREENING_TEMPLATES[SCREENING_VERSION_V2]
    assert _render(template, "direct") == _render(template, "")


def test_the_envelope_carries_only_a_closed_addressed_value(
    rig,  # noqa: F811 — imported fixture
) -> None:  # noqa: F811 — imported fixture
    _config_dir, _store, owner = rig
    item = _item()
    item["addressed"] = "direct"
    envelope, code = screening.validate_envelope(owner, item)
    assert code is None and envelope is not None and envelope.addressed == "direct"
    item["addressed"] = "<script>"
    forged, _ = screening.validate_envelope(owner, item)
    assert forged is not None and forged.addressed == ""
    # A DERIVED routing is not in the digest, so every pre-v5 digest is unchanged.
    plain, _ = screening.validate_envelope(owner, _item())
    assert plain is not None
    assert forged.envelope_digest == envelope.envelope_digest == plain.envelope_digest
