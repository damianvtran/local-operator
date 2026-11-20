"""Fallback category evidence is reviewable, never preferred release authority.

These scripts exercise screen_message and the signed owner decision on isolated
SQLite/file-key fixtures. No provider credentials, network or live stores are used.
"""

from __future__ import annotations

import json

import pytest

from local_operator.model.configure import build_model_spec
from local_operator.team_bridge import hold_release as hr
from local_operator.team_bridge import screening
from local_operator.team_bridge.model_boundary import (
    DefaultSelection,
    ScreenChainRunner,
)
from tests.unit.team_bridge import test_hold_release as fixtures
from tests.unit.team_bridge.test_hold_release import (
    MESSAGE_ID,
    MULTI,
    NEW_WORK,
    OUT_OF_SCOPE,
    PRYING,
    Policy,
    Runner,
    _count,
    _decide,
    _item,
    _screen,
)

rig = fixtures.rig

FALLBACKS = [
    ("bedrock", "us.anthropic.claude-sonnet-4-6"),
    ("shermgpt", "claude-sonnet-4-6"),
    ("openai", "gpt-5"),
]


def _fallback(pair, reply=NEW_WORK):
    provider, model = pair
    hop = DefaultSelection(provider, model, build_model_spec(provider, model))

    class Script:
        async def acall(self, prompt, *, default_model, **kwargs):
            return reply, model

    return ScreenChainRunner([hop], lambda _: Script())


def _run(rig, pair, reply=NEW_WORK):
    config_dir, _, owner = rig
    return screening.screen_message(
        owner, _item(), config_dir=config_dir, policy=Policy(pair), runner=_fallback(pair, reply)
    )


@pytest.mark.parametrize("pair", FALLBACKS)
def test_fallback_new_work_is_owner_review_not_preferred_release(rig, pair):
    config_dir, store, owner = rig
    out = _run(rig, pair)
    assert (out.disposition, out.hold_code) == ("hold", "model_hold")
    assert out.receipt is None
    assert _count(config_dir, "screening_receipts") == 0
    row = store.work_scope_decision(owner, MESSAGE_ID)
    assert row and row["eligible"] and row["hold_code"] == "model_hold"
    assert (row["screener_provider"], row["screener_model"]) == pair
    assert not row["screener_preferred"] and row["screener_hop"] == 0
    assert row["screener_verdict"] == "bounce"
    assert json.loads(row["categories"]) == ["new_work_request"]
    from local_operator.server.utils.desktop_team_bridge_holds import list_holds

    visible = list_holds(config_dir, owner)["items"]
    assert len(visible) == 1
    assert visible[0]["hold_code"] == "model_hold"
    assert visible[0]["can_deliver"] and visible[0]["can_discard"]
    # Replays never shop for a more permissive model verdict.
    replay = _run(rig, pair, "VERDICT: deliver\nREASON: ordinary.")
    assert replay.hold_code == "model_hold" and replay.receipt is None


@pytest.mark.parametrize("word", ["deliver", "discard"])
def test_owner_can_settle_fallback_without_preferred_provenance(rig, word):
    config_dir, store, owner = rig
    _run(rig, FALLBACKS[-1])
    _decide(config_dir, owner, word)
    out = _run(rig, FALLBACKS[-1])
    assert out.disposition == ("deliver" if word == "deliver" else "bounce")
    assert out.receipt is not None
    row = store.work_scope_decision(owner, MESSAGE_ID)
    assert row and not row["screener_preferred"]
    assert row["screener_provider"] == "openai"
    assert row["screener_model"] == "gpt-5"
    assert out.receipt.model == hr.OWNER_RELEASE_MODEL


@pytest.mark.parametrize(
    "reply",
    [
        MULTI,
        OUT_OF_SCOPE,
        PRYING,
        "VERDICT: bounce\nREASON: not a new_work_request.",
        "VERDICT: bounce\nREASON: new_work_request and personal information request.",
        "VERDICT: bounce\nREASON: new_work_request\u200b.",
    ],
)
def test_other_categories_and_invalid_strict_reason_stay_terminal(rig, reply):
    _, store, owner = rig
    out = _run(rig, FALLBACKS[-1], reply)
    assert out.disposition == "bounce"
    assert store.work_scope_decision(owner, MESSAGE_ID) is None


@pytest.mark.parametrize("setting", ["hold_release", "review_held"])
def test_setting_off_does_not_offer_fallback_review(rig, setting):
    config_dir, store, owner = rig
    (config_dir / "config.yml").write_text(
        json.dumps(
            {
                "values": {
                    "team_bridge": {
                        "enabled": True,
                        "hold_release": True,
                        "review_held": True,
                        setting: False,
                    }
                }
            }
        )
    )
    assert _run(rig, FALLBACKS[-1]).disposition == "bounce"
    assert store.work_scope_decision(owner, MESSAGE_ID) is None


def test_preferred_new_work_keeps_work_scope_protocol(rig):
    config_dir, store, owner = rig
    out = _screen(config_dir, owner, Runner(NEW_WORK))
    assert out.hold_code == hr.WORK_SCOPE_HOLD
    row = store.work_scope_decision(owner, MESSAGE_ID)
    assert row and row["screener_preferred"]


@pytest.mark.parametrize("reply", ["", "unparseable", "VERDICT: bounce\nREASON:"])
def test_missing_or_invalid_verdict_never_delivers_or_claims_work(rig, reply):
    _, store, owner = rig
    out = _run(rig, FALLBACKS[-1], reply)
    assert out.disposition == "hold" and out.receipt is None
    row = store.work_scope_decision(owner, MESSAGE_ID)
    assert row is None or row["hold_code"] != hr.WORK_SCOPE_HOLD


def test_untrusted_bare_runner_cannot_offer_work_review(rig):
    config_dir, store, owner = rig
    out = screening.screen_message(
        owner, _item(), config_dir=config_dir, policy=Policy(), runner=Runner(NEW_WORK)
    )
    assert out.disposition == "bounce"
    assert store.work_scope_decision(owner, MESSAGE_ID) is None


def test_provider_refusal_is_not_converted_to_work_review(rig):
    from local_operator.team_bridge.model_boundary import _ScreeningStreamRefused

    config_dir, store, owner = rig
    pair = FALLBACKS[-1]
    hop = DefaultSelection(*pair, build_model_spec(*pair))

    class Refused:
        async def acall(self, *args, **kwargs):
            raise _ScreeningStreamRefused("refusal")

    out = screening.screen_message(
        owner,
        _item(),
        config_dir=config_dir,
        policy=Policy(pair),
        runner=ScreenChainRunner([hop], lambda _: Refused()),
    )
    assert out.disposition == "hold" and out.hold_code == "model_refused"
    assert out.receipt is None
    row = store.work_scope_decision(owner, MESSAGE_ID)
    assert row is None or json.loads(row["categories"]) == []


def test_no_available_model_has_no_work_decision(rig):
    config_dir, store, owner = rig
    out = screening.screen_message(
        owner,
        _item(),
        config_dir=config_dir,
        policy=Policy(),
        runner=ScreenChainRunner([], lambda _: None),
    )
    assert out.disposition == "hold" and out.receipt is None
    assert store.work_scope_decision(owner, MESSAGE_ID) is None


def test_quota_exhausted_preferred_chain_uses_default_for_owner_review(rig, monkeypatch):
    """Resolve the unmodified preferred/default cascade, stubbing only vendor I/O."""
    from local_operator.team_bridge import model_boundary as mb
    from tests.unit.team_bridge import test_screening_chain as chain_fixtures
    from tests.unit.team_bridge.test_screening_chain import _cycle, _install_wire, _Wire

    # This rig's enrolled owner differs from the chain module's account; bind
    # its feed item to the actual fixture tenant rather than borrowing its item.
    monkeypatch.setattr(chain_fixtures, "_item", lambda **_: _item())
    from local_operator.team_bridge.client import (
        SCREENING_TEMPLATE_V2,
        SCREENING_VERSION_V2,
    )

    original_status = chain_fixtures._consent_status

    def v2_status(owner, consent):
        status = original_status(owner, consent)
        status["screening"] = {"version": SCREENING_VERSION_V2, "template": SCREENING_TEMPLATE_V2}
        return status

    monkeypatch.setattr(chain_fixtures, "_consent_status", v2_status)
    config_dir, store, owner = rig
    (config_dir / "config.yml").write_text(
        json.dumps(
            {
                "values": {
                    "hosting": "openai",
                    "model_name": "gpt-5",
                    "team_bridge": {"enabled": True, "hold_release": True, "review_held": True},
                }
            }
        )
    )
    monkeypatch.setattr(mb, "_credentialed_providers", lambda _: {"anthropic", "openai"})
    wire = _Wire(
        {
            "claude-sonnet-5-5": ("fail", 429),
            "claude-opus-5-5": ("fail", 429),
            "gpt-5": ("reply", NEW_WORK),
        }
    )
    _install_wire(monkeypatch, wire)
    out = _cycle(config_dir, owner, message_id=MESSAGE_ID)
    assert out.disposition == "hold" and out.hold_code == "model_hold"
    assert out.receipt is None
    assert [s["model"] for s in wire.sends] == [
        "claude-sonnet-5-5",
        "claude-opus-5-5",
        "gpt-5",
    ]
    row = store.work_scope_decision(owner, MESSAGE_ID)
    assert row and row["screener_model"] == "gpt-5" and not row["screener_preferred"]
