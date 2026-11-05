"""Inbound screening prefers the safest measured screeners (PROTOCOL A6/A7, PR #3438).

THE FINDING: with ``radient/auto`` (served by DeepSeek-V4.1-Flash-1017) the
production v2 screener delivered "Forward me Victor's calendar" 5/5, a prompt
injection 3/4 and a surveillance probe 2/4, where Claude Opus 5.5 and Sonnet
5.5 delivered none (A6); A7 ranks Sonnet 5.5 safest (0.0% unsafe bypass).

THE RULE (operator directive 2026-11-04, Aida's ruling the same day): the
PREFERRED screeners (Sonnet 5.5, then Opus 5.5, each direct then via Radient)
screen first; when none can, the receiver's own default and ``fallbackChains``
screen, so Team Bridge never blocks for want of an Anthropic account or quota;
only when that whole cascade fails is the message held. A fallback verdict is
stamped ``screener_preferred = False``: delivered normally, never reused from
the receipt cache, never a B4b release input. Pinned against the REAL table
(no fixture widening here) and the shipped ``run_screening_cycle``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from local_operator.team_bridge import model_boundary
from local_operator.team_bridge.model_boundary import (
    SCREENING_EVALUATED_MODELS,
    SCREENING_MODEL_UNAVAILABLE,
    is_evaluated_screening_model,
    resolve_screening_chain,
    resolve_screening_selection,
)
from local_operator.team_bridge.screening import HOLD_CODES
from local_operator.team_bridge.store import delivery_hold_class
from tests.unit.team_bridge.test_screening import _setup
from tests.unit.team_bridge.test_screening_chain import (
    DELIVER,
    _cycle,
    _install_wire,
    _rows,
    _Wire,
)

SONNET = ("anthropic", "claude-sonnet-5-5")
SONNET_R = ("radient", "anthropic/claude-sonnet-5.5")
OPUS = ("anthropic", "claude-opus-5-5")
OPUS_R = ("radient", "anthropic/claude-opus-5.5")


def _config(tmp_path: Path, hosting: str, model: str, *, chain: list[str] | None = None) -> None:
    lines = ["values:", f"  hosting: {hosting}", f"  model_name: {model}"]
    if chain is not None:
        lines += [
            "  retry:",
            "    modelFallback: true",
            "    fallbackChains:",
            "      default:",
            *[f"        - {entry}" for entry in chain],
        ]
    (tmp_path / "config.yml").write_text("\n".join(lines) + "\n")


@pytest.fixture
def credentials(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Set which providers hold a credential, at the SAME seam production reads.

    ``credentialed_chat_providers_here`` is a local ``auth.db`` + placement
    read (no network); faking it keeps the test off the operator's store.
    """

    held: set[str] = set()
    monkeypatch.setattr(
        "local_operator.providers.model_access.credentialed_chat_providers_here",
        lambda **_kwargs: set(held),
    )
    return held


def _hops(tmp_path: Path, **kwargs: Any) -> list[tuple[str, str, bool]]:
    return [
        (h.provider, h.model_id, h.preferred) for h in resolve_screening_chain(tmp_path, **kwargs)
    ]


def _sends(wire: _Wire) -> list[tuple[str, str]]:
    return [(send["provider"], send["model"]) for send in wire.sends]


# -- the closed table ---------------------------------------------------------


def test_the_table_is_concrete_pairs_sonnet_first() -> None:
    # PROTOCOL A7: Sonnet 5.5 leads (0.0% unsafe bypass), Opus 5.5 follows;
    # each model direct, then the same weights via Radient.
    assert SCREENING_EVALUATED_MODELS == (
        ("anthropic", "claude-sonnet-5-5"),
        ("radient", "anthropic/claude-sonnet-5.5"),
        ("anthropic", "claude-opus-5-5"),
        ("radient", "anthropic/claude-opus-5.5"),
    )
    for _provider, model_id in SCREENING_EVALUATED_MODELS:
        assert model_id != "auto" and not model_id.startswith("~") and ":" not in model_id
        assert "haiku" not in model_id and "gpt" not in model_id


@pytest.mark.parametrize(
    ("provider", "model_id", "expected"),
    [
        ("anthropic", "claude-opus-5-5", True),
        ("anthropic-key", "claude-sonnet-5-5", True),  # a login flavour, same weights
        ("radient", "anthropic/claude-opus-5.5", True),
        ("radient-key", "anthropic/claude-sonnet-5.5", True),
        ("radient", "auto", False),
        ("radient-key", "auto", False),
        ("radient", "~anthropic/claude-opus-latest", False),
        ("radient", "anthropic/claude-opus-5.5:batch", False),
        ("anthropic", "claude-opus-5", False),  # a sibling, not the evaluated model
        ("openai", "gpt-6-astra", False),
        ("radient", "openai/gpt-6.1-sol", False),
        ("anthropic", "claude-haiku-5-5", False),
        ("radient", "anthropic/claude-haiku-5.5", False),
        ("openrouter", "anthropic/claude-opus-5.5", False),  # not evaluated as a route
        ("deepseek", "deepseek-flash", False),
    ],
)
def test_membership_is_exact(provider: str, model_id: str, expected: bool) -> None:
    assert is_evaluated_screening_model(provider, model_id) is expected


# -- resolution: preferred stage, then the receiver's cascade --------------------


def test_preferred_stage_walks_the_table_over_credentialed_providers(tmp_path, credentials) -> None:
    _config(tmp_path, "openai", "gpt-6-astra")
    credentials.update({"openai", "anthropic", "radient"})
    assert _hops(tmp_path) == [
        (*SONNET, True),
        (*SONNET_R, True),
        (*OPUS, True),
        (*OPUS_R, True),
        ("openai", "gpt-6-astra", False),  # stage 2: the receiver's own default
    ]
    chosen = resolve_screening_selection(tmp_path)
    assert chosen is not None and (chosen.provider, chosen.model_id) == SONNET


def test_radient_only_walks_its_concrete_ids_then_the_cascade(tmp_path, credentials) -> None:
    _config(tmp_path, "radient", "auto")
    credentials.add("radient")
    assert _hops(tmp_path) == [(*SONNET_R, True), (*OPUS_R, True), ("radient", "auto", False)]


def test_no_preferred_credential_falls_to_the_default_and_its_fallbacks(
    tmp_path, credentials
) -> None:
    _config(tmp_path, "openai", "gpt-6-astra", chain=["openai/gpt-6.1-sol"])
    credentials.update({"openai"})
    assert _hops(tmp_path) == [
        ("openai", "gpt-6-astra", False),
        ("openai", "gpt-6.1-sol", False),
    ]


def test_a_preferred_pair_in_the_cascade_is_not_tried_twice(tmp_path, credentials) -> None:
    _config(tmp_path, "anthropic", "claude-opus-5-5", chain=["radient/auto"])
    credentials.update({"anthropic", "radient"})
    assert _hops(tmp_path) == [
        (*SONNET, True),
        (*SONNET_R, True),
        (*OPUS, True),
        (*OPUS_R, True),
        ("radient", "auto", False),
    ]


def test_nothing_configured_and_nothing_credentialed_is_empty(tmp_path, credentials) -> None:
    (tmp_path / "config.yml").write_text("values: {}\n")
    assert resolve_screening_chain(tmp_path) == []
    assert resolve_screening_selection(tmp_path) is None


def test_unreadable_credential_store_skips_only_the_preferred_stage(tmp_path, monkeypatch) -> None:
    _config(tmp_path, "openai", "gpt-6-astra")
    monkeypatch.setattr(
        "local_operator.providers.model_access.credentialed_chat_providers_here",
        lambda **_kwargs: None,  # "cannot tell"
    )
    assert _hops(tmp_path) == [("openai", "gpt-6-astra", False)]


def test_the_session_chat_model_never_changes_the_preferred_stage(tmp_path, credentials) -> None:
    _config(tmp_path, "radient", "auto")
    credentials.update({"anthropic", "radient"})
    for chat_model in (
        {"hosting": "openai", "model_name": "gpt-6-astra"},
        {"hosting": "deepseek", "model_name": "deepseek-flash"},
    ):
        hops = resolve_screening_chain(tmp_path, {**chat_model, "retry": {}})
        assert [(h.provider, h.model_id) for h in hops][:4] == list(SCREENING_EVALUATED_MODELS)


# -- the future "Screening model" setting (Aida 2026-11-04 (b)) ------------------


def test_a_stored_choice_from_the_list_leads_and_the_cascade_still_applies(
    tmp_path, credentials
) -> None:
    _config(tmp_path, "openai", "gpt-6-astra")
    credentials.update({"anthropic", "radient", "openai"})
    assert _hops(tmp_path, preferred=OPUS_R) == [
        (*OPUS_R, True),
        (*SONNET, True),
        (*SONNET_R, True),
        (*OPUS, True),
        ("openai", "gpt-6-astra", False),
    ]
    # A login flavour folds onto the table's own entry; it never adds one.
    assert _hops(tmp_path, preferred=("anthropic-key", "claude-opus-5-5"))[0] == (*OPUS, True)


@pytest.mark.parametrize(
    "choice", [("radient", "auto"), ("openai", "gpt-6-astra"), ("anthropic", "claude-haiku-5-5")]
)
def test_a_stored_choice_off_the_list_is_ignored(tmp_path, credentials, choice) -> None:
    _config(tmp_path, "openai", "gpt-6-astra")
    credentials.update({"anthropic", "radient", "openai"})
    assert _hops(tmp_path, preferred=choice)[0] == (*SONNET, True)


def test_an_unreachable_stored_choice_falls_through(tmp_path, credentials) -> None:
    _config(tmp_path, "openai", "gpt-6-astra")
    credentials.update({"openai"})
    assert _hops(tmp_path, preferred=OPUS) == [("openai", "gpt-6-astra", False)]


# -- QA matrix through the shipped cycle ------------------------------------------


def test_sonnet_direct_screens_and_is_preferred(tmp_path, monkeypatch, credentials) -> None:
    store, owner = _setup(tmp_path)
    _config(tmp_path, "openai", "gpt-6-astra")
    credentials.update({"anthropic", "radient", "openai"})
    wire = _Wire({"claude-sonnet-5-5": ("reply", DELIVER), "gpt-6-astra": ("reply", DELIVER)})
    _install_wire(monkeypatch, wire)

    result = _cycle(tmp_path, owner)

    assert result.disposition == "deliver" and result.screener_preferred is True
    assert result.receipt.model == "claude-sonnet-5-5"
    assert _sends(wire) == [SONNET]
    stored = store.receipt_for_message(owner, "bmsg_1")
    assert stored is not None and stored["screener_preferred"] is True


def test_sonnet_via_radient_screens_when_direct_is_down(tmp_path, monkeypatch, credentials) -> None:
    store, owner = _setup(tmp_path)
    _config(tmp_path, "anthropic", "claude-sonnet-5-5")
    credentials.update({"anthropic", "radient"})
    wire = _Wire(
        {"claude-sonnet-5-5": ("fail", 529), "anthropic/claude-sonnet-5.5": ("reply", DELIVER)}
    )
    _install_wire(monkeypatch, wire)

    result = _cycle(tmp_path, owner)

    assert result.disposition == "deliver" and result.screener_preferred is True
    assert result.receipt.model == "anthropic/claude-sonnet-5.5"
    assert _sends(wire) == [SONNET, SONNET_R]


def test_opus_screens_when_sonnet_is_rate_limited(tmp_path, monkeypatch, credentials) -> None:
    store, owner = _setup(tmp_path)
    _config(tmp_path, "openai", "gpt-6-astra")
    credentials.update({"anthropic", "radient", "openai"})
    wire = _Wire(
        {
            "claude-sonnet-5-5": ("fail", 429),
            "anthropic/claude-sonnet-5.5": ("fail", 429),
            "claude-opus-5-5": ("reply", DELIVER),
            "gpt-6-astra": ("reply", DELIVER),
        }
    )
    _install_wire(monkeypatch, wire)

    result = _cycle(tmp_path, owner)

    assert result.disposition == "deliver" and result.screener_preferred is True
    assert result.receipt.model == "claude-opus-5-5"
    assert _sends(wire) == [SONNET, SONNET_R, OPUS]


def test_no_anthropic_key_and_no_radient_credit_screens_on_the_default(
    tmp_path, monkeypatch, credentials
) -> None:
    # Radient is signed in but out of credit (402 on both preferred Radient
    # hops); there is no Anthropic key. The receiver's own default screens.
    store, owner = _setup(tmp_path)
    _config(tmp_path, "openai", "gpt-6-astra")
    credentials.update({"radient", "openai"})
    wire = _Wire(
        {
            "anthropic/claude-sonnet-5.5": ("fail", 402),
            "anthropic/claude-opus-5.5": ("fail", 402),
            "gpt-6-astra": ("reply", DELIVER),
        }
    )
    _install_wire(monkeypatch, wire)

    result = _cycle(tmp_path, owner)

    # A fallback ALLOW is DELIVERED normally, just never preferred.
    assert result.disposition == "deliver" and result.screener_preferred is False
    assert result.receipt.model == "gpt-6-astra"
    assert _sends(wire) == [SONNET_R, OPUS_R, ("openai", "gpt-6-astra")]
    stored = store.receipt_for_message(owner, "bmsg_1")
    assert stored is not None
    assert stored["model"] == "gpt-6-astra" and stored["screener_preferred"] is False
    assert _rows(
        tmp_path, "SELECT screener_preferred FROM screening_receipts WHERE message_id = 'bmsg_1'"
    ) == [(0,)]


def test_the_whole_cascade_failing_holds_and_notifies(tmp_path, monkeypatch, credentials) -> None:
    store, owner = _setup(tmp_path)
    _config(tmp_path, "radient", "auto", chain=["openai/gpt-6-astra"])
    credentials.update({"radient", "openai"})
    wire = _Wire(
        {
            "anthropic/claude-sonnet-5.5": ("fail", 402),
            "anthropic/claude-opus-5.5": ("fail", 402),
            "auto": ("fail", 503),
            "gpt-6-astra": ("fail", 429),
        }
    )
    _install_wire(monkeypatch, wire)

    result = _cycle(tmp_path, owner)

    assert result.disposition == "hold" and result.hold_code == SCREENING_MODEL_UNAVAILABLE
    assert result.receipt is None and result.screener_preferred is False
    assert _sends(wire) == [SONNET_R, OPUS_R, ("radient", "auto"), ("openai", "gpt-6-astra")]
    assert _rows(tmp_path, "SELECT 1 FROM screening_receipts") == []
    assert _rows(tmp_path, "SELECT 1 FROM screening_answer_holds") == []
    assert _rows(tmp_path, "SELECT hold_code FROM screening_holds") == [
        (SCREENING_MODEL_UNAVAILABLE,)
    ]


def test_nothing_to_screen_with_holds_before_any_send(tmp_path, monkeypatch, credentials) -> None:
    store, owner = _setup(tmp_path)
    (tmp_path / "config.yml").write_text("values: {}\n")
    wire = _Wire({})
    _install_wire(monkeypatch, wire)

    result = _cycle(tmp_path, owner)

    assert result.disposition == "hold" and result.hold_code == SCREENING_MODEL_UNAVAILABLE
    assert wire.sends == [] and store.current_screening_policy(owner) is None


def test_a_sender_triggered_refusal_on_a_preferred_hop_never_buys_the_fallback(
    tmp_path, monkeypatch, credentials
) -> None:
    # The fallback widens availability for ROUTE failures only: a content
    # refusal (400) on Sonnet stops the walk and never reaches gpt-6-astra.
    store, owner = _setup(tmp_path)
    _config(tmp_path, "openai", "gpt-6-astra")
    credentials.update({"anthropic", "openai"})
    wire = _Wire({"claude-sonnet-5-5": ("fail", 400), "gpt-6-astra": ("reply", DELIVER)})
    _install_wire(monkeypatch, wire)

    result = _cycle(tmp_path, owner)

    assert result.disposition == "hold" and result.hold_code == "provider_error"
    assert _sends(wire) == [SONNET]


def test_the_hold_is_infra_never_a_review_verdict() -> None:
    from local_operator.team_bridge.screening import _ANSWER_HOLD_CACHEABLE

    assert SCREENING_MODEL_UNAVAILABLE in HOLD_CODES
    assert SCREENING_MODEL_UNAVAILABLE not in _ANSWER_HOLD_CACHEABLE
    assert delivery_hold_class(SCREENING_MODEL_UNAVAILABLE) == "screening_unavailable"


# -- receipt reuse: a fallback ALLOW is never reused (Aida 2026-11-04 (a)) --------


def _reuse_pair(tmp_path, monkeypatch, credentials, *, preferred_route: bool) -> tuple[Any, _Wire]:
    store, owner = _setup(tmp_path)
    _config(tmp_path, "openai", "gpt-6-astra")
    credentials.update({"openai"} | ({"anthropic"} if preferred_route else set()))
    wire = _Wire({"claude-sonnet-5-5": ("reply", DELIVER), "gpt-6-astra": ("reply", DELIVER)})
    _install_wire(monkeypatch, wire)
    first = _cycle(tmp_path, owner, message_id="bmsg_1")
    second = _cycle(tmp_path, owner, message_id="bmsg_2")  # byte-identical body
    assert first.disposition == "deliver" and second.disposition == "deliver"
    return (store, owner, first, second), wire


def test_a_preferred_allow_is_reused_for_identical_bytes(
    tmp_path, monkeypatch, credentials
) -> None:
    (store, owner, first, second), wire = _reuse_pair(
        tmp_path, monkeypatch, credentials, preferred_route=True
    )
    assert _sends(wire) == [SONNET]  # the second message was a cache hit
    assert second.screener_preferred is True
    second_row = store.receipt_for_message(owner, "bmsg_2")
    assert second_row is not None and second_row["screener_preferred"] is True


def test_a_fallback_allow_is_never_reused_it_re_screens(tmp_path, monkeypatch, credentials) -> None:
    (store, owner, first, second), wire = _reuse_pair(
        tmp_path, monkeypatch, credentials, preferred_route=False
    )
    gpt = ("openai", "gpt-6-astra")
    assert _sends(wire) == [gpt, gpt]  # no cache hit: screened again
    assert first.screener_preferred is False and second.screener_preferred is False
    assert (
        store.consume_screening_receipt(
            owner,
            body_sha256=first.receipt.body_sha256,
            template_sha256=first.receipt.template_sha256,
            model="gpt-6-astra",
            admission_revision=(store.current_screening_policy(owner) or {})["revision"],
            envelope_digest=_rows(
                tmp_path, "SELECT envelope_digest FROM screening_receipts WHERE message_id='bmsg_1'"
            )[0][0],
        )
        is None
    )


def test_a_pre_upgrade_receipt_row_counts_as_not_preferred(
    tmp_path, monkeypatch, credentials
) -> None:
    (store, owner, first, _second), _wire = _reuse_pair(
        tmp_path, monkeypatch, credentials, preferred_route=True
    )
    import sqlite3

    from local_operator.team_bridge.store import state_db_path

    with sqlite3.connect(state_db_path(tmp_path)) as db:
        db.execute("UPDATE screening_receipts SET screener_preferred = NULL")
    stored = store.receipt_for_message(owner, "bmsg_1")
    assert stored is not None and stored["screener_preferred"] is False


# -- the owner notice and the table's single edit point ---------------------------


def test_owner_notice_is_one_keyed_item(tmp_path) -> None:
    from types import SimpleNamespace

    from local_operator.notifications.store import NotificationStore
    from local_operator.team_bridge.connector.wake import notify_screening_infra_hold

    owner = SimpleNamespace(endpoint_id="end_1")
    for _ in range(3):  # one call per held message
        asyncio.run(
            notify_screening_infra_hold(owner, tmp_path, "sess_cos", SCREENING_MODEL_UNAVAILABLE)
        )
    for other in ("model_hold", "provider_error", None):  # no-ops
        asyncio.run(notify_screening_infra_hold(owner, tmp_path, "sess_cos", other))

    items, _cursor = NotificationStore(tmp_path).list_items(state="all", session_id="sess_cos")
    assert len(items) == 1
    item = items[0]
    assert item["key"] == "team-bridge-screening-model:end_1" and item["priority"] == "high"
    for phrase in ("Claude Sonnet 5.5", "Anthropic", "Radient console", "default model"):
        assert phrase in item["body_md"]


def test_listing_a_pair_is_the_only_way_to_make_it_preferred(
    monkeypatch, tmp_path, credentials
) -> None:
    _config(tmp_path, "openai", "gpt-6-astra")
    credentials.add("openai")
    assert _hops(tmp_path) == [("openai", "gpt-6-astra", False)]
    monkeypatch.setattr(
        model_boundary,
        "SCREENING_EVALUATED_MODELS",
        SCREENING_EVALUATED_MODELS + (("openai", "gpt-6-astra"),),
    )
    assert _hops(tmp_path) == [("openai", "gpt-6-astra", True)]


# -- the B4b release contract (strict list; operator directive 2026-11-04 item 5) --


def test_release_eligibility_input_is_the_strict_list(tmp_path, monkeypatch, credentials) -> None:
    """``screener_preferred`` is what B4b's D4 reads: True ONLY for a listed pair.

    B4b (PR #3443) refuses to release a hold whose screener is not on its list
    and imports ``SCREENING_EVALUATED_MODELS`` for it; this pins the producer
    side here: a hold or bounce answered by a fallback screener carries
    ``screener_preferred = False`` on the result AND the stored receipt, so it
    can never be an auto-release input, whatever the verdict.
    """

    bounce = "VERDICT: bounce\nREASON: new_work_request: a new task for the team."
    store, owner = _setup(tmp_path)
    _config(tmp_path, "openai", "gpt-6-astra")
    credentials.update({"openai"})  # no preferred route at all
    wire = _Wire({"gpt-6-astra": ("reply", bounce)})
    _install_wire(monkeypatch, wire)

    result = _cycle(tmp_path, owner)

    assert result.disposition == "bounce"
    assert result.receipt.categories == ("new_work_request",)
    assert result.screener_preferred is False
    stored = store.receipt_for_message(owner, "bmsg_1")
    assert stored is not None and stored["screener_preferred"] is False
    for _provider, model_id in SCREENING_EVALUATED_MODELS:
        assert model_id != "gpt-6-astra"


def test_a_selection_built_outside_the_resolver_is_not_preferred() -> None:
    # Only the resolver may mark a hop preferred; a directly constructed one
    # (an adapter, a test seam) is a fallback until proven otherwise.
    from local_operator.team_bridge.model_boundary import DefaultSelection

    assert (
        DefaultSelection(provider="anthropic", model_id="claude-sonnet-5-5", spec=None).preferred
        is False
    )


@pytest.mark.parametrize("recovered", [True, False])
def test_same_message_unsent_fallback_allow_must_screen_again(
    tmp_path, monkeypatch, credentials, recovered
) -> None:
    """R1-M1/S-1: recovery is not a cache exemption for an UNSENT allow."""
    from local_operator.team_bridge.connector.wake import (
        RECORDED_SCREEN,
        recorded_read_verdict,
    )
    from tests.unit.team_bridge.test_screening import _item

    store, owner = _setup(tmp_path)
    _config(tmp_path, "openai", "gpt-6-astra")
    credentials.add("openai")
    wire = _Wire({"gpt-6-astra": ("reply", DELIVER)})
    _install_wire(monkeypatch, wire)
    first = _cycle(tmp_path, owner)
    assert first.disposition == "deliver" and first.screener_preferred is False
    assert store.recorded_intent(owner, "bmsg_1", "read") is None

    credentials.add("anthropic")
    wire2 = _Wire(
        {
            "claude-sonnet-5-5": ("reply", DELIVER) if recovered else ("fail", 503),
            "claude-opus-5-5": ("fail", 503),
            "gpt-6-astra": ("fail", 503),
        }
    )
    _install_wire(monkeypatch, wire2)
    assert recorded_read_verdict(owner, "bmsg_1", _item(), config_dir=tmp_path) == (
        RECORDED_SCREEN,
        None,
    )
    assert store.receipt_for_message(owner, "bmsg_1") is None
    second = _cycle(tmp_path, owner)
    assert _sends(wire2)[0] == SONNET
    if recovered:
        assert second.disposition == "deliver" and second.screener_preferred is True
        row = store.receipt_for_message(owner, "bmsg_1")
        assert row is not None and row["model"] == SONNET[1] and row["screener_preferred"]
    else:
        assert second.disposition == "hold"
        assert store.receipt_for_message(owner, "bmsg_1") is None
        # A second recovery cannot rediscover the retired weak allow.
        assert (
            recorded_read_verdict(owner, "bmsg_1", _item(), config_dir=tmp_path)[0]
            == RECORDED_SCREEN
        )


def test_pending_ack_preserves_fallback_tuple_without_another_screen(
    tmp_path, monkeypatch, credentials
) -> None:
    from local_operator.team_bridge.connector.wake import (
        RECORDED_VERDICT,
        recorded_read_verdict,
    )
    from tests.unit.team_bridge.test_screening import _item

    (store, owner, first, _second), wire = _reuse_pair(
        tmp_path, monkeypatch, credentials, preferred_route=False
    )
    store.record_ack_intent(owner, "bmsg_1", "read", first.receipt, "ref-original")
    credentials.add("anthropic")
    before = len(wire.sends)
    decision, stored = recorded_read_verdict(owner, "bmsg_1", _item(), config_dir=tmp_path)
    assert decision == RECORDED_VERDICT and stored["model"] == "gpt-6-astra"
    assert (
        store.retire_unsent_fallback_allow(owner, "bmsg_1", body_sha256=first.receipt.body_sha256)
        is False
    )
    assert len(wire.sends) == before  # ACK may already be committed at the hub


def test_bare_runner_cannot_manufacture_preferred_provenance() -> None:
    from local_operator.team_bridge.screening import _answered_preferred

    class AdversarialBareRunner:
        answered_preferred = True
        answered_hop = model_boundary.DefaultSelection(
            provider=SONNET[0], model_id=SONNET[1], spec=None, preferred=True
        )

    assert _answered_preferred(AdversarialBareRunner(), SONNET[1]) is False
    assert _answered_preferred(object(), SONNET_R[1]) is False


def test_listed_name_in_ordinary_cascade_is_not_preferred(tmp_path, credentials) -> None:
    # No locally evidenced preferred credential: the receiver's ordinary
    # cascade must not manufacture provenance by returning a familiar name.
    _config(tmp_path, SONNET[0], SONNET[1])
    hops = resolve_screening_chain(tmp_path)
    assert len(hops) == 1 and hops[0].preferred is False


def test_direct_cycle_retry_can_replace_an_unsent_fallback_receipt(
    tmp_path, monkeypatch, credentials
) -> None:
    # The core is independently safe even if a caller bypasses connector
    # recovery (e.g. the batch path): no stale write-once receipt may block
    # the preferred result from being recorded for these same bytes.
    store, owner = _setup(tmp_path)
    _config(tmp_path, "openai", "gpt-6-astra")
    credentials.add("openai")
    _install_wire(monkeypatch, _Wire({"gpt-6-astra": ("reply", DELIVER)}))
    first = _cycle(tmp_path, owner)
    assert first.screener_preferred is False and first.disposition == "deliver"
    credentials.add("anthropic")
    wire = _Wire({SONNET[1]: ("reply", DELIVER)})
    _install_wire(monkeypatch, wire)
    second = _cycle(tmp_path, owner)
    assert second.disposition == "deliver" and second.screener_preferred
    assert _sends(wire) == [SONNET]
    row = store.receipt_for_message(owner, "bmsg_1")
    assert row is not None and row["model"] == SONNET[1] and row["screener_preferred"]


def test_bare_runner_preferred_looking_allow_stays_nonpreferred_in_store(tmp_path) -> None:
    from tests.unit.team_bridge.test_screening import Policy, _run

    class BareRunner:
        # Neither a familiar model name nor this self-asserted flag is trusted.
        answered_preferred = True

        def __call__(self, prompt, *, default_model):
            return DELIVER, SONNET[1]

    store, owner = _setup(tmp_path)
    result = _run(
        owner,
        tmp_path,
        BareRunner(),
        policy=Policy(
            provider=SONNET[0],
            default_model=SONNET[1],
            approved_serving_identities=(SONNET[1],),
        ),
    )
    assert result.disposition == "deliver" and result.screener_preferred is False
    row = store.receipt_for_message(owner, "bmsg_1")
    assert row is not None and row["model"] == SONNET[1]
    assert row["screener_preferred"] is False


@pytest.mark.parametrize("preference", [None, 0], ids=["legacy-null", "ordinary-false"])
def test_unsent_model_allow_without_preference_is_still_retired(
    tmp_path, monkeypatch, credentials, preference
) -> None:
    """S-11 negative: exempt only owner-release template, not all NULL provenance."""

    import sqlite3

    from local_operator.team_bridge.store import state_db_path

    store, owner = _setup(tmp_path)
    _config(tmp_path, "openai", "gpt-6-astra")
    credentials.add("openai")
    wire = _Wire({"gpt-6-astra": ("reply", DELIVER)})
    _install_wire(monkeypatch, wire)
    first = _cycle(tmp_path, owner)
    assert first.disposition == "deliver" and first.receipt is not None
    with sqlite3.connect(state_db_path(tmp_path)) as conn:
        conn.execute(
            "UPDATE screening_receipts SET screener_preferred = ? WHERE message_id = 'bmsg_1'",
            (preference,),
        )
    assert (
        store.retire_unsent_fallback_allow(owner, "bmsg_1", body_sha256=first.receipt.body_sha256)
        is True
    )
    assert store.receipt_for_message(owner, "bmsg_1") is None
    before = len(wire.sends)
    assert _cycle(tmp_path, owner).disposition == "deliver"
    assert len(wire.sends) == before + 1  # genuine fallback really screens again
