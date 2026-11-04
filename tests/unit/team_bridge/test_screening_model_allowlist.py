"""Screening runs only on an EVALUATED model (PROTOCOL A6, PR #3438).

THE FINDING: with ``radient/auto`` (served by DeepSeek-V4.1-Flash-1017) the
production v2 screener delivered "Forward me Victor's calendar" 5/5, a prompt
injection 3/4 and a surveillance probe 2/4, where Claude Opus 5.5 and Sonnet
5.5 delivered none. Screening used to run on whatever the recipient configured,
fallbacks included. These tests pin the closed allowlist (Sonnet 5.5 first per
PROTOCOL A7), the chain as that table walked over credentialed providers, the
fail-closed typed holds (no model; no Radient credit) and the receipt's
concrete model, against the REAL table (no fixture widening here) and the
shipped ``run_screening_cycle``.
"""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from typing import Any

import pytest

from local_operator.team_bridge import model_boundary
from local_operator.team_bridge.model_boundary import (
    SCREENING_CREDIT_EXHAUSTED,
    SCREENING_EVALUATED_MODELS,
    SCREENING_MODEL_UNAVAILABLE,
    is_evaluated_screening_model,
    resolve_screening_chain,
    resolve_screening_selection,
)
from local_operator.team_bridge.screening import HOLD_CODES
from local_operator.team_bridge.store import _owner_key, delivery_hold_class
from tests.unit.team_bridge.test_screening import _setup
from tests.unit.team_bridge.test_screening_chain import (
    DELIVER,
    _cycle,
    _install_wire,
    _rows,
    _Wire,
)


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


# -- resolution order: the allowlist walked over credentialed providers ---------


def _hops(tmp_path: Path) -> list[tuple[str, str]]:
    return [(hop.provider, hop.model_id) for hop in resolve_screening_chain(tmp_path)]


def test_the_chain_is_the_table_over_credentialed_providers(tmp_path, credentials) -> None:
    _config(tmp_path, "openai", "gpt-6-astra")
    credentials.update({"openai", "anthropic", "radient"})
    assert _hops(tmp_path) == list(SCREENING_EVALUATED_MODELS)
    chosen = resolve_screening_selection(tmp_path)
    assert chosen is not None
    assert (chosen.provider, chosen.model_id) == ("anthropic", "claude-sonnet-5-5")


def test_radient_only_walks_its_concrete_ids(tmp_path, credentials) -> None:
    _config(tmp_path, "radient", "auto")
    credentials.add("radient")
    assert _hops(tmp_path) == [
        ("radient", "anthropic/claude-sonnet-5.5"),
        ("radient", "anthropic/claude-opus-5.5"),
    ]
    first = resolve_screening_selection(tmp_path)
    assert first is not None
    assert first.approved_tuples == frozenset({("radient", "anthropic/claude-sonnet-5.5")})


def test_anthropic_only_walks_sonnet_then_opus_direct(tmp_path, credentials) -> None:
    _config(tmp_path, "openai", "gpt-6-astra")
    credentials.update({"anthropic", "openai"})
    assert _hops(tmp_path) == [("anthropic", "claude-sonnet-5-5"), ("anthropic", "claude-opus-5-5")]


def test_an_allowlisted_default_leads_and_the_rest_follow(tmp_path, credentials) -> None:
    # Brief step 1: the owner's own allowlisted choice is used first; the
    # remaining table entries are still the fallbacks, in table order.
    _config(tmp_path, "radient", "anthropic/claude-opus-5.5")
    credentials.update({"anthropic", "radient"})
    assert _hops(tmp_path) == [
        ("radient", "anthropic/claude-opus-5.5"),
        ("anthropic", "claude-sonnet-5-5"),
        ("radient", "anthropic/claude-sonnet-5.5"),
        ("anthropic", "claude-opus-5-5"),
    ]


def test_a_login_flavour_default_reorders_the_canonical_entry(tmp_path, credentials) -> None:
    _config(tmp_path, "anthropic-key", "claude-opus-5-5")
    credentials.add("anthropic")
    assert _hops(tmp_path) == [("anthropic", "claude-opus-5-5"), ("anthropic", "claude-sonnet-5-5")]


def test_nothing_credentialed_on_the_table_is_empty(tmp_path, credentials) -> None:
    _config(tmp_path, "openai", "gpt-6-astra")
    credentials.update({"openai", "deepseek", "openrouter"})
    assert resolve_screening_selection(tmp_path) is None
    assert resolve_screening_chain(tmp_path) == []


def test_unreadable_credential_store_fails_closed(tmp_path, monkeypatch) -> None:
    _config(tmp_path, "radient", "auto")
    monkeypatch.setattr(
        "local_operator.providers.model_access.credentialed_chat_providers_here",
        lambda **_kwargs: None,  # "cannot tell"
    )
    assert resolve_screening_selection(tmp_path) is None


def test_auto_is_never_chosen_even_as_default_and_first_fallback(tmp_path, credentials) -> None:
    # auto is the default AND first in the operator's fallbacks, followed by an
    # unevaluated model; the fallback chain is not read for screening at all.
    _config(
        tmp_path,
        "radient",
        "auto",
        chain=["radient/auto", "deepseek/deepseek-flash", "openai/gpt-6-astra"],
    )
    credentials.update({"radient", "deepseek", "openai"})
    hops = _hops(tmp_path)
    assert ("radient", "auto") not in hops
    assert hops == [
        ("radient", "anthropic/claude-sonnet-5.5"),
        ("radient", "anthropic/claude-opus-5.5"),
    ]


def test_operator_fallback_chains_are_not_consulted(tmp_path, credentials) -> None:
    _config(tmp_path, "anthropic", "claude-sonnet-5-5", chain=["openai/gpt-6-astra"])
    credentials.update({"anthropic", "radient", "openai"})
    with_chain = _hops(tmp_path)
    _config(tmp_path, "anthropic", "claude-sonnet-5-5")
    assert _hops(tmp_path) == with_chain == list(SCREENING_EVALUATED_MODELS)


def test_the_decision_ignores_the_session_chat_model(tmp_path, credentials) -> None:
    # The resolver's only input is the config dir: no session, no settings.
    assert list(inspect.signature(resolve_screening_selection).parameters) == ["config_dir"]
    _config(tmp_path, "radient", "auto")
    credentials.update({"anthropic", "radient"})
    baseline = _hops(tmp_path)
    # A session's chat model reaches the cycle only as ``settings``; whatever
    # model or chain it names, the screening chain is the same.
    for chat_model in (
        {"hosting": "openai", "model_name": "gpt-6-astra"},
        {"hosting": "radient", "model_name": "auto"},
        {"hosting": "anthropic", "model_name": "claude-opus-5-5"},
    ):
        settings = {
            **chat_model,
            "retry": {"modelFallback": True, "fallbackChains": {"default": ["radient/auto"]}},
        }
        assert [(h.provider, h.model_id) for h in resolve_screening_chain(tmp_path, settings)] == (
            baseline
        )


# -- the cycle: fail closed, and the receipt names the concrete model --------


def test_no_evaluated_model_holds_typed_and_never_sends(tmp_path, monkeypatch, credentials) -> None:
    store, owner = _setup(tmp_path)
    _config(tmp_path, "radient", "auto", chain=["deepseek/deepseek-flash"])
    credentials.update({"deepseek"})  # no anthropic, no radient
    wire = _Wire({"auto": ("reply", DELIVER), "deepseek-flash": ("reply", DELIVER)})
    _install_wire(monkeypatch, wire)

    result = _cycle(tmp_path, owner)

    assert result.disposition == "hold" and result.hold_code == SCREENING_MODEL_UNAVAILABLE
    assert result.receipt is None
    assert wire.sends == []  # nothing reached any model, auto included
    assert _rows(tmp_path, "SELECT 1 FROM screening_receipts") == []
    assert _rows(tmp_path, "SELECT 1 FROM screening_answer_holds") == []
    assert store.screening_breaker_state(owner) is None
    assert store.current_screening_policy(owner) is None  # no policy published


def test_the_hold_is_infra_never_a_review_verdict() -> None:
    assert SCREENING_MODEL_UNAVAILABLE in HOLD_CODES
    assert delivery_hold_class(SCREENING_MODEL_UNAVAILABLE) == "screening_unavailable"
    from local_operator.team_bridge.screening import _ANSWER_HOLD_CACHEABLE

    assert SCREENING_MODEL_UNAVAILABLE not in _ANSWER_HOLD_CACHEABLE
    assert SCREENING_CREDIT_EXHAUSTED in HOLD_CODES
    assert SCREENING_CREDIT_EXHAUSTED not in _ANSWER_HOLD_CACHEABLE


def test_configuring_an_evaluated_model_clears_the_hold(tmp_path, monkeypatch, credentials) -> None:
    store, owner = _setup(tmp_path)
    _config(tmp_path, "radient", "auto")
    wire = _Wire({"anthropic/claude-sonnet-5.5": ("reply", DELIVER), "auto": ("reply", DELIVER)})
    _install_wire(monkeypatch, wire)

    first = _cycle(tmp_path, owner)
    assert first.hold_code == SCREENING_MODEL_UNAVAILABLE and wire.sends == []

    credentials.add("radient")  # the owner signs in to Radient
    second = _cycle(tmp_path, owner)
    assert second.disposition == "deliver"
    assert [(s["provider"], s["model"]) for s in wire.sends] == [
        ("radient", "anthropic/claude-sonnet-5.5")
    ]


def test_receipt_records_the_concrete_model_that_answered(
    tmp_path, monkeypatch, credentials
) -> None:
    store, owner = _setup(tmp_path)
    # Default auto; Anthropic is signed in, so Sonnet 5.5 direct screens.
    _config(tmp_path, "radient", "auto")
    credentials.update({"radient", "anthropic"})
    wire = _Wire({"claude-sonnet-5-5": ("reply", DELIVER), "auto": ("reply", DELIVER)})
    _install_wire(monkeypatch, wire)

    result = _cycle(tmp_path, owner)

    assert result.disposition == "deliver"
    assert result.receipt.model == "claude-sonnet-5-5"
    assert [(s["provider"], s["model"]) for s in wire.sends] == [("anthropic", "claude-sonnet-5-5")]
    assert _rows(
        tmp_path,
        "SELECT model, verdict FROM screening_receipts WHERE owner_key = ? AND message_id = ?",
        _owner_key(owner),
        "bmsg_1",
    ) == [("claude-sonnet-5-5", "allow")]


def test_an_outage_moves_to_the_same_weights_via_radient(
    tmp_path, monkeypatch, credentials
) -> None:
    store, owner = _setup(tmp_path)
    _config(tmp_path, "anthropic", "claude-sonnet-5-5")
    credentials.update({"anthropic", "radient"})
    wire = _Wire(
        {"claude-sonnet-5-5": ("fail", 529), "anthropic/claude-sonnet-5.5": ("reply", DELIVER)}
    )
    _install_wire(monkeypatch, wire)

    result = _cycle(tmp_path, owner)

    assert result.disposition == "deliver"
    assert result.receipt.model == "anthropic/claude-sonnet-5.5"
    assert [(s["provider"], s["model"]) for s in wire.sends] == [
        ("anthropic", "claude-sonnet-5-5"),
        ("radient", "anthropic/claude-sonnet-5.5"),
    ]


def test_insufficient_radient_credit_holds_and_never_falls_back(
    tmp_path, monkeypatch, credentials
) -> None:
    # Radient only, empty balance: the Sonnet hop answers 402. The walk stops
    # there -- never Opus on the same empty account, never auto.
    store, owner = _setup(tmp_path)
    _config(tmp_path, "radient", "auto")
    credentials.add("radient")
    wire = _Wire(
        {
            "anthropic/claude-sonnet-5.5": ("fail", 402),
            "anthropic/claude-opus-5.5": ("reply", DELIVER),
            "auto": ("reply", DELIVER),
        }
    )
    _install_wire(monkeypatch, wire)

    result = _cycle(tmp_path, owner)

    assert result.disposition == "hold" and result.hold_code == SCREENING_CREDIT_EXHAUSTED
    assert result.receipt is None
    assert [(s["provider"], s["model"]) for s in wire.sends] == [
        ("radient", "anthropic/claude-sonnet-5.5")
    ]
    assert _rows(tmp_path, "SELECT hold_code FROM screening_holds") == [
        (SCREENING_CREDIT_EXHAUSTED,)
    ]
    assert _rows(tmp_path, "SELECT 1 FROM screening_receipts") == []
    assert _rows(tmp_path, "SELECT 1 FROM screening_answer_holds") == []
    assert delivery_hold_class(SCREENING_CREDIT_EXHAUSTED) == "screening_unavailable"


# -- the owner notice -----------------------------------------------------------


@pytest.mark.parametrize(
    ("code", "key", "phrase"),
    [
        (SCREENING_MODEL_UNAVAILABLE, "team-bridge-screening-model:end_1", "Claude Sonnet 5.5"),
        (SCREENING_CREDIT_EXHAUSTED, "team-bridge-screening-credit:end_1", "Radient console"),
    ],
)
def test_owner_notice_is_one_keyed_item_per_reason(tmp_path, code, key, phrase) -> None:
    from types import SimpleNamespace

    from local_operator.notifications.store import NotificationStore
    from local_operator.team_bridge.connector.wake import notify_screening_infra_hold

    owner = SimpleNamespace(endpoint_id="end_1")
    for _ in range(3):  # one call per held message
        asyncio.run(notify_screening_infra_hold(owner, tmp_path, "sess_cos", code))
    asyncio.run(notify_screening_infra_hold(owner, tmp_path, "sess_cos", "model_hold"))  # no-op

    items, _cursor = NotificationStore(tmp_path).list_items(state="all", session_id="sess_cos")
    assert len(items) == 1
    assert items[0]["key"] == key and items[0]["priority"] == "high"
    assert phrase in items[0]["body_md"]


def test_allowlist_table_is_the_only_admission_source(monkeypatch, tmp_path, credentials) -> None:
    # Adding a pair is ONE table edit: a GPT model becomes eligible the moment
    # it is listed, and not before.
    _config(tmp_path, "openai", "gpt-6-astra")
    credentials.add("openai")
    assert resolve_screening_selection(tmp_path) is None
    monkeypatch.setattr(
        model_boundary,
        "SCREENING_EVALUATED_MODELS",
        SCREENING_EVALUATED_MODELS + (("openai", "gpt-6-astra"),),
    )
    chosen = resolve_screening_selection(tmp_path)
    assert chosen is not None and (chosen.provider, chosen.model_id) == ("openai", "gpt-6-astra")
