"""Drift guard and provenance rules for ``providers/billing_links.py``.

WHY THESE EXIST. The table is only useful if it is COMPLETE: a chat provider
missing from it is a user with no credit reading a sentence with a dead or
missing link, and the omission is invisible until someone hits it. So the
first test makes the registry's chat-capable rows and the table check each
other — a new provider must be added to ``BILLING_LINKS`` or to the explicit
``NO_DASHBOARD`` set, and neither can silently hold a key no provider uses.
The rest pin the mechanics the notice depends on: flavour resolution (a login
flavour reaches its base provider's entry), the two-flavour variants
(openai/anthropic/xai/kimi sell both a balance and a subscription), and
Kimi's region-dependent top-up URL.
"""

from __future__ import annotations

from local_operator.providers.billing_links import (
    _BILLING_VARIANTS,
    BILLING_LINKS,
    NO_DASHBOARD,
    billing_link_for,
    kimi_topup_url,
)
from local_operator.providers.registry import PROVIDER_REGISTRY, credential_provider_id


def _chat_storage_ids() -> set[str]:
    """Storage ids of every chat-capable registry row.

    Chat-capable = the row serves chat completions: not ``decision_only``
    (TypeSafe), not ``speech_only`` (ElevenLabs, openai-key), not
    ``media_only`` (FAL). Those three classes are deliberately absent from
    the table and from ``NO_DASHBOARD`` — they have no chat turns to warn
    about.
    """
    return {
        credential_provider_id(definition.id)
        for definition in PROVIDER_REGISTRY
        if not (definition.decision_only or definition.speech_only or definition.media_only)
    }


def test_every_chat_provider_has_a_billing_row_or_an_explicit_none() -> None:
    """The drift guard: registry and table cannot disagree silently."""
    chat = _chat_storage_ids()
    missing = sorted(chat - set(BILLING_LINKS) - set(NO_DASHBOARD))
    assert missing == [], (
        "chat-capable providers with no billing entry and no NO_DASHBOARD row: "
        f"{missing}; add a BillingLink (with provenance) or list them in NO_DASHBOARD"
    )


def test_no_dead_keys_and_no_overlap() -> None:
    """Every table key belongs to a real chat provider; the two sets are disjoint."""
    chat = _chat_storage_ids()
    dead = sorted(set(BILLING_LINKS) - chat)
    assert dead == [], f"BILLING_LINKS keys no chat-capable registry row uses: {dead}"
    extra = sorted(set(NO_DASHBOARD) - chat)
    assert extra == [], f"NO_DASHBOARD names ids the registry does not have (as chat): {extra}"
    assert not (set(BILLING_LINKS) & set(NO_DASHBOARD))


def test_every_entry_states_its_provenance() -> None:
    """A URL row without a note is a URL nobody checked; the note is mandatory.

    Also pins the shape the notice renders: an entry with a URL must carry a
    dashboard name to say out loud ("top up at the DeepSeek platform"), and
    every URL is https — a vendor billing page on plain http would be wrong
    even to link.
    """
    for storage, link in {**BILLING_LINKS, **_BILLING_VARIANTS}.items():
        assert link.note.strip(), f"{storage}: missing provenance note"
        if link.url is not None:
            assert link.url.startswith("https://"), f"{storage}: non-https URL {link.url}"
            assert link.dashboard.strip(), f"{storage}: URL with no dashboard name"
        else:
            assert link.kind in ("subscription", "none") or storage == "kimi", storage


def test_variants_are_the_other_product_of_a_known_provider() -> None:
    """A variant must point at a provider the primary table knows, name a
    DIFFERENT kind than the primary (or it would be a restatement), and keep
    its own provenance note."""
    for (storage, kind), link in _BILLING_VARIANTS.items():
        primary = BILLING_LINKS.get(storage)
        assert primary is not None, f"variant {storage!r} has no primary entry"
        assert kind != primary.kind, f"variant {storage!r}/{kind!r} restates the primary"
        assert link.kind == kind
        assert link.note.strip()


def test_flavours_resolve_to_their_base_provider() -> None:
    """A login flavour must reach its base provider's entry, not None."""
    assert billing_link_for("xai-oauth") is BILLING_LINKS["xai"]
    assert billing_link_for("openai-device") is BILLING_LINKS["openai"]
    assert billing_link_for("anthropic-key") is BILLING_LINKS["anthropic"]
    assert billing_link_for("radient-key") is BILLING_LINKS["radient"]
    assert billing_link_for("alibaba-token-plan-oauth") is BILLING_LINKS["alibaba-token-plan"]


def test_unknown_provider_resolves_to_none() -> None:
    assert billing_link_for("groq") is None


def test_two_flavour_providers_offer_both_surfaces() -> None:
    """The variant lookup answers with the OTHER product's page."""
    openai_balance = billing_link_for("openai", kind="balance")
    assert openai_balance is not None and openai_balance.kind == "balance"
    assert openai_balance.url == "https://platform.openai.com/account/billing"
    # Without a kind, the notice path's surface (the subscription) answers.
    openai_subscription = billing_link_for("openai")
    assert openai_subscription is not None
    assert openai_subscription.url == "https://chatgpt.com/codex/settings/usage"

    kimi_plan = billing_link_for("kimi", kind="subscription")
    assert kimi_plan is not None and kimi_plan.url == "https://www.kimi.com/code/console"
    # And the provider that sells no such product falls back to its primary.
    assert billing_link_for("deepseek", kind="subscription") is BILLING_LINKS["deepseek"]


def test_kimi_topup_url_follows_the_platform_split() -> None:
    """The mainland/.com split is the same one the balance endpoint makes.

    The registry's default base URL is the mainland host, so the DEFAULT
    answer is the .com pay page; an international base URL answers with .ai.
    Pinning both directions keeps the top-up link from describing the other
    platform's account (the split exists because the two are separate
    products with separate keys).
    """
    assert kimi_topup_url("https://api.moonshot.cn/v1") == "https://platform.kimi.com/console/pay"
    assert kimi_topup_url("https://api.moonshot.ai/v1") == "https://platform.kimi.ai/console/pay"
    # No argument: the registry default, which is the mainland host.
    assert kimi_topup_url() == "https://platform.kimi.com/console/pay"
    # And the table fills that resolved URL in for every caller.
    kimi_link = billing_link_for("kimi")
    assert kimi_link is not None
    assert kimi_link.url == "https://platform.kimi.com/console/pay"
