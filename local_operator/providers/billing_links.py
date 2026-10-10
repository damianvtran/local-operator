"""Provider -> billing-dashboard table: where a user fixes an empty account.

WHY THIS EXISTS. Two features ask the same question with different evidence —
"the account cannot send, where does the user go?" — and each would otherwise
spell its own list of vendor URLs. The first is the pre-emptive quota notice
(``providers/quota_notice.py`` and ``server/routes/desktop_quota.py``): it
fires BEFORE a send is refused, from the usage endpoints' own numbers. The
second is post-failure guidance (the Radient out-of-credits work), which sees
a refusal or an HTTP 402. A URL is a claim about a vendor's dashboard: two
hand-kept copies drift, and a dead link inside a "top up here" sentence is
worse than no link at all, because it spends the one click the notice
promised. So the destinations live here once, with the provenance of every
one of them stated, and both consumers import this module.

KEYED BY CREDENTIAL STORAGE ID (``registry.credential_provider_id``): the id
the AuthStore row, the usage cache and the fetchers all agree on, so a login
FLAVOUR (``xai-oauth``, ``anthropic-key``, ``openai-device``) resolves to the
same entry as its base provider. A provider that sells BOTH a pay-as-you-go
balance and a subscription — openai, anthropic, xai, kimi — is the one case
the shape cannot express in a single kind, and the URL a fix-landing surface
needs differs by WHICH product is empty. Those get a primary entry plus a
variant under ``_BILLING_VARIANTS``, resolved by :func:`billing_link_for` with
the kind the caller established from the evidence (the notice derives it from
the report shape: denominator-less balances are "balance", measured windows
are "subscription"). The variant map also serves the post-failure consumer,
which reaches the API-key case the notice cannot: API-key openai, anthropic,
google, mistral and xai have no usage fetcher, so they can never produce a
pre-emptive notice at all.

DRIFT GUARD: ``tests/unit/providers/test_billing_links.py`` requires every
chat-capable ``PROVIDER_REGISTRY`` id to appear here or in
:data:`NO_DASHBOARD`. A new chat provider whose billing page nobody recorded
must be an explicit decision, not an omission discovered when a user with no
credit is shown a sentence with no link in it.

PROVENANCE. Every URL was checked 2026-10-09 (as first recorded in the
pre-emptive-notice design doc), with ``curl -L`` where the host answered at
all — some hosts sit behind a bot wall that answers 403 to curl and can only
be confirmed by a human click, and those are marked ``verified=False``. The
comments below carry the design's per-URL findings verbatim in substance; the
one shared rule is that a path which redirects to a login page PROVES NOTHING
(unknown paths redirect too), so it is unverified even when the redirect
looks right.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal
from urllib.parse import urlsplit

from local_operator.providers.registry import (
    credential_provider_id,
    get_provider_definition,
)

#: What the provider bills: a pay-as-you-go balance, a subscription/plan, the
#: Radient console (its own thing — free signup grants and credit floors make
#: its remedy neither of the other two), or nothing to bill at all.
BillingKind = Literal["balance", "subscription", "radient", "none"]


@dataclass(frozen=True)
class BillingLink:
    """One provider's billing surface, as the two consumers need it.

    ``kind``/``url``/``dashboard`` describe the surface the PRE-EMPTIVE NOTICE
    uses — the product the provider's usage endpoint actually reports. That is
    the subscription for openai/anthropic/xai (their API-key routes have no
    fetcher, so a notice can only ever describe the plan), the API-key balance
    for deepseek/openrouter/kimi, and the balance journal for Radient's own
    states. ``dashboard`` is display copy, not a hostname: it is what a
    sentence can say instead of a URL ("top up at the DeepSeek platform").

    ``verified`` records whether SOMEONE confirmed the URL behind the bot wall
    at the time of the sweep, and ``note`` is the provenance the design doc
    carried for it. Neither is read by code — they exist so the next person
    editing a URL knows what was actually checked and when, because the
    failure mode here is silent: a dashboard URL that 404s only shows up in
    front of a user who is already out of credit.
    """

    kind: BillingKind
    url: str | None
    dashboard: str
    verified: bool
    note: str


#: The primary surface per credential storage id. Radient is its own kind
#: because its remedy text is NOT authored here at all — it is
#: ``radient_recovery.recovery_line``'s sentence — and its "billing" is a
#: signup-grant/credit-floor system, not a balance or a plan.
BILLING_LINKS: dict[str, BillingLink] = {
    "radient": BillingLink(
        kind="radient",
        url="https://console.radienthq.com/dashboard/billing",
        dashboard="the Radient console",
        # The paths are constants in agent-server: ``/dashboard/billing`` is
        # ``billingURL()``'s literal (internal/services/auto_reload_service.go:189)
        # and ``/dashboard/verification`` is ``signupClaimPath``
        # (internal/controllers/me_verification.go:24) — both on main ``add2e7e``.
        # The live host redirects unknown paths to login with the callback
        # preserved, and unknown paths redirect too, so the redirect alone
        # proves nothing — hence unverified. The verification page is not
        # repeated here: ``radient_recovery`` already quotes it as
        # ``CLAIM_URL``.
        verified=False,
        note=(
            "dashboard paths are agent-server main constants "
            "(auto_reload_service.go:189, me_verification.go:24); the live host "
            "redirects any unknown path to login, so the redirect proves nothing "
            "(checked 2026-10-09)"
        ),
    ),
    "anthropic": BillingLink(
        kind="subscription",
        url="https://claude.ai/settings/usage",
        dashboard="Claude settings",
        # Unverified: 403 bot wall. The pricing fallback claude.com/pricing
        # returned 200 but is a marketing page, not the account's usage view.
        verified=False,
        note=(
            "claude.ai/settings/usage unverified (403 to curl); fallback claude.com/pricing "
            "answered 200. API-key balances live at platform.claude.com/settings/billing "
            "(VERIFIED: console.anthropic.com redirects to it and a bogus sibling path 404s)"
        ),
    ),
    "openai": BillingLink(
        kind="subscription",
        url="https://chatgpt.com/codex/settings/usage",
        dashboard="ChatGPT usage settings",
        # Unverified: 403. OpenAI's help centre only says "settings -> usage"
        # without a URL, so the path is our best reading, not a confirmation.
        verified=False,
        note=(
            "chatgpt.com/codex/settings/usage unverified (403). API-key balances live at "
            "platform.openai.com/account/billing, documented in OpenAI's prepaid-billing "
            "help article (curl 403)"
        ),
    ),
    "deepseek": BillingLink(
        kind="balance",
        url="https://platform.deepseek.com/top_up",
        dashboard="the DeepSeek platform",
        # Unverified: 403 on every path tried. Search results confirm a
        # "Top up" section exists on platform.deepseek.com.
        verified=False,
        note=(
            "platform.deepseek.com/top_up unverified (403 on every path); "
            "'Top up' section confirmed by search"
        ),
    ),
    "xai": BillingLink(
        kind="subscription",
        # grok.com loads, but no usage page for the OAuth sign-in was found —
        # the design's own finding. The API-key balance surface exists and is
        # the variant below; the notice path cannot reach it (`xai` reports
        # only through the OAuth fetcher).
        url=None,
        dashboard="",
        verified=False,
        note=(
            "no dashboard found for the Grok OAuth sign-in (grok.com loads, no usage page). "
            "API-key balances: console.x.ai/team/default/billing, documented in xAI's billing "
            "docs (curl 403)"
        ),
    ),
    "zai": BillingLink(
        kind="subscription",
        url="https://z.ai/manage-apikey/billing",
        dashboard="Z.AI",
        # The API reports only a coding-PLAN quota, so the plan surface is the
        # primary one. The SPA answers 200 for any path, so neither URL is
        # verified; z.ai/subscribe is the plan-purchase page.
        verified=False,
        note="SPA answers 200 for any path, so unverified; plan purchase page is z.ai/subscribe",
    ),
    "kimi": BillingLink(
        kind="balance",
        # Resolved by :func:`kimi_topup_url` at lookup time, because the
        # platform (mainland vs international) is a runtime fact of the
        # registry's base_url — the same split `usage.moonshot_balance_target`
        # makes for the balance endpoint itself. The old
        # platform.moonshot.cn/.ai/console/pay URLs redirect to these with the
        # path kept.
        url=None,
        dashboard="the Kimi platform",
        verified=False,
        note=(
            "region top-up page: a .cn balance host -> platform.kimi.com/console/pay, otherwise "
            "platform.kimi.ai/console/pay; old moonshot.cn/.ai/console/pay redirect with path "
            "kept. Coding-plan console (subscription variant): www.kimi.com/code/console "
            "(VERIFIED: a bogus /code/* path redirects to /)"
        ),
    ),
    "google": BillingLink(
        kind="balance",
        url="https://ai.google.dev/gemini-api/docs/billing",
        dashboard="the Gemini API billing docs",
        # Verified: the docs page answers 200. aistudio.google.com/billing
        # redirects to sign-in for any path, so it is not discriminating.
        verified=True,
        note=(
            "ai.google.dev/gemini-api/docs/billing answered 200; "
            "aistudio.google.com/billing is not discriminating"
        ),
    ),
    "mistral": BillingLink(
        kind="balance",
        url="https://admin.mistral.ai/plateforme/billing",
        dashboard="the Mistral admin console",
        verified=False,
        note="admin.mistral.ai/plateforme/billing redirects to login for any path (unverified)",
    ),
    "openrouter": BillingLink(
        kind="balance",
        url="https://openrouter.ai/settings/credits",
        dashboard="the OpenRouter credits page",
        # Unverified: redirects to sign-in for any path. Separately (design
        # doc, section 3) `fetch_openrouter` reports only the key's own spend
        # cap, so an account with zero credits and no key limit reads
        # "unknown" and no notice fires for it — an endpoint gap, not a URL
        # one.
        verified=False,
        note="openrouter.ai/settings/credits redirects to sign-in for any path (unverified)",
    ),
    "alibaba": BillingLink(
        kind="balance",
        url="https://modelstudio.console.alibabacloud.com",
        dashboard="the Alibaba Model Studio console",
        verified=False,
        note="root answered 200; billing-intl.console.alibabacloud.com is unreachable",
    ),
    "alibaba-token-plan": BillingLink(
        kind="subscription",
        # Quoted from the registry's own login URL rather than spelled again
        # (registry.py, the `alibaba-token-plan` login instruction), which is
        # the page the product itself sends users to — SHIPPED user-facing,
        # but not independently verified: a bogus sibling path answers 200
        # (the host is a catch-all SPA), so no probe can confirm it
        # (round-1 m2, checked 2026-10-09).
        url="https://home.qwencloud.com/billing/subscription/token-plan-individual",
        dashboard="the QwenCloud console",
        verified=False,
        note=(
            "quoted from the registry's own token-plan login URL (shipped "
            "user-facing); the host answers 200 for any path, so unverifiable"
        ),
    ),
}

#: Flavour variants: ``(storage_id, kind)`` -> the OTHER product's surface,
#: for providers that sell both a balance and a subscription and whose
#: dashboards differ. The primary entry keeps the notice path's surface; a
#: consumer that knows the kind it is describing (the notice from the report
#: shape; post-failure guidance from the credential it failed on) asks
#: :func:`billing_link_for` with it and lands on the right page.
_BILLING_VARIANTS: dict[tuple[str, BillingKind], BillingLink] = {
    ("anthropic", "balance"): BillingLink(
        kind="balance",
        url="https://platform.claude.com/settings/billing",
        dashboard="the Anthropic Console billing page",
        verified=True,
        note="VERIFIED: console.anthropic.com redirects to it; a bogus sibling path returns 404",
    ),
    ("openai", "balance"): BillingLink(
        kind="balance",
        url="https://platform.openai.com/account/billing",
        dashboard="the OpenAI platform billing page",
        verified=False,
        note="documented in OpenAI's prepaid-billing help article (curl 403)",
    ),
    ("xai", "balance"): BillingLink(
        kind="balance",
        url="https://console.x.ai/team/default/billing",
        dashboard="the xAI console",
        verified=False,
        note="documented in xAI's own billing docs (curl 403)",
    ),
    ("kimi", "subscription"): BillingLink(
        kind="subscription",
        url="https://www.kimi.com/code/console",
        dashboard="the Kimi coding-plan console",
        verified=True,
        note="VERIFIED: a bogus /code/* path redirects to /",
    ),
}

#: Chat-capable storage ids with NOTHING to link. The local runtimes bill
#: nobody (they run on the user's own machine) and the mock serves no wire;
#: the drift guard accepts a row here in place of a table entry so "no
#: dashboard" stays an explicit decision rather than a silent omission.
NO_DASHBOARD: frozenset[str] = frozenset(
    {"lmstudio", "ollama", "vllm", "llamacpp", "openai-compatible", "test"}
)


def kimi_topup_url(base_url: str | None = None) -> str:
    """The Moonshot/Kimi top-up page for ``base_url``'s platform.

    Moonshot's two platforms are separate products with separate accounts
    (``api.moonshot.cn`` vs ``api.moonshot.ai``), and the pay pages followed
    the split when the product moved: ``platform.kimi.com/console/pay`` for
    mainland, ``platform.kimi.ai/console/pay`` otherwise. ``base_url``
    defaults to whatever the registry configures for ``kimi`` — the same value
    ``usage.moonshot_balance_target`` classifies for the balance endpoint, so
    the top-up link and the balance number cannot end up describing two
    different accounts.
    """
    if base_url is None:
        definition = get_provider_definition("kimi")
        base_url = (definition.base_url if definition else None) or "https://api.moonshot.cn/v1"
    host = (urlsplit(base_url.rstrip("/")).hostname or "").lower()
    if host.endswith(".cn"):
        return "https://platform.kimi.com/console/pay"
    return "https://platform.kimi.ai/console/pay"


def billing_link_for(provider: str, *, kind: BillingKind | None = None) -> BillingLink | None:
    """The billing surface for ``provider`` (id, alias or login flavour).

    ``kind`` selects a flavour variant when the provider sells more than one
    product (see :data:`_BILLING_VARIANTS`); the primary entry answers when
    the provider does not sell that kind, or when the caller has no opinion.
    ``None`` means the provider is not in the registry at all — callers with
    a stronger validation already ran (the desktop route 422s unknown
    providers before it gets here).
    """
    storage = credential_provider_id(provider)
    link = _BILLING_VARIANTS.get((storage, kind)) if kind is not None else None
    if link is None:
        link = BILLING_LINKS.get(storage)
    if link is None:
        return None
    if link.kind == "balance" and storage == "kimi":
        # The one entry whose URL is not a constant: its platform (mainland
        # vs international) is a registry fact, so it is filled in here for
        # every consumer rather than re-derived at each call site.
        return replace(link, url=kimi_topup_url())
    return link
