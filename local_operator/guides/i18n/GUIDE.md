---
name: i18n
description: Contribute localized product copy, add message keys or locales, check language fallback and audit eligibility, and verify i18n programme holds.
---

# Localizing product copy

1. Read `docs/I18N.md` in the source checkout for the contributor HOWTO and
   `docs/design/i18n.md` for the accepted contract (§§2, 4, 6). Re-read primary
   programme PRs before reporting completion; the RFC plan is not shipped status.
   Packaged installs do not bundle those repo docs: fetch/read their source
   revision, do not invent a `guide://` reference for them.
2. Key rendered product copy in a small owned namespace with immutable dotted
   keys. Author English source plus role/tone/width/placeholder context sidecars.
   Keep technical/model-facing docs, prompts, machine tokens and identifiers
   English; migrate English-parsing clients to stable codes before translation.
3. Use generated Python per-key functions and the existing render/envelope path.
   Run `scripts/i18n/generate.py`, `generate.py --check`, and `check.py` with the
   checkout's interpreter and Node. Never hand-edit generated keys/shipped data
   or raise a baseline; lower only touched extracted files with `--update PATH`.
   Core generation also emits/checks the relay type-only
   `local_operator/mobile/web/src/i18n/keys.gen.ts` from the same ICU arguments.
   This is not all-client generation/typechecking; use each client's actual scripts.
4. Treat backend `language` as the single source for config, TUI and connected
   UI settings: explicit config → `LOP_LANG` → OS → `en`, then shipped eligibility.
   Missing namespace/key uses English **source and formatting/plural rules**.
   Recognized `SUPPORTED` is not audited `SHIPPED`; a held locale is no exemption.
5. Delegate translation only to an LLM translator and complete independent
   `i18n-auditor`; record actual agent/model identities (including fallback),
   source hash, native glossary/style choices and per-carried-namespace results.
   Require placeholder/CLDR/RTL/pseudo and rendered evidence; high-visibility
   copy needs a second independent audit. Mechanical green is not native/design
   sign-off. Do not translate/audit your own contribution or erase a failed row.
6. Prove counterfactual guard failures and passing controls in isolated fixtures.
   The current Python ratchet blocks only activated scopes; advisory green does
   not prove global enforcement. Exempt only legitimate non-product text with
   narrow reason, owner and expiry/removal condition; owner/expiry are review
   obligations, not automatically validated allowlist fields today.
7. Preserve the English stable-prefix language preference once per main session
   and subagent spawn; context may override the reply language. Setting changes
   affect the next prefix/session, never mutate a cached prefix mid-turn.

**Dated state (2026-10-12):** installed 0.70.4; en/fr/es/ru/vi partially active.
Hindi core #2267, relay #2281/#2283 and UI #1002 are held on programme gates,
including required actual-Sonnet design. Chinese slash 50/51 is an incomplete
audit, **SECURITY HELD**, never a waiver. Urdu/native RTL, remaining clients and
global enforcement are unfinished. Use the HOWTO's primary links and verify
fresh state; this guide neither completes these streams nor performs their audits.

**Source supersession (2026-10-13 UTC):** at `641a392040bf`, six partial eligible
locales include hi; the four PRs above have merged, so their historical OPEN/design
holds are not current states. Chinese SECURITY HOLD, Urdu/native RTL, remaining
clients and global enforcement are still unfinished. A new General-language
picker is requested, not shipped. The HOWTO distinguishes current source, the
already-cut v0.70.11 tag, measured PyPI and installed 0.70.10; do not conflate them.
Release invariants run unconditionally at the exact event SHA in `publish.yml`;
follow the combined-target checks in AGENTS.md, not a narrow docs/version CI green.
