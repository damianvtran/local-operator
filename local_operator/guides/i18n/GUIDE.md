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

**Dated state (2026-10-29):** at core `8e1b54dad165`, the shipped set is
`en`, `fr`, `es`, `zh-CN`, `ru`, `vi`, `ur` and `hi`;
`SECURITY_HELD_LOCALES` is empty. The installed generation is
`20261029T083642Z-9c83492a15f9` (`lop --version` reports `v0.91.7`), and the
latest published release is `v0.91.7` (2026-10-29). Eligibility is not
namespace completeness: translation and the required independent LLM audit are
per-namespace, and native-speaker review is explicitly not a gate for any
locale (operator decision, 2026-10-29). The current allowance debt is 326
locale/namespace pairs and 7,881 missing keys: `fr` 27/965; `es` 60/1,386;
`zh-CN` 4/52; `ru` 60/1,386; `vi` 60/1,386; `ur` 56/1,337; `hi` 59/1,369.
English-pinned namespaces under caller contract remain unfinished, including
`wire.errors.sdk_spec` / `wire.errors.sdk_startup` and the English-only
`wire.errors.agents` / `wire.errors.desktop` lane recorded in
`docs/design/backend-originated-strings.md`. Use the HOWTO's primary links and
verify fresh repository, release and installed state; never treat this dated
snapshot as proof that a namespace is translated, audited or contract-complete.
