# zh-CN provider-state — author receipt, NOT an audit

Date: 2026-11-22. Lane: fix; bounded locale-author slice, no commit/stage/push.
Status: **translated**, independent complete-namespace audits A/B still required.

## Authorship and immutable inputs

Translator: i18n-translator, session `298f6fe9722a`.
Actual model: `openai/gpt-6.1-sol`; host explicitly announced temporary fallback
from `anthropic/claude-opus-5-5` for provider quota failure in this transcript.
No translation API, SDK, MT service, website or additional agent was used.
English source author is coder session `93b7b02899af`, NOT this translator.
Source provenance: that session's `provider-state-source-receipt.md`, private
commit `1b77e201977c7f252a1fd8bac4baed36fc46c83a`. Source/context SHA-256 pins
are rechecked in the actual execution log and retained in record.json.

Read guide://i18n, docs/I18N.md contributor sections, docs/design/i18n.md
catalogue/pipeline/terminal contracts, AGENTS.md i18n/isolation sections,
i18n/style-guides/zh-CN.md and i18n/glossary.md. English source/context and
existing zh-CN tui.welcome copy were read through read_public_copy. An attempted
read of nonexistent/uncommitted zh-CN tui.settings.json was refused; no alternative
reader was used for it. Existing welcome copy uses 登录 and 密钥.

## Native author choices (not independent linguistic approval)

| Key suffix | Translation | Reasoning | Rich cells |
|---|---|---|---:|
| logged_in | 已登录 | Completed authentication state, not a command; the stored-login meaning is supplied by the source contract. | 6 |
| env_key | 环境密钥 | Compact credential-source label, distinct from stored login; 环境 translates human “env”, 密钥 follows existing Chinese API-key terminology. No variable identifier is translated. | 8 |
| needs_login | 需登录 | Requirement status, not a button or imperative; avoids claiming an attempted login failed. | 6 |
| local_ready | 服务器已配置 | Completed server configuration only; does not claim online/reachable/healthy or authenticated. | 12 |
| local_unconfigured | 请配置服务器 | Explicit configuration instruction contrasting with 已配置; 请 follows the guide for next-step instructions, not button labels. | 12 |

No sentence-final punctuation, manual bidi controls, placeholders, plural/select
branches, technical identifiers or product names appear in any of the five labels.
The JSON keys retain the exact source order and bytes. Machine state codes,
provider names, environment-variable names and all consumers remain unchanged.

Glossary proposals (notes only; glossary was NOT edited):
- human “environment key” / “env key” → 环境密钥, meaning an API credential
  supplied through an environment variable, never an actual variable identifier;
- login / log in → 登录; API key → API密钥 (already used by adjacent welcome copy).
服务器 is explicitly mandated by the zh-CN style guide. No general credentials
translation is proposed; that glossary cell remains deliberately undecided.

## Executed mechanical validation

Executable evidence is `validation.log`, not this reasoning note.
`validator.py.txt` preserves the exact executed program. It imports the primary
checkout's `.venv/bin/python` with this worktree's explicit PYTHONPATH, verifies
both product and checker module paths, and checks fresh HOME/config isolation
with CMUX_* and LOP_* absent. PYTHONDONTWRITEBYTECODE=1 prevents repo pycache writes.

Actual shell execution (from the worktree root; scratch HOME explicitly created):

```sh
env -i PATH=/usr/bin:/bin \
 HOME=/Users/damian/.local-operator/sessions/298f6fe9722a/scratchpad/provider-validation-home \
 VALIDATION_HOME=/Users/damian/.local-operator/sessions/298f6fe9722a/scratchpad/provider-validation-home \
 LOCAL_OPERATOR_CONFIG_DIR=/Users/damian/.local-operator/sessions/298f6fe9722a/scratchpad/provider-validation-home/.local-operator \
 PYTHONPATH=/Users/damian/local-operator/.worktrees/i18n-provider-state-labels \
 PYTHONDONTWRITEBYTECODE=1 \
 /Users/damian/local-operator/.venv/bin/python \
 i18n/audits/evidence/zh-CN/tui.provider_state/zh22author/validator.py.txt \
 > i18n/audits/evidence/zh-CN/tui.provider_state/zh22author/validation.log 2>&1
```

Exit code: **0** (actual bash result). The official check.check_catalogues() ran
with discovery restricted in-memory to `en`, `zh-CN`, and `tui.provider_state`;
its actual loader/context/ICU/plural rules remain unchanged. Result: `[]`.
No full-tree/base checker, generated output, ledger or eligibility changes.
Exactly five ordered keys, all zero-argument, no plural/select, NFC, no introduced
protected tokens/controls. Official render_message and render_record returned
all ten exact expected locale values with en/zh-CN source provenance.

Same-code English control and Chinese Rich diagnostics used the documented
31-cell pane formula `max(31 - ID_cells - 4, 3)` with two synthetic display IDs.
For `openrouter` (10 cells), segment=17 and all five Chinese labels fit uncut.
For `synthetic-long-id` (17 cells), segment=10 and the two 12-cell Chinese
configuration labels truncate, as do the longer English controls. There is no
source maxCells/maxChars, so this is a reported dynamic-fit limitation, NOT a
failed invented cap or a universal no-truncation claim. Expanded pseudo strings
were also fitted with Rich for both locales (20 diagnostics). This is a custom
synthetic expansion probe; the repository advertises no one-command pseudo tool.

## Scope and unproven surfaces

Plain Rich output is NOT real SettingsView, terminal-native font, screenshot,
integration, global UI or linguistic approval evidence. No app/server/browser/
native test was started. Environment isolation is NOT network isolation; the
original French reproduction's offline qualification remains unverified here.
Current consumers still render English until the root's separately gated adapter
implementation. No external wire API changes, cap changes, source/context changes,
modelnotice changes, generated files, ledger rows or audit-history writes.
The owner must privately commit these bytes before independent A/B public-reader
access; no uncommitted-reader refusal may be bypassed.
