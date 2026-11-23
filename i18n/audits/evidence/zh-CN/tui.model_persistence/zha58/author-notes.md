# zh-CN author notes — translated, not independently audited

Author: i18n-translator / model-notice-zhCN-author; session 58dee179aac0.
Actual authoring model: openai/gpt-6.1-sol. The explicit model-switch message reports temporary fallback from anthropic/claude-opus-5-5 due to provider quota failure. All four translations were authored after that switch. No translation API, MT service, SDK or translation script was used; the strings were reasoned and authored directly by the LLM.

## Native choices

- 模型 and 会话 follow the shared glossary. 新会话的默认模型 makes the boot default's scope explicit without the unnatural 启动默认值.
- `/model default将当前模型保存为新会话的默认模型` expressly saves the current model for NEW sessions.
- `/model saved将本会话切换至已保存的默认模型` expressly switches THIS session to the already-saved default. It does not say this command saves anything or changes future sessions.
- `/settings也可修改新会话的默认模型` remains a complete settings-only clause, while 也可 preserves the source's “too” linkage in combined notices. It adds no unavailable /model alternative.
- Command-first declarative clauses describe command behavior rather than issuing a bare imperative in an error body; therefore no directive 请 is needed. Commands retain exact ASCII spelling/internal spaces. No space is inserted between commands/opaque IDs and Chinese prose, per style-guide typography.
- 模型： translates the human model prefix and uses the locale's full-width colon. Both placeholders stay exact and occur once. The em-dash separator is retained. No sentence-final punctuation is added to these terse status clauses.
- Glossary proposal: boot default → 新会话的默认模型 (explanatory phrase, not a new product name). No glossary file was edited.
- No fixed maxCells exists in the sidecar. No cap was invented; production wrapping at 80/100/120 remains the owner's integration and independent review scope.

## Validation provenance

`validation.log` is actual stdout/stderr from the executed command below, exit 0. `validate.py.txt` is the exact scoped validation script used. This document is author reasoning, NOT an execution log or an independent audit report.

```sh
set -o pipefail
cd /Users/damian/local-operator/.worktrees/i18n-model-persist-notice
mkdir -p "$LOCAL_OPERATOR_SCRATCHPAD/zh-author-home"
env -i HOME="$LOCAL_OPERATOR_SCRATCHPAD/zh-author-home" LOCAL_OPERATOR_CONFIG_DIR="$LOCAL_OPERATOR_SCRATCHPAD/zh-author-home/config" PATH="$PATH" TERM=xterm-256color PYTHONPATH="$PWD" /Users/damian/local-operator/.venv/bin/python i18n/audits/evidence/zh-CN/tui.model_persistence/zha58/validate.py.txt 2>&1 | tee i18n/audits/evidence/zh-CN/tui.model_persistence/zha58/validation.log
```

The repository's existing catalogue/context and ICU parse/argument/render APIs were used, with the primary interpreter and the worktree module path verified. Fresh isolated HOME/config and `env -i` removed CMUX variables. The frozen source/context hashes matched before authoring and during validation. All four keys preserve source order. The first three have zero arguments; `with_model` has simple `label` and `routes` arguments, rendered from exact strings. Source and target have zero plural/select arms. All protected commands are exact and appear once. Render checks cover short and long opaque IDs, already-localized routes and the standalone settings route. No manual bidi controls were added.

Global/base-aware checking was intentionally not run (centralized by owner). No generated keys, ledger, eligibility, consumers or another locale were changed. No commit/stage/push was performed. No live UI, wrapping, shaping, integration, independent audit or shipping claim is made. Independent A and B audits of all four keys remain required; status is translated ONLY.
