# zh-CN core literals — translation handoff

Translator session: 9950d6b42e2e. Actual translating model: openai/gpt-6.1-sol; fallback after provider-quota failures from anthropic/claude-sonnet-5-5 and anthropic/claude-opus-5-5. No translation service used. This is translation, not an independent audit.

## Files and keys

Three catalogue files in local_operator/i18n/catalogues/zh-CN/: cli.failures.json (4), cli.group_help.json (10), tui.setup_cues.json (2). Total: 16 keys.

- cli.failures.error_line
- cli.failures.required
- cli.failures.invalid_choice
- cli.failures.invalid_command
- cli.group_help.skills
- cli.group_help.skills_description
- cli.group_help.tunnel
- cli.group_help.secret
- cli.group_help.network
- cli.group_help.disk
- cli.group_help.operator
- cli.group_help.pair
- cli.group_help.search
- cli.group_help.fetch
- tui.setup_cues.setup_placeholder
- tui.setup_cues.no_provider_cue

## Terminology and rationale

- Error wrapper uses 错误 with the original ASCII colon/space shape; other prose uses Chinese punctuation. Choice placeholders (including supplied quote marks) remain unchanged.
- Agent Hub and Radient remain product names. 技能包/文件包 distinguish installable bundles from plain documents. 执行权限位 explicitly preserves the refusal of any Unix execute bit.
- harness: 智能体运行环境, making the local runtime clear; mesh network: 网状网络; approval gate: 审批限制. Existing target terminology was retained where suitable.
- The encrypted-store help describes long-term encrypted confidential values and retrieval without asking every time; it is not permission to bypass approval gates. Proposed glossary addition: secret (stored value) → 机密; harness → 智能体运行环境; mesh network → 网状网络. No glossary file changed.
- Pairing: 签名授权一次即可 means signing authorises pairing once, not that each future operation is pre-authorised.
- Setup uses 账户, consistent with the existing welcome connection steps, and keeps /login radient byte-identical.

## Width and verification

Measured rendered strings by Unicode East Asian Width (W/F=2, ASCII=1; combining marks=0):
- setup_placeholder: 输入/login radient开始 — 22 cells.
- no_provider_cue: 先连接AI账户：输入/login radient — 32 cells, within maxCells 74 (42 cells spare).
Nothing exceeds the specified budget.

Read-back checks: valid JSON, canonical ensure_ascii=False/indent=2/trailing newline, English key order, exact key parity and placeholder lists for all 16 keys. Checked no literal redaction marker is stored.

Catalogue validator, generate.py, tests and git commands NOT run, as explicitly instructed for this translation-only assignment. No ledger or generated files changed. Independent auditing remains for the separate auditor.
