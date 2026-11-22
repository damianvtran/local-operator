# Independent Vietnamese A audit — tui.provider_state

Verdict: **passed**, all five ordered keys reviewed; zero unresolved BLOCKER/MAJOR, zero MINOR/NIT findings. This is independent lexical approval, not UI/native-font, eligibility or release clearance.

Auditor: i18n-auditor / provider-state-vi-audit-A, session `6640da3c0b72`. Actual host-reported fallback: `openai/gpt-6.1-sol`, replacing `anthropic/claude-opus-5-5` for quota failure. Per-response provider telemetry is unavailable; this is the explicit host model-switch disclosure, not an assertion of independently measured serving identity. Translator: session `5644d4838456`, `i18n-translator / provider-state-vi-author`, `openai/gpt-6.1-sol`, as read from `author-5644/record.json`. Its auditor-shaped field explicitly says independent=false and verdict=translated; it is author provenance, not approval. No B audit or other auditor conclusions were read. No translation/MT service, API, or outside comparison was used.

## Native product-copy judgment, in source order

| Key suffix | Current Vietnamese | Judgment |
|---|---|---|
| logged_in | đã đăng nhập | Natural concise completed login status. `đã` marks the completed state, rather than requesting a login. Fits the source's stored-login presentation without adding a health or credential-validity guarantee. |
| env_key | khóa từ biến môi trường | Natural, accurate source-of-credential label: `từ` explicitly means the key comes from an environment variable. `biến môi trường` is established Vietnamese technical usage. Not an actual variable identifier, and no identifier is translated. `khóa` follows the guide's first-vowel tone placement. The glossary's generic credentials rendering does not prohibit the narrower API-key noun here. |
| needs_login | cần đăng nhập | Idiomatic requirement/status, not a command label. `cần` retains the distinction from completed `đã đăng nhập`. No artificial pronoun or extra politeness. |
| local_ready | máy chủ đã cấu hình | Compact and idiomatic noun-led status in product copy: the server is configured. Vietnamese technical labels naturally omit a passive marker here; adding `được` is unnecessary. `đã cấu hình` reports configuration completion, not online/reachable/healthy status. `máy chủ` conforms to the guide. The local-provider context supplies locality; English likewise does not repeat “local.” |
| local_unconfigured | cấu hình máy chủ | Natural direct action wording, with verb first. Contrasts clearly with the completed-state `máy chủ đã cấu hình`. Does not falsely say configuration succeeded or that login failed. `máy chủ` is consistent with local_ready. |

All five use the expected terse lower-case register, correct Vietnamese diacritics and NFC, no terminal punctuation or gratuitous “bạn”/“vui lòng.” No glossary violations, omitted negation, added permissions or unsafe destructive wording were found. These strings contain no protected commands, tags, paths, identifiers, placeholders, numbers or RTL text.

## Execution evidence

`audit.log` is the actual output of the companion `script.py.txt`, executed using the primary checkout's venv with this worktree on PYTHONPATH. HOME/config were scratch-isolated using `env -i`; inherited CMUX variables were absent. Actual command return code: **0**. Module paths prove that validator/catalogue/runtime code came from this worktree.

The official `scripts.i18n.check.check_catalogues()` ran against a fixture containing exactly English, English context and Vietnamese for this namespace. Initial and restored results were `[]`. Removing a key produced the expected missing-key error; adding `{unexpected}` produced the expected argument-parity error. No tracked catalogue was changed. Ordered key parity is exactly five; ICU parsing, argument parity and absence of plural/select constructs were additionally checked independently. No plural categories are applicable to these zero-argument plain strings. Official ICU rendering returned the source strings for en and vi; Rich markup/plain-text checks and NFC/protected-token checks passed. Synthetic padded pseudo text was measured with Rich; this is an explicitly local stress diagnostic, not an official pseudo locale or a rendered app frame.

Plain Rich cell widths: English **9/7/11/17/16**; Vietnamese **12/23/13/19/16**. No context maxCells exists. The 23-cell environment label and 19-cell configured-server label may be truncated by narrow consumers, but no universal 17-cell cap exists. This observation is deferred to actual styled integration geometry; it is not evidence of an overflow defect and does not warrant forced retranslation. No native-font or app-level rendering was exercised; current provider UI integration remains outside this lexical audit.

## Findings and boundaries

Finding IDs: none. No wording changes requested. A separate independent B pass remains the manager's responsibility for this high-visibility surface. No source/target/context/history/generated/baseline/eligibility files were edited. No app, server, browser, install, build, full checker/test suite, stage, commit or push was run. The direct ledger receipt is prepared but **not applied**; no ledger history is modified by this bundle.

This report is `audit_notes_not_execution_log`; only `audit.log` proves validator execution. Artifact hashes, input hashes and identities are in `record.json` and `artifact-manifest.json`.
