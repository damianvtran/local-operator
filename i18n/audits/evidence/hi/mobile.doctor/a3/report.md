# hi / mobile.doctor — independent translation audit a3 (PR #4193)

- Verdict: **passed** — blocker 0 · major 0 · minor 0 · nit 0 (unresolved blocker/major: 0)
- Scope: 4 new keys `mobile.doctor.watches.{ok,near,at_limit,unknown}` at ce4f2e4e576baaf3ed899b55c6b0ecb61442da94
- Source sha256 (en): `bf2cc180fd8bc8a7f5445eb4c4a5708fbb7915e6271fefe69f04c425babbcfef`; target `bfaa5bd734d62be0e62cd7950c6ceb1d04040c496128e8a6c28952d034d2016b`
- Translator: coder subagent of PR #4193 (commit ce4f2e4e5; wrote the English source and all 7 translations) — model: not reported in the commit; served model of the coder child unverified
- Auditor: copy-reviewer acting as independent i18n auditor, lopdev team member (`task` subagent), session c47ffb8607fc; did not write the translations; edited no catalogue value — anthropic/claude-opus-5-5 (served model of this session; no provider fallback observed)

## Findings

| id | severity | keys | change | status |
|---|---|---|---|---|
| — | — | — | none | — |

## Checks

- placeholder multiset identical to English for all 4 keys
- fs.inotify.max_user_watches / fs.inotify.max_user_instances and “No space left on device” byte-identical, in the locale's quotation marks
- NFC, no bidi control characters
- formality matches the locale guide; process/administrator/support terms match sibling mobile.doctor rows

## Method and limits

LLM audit, same model family as the translator (disclosed); no native-speaker pass and no rendered terminal/desktop frame claimed.
