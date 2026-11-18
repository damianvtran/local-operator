# fr / mobile.doctor — independent translation audit a4 (PR #4193)

- Verdict: **passed** — blocker 0 · major 0 · minor 1 · nit 0 (unresolved blocker/major: 0)
- Scope: 4 new keys `mobile.doctor.watches.{ok,near,at_limit,unknown}` at ce4f2e4e576baaf3ed899b55c6b0ecb61442da94
- Source sha256 (en): `bf2cc180fd8bc8a7f5445eb4c4a5708fbb7915e6271fefe69f04c425babbcfef`; target `7c437699b3e8e35f55eea1ae4977323722ac479f45a99e6e6d3fa111b867edec`
- Translator: coder subagent of PR #4193 (commit ce4f2e4e5; wrote the English source and all 7 translations) — model: not reported in the commit; served model of the coder child unverified
- Auditor: copy-reviewer acting as independent i18n auditor, lopdev team member (`task` subagent), session c47ffb8607fc; did not write the translations; corrected the values listed in findings — anthropic/claude-opus-5-5 (served model of this session; no provider fallback observed)

## Findings

| id | severity | keys | change | status |
|---|---|---|---|---|
| FR-W1 | minor | near, at_limit, unknown | U+0020 → U+202F before “;” and “:” and inside « … » (fr guide: Punctuation & typography; 1122 U+202F vs 72 plain spaces before :/; across fr catalogues) | corrected in catalogue |

## Checks

- placeholder multiset identical to English for all 4 keys
- fs.inotify.max_user_watches / fs.inotify.max_user_instances and “No space left on device” byte-identical, in the locale's quotation marks
- NFC, no bidi control characters
- formality matches the locale guide; process/administrator/support terms match sibling mobile.doctor rows

## Method and limits

LLM audit, same model family as the translator (disclosed); no native-speaker pass and no rendered terminal/desktop frame claimed.
