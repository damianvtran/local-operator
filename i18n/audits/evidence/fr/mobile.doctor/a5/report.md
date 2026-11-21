# fr / mobile.doctor — independent translation audit a5 (PR #4216)

- Verdict: **passed** — blocker 0 · major 0 · minor 0 · nit 0 (unresolved blocker/major: 0)
- Scope: changed `mobile.doctor.watches.{ok,near,at_limit,unknown}` and new `mobile.doctor.watches.{instances_near,instances_at_limit}` at 4e6a954306b5bf8e42778e24b389007ced2ca6ae
- Source sha256 (en): `322230f946a1ac493499986fba909735071e13415791de3dbbbb1c686e83c9fb`; target `5938854de08b6a0cea8930f401d5367049e60ee4cf817c3803dc9acab05009c7`
- Translator: coder subagent of PR #4216 (commit 4e6a95430; wrote the English source and all 7 translations) — model: not reported in the commit; served model of the coder child unverified
- Auditor: copy-reviewer acting as independent i18n auditor, lopdev team member (`task` subagent), session 760414253f82; did not write the translations; corrected the values listed in findings — anthropic/claude-opus-5-5 (served model of this session; no provider fallback observed)

## Findings

| id | severity | keys | change | status |
|---|---|---|---|---|
| — | — | — | none | — |

## Checks

- placeholder multiset identical to English for all 6 keys (runtime message_arguments)
- {how} select parsed by local_operator.i18n.runtime and rendered in both arms (command, other) with synthetic params; no unrendered braces
- fs.inotify.max_user_watches / fs.inotify.max_user_instances, {command} and the OS errors “No space left on device” / “Too many open files” byte-identical English inside the locale's quotation marks, as the existing #4193 rows treat them
- 'file watch' term unchanged from #4193; 'watch handle' term used identically in ok, instances_near and instances_at_limit and distinct from it
- NFC, no bidi control characters
- copy_scan.py (design-qa): 0 findings
- fr: U+202F before “;” “:” and inside « … »

## Method and limits

LLM audit, same model family as the translator (disclosed); no native-speaker pass and no rendered terminal/desktop frame claimed.
