# zh-CN / mobile.doctor — independent translation audit a5 (PR #4216)

- Verdict: **passed** — blocker 0 · major 0 · minor 1 · nit 0 (unresolved blocker/major: 0)
- Scope: changed `mobile.doctor.watches.{ok,near,at_limit,unknown}` and new `mobile.doctor.watches.{instances_near,instances_at_limit}` at 4e6a954306b5bf8e42778e24b389007ced2ca6ae
- Source sha256 (en): `322230f946a1ac493499986fba909735071e13415791de3dbbbb1c686e83c9fb`; target `dd8d652e9f41b6231f919aff9e05fb52bafa17fb310ea701267b2450356b5ae2`
- Translator: coder subagent of PR #4216 (commit 4e6a95430; wrote the English source and all 7 translations) — model: not reported in the commit; served model of the coder child unverified
- Auditor: copy-reviewer acting as independent i18n auditor, lopdev team member (`task` subagent), session 760414253f82; did not write the translations; corrected the values listed in findings — anthropic/claude-opus-5-5 (served model of this session; no provider fallback observed)

## Findings

| id | severity | keys | change | status |
|---|---|---|---|---|
| ZH-W2 | minor | near, at_limit, instances_near, instances_at_limit | 请运行{command}重启它 → 请运行 {command} 重启它: {command} is a literal shell command; all 12 sibling mobile.doctor rows space a literal command (请运行 lop mobile restart), and unspaced it rendered 'lop services restart重启它', losing the command boundary | corrected in catalogue |

## Checks

- placeholder multiset identical to English for all 6 keys (runtime message_arguments)
- {how} select parsed by local_operator.i18n.runtime and rendered in both arms (command, other) with synthetic params; no unrendered braces
- fs.inotify.max_user_watches / fs.inotify.max_user_instances, {command} and the OS errors “No space left on device” / “Too many open files” byte-identical English inside the locale's quotation marks, as the existing #4193 rows treat them
- 'file watch' term unchanged from #4193; 'watch handle' term used identically in ok, instances_near and instances_at_limit and distinct from it
- NFC, no bidi control characters
- copy_scan.py (design-qa): 0 findings

## Method and limits

LLM audit, same model family as the translator (disclosed); no native-speaker pass and no rendered terminal/desktop frame claimed.
