# Vietnamese independent translation audit — first pass

Verdict: **passed** (9/9 keys; 0 BLOCKER, 0 MAJOR, 0 MINOR, 0 NIT).
Audit date: 2026-11-19. Frozen source revision: e70bc818b8553dca1fad007705257817be49ee37.
Auditor: i18n-auditor, session cef644ceab2f, actual model openai/gpt-6.1-sol (system runtime declaration; no fallback). Translator is a different session, bdce0d00e792; shared model is not shared identity. No translation edits or MT/API/service used.

Scope: all nine Vietnamese wire.teams.effective messages against pinned English and context, glossary and Vietnamese guide. Independent native-language reasoning, not a translator-receipt endorsement. Mechanical checks executed first in isolated scratch; command output is audit.log, not these notes.

## Per-key semantic, idiom and remedy review

| Key suffix | Current translation | Judgment |
|---|---|---|
| team_missing | Không tìm thấy nhóm có tên {role}. | PASS. Natural Vietnamese absent-result construction; opaque team name preserved; nhóm matches glossary. Does not mistake role for a role label. |
| registry_unreadable | Không thể đọc danh mục tác nhân. Kiểm tra các định nghĩa tác nhân rồi thử lại. | PASS. Registry correctly expressed as a catalogue/collection of agent definitions, not registration, a log, or a named corrupt file. tác nhân agrees with existing vi wire.errors.agents. danh mục is an appropriate native noun here; no binding glossary rule requires a different registry term. Definition inspection and retry retained. |
| config_unreadable | Không thể đọc cấu hình mô hình. Kiểm tra cấu hình rồi thử lại. | PASS. Read failure, not invalidity or absence. mô hình is mandated glossary term. Direct neutral retry instruction; dropped possessive natural, no lost remediation. |
| tier_invalid_value | Giá trị tại subagents.models.{tier} không hợp lệ. Đặt thành chuỗi định danh dạng provider/model hoặc đối tượng cấu hình mô hình. | PASS. Configuration location and invalid value preserved. Both admissible remedies retained: selector string OR mapping/configuration object. Elliptical Đặt thành has an unambiguous value antecedent and is normal concise diagnostic copy. Technical syntax literal. |
| tier_invalid_mapping | Đối tượng cấu hình tại subagents.models.{tier} không hợp lệ. Đặt model thành chuỗi định danh dạng provider/model; có thể đặt thêm effort thành mức suy luận được hỗ trợ. | PASS. Mapping understood as configuration object. Imperative Đặt model makes model required; có thể đặt thêm effort makes effort optional, not both required or both optional. Supported reasoning effort preserved with native mức suy luận. Semicolon cleanly separates mandatory and optional clauses without changing meaning. |
| tier_unconfigured | Chưa cấu hình mô hình tại subagents.models.{tier}. | PASS. No configured model at the specified tier, not invalid selector and not missing provider. Natural concise diagnostic; no unsupported remedy invented. |
| tier_invalid_selector | Chuỗi định danh mô hình tại subagents.models.{tier} không hợp lệ. Dùng định dạng provider/model. | PASS. Selector is the configuration identifier string, not a UI selection widget. Exact provider/model remedy retained; does not expose raw invalid value. |
| model_unresolvable | Không xác định được mô hình {selector}. Kiểm tra chuỗi định danh mô hình và cấu hình nhà cung cấp. | PASS. Resolution failure expressed naturally without falsely asserting network failure or provider absence. Both selector and provider configuration checks preserved. Opaque selector argument unchanged. |
| unavailable | Thành viên nhóm này hiện không khả dụng. Kiểm tra cấu hình mô hình và định nghĩa tác nhân của thành viên này. | PASS. Neutral unavailable state, not a launch outcome. Configuration and agent-definition checks both retained. hiện is natural present-state phrasing, not a promise of recovery. |

## Cross-key checks

- All nine values are Vietnamese diagnostic prose. Only required technical configuration syntax remains English: subagents.models., provider/model, model, effort and opaque placeholder values. No added English prose.
- Direct product register, no added please/vui lòng, routine bạn, exclamation marks, or stilted honorifics. Consistent mô hình / nhóm / tác nhân and chuỗi định danh phrasing.
- All nine messages independently checked for NFC; zero format controls. No added keys, dropped keys, duplicate JSON members or reordered placeholders; placeholder names and parser types match English.
- No plural or select expression exists in either source or translation. Vietnamese CLDR other-only requirement is therefore not invoked; no missing category claim is hidden.
- No source markup, Rich tags, backtick code, commands or flags; none introduced. Literal configuration syntax preserved.
- Context has no maxCells/maxChars; explicitly allows natural wrapping. Values contain 34–168 Unicode characters. No fixed-budget overflow finding. Actual client wrapping/fit remains untested; character counts are not rendered proof.

## Actual execution and remaining gaps

Executed command (exit 0):

```sh
env -u XPC_FLAGS PYTHONDONTWRITEBYTECODE=1 /Users/damian/local-operator/.venv/bin/python /Users/damian/.local-operator/sessions/cef644ceab2f/scratchpad/effective-reasons-vi-a1/validate.py > /Users/damian/.local-operator/sessions/cef644ceab2f/scratchpad/effective-reasons-vi-a1/audit.log 2>&1
```

Script copies only en/context/vi into its own scratch fixture, patches catalogue_root in process, calls repository check.main(['--catalogues-only']), then independently checks strict JSON/duplicates, key order, argument names/types, ordered placeholders, tokens, NFC and controls. Result: check.py catalogue checks: 0 problem(s); 9/9 passed. Input and translator-receipt hashes match all requested pins.

Repository/ledger append, generation, build, install and PR operations: NOT RUN; forbidden by this task. Ledger receipt structurally validated through repository ledger._rows only, not appended or dry-run-appended.

Second independent high-visibility audit: STILL REQUIRED. Runtime, pseudo-localization, fallback, client bidi/mixed-token handling and rendered visual/fit proof: NOT RUN. LTR Vietnamese does not require an Urdu-specific RTL copy audit, but this does not certify client bidi behavior.

## Retention handoff

This bundle is scratch staging only, as explicitly requested; no claim that it is tracked or present on main. Manager must preserve/promote companions under i18n/audits/evidence/vi/wire.teams.effective/vi-a1-cef644/ before ledger append and final completion. receipt.json references that intended canonical record path; until promoted, that path is not an existing evidence claim. Keep these unique scratch originals until a tracked commit exists. audit.log is actual execution output; report.md is audit_notes_not_execution_log.
