# Independent zh-CN audit — wire.teams.effective (9 keys)

## Verdict

**PASSED — linguistic and scoped mechanical audit, 0 unresolved BLOCKER/MAJOR; no MINOR/NIT findings.** All nine keys reviewed against English and context, not against translator assertions. This is one independent audit, **not** full integration, visual, or second-independent-auditor approval.

Auditor: `i18n-auditor`, session `2b66cec83cdb`, actual model `openai/gpt-6.1-sol` (host session declaration; no fallback used). Independent from translator session `530f154495de` (`i18n-translator`, same model, different session). No translation service, SDK, MT website or back-translation used. Frozen source revision: `e70bc818b8553dca1fad007705257817be49ee37`, read directly with `git rev-parse HEAD`.

The assigned group label `wire.teams.effective9keys` denotes the nine-key scope; the **actual catalogue/ledger namespace is `wire.teams.effective`**, as established by filename, source key prefixes, and translator receipt. Do not append under the scope label.

## Inputs and policy

Read `guide://i18n`, frozen `AGENTS.md` i18n/environment sections, `docs/I18N.md` contribution/receipt procedure, RFC `docs/design/i18n.md` §§2.2, 2.8, 4, 6, `i18n/glossary.md`, and `i18n/style-guides/zh-CN.md`. The style guide and glossary preceded English/context and translation review. SHA-256 of all four supplied inputs independently matched the brief; exact paths and hashes are in `record.json` and actual execution output in `audit.log`.

Binding choices: 简体中文; neutral developer-facing product voice; full-width prose punctuation; no invented spaces between Chinese/Latin runs; 请 at first next-step clause; 智能体/团队/模型 terminology; technical config syntax untranslated; Chinese plural categories other-only when plurals actually occur.

## Every-key semantic/idiom review

| Key suffix | Current translation | Assessment |
|---|---|---|
| team_missing | 未找到名为{role}的团队。 | Natural neutral lookup failure; role denotes an opaque team name, not an agent role. No claim of deletion or execution. |
| registry_unreadable | 无法读取智能体注册表。请检查智能体定义后重试。 | Preserves broad read failure, agent definitions and retry remedy. Does not invent a corrupt file or expose an exception. |
| config_unreadable | 无法读取模型配置。请检查配置后重试。 | Preserves model-config read failure and inspect/retry remedy without implying a launch. |
| tier_invalid_value | subagents.models.{tier}的值无效。请将其设为provider/model格式的模型选择器或模型映射对象。 | 两种有效值明确；映射对象 is a configuration object, not a model-conversion operation. Path and selector syntax preserved. |
| tier_invalid_mapping | subagents.models.{tier}的映射对象无效。请将model设为provider/model格式的模型选择器；effort可选，填写时请设为受支持的推理强度。 | Required model versus optional effort is explicit and accurate. Second 请 belongs to the separate conditional directive after the semicolon, compliant with the guide; no spurious duplicate in one clause. effort is reasoning effort, not computational cost. |
| tier_unconfigured | subagents.models.{tier}未配置模型。 | Concise missing-configuration state, distinct from invalid configuration. |
| tier_invalid_selector | subagents.models.{tier}的模型选择器无效。请使用provider/model格式。 | Accurate invalid-selector diagnosis and literal required format. |
| model_unresolvable | 无法解析模型{selector}。请检查模型选择器和提供商配置。 | Natural technical 解析 for resolution; names both selector and provider configuration, not an availability or launch result. |
| unavailable | 此团队成员不可用。请检查其模型配置和智能体定义。 | Correct defensive fallback and both remedies; no launch claim. 此 is acceptable concise product register, not a different form of address. |

No suggested replacement wording: no actionable linguistic defect found. No English diagnostic prose remains; retained Latin runs are placeholders/config paths/field names or the required `provider/model` syntax.

## Mechanical execution and self-verification

Executed `validate_zh_effective.py` with the primary checkout interpreter and frozen worktree imports, `PYTHONDONTWRITEBYTECODE=1`, no installs/build/server/test suite. The script copied only this namespace's English, context and target bytes into an owned scratch fixture, selected that fixture via the repository's documented catalogue-root fixture hook, then invoked the **actual** `scripts.i18n.check.check_catalogues()` function. It returned `[]`; process exit 0. This is not a claim of executing `check.py` CLI, AST ratchet, all-namespace validation or generation.

Separate checks: strict JSON duplicate-key rejection; exactly 9/9 source-target-context keys; ICU parser argument name/type tuples identical for every key; independently compared raw placeholder occurrences; no added/omitted placeholders; literal path/selector occurrence counts equal; required model/effort names retained; no bidi controls inserted. No plural/select expression exists in either input; no CLDR arm is missing because no arm is applicable. The source has no Rich/HTML tags, backtick spans, executable commands or flags to preserve. Literal config examples are not prose to translate, and adding translated quotes or changing punctuation inside their tokens would be wrong. Source does not require backtick quoting; translations do not invent it.

Called repository `runtime.render_message` for all nine target strings with synthetic `role=demo-team`, `tier=hi`, `selector=provider/model` as applicable. All formatted successfully, with no remaining braces. **These formatted strings are not GUI-rendered frames.** Actual output retained in `audit.log`; this report is reasoning (`audit_notes_not_execution_log`), not a substitute for that log.

## Fit, LTR/RTL and outstanding integration evidence

- No fixed maxCells/maxChars hint exists; context explicitly requires natural wrapping. CJK/full-width punctuation is wide, and the mapping remedy is the longest; a desktop narrow-panel/wrapping capture remains required. No proven overflow defect from strings alone.
- zh-CN is LTR. Preserve logical catalogue order and literal Latin tokens. No inline RTL controls needed or inserted. Arbitrary user team names can contain RTL text, and technical identifiers still need renderer-boundary isolation appropriate to the owning client. This session did not inspect/prove that client boundary; do not claim it was tested. This is not a Chinese-copy defect.
- Full UI mirroring and Urdu shaping tests are not locale-specific requirements for zh-CN. Cross-locale Urdu RTL validation belongs to the integration owner, not this receipt.
- Pseudo-localization and rendered long-text, real fallback and mixed-direction captures remain integration evidence gaps. No one-command pseudo workflow is advertised in the HOWTO. No GUI/browser/rendered tests performed here, per the scratch-only/no-build scope.
- Full `check.py`, `generate.py --check`, generation, append dry-run/apply and integration eligibility were **not executed**. The manager must integrate and run the appropriate gates; this receipt does not declare shipping eligibility.

## Second independent audit applicability

**Required.** All nine are user-visible inline unavailability/error diagnostics in the saved-team read view and editor; several tell the user how to repair model configuration. HOWTO step 3 explicitly requires a second independent audit for high-visibility/safety copy. Their inline placement does not exempt error/configuration remedies. Another auditor session must review this frozen source/target hash; my own repeat reading is not that second independent audit. This report neither assumes a previous audit nor certifies a second one.

## Retention / append handoff

No repository, translation, generated assets or ledger written; no PR/commit/push. Explicit task scope requires scratch-only output. `receipt.json` is a single append-compatible object, structurally checked with frozen `ledger._rows`, **not appended**. Its note references the intended tracked canonical record path:
`i18n/audits/evidence/zh-CN/wire.teams.effective/a2b66/record.json`.

Before append, the manager must preserve `record.json`, `report.md`, `audit.log`, `validate_zh_effective.py` and `receipt.json` together under that path's directory, verify all hashes, and retain original input bytes via the source/translation provenance. This is a retention handoff, not a claim the bundle is tracked or merged. Record companion receipt has an explicit `record_reference`; record lists its hash. No historical bundles were touched and no original evidence was lost or recreated.
