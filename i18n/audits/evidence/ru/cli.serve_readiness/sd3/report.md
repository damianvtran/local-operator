# Russian first audit of revised source — cli.serve_readiness / sd3

Date: 2026-11-16. **Verdict: changes_requested; 0 BLOCKER, 1 MAJOR, 0 MINOR, 0 NIT.**

Auditor: independent-ru-auditor; actual session `b8c027b19533`; actual model `openai/gpt-6.1-sol` from the session runtime declaration; no fallback observed. Translation/remediation job `170903662c6e` is the original translator, per manager; its current model was not separately supplied. The prior translator identity `acf0a84cd992`/`openai/gpt-6.1-sol` remains historical rather than being invented as a new run's actual model.

Provenance: `i18n/audits/evidence/ru/cli.serve_readiness/sd3/record.json`.

## Scope and retained coverage

This is a fresh audit of the changed English/source contract over `1fc11dab96954dcc4c31abcc695c696baf2ead8c`, not a reuse of the prior source-hash approval. Independently assessed all seven changed and four new Russian keys against current English/context, current Russian style/glossary, the HOWTO and the readiness selection code. Full mechanics covers all 15 keys. Prior coverage for the four unchanged keys is retained from sd1/sd1r; the reproduction proves the base target SHA-256 equals the sd1r audited target and each unchanged key's raw UTF-8 catalogue line is identical. Existing sd1/sd2/sd1r evidence and history remain untouched. Deferred cgroup wording N1 was not reopened.

## Per-key assessment

| Suffix | Scope | Assessment |
|---|---|---|
| unproven | Changed | «есть запись» is neutral record wording; «идентичность процесса нельзя подтвердить» preserves inability to prove identity and its start-time explanation. No live-process identity claim. |
| not_reporting | Changed | Neutral «есть запись, но ответа нет» plus missing activity signals preserves a stopped heartbeat without claiming death or PID replacement. Native factual tone. |
| not_this_daemon | Changed | Individual-PID meaning is accurate, but the deictic singular «этот PID» is not neutral when the same diagnosis is attached to a plural PID list. RU-SD3-01. |
| unknown_state | Changed | «запись с неизвестным состоянием» is a neutral diagnosis suitable after singular or plural PID labels. |
| recorded_gap | Changed | Explicitly says the record gives one build and another is installed. Does not claim the live PID runs the recorded build; removes the redundant identity parenthetical because unproven immediately precedes it. |
| unknown | Changed | Em-dash composition avoids the nested colon structure. Every plural arm preserves unknown build currency AND unknown supervision. one/few/many/other and pronouns agree with the full daemon count, not capped visible PIDs. |
| identity_restart | Changed | Literal `lop services restart`; aims to move the daemon described by the record onto the installed build. Retains checks before action, without promising identity proof. Instruction uses correct вы-register. |
| pid_reason | New | Singular «идентификатор процесса» / plural «идентификаторы процессов» correctly localize PID/PIDs as authorized by the sidecar. Em dash precedes the shared complete diagnosis; no English plural suffix survives. PID lists and reason are inserted without modification. Since the numeric count is not displayed, few/many use nominative plural, not a counted noun construction. |
| and_more | New | «и ещё #» is idiomatic, concise, and correct for all four categories. The formatter supplies Latin digits and locale decimal punctuation. |
| identity_no_action | New | In every arm, the RECORD with unconfirmed process identity matches the installed build; the text does not say the live process matches it. No action is scoped to THIS optional check, not a general declaration of daemon health. Record singular/plural agreement is correct. |
| identity_restart_manual | New | Manual `lop serve` restart explicitly moves the recorded daemon to the installed build and explains inability to reload within the current process. No automatic reload or identity-proof promise. |
| install_unknown | Retained | Exact unchanged bytes; prior acceptable build-comparison uncertainty retained. |
| cgroup_unknown | Retained | Exact unchanged bytes; prior coverage retained. Manager-deferred N1 not reopened. |
| stale | Retained | Exact unchanged bytes and all plural arms mechanically rechecked. Retained remote-session/build consequence and glossary «сессии». |
| other_unit | Retained | Exact unchanged bytes and all plural arms mechanically rechecked. Retained distinction between another service and a lop unit; glossary «служба», «юнит», «под управлением». |

## Selection and composition review

Read readiness.py's record/identity branch and remedy selection. `identity != proven` records uncertainty and bypasses proven-process verdicts. An unproven current-build record increments unproven_current; only a record/install gap appends recorded_gap and enables gap advice. The automatic gap remedy requires all relevant records reloadable; otherwise the manual key is selected. A stale neighbor already supplies its single restart instruction. identity_no_action is selected only in the branch without stale/held/other diagnoses and remains an optional-check statement, not identity proof.

Grouping keys are equal complete reason tuples, so different identity/build diagnoses do not inherit each other's PID list. PIDs are capped before grouping; full daemon count still selects the outer plural; overflow uses keyed and_more. Single and multi-PID composed examples for all four identity diagnoses were parser-rendered and read. They retain one shared diagnosis per group, no doubled colons, and no duplicated identity parenthetical with the build gap. Reading the actual multi-PID output, rather than trusting mechanical green, exposed the residual singular deictic in not_this_daemon (RU-SD3-01). These are source inspection plus pure string renders, **not execution of the live readiness/service path**.

## Mechanical and fit results

Actual commands/output are in audit.log: repo catalogue-only validator, retained reproduction, ledger help/history-sha and append dry-run. The separate reproduction checks strict JSON and duplicate members, all 15 source/context/target keys, exact placeholder names/kinds/order, preserved command spans and tag parity, every Russian plural category and representative counts 0/1/2/4/5/11/21/22/25/101/1.5. All simple parameters are also exercised with opaque brace/backtick/path-bearing synthetic values to prove they are inserted verbatim rather than reparsed.

No maxCells/maxChars exists in the current sidecar. Diagnostic prose wraps at narrow widths; «идентификаторы процессов» is longer than PID, but there is no fixed button/tab budget to violate. Cell widths are measured, not a live wrapping/truncation proof. Cyrillic glyphs are single-cell in these measurements; no catalogue combining-mark or bidi-control risk was found. Russian dash spacing and formal imperatives conform. The Urdu-specific RTL audit is inapplicable to ru, and no RTL layout claim is made.

## Findings and handoff

### RU-SD3-01 — MAJOR — cli.serve_readiness.not_this_daemon

**Current:** «есть запись, но этот PID теперь принадлежит другому процессу».

**Observed composition:** «идентификаторы процессов 123, 456 — есть запись, но этот PID теперь принадлежит другому процессу …» (audit.log, multi-PID not_this_daemon sample).

**Problem:** The singular deictic «этот PID» has no single referent after two or more listed PIDs. The English source also retains “that PID”; this is a source/target composition problem, not a missing placeholder or failed plural parser. The shared diagnosis must be number-neutral for all PIDs to which it is attached. Current single-PID meaning is safe, but plural native product copy and the explicit neutral-grouping requirement are not satisfied.

**Suggested Russian wording:** «есть запись, но обнаружено повторное использование PID».

This impersonal noun diagnosis describes PID reuse without assigning singular agreement or claiming process identity has been established. Corresponding source suggestion: “recorded, but PID reuse was detected”. No added count placeholder is needed. The manager/translator should settle the source and target together, then have the changed diagnosis and its grouped render re-audited; do not silently add target-only plural parameters.

All other reviewed keys have no findings. Existing sd1 findings remain historically fixed, not erased or reopened. This is the first independent sd3 pass, not the second independent auditor gate. No source/translation edits, pytest, services, providers, installs, history appends or commits. New bundle files alone are eligible for tracking; parent owns append/commit.

ledger-receipt.json is the changes_requested append-ready object; receipt.json is the companion hash manifest. The final receipt dry-run follows the initial mechanically valid candidate in audit.log; only the final report verdict is authoritative. The note links record.json within 280 characters. All report prose is **audit_notes_not_execution_log**; only audit.log's actual responses prove command execution.
