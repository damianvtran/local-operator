# Independent Chinese audit — cli.serve_readiness

- Locale: `zh-CN`; namespace: `cli.serve_readiness`; audit ID: `sd1`.
- Audit date: 2026-11-16 (UTC).
- Auditor: `i18n-auditor` / session `9d590e061719`, actual model `openai/gpt-6.1-sol` (host session metadata; no fallback observed).
- Translator: task/session `acf0a84cd992`, actual model `openai/gpt-6.1-sol`, as supplied by the requesting task. Different session; this auditor did not translate or edit the catalogues.
- Audited working-tree base: `e4a6b6e1d01ef269604fe6d005e6bce1453b66ca` plus uncommitted catalogue files. These are NOT represented as committed or on main.
- Verdict: **passed**, with **0 unresolved BLOCKER/MAJOR**, 1 MINOR, 0 NIT.

## Finding

### ZH-SD1-01 — MINOR — cgroup diagnostic precision

Key: `cli.serve_readiness.cgroup_unknown`

Current: `运行于此检查无法定位的cgroup中`

Problem: “无法定位” ordinarily suggests that the check cannot locate the cgroup. In the caller, `serve_supervision()` already receives the cgroup path but cannot classify its owning scope/unit (`kind == "unknown"`). The message should describe uncertain ownership/classification rather than unavailable location. This is a precision improvement, not an assertion that supervision is known or an unsafe restart instruction; the surrounding unknown verdict remains conservative.

Suggested: `本次检查无法判定其所在cgroup的归属`

No translation edits were made. This minor suggestion is not a release-blocking finding.

## Full key coverage (human reasoning, not an execution log)

All keys below have the `cli.serve_readiness.` prefix. English source/context and the glossary/style guide were read before the target catalogue. Each value was reviewed for mechanics, meaning, native idiom/register, terminology and fit. A final safety-focused reread checked uncertainty, recorded-vs-running builds and guarded restart wording; that reread is NOT a second independent auditor approval.

| Key | Semantic/native assessment |
| --- | --- |
| `unproven` | Pass. “无法证实其进程身份” preserves uncertainty; lack of comparable start timestamp is not recast as proof of a stale or different process. |
| `not_reporting` | Pass. No response and stopped heartbeat remain distinct from proven death. |
| `not_this_daemon` | Pass. PID reuse by another process is stated; no inference about that process's build or permissions. PID remains Latin and uppercase. |
| `unknown_state` | Pass. Recorded state is unknown, without inventing a lifecycle state. |
| `install_unknown` | Pass. The running build cannot be compared to the installed build; it does not claim the installation is absent or the daemon is outdated. |
| `cgroup_unknown` | Pass with MINOR ZH-SD1-01 above. Classification uncertainty is retained, but location terminology can be more precise. |
| `recorded_gap` | Pass. Recorded build and installed build remain separately labelled; parenthetical “进程身份未经证实” prevents the record being mistaken for evidence about the current PID. |
| `unknown` | Pass. Current-install consistency and supervision remain unknown. Chinese `other` alone works for singular/plural and capped lists; `count` stays a plural argument even though no count is displayed. |
| `stale` | Pass. Distinguishes running build(s) from installed build and explains the hosted remote sessions use the former. “而非已安装的构建” resolves which build “上述构建” means. No claim that a merely recorded build is running. |
| `other_unit` | Pass. Other services, not a lop serve unit, manage these daemons. 服务 / 单元 match the glossary; both spellings of the `lop serve` command remain intact as in source. |
| `identity_restart` | Pass. Uses directive-initial 请; commands and order preserved. Checks each daemon before acting, without promising every record will be restarted or treating an unproven PID as safe. |

## Mechanical verification and execution evidence

See `audit.log` for actual captured commands, return codes and stdout/stderr, and `verify.py.txt` for the retained read-only reproduction script.

1. Initial system `python3 scripts/i18n/check.py --catalogues-only` failed at import (`tomllib` absent). This failure is retained; it is not presented as validation success.
2. Existing checkout interpreter `.venv/bin/python scripts/i18n/check.py --catalogues-only` succeeded: exit 0, `check.py: catalogue checks: 0 problem(s)`.
3. Additional strict parser verification succeeded for all 11 keys: valid JSON with duplicate-member rejection, exact key sets including context, identical argument names/kinds, Chinese `other`-only plural structure, retained backtick code/tag literal sets, and no inserted bidi formatting controls.
4. The three plural messages (`unknown`, `stale`, `other_unit`) rendered at counts 0, 1, 2, 5, 21 using the repository ICU runtime. All other keys also rendered. These are deterministic formatting samples, not live providers or daemon execution.
5. Read the ledger `append --help`, prepared the receipt against its actual schema, and performed append **dry-run only**. The parent owns history append. The dry-run command/response is retained separately in `ledger-dry-run.log`.

No `%` placeholders, flags, paths, Rich/HTML tags, select structures or destructive authorization prompts occur in these values. Source backtick commands are unchanged. `cgroup`, PID and literal commands retain technical spelling. No MT/translation service, installs, test suites or live providers were used.

## Style, fit and RTL boundaries

- Simplified Chinese, natural dropped pronouns, no 您, no Chinese/Latin insertion spaces, full-width prose punctuation; commands retain ASCII punctuation. Short reason/status fragments omit final stops, consistently with the source diagnostic composition. 请 is present on the first next-step clause.
- Terminology includes 会话, 服务, 单元 and 受监督 per the glossary; no conflicting product naming appears.
- No `maxCells`/`maxChars` hints exist in any of the 11 context entries. This is wrap-capable diagnostic prose, not a button/tab label. Sample raw-text widths range from 16 to 143 cells using CJK/full-width characters as 2 cells and ASCII as 1. Backtick source notation is included in those estimates; they are not actual Rich layout measurements. Real token lengths and aggregate `{reasons}` vary and can increase line length. No translation-specific fixed-budget breach was established.
- No combining-mark width or bidi-control hazard appears in the Chinese catalogue. Locale is LTR; Urdu sentence order, mirrored glyphs and native RTL rendering are out of scope. No screenshot, terminal-layout, pseudo-localization, visual design or release-eligibility approval is claimed.
- This is one independent audit. Any required second independent high-visibility/safety audit remains the requesting manager's separate gate; this record does not waive it.

## Retention and authority

`record.json` binds all inputs, actual auditor provenance, finding IDs and companion hashes. `receipt.json` is the ledger-compatible row and hashes the finalized non-circular companions; `ledger-dry-run.log` explicitly references the record. Audit reasoning above is `audit_notes_not_execution_log`; only captured execution logs prove commands. Files are retained under the canonical repository evidence root, initially untracked for the parent to include in its tracked commit; no commit or history append was made by this auditor. No existing dated bundle, manifest, authority addendum or ledger snapshot was modified.
