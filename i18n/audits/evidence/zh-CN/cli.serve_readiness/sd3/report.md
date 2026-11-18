# Chinese audit — #4112 review remediation, first pass / sd3

**Verdict: passed. Zero open findings (BLOCKER 0, MAJOR 0, MINOR 0, NIT 0).**

Audit date: 2026-11-16 UTC. Auditor: `i18n-auditor`, session `9d590e061719`, actual model `openai/gpt-6.1-sol`, as disclosed by host session metadata; no fallback observed. Translator: original translator session `acf0a84cd992`, new translation job `170903662c6e`, per requesting task. Original task disclosed translator model `openai/gpt-6.1-sol`; this new job's actual model/fallback was not supplied or independently queried, so its current-job model is recorded as unknown, not silently asserted from the old task. Auditor is not that translator and made no translation edits.

## Scope and retained coverage

Read current glossary, Chinese style guide, English source and context sidecar, Chinese target, contributor audit procedure, evidence contract, and the relevant `readiness.py` grouping/remedy-selection clauses. Base is `1fc11dab96954dcc4c31abcc695c696baf2ead8c` plus uncommitted remediation, not a claim of merge or release.

There are 15 keys: 7 existing keys changed, 4 new keys, 4 retained byte-identical members. Native semantic review covers the 11 changed/new keys below. The other four retain the completed full Chinese audit from `../sd1r/record.json` and the original full coverage in `../sd1/report.md`. The verifier compares each retained member's literal UTF-8 JSON line against `git show 1fc11dab:<target>` and emits its SHA-256. The entire baseline target SHA-256 is `ecfbee6fb7fadc73b9c22352970b2c0643d851ab18b31ad4184f2af53a5d4075`, identical to the prior sd1r input. No historical bundle was altered or reclassified as a new audit.

## Changed and new key review

All keys use prefix `cli.serve_readiness.`.

| Key | Assessment |
| --- | --- |
| `unproven` (changed) | Pass. “已有记录，但无法证实进程身份” is a neutral grouped diagnosis. It does not apply one PID's attributes to other PIDs. The lack of comparable start-time evidence remains the reason for uncertainty. |
| `not_reporting` (changed) | Pass. Records exist but there is no response; stopped heartbeat is preserved. In the PID-labelled diagnostic context it does not claim process death, authenticated identity or a known build. |
| `not_this_daemon` (changed) | Pass. Another process now occupies that PID. No build, supervision or permission conclusion is attached to the replacement process. PID spelling is preserved. |
| `unknown_state` (changed) | Pass. “记录存在，状态未知” works as a neutral per-group diagnosis without a singular pronoun. No unobserved lifecycle state is invented. |
| `recorded_gap` (changed) | Pass. “记录中注明构建为{recorded}；已安装的构建为{install}” describes the record, never the live process. Parameter names/types/order are unchanged. It intentionally omits the duplicate unproven-identity parenthetical; the immediately preceding diagnosis already carries uncertainty. |
| `unknown` (changed) | Pass. An em-dash pair separates the subject from PID-group diagnoses; no stacked colon is introduced. Current-install consistency and supervision explicitly remain unknown. `other` alone handles all Chinese counts, including capped lists. |
| `identity_restart` (changed) | Pass. 请 marks the directive. The literal `lop services restart` moves the recorded daemon toward the installed build; the sentence retains pre-action checks and makes no promise to prove identity. The source caller selects this only for recorded gaps that are reloadable, without duplicating a proven-stale neighbor's restart instruction. |
| `pid_reason` (new) | Pass. “进程ID{pids}——{reason}” localizes the permitted PID/PIDs label naturally. Chinese needs no plural noun inflection; `count` remains a plural argument. PIDs and complete diagnosis are inserted unchanged, with neither English verb-agreement residue nor an extra colon. |
| `and_more` (new) | Pass. “以及另外#个” is concise overflow wording, with the numeric count and natural default classifier 个. It joins the bounded PID-group diagnostic list, rather than modifying a build label. |
| `identity_no_action` (new) | Pass. It states that the **recorded build** matches the installed build and explicitly says process identity is unproven. “此项非必需检查无需采取任何操作” limits no-action advice to this non-required check. It does not turn unknown into healthy/proven or waive checks elsewhere. |
| `identity_restart_manual` (new) | Pass. 请 and 手动重启 preserve a manual next step; `lop serve` remains literal. “无法在当前进程内自行重新加载” conveys inability to reload in place. It does not promise verified identity or suggest automatic reload for a non-reloadable record. |

## Retained four keys

- `install_unknown`: unchanged member bytes; existing uncertainty and build-comparison coverage retained.
- `cgroup_unknown`: unchanged member bytes; retained accepted text “本次检查无法判定其所在cgroup的归属”. The separately deferred cgroup N1 discussion is outside this delta; no finding or edit was introduced for it. Historical `ZH-SD1-01` remains fixed as recorded in sd1r.
- `stale`: unchanged member bytes; distinction between actual running build and installation remains covered by the prior audit.
- `other_unit`: unchanged member bytes; other service versus lop serve unit distinction and glossary choices remain covered by the prior audit.

## Caller selection and composition review

Read `readiness.py` lines 3036–3052, 3139–3164 and 3181–3229. Unproven records are compared with install metadata separately from live identity. A matching unproven record increments `unproven_current`; a gap instead sets `unproven_gap` and combines all affected records' reloadability. Only identical complete diagnoses are grouped; the PID cap precedes grouping, with numeric PID lists and localized overflow. A proven stale neighbor supplies the existing single restart remedy. Absent such a neighbor, gap advice chooses automatic versus manual restart from reloadability. The row still returns `ok=False` and unknown when there is no proven stale/supervision verdict. These are **static source observations**, not execution of the service or a test of live daemon behavior.

Rendered catalogue compositions exercise two PIDs sharing one unproven-plus-gap diagnosis, an overflow count, and a separate unknown-plus-no-action sequence. The grouped render has one unproven statement and no `::` / `：：`; the gap text has no duplicated identity parenthetical. Chinese double em dashes are deliberate punctuation, not mirrored arrows. Caller-owned ASCII separators and opaque inserted values were not rewritten by this auditor.

## Mechanical checks and fit

Actual commands/results are in `audit.log`, distinct from these `audit_notes_not_execution_log`.

- Existing `.venv/bin/python scripts/i18n/check.py --catalogues-only`: exit 0, **0 problems**.
- Read-only `verify.py.txt`: strict duplicate-rejecting JSON, exact 15-key/context parity, identical placeholder names/kinds, retained commands/backtick and markup literals, Chinese `other`-only plural structures, no bidi formatting controls; all succeeded.
- Full current catalogue formatted successfully. Six plural keys were sampled at 0, 1, 2, 5, 21 and 1000; all other keys rendered too, for 45 ordinary message samples. Counts select grammar; other inserted parameters are independent synthetic values, not simulated daemon facts.
- Every simple parameter also received an opaque value containing braces, Rich-like text, backticks and a path. Rendered values remained byte-exact, not reinterpreted as ICU. This is not a Rich escaping or terminal safety proof.
- No `maxCells` or `maxChars` hints exist. Ordinary samples measured 11–135 raw cells with wide CJK/full-width glyphs counted as 2 and ASCII as 1. Longer real PID/build/device values or composite diagnostics may wrap; no fixed-budget overflow was established. No combining-mark width hazard was introduced. Chinese is LTR; no Urdu/RTL approval is claimed.
- Tone, 请 directives, Simplified script, Chinese/Latin adjacency, 服务 / 单元 / 会话 / 受监督 terms and literal commands conform to the guide. No new out-of-scope polish findings were added.

No MT service, installs, pytest, services, live providers or heavy tests were used. No screenshot/terminal layout/pseudo-localization or release-eligibility certification is claimed. This is the FIRST independent audit of this new source/target revision, not a second-independent approval; the requesting manager retains that gate.

## Provenance and retention

`record.json` pins current source/context/target and caller hashes, coverage, actual auditor identity and companion hashes. `ledger-receipt.json` is append-ready; `ledger-dry-run.log` captures the actual append dry-run and before/after history digest. Parent owns append and commit. Files are repo-only process artifacts; newly untracked sd3 evidence awaits the parent's tracked commit. No translations, source, prior bundles, snapshots or history were edited by this auditor.
