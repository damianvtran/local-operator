# Chinese remediation audit — cli.serve_readiness / sd1r

**Terminal verdict: passed.** No open findings of any severity. `ZH-SD1-01` is **fixed**. No new out-of-scope findings were raised.

Auditor: `i18n-auditor`, session `9d590e061719`, actual model `openai/gpt-6.1-sol`, from host session metadata; no fallback observed. Date: 2026-11-16 UTC. Independent from original translator `acf0a84cd992` (actual model `openai/gpt-6.1-sol`, supplied by requesting task). This is that auditor's remediation follow-up, not a newly independent second auditor.

## Scope and retained coverage

The original `../sd1/record.json` and `../sd1/report.md` retain full independent native audit coverage of all 11 keys. Exact byte reconstruction proves this revision changes only `cli.serve_readiness.cgroup_unknown`: replacing its accepted new text with its old text reproduces the entire sd1 target SHA-256. English, context, glossary and Chinese style-guide hashes are unchanged. The previous report/log/verifier hashes are also intact. The other 10 keys retain their original full coverage; this is not a repeated whole-diff audit.

Re-audited the changed key and direct semantic neighbors: `unknown`, `other_unit`, `install_unknown`, `recorded_gap`, and `identity_restart`. No translation edits, history append, commits, installs, heavy tests or live provider calls were made.

## Remediation finding disposition

### ZH-SD1-01 — MINOR — FIXED

Key: `cli.serve_readiness.cgroup_unknown`

Previous: `运行于此检查无法定位的cgroup中`

Accepted suggestion and exact current text: `本次检查无法判定其所在cgroup的归属`

The exact suggested replacement was applied, with no other target-file byte changes. The wording naturally describes the check's inability to classify cgroup ownership rather than inability to locate a known cgroup path. “无法判定” explicitly limits knowledge; it does not claim that the cgroup is missing, unmanaged, safe to restart, or owned by a particular service. The retained technical token `cgroup` is unchanged; Chinese/Latin adjacency follows the guide. The reason has no placeholders or plural/select structure and remains a diagnostic, not a directive requiring 请. Its sample width is 34 cells / 20 code points; no maxCells/maxChars hint exists.

## Direct-neighbor semantic review

- `unknown`: the replacement inserts intelligibly into the per-PID reasons and preserves the overall “无法确定是否与当前安装一致并受监督” uncertainty. Composed count 1/2/21 samples were rendered.
- `other_unit`: a proven other-service classification remains distinct from unknown cgroup ownership. “其他服务” and “单元” keep their glossary terminology; the new text does not assert this classification.
- `install_unknown`: inability to compare the running build with the install remains a different unknown reason; no causal or polarity conflation was introduced.
- `recorded_gap`: recorded-vs-installed build information remains explicitly unproven as to process identity. A cgroup classification failure does not prove a recorded build is running.
- `identity_restart`: inspection and restart commands remain byte-identical and ordered, with 请 and the per-daemon pre-action checks intact. No unguarded restart guarantee was added.

## Mechanical execution and limitations

`audit.log` contains actual command outputs, not audit notes. The existing repository interpreter ran `scripts/i18n/check.py --catalogues-only`: exit 0, **0 problems**. `verify.py.txt` verifies the exact one-key delta and retained-input hashes, then reuses the sd1 read-only parser verification across all 11 current keys: strict JSON/key/argument-kind parity, Chinese other-only CLDR branches, preserved code/tag literals, no bidi format controls, formatting samples and composed new-reason samples. All succeeded.

No fixed width hints exist. Chinese wide-glyph accounting remains as documented by sd1; this delta introduces no combining characters or RTL requirement. These are source/formatting checks, not rendered-frame or release-eligibility approval. Existing required second-independent-audit gates remain unchanged.

`record.json` pins the current inputs and companions. `ledger-receipt.json` is append-ready and its actual ledger append **dry-run** result is in `ledger-dry-run.log`. The parent owns append and commit. Existing sd1/sd2 bundles and history are preserved; newly created sd1r evidence is repository-only and awaits inclusion in the parent's tracked commit. This report is `audit_notes_not_execution_log`, not proof of command execution.
