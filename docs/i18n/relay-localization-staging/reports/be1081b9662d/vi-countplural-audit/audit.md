# PR #2329 — independent Vietnamese count-noun delta audit

**Verdict: native PASS / terminal; no findings.** Scope is only three changed unit nouns and their direct amount-composition contract. The previous **62 unchanged messages retain their native approval through independently proved byte compatibility**. Rendered certification remains **HOLD/BLOCKED by the customer resource hold**, not waived.

Auditor: lopdev task, session `be1081b9662d`, actual model `openai/gpt-6.1-sol`, no fallback. This auditor is independent of the candidate translator/implementer; same-model provenance remains disclosed. No self-audit, repository/ledger edits, activation or shipped claim.

## Source identity — working-tree delta, not committed HEAD

Checkout: `~/local-operator/.worktrees/coder-2329-convergence-20261014`; base HEAD `de9b9a490c839d6de2c2e41ea03ec819f66be489`. The new English/count caller content is **uncommitted at audit time**. Approval attaches to the exact below bytes, not to a claim that base HEAD already contains the delta.

Candidate root: `historical-scratch/590f5d5140c6/vi-final65-countplural`.

| Namespace | Source SHA-256 | Target SHA-256 |
|---|---|---|
| relay.shell | `f53b23560b88b531ef8fc3274bc396676f26e755ad4ff9ad87ce368cebdee560` | `f475dbefccd75d713fdea93fd8c5a12b541f94d00ec00c23b00fe72e2e942319` |
| relay.session | `a4ae148de969d4a2e8541aca5b4b99ad311d9ef112a0d056fc21fcf90e7e0129` | `7743c7b5d018b84395478e99231714a46cd998e8c78ce465333be8df62fd517a` |
| relay.login | `9a59686bcd0e3641a5a807ea2b778cb385e591060c26c7fb994bea95e4de5bc5` | `a87e8ed3381143ee8d61dada52b0df604c72191cdde1747b9ea8fd6c46537020` |

English shell context SHA-256: `c1eb5190cc1c0737bc53320e08a8ef28dd1ec01d8ccf9ba9d5a7b315d2c9a8a1`. New English source/context scratch files independently match the working checkout bytes. `validation.json` was read as an input; its claims were independently re-executed.

## Three-key native review

| Key suffix | New English | Vietnamese | Result |
|---|---|---|---|
| unit_credits | `{count, plural, one {credit} other {credits}}` | `{count, plural, other {tín dụng}}` | PASS |
| unit_tokens | `{count, plural, one {token} other {tokens}}` | `{count, plural, other {token}}` | PASS |
| unit_requests | `{count, plural, one {request} other {requests}}` | `{count, plural, other {yêu cầu}}` | PASS |

Vietnamese has only CLDR `other`, no noun singular/plural agreement. These invariant labels remain idiomatic for zero, one, two, fractional, negative and million amounts. They render the **noun only**, not `#` or another numeric count. Surrounding `Còn lại {value, number} {unit}` / `Đã dùng {value, number} {unit}` owns numeric placement; no added classifier/preposition or duplicate amount is needed. Credits stay `tín dụng`, distinct from token counts; `token` is the previously approved developer borrowing, not an English plural `tokens` leak. Requests remain `yêu cầu`.

Typed argument checks independently confirm source/target `count:plural` with **no skeleton**, target selector exactly `other`, NFC and exact key sets. The change is semantically correct, not a mechanical English one/other copy into Vietnamese.

## Retained audit proof

Prior audit: `reports/be1081b9662d/vi-native-audit/audit.json` (hash recorded in new `audit.json`). Compared to original `vi-final65` delivery:

- Exactly three English values and exactly the same three Vietnamese values changed.
- Removing only those three keyed lines leaves **byte-identical English and Vietnamese shell files**: 41 unaffected keys retained.
- Session 17 and login 4 **entire files are byte-identical** in source and target.
- Total preserved prior native decisions: **41 + 17 + 4 = 62**. Their native findings are not reopened. Prior rendered hold also remains; no retrospective visual approval.

## Actual execution

Python: `env -i HOME=<this-audit>/home LOCAL_OPERATOR_CONFIG_DIR=<this-audit>operator-home PATH="$PATH" TERM=xterm-256color source-checkout/.venv/bin/python <this-audit>/probe.py`.

Node: `env -i HOME=<this-audit>/home LOCAL_OPERATOR_CONFIG_DIR=<this-audit>operator-home PATH="$PATH" TERM=xterm-256color TZ=UTC node --experimental-strip-types <this-audit>/probe.mjs`.

Both exited **0**. Existing interpreter/dependencies only, isolated HOME/config, inherited CMUX stripped. Python imports candidate checkout first; Node executes exact `runtime.ts`/IntlMessageFormat. No browser/native app, heavy suite, dependency install or repository edit.

For each of credits/tokens/requests, counts **0, 1, 2, 1.5, -1, -1.5, 1,000,000** were passed as the actual numeric count and value to both remaining/used compositions: **42 cases in Python and 42 in exact JavaScript runtime; every composed result byte-identical across runtimes**.

| Surface | Actual output | Result |
|---|---|---|
| Fractional credits | `Còn lại 1,5 tín dụng`; `Đã dùng 1,5 tín dụng` | PASS |
| Negative credits | `Còn lại -1 tín dụng`; `Đã dùng -1,5 tín dụng` | PASS numeric/grammar probe, no claim that providers normally return negatives |
| One token | `Còn lại 1 token`; `Đã dùng 1 token` | PASS |
| Two requests | `Còn lại 2 yêu cầu`; `Đã dùng 2 yêu cầu` | PASS |
| Million units | `Còn lại 1.000.000 tín dụng` / `… token` / `… yêu cầu`; matching used forms | PASS |
| Missing count, all three Python nouns | `message argument 'count' was not supplied (have: none)` | PASS negative control |
| Missing count, all three JS nouns | Respective `relay.shell.usage.unit_*` key returned, not accepted as localized result | PASS negative control |
| Catalogue drops count | `arguments () != en's (('count', 'plural'),)` | PASS negative control |
| Wrong count type | `arguments (('count', 'number'),) != en's (('count', 'plural'),)` | PASS negative control |
| Illegal plural style | Parser rejects `{count, plural, percent other {tín dụng}}` with `expected '{'` at offset 24 | PASS negative control |
| Restored catalogue | Checker `[]` | PASS |

Actual production caller diff was read: `measured = amount.remaining ?? amount.used`; only finite measured values enter noun selection; **the same `measured`** goes to noun `{count: measured}` and amount `{value: measured}`. The direct runtime probe repeats that choice for both branches. This is source-plus-real-runtime proof, **not mounted UsageSheet or screenshot evidence**.

## Remaining applicability

Rendered **HOLD** retained under customer resource hold. Vietnamese is LTR; native RTL mirroring is not applicable, and no bidi visual claim is made. Pseudo/rendered long-text fit remains blocked under the hold and cannot substitute for a native audit. No stale fingerprint or ledger is updated here. Terminal native approval is three-delta PASS plus retained62 PASS, exact hashes above.

Durable adjacent evidence: `audit.json`, `mechanical.json`, `js-mechanical.json`, `probe.py`, `probe.mjs`.
