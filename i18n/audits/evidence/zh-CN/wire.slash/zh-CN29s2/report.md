# Independent zh-CN audit — wire.slash — zh-CN29s2

- Locale: `zh-CN` · namespace: `wire.slash` (52 keys)
- Auditor: `i18n-auditor`, session `7b1ecf2ef33b`, instance `7b1ecf2ef33b-slash-zh-cn`, actual model `radient/auto`, 2026-10-29
- Translator: i18n-translator (session 1fe5533c9a8f), actual model `radient/auto`, 2026-10-29
- Verdict: **pass** — 0 BLOCKER, 0 MAJOR, 0 MINOR, 1 NIT
- Bytes audited: en `b20f4be0dae6446424862ba227313b6c3384bd115a033edc57c0a289bfe74202` · target `ebc0ae0101022d4ff31ca7e54f0e0d34d000c7e3b46e62b54ac3ed132a88cd55` (whole files, SHA-256)
- Keys: 52 read whole-file; 5 changed by this commit, out of 70 across the seven locales

## Why this commit exists

Head commit `7159e2a30927831f5dfb88f84c588eac07770e5d` rewrote the 52 slash-command descriptions per locale to fit the
picker width, and the rewrite dropped verbs, technical tokens and scoping clauses — and,
in French, reversed an action (round-1 BLOCKER on `wire.slash.agent`: « leur parler »
inverts `/agent`'s attach direction). It carried no independent audit, and the
translation ledger still described the previous bytes. Round 1 of this audit rejected
those bytes: 0 BLOCKER / 1 MAJOR / 3 MINOR / 2 NIT.

## Rounds

| Round | Instance | Bytes | Verdict | BLOCKER / MAJOR / MINOR / NIT |
|---|---|---|---|---|
| 1 | `a303f65af544-slash-zh-CN` | the shortened values from head `7159e2a30` | changes_requested | 0 / 1 / 3 / 2 |
| 2 | `7b1ecf2ef33b-slash-zh-cn` | the rewritten values this commit carries | pass | 0 / 0 / 0 / 1 |

## What the audit covered

Whole file, all 52 keys: key-set and order parity against English, placeholder and
technical-token parity, plural categories, typography, and `rich.cells.cell_len` width
against the picker's description budget. Full finding list, with per-finding evidence, is
in `receipt.json` (`namespaces.wire.slash.findings`); the auditor's verdict text is
`verdict.txt`.

## Not claimed

- No English change: `en/wire.slash.json` is byte-identical at `b20f4be0dae6446424862ba227313b6c3384bd115a033edc57c0a289bfe74202`, which is why
  the ledger row's `source_sha256` is unchanged. Generation output is unaffected.
- No native-terminal RTL/shaping certification; the width figures are `cell_len`
  measurements and real-widget paints inside the receipt, not a claim about every terminal.
- Raw agent transcripts stayed in the auditing session's scratch. `receipt.json`,
  `verdict.txt` and `verdict-round1.txt` are the portable evidence; this report summarises
  them and claims nothing they do not.
