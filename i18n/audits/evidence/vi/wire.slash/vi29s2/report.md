# Independent vi audit — wire.slash — vi29s2

- Locale: `vi` · namespace: `wire.slash` (52 keys)
- Auditor: `i18n-auditor`, session `f3ea10304234`, instance `f3ea10304234-slash2-vi`, actual model `radient/auto`, 2026-10-29
- Translator: i18n-translator (session 137c362d5bba), actual model `radienthq/auto`, 2026-10-29
- Verdict: **pass** — 0 BLOCKER, 0 MAJOR, 1 MINOR, 4 NIT
- Bytes audited: en `b20f4be0dae6446424862ba227313b6c3384bd115a033edc57c0a289bfe74202` · target `d228f91dad840c65253a5d46919cb99c0cb0eaa21c99d315e79f55f8490dfc23` (whole files, SHA-256)
- Keys: 52 read whole-file; 11 changed by this commit, out of 70 across the seven locales

## Why this commit exists

Head commit `7159e2a30927831f5dfb88f84c588eac07770e5d` rewrote the 52 slash-command descriptions per locale to fit the
picker width, and the rewrite dropped verbs, technical tokens and scoping clauses — and,
in French, reversed an action (round-1 BLOCKER on `wire.slash.agent`: « leur parler »
inverts `/agent`'s attach direction). It carried no independent audit, and the
translation ledger still described the previous bytes. Round 1 of this audit rejected
those bytes: 0 BLOCKER / 2 MAJOR / 5 MINOR / 4 NIT.

## Rounds

| Round | Instance | Bytes | Verdict | BLOCKER / MAJOR / MINOR / NIT |
|---|---|---|---|---|
| 1 | `86c035febe3e-slash-vi` | the shortened values from head `7159e2a30` | changes_requested | 0 / 2 / 5 / 4 |
| 2 | `f3ea10304234-slash2-vi` | the rewritten values this commit carries | pass | 0 / 0 / 1 / 4 |

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
