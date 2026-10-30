# Independent fr audit — wire.slash — fr29s3

- Locale: `fr` · namespace: `wire.slash` (52 keys)
- Auditor: `i18n-auditor`, session `848874ddfe7a`, instance `848874ddfe7a-slash3-fr`, actual model `radient/auto`, 2026-10-29
- Translator: i18n-translator (session 0347acaa76f7, 16 keys; session 53ebb4e7a4a0, wire.slash.approvals), actual model `radient/auto`, 2026-10-29
- Verdict: **pass** — 0 BLOCKER, 0 MAJOR, 1 MINOR, 0 NIT
- Bytes audited: en `b20f4be0dae6446424862ba227313b6c3384bd115a033edc57c0a289bfe74202` · target `bb580c5bc5a2b2d91b03b97b92be5341db0c731feaf7ea592f08b70d3520983e` (whole files, SHA-256)
- Keys: 52 read whole-file; 16 changed by this commit, out of 70 across the seven locales

## Why this commit exists

Head commit `7159e2a30927831f5dfb88f84c588eac07770e5d` rewrote the 52 slash-command descriptions per locale to fit the
picker width, and the rewrite dropped verbs, technical tokens and scoping clauses — and,
in French, reversed an action (round-1 BLOCKER on `wire.slash.agent`: « leur parler »
inverts `/agent`'s attach direction). It carried no independent audit, and the
translation ledger still described the previous bytes. Round 1 of this audit rejected
those bytes: 1 BLOCKER / 4 MAJOR / 4 MINOR / 5 NIT.

## Rounds

| Round | Instance | Bytes | Verdict | BLOCKER / MAJOR / MINOR / NIT |
|---|---|---|---|---|
| 1 | `9412375334be-slash-fr` | the shortened values from head `7159e2a30` | changes_requested | 1 / 4 / 4 / 5 |
| 2 | `b682afa67e4b-slash2-fr` | the 16 rewritten values | changes_requested | 0 / 1 / 4 / 1 |
| 3 | `848874ddfe7a-slash3-fr` | `wire.slash.approvals` after the single-key fix | pass | 0 / 0 / 1 / 0 |

Round 2 accepted 51 of 52 keys and rejected one: `wire.slash.approvals` read
« Régler l’approbation des outils de cette session », which drops the *report* half of
English "Show or set" (a bare `/approvals` reports, per the handler docstring). The
translator's second session `53ebb4e7a4a0` rewrote that one value; round 3 confirmed it
and proved by byte reconstruction that the other 51 keys are byte-identical to the
round-2 audited file.

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
