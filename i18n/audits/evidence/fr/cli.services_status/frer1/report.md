# fr / cli.services_status — independent evidence-recovery audit (frer1)

14 keys, all 14 audited against the committed en source and the en context
sidecar. Verdict: **PASS** for copy correctness. **0 new findings**, plus **2
accepted minors carried forward** from the ledger row and not re-derived here.

## What was judged

- **Idiom.** Every string was read as French, not back-translated from the
  English. `supervisé par : {unit}` and `non supervisé : {scope} ; systemd ne
  peut pas le redémarrer` read as French operations copy. `rien n'a été
  activé`, `ce HOME n'est pas celui du gestionnaire systemd de l'utilisateur`
  and `ne sert pas son propre processus` keep the source's precise claim
  (a written unit file is not an enablement) without translating the English
  word order.
- **Register.** vous throughout (`installez-le`, `lancez`), matching
  `i18n/style-guides/fr.md`. CLI flag help uses the infinitive (`installer et
  activer`), which is the French help convention.
- **Glossary.** No glossary term is misrendered. `unit` renders as `unité`,
  `service` as `service`, and the kept-English technical tokens (`systemd`,
  `--user`, `linger`, `loginctl enable-linger`, `local-operator-serve.service`,
  `Linux`, `macOS`, `HOME`, `cgroup`, `PID`) stay byte-identical in every key
  that carries them.
- **Placeholders and code spans.** `{unit}`, `{scope}`, `{command}`, `{error}`
  are present, named and ordered exactly as in en; the backticked commands are
  byte-identical between en and fr. Checked mechanically, zero mismatches.
- **Typography.** 13 U+202F narrow no-break spaces, every one before a `:` or
  `;` in prose and none inside a code span; no U+00A0; U+2019 apostrophes and
  no ASCII apostrophe. This reproduces the "13 typography sites" claim in the
  ledger note exactly.

## Findings: 2 accepted minors, carried forward

Neither is re-derived by this audit. No key in this namespace declares
`maxCells`, so the declared-budget sweep cannot see them; the ledger row
measured them against the 80/78-column status-row budget and recorded the
exception in `i18n/style-guides/fr.md`.

| Key | fr cells | budget | status |
|---|---|---|---|
| `serve_linger_required` | 102 | 80 | accepted by the row, recorded in the style guide |
| `unsupervised_unknown` | 92 | 78 | accepted by the row, recorded in the style guide |

Both are marked `carried_forward: true` in `findings.json` with
`source: ledger row … carried forward unverified`. `counts.minor: 2` therefore
reconciles with `auditor.findings: 2` on the ledger row, and `findings.json`
carries a `counts_note` spelling out the split (0 new + 2 carried).

## Target-hash binding (R-4)

The ledger row for `cli.services_status` carries **only** `source_sha256`
(`ae14884d…`) and its note quotes no SHA-256 at all. The audited French bytes
were therefore **never bound by that row**: there is no target hash to compare
against. This bundle records the target hash it actually read (`23422bc3…`),
so the bytes are bound from today forward — but that is this audit's record,
not the row's.

## Structure note (not a defect)

The fr file stores its keys alphabetically while en stores them in insertion
order. The gate compares key SETS (`scripts/i18n/check.py` iterates
`sorted(set(...) & set(...))`), so order is not enforced, and this namespace is
the only one of 69 fr namespaces that differs — the same sorted ordering
appears in es and zh-CN for this namespace. Recorded as an observation only.

## Limitations

- Independent LLM review, not native-speaker certification.
- Scalar text and ICU rendering only: no rendered frame, no terminal or device
  geometry.
- The bundle records THIS audit. It does not reconstruct the audit the ledger
  note cites.
