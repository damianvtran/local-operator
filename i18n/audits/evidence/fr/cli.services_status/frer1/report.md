# fr / cli.services_status — independent evidence-recovery audit (frer1)

14 keys, all 14 audited against the committed en source and the en context
sidecar. Verdict: **PASS**, 0 findings.

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
- **Width.** All 14 keys render inside their declared `maxCells` (measured
  with `rich.cells.cell_len` over substituted placeholders), so the two
  accepted overruns the ledger records (`serve_linger_required`,
  `unsupervised_unknown` against the 80/78 status budget) are measured
  against the status-row budget rather than `maxCells` — consistent with the
  recorded exception in the style guide.

## Structure note (not a defect)

The fr file stores its keys alphabetically while en stores them in insertion
order. The gate compares key SETS (`scripts/i18n/check.py` iterates
`sorted(set(...) & set(...))`), so order is not enforced, and this namespace
is the only one of 69 fr namespaces that differs — the same sorted ordering
appears in es and zh-CN for this namespace. Recorded as an observation only.

## Limitations

- Independent LLM review, not native-speaker certification.
- Scalar text and ICU rendering only: no rendered frame, no terminal or
  device geometry.
- The bundle records THIS audit. It does not reconstruct the audit the ledger
  note cites.
