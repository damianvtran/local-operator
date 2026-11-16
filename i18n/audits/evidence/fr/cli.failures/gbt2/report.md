# fr · cli.failures · audit gbt2 (fresh independent round)

- Date: 2026-11-16 · Worktree commit 80a55b208 (base f0689a27)
- Auditor: i18n-auditor (audit2-fr), session e73cdb50a745. Started on
  openai/gpt-6.1-sol; provider quota failure (HTTP 429) switched the session to
  the fallback anthropic/claude-opus-5-5 before any catalogue was read, so the
  audit reasoning, findings and verdict were produced by anthropic/claude-opus-5-5.
- Translator (not this auditor): i18n-translator (timeout-fr), openai/gpt-6.1-sol
  with fallback anthropic/claude-opus-5-5 during its run, 2026-11-16 (as reported
  by the parent session). Same fallback model family as this auditor, different
  session and role; no translator notes were read.
- Inputs read: i18n/style-guides/fr.md, i18n/glossary.md, docs/I18N.md, en
  catalogue + context sidecar, fr catalogue. No earlier audit records/receipts
  for this key were consulted. No MT service used.
- Scope: all 5 keys of `cli.failures`.

## Verdict: passed — 0 BLOCKER, 0 MAJOR, 0 MINOR, 4 NIT

## Mechanical (see audit.log)

`check.py --catalogues-only` → 0 problems, rc 0. 5/5 keys, none added or
dropped; placeholder sets identical per key; no ICU plural/select in source;
`GitHub` kept; U+202F before `;` in the new key; apostrophes U+2019; valid JSON.
No maxCells declared (fr 92 cells vs en 83, a single stderr line — no fit issue).

## Accuracy — `cli.failures.github_borrow_timeout`

fr: « L’identifiant GitHub de {owner} n’a pas été reçu dans les {seconds} s ; relancer la commande »

Meaning preserved: credential not received within the deadline; remedy is to
re-run the command. "n’a pas été reçu" states non-arrival only — no denial,
revocation, login requirement, automatic retry or token/lease extension is
implied. `{seconds} s` keeps the existing numeric deadline. "identifiant" follows
glossary `credentials → identifiants` and the shipped singular usage
("Mettre à jour un identifiant", "l’identifiant OAuth stocké").

The four pre-existing keys are accurate and unchanged by this commit.

## Findings

- **N1 (NIT)** `cli.failures.github_borrow_timeout` — "relancer la commande".
  Infinitive is consistent with this namespace ("choisir parmi {choices}"), but
  the shipped fr diagnostics almost always address the reader with the vous
  imperative for a remedy ("relancez la même commande", "relancez cette
  commande"). Optional: « … ; relancez la commande ».
- **N2 (NIT)** same key — `{seconds} s` uses U+0020 between number and unit; French
  typography keeps the unit with its number (U+202F or U+00A0) so a wrapped
  terminal line cannot strand "s". Corpus is mixed (tui.app uses a no-break
  space). Optional: `{seconds}\u202Fs`.
- **N3 (NIT)** same key — "L’identifiant GitHub de {owner}" may momentarily read
  as "{owner}’s GitHub ID/username"; "de" also reads as possession rather than
  origin. Glossary-conformant and acceptable; a marginally clearer option:
  « L’identifiant GitHub provenant de {owner} n’a pas été reçu … ».
- **N4 (NIT)** `cli.failures.error_line` — "{prog}: erreur: {message}" has no U+202F
  before the colons, unlike `invalid_choice`/`required`. Defensible: the sidecar
  asks to keep the argparse-style '<command>: <error word>: <message>' shape (a
  structured prefix, not prose). Pre-existing; no change required.

No BLOCKER or MAJOR. Limits: no rendered terminal frame; no human native review.
