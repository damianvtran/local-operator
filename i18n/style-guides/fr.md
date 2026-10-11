# Style guide — French (`fr`)

Read with `i18n/glossary.md`; everything here is fr-specific and binding for
translators and the auditor. Wave 1 locale. The guide supersedes the template
for `fr`; when this file and a habit disagree, this file wins.

## Formality & tone

- Address: **vous**, lowercase mid-sentence ("vous", not "Vous"). Recorded
  decision: a professional tool — "tu" would read over-familiar or juvenile,
  and the fr developer surfaces users already read (JetBrains, MDN fr, GitHub
  fr) address the user as vous. The terseness of the English lives in sentence
  *length*, not in pronouns.
- Imperative for actions: "Enregistrer", "Reprendre", "Ignorer". No
  "Veuillez…" (bureaucratic); "Merci de…" only where the English says
  "please".
- Terse: statuses are past participles ("Enregistré", "Repris") — never
  "Votre session a bien été enregistrée avec succès".
- Sentence case; compact fragments and statuses may stay lowercase ("en
  cours") per fr UI convention. Never Title Case.

## Product-name policy

- Keep exactly as in glossary §1: "Local Operator", `lop`, tool names,
  `--flags`, config keys, paths, identifiers. No article: "Local Operator",
  not "le Local Operator".
- Kept-English technical list (settled fr dev vocabulary that stays English):
  commit, push, pull, merge, rebase, fetch, tag · token, cache, endpoint,
  script, prompt · API, CLI, TUI, JSON, YAML, HTTP, URL, GPU · bug. Reason:
  translating them ("validation", "bogue") reads foreign to fr developers.
- Wave-1 recorded addition: runtime (kept English; mirrors the es guide).
- Prefer French where settled: serveur, fichier, dossier, réseau, stockage,
  paramètres (settings), appareil (device), équipe, compétence (skill).

## Locale calls (fr)

- session → session (feminine: "une session")
- wake → réveil ("réveils programmés"); never the `wake` tool name in prose.
- tool trace → trace · transcript → transcription
- approvals → approbations; `/approvals` stays.
- schedule → planification (Windows fr precedent: "Planificateur de tâches");
  one scheduled item reads "prompt planifié".
- usage → utilisation · cost → coût
- job → tâche (a backgrounded shell job — recorded 2026-10-11).

## Plurals

- Categories: `one` (0 and 1 — "0 session"), `many` (exact multiples of 10⁶
  only — "1 000 000 de sessions"; 1 500 000 → `other`), `other`. `few` is
  not a French category — never use it.
- Example: `{count, plural, one {# session} many {# de sessions} other {# sessions}}`
- `#` renders the formatted number; never pre-format inside the branch.

## Numbers, dates, digits

- Never hand-written — the formatter renders "1 234,5" (espace fine insécable
  U+202F) and dates "10 oct. 2026"; 24-hour times.
- Digits: Latin (v1, all locales).

## Width rules

- fr runs roughly 15-25% longer than en. Tighten: prefer "Enregistrement
  auto" over "…automatique" when a budget is tight; shorter synonyms over
  truncation, always.
- Budgets: `maxCells` from the en sidecar; pinned: PERSIST_HINT 42/43 cells,
  tip pool 59, `/help` ≈ 55. French typographic spaces (U+202F) count one
  cell each — account for them.

## Punctuation & typography

- Quotes: « … » (U+202F inside); apostrophe ’ (U+2019, not '); ellipsis …;
  em dash " — " with spaces.
- Espace fine insécable (U+202F) before : ; ! ? — required within prose;
  never inside code spans, flags or paths (those stay byte-identical).
