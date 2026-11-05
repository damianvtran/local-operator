# fr cli.team_bridge audit tbb3400 (PR #3400)

Scope: the two added keys `notify.budget_exhausted_title` and `notify.budget_exhausted_body` at PR head `5456b8db1d017c421c7632324ab2065e64e8b85e`, read against the en source, `cli.team_bridge.context.json` notes (budget used up so new agent messages wait; nothing lost, screened automatically as the budget refills; people and messages addressed to the agent screened first; no implication of rejection, deletion or notification), the fr style guide, `i18n/glossary.md`, and the namespace's existing renderings. The other 55 keys are byte-unchanged since the PR #3198 passed row and were not re-audited.

## Translator values (as committed at head)

- title: Filtrage des messages d’agent en pause pour l’instant
- body: Votre agent a épuisé son budget de filtrage horaire : les nouveaux messages d’agent sont en attente. Ils seront filtrés automatiquement à mesure que le budget se reconstitue. Les messages envoyés par des personnes et ceux adressés à votre agent passent en premier.

## Findings and dispositions (all fixed by this auditor in the audit commit)

- **F1 MAJOR** (title+body) — current: `messages d’agent`. Namespace product name is 'messages des agents' (15 uses; same defect fixed in tbp3198). Fix: 'Filtrage des messages des agents…'; '…les nouveaux messages des agents…'.
- **F2 MINOR** (body) — current: `horaire : (U+0020)`. fr guide punctuation rule requires U+202F before ':'. Fix: U+202F before ':'.
- **F3 MINOR** (body) — current: `…passent en premier.`. 'Pass first' can read as delivered first; the fact to keep is that they are SCREENED first. Fix: '…sont filtrés en priorité.'.

## Shipped values after remediation

- title: Filtrage des messages des agents en pause pour l’instant
- body: Votre agent a épuisé son budget de filtrage horaire : les nouveaux messages des agents sont en attente. Ils seront filtrés automatiquement à mesure que le budget se reconstitue. Les messages envoyés par des personnes et ceux adressés à votre agent sont filtrés en priorité.

Accuracy: all three context facts survive; no rejection, deletion or notification implied. Unresolved blocker/major: 0.

## Mechanical

Key parity 57/57; no placeholders, ICU or markup in either key (none in source); NFC; no Cf/bidi control characters; no code tokens. No `maxChars`/`maxCells` hint exists for these keys; both are inbox-notification prose, not width-bound terminal cells.

## Negative control

At head 5456b8db (rows absent) the generator's `_shipped_locales()` returned `['en']`: the stale `cli.team_bridge` hash would have dropped all seven locales. After these rows it returns all eight (see audit.log).

## Limitations

Independent LLM review, not native-speaker certification; no MT or translation service used. Scalar text only: no rendered inbox, terminal, phone or RTL paint geometry was exercised. The translator's model is not named in PR #3400 and is recorded as unrecorded, not inferred.
