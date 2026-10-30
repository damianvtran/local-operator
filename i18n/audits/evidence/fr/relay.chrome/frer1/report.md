# fr / relay.chrome — independent evidence-recovery audit (frer1)

85 keys, all 85 audited against the committed en source and the en context
sidecar. Verdict: **PASS** for copy correctness, with **5 minor layout
findings** (declared-width overruns, none a rendering error).

## What was judged

- **Idiom.** The namespace is a phone relay SPA; the French reads as French app
  chrome. `Vous attend` for "Waiting for you", `Aucun message pour l'instant`,
  `Pas encore de conversation`, `Démarrez-en une avec « Nouvelle »` are
  idiomatic rather than calqued. `Terminée · historique conservé` carries the
  gender split French requires (session feminine, historique masculine).
- **Busy-state register.** Every in-progress state uses the French action-noun
  plus ellipsis: `Marquage…`, `Démarrage…`, `Reprise…`, `Connexion…`,
  `Enregistrement…`, `Chargement de la conversation…`. That is one consistent
  pattern across 6 keys, not six independent choices.
- **Cross-key consistency.** `« Reprendre »` and `« Terminée »` in the health
  copy are the same words as `relay.chrome.session.resume` ("Reprendre") and
  `relay.chrome.session.subtitle.ended` ("Terminée"), so the button the copy
  tells the reader to tap is the button that exists. The « Nouvelle » quoted in
  the empty-list hint is the `relay.chrome.list.new` button label.
- **Glossary.** `agent` → `agent`, `session` → `session`, `model` → `modèle`,
  `project` → `projet`, `usage` → `utilisation`, `settings` → `paramètres`,
  `library` → `bibliothèque`. `Local Operator` is unarticle'd, per the style
  guide. Tool names and `task(agent=…)` roles are not present in this
  namespace.
- **Plurals.** Four ICU plural keys carry `one` / `many` / `other`, exactly the
  three French categories, and never `few`. The `many` arms read `# de
  secondes` / `# de minutes` / `# de questions`, the correct French
  construction for the 10⁶ multiples that category exists for.
- **Placeholders and ICU.** Argument names, kinds, order and nesting are
  identical to en in all 85 keys; select arms are `online` / `retrying` /
  `other` in both. Checked mechanically, zero mismatches.
- **Typography.** U+202F before every prose `:`; « … » with U+202F inside;
  U+2019 only; `…` rather than three dots; spaced em dashes. Zero violations.

## Findings (all minor, all layout)

Five keys exceed the `maxCells` budget declared in the en context sidecar. In
every case the French is the correct rendering and no shorter *correct* synonym
exists, so the recommendation is a recorded width exception rather than a
rewrite. Each is exact, not sampled: these five keys carry no ICU placeholders,
so their cell counts do not depend on substitution values.

| ID | Key | fr cells | en cells | budget | over | role |
|---|---|---|---|---|---|---|
| F1 | `list.search_close` | 19 | 12 | 16 | +3 | accessible name (never laid out) |
| F2 | `list.mark_all` | 21 | 13 | 20 | +1 | visible header action |
| F3 | `row.unpin` | 11 | 5 | 10 | +1 | visible: swipe item + sheet row |
| F4 | `row.saving` | 15 | 7 | 12 | +3 | visible busy state |
| F5 | `session.loading` | 30 | 21 | 28 | +2 | visible loading state |

F1 is an accessible name: it is announced, never laid out, so it cannot
truncate — recorded because the declared budget is exceeded, not because it
renders wrong. The other four are visible and overrun by 1, 1, 3 and 2 cells
respectively.

All five are now recorded in `i18n/style-guides/fr.md` under "Width
exceptions", alongside the `cli.services_status` pair that already lived there.

These four (F1–F4) are now recorded in `i18n/style-guides/fr.md` under "Width
exceptions", alongside the `cli.services_status` pair that already lived there.

## Note on sweep totals

Earlier drafts quoted a sweep of "245 keys, 10 over, 10 French-only". Those
totals are **not reproducible**: the count depends on which values are
substituted for ICU placeholders, and `relay.chrome`'s own plural and counter
keys change category depending on the number rendered. The totals are
therefore dropped. What stands is the five exact cells above, which need no
substitution, plus the observation that these five are the only declared-budget
overruns in `relay.chrome`.

## Observation on the English source (not an fr defect)

`relay.chrome.session.degraded` renders "Not answering — sends will wait" and
the fr mirrors it (`les envois attendent`). `relay.health.dropped_detail`,
describing the same yellow "Not answering" state, instead says a send "fails
straight away" and that the text stays in the box. The two English sources make
opposite delivery promises; the French is faithful to each, so there is nothing
to fix in fr. Flagged because a reader moving between the two sheets would see
the contradiction.

## Limitations

- Independent LLM review, not native-speaker certification; no human French
  speaker reviewed these bytes.
- Scalar text, ICU rendering and `rich.cells.cell_len` only. No relay app was
  launched, no phone or browser rendered, no geometry measured: the ledger's
  "rendered relay geometry pending in the PR" caveat still holds.
