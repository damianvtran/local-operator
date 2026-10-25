# relay.chrome → fr — unified translation remediation round

**Locale:** French (`fr`) · **Namespace:** `relay.chrome` · **Keys:** 83/83
**Translator:** translator (this session `132edc7afe5f`) · **Model:** `radient/auto` as the session header reports; no model pin and no fallback selected. The header exposes no resolved provider model id, and none is guessed.
**Worktree:** `~/local-operator/.worktrees/relay-chrome-i18n` @ `535cea359825411741069cbffcee42e2927df9d3` — untracked target, **no commit**.
**Inputs:** first locale audit `chrome25-fr-a1` (session `3d61f1a19402`), independent safety audit `chrome25-safety-b1` (session `acd16c49df72`).

## Hashes

| Artefact | Path | SHA-256 |
|---|---|---|
| English source (unchanged) | `local_operator/i18n/catalogues/en/relay.chrome.json` | `f30049d1ec7bce1d0586d8d5d380368e64fb0ac841d91b9b04c1625d17688aac` |
| English context (unchanged) | `local_operator/i18n/catalogues/en/relay.chrome.context.json` | `1c727d1c6c8ddff7c33cbd3abb1ed716e89ab84c864cdb9c319b5df2595fb69a` |
| fr target — **old** | scratch `chrome25-fr-remediation/candidate-pristine.json` (byte copy of the audited candidate) | `b3a97316412b6b35c0d883a0e79a7cf8b9ed88c99bf298920cd235a2d7a6916f` |
| fr target — **new** | `local_operator/i18n/catalogues/fr/relay.chrome.json` | `5d3a3387e7324a1ed5b1b51104fd06da1470285674f3ce62e24e0eaf33bbb6d2` |

The pristine snapshot was taken **before** any edit, so the delta below is exact.

## Delta — 8 keys changed

| Key | Old | New | Findings |
|---|---|---|---|
| `relay.chrome.composer.folder_label` | `Dossier de travail␟ : {name}` | `Dossier de travail␟: {name}` | F01 |
| `relay.chrome.composer.model_label` | `Modèle␟ : {label}` | `Modèle␟: {label}` | F01 |
| `relay.chrome.composer.spend_label` | `Dépenses et contexte␟ : {label}` | `Dépenses et contexte␟: {label}` | F01 |
| `relay.chrome.session.subtitle.idle` | `Inactif` | `Inactive` | F02 |
| `relay.chrome.session.subtitle.waiting` | `En attente de vous` | `Vous attend` | MIN-03 |
| `relay.chrome.session.ended` | `Session terminée · historique conservé` | `Terminée · historique conservé` | F09, MAJ-04 |
| `relay.chrome.session.ended_hint` | `Un envoi ou Reprendre la rouvre dans votre dossier personnel` | `Envoyer un message ou «␟Reprendre␟» la rouvre dans votre dossier personnel` | F03, MIN-04 |
| `relay.chrome.session.degraded` | `Ne répond pas — les envois attendront` | `Ne répond pas — les envois attendent` | F10, MIN-06 |

`␟` is U+202F (espace fine insécable). No key added, removed or renamed; key order unchanged.

## Dispositions — every finding accounted for

| ID | Severity | Key(s) | Disposition | Reason / evidence |
|---|---|---|---|---|
| `fr-a1-F01` | minor | `composer.folder_label`, `model_label`, `spend_label` | **fixed** | U+202F + U+0020 left a two-character gap before `:`. Measured on the fr corpus (excluding this file): **249 values use U+202F immediately before `:`, 0 use U+202F+space**. All three now `U+202F:`. |
| `fr-a1-F02` | minor | `session.subtitle.idle` | **fixed** | The slot's sibling statuses agree with *session* (feminine): `Terminée`. The corpus already uses the feminine with sessions: `sessions inactives`, `si la cible est inactive`. Now `Inactive`. |
| `fr-a1-F03` | nit | `session.ended_hint` | **fixed** | Button reference now quoted `« Reprendre »` (U+00AB / U+202F / U+00BB), matching `« Nouvelle »` in `list.empty_hint` and the fr guide's quote rule. |
| `fr-a1-F04` | nit | `list.search_close` | **deferred** (no change) | 19 vs `maxCells` 16, but the role is an **accessible name only** — nothing is laid out, so nothing can clip. `Fermer` (6) fits but strips the object for screen-reader/voice users. Rendered/TTS confirmation deferred to the rendered QA pass. |
| `fr-a1-F05` + `safety-MIN-05` | minor | `list.mark_all` | **kept; alternative rejected** | `Tout marquer comme lu` 21/20. The first audit's own recommendation was to keep it. The safety audit's `Tout marquer lu` (15) is not grammatical French — `marquer lu` has no valid construction; `marquer comme lu` is the standard form (cf. corpus `marqué comme prêt`). Accurate idiom retained over a misleading abbreviation; flagged for rendered fit. |
| `fr-a1-F06` | minor | `row.unpin` | **deferred** (no change) | `Désépingler` 11/10 — the standard French unpin term. `Détacher` (8) was considered and rejected: it drops the pin metaphor and no longer pairs with the sibling 72 px tile label `Épingler`. Flagged for the action-sheet/swipe-tile render. |
| `fr-a1-F07` | minor | `row.saving` | **deferred** (no change) | `Enregistrement…` 15/12. `Sauvegarde…` (11) fits but drifts from the guide's Enregistrer/Enregistré vocabulary and from the existing fr busy state `enregistrement…` (`relay.session.tags.saving`). The busy state stays honest as an in-progress label. Flagged for rendered fit. |
| `fr-a1-F08` | minor | `session.loading` | **deferred** (no change) | `Chargement de la conversation…` 30/28; bare `Chargement…` fits but loses the object. Flagged for the centred loading state. |
| `fr-a1-F09` | minor | `session.ended` | **fixed** via MAJ-04 | 38 → 30 cells against `maxCells` 32. |
| `fr-a1-F10` | minor | `session.degraded` | **fixed** | Present tense keeps the wait sense and lands exactly on the budget: 37 → 36/36. |
| `safety-MAJ-04` | **major** | `session.ended` | **fixed** | Ended banner shortened to `Terminée · historique conservé` (30/32) — "history kept" is retained as a state; the subject is safe to drop on the session screen. |
| `safety-MIN-03` | minor | `session.subtitle.waiting` | **fixed** | `Vous attend` (11/24) replaces the literary `En attente de vous` (18/24); crisper native status label, same meaning. |
| `safety-MIN-04` | minor | `session.ended_hint` | **fixed** | `Un envoi` nominalised "a send"; now the verb form `Envoyer un message …`. The button word and `dossier personnel` are preserved. No `maxCells` on this key. |
| `safety-MIN-06` | minor | `session.degraded` | **width fixed; nominalisation rationale rejected** | 37 → 36/36. `envoi(s)` is this locale's settled noun for message sends — `relay.asks.status.delivering` = `Répondu — envoi en cours`, `wire.settings.send_journal_unconfirmed.label` = `Journaliser les envois non acquittés`. The proposed `envois en attente` was rejected: it asserts a queued state the English does not claim. No delivery guarantee is implied by `les envois attendent`. |

No finding was left unanswered. Nothing was deferred that changes meaning; the five residual width signals are listed below.

## Verification executed (actual output)

| Check | Command | Actual output |
|---|---|---|
| Catalogue gate | `PYTHONPATH=. ~/local-operator/.venv/bin/python scripts/i18n/check.py` | `check.py: ok — 182 file(s) scanned, 2253 literal(s) at or below the baseline, catalogues consistent; 34 advisory finding(s) (report-only until activated)` — exit 0, no line naming any locale. The 34 `advisory:` lines are AST-ratchet counts on Python files elsewhere in the tree. |
| Key parity/order | `json` load both sides | `keys: 83 order==en: True` |
| ICU arguments (name AND kind) | `runtime.message_arguments(en[k]) == runtime.message_arguments(fr[k])` | `args identical: True` (all 83) |
| Plural branches | `runtime.plural_selectors` vs generated fr CLDR table | `link.since_seconds ['one','many','other'] cls-ok=True`; `link.since_minutes ['one','many','other'] cls-ok=True`; `session.tray.questions ['one','many','other'] cls-ok=True`; `session.tray.agents_failed ['one','many','other'] cls-ok=True` |
| Select branches | `runtime.render_message(drawer.status, …)` | `online → 'Connecté · ce téléphone'`; `retrying → 'Hors ligne · reconnexion'`; `other → 'Connexion…'` |
| Changed-key render + width | `runtime.render_message(…, 'fr')` + `rich` cell length | `waiting 'Vous attend' 11/24`; `idle 'Inactive' 8/24`; `ended 'Terminée · historique conservé' 30/32`; `ended_hint` 74 cells, no cap; `degraded 'Ne répond pas — les envois attendent' 36/36`; `folder_label 'Dossier de travail␟: minerva' 28/40`; `model_label 'Modèle␟: Opus 5.5 · high' 24/40`; `spend_label 'Dépenses et contexte␟: 31% · $2.47' 34/40` |
| Typography | codepoint scan of all 83 values | `typography: clean` — no ASCII apostrophe, no `...`, no U+202F+space before `:`, guillemets delimited by U+202F, no double spaces |
| Encoding | raw bytes | `bytes: 5404 BOM: False CR: False ends_nl: True` |
| Frozen inputs | `shasum -a 256` | en `f30049d1…6aac`, context `1c727d1c…b69a` — unchanged |

## Residual width signals — deferred to rendered QA (no clipping proof claimed)

| Key | fr cells | `maxCells` | Why kept |
|---|---:|---:|---|
| `relay.chrome.list.search_close` | 19 | 16 | accessible name only — nothing rendered to clip |
| `relay.chrome.list.mark_all` | 21 | 20 | standard French phrasing; shorter form is ungrammatical |
| `relay.chrome.row.unpin` | 11 | 10 | only correct unpin term, pairs with `Épingler` |
| `relay.chrome.row.saving` | 15 | 12 | guide vocabulary (`Enregistrer`), matches existing busy state |
| `relay.chrome.session.loading` | 30 | 28 | dropping the object would mislead |

Three keys sit exactly **at** cap and need no action: `list.clear_search` 20/20, `session.empty_title` 28/28, `session.degraded` 36/36.

## Not addressed by me

- `tests/unit/i18n/test_catalogues.py::test_wave3_hindi_carries_its_namespace_snapshot_with_parity` fails because its expected namespace snapshot predates the new `relay.chrome` namespace (`At index 4 diff: 'relay.chrome' != 'relay.session.tags'`). That is slice-integration bookkeeping for the PR, reported by the first audit, not an fr defect — untouched.
- No other locale, English catalogue, ledger, generator or gate script was read-modified or edited; the six peer catalogues in the worktree were left untouched.
