# Translation audit — `relay.chrome` / French (`fr`)

**Audit ID:** `chrome25-fr-a1`  
**Date:** 2026-10-25  
**Agent:** translation-auditor (subagent origin), session `3d61f1a19402`, parent `44f93fd45b09`  
**Model:** `radient/auto` — as reported by the session header. `origin.json` and `transcript.jsonl` record no resolved provider/model id; none is guessed.  
**Worktree:** `~/local-operator/.worktrees/relay-chrome-i18n`, base/HEAD `535cea359825411741069cbffcee42e2927df9d3` (read-only; no commits)  
**Scope:** every key of `relay.chrome` in `fr` — **83/83 keys read**.

Independence: this audit ran in a different session from the translators. It read the English source, the English context sidecar, the French target, `i18n/glossary.md`, `i18n/style-guides/fr.md`, existing `fr` catalogues for consistency, and the repository's i18n validation code. No translator report or provenance file was read. No repository file was edited and no ledger entry was touched.

## Artefact hashes (`shasum -a 256`)

| Artefact | Path | SHA256 |
|---|---|---|
| English source | `local_operator/i18n/catalogues/en/relay.chrome.json` | `f30049d1ec7bce1d0586d8d5d380368e64fb0ac841d91b9b04c1625d17688aac` |
| English context sidecar | `local_operator/i18n/catalogues/en/relay.chrome.context.json` | `1c727d1c6c8ddff7c33cbd3abb1ed716e89ab84c864cdb9c319b5df2595fb69a` |
| French target | `local_operator/i18n/catalogues/fr/relay.chrome.json` | `b3a97316412b6b35c0d883a0e79a7cf8b9ed88c99bf298920cd235a2d7a6916f` |

All three match the hashes supplied with the task. The French file is untracked (`??`), 5,402 bytes, UTF-8 without BOM, LF endings, trailing newline.

## Method

1. Read the English source, the context sidecar (role, tone, `maxCells`, per-key notes) and the French target side by side, key by key, in source order.
2. Parsed every message with the project's own ICU subset (`local_operator.i18n.runtime`) and compared argument tuples — names **and** kinds — across all 83 keys.
3. Checked plural selectors against the generated CLDR table, then rendered representative branches (0, 1, 2, 59, 10⁶, 10⁶+1) and every `{state}` select branch.
4. Scanned code points for typography: guillemets, apostrophe, ellipsis, middle dot, em dash, narrow no-break space, ASCII fallbacks, NBSP, double spaces.
5. Rendered every key with realistic parameters, compared the result to the sidecar `maxCells` budgets and to the English rendered width.
6. Checked register (`vous`/`tu`), sentence case, product-name and kept-English policy, and the fr guide's locale calls; compared terminology with existing `fr` catalogues.
7. Ran the repository catalogue gate and, as an observation only, the catalogue unit tests.
8. Ran in-memory negative controls (no file mutation) to prove the checks can fail.

## Executed checks — actual output

| Check | Command | Actual output |
|---|---|---|
| sha256 source/context/target | `shasum -a 256 <en/context/target>` | en f30049d1ec7bce1d0586d8d5d380368e64fb0ac841d91b9b04c1625d17688aac / context 1c727d1c6c8ddff7c33cbd3abb1ed716e89ab84c864cdb9c319b5df2595fb69a / target b3a97316412b6b35c0d883a0e79a7cf8b9ed88c99bf298920cd235a2d7a6916f |
| repo catalogue gate | `python3 scripts/i18n/check.py` | check.py: ok — 182 file(s) scanned, 2253 literal(s) at or below the baseline, catalogues consistent; 34 advisory finding(s) (report-only until activated) |
| ICU argument tuples (name AND kind) | `runtime.message_arguments(en[k]) == runtime.message_arguments(fr[k]) for all 83 keys` | arg mismatches: none / fr key order == en order: True |
| CLDR plural categories | `format.plural_categories('fr'); plural_category('fr', n)` | fr categories: ('many', 'one', 'other') / 0 one / 1 one / 2 other / 59 other / 1000000 many / 1000001 other / 1500000 other |
| plural branch audit | `runtime.plural_arguments(fr[k])` | since_seconds count [('one', True, False), ('many', True, False), ('other', True, False)] / since_minutes count [('one', True, False), ('many', True, False), ('other', True, False)] / tray.questions count [('one', True, False), ('many', True, False), ('other', True, False)] / tray.agents_failed failed [('one', True, False), ('many', True, False), ('other', True, False)]  # True = branch renders #, False = branch non-empty |
| rendered ICU samples | `runtime.render_message(fr[k], params, 'fr')` | since_seconds 1 -> 'Reconnexion · synchro il y a 1 seconde' / since_seconds 2 -> 'Reconnexion · synchro il y a 2 secondes' / since_seconds 1000000 -> 'Reconnexion · synchro il y a 1\u202f000\u202f000 de secondes' / tray.questions 0 -> '0 question' / tray.questions 3 -> '3 questions' / tray.questions 1000000 -> '1\u202f000\u202f000 de questions' / agents_failed {failed:1,running:2,total:4} -> 'Agents 2/4 · 1 en échec' / drawer.status online -> 'Connecté · ce téléphone' / drawer.status retrying -> 'Hors ligne · reconnexion' / drawer.status other -> 'Connexion…' / drawer.status unknown('up') -> 'Connexion…' (falls to other) |
| typography inventory | `codepoint scan of the 83 fr values` | U+00AB 1, U+00BB 1, U+2019 1, U+2026 9, U+202F 5, U+2014 2, U+00B7 15 / ASCII three dots: [] / straight apostrophe: [] / U+00A0: [] / double regular space: [] / U+202F+space: ['relay.chrome.composer.folder_label','relay.chrome.composer.model_label','relay.chrome.composer.spend_label'] / other fr catalogues: U+202F+space+colon = 0; U+202F+colon = 249 |
| encoding/key integrity | `raw byte + duplicate-key scan` | BOM: False, ends_newline: True, CR: False, size: 5402 / raw key occurrences: 83, unique: 83 / fr extra/missing vs en: [] / [] |
| width scan | `render every key with representative params; compare with context maxCells` | 7 over cap: search_close 19/16, mark_all 21/20, unpin 11/10, saving 15/12, loading 30/28, ended 38/32, degraded 37/36; at cap: clear_search 20/20, empty_title 28/28; 74 within cap |
| register scan | `regex scan for tu/ton/ta/tes/te/toi/Veuillez/Merci de` | 0 hits for every informal or bureaucratic form; vous/votre forms only |
| negative controls (in memory, no file mutation) | `compare argument tuples; check._check_plural_categories` | renamed {name}->{nom} detected: True / number -> plural re-kind detected: True / few (invalid for fr): ["fr/relay.chrome: synthetic: plural category 'few' not in ['many', 'one', 'other']"] / no other: ["fr/relay.chrome: synthetic: plural has no `other` branch"] / missing many: no problem raised (the repo check does not enforce a direct plural's full CLDR set; verified manually here) |
| repo unit tests (observation) | `PYTHONPATH=. ../../.venv/bin/python -m pytest tests/unit/i18n/test_catalogues.py -q` | 1 failed, 21 passed, 232 warnings in 4.00s / FAILED tests/unit/i18n/test_catalogues.py::test_wave3_hindi_carries_its_namespace_snapshot_with_parity /   AssertionError: At index 4 diff: 'relay.chrome' != 'relay.session.tags' /   (expected namespace snapshot predates the new relay.chrome namespace; not an fr translation defect) |

**Notes on the two non-clean outputs.** The catalogue gate's `check.py: ok` is authoritative and passes: per-namespace key sets, argument tuples and plural selectors are consistent, every plural has `other`, every selector is a category of the locale. Its 34 `advisory:` lines are AST-ratchet literal counts elsewhere in the tree, unrelated to this catalogue. The unit-test failure is `test_wave3_hindi_carries_its_namespace_snapshot_with_parity`, whose expected namespace snapshot predates the new `relay.chrome` namespace (`At index 4 diff: 'relay.chrome' != 'relay.session.tags'`); that is slice-integration bookkeeping for the PR, **not an fr translation defect** — observed and reported, not fixed.

Negative controls: a renamed placeholder (`{name}`→`{nom}`) is detected; a `number`→`plural` re-kind is detected; `few` is rejected for `fr`; a plural without `other` is rejected. A *missing* `many` is **not** caught by `check.py` for a plural that is already plural in English, so fr `many` coverage was verified manually: all four plural keys carry `one`, `many`, `other`, and every branch renders `#`.

## Findings

No **blocker** and no **major** finding. Two minor language/typography fixes (F01, F02), one typography nit (F03), and seven width signals (F04 nit, F05–F10 minor).

### chrome25-fr-a1-F01 — minor — typography

The narrow no-break space is shown escaped as `\u202f` so the two characters before the colon are visible.

| Key | Current | Proposed |
|---|---|---|
| `relay.chrome.composer.folder_label` | `Dossier de travail\u202f : {name}` | `Dossier de travail\u202f: {name}` |
| `relay.chrome.composer.model_label` | `Modèle\u202f : {label}` | `Modèle\u202f: {label}` |
| `relay.chrome.composer.spend_label` | `Dépenses et contexte\u202f : {label}` | `Dépenses et contexte\u202f: {label}` |

**Why:** U+202F followed by U+0020 leaves a two-character gap before the colon. The fr guide requires a single espace fine insécable before ':', and the rest of the fr locale agrees: 249 occurrences (219 distinct values) use U+202F immediately before the colon and **zero** use the U+202F+space pattern. Fixing `spend_label` also brings its rendered width to 40, exactly at its `maxCells`.

### chrome25-fr-a1-F02 — minor — grammar/consistency

| Key | Current | Proposed |
|---|---|---|
| `relay.chrome.session.subtitle.idle` | `Inactif` | `Inactive` |

**Reason:** the status line describes the session (*la session*): the same slot renders `Terminée` (feminine), `En attente de vous` and `Ne répond pas`. `Inactif` is the only masculine form in the set, so the gender flips between states. Making the set masculine (`Terminée` → `Terminé`) is the alternative, but `Terminée` is the established fr rendering and matches the session noun.

### chrome25-fr-a1-F03 — nit — typography/consistency

| Key | Current | Proposed |
|---|---|---|
| `relay.chrome.session.ended_hint` | `Un envoi ou Reprendre la rouvre dans votre dossier personnel` | `Un envoi ou « Reprendre » la rouvre dans votre dossier personnel` |

**Reason:** the same file marks the other referenced button as `« Nouvelle »` (`list.empty_hint`), and the fr guide prescribes `« … »`. An unquoted capitalised verb mid-sentence reads as a proper noun. No `maxCells` on this key.

### chrome25-fr-a1-F04 — nit — width signal

| Key | Current | Proposed |
|---|---|---|
| `relay.chrome.list.search_close` | `Fermer la recherche` | keep; a11y-only, no visual clipping — 'Fermer' (6) only if the cap is enforced for accessible names |

**Reason:** 19 cells vs maxCells 16. Role is 'search field close button accessible name', so the overage does not truncate a visible label; flagged for the rendered/TTS check.

### chrome25-fr-a1-F05 — minor — width signal

| Key | Current | Proposed |
|---|---|---|
| `relay.chrome.list.mark_all` | `Tout marquer comme lu` | keep (standard fr; no shorter idiomatic form) — confirm at 360 px |

**Reason:** 21 cells vs maxCells 20 on a visible ghost button; over by 1. 'Tout marquer comme lu' is the established French phrasing, so shortening would cost idiom.

### chrome25-fr-a1-F06 — minor — width signal

| Key | Current | Proposed |
|---|---|---|
| `relay.chrome.row.unpin` | `Désépingler` | keep — no shorter synonym — confirm on the action-sheet row |

**Reason:** 11 cells vs maxCells 10; 'Désépingler' is the only correct French term for unpin.

### chrome25-fr-a1-F07 — minor — width signal

| Key | Current | Proposed |
|---|---|---|
| `relay.chrome.row.saving` | `Enregistrement…` | keep; 'Sauvegarde…' (11) only if it clips, at the cost of drifting from the guide's Enregistrer/Enregistré vocabulary |

**Reason:** 15 cells vs maxCells 12 for the visible busy label that replaces the pin action.

### chrome25-fr-a1-F08 — minor — width signal

| Key | Current | Proposed |
|---|---|---|
| `relay.chrome.session.loading` | `Chargement de la conversation…` | keep; 'Chargement…' only if it clips (loses the object) |

**Reason:** 30 cells vs maxCells 28 on the conversation loading state.

### chrome25-fr-a1-F09 — minor — width signal

| Key | Current | Proposed |
|---|---|---|
| `relay.chrome.session.ended` | `Session terminée · historique conservé` | keep; if it clips: 'Terminée · historique conservé' (30) |

**Reason:** 38 cells vs maxCells 32 — the largest visible-banner overage; meaning is correct, so this is a fit question for the rendered check.

### chrome25-fr-a1-F10 — minor — width signal

| Key | Current | Proposed |
|---|---|---|
| `relay.chrome.session.degraded` | `Ne répond pas — les envois attendront` | Ne répond pas — les envois attendent |

**Reason:** 37 cells vs maxCells 36; the present tense keeps the warning's meaning ('messages are kept and delivered when it answers') and lands exactly on the budget.

## Safety-related copy — explicit review

| Surface | Keys | Verdict |
|---|---|---|
| Approve pill | `session.tray.approve`, `session.tray.approve_label` | **PASS.** Visible pill `Approuver`; the accessible name starts with the same word (a sidecar requirement) and says `une demande attend votre réponse`, matching existing `relay.approvals` wording `Cette demande attend toujours votre réponse.` Nothing implies the approval is already given or automatic, and the request is called a `demande`, not a `question` — the same distinction the English draws. |
| Ended / Resume | `session.subtitle.ended`, `session.ended`, `session.ended_hint`, `session.resume`, `session.resuming` | **PASS.** `Session terminée · historique conservé` conveys ended-but-kept; `Un envoi ou Reprendre la rouvre dans votre dossier personnel` conveys that either sending or Resume reopens it, in the home folder rather than the project folder. `Reprendre` matches `relay.chrome.session.resume` and existing fr CLI wording. Only the typography nit F03 (quoting the button label) applies. |
| Degraded banner | `session.subtitle.not_answering`, `session.degraded` | **PASS.** `Ne répond pas — les envois attendront` matches “messages typed now are kept and delivered when it answers again”, and is consistent with the subtitle `Ne répond pas`. No action is implied. |

No safety-relevant mistranslation was found.

## Verdict per namespace group

| Group | Keys | Verdict |
|---|---|---|
| `relay.chrome.drawer.*` | 12 | PASS — clean |
| `relay.chrome.link.*` | 3 | PASS — clean; 'synchro' considered idiomatic clipping, kept because 'dernière synchro' would be 49 > 44 cells |
| `relay.chrome.list.*` | 21 | PASS — F04 (nit), F05 (minor width) |
| `relay.chrome.snackbar.*` | 1 | PASS — clean |
| `relay.chrome.row.*` | 10 | PASS — F06, F07 (minor width) |
| `relay.chrome.appearance.*` | 5 | PASS — clean; 'Oui/Non' justified by the 6-cell cap (Activé/Désactivé do not fit) |
| `relay.chrome.session.*` | 25 | PASS — F02 (minor), F03 (nit), F08–F10 (minor width); safety copy correct |
| `relay.chrome.composer.*` | 4 | PASS — F01 (minor, 3 keys) |
| `relay.chrome.spend.*` | 1 | PASS — clean |
| `relay.chrome.model.*` | 1 | PASS — clean |
| **Total** | **83** | |

## Width signals (rendered with the in-house formatter)

`maxCells` comes from the English sidecar. Values are cells of the rendered string with representative parameters (`name=minerva`, `label=31% · $2.47`, counts 1/2/59, tasks 2/5, agents 2/4). These are **signals**, not semantic defects; rendered app evidence is a separate follow-up and is not claimed here.

| Key | EN rendered | FR rendered | maxCells | Status |
|---|---|---|---|---|
| `relay.chrome.list.search_close` | 12 | 19 | 16 | **OVER** |
| `relay.chrome.list.mark_all` | 13 | 21 | 20 | **OVER** |
| `relay.chrome.list.clear_search` | 12 | 20 | 20 | **AT_CAP** |
| `relay.chrome.row.unpin` | 5 | 11 | 10 | **OVER** |
| `relay.chrome.row.saving` | 7 | 15 | 12 | **OVER** |
| `relay.chrome.session.loading` | 21 | 30 | 28 | **OVER** |
| `relay.chrome.session.empty_title` | 15 | 28 | 28 | **AT_CAP** |
| `relay.chrome.session.ended` | 28 | 38 | 32 | **OVER** |
| `relay.chrome.session.degraded` | 31 | 37 | 36 | **OVER** |

All other 74 keys render within budget; the longest of those are `link.since_seconds` 40/44, `composer.folder_label` 39/40 (38 after F01) and `session.tray.approve_label` 44/48.

## Observations accepted (no change requested)

- link.since_*: 'synchro' is an idiomatic French clipping of synchronisation; 'dernière synchro' would be 49 cells > maxCells 44, so the short form is justified. Meaning ('synced # seconds ago') is preserved.
- list.no_matches: 'Aucun résultat' generalises 'No matching conversations', the standard French search empty state; no ambiguity in a search field.
- list.filter.needs_you: 'À traiter' is a compact idiomatic rendering of 'Needs you'; the chip's accessible group label carries the 'conversations' context.
- list.new: 'Nouvelle' is an elliptical label agreeing with 'Nouvelle session' (the accessible name); the 10-cell budget excludes the noun.
- appearance.on/off: 'Oui'/'Non' rather than 'Activé'/'Désactivé' because 'Désactivé' is 9 > maxCells 6; the value answers the toggle's label naturally.
- Glossary check: session→session, project→projet, usage→utilisation, cost→coût (spend rendered as 'dépenses' here), job/task→tâche, settings→paramètres, appearance→apparence, library→bibliothèque, agent→agent; the kept-English set is respected (Local Operator, tags; no tool-name prose in this namespace). No drift between the chrome strings and the existing fr catalogues.

## Limitations

- No rendered app/browser evidence in this pass; rendered evidence is explicitly deferred and not claimed here. Width findings are cell-count signals against the en sidecar maxCells and the in-house renderer, not pixel proof.
- The repo catalogue gate does not enforce the full CLDR category set for plurals that are already plurals in en; the fr many category was therefore verified manually (present in all four plural keys).
- The unit-test failure is a slice-integration expectation (namespace snapshot not yet updated for relay.chrome), not an fr translation defect; it was observed, not fixed.

RTL: not applicable — `fr` is LTR and this namespace has no direction-sensitive token or embedded RTL run.

## Overall verdict

**PASS with minor findings** — no blocker, no major. All 83 keys read. Accuracy, idiom and register are consistently French (professional `vous`, sentence case, terse statuses); ICU arguments match exactly by name and kind; all four plurals carry the full French CLDR set (`one`/`many`/`other`) with the number present in every branch; no glossary drift; safety copy for Approve, ended/Resume and degraded is correct.

Recommended before merge: F01 (three composer accessible names, one-cause typography fix) and F02 (`Inactif`→`Inactive`). F03 is a nit. F04–F10 need the rendered pass to decide whether the budgets actually clip.

## Appendix — key coverage (83/83)

Each key was read; keys touched by a finding carry its ID.

**drawer.** `relay.chrome.drawer.open`, `relay.chrome.drawer.close`, `relay.chrome.drawer.label`, `relay.chrome.drawer.conversations`, `relay.chrome.drawer.questions`, `relay.chrome.drawer.past`, `relay.chrome.drawer.projects`, `relay.chrome.drawer.library`, `relay.chrome.drawer.usage`, `relay.chrome.drawer.appearance`, `relay.chrome.drawer.settings`, `relay.chrome.drawer.status`

**link.** `relay.chrome.link.reconnecting`, `relay.chrome.link.since_seconds`, `relay.chrome.link.since_minutes`

**list.** `relay.chrome.list.search_open`, `relay.chrome.list.search_close` — F04 (nit), `relay.chrome.list.search_label`, `relay.chrome.list.filter_label`, `relay.chrome.list.filter.all`, `relay.chrome.list.filter.needs_you`, `relay.chrome.list.filter.unread`, `relay.chrome.list.filter_empty`, `relay.chrome.list.mark_all` — F05 (minor), `relay.chrome.list.marking`, `relay.chrome.list.section.pinned`, `relay.chrome.list.section.active`, `relay.chrome.list.section.previous`, `relay.chrome.list.connecting`, `relay.chrome.list.empty`, `relay.chrome.list.empty_hint`, `relay.chrome.list.no_matches`, `relay.chrome.list.clear_search`, `relay.chrome.list.new`, `relay.chrome.list.new_label`, `relay.chrome.list.starting`

**snackbar.** `relay.chrome.snackbar.dismiss`

**row.** `relay.chrome.row.actions`, `relay.chrome.row.pin`, `relay.chrome.row.unpin` — F06 (minor), `relay.chrome.row.saving` — F07 (minor), `relay.chrome.row.tags`, `relay.chrome.row.tags_none`, `relay.chrome.row.mark_read`, `relay.chrome.row.open`, `relay.chrome.row.swipe_pin`, `relay.chrome.row.more`

**appearance.** `relay.chrome.appearance.theme`, `relay.chrome.appearance.wide_view`, `relay.chrome.appearance.group_by_tag`, `relay.chrome.appearance.on`, `relay.chrome.appearance.off`

**session.** `relay.chrome.session.back`, `relay.chrome.session.connecting`, `relay.chrome.session.loading` — F08 (minor), `relay.chrome.session.empty_title`, `relay.chrome.session.empty_hint`, `relay.chrome.session.subtitle.working`, `relay.chrome.session.subtitle.working_plain`, `relay.chrome.session.subtitle.waiting`, `relay.chrome.session.subtitle.idle` — F02 (minor), `relay.chrome.session.subtitle.ended`, `relay.chrome.session.subtitle.not_answering`, `relay.chrome.session.tray.label`, `relay.chrome.session.tray.approve`, `relay.chrome.session.tray.approve_label`, `relay.chrome.session.tray.questions`, `relay.chrome.session.tray.tasks`, `relay.chrome.session.tray.agents`, `relay.chrome.session.tray.agents_failed`, `relay.chrome.session.tasks_title`, `relay.chrome.session.agents_title`, `relay.chrome.session.ended` — F09 (minor), `relay.chrome.session.ended_hint` — F03 (nit), `relay.chrome.session.resume`, `relay.chrome.session.resuming`, `relay.chrome.session.degraded` — F10 (minor)

**composer.** `relay.chrome.composer.label`, `relay.chrome.composer.folder_label` — F01 (minor), `relay.chrome.composer.model_label` — F01 (minor), `relay.chrome.composer.spend_label` — F01 (minor)

**spend.** `relay.chrome.spend.title`

**model.** `relay.chrome.model.effort`

