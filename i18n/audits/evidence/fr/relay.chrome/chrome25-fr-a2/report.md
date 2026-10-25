# Terminal delta verification — `relay.chrome` / French (`fr`)

**Audit ID:** `chrome25-fr-a2` · **Round:** 2 (terminal delta)  
**Date:** 2026-10-25  
**Agent:** translation-auditor (subagent origin, label “Audit chrome fr”), session `3d61f1a19402`, parent `44f93fd45b09`  
**Model:** `radient/auto` — the current runtime header, recorded verbatim. No resolved provider model id is exposed by the harness and none is inferred.  
**Worktree:** `~/local-operator/.worktrees/relay-chrome-i18n` @ `535cea359825411741069cbffcee42e2927df9d3` — read-only; no target, repo or ledger edits.  
**Scope:** delta only — all 8 changed keys, all 10 R1 findings, and direct safety/ICU/terminology neighbours. The 75 byte-identical keys retain their R1 coverage; no full re-audit.

Independence: this verification ran in my R1 session (`3d61f1a19402`), distinct from the translator (`132edc7afe5f`) and the separate safety reviewer. I read my own R1 record and the remediation evidence the task named — `remediation.md`, `remediation.json` and `candidate-pristine.json` — plus the English source, context sidecar, glossary, fr style guide and current target. No translator final summary was used; the on-disk remediation record is the input. The second safety-auditor delta signoff is not mine to claim.

## Hashes (all recomputed with `shasum -a 256`)

| Artefact | Path | SHA-256 |
|---|---|---|
| English source (unchanged since R1) | `local_operator/i18n/catalogues/en/relay.chrome.json` | `f30049d1ec7bce1d0586d8d5d380368e64fb0ac841d91b9b04c1625d17688aac` |
| English context (unchanged since R1) | `local_operator/i18n/catalogues/en/relay.chrome.context.json` | `1c727d1c6c8ddff7c33cbd3abb1ed716e89ab84c864cdb9c319b5df2595fb69a` |
| Target — old (pristine snapshot, = R1 audited target) | `~/.local-operator/sessions/132edc7afe5f/scratchpad/chrome25-fr-remediation/candidate-pristine.json` | `b3a97316412b6b35c0d883a0e79a7cf8b9ed88c99bf298920cd235a2d7a6916f` |
| Target — new (current) | `local_operator/i18n/catalogues/fr/relay.chrome.json` | `5d3a3387e7324a1ed5b1b51104fd06da1470285674f3ce62e24e0eaf33bbb6d2` |

**Final target SHA-256:** `5d3a3387e7324a1ed5b1b51104fd06da1470285674f3ce62e24e0eaf33bbb6d2` — 5,404 bytes, UTF-8 without BOM, LF, trailing newline. The old snapshot hash equals the target hash recorded in my R1 audit, so the delta is measured against exactly the text R1 reviewed.

## Change set — exactly 8 keys

| Key | Old | New | Findings |
|---|---|---|---|
| `relay.chrome.session.subtitle.waiting` | 'En attente de vous' | 'Vous attend' | chrome25-safety-b1-MIN-03 |
| `relay.chrome.session.subtitle.idle` | 'Inactif' | 'Inactive' | chrome25-fr-a1-F02 |
| `relay.chrome.session.ended` | 'Session terminée · historique conservé' | 'Terminée · historique conservé' | chrome25-fr-a1-F09, chrome25-safety-b1-MAJ-04 |
| `relay.chrome.session.ended_hint` | 'Un envoi ou Reprendre la rouvre dans votre dossier personnel' | 'Envoyer un message ou «\u202fReprendre\u202f» la rouvre dans votre dossier personnel' | chrome25-fr-a1-F03, chrome25-safety-b1-MIN-04 |
| `relay.chrome.session.degraded` | 'Ne répond pas — les envois attendront' | 'Ne répond pas — les envois attendent' | chrome25-fr-a1-F10, chrome25-safety-b1-MIN-06 |
| `relay.chrome.composer.folder_label` | 'Dossier de travail\u202f : {name}' | 'Dossier de travail\u202f: {name}' | chrome25-fr-a1-F01 |
| `relay.chrome.composer.model_label` | 'Modèle\u202f : {label}' | 'Modèle\u202f: {label}' | chrome25-fr-a1-F01 |
| `relay.chrome.composer.spend_label` | 'Dépenses et contexte\u202f : {label}' | 'Dépenses et contexte\u202f: {label}' | chrome25-fr-a1-F01 |

`repr` is Python string syntax, so U+202F shows as `\u202f`. Key set and order are identical to `en` (83/83); no key was added, removed or renamed.

## Verification executed — actual output

| Check | Command | Actual output |
|---|---|---|
| sha256 (four artefacts) | `shasum -a 256 <en/context/old/target>` | en f30049d1ec7bce1d0586d8d5d380368e64fb0ac841d91b9b04c1625d17688aac / context 1c727d1c6c8ddff7c33cbd3abb1ed716e89ab84c864cdb9c319b5df2595fb69a / old b3a97316412b6b35c0d883a0e79a7cf8b9ed88c99bf298920cd235a2d7a6916f / new 5d3a3387e7324a1ed5b1b51104fd06da1470285674f3ce62e24e0eaf33bbb6d2 |
| delta diff | `python3: json compare pristine vs current` | key order same: True; counts 83/83; changed 8; exact expected set True; 75 unchanged keys; carry-forward payload sha256 c576f76f… identical |
| repo catalogue gate | `python3 scripts/i18n/check.py` | check.py: ok — 182 file(s) scanned, 2253 literal(s) at or below the baseline, catalogues consistent; 34 advisory finding(s) (report-only until activated) |
| ICU argument tuples (name AND kind) | `runtime.message_arguments(en[k]) == runtime.message_arguments(fr[k]) for all 83 keys` | arg mismatches: none; keys 83 83 |
| plural categories and selectors | `format.plural_categories('fr'); runtime.plural_selectors(...)` | fr categories ('many','one','other'); since_seconds/since_minutes/questions/agents_failed each (('one','many','other'),); 0→one, 1→one, 2→other, 59→other, 1000000→many, 1000001→other |
| changed-key renders and widths | `runtime.render_message(fr[k], params, 'fr')` | waiting 'Vous attend' 11/24; idle 'Inactive' 8/24; ended 'Terminée · historique conservé' 30/32; ended_hint 74 cells no cap; degraded 'Ne répond pas — les envois attendent' 36/36; folder 'Dossier de travail\u202f: minerva' 28/40; model 'Modèle\u202f: Opus 5.5 · high' 24/40; spend 'Dépenses et contexte\u202f: 31% · $2.47' 34/40 |
| full width scan | `render every key with representative params; compare maxCells` | over cap: search_close 19/16, mark_all 21/20, unpin 11/10, saving 15/12, loading 30/28 (the unchanged F04–F08 signals); no other key over cap |
| typography inventory | `codepoint scan of the 83 values` | U+00AB 2, U+00BB 2, U+00B7 15, U+2014 2, U+2019 1, U+2026 9, U+202F 7; U+202F+space 0; ASCII '...' 0; straight apostrophe 0; U+00A0 0; double regular space 0 |
| encoding and key integrity | `raw byte + duplicate-key scan` | 5404 bytes; BOM False; CR False; ends newline True; 83 keys, order == en |
| repo unit tests (observation) | `PYTHONPATH=. ../../.venv/bin/python -m pytest tests/unit/i18n/test_catalogues.py -q` | 1 failed, 21 passed, 232 warnings in 5.95s — FAILED test_wave3_hindi_carries_its_namespace_snapshot_with_parity (namespace snapshot predates relay.chrome); unchanged slice bookkeeping, not an fr defect |

### R1 carry-forward proof

Canonical JSON (sorted keys, UTF-8) of the 75 keys whose values are identical in the pristine snapshot and the current target, hashed in both files:

- old: `c576f76f491f1c8f6e694790bf8fa3d462d115acd0c5dc4facb698061008ea86d`
- new: `c576f76f491f1c8f6e694790bf8fa3d462d115acd0c5dc4facb698061008ea86d`
- identical: **true**; 75 keys

Those 75 keys are byte-identical to the target audited in `chrome25-fr-a1`; R1's accuracy, idiom, register, glossary and placeholder verdicts carry forward for them without re-reading (ICU arguments were re-checked for all 83 anyway: `arg mismatches: none`).

## Disposition of every R1 finding

| Finding | Severity | Keys | Disposition | Verification / reason |
|---|---|---|---|---|
| `chrome25-fr-a1-F01` | minor | `relay.chrome.composer.folder_label`, `relay.chrome.composer.model_label`, `relay.chrome.composer.spend_label` | fixed — accepted | All three now contain U+202F immediately followed by ':' with no second space (ascii: 'Dossier de travail\u202f: {name}', 'Modèle\u202f: {label}', 'Dépenses et contexte\u202f: {label}'). The rest of the fr locale uses 249 occurrences (219 distinct values) of U+202F+':' and zero of U+202F+space+colon. Rendered widths 28/40, 24/40, 34/40. |
| `chrome25-fr-a1-F02` | minor | `relay.chrome.session.subtitle.idle` | fixed — accepted | 'Inactif' → 'Inactive' (8/24 cells). Feminine now agrees with the session and with the sibling 'Terminée'; the fr corpus uses the feminine for sessions ('supprimer les sessions inactives depuis…', 'si la cible est inactive'). |
| `chrome25-fr-a1-F03` | nit | `relay.chrome.session.ended_hint` | fixed — accepted | Button label is now '«\u202fReprendre\u202f»' (U+00AB, U+202F, U+00BB), matching '«\u202fNouvelle\u202f»' in list.empty_hint and the fr guide. The sentence is now 'Envoyer un message ou « Reprendre » la rouvre dans votre dossier personnel' — two coordinated infinitives with singular 'la rouvre', grammatical; 74 cells, no maxCells. |
| `chrome25-fr-a1-F04` | nit | `relay.chrome.list.search_close` | deferred — accepted | Unchanged 19 cells vs maxCells 16, but the role is an accessible name only. Nothing is laid out, so no visible clipping is possible; 'Fermer' would fit but removes the object for screen-reader and voice-control users. Rendered/TTS confirmation remains with the rendered QA pass. |
| `chrome25-fr-a1-F05` | minor | `relay.chrome.list.mark_all` | kept / alternative rejected — accepted | Unchanged 21/20. 'Tout marquer comme lu' is the established French form (the same file's row.mark_read uses 'Marquer comme lu'); the proposed 'Tout marquer lu' is not grammatical French. One cell over budget is a rendered-fit question, not a semantic defect. |
| `chrome25-fr-a1-F06` | minor | `relay.chrome.row.unpin` | deferred — accepted | Unchanged 11/10; 'Désépingler' is the only correct French term and pairs with the sibling 'Épingler'. Flagged for the action-sheet/swipe-tile render. |
| `chrome25-fr-a1-F07` | minor | `relay.chrome.row.saving` | deferred — accepted | Unchanged 15/12; 'Enregistrement…' follows the guide's Enregistrer/Enregistré vocabulary and the existing busy-state noun pattern. 'Sauvegarde…' would fit but drift. Rendered fit pending. |
| `chrome25-fr-a1-F08` | minor | `relay.chrome.session.loading` | deferred — accepted | Unchanged 30/28; bare 'Chargement…' would lose the object. Rendered fit in the centred loading state pending. |
| `chrome25-fr-a1-F09` | minor | `relay.chrome.session.ended` | fixed — accepted | Fixed as 'Terminée · historique conservé' (30/32 cells). Semantics preserved: the session is ended and its history is kept; the adjective agrees with 'session' and matches the unchanged subtitle 'Terminée'. On the session banner the subject is unambiguous. |
| `chrome25-fr-a1-F10` | minor | `relay.chrome.session.degraded` | fixed — accepted | Fixed as 'Ne répond pas — les envois attendent' (36/36 cells). Present tense keeps the wait sense without asserting a queue; 'envois' is the locale's settled noun for message sends (relay.asks.status.delivering 'Répondu — envoi en cours'; wire.settings.send_journal_unconfirmed.label 'Journaliser les envois non acquittés'). |

No R1 finding is unanswered. F04–F08 are width signals retained for the rendered-fit pass; none is a blocker or major, and no cosmetic churn is requested. The remediation's rejection of a shorter `Tout marquer lu` is correct — `marquer comme lu` is the grammatical form used by this file's own `row.mark_read`.

## Safety and terminology neighbours

- **Approve:** PASS — session.tray.approve 'Approuver' and session.tray.approve_label 'Approuver — une demande attend votre réponse' are byte-identical to the R1-audited target (carried forward).
- **Ended / resume:** PASS — session.ended now 'Terminée · historique conservé' and session.ended_hint 'Envoyer un message ou « Reprendre » la rouvre dans votre dossier personnel'; both preserve ended-but-kept and reopen-in-home-folder semantics, and session.resume/resuming are unchanged. No ambiguity introduced.
- **Degraded:** PASS — 'Ne répond pas — les envois attendent' still states sends wait; no delivery guarantee. session.subtitle.not_answering 'Ne répond pas' unchanged.
- **Waiting:** PASS — 'Vous attend' accurately renders 'Waiting for you'; consistent with the 'attend' verb in approve_label; 11/24 cells.
- **Terminology:** `Vous attend` reuses the same `attendre` as `approve_label` (`une demande attend votre réponse`); `Terminée` matches `subtitle.ended`; `envois` matches the settled corpus nouns `Répondu — envoi en cours` (`relay.asks.status.delivering`) and `Journaliser les envois non acquittés` (`wire.settings.send_journal_unconfirmed.label`); `la rouvre` has existing precedent (`wire.slash.stop`); `dossier personnel` is unchanged from R1. No drift introduced.
- **ICU:** changed keys carry no ICU structure; all 83 argument tuples (name and kind) still match `en`, all four plurals keep `one`/`many`/`other` with `#` in every branch, and the `{state}` select branches are unchanged.

## New findings

**None.** No changed key introduces an accuracy, idiom, register, glossary, placeholder, plural, overflow or safety defect. No out-of-scope issue was found in the neighbours inspected; no nit loop is opened.

## Verdict

**TERMINAL PASS.** 83/83 keys read across R1; 8 changed keys re-reviewed here; 75 unchanged keys proven byte-identical and carried forward. Zero blocker, zero major, zero new findings. Residual width signals (`search_close` 19/16 a11y-only; `mark_all` 21/20; `unpin` 11/10; `saving` 15/12; `loading` 30/28) remain deferred to the rendered-fit pass, as instructed — they are signals, not semantic defects.

## Limitations

- No rendered app evidence in this pass; width signals are cell counts against the en sidecar maxCells and are deferred to the rendered QA pass.
- The pytest namespace-snapshot failure is slice-integration bookkeeping for the PR, unchanged by this delta and not an fr translation defect.

Hash-bound receipt: locale `fr`, namespace `relay.chrome`, old `b3a97316412b6b35c0d883a0e79a7cf8b9ed88c99bf298920cd235a2d7a6916f`, new `5d3a3387e7324a1ed5b1b51104fd06da1470285674f3ce62e24e0eaf33bbb6d2`, en/context unchanged, verdict TERMINAL PASS.
