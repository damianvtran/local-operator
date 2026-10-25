# `hi/relay.chrome.json` — round-2 terminal DELTA audit

**Audit ID:** chrome25-hi-a2 · **Locale:** `hi` · **Namespace:** `relay.chrome` (83 keys) · **Round:** 2 (delta)
**Verdict: TERMINAL PASS** — no blocker or major remains. 7 changed keys re-audited; 76 keys carried forward by byte equality; 1 documented width residual for the rendered pass.

---

## 1. Hash-binding receipt

Every value below was recomputed in this run from the bytes on disk — none copied from prose.

| Artefact | SHA-256 | How verified |
|---|---|---|
| target **before** (R1 target = the record's pre-remediation snapshot) | `a29563f5dd0a9f425b717639c28e6a8f4c97382d45b0928bacdf5e12d31285b3` | `hashlib.sha256` on `~/.local-operator/sessions/6f3832cb71c0/scratchpad/hi-relay-chrome.candidate-pre-remediation.json`; equals my own R1 `record.json` target |
| target **after** (current worktree file) | `eb9f326ca6689ed0d35311cd75ae8cb5683b515e8b638c938790759eeec96784` | `hashlib.sha256` on `~/.local-operator/.worktrees/relay-chrome-i18n/local_operator/i18n/catalogues/hi/relay.chrome.json` |
| `en/relay.chrome.json` | `f30049d1ec7bce1d0586d8d5d380368e64fb0ac841d91b9b04c1625d17688aac` | unchanged vs R1, vs the brief, and vs the remediation record |
| `en/relay.chrome.context.json` | `1c727d1c6c8ddff7c33cbd3abb1ed716e89ab84c864cdb9c319b5df2595fb69a` | unchanged vs R1, vs the brief, and vs the remediation record |
| base commit | `535cea359825411741069cbffcee42e2927df9d3` | `git rev-parse HEAD` in the worktree |

Target remains an **untracked** working-tree file (`git status --porcelain` shows the same seven untracked locale catalogues as at R1); re-hashed after all my reads, unchanged.

## 2. Change scope — exact, byte-wise

Byte diff of the pre-remediation snapshot against the current file: **7 changed keys, 76 identical, 83 keys, key order identical, one JSON line each, values only** (23 diff lines, all value lines — no reformatting, no key added/removed/reordered).

| key | before | after |
|---|---|---|
| `drawer.status` | `… retrying {ऑफ़लाइन · फिर कोशिश} other {कनेक्ट हो रहा है…}}` | `… retrying {ऑफ़लाइन · फिर से कोशिश} other {…}}` (only the `retrying` branch) |
| `list.filter.needs_you` | `आपके लिए · {count, number}` | `आपका ध्यान चाहिए · {count, number}` |
| `list.mark_all` | `सभी पढ़ा हुआ चिह्नित करें` | `सभी को पढ़ा हुआ चिह्नित करें` |
| `row.actions` | `{name} के लिए क्रियाएँ` | `{name} के लिए कार्रवाइयाँ` |
| `row.saving` | `सहेज रहे हैं…` | `सहेजा जा रहा है…` |
| `session.subtitle.waiting` | `आपका इंतज़ार` | `आपका इंतज़ार है` |
| `session.ended_hint` | `संदेश भेजने या “जारी रखें” से यह आपके होम फ़ोल्डर में फिर खुलेगा` | `संदेश भेजने पर या “जारी रखें” दबाने पर यह आपके होम फ़ोल्डर में फिर खुलेगा` |

This matches the remediation record's `changed_keys` exactly (independent comparison, not a read of its table).

## 3. R1 carry-forward proof

R1 audited all 83 keys at exactly `a29563f5…`. Per-key comparison of that snapshot against the current file: **76 keys byte-identical**, and those 76 include every safety-critical string:

`session.tray.approve`, `session.tray.approve_label`, `session.subtitle.ended`, `session.ended`, `session.resume`, `session.resuming`, `session.degraded`, `session.subtitle.not_answering`, `session.empty_title/hint`, `session.tray.label`, `composer.*` — all `True` for `pre[k] == cur[k]`.

`drawer.status` is the only safety-adjacent key in the change set; its `online` and `other` branches and the `{state, select, …}` structure are byte-identical, and only the non-safety `retrying` descriptor changed.

No unchanged key was re-litigated; no full namespace re-audit was performed.

## 4. Dispositions — each R1 finding

| R1 finding | Sev | Disposition | Auditor verdict on the disposition |
|---|---|---|---|
| a1-01 `needs_you` | major | **fixed**, as `आपका ध्यान चाहिए · {count, number}` (my proposal was `आपका इंतज़ार · …`) | **Accepted.** The original defect — a "for you" reading that loses "the operator must act" — is gone; the applied form states the need explicitly. Its collision reasoning is sound: keeping `आपका इंतज़ार` there would have duplicated the liveness line's claim. |
| a1-02 `subtitle.waiting` | minor | **fixed**, as `आपका इंतज़ार है` (my proposal `आपके इंतज़ार में`) | **Accepted.** The defect was a missing predicate; `है` completes it, and the reviewer is right that my locative also needed `है` (`आपके इंतज़ार में है`). Same vocabulary, 10/24 cells. |
| a1-03 `model.effort` | minor | **rejected — retained `तर्क-प्रयास`** | **Accepted as correct.** My premise was wrong: the corpus ships `तर्क-प्रयास` three times — `wire.settings.model_effort.label` (`डिफ़ॉल्ट तर्क-प्रयास`), `wire.settings.model.description`, `wire.slash.effort`. The model sheet therefore matches the control it opens; moving it to `रीज़निंग प्रयास` would have split this screen from that control. No churn requested. |
| a1-04 `row.actions` | minor | **fixed** → `कार्रवाइयाँ` | **Accepted**, with a correction recorded: `क्रिया` *is* attested elsewhere (`wire.settings.aida.description`, `अगली क्रिया`), so my "only occurrence" premise was wrong. The disposition stands on the corpus term for an actions menu (`tui.transcript.group.actions`), not on absence. |
| a1-05 `ended_hint` | minor | **fixed** → `… भेजने पर या “जारी रखें” दबाने पर …` | **Accepted**, applied close to the proposal; both coordinates now conditional. |
| a1-06 `mark_all` | nit | **fixed** → `सभी को …` | **Accepted**, 18/20 cells measured. |
| a1-07 busy voice | nit | **fixed, opposite direction** — passive restored on `row.saving` | **Accepted; my R1 claim was wrong.** I asserted `सहेजा जा रहा है…` was 16 cells without executing the measurement. Re-measured now with `rich.cells.cell_len` on the exact string: **11 cells** of `maxCells` 12. The passive fits, and it is byte-identical to the already-shipped `relay.session.tags.saving` (verified `True`). Their direction is better than mine. |
| a1-08 `drawer.status` | nit | **fixed** → `फिर से कोशिश` | **Accepted**; corpus majority confirmed (15 `फिर से कोशिश` incl. the new string; 7 bare `फिर कोशिश`, all imperative `… करें`). |

Safety-stream findings on the same keys (`chrome25-safety-b1 MIN-07` → `needs_you`, `MIN-08` → `mark_all`) land on identical strings; that stream's own delta signoff is its auditor's, and is **not** claimed here.

## 5. Re-audit of the changed strings (accuracy · idiom · register · glossary)

- `आपका ध्यान चाहिए · {count, number}` — "needs your attention · N". Agreement correct (`आपका` with masc. `ध्यान`), `आप` register, chip idiom natural, middle dot and `{count, number}` intact. `ध्यान` is new to this locale (1 occurrence corpus-wide) but plain Hindi, not a calque and not a term the glossary fixes. Neighbours `सभी` / `अपठित · {count}` / `filter_empty` unaffected.
- `आपका इंतज़ार है` — complete predicate, matching the four sibling liveness labels (`कार्यरत`, `आइडल`, `समाप्त`, `उत्तर नहीं दे रहा`); no longer duplicates the approve pill's wording (that line keeps `आपका इंतज़ार कर रहा है`).
- `सभी को पढ़ा हुआ चिह्नित करें` — object marker restored; identical complement to `row.mark_read`, consistent with `list.marking`'s `चिह्नित` root.
- `{name} के लिए कार्रवाइयाँ` — the corpus's actions-menu term, and the direct plural is right because the noun is the head, not governed by `के लिए`. 21/40 cells at a realistic name.
- `सहेजा जा रहा है…` — passive busy state, now identical to `relay.session.tags.saving`; with `list.marking` both busy labels in this file share one voice.
- `ऑफ़लाइन · फिर से कोशिश` — selectors `online`/`retrying`/`other` unchanged; matches `फिर से कनेक्ट हो रहा है` in `link.reconnecting`.
- `संदेश भेजने पर या “जारी रखें” दबाने पर यह आपके होम फ़ोल्डर में फिर खुलेगा` — parallel conditionals, `दबाने` matches the file's `दबाएँ`; still quotes the Resume label verbatim (`… “जारी रखें” …` = `session.resume`) and still states the home-folder destination (sidecar's safety-relevant point). 51 cells, no budget.

No new issue was found in the change set or its direct neighbours. Nothing here warrants another round.

## 6. Executed checks (actual output, this run)

```
$ env -i HOME=$ISO … ../../.venv/bin/python scripts/i18n/check.py
check.py: ok — 182 file(s) scanned, 2253 literal(s) at or below the baseline,
catalogues consistent; 34 advisory finding(s) (report-only until activated)     EXIT=0
```
No advisory line mentions `hi/` or `relay.chrome`.

- Key parity: `en 83 · context 83 · hi 83`, sets and **order** identical.
- ICU arguments, all 83 keys, name **and** kind: **0 mismatches**; plural selectors `[]` differences (four plural keys all `one`+`other`; `drawer.status` `online`/`retrying`/`other`).
- `hi` CLDR categories from the shipped table: `['one','other']` — exactly those are used.
- Render matrix through `local_operator.i18n.runtime`: **30 cases, 0 failures** (counts 0/1/2/3/11/121; `failed` 0…121; select `online`/`retrying`/`unpaired`/`offline`). Changed-key samples: `'आपका ध्यान चाहिए · 3'`, `'आपका ध्यान चाहिए · 12'`, `'आपका इंतज़ार है'`, `'सभी को पढ़ा हुआ चिह्नित करें'`, `'सहेजा जा रहा है…'`, `'ऑफ़लाइन · फिर से कोशिश'`, `'minervaai के लिए कार्रवाइयाँ'`.
- Negative controls still fire in this run (checks are not no-ops): `argument 1 is 'counts', en's is 'count' (renamed or reordered)`; `argument 'count' changed kind 'number' -> 'simple'`; `plural category 'few' not in ['one', 'other']`; valueless branch `other:False`; `plural 'count' is missing locale categories ['one']`; `en has 2 argument(s), translation has 1`.
- Cross-key integrity: `approve_label` starts with the visible pill (`True`); `ended_hint` quotes `session.resume` verbatim (`True`); `empty_hint` quotes `list.new` verbatim (`True`).
- Typography on the change set: no ASCII quotes, no `...`, no ZWJ/ZWNJ; U+2026 kept where English has it; middle-dot separator parity `[]` mismatches.
- Widths (`rich.cells.cell_len` on rendered values): **0 of 85 rendered rows over budget** at sample counts. 3-digit/4-digit sweep across every count-bearing key — the only overflow anywhere is the residual below.

## 7. Residual (verified, non-gating)

`list.filter.needs_you` = `आपका ध्यान चाहिए · {count, number}`: **15/16** cells at a 1-digit count, **16/16** at 2 digits, **17/16** at 3 digits (19/16 at 4). Every other count-bearing key stays inside budget even at 4 digits (`since_seconds` 38/44, `agents_failed` 22/28, `tray.questions` 9/16, `filter.unread` 12/16, `tray.tasks` 11/16 at 3 digits and 15/16 at 4, `tray.agents` 11/20 and 15/20). The sidecar says the chip row scrolls sideways and asks only to keep it short; no measured clipping. Owner: rendered/design fit pass. Not a semantic defect and not a merge gate.

## 8. Limitations

- **R1-stream delta only.** The second safety-stream delta signoff on these keys is another auditor's; not claimed here.
- **Source-only.** No rendered app/browser evidence and no runtime lifecycle coverage in this round; widths are `cell_len` measurements on rendered strings, not laid-out frames.
- The target is untracked working-tree state; the receipt is bound to `eb9f326c…` and lapses if that file changes.
- No repository, target, ledger or other locale was modified by this audit; nothing was committed.

## 9. Receipt

```
audit      chrome25-hi-a2   locale hi   namespace relay.chrome   round 2 (delta)
en         f30049d1ec7bce1d0586d8d5d380368e64fb0ac841d91b9b04c1625d17688aac   (unchanged)
context    1c727d1c6c8ddff7c33cbd3abb1ed716e89ab84c864cdb9c319b5df2595fb69a   (unchanged)
before     a29563f5dd0a9f425b717639c28e6a8f4c97382d45b0928bacdf5e12d31285b3   (R1 target / pre-remediation snapshot)
after      eb9f326ca6689ed0d35311cd75ae8cb5683b515e8b638c938790759eeec96784   (current target)
changed 7 keys   carried forward 76 keys   total 83   base 535cea359825411741069cbffcee42e2927df9d3
findings: 7 fixed (2 also safety-stream), 1 rejected-correctly (verified)   open blocker/major: 0
verdict    TERMINAL PASS
```
