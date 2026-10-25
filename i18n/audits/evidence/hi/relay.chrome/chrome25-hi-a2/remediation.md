# Hindi remediation round — `hi/relay.chrome.json`

**Locale:** `hi` (Hindi) · **Namespace:** `relay.chrome` (83 keys) · **Round:** unified remediation 1
**Inputs read:** first audit `chrome25-hi-a1` (session `e86bee585698`) and second safety audit
`chrome25-safety-b1` (session `acd16c49df72`), hi sections only.
**File:** `~/local-operator/.worktrees/relay-chrome-i18n/local_operator/i18n/catalogues/hi/relay.chrome.json`

| Artefact | SHA-256 |
|---|---|
| candidate before remediation (snapshot kept at `$SCRATCH/hi-relay-chrome.candidate-pre-remediation.json`) | `a29563f5dd0a9f425b717639c28e6a8f4c97382d45b0928bacdf5e12d31285b3` |
| after remediation | `eb9f326ca6689ed0d35311cd75ae8cb5683b515e8b638c938790759eeec96784` |
| `en/relay.chrome.json` (frozen, unchanged) | `f30049d1ec7bce1d0586d8d5d380368e64fb0ac841d91b9b04c1625d17688aac` |
| `en/relay.chrome.context.json` (frozen, unchanged) | `1c727d1c6c8ddff7c33cbd3abb1ed716e89ab84c864cdb9c319b5df2595fb69a` |

**Delta scope:** 7 keys changed, 76 keys byte-identical to the candidate. No key added, removed or
reordered; no other locale, English catalogue, ledger, generator or gate touched; nothing committed.

Canonical diff (old → new):

| key | old | new |
|---|---|---|
| `drawer.status` (retrying) | `ऑफ़लाइन · फिर कोशिश` | `ऑफ़लाइन · फिर से कोशिश` |
| `list.filter.needs_you` | `आपके लिए · {count, number}` | `आपका ध्यान चाहिए · {count, number}` |
| `list.mark_all` | `सभी पढ़ा हुआ चिह्नित करें` | `सभी को पढ़ा हुआ चिह्नित करें` |
| `row.actions` | `{name} के लिए क्रियाएँ` | `{name} के लिए कार्रवाइयाँ` |
| `row.saving` | `सहेज रहे हैं…` | `सहेजा जा रहा है…` |
| `session.subtitle.waiting` | `आपका इंतज़ार` | `आपका इंतज़ार है` |
| `session.ended_hint` | `संदेश भेजने या “जारी रखें” से यह आपके होम फ़ोल्डर में फिर खुलेगा` | `संदेश भेजने पर या “जारी रखें” दबाने पर यह आपके होम फ़ोल्डर में फिर खुलेगा` |

---

## 1. Dispositions

### `chrome25-hi-a1-01` (major) + `chrome25-safety-b1 MIN-07` (minor) → **fixed**
`relay.chrome.list.filter.needs_you`

Both audits name the same concrete defect: a triage chip that must say **the operator has to act**
(an approval to give, a question to answer), while `आपके लिए` says "for you" — a curated-feed
reading. Adopted the safety audit's action-explicit rendering `आपका ध्यान चाहिए` ("needs your
attention"), not the first audit's `आपका इंतज़ार` ("waiting on you"), because:
- the brief's requirement for this key is *needs operator action*; "needs your attention" states
  the need, where "waiting on you" states a wait that a reader can take as passive observation;
- `आपका इंतज़ार` would also collide with the liveness line `session.subtitle.waiting` ("Waiting for
  you" — the agent is waiting on your approval), which is a different surface and a different claim.
- Width: 15 cells at a 1-digit count, 16 at 2 digits (budget 16). At a 3-digit count it is 17 —
  see §4 residual, retained deliberately.

### `chrome25-hi-a1-02` (minor) → **fixed, different wording**
`relay.chrome.session.subtitle.waiting`: `आपका इंतज़ार` → `आपका इंतज़ार है`

Accepted the concrete defect — a bare possessive noun phrase reads as "your wait" while its four
sibling labels are predicates — but not the proposed locative `आपके इंतज़ार में`, which is itself a
fragment: a bare `में` phrase needs `है` to become a statement (`आपके इंतज़ार में है`), so the
proposal does not actually complete the predicate it was raised to complete. `आपका इंतज़ार है` is
the standard complete form of the same phrase, keeps the established vocabulary and register, and
measures 10 of 24 cells.
The finding's second rationale (removing a duplicate with `needs_you`) is moot after the fix above,
because those two keys no longer share wording.

### `chrome25-hi-a1-03` (minor) → **rejected**
`relay.chrome.model.effort`: `तर्क-प्रयास` retained; `रीज़निंग प्रयास` rejected.

The claim is that `तर्क-प्रयास` is hyphen-ambiguous and not the corpus form. Verified against the
corpus (`local_operator/i18n/catalogues/hi/`, grep): `तर्क-प्रयास` is **shipped in three places**,
including the label and the help text for exactly this control —
`wire.settings.model_effort.label` = `डिफ़ॉल्ट तर्क-प्रयास`,
`wire.settings.model.description` = `…प्रोवाइडर, मॉडल और तर्क-प्रयास…`,
`wire.slash.effort` = `तर्क-प्रयास दिखाएँ या सेट करें (shift+tab बदलता है)`.
So the hyphenated compound is the corpus's established term for the reasoning-effort knob, used by
the settings row that configures it and the `/model` command that changes it; adopting
`रीज़निंग प्रयास` here would drift the relay model sheet away from both. The audit's own cited
corpus form `रीज़निंग प्रयास` does exist (`tui.notices.effort_default_changed`) — the corpus carries
both forms, and this namespace matches the control-side one. That pre-existing split is recorded
below as not addressed, not introduced here.

### `chrome25-hi-a1-04` (minor) → **fixed**
`relay.chrome.row.actions`: `क्रियाएँ` → `कार्रवाइयाँ` (21 of 40 cells).

Accepted: the corpus word for an "actions" menu is `कार्रवाई` (`tui.transcript.group.actions` =
`{label} कार्रवाई`/`… कार्रवाइयाँ`; `wire.errors.desktop.credential_operation_failed`). Correction to
the finding's supporting claim, recorded for accuracy: `क्रिया` is *not* unattested — `अगली क्रिया`
appears in `wire.settings.aida.description` — so the "only occurrence" premise is wrong; the
disposition rests on the corpus term for the actions-menu concept, not on absence.

### `chrome25-hi-a1-05` (minor) → **fixed**
`relay.chrome.session.ended_hint` → `संदेश भेजने पर या “जारी रखें” दबाने पर यह आपके होम फ़ोल्डर में फिर खुलेगा`

Accepted: the second coordinate was verbless (`भेजने … या “जारी रखें” से`), leaving an instrumental
`से` hanging off a quoted button label. The parallel conditional (`… भेजने पर … दबाने पर`) gives the
button a verb and matches the corpus's button wording `दबाएँ` already used in this file's
`list.empty_hint`. 51 cells, no `maxCells` (the sidecar says it wraps to two lines). Cross-key
integrity re-verified: the sentence still contains `“जारी रखें”` verbatim equal to
`session.resume`, and still states the home-folder destination.

### `chrome25-hi-a1-06` (nit) + `chrome25-safety-b1 MIN-08` (minor) → **fixed**
`relay.chrome.list.mark_all` → `सभी को पढ़ा हुआ चिह्नित करें` (18 of 20 cells).

Accepted from both audits: the dative `को` is the grammatical object marker and the earlier elision
was budget-driven optimism. Measured with `rich.cells.cell_len` — the measure the hi style guide
names — it fits with 2 cells spare, so the elision buys nothing.

### `chrome25-hi-a1-07` (nit) → **fixed, opposite direction to the proposal**
Busy-state voice: unified on the **passive**, not the active.
- `relay.chrome.row.saving`: `सहेज रहे हैं…` → `सहेजा जा रहा है…`
- `relay.chrome.list.marking`: unchanged `चिह्नित हो रहा है…`

The finding's direction rested on "unifying on the passive is impossible — `सहेजा जा रहा है…` is
16 cells against maxCells 12". That measurement does not reproduce: with `rich.cells.cell_len`
(the measure the hi style guide names, and the one the first audit itself used in §2.5) the string
is **11 cells of 12**. So the passive fits, and it is the form already shipped for the identical
English string in `relay.session.tags.saving` (`सहेजा जा रहा है…`) — a cross-file consistency the
finding itself flagged as unsettleable from this namespace, but which this namespace can meet by
matching it. Both busy labels are now passive, which resolves the underlying issue (two voices on
one screen) without breaking the shipped term or hugging the 12-cell tile worse than 11/12.

### `chrome25-hi-a1-08` (nit) → **fixed**
`relay.chrome.drawer.status` (retrying branch): `ऑफ़लाइन · फिर कोशिश` → `ऑफ़लाइन · फिर से कोशिश`
(16 of 32 cells).

Accepted: corpus counts `फिर से कोशिश` 14 occurrences vs `फिर कोशिश` 8, and the bare `फिर कोशिश`
forms are imperatives (`फिर कोशिश करें`). Only the `retrying` branch text changed; the `online` /
`retrying` / `other` selectors and the `{state, select, …}` structure are byte-identical.

### Not a finding, re-checked → **no change**
- `session.degraded` `उत्तर नहीं दे रहा — भेजे संदेश रुके रहेंगे`: the safety audit passed this for
  hi ("keeps the wait/held sense; no delivery claim"). Re-read against the source "Not answering —
  sends will wait" and the sidecar ("messages typed now are kept and delivered when it answers
  again"): `रुके रहेंगे` states the hold and promises no delivery, so the brief's "no new
  queued/delivered guarantee" constraint holds. Left unchanged rather than churned.
- Identical `one`/`other` branches (`since_seconds`, `since_minutes`, `tray.questions`,
  `tray.agents_failed`): correct Hindi, both branches present and both render `#` — verified again
  after the edit.

---

## 2. Validation after the change (actual output)

Key/ICU/plural/branch semantics, via the shipped runtime `local_operator.i18n`:

```
keys: 83 83 | order identical: True | set equal: True
arg/selector mismatches: []
plural categories present: ['other']            # literal selectors found; both one+other present per key
all plurals keep other: True
value-less branches: none
approve_label starts with approve pill: True
ended_hint contains “जारी रखें” (resume label): True
empty_hint contains “नया” (new label): True
ended_hint keeps home-folder destination: True
```

Rendered samples of every changed key (runtime `render_message`, locale `hi`):

```
list.filter.needs_you   count 1   -> 'आपका ध्यान चाहिए · 1'
list.filter.needs_you   count 12  -> 'आपका ध्यान चाहिए · 12'
session.subtitle.waiting          -> 'आपका इंतज़ार है'
row.actions             name=minervaai -> 'minervaai के लिए कार्रवाइयाँ'
list.mark_all                     -> 'सभी को पढ़ा हुआ चिह्नित करें'
row.saving                        -> 'सहेजा जा रहा है…'
drawer.status online              -> 'कनेक्टेड · यह फ़ोन'
drawer.status retrying            -> 'ऑफ़लाइन · फिर से कोशिश'
drawer.status other               -> 'कनेक्ट हो रहा है…'
session.ended_hint                -> 'संदेश भेजने पर या “जारी रखें” दबाने पर यह आपके होम फ़ोल्डर में फिर खुलेगा'
```

Repo checker (`cd ~/local-operator/.worktrees/relay-chrome-i18n && PYTHONPATH=. ~/local-operator/.venv/bin/python scripts/i18n/check.py`, exit 0):

```
check.py: ok — 182 file(s) scanned, 2253 literal(s) at or below the baseline, catalogues consistent; 34 advisory finding(s) (report-only until activated)
```

The 34 `advisory:` lines above it are the repo-wide literal ratchet on unrelated Python files
(`local_operator/update.py`, `tools/builtin.py`, …), report-only; **no line mentions `hi/` or
`relay.chrome`**. No incidental wave-in-progress failure occurred in this run; an earlier run in
this window did fail once on a peer's in-flight `fr/relay.chrome.json` (argument-kind error), which
is not this locale and was not touched.

## 3. Width after the change (`rich.cells.cell_len` on rendered values)

Over budget: **none** except the flagged 3-digit case in §4. Tightest: `row.saving` 11/12,
`list.mark_all` 18/20, `list.starting` 10/12 (unchanged), `session.resuming` 8/12.
Nothing in this delta grew toward a clip; the two changed headers moved 16→18 (`mark_all`, budget 20)
and 12→16 (`drawer.status` retrying branch, budget 32).

## 4. Residual / flag for rendered fit (the only actionable item)

`relay.chrome.list.filter.needs_you` — `आपका ध्यान चाहिए · {count, number}`:
15 cells at a 1-digit count, **16/16** at 2 digits, **17/16** at a 3-digit count (1 cell over).
Not abbreviated, deliberately: the shorter options drop the action-required semantics the brief
requires for this key (`आपके लिए` — the rejected original; `आपका इंतज़ार` — a wait, not a need).
The sidecar says the chip row scrolls sideways and only asks to "keep it short", so there is no
measured clipping; this is flagged for the rendered/design pass, not claimed as verified fit.
Counts of 100+ conversations waiting on one operator are the only case affected.

## 5. Not addressed (out of scope for this namespace; recorded, not deferred to a ticket)

- The Hindi corpus carries two renderings for the same English term "reasoning effort":
  `तर्क-प्रयास` (`wire.settings.model_effort.label`, `wire.settings.model.description`,
  `wire.slash.effort`) and `रीज़निंग प्रयास` / `रीज़निंग-प्रयास स्तर` (`tui.notices.*`,
  `wire.errors.desktop.effort_levels_missing`). This wave keeps the control-side term; harmonising
  the notices is a cross-namespace call for a later wave, not a `relay.chrome` defect.

## 6. Provenance of this round

- Role: `translator` (team `lopdev`), session `6f3832cb71c0`.
- Model: `radient/auto` as this session header shows (`provider: radient`, `model: auto`). `auto`
  is a provider alias; the resolved upstream model id is not recorded in the session store, so no
  specific model is claimed. **No fallback was invoked or observed** in this round.
- Harness: `lop v0.90.2`; worktree `~/local-operator/.worktrees/relay-chrome-i18n` @
  `535cea359825411741069cbffcee42e2927df9d3`.
- Nothing committed; the target remains an untracked working-tree file, so the SHA-256 above is its
  identity.
