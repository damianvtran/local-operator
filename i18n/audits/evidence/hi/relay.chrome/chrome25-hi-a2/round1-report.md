# Independent translation audit — `hi/relay.chrome.json`

**Audit ID:** chrome25-hi-a1 · **Locale:** `hi` (Hindi) · **Namespace:** `relay.chrome` · **Keys read:** 83 / 83
**Verdict:** **FINDINGS — not clean** (1 major · 4 minor · 3 nit). Mechanical layer PASS.

---

## 1. Provenance and scope

| Item | Value |
|---|---|
| Auditor role | `translation-auditor` (subagent), team `lopdev` |
| Auditor session | `e86bee585698` — delegating job `7c6b736a1841` ("Audit chrome hi"), coordinator session `44f93fd45b09` |
| Provider / model | `radient/auto` — read from this session's own transcript metadata (`provider: radient`, `model: auto`, `model_id: auto`). `auto` is a provider alias; the resolved upstream id is not recorded in the session store, so **no specific model is claimed here**. |
| Harness | `lop v0.90.2`, generation `20261025T122248Z-0.90.2` (`readlink ~/.local/share/lop/current`) |
| Date (UTC) | 2026-10-25T18:01Z |
| Worktree (read-only) | `~/local-operator/.worktrees/relay-chrome-i18n` @ `535cea359825411741069cbffcee42e2927df9d3` |
| Target file state | `local_operator/i18n/catalogues/hi/relay.chrome.json` — **untracked in that worktree**; the SHA-256 below is its identity |
| Independence | No translator report was read, and no other locale's `relay.chrome.json` was opened. Consistency was judged against the English source, `i18n/glossary.md`, `i18n/style-guides/hi.md`, and the **pre-existing** Hindi catalogues (28 files, 1667 values). Nothing in the repo was modified. |

Hashes (SHA-256, computed here):

| Artefact | SHA256 |
|---|---|
| `en/relay.chrome.json` | `f30049d1ec7bce1d0586d8d5d380368e64fb0ac841d91b9b04c1625d17688aac` — matches the brief |
| `en/relay.chrome.context.json` | `1c727d1c6c8ddff7c33cbd3abb1ed716e89ab84c864cdb9c319b5df2595fb69a` — matches the brief |
| `hi/relay.chrome.json` (target) | `a29563f5dd0a9f425b717639c28e6a8f4c97382d45b0928bacdf5e12d31285b3` |

Read in full: English source (4914 B), context sidecar (34149 B, all 83 `role`/`tone`/`maxCells`/`note` entries), target (7235 B). Every one of the 83 keys was read against its sidecar note, not merely compared mechanically.

---

## 2. Executed checks and their actual output

### 2.1 Repo catalogue validation (the repo's own checker)

```
$ env -i HOME="$ISO" … ../../.venv/bin/python scripts/i18n/check.py
check.py: ok — 182 file(s) scanned, 2253 literal(s) at or below the baseline,
catalogues consistent; 34 advisory finding(s) (report-only until activated)
EXIT=0
```

`catalogues consistent` is the meaningful line: key parity, ICU argument tuples **by name and kind**, plural-category legality and context-sidecar shape all pass for this file. The 34 `advisory:` lines are the repo-wide literal ratchet on unrelated Python files, report-only.

### 2.2 Key and argument parity

- `en` 83 · `en.context` 83 · `hi` 83 — key sets identical (`True`).
- Per-key ICU comparison across all 83: **0 problems**.
- 15 keys carry arguments; every tuple matches `en` in name, **kind** and order:

| key | en args | hi args | plural branches |
|---|---|---|---|
| `drawer.status` | `state:select` | `state:select` | — (online/retrying/other) |
| `link.since_seconds` | `count:plural` | `count:plural` | `one`,`other` ↔ `one`,`other` |
| `link.since_minutes` | `count:plural` | `count:plural` | `one`,`other` ↔ `one`,`other` |
| `list.filter.needs_you` | `count:number` | `count:number` | — |
| `list.filter.unread` | `count:number` | `count:number` | — |
| `row.actions` | `name:simple` | `name:simple` | — |
| `session.subtitle.working` | `elapsed:simple` | `elapsed:simple` | — |
| `session.tray.questions` | `count:plural` | `count:plural` | `one`,`other` ↔ `one`,`other` |
| `session.tray.tasks` | `done:number, total:number` | identical | — |
| `session.tray.agents` | `running:number, total:number` | identical | — |
| `session.tray.agents_failed` | `failed:plural, running:number, total:number` | identical | `one`,`other` ↔ `one`,`other` |
| `composer.folder_label` | `name:simple` | `name:simple` | — |
| `composer.model_label` | `label:simple` | `label:simple` | — |
| `composer.spend_label` | `label:simple` | `label:simple` | — |

- Hindi CLDR categories from the **shipped** table (`local_operator/i18n/data/plural_rules.json`): `['one','other']`. All four plural messages use exactly those; no `few`/`many`/`zero` needed or present.

### 2.3 Rendered ICU branches (shipped runtime `local_operator.i18n.runtime`)

30 render cases, **0 failures**. Actual output:

```
drawer.status online      -> 'कनेक्टेड · यह फ़ोन'
drawer.status retrying    -> 'ऑफ़लाइन · फिर कोशिश'
drawer.status unpaired    -> 'कनेक्ट हो रहा है…'      (other branch)
drawer.status offline     -> 'कनेक्ट हो रहा है…'      (other branch)
link.since_seconds  0/1/2/3/11/121 -> 'फिर से कनेक्ट हो रहा है · {0,1,2,3,11,121} सेकंड पहले सिंक हुआ'
link.since_minutes  0/1/2/3/11/121 -> 'फिर से कनेक्ट हो रहा है · {…} मिनट पहले सिंक हुआ'
tray.questions      0/1/2/3/11/121 -> '0 सवाल' / '1 सवाल' / '2 सवाल' / '3 सवाल' / '11 सवाल' / '121 सवाल'
tray.tasks                         -> 'कार्य 3/8'
tray.agents                        -> 'एजेंट 1/8'
tray.agents_failed  failed=0/1/2/3/11/121 -> 'एजेंट 2/3 · {0,1,2,3,11,121} विफल'
```

`0` rendering through the `one` branch (`0 सवाल`) is correct for Hindi per the style guide ("`one` covers 0 and 1").

### 2.4 Negative controls (the checks are not no-ops)

Run on probe strings only — no shared file was mutated:

| control | actual result |
|---|---|
| renamed placeholder (`{count}`→`{counts}`) | `argument 1 is 'counts', en's is 'count' (renamed or reordered)` |
| kind change (`number`→`simple`) | `argument 'count' changed kind 'number' -> 'simple'` |
| non-CLDR category (`few`) | `plural category 'few' not in ['one', 'other']` |
| branch that drops the number | `one:True, other:False` (`renders_value`) |
| lossy `number`→`plural` upgrade | `plural 'count' is missing locale categories ['one']` |
| dropped argument | `en has 2 argument(s), translation has 1` |

Each control fired with the expected diagnostic, so §2.2's clean result carries weight.

### 2.5 Width (rendered strings, `rich.cells.cell_len` — the measure the hi guide names)

85 rendered rows (83 keys; the `select` key counted per branch). **0 over `maxCells`.** Hindi is shorter than English in 68 rows, equal in 4, longer in 13 — largest growth **+4 cells** (`link.reconnecting` 13→17 of 32; `list.marking` 8→12 of 20). Nearest to budget: `list.starting` (10 of 12, 2 spare); longest budgeted string: `link.since_seconds` at 36 of 44 with a three-digit count.

*Note on method:* measuring the raw ICU **source** instead would falsely flag 9 keys (`since_seconds` "100 cells of 44") because the syntax itself occupies cells. The table above measures what the label shows.

### 2.6 Typography, character and cross-key integrity

- ZWJ/ZWNJ: **0**. ASCII quotes: **0**. ASCII `...`: **0** — U+2026 appears in exactly the 9 keys where English has it (key sets identical).
- ` · ` separator: present in every key where English has it, and absent where it does not (0 mismatches).
- Devanagari digits: 0 (Latin digits per §2.8). `कृपया`: 0. `!`: 0. Trailing period/danda: 0. Double spaces / padding: 0.
- The only Latin run in the file is `Local Operator` (product name, glossary §1 — correctly untranslated).
- Cross-key label integrity (context-mandated): `session.tray.approve_label` **starts with** `session.tray.approve` (`True`); `session.ended_hint` contains `"“जारी रखें”"` verbatim equal to `session.resume` (`True`); `list.empty_hint` contains `"“नया”"` verbatim equal to `list.new` (`True`).

### 2.7 Register and glossary survey against the pre-existing Hindi corpus

Attested and consistent (spot-checked in 28 pre-existing hi files): `सत्र` (glossary), `एजेंट`, `प्रोजेक्ट`, `उपयोग`, `लाइब्रेरी` (`wire.errors.desktop.library_*`), `दिखावट` (`wire.settings.appearance.title`), `सेटिंग` (`tui.palette.kind_setting`), `टैग`, `संदर्भ` (context), `ख़र्च`, `चालू`/`बंद` (`tui.tags.on`), `सवाल` (`relay.asks.*`), `अपठित` (`wire.slash.notifications`), `स्वीकृत करें` (`relay.asks.pending.approve`), `जारी रखें` (`tui.welcome.tip.resume`), `इंतज़ार` (`relay.ask.dismiss.moot_note`).

Divergences found → findings 03, 04, 07, 08 below.

---

## 3. Per-area verdicts

| Area | Keys | Verdict |
|---|---|---|
| `relay.chrome.drawer.*` | 12 | **Pass with 1 nit** (08: retrying branch clipped). Destinations, landmark name and the `select` are accurate; `दिखावट`/`सेटिंग`/`लाइब्रेरी` match the corpus. |
| `relay.chrome.link.*` | 3 | **Clean.** Identical plural branches are correct Hindi and attested (`tui.review.age_minutes`); `#` order "N सेकंड पहले" is idiomatic. |
| `relay.chrome.list.*` | 21 | **Findings** — 01 major (`filter.needs_you`), 06 nit (`mark_all`). The other 19 keys read as native copy; search/filter/empty states are accurate. |
| `relay.chrome.row.*` | 10 | **Findings** — 04 minor (`actions`), and 07's `saving` half. Pin/unpin/more/tags/open are clean and inside tight budgets. |
| `relay.chrome.appearance.*` | 5 | **Clean.** `चालू`/`बंद` match `tui.tags.on`; `group_by_tag` renders "show conversations grouped by tag" — a natural Hindi strategy, 26 of 36 cells. |
| `relay.chrome.session.*` | 25 | **Findings** — 02 minor (`subtitle.waiting`), 05 minor (`ended_hint`), 07's `marking` relation. Approve/ended/resume semantics verified correct (§4). |
| `relay.chrome.composer.*` + `spend.title` | 5 | **Clean.** The settled-loan policy holds (`फ़ोल्डर` for folder), `संदर्भ` for context, placeholders verbatim. |
| `relay.chrome.model.effort` | 1 | **Finding** — 03 minor. |
| `relay.chrome.snackbar.dismiss` | 1 | **Clean.** |

---

## 4. High-visibility safety copy — checked explicitly

- **Approve.** `session.tray.approve` = `स्वीकृत करें`, byte-identical to the corpus's approval verb (`relay.asks.pending.approve`) and paired with the corpus's reject (`अस्वीकार करें`) — the approve/deny polarity cannot be mistaken. `approve_label` = `स्वीकृत करें — एक अनुरोध आपका इंतज़ार कर रहा है` begins with the visible pill string, as the sidecar requires, so voice control can say what is on screen. **Correct.**
- **Ended.** `subtitle.ended` = `समाप्त`; `session.ended` = `सत्र समाप्त · इतिहास सुरक्षित` — "session ended · history preserved" keeps the English promise (history retained, resumable) and makes no lost-data claim. **Correct.**
- **Resume.** `session.resume` = `जारी रखें`, the corpus's own `/resume` wording (`tui.welcome.tip.resume`). `ended_hint` quotes that label verbatim and keeps the safety-relevant destination — `आपके होम फ़ोल्डर में` ("in your home folder"), not the project folder. Its **grammar** is finding 05; the **semantics** are right. `resuming` = `खुल रहा है…` describes the reopen and stays within 12 cells. **Correct in substance.**
- **Not answering / degraded.** `उत्तर नहीं दे रहा — भेजे संदेश रुके रहेंगे` tells the user sends are held, not dropped, matching the sidecar ("messages typed now are kept and delivered when it answers again"). **Correct.**
- A separate second independent pass covers safety copy; this section is corroborative, not a substitute.

---

## 5. Findings

### chrome25-hi-a1-01 · **major** · `relay.chrome.list.filter.needs_you`
- Current: `आपके लिए · {count, number}`
- Proposed: `आपका इंतज़ार · {count, number}`
- Reason: the English is a triage filter — the sidecar says "conversations blocked on the operator: an approval to give or a question to answer" — while `आपके लिए` means "for you", which reads as a curated-for-you feed and drops the "needs/blocked" semantics that makes the chip actionable. `आपका इंतज़ार` ("waiting on you") uses vocabulary already in the corpus (`relay.ask.dismiss.moot_note`, `tray.approve_label`) and keeps the chip within budget: 12 cells of 16 with a 1-digit count.
- Rejected alternative: `आपके लिए ज़रूरी` = 14 of 16 — semantically closer but leaves almost no room for a 3-digit count.

### chrome25-hi-a1-02 · **minor** · `relay.chrome.session.subtitle.waiting`
- Current: `आपका इंतज़ार`
- Proposed: `आपके इंतज़ार में`
- Reason: a bare possessive noun phrase most literally means "your wait"; its four siblings in the same slot are predicates (`कार्यरत`, `समाप्त`, `आइडल`, `उत्तर नहीं दे रहा`). The locative makes "awaiting you" explicit and keeps the parallel shape. 10 cells of 24.
- Also removes an exact duplicate: with finding 01 adopted, two different surfaces would otherwise both read `आपका इंतज़ार`.

### chrome25-hi-a1-03 · **minor** · `relay.chrome.model.effort`
- Current: `तर्क-प्रयास`
- Proposed: `रीज़निंग प्रयास`
- Reason: Hindi hyphens join coordinate or echo pairs (`आना-जाना`), so `तर्क-प्रयास` can be read as "logic-and-effort". The corpus already renders this exact English phrase as `रीज़निंग प्रयास` (`tui.notices.effort_default_changed`) and `रीज़निंग-प्रयास स्तर` (`wire.errors.desktop.effort_levels_missing`); `तर्क` alone is the corpus's reasoning word (`wire.settings.display_reasoning`). 9 cells of 24.
- If the wave prefers the native word, `तर्क प्रयास` (unhyphenated) is acceptable; the hyphen is the defect, the term choice is a policy call.

### chrome25-hi-a1-04 · **minor** · `relay.chrome.row.actions`
- Current: `{name} के लिए क्रियाएँ`
- Proposed: `{name} के लिए कार्रवाइयाँ`
- Reason: the corpus's word for "actions" is `कार्रवाई` / `कार्रवाइयाँ` (`tui.transcript.group.actions`, `wire.errors.desktop.credential_operation_failed`); `क्रियाएँ` is the only occurrence in 1667 Hindi values, and `क्रिया` also means "verb" — a poor fit for an action-sheet's accessible name. 17 cells of 40.

### chrome25-hi-a1-05 · **minor** · `relay.chrome.session.ended_hint`
- Current: `संदेश भेजने या “जारी रखें” से यह आपके होम फ़ोल्डर में फिर खुलेगा`
- Proposed: `संदेश भेजने पर या “जारी रखें” दबाने पर यह आपके होम फ़ोल्डर में फिर खुलेगा`
- Reason: the second coordinate has no verb, so `से` hangs off a quoted button label; the idiomatic conditional is `…भेजने पर … दबाने पर`. This is the sentence that tells a user how to bring an ended session back, so it should not read as translated. 43→51 cells; the key has no `maxCells` and wraps.

### chrome25-hi-a1-06 · **nit** · `relay.chrome.list.mark_all`
- Current: `सभी पढ़ा हुआ चिह्नित करें`
- Proposed: `सभी को पढ़ा हुआ चिह्नित करें`
- Reason: the definite object of a transitive imperative normally takes `को`; the elision is tolerated in telegraphic UI copy but reads loose next to `row.mark_read`. 18 cells of 20 — still fits.

### chrome25-hi-a1-07 · **nit** · busy-state voice: `relay.chrome.list.marking` + `relay.chrome.row.saving`
- Current: `चिह्नित हो रहा है…` (impersonal passive) vs `सहेज रहे हैं…` (respectful active)
- Proposed: `चिह्नित कर रहे हैं…` (19 of 20) and keep `सहेज रहे हैं…` (9 of 12)
- Reason: two busy labels on one screen speak in two voices. Unifying on the passive is impossible — `सहेजा जा रहा है…` is 16 cells against `row.saving`'s `maxCells` 12 — so the active respectful voice is the coherent direction. Noted for the wave: `relay.session.tags.saving` uses the passive form for the same English "Saving…", a cross-file call this namespace cannot settle alone.

### chrome25-hi-a1-08 · **nit** · `relay.chrome.drawer.status` (`retrying` branch)
- Current: `ऑफ़लाइन · फिर कोशिश`
- Proposed: `ऑफ़लाइन · फिर से कोशिश`
- Reason: the corpus's majority form is `फिर से कोशिश` (`relay.asks.sheet.retry`, `tui.notices.*`); bare `फिर कोशिश` is attested only with its imperative (`फिर कोशिश करें`), so the status branch reads as a clipped noun fragment. 16 cells of 32.

---

## 6. Explicitly checked — no finding (and why)

- **Identical `one`/`other` branch text** in `since_seconds`, `since_minutes`, `tray.questions`, `tray.agents_failed`: correct, not a defect. `सेकंड`, `मिनट`, `सवाल` and `विफल` do not inflect after a numeral in the direct case, and the same pattern is already shipped in `tui.review.age_minutes` and `relay.asks.dock.waiting`. Both branches still render `#` (verified: no value-less branch).
- **`# सवाल पहले` word order:** Hindi places the count before the unit and the verb last; the rendering `3 सेकंड पहले सिंक हुआ` is idiomatic, not a calque.
- **`drawer.conversations` = `बातचीत`** as both screen title and back-target (`session.back` = `बातचीत पर वापस`): one word, one meaning, and it matches how the corpus names a conversation (`relay.asks.card.gone`).
- **`subtitle.idle` = `आइडल`:** both `आइडल` and `निष्क्रिय` exist in the corpus for this concept (often in the same settings key); the loan is the better fit for a 24-cell status slot and the style guide allows a living loan.
- **`session.ended` `इतिहास सुरक्षित`:** "history preserved" is not an overclaim — the sidecar says history is kept and the session can be resumed.
- **`समाप्त` shared by session-ended and question-expired:** the two surfaces never co-occur, and Hindi has one natural word; no drift to fix.
- **`row.saving` = `सहेज रहे हैं…`:** deviates from `relay.session.tags.saving`'s passive, but the passive cannot fit `maxCells` 12 — budget-driven, recorded under 07 rather than as a separate defect.

---

## 7. Limitations

- **No rendered app evidence in this pass.** Widths are `cell_len` measurements on rendered strings, not laid-out frames; a screenshot/visual regression pass follows separately. No claim of rendered-frame or design/UX coverage is made here.
- Consistency was checked against English + the pre-existing Hindi corpus + glossary + style guide only. No translator report and no other locale's chrome catalogue was read, so independence holds; it also means cross-locale harmonisation is out of scope.
- Single file, single locale: `relay.chrome` has 83 keys and no sub-namespaces; "per-namespace" verdicts are therefore the per-area table in §3.
- The target is an **untracked** working-tree file — if it changes, the hash changes with it and this audit no longer applies.

---

## 8. Overall

**Mechanical layer: PASS** — parity, ICU argument names/kinds, CLDR categories, rendered branches, typography tokens and every width budget hold, and the controls show those checks bite.

**Semantic layer: FINDINGS** — 1 major (a triage filter that no longer says it needs you), 4 minor (status ellipsis, hyphens in a compound term, "actions" drift, a verbless coordinate in the resume hint), 3 nit. The translation is otherwise strong: natural word order, correct address form, and the safety-critical approve/ended/resume wording is right.

**Recommendation:** fix 01 before this namespace is called clean; 02–05 are cheap and fit budget; 06–08 are optional polish. Re-audit scope for round 2 should be the remediation diff (`a29563f5…`..new head) plus the direct neighbours named in each finding.
