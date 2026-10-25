# Translation audit — `ur/relay.chrome.json` (`chrome25-ur-a1`)

**Verdict: PASS on blockers and majors — 0 blocker, 0 major, 6 minor, 3 nit.**
Full namespace read: **83 of 83 keys**, source and target compared value by value.

| Item | Value |
|---|---|
| Locale / namespace | `ur` / `relay.chrome` |
| Audit id | `chrome25-ur-a1` |
| Repository revision | `535cea359825411741069cbffcee42e2927df9d3` (`~/local-operator/.worktrees/relay-chrome-i18n`) |
| English source | `local_operator/i18n/catalogues/en/relay.chrome.json` — `f30049d1ec7bce1d0586d8d5d380368e64fb0ac841d91b9b04c1625d17688aac` |
| Context sidecar | `local_operator/i18n/catalogues/en/relay.chrome.context.json` — `1c727d1c6c8ddff7c33cbd3abb1ed716e89ab84c864cdb9c319b5df2595fb69a` |
| Target | `local_operator/i18n/catalogues/ur/relay.chrome.json` — `6059005113786f1d74fc848a9f254fadd5fba259be11aecfcbf59ebe2cd24d28` |
| Key count | 83 (en 83 / context 83 / ur 83; key order identical) |
| Auditor | agent `translation-auditor` (team `lopdev`), session `439789cb67e3` |
| Model (as reported) | `radient/auto` — the session header's model string; the provider/model behind the auto route is not exposed to the session, so no underlying model id is claimed |
| Independence | separate session from the translators; translator reports/evidence were not opened; target bytes read directly |
| Fallback | none observed in this session (no fallback or session-incident notice present) |

Read-only audit: nothing under the worktree was created, edited, staged or committed. All helper
scripts and logs live in this session's scratchpad bundle.

---

## 1. What was checked

1. **Byte identity of inputs** — SHA-256 of English, context sidecar and target (table above); target
   is UTF-8, no BOM, trailing newline present, 6,037 bytes.
2. **Key parity** — 83/83, same order, no missing/extra keys.
3. **ICU structure** — every argument name, every argument *kind* (`number`, `plural`, `select`,
   `simple`) and every selector tuple compared against English by walking the repository's own
   parser (`local_operator.i18n.runtime.parse_message`): **0 mismatches across 83 keys**.
4. **CLDR categories** — the repo-generated table gives Urdu `one`/`other` (`0`→`other`, `1`→`one`,
   `2..`→`other`). Every plural message carries both categories and no others; every branch renders
   its own count (`#` present in both branches of `tray.questions`, `agents_failed`,
   `since_seconds`, `since_minutes`).
5. **Renders through the shipped formatter** (`messages.render(locale="ur")`) for every ICU message,
   including all `select` branches and plural counts 0/1/2/5/11 — see `audit.log`. Unknown `select`
   tokens fall back to `other` (`منسلک ہو رہا ہے…`), which is the correct behaviour.
6. **Negative controls** (in-memory only, files untouched): renamed placeholder, dropped placeholder,
   kind change `number`→`simple`, dropped plural branch, non-CLDR plural category `few`. The repo's
   gates catch placeholders and plural categories; a dropped `one` branch and a renamed `select`
   branch are *not* flagged by `scripts/i18n/check.py` (a checker gap, not a defect in this file —
   this file has `one`/`other` and the same three `select` keys as English).
7. **Repository catalogue check** — `python scripts/i18n/check.py` → exit 0,
   `catalogues consistent` (182 files scanned; 34 pre-existing advisory ratchet lines only).
8. **Urdu script hygiene** — the target uses only Urdu letter forms (`ک` U+06A9, `ی` U+06CC,
   `ہ` U+06C1, `ے` U+06D2, `ں` U+06BA, `ھ` U+06BE); no Arabic `ي/ك/ه/ة`, no Devanagari danda, no
   ASCII `!`, no Urdu-Indic digits (Latin digits per the style guide), no bidi control characters
   (LRM/RLM/LRI…PDI) — control characters are the render layer's job and are correctly absent.
9. **Punctuation inventory** — `·` ×12, `—` ×2, `…` ×9, `“ ”` ×4, Urdu comma `،` ×1, Latin `:` ×3;
   every English `U+2026`/`·`/`—` has a counterpart, no value has a trailing full stop, no value has
   padding or double spaces, and no value is left identical to English.
10. **Terminology against the existing Urdu corpus** (`ur/relay.*`, `ur/wire.*`, `ur/tui.*`) and
    `i18n/glossary.md` + `i18n/style-guides/ur.md` — details in §4.
11. **Width** — every value rendered with worst-case parameters and measured against the sidecar's
    `maxCells` (Urdu ≈ 1 cell/char per the style guide; combining marks 0). One string exceeds its
    budget (F9). This is a signal for the rendered pass, not a semantic defect.

Not performed here (and not claimed): rendered app/device evidence, and the web `IntlMessageFormat`
path — that package is not installed in this checkout and no scratch dependency install is permitted.
The web runtime resolves `ur` as RTL (`mobile/web/src/i18n/runtime.ts:18-21`) and delegates plural
selection to `Intl.PluralRules` for `ur`, i.e. the same `one`/`other`, `0`→`other` categories
verified against the repo table.

---

## 2. Findings

### F1 — minor — `relay.chrome.row.saving`

- Current: `محفوظ…`
- Proposed: `محفوظ ہو رہا…` (13 cells vs `maxCells` 12 — needs a rendered check; if the row must stay
  ≤ 12, `پن محفوظ…` is the pin-specific fallback)
- Reason: bare `محفوظ` is the completed-state participle ("saved"), but this label replaces the pin
  action *while the computer has not confirmed* the change; the locale's own busy state is
  `محفوظ ہو رہا ہے…` (`relay.session.tags.saving`). As written it reads "saved…" before the save happened.

### F2 — minor — `relay.chrome.list.starting`

- Current: `شروع…`
- Proposed: `شروع ہو رہا…` (12 cells = `maxCells` 12)
- Reason: reads as "start/started", not "starting". Every other busy state in this file is a
  progressive (`نشان زد ہو رہا ہے…`, `منسلک ہو رہا ہے…`, `لوڈ ہو رہی ہے…`); this one is the outlier.

### F3 — minor — `relay.chrome.drawer.status` (`retrying` branch)

- Current: `آف لائن · دوبارہ کوشش`
- Proposed: `آف لائن · دوبارہ کوشش ہو رہی ہے` (31 cells ≤ `maxCells` 32)
- Reason: the `online` and `other` branches are states/clauses; the bare noun reads like a retry
  control ("Retry"), and the locale already uses `دوبارہ کوشش کریں` as the *button*. The footer says
  what the app is doing, so it needs the ongoing form.

### F4 — minor — `relay.chrome.session.degraded`

- Current: `جواب نہیں دے رہا — پیغام روکے ہیں`
- Proposed: `جواب نہیں دے رہا — پیغام قطار میں ہے` (36 cells = `maxCells` 36)
- Alternative minimal fix: `پیغامات روکے ہیں` (35 cells)
- Reason: singular `پیغام` with plural `روکے ہیں` is an agreement slip, and "روکے" reads as *blocked*
  rather than *queued until it answers* (English: "sends will wait"). `قطار میں` is the locale's
  established queued phrasing (`wire.settings.asks.description`). The waiting/delivery semantics is
  safety-relevant copy and should not read as a permanent block.

### F5 — minor — `relay.chrome.appearance.group_by_tag`

- Current: `گفتگوئیں ٹیگ سے گروپ کریں`
- Proposed: `گفتگوئیں ٹیگ کے حساب سے گروپ کریں` (33 cells ≤ `maxCells` 36)
- Reason: `ٹیگ سے گروپ کریں` is a Hindi-ised calque; the Urdu instrumental is `کے حساب سے` (or
  `کے مطابق`), and `گروپ کریں` itself is fine.

### F6 — minor — `relay.chrome.drawer.library` (cross-namespace term drift)

- Current: `لائبریری`
- Other rendering of the same term: `Library`, untranslated, in `wire.settings.librarian.description`
  (`Librarian عالمی Library منظم رکھتی ہے…`)
- Proposed: record one decision. Recommendation: keep `لائبریری` for the drawer — a navigation label
  needs a native word — and either align the settings prose or add a glossary row ("Library" is not
  in glossary §1, so the keep-English rule does not currently cover it).
- Reason: the same English surface term maps to two different local terms; whichever way it is
  decided, one of the two files must change.

### F7 — nit — `relay.chrome.session.resuming`

- Current: `کھل رہا ہے…`
- Proposed: `جاری ہو رہا…` (12 cells = `maxCells` 12)
- Reason: the button beside it says `جاری رکھیں`; the busy state switches verb to "opening".
  (`دوبارہ کھولیں` is attested for reopening a session in `relay.asks`, so this is a nit, not an error.)

### F8 — nit — `relay.chrome.link.since_seconds`, `relay.chrome.link.since_minutes`

- Current: `دوبارہ منسلک ہو رہا ہے · # سیکنڈ پہلے` / `… · # منٹ پہلے`
- Proposed: `دوبارہ منسلک… · آخری اپ ڈیٹ # سیکنڈ پہلے` (41 cells ≤ 44) and `… · آخری اپ ڈیٹ # منٹ پہلے` (39 ≤ 44)
- Reason: English pins the time to "last synced"; without a referent, `# سیکنڈ پہلے` can be read as how
  long the link has been down. The proposal restores the referent inside the budget by using the
  short reconnect form; if the full `دوبارہ منسلک ہو رہا ہے` is preferred, the sync phrase does not fit
  in 44 cells and the current compression is defensible (hence nit).

### F9 — nit — `relay.chrome.list.filter.needs_you` (width only)

- Current: `آپ کا انتظار · {count, number}`
- Measured: 16 cells at a one-digit count (at the `maxCells` 16 limit), **17 cells at two digits** —
  the only over-budget string in the namespace.
- Proposed: no wording change; flag for the rendered pass. The sidecar already says the chip row
  scrolls sideways, so a one-cell overrun at two-digit counts is a layout check, not a copy defect.
- Reason: `آپ کا انتظار` is the natural phrasing and matches `session.tray.waiting`; this is a layout
  risk, not a translation error.

---

## 2b. Checked and accepted (not findings)

- `گفتوگو` terms: `گفتگو` / `گفتگوئیں` / `گفتگوؤں` — correct Urdu forms, matching `wire.settings`
  and `wire.slash` usage (10 + 7 occurrences in the existing corpus).
- `ناپڑھی` for "Unread" — matches the established `wire.slash.notifications` rendering
  (`ناپڑھی تکمیل کی اطلاعات`); the "illiterate" homograph is not live in a counted filter chip.
- `کام جاری` for "Working" — identical to the existing `tui.status.band.working`.
- `استدلال کی شدت` for "Reasoning effort" — identical to `wire.settings.model_effort.label/help`.
- `ظاہری انداز` (Appearance), `تھیم` (Theme), `استعمال` (Usage), `ترتیبات` (Settings),
  `منظور کریں` (Approve — same string as `relay.asks.pending.approve`), `جاری رکھیں` (Resume — the
  `/resume` term), `پن شدہ`/`پن کریں` (Pin) — all match the existing corpus.
- `سب پڑھا نشان زد کریں` (Mark all read) — terse, mirrors the English terseness and sits exactly at
  the 20-cell budget; `پڑھا ہوا نشان زد کریں` is used where the budget allows it. Accepted.
- `ورکنگ فولڈر` (Working folder) — hybrid loanword in Urdu script; permitted by the style guide's
  settled-loanword rule (no Latin leaked) and everyday Pakistani register. Accepted.
- `آپ کا انتظار` (Needs you) — idiomatic "awaiting you"; same phrase as "Waiting for you" on a
  different surface, which is accurate rather than confusing.
- Latin/isolation: the only Latin run in content is the product name `Local Operator` (kept verbatim
  in `list.empty_hint`); ICU argument names are untouched. No Latin token is wrapped in Urdu
  punctuation or reordered, and storage order is logical (renderers apply FSI/PDI). `row.actions`
  correctly moves `{name}` to the front for Urdu word order, and the name stays an isolated LTR run.
- Numeral orientation: Latin digits only; `کام 2/5`, `ایجنٹس 2/3` and `· 12` place the number in
  reading order and keep the slash/middle-dot forms the English uses.

---

## 3. Safety-relevant copy (explicit pass)

- **Approve** (`session.tray.approve`): `منظور کریں` — same imperative as the approvals sheet; not a
  softened or ambiguous form.
- **Approve accessible name** (`session.tray.approve_label`): `منظور کریں — ایک درخواست آپ کی منتظر ہے`
  — **starts with the same word as the visible pill** (the voice-control requirement in the sidecar)
  and keeps the em dash; `منتظر` agrees with `درخواست` (feminine). Correct.
- **Ended** (`session.subtitle.ended`, `session.ended`): `ختم ہو گیا` / `سیشن ختم ہو گیا · تاریخچہ محفوظ`
  — ended, history kept; no claim of deletion, nothing alarming.
- **Ended hint** (`session.ended_hint`): `کوئی پیغام بھیجیں یا “جاری رکھیں” دبائیں تو یہ آپ کے ہوم فولڈر میں دوبارہ کھل جائے گا`
  — send **or** Resume reopens it, and the folder named is the home folder (not the project folder),
  exactly as the sidecar requires; the quoted word is byte-identical to the Resume button.
- **Resume** (`session.resume`): `جاری رکھیں` — the established term; no "restart/delete" ambiguity.
- **Degraded** (`session.degraded`): see F4 — the only safety-adjacent wording issue; the current
  text says messages are held, not lost, so it is a refinement rather than a misleading banner.

---

## 4. Per-group verdict

| Group | Keys | Verdict |
|---|---|---|
| `drawer` | 12 | PASS with findings — F3 (minor), F6 (minor) |
| `link` | 3 | PASS with findings — F8 (nit) |
| `list` | 21 | PASS with findings — F2 (minor), F9 (nit) |
| `snackbar` | 1 | CLEAN |
| `row` | 10 | PASS with findings — F1 (minor) |
| `appearance` | 5 | PASS with findings — F5 (minor) |
| `session` | 25 | PASS with findings — F4 (minor), F7 (nit) |
| `composer` | 4 | CLEAN |
| `spend` | 1 | CLEAN |
| `model` | 1 | CLEAN |

**Namespace verdict: PASS on blockers/majors (0/0).** The file is structurally sound, uses the proper
Urdu script and the locale's established terminology, keeps every placeholder and CLDR category, and
uses a consistent respectful register (`آپ` + `کریں` imperatives; no `تم/تو`, no `!`, no `براہ کرم`
padding). The six minors are state/idiom refinements, two of them in busy-state labels; none changes
what an action does.

---

## Appendix — all 83 keys read

| # | key | English | اردو (target, as stored) | disposition |
|---:|---|---|---|---|
| 1 | `relay.chrome.drawer.open` | Open navigation | نیویگیشن کھولیں | read, no finding |
| 2 | `relay.chrome.drawer.close` | Close navigation | نیویگیشن بند کریں | read, no finding |
| 3 | `relay.chrome.drawer.label` | Navigation | نیویگیشن | read, no finding |
| 4 | `relay.chrome.drawer.conversations` | Conversations | گفتگوئیں | read, no finding |
| 5 | `relay.chrome.drawer.questions` | Questions | سوالات | read, no finding |
| 6 | `relay.chrome.drawer.past` | Past sessions | گزشتہ سیشنز | read, no finding |
| 7 | `relay.chrome.drawer.projects` | Projects | پروجیکٹس | read, no finding |
| 8 | `relay.chrome.drawer.library` | Library | لائبریری | F6 (minor) |
| 9 | `relay.chrome.drawer.usage` | Usage | استعمال | read, no finding |
| 10 | `relay.chrome.drawer.appearance` | Appearance | ظاہری انداز | read, no finding |
| 11 | `relay.chrome.drawer.settings` | Settings | ترتیبات | read, no finding |
| 12 | `relay.chrome.drawer.status` | {state, select, online {Connected · this phone} retrying {Offline · retrying} other {Connecting…}} | {state, select, online {منسلک · یہی فون} retrying {آف لائن · دوبارہ کوشش} other {منسلک ہو رہا ہے…}} | F3 (minor) |
| 13 | `relay.chrome.link.reconnecting` | Reconnecting… | دوبارہ منسلک ہو رہا ہے… | read, no finding |
| 14 | `relay.chrome.link.since_seconds` | {count, plural, one {Reconnecting · last synced # second ago} other {Reconnecting · last synced # seconds ago}} | {count, plural, one {دوبارہ منسلک ہو رہا ہے · # سیکنڈ پہلے} other {دوبارہ منسلک ہو رہا ہے · # سیکنڈ پہلے}} | F8 (nit) |
| 15 | `relay.chrome.link.since_minutes` | {count, plural, one {Reconnecting · last synced # minute ago} other {Reconnecting · last synced # minutes ago}} | {count, plural, one {دوبارہ منسلک ہو رہا ہے · # منٹ پہلے} other {دوبارہ منسلک ہو رہا ہے · # منٹ پہلے}} | F8 (nit) |
| 16 | `relay.chrome.list.search_open` | Search | تلاش | read, no finding |
| 17 | `relay.chrome.list.search_close` | Close search | تلاش بند کریں | read, no finding |
| 18 | `relay.chrome.list.search_label` | Search conversations | گفتگوئیں تلاش کریں | read, no finding |
| 19 | `relay.chrome.list.filter_label` | Filter conversations | گفتگوئیں فلٹر کریں | read, no finding |
| 20 | `relay.chrome.list.filter.all` | All | سب | read, no finding |
| 21 | `relay.chrome.list.filter.needs_you` | Needs you · {count, number} | آپ کا انتظار · {count, number} | F9 (nit) |
| 22 | `relay.chrome.list.filter.unread` | Unread · {count, number} | ناپڑھی · {count, number} | read, no finding |
| 23 | `relay.chrome.list.filter_empty` | No conversations match this filter | اس فلٹر سے کوئی گفتگو نہیں ملتی | read, no finding |
| 24 | `relay.chrome.list.mark_all` | Mark all read | سب پڑھا نشان زد کریں | read, no finding |
| 25 | `relay.chrome.list.marking` | Marking… | نشان زد ہو رہا ہے… | read, no finding |
| 26 | `relay.chrome.list.section.pinned` | Pinned | پن شدہ | read, no finding |
| 27 | `relay.chrome.list.section.active` | Active | فعال | read, no finding |
| 28 | `relay.chrome.list.section.previous` | Previous | پچھلے | read, no finding |
| 29 | `relay.chrome.list.connecting` | Connecting… | منسلک ہو رہا ہے… | read, no finding |
| 30 | `relay.chrome.list.empty` | No conversations yet | ابھی کوئی گفتگو نہیں | read, no finding |
| 31 | `relay.chrome.list.empty_hint` | Start one with New, or from Local Operator on your computer | “نیا” سے شروع کریں، یا اپنے کمپیوٹر پر Local Operator سے | read, no finding |
| 32 | `relay.chrome.list.no_matches` | No matching conversations | کوئی مماثل گفتگو نہیں | read, no finding |
| 33 | `relay.chrome.list.clear_search` | Clear search | تلاش صاف کریں | read, no finding |
| 34 | `relay.chrome.list.new` | New | نیا | read, no finding |
| 35 | `relay.chrome.list.new_label` | New session | نیا سیشن | read, no finding |
| 36 | `relay.chrome.list.starting` | Starting… | شروع… | F2 (minor) |
| 37 | `relay.chrome.snackbar.dismiss` | Dismiss | بند کریں | read, no finding |
| 38 | `relay.chrome.row.actions` | Actions for {name} | {name} کے لیے کارروائیاں | read, no finding |
| 39 | `relay.chrome.row.pin` | Pin to top | اوپر پن کریں | read, no finding |
| 40 | `relay.chrome.row.unpin` | Unpin | پن ہٹائیں | read, no finding |
| 41 | `relay.chrome.row.saving` | Saving… | محفوظ… | F1 (minor) |
| 42 | `relay.chrome.row.tags` | Tags | ٹیگز | read, no finding |
| 43 | `relay.chrome.row.tags_none` | None | کوئی نہیں | read, no finding |
| 44 | `relay.chrome.row.mark_read` | Mark as read | پڑھا ہوا نشان زد کریں | read, no finding |
| 45 | `relay.chrome.row.open` | Open | کھولیں | read, no finding |
| 46 | `relay.chrome.row.swipe_pin` | Pin | پن کریں | read, no finding |
| 47 | `relay.chrome.row.more` | More | مزید | read, no finding |
| 48 | `relay.chrome.appearance.theme` | Theme | تھیم | read, no finding |
| 49 | `relay.chrome.appearance.wide_view` | Wide view | چوڑا نظارہ | read, no finding |
| 50 | `relay.chrome.appearance.group_by_tag` | Group conversations by tag | گفتگوئیں ٹیگ سے گروپ کریں | F5 (minor) |
| 51 | `relay.chrome.appearance.on` | On | آن | read, no finding |
| 52 | `relay.chrome.appearance.off` | Off | آف | read, no finding |
| 53 | `relay.chrome.session.back` | Back to conversations | گفتگوؤں میں واپس | read, no finding |
| 54 | `relay.chrome.session.connecting` | Connecting… | منسلک ہو رہا ہے… | read, no finding |
| 55 | `relay.chrome.session.loading` | Loading conversation… | گفتگو لوڈ ہو رہی ہے… | read, no finding |
| 56 | `relay.chrome.session.empty_title` | No messages yet | ابھی کوئی پیغام نہیں | read, no finding |
| 57 | `relay.chrome.session.empty_hint` | Send a message to get started | شروع کرنے کے لیے پیغام بھیجیں | read, no finding |
| 58 | `relay.chrome.session.subtitle.working` | Working · {elapsed} | کام جاری · {elapsed} | read, no finding |
| 59 | `relay.chrome.session.subtitle.working_plain` | Working | کام جاری | read, no finding |
| 60 | `relay.chrome.session.subtitle.waiting` | Waiting for you | آپ کا انتظار | read, no finding |
| 61 | `relay.chrome.session.subtitle.idle` | Idle | فارغ | read, no finding |
| 62 | `relay.chrome.session.subtitle.ended` | Ended | ختم ہو گیا | read, no finding |
| 63 | `relay.chrome.session.subtitle.not_answering` | Not answering | جواب نہیں دے رہا | read, no finding |
| 64 | `relay.chrome.session.tray.label` | Session status | سیشن کی حالت | read, no finding |
| 65 | `relay.chrome.session.tray.approve` | Approve | منظور کریں | read, no finding |
| 66 | `relay.chrome.session.tray.approve_label` | Approve — a request is waiting for you | منظور کریں — ایک درخواست آپ کی منتظر ہے | read, no finding |
| 67 | `relay.chrome.session.tray.questions` | {count, plural, one {# question} other {# questions}} | {count, plural, one {# سوال} other {# سوالات}} | read, no finding |
| 68 | `relay.chrome.session.tray.tasks` | Tasks {done, number}/{total, number} | کام {done, number}/{total, number} | read, no finding |
| 69 | `relay.chrome.session.tray.agents` | Agents {running, number}/{total, number} | ایجنٹس {running, number}/{total, number} | read, no finding |
| 70 | `relay.chrome.session.tray.agents_failed` | {failed, plural, one {Agents {running, number}/{total, number} · # failed} other {Agents {running, number}/{total, number} · # failed}} | {failed, plural, one {ایجنٹس {running, number}/{total, number} · # ناکام} other {ایجنٹس {running, number}/{total, number} · # ناکام}} | read, no finding |
| 71 | `relay.chrome.session.tasks_title` | Tasks | کام | read, no finding |
| 72 | `relay.chrome.session.agents_title` | Agents | ایجنٹس | read, no finding |
| 73 | `relay.chrome.session.ended` | Session ended · history kept | سیشن ختم ہو گیا · تاریخچہ محفوظ | read, no finding |
| 74 | `relay.chrome.session.ended_hint` | A send or Resume reopens it in your home folder | کوئی پیغام بھیجیں یا “جاری رکھیں” دبائیں تو یہ آپ کے ہوم فولڈر میں دوبارہ کھل جائے گا | read, no finding |
| 75 | `relay.chrome.session.resume` | Resume | جاری رکھیں | read, no finding |
| 76 | `relay.chrome.session.resuming` | Resuming… | کھل رہا ہے… | F7 (nit) |
| 77 | `relay.chrome.session.degraded` | Not answering — sends will wait | جواب نہیں دے رہا — پیغام روکے ہیں | F4 (minor) |
| 78 | `relay.chrome.composer.label` | Message | پیغام | read, no finding |
| 79 | `relay.chrome.composer.folder_label` | Working folder: {name} | ورکنگ فولڈر: {name} | read, no finding |
| 80 | `relay.chrome.composer.model_label` | Model: {label} | ماڈل: {label} | read, no finding |
| 81 | `relay.chrome.composer.spend_label` | Spend & context: {label} | خرچ اور سیاق و سباق: {label} | read, no finding |
| 82 | `relay.chrome.spend.title` | Spend & context | خرچ اور سیاق و سباق | read, no finding |
| 83 | `relay.chrome.model.effort` | Reasoning effort | استدلال کی شدت | read, no finding |

Counts: **0 blocker, 0 major, 6 minor (F1–F6), 3 nit (F7–F9)**.
Every row above was read against the English source and its context note; the ICU-bearing
rows were additionally rendered (see `audit.log`).
