# Hindi audit — tui.notifications (ntt1)

Auditor: translation-auditor (`tui-audit-hi`), instance d184e3a74566. Not the
translator; no catalogue, generated file, git state or ledger was edited.
Target bytes audited: `8d503495a5aed2562306b7c06ed15eac6bbb5486c2717bb10bbda65e3303fc1e`
(copied out of the worktree first; the live file still matched at copy time).
Source/context at pin `d25392f3553c38ce1740c98e53c9c66a700e9e64`:
`b415bc…70b68` / `0a9ee2…85907`, both verified before any reading.

**Verdict: passed — 72/72 keys read; 0 blocker, 0 major, 1 minor, 2 nit.**

| severity | count | gating |
|---|---|---|
| blocker | 0 | — |
| major | 0 | — |
| minor | 1 | no (code-side, not this file's text) |
| nit | 2 | no |

The Hindi is accurate, idiomatic and register-consistent; every ICU argument,
category and literal token survives. Nothing found blocks the ledger row.

## What was actually run

Both passes ran outside the worktree against a scratch export of the pinned
tree, with the audited hi bytes placed in that tree's catalogue root, using the
repo's own modules (`local_operator/i18n/runtime.py`, `scripts/i18n/check.py`,
`rich.cells.cell_len`, and the pinned `local_operator/tui/notifications_render.py`).

1. `audit_hi.py` — 466 PASS / 0 FAIL (`audit.log`):
   - input identity (3 sha256 vs pins), key-set parity (72/72/72), `load_catalogue` equality
   - `check_catalogues()` over the pinned tree: **0 problem lines**
   - ICU argument parity per key via `_arguments_problem` (name, order, kind)
   - `_check_plural_categories` + hi CLDR coverage: 7 plural-bearing keys, all define
     `one`+`other` and cover hi's `['one','other']`
   - render at 0/1/2/5/21/1000 and realistic strings: every branch non-empty
   - fit: every rendered variant (worst-case params) within the en sidecar `maxCells`
   - literal tokens byte-exact; NFC; no bidi/zero-width controls; no doubled spaces
   - **negative controls, all fail-closed**: dropped `{source}` → caught; renamed
     `{error}`→`{err}` → caught; plural category `few` for hi → caught; plural
     without `other` → caught; a legal single-`{count}` form → stays green; an
     over-budget string → caught by the fit measurer.
2. `render_hi.py` — 31 PASS / 0 FAIL (`render.log`): the pinned page renderer
   painted with hi copy at 40/60/76/100/120 columns — rows exact width, header
   within width, detail lines within width (bodies wrap), every detail section
   and the footer ladder measured hi vs en.

## Findings

### NTT1-M1 · minor · `tui.notifications.row_now` — time column loses one cell of alignment
- en `now` · current `अभी` · proposed **no text change** (the word is right)
- Evidence (`render.log`, width 40): `'    ▁▃  Plan updated                 अभी '`
  against `'▸ • ▁▃▅ Nightly…  उच्च                 6m'`; measured stamp cells **4**
  vs the 5-cell `TIME_CELLS` budget, and `cell_len("अभी".rjust(5))` = 4 while
  `cell_len("now".rjust(5))` = 5.
- Reason: `notifications_render.py` (pinned line 164) does
  `stamp = row_time(...).rjust(TIME_CELLS)` — `str.rjust` counts **code points**,
  and `अभी` is 3 code points but 2 cells, so the right-aligned time column sits
  one cell short on rows under a minute old. Hindi is the only locale whose
  `row_now` is not ASCII, so only hi shows it. Fix belongs in code
  (`' ' * max(0, TIME_CELLS - cell_len(stamp)) + stamp`), not in this file.

### NTT1-N1 · nit · `tui.notifications.group_week` — "earlier this week" can read "before this week"
- en `Earlier this week` · current `इस हफ़्ते पहले` · proposed `इसी हफ़्ते पहले`
- Reason: without the emphatic `इसी`, the section header can be parsed as "before
  this week" — which is the `पुराने` section's own range. Same measured width
  (10 cells, cap 30), so the fix costs nothing.

### NTT1-N2 · nit · `tui.notifications.empty_filtered_title` — oblique plural of a plural demonstrative
- en `Nothing matches these filters` · current `इन फ़िल्टर से कुछ भी मेल नहीं खाता`
- proposed `इन फ़िल्टरों से कुछ भी मेल नहीं खाता` (23 cells, cap 40)
- Reason: the hi style guide asks for the natural plural in oblique cases
  (`फ़ाइल → फ़ाइलें`, `सत्र → सत्रों`); plural `इन` wants `फ़िल्टरों`. Caveat: the
  string is byte-identical to the already-audited **desktop** copy, so changing it
  here alone creates a new cross-surface split — fix both or deliberately keep both.

## Translator notes checked in the bytes (not taken on trust)

- **`asks_stay` / `stale_skipped` replace a literal `1` with `#`** — confirmed and
  correct: hi puts 0 in `one`, so `@0` renders `0 सवाल … बना रहेगा।` and
  `0 सूचना … बदल गई, इसलिए छोड़ दी गई।`; a literal `1` would have printed "1" for
  zero. (`audit.log`, render section @0/1/2/5/21/1000.)
- **`confirm_dismiss_session` moves `{source}` inside the plural arms** — the
  argument contract is unchanged: `(count, plural), (source, simple)` in both
  locales, same order, `check_catalogues()` clean, and renders read
  `nightly-build का 1 अपडेट हटाएँ? y/n` / `nightly-build के 2 अपडेट हटाएँ? y/n` —
  the का/के agreement with the counted `अपडेट` is right at both categories.
- **`seen_in_chat` uses `चैट` for width** — measured: `चैट में देखा गया` = 10 cells,
  `बातचीत में देखा गया` = 12 of a 12-cell cap. Both fit, so this is headroom for
  the row title rather than a forced choice; see the cross-surface note below.
- **`hint_read` / `hint_undo` are exactly 8/8 cells, Devanagari caveat assessed** —
  no overflow risk, and the numbers say why: `cell_len` gives one cell per base
  consonant and zero per matra/nukta/virama, so `पढ़ा हुआ मानें` = 6 bases + 2
  spaces = 8 and `पूर्ववत करें` = 7 bases + 1 space = 8. The only conjunct (`र्व`)
  can render *narrower* (one ligature), never wider, so cell_len is an upper bound
  on a standard terminal's advance. Zero headroom stays, and it is visible in the
  footer ladder: at 60 columns hi costs 59/59 (en 46/59) and at 40 columns hi drops
  the `u` read hint that en keeps — absorbed by the page's whole-hint rule
  (§8.4), so nothing clips.

## Cross-surface notes (not this file's defect, not gating)

- **Mark-read lemma split.** This file uses `पढ़ा हुआ मानें` / `पढ़ा हुआ माना गया`
  (6 keys) and `अपठित मानें` / `अपठित माना गया` (2 keys) — internally consistent and
  the form the earlier audit (c9c5d8079e51) preferred. The desktop hi still says
  `पढ़ा गया चिह्नित करें`, and core wire says `read उन्हें पढ़ा हुआ चिह्नित करता है`
  (`wire.slash.notifications`) and `read सब पढ़ा हुआ करता है`
  (`wire.slash.notifications_page.help`) — three Hindi forms for one action across
  surfaces. Recorded as a cross-surface minor only; this file must not be "fixed"
  toward the older desktop wording.
- **`seen_in_chat` term split** (`चैट` here, `बातचीत` in the desktop catalogue for
  the same fact, while this file itself says `बातचीत` in 6 other keys). The
  English differs too (`seen in chat` vs `Seen in conversation`), so this is not a
  glossary-table violation; it is a same-fact term split worth one line in the
  programme notes.
- **Untranslated priority token in the detail header.** The pinned renderer builds
  that chip from the raw data word: `right.append(BARS[...].rstrip() + " " + priority)`
  — the hi detail renders `▁▃ normal · अपडेट · 1/42` (`render.log`). No catalogue
  key exists for it, so this is a source-side gap, not a translation defect.

## Judged and accepted (no finding raised)

- Imperative voice in `unread_mark_all` / `unread_mark_these` (`/unread read से सभी
  {count} को पढ़ा हुआ मानें`) where English is declarative: in a receipt listing the
  instruction reading is natural Hindi and the action is unchanged.
- `banner_digest_title` `# और उच्च प्राथमिकता वाली सूचनाएँ` — `और` = "more" reads
  correctly before `उच्च प्राथमिकता वाली`; the "# and …" misreading does not survive
  the `वाली` construction.
- `group_waiting` / `header_counts` use two phrasings (`आपके जवाब का इंतज़ार`,
  `आपके इंतज़ार में`) for two roles (section header vs count), matching the desktop
  catalogue's own split.
- `ask_expires` `{time} में समाप्त होगा` covers both a duration (`38m में`) and a
  clock time (`14:05 में`) correctly — the locative does both jobs.
- `priority_normal_up` keeps the `+` (`सामान्य+`, 5/8) rather than expanding, as the
  English does; `position` `{index}/{total}` is an accepted compact form of
  "{index} of {total}" in a header tail.

## Also verified

- 9 of 72 hi strings are longer than their en source (`banner_digest_body` +3,
  `restored` +3, `hint_read`/`hint_undo` +4 each, others +1..+2); every one is inside
  its `maxCells`, and the two doubled ones are the 8/8 hints above.
- `header_counts` worst case `1,000 अपठित · 1,000 आपके इंतज़ार में` = 29 of 40 cells;
  `sidebar_note` `सूचनाएँ 1,000` = 10 of 18; `header_partial` `शुरुआती 1,000` = 10 of 12.
- Header at 40/60/76/100/120 columns stays within width; the filter edge drops whole
  segments (`अपठित │ अपडेट │ सामान्य+ │ समय` present at ≥60).
- Register: `आप` throughout, no `तुम`/`तू`, no added `!`, danda for full stops,
  `?` preserved, `y/n` literal, `/notifications`, `/unread`, `/unread read` and
  `read` byte-exact (both `/unread read` lines start with the token).
- NFC everywhere; no U+200E/F, U+202A–E, U+2066–9, ZWSP/ZWNJ/ZWJ; no Devanagari
  digits (`1,000` is the formatter's Indian grouping, as the guide requires).
- Formatting matches the house style (2-space indent, sorted keys, UTF-8, no BOM,
  trailing newline) — byte-identical to a re-dump of itself.
- RTL: not applicable (hi is LTR).

## Limitations

- LLM semantic judgement, not native-speaker attestation.
- The page renderer was driven directly at real widths with the SHIPPED gate shimmed
  to `hi` (hi is precisely what this ledger row would ship, so a live process cannot
  paint it yet); no live TUI process on a real terminal was exercised.
- No release-eligibility, runtime-service or native/design sign-off is claimed here.

## Model disclosure

Requested `anthropic/claude-opus-5-5`. During this turn the session received
provider-failure model-switch notices: `anthropic/claude-sonnet-5-5`, then
`openai/gpt-6.1-sol`, latest `radient/auto` (reason: provider failure: quota).
The work above was produced across those switches; the latest notice names
`radient/auto` as the serving model.
