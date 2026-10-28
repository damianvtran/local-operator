# Vietnamese audit — `tui.notifications` (72 keys)

Auditor: translation-auditor (lopdev team), session `8408dcd6e3c9`, independent of the
translator. Model actually serving this audit: **radient/auto** — the declared primary
`anthropic/claude-opus-5-5` was replaced mid-audit by three provider-quota fallback notices
(`anthropic/claude-sonnet-5-5`, `openai/gpt-6.1-sol`, `radient/auto`); the latest notice governs.
Pinned commit: `d25392f3553c38ce1740c98e53c9c66a700e9e64`.

Pins verified from the bytes, not taken on trust:
`en` `b415bc70…70b68`, context sidecar `0a9ee260…85907`, target `98cc6e1b…d341` — all three
match the delegation. 72/72 keys read, each against its `context.json` entry (role, note, tone,
`maxCells`). Siblings read for terminology: the audited desktop `ui.notifications` vi catalogue
(130 keys) and the core vi `wire.errors.notifications`, `wire.slash.notifications`,
`wire.slash.notifications_page` at the same commit.

**Verdict: passed — 2 minor (V1, V2), 0 blocker, 0 major, 0 nit. The ledger row may ship.**

---

## Findings

### V1 — minor — `tui.notifications.seen_live`

- English: `seen live` · current vi: **`xem khi nhận`** · proposed: **keep, recorded deviation**
  (or `xem trực tiếp` if `maxCells` can rise to 13).
- Reason: the desktop catalogue renders the same English term (`Seen live`) as **`Đã xem trực tiếp`**
  (`ui.notifications.seen.live`). Two Vietnamese phrasings now name one English term. The 12-cell
  budget is what forces it: `Đã xem trực tiếp` = 16 cells, `xem trực tiếp` = 13 (measured, both
  over), and the current `xem khi nhận` = 12/12. The chosen phrase is semantically sound — the
  desktop hint behind the same fact is "marked read because you were in the conversation when it
  arrived", and `nhận` carries "arrived" — and it stays parallel with `seen_in_chat` (`xem khi hiện`).
  This is a recorded deviation rather than a wording defect; it gates nothing.

### V2 — minor — combined footer receipt clips below the third clause

- English (worst composition): `Dismissed 3 updates. 1 ask stays until it's answered.
  2 changed since you looked and were left.` = **95 cells**
- Current vi: `Đã ẩn 3 cập nhật. 1 câu hỏi được giữ lại đến khi có câu trả lời. 2 mục đã thay đổi
  kể từ lúc bạn xem nên được giữ nguyên.` = **121 cells**
- Container: the notice row is `.settings-view-detail` inside `.notifications-view` — measured at
  80×24 as **76 cells wide, 1 row tall** (`GEO {'notice': [76, 1]}`).
- Measured: vi folds to 2 rows → clipped; the rendered frame paints
  `Đã ẩn 3 cập nhật. 1 câu hỏi được giữ lại đến khi có câu trả lời. 2 mục đã` and stops.
- Control: the **English line clips too** (95 → 2 rows, frame ends `…2 changed since you lo`).
  So the composition limit is the source's, not the translation's; vi aggravates it by 26 cells.
  No rewording fixes it: with `dismissed` + `asks_stay` already spending 64 cells, the third clause
  would need ≤ 12 cells to fit, and the shortest honest Vietnamese for "changed since you looked
  and were left" does not fit in 12.
- Reason: a fit failure reachable only when **two** skip reasons and at least one changed row
  coincide (asks are skipped by design; the stale skip needs a concurrent update between paint
  and write). Fix belongs in composition (`_reports`/`_compose` join) or in allowing the notice row
  two lines — not in the catalogue. Filed as minor, gates nothing; the coder/design owner should
  decide. Each receipt **alone** fits: dismissed 17, marked_read(100) 30, restored(100) 21,
  nothing_unread 26, stale_skipped(100) 58, asks_stay(1) 46, all ≤ 76.

---

## Assessed deviations — no finding, budget arithmetic recorded

The translator's flagged choices were checked against the layout, not accepted on trust:

| Key | Chosen | Alternative | Why the alternative cannot ship |
|---|---|---|---|
| `row_now` (5) | `mới` (3) | `vừa xong` (8) | 8 > 5. Rendered in a real frame: the time column paints `  mới` whole. `mới` reads as "new/just arrived" in a time column, and the source itself splits `now` (row) from `just now` (detail, `vừa xong`, 12 cells) — so the split mirrors English. |
| `group_conversation` (12), `hint_conversation` (14) | `trò chuyện` (10) | `cuộc trò chuyện` (15) | 15 > 12 and > 14. The bare noun is the classifier-dropped form of the corpus term (83 hits of `cuộc trò chuyện`); dropping `cuộc` in a terse label is ordinary Vietnamese. |
| `priority_normal_up` (8) | `thường+` (7) | `Bình thường trở lên` (19), `bình thường+` (12) | 12 > 8. `+` mirrors the source's `normal+`, and `cao` sits beside it in the same strip. |
| `seen_in_chat` (12) | `xem khi hiện` (12/12) | desktop `Đã xem trong cuộc trò chuyện` (28) | 28 > 12. Not glossary drift: the English itself differs (`seen in chat` here vs `Seen in conversation` on the desktop). |
| `hint_dismiss` (10) | `ẩn` (2) | `bỏ qua` (6) | **Verified correct.** `relay.ask.dismiss` uses `bỏ qua` for dismissing an *ask*, but this page never labels that action: `notifications_view.py` guards the dismiss hints with `if not is_ask` (lines 831, 839) and asks only offer `a`/`↵` answer. For update dismissal `ẩn` matches the desktop (`row.dismiss` = `Ẩn`, `toast.dismissed` = `Đã ẩn …`). Choosing `bỏ qua` here would have been the defect. |

At-budget strings render whole: `arg_read` 44/44 paints intact in the command picker at 80 columns
(frame 06: `❯  read        Đánh dấu tất cả thông báo chưa đọc là đã đọc`, no `…`, 76-cell row,
primary column 12, 63 cells of room); the six 8-cell labels (`filter_unread`, `hint_back`,
`hint_undo`, `hint_unread`, `kind_updates`, `update_kind`) are shown whole or not at all, because
the footer drops whole hints widest-first and never clips one; `seen_in_chat`/`seen_live` (12/12)
render whole as row chips at 76 columns and are dropped entirely at 40 columns, the same rule
English follows.

---

## Clean, with counts

- **Accuracy** — 72/72. Every action is the right action: dismiss (`ẩn`) vs mark read
  (`đánh dấu … đã đọc`) vs mark unread vs undo (`hoàn tác`) vs restore (`khôi phục`) are
  four distinct verbs, matching the four distinct writes. `pointer_unread` names `/unread` as the
  new home of completion receipts; `refused_argument` refuses a non-`read` argument; `arg_read`
  and `confirm_mark_all` say what is marked and that it is confirmed. `restored` adds `mục`
  (Vietnamese needs the noun where English's "Restored 3." does not).
- **Idiom** — 72/72. No calques. `Không có gì đang chờ bạn trong các cuộc trò chuyện`,
  `Câu hỏi được giữ lại đến khi có câu trả lời`, `Đang chờ bạn trong cuộc trò chuyện này` read as
  Vietnamese, not as translated English; `empty_none_body` is byte-identical to the already-audited
  desktop string.
- **Register** — 72/72. `bạn` only where the sentence needs an addressee (7 values); no `Vui lòng`,
  no `!`, sentence case, `…`/`—` per the style guide, `←` and `y/n` untouched.
- **Glossary consistency** — `thông báo` (8), `cập nhật` (8), `câu hỏi` (4), `ẩn` (5),
  `trò chuyện` (4), `đã đọc`/`chưa đọc` (6 each), `mục` (10) — one term per concept across all 72
  keys and against the desktop/wire siblings. No drift except V1.
- **Placeholder integrity** — every `{age}`, `{time}`, `{source}`, `{error}`, `{index}`, `{total}`,
  `{count}`, `{unread}`, `{waiting}` preserved with the same name, kind and order; repo checker
  reports 0 argument problems; the renderer produces a value for every key at every sample.
- **Plurals** — 7 plural keys, all `other`-only per the vi style guide, all render the count at
  **0, 1, 2, 5, 21, 1000** (`1.000` grouping correct via the formatter); en's `one` correctly
  collapsed; no empty branch; no branch that drops `#`.
- **Literals** — `/notifications`, `/unread`, `/unread read`, `read`, `y/n`, `←`, and the key
  letters `a`/`f` kept byte-exact (negative control nc4 proves the check fails closed when one is
  lost).
- **NFC / bidi** — file is NFC; no manual bidi controls (vi is LTR, none permitted). NFD mutant
  (nc5) fails closed.
- **RTL** — not applicable.

---

## Out of scope, reported to the manager

The detail header paints the **raw English priority token** in the Vietnamese UI: frame 07 shows
`← Thông báo … ▁▃ normal · cập nhật · 1/1`. `notifications_render.py:336` builds that tail as
`BARS[...].rstrip() + " " + priority`, so `normal`/`high`/`low` never pass through a catalogue.
There is no key for it in the 72, so this is **not** a defect of the translation and does not
affect the ledger row; the desktop has `row.priority.normal` = `Bình thường` if the coder wants a
key. Also noted, out of scope: opening an ask from the page hands off to the ask sheet, which is
still English (`enter answer`, `d decline`) — a different namespace, not this one.

---

## Evidence

`audit.log` holds the actual stdout: pins, `check_catalogues()`, ICU parity, the seven plural
renders at six counts, `maxCells` for 53 budgeted keys, NFC/bidi/literal checks, six negative
controls (each fails closed), the two rendered passes, and the pre-render arithmetic.
`frames/` holds the SVG stills from the real app at 80×24 in an isolated clone of the pinned
commit (`01-list`, `02-receipt-*`, `06-picker`, `07-detail`, `08-receipt-control-{vi,en}`,
`09-empty`).

Limits of this audit: one rendered surface at 80×24 (narrow mode); no 120-column wide-mode frame;
LLM semantic judgement, not native-speaker attestation; no runtime-service or release claim.
