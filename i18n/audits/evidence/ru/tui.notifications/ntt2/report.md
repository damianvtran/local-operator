# Russian audit, round 2 — `tui.notifications` delta (audit_id `ntt2`)

Auditor: translation-auditor (lopdev), session `ca5d33d6f6aa`, actual model
**radient/auto** (session model-switch notices: requested
anthropic/claude-opus-5-5 fell back → claude-sonnet-5-5 → openai/gpt-6.1-sol →
radient/auto, which served both rounds). Independent of the translation: I
authored none of the catalogue, applied none of these bytes, and edited no
catalogue, code, generated file, git state or ledger.

Scope: the remediation delta only — the three copy changes I accepted from
round 1, plus the ru style-guide note. Round-1 findings T1/T3 were resolved by
recording them as budget exceptions; T6/T7 were nits with no change required.
Unchanged keys were not re-audited semantically; the full mechanical harness was
re-run over all 72 keys as a regression net.

Verdict: **PASSED** — 0 blocker, 0 major, 0 minor, 1 nit (documentation
precision only, does not gate). **The ru ledger row may ship at this hash.**

## Pinned inputs (all read at `77b3c2cf45ea5a21e316c11e1fd7a56c9a135aa6`)

| Input | sha256 |
|---|---|
| `ru/tui.notifications.json` (untracked on disk) | `eee1186170a1856f4e59528b4769b8b834c0c928ce7cad325ceedccc789058fc` |
| `en/tui.notifications.json` | `b415bc703840afabc1c3ecaeecdad24a84d9a6ab3367833cad4f17f0e0e70b68` |
| `en/tui.notifications.context.json` (branch) | `9d10eb5db735fc9cbfb5152f9106a296f39bc85a722ff67b38ee5233c4d645b6` |
| round-1 target (for the diff) | `03cdb4d6fc69764229a5dd15611a0d64e1856dca54107904bfdca223f8b947c1` |
| round-1 context (for the budget diff) | `0a9ee26060dcc5a5dc44c784fcdff2d3b64c6457899d7b3c382c0c7e32d85907` |
| `i18n/style-guides/ru.md` (committed) | `89b459c54bf10b2d3efa9d9b28de4c7a0d57f3b6cf8afc56019f2931d31e0efc` |

On-disk bytes were copied to this bundle's `inputs/` **before** reading, hashed,
and matched the reported `eee11861…` exactly. Re-verified unchanged after the
audit.

## 1. The delta is exactly the three keys

`changed keys (3): ['chip_high', 'pointer_unread', 'stale_skipped']`; key set
unchanged (72/72), key order still identical to English and still sorted. The
unified byte diff is **6 ± lines — three replaced lines, nothing else**:

```
-  "tui.notifications.chip_high": "важно",
+  "tui.notifications.chip_high": "выс.",
-  "tui.notifications.pointer_unread": "Отчёты о завершении сессий перенесены в /unread.",
+  "tui.notifications.pointer_unread": "Завершённые сессии перенесены в /unread.",
-  "tui.notifications.stale_skipped": "{count, plural, one {… с момента просмотра — оставлено как есть.} …}",
+  "tui.notifications.stale_skipped": "{count, plural, one {… после просмотра — не тронуто.} …}",
```

No unexpected byte: no CR, no tab, no trailing whitespace on any line, still
UTF-8, still NFC, still ends `}\n`, 7136 → 7031 bytes (the whole saving is those
three lines).

## 2. The three changed keys

- **`chip_high` = «выс.»** (4 cells, budget 6 — round 1 was 5 cells). Exactly the
  agreed text; no arguments; NFC clean. It now matches the same English "High"
  in the audited desktop sibling («Высокий») and this file's own priority filter
  («высокий»). T2 resolved.
- **`pointer_unread` = «Завершённые сессии перенесены в /unread.»** (40 cells,
  was 48; no budget declared). Exactly the agreed sentence, and it now uses the
  term core ru already has for these rows («Непрочитанные завершённые сессии»,
  `wire.slash.notifications`). T4 resolved.
- **`stale_skipped` = the exact ICU value I supplied**, arm by arm verified
  against the agreed text (not by eye — each arm compared as a string):
  `one` «# уведомление изменилось после просмотра — не тронуто.»,
  `few` «# уведомления изменились после просмотра — не тронуты.»,
  `many` «# уведомлений изменилось после просмотра — не тронуто.»,
  `other` «# уведомления изменилось после просмотра — не тронуто.»;
  wrapper `{count, plural, …}` with exactly four arms. T5 resolved.

Argument contract for all three: identical to English (name, kind, order);
`_arguments_problem` → `None`. Plural completeness on `stale_skipped`
(`_number_upgrade_problem`): `None` — all four ru categories present, no empty
branch, every branch renders `#`.

## 3. CLDR renders at the required counts (measured, not asserted)

| count | ru | cells | en | cells |
|---|---|---|---|---|
| 0 | `0 уведомлений изменилось после просмотра — не тронуто.` | 54 | `0 changed since you looked and were left.` | 41 |
| 1 | `1 уведомление изменилось после просмотра — не тронуто.` | 54 | `1 changed since you looked and was left.` | 40 |
| 2 | `2 уведомления изменились после просмотра — не тронуты.` | 54 | `2 changed since you looked and were left.` | 41 |
| 3 | `3 уведомления изменились после просмотра — не тронуты.` | 54 | `3 changed since you looked and were left.` | 41 |
| 5 | `5 уведомлений изменилось после просмотра — не тронуто.` | 54 | `5 changed since you looked and were left.` | 41 |
| 11 | `11 уведомлений изменилось после просмотра — не тронуто.` | 55 | `11 changed since you looked and were left.` | 42 |
| 21 | `21 уведомление изменилось после просмотра — не тронуто.` | 55 | `21 changed since you looked and were left.` | 42 |
| 22 | `22 уведомления изменились после просмотра — не тронуты.` | 55 | `22 changed since you looked and were left.` | 42 |
| 25 | `25 уведомлений изменилось после просмотра — не тронуто.` | 55 | `25 changed since you looked and were left.` | 42 |
| 101 | `101 уведомление изменилось после просмотра — не тронуто.` | 56 | `101 changed since you looked and were left.` | 43 |
| 1234 | `1 234 уведомления изменились после просмотра — не тронуты.` | 58 | `1,234 changed since you looked and were left.` | 45 |

Category selection is correct at every count (0/5/11/25 → `many`, 1/21/101 →
`one`, 2/3/22 → `few`, **1234 → `few`**: 1234 % 10 = 4, 1234 % 100 = 34); NBSP
grouping renders; no literal `#` leaks into any render; each arm's verb and
participle agree with its own numeral. Worst realistic count now 54 (was 66) —
12 cells saved, delta vs English constant at +13, so the footer stays on one
line from 55 columns instead of 67.

## 4. `pointer_unread` literal

`index=32 segment='/unread'`, codepoints `0x2f 0x75 0x6e 0x72 0x65 0x61 0x64`
— byte-exact Latin, undeclined, no Cyrillic inside, sentence ends ` /unread.` ✓.

## 5. The style-guide note (committed at `77b3c2cf45e`)

Records both accepted exceptions under "Recorded budget exception (notifications
inbox, wave 3)": `filter_unread` keeps «новые», `seen_in_chat` keeps «было в
чате», with «непрочитанные» (13 cells) and «видели в чате» (13) exceeding the 8-
and 12-cell budgets. Every load-bearing claim checks out against my round-1
measurements (13 vs 8; 13 vs 12; no shorter accurate synonym — «непрочит.» is 9
cells, still over 8), and the exceptions are recorded in the guide rather than
the budgets being silently widened, which is exactly the rule the guide sets for
itself. This is the correct resolution of T1/T3 and I sign it off.

### D1 — nit — the note's two imprecise descriptions (documented, non-gating)

- `i18n/style-guides/ru.md` (wave-3 exception, lines 91–96)
- Current: "…both exceed their **chip** budgets (8 and 12) … the chip is absorbed
  by the **title column** and drops wholesale below **~45** columns".
- Why it is imprecise: the 8-cell slot is the list header's right-edge segment
  (`filter_unread` — dropped whole when the header line is too narrow), not a row
  chip; title-column absorption is the row-chip mechanism and applies to
  `seen_in_chat` only. Measured drop thresholds (real `row_line`, ru copy, at
  `77b3c2cf`): the lone `seen_in_chat` chip is dropped through 43 columns and
  appears at 44; with the priority chip riding along it is dropped through 48 and
  appears at 49; `seen_live` alone 44/45; `выс.` alone 35/36 — and the threshold
  is width-only, not title-length-dependent (an 80-cell title gives the same 36).
- Proposed (optional, no re-audit needed): "…their budgets (8 for the header's
  read-filter segment, 12 for the row chip)… the row chip is dropped wholesale
  at ≤48 columns (≤43 alone)…".
- Reason: a future reader measuring the header segment cannot reproduce "~45
  columns" from the row-chip rule; the numbers it cites for the two strings are
  correct, so the exception itself is unaffected.

## 6. Mechanical harness re-run (regression net, all 72 keys)

On a scratch `git archive` of `77b3c2cf45e` with the on-disk ru file dropped in:
`check_catalogues()` **0 problems**; argument contract **0 failures / 72**; plural
completeness PASS on all 8 plural arguments; literal tokens intact; NFC/bidi/
typography clean; budgets met (`chip_high` 4/6); `row_line` exact width at
40/60/76/80/120/140; 8 negative controls fail closed. `stale_skipped`'s grammar
probes were updated to the remediated forms, and two fresh controls were added
for the new bytes (`#` dropped from `few` → the completeness rule reports it;
wrong case in `few` → the grammar probe rejects it).

## 7. Context sidecar move (`9d10eb5d…` vs round 1)

Exactly two budgets moved, both Spanish-driven and both with their notes:
`hint_answer` 8 → 9 and `hint_dismiss_all` 12 → 14. **None** of the three
audited keys' budgets moved (`chip_high` 6 → 6, `pointer_unread` none,
`stale_skipped` none), and the ru strings for the two widened keys fit both the
old and the new budget (`hint_answer` «ответить» 8 ≤ 8; `hint_dismiss_all`
«скрыть все» 10 ≤ 12), so nothing ru-relevant changes. Verified, not assumed.

## 8. Noted outside scope (not findings against this file)

1. The ledger at `77b3c2cf45e` carries `tui.notifications` rows for es, fr, vi,
   hi, ur and zh-CN but **no ru row yet** — so "the ledger row may ship at this
   hash" means the row still has to be written; per the sibling rows its note
   should cite `ntt2 (i18n/audits/evidence/ru/tui.notifications/ntt2/)` and
   target `eee11861`.
2. Round-1 outside-scope items stand unchanged: the detail header paints the raw
   Latin priority value, the sidebar footer mixes «уведомлений: N» with the
   English-only `asks: N`, and `ask_expires` can render the `now` sentinel in
   every locale.

## Verdict

| Area | Scope | Verdict |
|---|---|---|
| Delta containment | 3 keys, byte diff | clean — exactly the three keys |
| argument tuple / plurals / CLDR 0–1234 | 3 keys | clean |
| Literals (`/unread`) | 1 key | clean, byte-exact |
| Widths | 3 keys | clean (54 vs 66 at the realistic worst count) |
| NFC / bidi / byte hygiene | whole file | clean |
| Style-guide exception note | 1 note | accurate → 1 nit (D1) |
| Mechanical harness, all keys | 72 | clean, 0 failures |
| **Overall** | **72 read, 3 re-audited semantically** | **PASSED — ru ledger row may ship at `eee11861…`** |

Limitations: LLM semantic judgement, not native-speaker attestation; static copy
plus committed render-function geometry, not a live TUI frame; no release claim.
The audited bytes are an untracked working-tree file — if they are edited after
this audit, the hash no longer binds them.
