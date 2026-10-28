# Spanish audit — `tui.notifications` (round 2, delta)

Auditor: translation-auditor, session `288dd1136395`, actual model **radient/auto** (session
model-switch notices: requested `anthropic/claude-opus-5-5`, served after fallbacks via
`radient/auto`). Independent of whoever produced this change.

**Verdict: PASS — 0 blocker, 0 major.** 71/72 strings byte-identical to the round-1 set; the
single change is correct and exactly at budget; both widened budgets equal the measured need.

**Round-scope statement.** This is a delta audit against `tui.notifications` (es), covering
only the changes named in the round-2 task. Everything else was audited in `../ntt1`; the
text below re-runs the full mechanical sweep as a regression check but does not re-audit the
unchanged strings' translation judgement. `ntt2` supersedes `ntt1` for the changed items;
`ntt1` remains the audit of record for the unchanged 71.

## Exported inputs and hashes

| input | source | sha256 |
|---|---|---|
| es target (on disk, uncommitted) | `local_operator/i18n/catalogues/es/tui.notifications.json` | `c4c2c502cd3805687bdaeb505b579c629a259bdbee3c453ba72539cca4588ac4` |
| en source @`8976104b83b` | committed tip | `b415bc703840afabc1c3ecaeecdad24a84d9a6ab3367833cad4f17f0e0e70b68` |
| en context @`8976104b83b` (committed) | committed tip | `0a9ee26060dcc5a5dc44c784fcdff2d3b64c6457899d7b3c382c0c7e32d85907` |
| en context (on disk, uncommitted) | working tree | `9d10eb5db735fc9cbfb5152f9106a296f39bc85a722ff67b38ee5233c4d645b6` |

The committed tip `8976104b83b` (branch `feat/notifications-tui`) has `M` on the context sidecar, so the two budgets you asked me to pin were still `8` and `12` **in the commit**. The new values exist only as an uncommitted working-tree edit:

- committed context `0a9ee260…85907`: `hint_answer` maxCells 8, `hint_dismiss_all` maxCells 12 — byte-identical to round 1.
- on-disk context `9d10eb5d…645b6`: `hint_answer` 9, `hint_dismiss_all` 14.

That is the one unexpected byte in this round; details under "Process notes" below. The audit is run against the on-disk context because that is the version that would ship, and the committed values are kept in the bundle as the control.

## 1. `hint_unread` — `no leída` → `sin leer`: accepted, resolves T-02

- Grammar/meaning: `sin leer` is the established Spanish state phrase for "unread"; as a footer label it reads exactly as the English `unread` does — the `u` key toggles a row to unread. Paired with `hint_read` `leer`, the footer now reads `u sin leer` / `u leer`, which is clearer than the previous `u no leída`.
- Width: `sin leer` = **8 cells against a budget of 8** (measured with `rich.cells.cell_len`), identical to the string it replaced, so this is a zero-fit-cost change.
- Consistency, and this is the substantive gain: the catalogue now uses one state term. `filter_unread` = `sin leer`, `header_counts` = `{n} sin leer`, `nothing_unread` = `No hay nada sin leer.`, and now `hint_unread` = `sin leer`. The participle form `no leída(s)` survives only where a participle is grammatically required — `marked_unread` `Marcadas como no leídas: {count}.` and `arg_read` `Marcar todas las no leídas como leídas`.
- That split is exactly the audited desktop pattern, not a workaround: desktop es has `filter.unread` = `Sin leer`, `flag.unread` = `Sin leer`, `drawer.group.unread` = `Sin leer`, against `toast.marked_unread` = `Marcadas como no leídas: {count}` and `row.mark_unread` = `Marcar como no leído`. This file now matches it key for key.
- Register: lowercase, matching the other footer hints (the desktop's capitalized `Sin leer` is a filter/flag label, a different role).

T-02 is **resolved**; no replacement finding.

## 2. The two widened budgets — confirmed, and they equal what the measurements demanded

| key | es string | measured cells | old budget | new budget | verdict |
|---|---|---|---|---|---|
| `hint_answer` | `responder` | 9 | 8 | **9** | exactly the measured minimum; still `+3` vs en (6) |
| `hint_dismiss_all` | `descartar todo` | 14 | 12 | **14** | exactly the measured minimum; still `+3` vs en (11) |

Re-measured on the round-2 bytes with the repo's `cell_len` and checked against the on-disk sidecar: both keys are now at budget, neither over, and neither has slack. With those two corrected, the mechanical sweep finds **exactly two** over-budget keys left in the whole namespace — `seen_live` and `seen_in_chat` — down from four (report line: `FAILURES=[] … over_budget_keys=2`).

**What widening the budget does not do**, stated plainly because it matters for the ledger: `maxCells` is not an input to any renderer. `hint_row(labels, width-1)` still costs the same 103 cells for the eight es labels, so the **E `descartar todo` hint still disappears at pane widths 92–103**, where English keeps all eight (all-hints-fit threshold: en 92 cols, es 104 cols — unchanged by the metadata edit). The widening converts a false constraint into an accurate one and records the trade in the note; it does not buy back the footer slot. That was the disposition I proposed in round 1 ("raise to 14, or accept the hint loss") and it is what shipped.

Sidecar integrity: 72 keys before and after, identical key set, all 72 still carry `note`, `role` and `tone`, and the count of keys carrying `maxCells` is 53 in both copies. Nothing else in the context file moved.

## 3. es message delta — exactly one string

`71/72` strings are byte-identical to the `559d0972…f783` set I audited in round 1. The only difference is `tui.notifications.hint_unread`, `no leída` → `sin leer`. No key added, none removed, no other differing byte. The en message file is unchanged (`b415bc70…70b68` at both commits), so no source string moved under the translation.

## 4. Status of the round-1 findings

**Resolved (2 minor / 2 of the four budget keys):**

| finding | what closed it |
|---|---|
| T-02 `hint_unread` `no leída` vs `sin leer` | changed to `sin leer`; verified above |
| T-03 (part) `hint_answer` 9/8 | budget widened to 9 |
| T-03 (part) `hint_dismiss_all` 14/12 | budget widened to 14 |

**Open, unchanged in status (2 minor, 6 nit) — none is a blocker or major:**

| finding | severity | status now |
|---|---|---|
| T-01 update term split: `kind_updates`/`update_kind` `novedad` vs this file's own `Actualizaciones anteriores`/`actualizaciones descartadas` and desktop `Actualización` | minor | stays drift-recorded. The budget was **not** widened for these (still 8), so the forcing constraint is unchanged and the disposition stands: keep `novedad`, or widen the kind budget to 12 as a product call. Note the precedent is real — audited core es `wire.slash.notifications_page` says `Preguntas y novedades`. |
| T-03 (part) `seen_live` `visto en vivo` 13/12, `seen_in_chat` `visto en chat` 13/12 | minor | **still acceptable** — re-measured on the round-2 bytes: both are 13 cells, budget unchanged at 12, and the chip remains a whole unit in `row_line`. It costs 4 title cells (`seen_live`) and 1 (`seen_in_chat`) at width 80, and is dropped wholesale below width 45 (en 41/44). Nothing clips, wraps or overflows; the row's own arithmetic keeps the total at `width`. Confirmed acceptable. |
| T-04 `updates_since` `desde {time}` vs desktop `desde las` | nit | stands, and this round touched nothing near it: its budget stays 30, the text still measures 27/29/30/30 cells at 1/9/12/99 (33 at 1234), the detail meta is `Text(no_wrap=False)` so it soft-wraps rather than crops, and the desktop's `desde las` would add 4 cells and break the budget. Keep as accepted deviation. |
| T-05 `hint_select` `elegir` vs desktop `Seleccionar` | nit | stands |
| T-06 `seen_in_chat` `chat` where the file/desktop say `conversación` | nit | stands |
| T-07 `banner_digest_title` `más de prioridad alta` garden path | nit | stands (unaddressed; still a nit) |
| T-08 `por responder` vs desktop `en espera` | nit | stands |
| T-09 `priority_normal_up` `normal+` | nit | stands |

## 5. Regression sweep on the round-2 bytes

Full re-run, not an inference from round 1: hashes pinned fail-closed; 72/72 key parity; 72/72 ICU argument tuples identical in name, kind and order; all 8 plural nodes carry the CLDR es categories `one`/`many`/`other`; literal tokens byte-exact (`/notifications`, `/unread`, `/unread read`, `read`, `y/n`, `←`, `·`, `normal+`, key glyphs `a`/`f`); NFC clean; no manual bidi controls; no space before `:;?!`; `¿…?` on both questions; no `usted`/`vosotros`; no exclamation marks. `scripts/i18n/check.py check_catalogues()` reports 0 problems for the whole tree.

Six negative controls on scratch copies, all still detected: dropped `{time}`; renamed `{count}`→`{total}`; es plural missing `many`; anglicised `/unread read`; NFD accent; budget shrunk under a clean key. (The last one now lists `hint_back` plus the two `seen_*` keys and no longer lists the two widened keys — consistent with the budget change, which is itself a control that the sweep reads the new sidecar rather than a cached one.)

## Process notes (not translation findings)

1. **The sidecar change is uncommitted at tip `8976104b83b`.** As committed, the budgets are still 8 and 12 and both notes are the old one-liners. If the branch were merged without committing `local_operator/i18n/catalogues/en/tui.notifications.context.json`, the widening would not ship and the two `maxCells` values I confirmed above would not exist in `main`. The es file itself is also still untracked. Both need committing for this audit to bind the shipped bytes.
2. **Four fields changed, not two.** You said "exactly the two maxCells values"; the diff shows **two `maxCells` values plus both `note` strings** (`hint_answer` 36 → 334 chars, `hint_dismiss_all` 34 → 390 chars), each gaining an audit rationale that cites the Spanish measurements. Nothing else moved. The added prose is accurate against my measurements — it quotes `responder`/`contestar` at 9, `responde` at 8, the French `répondre` calibration, the 92–103 column band, and the rejected `ignorar todo` — so I have no objection to it; I am naming it because you asked me to name anything that moved beyond the two numbers.

## Verdict

**PASS.** 0 blocker, 0 major, 2 minor (T-01, T-03-`seen_*`), 6 nit open; 2 minor resolved this round (T-02 and two of the four budget keys). 72 keys read again end-to-end; one string changed and verified; both widened budgets equal the measured need; the two remaining over-budget chips are harmless by the renderer's own arithmetic.

**The es ledger row may ship at `c4c2c502cd3805687bdaeb505b579c629a259bdbee3c453ba72539cca4588ac4`** — provided the context sidecar is committed with it (process note 1), since the budgets this round approves exist only in the working tree.

## Limitations

- LLM semantic judgement, not native-speaker attestation.
- Fit conclusions are arithmetic from the pinned renderer (`hint_row`/`hints_text`/`row_line` constants at `8976104b83b`) and `rich.cells.cell_len`; no live TUI capture, so no pixel-level claim.
- The audit binds the on-disk bytes named above. The es file is untracked and the context file is modified; if either changes before commit, this audit no longer applies to the shipped bytes.
