# Urdu audit — `tui.notifications`, round 2 (`ntt2`) — DELTA RE-AUDIT

Auditor: independent `translation-auditor` (lopdev), session 6cd869512279; actual serving model
`radient/auto` (declared primary `anthropic/claude-opus-5-5`, moved by provider-quota
[model switch] notices through `anthropic/claude-sonnet-5-5` and `openai/gpt-6.1-sol` in the
previous round of this same session). I authored none of the English or Urdu copy and edited
nothing: the audited bytes were copied to scratch first, and every judgement below is made
against that copy.

**Audit id:** `ntt2` · **locale:** `ur` · **namespace:** `tui.notifications` · **date:** 2026-10-25
**Keys read:** 72 of 72 English values, 72 of 72 Urdu values, 72 of 72 context entries. The four
changed keys were examined key by key; the other 68 are proven byte-identical to round 1.

**Verdict: PASSED — 0 blocker, 0 major, 0 minor, 0 nit.**

The round-1 major (NT-UR-1, the `back` arrow) and minor (NT-UR-2, `hint_undo`) are fixed and
verified **through the real paint path**, the missing terminal `۔` nit (NT-UR-5) is applied, and
the render-layer defect I reported in round 1 as code-side is fixed at commit `8976104b83b` —
measured, not assumed.

**The ur ledger row may ship at this hash: `07290844e42e8c147a1f342d359d1569446c8634660da62cabb5f479be63d52a`** — the on-disk, currently untracked
`local_operator/i18n/catalogues/ur/tui.notifications.json`.

## Inputs (pinned before any judgement)

| input | value |
|---|---|
| audited target (copied to scratch first) | `07290844e42e8c147a1f342d359d1569446c8634660da62cabb5f479be63d52a`; the worktree file hashes identically |
| round-1 target | `a0cd78685b89231a7c282cf7f7b68020cbd4a8e29593e84c68e63105a9a96ea0` |
| English source / sidecar (unchanged since round 1) | `b415bc703840afabc1c3ecaeecdad24a84d9a6ab3367833cad4f17f0e0e70b68` / `0a9ee26060dcc5a5dc44c784fcdff2d3b64c6457899d7b3c382c0c7e32d85907` |
| code under test | `8976104b83b1f4af2fa6014ae5c73d9bbcdebc0a` (`feat/notifications-tui`, also the worktree HEAD), extracted with `git archive` so the target and the code are one generation |
| round-1 code, for the before/after control | `d25392f3553c38ce1740c98e53c9c66a700e9e64` |
| English catalogue at the code commit | byte-identical to the pinned English source: **true** |

The ur catalogue is still `??` untracked in the worktree; nothing was staged, stashed, moved or
committed by me. The code extractions live outside the session scratchpad
(`~/workspace/lo-ntt2/`) because two `git archive` trees hit the 256 MiB scratchpad cap mid-round;
that is disclosed in `audit.log`.

## What changed — exactly four keys, nothing else

| key | round 1 | round 2 |
|---|---|---|
| `back` | `← اطلاعات` | `→ اطلاعات` |
| `hint_undo` | `کالعدم` | `بحال` |
| `unread_mark_all` | `… نشان زد کریں` | `… نشان زد کریں۔` |
| `unread_mark_these` | `… نشان زد کریں` | `… نشان زد کریں۔` |

All 72 keys present, none added or removed, every other value byte-identical, and the whole-file
unified diff is exactly those four lines — no formatting, ordering or trailing-byte drift
(`audit.log` §1, verbatim).

## NT-UR-1 (was major) — FIXED, proven through the real paint path

Rendered through the app's own `display_tr` resolver (`notifications_render._t`, active locale
`ur`, run from the extracted commit), `back` paints as:

```
painted (escaped): '\u2067→ اطلاعات\u2069'
painted (marked) : <RLI>→ اطلاعات<PDI>
FriBidi visual   : <RLI>اطلاعات →<PDI>
cells            : 9   (maxCells 24)
```

Inside the isolate the logical first element is placed rightmost, so the arrow sits at the
**right end of the word and points right** — toward where the reader came from, i.e. back — which
is what RTL requires. It stays `→` because U+2190/U+2192 are `mirrored=0` in the UCD; nothing
flips it automatically.

The decisive before/after uses **the same audited bytes** in both renderer generations, so the
renderer is the only variable (`audit.log` §4, §5):

| state | composed detail header (width 80), first line | visual |
|---|---|---|
| round-1 code + round-1 bytes (the true pre-fix page) | ` ← اطلاعات … ▁▃ normal · اپ ڈیٹ · 3 / 41 ` | `41 / 3 · اپ ڈیٹ · normal ▁▃ … اطلاعات ←` |
| round-1 code + round-2 bytes | ` → اطلاعات … ▁▃ normal · اپ ڈیٹ · 3 / 41 ` | `41 / 3 · اپ ڈیٹ · normal ▁▃ … اطلاعات →` |
| round-2 code + round-2 bytes (the shipped state) | ` <RLI>→ اطلاعات<PDI> … ▁▃ · <RLI>اپ ڈیٹ<PDI> · <RLI>3 / 41<PDI> ` | `<RLI>اطلاعات →<PDI> … ▁▃ · <RLI>اپ ڈیٹ<PDI> · <RLI>41 / 3<PDI>` |

Two things are visible in the first row: the arrow pointing left at the right end of the
auto-detected RTL line (the round-1 defect), and the raw Latin data token `normal` sitting inside
every locale's header. Both are gone in the shipped state — the header now carries the localized
`▁▃` bars and each translated segment is isolated.

## NT-UR-2 (was minor) — FIXED

`hint_undo` is now `بحال` (4 cells, cap 8). It matches the namespace's own undo receipt
(`restored` = `بحال: {count}۔`) and the desktop's `toast.undone` (`بحال ہو گیا`); `کالعدم` is gone: 0 occurrences across the Urdu catalogue directory at
the audited bytes (it was 1 — this key — in round 1), while `بحال` already renders other states
elsewhere (`tui.projects_edit.json`). Residual, recorded rather than re-raised: the desktop's *button*
for the same action is still `واپس کریں`, which does not fit the TUI's 8-cell hint budget (9 cells)
and whose short form `واپس` would collide with `hint_back` on the same footer row. The wave should
record that budget-forced difference in `i18n/style-guides/ur.md`.

## NT-UR-5 (was nit) — APPLIED

Both `/unread read` receipt lines now end with `۔`, matching every other receipt in the namespace
(`dismissed`, `marked_read`, `marked_unread`, `restored`, `nothing_unread`, `stale_skipped`,
`asks_stay`).

## The render-layer fix — per-token islands, measured

Round 1 reported that five keys carry a leading `/` literal inside RTL sentences and that
per-string isolation alone reorders it. The commit declares those literals to `display_tr`
(`model.SHELL_COMMANDS = ("/notifications", "/unread")`,
`SHELL_COMMAND_TEMPLATES = ("/unread read",)`), so each token becomes its own LTR island. Painted
strings and their FriBidi visuals (`audit.log` §3), with the "control" being the same painted
string reduced to per-string isolation only:

| key | feasible token | shipped paint | control (no island) |
|---|---|---|---|
| `pointer_unread` | `/unread` | `<RLI>… <LRI>/unread<PDI> …<PDI>` → token `/unread` contiguous | `unread/` |
| `refused_argument` | `/notifications` | `<RLI><LRI>/notifications<PDI> …<PDI>` → `/notifications` | `notifications/` |
| `banner_digest_body` | `/notifications` | token intact | `notifications/` |
| `unread_mark_all` | `/unread read` | `<RLI><LRI>/unread read<PDI> …<PDI>` → one island, both words | `unread read/` |
| `unread_mark_these` | `/unread read` | token intact, one island | `unread read/` |

Island accounting on the painted forms: exactly one outer `RLI … PDI` per string and one inner
`LRI … PDI` in each of the five command-bearing keys, none elsewhere in the spot-rendered set. The
copy itself still contains **no** manual bidi controls (verified byte-wise), so nothing is
double-applied.

The projection is correctly scoped to Urdu: the same seven keys painted for `en`, `es`, `fr`,
`hi`, `ru`, `vi` and `zh-CN` are byte-equal to each locale's own catalogue with zero control
characters (`audit.log` §6).

## Mechanical re-checks on the delta

- Argument tuples unchanged: `back`/`hint_undo`/`unread_mark_these` take none, `unread_mark_all`
  takes `{count}` (simple) — identical to English, and all 72 keys pass the repo's argument
  parity rule.
- No plurals in the changed keys; all seven plural messages still render their count at
  0/1/2/5/21/1000.
- `check_catalogues()` over the extracted commit tree with the audited ur bytes: **0 problems**
  for this namespace and 0 overall.
- File hygiene: NFC clean, no manual bidi controls, no ZWNJ/ZWJ/NBSP, no non-ASCII digits, no
  doubled `۔`.
- Widths: `back` 9/24, `hint_undo` 4/8, `unread_mark_these` 44 (uncapped), `unread_mark_all` 52
  (uncapped, +1 for the new `۔` at a 4-digit count). Every declared budget holds.

## A correction to my own round-1 reasoning

In round 1 I wrote that the glyph fix had to land *with* the isolating change, because "while the
page still paints LTR the arrow renders leftmost". Measured now, that caveat is wrong: the
composed detail header begins with the Urdu label, so the whole physical line is auto-detected
RTL even under the round-1 renderer, and the arrow lands rightmost — where `→` reads correctly and
`←` did not (rows 1 and 2 of the before/after table). The caveat would only hold in a terminal that
forces an LTR paragraph base, which I did not measure. The finding and the fix stand; the
sequencing reason I gave for it does not, and I would rather withdraw it than let it harden into
guidance. The isolating change is still the right shape for a different, measured reason: it stops
the header's segments from being reordered as one RTL paragraph and keeps `/command` tokens intact.

## Nits left as they were (no change requested, recorded for the guide)

- `hint_clear` `صاف` — bare adjective where sibling hints name the action; `صاف کریں` is exactly
  8 cells (cap 8, zero headroom).
- Four grammatical forms of "unread" coexist (`نہ پڑھی گئی`, `نہ پڑھی`, `نہ پڑھا`, `نہ پڑھا ہوا`);
  same term, so not a glossary violation — record the convention so a later wave does not add a
  fifth.

## Limitations

- LLM semantic judgement, not native-speaker attestation.
- Bidi conclusions come from GNU FriBidi 1.0.16 run over the app's own painted strings — not from
  a rendered terminal frame. No claim about glyph shaping, terminal fonts, or a specific
  terminal's UAX#9 implementation.
- Widths are `rich.cells.cell_len` on rendered strings.
- The audited bytes are untracked working-tree content: this binds a hash, not a commit.
- The audit ran across a provider fallback chain; `radient/auto` served the final judgement.
