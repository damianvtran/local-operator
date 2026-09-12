# PR #1018 — review round 3 remediation evidence

Head `62eb50d19` (remediation) vs `afbf1cc66` (reviewed head), with
`2c43bb024` (the head before the hint existed) kept as the "is this new?"
control.

Same capture mode as the previous rounds: headless Chrome (`--headless=new`),
`Emulation.setDeviceMetricsOverride` at `dpr=2`, `mobile=true`, one throwaway
profile per run. Before-frames come from a worktree detached at `afbf1cc66`
with its own `pnpm build` (bundle `index-B1dgRUqV.js`, matching the hash the QA
round reported for that head) — never a `git stash` or a `checkout --` in a
live checkout. Both fixtures run `dial_registrants=False` under an isolated
`HOME`, so the operator's daemon was never touched.

## The fixture: a TWO-DIGIT failure count

The shipped `failures-pending` projection fails 3 of 22 agents, and `· 3` fits
at every width — it cannot show this defect. These frames use a wrapper that
reuses that projection verbatim and only widens the failure count to 12,
keeping the pending card that holds the panels shut. The threshold UX measured
is exactly `· 10 failed`.

## How the wrap was measured, and why the previous round missed it

**Painted line-box count**, from `Range.getClientRects()` over the count span's
own text, grouped by rounded `top` — one entry per line box the range paints
on, so a count broken as `· 12` / `failed` returns 2 and an intact one returns
1.

The previous round reported `wrapped: false` from `scrollWidth > clientWidth`,
which **structurally cannot observe this defect**: a flex row whose children
wrap has `scrollWidth == clientWidth` by construction. That test reads
"no overflow" in every cell below, wrapped or not — it is recorded in the
`naiveOverflow` column of `measurements.json` precisely to show it staying
`false` while the count visibly wraps.

Line count alone is also not sufficient, so each cell additionally samples
**painted danger-red pixels inside the count's own rect**, which is what caught
the paint-order side effect described below.

| viewport | before `afbf1cc66` | after `62eb50d19` |
|---|---|---|
| 390x844 | 1 line, h=14px, 509 red px | 1 line, h=17.4px, 491 red px |
| 360x780 | **2 lines**, h=33.5px, 493 red px | 1 line, h=17.4px, 491 red px |
| 320x568 | **2 lines**, h=33.5px, **41 red px** (wrapped word behind the card) | 1 line, h=17.4px, 485 red px |

Row height is **44px in every cell on both sides** — the wrap never stole a
row, which is why all three streams filed this as MINOR.

`before-*` / `after-*-failures-pending-2digit.png` are the pairs;
`compare-headers-2digit.png` crops all six header rows into one sheet.

## The paint-order side effect, found by the pixel sample

Making the header a flex row **blockifies its children**, which moves them out
of the inline paint phase. Below the viewport ladder — where this panel's
container collapses to ~12px and the pending card, a LATER sibling, overlaps
the row — the count then painted *under* the card's background: **0** danger-red
pixels at 320x568 while keeping its exact box (`x=151.6 y=115.6 w=79.5
h=17.4`). Isolated by setting `position: relative` on the count alone, which
restored 485 red px with a byte-identical rect, confirming paint order rather
than layout.

`relative` on the header row is the shipped remedy: it restores what inline
content had for free, without a z-index. At 390x844 and 360x780 the class is
inert (491 red px before and after).

A contradiction worth recording, since it produced plausible wrong numbers
first: an early probe read 485 red px at 320 and a later one read 0 on the same
build. The first had hidden the pending card in an earlier viewport cell and
then navigated to the same hash route — a same-document navigation, so the
inline `visibility: hidden` survived into every later cell. That is C1's own
mechanism, hit by accident. Every number above was re-taken with **one viewport
per browser session** and no mutation before the baseline.

## The label now truncates, which is what made the count safe

`compare-longlabel.png`. With a long roster label forced in:

| viewport | before | after |
|---|---|---|
| 390x844 | label wraps to 3 lines, row 44px | label ellipsizes (`scrollW 390 > clientW 192`), count 1 line, row 44px |
| 360x780 | label wraps, **count wraps to 2 lines**, row **58.5px** | label ellipsizes (390 > 163), count 1 line, row 44px |
| 320x568 | label wraps to 3 lines, row **58.5px** | label ellipsizes (390 > 125), count 1 line, row 44px |

`textOverflow: ellipsis` is computed and the `…` is visible in the frames. The
`shrink-0` on `· answer first` in `disclosure.tsx` is unchanged — its comment
said a long label truncates first, and that is now true rather than assumed.

## Contrast: the count is still undimmed

The count keeps `text-danger` at full opacity as a SIBLING of the dimmed label
span — `rgb(239, 128, 120)`, `opacity: 1`, the 7.08:1 D4 restored. Verified on
both held headers at all three viewports; nothing in this commit dims, hides or
truncates the count itself.

## Harness (C1)

`scripts/mobile_reachability_check.py` R2 block: the "unpin" re-`goto` never
unpinned, because the app is a hash-router SPA and the pin is inline style on a
live node. Now cleared for real and **asserted** (`clientH == innerH`), since
R1's first act re-pins and would hide a no-op reset. The run goes 62/62 → 64/64
(the two new assertions), exit 0, on the PR's own fixture.
