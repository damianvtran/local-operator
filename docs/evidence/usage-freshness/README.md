# Evidence: `/usage` header freshness

Captured on macOS (Darwin 25.6.0, arm64) on 2026-09-02, against
`dev-usage-freshness` at 100x30, with a pinned clock so before/after frames
differ only by the code under test.

| File | What it shows |
|---|---|
| `before.svg` | `origin/main`: the title reads `Usage  2h ago` above five Anthropic accounts refreshed 1.8 minutes earlier |
| `before-stuck-account.svg` | The same frame scrolled to the bottom: `kimi cred:8 · last known 2h ago` — the one account the header was reporting |
| `after.svg` | This branch, same state: the title reads `Usage  1m ago` |
| `after-stuck-account.svg` | The same, scrolled: the header is fresh, carries `· 1 stale`, and the stuck block says `last known 2h ago` in amber with a dimmed status dot |
| `before-short-pane.svg` / `after-short-pane.svg` | The 100x18 frame from design finding D1, where the row budget squeezes the note out: on `main` nothing marks the stale block; here the pinned `· 1 stale` and the dimmed dot survive |
| `*.png` | The same frames rendered, so they are viewable inline |
| `usage_shot.py` | Reproduces every frame (run from the worktree root) |
| `rasterise.py` | SVG → PNG without collapsing the spacing (see below) |
| `verify_refresh.py` | Drives the real `r` key against a controller with one permanently-stuck account, printing the header's `fetched_ms` after each press |

## The reported state, from the operator's own cache

Read out of a copy of `~/.local-operator/usage_cache.db` (copied before opening;
the live file is 0600 + WAL and was never written to). 21 account rows, of which
the visible providers were:

```
key                                identity                    age_min  fails  probe_in_min
kimi:39624648dfb455c7              cred:8                        177.2      1          -9.4
openai:87fdefa5f58038e1            damian@gominerva.com            9.6      0          None
openrouter:98118c395db3e721                                       9.6      0          None
zai:28758706a133a255               damian@gominerva.com            9.6      0          None
xai:c1cb41263c3a4df8               damianvtran@gmail.com           9.6      0          None
anthropic:cb8e36a1a3f6c56b         damian@gominerva.com            4.3      0          None
anthropic:cb8e36a1a3f6c56b         damian@radienthq.com            4.3      0          None
anthropic:cb8e36a1a3f6c56b         damian@pergamonhq.com           4.3      0          None
anthropic:cb8e36a1a3f6c56b         damianvtran@gmail.com           4.3      0          None
anthropic:cb8e36a1a3f6c56b         damian@local-operator.com       4.3      0          None
```

Every account but one had been confirmed within ten minutes. The header said
`2h ago`, because it took the `min` of `fetched_at` across the set and
`kimi cred:8` had been serving last-good from its per-account backoff for 169
minutes.

## The reproduction

`usage_shot.py` builds exactly that set — five fresh Anthropic logins plus one
Kimi account with a 169-minute-old stamp and `consecutive_failures=1` — and
computes the header through the app's own `_usage_data_fetched_ms`:

```
                     origin/main            this branch
title age shown  :   Usage  2h ago          Usage  1m ago
newest report    :   1.8 min old            1.8 min old
stalest report   :   169.0 min old          169.0 min old
account note     :   last known 2h ago      last known 2h ago
```

The per-account note is unchanged, which is the point: the individual staleness
was always reported correctly and specifically, on the block it belongs to.

## Why `r` looked dead

A forced refresh does re-probe — `_reset_account_for_force` clears the streak —
but when the probe misses again `_mark_account_failure` returns the PREVIOUS
report object, keeping its old `fetched_at`. A header taken from the oldest
stamp therefore returns the same reading no matter how many times `r` is
pressed. `verify_refresh.py` presses the real key three times against a
controller that behaves that way:

```
                    origin/main                        this branch
after open      :   'Usage  2h ago'  fetched_ms=…860000    'Usage  just now'  fetched_ms=…000000
after r #1      :   'Usage  2h ago'  fetched_ms=…860000    'Usage  just now'  fetched_ms=…300000
after r #2      :   'Usage  2h ago'  fetched_ms=…860000    'Usage  just now'  fetched_ms=…600000
after r #3      :   'Usage  3h ago'  fetched_ms=…860000    'Usage  just now'  fetched_ms=…900000
stuck account   :   'last known 3h ago'                    'last known 3h ago'
stuck fetched_at:   169 min old (unchanged)                169 min old (unchanged)
```

On `origin/main` `fetched_ms` is byte-identical across three forced refreshes —
the panel could not report the work it was doing. On this branch it advances
with each press while the stuck account keeps its own honest note.

## Geometry behind the frames

```
virtual/actual  : Size(width=98, height=28) / Size(width=98, height=28)
screen scrollbar: False
```

Virtual size equals actual size and no screen scrollbar appeared, so the overlay
did not make the screen scrollable (the failure mode AGENTS.md records from the
usage-card round). Five consecutive captures are byte-identical apart from the
boot splash's animating glyph colour, which drifts the same way on `origin/main`
— the panel itself is settled, with no post-paint reflow.

## Round 1 review follow-ups

The first round of this PR moved the header to the newest stamp, which made the
per-account note load-bearing: it is what stops a block the title does not
describe from being read at the title's age. Review found three ways that
marking could be absent, and the design round found two more where it was
present but not legible. All are closed, and re-verified here:

```
R1  a stale row with consecutive_failures == 0 renders unmarked
      note rendered : 'last known 3h ago'          (was '')
R2  a never-successful account's failed-probe stamp becomes the header
      header WITHOUT stub : 2h ago
      header WITH    stub : 2h ago                 (was 'just now')
D1  the note dropped by compaction while its stale meter survives
      misread frames: 0 / 90 sizes swept           (was 18 / 40)
```

`verify_refresh.py` still shows the originally reported bug fixed, now with the
title naming its own exception:

```
after open      : 'Usage  just now  · 1 stale'  fetched_ms=…000000
after r #1      : 'Usage  just now  · 1 stale'  fetched_ms=…300000
after r #2      : 'Usage  just now  · 1 stale'  fetched_ms=…600000
```

**On the PNGs.** The first round's PNGs collapsed every two-space separator
(`Usage1m ago`, `rrefresh`), which is exactly the property a design reviewer
inspects. The cause is not a missing font: Textual writes each styled run as its
own `<text>` node and encodes the gaps as leading `&#160;`, and librsvg applies
XML whitespace collapsing unless the element opts out, then stretches what
remains to satisfy `textLength`. `rasterise.py` adds `xml:space="preserve"` to a
COPY before converting, so the committed SVGs stay byte-identical to Textual's
export and remain authoritative.
