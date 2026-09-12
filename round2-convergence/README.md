# PR #1018 — round 2 convergence evidence

Head `afbf1cc66` (convergence) vs `2c43bb024` (the round-2 reviewed head).

Same capture discipline as `round1-remediation/`: headless Chrome
(`--headless=new`), `Emulation.setDeviceMetricsOverride` at 390x844 and
360x780, `dpr=2`, `mobile=true`, both sides in the SAME mode. Before-frames
come from a throwaway worktree detached at `2c43bb024` with its own
`pnpm build`; the only file copied into it is the new fixture, because the
compounding scenario D4 needs did not exist at that head and a before/after
pair has to be shot by one instrument. Components on the before side are
untouched `2c43bb024`.

## D4 — the dim was taxing the glyph U5 exists to surface

`*-failures-pending.png`. The compounding case: failed agents AND a pending
request at once, so the roster is held shut at the moment a failed fan-out
matters most. No pre-existing fixture had both, which is why round 1 could not
have caught it.

Contrast measured from the PAINTED pixel — the text colour composited over its
real background at the effective opacity, not the declared token, because
opacity composites and the token says nothing about what lands on screen.

| | before | after |
|---|---|---|
| `· 3 failed` painted | `rgb(152,84,78)` @ opacity 0.6 | `rgb(239,128,120)` @ opacity 1 |
| contrast on `rgb(22,19,14)` | **3.29:1** | **7.08:1** |

The label and running count still dim, so the row stays visibly inert; only the
danger span is exempt. The dim could not be undone by a descendant, so the
count is a SIBLING of the dimmed span rather than a child of it.

## U8 — the held state now states its rule

Same frames. `· answer first` in the header, plus a `title`. Measured cost:
zero rows — the header is 44px at 390x844, 360x780 and 320x568, `wrapped:
false`, `overflowing: false` at all three. A wrap would have cost exactly the
row the panel budget exists to protect.

## D5 — the two cards' footers now share one rhythm

`*-ask-free.png` against `*-approval.png`, measured at 390x844:

| variant | scroller → first control | bottom padding |
|---|---|---|
| approval (unchanged) | 60px | 11px |
| secret, before | 8.0px | 32.4px |
| secret, after | 33.4px | **11px** |

Normalised to the approval's 11px bottom padding. The reassurance line moved
ABOVE the input rather than a hairline being added beside it: a second divider
idiom next to the approval's `remember` row is the competing-patterns bug, and
a warning about a credential is one to read before pasting, not after.

## R1/U7 + R2 — the guard can now observe the defect it is named for

`head-reachability.txt` (62/62, exit 0) and `mutant-dvh-reachability.txt`
(32/62, exit 1). The mutant is `afbf1cc66` with `columnCap()` reverted to
`calc(100dvh * fraction)` in a throwaway worktree, rebuilt, served on its own
port.

The point is WHICH assertions fail. Before this change, a fully `dvh`-reverted
build scored `R1 … send reachable` at 44/44px PASS. Now:

```
[FAIL] R1 360x780 kb=336 send reachable (secret variant):
       0/44px visible (height 44px), hit=False — centre lands on None;
       column 444, dvh 780, cap 468px
[FAIL] R1 360x780 kb=300 send reachable (secret variant):
       10/44px visible (height 44px), hit=False — centre lands on None
[FAIL] R2 360x780 kb=336 approve visible on arrival:
       0/44px visible (height 44px), hit=False — centre lands on None
```

12 of the 30 mutant failures are per-control reachability assertions that ALL
passed before this commit. The count rose 44 → 62 because R2's on-arrival
assertions now also run under each keyboard height (R2 finding).

Residual, and stated rather than papered over: at 390x844 kb=260 the mutant
still passes every check (card bottom 563 vs column 584 — the cap is wrong by
less than the slack at that one viewport). The reviewer flagged this near-miss
in round 2; it is narrowed, not eliminated, by adding the pinned on-arrival
passes.
