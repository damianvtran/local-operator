# Quiet groups on the relay web — the bar, before and after

Slice S4 of the quiet-turn design (`docs/design/quiet-turns.md` §5 + §8): a
client-derived group over consecutive delivery receipts in the phone
transcript — one collapsed `Peer messages · N` bar, expandable to the
receipts with their tool rows, instead of one card per receipt. No wire kind,
no capability flag; the derivation is pinned across clients by
`local_operator/mobile/web/src/lib/quiet-groups.parity.json`.

- **After** = PR #2160's branch, code at `4f7c61f25e` — the bar states.
- **Before** = `origin/main` `941e9d0711` — the same sessions, one row per
  receipt.

## Reproduce

```sh
# after (the PR worktree, deps shared with the primary checkout):
cd <worktree> && PYTHONPATH=. .venv/bin/python scripts/mobile_quietgroups_shot.py <outdir>

# before (a throwaway worktree at origin/main, the same rig scripts copied in,
# its node_modules linked to the primary checkout's tree and its bundle built):
git worktree add --detach <dir> 941e9d0711
cp scripts/mobile_quietgroups_fixture.py scripts/mobile_quietgroups_shot.py <dir>/scripts/
```

One command starts `scripts/mobile_quietgroups_fixture.py` (the real daemon
over synthetic projections — `dial_registrants=False`, per-run password never
printed, HOME/config re-homed) on a free port, taps into each session with
real touches, shoots 390x844 and 360x780 at dpr 2, writes
`<outdir>/quietgroups-mobile-geometry.json`, and tears both processes down
(reaping by exact PID; the Chrome teardown fails closed if a helper survives).

## States (`after/`; the same sessions in `before/`)

| frame | session | reads (after) | reads (before) |
|---|---|---|---|
| collapsed-minimum | Quiet group collapsed | `Peer messages · 3+` — first paint, older rows unproven | four rows, one per receipt |
| collapsed-resolved | Quiet group resolved | `Peer messages · 2` — after a real drag to the top; the empty history answer settles the minimum | four rows |
| expanded | Quiet group expanded | the bar pressed: three receipts + the `bash` row | four rows |
| mixed | Quiet group mixed | `Messages · 2` (a peer + a wake receipt) | two rows |
| wake-only | Quiet group wake only | `Wake messages · 2` | two rows |
| single-receipt | Quiet group single | one receipt is not a group — the ordinary card | the same card |

Light set (`light/`): `collapsed-minimum` + `expanded` at 390, captured by
the design round (design r1) — canvas `#f5f0e6`, label 12px, bar measured
44px; the designer's own `light-report.json` rides beside the frames.

## The retired window-cut state (measured, PR #2160)

A seventh state — a group cut by the client's window with the count stated as
a minimum over LOADED-but-off-window rows — was written and then retired: the
first capture on the folded tree showed it cannot be held still. Since #2113
the transcript auto-grows the window the moment the top edge comes into reach
(`transcript.tsx` `onScroll`: `hiddenCount > 0 → setWindowSize(n + PAGE)`) and
renders no "show N more loaded" control, so the growth dissolves the cut
before any stable scroll position can show it. The only stable minimum is the
hasMore form, `collapsed-minimum`.

## Isolation

One throwaway headless Chrome per run (its own `mktemp` profile,
`--use-mock-keychain`, zero-survivor teardown — `scripts/mobile_overflow_capture.py`),
the fixture on an ephemeral port, both reaped by exact PID. Nothing here
touches a live daemon or the operator's sessions.
