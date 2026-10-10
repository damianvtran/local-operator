# The pre-emptive no-quota notice on the empty splash

One advisory status row beside the credential warning, shown while the session
is still empty: the session's own provider has DEFINITE, FRESH evidence that no
account behind the selected model can send, and the row says where the user
fixes it. This is the TUI half of the design whose other surfaces are the
desktop route (`GET /v1/desktop/quota-notice`, #2128 and #2135) and the
desktop line (local-operator-ui #943). Every sentence is the shared verdict's
(`providers/quota_notice.evaluate_quota_notice`); this surface reads it
CACHED-ONLY — the usage row the 60 s warmer maintains — so a splash paint
never crosses the network.

Reproduce any frame with:

```sh
env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
    scripts/quota_notice_shot.py out.svg 100x30 <state>
    # sizes default per state: `narrow` 60x24, `radient-unverified-wide` 200x30
```

The fixtures live in `tests/unit/tui/test_quota_notice.py` (`frame_app`), the
same ones the tests pin, so a frame cannot drift from what the tests assert.

| file | state | reached by |
| --- | --- | --- |
| `before-depleted.*` | origin/main `1846ad08f0`, the branch base — no row | the same session fixture as `depleted` |
| `depleted.*` | a spent DeepSeek balance | fresh zero-balance report in the cache |
| `plan-window.*` | a spent Anthropic plan window | fresh 100 %-used window, resets in 59 min |
| `radient-unverified.*` | Radient free credits behind email verification | pending facts in the process cache; the sentence is the real `recovery_line` output |
| `radient-unverified-wide.*` | the same state at 200 columns | content widths 181–227, where the in-sentence URL used to be cut mid-address; now the tail drops WHOLE at `…verification email…` |
| `free-model.*` | a spent account on a stated-free model | listing row priced 0.0/0.0 → NO row |
| `unknown.*` | a stale row (10 min old) | NO row — nothing definite to say |
| `connected.*` | a healthy plan window | NO row |
| `coexist.*` | the quota row AND the credential warning | a render-level fixture: both rows present, neither doubled |
| `narrow.*` | depleted at 60 columns | the appended URL is dropped WHOLE; the sentence clips to its head |

Geometry, from the `.geometry.json` beside each frame (native 8x17 px cells;
the SVG text runs carry the same numbers at 8 px per cell):

```
before-depleted  100x30  welcome region 96x22   screen virtual [98,28] == size, no scrollbar
depleted         100x30  welcome region 96x23   screen virtual [98,28] == size, no scrollbar
narrow           60x24   welcome region 56x10   screen virtual [58,22] == size
radient-unverified-wide  200x30  welcome region 196x23  screen virtual [198,28] == size, no scrollbar
```

The pair differs by exactly the one measured row the block draws — and by
nothing horizontally: the status/hint stack holds at pad 35 cells (x=296 px)
with the row on and off, because the notice row centres on its OWN width
(pad 0 here, x=16 px) instead of joining the shared pad, which would collapse
to zero the moment the row lands (design round 1, D1). The block stays
content-sized (`len(lines) == welcome.size.height`), the app screen stays
non-scrollable, and the tests pin both, plus the pad invariance
(`test_the_splash_shows_the_row_and_adds_exactly_one_measured_row`,
`test_the_notice_row_never_moves_the_shared_pad`). The rest of the frames pin
the absent cases (free model, stale row, healthy account) and the
narrow-width degradation.

Copy rules the frames show: the sentence is the verdict's verbatim body
(newlines folded for the single row); the first `open_url` URL is appended
only when the sentence does not already carry it, and is dropped WHOLE — never
half-printed — when the row cannot hold it (the `/login <provider>` rule).
A URL INSIDE a sentence (Radient's) gets the same whole-drop: when the cut
would land in the address, the tail goes at the URL's start — the introducer
(`…, or open`) with it — and is swept across every width by
`test_no_url_is_half_printed_at_any_width`. `radient-unverified-wide` shows the
band (content widths 181–227) where a partial address used to print; the
100-column frame shows the other end of the ladder: the head of the claim, an
ellipsis, and no URL at all.

Both the row and its position are covered by tests in
`tests/unit/tui/test_quota_notice.py`: the cached-only property (a real
controller seeded through a real fetch, then the fetch layer armed as a
recorder — plus the review-round-1 cold-cache case where a stored credential
must not tempt a probe), the verdict reuse (the verdict is spied; the row must
follow it, including the `unverified` state), the fit/shed/order geometry,
the pad invariance and the URL sweep, and the two pilots (the measured row
delta; the warm worker re-snapshotting the splash).
