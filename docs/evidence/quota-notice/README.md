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
    scripts/quota_notice_shot.py out.svg 100x30 <state>   # `narrow` defaults to 60x24
```

The fixtures live in `tests/unit/tui/test_quota_notice.py` (`frame_app`), the
same ones the tests pin, so a frame cannot drift from what the tests assert.

| file | state | reached by |
| --- | --- | --- |
| `before-depleted.*` | origin/main `1846ad08f0`, the branch base — no row | the same session fixture as `depleted` |
| `depleted.*` | a spent DeepSeek balance | fresh zero-balance report in the cache |
| `plan-window.*` | a spent Anthropic plan window | fresh 100 %-used window, resets in 59 min |
| `radient-unverified.*` | Radient free credits behind email verification | pending facts in the process cache; the sentence is the real `recovery_line` output |
| `free-model.*` | a spent account on a stated-free model | listing row priced 0.0/0.0 → NO row |
| `unknown.*` | a stale row (10 min old) | NO row — nothing definite to say |
| `connected.*` | a healthy plan window | NO row |
| `coexist.*` | the quota row AND the credential warning | a render-level fixture: both rows present, neither doubled |
| `narrow.*` | depleted at 60 columns | the appended URL is dropped WHOLE; the sentence clips to its head |

Geometry, from the `.geometry.json` beside each frame (native 8x17 px cells):

```
before-depleted  100x30  welcome region 96x22   screen virtual [98,28] == size, no scrollbar
depleted         100x30  welcome region 96x23   screen virtual [98,28] == size, no scrollbar
narrow           60x24   welcome region 56x10   screen virtual [58,22] == size
```

The pair differs by exactly the one measured row the block draws: the block
stays content-sized (`len(lines) == welcome.size.height`), the app screen
stays non-scrollable, and the tests pin both
(`test_the_splash_shows_the_row_and_adds_exactly_one_measured_row`). The rest
of the frames pin the absent cases (free model, stale row, healthy account)
and the narrow-width degradation.

Copy rules the frames show: the sentence is the verdict's verbatim body
(newlines folded for the single row); the first `open_url` URL is appended
only when the sentence does not already carry it, and is dropped WHOLE — never
half-printed — when the row cannot hold it (the `/login <provider>` rule).
The Radient sentence is two lines and longer than any sensible terminal row,
so that frame shows what a one-row surface can: the head of the claim, an
ellipsis, and no half-printed URL. If that trade wants revisiting, the design
round is the place — the frame is the evidence for it.

Both the row and its position are covered by tests in
`tests/unit/tui/test_quota_notice.py`: the cached-only property (a real
controller seeded through a real fetch, then the fetch layer armed as a
recorder), the verdict reuse (the verdict is spied; the row must follow it,
including the `unverified` state), the fit/shed/order geometry, and the two
pilots (the measured row delta; the warm worker re-snapshotting the splash).
