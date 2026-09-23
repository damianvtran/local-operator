# Mobile session list — stable ordering on the shared catalogue key, and shared pins

## Defect

Reported: *"significant jitter in the mobile session list rows where they keep
switching based on activity."*

The phone's home list moved under the reader as sessions streamed. The cause is
that the mobile daemon re-derived its own order key from whatever live state the
merge happened to hold, while the terminal sidebar and the desktop app both sort
on the shared catalogue key (`session.catalog.CatalogEntry.rank`). Three
concrete instabilities fell out of that second key:

1. **Section membership diverged.** The phone set `section` from "a live
   `SessionEntry` exists"; the shared rule is `pending or unseen or live_state`.
   A durable-only conversation that finished while nobody was watching carries
   an unseen receipt, so it ranks tier 1 in the catalogue and is **Active** on
   the terminal and the desktop — and landed under **Previous** on the phone.
2. **A live session tied at birth zero.** `created_at` fell back to `0.0` until
   the 1 s durable refresh resolved a birth, so a just-started conversation
   ranked at the BOTTOM of its tier and then JUMPED to the top the moment its
   real date landed.
3. **`busy` was the projection's `streaming`, not the record's own flag** — a
   different fact, and one the other surfaces do not rank on.

## Fix

`_rank_row` (`local_operator/mobile/daemon.py`) builds the same `SessionRow` the
catalogue's `decorate_rows` builds, hands it the same attention state, and asks
`catalog.entry_for(...)` for `(rank, active)`. The list sorts on that rank and
takes its `active`/`previous` partition from the same call, so the three
surfaces cannot disagree about a row's tier, its section, or its place. An
unresolved birth falls back to the runtime's own `started_at` rather than `0.0`,
so a row holds its place across the durable refresh.

Pins ride the **shared durable store** (`local_operator.tui.sidebar_pins`,
`sidebar-pins.json`) through a new `POST /api/sessions/{id}/pin` — the same file
the TUI's `F10` and the desktop app's pin action read and write. The list gains
a `★ Pinned` section (the sidebar's section order; a pin is a display-only lift
and never reorders the ranking) with a long-press action sheet on each row, and
the session view header carries a discoverable ☆/★ control.

## Evidence

`order_fixture.py` is the harness: the real mobile bundle and API (`MobileDaemon`
+ `build_app`), synthetic sessions, no runtime scanner or registrant sockets. It
adds the shape the bug is about — a durable-only conversation with an unseen
receipt and one without a live runtime at all.

Run (isolated `HOME`/config via `scripts.probe_isolation`):

```
cd <worktree> && PYTHONPATH=. .venv/bin/python docs/evidence/mobile-session-order-pins/order_fixture.py 4200
```

### The divergence, before and after

`order-before.txt` / `order-after.txt` are the real `GET /api/sessions` rows.
The only difference between the two runs is the daemon change; same fixture, same
dates, same data.

| session_id | before (origin/main) | after |
|---|---|---|
| done-new | active, unseen | active, unseen |
| done-old | active, unseen | active, unseen |
| **cold-unread** | **previous, unseen** ← wrong section | **active, unseen** |
| busy-a | active | active |
| cold-read | previous | previous |

`order-after.txt` also records `POST /fixture/tick` five times: the order does
not move on activity.

### Frames

`before-list.png` and `after-list.png` are the same fixture list in the real
bundle at the phone viewport. Before: *"Finished while away"* (unread, no live
runtime) sits stranded under **Previous Sessions** below the working session.
After: it leads **Active Sessions**, where the terminal and the desktop put it.

`after-pinned-section.png` shows the `★ Pinned` section: a row pinned through
`POST /api/sessions/{id}/pin` is lifted out of Active into its own section with
the `★` mark, and a `pinned: true` appears on its frame row.

`after-session-view-pin.png` shows the session view header's ☆/★ control (here
★, accent ink) — the discoverable half of the pin gesture.

### The pin is the shared pin

A pin written by the phone lands in `sidebar-pins.json` — the file the terminal
reads with its own reader. Verified directly:

```
$ cat <sandbox>/config/sidebar-pins.json
["done-new"]
```

where `<sandbox>` is the fixture's isolated config root (the pin was set from
the session view header's ★ control).

## Tests

- `tests/unit/mobile/test_relay_perf.py`:
  - `test_a_live_session_does_not_jump_when_its_birth_resolves` — the birth-zero
    jump, driven through the real merge.
  - `test_section_membership_is_the_shared_active_rule` — the section divergence,
    asserted against `entry_for` too so it is about *agreement*.
  - `test_a_durable_unread_row_sorts_where_the_catalogue_puts_it` — the merged
    order equals the shared ranking of the same rows.
  - `test_a_phone_woken_session_is_active_before_its_record_arrives` — the
    provisional wake window does not flash under Previous.
  - `test_set_pins_does_not_touch_the_event_loop` — the pin write leaves the SSE
    wake to the route, on the loop.
  - `test_pin_route_writes_the_shared_store_and_the_frame_carries_it` — the route
    round-trips, and the pin is read back with the TERMINAL's `read_pins`.
  - `test_pin_route_refuses_an_unknown_session_and_a_non_boolean` — 404 for an
    unknown id, 422 for a truthy non-boolean.
- `local_operator/mobile/web/src/session-list.pinned.test.tsx` — the `★ Pinned`
  section, the pinned mark, the long-press opening the sheet, a scroll cancelling
  it (with an unmoved-press control so the cancel test cannot pass vacuously),
  and that a long-press does not navigate.
- `local_operator/mobile/web/src/session-view.seen.test.tsx` — the session view
  retains the list stream, so the header pin can receive the authoritative
  repaint.

## Review round 1

Agent review posted five findings; all addressed in this PR:

- **MAJOR 1** — the phone-woken window (`provisional_active`) was dropped, so a
  woken row flashed under Previous. Ranked as a live `idle` row instead; tested.
- **MAJOR 2** — `set_pins` woke the SSE stream from a worker thread. The notify
  moved to the route, on the loop; tested by monkeypatching the notifier.
- **MINOR 1** — the session view read the list store without retaining its
  stream. It retains it now; tested.
- **MINOR 2** — the long-press timer had no unmount cleanup. Added.
- **MINOR 3** — parity prose over-reached on the wake band. Narrowed here and in
  `docs/mobile.md` / `docs/SESSION_SIDEBAR.md`.
- **NIT 1** — `suppressClick` could latch. A new press clears it.

## Review round 2

Verdict **clean (terminal)** — no BLOCKER, no MAJOR. Four MINOR and one NIT, all
addressed:

- **M2-1** — the narrowed prose still over-claimed tier/section agreement. Both
  docs now name the two real asymmetries (the wake band, and the phone-woken
  window) instead of one.
- **M2-2** — "or a start" was wrong; only the wake path marks
  `provisional_active`. Prose corrected in `_rank_row`, the merge comment, and
  the test docstring.
- **M2-3** — the "free" refcount claim was wrong (React destroys before it
  creates, so a list↔session navigation reconnects the SSE). The comment now
  says so with the measurement.
- **M2-4** — the route half of MAJOR 2 was untested. Added
  `test_pin_route_wakes_the_list_stream`, mutation-checked (deleting the route's
  `notify_list_changed` fails it).
- **N2-1** — the stubs' returned unsubscribe is a no-op, so the effect's unmount
  path is hit but not observed. Left as-is deliberately: the neighbouring
  `retainProjectionStream` stubs in these same files do the same, and deepening
  only this one would make the two stream mocks inconsistent for a nit. A
  comment now records the choice at each stub.
