# Design: an idle parent that owns running subagents

Status: proposed. Base: `origin/main` @ `e7be1fe1`; all `file:line` are against that commit. The
shape is already decided (the catalogue derives a distinct state when the parent's own turn is not
busy and N>=1 children run; the count travels as data); this file records WHY.

## The defect

A session whose own turn ended while it still owns running subagents renders as **no activity**
everywhere: TUI `●`/"Ready", Electron `Circle`/"Ready", phone nothing. The cause is a measured fix
one layer down: `ServingSessionHandle.is_conversationally_active` (`session/runtime/serving.py:1380-1421`)
EXCLUDES subagents from the busy bit (publishing residency there made 8/8 live sessions claim to be
working), so `decorate_rows` (`session/catalog.py:678-690`) derives `live_state = "idle"` and
`status_code` (`:186-210`), `status` (`:286-370`) and `row_state_mark`
(`tui/widgets/session_picker.py:975-1040`) follow. Membership is not the bug —
`CatalogEntry.active` (`catalog.py:181-183`) already files the row under "Active"; the words and the
glyph are.

The fact is already in the readers' hands: `SessionRecord.subagents_running` / `subagents_queued`
(`session/runtime/types.py:1063-1075`), written by `set_subagents`
(`session/runtime/server.py:3645-3662`) from `subagent_counts` (`serving.py:6146-6179`; `None` =
"this build does not report", never 0). Both row builders — `decorate_rows` and the feed's `_row_for`
(`server/utils/desktop_feed.py:1374-1434`) — already hold that record, so nothing here adds a read.

| surface | today | after |
| --- | --- | --- |
| code | `idle` / `attached` | **`delegating`** |
| label | "Ready" | **"Delegating (2 subagents)"** / "Delegating (1 subagent)" |
| TUI glyph | `●` | **`⇉`**, ink `accent`, static |
| desktop icon | `Circle` | **`CornerDownRight`**, ink `text-info`, static |
| phone slot | empty | **static dim ring**; chip stays `N subagent(s)` |

## 1. A new CODE, not a count-only field

The `leaving` precedent (`resume.py:1875-1898`) refused a new `live_state` token and moved that state
into the LABEL, because `live_state` is a token consumers branch on. It cannot be reused, and the
requirement is why: the desktop renderer derives glyph AND ink from the CODE STRING ALONE
(`chat-session-status.tsx:17-60`), so a label-only change is invisible there — `leaving` has that
limitation today (a draining idle row still paints `Circle`). Here the absent mark IS the complaint,
and a count beside an `idle` code is data no surface may draw.

Two ladders are avoided by ONE predicate: the code names the state, the count is spelled inside the
label, and both come from the row (section 4). An unknown code fails visibly (`HelpCircle`,
`chat-session-status.tsx:41-50`), so publisher and renderers ship in lockstep.

## 2. Spelling and the noun

- Code `delegating`, a present participle like `busy`/`leaving`. Rejected: `subagents` (a noun is not
  a state) and `delegated` (reads finished).
- Label `Delegating ({n} subagent{'s' if n != 1 else ''})` — the `Scheduled (2 wakes)` shape — and it
  must not pluralise at 1 (`tests/unit/tui/test_session_sidebar.py:1445`).
- Noun **subagent(s)**: the record's field name, `/info`'s word (`info/render.py:435`), the composer's
  word; the phone's `{n} agent(s)` chip (`mobile/web/src/screens/session-list.tsx:187-193`) is the
  outlier and is renamed in the mobile PR. Counter-argument: "subagent" is longer in a right-cluster
  chip, so a narrow phone truncates the title sooner — geometry is unaffected, and design/UX may
  overrule on a rendered frame.
- One string serves the TUI tooltip, the desktop `title`/`sr-only` (both `row.status.label`) and the
  chip beside the phone's slot.

## 3. Precedence rung

`status_code` (`catalog.py:186-210`), inserted between `attached` and `scheduled` and gated on
`not row.leaving`; `status` (`:286-370`) takes the same rung, with `leaving` above it exactly as it
sits above `busy` today (`:315-316`).

```
pending -> approval | answer     wedged | busy
shows_completion_mark -> complete | error | interrupted
attached -> attached ("Open")
+ delegating -> delegating ("Delegating (N subagents)")   <-- NEW
armed wake -> scheduled          idle -> idle ("Ready" / "Running headless (exec)")
dormant -> dormant               cold receipt -> complete | error | interrupted
                                 else recent
```

| it sits below | why | counter-argument, and why it loses |
| --- | --- | --- |
| `pending`, `wedged`, `busy` | the parent is by construction not busy; a parked gate or a silent runtime is a louder fact | — |
| an unseen completion (`shows_completion_mark`, `:213-282`) | that predicate is THE single arbiter for "mark or live state", read by two callers; extending it suppresses an actionable unread outcome on a row this state declares conversationally idle — precisely when the user CAN read the reply | `busy` does suppress the mark. Loses: the work in flight is not the parent's, whose own lane is free |
| `attached` | D2 (`session_picker.py:997-1005`) — `○` answers "where am I?", and the row the user is sitting in is not the row needing to be told about. The residency family's order (`attached > armed wake > idle > dormant`) is untouched; the new rung is its head | "which is actually working" would prefer `delegating` above `attached`. Loses: that spends the user's own location marker |
| an armed wake | the wake outranks `idle` because `●` is the least informative thing true of a live row (`session_picker.py:985-996`); `⇉` is not that — happening NOW beats happening later. This is the complaint's own case (idle + wake + children currently says "Scheduled") | a forward-looking wake could be argued above work you are not doing. Loses: `⇉` carries work in progress, which is strictly more than `●` |
| `leaving` (a gate, not a rung) | a runtime committed to exiting is a stronger fact | without the gate a leaving row draws `⇉` beside "Leaving…", needing an exception in the glyph->words `ALLOWED` map |

One ladder, three surfaces: the TUI mark, the desktop icon chain and the phone's slot put this below
unread/pending and above presence-adjacent resting states (the phone has no `attached`/wake arms, so
it is a no-op there). Non-goal: the terminal title's separator vocabulary
(`tui/terminal_title.py:10-26`) and the CLI STATE column — `leaving` already refused to teach STATE a
new word, for this reason.

## 4. What carries the count

- `SessionRow` (`resume.py:1821`, live block `:1855-1898`): add `subagents_running: int | None = None`
  and `subagents_queued: int | None = None`, defaulted like `leaving`, so every other construction
  site is unchanged. `decorate_rows` fills them at `catalog.py:708-720` and `_row_for` transcribes the
  same two lines (the hand transcription is already held in parity by
  `tests/unit/server/test_desktop_feed.py`). No new IO: both come off a record the callers already
  read, so the 10 Hz doorbell budget (`desktop_feed.py:110-119`) is untouched.
- ONE predicate, on the ROW, so glyph and words cannot disagree: delegating is `not leaving and
  live_state in {"idle","attached"} and (subagents_running or 0) >= 1`. `status_code`/`status` read it
  and so does `row_state_mark`. The row-level home also avoids a new import edge between
  `session_picker` and `session.catalog`, and makes the pairing structural rather than a promise —
  the reason `shows_completion_mark` exists.
- Wire LIST row: free — `desktop_sessions.py:3284-3290` projects `entry.row._asdict()`. Declare both
  on `server/models/desktop_sessions.py::SessionRow` and on the UI's hand-written mirror
  `SessionCatalogueRow` (`src/shared/desktop-session-contract.ts:9-33`); `None` serialises as JSON
  `null`, still distinct from `0`.
- The `session_status` frame stays a `{code, label, revision}` TRIPLE, and a count change still
  produces a frame: the count lives INSIDE the label, so the dedupe key `(code,label)`-minus-clock
  (`catalog.py:455-481`) changes on every count change — on a row whose rung is `delegating`, `0 -> 2`
  and `2 -> 1` each emit exactly one frame, while the 15 s heartbeat rewrite emits none. A count change
  on a row whose rung is above it (unseen mark, `busy`, `attached`) emits NO frame, because that label
  does not carry the count. Consequence: a client's numeric copy can go stale until the next list read
  — inert today (no surface draws the bare number on desktop or TUI; the phone reads its own summary
  rows), but a future desktop chip needs the count on the frame plus the projection in
  `use-desktop-feed.ts:95-101`.
- Subagent rows stay hidden; nothing here surfaces `CatalogEntry.subagent`.

## 5. The phone's second derivation

The phone moves onto the record; it does not keep its projection count. `daemon.py:550-552` counts
`status == "running"` over the live projection while the record counts `RUNNING_SUBAGENT_STATUSES` —
two definitions of one number, and the projection can be stale or absent. The fix follows the same
function's pattern for `leaving`/`updating` (`daemon.py:538-546`): read
`entry.record.subagents_running`, preserving `None`. A durable-only row has no entry and so no count:
no ring, no chip — honest (no live record, nothing claimed), NOT a divergence to fix. Cost:
`number | null` in `types.ts`, and the projection derivation dies with its test.

## 6. PR split and order

1. **local-operator — the state model** (Python + TUI + the phone's daemon field + this document):
   `resume.py`, `catalog.py`, `desktop_feed.py`, `session_picker.py`, `daemon.py`, tests. Everything
   else depends on it.
2. **local-operator-ui**: the `delegating` arm in `chat-session-status.tsx`, the contract mirror,
   `.mjs` cases, a Neighbours story row, a rendered before/after pair. Parallel development, merged
   AFTER (1) so its evidence comes from a backend that actually emits the code.
3. **local-operator — mobile web only** (`local_operator/mobile/web/**`): slot rung, noun, `types.ts`,
   web test, rendered phone frames. Its own PR because `ci_scope.py:130-141` makes a web-only diff
   INERT for every Python job (`:300-310`) — fast CI, and `mobile-web.yml` owns its only typecheck;
   folding it into (1) would attach web evidence to a Python review round. Watch `web/.gitignore:9-46`:
   a source-only change can emit an unstyled bundle that exits 0, so the rendered frame is the
   evidence, not the build.

**Overlap — coordinate, do not edit:** PR #1436 touches `resume.py`, `session/catalog.py`,
`tui/widgets/session_picker.py`, `server/models/desktop_sessions.py`,
`tests/unit/server/test_desktop_feed.py`; UI PR #448 touches `chat-sidebar.tsx`,
`canonical-sessions-store.ts`, `desktop-session-contract.ts`. Land this after (or rebased onto) #1436;
the parity test and the glyph->words `ALLOWED` map are the conflict hotspots.

## 7. Tests that must move, and must be added

Move (vocabulary pins): `tests/unit/tui/test_session_sidebar.py:1479-1556` (the exhaustive
glyph->words map and `checked == 75`: add the count dimension `None, 0, 2` plus the new entry and
update the product — round 1's finding was exactly a pairing no frame covered), `:1423` (tooltip vs
glyph), `:788` (mark column); `tests/unit/server/test_desktop_feed.py` (parity gains the two fields);
`tests/unit/session/runtime/test_registry.py:562-590` (carry the absent-≠-0 pin to the row); UI
`scripts/chat-session-status.test.mjs` (add `delegating`; re-point the unknown-code case at a
still-unknown code), `chat-session-status.stories.tsx`, `scripts/session-status-feed.test.mjs` (assert
the frame is STILL a triple); mobile `types.ts` + a slot-ladder web test + the daemon summary test.

Add (the state must be falsifiable):

- catalogue truth table: `idle`+2 -> `delegating` with the exact label; `idle`+1 -> "Delegating
  (1 subagent)"; `idle`+`None` -> `idle` (never `delegating`, never "0"); `idle`+0 -> `idle`;
  `busy`/`wedged`/`pending`/unseen-complete/`attached`/armed-wake/`leaving`/cold rows keep their rung.
- feed: `0 -> 2` and `2 -> 1` each publish exactly one `session_status` frame; a heartbeat rewrite
  publishes none; a `None` row publishes none.
- TUI: `cell_len(DELEGATING_MARKER) == 1`.
- ORDER: a delegating row keeps `session_category`'s tier and its `rank` — adding a state must not
  move a row (`catalog.py:82-163` is the record of that mistake).

## Risks to watch

- **Glyph crowding**: `⇉` (U+21C9, Neutral width, `cell_len == 1`) shares ink with the busy spinner;
  shape is the discriminator. If a rendered column shows the pair reading as one thing, re-ink before
  re-shaping.
- **`leaving` + children** is rare but reachable; one test keeps the `ALLOWED` map from ever needing
  an exception.
- **Stale numeric count** in a client that later draws it, and **renderer coupling**: a backend
  shipping `delegating` before the app shows `HelpCircle` — loud and correct, but the PR order above
  is what makes the app's evidence real.
