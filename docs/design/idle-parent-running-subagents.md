# Design: an idle parent that owns running subagents

Status: shipped in this PR. Base: `origin/main` @ `dad3b92f`; all `file:line` are against that commit
(or the tree this PR produced, where the line moved). The shape was decided before the code was
written (the catalogue derives a distinct state when the parent's own turn is not busy and children
run; the count travels inside the label); this file records WHY.

**Revised after the shape was implemented.** Two spellings changed between the proposal and the
landed change, and this file has been corrected rather than left as a plausible-looking record of a
state that does not exist: the LABEL is the count's own sentence (`2 subagents running`, and its
`· N queued` addend) rather than `Delegating (2 subagents)`, and the desktop icon is `Share2` — the
app's own "Delegated work" glyph — rather than `CornerDownRight`. The phone's mark is two accent
dots, not a ring. The reasons are in §2 and §5.

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
| label | "Ready" | **"2 subagents running"** ("1 subagent running"; "· N queued" addend; "N subagents queued" alone) |
| TUI glyph | `●` | **`⇉`**, ink `accent`, static |
| desktop icon | `Circle` | **`Share2`**, ink `text-accent`, static |
| phone slot | empty | **two 4px accent dots**; chip `N subagent(s)`, or `N queued` when nothing is running |

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
- Label **the count's own sentence**, built by `catalog.delegating_label` in three shapes:
  `2 subagents running` / `1 subagent running` (singular at one, the `Scheduled (1 wake)` rule —
  `tests/unit/tui/test_session_sidebar.py`), `2 subagents running · 3 queued` when both facts exist,
  and `2 subagents queued` when nothing is spending yet.
- **Lead with a count, not with "Delegating".** Rejected on the words: a state whose whole content is
  "the parent's own turn is NOT running" must not open with a gerund that implies the opposite. It
  also puts the number first, where a one-cell glyph cannot carry it.
- Noun **subagent(s)**: the record's field name, `/info`'s word (`info/render.py:435`), and the
  TUI's own stop notice (`tui/app.py:20950`). The phone's chip is renamed to it in this PR;
  "agent" alone is ambiguous in a product with an "Agents" page of reusable profiles.
- One string serves the TUI tooltip, the desktop `title`/`sr-only` (both `row.status.label`) and the
  phone's chip.

## 3. Precedence rung

`status_code` (`catalog.py:186-210`), inserted between `attached` and `scheduled` and gated on
`not row.leaving`; `status` (`:286-370`) takes the same rung, with `leaving` above it exactly as it
sits above `busy` today (`:315-316`).

```
pending -> approval | answer     wedged | busy
shows_completion_mark -> complete | error | interrupted
attached -> attached ("Open")
+ delegating -> delegating ("2 subagents running")   <-- NEW
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

One rung, two and a half surfaces — and THE PHONE IS THE HALF, deliberately. The TUI mark and the
desktop icon chain both read `status_code`, so they share this ladder exactly. The phone cannot: its
summary carries no code (adding one would be a wire change of its own for one rung), so it ranks the
facts it does carry — `streaming`, the unread receipt, and now `leaving` — and the daemon withholds the
counts for any session it cannot vouch for (`_advertisable_counts`: a degraded dial, a stopped
heartbeat, a runtime on its way out), which is what keeps the phone's number from advertising work the
other two surfaces are calling "Leaving…" or "Not answering" (UX round 1, U1). The phone's ladder is
therefore shallower than this one BY CONSTRUCTION, and the place to enforce the rung is the wire, not
the client. Non-goal: the terminal title's separator vocabulary (`tui/terminal_title.py:10-26`) and the
CLI STATE column — `leaving` already refused to teach STATE a new word, for this reason.

## 4. What carries the count

- `SessionRow` (`resume.py:1821`, live block `:1855-1898`): add `subagents_running: int | None = None`
  and `subagents_queued: int | None = None`, defaulted like `leaving`, so every other construction
  site is unchanged. `decorate_rows` fills them at `catalog.py:708-720` and `_row_for` transcribes the
  same two lines (the hand transcription is already held in parity by
  `tests/unit/server/test_desktop_feed.py`). No new IO: both come off a record the callers already
  read, so the 10 Hz doorbell budget (`desktop_feed.py:110-119`) is untouched.
- ONE predicate, on the ROW, so glyph and words cannot disagree:
  `SessionRow.delegating` (`resume.py`) is `not leaving and
  (subagents_running or 0) + (subagents_queued or 0) >= 1`, returning the normalised pair (or
  `None`). `status_code`/`status` read it and so does `row_state_mark`. **It answers only the COUNT
  question**: `pending`/`busy`/`wedged`/`attached`/an unseen completion are louder facts each ladder
  already tests above this rung in its own order, and folding them into the property would give the
  two callers a second, hidden precedence to keep in step — the failure it exists to remove. The one
  exception is `leaving`, which is a GATE rather than a rung: `status_code` has no leaving arm at
  all, so without it a draining row whose `live_state` was idle would publish `delegating` beside a
  "Leaving…" tooltip and draw `⇉` next to the words. The row-level home also avoids a new import
  edge between `session_picker` and `session.catalog`, and makes the pairing structural rather than a
  promise — the reason `shows_completion_mark` exists. `reported_subagent_count`
  (`session/runtime/types.py`, beside the fields it validates) refuses a non-`int`, a `bool`, a
  negative and anything past `MAX_REPORTED_SUBAGENT_COUNT`, and it is the ONE implementation for three
  readers: this predicate's `resume._counted`, `info.collect`'s fleet tally, and the desktop listing's
  response model — rather than letting a corrupt record raise inside the sidebar's poll loop (`from_json`
  does no type validation and this is the first ARITHMETIC on those fields) or 500 the conversation list
  at the wire edge (review round 1, R4).
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

The phone moves onto the record; it does not keep its projection count. The old `_merge_summaries`
counted `status == "running"` over the live projection while the record counts
`RUNNING_SUBAGENT_STATUSES` — two definitions of one number, and the projection can be stale or absent.
The fix follows the same function's pattern for `leaving`/`updating` (`daemon.py`): read
`entry.record.subagents_running` and `subagents_queued`, preserving `None`. A durable-only row has no
entry and so no count: no mark, no chip — honest (no live record, nothing claimed), NOT a divergence
to fix. Cost: `number \| null` in `types.ts`.

**The phone's two arms, and why its mark is not a ring.** The slot is the only place this app can
carry the state — it has no `attached` or wake arms to lose to — so two 4px accent dots keep the
12px slot's geometry and give the state a SHAPE distinct from the single unread dot one rung above it
in the same ink (the TUI's own argument for the arrow). A ring read as a quieter dot at this size. With
the pair's 2×4px inside the reserved box the title's start x is untouched.

**The chip sums the two counts, and that is NOT the catalogue's wording.** `delegatedChip` prints
`N subagent(s)` from `running + queued`, where `delegating_label` prints `2 subagents running ·
1 queued`. The reason is the tap: the session view the row opens onto prints its roster header as
`{running}/{direct.length} running`, and both halves of that come from the folded projection, which
draws a parked child as running (`mobile/projection.py` maps `queued` → the `running` mobile status). A
chip that counted only the running ones therefore read `1 subagent` on a row whose session view said
`2/2 running` one tap later (UX round 1, U2). One number for one population across the tap wins over
spelling the split on a chip this narrow; the precise sentence stays on the catalogue, the desktop
`title` and the TUI tooltip, where there is room for it. The chip is hidden on `null` and on 0, always
carries the noun (a count-only chip left the reader guessing WHAT was queued, U3), and caps at `99+`
so three digits cannot `shrink-0` the row's name away at 360px (U4) — the exact figure is one tap away
in that same roster header.

> **"One population" holds at DEPTH 1, and that is the whole of the claim.** The chip counts this
> session's whole subtree; the roster header counts its direct children. Past depth 1 they diverge by
> design (`4 subagents` beside `2/2 running`), which is pre-existing and deferred with both
> populations named in "Not in scope" below — read that bullet before repeating this sentence.

**And it is not the sidebar's `⌥N` chip.** Two delegated-work numbers can appear on one TUI frame:
`⌥14` in the sidebar's footer is the store's GLOBAL hidden-run population (`catalog.subagent_population`,
newest-first and capped, across every session), while this rung counts THIS session's children. Both
can be true at once and neither is a bug; the new mark is simply the first thing that makes a reader
expect the sidebar's children affordance to be this session's. Naming which population each one is
remains open (UX round 1, U5, recorded rather than re-pointed).

## 6. PR split and order

1. **local-operator — the state model, TUI and the phone** (Python + TUI + the phone's daemon field +
   `mobile/web/**` + this document): `resume.py`, `catalog.py`, `desktop_feed.py`,
   `session_picker.py`, `daemon.py`, `mobile/web/src/{types.ts,screens/session-list.tsx}`, tests.
   Everything else depends on it.
2. **local-operator-ui**: the `delegating` arm in `chat-session-status.tsx`, the contract mirror,
   `.mjs` cases, a Neighbours story row, a rendered before/after pair. Parallel development, merged
   AFTER (1) so its evidence comes from a backend that actually emits the code.

The mobile web change was originally proposed as its own PR — a web-only diff is INERT for every
Python job (`ci_scope.py`) and `mobile-web.yml` owns its only typecheck — and it now rides PR 1 with
the rest. The cost is that web evidence lands on a round that also reviews Python; the gain is that a
single round reviews the ONE vocabulary in every place it is rendered, which is the property this
change is most likely to regress. Watch `web/.gitignore` either way: a source-only change can emit an
unstyled bundle that exits 0, so the rendered frame is the evidence, not the build (the build's own
`check-bundle.mjs` is the other guard — it reported `293 classes for 230 tokens` here).

**Overlap — coordinate, do not edit:** PR #1436 touches `resume.py`, `session/catalog.py`,
`tui/widgets/session_picker.py`, `server/models/desktop_sessions.py`,
`tests/unit/server/test_desktop_feed.py`; UI PR #448 touches `chat-sidebar.tsx`,
`canonical-sessions-store.ts`, `desktop-session-contract.ts`. Land this after (or rebased onto) #1436;
the parity test and the glyph->words `ALLOWED` map are the conflict hotspots.

## 7. Tests that moved, and that were added

Moved (vocabulary pins): `tests/unit/tui/test_session_sidebar.py` (the exhaustive glyph->words map,
now `checked == 675` — the count dimension `None, 0, 2` on BOTH counts, because a pairing no
hand-picked row covers is exactly how this mark goes wrong; round 1's finding was a pairing no frame
covered), the tooltip-vs-glyph test, the rendered mark-column test;
`tests/unit/server/test_desktop_feed.py` (parity gains the delegating arm);
`tests/unit/mobile/test_daemon.py` (the summary shape, plus the daemon's own gate — UX round 1, U1: a
degraded entry, a stopped heartbeat and a `leaving` runtime each report NO counts, a fresh live entry
reports them).

Added (the state must be falsifiable), all landed in this PR:

- `tests/unit/session/test_catalog_delegating.py` — the catalogue truth table: the three label shapes
  with the singular at one; `idle`+`None` -> `idle` (never `delegating`); `idle`+0 -> `idle`; a
  corrupt count neither raising nor inventing a state; and every precedence collision
  (`pending`/`busy`/`wedged`/unseen-complete/interrupted/error/`attached`/armed-wake/`leaving`/cold)
  asserted TWICE per case — with and without children — so the rung can only be added below them.
- `tests/unit/server/test_desktop_feed.py` — the U3 edge test: `0 -> 2` and `2 -> 1` each publish
  exactly one `session_status` frame, a heartbeat rewrite publishes none, a record reporting nothing
  republishes the rung it leaves, and the frame is asserted to still be a `{code, label, revision}`
  TRIPLE.
- `tests/unit/tui/test_session_sidebar.py` — `cell_len(DELEGATING_MARKER) == 1`, and a RENDERED row
  showing the arrow in the mark column without moving the title.
- `tests/unit/session/test_catalog_delegating.py` — ORDER: a delegating row keeps
  `session_category`'s tier and its `rank` (adding a state must not move a row; `rank`'s docstring is
  the record of that mistake), and the absent-≠-0 case carried from the record to the row through the
  real scan.
- `mobile/web/src/session-list.delegating.test.tsx` — the slot rung and the chip against the real
  card: the pair of dots, the singular, the sum (one running + one parked reads `2 subagents`, the
  population the session view counts), the `99+` cap, `leaving` suppressing the mark, the two rungs that
  outrank it (unread, then streaming — which still count their children), and `null`/0 rendering
  NOTHING rather than a zero.

## Risks to watch

- **Glyph crowding**: `⇉` (U+21C9, Neutral width, `cell_len == 1`) shares ink with the busy spinner;
  shape — and being STATIC — is the discriminator. If a rendered column shows the pair reading as one
  thing, re-ink before re-shaping.
- **`leaving` + children** is rare but reachable; one test keeps the `ALLOWED` map from ever needing
  an exception.
- **Stale numeric count** in a client that later draws it, and **renderer coupling**: a backend
  shipping `delegating` before the app shows `HelpCircle` — loud and correct, but the PR order above
  is what makes the app's evidence real.
- **The desktop row has room for exactly one trailing statement** (`chat-sidebar.tsx`
  `rowTrailingStatement`, three failed layouts recorded), so the count deliberately does NOT get its
  own chip there: it rides `status.label` into the row's `title` and `sr-only`, and the row shows the
  icon alone. A future design that wants the number visible inline is a change to that slot's
  contract, not to this state.

## Not in scope — deferred, with the repro

A pre-existing GATE-LIFECYCLE defect is visible in the same neighbourhood and is deliberately NOT
fixed here: a settled child's gate clears the parent's published `pending` while another card is
still parked, because `ServingSessionHandle._announce_settled()` clears unconditionally. The repro
lives in `tests/unit/session/runtime/test_parked_gates.py`, and the fix sketch is one condition — only
clear the published gate when the card being settled is the one that published it. It is unfixed
because it is a change to gate lifecycle rather than to this row's vocabulary, and it would be
reviewed as a different change with a different blast radius. Recorded on the PR as a
`deferred — <reason>` finding so it is not lost.

A second pre-existing presentation defect is deferred for the same reason, and this change is what made
it visible: the session view's roster folds a PARKED child into the running lane. `mobile/projection.py`
maps the `queued` lifecycle status onto the `running` mobile status (only `paused`/`pausing` become
`parked`), so the roster header prints `2/2 running` for a parent with one running and one waiting
child, and draws the waiting child with the running `⟳` and its pulse. The phone's chip was moved onto
THAT population rather than this one's (see §5) so the two surfaces agree across a tap — but the fold
itself is a projection/presentation question with its own blast radius (every roster, every platform
client, and the `parked` literal's meaning), not a row-vocabulary one, so it is recorded on the PR as
`deferred — <reason>` with the anchors above rather than changed here.

**A third, also pre-existing, divergence is NAMED here rather than left in the code's head: the two
numbers this change is about count two populations, and they part company past depth 1.** The chip
(and the catalogue label) reads the RECORD, whose count is this session's WHOLE SUBTREE — the runtime
counts over ``comms.nodes()``, "already contains every nested descendant"
(``session/runtime/types.py::SessionRecord.subagents_running``) — while the session view's roster
header counts DEPTH 1 ONLY (``SubagentsPanel`` passes ``parentJobId={null}`` and ``AgentRoster``
filters ``agent.parent_job_id === parentJobId``, so ``{running}/{direct.length}`` is direct children by
construction). A parent with two direct children, one of which spawned two of its own, is therefore
``4 subagents`` on the list and ``subagents 2/2 running`` on the view one tap later. Both numbers are
true and neither surface is broken: they answer different questions about the same session.

PRE-EXISTING, measured at the merge base ``dad3b92ff6ef0e166f5e0caed0ffa8dcd3539402`` — the anchor is
the pre-change chip itself: ``sum(1 for subagent in (p.subagents if p else []) if subagent.status ==
"running")`` (``local_operator/mobile/daemon.py:631-633`` at that ref) walked the projection's FULL node
list, i.e. the same whole-tree population, and the roster's ``direct.length`` was depth 1 then too. So
what this change fixed is which population the chip reads at depth 1 (the record's, so ``1 running +
1 parked`` stops reading ``1 subagent``) and the word for a parked child — not the depth divergence,
which is a decision about which question each surface answers (round the chip down to direct children,
or teach the header the tree total with the direct count secondary), not a bug fix in a row vocabulary.

THE RIG CAN NOW DRAW BOTH: ``scripts/mobile_delegating_fixture.py`` carries the depth-2 row (two direct
children, one of which spawned two of its own) and `scripts/mobile_delegating_shot.py` taps it and
captures the view it lands on, composing ``nested-list-vs-view-{390x844,360x640}.png`` — the list's
``4 subagents`` beside the view's ``subagents 2/2 running`` in one image. Deferred by the manager's
ruling on design round 2's D1, with this bullet and that row as the recording.
