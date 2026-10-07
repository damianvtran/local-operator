# Mesh wire honesty: true stamps, and unreachable is not missing

**Slices in this note — three small PRs, sequential, design-first (this commit precedes
code):**

- **S1 — true stamps for wire-served rows.** A row served without a true entry time must
  SAY SO on the wire; the consumer contract for a row without a true stamp is written
  down here. (Unblocks the UI lane's ordering finality, `local-operator-ui#870`: the
  mid-turn join renders `[tool above user]` because a wire row carries a fabricated
  serve-time while a tool row carries its true `started_at_epoch` — two clocks that are
  not comparable.)
- **S2 — unreachable is not missing (the UI lane's U2).** A resolution miss that is due
  to SILENCE must be distinguishable from "no device holds it": the surface says
  *unreachable* and keeps the composer open; an unprovable absence is never presented as
  a deletion, and a silent peer is never pinned to an id.
- **M — the two deferred shutdown-window minors from #2024** (PR comment `6018349987`):
  `_stored_page_read` submits outside `_page_loop_lock`; the parked-read-during-stop path
  logs asyncio's `Task was destroyed but it is pending!`. Both still present on
  `e556fb984` (`network/relay.py:7069`, `:7114-7136`). Dropped into their own micro-PR
  after S1 so neither large PR mixes unrelated hunks.

Context: D5-core (`PR #2024`, merged `e556fb98`) fixed the cold READ; these are the two
adjacent honesty failures the same lane surfaced — both verified against `e556fb984`
(cites below, from a read-only fact pass over this tree; UI cites from
`~/local-operator-ui` main and its `remote-load-hydration` worktree).

---

## S1 — true stamps for wire-served rows

### The defect, as the code states it

`DesktopSessionBridge._remote_history` (`server/utils/desktop_sessions.py:3343`)
fabricates one serve-time stamp per page and stamps every wire row with it:

```python
3463|        stamp = time.time()
3464|        entries = [{"id": …, "ts": stamp, "type": "message", "payload": …} for row in page]
```

The rationale is documented at `:3389-3400` ("a message has no entry time of its own …
dates the user's own message to whenever they opened the window"). It is a STATED LIMIT,
and it is exactly what the mid-turn join breaks on: `withTimeOrder`
(`local-operator-ui`, `transcript-reducer.ts:989-1038` on main / `:1081-1130` in the lane
worktree) sorts by `record.ts` ascending, ties by position (`:1125-1127`) — while a tool
row is seeded from `started_at_epoch` (`:3803`, `seededClock` `:3816-3834`). Two clocks,
one order; the serve-stamp loses. The consumer already refuses an absent instant "as a
value, never defaulted" (`:3822`) — so the fix cannot simply omit `ts` for everyone.

**The asymmetry:** only the WIRE branch fabricates. The local branch serves journal rows
with real `ts` (`:3337`), and the cold peer path (this lane's `net_session_history`) serves
journal rows with REAL timestamps (`network/relay.py:7057-7059`). Same DTO, one source
lying.

### Requirement (routed from Aida, verbatim intent)

1. A row served without a true entry time must SAY SO on the wire — never a silently
   substituted reader-clock stamp. An absent (or labelled) stamp the reader can see is
   better than an invented one it cannot.
2. The consumer contract for a served row without a true stamp is written down, so
   ordering rules downstream are written against a contract, not a guess.

### Facts this design stands on (verified on `e556fb984`)

- **The truth exists and is joinable.** The window capture (`history_window.py`,
  `_capture_display_window:662`) holds the `Transcript`; the message id IS the entry id
  (`transcript.py:410-412`, `:3175-3179`; wake rows the same, `history_window.py:720-722`;
  entry time on the journal envelope `transcript.py:371/:377/:389`). A `{entry.id:
  entry.ts}` map is total over every row the window can carry — subtracted rows are
  ABSENT, not joinless (`history_window.py:723-734`).
- **The window is small and bounded**: `DISPLAY_HISTORY_MESSAGES = 120`,
  `DISPLAY_HISTORY_BYTES = 512 KiB` (`history_window.py:52-53`); the map is ≤ ~120 pairs
  (~2 KB) — inside the existing `oversized_frame_report` guard (`runtime/server.py:3795`).
- **The additive-field hazard has a precedent to copy**: `display-history-audit-v1`
  (`history_window.py:51`) + `AUDIT_WIRE_FIELDS`/`strip_audit_fields` (`:161-186`) applied
  on ALL THREE serialization routes (`runtime/server.py:3757-3760`, `:7486-7487`,
  `:7499-7505`). The comment at `:41-50` names the failure mode: forbidding extras +
  mixed builds ⇒ an unstripped new field is "a FAILED ATTACH — not a degraded one".
  Negotiation gate precedent: `:3604-3606`
  (`"display-history-window-v1" in self._record.capabilities`).
- **The desktop renderer never reads the attach window at all.** `grep` for
  `display_history|displayHistory` across `~/local-operator-ui/src/renderer/src` returns
  nothing; it is fed by the daemon's `sessions.history`/`sessions.get` pages
  (`use-canonical-session.ts:2653`, `:4590`, `:4670`), i.e. `HistoryPage` /
  `DesktopSnapshot.history`. So the desktop fix rides the ENTRY ENVELOPE, whose model
  (`server/models/desktop_sessions.py:548-558`) has no `model_config` → pydantic default
  `extra="ignore"`: an added sibling field is ignored by old renderers.
- **The UI consumes `ts` in many places** (`durableRecord:2008` `(entry.ts ?? 0)`; live
  rows `:2819`…; `monotonicStamp:4742-4770`, whose doc names the two-clock defect). The
  consumer contract below is written for these exact call sites.

### The decision

**Truth on the wire wherever it exists; an explicit, negotiated vocabulary where it does
not; legacy bytes preserved for renderers that do not negotiate.**

1. **Owner (runtime) ships the join.** `_capture_display_window` builds `{id: ts}` (or an
   inline per-row field — implementer's choice, same contract) and serializes it ONLY when
   the viewer advertised the new capability (working name
   `display-history-entry-times-v1`), through a `strip_*` helper exactly as the audit
   fields do, on all three routes. Unnegotiated viewers receive byte-identical frames —
   the failed-attach hazard is closed by construction.
2. **Daemon `/history` (+ the snapshot's embedded page) carries the vocabulary.** Entry
   envelope, additive, closed set:
   | truth available? | renderer negotiated `entry_ts=1`? | `ts` | `ts_source` |
   |---|---|---|---|
   | yes | either | true entry time | `"entry"` |
   | no | yes | `null` | `"unstated"` |
   | no | no | legacy serve-stamp | `"served"` |

   `entry_ts=1` is a per-request signal (the `frontend_replace` precedent,
   `routes/desktop_sessions.py:4680-4686`). `ts_source` on its own is additive and
   ignored by old readers; old readers keep today's ordering behaviour because the
   no-flag row still carries the serve-stamp. (The cold stored page
   (`net_session_history`) serves journal rows directly — its entries are `"entry"`
   by construction.)
3. **Consumer contract (for the UI lane — quote into the renderer's rule):**
   - `"entry"` — `ts` is the row's true entry time (seconds); safe to order by and to
     display.
   - `"unstated"` — no instant exists for this row; order by position/arrival and display
     no time. Never default it to 0 (the reducer already refuses this, `:3822`).
   - `"served"` — a transport arrival approximation, NOT a stated instant; do not order
     it as a clock and never display it as the message's time.
   - `ts_source` ABSENT — a legacy daemon; keep today's behaviour (opaque ordering clock).
4. **Not in scope:** consumption of entry times by window-parsing surfaces (TUI/phone,
   `attached.py:5274`). The carrier ships capability-gated and stripped; consumption is a
   named follow-up, not smuggled in.

### Evidence plan

Unit: join totality (every carried row resolves; subtracted rows absent); strip-on-
unnegotiated (an old-viewer frame is byte-identical — the audit-field discipline, tested
the same way); the `/history` value table above for each of the three rows incl. the
legacy corner; `ts_source` ignored by an extra-ignoring reader. QA: drive a real two-relay
pair (mid-turn join) and diff entry `ts` against the owner's journal for the same ids;
render the values through the contract table. Docs: `DESKTOP_API.md` (history entry row +
the vocabulary), mobility §3.4's ts note updated to point here.

---

## S2 — unreachable is not missing (U2)

### The chain, verified

`GET …/events` (and snapshot/history) → `DesktopSessions.session` →
`locate()` (`desktop_sessions.py:7295`) → the id resolves through
`remote_open.remote_row_for` (`:54`, body `:76-85`): cache-first, then ONE forced read
(`peer_session_rows(root, ttl_s=0)`), then `None` → `KeyError("Unknown session")`
(`:7386-7388`) → `errors()` `except KeyError` → **404**
(`routes/desktop_sessions.py:1876-1879`). The reader turns exactly a 404 into
`missing` → "This conversation is no longer on this machine…" + a refused composer
(`use-canonical-session.ts:3927-3953`; `canonical-transcript.tsx:3306-3336`).

The reachable-but-refusing branch already exists and is the model: a row with
`row.reachable == False` raises `PeerSessionUnreachable` (`:7398`) → 409
`session_is_remote` + `remote_open.unreachable_peer_sentence` (`routes:1804-1817`).
**The gap is the miss path**: when the forced read could not REACH anyone, the silence is
recorded — `unanswered_peers()` (`peer_rows.py:133-163`, `UnansweredPeer(device_id, name,
reason)` `:73-92`) rides the SAME cache entry as the rows (`_read_all:430-442` writes all
three under one key/moment), so a check after the forced read needs **no second dial**
(proven on the failure path too: `_read` still writes). Nothing consults it there.

### The decision

**At the seam that answers a surface, a resolution miss consults the same read's silence;
when peers were silent, the answer is a distinguishable "unresolved — did not answer"
state, never "unknown". Attribution never pins a silent peer to an id.**

- The consult lives at the raise sites, NOT inside `remote_row_for` (six call sites:
  `cli.py:13721/:14745`, `network/cli.py:4912`, `desktop_mesh.py:652`,
  `desktop_sessions.py:5988` + `:7386`, `remote_open.py:125`). The pool's `locate()`
  miss path and `open_remote_viewer` consult `unanswered_peers(root)`; with silent
  devices present they raise a new typed state (working name `PeerSessionUnresolved`),
  whose sentence names the SILENT DEVICES as silent ("<names> did not answer; this
  conversation may be on one of them") and never claims ownership.
- Routes map it to a distinguishable response with its OWN code (409
  `"session_unresolved"` — same family as `session_is_remote`, different remedy), so the
  UI can render *unreachable* and keep the composer open. The rule for the surface (UI
  lane): an unprovable absence must not present as a deletion and must not close the
  composer; a Retry/reconnect affordance replaces "Start a new chat" as the only action.
- With every device answering and none holding the id: still 404 — a real "unknown".
  With a cached row marked unreachable: unchanged 409 `session_is_remote` (already
  truthful). Phone daemon shares none of this path (verified: no matches in
  `local_operator/mobile/`), so this is desktop-daemon + CLI scope.
- Open-question left to the design review, stated rather than decided: the genuinely-
  unknown copy's wording once every device HAS answered ("no longer on this machine" is
  the UI lane's to refine; flagged, not silently changed here).

### Evidence plan

Unit: silent peer + unknown id → the new state with names and no ownership claim; all
peers answered → 404 unchanged; unreachable-marked row → 409 unchanged; the consult
issues no dial (cache-shared, asserted at the relay's op counter). QA: kill a peer's
relay, open its cached session's stream/history through the real daemon → the new code,
composer-open contract handed to the UI lane with the exact response; restart the relay →
resolves again.

---

## M — the deferred loop minors

`network/relay.py`: (1) `_stored_page_read` (`:7069`) submits via
`asyncio.run_coroutine_threadsafe` (`:7082`) after `_page_loop_lock`'s lock is released —
take the submit inside the lock (or re-check the loop's liveness under it) so a
shutdown-concurrent read cannot hit `RuntimeError: Event loop is closed`; (2) the parked-
read-during-stop path logs `Task was destroyed but it is pending!` — cancel/close the
parked future from `_page_loop_main`'s `finally` (`:7133-7136`) so the doomed read ends
without asyncio's complaint. Each with a test at the same grain as the #2024 cells
(red-before provable from the recorded reproductions). Own micro-PR after S1.

---

## Delivery

Per lopdev: coder + reviewer + QA rounds per PR; PRs non-draft, no reviewers, nobody
tagged; heads reported to Aida; releases ride the window lane. The UI halves (the
consumer contract in S1, the unreachable surface in S2) are the UI lane's; this note is
the contract they can write against before our code lands.
