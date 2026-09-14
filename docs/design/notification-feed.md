# The machine-wide notification feed: decision record

Covers the backend half of the "background completions never reach the desktop
app" work. The routes, frames and the ladder are documented for consumers in
`docs/DESKTOP_API.md` and `docs/ATTENTION.md`; this file records WHY, and the two
places the implemented behaviour differs from the design briefs that preceded
it.

## The defect

`notification` frames ride the SSE stream of a session's BRIDGE, and a bridge
exists only while a route holds one. The desktop app holds exactly one — the
session it displays — so a completion in session B while the app showed session
A produced no composed frame, no banner and no sidebar mark beyond a 5 s
`visibilityState`-gated `sessions.list` poll. The only remaining announcer was a
running TUI's 1 s tick; with no TUI running, a finished turn was announced by
nobody.

## What was built

- `local_operator/server/utils/desktop_feed.py` — the process singleton behind
  `GET /v1/desktop/events`. One 100 ms `os.stat` doorbell on the attention
  store, SQL only when it moved, and a delta read (`AttentionStore.published_since`
  / `acknowledgement_map`) so a tick costs what happened rather than what exists.
  It composes `notification` frames for sessions with no bridge and emits one
  `attention` frame per changed session. **It acquires no bridge and spawns no
  runtime** — asserted in `tests/unit/server/test_desktop_feed.py`, because the
  tempting "reuse the bridge's composer" refactor would make watching a
  catalogue into spawning one.
- `local_operator/server/utils/desktop_presence.py` +
  `local_operator/session/runtime/presence.py` — the machine-wide delivery
  lease: a route the app beats, an aggregate at `run/desktop/delivery.json`
  (0700/0600, staged) that every sibling process reads, reaped on a dead pid or a
  stale heartbeat (the same two rules as `scan_viewers`).
- `notifications.compose.notification_payload` — the payload builder the bridge
  and the feed now share, so `dedupe_key` is byte-identical and one completion
  cannot become two banners.
- The four-rung ladder in `session/runtime/serving.py` (rung 4 is new), the
  visibility predicate pinned to `watching_surfaces()`, the TUI's deference to
  rung 2, and `desktop.launch_command` + the launch rung in the click ladder.

## Correction 1 — `focus_policy` is a routing field, not content

The design said the feed ships the per-session payload "byte-for-byte". That is
wrong for one field, and shipping it literally would have made the operator's
report permanent: the per-session frame hard-codes `focus_policy:
"when_unfocused"`, and the app suppresses exactly that value whenever ANY window
is focused. In the commonest state of all — the user in the app on session A
while B finishes — rung 2 had already silenced the runtime and the TUI, and the
app then suppressed the frame. The completion reached nobody.

So `focus_policy` is derived per completion: a completion for the session a
desktop window is displaying raises no feed frame at all (rung 1 applied, the
card is in band), and any other completion carries `always`. The "verbatim
payload" decision becomes "verbatim EXCEPT `focus_policy`", and the parity test
asserts the two builders differ on exactly that field and nothing else.

## Correction 2 — rung 1 is VISIBILITY, not reachability, and the presence is narrowed by kind

`notification_surfaces()` answers "could a banner reach this person somewhere on
this machine". Using it as a suppression predicate made that read as "a human is
reading this session": with the panel on X and the window behind another app,
every OS surface went quiet while nobody was looking. Rung 1 is therefore pinned
to `watching_surfaces()` → `_visible_attach_surfaces()`, and the desktop's
contribution to it is the window's REAL state (focused AND visible AND not
minimised, reported by Electron main) rather than the renderer's
`document.visibilityState`/`hasFocus()`, which the UI's own code documents as
unsound.

A desktop connection still counts as watching from its per-session lease when
NO machine-wide presence exists at all, so an app that predates this feature
keeps its behaviour verbatim in both directions.

Second half: making the presence machine-wide must not suppress a GATE it cannot
replace. The feed carries completions only, so a parked `ask`/`approval` would
have been silenced with nothing to announce it. The lease therefore advertises
`can_notify_kinds` (`["complete","error"]`), a reader asks for the kind it is
about to route, and the gate path keeps its per-session lease and its
per-session toast untouched. Carrying gate frames on the feed is a follow-up;
the user-visible gate win is delivered by the click ladder instead.

## Constants, and where they live

| Constant | Value | Why |
|---|---|---|
| `DOORBELL_INTERVAL_S` | 0.10 | two stats per tick, no SQL; the detection floor the design's latency target is built on |
| `CATALOGUE_PROBE_INTERVAL_S` | 1.0 | the invalidation token needs one `readdir`; running it at 10 Hz would spend the doorbell's whole budget on a page of rows |
| `HEARTBEAT_INTERVAL_S` | 15.0 | the client's silence watchdog is 3x this, so a half-open socket is detectable |
| `BURST_LIMIT` | 3 | shared with the TUI's per-tick cap and asserted equal by a test |
| `PRESENCE_TTL_S` / `PRESENCE_BEAT_S` | 45 / 15 | the same pair `WATCH_TTL`, `DESKTOP_WATCH_LEASE_S` and `VIEWER_HEARTBEAT_TIMEOUT_S` use |
| `PRESENCE_CACHE_TTL_S` | 2.0 | bounds how stale a revocation can be, and keeps a directory read off the announce path |

## What this deliberately does not do

- **No gate frames on the feed** (see correction 2).
- **No notification-time pre-warm.** A warm is cancelled unless a subscription
  holds the bridge, so it either pins a runtime per completed session or buys
  nothing; mounting the panel and taking a visible lease already creates the
  runtime.
- **No change to `notification_contract`.** It stays 1: the payload is unchanged
  apart from the derived routing field, and both capability keys are new.
- **No mobile push**, no resident hidden window, no second transport.
- **A catalogue token that does not notice an in-place transcript append.** The
  token is the sessions directory's own `(inode, mtime_ns)` plus its name set,
  so a row set that changes is invalidated within a second; a title or preview
  rewritten under a stable row set is left to the sidebar's 30 s safety poll and
  its refetch on focus. Noticing it would mean walking the store on every tick,
  which is the cost the retired 5 s `sessions.list` poll was paying.
