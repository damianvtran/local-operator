# Reaping idle runtimes: parked sidebar sources and the 5-minute clock

Status: **superseded in part by the Decision below.** Branch
`fix/reap-idle-runtimes`. The analysis in §§1–2 is what was implemented; the
headline recommendation in §3.1 was **not**.

Every file:line below was read on this branch (base `0cd3a9434`).

---

## Decision (recorded at implementation time)

**Option A — delete the LRU clause and evict the presentation with the source
— was REJECTED. A timed viewer-side release was built instead.**

Option A is one line and cannot be wrong, and §3.1 argues it well. It was
rejected for a reason that sits outside this note's frame: it pays its cost in
exactly the interaction `RETAINED_PRESENTATIONS` was raised from 4 to 12 to
fix, and that raise was made against measurements (`app.py`: 246 ms loop-CPU
per cold switch versus 59–65 ms on a hit; 12/40 cache hits at capacity 4
against 39/40 at 8). Under option A *every* parked conversation becomes a cold
rebuild on switch-back, which reintroduces precisely the switch-back jitter
that constant exists to prevent. §3.1 acknowledges this as "the strongest
objection to the viewer-side fix" and proposes measuring afterwards; the
measurement that motivated the constant already exists, so the regression was
taken as known rather than as hypothetical.

**What was built.** A parked source keeps its socket *and* its presentation for
`SIDEBAR_IDLE_RELEASE_S` (5 minutes). Past that, with no attention in the
interval, the source is released — socket closed, frontend unsubscribed,
controller disposed, session disposed — and its presentation is evicted at the
same time. The runtime then sees `attach_clients()` fall to 0 and reaps itself
through the existing 3-second residency drain. No wire op, no new frame, no
protocol change.

This keeps both properties the note treats as mutually exclusive: the
alternating working set is completely unaffected (a user returning inside the
window never released anything, so there is zero perf regression on the common
path), while a conversation genuinely abandoned stops pinning a ~283 MB child.
The presentation is evicted *with* the source for the reason §3.1 already
identifies — a released source mints a new `SessionInteraction` and therefore a
new `token`, so a retained presentation could never hit again.

**What this note got right and was kept verbatim:** the viewer is the only safe
initiator (§2 — the `retiring` churn loop and the EOF redial storm are both
real, and both were re-verified in the code before implementing); the clock
must not reset on socket presence, the band poll or the sidebar refresh (§3.4);
`is_busy()`/`_should_exit` must not be duplicated viewer-side (§3.4); and the
`retire_if_unused` offer on the sidebar-leave path (§5) was implemented as
specified.

**The empty conversation, and an idea that was tried and reverted.** The
`retire_if_unused` offer on the sidebar-leave path (§5) makes a pristine
runtime go *immediately* when its viewer lets go, rather than waiting for the
residency drain. The judgment stays the runtime's (`retire_if_unused` →
`is_pristine`), never a viewer-side guess at emptiness: only the runtime can
see an armed wake, a just-arrived peer message, or a second attached viewer.

An earlier revision of this change went further and released an
apparently-empty source at *park* time, on the reasoning that retaining it buys
no switch-back because there is no transcript to rebuild. **That was wrong and
was reverted.** A parked empty conversation is still reachable — `/new`, click
a session with history, click back is a shipped flow — so disposing it at park
destroys a source the user can still return to. It was caught by
`test_sidebar_swap_reset.py::test_returning_to_an_empty_conversation_shows_the_splash_under_a_notice`
and is now pinned directly in
`test_sidebar_idle_reap.py::test_an_empty_parked_source_still_waits_out_the_clock`.
Emptiness changes what the runtime does when the clock expires; it is not a
licence for the viewer to let go early.

**Consequence for the committed repro tests.** They were written during
diagnosis and encode "release immediately", so they are not compatible with a
5-minute clock as literally written. The compressed timeout they now use
patches `SIDEBAR_IDLE_RELEASE_S`; every assertion is unchanged. See the header
of `tests/e2e/test_sidebar_runtime_reap_e2e.py`.

---

## 1. The problem as I found it

One TUI process held 15 established attach sockets and 15 runtime children
(~1.9 GB RSS) while idle for ~30 minutes. Two independent defects produce
that, and only one of them is the leak.

### Defect 1 — the presentation LRU gates process release

`_sidebar_source_releasable` (`app.py:4618-4646`) ends with:

```python
and source.session.session_id not in self._sidebar_presentations
```

`_sidebar_presentations` is the retained-widget LRU, capacity
`RETAINED_PRESENTATIONS = 12` (`app.py:1372`). Its documented purpose is
fast switch-back of *prepared Textual widgets* (`app.py:1342-1362`), and the
constant's own comment prices a parked slot at "~1.5-2.4 MiB RSS ... plus one
live socket whose owner deltas are still delivered while parked"
(`app.py:1367-1368`).

**That price is wrong, and the omission is the defect.** A parked
presentation pins its leased `RemoteSession`, whose attach socket makes the
runtime's `attach_clients()` return 1 (`server.py:1536-1551`), which is
term 3 of `process._should_exit` (`process.py:282-283`). So a widget cache
sized for a *rendering* working set silently became a **process** working
set: 12 parked conversations pin up to 12 runtime children at ~283 MB each
(the figure `process.py:75` uses), not 12 × 2 MiB of widgets.

The measured probe in the task confirms nothing else was holding them:
`retained_for_local_work=False`, `retained_for_auto_work=False`,
`preparations=0`, no pending gate — the predicate's other five clauses
(`app.py:4633-4644`) all permitted release. Only LRU membership refused.

Note the asymmetry that proves the LRU clause is not load-bearing for
correctness: the sidebar-*close* path already drops every source
unconditionally, LRU membership and all (`app.py:5423` clears
`_sidebar_presentations` wholesale, then `app.py:5452` releases every
remaining source with `reason="closed"`). Mass viewer-side release of parked
sources is therefore an **established, shipped path**, not a new mechanism.

### Defect 2 — no signal distinguishes "attached" from "in use"

`_viewer_attached` (`process.py:230-250`) is a pure presence check on
`attach_clients()`. `_ClientConn.last_seen` (`server.py:426`) is stamped only
in `_handle_client`'s request loop (`server.py:1345`, immediately before
`_on_request`), so for a viewer that sends nothing it is the attach time and
never moves. The task's 12-second spy on `_on_request` measuring zero ops is
consistent with the code: I found no periodic op from an attach client —
`remote.py` has no `set_interval`/keepalive, and the only recurring wire op
in the tree is the desktop host's `desktop_watch` lease
(`desktop_sessions.py:354-376`), which is not a terminal TUI.

So today an attached viewer pins its runtime forever, by design
(`process.py:261-275` argues term 3 as a READINESS signal), and there is no
existing evidence that could time it out.

---

## 2. The constraint that decides the whole design

**A runtime must not idle-reap itself while a TUI is displaying it.**

`_adopt_session` installs `set_refresh_callback(self._on_runtime_refreshed)`
on the adopted session (`app.py:6039-6041`). `_on_runtime_refreshed`
(`app.py:12455-12502`) ends in `self._start_runtime_engage(reason="refresh")`
— eager and unconditional, by explicit design (`app.py:12458-12461`: "the
band would show the cold state until the user typed"). The `retiring` frame
(`server.py:849-880`) lands as `RETIRING_REASON`
(`attach_client.py:398-404`), which `_on_disconnected` routes to
`_go_cold(refresh=True)` (`remote.py:3375-3384`), which fires that callback
(`remote.py:3651`).

Therefore a runtime that idle-reaps itself and announces `retiring` to the
current TUI gets a **brand-new runtime spawned within ~1 s**: exit → spawn →
idle → exit, forever. That is strictly worse than the leak — it converts a
static 1.9 GB into 1.9 GB plus a permanent respawn treadmill.

The frame choice is not a way out. The two available frames are the only
two, and both are wrong for an idle reap of a displayed session:

| frame | viewer behaviour | verdict for idle-reap |
|---|---|---|
| `retiring` | `_go_cold(refresh=True)` → eager re-engage (`remote.py:3383`) | **churn loop** |
| bare EOF | `_on_disconnected` → `_recovering = True` → `_recover_owner` (`remote.py:3400-3407`) | **worse** — see below |
| `stopping` | parks in the stopped state, tells the user `/resume` (`server.py:854-857`) | lies; the session did not end |

Bare EOF is worse than it looks for exactly the sources we care about.
`_can_go_cold` is set from `surface == "desktop"` (`remote.py:485`), and
`_lease_sidebar_source` calls `RemoteSession.connect` (`app.py:4056-4062`)
without a `surface` argument, so it defaults to `"terminal"`
(`remote.py:782`) and `_can_go_cold` is **False**. A parked source that sees
EOF therefore enters runtime-death recovery and, per
`remote.py:3768-3798`, chases a record for `COLD_FALLBACK_S = 8.0`
(`remote.py:129`) and then — because it cannot go cold — keeps retrying
forever while the legacy contract tries to **take over** the session
(`no_takeover` raises by construction, `app.py:4040-4043`, so it loops).
A runtime-initiated EOF against a parked sidebar source is a redial storm.

### Confirmed: parked sources install neither callback

I verified the parent's suspicion. `_lease_sidebar_source`
(`app.py:4027-4085`) installs the event controller (4077-4079) and the
frontend watch (4080) and **nothing else**. The three lifecycle callbacks —
`set_takeover_callback`, `set_stopped_callback`, `set_refresh_callback` — are
installed only in `_adopt_session` (`app.py:6025-6041`), for the CURRENT
session. `set_went_cold_callback` is defined (`remote.py:3711`) but has no
caller anywhere in `local_operator/`.

So a parked source has no refresh path (no churn loop) but also no cold path
(no graceful recovery) — it falls into the redial storm above. Both defects
point the same way: **the viewer must initiate, because only the viewer knows
what it is showing.**

---

## 3. Recommendation

### 3.1 Fix defect 1 by deleting the LRU clause and decoupling the two caches

Drop `and source.session.session_id not in self._sidebar_presentations` from
`_sidebar_source_releasable` (`app.py:4645`). Releasing the *source* (socket,
subscription, runtime pin) while retaining the *presentation* (widgets) is
what the two caches were always supposed to mean.

Teardown is already correct for this. `_release_sidebar_source`
(`app.py:4648-4679`) saves the draft, unsubscribes the frontend, disposes the
controller and calls `session.dispose()` — which closes the client socket
(`remote.py:4978-4983`, inside `dispose` at 4964) and nothing else. The runtime
then sees a client disappear, `attach_clients()` drops to 0, and **the
existing 3-second drain reaps it** (`process.py:341-359`). No new wire op, no
new frame, no protocol change, and the runtime keeps full authority over its
own exit via the unchanged `_should_exit`.

**One consequence must be handled, and it is the real cost of this fix.**
`SessionPresentation.source_token` is compared in
`_sidebar_presentation_current` (`app.py:3920-3921`), and `token` is a
per-`SessionInteraction` UUID (`session_interaction.py:115`) whose docstring
says it "identifies this exact owner-facing incarnation"
(`session_interaction.py:3-6`). Releasing a source and re-leasing it mints a
**new** `SessionInteraction` and therefore a new token, so the retained
presentation can never hit again — it is dead weight that will be rebuilt
anyway.

That is not a reason to keep the pin; it is the measurement that tells us
which of two designs to pick:

- **(A) Release the source and evict its presentation together.** Honest, one
  line of extra bookkeeping, and it makes the cache mean what its comment
  says. Cost: every parked conversation becomes a cold rebuild on
  switch-back. The constant's own measured data says what that costs —
  246 ms loop-CPU per switch at capacity 4 versus 59-65 ms on a hit
  (`app.py:1360-1362`). That is a real, user-visible regression in exactly
  the interaction `RETAINED_PRESENTATIONS` was raised from 4 to 12 to fix,
  and it is the strongest objection to the viewer-side fix.

- **(B) Release the source, keep the presentation, and let it re-validate.**
  Requires the retained presentation to survive a source swap — i.e. the
  currency check keys on something that survives re-leasing rather than on
  the incarnation token.

**I recommend (A) for the first PR, and (B) only if measurement shows the
regression matters.** Reasons: (A) is strictly smaller and cannot be wrong —
it removes a pin and takes a known, bounded latency cost on a path that was
*already* cold for any conversation past the LRU capacity. (B) changes a
correctness-critical cache-validity predicate whose docstring
(`app.py:3898-3917`) enumerates four distinct staleness classes that must
still miss; getting that wrong paints a stale or wrong transcript, which is
far worse than a 250 ms switch.

The evidence that would settle (A)-vs-(B): instrument switch-back loop-CPU
across a realistic 8-12 conversation working set with the clause removed,
using the same method that produced the `app.py:1360-1362` numbers. If the
p90 lands near the 246 ms cold figure on conversations the user actually
alternates between, do (B) as a follow-up. My prior is that it will matter
less than the raw number suggests, because a released source's rebuild no
longer competes with 11 sibling sockets delivering runtime deltas — but that
is a prior, not a measurement, and I would not ship (B) on it.

A cheaper middle path worth pricing during implementation: keep a **small**
presentation retention for the 2-3 most recent conversations and release
sources beyond that. That preserves the alternating-pair case (the common
one) while capping pinned runtimes at a number a laptop can hold. It is a
constant change plus the clause removal, not a new mechanism.

### 3.2 Defect 2: do NOT build the 5-minute runtime-side clock

Answering question 1 directly — of the three options:

- **(a) Viewer sends periodic activity heartbeats.** Reject. It is a wire
  change (`ControlOp` in `mobile/types.py:258-313`, and the additive-op
  comments at 234-238 and 296-299 show the bar), and the task's own framing
  names the fatal flaw: a viewer that periodically says "I am alive" is
  precisely what defeats an inactivity reaper. To be correct it would have to
  send only on *human* activity, which means the viewer already knows the
  answer — at which point it should act on it locally rather than telling a
  remote process to act for it. Worse, whatever the runtime concludes, it can
  only act via `retiring` (churn loop, §2) or EOF (redial storm, §2).

- **(b) Runtime-side, derived from existing evidence.** Reject — there is no
  such evidence. `last_seen` (`server.py:426`) does not move for an idle
  viewer, and per §1 nothing else periodic exists on a terminal attach. This
  option is not available without (a).

- **(c) Viewer-side release.** **Recommend.** The viewer is the only party
  that knows a conversation is parked behind a `100vw` offset
  (`app.py:4196`) rather than being read. Releasing drops the socket; the
  runtime's own unchanged predicate then decides its fate. Fail-closed is
  free: the five surviving clauses of `_sidebar_source_releasable` all keep
  the source on any doubt.

**So: fixing defect 1 subsumes the leak, and the 5-minute runtime-side clock
is not needed for it.** Question 3 asked me to say this plainly, so: the two
mechanisms are not complementary for the reported symptom. The 1.9 GB was
caused entirely by defect 1. A 5-minute clock would not have fixed it any
faster than the 3-second drain does once the socket drops, and building both
means maintaining two answers to one question.

### 3.3 The residual case, and my recommendation on it

Question 3 also asks what a runtime-side clock still covers if the viewer
releases properly. Honestly enumerated, the residual is:

1. **A wedged or `SIGKILL`ed viewer.** Its socket dies with the process, so
   the OS closes it and `attach_clients()` drops — the 3 s drain handles it.
   **Not residual.** A viewer wedged but *alive* (event loop frozen, socket
   still open) is genuinely residual.
2. **A viewer on another machine.** No such client exists today; the listener
   is loopback-only (`server.py:1190-1195`). Not residual until it does.
3. **The desktop lease.** Already time-bounded — `DESKTOP_WATCH_LEASE_S =
   45.0` (`runtime/types.py:60`), enforced in `_desktop_lease_live`
   (`server.py:1553-1557`) and renewed by the host
   (`desktop_sessions.py:354-360`). An abandoned desktop surface stops
   counting after 45 s. **Not residual — this case is already solved, and it
   is the precedent that a lease, not a heartbeat, is how this codebase
   answers "is anyone still there".**
4. **The phone.** `phone_watchers` is deliberately outside the predicate
   (`process.py:272-275`) and `watch`/`unwatch` already mark the 0↔N
   transition (`types.py:284-291`, `server.py:1693-1697`).

That leaves **one** genuinely residual case: a live-but-wedged terminal
viewer holding an open socket. **I recommend not building a wire protocol
change for it**, because the only actions a runtime could take on that
conclusion are the two broken ones from §2 — and a wedged viewer is exactly
the client least able to handle a `retiring` re-engage or an EOF recovery
correctly.

If the operator still wants a runtime-side backstop after §3.1 ships, the
smallest honest version is a **lease, not a heartbeat**, reusing the desktop
pattern verbatim: an attach client's pin expires unless renewed, and the TUI
renews only for the session it is actually displaying. That is still a wire
addition, but it reuses an existing shape, it is fail-closed in the right
direction (a viewer that cannot renew was not usable anyway), and it needs no
`PROTOCOL_VERSION` bump under the additive rule (`types.py:234-238`) — an old
runtime ignores the op and behaves exactly as today. I would build it only
after §3.1 is deployed and measured, because I expect the residual to be
approximately zero occurrences per week, and shipping it first is how the
churn loop in §2 gets discovered in production instead of in this note.

### 3.4 What resets the clock, if one is ever built

Recorded for completeness (question 2), and stated in the existing
vocabulary rather than a new one. `is_busy()` (`owned.py:688-740`) already
covers turns, compaction, subagents, jobs, queued prompts and parked gates,
and is checked first and alone (`process.py:277-279`).

**Must reset** (human intent, none of which `is_busy` sees):
a keystroke into the composer for that session; the session becoming the
displayed one; a routed slash command; an answered gate; a scroll or
navigation action in its transcript.

**Must NOT reset** (presence, not attention): mere socket establishment; the
1 Hz band repaint poll (`app.py:3424`) and the 2 s sidebar refresh
(`app.py:5772`); any `daemon`-class client (`process.py:234-240` and
`server.py:1588-1596` both already argue this); the phone SSE watcher
(`process.py:272-275`).

The critical observation: **every entry in the "must reset" column is a TUI
event, and none of them is on the wire today.** That is the same fact that
makes (a) and (b) unavailable in §3.2, restated. It is also why the clock, if
built, belongs on the viewer side of the seam — which is §3.1.

---

## 4. Interactions

**`WARM_WINDOW_S = 90` wakes** (`process.py:78`). Untouched. Term 2
(`process.py:280-281`) is checked before term 3, so a released socket cannot
reap a runtime with a wake due inside 90 s. Question 6's "wake fires while
nobody is attached" case is exactly what this constant exists for.

**Build refresh / `retiring`.** Untouched, and this is important: `retiring`
keeps its single current meaning ("a newer build is on disk; engage a
successor"), which is what makes `_on_runtime_refreshed`'s unconditional
re-engage (`app.py:12502`) correct. Overloading it with "you were idle" is
what creates the churn loop. **An idle reap must use no frame at all** — the
viewer closes its socket and the runtime exits quietly through the existing
`_clean_exit` (`process.py:287-301`). That is the answer to question 4: not
`retiring`, not EOF-from-the-runtime, but *viewer-initiated socket close*,
which is neither.

**`retire_if_pristine`.** Untouched. Already correct and already re-checks
under itself (`server.py:1698-1747`, `_retire_if_pristine` at 1881+, with the
post-broadcast re-check at 1930-1948).

**Desktop lease.** Untouched; already handled inside `attach_clients`
(`server.py:1547-1551`).

---

## 5. The pristine / "Untitled conversation" case

Question 5: **yes, it needs one thing beyond calling the existing op — but
the op itself needs no change.**

`retire_if_unused` (`remote.py:4927-4962`) is called from exactly three
places, all of which ABANDON the *current* session: `/resume` onto a saved
session (`app.py:8291`), `/resume` onto a live one (`app.py:9496`), and
unmount/quit (`app.py:17642`). **No sidebar-leave path calls it.** So a user
who clicks onto an empty "Untitled conversation" from the sidebar and clicks
away leaves a pristine runtime that only the 3 s drain can collect — and
today it cannot, because defect 1 keeps the socket open, which is precisely
what the second committed repro
(`test_an_empty_conversation_visited_and_left_retires_its_runtime`) asserts.

The fix is to offer the runtime back on the sidebar-leave path too, before
`session.dispose()` in `_release_sidebar_source` (`app.py:4679`), guarded by
the same `getattr` probe the existing caller uses (`app.py:12949-12951`).
Nothing else is required: the op already refuses when another viewer is
attached (`server.py:1743-1745`) or anything durable exists
(`is_pristine`, `owned.py:809-894`, every probe failing closed), and its
failure mode is "the runtime stays up and the drain gets it"
(`remote.py:4941-4944`).

Strictly speaking §3.1 alone makes the empty case work — the socket drops and
the drain reaps in 3 s. Adding the offer makes it *immediate* rather than
3 seconds later, which is the same rationale the op already carries
(`server.py:1739-1742`). Worth doing in the same PR; not worth doing alone.

---

## 6. Risks

| scenario | what keeps it alive |
|---|---|
| Reading a long conversation for 6 min, then typing | It is the CURRENT session. `_sidebar_source_releasable` refuses on `source is not self._interaction` (`app.py:4638`), which is checked before any other clause and is not time-based. Nothing in this design can reap the displayed session — that is the entire reason I reject the runtime-side clock. |
| Session parked on an approval | Three independent guards: `is_busy()` counts a parked gate as a running turn (`owned.py:692-695`, `724-727`), so `_should_exit` refuses; `has_pending_gate_reply` refuses release (`app.py:4642`); `gate_draft` makes it `retained_for_local_work` (`session_interaction.py:170`). |
| Detached background job | `is_busy()` counts running jobs (`owned.py:728-737`, uncertainty failing closed at 736-737) and background tasks (738-739). The runtime refuses to exit regardless of viewer state. Note this is genuinely *stronger* than the viewer's own `retained_for_auto_work`, which is gated on `approve_all` (`session_interaction.py:186`) — but that only means the viewer may release a socket, never that the runtime dies. |
| Phone attached | An interactive phone attach dials as `"attach"` (`process.py:272-275`) and counts in `attach_clients()`; the SSE watcher registers through `watch` (`server.py:1598-1608`). A TUI releasing its own socket does not touch either. |

**The risks I would actually watch during rollout**, none of which the table
above covers:

1. **Switch-back latency regression.** The known cost of §3.1(A). Watch p90
   switch loop-CPU against the `app.py:1360-1362` baseline. Mitigation is the
   small-retention middle path in §3.1.
2. **Release/re-lease races.** `_release_sidebar_source` already yields at
   its draft save and re-checks the predicate afterwards
   (`app.py:4664-4667`), and `_start_sidebar_connection`'s done-callback
   re-checks after the connection task settles (`app.py:5161-5162`). Removing
   the LRU clause widens the window in which a source can be released while a
   prepare is in flight. `preparations` (`app.py:4641`) is the guard;
   verify it is incremented before any await on every entry path
   (`app.py:4036`, `4073`).
3. **Sidebar prewarm churn.** `PREWARM_PER_REFRESH = 2` (`app.py:1375`) warms
   two entries per 2 s poll (`app.py:5772`). With sources releasing eagerly,
   confirm prewarm does not immediately re-lease what release just dropped —
   that would trade a static leak for a spawn treadmill, the same failure
   mode as §2 by a different route. `_sidebar_unretainable`
   (`app.py:4754-4767`) is the existing damper, and it has a wrinkle that
   matters here: its docstring notes "a source that has been released has no
   readable stamp; the refusal is then trusted until the sidebar closes"
   (`app.py:4750-4752`). Releasing sources eagerly therefore makes refusals
   stickier than they are today. Benign by that docstring's own argument (it
   suppresses only speculation, never an explicit click), but verify.
4. **Runtime spawn cost on the alternating pair.** If a user ping-pongs
   between two conversations faster than the 3 s drain, each switch may pay a
   cold start. The drain's own 3 s window absorbs a fast return; a slow
   alternation (>3 s, <1 min) is the worst case. Measure before assuming it
   is fine.

---

## 7. Summary of the recommendation

1. Remove the `_sidebar_presentations` clause from
   `_sidebar_source_releasable` (`app.py:4645`); evict the presentation with
   the source (option A). This alone fixes the 1.9 GB leak, via the existing
   3-second drain and with no wire change.
2. Call `retire_if_unused` on the sidebar-leave path in
   `_release_sidebar_source` (`app.py:4648`), probed, best-effort — the
   "Untitled conversation" case, made immediate instead of 3 s later.
3. **Do not build the 5-minute runtime-side reaper now.** It does not address
   the reported leak, its only residual case is a live-but-wedged local
   viewer, and both frames it could act through are broken for a displayed
   session (churn loop / redial storm). If a backstop is wanted later, build
   a *lease* in the shape of `DESKTOP_WATCH_LEASE_S`, not a heartbeat, and
   only after (1) is measured in production.
