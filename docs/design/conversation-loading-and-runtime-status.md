# Design: conversation loading and runtime status, instant and independent of runtime compute

Status: **proposal — analysis and design only, no production code.** Written by
the architect against `origin/main` `a3228afd4` (v0.63.0), in the worktree
`~/local-operator-worktrees/convload-arch-1790433460`. Every claim about current
behaviour is cited to `file:line` **at that revision** or to a command whose
output is quoted; every number carries the load average it was taken at, because
on this host a wall time without a load figure is weather, not evidence.

Read this first, because it changes what the next phase should build:

> Three design efforts have already been at this surface
> (`session-load-central-cache.md`, `attached-interface-signal.md`,
> `full-history-audit-window.md`), and **most of what the operator is asking for
> is already built and working**. The measured gap is narrower and more
> specific than the request implies:
>
> 1. **First paint is already decoupled from the runtime on both interfaces.**
>    It is at the budget (200-297 ms p50) and needs ~2x headroom, not a redesign.
> 2. **The live overlay is not slow because of compute contention.** It is slow
>    because a cold runtime is *spawned and booted* (1.3-1.5 s quiet, 4.4-4.9x
>    over budget) while a warm one answers in 30-80 ms (4-10x under budget). This
>    is a **warmth** problem, and the warmth mechanism already exists with a cap
>    of 4.
> 3. **The cap cannot be raised enough to fix it inside the operator's own
>    memory envelope.** Therefore the load-bearing change is *make the cold path
>    cheap*, not *keep more warm*.
> 4. **The TUI has a real, different coupling the server-side rigs cannot see**:
>    its switch cache deliberately switches itself OFF while a session is
>    streaming — i.e. exactly when several runtimes are going — and its
>    speculative warm-up does 93 ms of synchronous work on the UI event loop,
>    twice per one-second refresh.
>
> The single most important correction to the operator's framing: **conversation
> loading is not being slowed down by runtime compute.** It is slowed down by
> (a) a cold process that has to exist before the *live* half can be answered,
> and (b) on the TUI only, by work the UI does to itself in proportion to how
> many sessions are live. Those are two different fixes and neither is "ignore
> the runtimes".

---

## 0. Table of contents

1. [The four asks, adjudicated](#1-the-four-asks-adjudicated)
2. [The operator's hypotheses vs the code](#2-the-operators-hypotheses-vs-the-code)
3. [Method, and what these numbers are worth](#3-method-and-what-these-numbers-are-worth)
4. [Where the coupling actually is](#4-where-the-coupling-actually-is)
5. [Is <300 ms reachable as-is?](#5-is-300-ms-reachable-as-is)
6. [The not-stale contract](#6-the-not-stale-contract)
7. [Push versus poll](#7-push-versus-poll)
8. [DSH as a reference implementation](#8-dsh-as-a-reference-implementation)
9. [Ranked changes](#9-ranked-changes)
10. [What I would not do](#10-what-i-would-not-do)
11. [Open questions, and the evidence that would settle them](#11-open-questions-and-the-evidence-that-would-settle-them)
12. [What I ran](#12-what-i-ran)

---

## 1. The four asks, adjudicated

| # | The ask | State on `a3228afd4` | Evidence |
|---|---|---|---|
| 1 | "Optimize runtime attach to interfaces and reduce the amount of repeat work" | **Partly built, partly real.** The attach *bind* is off the owner's loop and works in all four owner states; the lock-held-across-attach defect is fixed. What is genuinely repeated is (a) a **cold spawn + whole-journal parse** on every attach whose runtime was reaped, and (b) on the TUI, a **256 KB journal parse per switch/prewarm** that goes around the existing page cache. | §4.2, §4.4, §4.5 |
| 2 | "Can the UI cache across conversation switches, and where does the cache belong?" | **Already answered, twice.** The desktop has a process-wide page cache (`session/page_cache.py`) plus a resident-bridge pool; the TUI has a 12-slot prepared-presentation cache plus a 1 Hz speculative warm-up. The open question is not *whether* to cache but **where the existing caches do not reach** — §4.5 names three holes. | §4.5 |
| 3 | "Study DSH and take what applies" | Done — §8. Four patterns transfer (list reads that never open a body; live-preferred but non-activating reads; an explicit `asOfSeq` watermark per value; a cache that may be *stale but never ahead*). Three do not, and the reason is that DSH's projection registry folds events **synchronously in one process**, which lop's per-session runtime processes cannot do. | §8 |
| 4 | "Instant conversation loading, under 300 ms, measured" | **Met on a warm runtime (30-80 ms), missed by 4.4-4.9x on a cold one (1.3-1.5 s), and first paint is at the budget (200-297 ms p50) rather than under it.** Both halves are measured; both have a named cause. | §5 |

### What is already solved and must not be re-proposed

* **The pool-wide lock is gone from the slow halves.**
  `DesktopSessions.session` (`server/utils/desktop_sessions.py:6253`) takes the
  pool lock only for the handout reservation, the cached lookup and the
  insertion; the cold lookup and the attach run outside it. Its own docstring
  carries the before/after: a 642-byte session answered in **829 ms** while the
  261 MB open was in flight, against **24 requests completing inside the open**
  (median 70 ms, worst 308 ms) after. Its evidence is committed at
  `docs/evidence/session-load-central-cache/`.
* **The attach bind is off the owner's workload loop, for every attach.**
  `subscribe_frontend` carries `@_on_session_loop`
  (`session/runtime/serving.py:363,2413`), the serving plane gives it
  `_ONLOOP_BIND_GRACE_S = 0.1` (`session/runtime/server.py:215,3194`) and past
  that binds through `subscribe_frontend_nowait`
  (`server.py:3226-3364`, with the `frontend_off_loop_binds` counter). Coder
  measured it with the grace off and the owner's loop blocked by a 2 s
  synchronous step: **sync 1.1/1.9 ms against 10.5/15.5 ms idle**. "A busy owner
  must not delay an attach" is already true at the bind.
* **The read path does not wait 2 s for a bind.** `READ_ATTACH_BUDGET_S = 2.0`
  (`session/attached.py:203`) is the *background* task's budget; the route stops
  waiting at `READ_FIRST_FRAME_GRACE_S = 0.05` (`session/attached.py:220`, used
  at `server/utils/desktop_sessions.py:1394-1401`). The earlier reading of this
  as a 2 s critical-path wait was wrong and is corrected here.
* **The status list does not do a per-row process census.**
  `registry.scan` (`session/runtime/registry.py:549`) spends **at most one `ps`
  fork for a whole quiet population** (`procstate.zombie_states`), and
  `decorate_rows` (`session/catalog.py:908`) reads exactly two things for a
  whole list. `cached_session_rows` (`session/catalog.py:1387`) memoises each
  row on the transcript's stat and resolves the store's realpath once, not per
  row.
* **A machine-wide notification feed exists that spawns nothing.**
  `server/utils/desktop_feed.py` costs **four `os.stat` calls per 100 ms tick**
  and "acquires no bridge and spawns no runtime".

---

## 2. The operator's hypotheses vs the code

The operator asked to be told the best approach rather than to have his ideas
implemented. Four of his six hypotheses survive; two do not.

| # | His words | Verdict | Why |
|---|---|---|---|
| H1 | "conversation loading should not be affected by runtimes" | **SURVIVES, with a correction.** The *durable* half is already independent of runtimes on both interfaces. The *live* half is not affected by runtime **compute** — it is affected by whether a runtime **exists at all**. A busy owner does not delay a read (off-loop bind, 1.1-1.9 ms); a reaped owner costs 1.3-1.5 s of spawn and boot. | §4.2, §4.3 |
| H2 | "when several agent runtimes are going, things slow down with new message sending and conversation loading" | **SURVIVES on the TUI, REFUTED on the desktop server.** Desktop reads are not serialised by fleet occupancy (the pool lock fix; 24 requests complete inside a 2.45 s open). The TUI is a different story and it is measurable in the tree: the prepared-presentation cache is *disabled for any session that is streaming*, and the speculative warm-up costs 93 ms of synchronous work on the event loop, twice per second while the sidebar is open. More live sessions ⇒ more of both. | §4.4 |
| H3 | "make conversation loading and status instant and independent of the compute burden of the runtime" | **SURVIVES.** This is precisely the durable-read / live-overlay split, and the protocol already carries the fields (`cold`, `cold_reason`, `attaching`). | §6 |
| H4 | "not lagged or behind the actual current state of the runtime" | **SURVIVES and is currently VIOLATED, measurably.** With the owner `SIGSTOP`ped, a re-open reported `cold: false` (a live-looking session), `POST /warm` returned **200 `{"state": "warm"}`**, and only `POST /messages` told the truth — after **15.0-15.6 s**, as a 503 `runtime_unreachable`. The registry's own resolution is `HEARTBEAT_TIMEOUT_S = 45.0` (`session/runtime/types.py:370`) against a 15 s beat, so "live" can be ~45-60 s behind the truth. | §6.1 |
| H5 | "the UI can cache across conversation switches" | **SURVIVES, already built.** What is missing is coverage, not the cache: the TUI switch path bypasses `session/page_cache.py` entirely, and its own cache misses on `streaming`. | §4.5 |
| H6 | "a read model / projection the UI reads without touching the runtime" (his candidate list) | **SURVIVES as the right shape, but it is not the win it looks like** — the desktop already *is* a read model over the durable journal, and it is already fast (200-300 ms). Adding a second projection layer in front of a 200 ms read does not reach 300 ms reliably. The win is at the *live* edge: publish the runtime's own cold answer instead of no answer. | §9 R1 |

---

## 3. Method, and what these numbers are worth

**Rig.** `scripts/bench_desktop_open_attach.py` (already in the tree, written for
the 2026-09-22 desktop load diagnosis) driving a **real `local-operator serve`**
started from this worktree, against fresh APFS clones of fixture journals, with a
fresh `HOME`/`LOCAL_OPERATOR_CONFIG_DIR` per run, `LOP_*`/`CMUX_*` stripped, its
own port and its own desktop token. Command in §12.

**Conditions, stated because they bound the claims.** Host load average
**74-116** for the run below; this host carries ~25 concurrent agent sessions and
sat at load 82-92 at the time. **A 300 ms target measured at load 90 is not a
laptop measurement**, and the honest use of these numbers is (i) the *shape* —
which term dominates, and how it scales with journal size and with runtime count
— and (ii) *structural counts*, which are load-independent. Where a wall time is
quoted it is accompanied by its load.

**Structural instruments used instead of wall clock where possible:** the page
cache's own decoded-row counters, `read_transcript_page`'s byte-windowed cursor
locator, the registry's batched `ps` fork, and the counts already recorded in the
tree's own docstrings (each of which names the measurement it came from).

**A rig-instrumentation experiment that did NOT happen, and its consequence.**
I built an audit-hook counter (file opens by basename, directory scans, `stat`
calls, `subprocess.Popen`, and `asyncio.to_thread` queue latency) installed
through a `.pth` in the worktree venv, intending to count the per-open repeat
work rather than characterise it. It was never run: the peer coder was already
producing the repeat-work inventory with direct probes, and duplicating it would
have cost a second `serve` fleet on an already-saturated host. **The counts in
§4.6 therefore come from the tree's own counters and the prior design doc's
audit, not from my instrument** — say so rather than implying otherwise. The rig
is in `$LOCAL_OPERATOR_SCRATCHPAD/audit/` if the next phase wants it; it is
deliberately not left installed (the `.pth` was removed at the end of this
phase).

---

## 4. Where the coupling actually is

The task named six candidate mechanisms. Five are refuted with evidence; one is
the answer, and it has two parts (server: spawn; TUI: self-inflicted loop work).

### 4.1 The candidate table

| Candidate | Verdict | Evidence |
|---|---|---|
| Blocking shared resource (sqlite/WAL, `flock`) | **REFUTED as the dominant term.** Present at a bound, not on the critical path: the snapshot's attention read is capped at `ATTENTION_SNAPSHOT_WAIT_S = 0.05` (`server/utils/desktop_sessions.py:291`) and serves the last-known value on timeout. | `desktop_sessions.py:2690-2713` |
| The asyncio loop starved by synchronous work | **REFUTED on the server, CONFIRMED on the TUI.** The server's cold lookup and attach are off-loop (`to_thread`, `_locate_flights`); the TUI's `_prepare_sidebar_session` is 93 ms **on the event loop**, twice per refresh. | `events.py:620-622` |
| A subprocess census on the load path | **REFUTED.** `registry.scan` spends one `ps` fork per *whole quiet population*, none on the healthy path; the catalogue reads two sources per list. A second per-row `ps` was explicitly removed (QA round 2 Q5: 201 forks at 200 zombie records). | `registry.py:549-700`, `catalog.py:908-1000` |
| A synchronous whole-transcript decode | **CONFIRMED — but not where the operator would guess.** It is not on the *paint* path (pages are 1-4 ms, backward, page-cached); it is on the **cold runtime's boot** (`Transcript.__init__`, `transcript.py:1087` and its streaming parse at `:1161`) and inside `create_session` (`serving.py:7896`). | §4.2, §4.3 |
| An O(n) scan per call | **CONFIRMED, bounded, and mostly already fixed.** Row scans per poll were replaced by stat-keyed memoisation; the deep-page cursor walk is now byte-located. | `catalog.py:1387-1450`, `transcript.py:624-680` |
| A lock held across an `await` | **REFUTED — this was the previous defect and it is fixed.** `acquire()` holds `self.lock` only around `_ensure_facade()` and single-flight creation. | `desktop_sessions.py:1326-1440`, `:6253-6640` |

### 4.2 The mechanism that is actually there: a cold runtime is spawned and booted

Two independent measurement chains agree on the same number, from different
directions.

**Chain 1 — the desktop click (mine).** `scripts/bench_desktop_open_attach.py
--scenario attach` measures `t_attached`, the first SSE frame whose `cold` is
false, after a `POST .../watch {visible:true}`. The `watch` POST itself returns
in 7-201 ms; the wait is inside the engage. My run, load 74-116, n=5 per profile:

| profile | journal | rows | `t_attached` p50 | max |
|---|---|---|---|---|
| tiny | ~25 KB | 21 | 1 823 ms | 2 189 ms |
| p50 | ~0.7 MB | ~250 | 2 244 ms | 3 886 ms |
| p90 | ~1.6 MB | ~900 | 3 049 ms | 3 503 ms |
| p99 | ~6 MB | ~2.4 k | 1 810 ms | 1 869 ms |
| **xl** | **~200 MB** | **~18 k** | **6 124 ms** | **11 800 ms** |

Load for the tiny/p50 rows was 74-85; for the xl rows 98-116.

**Chain 2 — the same session attached three times while its runtime stays alive
(coder).** Load 66.9-70.5, n=3x3:

```
cold attach (leg 0, runtime must spawn): 1470 / 1303 / 1321 ms
warm attach (legs 1-2, same runtime alive): 61.0, 78.7 / 57.8, 57.4 / 31.9, 29.6 ms
```

and on the warm legs `t_attached == t_snapshot`: the snapshot frame already
carries `cold: false`, so there is no second wait to design away.

**Chain 3 — the tree's own words.** `session/runtime/process.py:104-124`
documents the same cost and the mechanism that already avoids it:

> "closing a conversation and opening it a minute later paid a cold spawn
> (measured **1.1-1.5 s of wall time and ~0.9-1.0 s of CPU** on this fleet) for
> session state that was already in a live process's memory. With this window the
> re-open is a live attach (measured **16/22 ms p50/p95** on a 0.56 MB session and
> 55/101 ms on a 48 MB one at load ~98)."

**Two terms, and they need different fixes.**

* **A fixed term — spawn + boot.** Coder's probe: the pid appears
  **126.9-248.4 ms** after the click, and the engage completes at
  **1303-1470 ms**. So ~0.1-0.25 s is "make a process exist", and **~1.05-1.25 s
  is boot inside it**. The boot is `create_session`
  (`serving.py:7896`), which the tree itself records as "one long SYNCHRONOUS
  stretch" — its own measurement is **median 165.7 ms of contiguous loop stall
  for `_prepare` alone**, 77.3 ms with imports warm (`serving.py:7850-7875`).
* **A size-dependent term — the whole-journal parse.**
  `Transcript.__init__` (`session/transcript.py:1087`) streams and JSON-decodes
  **every row of the journal** when the file exists (`:1161-1166`, one row at a
  time to keep the peak bounded). Recorded baselines: **2 286.7 ms on a 261 MB
  journal, 248.2 ms on a 107.6 MB one** (`docs/evidence/session-load-central-cache/README.md`,
  measured at load 176-312), and 1.37 s / 231 MB in
  `docs/design/full-history-audit-window.md:185-191`.

**That second term is why my `xl` attach is 6.1 s p50 while `p99` (6 MB, same
load band) is 1.8 s.** Attach is *not* flat in transcript size; it is flat *only
up to the size where the parse stops being noise*. Coder's "flat 1.4-2.3 s"
result on tiny/p50/p90/p99 is correct and does not generalise to the operator's
261 MB conversation, which is the one he is complaining about. **This is the
reconciliation of the two measurement sets, and it matters for the ranked list:
warmth fixes the fixed term; only the parse fix touches the size-dependent one.**

### 4.3 The warmth policy, and the ceiling on it

The mechanism that avoids the spawn is already implemented and already on:

```
DEFAULT_KEEP_ALIVE_SECONDS = 300     session/runtime/process.py:124
DEFAULT_KEEP_ALIVE_MAX     = 4       session/runtime/process.py:156
KEEP_ALIVE_SCAN_S          = 5.0     session/runtime/process.py:163
```

A runtime a viewer has left stays resident for five minutes; a machine-wide LRU
keeps at most **four** idle clientless runtimes. The operator's
`~/.local-operator/config.yml` has **no `runtime` section** (read-only check), so
both defaults are in force.

**Coder's read-only census of the operator's real store** puts the cap's coverage
in perspective:

```
sessions_total          12 258
transcript bytes total   7 320 MB
touched last 5 min           34   (8 user sessions + 26 subagent runs)
touched last 10 min          36   (8 user + 28 subagent)
touched last 60 min          66
touched last 6 h            154
```

**`keep_alive_max = 4` covers half his five-minute user working set.** And the
cost per idle runtime is measured at **~73-130 MB RSS**
(`process.py:135`, `test_runtime_keep_alive`); the standby mechanism measures
**47-158 MB, median ~130 MB** (`session/runtime/standby.py:109`):

| cap | idle-runtime RSS | inside the operator's stated ~200-300 MB envelope? | covers his 5-min working set? |
|---|---|---|---|
| 4 (today) | ~292-520 MB | **already at/over** | 4 of 8 user sessions |
| 6 | ~438-780 MB | no (1.5-2.6x) | 6 of 8 |
| 8 | ~584-1 040 MB | no (2-3.5x) | all 8, no LRU headroom |
| 12 | ~876-1 560 MB | no (3-5x) | all 8 with headroom |

Two facts that make this less bad than it looks: the cap is charged **only to
idle clientless runtimes**, so the 26-28 subagent runs per five minutes do not
consume slots; and a busy runtime is never kept alive by this policy. But the
arithmetic is the arithmetic: **the cap cannot be raised to cover the working set
inside the memory budget the operator himself gave us.** At best it buys 2-3
extra warm conversations (cap 6) at ~1.5-2.6x the envelope.

**Therefore the load-bearing change is to make the cold path cheap, not to hold
more warm.** If a cold engage is 150 ms rather than 1.4 s, a 50% miss rate stops
mattering and the memory question mostly evaporates.

### 4.4 The coupling the server rigs cannot see: the TUI, and itself

`session/page_cache.py` and the desktop rigs measure the *server* plane. The
operator's "the UI" includes the TUI, and the TUI has two costs the server rigs
cannot see: one that scales with **how many sessions are live**, and one that
scales with **how many sessions exist at all**.

**(a) The switch cache is not self-disabling — it is disabled BY the target being
busy, and that is the operator's case.** `_sidebar_presentation_current`
(`tui/app.py:6552`) deliberately *accepts* durable rows appended while a session
was parked; the docstring records that the strict all-stamps-equal form "made
nearly every click cold" and that this was fixed. Coder measured the two arms on
29 switches: **19 of 19 revisits of a non-streaming, gate-free session hit the
cache at 0.21 ms**, while a switch to a **streaming** target returns `False`
(`:6585-6586`) and pays **31.91 ms**. So the cache works, and the arm that matters
is the streaming one — the one that fires exactly when several runtimes are
going.

Two corrections to how I first read this, both worth keeping:

* It is **not** "the cache switching itself off". The other miss arms are
  deliberate: `:6582-6583` (a pending gate is presentation-only state a delta
  cannot supply) and `:6587-6588` (a moved `display_history_revision` means
  compaction rewrote rows already painted). Only the streaming arm is arguable.
* The cost is **31.91 ms, not a full switch's worth of work.** My first framing
  implied a rebuild on the order of the whole switch; it is ~32 ms of prepare
  against a 0.21 ms hit. That is a jank-scale item, not a load-time item — it
  belongs below the census in (b), not above it.

The streaming arm is still worth fixing, and the shape is unchanged: the commit
path *already* projects appended durable rows onto a parked presentation
("the reveal path already projects appended rows at commit"), and the owner's
canonical live seed is already available to rebuild the live block
(`EventController.restore_live_projection`, `tui/events.py:697`). The live block
is a delta the commit already knows how to apply; refusing the whole cache for it
throws away the cached base — for ~32 ms.

**(b) The larger TUI repeat cost is a whole-store census every 2 seconds, and it
scales with the store, not the runtime — which is how it couples to the runtime
count anyway.** `on_mount` arms `set_interval(2.0, self._refresh_sidebar,
pause=True)` (`tui/app.py:10228`, gated on the sidebar being open). It calls
`load_catalog(root, ...)` (`:9903`) inside a worker
(`await asyncio.to_thread(collect)`, `:9939`), and coder counted that poll on a
12,200-directory store — the operator's is **12,258**:

| | the 2 s poll (`load_catalog`) | a bare `cached_session_rows(root)` |
|---|---|---|
| syscalls | **21,205 `stat` + 3 `scandir`** | 12,802 `stat` + 1 `scandir` |
| wall | 276.5-311.8 ms | 84.5-136.2 ms |
| CPU | **221.5-227.1 ms** | 72.5-78.6 ms |
| first poll | 3,234.7 ms / 1,537.0 ms CPU / 29,604 stats | — |
| reads | `open=0`, `read_bytes=0` — the cost is entirely the census | — |

**Where the 8,403 `stat` calls and ~145 ms of CPU actually are — a correction to
my own first reading.** The per-row memo is *already* on the poll's path:
`load_catalog` (`session/catalog.py:1818`) ends in `_hydrate` (`:1914`), whose
loop is `for row in cached_session_rows(...)` at **`session/catalog.py:1930`**.
So "the poll does not call the memoised reader" was wrong — the call is one level
down, which is why grepping `app.py` for the symbol finds nothing. What survives,
and is the real finding, is one level further up: **`_ranked_candidates`
(`session/catalog.py:1518`) runs `_scan_sessions(directory, strict=True, ...)`
(`:1566`) on every poll**, plus decoration and ranking. A bare
`cached_session_rows(root)` skips that scan; the poll cannot, because the TUI
needs `include_subagents`, `pinned_hidden_ids` and the ranking.

**The change is therefore to memoise or gate `_ranked_candidates`' scan on a
store-change key** — not to swap in a different reader. The module already does
this three times (`_ROW_CACHE:1206`, `_BIRTH_MEMO:1243`, `_BINDING_MEMO:1318`,
each stat-before-read with a documented blind spot), so the discipline exists. Its
risk is *not* the same as theirs, and the difference is worth stating: those memos
are safe because a wrong key can only cost a re-read or serve a value that is
itself stat-validated, whereas **a memo on the ranking can serve a stale ORDER**.
The key must therefore be the store's own change signal, and the doc that lands
this must name which fact makes the order change and why the key covers it.

**How this couples to "several agent runtimes going":** the census is
O(store directories) and the store only grows. Coder's census of the operator's
own store shows **26-28 subagent runs starting per five minutes**, each creating
a session directory that never goes away. So a fleet that runs more agents makes
the sidebar's fixed 2 s poll permanently slower — not because the runtimes are
consuming CPU, but because they are consuming *directories*. That is a coupling
to runtime count, and it is a store-growth coupling rather than a compute one.

**(c) The speculative warm-up costs 93 ms on the event loop, twice per refresh.**
The tree measured this and recorded the retraction of the competing explanation
in `tui/events.py:596-624`:

> "Measured on 12 background sessions: ~229 such events/s, every one
> hidden-source, +9 points of a core. The full delivery cost of a result thrown
> away." … "Typing is measurably slower with the sidebar open than closed, and
> THIS PATH IS NOT THE CAUSE … An ABBA A/B put this change at **153.8 ms against
> 143.1 ms** without it (within noise) while closed sat at **121.1 ms** … **The
> leading open suspect is `_prepare_sidebar_session` (93 ms median, on the event
> loop, two per prewarm refresh)**, not event fan-in."

The prewarm worker (`tui/app.py:9962-10072`) runs at most
`PREWARM_PER_REFRESH = 2` preparations per cycle (`:2256`), selected from live
sessions the catalog polls at `LIVE_REFRESH_INTERVAL_S = 1.0`
(`tui/widgets/session_picker.py:254`), and it holds up to
`RETAINED_PRESENTATIONS = 12` parked sources (`:2174`) with live subscriptions
whose delivery cost is ~0.26 ms/frame (measured, `events.py:668-673`).

So on the TUI the operator's observation is literally true and has a mechanism:
**more live sessions ⇒ more speculative preparation on the UI loop, and more
full rebuilds on every switch to a busy conversation.** The desktop is not
affected this way; its reads are served by a different process.

**(c) The TUI's click path is otherwise well built and should be left alone.**
`_lease_sidebar_source(speculative=False)` calls
`AttachedSession.saved_preview` (`session/attached.py:1594`), which reads the
**last 256 KB** of the journal off-loop (`session/saved_preview.py:29,47-52`),
paints it, and starts the live attach in a **background task**
(`_start_sidebar_connection`, `tui/app.py:8977,9019`). No click waits on an
engage. The 15 s readiness gate (`tui/session_navigation.py:23-41`) is a paint
gate for a *committed* session, not a spawn wait, and it is already fenced
against retry multiplication.

### 4.5 Where the existing caches do NOT reach

This is the answer to ask #2. The caches exist; the holes below are real, and one
of them I initially over-priced — the correction is recorded rather than dropped.

| Hole | What it costs | Evidence |
|---|---|---|
| **H-1. The TUI's conversation load path bypasses `session/page_cache.py`** — the page cache's only consumers are the desktop `history` route (`server/utils/desktop_sessions.py:82`) and the TUI's **subagent** pager (`tui/widgets/subagent_view.py:75`). The main load path is `read_saved_preview` (`session/saved_preview.py:39`), which reads raw bytes, and the connect path uses `read_replay_suffix` directly (`attached.py:5337`). | **Smaller than it looks, and I am not going to sell it as more.** The read is *bounded* at `PREVIEW_BYTES = 256 * 1024` (`saved_preview.py:29`), so this is a repeat parse of a fixed window, not of the journal. It repeats on every click and every prewarm cycle; it does not scale with the conversation. Coder's measurement: wiring the page cache in "would buy little on this path". | `grep -n 'page_cache' local_operator/**` → 4 hits, none in `saved_preview.py`; the 256 KB bound at `:29,47-52` |
| **H-2. The presentation cache misses on `streaming`** (and, deliberately, on a moved `replay_revision` and on any pending gate). | A **31.91 ms** prepare on a switch to a busy conversation, against **0.21 ms** on a hit (coder, 29 switches; 19/19 hits on the non-streaming arm). Real, but jank-scale — see §4.4(a). | `tui/app.py:6552-6588` |
| **H-3. Nothing is durable across a restart.** The caches are in-process: `_ROW_CACHE` in `catalog.py`, `_sidebar_presentations` in the TUI, and the page cache in `page_cache.py`. A new `serve` or a new TUI pays every cold cost again. | The operator restarts `lop` and re-pays the paint; a second interface (desktop + TUI) never shares a warm page. | `session/page_cache.py` (process-wide, byte-budgeted, not persisted) |
| **H-4. The sidebar's 2 s poll re-pays a whole-store census that a memoised sibling already answers** (§4.4(b)). | 221.5-227.1 ms of CPU and 21,205 `stat` calls per 2 s at his store size, against 72.5-78.6 ms and 12,802 stats. | `tui/app.py:9903` vs `session/catalog.py:1387` |

**Two cache layers exist and neither substitutes for the other**, which is worth
writing down because it is the obvious wrong conclusion to draw: the TUI's
`app._sidebar_presentations` (`tui/app.py:2174`, 12 entries) caches **built
widgets**; `session/page_cache.py` caches **decoded rows**. A hit in one is not a
hit in the other, and H-4 is a third thing again (a catalogue census).

Everything else the operator's ask #2 implies is already covered: the desktop
pool keeps resident bridges, the catalog memoises rows on the transcript's stat,
the SSE snapshot and `/history` share one page-cache key, and the single-flight
`_locate_flights` stops two concurrent cold callers from parsing twice.

### 4.6 What the attach path redoes — inventory

The column that matters is "does a **click** pay it". Sources are the tree's own
route/call structure and the prior audit in
`docs/design/session-load-central-cache.md` §1 (whose six-row table was the
design input for the shipped page cache).

| Work | When | Paid by a click? | Count / cost |
|---|---|---|---|
| Journal tail page (100 rows) | snapshot + `/history` | yes, twice — **deduped by the page cache** since (B) shipped | 1-4 ms cold, **0 rows decoded on a hit** |
| `read_latest_custom*` metadata row | bridge `locate()` | yes | backward read, 10-12 ms (was a whole-journal parse, 2 059-2 313 ms) |
| `_cwd_is_unconfirmed` | bridge build | yes | off-loop, run-dir discovery |
| `draft_birth_selection` | cold facade | yes | one `to_thread` config/marker read |
| Attention read (`attention.db`) | every snapshot | yes | capped at 50 ms; may serve stale |
| Runtime attach: dial | watch / read envelope | yes | 8.9-18.2 ms warm |
| Runtime attach: frontend sync | watch | yes | 10.5-15.5 ms idle; **1.1-1.9 ms with the owner blocked** |
| **Cold spawn + boot** | watch, when no warm runtime | **yes — and this is the 1.3-1.5 s** | pid at 127-248 ms; engage at 1 303-1 470 ms |
| **`Transcript.__init__` whole-journal parse** | inside the cold boot | yes, when cold | 248 ms / 107 MB; **2 287 ms / 261 MB**; 1.37 s / 231 MB |
| TUI: 256 KB preview parse (+ image hashes) | every TUI switch and prewarm | yes, on the TUI | bounded window, goes around the page cache (H-1) |
| TUI: prepare on a switch to a **streaming** session | every such switch | yes, on the TUI | 31.91 ms vs 0.21 ms on a cache hit (coder, 29 switches); cache refused (H-2) |
| TUI: whole-store census | every 2 s while the sidebar is open | yes, on the TUI | **21,205 `stat` + 3 `scandir`, 221.5-227.1 ms CPU** at 12 200 dirs (H-4) |
| TUI: speculative warm-up projection | ≤ `PREWARM_PER_REFRESH` per poll | yes, on the TUI | 93 ms median **on the event loop**, two per refresh |
| Registry `registry.scan` | sidebar poll, 1 Hz | yes | **one `ps` fork per whole quiet population** when any exist; zero on the healthy path |
| Wake index read | sidebar poll | yes | one read per list |
| Catalogue authoring probe | 1 s cadence | yes | 34 `agent.yml` read+filter+`crc32` = 1.16 ms; warm stat memory = 0 reads |
| Machine-wide feed tick | 100 ms, when a subscriber exists | no (no bridge, no spawn) | **4 `os.stat`** |

### 4.7 First paint: the remaining gap

First paint (the SSE `snapshot` frame — state plus a 100-row page) does **not**
wait on a runtime: the server builds a cold facade and paints from the journal.
My run and coder's agree on the shape and disagree in an instructive place —
`tiny`:

| profile | mine: cold p50 (min-max) | mine: cold over 300 ms | mine: warm reopen p50 (min-max) | coder: cold p50 |
|---|---|---|---|---|
| tiny (~25 KB) | **36.8 ms** (29.5-251.0) | 0/5 | 26.6 ms (22.3-57.6) | 236.6 ms |
| p50 (~0.7 MB) | 237.5 ms (81.7-681.7) | 2/5 | 68.4 ms (35.0-502.6) | 200.8 ms |
| p90 (~1.6 MB) | **490.0 ms** (71.2-671.6) | 4/5 | 187.7 ms (31.6-310.8) | 274.9 ms |
| p99 (~6 MB) | 147.7 ms (78.5-516.3) | 1/5 | 100.9 ms (27.2-186.4) | 244.0 ms |
| **xl (~200 MB)** | **789.0 ms** (738.3-933.5) | **5/5** | 75.3 ms (52.0-558.8) | — |
| m5000 (~12 MB) | — | — | — | 297.3 ms |

**Read the spread, not the medians.** Both runs were taken at load 74-98 and the
same profile moved by an order of magnitude between samples (tiny: 29.5 ms to
251.0 ms; p90 cold: 71.2 ms to 671.6 ms). Mine and coder's differ because the
`open` scenario ran at a different point in each session against a differently
warmed host, which is the honest limit of any wall-clock claim here: **the p50 is
weather; the `over 300 ms` counts and the size ordering are the finding.** What
both runs agree on: first paint is *at* the budget on small profiles, misses
badly on the 200 MB shape, and is 3-9x cheaper on a warm reopen.

Two facts fix the shape of the fix: the `validate` leg
(`GET /v1/desktop/sessions/{id}` → `snapshot()` at
`server/routes/desktop_sessions.py:2334`) is **~84% of first paint in coder's run
and flat in journal size there** (199 ms tiny vs 258 ms m5000) — because it is the
*cold facade and bridge construction*, not a read; and coder's cold-vs-warm split
of that same leg is **105.9-123.7 ms cold against 23.1-39.2 ms warm on one
unchanged session**. So first-paint headroom is on the same axis as the live
overlay: the cost appears when there is no warm state, and disappears when there
is — with one exception, the `xl` row, where 789 ms p50 / 5-of-5 over budget is
size-dependent and survives a warm reopen on 1 of 5 samples. That is the row to
optimise against and it is the operator's own shape.

---

## 5. Is <300 ms reachable as-is?

**Plainly: yes for a warm runtime, no for a cold one — and the distance is
measured.**

| Measurement | Result | vs 300 ms |
|---|---|---|
| Warm attach, same runtime alive (coder, n=9) | 29.6-78.7 ms | **pass, 4-10x headroom** |
| Warm first paint / reconnect (both, n=5) | 26.6-187.7 ms p50 | pass (individual samples up to 558.8 ms) |
| Cold first paint, small profiles (both) | 36.8-490.0 ms p50 | **at or over the budget**, 0/5-4/5 samples over per profile |
| Cold first paint, xl / 200 MB (mine) | 789.0 ms p50, 5/5 over | fail, 2.6-3.1x |
| Cold attach (coder) | 1 303-1 470 ms | fail, 4.4-4.9x |
| Cold attach, xl / 200 MB at load 98-116 (mine) | 5 654-11 800 ms | fail, 19-39x |

**What the target costs, in the two changes that reach it:**

1. **Publish the runtime's cold answer instead of no answer** (§9 R1). The
   viewer's first frame then arrives when the pid exists — **127-248 ms**, inside
   the budget — carrying the durable page and an explicit `attaching` flag, with
   the live frame following as a `frontend.replace`. This is *not* a new
   protocol: `cold`, `cold_reason` and `attaching` already exist and the desktop
   renderer already consumes the late authoritative replacement
   (`desktop_sessions.py:1446-1460`). Cost: a second state shape on the serving
   plane, and the discipline that nothing in it may claim liveness it has not
   observed (§6).
2. **Remove the whole-journal parse from the boot's critical path** (§9 R2) for
   the case where a runtime must genuinely be started. This is the deferred (D)
   of `docs/design/session-load-central-cache.md` §3(D), and its deferral reason
   survives — see §9 R2.

**R1 alone reaches <300 ms for status and attach, including for the operator's
261 MB conversation, without buying a byte of memory.** It does not make the
*first prompt* on a cold session instant: a turn needs LLM history, which needs
the parse. That is R2's job, and it is a bigger change.

---

## 6. The not-stale contract

A cache that is fast and wrong is worse than no cache. This is the hard half, and
the current code already violates it in one measured way.

### 6.1 The measured violation

From `--scenario busy` (my run; the owner is `SIGSTOP`ped with a live pid and a
live record — the wire shape of a loop blocked in a long synchronous step).
**Every one of the 23 rows the run completed says the same thing** — all five
profiles, tiny through 200 MB `xl`:

```
n = 23 rows across tiny / p50 / p90 / p99 / xl

reopen_cold          false   (23/23)        reopen_cold_reason  null   (23/23)
reopen_snapshot_ms   44.6 .. 320.3  (p50  66.2)
reopen_validate_ms   30.5 .. 245.7  (p50  46.2)
during_send_cold     false   (23/23)
warm_status          200     (23/23)   {"state": "warm"}   warm_ms 5.4 .. 95.3
send_status          503     (23/23)   runtime_unreachable
send_ms              15 020.7 .. 15 565.3   (p50 15 041.6)
pid_state_before     Ts      (23/23)   ← the owner is stopped, not dead
```

Three surfaces confidently report a healthy session — `cold: false`, a `warm`
receipt with status 200, and a fast `validate` — while the owner cannot answer a
single message. Only the mutating route tells the truth, and it takes
**15.0-15.6 s** to do it, with a spread of 0.5 s across 23 samples: that 15 s is
a *budget*, not weather. The registry's resolution is
`HEARTBEAT_INTERVAL_S = 15.0` and `HEARTBEAT_TIMEOUT_S = 45.0`
(`session/runtime/types.py:369-370`), so `live → wedged` can lag the truth by
**45-60 s**.

**That is the operator's "lagged or behind the actual current state of the
runtime", reproduced.** It is not a caching defect; it is that
`cold: false` is currently derived from *the local facade's memory of its own
bind*, not from a fresh answer from the owner.

### 6.2 The contract, stated so it can be tested

Definitions, and every one of them is already a field or a concept in the tree:

* **Durable state** — rows in the journal. Rule: it is monotone and its authority
  is the file. A reader can always have it. Watermark: the newest entry id it
  contains (`history_cursor` already exists,
  `session/frontend_state.py`).
* **Live state** — what the runtime is doing now. Authority: the owner. A reader
  cannot have it without an answer, and must be able to tell whether it has one.
* **`verified_at`** — the moment a **round trip to the owner was answered**. This
  is the missing field, and it is the whole contract:

```
AttachView := {
    session_id,
    durable:  { rows, watermark: <newest entry id>, read_at },
    live:     { state, generation, verified_at } | null,
    attaching: bool,
    cold_reason: str | null,
}
```

**The five rules.**

1. **Nothing may report `live` without a `verified_at`.** `cold: false` must be
   derived from an *answered* subscribe/dial (`verified_at` set in that
   round trip), never from a resident facade's memory of an earlier one. Today's
   `reopen_cold: false` with a SIGSTOPped owner is exactly the forbidden state.
   *Test:* freeze the owner, wait past one attach round trip, assert the frame
   carries `live: null` (or `live.verified_at` older than its own budget) —
   never `cold: false` with a fresh timestamp.
2. **Age is checked by the reader, not hoped by the writer.** A reader that
   needs "now" compares `now - verified_at` against its own budget. A reader that
   only needs paint uses `durable` and ignores `live`.
3. **The writer's death needs no special case, because `verified_at` ages.**
   This is the answer to "what does a reader see when the writer dies
   mid-update": the last `live` value with a `verified_at` that stops advancing.
   A kill -9 mid-write cannot produce a *fresh* stamp, so it cannot produce a
   false "current" — only a narrowing staleness window that the reader is
   already computing. **This is the property that makes the design checkable
   rather than hoped-for: staleness is observable, never asserted away.**
4. **`cold: false` and `attaching: true` are mutually exclusive on the same
   frame for the same session.** The renderer's "live now, or explicitly
   attaching" choice is then total.
5. **The cache may be behind, never ahead.** The projection may miss rows the
   journal has; it may not contain rows the journal does not. This is
   `session-projection-cache`'s rule in DSH's own words ("a crash can leave a
   checkpoint stale, but never ahead of committed events"), and it is the rule
   that makes rule 3 safe.

**A cache that shows a session as idle while it is mid-turn is impossible under
rules 3+5**, and here is the argument, not the hope: "mid-turn" is live state.
Live state is only ever published with a `verified_at` from an answer that
observed it. A cache that has not been told is either (a) publishing the last
answer with its old `verified_at` — which the reader rejects on age, or (b)
publishing nothing (`live: null`). Neither renders as idle. The forbidden output
requires fabricating liveness, and rule 1 is mechanically checkable: a `live`
block with a null `verified_at` fails parsing.

### 6.3 What the design must preserve

The tree's existing discipline is load-bearing and must not be defeated:

* **`_detach` at `users == 0`** (`desktop_sessions.py:1326-1440`): the facade
  cannot be disposed under a caller still attaching it. A read model that
  outlives the bridge must not resurrect the facade refcount; it holds the
  *durable* half only.
* **`_bind_lock` serialises dials on the facade.** A new early-cold answer must
  not become a second dialler for the same facade.
* **The epoch/replay reset on detach.** A projection keyed on the runtime's
  generation must be invalidated with the epoch, not with a timeout.
* **`_locate_flights` single-flight.** Any new durable read introduced for the
  cold answer must join the existing flight, not add a second parse of the same
  journal.
* **The desk of "no await between the sequence read and the return"** in
  `snapshot()` (`desktop_sessions.py:2795`) — the pre-open supersession
  proof depends on it. Adding a `verified_at` read must not introduce an await
  between those lines.

---

## 7. Push versus poll

**Recommendation: keep SSE push for live state (it already exists), add no new
transport, and reject "push as the fix".**

What exists today: the desktop session stream (`/events`, SSE with a heartbeat
every 15 s) carries `snapshot`, `frontend.replace` and `attention` frames; the
machine-wide feed (`desktop_feed.py`) publishes `notification`/`attention` with a
100 ms tick and a live-only, never-replayed contract; the TUI holds live
subscriptions per parked source with an owner-side mute negotiated on the wire
(`set_event_mute`, `tui/events.py:668-673`). The TUI's own A/B already retired
"viewer-side gating" as a class of fix (`events.py:616-619`), which is evidence
against adding more filtering machinery here.

**Why push is the right shape, and already satisfied:** "status must not be
lagged" is a *latency of truth* problem, and a push channel is the only shape
that can shrink it below the poll period. But the measured lag is not a polling
period — it is `HEARTBEAT_TIMEOUT_S = 45` plus a facade that asserts liveness it
has not verified. **Pushing the same unverified claim faster would deliver a
wrong answer sooner.** §6's `verified_at` is what shrinks the lag, and it is
orthogonal to push-vs-poll.

**Why a new push channel is not justified:** the fan-out the task asks about
(TUI + server + mobile relay) is already the reason each interface has its own
stream, and adding a fifth multiplexed channel buys lifecycle, reconnect and
missed-event problems that the tree has already paid for once each. The cheap
versioned projection in §6 gets the same result: the *only* thing the extra
channel would add is earlier delivery of a value whose freshness is currently
unverifiable.

**Where push does need one addition:** the runtime's early cold answer (§9 R1)
should be *published on the existing stream* as a frame, not returned as a new
endpoint. That is a frame shape, not a transport.

---

## 8. DSH as a reference implementation

Read from the checkout at
`/Users/damian/.local/share/dsh-cli/lib/node_modules/@deepseek-ai/dsh/`
(v0.1.7-rc.2 packages). **The live GUI could not be inspected from this session
and I am not going to pretend otherwise**: `GET http://127.0.0.1:3080/` returns
`401 dsh web authentication required` without the launch token (the documented
browser-trust fence, `dsh-client-connection`), and this session has no
Chrome/Chromium binary for the browser panel. So this section is a **source
study** — package READMEs and shipping code — not an observed-UI study. That is a
real limit on ask #3's evidence and the next phase should close it with a
rendered capture.

### 8.1 Transferable

| Pattern | DSH's own words | Why it transfers to lop |
|---|---|---|
| **List reads never open a body** | "List reads only stored headers and projection-cache rows: it never calls per-session stat or opens a cold Session body." (`dsh-api-session-controller`) | lop's sidebar already does two reads for a whole list (`catalog.py:908`); the rule to adopt is the *prohibition* — a store-scale list must never touch a per-session body. At 12 258 sessions this is the difference between O(working set) and O(store). |
| **Live-preferred but non-activating reads** | "`session.projections` reads one complete baseline through a live-preferred Session observation **without activating an Agent**." | This is exactly the split lop needs at the attach: prefer live, but never let a read *create* the runtime. lop currently activates on `/warm` (correctly) and paints cold on read (correctly) — DSH's phrasing is the invariant to write down and test. |
| **A watermark per value** | "Snapshots identify the last event reflected by every returned value, so carriers can pair state with the matching history cut." (`asOfSeq`) | lop has the raw material — `frontend_state.history_cursor`, `epoch`, the bridge's `sequence` — but not a per-value `asOf`. §6's `verified_at` + watermark is this pattern, and DSH proves it is enough. |
| **Stale, never ahead** | "The session log remains authoritative: a crash can leave a checkpoint stale, but never ahead of committed events." (`dsh-session-projection-cache`) | The correctness rule for §6 rule 5, already written in words that fit lop's journal exactly. |
| **Cached rows lose to live rows, whatever the watermark** | "a cached list block, viewed from the projection cache for a cold Session, **yields to the connected Session's baseline whatever watermark it carries**." | A one-line precedence rule that resolves the hard case in a cache-plus-live design: the live answer wins on conflict, unconditionally. Adopt verbatim. |
| **Generation semantics on reconnect** | "every generation opens with a complete process-local baseline, so reconnect replaces projection state instead of treating transient values as durable events"; "the first control stream waits for generation readiness, so its opening values cannot precede invalidation." | lop already has epochs and a `frontend.replace`; DSH's contribution is the *ordering rule* (subscribe-first, then page, then repair gaps through a tail page) and the sequencing that makes a delayed baseline unable to clobber a newer value. lop's `desktop_sessions.py` already implements the queueing half; the ordering rule is worth stating as a test. |
| **Identity stability suppresses jitter** | "returns one identity-stable Conversation binding … It does not open another event source"; "Client list refreshes retain unchanged row objects and reuse the items array when order and values match." | Directly answers "reduce UI jitter": a switch that produces the *same* objects produces no repaint. lop's `_sidebar_presentation_current` is the analogue and it is *stricter* (it refuses on streaming) — see §9 R4. |
| **Optimistic local echo on submit** | "`session.beginSubmission` inserts one into `SessionSnapshot.pendingSubmissions` **synchronously, before the caller serializes and prompts**, so a conversation UI can show the message on the submit click's own frame." | The operator's other complaint is "new message sending". Whatever the engine costs, the *echo* must not wait for an engage. This is cheap and transferable, and it is worth checking whether lop's send path paints before the round trip. |

### 8.2 NOT transferable, and why

| Pattern | Why it does not transfer |
|---|---|
| **The projection registry itself** (`sessionProjections.register`, synchronous `apply(state, event)`, `Object.is` change suppression) | It works because **one Node process folds every committed event into memory**: registration is a Cordis fiber effect, the fold is synchronous, and a change is detected by reference identity. In lop the writer of the journal is a **separate process per session** (`session/runtime/*`), and the UI's reader is another process again. A synchronous in-process fold has no counterpart across a process boundary; importing the shape means re-establishing it as a *tail-and-fold reader* over the JSONL, which is a different design with different guarantees (missed rows on a torn tail, ordering across `compact_file`). |
| **In-memory snapshot store as the source of truth** | DSH's client store is the same process as the gateway and is rebuilt from durable events on reload ("Echoes are Client memory only; reload and reconnect rebuild the conversation from durable events alone"). lop's TUI is one process and *could* do this, but its journal is not the only durable authority — the **runtime holds the LLM context**, and a TUI-side fold can be ahead of what the model has actually been given. Any lop projection must be labelled as *display* state, never as context state. |
| **WebSocket multiplexed streams** (`/api/remote.mux`) | A transport preference, not an architectural one. lop's SSE + HTTP works and is already push-shaped; swapping transports buys no measured latency and costs a reconnect/lifecycle rewrite on three surfaces. |
| **`localStorage` persistence** and the browser-only store | No analogue worth copying; lop's durable state is the journal. |
| **Writer-lock semantics** (`session/writer-held` → "try another blank") | DSH's answer to contention is to make the caller pick a different session. lop's is takeover/adoption with an explicit viewer/owner split (`docs/design/attached-interface-signal.md`), which is a different and *better* contract for a personal harness — copying the refusal would regress "reconnect and reconcile". |

**The one-line summary of the DSH study:** its *contracts* (watermarks, stale-not-
ahead, live-preferred-but-non-activating, cached-yields-to-live, identity
stability) are exactly the contracts §6 needs and should be adopted as written;
its *mechanisms* assume a single-process in-memory fold and must not be.

---

## 9. Ranked changes

Each entry: expected effect on the 300 ms target, risk, blast radius,
independence, and **how it is measured**. "Independent" means it can ship alone.

### R1 — Publish the runtime's cold answer instead of no answer ★ rank 1

**What.** Make the runtime's serving plane answer a viewer as soon as the
process exists, with a **durable** snapshot from the journal, an explicit
`attaching: true`, and **no live block**; publish the live frame later as a
`frontend.replace`. The gate pattern already exists one screen away: MCP wiring
defers behind `mcp_publication_gate` (`serving.py:723-724,7945`) which
`RuntimeServer._serve` (`server.py:2641`) sets "the moment the publisher exists",
precisely so that "no integration configuration can sit between the user and a
bound session".

**Effect.** `t_attached` cold: **1 303-1 470 ms → 127-248 ms** (the time to get a
pid). For the operator's 261 MB conversation this removes the entire 1.8-2.3 s
parse from the attach path — **the target is met for attach and status with no
memory cost**. First paint is unchanged (already 200-300 ms) but its *live* leg
stops being the thing the user waits for.

**Risk: medium-high.** A second state shape on the serving plane. The
`cold`/`cold_reason`/`attaching` vocabulary exists and the desktop renderer
already applies the late authoritative `frontend.replace`, so the protocol
change is smaller than it sounds — but every rule in §6.3 must hold, and "no live
block before an answer" has to be enforced by construction rather than by
convention.

**Blast radius.** `session/runtime/serving.py`, `session/runtime/process.py`,
`session/runtime/server.py`, `session/attached.py` (the viewer's early frame),
`server/utils/desktop_sessions.py` (the cold facade), and the frontend-state
protocol tests. Not the TUI's in-process path.

**Independent: yes.**

**Measurement.** `--scenario attach` before/after: `t_attached` on
tiny/p50/p90/p99/xl, plus a new **structural** assertion — the number of journal
rows JSON-decoded before the first frame (`== 100`, the page), which is
load-independent and therefore the real acceptance criterion. Plus coder's
`probe_warm_reattach.py` for the warm legs, which must not regress (29.6-78.7 ms).

### R2 — Take the whole-journal parse off the boot's critical path ★ rank 2

**What.** The deferred (D) of `docs/design/session-load-central-cache.md` §3(D),
whose follow-up shape that doc already names: **not** lazy `Transcript._entries`,
but a **windowed boot replay** through the machinery that already exists
(`context_cut_index`, `first_message_index`, `audit_slice`, `replay_entries`) —
the same treatment the *viewer* already got at `session/attached.py:5337`, whose
comment gives the identical argument: "the replay only ever uses rows from the
latest compaction's kept window".

**Does the deferral reason still hold?** **Yes for `_entries` laziness, and it is
sharper than the doc states.** I re-read the hazard at this revision:
`_write_entries`' rebuild branch (`session/transcript.py:1552-1568`) fires on
`FileNotFoundError` and writes `(*self._entries, *entries)`. With an eager
`_entries`, a journal that a sweep removed underneath a live session is
*recovered* into memory and rewritten. With a lazy `_entries` that recovery
silently becomes "write only the new batch". That is a real regression on a path
that exists on purpose. Add §3(D)'s own list — ~30 call sites, the `hasattr`
duck-typing in `session_factory.py:684-686,749-750`, and a `threading.Lock`
guarded load reachable from a loop thread being the #401 freeze shape — and
**the deferral stands**.

So R2 is the *narrow* version: keep `_entries` eager-on-demand, and stop the
**boot** from needing the whole file. Prerequisites, in the order §3(D) gives
them: (1) a windowed `build_llm_history` with a proven equivalence test against
the whole-file replay; (2) an explicit `await transcript.materialise()` boundary
the append path calls before computing a rollback size; (3) the duck-typed
`hasattr` call sites converted so they cannot be answered by a partially-loaded
object.

**Effect.** Removes **248 ms / 107 MB** and **2 286 ms / 261 MB** from the cold
engage — the size-dependent term my `xl` row exposed, and the term warmth cannot
fix because it is paid *inside* every cold boot. This is the change that makes
"cold is cheap" true, which is what lets the cap stay at 4.

**Risk: high** (data-loss adjacency; the reason it was deferred). **Blast
radius: wide** — `session/transcript.py`, `session_factory.py`, and every reader
of `entries`/`build_llm_history`. **Independent: yes** (R1 does not need it; it
is needed for the first *prompt* on a cold session).

**Measurement.** Structural, load-independent: **rows JSON-decoded during
`create_session`**, counted from the boot path, asserted equal to the
post-compaction window rather than the file. Wall-clock second:
`--scenario attach` on xl. Equivalence: the existing whole-file oracle test for
`build_llm_history` extended to the windowed path.

### R3 — Warmth, priced and capped inside the envelope ★ rank 3

**What.** Do **not** treat the cap as the fix. If any memory is spent, spend it
on the *right* sessions: today's LRU is over "idle clientless runtimes" in detach
order, which mixes user conversations and subagent runs. A keep-set of "the
recently *user*-visited conversations, up to N" is the same mechanism aimed at the
working set. **Recommendation: 4 → 6 at most**, and only with the operator's
explicit budget, because the arithmetic in §4.3 puts even the current cap at the
edge of his stated ~200-300 MB envelope.

**Effect.** Each additional warm slot converts one 1.3-1.5 s cold attach into a
30-80 ms warm one. Cap 4→6 covers 6 of his 8 recent user sessions; the remaining
2 pay cold, which R1+R2 make cheap.

**Risk: low** (a policy constant and its candidate predicate). **Blast radius:**
`session/runtime/process.py` keep-alive predicates + the config schema.
**Independent: yes.**

**Measurement.** `test_runtime_keep_alive` extended with the RSS figure per
resident; a probe that visits 8 conversations and reports how many attaches were
warm; the memory delta measured on an isolated root, never the operator's.

### R4 — Stop the TUI refusing a cached presentation for a STREAMING target

**What.** `_sidebar_presentation_current` (`tui/app.py:6552`) refuses a parked
presentation when the session is streaming (`:6585-6586`). Extend the commit
path's existing delta projection — which already handles "more durable rows were
appended" — to the streaming case, using the owner's canonical live seed
(`EventController.restore_live_projection`, `tui/events.py:697`) to rebuild the
live block on top of the cached base. Keep the refusals that are *correct*: a
changed `replay_revision` (compaction rewrote painted rows, `:6587-6588`) and a
pending gate (presentation-only state the delta cannot supply, `:6582-6583`).

**Effect — restated, because my first framing over-priced this.** The cache is
**not** self-disabling: coder measured **19 of 19** revisits of a non-streaming,
gate-free session hitting it at **0.21 ms**. The streaming arm is a **31.91 ms**
prepare instead of 0.21 ms on the case that fires when several runtimes are going.
So this is a **~32 ms jank-scale** item, not a load-time one, and it sits below
the census in §4.4(b) on effect. It stays on the list because it is the one miss
arm that is arguably wrong rather than deliberate, and because it is the operator's
exact case.

**Risk: medium** — correctness of the live block on a reveal (a mid-turn join is
the same problem and already has a path, so the risk is bounded by reusing it
rather than inventing one). **Blast radius:** `tui/app.py` presentation cache +
`tui/events.py` restore path. **Independent: yes.**

**Measurement.** A TUI probe that (i) starts N streaming sessions, (ii) switches
between them, and counts **cache hits vs prepares and their two costs** — a count
and two numbers, not a wall time. Plus the existing TUI test battery for reveal
correctness.

### R5 — Move the speculative warm-up's synchronous work off the event loop ★ rank 5

**What.** `_prepare_sidebar_session` is measured at **93 ms median on the event
loop, twice per prewarm refresh** (`tui/events.py:620-622`), and the tree's own
A/B puts the open-vs-closed typing gap at **121.1 ms closed against 143.1-153.8 ms
open**. Move the projection/parse half to a worker thread, leaving only widget
mutation on the loop. Alternatively (or additionally) widen the prewarm cadence
while the user is typing.

**Effect.** Removes a measured UI-loop stall that scales with the number of live
sessions. This is the TUI half of "several runtimes slows things down" that no
server-side rig can see.

**Risk: medium** — off-loop widget construction is a known hazard here; the work
must be split so that only pure projection moves. **Blast radius:** `tui/app.py`
prewarm + `session_presentation.py`. **Independent: yes.**

**Measurement.** The tree's own instrument: an ABBA A/B of typing latency with
the sidebar open, plus a count of **event-loop stalls > 50 ms per minute** while
N sessions stream. Report at a stated load.

### R6 — Route the TUI's own load path through the existing page cache

**What.** `read_saved_preview` (`session/saved_preview.py:44-52`) does its own
256 KB read + JSON decode + `replay_entries` + one content read and one
full-digest re-hash per image reference, on **every** switch and **every** prewarm
cycle, for a journal that usually has not changed. `session/page_cache.py` exists
and is consumed by exactly two paths, neither of them this one (`grep -n
'page_cache' local_operator/**` → 4 hits).

**Effect — and this one I over-priced too, so it is stated as measured.** The
bypass is real but the read is **bounded at `PREVIEW_BYTES = 256 * 1024`**
(`saved_preview.py:29`), so this is a repeat parse of a fixed window, not of the
journal, and it does not grow with the conversation. Coder measured that wiring
the page cache in "would buy little on this path". It therefore stays on the list
as the **cheapest item to land** and the direct answer to ask #1's "reduce the
repeat work" — not as a latency win.

**Risk: low** (an existing, tested cache; a new key). **Blast radius:**
`saved_preview.py` (+ a cache identity for a byte-range read, which page_cache
already has for pages). **Independent: yes.**

**Measurement.** Rows decoded per switch, counted on both sides: the window's
rows (today) vs **0** on an unchanged-journal hit.

### R9 — Gate the TUI's 2 s whole-store census on a store-change key

*(added after coder's corrected measurement; the numbering keeps R1-R6 as the
manager has them)*

**What.** `_ranked_candidates` (`session/catalog.py:1518`) calls
`_scan_sessions(directory, strict=True, ...)` (`:1566`) on **every** poll. The
hydration memo is already on the poll's path — `load_catalog` (`:1818`) →
`_hydrate` (`:1914`) → `for row in cached_session_rows(...)` (`:1930`) — so the
recoverable cost is the scan, decoration and ranking above it, not a missing
memo. Memoise or gate that scan on the store's own change key, using the
discipline the module already applies three times (`_ROW_CACHE:1206`,
`_BIRTH_MEMO:1243`, `_BINDING_MEMO:1318`).

**Effect.** Up to **8,403 `stat` calls and ~145 ms of CPU per 2 seconds** at the
operator's store size (21,205 `stat` + 3 `scandir` / 221.5-227.1 ms CPU today,
against 12,802 `stat` + 1 `scandir` / 72.5-78.6 ms for the scan-free reader). This
is the largest *repeat* cost on the TUI path, it is paid whenever the sidebar is
open, and it grows with the store — which is how it couples to the runtime count
(26-28 new session directories per five minutes).

**Risk: low-medium, and the reason is specific.** The module's existing memos are
safe because a wrong *key* costs a re-read or serves a value that is itself
stat-validated; **a memo on the RANKING can serve a stale ORDER**. The key must be
the store's own change signal, and whoever lands this must name which fact makes
the order change and why the key covers it.

**Blast radius:** `session/catalog.py` + the TUI sidebar poll. **Independent: yes.**

**Measurement.** Syscall counts per poll (`stat`, `scandir`) on a 12,200-directory
fixture store, plus CPU ms per poll — both load-independent in the way that
matters, since the cost is `stat` volume and not scheduling.

### R7 — Make liveness verified, and make the receipts honest ★ rank 6 (correctness, not speed)

**What.** §6's `verified_at` on the live block; `cold: false` only from an
answered round trip; and the mutating routes' receipts (starting with `POST
/warm`, which answered **200 `{"state":"warm"}`** for a `SIGSTOP`ped owner) must
not assert an outcome they did not observe.

**Effect.** None on latency; it removes the measured false-live window
(45-60 s by heartbeat, and "immediately and wrongly" for a resident facade). The
operator's "without being lagged or behind" is this, and it is the difference
between a design that is fast and one that is trustworthy.

**Risk: medium** — it changes what a renderer sees for a busy owner, so the
renderer's `attaching`/`cold_reason` handling has to be exercised against a
frozen owner (the `--scenario busy` rig already does the freezing).

**Blast radius:** the frontend-state wire shape + `desktop_sessions.py` +
`registry.classify` consumers. **Independent: yes.**

**Measurement.** `--scenario busy` as the acceptance test: assert the read path
never reports `cold: false` with a fresh `verified_at` while the owner is
`SIGSTOP`ped, and that a failure surfaces in **< 2 s** instead of 15.0-15.6 s.

### R8 — First-paint headroom in its own right

**What.** First paint is at the budget (200-297 ms p50, p95 over on 3 of 5
profiles) and `validate` is 84% of it. It is cold-spawn-coupled, so R1/R3 move
it; the residual items are the 50 ms attention wait
(`ATTENTION_SNAPSHOT_WAIT_S`), the duplicate tail read between snapshot and
`/history`, and the xl row (738-934 ms cold), which the page cache does not help
because the journal is cold by construction in every sample.

**Effect.** ~2x headroom on first paint; the p95 failures close.
**Risk: low-medium.** **Blast radius:** `server/utils/desktop_sessions.py`.
**Independent: yes.**

**Measurement.** The existing `--scenario open` table, with the page-cache
`cached_page_hit` / rows-decoded counters beside it, at a stated load.

### Ranked summary

| rank | change | effect on 300 ms | risk | independent |
|---|---|---|---|---|
| 1 | R1 publish the cold answer | cold attach 1 303-1 470 ms → **127-248 ms** | medium-high | yes |
| 2 | R2 windowed boot replay | −248 ms / 107 MB; **−2 286 ms / 261 MB** | high | yes |
| 3 | R9 gate the 2 s store census | **−8 403 `stat`, −~145 ms CPU per 2 s** | low-medium | yes |
| 4 | R3 warmth 4 → 6 | one cold attach → 30-80 ms per extra slot | low | yes |
| 5 | R5 prewarm off the loop | removes 93 ms loop stalls, 2/refresh | medium | yes |
| 6 | R4 TUI cache: the streaming arm | 31.91 ms prepare → 0.21 ms hit | medium | yes |
| 7 | R7 verified liveness | removes a 45-60 s false-live window | medium | yes |
| 8 | R8 first-paint headroom | closes the over-budget samples | low-medium | yes |
| 9 | R6 TUI load path → page cache | bounded repeat parse → **0 rows decoded** | low | yes |

**Ordering rationale.** R1 before R2 because R1 reaches the target for the thing
the user actually does (open, switch, watch) without touching the hazardous
contract; R2 is bigger, riskier, and only strictly required for the first
*prompt* on a cold session. **R9 (the census) is third on measured effect, not on
novelty** — it is the largest *recurring* cost any of these changes removes, and
it is the answer to ask #1's "reduce the amount of repeat work" that the operator
pays whether or not he opens anything. R3 before R5/R4 because it is one constant
on the primary interface. R4 and R6 are last on effect and are ranked for
completeness rather than for expected gain — both were over-priced in my first
pass and are corrected above rather than quietly dropped.

---

## 10. What I would not do

* **Do not build a second cache.** Three exist (`page_cache.py`,
  `_sidebar_presentations`, `_ROW_CACHE`) plus the resident-bridge pool. The work
  is coverage and invalidation, not another layer. In particular, do not add a
  desktop-side projection cache in front of a read that is already 200 ms.
* **Do not make `Transcript._entries` lazy.** §3(D)'s reason survives, and I
  found it sharper than documented (the rebuild branch's recovery of a swept
  journal). R2 is the narrow version.
* **Do not add a push transport.** §7: pushing an unverified claim faster
  delivers a wrong answer sooner.
* **Do not raise the keep-alive cap to cover the working set.** §4.3: it costs
  2-5x the operator's own stated memory envelope, and R1+R2 make the cold path
  cheap enough that the cap stops being the lever.
* **Do not gate anything on a per-session cost that scales with the store.** His
  store is 12 258 sessions / 7.3 GB with 26-28 subagent runs per five minutes; a
  design whose per-open cost is O(store) fails at his scale. Every change above
  is O(journal window) or O(working set) — and R9 removes one place where the
  O(store) is already being paid every 2 seconds.
* **Do not re-propose wiring the page cache into `read_saved_preview` as a
  latency win** (R6). The bypass is real and worth closing for tidiness, but the
  read is bounded at 256 KB, so the honest expectation is "less repeat work",
  not "faster switching".
* **Do not design against a multi-second cold-engage tail.** At `--runs 5` a
  "p95" is just the maximum, and a figure of 824 ms (564 KB) / 1 340 ms (10 MB)
  did not reproduce at 60 samples per cell (16.5-20.8 / 280.7-314.7 ms, i.e.
  1.2-1.5x the median). The withdrawn number is named here so the next reader
  does not re-derive it.
* **Do not tune against wall clock alone.** On this host (load 74-116) the same
  operation moved 12x within one cell (coder's tiny cold `validate`:
  16.2 → 261.2 ms; my own tiny cold first paint: 29.5 → 251.0 ms). Counts — rows
  decoded, journal opens, forks, `stat` calls, prepares vs hits — are the
  acceptance criteria; wall time is context.

---

## 11. Open questions, and the evidence that would settle them

1. **Is `read=True`'s attach attempt needed for correctness at all?** Its stated
   purpose is to make a warm conversation show as attached on first paint, and a
   warm attach is 30-80 ms. The evidence that would settle it: count how often
   the read-envelope attach *changes the frame it returns* (i.e. `cold` flips
   from true to false inside the 50 ms grace) across a week of production logs.
   If it is rare, the grace can drop to 0 and `validate` becomes a pure durable
   read.
2. **What is the real distribution of "user returns to a conversation"?** The cap
   discussion rests on coder's 5-minute census (8 user sessions). The evidence
   that would settle the right keep-set is the revisit interval distribution
   across, say, a month of the operator's own store — which is a read-only
   analysis over `sessions/*/transcript.jsonl` mtimes, and it would also tell us
   whether `keep_alive_seconds = 300` is the right window.
3. **Does R1's early-cold frame break any renderer assumption?** The desktop
   renderer is documented to apply the late `frontend.replace`; the TUI's
   in-process path is not affected. The evidence: a headless pilot of the
   desktop stream against a runtime held in the early-cold state, asserting the
   panel moves `attaching → live` and never renders a live block the owner has
   not answered for.
4. **How much of the TUI's switch cost is the rebuild vs the paint?** R4 assumes
   the rebuild dominates; `_prepare_sidebar_session`'s 93 ms is *all* of it
   (projection + widget preparation). The evidence: split that measurement into
   its parse half and its Textual half before committing to R4's shape.
5. **Is the 256 KB `saved_preview` window the right size for the operator's
   conversations?** A 6 MB journal's tail 256 KB may hold only a handful of rows,
   in which case a switch paints a short transcript and the rest arrives with the
   live attach. The evidence: rows-per-preview over his real store, read-only.

---

## 12. What I ran

Worktree: `~/local-operator-worktrees/convload-arch-1790433460` at `origin/main`
`a3228afd4`, own venv (`uv venv --python 3.12` + `uv pip install -e ".[all,dev]"`,
verified `local_operator.__file__` resolves inside the worktree). Scratch:
`$LOCAL_OPERATOR_SCRATCHPAD` (`~/.dsh/scratch/d98d97c9-4eb6-496c-9f59-0645e8d38b45`);
the operator's live `~/.local-operator` was read only, never written, and no
running fleet was signalled.

```sh
# the four-scenario table (open / attach / busy / list), load 74-116, n=5/cell
env -u XPC_FLAGS .venv/bin/python scripts/run_bounded.py --timeout 2400 -- \
  .venv/bin/python scripts/bench_desktop_open_attach.py \
  --scenario all --profiles tiny,p50,p90,p99,xl --runs 5 \
  --root "$SCRATCH/benchroot-open" --tree "$WT" --keep --json "$SCRATCH/bench-open.json"
```

**The run was cut short and that is disclosed rather than smoothed over.** The
wrapper logged `signal 15 received — process group … signalled (rc=143)` after
1 690 s, during the last cell (`xl` busy, 3 of 5 samples), so the driver's own
summary block and `bench-open.json` were never written. What the log carries is
complete for `open` (5 profiles x 5, cold + reopen) and `attach` (5 x 5), and
**23 of 25** `busy` rows spread over all five profiles — enough that every busy
claim in §6.1 is unanimous across 23 samples. The per-scenario figures quoted
above were recomputed from the driver's own stdout lines, not transcribed by
hand.

Read-only checks against the operator's environment (no writes):
`git -C ~/local-operator fetch origin main`, `git show origin/main:pyproject.toml`,
and a `grep -c` of `~/.local-operator/config.yml` for a `runtime` section (absent,
so the keep-alive defaults are in force).

**Corrections this document makes to its own inputs — and to two of its own
first drafts.** The peer coder's and the manager's intermediate readings were
right about the mechanism and each carried one claim this document does not
repeat: (a) that the read route waits up to 2 s for a bind — it stops at 50 ms;
(b) that a `runtime_build_session_ms` p95 of 824/1 340 ms was a contention
signature — it was small-sample contamination, and the steady-state medians are
15.3-16.5 ms (564 KB) and 189.9-205.0 ms (10 MB); (c) that attach is flat in
transcript size — true to ~12 MB, false at 200 MB, where the whole-journal parse
dominates and my `xl` row measures 6.1-11.8 s; (d) that the sidebar poll skips
`cached_session_rows` — it does not, `load_catalog:1818` → `_hydrate:1914` →
`cached_session_rows:1930`, and the surviving finding is the unmemoised
`_scan_sessions` in `_ranked_candidates:1518,1566` (R9). My own first drafts
over-priced two items and both are corrected in place rather than removed: the
TUI cache miss on a streaming target is ~32 ms, not a whole switch (R4), and the
page-cache bypass in `read_saved_preview` is a bounded 256 KB re-parse, not a
journal read (R6).
