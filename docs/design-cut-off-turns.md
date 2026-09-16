# Design: a turn cut off by anything but a deliberate stop is an ERROR, with a reason

Status: **proposal (architect)**. Base: `~/local-operator` @ `633baf258`
(0.54.14). Two PRs (§7). **No `pyproject.toml` bump in either** — the 0.54.14
window is owned by another session.

Every file:line below was read on this base. Where the analysis rests on a
claim I did not verify, it says so and names the evidence that would settle it.

Operator's requirement, verbatim:

> occasionally, sessions like this will just get interrupted after running for a
> while, I think it has something to do with updates ... ends up being quite
> disruptive and results in some sessions being forgotten about if you're running
> many at once for long runtimes

> make sure that errors are properly called out in all situations so at least we
> see it in active sessions as errored

Deliverable: this proposal, the implementation plan in §7, and the test plan in
§8. Copy that design/UX rounds will review is written out concretely in §6.

---

## 1. The problem as found in the code, and what I verified

### 1.1 An in-flight run leaves no durable outcome, and "no outcome" is published as `interrupted`

`Session._prompt_messages` journals `attention_started` with a fresh token
**before** the turn runs (`session/session.py:6508-6520`), and the durable
outcome — `completion_attention` — is written only from that turn's `finally`
via `_publish_attention_outcome` (`session/session.py:6534`, `5949-6003`). A
process that dies mid-turn therefore leaves `attention_started` with no
matching outcome.

`AttentionStore` boot repair then *invents* the outcome
(`session/attention.py:192-244`):

```python
started = transcript.latest_custom("attention_started")
saved = transcript.latest_custom(ATTENTION_CUSTOM_TYPE)
if (isinstance(started, dict) and started.get("conversation_id") == identity
        and (not isinstance(saved, dict) or saved.get("token") != started.get("token"))):
    token = started["token"]
    store.publish(identity, token, provisional_anchor(token), "interrupted")   # :212
    return
```

The kind is the literal `"interrupted"`, with no way to tell a user's `/stop`
from a `SIGKILL`. `Session.__init__` calls this (through `bootstrap_transcript`)
on **every** runtime boot and the mobile daemon calls it on its sweep, so the
lie is written retroactively the next time anything opens the session.

**Verified on this machine** (the manager's case study, session
`80efe201ea61`):

- `attention.db` seq 6440: `conversation=session/80efe201ea61`,
  `token=91a9cc1c-ad8a-44de-b673-27a321910f2f`,
  `anchor=completion-91a9cc1c-…`, **`kind=interrupted`**.
- The transcript's row 0 (09-10 22:11:38) is `attention_started` with exactly
  that token; the transcript contains **two** `attention_started` entries and
  **zero** `completion_attention`.
- `session_incident` count in that session and in its four children: **0**.
- No wake-index entry exists for that session id at all (`~/.local-operator/wakes/`
  has no `80efe201ea61.json`).
- Current runtime for that session, pid 81222, `started_at` 1789135478
  (= 09-11 10:04:38) — i.e. a successor was engaged at the 10:04 restore and is
  alive now. The original runtime is gone with **no** record under
  `~/.local-operator/run/mobile/` (only `81222.json` names the session).

### 1.2 Nothing journals the failure, so no surface and no model learns it

`incidents.py`'s `session_incident` path fires only from in-process error
handling: `Session.journal_incident` (`session/session.py:7364-7395`) is called
from the pipeline's normal path (`:6863-6866`) and from the MCP breaker
(`:7590`). `session/session.py:675-697` renders such an entry into the next
turn's LLM history. A process death writes nothing, so:

- the TUI shows whatever the viewer's local synthesis decided (§1.3);
- the sidebar shows the published kind mapped straight through
  (`session/catalog.py:154-157, 168-171, 255-258, 296-298` → "Interrupted" /
  "Unseen interruption");
- the phone shows the daemon's copy (`mobile/daemon.py:872-878`) — or would,
  where it is not suppressed;
- the next turn and the next `--resume` start blind, having replayed neither an
  incident nor a reason.

### 1.3 The viewer's owner-death path ends the turn with **no reason**

`AttachedSession._go_cold` (`session/attached.py:4091-4175`) ends an in-flight
turn with

```python
self._end_turn_locally(direct=True)          # :4161
```

and `_end_turn_locally` builds `AgentEndEvent(aborted=aborted, generation=0,
error=error)` with `error=None` — `aborted=True, error=None`
(`session/attached.py:3927-3972`). That is the exact shape a user's abort
produces, so the TUI paints `NoticeBlock("interrupted", "warning")`
(`tui/app.py:34287-34298`) for both. `_settle_suspect_turn`
(`session/attached.py:3996-4047`) can do better only from the successor's
`last_turn_outcome`, and only when a successor snapshot exists: it synthesises
`error="turn failed"`, which is a class marker with no cause
(`session/attached.py:4013-4018`, `:4039`).

### 1.4 An involuntary abort that *does* settle is still published `interrupted`

`_publish_attention_outcome` classifies purely off the end event
(`session/session.py:5970`):

```python
kind = "error" if outcome.error else "interrupted" if outcome.aborted else "complete"
```

Every involuntary stop converges on `Session.abort` with `aborted=True` and
`error=None`, so they all become `interrupted`:

- `Session.dispose` → `self.abort("session disposed")`
  (`session/session.py:11938`), reached from `ServingSessionHandle.dispose`
  (`session/runtime/serving.py:807-854`), reached from the runtime's shutdown
  path — SIGTERM/SIGINT **and** the socket `stop` op both set the same event
  (`session/runtime/process.py:560-561`, `:613-616`), and from the reaper's
  `_clean_exit` (`session/runtime/process.py:287-301`);
- the viewer's go-cold synthesis above.

The abort *cause* does exist at the moment it happens — `AbortSignal.reason`
(`harness/types.py:1116-1135`) — and `Session.abort(reason)` already receives
it (`session/session.py:5511-5545`). It is simply dropped: the loop turns an
aborted turn into `AgentEndEvent(aborted=True, error=stream_error)` with no
reason (`harness/loop.py:657-663`, `:1054-1056`).

### 1.5 Restored subagent rows are indistinguishable from a user cancel

`_restored_job_rows` (`session/attached.py:490-526`) rewrites every
non-terminal, non-parked row to

```python
job.model_copy(update={"status": "interrupted", "restored": True})   # :523
```

for the honest reason that "was cut off mid-run" is what the panel can offer to
resume. The row carries no cause, and the roster **record** that does carry one
is not consulted: in the case study's `subagent-roster.v1.json`, job
`da292d4e1688` came back `status: interrupted, restored: true, settled_at:
null`, while its record on the same sidecar reads `outcome: None,
settled_at: None, session_dir: /Users/damian/.local-operator/sessions/fa38095fff26`
— and that child's transcript ends on a `tool` result at 02:46:55 with no
assistant reply, i.e. it was genuinely mid-turn. A sibling record
(`77c35ee4b827`) reads `outcome: completed`, so the evidence to distinguish them
was on disk the whole time.

### 1.6 Update linkage — what I verified, and what I did not

The operator's hypothesis is right, for two distinct reasons. Both verified:

- **Self-refresh retirement is real and frequent.** `~/Library/Logs/local-operator/mobile.log`
  contains **116** `session runtime: retiring for <build>` lines, each preceded
  by `session runtime: build on disk is <new> but this process loaded <old>;
  idle, retiring in Ns so the next engage runs the new build`. Both are emitted
  by `_refresh_for` (`session/runtime/process.py:362-418`). Confirmed by
  `mobile.log:6735059-6735067` (0.54.11@b133eba → 0.54.12@402af7f, 8.4 s
  stagger).
- **A running process can import a mismatched file mid-update.** The log carries
  **605** occurrences of
  `ImportError: cannot import name '_journal_injection_ids' from 'local_operator.session.transcript'`,
  raised from `mobile/durable.py:63` via `daemon.py:110` (`_durable_fold_cache`)
  and surfaced as `durable fold failed for session <id>` — including once for
  `80efe201ea61` (`mobile.log:6731482`). `lop-update` runs
  `uv tool install --force --from <snapshot> local-operator` and writes
  `.lop-source` **after** it (out-of-tree script, `~/.local/bin/lop-update`), so
  a live process has a window in which the installed tree is a mix of two
  builds.

**What I could not prove, and will not assert:** which exit path killed the
case study's original runtime. The evidence is consistent with three, and the
log has no line naming a session on its exit paths, so it cannot discriminate:
(a) a refresh retirement that raced a live turn — but the runtime was parked on
four subagent jobs, so `is_busy()` was True (`session/runtime/serving.py:856-908`:
`_turn_lock` is held for the whole turn, `session/session.py:6368-6378`, and
`running_subagents() > 0`), and `may_refresh` would have returned `"busy"`, so
this needs the §1.6 race *plus* a spurious idle reading; (b) a `SIGTERM` from a
bounce (a descendant of the daemon's LaunchAgent job is the obvious candidate —
`mobile/install.py:301`); (c) a torn-install exception that killed the process.
**Evidence that would settle it:** a per-runtime exit record written on every
exit path (this design's §5.3 marker), or an exit reason line in `amain`
(`session/runtime/process.py:613-653`) — today an exiting runtime logs nothing
about *why*. Settling it is not a prerequisite for the fix: §3 makes every one
of the three visible, and §5 closes the two that are ours to close.

### 1.7 What is already right and is not touched

- A deliberate stop is recognised at the *viewer* when its wire signal is
  present: `AttachedSession._session_was_stopped` distinguishes a stop from
  owner death (`session/attached.py:4049-4076`), and `_reason == STOPPED_REASON`
  short-circuits recovery (`:3903-3917`). The problem is what happens when the
  wire signal is absent (owner death) or when the classifier downstream defaults
  to `interrupted`.
- `may_refresh` is the right *predicate* (`session/runtime/serving.py:1064-1103`);
  §5.1 keeps it and adds a latch, it does not replace it.
- The `retiring` frame and `_go_cold(refresh=True)` are exactly the right
  mechanism for a planned retirement (`session/runtime/server.py:926-959`,
  `session/attached.py:4099-4158`) — this design does not regress
  `docs/design-runtime-autorefresh.md`.
- `_supersedes_provisional` (`session/attention.py:115-147`) is the one place
  that has to be relaxed, and §3.4 says why.

---

## 2. Taxonomy (D1)

`kind` stays a three-value enum. Only its *assignment* changes.

| kind | meaning | assigned when |
|---|---|---|
| `complete` | the turn reached its own end | the turn's terminal event carries no error and no cut-off |
| `interrupted` | a **deliberate** stop: Esc/Ctrl+C mid-turn, `/stop`, another front end's `lop stop` | a deliberate-stop cause was recorded for this turn |
| `error` | the turn was cut off by **anything else**, with a `reason` | an involuntary cut-off, or a provider/tool error (unchanged path) |

**The default flips.** Today "no evidence" means `interrupted`; after this
change "no evidence" means `error`, and `interrupted` requires positive
evidence of a deliberate stop. That is exactly the operator's ask — a cut-off we
cannot explain is not a stop — and it is the safe direction: a deliberate stop
misreported as an error is an annoyance, an involuntary cut-off misreported as a
user abort is the bug being fixed.

Consequences to accept deliberately:

- **A forced stop reports an error.** `lop stop --force` SIGTERMs a runtime that
  has not answered the graceful rung (`session/runtime/control.py`). A live turn
  killed that way is honestly "cut off, not gracefully stopped", and §6's copy
  says so. We do not try to correlate a bare SIGTERM back to an operator intent.
- **A wakeless session's deliberate stop is not recoverable from the wake
  index.** `_mark_wakes_dormant` writes `stopped_at` only when the session has
  schedules (`session/runtime/control.py:550-582`; the TUI mirror returns 0 for
  a schedule-less session, `tui/app.py:25194-25213`), and `announce_stop`'s own
  docstring says the marker approach was dropped for exactly that reason
  (`session/runtime/server.py:870-873`). So §3 records the deliberate stop in
  the *outcome marker itself*, which is the durable artifact restore already
  reads — not in the wake index.

Document to update: `docs/ATTENTION.md` lines 17-19 and 38 (`Error and
interrupted outcomes can have an explicit outcome marker…` / `Resuming an
unfinished journaled run records an interruption…`). That contract must now
read: *resuming an unfinished journaled run records an **error** naming the
cause; only a recorded deliberate stop records an interruption.*

`docs/design-runtime-autorefresh.md` §4.1's table row for a rebind synthesising
`aborted=True` is narrowed, not contradicted: a rebind with
`last_turn_outcome == "error"` now carries the real reason instead of the
`"turn failed"` placeholder (§3.3).

---

## 3. The reason survives (D2)

### 3.1 One additive reason pair, four carriers

Add **two** additive fields, `cause` (machine token) and `reason` (one
operator-facing sentence), and thread them through the four carriers that
already exist. No new vocabulary, no new file format, no `PROTOCOL_VERSION`
bump.

| carrier | today | after |
|---|---|---|
| `AgentEndEvent` (`harness/types.py:1217-1232`) | `aborted`, `error` | `+ cut_off: str = ""`, `+ cut_off_cause: str = ""` — `AgentEvent` is `extra="allow"` (`harness/types.py:1202-1205`), so an **old viewer keeps the field as an extra and never fails validation** |
| `completion_attention` custom entry (`session/attention.py:35`, written at `session/session.py:5987-5995`) | `{conversation_id, token, anchor, kind}` | `+ cause`, `+ reason` |
| `attention.db` `completions` (`session/attention.py:266-271`) | 5 columns | `+ reason TEXT NOT NULL DEFAULT ''` and `+ cause TEXT NOT NULL DEFAULT ''`, added by the **additive-migration pattern already in this file** (`session/attention.py:285-334`): create-if-absent inside the init transaction, deliberately **not** restated in the corruption probe at `:297-299` |
| canonical frontend / mobile projection `attention` state | `AttentionStore._state` dict (`:360-367`) | `+ "reason"`, `+ "cause"`; `state_many` already does `SELECT c.*` (`:400-407`) so the column flows with a two-line change, and the synth-empty dict (`:377-387`) needs both keys |

`AttentionStore.publish(conversation, token, anchor, kind, *, baseline_seen=None)`
(`session/attention.py:422`) gains `reason: str = ""` and `cause: str = ""`;
the `INSERT OR IGNORE` at `:513-516` and the supersede `UPDATE` at `:504-507`
both carry them.

### 3.2 The live path: classify at the abort boundary

`Session` gains one field and one classification, both small:

```python
# session/session.py, near _last_turn_outcome (:2150)
#: Why the CURRENT turn ended involuntarily, or "" when it ended on its own or
#: was deliberately stopped. Set by the runtime's own exit paths (a retire that
#: caught a live turn, a termination signal, disposal) and consumed by
#: ``_classify_cut_off`` and ``_publish_attention_outcome``. Deliberately NOT
#: derived from ``AbortSignal.reason``: ``abort("session disposed")`` is
#: produced by BOTH a user's /stop and a SIGTERM, so the reason string cannot
#: classify the trigger.
self._cut_off_cause: str = ""
```

```python
def _classify_cut_off(self, event: AgentEvent) -> AgentEvent:
    """Re-label an involuntary abort as a CUT-OFF error before it is emitted.

    A deliberate stop keeps today's shape (``aborted=True, error=None`` →
    `interrupted`). An involuntary one is rewritten to ``aborted=False,
    error=<sentence>`` so every existing surface does the right thing without
    being taught a new field: ``on_turn_ended`` appends its error notice
    (`tui/app.py:34007-34008`), ``_finalize_turn`` does NOT append the
    `interrupted` notice (`:34287`), the title takes the ✗ mark
    (``failed=bool(error) and not aborted``, `:34259`), ``_last_turn_outcome``
    becomes ``"error"`` (`session/session.py:6255-6256`), and
    ``_publish_attention_outcome`` publishes ``error`` (`:5970`). An OLD viewer
    that has never heard of ``cut_off`` gets the same behaviour, which is the
    backwards-compatibility requirement.
    """
```

Called from the top of `Session._emit` (`session/session.py:6248`) for
`AgentEndEvent` instances only. The `cut_off`/`cut_off_cause` fields ride along
so a *new* viewer can name the cause precisely (§3.3) without re-deriving it.

`_publish_attention_outcome` becomes:

```python
cut_off = bool(self._cut_off_cause)
kind = "error" if (outcome.error or cut_off) else "interrupted" if outcome.aborted else "complete"
reason = outcome.error or _render_cut_off(self._cut_off_cause)
cause = self._cut_off_cause if cut_off else ""
```

and passes `reason=`/`cause=` to both the journal entry and `AttentionStore.publish`.
It also journals the incident once (§3.5).

`self._cut_off_cause` is cleared at the start of every turn alongside
`self._attention_outcome` (`session/session.py:6508`), so a stale cause cannot
label a later, healthy turn.

### 3.3 The `cause` vocabulary

Authored by the runtime, not classified from provider text — so it is a small
module constant, rendered with the incident formatter's shape rather than a
parallel prose system. `incidents.py` gains:

```python
#: Why a turn was cut off. Harness-authored, so unlike _RULES these are exact
#: tokens rather than substring guesses over vendor text; the rendering borrows
#: Incident's shape but not its classifier.
CUT_OFF_CAUSES: dict[str, str] = {
    "user-stop": "the session was stopped by the user",
    "runtime-retired": "the runtime retired so the next engage would run a newer build",
    "runtime-shutdown": "the runtime was terminated while this turn was running",
    "runtime-killed": "the runtime disappeared without exiting cleanly while this turn was running",
    "install-mid-update": "a local-operator install was being replaced on disk while this turn was running",
    # About what the VIEWER can verify, not about what happened to the process:
    # the arm that paints it is the recovery loop's give-up, which also fires
    # for a live-but-silent owner (record present, pid alive, nothing answers).
    "owner-lost": "the session's runtime stopped answering while this turn was running",
    "disposed": "the session was disposed while this turn was running",
}
```

and `_HINTS["cut-off"]` (the same shape as the existing entries at
`incidents.py:217-241`):

> "The runtime was cut off before this turn produced a result. The transcript
> holds whatever was written before it stopped and nothing after. Check the
> state of anything it was mid-way through before repeating the work; do not
> assume the request completed."

plus `format_cut_off_message(cause, *, detail="") -> str`, which builds an
`Incident(category="cut-off", raw=<CUT_OFF_CAUSES[cause] + detail>)` and calls
`Incident.render()` — the same `[session incident] cut-off: … / suggested action:
… / This is why the previous turn ended.` text (`incidents.py:257-267`), so the
next turn's card and the resume replay are the existing surface, not a new one.

The `detail` is where the build pair goes, e.g. for `runtime-retired`:
`" (0.54.11@b133eba → 0.54.12@402af7f)"`. `_refresh_for` already has both stamps
(`session/runtime/process.py:384-392`), so it passes them to the handle.

### 3.4 Restore: default to `error`, reconstruct the cause

`_import_transcript_outcome` (`session/attention.py:192-244`) changes in three
ways.

1. **Do not publish for a run that may still be live.** Before publishing the
   provisional marker, consult the run registry: if
   `find_runtime_record(config_dir, identity_session_id)` returns a record whose
   pid is alive and whose record still describes that session (the liveness and
   pid-reuse checks already exist in `session/runtime/control.py:480-516`),
   publish **nothing** — the turn is in flight and will publish its own outcome.
   This is the false-alarm guard. Without it a daemon sweep mid-turn would
   write a wrong `error` row for a healthy run, and that row would be
   *silently* wrong rather than loudly so: the phone and the sidebar both
   suppress an outcome mark while the session's live state is `busy`
   (`tui/app.py:19080`, `session/catalog.py:240-242`) and the daemon
   suppresses the notice while `projection.streaming` is true
   (`mobile/daemon.py:872`). The guard removes the class instead of
   relying on every surface's suppression.
2. **Kind from evidence, not by default.** With no live owner:
   - a `stopped_at` marker on the wake index (when the session has schedules)
     → `interrupted`, `cause="user-stop"`;
   - a record present whose pid is **dead** → `error`,
     `cause="runtime-killed"`, detail from the record (`<version>@<ref>`, pid,
     `started_at`);
   - a `turn_cut_off`/outcome marker from an earlier run of the same token
     (§3.2's live path) → that marker's `kind`/`cause`/`reason`, verbatim;
   - nothing at all → `error`, **no cause**, reason `CUT_OFF_UNKNOWN`
     ("the turn was cut off and the cause could not be determined"). The first
     draft named `runtime-killed` here with a "(the cause could not be
     determined)" parenthetical, and every surface that trims a parenthetical —
     the sidebar's `_error_label`, which drops one by design for the build pair
     — asserted a mechanism in the one case where nothing is known (design/UX
     review round 1, D3/U3). Leaving `cause` empty also keeps
     `cause_from_reason(reason)` — the inverse every reader uses — agreeing with
     the sentence instead of naming an unobserved cause.
3. **The provisional anchor stays.** `provisional_anchor(token)` is the real
   "this record is replaceable" signal. `_supersedes_provisional`
   (`session/attention.py:115-147`) currently *also* requires
   `stored_kind == "interrupted"` (`:145`); that clause must be dropped so a
   provisional record published as `error` is still superseded by the same
   token's real `complete`. The anchor, not the kind, is the shape. Without this
   the change would brick exactly the session the docstring at `:433-443`
   describes. Pin it with the test in §8.1.

### 3.5 Journals on restore, once per orphaned run

**The saved-marker branch journals too** (QA round 1, Q-2; review MINOR-2). A
runtime that publishes its own outcome and then dies cannot narrate it:
`journal_incident` refuses once `_disposed` is set, and the dispose rung sets
that flag before the turn's `finally` publishes — so the whole update/shutdown/
retire family reached every SURFACE and no MODEL, which is exactly the operator's
reported family ("interruptions … have something to do with updates"). The
successor is the only writer left, so `_import_transcript_outcome` returns the
tuple for a saved marker whose `kind == "error"` **and** whose `cause` is a
`CUT_OFF_CAUSES` token. Both conditions are load-bearing: a provider error
carries no cause and a deliberate stop carries `user-stop`, and narrating either
as a cut-off incident would commit the taxonomy's own misclassification in the
reader meant to repair it. Deduped on the token like the orphaned-run half, so a
second boot narrates nothing.

`bootstrap_transcript` is called from `Session.__init__` (where a `Session`
exists to journal) **and** from the mobile daemon sweep (where it does not). So
the journaling belongs at the caller that has a session:

- `_import_transcript_outcome` returns the `(kind, cause, reason, token)` it
  published (or `None`);
- `Session.__init__`/`refresh_attention` calls `journal_incident(...)` with
  `format_cut_off_message(...)` **once per token** — dedupe on the token already
  journaled (`transcript.latest_custom(SESSION_INCIDENT_MESSAGE_TYPE)` carrying
  the token in `details`), so re-opening a session does not re-narrate the same
  death. `journal_incident` already persists then parks the live-context append
  (`session/session.py:7379-7395`, `_append_or_park_journal:7308-7339`), so it is
  durable even if the process dies immediately after; the parked notice flushes
  at the next turn boundary or is replayed into the LLM history on resume
  (`session/session.py:675-697`).

---

### 3.6 Where the deliberate verdict is written (review round 1, BLOCKER-1)

`interrupted` requires POSITIVE evidence of a deliberate act, and the verdict has
to be recorded while the act is still knowable — `Session.dispose()` notes
`disposed` unconditionally, so an un-noted dispose publishes the user's own cancel
as `kind=error, cause=disposed`.

Only the `stop` control op used to record it. The route a bare `/stop` actually
takes on a TUI-owned session is `tui/app.py::_stop_local_session` → `await
session.dispose()`, with a peer's `lop stop` reaching the same method through
`TuiSessionHandle.request_stop` — so the DEFAULT path published a false failure,
and neither the unit guard (which set `_attention_outcome` directly) nor the e2e
cell (which used `stop_session`, the one rung that did record it) could see it.
Measured with the real `Session.dispose()`: unnoted → `error`/`disposed`, noted →
`interrupted`/`user-stop`.

The verdict is therefore written at the deliberate rungs, and only there:

* `OperatorApp._stop_local_session` (before the dispose),
* `ServingSessionHandle._note_deliberate_stop`, called by `request_stop`
  (unchanged), `abort` (the phone's stop button) and `cancel_gracefully` (a
  supervisor's cancel) — one helper, so the three stay in step,
* and NOT in `Session.dispose()` itself, whose involuntary teardowns (a reload,
  an unmount, `_mobile_teardown`) are cut-offs and must keep saying so.

A retirement outranks all three: while `_retiring_cause` is latched, the runtime
is already ending that turn for its own reason and the cut-off verdict belongs to
the retire path, which recorded it when it latched.

## 4. Restored subagent rows (D3)

`_restore_cold_subagents` already reads the roster **records** for accounting
(`session/attached.py:1169-1208`) — the same sidecar that carries
`records[].outcome`, `records[].settled_at` and `records[].session_dir`
(verified on the case study's sidecar). So `_restored_job_rows`
(`session/attached.py:490-526`) takes the records too, and a row resolves in
this order:

1. **The record settled it.** `outcome` in `{completed, failed, error,
   interrupted}` → the row says that, not `interrupted`. This alone repairs job
   `77c35ee4b827` (record `outcome: completed`, row restored `interrupted`).
2. **The record did not, and the child's own transcript decides.** Read
   `records[i].session_dir`'s transcript tail:
   - last row is a `tool` result, or an assistant message whose `tool_calls`
     have no results → the child was mid-turn → `interrupted`, with a
     `cut_off_cause` (the parent's death, or the child's own stop) on the row;
   - last row is an assistant message with no pending tool call → the child
     **finished** but the parent never recorded it → report the child's own
     last-turn outcome (`completed`), not `interrupted`.
3. **No record and no transcript** → `interrupted` with `cause="owner-lost"`,
   today's behaviour.

The row gains an additive `cut_off_cause: str = ""` field, so the subagent panel
can show *why* rather than only that it stopped.

**RESOLVED ON BOTH WRITERS, from one function** (UX round 1, U2). There are two
restorers of the same roster: this cold viewer, and the successor runtime's
`AsyncJobManager` — whose table is what the session publishes as the live
roster. Written separately, the viewer's resolved rows survived under a second:
measured at t+1.2 s `completed|interrupted|completed|interrupted`, and by
t+1.9 s all four back to a blanket `interrupted` with the cause dropped, because
the owner's restore replaced them. The resolution therefore lives in
`session/restored_rows.py` and both paths call it (`Session._load_subagent_roster`
before `jobs.restore`, and `_restore_cold_subagents`).

`cut_off_cause` is **deliberately not persisted**: the sidecar's `jobs` rows are
validated by strict `extra="forbid"` models, so a field an older owner does not
know makes it DROP the whole row at resume — a worse degradation than losing a
cause that costs nothing to re-derive. `_ROSTER_ROW_FIELDS` stays as it is, and
`JobState.from_job` carries the value from the manager's row to the wire. The three-surface vocabulary
already exists (`tui/widgets/subagent_panel.py:230-260`, `:385-390`,
`subagent_view.py:466`, `:1186-1189`) and a design round decides whether the
panel prints the cause inline or behind the existing row detail.

Cost note: rule 2 reads up to N child transcripts on a cold restore. Bound it —
only for rows whose record's `outcome` is `None` (the case study had two of
four), and read the tail only, off the loop (the call site already hops to a
worker for the sidecar read). If a tail read raises, fall through to rule 3.

---

## 5. The update path cannot silently cut a live turn (D4)

### 5.1 Is `may_refresh` sufficient? No — it is the right predicate, sampled, not held

`may_refresh` (`session/runtime/serving.py:1064-1103`) is `not is_busy()` and
`not _wake_within_window()`, which is the right authority. But both retire paths
sample it and then act across an `await`:

- the reaper re-checks immediately before `_clean_exit`
  (`session/runtime/process.py:409-417`) — and the code **admits the race** in
  its own comment at `:376-379`: *"A turn that starts between THIS re-check and
  `_clean_exit` is aborted by the dispose exactly as a `stop` op racing a turn
  is"*;
- the viewer-driven `_retire_for` re-asks after `announce_retiring` and then
  calls `request_stop()` (`session/runtime/server.py:2298-2314`);
- the quiet idle exit has the same shape at the top of `_reaper`
  (`session/runtime/process.py:335-359`).

`dispose()` is async, so the loop is free in that gap, and a `prompt`/`peer_message`
arriving over the socket can open a turn that the dispose then aborts. So **the
"idle" claim is checkable but only at one instant**, which is the defect.

**Recommendation: keep `may_refresh` and add a latch, so the claim is true by
construction.**

```python
# ServingSessionHandle
def begin_retire(self, cause: str) -> bool:
    """Commit this runtime to retiring, iff it is idle RIGHT NOW.

    Sets ``_retiring_cause`` in the same synchronous step that checks
    ``may_refresh()``, and from that instant the admission paths
    (``prompt`` / ``receive_peer_message``) REFUSE rather than queue: a
    runtime that is leaving must not start a turn it will abort one await
    later. The refusal names the cause, and the sender's next engage spawns
    the new build — which is what ``_refresh_for``'s comment already claims
    happens today, made true instead of merely likely.
    """
```

The two retire paths call `begin_retire(cause)` instead of a bare re-check and
dispose only on `True`; `"kept: <reason>"` otherwise (the existing answer shape
at `session/runtime/server.py:2296-2309`). The quiet idle exit uses the same
latch with `cause="idle-exit"`. Roughly 40 lines plus tests.

### 5.2 A mismatched import must be a loud, named failure

Verified shape (§1.6): a lazy `from local_operator… import x` inside a running
process raises against a half-replaced tree, and today it surfaces as an
arbitrary traceback (`durable fold failed for session X`) or a random tool
error. **Recommendation: name it, and feed it into §3's vocabulary as
`install-mid-update`.**

- New helper in `update.py` (next to `installed_build`, `:804`):
  `classify_import_failure(exc: BaseException, module: str, *, boot: BuildStamp | None) -> str | None`
  — returns a rendered reason when the exception is an `ImportError` naming a
  `local_operator.*` module **and** `installed_build()` differs from `boot`
  (the stamp the process loaded at boot, `session/runtime/server.py`'s
  `_boot_build`). It returns `None` for a genuine packaging bug, so we do not
  mislabel one.
- The two seams that actually see it — the mobile daemon's
  `_durable_fold_cache`/`_durable_projection` (`mobile/daemon.py:110`, `:535`)
  and the harness's own lazy imports inside a turn — log at ERROR with that
  reason instead of a bare traceback, and a turn aborted by it ends as an
  `error` with `cause="install-mid-update"`.
- **Out of scope, deliberately:** preventing the torn window. `uv tool install
  --force` replaces the tree in place; the only complete fix is an atomic
  install (write a new directory, swap a symlink), which is uv's behaviour, not
  ours. The settle window (`BUILD_SETTLE_S`, `session/runtime/process.py:93`)
  and §5.1's latch prevent the *retirement* half; the import half is reported
  honestly and stops being an unexplained traceback. An optional follow-up (not
  this PR): a `<tool dir>/.lop-updating` sentinel written by `update.py` and by
  the out-of-tree `lop-update` around the install, so the reason does not have
  to be inferred from the stamp. It requires editing `~/.local/bin/lop-update`,
  which is not in this repo, so it cannot ride a repo PR unaltered.

### 5.3 Every exit path should say why it exited

Cheap and it is what would have settled §1.6. `amain`'s shutdown
(`session/runtime/process.py:613-653`) and `_clean_exit` (`:287-301`) log one
INFO line naming the trigger and pid, e.g.
`session runtime: exiting (SIGTERM, pid 1234, 0.54.11@b133eba)`. Not a
requirement of the operator's ask, but it is one line each and it makes the next
occurrence diagnosable from the log instead of by inference.

---

## 6. Surfaces and copy (for the design/UX rounds)

Every string below is user-visible and is what the design round reviews. The
cause sentences are §3.3's, rendered as `format_cut_off_message`.

### 6.1 TUI — the moment it happens (a watched session)

`tui/app.py:34007-34008` already appends `NoticeBlock(self._with_recovery_hint(message.error), "error")`
on every terminal event carrying `error`, and `_finalize_turn` does **not**
append its `interrupted` row when `aborted=False`. So the live copy is just the
`error` sentence:

> ✗ turn cut off — the runtime was terminated while this turn was running. The
> transcript holds what it wrote before that and nothing after.

`_with_recovery_hint` (`tui/app.py:34074-34139`) appends a remedy only when the
text is recognised as an MCP or provider auth failure, so a cut-off sentence
passes through unchanged.

**Tool cards.** `_finalize_turn` retires stranded live cards, and the WORD comes
from the turn's verdict (`widgets/tool_card.py::mark_interrupted(cut_off=...)`).
This was offered to the design round as a choice between reusing `⊘ interrupted`
and adding a state; the design round took the word (round 1, D2) and it is the
cheaper of the two — the glyph, the column and the dim tier are untouched, so no
new design surface, and the row stops calling the same death `interrupted` under
a `✗ turn cut off` notice. Rendered: `bash  sleep 30   cut off ⊘  9.7s`.

**The bound the paint lands on.** The recovery loop checks its deadline at the
top of a pass, so a pass that slept its pacing delay past `COLD_FALLBACK_S`
reported the cut-off one sleep late — measured at 9.4 s against a bound of 8.0.
The sleep is now capped at the deadline (`paced` in `_recover_runtime`), and the
watched SIGKILL paints at 8.03 s measured from the kill (the residual hundredth
is the notification lag before the loop's deadline even starts: the loop's clock
begins when it notices the drop, not when the process died).

### 6.2 TUI — the next resume / an unwatched session

The attention poller already paints, for an unseen `error`/`interrupted`
outcome (`tui/app.py:19238-19266`):

```python
block = NoticeBlock("Stopped with an error" if state["kind"] == "error" else "Interrupted")
```

Change the error branch to carry the reason:

> ✗ Stopped with an error — the runtime disappeared without exiting cleanly
> while this turn was running.

and generalise `_adopt_own_interrupt_notice` (`tui/app.py:18467-18511`) to
adopt an `error` row too. **The row's tier comes with it** (design round 1, D1):
the error branch passed no kind, so it took `NoticeBlock`'s `info` default — a
dim `·` in the same fill as the routine `Interrupted` control one branch down —
for an outcome the live surface paints `✗` danger. Both surfaces now take the
sentence AND the tier from `harness/rows.py::completion_notice`, which is where
this repo's row decisions belong; `interrupted` deliberately stays `info` on the
replay (the live row's `warning` is a turn-scoped statement a receipt is not
making). **Without that generalisation a cut-off paints twice**:
the live `on_turn_ended` error notice has no `completion_anchor_id`, so the
poller cannot dedupe it and appends its own. The same duplicate is believed to
exist **today for provider errors** — `_own_interrupt_notice` is only ever set
in the aborted branch (`tui/app.py:34287-34308`) and `_adopt_own_interrupt_notice`
returns `False` for any other kind (`:18495`) — so this repair is also a
pre-existing bug fix, not just new plumbing. Verify it with the two-line test in
§8.1 before relying on it.

### 6.3 Sidebar tooltip / status (`session/catalog.py`)

No mapping change is needed — `error` already renders "Error" / "Unseen error"
(`catalog.py:154-157, 168-171, 255-258, 296-298`). Append the reason to the two
`Unseen …`/`…` error spellings:

> `Unseen error — the runtime retired so the next engage would run a newer build`

Long reasons get the cause sentence only (not the build pair); the tooltip has
one line. `shows_completion_mark` keeps its existing precedence
(`catalog.py:174-242`) — a busy session still shows the spinner, which is
correct and unchanged.

### 6.4 Mobile daemon (`mobile/daemon.py:872-878`)

```python
text="Stopped with an error" if attention["kind"] == "error" else "Interrupted"
```

becomes

```python
text, severity = completion_notice(attention["kind"], attention.get("reason") or "")
# ... TranscriptEntry(..., text=text, details={"severity": severity})
```

BOTH halves matter, and the second was missing at first (design round 1, D4):
the frame carried an empty `details`, and the phone's `NoticeRow` picks its glyph
and ink from `details.severity` — so a cut-off rendered as the same dim `·` as
`Interrupted`, a routine receipt, on the surface the operator reads from a phone.
The severity is derived in `harness/rows.py` (the module that owns these rows for
both surfaces) rather than at the serialization boundary, so the TUI's poller and
the phone cannot disagree about the tier of one outcome.

**And `stop_reason` must not read a cut-off as a completion** (review round 1,
MAJOR-1). `fold_event` set `stop_reason = "aborted" if event.aborted else
"completed"`, and the taxonomy flip reports a cut-off as `aborted=False` — so the
phone told the user the turn had FINISHED and silently dropped
`interrupted — tap to resume`, the only recovery affordance there is, for exactly
the sessions the operator reports losing. It now prefers `cut_off`/`cut_off_cause`
(§9 risk 2's third reader of `aborted`, resolved in favour of the reader).

Phone notification copy is unchanged (`tui/notify.py:139-146, 197-207` already
has a distinct `interrupted` category and an `error` category —
`CONTEXT_ATTENTION` / `BODY_ERROR`); a cut-off now takes the error one, which is
the operator's ask. Background banners stay gated on `live_state in {"", "idle"}`
(`tui/app.py:666`, `:19080`), so a live mid-turn session still never banners.

### 6.5 The next turn's `[session incident]` card

Rendered by `_default_convert_to_llm` (`session/session.py:675-697`) from the
`session_incident` entry, via `format_cut_off_message`:

> [session incident] cut-off: the runtime retired so the next engage would run a
> newer build (0.54.11@b133eba → 0.54.12@402af7f).
> suggested action: The runtime was cut off before this turn produced a result.
> The transcript holds whatever was written before it stopped and nothing after.
> Check the state of anything it was mid-way through before repeating the work;
> do not assume the request completed.
> This is why the previous turn ended. Take it into account before repeating the
> same request.

---

## 7. Implementation plan

Split into two PRs so the taxonomy/reason work (which touches the wire, the DB
and every surface) lands and is reviewed before the runtime-lifecycle work.

### 7.1 PR A — taxonomy, reason, restore, surfaces

| file | change |
|---|---|
| `session/attention.py` | `_state`/`state_many`/synth state carry `reason`,`cause` (`:351-417`); `publish(..., reason="", cause="")`; additive `ALTER TABLE` migration for the two columns in the `:285-334` pattern; `_supersedes_provisional` drops the `stored_kind == "interrupted"` term (`:145`); `_import_transcript_outcome` returns `(kind, cause, reason, token)` and defaults to `error` with the live-owner guard (§3.4) |
| `session/session.py` | `_cut_off_cause`; `_classify_cut_off` called from `_emit` (`:6248`); `_attention_outcome` rewrite in `_publish_attention_outcome` (`:5970`); `reason`/`cause` on the journal entry (`:5987`) and `publish` (`:5996`); clear `_cut_off_cause` at `:6508`; journal the restored cut-off once per token (§3.5); `note_cut_off(cause)` / `note_deliberate_stop()` |
| `incidents.py` | `CUT_OFF_CAUSES`, `_HINTS["cut-off"]`, `format_cut_off_message` (`:217-282` area) |
| `harness/types.py` | `AgentEndEvent.cut_off`, `.cut_off_cause` (`:1217`) |
| `session/frontend_state.py` | `last_turn_cut_off: str = ""` alongside `last_turn_outcome` (`:1586`), set in the same `_run_turn_pipeline` `finally` (`session/session.py:6256`), copied in `_last_turn_outcome_from`'s neighbours (`:3904`) |
| `session/attached.py` | `_settle_suspect_turn` synthesises `error=last_turn_cut_off` (real reason) instead of `"turn failed"` (`:4039`); `_go_cold`'s non-refresh branch ends the turn as `owner-lost` rather than a bare abort (`:4161`) |
| `session/runtime/serving.py` | `note_cut_off("runtime-shutdown")` / `note_deliberate_stop()` on the stop rung; `may_refresh` unchanged |
| `tui/app.py` | poller copy (`:19261-19263`); `_adopt_own_interrupt_notice` generalised to `error` (`:18467-18511`) |
| `mobile/daemon.py` | reason in the notice copy (`:872-878`) |
| `session/catalog.py` | reason appended to the two error spellings (`:154-171, 255-258, 296-298`) |
| `docs/ATTENTION.md` | taxonomy contract (§2) |

### 7.2 PR B — the update path and restored rows

| file | change |
|---|---|
| `session/runtime/serving.py` | `begin_retire(cause)`; refuse `prompt`/`receive_peer_message` once retiring (§5.1) |
| `session/runtime/process.py` | `_refresh_for`/quiet-exit use `begin_retire`; pass the build pair to the handle; exit-reason log line (§5.3) |
| `session/runtime/server.py` | `_retire_for` uses `begin_retire` (`:2273-2314`) |
| `session/runtime/control.py` | the stop rung records a deliberate cause before triggering |
| `update.py` | `classify_import_failure` (§5.2) |
| `mobile/daemon.py` | the fold/projection seams log the named failure (`:110`, `:535`) |
| `session/restored_rows.py` | NEW: the one row-resolution policy, called by BOTH restorers (§4) |
| `session/attached.py` | `resolve_restored_rows(jobs, records=...)` on the cold path; the named `owner-lost` verdict on both legacy synthesised ends (§6.1) |
| `session/session.py` | `_load_subagent_roster` resolves the rows through the same function before `jobs.restore` (§4) |
| `harness/jobs.py` | the non-persisted `cut_off_cause` on `AsyncJob`, carried so the wire row can say why (§4) |
| `harness/rows.py` | `completion_notice` — the returned-to row's sentence and tier, for both surfaces (§6.2, §6.4) |
| `mobile/projection.py` | `stop_reason` prefers `cut_off`/`cut_off_cause`, so a cut-off keeps the phone's resume affordance (§6.4) |
| `tui/app.py`, `tui/widgets/tool_card.py`, `tui/events.py` | the end event's cut-off verdict reaches the stranded card WORD (§6.1); `_stop_local_session` records the deliberate verdict (§3.6) |
| `docs/design-cut-off-turns.md` | this file, marked implemented |

Order: PR A first (PR B's `begin_retire` reason and the row causes render
through PR A's vocabulary).

---

## 8. Test plan

### 8.1 Unit

`tests/unit/session/test_attention.py` (extend):
1. `_import_transcript_outcome` with `attention_started` and no outcome, no
   record → publishes **`error`**, `cause="runtime-killed"`, non-empty `reason`.
   *This fails on `633baf258`* — the same fixture asserts `interrupted` today at
   `:198`.
2. Same, with a live record for the session → publishes **nothing** (the
   false-alarm guard).
3. Same, with a dead-pid record → `error` with the record's build/pid in the
   detail.
4. Same, with a `stopped_at` wake-index entry → `interrupted`,
   `cause="user-stop"`.
5. Supersede: a provisional record published as **`error`** is still superseded
   by the same token's real `complete` (the `_supersedes_provisional`
   relaxation — this fails before the change).
6. `publish(..., reason=, cause=)` round-trips through `state`/`state_many`, and
   an **old** DB (created without the columns) migrates additively without
   reading as corrupt.

`tests/unit/session/test_cut_off_turns.py` (new):
7. `_classify_cut_off`: `_cut_off_cause=""` + `aborted=True` → unchanged
   (`aborted=True, error=None`); `_cut_off_cause="runtime-shutdown"` +
   `aborted=True` → `aborted=False, error=<sentence>`, `cut_off_cause` set.
8. `_publish_attention_outcome` with a cut-off outcome → kind `error`, the
   reason and cause on the journal entry and in the store.
9. A deliberate stop (`abort("interrupted")`) still publishes `interrupted` with
   no reason — the regression guard for the whole taxonomy change.
10. The next turn's rendered LLM history contains the `[session incident]`
    text after a restore (replay through `_default_convert_to_llm`).

`tests/unit/session/runtime/test_process_cut_off.py` (new):
11. `begin_retire` returns True when idle and False when a turn is held; after
    True, a `prompt`/`peer_message` is refused with a named reason and no turn
    opens.
12. `_refresh_for` with a turn held → no announce, no exit; released → retires.
13. `_classify_import_failure` returns a reason for an `ImportError` naming a
    `local_operator.*` module when the stamp moved, and `None` when it did not.
14. `_restored_job_rows` with records: `outcome="completed"` → `completed`; a
    child tail ending on a tool result → `interrupted` + cause; a child tail
    ending on an assistant message → the child's outcome.

`tests/unit/tui/test_cut_off_notices.py` (new):
15. A cut-off end paints exactly **one** row: the error notice with the reason,
    never the `interrupted` notice (the `aborted=False` consequence).
16. A provider-error end paints exactly one row (the pre-existing duplicate,
    now adopted).
17. The poller's row carries the reason and is adopted when the live row exists.

Plus copy assertions: `"Interrupted" not in transcript_text(app)` for a cut-off,
`"Stopped with an error —"` present.

### 8.2 E2E — `tests/e2e/test_cut_off_turns_e2e.py` (new; follow
`tests/e2e/test_runtime_refresh_e2e.py`'s shape: production `process.py` in a
subprocess, isolated `HOME` + config dir, every `CMUX_*` unset, synthetic ids)

| cell | drive | assert |
|---|---|---|
| A (the case-study shape) | boot a real runtime, hold a turn open on a `wait`-shaped tool, `SIGKILL` it (leaves the record), then boot a successor | record present+dead ⇒ `kind=error`; `reason` non-empty; exactly one `session_incident` in the transcript; the next turn's history contains `[session incident]`; `catalog` status contains "Unseen error"; the phone projection's notice contains the reason |
| B (graceful termination) | as A but `SIGTERM` mid-turn with an attached headless viewer | the dying runtime writes the marker itself; the viewer paints an error notice naming the cause within 1 s; durable kind `error`, `cause="runtime-shutdown"` |
| C (deliberate stop) | `lop stop` mid-turn | durable kind `interrupted`, `cause="user-stop"`, **no** `session_incident` |
| D (retire latch) | hold a turn open, flip the build stamp, then fire a `prompt` in the window | the runtime does **not** retire; the prompt is refused with a named reason; the next engage runs the new build; no turn is aborted |
| E (torn install) | monkeypatch a lazy `local_operator.*` import to raise mid-turn | the turn ends `error` with `cause="install-mid-update"`; the log line names it; no bare traceback as the only evidence |

**Reproduction recipe for cell A, runnable by hand** (this is the shape the
manager's case study took, and the pre-fix run must be recorded first):

```sh
export HOME=$(mktemp -d); export LOCAL_OPERATOR_CONFIG_DIR=$HOME/.local-operator
unset ${(k)CMUX_*} 2>/dev/null || true
# 1. start a runtime for a fresh session, scripted to park in a long tool
.venv/bin/python -m local_operator.cli --new <<<"run the scripted park"
PID=$(cat "$LOCAL_OPERATOR_CONFIG_DIR/sessions/<id>/.session.pid")
# 2. SIGKILL it mid-turn — this is the case study's death, hard-kill shape
kill -9 "$PID"
# 3. pre-fix: boot the successor and read the store
.venv/bin/python -c "from local_operator.session.attention import AttentionStore; print(AttentionStore().state('session/<id>'))"
#    => {'kind': 'interrupted', ...}   ← the bug
# 4. post-fix: same boot
#    => {'kind': 'error', 'reason': '...', 'cause': 'runtime-killed'}
grep -c session_incident "$LOCAL_OPERATOR_CONFIG_DIR/sessions/<id>/transcript.jsonl"   # 1, was 0
```

### 8.3 QA matrix cells (qa-tester, real execution, isolated config per cell —
never the operator's live `~/.local-operator`)

- A/B/C/D/E above by hand, each with commands and captured output.
- A re-run of A against **pre-fix** `633baf258` to show `interrupted` + 0
  incidents, then against the branch.
- A `SIGKILL` of a **watched** session, to confirm the viewer's cold path paints
  the error inside `COLD_FALLBACK_S` and never `Interrupted`.
- A `lop stop` of a **wakeless** session with a live turn, to confirm the
  deliberate path still reports `interrupted` (the §2 consequence).
- An old-DB upgrade cell: copy a pre-change `attention.db`, boot, confirm the
  additive migration and that no historical row is re-announced.

---

## 9. Risks, and what I deferred

**Risks to watch in rollout**

1. **The false-error risk is the one that matters.** Defaulting to `error` means
   any bug in the "is this run live?" guard (§3.4) paints a healthy session as
   failed. The guard and its unit test (8.1 test 2) are the load-bearing part of
   PR A, and the daemon sweep is the path most likely to trip it. Watch the
   phone in rollout for an error badge on a session that is working.
2. **The `aborted=False` rewrite is the riskiest single semantic choice.**
   Anything that reads `aborted` as "the turn was interrupted" now sees
   `error` instead. §6.1's tool-card vocabulary is the visible consequence, and
   test 15 pins the notice count. **The third reader this risk predicted DID turn
   up**: `mobile/projection.py`'s `stop_reason` (round 1, MAJOR-1), which the
   phone uses to decide whether to offer `interrupted — tap to resume` — the
   predicted fallback (prefer `cut_off`/`cut_off_cause`) is what was applied, so
   `aborted` is left alone and no surface has to be taught a new field. Treat this
   entry as answered rather than as a standing option.
3. **Two rows for one outcome.** The dedupe between the live turn-end notice and
   the attention poller is anchored on `completion_anchor_id`, which the live row
   cannot carry at paint time. §6.2's generalisation of `_adopt_own_interrupt_notice`
   is what keeps it at one row; test 15/16 pin it.
4. **Restore read amplification.** D3 rule 2 reads child transcripts on a cold
   restore; bounded to records with `outcome is None`, but a session with
   dozens of unsettled children is the case to measure.

**Deferred, with the reason**

- **Atomic install** (write a new prefix, swap a symlink) so there is no torn
  window at all. It is `uv tool install --force`'s behaviour, not this repo's;
  §5.2 makes the failure honest instead. Reason for deferring: a repo PR cannot
  fix it.
- **The `.lop-updating` sentinel.** Requires editing `~/.local/bin/lop-update`,
  which is outside this repository. The stamp comparison in
  `classify_import_failure` covers the same ground with no out-of-tree edit.
- **Rekeying the wake index for wakeless sessions** so `stopped_at` always
  exists. Not needed: §3 records the deliberate stop in the outcome marker
  itself, and `announce_stop`'s docstring (`session/runtime/server.py:870-873`)
  is an explicit decision against the marker approach.
- **Retro-classifying historical `interrupted` rows.** The cause is not
  recoverable from the store; a migration would guess. Left alone.
- **Settling the case study's exact death (a/b/c in §1.6).** §5.3's exit-reason
  log is the cheap answer and lands in PR B; a py-spy/`faulthandler` capture at
  the moment of a future death would be conclusive and is not needed to ship.

**Deliberately NOT done**

- No `PROTOCOL_VERSION` bump: every wire addition is a new field on an existing
  frame or a new optional key in the additive `attention` state, and
  `AgentEvent` is `extra="allow"`, so an older runtime or viewer degrades to
  today's behaviour.
- No second vocabulary: the reason rides `Incident`'s renderer and
  `journal_incident`, and `interrupted` keeps its existing spelling everywhere.
- No version bump in either PR.

---

Model: openrouter/deepseek/deepseek-v4.1-flash. No provider fallback occurred
during this round.
