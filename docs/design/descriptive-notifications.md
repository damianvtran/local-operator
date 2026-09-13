# Design: descriptive desktop notifications (`notification` frame + shared composer)

Status: proposal for implementation. Author: architect (anthropic/claude-opus-5).
Base: `origin/main` @ `1c2933509` (backend), `2217ea59a` (local-operator-ui).
All `file:line` references are against those trees.

Scope: two repositories, two coders. The backend coder owns §3, §4, §5, §6, §7,
§10.1 and §11.1. The UI coder owns §8, §10.2 and §11.2. Nothing in here requires
either of them to choose a semantic.

---

## 0. The problem, as found

Two defects, one of them a correctness bug and one of them a content gap.

**Defect A — the false "turn ended".** `desktop-notifier.ts:101-106` raises a
toast on `frame.type === "event"` when the event type is `agent_end` **or**
`turn_end`. Those are not the same fact and neither is "the turn is over":

- `turn_end` is **one model call finishing** — `TurnBoundaryEnd` in
  `local_operator/tui/events.py:191-192` says so, and `TurnEndEvent`
  (`local_operator/harness/types.py:1275-1278`) carries `message` +
  `tool_results` for a single loop step. An agentic turn is many of these. Every
  one of them currently produces "Turn complete / The agent finished its turn."
  This is the user's report, exactly.
- `agent_end` is the parent's logical turn end, but it routinely arrives while
  `task` children are still running. The `task` tool returns at registration, so
  the model stops talking long before the delegated work lands
  (`local_operator/tui/notify.py:65-77`, and `Notifier.notify_turn_complete` at
  `notify.py:1047-1068` refuses while `running_children > 0`).

So the desktop app notifies on a per-step event the TUI has never treated as
notifiable, and on a parent end the TUI deliberately suppresses. Both produce a
toast asserting something false.

**Defect B — the content is canned.** `desktop-notifier.ts:134` hardcodes
`"Turn complete"` / `"The agent finished its turn."`. The session name is on the
wire already (`conversation_title`, `frontend_state.py:1548`) and unused; the
status is not expressed at all; the last assistant line is reachable
(`session_preview`, `resume.py:1926`) and not sent.

**What is already right and must not be broken.** The gate path
(`desktop-notifier.ts:116-127`) is correct: it toasts title/detail from
`pending_gate`, keys dedupe on `session:epoch:request_id`, and deliberately
ignores focus because the user may be reading another conversation in the same
window. The watch lease (`desktop_sessions.py:357-400`) and the
claim-then-deliver arbiter (`attention.py:585-655`) are both sound. This design
adds one frame type and one composer; it changes no existing frame.

### 0.1 Corrections to the framing

The brief asked to be corrected rather than designed on top of. Five items.

1. **"notifications generally originate in the backend" is only half true
   today, and the half that is true is the wrong half for this feature.** The
   backend emits OS toasts from two places — the TUI's own `Notifier`
   (`notify.py:1084`) and the *detached runtime's* gate fallback
   (`serving.py:2260-2352`) — but both are **delivery**, and both are explicitly
   suppressed when a desktop lease claims `can_notify`
   (`serving.py:2293-2298`). There is no backend path that composes notification
   *content* for the desktop. That is the gap this design fills, and it fills it
   with composition, not with a second delivery transport. `local-operator serve`
   must stay silent as a *deliverer* (`notify.py:53-59`); composing a payload the
   UI delivers does not violate that rule, and §3 keeps them separate on purpose.

2. **`_outstanding_delegated_jobs` counts `running` + `task` only, not "queued
   and backgrounded work too".** The code is `tui/app.py:23405-23422`: `status ==
   "running" and type == "task"`. Queued children are included **because a queued
   child's `JobState.status` is still `"running"`** and the queued-ness lives in a
   separate `queued` flag (`harness/jobs.py:345`, and `frontend_state.py:2653`
   reconstructs the split). Backgrounded `bash` jobs are deliberately **excluded**
   (`app.py:23377-23391`) — the comment at `app.py:34464-34466` claiming
   otherwise is stale. This matters for §5: the bridge must reproduce
   `type == "task" and status == "running"`, and must **not** subtract queued
   children.

3. **The brief asks "what about a turn that ended while a child was killed?"
   The bridge never has to answer it.** See §5: the authoritative signal is not a
   job count read at `agent_end` time but the durable attention publication,
   whose own delegated check (`session.py:5962`) runs inside the session process
   at the same instant, and whose re-entry path (`_on_job_completed`,
   `session.py:7604`) opens a *fresh turn* for each settled child. A cancelled
   child never settles into a turn, so the eligibility decision is simply
   deferred to whatever turn does complete. There is no killed-child edge case to
   encode because the bridge does not make that decision at all.

4. **`session_preview` returns at most `PREVIEW_MAX_CHARS = 200`
   (`resume.py:378`), not the notification budget.** The banner budget is
   `BACKGROUND_SNIPPET_MAX_CHARS = 120` (`notify.py:335`) and is passed as
   `max_chars` so the word-boundary ellipsis is computed against the real budget
   (`app.py` background body, which passes `max_chars=`). Do not trim after the
   call.

5. **`desktop_sessions.py:614` calls `session_preview` in the session *catalog*
   (`DesktopSessions.list`), not in the bridge.** The bridge itself never reads a
   preview today. So §4's claim "a preview is reachable from the bridge" is true
   only in the sense that the function is importable; the plumbing is new work.

---

## 1. Decision summary

| # | Question | Decision |
|---|---|---|
| 1 | Where content is composed | **Backend, shared composer** in `local_operator/notifications/compose.py`, consumed by the desktop bridge and (§9) the TUI foreground path. |
| 2 | Wire contract | **New frame type `notification`**, additive, replay-exempt, versioned by `notification_contract` in `/v1/capabilities`. |
| 3 | What may notify | Exactly four kinds: `complete`, `ask`, `approval`, `error`, plus `interrupted` **suppressed by default** (§6.5). Emission is driven by the **attention publication**, not by `agent_end`. |
| 4 | Content | `title` = session name (privacy-gated), `status` = `CONTEXTS[kind]`, `body` = snippet for `complete` only, else `BODIES[kind]`. One privacy flag gates **both** name and snippet. |
| 5 | Arbitration | **Both**, at different layers: `can_notify` decides *whether the backend suppresses its own fallback* (unchanged); `claim_delivery` decides *which surface owns this completion*. The desktop **claims in main, and releases on suppression** (§7.3). |
| 6 | TUI parity | Yes — `Notifier.send` gains an optional `body` via the same composer. |
| 7 | Rollout | **Backend first.** The UI change is safe to ship after, never before. |

---

## 2. Option 1 — where content is composed

### The alternatives

**(A) Backend shared composer.** The backend emits a fully-composed
`{title, status, body}` and the UI renders it verbatim.

**(B) UI-side composition from richer facts.** The backend sends structured
facts (`kind`, `conversation_title`, `last_assistant_text`) and
`desktop-notifier.ts` assembles the strings.

### Trade-offs

(B) is cheaper on the wire and lets the UI localise later. It is also how the
gate path works today — `pending_gate` carries `title`/`detail` and the UI picks
the fallback `"Approval needed"` (`desktop-notifier.ts:126`).

But (B) puts four decisions in a signed Electron binary the user may be running
against any backend version:

1. **The privacy gate.** `session_names_in_notifications()`
   (`notify.py:398-416`) reads `display.notification_session_name` from the
   backend's config. Its docstring is explicit that it governs *both*
   notification legs so the settings copy ("a session's name appears on
   banners, including the lock screen") is true of every banner. A UI that
   composes locally either re-reads that config over HTTP on every toast, or
   ships a banner the user opted out of. An old UI against a new backend would
   leak the name *forever*, because the opt-out is a backend setting the old
   binary has never heard of. This alone decides it.

2. **The snippet-iff-complete rule.** `app.py:18830-18880` documents at length
   why an `error`/`interrupted` session must not carry its last assistant line:
   `session_preview` filters to `role == "assistant"`, so on a failed turn it
   returns the last thing the model said *before* the failure — typically a
   success sentence under a `Needs attention` subtitle, "the two content lines
   of the frame asserting opposite things" (review round 1 M1, design round 1
   D1). The attention anchor already refuses this (`session.py:5985`). A rule
   this subtle, re-implemented in TypeScript, will drift.

3. **Sanitisation.** `sanitize_text` (`notify.py:419-431`) strips
   `[\x00-\x1f\x7f-\x9f]` because the text is model-generated and reaches argv
   and an AppleScript literal. The Electron `Notification` constructor is a
   softer target than `osascript`, but the composer output is *also* consumed by
   the TUI (§9) and by the cmux/`notify-send` legs, so the scrub has to happen
   once, at the source, or two consumers disagree about whether the text is
   shape-safe.

4. **Wording parity.** The whole point of the operator's report is that "every
   surface agrees". `CONTEXTS`/`BODIES` (`notify.py:140-206`) are the house
   vocabulary, carrying design-round decisions (D3: `interrupted` is its own
   category, not a synonym for `error`). Parity is free with (A) and is an
   ongoing manual sync with (B).

### Recommendation: (A), backend composer

New module `local_operator/notifications/compose.py`. **Not** inside `tui/`:
`notify.py`'s module docstring makes "this module is imported only from `tui/`"
a load-bearing rule about *delivery*, and the bridge importing `tui.notify`
directly would erode it. (Verified: `notify.py` pulls in no Textual —
`import local_operator.tui.notify` loads no `textual` module — and
`serving.py:2300` already imports it from outside `tui/`. So the import is
*safe*; it is the **rule** that is worth preserving, which is why the new module
re-exports rather than relocating.)

`compose.py` imports the vocabulary from `tui/notify.py` and adds no delivery.
The existing constants stay where they are; moving them would churn ten import
sites for no gain.

**Cost of choosing wrong.** If (A) turns out wrong, we have one Python module
with a pure function and one frame field set — the UI would keep the structured
fields (§4 carries `kind` and `session_name` *beside* the composed strings for
exactly this reason) and start composing from them. That is a one-file UI change.
If (B) turned out wrong, the fix requires shipping a new signed Electron build to
every user, because the wrong strings live in the binary. The asymmetry favours
(A) independently of the arguments above.

---

## 3. Module and function contract (backend)

### 3.1 New module: `local_operator/notifications/__init__.py`, `compose.py`

```python
# local_operator/notifications/compose.py

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

#: The kinds a composed notification may carry. Deliberately the CONTEXTS key
#: set (tui/notify.py:140) plus the two gate kinds, so the vocabulary cannot
#: drift from the TUI's.
NotificationKind = Literal["complete", "error", "interrupted", "ask", "approval"]

#: Wire contract version for the `notification` frame's payload shape.
#: Bumped only on a BREAKING change to field names/types; additive fields do
#: not bump it (see docs/design/descriptive-notifications.md §4.3).
NOTIFICATION_CONTRACT_VERSION = 1


@dataclass(frozen=True)
class ComposedNotification:
    """One notification's rendered text plus the facts it was rendered from.

    Both halves travel. The strings are what every surface shows, so wording
    parity is free; the facts are what lets a future consumer re-render
    (localisation, a different budget) without the backend having to guess
    which surface is asking.
    """

    kind: NotificationKind
    #: Banner title: the session's name, or APP_NAME when the privacy flag is
    #: off, or BACKGROUND_FALLBACK_TITLE when the session has no stored title.
    title: str
    #: Short state category from CONTEXTS: "Complete", "Input required",
    #: "Needs attention", "Interrupted". Rendered as the subtitle where a
    #: surface has one and folded into the body where it does not.
    status: str
    #: The content line. A last-assistant-line snippet for `complete` when the
    #: privacy flag allows and one exists; the BODIES sentence otherwise.
    body: str
    #: True when `body` is model-written text rather than a house constant.
    #: The UI uses it for nothing today; it exists so a surface that must not
    #: show conversation content (a shared screen mode) can degrade without
    #: re-deriving the privacy rule.
    body_is_snippet: bool
    #: Whether `title` is the conversation's real name. False when the privacy
    #: flag is off or the session is untitled.
    title_is_session_name: bool


def compose(
    kind: NotificationKind,
    *,
    session_dir: Path | None,
    session_name: str = "",
    gate_title: str = "",
    gate_detail: str = "",
) -> ComposedNotification:
    """Render one notification. Pure apart from two best-effort disk reads.

    `session_dir` is read for the stored title and, for `complete`, the last
    assistant line. `session_name` is the caller's already-resolved name (the
    TUI has it in hand; the bridge reads `conversation_title` off the frontend
    snapshot) and WINS over the stored title when non-empty, because a rename
    reaches the live state before it reaches the sidecar.

    `gate_title`/`gate_detail` are used only for `ask`/`approval`; they carry
    the tool name and the action being authorised, in the shape
    `serving.py::_announce_pending` already builds. Ignored for other kinds.

    Never raises: a notification is chrome. Every read is guarded and degrades
    to the house vocabulary.
    """
```

### 3.2 Exact composition rules (this is the whole of §4 of the brief)

Let `names_ok = session_names_in_notifications()`, read **per call** (the
`/settings` page writes it live; `notify.py:1096-1100` documents why caching it
at construction leaks the name for the rest of the session).

**`title`**

| Condition | Value | `title_is_session_name` |
|---|---|---|
| `not names_ok` | `APP_NAME` (`"Local Operator"`) | `False` |
| `names_ok` and a name resolves | `sanitize_text(name, limit=MAX_TITLE_CHARS)` where `name = session_name or stored_session_title(session_dir)` | `True` |
| `names_ok`, no name | `BACKGROUND_FALLBACK_TITLE` (`"A session finished"`) for `complete`/`error`/`interrupted`; `APP_NAME` for `ask`/`approval` | `False` |

The split on the nameless case is deliberate: "A session finished" is a true
sentence for a completion and a false one for a parked question. For the gate
kinds the brand is the honest fallback, which is what `serving.py:2318-2324`
already does (`name = APP_NAME` then conditionally the conversation name).

`MAX_TITLE_CHARS` is 80 (`notify.py:377`). macOS clips a title at ~43 characters
(`notify.py:230-240`), which is why the status never rides in the title.

**`status`** — `CONTEXTS[kind]` verbatim (`notify.py:140-146`):

| kind | status |
|---|---|
| `complete` | `Complete` |
| `ask` | `Input required` |
| `approval` | `Input required` |
| `error` | `Needs attention` |
| `interrupted` | `Interrupted` |

**`body`**

| kind | Condition | Value | `body_is_snippet` |
|---|---|---|---|
| `complete` | `names_ok` and `session_preview(session_dir, max_chars=BACKGROUND_SNIPPET_MAX_CHARS)` is non-empty | `sanitize_text(preview, limit=BACKGROUND_SNIPPET_MAX_CHARS)` | `True` |
| `complete` | otherwise | `BODIES["complete"]` = `"Task complete"` | `False` |
| `error` | always | `BODIES["error"]` = `"Stopped with an error"` | `False` |
| `interrupted` | always | `BODIES["interrupted"]` = `"Stopped before finishing"` | `False` |
| `ask` / `approval` | `gate_detail` non-empty | `gate_detail`, prefixed with `gate_title` **only when it does not already start with it** (see below) | `False` |
| `ask` / `approval` | otherwise | `BODIES[kind]` | `False` |

The gate body reuses `serving.py:2337-2347` exactly, including its bug fix: a
tool's `describe_approval` already leads with its own action word
(`_describe_path_approval` emits `"write: /path"`) and the title **is** the tool
name, so a naive `f"{title}: {detail}"` renders `"write: write: /path"` — round
4, Q3. The rule is:

```python
subject = (gate_detail or "").strip()
if subject and gate_title and not subject.lower().startswith(gate_title.lower()):
    subject = f"{gate_title}: {subject}".strip().rstrip(":").strip()
body = subject or BODIES[kind]
```

Extract this into a `compose.py` helper and have `serving.py::_announce_pending`
call it, so the two cannot drift. That is the only change to `serving.py`.

**Budgets, restated as constants the coder must not re-pick**

- title: `MAX_TITLE_CHARS = 80` (`notify.py:377`)
- snippet: `BACKGROUND_SNIPPET_MAX_CHARS = 120` (`notify.py:335`), passed as
  `max_chars` to `session_preview`, **not** applied afterwards — the
  word-boundary ellipsis must be computed against the real budget.
- The UI's `body.slice(0, 240)` (`desktop-notifier.ts:153`) stays as a
  belt-and-braces cap and will never bind, since 120 < 240.

**Why one flag gates both the name and the snippet.** `app.py:18886-18894` is
explicit: the flag exists to keep model-written session text off a screen other
people can see, and "a snippet is strictly more session-derived than the name —
a name is a topic, a snippet is content". A user who opted out of the name has
necessarily opted out of the snippet. No new setting; `display.notification_
session_name` governs both. The settings copy stays true.

---

## 4. Option 2 — the wire contract

### 4.1 The frame

A **new frame type**, not an extension of `event` or `frontend.update`.

```
{
  "session_id": "<12 lowercase hex>",
  "epoch": "<bridge epoch hex>",
  "seq": <int>,
  "type": "notification",
  "payload": {
    "contract": 1,
    "kind": "complete" | "error" | "interrupted" | "ask" | "approval",
    "title": "<string, <=80 chars, sanitised>",
    "status": "<string, one of CONTEXTS values>",
    "body": "<string, <=120 chars for snippets, sanitised>",
    "body_is_snippet": <bool>,
    "title_is_session_name": <bool>,
    "dedupe_key": "<string, see 4.2>",
    "completion_token": "<uuid string>" | null,
    "session_name": "<string>" | null,
    "focus_policy": "when_unfocused" | "always"
  }
}
```

**Why a new type and not an extension of `event`.** Three reasons, in order of
weight.

1. **`event` is a typed canonical `AgentEvent`** — `_event()` at
   `desktop_sessions.py:198-199` is `publish("event", event.model_dump())` and
   `DESKTOP_API.md:224` pins it as "a typed canonical AgentEvent". A notification
   is a bridge-composed decision, not an engine event. Putting it there would
   mean inventing a fake `AgentEvent` subtype, which every existing consumer
   (`use-canonical-session.ts:323`, `applyEvent`) would then try to paint into
   the transcript.
2. **An older UI must ignore it silently**, and the renderer's frame loop
   already does exactly that for an unknown type: `use-canonical-session.ts:183-
   334` is a chain of `if (frame.type === ...)` with **no `else` and no throw**,
   and `frame.type === "open" || "seq" in frame` (line 201) advances the receipt
   cursor for any frame carrying a `seq` — so an old renderer advances its cursor
   correctly and paints nothing. `desktop-notifier.ts:87-107` is the same shape.
   Verified by reading both, not assumed.
3. **`frontend.update` is a field delta of canonical state.** A notification is
   an *edge*, not a state — `notify.py:12-16` draws precisely this distinction
   (the title is a persistent state, a notification is a one-shot edge). A
   notification in `changes` would be re-delivered by every snapshot.

### 4.2 `dedupe_key`

The backend mints it; the UI treats it as an opaque string and keys its existing
TTL map on it. Shape:

- completions: `{kind}:{session_id}:{completion_token}` — the frame's own
  `kind` (`complete` or `error`), so a store or dedupe-map dump reads
  consistently beside the frame it keys. A token has exactly one kind, so the
  prefix can never widen or narrow a collision; it is a label, not a
  discriminator.
- gates: `gate:{session_id}:{bridge_epoch}:{request_id}`

**Why the backend mints it.** The UI's current completion key is
`turn:{session}:{frontendEpoch}:{seq}` (`desktop-notifier.ts:130`), keyed on the
*bridge's* sequence. That is wrong across a reconnect: `acquire()` mints a new
epoch and resets `sequence` to 0 (`desktop_sessions.py:108-109`), so the same
completion re-delivered after a detached interval gets a different key and
toasts twice. The `completion_token` is the durable, conversation-bound identity
(`attention.py:487`, a validated UUID) and is stable across every epoch.

The gate key keeps the bridge epoch because a `request_id` is only unique within
a runtime generation, which is exactly what the current UI code already assumes.

### 4.3 Versioning and both directions of skew

`payload.contract` is an integer, `1` today. **Additive fields do not bump it.**
It bumps only if a field is removed or changes type — and if that ever happens,
the *old* field is kept beside the new one for one release.

`/v1/capabilities` (`local_operator/server/routes/capabilities.py`) gains one
entry under `features`:

```python
"features": {
    ...
    "notification_contract": 1,
}
```

This follows the existing `<subsystem>: <version>` convention the file's own
comment establishes (`capabilities.py:31-34`).

**New backend, old UI.** The old UI does not know `"notification"`. Its
`observe()` falls through every branch and returns; its renderer advances the
receipt cursor and paints nothing. **But it still toasts on `agent_end`/
`turn_end`**, because that code is in the shipped binary. So the user keeps the
buggy behaviour they have today and gains nothing — no double toast, because the
new frame is ignored. This is the "degrade to something sane" requirement: the
old UI is no worse than before. It is not *fixed*, and it cannot be, because the
bug lives in a signed binary.

> **This is the reason the backend must NOT stop emitting `agent_end`.** A
> tempting "fix" is to suppress `agent_end` on the desktop stream so old UIs
> stop toasting. Do not: `agent_end` is a canonical event several renderer paths
> depend on (`TERMINAL_EVENTS` at `use-canonical-session.ts:325`, and the
> transcript reducer), and removing it to fix a notification would break
> transcript settling on **every** UI version.

**Old backend, new UI.** The new UI reads `features.notification_contract` from
`/v1/capabilities` once at startup (the `useDesktopCapabilities` hook already
fetches this — `backend-compatibility-banner.tsx:27-29`). Two branches, and the
branch is a *hard* switch, not a heuristic:

- **`notification_contract >= 1` present** → the UI **stops** toasting on
  `agent_end`/`turn_end` entirely and toasts only on `notification` frames.
- **absent** → the UI keeps its legacy path, but **narrowed to `agent_end`
  only** with the canned copy. Dropping `turn_end` from the legacy path is safe
  against any backend (it is a per-step event on every version) and fixes the
  loudest half of the bug even on an old backend. Keeping `agent_end` there
  preserves *some* completion signal against an old backend rather than going
  silent.

`notification_contract` unknown/unfetchable is treated as **absent** (legacy
path). Failing toward the legacy path rather than toward silence is deliberate:
a missing capability response is a transient HTTP failure far more often than it
is an old backend, and going silent on a transient failure loses completions.

### 4.4 Replay and the frame budget

The bridge retains `REPLAY_COUNT = 256` frames / `REPLAY_BYTES = 8 MiB`
(`desktop_sessions.py:49-50`), and `publish()` pushes every frame into
`self.replay`. A notification frame is ~400 bytes, so the byte budget is not the
concern — **the count is**. A notification frame occupying a replay slot pushes
out a real transcript event.

**Decision: `notification` frames are NOT retained for replay.**

```python
def publish(self, kind: str, payload: dict[str, Any], *, replay: bool = True) -> None:
    ...
    if replay:
        self.replay.append((frame, size))
        self.replay_bytes += size
        while self.replay and (...):
            ...
    for sub in self.subscribers.values():
        ...
```

and the notification path calls `self.publish("notification", payload, replay=False)`.

**Why.** A notification is an edge whose whole value is timeliness. Replaying it
on reconnect means toasting the user about a turn that finished while their
laptop lid was shut, possibly hours later — which is precisely the "one-shot
edge" contract `notify.py:12-16` describes. The durable signal is not lost: the
`attention` frame and the sidebar's unseen mark both survive a reconnect and are
the correct surface for "you missed something".

**The `seq` numbering stays global**, i.e. a non-replayed frame still increments
`self.sequence`. This is load-bearing: `events()` computes `gap` from
`after_seq < first - 1` (`desktop_sessions.py:415`) where `first` is the oldest
*retained* frame's seq. If a notification consumed a seq and was not retained,
`first` naturally skips it and a client reconnecting at that cursor is **not**
flagged as gapped — because `after_seq` equals the notification's seq, which is
`>= first - 1` for the next retained frame. Verified against the arithmetic. The
alternative (not incrementing) would make `seq` non-monotonic across the two
paths and break the receipt cursor.

**One consequence the UI coder must handle:** a notification frame carries a
`seq` the renderer will record as its receipt cursor
(`use-canonical-session.ts:201-203`, `"seq" in frame`). That is correct and
desirable — it means a reconnect after a notification does not re-request frames
it already saw. No change needed; stated so it is not "fixed".

---

## 5. Option 3 — the gating rules

### 5.1 The source of truth, named

**For "the turn is genuinely over": the attention publication, i.e. a new
`completions` row in `attention.db`.**

Not `agent_end`. The reasoning:

- `Session._publish_attention_outcome` (`session.py:5949-5999`) runs in the
  `finally` of `_run_turn_pipeline` (`session.py:6534`), after
  `_flush_held_end()`. It is the one place in the codebase that already answers
  "did this turn produce a notifiable outcome" — and it answers it with the
  *same* delegated-children check the TUI uses: `delegated = any(job.type ==
  "task" and job.status == "running" for job in self.jobs.list())`
  (`session.py:5962`).
- When `kind == "complete"` and (`not messages` or `delegated`), it writes an
  `eligible: False` marker and **publishes nothing** (`session.py:5970-5980`).
  So a delegating parent's premature `agent_end` produces no completion row at
  all. The false-finish problem is already solved, one layer below every
  frontend.
- It runs **inside the session process**, which owns the job manager. The bridge
  does not have to reconstruct the job count, cannot disagree with the TUI about
  it, and does not need to answer the killed-child question (§0.1 item 3).
- It is durable and cross-process, which is what makes §7's arbitration possible
  at all.

**This is the single most important decision in the design.** Every alternative
("count jobs in the bridge from `frontend_state.jobs`", "read `last_turn_outcome`
from the snapshot") reconstructs a decision that is already made correctly, in a
process with less information, and can therefore disagree with the TUI. The
bridge's job is to *observe* the publication, not to re-derive it.

**How the bridge observes it.** `_poll_attention` (`desktop_sessions.py:237-293`)
already runs a 1 s loop per bridge, gated on the cheap `store.revision()` change
detector, and `refresh_attention` already computes the new state and publishes an
`attention` frame when it differs (`desktop_sessions.py:228-234`). The
notification hook goes **inside `refresh_attention`**, immediately after the
`attention` publish, and fires when:

```
previous is non-empty                      # not the bridge's first read
and state["completion_token"] is not None
and state["completion_token"] != previous.get("completion_token")
and state["unseen"] is True
and state["kind"] in NOTIFIABLE_COMPLETION_KINDS
```

The `previous` guard is the existing baseline rule (`desktop_sessions.py:232-234`:
"the initial snapshot owns the baseline") and is what keeps opening a session
from toasting its last completion.

**For "children still running": nothing in the bridge.** Answered upstream by
`session.py:5962`, in the process that owns `self.jobs`. Stated explicitly
because the brief asked which of `frontend_state.jobs` or the attention store is
authoritative: **neither is consulted in the bridge**, because the question is
already answered before the fact the bridge observes even exists. This is why
the TUI and the bridge cannot disagree — they are reading two projections of one
decision, not making the same decision twice.

### 5.2 For the gate kinds: the existing `pending_gate`, unchanged

`ask` and `approval` are **not** routed through the new frame. The existing path
(`desktop-notifier.ts:89-99` reading `pending_gate` out of `snapshot` and
`frontend.update`) already works, already ignores focus for the right reason,
and already dedupes on `request_id`.

**The new path must not duplicate it.** The rule, stated so a coder cannot get
it wrong:

> **The bridge emits `notification` frames with `kind` in `{complete, error}`
> ONLY. It never emits `kind` `ask` or `approval`.**

The `ask`/`approval` kinds exist in `compose()` because the *TUI* path (§9) and
`serving.py::_announce_pending` use them. They are unreachable on the desktop
wire. The `NotificationKind` literal keeps them so one vocabulary serves all
surfaces; the bridge has its own narrower allowlist:

```python
#: The kinds the DESKTOP BRIDGE may put on the wire. Narrower than
#: NotificationKind on purpose: `ask` and `approval` already reach the desktop
#: as `pending_gate` in the snapshot/update frames, and a second channel for
#: the same card is the duplicate this whole design exists to prevent.
BRIDGE_NOTIFIABLE_KINDS = frozenset({"complete", "error"})
```

What this design *does* improve for gates is the **fallback title**: the UI
currently renders `title || "Approval needed"` (`desktop-notifier.ts:126`),
which is wrong for an `ask`. §8.3 fixes that UI-side, using
`gate.kind` which is already on the wire (`PendingDesktopGate.kind`,
`desktop-session-contract.ts:99`).

### 5.3 The permitted transitions, exhaustively

| Transition | Emits? | Exact condition | How it is proven |
|---|---|---|---|
| Real turn end | **YES**, `kind="complete"` | A new `completions` row with `kind="complete"` and `unseen=true` appears for this conversation | `session.py:5970-5999` refuses to publish when `delegated` or when the turn produced no durable assistant message. The bridge observes the row; it does not judge it. |
| Turn ended with children still running | **NO** | — | No row is ever written (`session.py:5970-5976` writes `eligible: False` and returns). Nothing for the bridge to see. |
| Turn ended, a child was killed | **NO** for that turn | — | Same: no row. Each settled child re-enters as a fresh turn (`session.py:7604`) whose own completion publishes normally. A cancelled child never re-enters, so the next real turn carries the notification. §0.1 item 3. |
| Error | **YES**, `kind="error"` | A new `completions` row with `kind="error"` | `session.py:5971`: `kind = "error" if outcome.error else ...`. Published even with no assistant message, anchored to `provisional_anchor(token)`. |
| Interrupted | **NO** by default | See §6.5 | Suppressed by `BRIDGE_NOTIFIABLE_KINDS`. |
| `ask` awaiting input | **YES, via `pending_gate`** — no new frame | `frontend.snapshot.pending_gate` / `changes.pending_gate` non-null with `kind == "ask"` | Existing path, `desktop-notifier.ts:89-99`. |
| Approval gate | **YES, via `pending_gate`** — no new frame | same, `kind == "approval"` | Existing path. |
| Per-step `turn_end` | **NEVER** | — | It is one model call (`events.py:191`, `types.py:1275`). The defect. |
| `agent_end` | **NEVER** as a notification trigger | — | Still forwarded as an `event` frame for the transcript (§4.3 warning). It is not the turn-over signal. |
| `subagent_end` / child completion | **NEVER** | — | `notify.py:65-77`: a child finishing is an implementation detail of the parent's task; five children would fire five toasts for one task. |
| `subagent_start` / `subagent_progress` | **NEVER** | — | Progress, not an edge the user owes anything for. |
| `tool_execution_start/update/end`, `tool_call_compose` | **NEVER** | — | Steps. |
| `message_start` / `message_update` / `message_end` / streaming deltas | **NEVER** | — | Fragments of one message. |
| `history_delta` | **NEVER** | — | Durable rows painted on reconnect; every one is already settled. |
| Session rename | **NEVER** | — | A `conversation_title` change in `frontend.update`. Not an edge; it changes the *title* of a future toast. |
| `compaction_start` / `compaction_end` | **NEVER** | — | Internal context management. `session.py:6457-6468`: the pipeline deliberately holds the boundary events so compaction does not look like a turn ending and restarting. |
| Todo updates | **NEVER** | — | `frontend.update` field delta. |
| `retry_start` / `retry_end`, `model_change` | **NEVER** | — | Provider mechanics. |
| `wake_delivered` / `peer_message_delivered` / `steering_delivered` | **NEVER** | — | Inbound, not an outcome. A wake that drives a turn produces a completion, which notifies. |
| `notice` | **NEVER** | — | Infrastructure chatter (an MCP server that failed to connect). |
| Bridge reconnect / gap / new epoch | **NEVER** | — | `previous is non-empty` guard, plus §4.4 replay exemption. |
| Opening a session with an old unseen completion | **NEVER** | — | Same `previous` guard: the first `refresh_attention` of a bridge's life sets the baseline and publishes nothing (`desktop_sessions.py:232-234`). |

### 5.4 The NEVER list, as one sentence for the coder

> The bridge emits a notification frame **if and only if** `refresh_attention`
> observes a *newly published, unseen* completion row whose kind is `complete`
> or `error`, on a bridge that has already taken a baseline reading. No engine
> event triggers a notification. Ever.

---

## 6. Backend implementation sketch

### 6.1 `refresh_attention` gains the hook

```python
async def refresh_attention(self) -> dict[str, Any]:
    state = await asyncio.to_thread(
        AttentionStore(self.root / "attention.db").state, f"session/{self.session_id}"
    )
    remote = self.remote
    state["supported"] = bool(...)
    if state != self.attention:
        previous = self.attention
        self.attention = state
        if previous:
            self.publish("attention", state)
            # THE NOTIFICATION EDGE. After the attention frame, because a
            # reader that toasts should already have the receipt state that
            # explains the toast. `previous` being non-empty is the same
            # baseline rule the attention publish uses: a bridge's FIRST read
            # is the session's history, not news.
            self._maybe_publish_notification(previous, state)
    return state
```

### 6.2 The emitter

```python
def _maybe_publish_notification(
    self, previous: dict[str, Any], state: dict[str, Any]
) -> None:
    """Turn a newly published completion into one notification frame.

    Guarded end to end: a notification is chrome and this runs inside the 1 s
    attention poll, whose loop already treats a store error as costing one
    tick rather than the feature (see `_poll_attention`).
    """
    token = state.get("completion_token")
    if (
        not token
        or token == previous.get("completion_token")
        or not state.get("unseen")
        or state.get("kind") not in BRIDGE_NOTIFIABLE_KINDS
    ):
        return
    try:
        composed = await_in_thread_or_inline(...)  # see note
        payload = {
            "contract": NOTIFICATION_CONTRACT_VERSION,
            "kind": composed.kind,
            "title": composed.title,
            "status": composed.status,
            "body": composed.body,
            "body_is_snippet": composed.body_is_snippet,
            "title_is_session_name": composed.title_is_session_name,
            "dedupe_key": f"{composed.kind}:{self.session_id}:{token}",
            "completion_token": token,
            "session_name": composed.title if composed.title_is_session_name else None,
            "focus_policy": "when_unfocused",
        }
        self.publish("notification", payload, replay=False)
    except Exception:
        logger.debug("notification compose failed for %s", self.session_id, exc_info=True)
```

**Two implementation notes the coder must not improvise.**

1. **`compose()` does disk I/O** (`stored_session_title` reads up to 128 KiB
   from both ends of the transcript, `resume.py:336`; `session_preview` reads
   64 KB from the tail, `resume.py:373`). `refresh_attention` is on the event
   loop. Wrap it: `composed = await asyncio.to_thread(compose, kind, session_dir=..., session_name=...)`.
   Make `_maybe_publish_notification` `async` and `await` it from
   `refresh_attention`. This is why the sketch above is deliberately
   incomplete — write it as an `async def` that `await`s `asyncio.to_thread`.

2. **`session_name` comes from the live snapshot, not the sidecar**, when a
   runtime is attached: `self.remote.frontend_state.conversation_title`
   (`frontend_state.py:1548`, populated from `session.conversation_name` at
   `frontend_state.py:3195`). A rename reaches the live state before the
   sidecar. Guard it — `self.remote` may be `None` for a cold bridge, in which
   case `compose()` falls back to `stored_session_title`.

### 6.3 `publish()` gains `replay`

Signature change per §4.4. Default `True`, so no existing call site changes.

### 6.4 `serving.py::_announce_pending` uses the composer

Replace the inline body-building at `serving.py:2337-2347` with a call to the
extracted helper. Behaviour-identical; this is what keeps the gate wording
identical across the detached-runtime OS fallback and the TUI.

### 6.5 `interrupted`: suppressed, and why

`interrupted` composes correctly and is deliberately **not** in
`BRIDGE_NOTIFIABLE_KINDS`.

The TUI already refuses it: `app.py:34457-34459` — "aborted → none. The user
pressed Ctrl+C or Esc, so they were at the keyboard a moment ago and already
know; telling them their own stop worked is the definition of a notification
nobody wants."

The counter-argument is real: on the *desktop*, an interruption can come from
another surface (a phone, a second TUI), so the desktop user may not have been
the one who pressed the key. But we cannot tell those apart — `AgentEndEvent`
carries `aborted: bool` with no actor — and the maintainer's store has 318
complete / 0 error / 14 interrupted (`notify.py:133-136`), so the population is
small and dominated by the user's own Ctrl+C.

**Recommendation: ship suppressed.** It is one frozenset entry to change if the
user asks for it, and shipping it on would add a toast for an action the user
almost always just took themselves. The evidence that would settle it: an actor
field on the abort path (who requested the interrupt), which does not exist
today and is not worth building for this.

---

## 7. Option 5 — cross-surface arbitration

Three surfaces can see one completion: a TUI in a terminal, the desktop app, and
the mobile relay.

### 7.1 What each mechanism actually does

These are **two different questions** and conflating them is the trap.

**`can_notify` (the watch lease) answers: may the backend's own OS-toast
fallback stay quiet?** It is consumed at `serving.py:2293-2298` (gate routing)
and `server.py:1829-1837` (`notification_surfaces`). A live desktop lease with
`can_notify=true` suppresses the detached runtime's `detached_notify` for gates.
It says nothing about which *frontend* owns a completion.

**`claim_delivery` (the deliveries watermark) answers: which observer process
owns this completion's toast?** `attention.py:585-655`. `BEGIN IMMEDIATE`
serialises N observers; exactly one wins per `completion_token`. It is the
arbiter, and it is clock-free by construction (`attention.py:43-49`: the
`delivered_at` column is diagnostics only, because "a wall clock must never
enter the claim, or two observers whose clocks disagree would both deliver").

### 7.2 Who claims what

**Decision: the desktop claims in the Electron main process, via a new
`sessions.notified` op — not in the bridge.**

This is the subtle part and the brief is right to flag it.

The naive design claims in `_maybe_publish_notification` before putting the
frame on the wire. That is wrong, for the reason `claim_delivery`'s own
docstring gives (`attention.py:601-607`): **claim-then-deliver means the claimant
must actually be the deliverer.** The bridge is not. Between the bridge's claim
and an OS toast lie: the SSE socket, the Electron main process, the
`Notification.isSupported()` check, and the focus gate. A bridge-side claim that
the UI then suppresses (focused window) would mark the completion delivered while
nobody was told — and no other surface could ever tell them, because
`claim_delivery` returns `False` for good.

So:

1. The bridge publishes the `notification` frame **without claiming**. A frame
   is an offer, not a delivery.
2. `desktop-notifier.ts` receives it, checks `Notification.isSupported()`, its
   local TTL dedupe, and the focus gate.
3. **Only if it is actually going to call `new Notification(...)`** does it
   claim, through a new desktop op, and only shows the toast if the claim wins.

New op in `local_operator/server/routes/desktop_sessions.py`:

```
POST /v1/desktop/sessions/{session_id}/notified
body: { "completion_token": <uuid> }
200 -> { "claimed": true | false }
404 -> unknown session
```

Backed by a `DesktopSessions.claim_notification(session_id, token)` that mirrors
`acknowledge_attention` (`desktop_sessions.py:467-485`) exactly — **cold path,
no bridge acquire, no runtime spawn**, same `SESSION_ID.fullmatch` +
`is_user_session` validation — and calls:

```python
AttentionStore(self.root / "attention.db").claim_delivery(
    f"session/{session_id}", token, "desktop"
)
```

`backend="desktop"` (the `backend` column is diagnostics only —
`attention.py:47-49` — but naming it correctly is what makes a store dump
readable).

**Why not reuse `sessions.seen`.** Because notifying is not reading.
`attention.py:9-15` is unambiguous: two watermarks, deliberately separate;
`claim_delivery` "NEVER ACKNOWLEDGES", and `docs/SESSION_SIDEBAR.md` pins that
routing a notification never marks anything read. Reusing `seen` would clear the
sidebar's unseen mark for a session the user never opened. The new op must
therefore **not** go through `guardForegroundReceipts`
(`desktop-ipc.ts:29-59`) — that guard exists for `sessions.seen` and demands a
focused window, which is the exact opposite of when a notification fires.

### 7.3 The focused-window case (the subtle one the brief names)

**Answer: the claim is never taken in that case, because the focus gate runs
BEFORE the claim.** Order is load-bearing:

```
observe(notification frame)
  → isSupported()?              no  → return, no claim
  → local TTL dedupe fresh?     no  → return, no claim
  → focus_policy === "when_unfocused" && anyFocused()?  yes → return, NO CLAIM
  → POST /notified              claimed=false → return (another surface won)
  → new Notification(...)       claimed=true  → show
```

So there is nothing to release: a suppressed toast never claimed. The completion
stays unclaimed and **another surface can still notify it** — a TUI observer on
the same machine running `_notify_background_completions` will pick it up on its
next 1 s poll and toast it. That is the right outcome: the desktop window being
focused on session A says nothing about whether session B's completion deserves
a banner elsewhere.

`release_delivery` is therefore **not needed on the desktop path**, and must not
be wired in. It exists for the case where a backend claims and then *fails to
deliver* (`attention.py:657-671`), which the ordering above makes unreachable —
the claim is the last thing before `notification.show()`, and
`Notification.show()` on a supported platform does not fail synchronously.

> One residual risk, accepted and named: Electron's `Notification.show()` can be
> silently dropped by the OS (Focus Assist / Do Not Disturb / a denied
> permission that `isSupported()` still reports true for). The claim is then
> taken for a banner nobody saw. This is the identical accepted trade
> `attention.py:609-621` already documents for the TUI path ("bounded in
> CONSEQUENCE, unbounded in TIME"), and the durable signal survives: `unseen`
> stays true and the sidebar keeps its mark. Do not add a lease to fix it; a
> lease needs a clock, and a clock in this predicate is what lets two observers
> both deliver.

### 7.4 The TUI's side is already correct and needs no change

`_notify_background_completions` (`app.py:19130-19140`) skips a row when
`entry.row.live_state not in _BACKGROUND_NOTIFY_ANNOUNCEABLE_STATES`
(`{"", "idle"}`, `app.py:666`) — i.e. it stays quiet when another TUI holds the
session attached, busy, or wedged.

**A desktop bridge does not make a session `attached`.** `live_state` comes from
the record's `detached` flag (`catalog.py:374-386`), which is
`not bool(self._visible_attach_surfaces())` (`server.py:1788`), and a desktop
surface counts as visible there only while its lease says `desktop_visible`
(`server.py:1818-1826`) — which the renderer sets to `visible && focused`
(`desktop-notifier.ts:77`). So:

- Desktop window **focused on this session** → `live_state == "attached"` → the
  TUI observer skips it → only the desktop can notify → and the desktop's focus
  gate suppresses it. Correct: the user is looking at it.
- Desktop window **not focused** → `live_state` is `"idle"` → both the TUI
  observer and the desktop are eligible → `claim_delivery` picks exactly one.
  Correct: one toast.

The mobile relay is not a competitor: it is a `daemon` client
(`server.py:1846-1856`), never counted as watching, and it emits no OS toasts —
`grep` for `detached_notify|claim_delivery` under `local_operator/mobile/`
returns nothing. It participates in the *read* watermark (`POST
/api/sessions/{id}/seen`), not the delivery one.

### 7.5 What about the same session open in two Electron windows?

`desktop-notifier.ts` holds one `delivered` map across every window
(`desktop-notifier.ts:39`) and `anyFocused()` scans all of them
(`desktop-notifier.ts:109-114`). Unchanged. The stream relay is per-window, so
two windows on one session produce two `notification` frames with the *same*
`dedupe_key` — and the shared map collapses them to one before any claim is
attempted. Good: the HTTP claim is not even reached twice.

---

## 8. UI contract

### 8.1 `src/shared/desktop-session-contract.ts`

```ts
/**
 * One composed notification, as the backend rendered it.
 *
 * The strings are authoritative: the backend owns wording parity across the
 * TUI, the detached-runtime fallback and this app, and it is the only place
 * that can read the `display.notification_session_name` privacy flag. The
 * structured fields travel beside them so a future surface can re-render
 * without this app re-deriving a rule it cannot see.
 */
export type DesktopNotification = {
	/** Payload shape version. 1 today; additive fields do not bump it. */
	contract: number;
	kind: "complete" | "error" | "interrupted" | "ask" | "approval";
	title: string;
	/** Short state category ("Complete", "Needs attention"). */
	status: string;
	body: string;
	/** True when `body` is model-written text rather than a house constant. */
	body_is_snippet: boolean;
	/** False when the privacy flag is off or the session has no stored name. */
	title_is_session_name: boolean;
	/** Opaque; key the dedupe map on this and nothing else. */
	dedupe_key: string;
	/** Durable completion identity; the argument to `sessions.notified`. */
	completion_token: string | null;
	session_name: string | null;
	/** `when_unfocused` for completions; `always` for a gate. */
	focus_policy: "when_unfocused" | "always";
};
```

and one arm added to the union:

```ts
	| Receipt<"notification", DesktopNotification>
```

### 8.2 `src/main/desktop-notifier.ts`

```ts
observe(sessionId: string, frame: DesktopSessionFrame): void {
	if (!this.canNotify) return;
	if (frame.type === "snapshot") { /* unchanged */ }
	if (frame.type === "frontend.update") { /* unchanged */ }
	if (frame.type === "notification") {
		void this.composed(sessionId, frame.payload);
		return;
	}
	if (frame.type === "event") {
		// LEGACY PATH ONLY. A backend advertising `notification_contract` owns
		// every completion toast, so this must stay silent against it or the
		// user gets two banners for one turn. `turn_end` is dropped
		// unconditionally: it is ONE MODEL CALL on every backend version
		// (local_operator/tui/events.py TurnBoundaryEnd), never a finished turn.
		if (this.notificationContract > 0) return;
		if (String(frame.payload.type ?? "") === "agent_end") {
			this.turn(sessionId, frame.seq);
		}
	}
}

private async composed(sessionId: string, n: DesktopNotification): Promise<void> {
	if (!this.claim(n.dedupe_key)) return;
	if (n.focus_policy === "when_unfocused" && this.anyFocused()) return;
	if (n.completion_token) {
		// Claim LAST, immediately before delivery: an unclaimed completion
		// stays available to another surface (a TUI on this machine), while a
		// claim taken for a toast we then suppressed would be delivered to
		// nobody, for good. See docs/design/descriptive-notifications.md 7.3.
		const won = await this.claimDelivery(sessionId, n.completion_token);
		if (!won) return;
	}
	this.show(sessionId, n.title, n.status, n.body);
}
```

`claimDelivery` posts `{op: "sessions.notified", sessionId, completionToken}`
through the **ungated** `request` (not the `guardForegroundReceipts` wrapper —
see §7.2). A rejected/failed request returns `false` and the toast is skipped:
failing closed is right, because the failure mode of failing open is a duplicate
banner on every surface.

`notificationContract` is a number on the notifier, default `0`, set from
`/v1/capabilities` — `setNotificationContract(n: number)` called once at
backend-connect. `0` means legacy.

### 8.3 `show()` gains the status line, and the gate fallback is fixed

```ts
private show(sessionId: string, title: string, status: string, body: string): void {
	const notification = new Notification({
		title,
		// macOS has no subtitle on this API, so the state category leads the
		// body. It is the word the user reads in under a second, and it must
		// not be the line the OS clips first.
		body: `${status ? `${status} — ` : ""}${body}`.slice(0, 240),
		silent: false,
	});
	...
}
```

> **The `status — body` concatenation is a guess about a rendered frame and must
> be validated visually before merge (§11.2).** Electron's `NotificationConstructorOptions`
> has `subtitle`, but it is documented macOS-only; if it renders correctly on the
> maintainer's machine, prefer `subtitle: status` with a bare `body`, which is
> what every backend leg already does (`cmux_command` takes title/subtitle/body,
> `notify.py:624-646`). The coder decides from the screenshot, not from the docs.
> Either way the *strings* come from the backend unchanged.

The gate path's fallback title (`desktop-notifier.ts:126`) changes from
`title || "Approval needed"` to a kind-aware one, since `gate.kind` is already on
the wire:

```ts
this.show(sessionId, title || (kind === "ask" ? "Question" : "Approval needed"), "", detail);
```

### 8.4 What the UI must NOT do

- Must not compose or re-word `title`/`status`/`body`.
- Must not toast on `turn_end`, on any backend version.
- Must not toast on `agent_end` when `notification_contract >= 1`.
- Must not call `sessions.seen` from the notification path.
- Must not claim before the focus gate.

---

## 9. Option 6 — TUI parity

**Yes, `Notifier.send` gains the body, via the same composer.**

Today `Notifier.send` (`notify.py:1084-1122`) sends `title = label or APP_NAME`,
`subtitle = CONTEXTS[kind]`, `body = BODIES[kind]` — no snippet. Meanwhile the
*background observer* path in the same app already sends the snippet
(`app.py:18752+`). So a user gets a richer banner for a session they were **not**
in than for the one they were. That asymmetry is backwards, and closing it is the
"every surface agrees" half of the operator's request.

Signature change, additive and back-compatible:

```python
def send(self, kind: NotifyKind, *, body: str = "") -> bool:
    """Deliver one notification of `kind`; return whether anything was sent.

    `body` overrides the house sentence from BODIES when supplied. It is
    already composed and sanitised by `notifications.compose` — this method
    does not re-derive the privacy gate, because the composer read it at the
    same instant it read the text, and a second read here could disagree with
    the text it is gating.
    """
```

and `OperatorApp._notify` (`app.py:18490-18520`) resolves it:

```python
def _notify(self, kind: str, *, running_children: int | None = None) -> bool:
    ...
    notifier.set_label(self._notify_label())
    composed = compose(kind, session_dir=self._session_dir(), session_name=self._notify_label())
    if kind == "complete":
        return notifier.notify_turn_complete(running_children=running_children or 0,
                                             body=composed.body)
    ...
```

`notify_turn_complete`, `notify_waiting` and `notify_error` each take
`body: str = ""` and forward it. Every existing call site keeps working.

**One constraint the coder must respect:** `compose()` does disk I/O and
`_notify` runs on the Textual event loop. The reads are bounded (64 KB tail,
128 KB for the title) and the method is already wrapped in a `try/except` that
swallows everything, but a synchronous read on the UI thread at turn end is a
frame hitch. Use the app's existing worker pattern (`run_worker(thread=True)`)
or accept the hitch after measuring it. **Measure before deciding** — if
`session_preview` over the maintainer's largest transcript is under ~5 ms,
inline is fine and simpler.

> **MEASURED, and resolved to inline.** Over a synthetic 42 MB transcript
> (20,000 assistant messages of 2 KB each — larger than any real session on the
> maintainer's machine), `compose()` runs in **0.19 ms median, 0.31 ms worst of
> 20 runs**. That is two orders of magnitude under the 5 ms bar and under a
> frame at any refresh rate, because both reads are bounded windows rather than
> whole-file scans, so the cost is independent of transcript size. `_notify`
> therefore calls the composer directly; a `run_worker` hop would buy nothing
> and would cost the turn-end toast a scheduling delay. The measurement is
> recorded here rather than in a comment alone so a future reader can see what
> was actually tested rather than re-deriving the decision.
>
> The bridge's call is still wrapped in `asyncio.to_thread` (§6.2 note 1) and
> that is not inconsistent: it runs inside the 1 s attention poll shared by up
> to `BRIDGE_COUNT` bridges, where a burst of simultaneous completions stacks
> the cost on one event loop serving every session, rather than on one app's UI
> thread at one turn's end.

**What does not change:** the focus gate (`notify.py:1094-1095`), the cmux-first
routing (`notify.py:1104-1107`), and the fact that the snippet reaches
`cmux_command`/`notify-send` argv but **never** an OSC escape
(`app.py:18898-18906` — `notification_writes` is only ever called with the fixed
`BODIES` constant). If `send()` now passes a model-written body into
`notification_writes`, that invariant breaks and the OSC injection surface opens.

> **BLOCKER-CLASS CONSTRAINT, stated so it cannot be missed:** in
> `Notifier.send`, the composed body may be passed to the **cmux leg** and the
> **D-Bus leg**, but the **in-band OSC leg must keep using `BODIES[kind]`**.
> `sanitize_text` strips ESC and BEL (`notify.py:373`) so it is defence in
> depth either way, but the invariant that model text never reaches an OSC
> string is worth keeping intact rather than relying on one regex. A test pins
> this (§10.1, T-N4).

---

## 10. Test plan

### 10.1 Backend

**Extend `tests/unit/tui/test_notify.py`** (55 tests today) — or add
`tests/unit/notifications/test_compose.py` for the pure composer, which is
cleaner since the module is new:

| ID | Assertion |
|---|---|
| T-C1 | `complete` with a stored title and assistant text → title is the name, status `Complete`, body is the snippet, `body_is_snippet is True`. |
| T-C2 | `complete` with `display.notification_session_name` off → title is `APP_NAME`, body is `BODIES["complete"]`, both flags `False`. **The snippet is absent.** |
| T-C3 | `error` with assistant text present → body is `"Stopped with an error"`, never the snippet. Pins review round 1 M1. |
| T-C4 | `interrupted` likewise. |
| T-C5 | A 400-char assistant line → body ≤ 120 chars **and** ends on a word boundary (proves `max_chars` was passed to `session_preview`, not applied after). |
| T-C6 | A title/snippet containing `\x1b]0;pwned\x07` → both scrubbed. |
| T-C7 | Untitled session, `complete` → `BACKGROUND_FALLBACK_TITLE`; untitled session, `ask` → `APP_NAME`. |
| T-C8 | `approval` where `gate_detail` already starts with `gate_title` → no `"write: write:"` duplication. Pins round 4 Q3. |
| T-C9 | Unreadable/missing `session_dir` → returns the house vocabulary, never raises. |

**Extend `tests/unit/tui/test_notify_wiring.py`** (real app via `run_test`):

| ID | Assertion |
|---|---|
| T-N1 | A completed turn's toast body is the last assistant line, not `"Task complete"`. |
| T-N2 | With the privacy flag off, the same turn's toast carries neither the name nor the snippet. |
| T-N3 | An errored turn's body is the house sentence even with assistant text present. |
| T-N4 | **The OSC leg never carries model text**: with no cmux surface and an `osc9` terminal, the bytes written to the driver contain `BODIES[kind]` and not the snippet. Pins §9's invariant. |

**Existing `test_background_completion_notify.py` (34 tests) must stay green
untouched** — it already covers the claim/release path. Its continued passing is
the evidence that §7 changed nothing about the TUI's arbitration.

**New `tests/unit/server/test_desktop_notifications.py`** (mirroring
`test_desktop_attention.py`'s cold-path style):

| ID | Assertion |
|---|---|
| T-B1 | Publishing a `complete` completion for a bridged session emits exactly one frame with `type == "notification"`, after the `attention` frame. |
| T-B2 | The **first** `refresh_attention` of a bridge's life (baseline) emits **no** notification, even with an unseen completion already in the store. |
| T-B3 | The same completion observed twice (revision churn, no new token) emits one frame, not two. |
| T-B4 | A `kind == "interrupted"` completion emits **no** frame. |
| T-B5 | An `agent_end` event on the bridge emits an `event` frame and **no** `notification` frame. The anti-regression test for defect A. |
| T-B6 | A `turn_end` event likewise. |
| T-B7 | The notification frame is **not** in `bridge.replay` — reconnecting with `after_seq` before it replays the transcript events around it and not the notification. |
| T-B8 | `seq` still advances monotonically across a notification, and a client reconnecting at the notification's `seq` is **not** flagged `gap`. |
| T-B9 | `POST /notified` claims once: the second call for the same token returns `claimed: false`. |
| T-B10 | `POST /notified` does **not** change `unseen` or the `receipts` watermark — read `state()` before and after. Pins the two-watermark separation. |
| T-B11 | `POST /notified` never acquires a bridge or starts a runtime (monkeypatch `DesktopSessions.session` to raise, as `test_desktop_attention.py:39-44` does). |
| T-B12 | `POST /notified` with a foreign/unknown token returns `claimed: false` without writing. |
| T-B13 | An unreadable transcript during compose costs the notification, not the attention poll: the `attention` frame is still published. |

**Extend `tests/unit/session/test_attention.py`** (26 tests): one test that
`claim_delivery(..., backend="desktop")` and a TUI claim for the same token
produce exactly one winner. The primitive is already covered; this pins that the
desktop is just another claimant with no special path.

**`tests/unit/server/test_openapi.py`** picks up the new route automatically —
confirm the `/notified` shape is in the generated schema with `extra="forbid"`
on its `Input` subclass.

### 10.2 UI

**New `scripts/desktop-notifier.test.mjs`, added to the `test:desktop` list.**
Same in-memory esbuild bundling `desktop-contract.test.mjs:1-47` uses, with the
electron fixture extended to supply a **fake `Notification`** that records
constructor args instead of calling the OS:

```js
export class Notification {
  static isSupported() { return true; }
  constructor(options) { globalThis.__toasts.push(options); }
  on() {}
  show() { globalThis.__shown.push(this); }
}
```

This is the "test-visible delivery sink" the brief asks for: it proves the
*right payload* reached the OS boundary without anyone reading Notification
Center.

| ID | Assertion |
|---|---|
| T-U1 | A `notification` frame with `focus_policy: "when_unfocused"` and no focused window produces exactly one toast whose title/body are the backend's strings, verbatim. |
| T-U2 | The same frame with a focused window produces **no** toast **and no `sessions.notified` request**. The §7.3 ordering test. |
| T-U3 | Two frames with the same `dedupe_key` produce one toast and one claim. |
| T-U4 | `sessions.notified` returning `{claimed: false}` produces no toast. |
| T-U5 | A rejected/500 `sessions.notified` produces no toast (fail closed). |
| T-U6 | With `notificationContract = 1`, an `agent_end` event frame produces **no** toast. The anti-double-toast test. |
| T-U7 | With `notificationContract = 0`, `agent_end` produces the legacy toast and `turn_end` produces **none**. |
| T-U8 | An unknown future frame type (`type: "something_new"`) is ignored without throwing — the forward-compat test. |
| T-U9 | A `pending_gate` with `kind: "ask"` and an empty title falls back to `"Question"`, not `"Approval needed"`. |
| T-U10 | The notification path never emits an op of `sessions.seen`. |

**`pnpm check-types`, `pnpm lint`** cover the contract type addition.

### 10.3 End-to-end proof for the QA round

The real surface is an OS toast. Three layers of evidence, none of which is a
human staring at Notification Center:

1. **Bridge-side frame assertion (backend, real HTTP).** Extend
   `tests/e2e/test_desktop_sessions.py`, which already drives real loopback HTTP
   with the production `Session`/`RuntimeServer`/`AttachClient` and a scripted
   provider stream (`DESKTOP_API.md:248-252`). Run a turn that calls a tool and
   then answers, and assert on the **ordered frame log** from the SSE
   subscription: exactly one `notification` frame, following N `turn_end`
   frames, with a body equal to the model's last line. This is the proof that
   the *right thing* was emitted, at the real transport, over a real turn.
   Run it with `env -u NO_COLOR TERM=xterm-256color .venv/bin/python -m pytest
   tests/e2e -m e2e -n0 -q`.

2. **Delegation case, same harness.** A turn that spawns a `task` child and
   returns while the child runs must produce **zero** notification frames; the
   frame appears only after the child settles and its re-entry turn completes.
   This is the user's defect, proven end to end.

3. **UI-side delivery sink (Electron main harness).** T-U1..T-U10 above with the
   fake `Notification`. Plus one **manual-but-captured** step for the QA report:
   run the real app against a real backend, trigger a completion with the window
   unfocused, and capture the banner with a screenshot (`screencapture -x` on
   macOS catches the banner). That single screenshot is what validates the
   `status — body` vs `subtitle` decision in §8.3, which no unit test can.

**The QA matrix must cover, per surface:** completion / error / interrupted /
`ask` / `approval`; privacy flag on and off; window focused and unfocused; old
backend + new UI; new backend + old UI (install the previous signed build and
confirm no double toast and no crash on the unknown frame); two windows on one
session; TUI and desktop both running with the same session idle (exactly one
toast, and `lop sessions` still shows the row unseen afterwards — proving §7.2's
watermark separation on a real store).

---

## 11. Rollout order and risks

### 11.1 Order: backend first, always

The backend PR is safe alone:

- The `notification` frame is additive and ignored by every shipped UI
  (§4.3, verified against `use-canonical-session.ts` and `desktop-notifier.ts`).
- `POST /notified` is a new route nothing calls yet.
- `features.notification_contract` is a new key in an existing map.
- The TUI parity change (§9) is self-contained and independently valuable.

The UI PR is **not** safe alone: without the backend it has no frames to render,
and if it drops the `agent_end` legacy path on the strength of a capability the
backend does not advertise, completions go silent. The capability check in §4.3
makes the UI safe *whenever* it ships, but shipping it first delivers nothing.

Recommended sequence:

1. **Backend PR** — composer, frame, route, capability, TUI parity, tests.
   Merge, release, `lop-update`.
2. **UI PR** — contract type, notifier rewrite, capability gate, tests,
   screenshot evidence. Merge and cut the Electron release.

There is no window where the user gets two toasts for one turn: an old UI
ignores the new frame, and a new UI stops the legacy path only when the backend
says it owns notifications.

### 11.2 Risks to watch during rollout

1. **The `status — body` concatenation may read badly on the real OS.** §8.3.
   Watch: the screenshot in the UI PR. Remedy: `subtitle: status` on macOS. This
   is the one genuinely undecided rendering question and it is decided by looking
   at a frame, not by a doc.

2. **The 1 s attention poll adds a compose on the loop.** Only on the tick where
   a token changes, so the steady state is unaffected
   (`desktop_sessions.py:252-269` gates on `revision()`). But
   `BRIDGE_COUNT = 64` bridges each with their own poll means a burst of
   completions could stack `to_thread` calls. Watch: the e2e run's wall time,
   and `logger` noise from the poll's failure-transition logging. Remedy: the
   existing guard already costs one tick rather than the feature.

3. **`claim_delivery` returning `False` looks like a bug and is not.** When both
   a TUI and the desktop are eligible, one of them silently drops its toast. Watch:
   user reports of "I only got the banner on one surface" — that is the feature.
   The diagnostic is the `backend` column in `deliveries`, which will read
   `desktop` or `detached`/`cmux`.

4. **A `notification` frame consumes a `seq` without being replayable.** §4.4
   argues the gap arithmetic is unaffected. Watch: T-B8, and any `gap` frames in
   the e2e log after a notification. If this is wrong, the symptom is a spurious
   full-snapshot refetch after every completion — visible as a transcript flash.

5. **The composer reads the transcript while the session is writing it.**
   `session_preview` already handles a half-written final line
   (`resume.py:1974-1976`). Watch: empty bodies degrading to `"Task complete"`
   more often than expected, which would mean the read is racing the flush. The
   emitter fires off the *durable* attention publication, which happens after
   message persistence (`session.py:5982-5988`), so the row should always be
   there — but the ordering is worth one assertion in T-B1.

6. **Two windows, two relays, one HTTP claim.** §7.5 argues the shared dedupe
   map collapses them first. Watch: duplicate `POST /notified` in the backend
   access log. Harmless if it happens (the second returns `claimed: false`), but
   it would mean the map is not as shared as `desktop-notifier.ts:39` implies.

### 11.3 Documentation to amend in the backend PR

- **`docs/DESKTOP_API.md`** — a `notification` row in the frame list at
  line ~211-234, a `/notified` row in the endpoint table at ~156-166, and an
  amendment to the sentence at line 243 ("its gate/turn notification
  dedupe/click behavior is not implemented by these backend routes"), which this
  change makes partly false.
- **`docs/ATTENTION.md`** — one paragraph under "Read APIs and transports"
  naming the desktop as a `claim_delivery` claimant with `backend="desktop"`,
  and restating that the claim never advances the read watermark. Its existing
  closing line ("Desktop adoption ... must not equate a watch/notification lease
  with a durable read") is already correct and this design honours it.
- **`local_operator/tui/notify.py`'s module docstring** — the "Only a terminal"
  gate (lines 53-59) needs one sentence: the server still never *delivers*, but
  it now *composes* for a frontend that does.
- **`docs/SESSION_SIDEBAR.md`** — no change; its rule that routing a
  notification never marks anything read is preserved by §7.2 and pinned by
  T-B10.

---

## 12. What this design deliberately does not do

- **No new setting.** `display.notification_session_name` gates both the name
  and the snippet (§3.2). A second flag would let a user opt out of names while
  leaking conversation content, which is strictly worse than one clear promise.
- **No notification for `interrupted` on the desktop** (§6.5). One frozenset
  entry away if asked for.
- **No backend-side OS delivery for completions.** The `can_notify` lease
  already suppresses the runtime's fallback for gates; completions were never
  delivered from the server and still are not.
- **No replay of notifications** (§4.4). An edge that arrives hours late is
  noise; the durable unseen mark is the right surface for a missed completion.
- **No relocation of `CONTEXTS`/`BODIES`.** Ten import sites, zero benefit.
- **No change to the `pending_gate` notification path** beyond the `ask` title
  fallback. It works.

---

## 13. Amendments from the UI PR's design round (applied in the backend PR)

Two findings from the UI half's design round landed on the backend, because the
user-facing result spans both halves and neither could be fixed in the
renderer without moving a decision the backend owns.

### D4 (MAJOR) — an `error` banner must name the failure

**As designed**, §3.2 gave every non-`complete` kind the house constant, so a
failure read `Needs attention — Stopped with an error`. That names a state the
user must act on while withholding the only fact that says WHICH action: top up
a quota, fix a credential, or simply retry. They have to open the session to
learn anything, on the surface whose entire purpose is to save them that trip.

**Resolved by adding the text, not by deferring it.** The failure text IS
durably reachable without inventing a path: `incidents.py` journals a
`session_incident` custom message on every classified failure, persisted
precisely so a resumed session can explain itself, and its `details.raw` holds
the unrendered provider text. `resume.session_failure_summary()` reads it from
the same bounded 64 KiB tail window `session_preview` already uses, so the cost
is one more read of a warm region rather than a second scan strategy.

Three constraints the implementation holds:

- **`raw`, never `text`.** The rendered `text` is the model-facing block, several
  lines long and tailed with "This is why the previous turn ended. Take it into
  account before repeating the same request." — an instruction addressed to the
  model, which on a lock screen reads as nonsense.
- **Same scrub and budget as the snippet.** `sanitize_text(...,
  BACKGROUND_SNIPPET_MAX_CHARS)`: a provider's error envelope is untrusted text
  on the same argv and AppleScript wires, and is not length-bounded at source.
- **Same privacy gate,** for a stronger reason than the snippet's — an error
  envelope can quote a prompt fragment, a file path or an account identifier.

`ComposedNotification` gained `body_is_failure`, defaulted and last, so a
surface can tell the two kinds of untrusted text apart. The two flags are never
both true. This does not weaken review round 1's M1: the snippet is still
`complete`-only, and the reason the failure text is safe where the last
assistant line was not is that it describes the FAILURE rather than the work,
so it cannot read as a success beside a state that says it failed.

### D3 (MAJOR) — a gate banner must be triageable

**As designed**, completions gained a session identity and gates kept none, so
the banner holding a run hostage could not be attributed: "Waiting for approval"
with three sessions open names none of them.

**Resolved without a second channel**, exactly as §5.2 requires. Gates keep
travelling on `pending_gate`; `PendingGateState` gained an additive, optional,
defaulted `session_name`, stamped at both publication sites
(`ServingSessionHandle._publish_pending_gate` and `RuntimeServer.set_pending`,
the latter delegating to the former's helper so the privacy rule cannot hold on
one and not the other).

Empty when `display.notification_session_name` is off, and the emptiness is
decided in the BACKEND for the reason §1 gives for composing there at all: only
the backend can read that setting, and a renderer re-deriving it is one that can
get it wrong inside a signed binary the user updates on their own schedule.
Additive in both skew directions — an old viewer ignores the key, a new viewer
reading an old payload gets `""` and falls back to the anonymous card it already
draws — which is what lets the backend ship before the UI renders it.
