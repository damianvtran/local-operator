# Design: an attached interface is not an attended one

Status: **shipped on `fix/attached-interface-signal`, remediated in round 2**.
Extended in round 3 (the transport-bound hole, §12) on `fix/desktop-attach-truth`:
the attach fact is session-scoped, a successor reads the app's own record
narrowly, and an explicit withdrawal op ends the attachment. §12 is written to
match the code on that branch, same rule as the rest of this file.
This file began as a proposal (architect) against `origin/main` `1392324b` in the
worktree `~/local-operator-worktrees/attached-interface-signal`; it is now the
**in-branch authority for what the branch does**, so every body, predicate and
number below is written to match the code on the branch, and a round that changes
the code changes this file in the same commit. Round 2 (review remediation,
2026-09-23) changed four things it documents: §2.2's desktop clause, §3's state
set and bodies, §5's browser text and wait budget, and §5.5's two hard-coded
sentences. Scope: **one backend PR** (§9), plus a second commit the reviewer may
split out (§9 change 2, the rung-1 deny arm). No `pyproject.toml` bump. No UI
change is required for the model-facing fix, though §8 names the one UI fact that
would make it moot.

Operator's requirement, verbatim:

> Agents sometimes announce "nobody is at a screen to approve" … while the
> Local Operator desktop app (local-operator-ui) is open and the operator is
> right there reading and clicking.

Three numbered asks follow from it: (1) parent agents **and** subagents need a
trustworthy, **byte-stable** answer to "is an interface attached" — computed per
turn, never injected per step and never journalled per attach/detach; (2) the
default must not be "the operator is unavailable"; (3) the signal belongs in the
`browser` tool's access request, with a **15-minute** recommended wait and a
re-request after it.

Every `file:line` below was read on `1392324b`. Two read-only probes were run
against the operator's **live** machine: the machine-wide delivery-presence
record was read from disk (§1.1) and the transcripts of the operator's own
sessions were read for the incident (§1.4). Nothing was written, no session was
signalled, and the product code was not modified.

---

## 1. The problem as I found it

### 1.1 The live measurement: a focused, visible desktop app that cannot name its conversation

`~/.local-operator/run/desktop/delivery/<instance>.json`, read while the app was
open and focused on the operator's screen:

```
$ ls -la ~/.local-operator/run/desktop/delivery/
-rw-------@ 1 damian  staff  315 Sep 23 10:54 b01d6507fbd24452937d1be531518b02.json
$ cat ~/.local-operator/run/desktop/delivery/b01d6507fbd24452937d1be531518b02.json
{"pid": 1276, "instance_id": "b01d6507fbd24452937d1be531518b02", "can_notify": true,
 "can_notify_kinds": ["complete", "error"], "subscribers": 1,
 "window": {"exists": true, "focused": true, "visible": true, "minimized": false},
 "session_id": "", "written_at": 1790175294.9, "heartbeat_at": 1790175294.9}
```

Three facts in one 315-byte file, and each one decides a different leg:

- `subscribers: 1` and `can_notify: true` — a live watch lease exists for a
  conversation, and the app can raise a banner. `notification_surfaces()` is
  satisfied, which is why gate parking already worked in the incident (§1.5).
- `window.focused/visible: true` — `DesktopPresence.attended` is `True`
  (`presence.py:345-355`, `attended = has_window and focused and visible and not
  minimized`).
- **`session_id: ""`** — the app cannot name the conversation it is showing.

That last fact is the whole bug. It is not stale data: the publisher blanks the
field by design when it cannot vouch for it
(`server/utils/desktop_presence.py:382`: `newest.session_id if newest.has_window
and newest.can_notify else ""`), and the UI has a documented history of exactly
this shape — `~/local-operator-ui/src/main/desktop-notifier.ts:951-977` records
the review-round-2 finding R2-3, where the app "reported a focused, visible
window with `session_id: ""`, so the backend could not recognise the
conversation actually on screen".

### 1.2 The chain, and the one line where it goes wrong

```
RuntimeServer.__init__                      server.py:1302-1309
  └─ handle._install_interactivity_probe()  server.py:1303-1309
       serving.py:3581  holder.interactive_probe = lambda: bool(self._watching_surfaces())
         └─ serving.py:3594-3610  RuntimeServer.watching_surfaces()
              └─ server.py:3811-3840  watching_surfaces()
                   └─ server.py:3720-3750  _visible_attach_surfaces()
                        └─ server.py:3773-3791  _desktop_visible(conn)
                             └─ presence.py:345  attended = focused ∧ visible ∧ ¬minimized
```

`_desktop_visible` (`server.py:3773-3791`) is:

```python
try:
    presence = desktop_presence(getattr(self, "_config_root", None) or config_dir())
except Exception:
    return conn.desktop_visible
if not presence.present:
    return conn.desktop_visible
record = getattr(self, "_record", None)
session_id = str(getattr(record, "session_id", "") or "")
return presence.attended and presence.session_id == session_id
```

With `present=True`, `attended=True`, `presence.session_id=""` and a non-empty
`session_id` for this session, the machine-wide record **denies**. So the desktop
leg is dropped, `watching_surfaces()` is empty, the probe is `False`,
`GoalState.is_interactive()` is `False` (`goal.py:128-136`), and
`session_factory.py:2843` passes `interactive=False` into
`build_system_blocks` (`prompts_api.py:549`), which emits the `<interactivity>`
block at `prompts_api.py:710-737`.

The empty name is **absence of evidence being read as evidence against** — the
one failure mode the docstring at `server.py:3735-3740` says the fallback exists
to avoid ("an old UI's behaviour is byte-identical").

### 1.3 A second, independent defect on the same predicate: it answers the wrong question

Even with a perfect `session_id`, `_visible_attach_surfaces()` asks **"is a
person looking at this session right now"**. That is exactly the right question
for rung 1 of the notification ladder — `docs/DESKTOP_API.md:1421-1435` is
explicit that rung 1's predicate is the visibility one and must never be
`notification_surfaces()` — and it is exactly the wrong question for the model,
which needs to know whether a question can **be presented and be seen when the
operator returns**.

Consequences with a *correct* session id:

- An app window that is visible but **not focused** (the operator is reading a
  terminal beside it, or the app is behind another window) reports
  `attended=False` → "nobody is at a screen", while the pane is mounted, the
  lease is live, and a card painted now would be there when they look.
- A TUI holding this session in a tab it is not currently displaying sets
  `terminal_displaying=False` (`server.py:3962-3990`, the `viewer_watch` op) and
  is likewise dropped, though the session is open in a viewer that will present
  the card the moment the operator switches to it.
- Every focus change moves `attended`, so this question also **flaps** — which
  is the byte-stability problem in §3.

### 1.4 The incident, from the transcripts

Session `93d57660e002` is the subagent `post-analyst`
(`origin.json`: `{"origin": "subagent", "label": "post-analyst", "agent":
"post-analyst"}`). Its parent sent it a `<parent-message>` carrying the claim
verbatim:

```
There is no interactive surface attached to this session, so the operator cannot
click Allow on the https://www.linkedin.com origin approval right now. Do not
keep waiting on await_access: cancel the pending access request, report Part A as
"login-walled — origin approval unavailable this session, his own engagement
numbers not captured", and close the browser tab you opened
```

and the child reasoned about it, correctly, against the evidence in front of it:

```
Hmm. Options: ask the operator to sign in. But there's no interactive surface?
Actually there IS a browser tab in the desktop app. The parent said earlier
"There is no interactive surface attached to this session" — but now they
approved LinkedIn, so maybe there's a browser tab.
```

The operator then clicked **Allow** in the app the parent had declared
unwatchable. The false claim was made by a parent reading its own
`<interactivity>` block, relayed through `hub`, and acted on by a child that had
a better view of reality than its parent.

The same block is live across the fleet: `rg -l "nobody is watching a screen"` over
the operator's recent transcripts matches the system prompt of at least twenty
live sessions, e.g. `32ba88980028` ("Add approval request badges"),
`11f9a2692d57` ("Browser extension update enforcement failure"),
`4eabc50d61bd` ("Clear disk space and caches"). This is not one session's bad
luck.

### 1.5 What is NOT broken — proved, because the fix must not touch it

- **Gate parking already worked in the incident.**
  `serving.py:3484-3534` (`_gate_timeout_s`) parks when
  `self._watching_surfaces() or self._desktop_notification_available()`, and the
  second arm was satisfied (`can_notify: true`, live lease). The failure was
  "the model was told nobody can answer, so it declined to hold", not "the gate
  expired".
- **Residency is already lenient and must stay exactly as it is.**
  `process.py:637-657` (`_viewer_attached`) → `RuntimeServer.attach_clients()`
  (`server.py:3692-3712`) counts a desktop connection while
  `c.desktop_visible or c.desktop_can_notify`. That is a *different, wider*
  predicate than `_visible_attach_surfaces()`, and it is the one that keeps a
  runtime with an open pane alive. **Nothing in this design loosens or tightens
  residency**; a runtime must still exit when no visible panel is attached, and
  the §7 tests pin the negative.
- **Notification routing must not change.**
  `_announce_pending` (`serving.py:3738`) and the rung-1 comment at
  `serving.py:3720-3736` read `_watching_surfaces()`. Changing *that* to the new
  predicate is how "this machine can banner" once again became "a human is
  reading X".

---

## 2. The two-tier model

Two questions, two predicates, and each consumer reads exactly one of them.

**Tier A — ATTACHED.** *"An interface exists that can present a question, and
that the operator will see when they return to it."* No focus, no window z-order,
no `document.visibilityState`.

**Tier B — ATTENDED.** *"A person is looking at this session right now."*
Unchanged from today, in every respect.

### 2.1 Signal inventory

| Signal | Read at | Trustworthy for A? | For B? |
|---|---|---|---|
| Terminal `attach` conn, `kind=="attach"`, `surface!="desktop"` | `server.py:3720-3750` | **Yes.** The connection is the process that can paint the card; it stays counted while the socket lives, and `viewer_watch` is a *displaying* hint, not a membership test. | Only while `conn.terminal_displaying` — a multiplexer holding another session is not somebody reading this one (`server.py:3735-3740`). |
| Desktop `attach` conn, **lease live** | `server.py::_desktop_lease_live`; op at the `desktop_watch` handler | **Yes, and the lease alone** — see §2.2. The renderer sends this heartbeat only while it holds a live subscription **for this session** (`~/local-operator-ui/src/renderer/src/shared/hooks/use-desktop-watch-lease.ts:24-56`, withdrawn on pane leave at `:66-80`). "This conversation is open in a pane" is precisely presence, and it is the only desktop fact Tier A reads. | No. A lease is not attention. |
| Desktop `attach` conn, lease live, `desktop_visible` | same | Yes (a subset of the row above) — but **not read**: §2.2 | Yes, as today, **and** it is the sound fallback when the machine-wide record cannot name a conversation (§2.3). |
| Desktop `attach` conn, lease live, `desktop_can_notify` | same | Yes (a subset of the first row) — but **not read**: `can_notify` is reachability and it made the arm focus-dependent (§2.2) | No. `can_notify` is reachability, not attention (`presence.py:113-118`). |
| Machine-wide presence record, `present ∧ session_id == this session` | `presence.py:345, 360` | Yes — but redundant: an app showing this conversation holds the per-session lease above. Kept OUT of Tier A (§2.2), and used only where it already is: Tier B. | Yes, subject to `attended` (§2.3). |
| Machine-wide presence record with `session_id == ""` | `presence.py:345` | **No signal — absence of evidence.** Must not deny (§2.3). | Same. |
| Phone `watch` (`watch_supported ∧ phone_watchers > 0`) | `server.py:3811-3840` | **Yes.** | **Yes.** |
| `daemon` connection | `server.py:3811-3840` docstring | **Never.** The adoption dial covers every session on the machine and is held open permanently. | Never. |
| `notification_surfaces()` (desktop `can_notify` with a live lease) | `server.py:3793-3809` | Belongs to **reachability**, not attachment: a leased app with no pane on this session can be *told about* a card, it cannot *show* one. Read by parking (§9 change 1, item 4) and by rung 2, never by the model. | Never. |

### 2.2 The new predicate

New, additive method on `RuntimeServer`, beside `watching_surfaces()`:

```python
def attached_surfaces(self) -> frozenset[str]:
    """Which KINDS of interface can PRESENT a card that the operator will see.

    Sibling of ``watching_surfaces()``, NOT a replacement. That one answers "is
    a person looking at this session right now" and is the whole of rung 1 of
    the notification ladder (docs/DESKTOP_API.md §"The notification eligibility
    ladder"). This one answers "is there an interface that could show this
    session a question, and that the operator returns to" — the question the
    MODEL needs, because a question asked now is answered when they look, not
    when they are looking.

    Focus is deliberately absent. It flaps with window z-order, and every flap
    would move the model-facing block (§3).
    """
    attached: set[str] = set()
    for conn in list(self._clients.values()):          # snapshot: see C8 note, server.py:3724
        if conn.kind != "attach":
            continue
        if conn.surface == "desktop":
            # THE LEASE AND NOTHING ELSE (round 1, MINOR 5).
            if self._desktop_lease_live(conn):
                attached.add("desktop")
        else:
            attached.add("attach")
    if self.watch_supported and self.phone_watchers > 0:
        attached.add("viewer")
    return frozenset(attached)
```

The desktop clause is **the lease alone**, and that is a round-2 correction: it
was originally the `attach_clients()` clause verbatim
(`self._desktop_lease_live(conn) and (conn.desktop_visible or
conn.desktop_can_notify)`), on the theory that both callers asked "could this
front end present something". The theory cost Tier A its stability. The wire's
`desktop_visible` is the app's `visibilityState === 'visible' && hasFocus()`
(`~/local-operator-ui/src/main/desktop-notifier.ts`), so on a host with no
OS-notification channel (`can_notify` false — the JSON transport, browser dev)
the clause collapses to focus, and raising and lowering the window flips the
persisted block and writes a `[session-state]` row each way — the churn §3.3
forbids, in exactly the configuration §2.1 calls "reachability". The lease is
what the question asked for: it is renewed by a heartbeat naming THIS session's
own subscription and withdrawn when the pane leaves, with no window state and no
notification capability in it. `attach_clients()` keeps the extra clause, because
it answers a RESIDENCY question where an app that can neither show nor notify is
not a reason to stay up; the two expressions are now deliberately different.

The terminal clause deliberately **drops** `terminal_displaying`.

`serving.py` gains the matching reader next to `_watching_surfaces()`
(`serving.py:3594`):

```python
def _attached_surfaces(self) -> frozenset[str]:
    server = self._registrant
    reader = getattr(server, "attached_surfaces", None)
    if callable(reader):
        try:
            return frozenset(cast("frozenset[str]", reader()))
        except Exception:
            logger.debug("could not read the attached surfaces", exc_info=True)
    # An older registrant: attach_clients() is the same question one bit wide,
    # and it is the reading this handle's own probe already had available.
    return frozenset({"attach"}) if self._attached_clients() > 0 else frozenset()
```

The old-registrant fallback reuses the existing `_attached_clients()`
(`serving.py:3621-3631`), which already returns 0 on any raise; the failure mode
of a mixed-version fleet is therefore "attached", which is the direction this
design biases (§2.4).

### 2.3 The deny arm: `session_id == ""` is not evidence

In `_desktop_visible` (`server.py:3773-3791`), replace the unconditional
comparison with one that denies only on positive evidence:

```python
if not presence.present:
    return conn.desktop_visible
session_id = ...
if not presence.session_id:
    # The app has a window but cannot NAME the conversation on it. That is
    # absence of evidence, not evidence against: the publisher blanks the field
    # whenever it cannot vouch for it (server/utils/desktop_presence.py:382),
    # and a renderer-report lapse blanks it too. Denying here is what told the
    # operator's own focused, visible app that nobody was at a screen (§1.1).
    # The per-connection flag is the per-session answer — it is set by a
    # heartbeat that names THIS session's subscription (server.py:3947-3959) —
    # and falling through to it is exactly the pre-presence behaviour the
    # docstring at :3735-3740 promises for an older app.
    return conn.desktop_visible
return presence.attended and presence.session_id == session_id
```

Two consequences, both intended:

- **Tier B becomes marginally more suppressing** in the `session_id==""` case:
  an app that is focused *and* has this session's pane mounted now suppresses the
  in-band-duplicate banner where today it raises one. That is the UI's own
  R2-3 position (a focused window showing the conversation must not be bannered
  as background), and it is scoped precisely: the field is empty *and*
  `conn.desktop_visible` is true, i.e. a mounted, visible, focused pane. An
  unfocused window still yields `desktop_visible=False` and still banners.
- **The denied direction is preserved** where the record is evidence: an app that
  names a *different* conversation still denies, which is what stops the app
  from suppressing a background session's banner while it shows someone else.

This is the one change that touches rung 1, so it is scoped as its own commit
(§9 change 2) with its own test, and the reviewer may split it out without
disturbing the model-facing fix.

### 2.4 The bias, stated

Where the two tiers disagree, the model-facing answer reports **attached**. A
wrong "attached" costs a parked gate and a late answer; a wrong "unattached"
costs a turn that gives up on a question the operator was ready to answer — the
incident. The predicate is therefore built so that every uncertain path
(`getattr` miss, raise, old registrant, missing presence record) resolves to
attached, and only a positively-observed absence resolves to unattached.

---

## 3. The model-facing block: states, bytes, and cost

### 3.1 THREE states and TWO channels, and why not more

The block is a function of two **stable, measured** facts and nothing else.

**Tier A (three values).** `attached` / `unattached` / **`None` = nothing was
measured**. The third value is not a display convenience: `exec` runs, scheduled
runs and plain CLI sessions install no runtime probe (`_install_interactivity_probe`
lives on `ServingSessionHandle`), and the fail-open `is_interactive()` default —
which the PARK decision must keep, because a wrong "unattached" there costs the
incident — would otherwise ship a ~597-char claim about an attached interface to
every one of them. A block is a statement about what was measured, so with no
measurement it renders **nothing at all**, which is also the byte-shape those
hosts had before the positive arm existed. The model-facing reading is therefore
its own accessor (`GoalState.interactivity()`, `Session.interactivity()`) beside
the fail-open `is_interactive()`.

**The channel (three values).** `ask` / `hub` / `none`, stated by the caller
(`build_system_blocks(..., channel=...)`) from a fact only the caller has:

- `CHANNEL_ASK` — the live ask hook (`GoalState.can_ask()`, published by
  `Session.__init__` because `set_ask_handler` installs the hook AFTER the
  prompt-provider closure is built and the closure's `tools` list never gains
  `ask`). Read live, so the body follows the hook in both directions.
- `CHANNEL_HUB` — a delegated child (`harness/subagent.py`). Its attachment
  answer is the PARENT's, and its route to the operator is the session that
  delegated it. Inventory membership cannot derive this: a top-level session
  holds `hub` too (it is how ITS children reach it).
- `CHANNEL_NONE` — a served session with no ask hook, or a caller that stated
  nothing while holding no `ask` tool. `channel=None` falls back to tool
  membership, the same rule the inventory block already follows.

**A fourth state keyed on focus is still rejected** — it flaps (§3.3) — and so is
any state keyed on the surface KIND: a kind in the block would move the bytes when
the operator switched from the TUI to the app.

### 3.2 Exact bytes

Six constants, keyed on those two facts, with no timestamp, no session id, no
count, no host name, no surface kind and **no tool the reader does not have**:

| attachment | channel | constant |
|---|---|---|
| attached | `ask` | `_INTERACTIVITY_ATTACHED_ASK` |
| attached | `hub` | `_INTERACTIVITY_ATTACHED_HUB` |
| attached | `none` | `_INTERACTIVITY_ATTACHED_NONE` |
| unattached | `ask` | `_INTERACTIVITY_DETACHED_ASK` |
| unattached | `hub` | `_INTERACTIVITY_DETACHED_HUB` |
| unattached | `none` | `_INTERACTIVITY_DETACHED_NONE` |
| not measured | any | *no block* |

**Attached, channel `ask`** (byte-identical to round 1):

```
<interactivity>
An interface is attached to this session, so a question you ask WILL be
presented to the operator: `ask` puts it on that surface and waits for the
answer, parked for hours if necessary.

- Ask when the answer is genuinely the operator's to give, and not otherwise.
- The question is presented even if nobody is looking at this exact moment. It
  waits; it is not lost. A slow answer is not a refusal, and it is not a reason
  to decide on the operator's behalf.
- Write for a reader who may answer minutes later: say what you need and what
  you will do with it.
</interactivity>
```

**Attached, channel `hub`** (a child — round 1 told it to use `ask`):

```
<interactivity>
An interface is attached to the session this run was delegated from, so a
question you cannot settle yourself belongs to the operator: `hub` to that
session carries it there, and it waits for them — parked if necessary.

- Raise it through `hub` when the answer is genuinely the operator's to give,
  and not otherwise.
- Your question reaches the operator even if nobody is looking at this exact
  moment. It waits; it is not lost. A slow answer is not a refusal, and it is
  not a reason to decide on the operator's behalf.
- Write for a reader who may answer minutes later: say what you need and what
  you will do with it.
</interactivity>
```

**Attached, channel `none`:**

```
<interactivity>
An interface is attached to the session this run belongs to, but this run has no
way to put a question in front of the operator: no ask hook is wired into this
session, and it holds no channel to one that is.

- Say what you would have asked in your report — the question and the fact that
  would change your answer — and proceed on the best reading you have.
- Stating the question is not losing it: the operator reads this conversation
  when they return.
- Write for a reader who may answer minutes later: say what you need and what
  you will do with it.
</interactivity>
```

**Unattached, channel `ask`** (unchanged):

```
<interactivity>
No interface is attached to this session right now, so a question you ask cannot
be presented to anyone until a surface attaches: it waits, unread, and the turn
may block for hours.

- Prefer to PROCEED with what you have, or finish the turn with a clear
  statement of what you would have asked, over calling `ask`.
- That statement is a decision you already took and the fact that would change
  it, not a question left hanging.
- Do not take an irreversible or destructive action to avoid asking; when the
  choice genuinely needs a person, stop and say so — that is cheaper than a
  wrong guess.
- The operator will read this conversation when they return, so write for
  someone catching up, not for someone watching live.
</interactivity>
```

**Unattached, channel `hub`:**

```
<interactivity>
No interface is attached to the session this run was delegated from right now,
so a question raised through `hub` cannot be presented to the operator until a
surface attaches: it waits, unread.

- Prefer to PROCEED with what you have, or finish the turn with a clear
  statement of what you would have asked, over stalling on a question.
- That statement is a decision you already took and the fact that would change
  it, not a question left hanging.
- Do not take an irreversible or destructive action to avoid asking; when the
  choice genuinely needs a person, stop and say so — that is cheaper than a
  wrong guess.
- The operator will read this conversation when they return, so write for
  someone catching up, not for someone watching live.
</interactivity>
```

**Unattached, channel `none`:**

```
<interactivity>
No interface is attached to the session this run belongs to right now, and this
run has no way to put a question in front of the operator.

- Prefer to PROCEED with what you have, or finish the turn with a clear
  statement of what you would have asked, over stalling on a question.
- That statement is a decision you already took and the fact that would change
  it, not a question left hanging.
- Do not take an irreversible or destructive action to avoid asking; when the
  choice genuinely needs a person, stop and say so — that is cheaper than a
  wrong guess.
- The operator will read this conversation when they return, so write for
  someone catching up, not for someone watching live.
</interactivity>
```

The unattached bodies still **drop the sentence "nobody is watching a screen"** —
a claim the evidence does not support (the exact false negative of §1.1) and the
phrase the models parrot into `hub` messages. What remains is the fact that was
measured — whether a surface of THIS session is attached, or that none was
measured — and its consequence.

`prompts_api.py::_interactivity_block` renders them; `_interactivity_channel`
resolves the channel; `tests/unit/test_prompts_api.py` asserts each of the six
verbatim, so reintroducing a claim has to be deliberate.

### 3.3 Byte stability under churn — confirmed from the code, not asserted

- **The block lives in the tail, block index 3** (`prompts_api.py:762` returns
  `[instructions, inventory, env_block, tail]`), which is the section
  `Session._system_state_message` labels *"Knowledge and session state"*
  (`session.py:3635`). Index 0 is the frozen head.
- **It is re-emitted only when it changes.** `Session._system_state_delta`
  (`session.py:3594-3612`) compares each block against `_last_system_blocks`
  and emits a `[session-state]` custom message only for indices that differ
  (`session.py:3584-3589`). Byte-identical state ⇒ empty `changes` ⇒ **no
  transcript row, no message**.
- **The provider is cached on the same key.** `session_factory.py` keys the
  closure on `(…, interactive, channel, …)` and returns `list(cached_blocks)`
  when the key is unchanged, so a rebuild that changes nothing costs nothing —
  and the channel is in the key because the hook can be installed or removed
  mid-session.
- **50 focus changes cost zero.** Focus is not an input to Tier A (§2.2), so
  `attached_surfaces()` returns the same `frozenset`, the key is unchanged, the
  block is byte-identical, and `_system_state_delta` returns `{}`.
  `test_prompts_api.py::test_interactivity_costs_the_same_whatever_the_attach_churn`
  asserts the byte property against a CONSTANT; round 1 found that too weak
  (its NIT 8), so
  `tests/unit/session/runtime/test_server.py::test_fifty_real_focus_changes_move_neither_tier_a_nor_its_answer`
  now drives fifty real `desktop_watch` flips on a real connection in BOTH
  notification configurations and asserts the answer never moves — which is what
  the desktop clause change in §2.2 was for.
- **A genuine state change costs one row** and is thereafter silent. That is the
  intended trade and it is bounded by the number of *transitions*, not by the
  number of reattaches.

### 3.4 The positive case today emits nothing — what changes

Today `attached` renders no block at all, so the model receives no confirmation
that a question is answerable. After this change the positive block rides the
tail on **every** request of an attached session: **measured at 597 chars = 215
billed tokens** at the guard's 2.78 chars/token (§5 of the PR body; the ~90-token
estimate here was wrong, which is why the ratchet was raised rather than
discovered in CI), and unchanged by round 2 — the attached `ask` body is
byte-identical and the guard re-measured 84,009 chars = 30,219 billed against
30,365, i.e. 146 headroom. This
is a deliberate, permanent cost — it is the counterweight to the negative
block's existing advice, and without it the model's default reading of silence is
"probably nobody there". Stated here so the footprint is not discovered in
review: it is a constant cost per request, in the same cache prefix the tail
already occupies, and it is not a new tool (AGENTS.md, "The tool-surface
footprint ladder" — rung 1 does not apply, there is no new schema).

### 3.5 The install seam

`serving.py:3565-3583` (`_install_interactivity_probe`) stops reading the
attention predicate and reads the attachment one:

```python
holder.interactive_probe = lambda: bool(self._attached_surfaces())
```

The docstring's O(1)-in-churn claim (`serving.py:3566-3576`) stays true verbatim,
and becomes *more* true: the probe no longer flaps with focus.

---

## 4. Subagent propagation — the evidence

**A subagent's prompt never carries the block at all.** `_build_child_session`
builds the child's provider at `harness/subagent.py:1943-1982` and calls
`build_system_blocks(...)` at `:1968-1982` **without `interactive=`**, so the
default `True` (`prompts_api.py:561`) applies unconditionally. Children are
therefore silent on a question the operator can answer, which is the same defect
from the other side, and it is the reason the parent's false claim propagated
through `hub` into `post-analyst` (§1.4) unchecked.

**A child cannot answer the question by itself, and that is structural.** A child
`Session` is constructed in-process (`subagent.py:1996-2010`), holds no control
socket and has no registrant, so there is no `attached_surfaces()` for it to
read. Its only channel to a human is `hub` → parent: `build_ask_tool` returns
`None` when `context.ask_user is None` (`builtin.py:17017-17023`) and the
builder's own note says "A child session is built without an ask handler"
(`builtin.py:16995-16999`).

**The right answer is the parent's**, because the parent's session is the surface
the operator is attached to. The established pattern for exactly this already
exists one line above the call: `goal=parent_session.goal`
(`subagent.py:1976`), read live off the parent's holder via `Session.goal`
(`session.py:4103-4105` → `self._goal_state.text`).

So the child gets the parent's probe, installed on the child's own holder, and
the provider passes the live value:

```python
# in _build_child_session, after the child Session exists
parent_probe = parent_session.interactivity_probe      # new public read-only property
if parent_probe is not None:
    child._goal_state.interactive_probe = parent_probe

# in system_blocks_provider
return build_system_blocks(
    ...,
    interactive=parent_session.is_interactive(),
)
```

Reusing the probe object rather than its value keeps the child's answer live per
turn, exactly as `goal=` does, and keeps the child's `ask`-less browser text
(§5) in agreement with its own prompt.

The parent's holder is **not** shared: `GoalState` also carries `team_brief` and
`agent_brief` (`goal.py` and `session_factory.py:2833-2836`), and a child that
inherited those through the holder would silently start rendering the parent's
`<team>` block. Install the probe; do not hand over the object.

---

## 5. The browser access flow

### 5.1 Where the tool learns the answer

`ToolContext` currently exposes `ask_user` (`harness/types.py:1181`) and
`has_ui` (`:1017`) — and `has_ui` is explicitly **not** "a human is present"
(`builtin.py:16995-17014` documents the bug that reading it as such caused).
There is no attachment field, and the class docstring (`types.py:967-975`)
requires a capability a built-in tool looks for to be **declared**.

Add one declared field, wired from the session's own holder so it is live per
call:

```python
# types.py, beside ask_user
#: Live read of "an interface is attached to the SESSION this tool is running
#: in" (RuntimeServer.attached_surfaces via the goal-state probe). None means
#: no session is behind this context (a bare tool test), which reads as True —
#: the pre-existing default. NOT a synonym for has_ui, and NOT evidence that
#: anyone is looking right now: see docs/design/attached-interface-signal.md.
attached_probe: Callable[[], bool] | None = None
```

`Session._build_tool_context` (`session.py:8798-8830`) sets
`attached_probe=self._goal_state.is_interactive`. Passing the **bound method**
keeps it a live read rather than a snapshot, which is the reason
`session_name_provider` exists as a hook next door (`types.py:1018-1027`).
`tests/unit/session/test_tool_context_parity.py` is the enforcement that every
declared field is actually populated; it gets the new row.

### 5.2 `request_access` and the pending text

`_access_result_text` is the single renderer for every access state, and the
`pending` arm has variants keyed on three measured facts: the attachment of the
session (read through `ToolContext.attached_probe` AT RENDER TIME — see §5.3),
the notify channel this caller actually has (`_notify_channel`), and whether the
run is a DELEGATED one (`_delegated_here` — §5.2a).

**The wrapping is code, not authorship, and that is the fix for a whole class of
finding.** Every arm leaves through `_wrap_receipt`, which wraps to
`_RECEIPT_WRAP = 72` cells AFTER the interpolations: the card's lane is 76 cells
at 80 columns and its measure is 72, it paints ONE ROW PER RAW LINE, and it clips
a row that is longer. Round 1 measured the unwrapped bodies losing their
load-bearing clauses ("proceed with what you have", "15 MINUTES", "not a
refusal"); round 2 hand-wrapped the pending arms and still shipped three lines at
74-78 cells — one of them losing exactly the notify clause interpolation had just
added — plus `none`, `superseded` and `denied` as single 162-262-cell rows whose
recovery step clipped at EVERY width, `none` being the commonest post-expiry
state. A hand-measured line cannot be right when its length depends on
`{notify}`/`{origin}`/`{where}`; wrapping where those values are known can be.
`test_no_access_arm_has_a_row_the_card_would_clip` asserts the bound across every
combination (state × host × attachment × delegation × channel), and
`scripts/browser_access_shot.py` renders the real card for a look, at 80×30 and
100×32.

Rendered (attached, with `origin=https://www.linkedin.com/feed/`, `position 1 of
1`, `notify` the ask-bearing phrase, and `{where}` resolving to the extension
popup):

```
approval for https://www.linkedin.com/feed/ is pending (1 of 1).
The prompt is showing in the Local Operator extension popup (toolbar
icon, numbered badge showing the pending count) — the badge alone is not
reliably seen.

An interface is attached to this session, so the operator can answer it
as soon as they look — make sure they are told (a short message, or
`ask`).

- Wait UP TO 15 MINUTES in total for the decision: a person may be away
  from the desk, and a slow answer is NOT a refusal.
- Keep calling action='await_access' with the same url for that budget —
  each call waits at most 240s, so about four calls sized to that cap
  span it (an unsized call waits 120s, so eight of those do). That is
  the mechanism: there is no sleep shortcut here, because the `wait`
  tool awaits a background job and this flow has none.
- The prompt expires after about 10 minutes. If await_access returns "no
  live access request", call action='request_access' with the same url
  to raise a NEW prompt — that is what pings the operator again.
  Re-requesting while the old prompt is still live changes nothing and
  notifies nobody, so do it only once it has expired, and at most once
  per 15-minute window: after that, report what you have and move on.
- AN UNANSWERED PROMPT IS NOT A REFUSAL. Do not report it as refused —
  the request is still pending while an interface is attached — but say
  plainly if you proceeded without access.
```

Rendered (unattached — same inputs, `attached=False`):

```
UNapproval for https://www.linkedin.com/feed/ is pending (1 of 1).
The prompt is showing in the Local Operator extension popup (toolbar
icon, numbered badge showing the pending count) — the badge alone is not
reliably seen.

An interface is attached to this session, so the operator can answer it
as soon as they look — make sure they are told (a short message, or
`ask`).

- Wait UP TO 15 MINUTES in total for the decision: a person may be away
  from the desk, and a slow answer is NOT a refusal.
- Keep calling action='await_access' with the same url for that budget —
  each call waits at most 240s, so about four calls sized to that cap
  span it (an unsized call waits 120s, so eight of those do). That is
  the mechanism: there is no sleep shortcut here, because the `wait`
  tool awaits a background job and this flow has none.
- The prompt expires after about 10 minutes. If await_access returns "no
  live access request", call action='request_access' with the same url
  to raise a NEW prompt — that is what pings the operator again.
  Re-requesting while the old prompt is still live changes nothing and
  notifies nobody, so do it only once it has expired, and at most once
  per 15-minute window: after that, report what you have and move on.
- AN UNANSWERED PROMPT IS NOT A REFUSAL. Do not report it as refused —
  the request is still pending while an interface is attached — but say
  plainly if you proceeded without access.
```

#### 5.2a A delegated run credits the session it was delegated from

A child renders its PARENT's attachment answer — `harness/subagent.py` installs
the parent's live probe on the child's holder — so "an interface is attached to
this session" attributed the parent's pane to a run that owns none, while the
child's own `<interactivity>` block (§4) says "the session this run was delegated
from". The two are read by the same child in the same turn, and round 2's D9 is
exactly that disagreement. `_delegated_here(context)` settles it with the SAME
predicate `build_hub_tool` uses to choose the child-shaped `hub` tool
(`subagent_comms.is_child(context.job_id)`): the subject becomes "the session this
run was delegated from", and the notify channel becomes "a short message, or
`hub` to that session" — the route the block beside it names (round 2, Q9). A
top-level session holds `hub` too, so its children can reach IT, and is still
offered only "a short message": naming a route it cannot notify anyone through
would be the false instruction this flow exists to avoid.

Rendered for such a reader (the `child` case of the shot script):

```
approval for https://www.linkedin.com/feed/ is pending (1 of 1).
The prompt is showing in the Local Operator extension popup (toolbar
icon, numbered badge showing the pending count) — the badge alone is not
reliably seen.

An interface is attached to the session this run was delegated from, so
the operator can answer it as soon as they look — make sure they are
told (a short message, or `hub` to that session).

- Wait UP TO 15 MINUTES in total for the decision: a person may be away
  from the desk, and a slow answer is NOT a refusal.
- Keep calling action='await_access' with the same url for that budget —
  each call waits at most 240s, so about four calls sized to that cap
  span it (an unsized call waits 120s, so eight of those do). That is
  the mechanism: there is no sleep shortcut here, because the `wait`
  tool awaits a background job and this flow has none.
- The prompt expires after about 10 minutes. If await_access returns "no
  live access request", call action='request_access' with the same url
  to raise a NEW prompt — that is what pings the operator again.
  Re-requesting while the old prompt is still live changes nothing and
  notifies nobody, so do it only once it has expired, and at most once
  per 15-minute window: after that, report what you have and move on.
- AN UNANSWERED PROMPT IS NOT A REFUSAL. Do not report it as refused —
  the request is still pending while an interface is attached — but say
  plainly if you proceeded without access.
```

Three round-2 corrections are visible in that text, and each is the fix for a
review finding:

- **`{notify}` is what THIS caller can actually use** (`_notify_channel`): "a
  short message, or `ask`" where the session owns an ask hook, and only "a short
  message" otherwise. It used to name `ask` unconditionally, which a SUBAGENT
  does not have — and the same text told a child to "ask the user directly" on a
  deny.
- **The unattached variant no longer denies the surface it just named.** "the
  prompt is queued and nobody can act on it until a surface attaches"
  contradicted its own first sentence: `attached` counts Local Operator PANES,
  while the prompt is sitting in the extension popup (or the app's browser tab),
  which the operator can click. What is measured is that no pane of this session
  is attached, and that is what it says. The same correction removed the mirror
  over-claim from the attached variant ("the operator can see and answer it" was
  never what the predicate measured).
- **`ask` is no longer offered as a notify channel in the UNATTACHED variant**,
  whose advice is "do not block the turn": `ask` parks the turn for hours. The
  two arms now agree about whether to hold.

Round 3 added three more, all visible in the blocks above: the **subject** now
names the session that owns the pane (a delegated run's parent — §5.2a, D9); the
**arithmetic** states its assumption ("four calls sized to that cap", with "an
unsized call waits 120s" beside it) instead of a count that was only true of
calls the model sized to the cap itself; and the surface is named ONE way,
"Local Operator pane", where the timeout arm used to say a bare "pane"
(D8r2).

### 5.3 The await timeout text

The `remaining_ms <= 0` arm is now rendered by `_access_result_text`
(`state="await_timeout"`) rather than inline, because the two arms have to agree
and they did not: the pending text said "proceed with what you have rather than
blocking the turn", while the timeout arm told every caller — attached or not —
to keep waiting. It is attachment-aware, it names the wait that happened, and its
next step is executable:

```
still pending after 240s, and the operator has not decided on this
origin yet:
https://www.linkedin.com/feed/
Remind them to check the Local Operator extension popup. Then:
- An interface is attached to this session: keep calling
  action='await_access' — each call waits at most 240s — until about 15
  MINUTES in total have gone by.
- Once the prompt has expired, call action='request_access' with the
  same url to raise a new one: that is what pings them again.
AN UNANSWERED PROMPT IS NOT A REFUSAL.
```

with the first bullet replaced, when unattached, by:

```
- No Local Operator pane is attached to this session, so nothing in this
  run will present it: notify the operator (a short message) and proceed
  with what you have rather than blocking the turn.
```

The read of the attachment happens **here**, at render time, not once at the top
of `_bridge_access` (round 1, MINOR 3): an `await_access` that spent 240s waiting
must report the state at the END of that wait, or a surface that attached while
the model waited is told the session is unattached and advised to give up. The
three comments that claimed it was read after the wait (here, `harness/types.py`
and `session/session.py`) are now true of the code, and the `attached=` argument
no longer exists on the await-terminal call site where it could never be used.

### 5.4 The wait budget — decided, and the cap deliberately does NOT move

**`BROWSER_AWAIT_ACCESS_MAX_S = 240.0` stays, and the DEFAULT goes BACK to
`120.0`.** Three measured reasons for the cap, any one sufficient:

1. **The `browser` tool is `interruptible=False`.** A call that sits 15 minutes
   cannot be cut by a steer, a stop or an abort — the failure `ask`'s own builder
   documents. Raising the cap without flipping that flag converts a bounded wait
   into a hang.
2. **Each slice is a real RPC**, `_BRIDGE_AWAIT_SLICE_MS = 20_000`, so 15 minutes
   is ~45 round trips, and the extension's request TTL is 10 minutes
   (`extension/src/driver/access-queue.ts`, `ACCESS_REQUEST_TTL_MS = 10 * 60_000`).
   One prompt cannot serve a 15-minute wait anyway: at ~10 minutes the tool sees
   `state="none"` and returns.
3. **The budget is carried by REPEATED `await_access` CALLS**, which the pending
   text now says explicitly — about four calls SIZED to the 240 s cap, with the
   other half of the assumption beside it ("an unsized call waits 120s, so eight
   of those do"): the count was true only of calls the model sized to the cap
   itself (round 2, NIT 10). This replaces the original
   recommendation, which put the remainder in the `wait` tool and was **wrong**:
   `wait` awaits a background JOB (`WaitParams.job_id` is required) and a session
   awaiting an operator's click has none, so the named mechanism could not be
   called at all. The text states the executable path and names `wait` only to say
   why it is not it.

Round 1 also raised the DEFAULT to the cap "so an unsized call gets the longest
wait the tool can honestly serve". That is reverted: an unsized `await_access` is
exactly the call the pending text tells the model to make, so raising the default
doubles the uninterruptible block for the commoner case, which is the argument
reason 1 makes against touching the cap at all.

### 5.5 The `ask` refusal, and the only other hard-coded sentence

`rg "interactive surface"` over `local_operator/` returns exactly **two** hits:
`prompts_api.py` (the block, §3.2) and `builtin.execute_ask`. There is no third
in `hub`, `comms.py` or `control.py`.

`execute_ask`'s `ask_user is None` branch is unreachable through the advertised
tool (the builder refuses to create it without a hook): it is a host-wiring
fault, and the branch's own comment says so. But its text is the sentence the
models repeat, so it claims nothing about a screen and nothing about a ROSTER:

```
this host has no way to present a question to a person — no ask hook is
wired into this session, so this process cannot put one in front of the
operator. A delegated child's route to them is `hub` to its parent;
otherwise decide without them.
```

The roster it used to carry ("a subagent, an `exec` run and a scheduler run have
none") was **wrong about one of the three**: a supervised `exec --control` run
DOES have a hook (`exec_control.py` builds the handle with
`install_gates=supervised` and `serving.py` installs it via `set_ask_handler`),
which `build_ask_tool`'s own docstring says three lines above that string. A
roster is a second copy of a wiring fact, and this one had already drifted; the
condition is named instead, and the delegated child's real route is stated
because "the operator cannot be asked" is false for a child whose parent is one
`hub` call away.

**The gate-expiry notice, corrected.** `harness/render.py` rendered an expired
gate as "[system] … nobody was attached to this session and it expired", and this
document originally blessed it as true. That blessing was wrong once §2.2 and the
parking change land: attachment-first parking holds a gate for the configured
`unattended_gate_timeout` (24 h by default) when a pane IS attached, so the gate
can expire with a pane on it and the model is told nobody was there — the
sentence class this whole change exists to stop. Both arms now report the wait
they measured ("it expired unanswered after 24h"), through
`harness/rows.py::gate_waited_text` — ONE formatter, shared with the human row
(`gate_timeout_notice`), which carried the same unsupported "with nobody
attached" clause and now reads "waited 2h for approval, then expired unanswered".

## 6. Nothing else changes

- No new tool, no new op, no protocol field, no capability key. `desktop_presence`
  stays version 1 and the presence payload is untouched: the fix is in how one
  reader interprets a field it already has. `docs/DESKTOP_API.md:1508-1520`
  therefore needs no update.
- `docs/DESKTOP_API.md` §"The notification eligibility ladder" is **unchanged**
  except for one clarifying sentence in rung 1: its predicate is Tier B; Tier A
  is for the model and the gate, not for suppression.
- No UI change is *required*. The UI's own R2-3 fix
  (`desktop-notifier.ts:951-977`) would make the empty-name case rarer; it would
  not have fixed the incident's second cause (§1.3, focus), and it is not a
  dependency of this design.
- **Round 2 does change two files this plan did not list**, both of them copy:
  `harness/render.py` and `harness/rows.py` (§5.5 — the expired-gate sentence in
  two renderers of one event), and `scripts/bench_context_budget.py`'s ratchet
  comment (§3.4). The brief's own "one backend PR, four commits" shape is
  otherwise unchanged.

---

## 7. Tests

Existing seams are named, and each new test is listed with the failure it must
exhibit **before** the change (AGENTS.md, "Prove the test can still fail").

**The incident reproduction (fails before, passes after) — the one that matters.**
`tests/unit/session/runtime/test_server.py`, beside the existing
`test_watching_surfaces_*` (`:2053-2074`, `:2102-2200`):

- `test_a_focused_desktop_pane_that_cannot_name_its_conversation_is_still_attached`
  — write a **real** delivery record into an isolated config root
  (`{has_window: true, focused: true, visible: true, minimized: false,
  session_id: "", can_notify: true, subscribers: 1}`, the §1.1 payload verbatim,
  via `presence.delivery_record_path`/`DesktopDeliveryPublisher` — the shape
  `tests/unit/server/test_desktop_presence.py` already builds), register a live
  desktop `attach` conn with `desktop_can_notify=True`, `desktop_visible=False`,
  and assert `runtime.attached_surfaces() == {"desktop"}` **and**
  `runtime.watching_surfaces() == frozenset()`. Before the change there is no
  `attached_surfaces` at all, so the first assertion fails by `AttributeError`;
  after change 1 alone it fails on the shipped predicate, which is the
  regression this pins.
- `test_an_unfocused_desktop_pane_is_attached_though_nobody_is_watching` — the
  §1.3 case, with a *named* session id and `focused: false`.
- `test_a_multiplexed_terminal_away_from_this_session_is_attached_but_not_attended`
  — `terminal_displaying=False`: Tier A says `{"attach"}`, Tier B says empty.
- `test_a_daemon_connection_is_never_attached` — mirrors the rung-1 reasoning at
  `server.py:3811-3840`, so a machine running `lop mobile` cannot make every
  session claim an interface.
- **The negative that must not move (RESIDENCY):**
  `test_the_reaper_still_sees_no_viewer_without_a_visible_panel` — a desktop conn
  with `desktop_visible=False, desktop_can_notify=False` gives
  `attach_clients() == 0`. That is the half that proves residency was not
  loosened (§1.5). Since round 2 the same test also pins the SPLIT: the same
  connection has a LIVE lease, so `attached_surfaces() == {"desktop"}` — the pane
  holds this conversation, the reaper has no reason to stay up for it, and the
  two predicates are deliberately no longer one expression (§2.2).

**The deny arm (change 2).** `tests/unit/session/runtime/test_server.py`:
`test_a_presence_record_that_cannot_name_a_session_falls_back_to_the_connection`
(asserting `_desktop_visible` is `True` for `session_id==""` +
`desktop_visible=True`), `test_a_presence_record_naming_another_session_still_denies`,
and `test_a_presence_record_that_cannot_name_a_session_does_not_suppress_an_unfocused_banner`
(the `desktop_visible=False` case, which must keep bannerning).

**The block.** `tests/unit/test_prompts_api.py:961-996`:
`test_a_detached_session_is_not_told_the_operator_is_unavailable` renames and
re-authors the existing test to assert the negative body contains neither
"nobody is watching a screen" nor "nobody is at a screen";
`test_an_attached_session_is_told_a_question_will_be_presented` is new and asserts
`"<interactivity>" in attached[-1]`; the byte-churn test stays and gains the
attached state. **Every assertion is on the exact string**, not a substring of a
template, so a future edit that reintroduces a claim has to do it knowingly.

**Byte stability through the real journalling seam.**
`tests/unit/test_session_factory.py` (which already drives
`session_factory._make_system_blocks_provider` at `:2319` and a real child at
`:2384-2410`):
`test_fifty_focus_changes_journal_no_session_state_row` — build a session whose
probe flaps between two *identical* Tier-A answers, drive the provider and the
`_publish_state` path (`session.py:3546-3592`) 50 times, and assert the
transcript holds **zero** `session_state` custom messages and that block 3 is
byte-identical throughout; plus `test_one_attachment_transition_journals_exactly_one_row`.

**The probe wiring.** `tests/unit/session/runtime/test_serving.py`, beside
`test_background_desktop_owns_notification_without_becoming_interactive`
(`:1294-1324`, which must stay green unchanged — it is the routing half):
`test_the_model_facing_probe_reads_attachment_not_attention` — a fake registrant
with `attached_surfaces() == {"desktop"}` and `watching_surfaces() ==
frozenset()`, asserting the **installed probe returns True** while
`_watching_surfaces()` stays empty.

**The gate.** `tests/unit/session/runtime/test_parked_gates.py`:
`test_an_attached_pane_parks_a_gate_without_anyone_watching_it`.

**Subagents.** `tests/unit/test_session_factory.py`, following the
`_build_child_session` pattern at `:2384-2410`:
`test_a_child_is_told_whether_an_interface_is_attached_to_its_parent` — a parent
whose probe returns `False` (through a real `GoalState.interactive_probe`)
produces a child whose blocks carry the **unattached** body; and the mirror with
`True`. This test fails on today's tree because the child's provider never passes
`interactive=` (`subagent.py:1968-1982`).

**The browser tool.** `tests/unit/tools/test_browser_tool.py` and
`tests/unit/browser_bridge/test_tool_selection.py` own `_access_result_text`:
`test_a_pending_prompt_says_whether_an_interface_is_attached` (both variants),
`test_the_pending_prompt_names_the_fifteen_minute_wait_and_the_re_request`,
`test_an_unanswered_prompt_is_not_reported_as_a_refusal`, and
`test_the_await_timeout_names_re_request_not_an_endless_await`. These are pure
string tests over `_access_result_text` — no daemon, no browser.

**`ask`.** `tests/unit/tools/` — the existing ask-tool test that covers the
`ask_user is None` branch asserts the new text; `rg "No interactive surface"` in
`tests/` must return zero matches afterwards, which is cheap to assert as a
one-line guard in the same file.

**Run cost.** These are unit tests over fakes and isolated config roots; none
boots a TUI and none needs a live app. Per AGENTS.md, "Scoping the inner loop",
prefer `scripts/ci_scope.py --run`; the whole-tree suite remains CI's job.

---

### 7.1 Round 2 (remediation) — what the new tests pin

| test | the failure it exhibits before the round-2 change |
|---|---|
| `test_prompts_api.py::test_a_host_with_no_probe_is_told_nothing_about_attachment` | `interactive` had no unmeasured value, so an `exec`/scheduled/CLI host shipped the positive body |
| `test_prompts_api.py::test_a_delegated_child_is_told_the_route_it_actually_has` | the child's body named `ask`, which no child has, and said "attached to this session" about the parent's |
| `test_prompts_api.py::test_a_session_with_no_channel_promises_no_presentation` | the no-channel session inherited "a question you ask WILL be presented" |
| `test_prompts_api.py::test_an_unstated_channel_is_read_off_the_tool_inventory` | `channel=None` had no rule; `hub` must never be inferred (a parent holds it too) |
| `test_session_factory.py::test_a_child_of_an_unmeasured_parent_is_told_nothing` | a probe-less parent's child rendered the fail-open `True` default |
| `test_session_factory.py::test_one_attachment_transition_journals_exactly_one_row` | (extended) the row now carries the attached-`ask` body of a served session |
| `test_tool_selection.py::test_the_access_advice_names_only_tools_the_caller_has` | the browser text named `ask` for callers without a hook |
| `test_tool_selection.py::test_a_pending_prompt_says_whether_an_interface_is_attached` | (extended) the unattached text denied the surface it named |
| `test_tool_selection.py::test_the_await_timeout_names_re_request_not_an_endless_await` | (extended) the arm is attachment-aware and names the executable path |
| `test_server.py::test_fifty_real_focus_changes_move_neither_tier_a_nor_its_answer` | focus flapping moved Tier A on a `can_notify=False` host (NIT 8: the old test fed the renderer a constant) |

### 7.2 Round 3 (remediation) — what the new tests pin

| test | the failure it exhibits before the round-3 change |
|---|---|
| `test_tool_selection.py::test_no_access_arm_has_a_row_the_card_would_clip` | three arms were single 162-262-cell lines and three lines sat at 74-78 cells, so the card clipped the notify clause and every recovery step; the assertion is per row, over every state × host × attachment × delegation × channel |
| `test_tool_selection.py::test_a_delegated_run_credits_its_parents_interface` | a child's receipt said "an interface is attached to this session" about its parent's pane (D9), and its notify clause named no route at all |
| `test_tool_selection.py::test_the_real_path_gives_a_child_the_parents_interface` | the WIRING: `execute_browser` must read the delegated fact off the live context, or the renderer-level fix is dead code |
| `test_tool_selection.py::test_the_notify_route_names_only_tools_the_caller_has` | `hub` must be named for a child and NOT for a top-level session, which holds it for its own children (Q9) |
| `test_tool_selection.py::test_the_pending_prompt_names_the_fifteen_minute_wait_and_the_re_request` | (extended) the call count is qualified by the size it assumes — four sized calls, eight unsized (NIT 10) |
| `test_parked_gates.py::test_an_expired_question_is_not_reported_to_the_model_as_a_denial` | (extended) the model row said "the user" and said "expired unanswered" twice (D8r) |

## 8. Configuration

**No new key.** Every knob this design touches already exists:

- The gate's park duration is `runtime.unattended_gate_timeout`
  (read at `serving.py:3551-3562`, default `DEFAULT_UNATTENDED_GATE_TIMEOUT_H`).
  Unchanged.
- The notification switch is `notifications_enabled()`
  (`serving.py:3527-3534`). Unchanged.
- The browser flow's budget is the existing `timeout_s` parameter bounded by
  `BROWSER_AWAIT_ACCESS_MAX_S` (`builtin.py:11041`), which the caller already
  sizes and which stays put (§5.4). The 15-minute recommendation is a **constant
  in the result text**, next to the constant it describes, not a setting: nothing
  consumes it but the sentence, and a key that nothing reads is the failure mode
  AGENTS.md §"Adding a configuration key" exists to prevent.

If a future change does need one, it must be registered in `SETTINGS`
(`settings_io.py`), carry a module-level default constant mapped in
`_consumer_defaults()`, and pass `test_every_default_matches_its_consumer`.

---

## 9. Implementation brief

One PR, one branch (`fix/attached-interface-signal`), no version bump. Four
commits, in this order, each independently reviewable.

**Change 1 — `feat(runtime): an attached interface is not an attended one`**
*Files: `local_operator/session/runtime/server.py`, `serving.py`.*
1. Add `RuntimeServer.attached_surfaces()` exactly as §2.2, with the docstring
   that says why focus is absent and why the desktop clause is the LEASE alone —
   **not** the `attach_clients()` clause it originally shared (§2.2, round 2).
2. Add `LocalOperatorHandle._attached_surfaces()` beside `_watching_surfaces()`
   (`serving.py:3594`), with the old-registrant fallback through the existing
   `_attached_clients()` (`serving.py:3621`).
3. `_install_interactivity_probe` (`serving.py:3565-3583`) installs
   `lambda: bool(self._attached_surfaces())`.
4. `_gate_timeout_s` (`serving.py:3484-3534`) reads `_attached_surfaces()` where
   it reads `_watching_surfaces()` today; the
   `_desktop_notification_available()` leg and `notifications_enabled()` leg are
   untouched.
5. **Do not touch** `attach_clients()`, `_visible_attach_surfaces()`,
   `watching_surfaces()`, `notification_surfaces()`, `process.py` or
   `docs/DESKTOP_API.md`'s ladder in this commit.

**Change 2 — `fix(runtime): a presence record that cannot name a session is not evidence`**
*File: `local_operator/session/runtime/server.py`.* Apply the §2.3 rewrite of
`_desktop_visible` (`server.py:3773-3791`) with its comment, plus the three
tests, plus the clarifying sentence in `docs/DESKTOP_API.md`'s rung 1. Reviewer
may split this out; nothing in change 1 or 3 depends on it.

**Change 3 — `feat(prompts): tell the model whether a question can be presented`**
*Files: `local_operator/prompts_api.py`, `harness/subagent.py`,
`session/session.py`, `harness/types.py`, `tools/builtin.py`.*
1. `build_system_blocks` (`prompts_api.py:710-737`): replace the negative-only
   arm with both byte-stable bodies from §3.2, with the comment recording that
   the negative text no longer claims anything about who is looking, and that
   both bodies are constants so focus churn is free.
2. `harness/types.py`: declare `attached_probe` beside `ask_user` (`:1181`), with
   the §5.1 comment; add its row to
   `tests/unit/session/test_tool_context_parity.py`.
3. `session/session.py`: set `attached_probe=self._goal_state.is_interactive` in
   `_build_tool_context` (`:8798-8830`); add the public read-only
   `Session.interactivity_probe` property beside `goal` (`:4103`).
4. `harness/subagent.py`: install the parent's probe on the child's holder and
   pass `interactive=parent_session.is_interactive()` at `:1968-1982` (§4). Add
   the comment saying why the holder is not shared.
5. `tools/builtin.py`: reword the `ask_user is None` refusal at `:17097` (§5.5).

**Change 4 — `feat(browser): the access prompt says who can answer it`**
*File: `local_operator/tools/builtin.py`.*
1. `_access_result_text` `pending` arm (`:11077-11095`): the two variants from
   §5.2, keyed on `context.attached_probe` (defaulting to True when absent).
2. The `await_access` timeout arm (`:11198-11212`): the §5.3 text.
3. `BROWSER_AWAIT_ACCESS_DEFAULT_S` → `240.0`; **`BROWSER_AWAIT_ACCESS_MAX_S`
   stays `240.0`** and gets the comment from §5.4 naming `interruptible=False`
   and the extension's 10-minute TTL as the two reasons.
4. Add the `attached_probe` read to `_bridge_access`'s signature path so both
   `request_access` and `await_access` report the same fact.

**Gates, before the PR.** Every one of them whole-tree per AGENTS.md (there is no
affected-files shortcut): `flake8`, `black --check`, `isort --check-only`,
`make type-check` (which routes pyright through `scripts/run_bounded.py`; rc=124
is the bound, not a failure), and `.venv/bin/python -m pytest tests/unit -q` —
expect 40-55 minutes on this host under fleet load, so run it once, alone, and
check `uptime` first. `env -u NO_COLOR TERM=xterm-256color` for anything that
boots a TUI; unset every inherited `CMUX_*` variable in any test that does.

**Evidence for the PR.** The unit tests above are the mechanics; the evidence is
(a) the §1.1 presence file read live before the change, (b) the incident
transcript excerpt from §1.4, (c) the same 315-byte file and the same
`<interactivity>` block after the change on the operator's machine, and (d) the
transcript of a session that stopped carrying the block. `rg -l "nobody is
watching a screen"` over the operator's sessions, before and after, is a
fleet-scale before/after that needs no rig.

---

## 10. Risks to watch during rollout

1. **The block now asserts something in the positive case.** If `attached_surfaces()`
   is wrong in the *other* direction — a stale desktop conn with a live lease but
   no pane — the model will hold a turn waiting on a card nobody can see. The
   lease is 45 s (`DESKTOP_WATCH_LEASE_S`, mirrored by `PRESENCE_TTL_S`,
   `presence.py:83-86`) and the renderer withdraws on pane leave, so the window
   is bounded; watch for `wait` calls on sessions whose app just closed.
2. **Change 2 moves rung 1.** The failure it could reintroduce is a suppressed
   banner for a conversation nobody is looking at. Mitigated by keeping the deny
   on positive evidence only, and by the explicit unfocused-window test.
3. **A child now renders the parent's block.** If a parent is unattached, its
   children will say so — correct, but it is a new sentence in child prompts and
   will show up in token accounting.
4. **`_attached_surfaces()` on a mixed-version fleet** falls back to
   `attach_clients()`, which counts a desktop conn with `desktop_can_notify` even
   if the app never sends `desktop_watch`. That is the intended bias (§2.4), and
   it is worth stating in the PR so nobody "fixes" it into a false negative.
5. **Tier A now counts a leased pane whose window is hidden.** That is the point
   of the §2.2 change — the pane holds the conversation and the operator returns
   to it — but it does mean a gate on such a session parks for the configured
   `unattended_gate_timeout` where it used to fall back to the 30 s cap. The
   direction is the documented one (§2.4: a wrong "attached" costs a parked gate,
   a wrong "unattached" costs the incident), and `watching_surfaces()` still
   governs what has a human on it right now.
6. **The three-state block changes no bytes on the hosts it is meant to protect.**
   `interactive=None` renders nothing, which is what those hosts had before the
   positive arm existed; the only population that gains a body is the one with a
   probe, and its attached-`ask` body is byte-identical to round 1.

## 11. What I could NOT establish

- **Why the operator's app published `session_id: ""`.** The UI's own R2-3 fix
  (`desktop-notifier.ts:951-977`) reads the id from the renderer heartbeat map,
  and the live record still shows `""`. I did not read the running app's build
  or its IPC traffic, so I cannot say whether the installed app predates R2-3,
  whether the renderer report had lapsed
  (`RENDERER_REPORT_TTL_MS`), or whether that fix is incomplete. The design is
  deliberately independent of the answer: `""` is treated as absence of evidence
  either way. Settling it needs a UI-side probe, not a backend one.
- **Whether `desktop_can_notify` is `True` on this session's own desktop
  connection.** The machine-wide record proves a live claim exists somewhere on
  the backend; I did not read this session's `_clients` table (that would need a
  control-socket call into the operator's live runtime, which this design's rules
  forbid). Change 1 does not depend on it — the terminal and phone arms stand
  alone, and the fallback is deliberately biased to attached — but the §7
  incident test asserts the connection-level fact directly, so the PR's evidence
  will settle it.
- **The exact provenance of the sentence "There is no interactive surface
  attached to this session, so the operator cannot cli…"** in the operator's
  screenshot. `rg "interactive surface"` over `local_operator/` finds only
  `prompts_api.py:723` and `builtin.py:17097`, and neither reads "the operator
  cannot click". The transcripts show the parent **composing** that sentence from
  its own block and the model's prose (`93d57660e002`, §1.4), so I read it as a
  model paraphrase rather than a harness string — but I could not prove there is
  no third emitter in an installed build, because the uv tool install is not on
  this host at either path I checked
  (`~/.local/share/uv/tools/local-operator/` does not exist;
  `~/.local/share/lop/current` resolves to a generation directory holding only
  `bin/` and `tools/`).
- **Whether any *other* consumer reads `has_ui` as "a human is here".** I checked
  the `ask` builder (`builtin.py:16995-17023`) and left the field alone; a
  dedicated sweep of `has_ui` was outside this design's scope.

---

## 12. Round 3: the attach fact outlives the socket (`fix/desktop-attach-truth`)

### 12.1 The hole §2.2 left open

Everything in §2.2 is a fact about a SOCKET, and the socket is what a bridge
re-dial (`reader eof` / `reader reset`), a renderer stream restart or a runtime
swap takes away. Reproduced against the real `RuntimeServer` + a real control
socket (isolated config root), before the change:

```
desktop conn, before any desktop_watch   -> frozenset()
live desktop_watch lease                 -> frozenset({"desktop"})
socket lost, app still up, pane mounted  -> frozenset()   <-- FALSE "no interface"
re-dialed, re-assert not yet landed      -> frozenset()   <-- FALSE "no interface"
re-dial renewed the lease                -> frozenset({"desktop"})
conn alive, desktop_seen lapsed >45s     -> frozenset()   <-- FALSE "no interface"
```

Live evidence (2026-09-25): the session runtime log shows `dropped attach client
(… surface=desktop)` at 09:14:46 and storms at 09:17:10-22 / 09:19:02-39, each
coinciding with an injected "No interface is attached…" block; the app's log
shows the renderer's SSE re-subscribing with new epochs on each drop.

### 12.2 C1 — the heartbeat is remembered per session

`RuntimeServer._desktop_attach_seen`, renewed by every accepted `desktop_watch`
op and cleared only by the explicit withdrawal (§12.3). `attached_surfaces()`'s
desktop arm counts it for the SAME `DESKTOP_WATCH_LEASE_S` (45 s) window as the
per-connection lease — not a second constant, not a wider window — and after the
TTL with no heartbeat it is honest again. The comment on the field states what
renews it, why it is session-scoped (a live pane's attachment must not die with
the socket that carried it), and that no deliberate stop/exit path leaves the
runtime alive, so the withdrawal is the only in-place clear.

KEPT OUT of `attach_clients()` (the reaper), `watching_surfaces()`,
`_visible_attach_surfaces()` and `_desktop_visible()`: a stale memory must never
keep a runtime resident, and a dropped socket is not attention. Pinned by
`test_a_dropped_then_recent_heartbeat_moves_none_of_the_other_tiers`.

### 12.3 C2 — an explicit withdrawal, and why it is its own op

The obvious vehicle — `desktop_watch(visible=False, can_notify=False)` — CANNOT
carry this meaning, and the incident's own churn proves it: a transient renderer
stream end makes the bridge send exactly that pair (the post-pop refresh), and a
live pane on a host with no notification channel beats it every 15 s. Clearing
on `(False, False)` would wipe the memory at every stream restart — re-opening
the incident — and would flap the persisted block on no-notify hosts, which is
the round-2 churn pin. So the bridge sends a new, shape-gated, attach-only op,
**`desktop_withdraw`**, at the earliest moment this layer can honestly say "the
pane left": the lease EXPIRY path (`_expire_watches`'s final pass — 45 s with no
beat, each beat cancelling it), where a transient restart is already excluded by
its own re-subscription.

`AttachedSession` records the withdrawal as the desired state and `_dial`
replays it (the withdrawal arm of the re-assert), so a successor runtime engaged
under a closed pane starts and STAYS detached instead of resurrecting a 45 s
lease nobody holds; a stale, never-withdrawn record takes the same arm. Both
sends keep the existing `desktop_watch` envelope: bounded at
`_DESKTOP_WATCH_ACK_BOUND_S`, swallowed, because it is a presence hint whose
loss its own TTL repairs — and a cancellation still propagates.

PROMPTNESS is bounded by what this layer can know: the renderer's release
(`releaseWatchHeartbeat`) stops at main today, so the bridge cannot tell a leave
from a restart before the lease runs out. When the release reaches this bridge
(§8's UI item), the same op can be sent at once; the runtime semantics are
already right.

### 12.4 C3 — a successor reads the app's own record, narrowly

The desktop arm also counts `desktop` when the machine-wide presence record is
`present ∧ has_window ∧ session_id == THIS session's` (non-empty), bounded by
that record's own TTL (its reader already reaps it). It covers a successor
runtime booted under a still-open window before the bridge's re-dial lands.

**The rejected alternative, recorded because it was argued for:** granting on
`session_id == ""` (the architect's `∨ ""`). On this machine the record read
`""` WHILE the operator was watching — the UI withdraws the name on every
transient stream end (§8, C4) — so that grant would tell EVERY session on the
machine "an interface is attached". `_desktop_visible` (Tier B, §2.3) is
untouched; its fallback already denies nothing on the empty name.

### 12.5 C5 — the churn is readable now (logs only)

Bridge side (`server/utils/desktop_sessions.py`): one INFO/WARNING line per
ended subscriber stream naming the reason — client disconnect, subscriber
overflow, relay error, bridge dispose — and the session id; one line when a
re-dial lands, carrying the gap since the socket died. Runtime side
(`server.py::_drop_client`): the existing drop line gains, FOR DESKTOP CONNS
ONLY, whether the session-scoped memory survived the drop (`live` / `lapsed` /
`never`). No behaviour changes; the next storm is diagnosable from these lines.

### 12.6 What the extension changes for the tests

Fails on the pre-extension tree, passes after: `test_socket_lost_while_the_pane
_is_mounted_does_not_move_the_attachment_answer`,
`test_a_redial_before_its_re_assert_has_landed_does_not_move_the_attachment_answer`,
`test_a_lapsed_lease_past_the_ttl_reads_detached`,
`test_a_successor_runtime_reads_the_record_that_names_this_session`,
`test_a_dropped_then_recent_heartbeat_moves_none_of_the_other_tiers`,
`test_a_withdrawal_clears_the_memory_and_a_drop_does_not` (the C2 pin),
`test_a_real_drop_does_not_flip_the_interactivity_block` (prompt-level), plus the
bridge-side `test_the_last_lease_to_expire_is_withdrawn_explicitly` /
`test_a_transient_stream_end_renews_and_does_not_withdraw` and the dial-side
`test_a_recorded_withdrawal_replays_as_a_withdrawal_not_a_lease`.
`test_an_unnamed_record_does_not_grant` (the C3 overrule) and
`test_a_heartbeat_re_arms_the_attachment_answer` pin invariants that hold on both trees.

### 12.7 Not addressed here (UI repo)

The renderer withdraws its presence report (`releaseWatchHeartbeat` +
`noteLeftSession`) on EVERY transient stream end, not only on pane unmount —
which is why the machine-wide record read `session_id: ""` while the operator
was watching. That fix belongs in `~/local-operator-ui` (its own release
window); the backend here treats the empty name as absence of evidence either
way, and C2's withdrawal is what will make a genuine pane-leave prompt once the
release reaches this bridge.
