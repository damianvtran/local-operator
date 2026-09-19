# Design: operator authority for gate-loosening control requests

Status: implemented (Stage 1 + 1b). Issue:
`damianvtran/local-operator#1310`. Predecessor: #1282 / PR #1291, which closed
the *settings-write* half of the same invariant.

## 0. The invariant

```text
a constrained subject must not be able to mint the authority
that removes its own approval requirement
```

#1282 enforced it for `config.yml` writes: `tool_approval_mode` is live, a model
tool can write the file, so a loosening is authorised on the write's
**provenance** (`ConfigChange.source == "local"` in the process holding the
gate) rather than on its value.

This document covers the other boundary, which #1282 does not touch: the
**control plane**. A running session publishes a discovery record at
`<config>/run/mobile/<pid>.json`, mode 0600 under a 0700 directory, carrying
`control_key` and `control_port`. The runtime used to accept
`{"op": "slash_result", "command": "approvals", "args": "auto"}` — and
`{"op": "approval_answer", "approved": true}` — on that key alone.
`slash_result` rather than a bare `slash` is the op that reaches the sink: the
runtime answers a bare `slash` with "terminal-only here", so the route an
attacker actually uses is the `_result` one. A model-authored `bash` tool call
runs as the **same uid**, so it could read the record (it is 0600 *under its own
account*), dial loopback and remove its own gate. Same invariant, same subject,
one route over.

The two halves are not equally easy for that attacker, and the difference is
worth stating exactly: `/approvals auto` is a single frame, while the card half
needs the `request_id`, which is 8 random bytes and never guesses. It is not out
of reach, though — the welcome projection hands the pending card, id included, to
any client that authenticates with the record key, and a same-uid process can be
that client (that is what the negative control does). So the card half is
reachable *after one dial*, not by guessing (QA round 1, Q2).

The distinction #1310 draws, in one sentence: **same-account process identity is
not operator consent when the operation expands that process's own authority.**

## 1. The class

Exactly one predicate, `harness/approval.transition_authority`, beside
`loosening_is_authorised`:

```python
def transition_authority(command, args, *, approved=None) -> Authority
```

- **Authority-increasing** — the request removes or answers the gate that
  constrains the caller:
  - `/approvals auto|off|yolo` → `ServingSessionHandle._auto_approve = True`
    (or the app's `_set_approve_all` for the gate a TUI owns);
  - `approval_answer(approved=True)` → resolves the parked card.
- **Ordinary** — everything else: `ask|on|prompt`, any unknown word, a bare
  `/approvals`, `default …`, `read`/`status`/`stop`/`prompt`/`model`/`rename`,
  `peer_message`, `ask_answer`. An `approval_answer(approved=False)` is
  ordinary **on purpose**: a deny settles the card in the safe direction, so it
  must keep working from every surface that can reach the session.

Aliases resolve through `primary_slash_name` before matching — the same
resolution both dispatch hosts already perform — so an alias cannot slip past
the seam and reach a sink that would have honoured it.

The class is machine-testable rather than inferred from UI wording, which was
an explicit acceptance requirement: the predicate is a pure function, and the
seam's op table is re-derived from `session/runtime/server.py`'s own dispatch
source by `tests/unit/session/runtime/test_approval_authority_seam.py`.

## 2. The mechanism: one seam, one in-memory capability

**One seam.** `RuntimeServer._on_request` already holds both the connection and
the frame, and every route in the tree — the desktop
`POST /v1/desktop/sessions/{id}/commands` surface, the phone relay, a peer send,
a follower terminal, the CLI — reaches the handle through it, or through
`TuiSessionHandle` in the TUI's own process. So the guard lives there, before
dispatch: `harness/approval.AUTHORITY_OPS` names the ops in the class, and an
increasing frame is refused unless it presents the capability. The refusal
reuses the existing `{"op": "error"}` reply, carrying
`OPERATOR_CAP_REQUIRED_NOTICE` — or `CARD_APPROVAL_REFUSED_NOTICE` when the
refused frame was an `approval_answer`, because a person whose card was refused
needs a different sentence from a person whose command was (UX round 2, U8). The
op travels as a TOKEN and the sentence is rebuilt on the far side, exactly as the
typed category is.

**One sentence, both hosts, and the reader's machine.** `/approvals default …`
is answered by whichever host was asked, and the two handles had drifted into
two wordings of the same clause; it is now one shared builder
(`harness/approval.approvals_default_notice`). It also stopped saying "this
machine's `config.yml`": read from a phone, "this machine" is the phone (design
round 3, D16).

**The reports are told, not guessing.** A report (`/approvals` with a
divergence, `/approvals default …`) names remedies, and the same connection's
`/approvals auto` may be refused — so the seam passes `may_loosen`, judged by
its own predicate on the connection that asked, into the handle that builds the
sentence (design round 2 D10, UX round 2 U7/U9). A handle serves every
connection alike and cannot infer this; a report that offers a refused command
is how a follower was sent to a dead end twice.

**One capability.** 32 random bytes (`secrets.token_bytes(32)`), hex on the
wire, present in exactly two places: the memory of the gate-owning process (a
`RuntimeServer` field) and the memory of the console that started it (a
process-local pid → capability table in `harness/approval.py`). It is **never**
written to the record, the environment, argv, a log file, or `lop info` — and,
since the record is same-uid writable, it never crosses the wire either: what
crosses is a per-connection proof (see **Handoff** below).

**Handoff.** `session/runtime/launch._spawn_runtime` mints it, opens a
`socketpair`, passes the child's end with `pass_fds` and its *number* (not a
secret) in a new argv flag `--operator-fd`, writes the 32 bytes and closes both
ends immediately. `process.main` reads the descriptor, closes it, and hands the
value to the `RuntimeServer`. Tool subprocesses are spawned later with
`close_fds=True` and `start_new_session=True`, by which time the descriptor is
gone from this process's table too.

The console half is a **pid → capability** table rather than a field threaded
through every console object, because the question is a property of the process,
not of any one object: "did I start the runtime behind this record?".
`AttachClient.connect` resolves it from `record.pid`. The VALUE NEVER CROSSES
THE WIRE, and that is the second half of the design rather than a detail: the
record is same-uid WRITABLE, so `control_port` is not a trusted pointer and an
endpoint that receives a credential can replay it at the real runtime — measured
end to end with production clients before this was fixed (agent review round 1,
R1-1). What rides a frame is a per-connection HMAC **proof** of the capability
(`harness/approval._proof`), the runtime proves possession FIRST in its welcome
(`operator_proof` over the client's `operator_nonce` and a salt it mints for that
connection), and the client presents nothing at all to an endpoint that cannot
prove it holds the same value. `_request_frame` / `_request_payload` attach the
proof to exactly the frames `frame_authority` classes as increasing; the phone
relay's own writer does the same (`mobile/daemon._operator_request_proof`). That
is what makes the surface table below fall out of *one* rule instead of five
special cases.

**Windows.** `pass_fds` is POSIX-only, so the Windows path uses an inheritable
anonymous pipe with `close_fds=False`. The boundary is weaker there regardless
(any same-user process may read another's memory), so the runtime **reports the
level it can detect** rather than implying one
(`harness/approval.operator_cap_guarantee`): macOS `strong`, Linux `strong`
unless `/proc/sys/kernel/yama/ptrace_scope` is `0` (`not-a-boundary`), Windows
`weak`, Linux without `yama` `unreported`. The refusal copy names remedies; it
never claims a boundary the host does not have.

**Wire.** Three optional fields, all additive, so `PROTOCOL_VERSION` does not
move: `operator_nonce` on the auth frame (the client's half of the handshake),
`operator_proof`/`operator_salt` on the welcome (the runtime's half), and
`operator_cap` — the proof, never the value — on the increasing ops only.
`mobile/types.py` validates their shape; `AttachClient`'s request helper and the
relay's writer attach the request proof, and ordinary ops never carry it. A **rolling
upgrade fails closed** — an old console cannot loosen a new runtime (it sends no
capability) and a new console cannot loosen an old runtime's gate any more than
it could before (the old runtime simply ignores the field). The daemon's HTTP
boundary drops the field from request bodies outright: the relay is a local
process that attaches the capability itself, so a value arriving over HTTP can
only be a forgery.

## 3. Surface by surface

| surface | ordinary ops | authority-increasing |
| --- | --- | --- |
| TUI pane that OWNS the session (in-process gate) | unchanged | works, one step (the operator's own keyboard; not routed) |
| TUI pane viewing a runtime IT spawned | unchanged | works (presents the capability) |
| TUI pane viewing a runtime spawned by ANOTHER process (wake supervisor, peer send, `lop refresh`) | unchanged | refused; the copy names the remedies |
| TUI-hosted app reached by a follower (phone) | unchanged | refused |
| Desktop app | unchanged | works iff its backend spawned that runtime; else refused |
| Phone relay | unchanged | works iff the relay spawned that runtime; else refused |
| `lop` CLI / one-shot front ends | unchanged | only if this process spawned the runtime |
| `lop exec --control` (supervised one-shot), supervisor answering from another process | unchanged | **refused** — deny works, approve does not. The run's runtime is started by the `lop exec` process, which has exited or is backgrounded, so no live console holds the capability. Remedy for the next run: `--yolo`, or `tool_approval_mode: auto`; for an interactively approved run, start it where the approver is |
| `--yolo` | unchanged | n/a (the gate is born `auto`) |
| headless (non-TTY) | unchanged | **nothing may loosen it**: the gate is born **denying** (`session_factory.py`'s non-TTY default, `cli.py`'s one-shot path) and no console exists to hand it a capability, so `/approvals auto` is refused from every route. `--yolo` or `tool_approval_mode: auto` is the lever, at launch (agent review round 2, R2-3) |
| tightening `auto → ask`, any route | unchanged | unchanged |

Operator-visible regressions, stated plainly: a phone loses `/approvals auto`
**and card approval** for sessions whose runtime another live process started; a
pane attached to a background-started runtime cannot loosen it; and **a
supervised `lop exec --control` run cannot be APPROVED by a supervisor in
another process** — only denied — because its runtime is started by the `lop
exec` process and no live console holds the capability. Tightening, reporting
and everything else are untouched: the routes that may loosen are a proper
subset of the routes that may tighten.

### 4.1 The remedies, with their conditions

The refusal copy names only what the person reading it can do from where they
are, and this table is the fuller statement — several of these are CONDITIONAL,
which is why the copy names the event and the lever rather than promising one
total fix (design round 2, D11/D13; UX round 2, U9):

| remedy | what it does | when it works | in the refusal copy? |
| --- | --- | --- | --- |
| type `/approvals auto` in the window that started the runtime | loosens THIS session, in one step | that window is still live and still attached | yes, as the primary remedy |
| let the runtime retire, then reopen the session here | makes this window the one that starts the next runtime, so it owns the gate | always, but only when the runtime leaves — retirement is readiness-judged (`cli.refresh_command`), never forced | yes, as the background case |
| `lop refresh` | asks a live runtime to move to the install on disk and leave at its next boundary | the install on disk has MOVED (an update). Without a move a runtime answers "already current" and stays | **no** — conditional on a move, and the copy has no room to state the condition without pushing the reason off a narrow screen. It is here because a reader who HAS just updated should know it |
| `/approvals ask` | tightens | everywhere, including a follower, the phone and the desktop | yes |
| `--yolo`, or `tool_approval_mode: auto` in `config.yml` | the NEXT session starts loosened | at launch; the config write is a file edit (or the desktop app's settings), not a session command — no control connection may write it | yes |
| `lop stop <session>`, then reopen | ends this session so this window can own the next one | always, and it ENDS the running turn: named here for completeness, never as a remedy | **no** — it is not a remedy this change is willing to recommend |

Note the one the product does not have: an unconditional command that retires a
live runtime so a viewer can take over.

### 4.2 What the refusal looks like on a narrow screen

The copy is 345 characters (the card's is 162) and both are under the runtime's
400-character error-frame cap, which is asserted as a number
(`test_the_refusal_copy_names_the_remedies_and_not_a_rule`) so a longer copy
fails a test rather than a phone.

Rows are a property of the RENDERER, not of the characters: the notice block
wraps at its own content width — measured **40 cells** at a 44-column terminal,
not 44 — where the command's copy renders as **12 rows** and the card's as **6**.
The transcript's own content area at 44x20 is **11 rows** in a freshly booted app
and **13** once the transcript fills, so the block fits at every height measured
and nothing is clipped; the reason and the primary remedy lead the copy because
those are the rows that survive at the narrowest of them. Both block heights are
pinned as exact numbers in
`tests/unit/tui/test_approvals_ux.py::test_the_refused_card_notice_reaches_the_screen`
(design round 3 D14 and agent R3-5 corrected a wrap-based pin that said "9 rows"
and measured a wrapping the frame does not do; agent round 4 R4-2 corrected the
area figure this text used to claim).

## 4. The residual

The capability is a real boundary only where the operating system makes it one.
It does **not** defend against a process that can read the console's memory or
file descriptors: Linux with `ptrace_scope=0`, Windows, or an attached debugger.
That is the honest statement of what this closes and what it does not — it
removes "read a world-readable-under-this-uid record and dial a port" as an
attack, which is what a model-authored tool call can actually do today.

Also deliberately not fixed here, recorded so it is not mistaken for covered:

- **background-spawned runtimes have no console**, so nothing may loosen them
  until someone attaches one that does (a wake supervisor's runtime is exactly
  case 3 in the table). `lop exec --control` is the surface where this is most
  visible: its runtime is started by the `lop exec` process, so a supervisor in
  another process keeps the safe direction (deny) and loses the other one
  (approve). The spec that shaped this change asserted exec-control's gate is
  born `auto` and therefore unaffected; measured on `1effded9`, it is not —
  `tests/e2e/test_exec_startup_e2e.py::test_exec_supervisor_approval_ui` drives
  a parked `--control` gate and now pins the new behaviour. Whether to accept
  that, exempt `kind="exec"` (which would reopen the hole for the most
  unattended surface), or hand a capability to a supervisor through the exec
  ledger is the operator's call and is recorded as an open question on the PR;
- **a process that proxies the whole session** can relay a connection's proof
  and that connection's requests — which is what a proxy is — but it never
  learns the capability and cannot originate a request of its own: nothing that
  crosses the wire is reusable on another connection (agent review round 1,
  R1-1; `test_an_impostor_endpoint_learns_nothing_it_can_replay`);
- **the phone relay's own password** remains the authority for reaching a
  runtime the relay started, over a remote transport — Stage 3 replaces it with
  a device-bound credential;
- **the desktop `claim` handshake and the serve record's `claim_key`** are a
  different plane (governing a *daemon*, not a session gate) and are unchanged;
- **nothing identifies WHICH window owns a session when several are live.** The
  copy says "the window that started this session" and that referent is
  resolvable by the operator only by elimination: no `lop info` field, no band
  marker names the owner. This was raised in round 2 (UX U9) and answered with a
  route rather than an identity, and it is recorded here as still open;
- **the phone's card component renders the refusal body raw.** The sentence it
  now receives is the CARD's — the question survived, and a deny works from there
  — but `pending-card.tsx`'s `humanizeError` has no arm for it, so the copy is
  shown as sent rather than as a card-shaped message. That file is in another
  repository (UX review round 3, U11's remainder);
- **a relay dial that has ENDED leaves ``SessionEntry.authority_bearing`` set**
  until the next dial replaces the connection. Pre-existing and outside this
  delta, and no user-visible consequence has been produced from it — the next
  request is written on the next dial, which resets both fields — but it cost a
  reviewer one false reading, so it is recorded (QA review round 3, Q7);
- **D5's per-row "why" is deferred.** `auto` is tinted because this connection
  may not loosen THIS session and `default auto` because it is machine-locality
  and is refused from every surface, including a console — two different reasons
  behind one tint, distinguishable only by spending the keystroke. The refusal
  explains each; the list has no room at 60 or 44 columns (UX review round 3,
  U15).

## 5. Staged plan

- **Stage 1 (this change)** — the class, the seam, the in-memory capability, the
  handoff, the surface table above.
- **Stage 1b (this change)** — the guarantee-level report, the daemon's
  HTTP-boundary drop, and the not-at-rest probes.
- **Stage 2** — an OS boundary confining the model-code spawn sites, so the
  residual above stops being reachable by construction: macOS `sandbox-exec`,
  Linux Landlock/`bwrap`, Windows restricted token plus a deny ACE.
- **Stage 3** — a device-bound credential for the phone, replacing the relay's
  password as the authority for the remote path.
