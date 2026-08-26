# Design: peer-to-peer messaging between `lop` sessions (`lop send` / `lop sessions`)

Status: proposal for implementation. Author: architect (opus-4).
Base: `origin/main` @ `f2a52b53`. All file:line references are against that tree.

## 0. Problem and shape of the solution

The user wants one `lop` session to hand a message to another `lop` session on
the same machine **without cmux**, plus a command to **list active sessions and
their resource usage**. The delivered message must land in the target's message
history so the human can see it, marked as a cross-session message in both the
TUI and the mobile web surface.

The good news, confirmed by reading the tree: **the entire transport already
exists** in `local_operator/mobile/`. Every `lop` process already publishes a
discovery record and runs an authenticated loopback control server (the
"registrant"). The phone daemon is just one client of that server. `lop send`
is a *second kind of client* — a short-lived CLI process that dials the same
socket and speaks one new control op. We add:

1. one new `ControlOp` (`peer_message`) + its frame validation,
2. one branch in `Registrant._dispatch`,
3. one new `SessionHandle` method (`receive_peer_message`) implemented on the
   two real handles (`OwnedSessionHandle`, `TuiSessionHandle`) plus a new
   `Session.receive_peer_message`,
4. one new transcript custom type (`PEER_MESSAGE_MESSAGE_TYPE`) that renders
   with a cross-session indicator in TUI, mobile web, and the subagent/peek
   paths,
5. a small stand-alone sender client (reuse of `attach_client` dial mechanics),
6. two CLI subcommands (`send`, `sessions`),
7. a guide at `guides/peer-messaging/GUIDE.md`.

We do **not** invent an IPC stack, a new socket, a new auth scheme, or a new
registry. The trust boundary is already correct: the record is `0600` under a
`0700` dir (`registry.run_dir` at `registry.py:30`, `SessionRecord` docstring at
`types.py:216-219`), so anything that can read the `control_key` is already the
owning account — which is exactly the "same-account is the trust boundary"
decision.

### Where I diverge from the framing

- **The framing says "reuse the `parent_message` EntryKind/rendering path or add
  a new `peer_message` kind? Decide."** I recommend a **new `peer_message`
  EntryKind** (mobile) and a **new `PEER_MESSAGE_MESSAGE_TYPE` custom type**
  (transcript). Reusing `parent_message` would be wrong: `parent_message` means
  "your hub parent said this" and is rendered as "Parent" (`transcript.tsx:135`,
  `subagent_view.py:315,332`). A peer session is not a parent; overloading the
  kind would mislabel real hub messages and vice-versa, and the indicator must
  name the *sending session* (pid/conversation/model), which the parent path has
  no slot for. The cost of a new kind is small and localized (one Literal entry
  each in `types.py` and `types.ts`, one `case` in `transcript.tsx`, one branch
  in `subagent_view.py`, one block type in the TUI). See §3.

- **The framing suggests `receive_peer_message` "distinct from prompt/steer".**
  Confirmed necessary, and for a reason the framing did not state: there is
  currently **no path that both persists a message to the transcript AND injects
  it into the live `_context.messages` without driving a turn.** `queue_aside`
  (`session.py:3850`) persists lazily but only *materializes at a turn boundary*
  — for a genuinely idle session there is no boundary, so the message would sit
  un-injected until the next user turn triggers `_drain_pending`. Actually that
  is *acceptable* for record-only (the model sees it on its next turn either
  way), but the durable transcript write must happen *now* so the human sees it
  immediately and a crash cannot lose it. So record-only needs an eager
  transcript append + a context append, which is a new method. Details in §2/§3.

- **RSS vs footprint:** the framing is right that RSS under-reports. But
  `psutil` is **not a dependency** (`pyproject.toml` has none; `import psutil`
  fails in the venv). I therefore specify a **stdlib + `ps`/`top` subprocess**
  approach with graceful degradation, not psutil. See §4.4.

## 1. Wire protocol

### 1.1 The new op

Add exactly **one** op, `peer_message`, with a `mode` field — **not** two ops.
Justification: mailbox vs steer differ only in *when* the message is consumed,
not in *what* crosses the wire; both carry identical fields (text, sender
identity, flags). A single op with a `mode` discriminator keeps
`validate_control_frame` and `_dispatch` to one branch each and matches the
existing convention where `prompt` and `steer` are peers but a single validated
family. Two ops would duplicate validation and identity handling for no gain.

`local_operator/mobile/types.py` — extend the `ControlOp` Literal (search for
the existing definition; it lives near the top with the other op names) to
include `"peer_message"`.

### 1.2 Frame shape

```jsonc
{
  "op": "peer_message",
  "req": 1,                     // request id, matched by the ack (existing convention)
  "text": "…",                  // required, non-empty
  "mode": "mailbox",            // "mailbox" (default) | "steer"
  "wake": false,                // bool; only meaningful when mode == "mailbox"
  "sender": {                   // identity for the indicator; all optional strings/ints
    "pid": 12345,
    "session_id": "…",
    "conversation_name": "peer-send design",
    "model_label": "anthropic/claude-opus-4",
    "cwd": "/tmp/lop-peer-send"
  }
}
```

Semantics:
- `mode == "mailbox"` (default): queue as the next user turn, non-interrupting.
  With `wake == false` (default) → record-only: durable row appears now, the
  agent stays idle if idle, and reads it on its next turn. With `wake == true`
  → if the session is idle, drive a turn now; if busy, it is already going to be
  read at the next turn, so `wake` is a no-op while busy (documented).
- `mode == "steer"`: inject mid-turn exactly like the existing steer path. If
  the session is idle, "steer" degrades to opening a turn immediately (same as
  wake) — there is nothing to steer into, and dropping it would violate "the
  message MUST appear in history." `wake` is ignored in steer mode.

### 1.3 Validation

`local_operator/mobile/types.py`, in `validate_control_frame`
(`types.py:110-154`), add an `elif op == "peer_message":` clause:

```python
elif op == "peer_message":
    text = frame.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    mode = frame.get("mode", "mailbox")
    if mode not in ("mailbox", "steer"):
        raise ValueError("mode must be 'mailbox' or 'steer'")
    if "wake" in frame and not isinstance(frame.get("wake"), bool):
        raise ValueError("wake must be a boolean")
    sender = frame.get("sender", {})
    if not isinstance(sender, dict):
        raise ValueError("sender must be an object")
    # sender fields are advisory (indicator only); do not hard-fail on missing
    # keys — an older sender that omits them still delivers, just less labelled.
```

### 1.4 Additive versioning / graceful degradation

`PROTOCOL_VERSION` is currently `4` (`types.py:177`). **Do not bump it for this
feature** — the bump is only load-bearing when an *old client* must refuse a
*new registrant* (the attach-eviction case documented at `types.py:161-176`).
Here the opposite risk applies: a *new sender* dialing an *old registrant*. That
is handled without a version bump because `Registrant._dispatch` already ends
with `raise ValueError(f"unknown op: {op!r}")` (`registrant.py:675`), which the
sender receives as an `{"op": "error", ...}` frame (`registrant.py:611`). So an
old registrant answers a `peer_message` op gracefully with a clear error string,
and the sender prints "this session runs an older lop that cannot receive peer
messages (pid N)". The sender should treat that specific error as a soft failure
(exit non-zero, human-readable), not a crash.

If a future change makes peer messaging load-bearing for cross-version safety,
bump then. For now: purely additive, no bump. Document this reasoning in the op
comment so a later maintainer does not "helpfully" bump it.

## 2. Registrant dispatch

`local_operator/mobile/registrant.py`, `_dispatch` (`registrant.py:614-675`).
Add before the final `raise`:

```python
if op == "peer_message":
    receive = getattr(h, "receive_peer_message", None)
    if not callable(receive):
        # An owner host that predates peer messaging. Same optional-capability
        # pattern as recall_steer (registrant.py:657).
        raise ValueError("this session cannot receive peer messages")
    typed = cast(
        Callable[..., Awaitable[str]], receive
    )
    return await typed(
        frame["text"],
        mode=str(frame.get("mode", "mailbox")),
        wake=bool(frame.get("wake", False)),
        sender=frame.get("sender") or {},
    )
```

Note `_on_request` (`registrant.py:604-609`) already sends the ack with the
returned detail string and then calls `self._handle.refresh()` + `self._push()`,
so the projection (and thus the phone) repaints after delivery for free. No
extra push wiring needed.

### 2.1 New `SessionHandle` method

`SessionHandle` Protocol (`registrant.py:138-181`): add

```python
async def receive_peer_message(
    self,
    text: str,
    *,
    mode: str = "mailbox",
    wake: bool = False,
    sender: dict[str, Any] | None = None,
) -> str: ...
```

Make it **optional at the Protocol level** by treating it as a capability
(the dispatch uses `getattr`, matching `recall_steer` at
`registrant.py:657`), so fake/third-party handles in tests need not implement
it. Add it to the Protocol for documentation and typing on the real handles.

### 2.2 `OwnedSessionHandle.receive_peer_message`

`local_operator/mobile/owned.py` (add after `steer`, ~L623). This handle owns an
in-process `Session` on `self._loop`:

```python
async def receive_peer_message(
    self,
    text: str,
    *,
    mode: str = "mailbox",
    wake: bool = False,
    sender: dict[str, Any] | None = None,
) -> str:
    self._check_loop_thread()
    detail = await self._session.receive_peer_message(
        text, mode=mode, wake=wake, sender=sender or {}
    )
    # Mirror the phone fold the same way steer() does (owned.py:620-622): put a
    # visible row on the projection now so any attached phone paints it without
    # waiting for the next MessageStartEvent.
    self._fold.note_peer_message(text, sender=sender or {})
    self._notify()
    return detail
```

### 2.3 `TuiSessionHandle.receive_peer_message`

`local_operator/mobile/tui_handle.py` (add after `steer`, ~L306). This handle
bridges to the app's live `Session` via `self._on_app(...)` / the owner loop.
Follow the `steer` shape (`tui_handle.py:272-306`): run the session mutation on
the owner loop, then fold + notify:

```python
async def receive_peer_message(
    self,
    text: str,
    *,
    mode: str = "mailbox",
    wake: bool = False,
    sender: dict[str, Any] | None = None,
) -> str:
    session = self._session()
    detail = await self._on_app(
        lambda: session.receive_peer_message(
            text, mode=mode, wake=wake, sender=sender or {}
        )
    )
    # session.receive_peer_message is async; _on_app runs sync callables on the
    # Textual thread. Prefer scheduling the coroutine on the owner loop the same
    # way TuiSessionHandle.prompt does (tui_handle.py:256) via
    # run_coroutine_threadsafe, then awaiting the future. See implementation
    # note below.
    self._fold.note_peer_message(text, sender=sender or {})
    if self._on_projection is not None:
        self._on_projection()
    return detail
```

**Implementation note for the coder:** `Session.receive_peer_message` is a
coroutine that must run on the owner event loop (it touches `_context.messages`,
the transcript, and may spawn a turn). `TuiSessionHandle` already has the exact
machinery for "run a coroutine on the owner loop and await it from the
registrant's bridge coroutine" — see `prompt` (`tui_handle.py:231-270`) using
`asyncio.run_coroutine_threadsafe(run_turn(), owner_loop)` and `_await_future`.
Reuse that pattern rather than `_on_app` (which is for *sync* callables on the
Textual thread). The simplest correct form: get `owner_loop` (the session's
loop), schedule `session.receive_peer_message(...)` on it with
`run_coroutine_threadsafe`, and `await asyncio.wrap_future(...)`.

### 2.4 Why a new `Session.receive_peer_message` and how it maps to modes

`local_operator/session/session.py`. The three delivery outcomes map onto
machinery that already exists, but no single existing method combines them:

- **record-only (mailbox, no wake, idle)** — the message must be *durably
  persisted now* and *visible to the model on its next turn*, without opening a
  turn. Append the `CustomMessage` to the transcript immediately and append it
  to `_context.messages` so the next `_run_turn_pipeline` includes it. This is
  the branch with no existing home (`queue_aside` only materializes at a turn
  boundary, `seed_history` is pre-first-turn only). New code.

- **mailbox + wake, idle** — same durable persist, then drive a turn. Mirror the
  idle wake path: `self._spawn_background(self._prompt_messages([message]))`
  (`session.py:5619`). `_prompt_messages` persists via the pipeline; to avoid a
  double transcript write, the wake path passes the `CustomMessage` *into* the
  pipeline (which appends it once). Follow the wake precedent exactly.

- **mailbox, busy** — the agent is going to read `_context.messages` at the next
  turn regardless. Persist durably now (transcript + context append). `wake` is
  moot while busy. Optionally also enqueue as courtesy steering if you want it
  surfaced mid-turn, but the product decision says mailbox is *non-interrupting*,
  so do **not** steer in mailbox mode — just persist and let the running turn's
  natural context refresh pick it up, or the next turn read it.

- **steer mode, busy** — inject mid-turn via the existing steer path:
  `self.steer(text, ...)` (`session.py:2635`), which enqueues to
  `_steering_queue` and is folded into the running turn. The steer path already
  persists the steering message.

- **steer mode, idle** — nothing to steer into; degrade to a driven turn exactly
  like mailbox+wake idle.

Signature:

```python
async def receive_peer_message(
    self,
    text: str,
    *,
    mode: str = "mailbox",
    wake: bool = False,
    sender: dict[str, Any] | None = None,
) -> str:
    """Deliver a message from ANOTHER local lop session into this one.

    Returns a short human-readable detail string for the sender's ack.
    """
```

Body sketch (the coder fills in against the real helpers):

```python
    sender = sender or {}
    message = self._peer_custom_message(text, sender)   # see §3.1
    busy = self._is_streaming
    if mode == "steer":
        if busy:
            self.steer(str(message.details["text"]))    # existing mid-turn path
            # steer() persists its own row; do NOT also append here.
            return "delivered mid-turn (steered)"
        # idle steer degrades to a driven turn
        self._spawn_background(self._prompt_messages([message]))
        return "delivered (opened a turn)"
    # mode == "mailbox"
    if wake and not busy:
        self._spawn_background(self._prompt_messages([message]))
        return "delivered and woke the session"
    # record-only (idle or busy): persist now, model reads it next turn
    await self._transcript.append_message(message)
    self._context.messages.append(message)
    await self._emit(PeerMessageDeliveredEvent(...))    # see §3.2
    return "delivered to the mailbox (will be read on the next turn)"
```

**Critical correctness point the coder must not miss:** in the record-only
branch we append to *both* the transcript and `_context.messages`. The wake/steer
branches persist through the pipeline/steer path and must **not** double-append.
Mirror how the wake path (`session.py:5594-5619`) and job-result path
(`session.py:3756-3761`) hand a `CustomMessage` to `_prompt_messages` without a
separate transcript write.

**Allow-list gotcha:** `CustomMessage` types are only converted into an LLM user
turn if they are in the allow-list at `session.py:414-430`. `PEER_MESSAGE_...`
**must be added to that tuple** or the model never sees the peer message even
though it is in the transcript. This is the exact "trap a new aside type falls
into" the comment at `session.py:419-423` warns about. Add it beside
`HUB_MESSAGE_TYPE`.

## 3. History injection + indicator

### 3.1 Transcript representation

New custom type. Put the constant next to the other message-type constants (the
harness ones live in `harness/comms.py` / `harness/wake.py`; add
`PEER_MESSAGE_MESSAGE_TYPE = "peer_message"` in the most consistent home —
recommend a new small module `local_operator/session/peer.py` or beside
`WAKE_PROMPT_MESSAGE_TYPE`; the coder picks per local convention, but it must be
importable by `session.py`, `projection.py`, `tui/app.py`, and
`subagent_view.py`).

`_peer_custom_message` builds:

```python
CustomMessage(
    custom_type=PEER_MESSAGE_MESSAGE_TYPE,
    attribution="user",
    details={
        # "text" is what the MODEL reads — wrap it so the model knows the
        # provenance, mirroring the subagent-message envelope (comms.py:1392).
        "text": (
            f"<peer-session-message "
            f"from_pid={sender.get('pid')!r} "
            f"conversation={sender.get('conversation_name','')!r} "
            f"model={sender.get('model_label','')!r}>\n"
            f"{text}\n"
            "</peer-session-message>"
        ),
        "body": text,                 # the raw body the UIs render
        "sender": sender,             # pid/session_id/conversation_name/model_label/cwd
    },
)
```

Use `append_message` (not `append_custom`): a `CustomMessage` goes through
`Transcript.append_message` (`transcript.py:327`), which is how `HUB_MESSAGE_TYPE`
and `WAKE_PROMPT_MESSAGE_TYPE` are stored, and it is what `build_llm_history`
replays. `append_custom` (`transcript.py:393`) is for host bookkeeping entries
that are *not* LLM-visible (wake schedules, checkpoints) — wrong tool here.

### 3.2 Live receipt event (TUI + phone)

Add `PeerMessageDeliveredEvent` to `harness/types.py` beside `WakeDeliveredEvent`
(`harness/types.py:997`), carrying `body: str` and `sender: dict`. Emit it in the
record-only branch (§2.4) so the owner TUI paints the indicator the instant the
message lands, even while idle. For the wake/steer branches the row appears
through the normal turn render, but still emit the receipt so the *indicator* is
consistent. (Model this on `WakeDeliveredEvent`, which fires *before* the turn
spawn — `session.py:5599-5610`.)

### 3.3 TUI rendering (owner terminal)

Two paths, both in `local_operator/tui/`:

1. **Live:** add `on_peer_message_delivered` to `OperatorApp`
   (`tui/app.py`, beside `on_wake_delivered` at `app.py:13265`). It appends a new
   `PeerMessageBlock` (a small `NoticeBlock`/`WakeBlock`-style widget in
   `tui/widgets/transcript.py`, next to `WakeBlock` at `transcript.py:1144`).
   The block header states it is a cross-session message and names the sender,
   e.g. `↔ peer message from "peer-send design" (pid 12345, anthropic/claude-opus-4)`,
   with the body below. Record its identity in a `self._live_peer_receipts` set
   keyed by message id, exactly like `_live_wake_receipts`
   (`app.py:13275`), so replay does not double-paint.

2. **Resume replay:** in `_render_resumed_history` (`app.py:2421`), add a branch
   mirroring the wake branch (`app.py:2490-2501`): when
   `getattr(message, "custom_type", None) == PEER_MESSAGE_MESSAGE_TYPE`, append a
   `PeerMessageBlock` from `details["body"]` + `details["sender"]`, skipping ids
   already in `_live_peer_receipts`.

The block should read visually as an *inbound* cross-session message, distinct
from the user's own `UserBlock` and from `WakeBlock`. Recommend a left accent
rule + a `↔`/`⇄` glyph in the header and the sender label in the muted meta
style. **This is a user-visible change → designer round required** (see §6):
capture rendered before/after SVGs per AGENTS.md "Visual validation" using the
real `OperatorApp` host (not `_PanelHost`/`_PickerHost`, which skip the CSS —
`AGENTS.md:129-132`). A shot script seeds a transcript then calls
`app.on_peer_message_delivered(...)` directly.

### 3.4 Mobile web rendering

Two edits, mirroring the existing `parent_message`/`subagent_message` support:

1. `local_operator/mobile/types.py`: add `"peer_message"` to the `EntryKind`
   Literal (`types.py:305-314`).
2. `local_operator/mobile/projection.py`: in `fold_messages_to_entries`
   (`projection.py:153`) and in the live fold, emit a `TranscriptEntry(kind=
   "peer_message", text=..., details={"sender": ...})` for a `CustomMessage`
   whose `custom_type == PEER_MESSAGE_MESSAGE_TYPE` — parallel to the
   `HUB_MESSAGE_TYPE` branch at `projection.py:172-186`. Also add
   `ProjectionFold.note_peer_message(text, *, sender)` beside `note_user_message`
   (`projection.py:534`) for the optimistic live echo used by the handles (§2.2,
   §2.3).
3. `local_operator/mobile/web/src/types.ts`: add `"peer_message"` to `EntryKind`
   (`types.ts:10-18`). Because the daemon serializes with `asdict`, the
   `sender`/`details` fields ride automatically; expose `sender` on the web
   `TranscriptEntry` type if it is not already generic `details`.
4. `local_operator/mobile/web/src/components/transcript.tsx`: add a `case
   "peer_message":` beside `parent_message` (`transcript.tsx:131-146`). Render a
   distinct inbound card: a `↔` glyph, a meta line naming the sender
   conversation/model, and the body. **This is the mobile relay surface changing
   → designer round covers it too**; capture a browser screenshot of the peer
   card in the session view.

The web build is a `pnpm`/`vite` app (`mobile/web/package.json`). The bundled
static assets are what the daemon serves; the coder must rebuild the web bundle
per the mobile build process (check `mobile/web/scripts` and how CI/publish
bundles it) so the new case ships. Flag: if the repo commits a pre-built bundle,
it must be regenerated; if it builds at package time, no artifact commit needed.
Verify which before claiming the mobile change works.

### 3.5 Peek / hub-transcript rendering (optional but cheap)

`harness/comms.py:_render_custom_step` (`comms.py:1909-1923`) renders custom
messages for `hub op='peek'`. Add a `peer_message` branch so a parent peeking a
child that received a peer message sees `peer ← <conversation>`. Small, keeps
the peek view honest. Not load-bearing; include if time permits.

## 4. Sender client + CLI

### 4.1 The sender client

Create `local_operator/mobile/peer_client.py` (a sibling of `attach_client.py`).
Do **not** extend `AttachClient` — it authenticates as `client: "attach"`
(`attach_client.py:161`), which makes the registrant treat it as a follower
terminal with attach caps and counts it against `ATTACH_MAX_CLIENTS`
(`types.py:191`). A one-shot sender should dial as a **daemon-class** connection
(the default when the auth frame omits `client` — `types.py:180-184`), send one
frame, read the ack, and close. That is the *minimal* correct client and avoids
perturbing attach accounting.

Reference the daemon's own dial (`daemon.py:256-275`): open a `1<<20`-limit
connection to `127.0.0.1:record.control_port`, write `{"key": control_key}\n`,
then write the op frame and read one reply line. Because a daemon-class
connection receives an unsolicited `welcome`/`projection` push first
(`registrant._on_connection` sends a welcome), the sender must read frames until
it sees the `{"op":"ack"|"error","req":N}` matching its request id, ignoring
intervening `projection` frames — the same req-matching `_request` does
(`attach_client.py:266-286`). Keep it tiny:

```python
async def send_peer_message(
    record: SessionRecord,
    *,
    text: str,
    mode: str,
    wake: bool,
    sender: dict[str, Any],
    deadline_s: float = 5.0,
) -> str:
    reader, writer = await asyncio.open_connection(
        "127.0.0.1", record.control_port, limit=1 << 20
    )
    try:
        writer.write(json.dumps({"key": record.control_key}).encode() + b"\n")
        await writer.drain()
        req = 1
        writer.write(json.dumps({
            "op": "peer_message", "req": req, "text": text,
            "mode": mode, "wake": wake, "sender": sender,
        }).encode() + b"\n")
        await writer.drain()
        # Read until our ack/error; skip welcome/projection pushes.
        while True:
            line = await asyncio.wait_for(reader.readline(), timeout=deadline_s)
            if not line:
                raise ConnectionError("session closed the connection before acking")
            frame = json.loads(line.decode("utf-8", "replace"))
            if frame.get("req") == req and frame.get("op") in ("ack", "error"):
                if frame["op"] == "error":
                    raise RuntimeError(str(frame.get("message", "delivery failed")))
                return str(frame.get("detail", "delivered"))
    finally:
        writer.close()
```

64KB note (§7): the sender writes one frame; a peer *message* body over ~1MB
would exceed the `1<<20` read limit on the registrant side. Enforce a sane cap
in the CLI (reject bodies > 256KB with a clear error) so a huge paste never
becomes a silently dropped oversized line.

### 4.2 Target resolution

Resolve targets against `registry.scan(config_dir())` (`registry.py:86`). A
target selector accepts, in priority order:

1. **`--pid N`** — exact pid match. Unambiguous; the record filename is the pid.
2. **`--session ID`** — exact `session_id` match.
3. **positional `TARGET`** — a **substring, case-insensitive, matched against
   `conversation_name` first, then `session_id`, then `cwd` basename**. If it
   matches exactly one *live* record, use it. If it matches several, print the
   candidates (the `sessions` table, §4.4) and exit non-zero asking the user to
   disambiguate with `--pid`. If zero, error.

Only `state == "live"` records are eligible targets (a `wedged` record's owner
is stuck and will not service the socket promptly; `stale` is dead). If the best
match is `wedged`, say so explicitly rather than hanging on a dial.

### 4.3 `lop send` subcommand

`local_operator/cli.py`, in `build_cli_parser` after the `mobile` block
(~`cli.py:397`):

```python
send_parser = subparsers.add_parser(
    "send",
    help="Send a message to another local lop session (no cmux needed)",
    parents=[parent_parser],
)
send_parser.add_argument("target", nargs="?", help="conversation-name / session-id substring")
send_parser.add_argument("--pid", type=int, help="target by exact pid")
send_parser.add_argument("--session", dest="session", help="target by exact session id")
send_parser.add_argument("message", nargs="?", help="message text; omit to read stdin")
send_parser.add_argument("--now", "--steer", dest="steer", action="store_true",
                         help="inject mid-turn (steer) instead of the default mailbox")
send_parser.add_argument("--wake", action="store_true",
                         help="if the target is idle, drive a turn now (mailbox mode only)")
```

Argument shape note: `send <target> <message>` with both positional is the
natural form (`lop send "peer-send design" "gates are green"`). Support reading
the body from stdin when `message` is omitted (`lop send mytarget < note.txt`),
which is the ergonomic path for an agent piping a longer note. `--pid`/`--session`
override the positional `target`.

Dispatch in `main()` beside the others (`cli.py:2099`): `elif args.subcommand ==
"send": return send_command(args)`.

`send_command(args)`:
1. `_apply_run_in` not needed (no cwd semantics).
2. Resolve target (§4.2) → a `SessionRecord`; on ambiguity/none, print + return 1.
3. **Self-send guard:** if `record.pid == os.getpid()` — impossible here since
   the sender is its own short-lived process, but the *meaningful* self-send is
   "the calling agent's own session pid." The CLI can't know that reliably, so
   guard on the record being the sender's parent only if we can determine it;
   otherwise allow it (a session messaging itself is harmless — it lands in its
   own history). Document that a session *can* address itself and it simply
   records a note. See §7.
4. Build `sender` dict from the environment if discoverable — otherwise from
   `os.getppid()` and best-effort registry lookup of the *caller's* own record
   (scan for a record whose pid is an ancestor). Minimum viable: `sender = {"pid":
   os.getppid()}` plus any of conversation/model the caller passes via optional
   `--from-name`/`--from-model` flags. **Simplest robust approach:** the sender
   looks up its *own* session by walking `scan()` for a record whose pid equals
   `os.getppid()` (the `lop` TUI that spawned this `lop send` subprocess is the
   parent) and copies its `conversation_name`/`model_label`/`session_id`. If not
   found (headless/exec caller), `sender` carries just the pid. This keeps the
   indicator honest without new flags.
5. `asyncio.run(send_peer_message(record, text=..., mode="steer" if args.steer
   else "mailbox", wake=args.wake, sender=sender))`.
6. Print the returned detail and exit 0; on `RuntimeError`/`ConnectionError`
   print a red error and exit 1.

### 4.4 `lop sessions` subcommand (list + resource usage)

```python
sessions_parser = subparsers.add_parser(
    "sessions",
    help="List active lop sessions and their resource usage",
    parents=[parent_parser],
)
sessions_parser.add_argument("--json", action="store_true", help="machine-readable output")
```

Dispatch: `elif args.subcommand == "sessions": return sessions_command(args)`.

`sessions_command`:
1. `records = registry.scan(config_dir())`.
2. For each, gather resource usage (below).
3. Print a table; columns:
   `STATE  PID  KIND  CONVERSATION  MODEL  CWD  RSS  FOOTPRINT  UPTIME  HEARTBEAT_AGE`
   - `STATE` = live/wedged/stale.
   - `UPTIME` from `record.started_at`; `HEARTBEAT_AGE` = now − `heartbeat_at`
     (surfaces wedged-ness numerically — this is the "verify counts add up /
     messages deliver" self-test hook).
   - `--json` dumps the same fields as a list of dicts.

**Resource usage — the RSS-vs-footprint gap.** The framing is correct: RSS
under-reports because memory is compressed/swapped on macOS. `psutil` is not
available. Portable-enough approach with graceful degradation:

- **RSS (portable, always available):** `resource`/`/proc` is not portable to
  macOS; use `ps -o rss= -p <pid>` (KB) via subprocess. Works on macOS and
  Linux. Parse to bytes. This is the always-present baseline column.
- **True footprint (macOS):** shell out to
  `top -l1 -stats pid,mem -pid <pid>` and parse the `MEM` column, which reports
  the *phys footprint* (compressed + wired + …), the number Activity Monitor
  shows and the one that actually "adds up." Do this only on `sys.platform ==
  "darwin"`. `top -l1` for a single pid is fast (~sub-second). If parsing fails,
  leave `FOOTPRINT` as `—`.
- **Linux footprint:** read `/proc/<pid>/smaps_rollup` `Pss` (proportional set
  size) if present, else fall back to RSS. Guard with a file-exists check.
- Batch where possible: one `ps -o pid,rss= -p pid1,pid2,...` for all live pids
  is one subprocess instead of N. `top -l1` without `-pid` returns all processes
  in one shot — filter by pid — which is also one subprocess. Prefer the batched
  forms; N sessions on a laptop is single digits, but one call is cleaner.

Implement resource gathering in a small helper module
`local_operator/mobile/resources.py` (`session_resource_usage(pids: list[int])
-> dict[int, ResourceUsage]`) so it is unit-testable in isolation with a fake
subprocess runner. `ResourceUsage` = `{rss_bytes: int|None, footprint_bytes:
int|None}`. **All fields optional/None** — never fail the listing because a
`ps`/`top` call failed or a pid vanished between scan and measure.

This resource helper doubles as the monitoring/self-test the user asked for:
`lop sessions` after a `lop send` shows the target still live with a plausible
footprint, and the operator can eyeball that pids/counts line up.

## 5. Guide

Path: `local_operator/guides/peer-messaging/GUIDE.md`. **No registration line is
needed** — `discovery.py:discover_guides` auto-loads every `<dir>/GUIDE.md` with
valid frontmatter (`discovery.py:29-64`). The only requirement is the
frontmatter `name` + `description` (the description is the semantic routing
signal shown to the model).

Frontmatter:

```markdown
---
name: peer-messaging
description: Send a message from one local lop session to another with `lop send`, and list active sessions and their resource usage with `lop sessions`. Use when the user asks to hand a note or instruction to a different running lop session on this machine, or to see what sessions are running and how much memory they use. Works without cmux.
---
```

Body must contain:
- **What it is:** any lop session on this machine can message any other; the
  trust boundary is the account (loopback + 0600 control socket).
- **`lop sessions`:** the columns, what STATE/HEARTBEAT_AGE mean, how to read
  RSS vs FOOTPRINT (footprint is the true number on macOS). Show a sample table.
- **`lop send`:** the two delivery modes and when to use each:
  - default (mailbox, record-only): the message lands in the target's history
    now and it reads it on its next turn; the idle agent stays idle. Use for
    non-urgent hand-offs.
  - `--wake`: also drive a turn if the target is idle. Use when the target
    should act now.
  - `--now`/`--steer`: inject mid-turn like a steer. Use to correct/redirect a
    session that is actively working.
- **Target selection:** `--pid`, `--session`, or a conversation-name substring;
  what happens on ambiguity (disambiguate with `--pid`).
- **What the human sees:** the message appears in the target's transcript marked
  `↔ peer message from …` in both the TUI and the phone.
- **Examples**, including reading a body from stdin.
- A short **limits** section: same-account only, no cmux needed, a session
  running an older lop cannot receive (clear error), and message size cap.

## 6. Test plan

### 6.1 Unit tests (`.venv/bin/python -m pytest`, TUI ones with
`env -u NO_COLOR TERM=xterm-256color`)

- `tests/unit/mobile/test_types.py`: `validate_control_frame` accepts a valid
  `peer_message` and rejects: empty text, bad `mode`, non-bool `wake`, non-dict
  `sender`. Assert `PROTOCOL_VERSION` unchanged (guards against an accidental
  bump).
- `tests/unit/mobile/test_registrant.py` (or the existing dispatch test):
  `_dispatch("peer_message", frame)` calls `handle.receive_peer_message` with
  the parsed args; a handle *without* the method yields the "cannot receive"
  error; unknown-op path unchanged.
- New `tests/unit/mobile/test_peer_client.py`: `send_peer_message` against a fake
  in-process registrant (reuse the harness that existing attach/daemon tests use
  to stand up a `Registrant` with a fake `SessionHandle`) — asserts the ack
  detail is returned, that intervening `projection` frames are skipped, and that
  an `error` frame raises.
- `tests/unit/session/test_session_peer.py`: `Session.receive_peer_message`:
  - record-only (idle, mailbox, no wake): transcript gains one
    `PEER_MESSAGE_MESSAGE_TYPE` row, `history()` includes the wrapped user text,
    session stays idle (no turn ran).
  - mailbox + wake (idle): a turn runs (assert via the fake stream fn) and the
    row is persisted exactly once (no double append).
  - steer (busy): routed through `steer`, appears in the running turn.
  - allow-list: the peer message renders into `build_llm_history` as a user
    message (guards the `session.py:414` allow-list edit).
- `tests/unit/mobile/test_projection.py`: `fold_messages_to_entries` maps a peer
  `CustomMessage` to a `TranscriptEntry(kind="peer_message")` with sender detail;
  `note_peer_message` appends the optimistic row.
- `tests/unit/test_cli.py` (arg parsing): `lop send TARGET "msg"`, `--pid`,
  `--session`, `--now`, `--wake`, and `lop sessions --json` parse to the expected
  namespace; no regression to existing subcommands.
- `tests/unit/mobile/test_resources.py`: `session_resource_usage` with a fake
  subprocess runner returns parsed RSS/footprint, and degrades to `None` on a
  failing/absent `ps`/`top`/`smaps_rollup` without raising.
- TUI: a `tests/unit/tui` test that `on_peer_message_delivered` appends a
  `PeerMessageBlock` and that `_render_resumed_history` replays a peer custom
  message without double-painting a live receipt.

### 6.2 Live two-session manual validation (this is the real evidence — a green
suite is not testing evidence per the standing instructions)

Run from the checkout with `.venv/bin/local-operator` (not `lop`):

1. Terminal A: start a TUI session, give it a name (send it a first prompt so it
   is named), leave it idle.
2. Terminal B: start a second TUI session.
3. Terminal C (or B's shell): `.venv/bin/local-operator sessions` — capture the
   table; confirm both pids appear `live` with plausible RSS **and** a larger
   FOOTPRINT (prove the footprint column is the true number, not RSS).
4. `.venv/bin/local-operator send "<A's name>" "mailbox test"` → confirm the ack
   text, then confirm the `↔ peer message` row appears in **A's TUI** while A
   stays idle (record-only). Capture A's frame.
5. `... send "<A's name>" "wake test" --wake` → confirm A drives a turn and
   answers, with the indicator row above it.
6. Start a long turn in A, then `... send "<A's name>" "steer test" --now` →
   confirm mid-turn injection.
7. Phone surface: with `lop mobile` running, open the session on the phone and
   confirm the peer card renders with the sender label (browser screenshot).
8. Error paths: `send` to a non-existent name (clear "no match"); ambiguous
   substring (prints candidates); `send --pid <dead pid>` (clear failure).
9. Resume: Ctrl-C A, `--resume` it, confirm the peer rows replay with the
   indicator and are not doubled.

Capture commands + real output for the PR testing-evidence section, including
the unauthorized-analog (older-registrant "cannot receive" path — simulate by
pointing the sender at a record and stubbing an old handle, or note it as covered
by the unit test if a live old binary is unavailable).

### 6.3 Visual evidence (designer + ux-reviewer gates)

- Designer: before/after SVG stills of the TUI transcript (idle receipt, and a
  peer row above a driven turn) via the real `OperatorApp` host, plus a phone
  browser screenshot of the peer card. Loading/empty/populated as applicable.
- ux-reviewer: this adds new commands + a new inbound message affordance and a
  new keyboard-free interaction (a message arriving unbidden in an idle session)
  → walk the real flow: send in each mode, confirm the indicator reads as
  *inbound cross-session* and not as the user's own message, confirm the idle
  session is not disrupted in record-only mode.

## 7. Risks and edge cases

- **Self-send.** A session addressing itself just records a note in its own
  history — harmless, allowed, documented. The only guard worth adding: the
  `lop send` process is a child of the sending TUI, so `record.pid ==
  os.getppid()` is a *self*-send; permit it but label the sender honestly. Do
  not try to forbid it.
- **Target mid-restart (wedged).** `scan` returns `wedged` for a live pid whose
  heartbeat is stale (`registry.py:116`). The socket may still answer, but the
  owner is stuck. Policy: refuse to dial a `wedged` target with a clear message
  ("target pid N is not responding (heartbeat 90s old); try again"), rather than
  hang. `live` only.
- **Stale pid reuse.** A `stale` record's pid is gone (`registry.py:110`); `scan`
  reaps the file. If the OS has reused the pid for an unrelated process, the
  reaped record means we never target it. Because we dial by `control_port` +
  `control_key` from the *record*, and the record is gone once reaped, there is
  no wrong-process dial risk. If a race leaves a stale record momentarily, the
  dial fails auth (wrong/absent key) or connection-refused → soft error. Safe.
- **No registrant (exec/headless).** An `exec` session
  (`SessionRecord.kind == "exec"`) may not run a registrant or may not implement
  `receive_peer_message`. Two layers cover it: `find`/`scan` shows kind, and the
  dispatch returns "this session cannot receive peer messages" for a handle
  lacking the method. The `sessions` table shows `kind`, so the user sees which
  targets are addressable. Document that only interactive (`tui`) and
  daemon-owned sessions receive.
- **Auth failure.** If the `control_key` is wrong (record torn/rotated between
  scan and dial), the registrant closes the connection on the auth frame; the
  sender surfaces "session closed the connection before acking." Soft error,
  exit 1.
- **64KB / 1MB frame cap.** The registrant reads with a `1<<20` limit
  (`daemon.py:265` on the client side; the registrant's listener uses the same
  limit — verify in `registrant._serve`/`_on_connection`). A body approaching
  1MB risks an oversized line. The CLI caps bodies at 256KB with a clear error;
  document it. This is well under the limit and covers every realistic hand-off.
- **Ordering.** Two `lop send`s to the same idle target: each is its own dial;
  the record-only appends happen in the order the registrant's single event loop
  services them (the registrant serializes on its loop). For mailbox record-only,
  both land in `_context.messages` in arrival order and the model reads both on
  the next turn — correct. For `--wake`, the first spawns a turn; the second,
  arriving while now-busy, becomes record-only (busy branch) and is read within
  that turn's context or the next — no lost message, no interleave corruption
  (the turn lock serializes). Steer mode enqueues to `_steering_queue`, which is
  already ordered.
- **Double-append.** Called out in §2.4 — the single most likely implementation
  bug. The wake/steer branches must not append to transcript/context separately;
  only the record-only branch appends directly. A unit test asserts exactly one
  transcript row per delivery.
- **Allow-list omission.** Called out in §2.4 — if `PEER_MESSAGE_MESSAGE_TYPE`
  is not added to the `session.py:414` allow-list, the human sees the row but the
  model never does. A unit test asserts `build_llm_history` includes it.
- **Mobile bundle staleness.** If the web bundle is committed pre-built, the new
  `peer_message` case ships only after a rebuild. Verify the build path before
  claiming the phone change works (§3.4).

## 8. File-by-file change list (for the coder)

Backend/protocol:
- `local_operator/mobile/types.py`: `ControlOp` += `"peer_message"`;
  `validate_control_frame` new clause; `EntryKind` += `"peer_message"`;
  `TranscriptEntry` gains no new required field (sender rides `details`). No
  `PROTOCOL_VERSION` bump (comment why).
- `local_operator/mobile/registrant.py`: `_dispatch` peer branch;
  `SessionHandle` Protocol += `receive_peer_message`.
- `local_operator/mobile/owned.py`: `OwnedSessionHandle.receive_peer_message`.
- `local_operator/mobile/tui_handle.py`: `TuiSessionHandle.receive_peer_message`.
- `local_operator/session/session.py`: `Session.receive_peer_message`;
  `_peer_custom_message`; add `PEER_MESSAGE_MESSAGE_TYPE` to the custom-type
  allow-list (`~L414`); emit `PeerMessageDeliveredEvent`.
- `local_operator/session/peer.py` (new) or beside wake constants:
  `PEER_MESSAGE_MESSAGE_TYPE`.
- `local_operator/harness/types.py`: `PeerMessageDeliveredEvent`.
- `local_operator/mobile/projection.py`: fold peer custom → entry;
  `ProjectionFold.note_peer_message`.
- `harness/comms.py`: optional peek render branch.

Sender + CLI:
- `local_operator/mobile/peer_client.py` (new): `send_peer_message`.
- `local_operator/mobile/resources.py` (new): `session_resource_usage`.
- `local_operator/cli.py`: `send`/`sessions` parsers; `send_command`,
  `sessions_command`; dispatch branches in `main`.

UI:
- `local_operator/tui/widgets/transcript.py`: `PeerMessageBlock`.
- `local_operator/tui/app.py`: `on_peer_message_delivered`; resume-replay branch;
  `_live_peer_receipts`.
- `local_operator/mobile/web/src/types.ts`: `EntryKind` += `"peer_message"`.
- `local_operator/mobile/web/src/components/transcript.tsx`: `peer_message` case.
- Rebuild the web bundle per the mobile build path.

Guide:
- `local_operator/guides/peer-messaging/GUIDE.md` (new; auto-discovered).

Tests: as enumerated in §6.1.
