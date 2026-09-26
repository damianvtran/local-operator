# Console tool

The `console` tool drives a **real interactive terminal inside the Local Operator
desktop app**: a pty running a program, with a real terminal grid, that keeps
running and keeps its output while its pane is closed. It is the agent-facing half
of the Console tab (`docs/design/ui-console-tab.md`); `guide://console` is the
playbook a model reads, and this file is the operator's view of the same thing.

There is exactly **one host and no fallback**: the app. The tool is therefore
gated with one `createIf` line on a cheap, synchronous, file-only probe of the
app's discovery record — the same shape as `build_browser_tool`:

```python
def build_console_tool(context: ToolContext | None) -> AgentTool | None:
    if not ui_console_advertisable():      # FRESH-or-STALE, file only, never a socket
        return None
```

`ui_console_advertisable()` requires two things the app publishes in
`run/ui-browser/host.json`: a live heartbeat (FRESH, or STALE-but-alive), and the
record's `console` capability bit. A running app whose console is off — disabled
in Settings, `LOCAL_OPERATOR_UI_CONSOLE_HOST=0`, or a `node-pty` load failure —
offers **no tool at all** rather than a tool whose every call refuses. The tool is
absent, not hidden: a concealed tool cannot be discouraged, only missing, and an
agent asked to test a TUI has to know the capability exists.

The console rides the app's **existing** loopback host, key and safety rules (bind
`127.0.0.1`, require `X-Bridge-Key` on every request, no CORS headers, no CDP). It
adds a method namespace, not a second endpoint.

Implementation: `local_operator/ui_console/` (the record model and the client),
the **console** section of `local_operator/tools/builtin.py` (the tool, the
parameter validation, the secret path, the result rendering), `local_operator/
tools/registry.py` (the `createIf` row), `local_operator/prompts_api.py` +
`local_operator/prompts_md/system.md` (the gated prompt note),
`local_operator/terminals.py` and `local_operator/tui/glyphs.py` (the console's
own environment marker and the glyph decision it feeds).

Tests: `tests/unit/ui_console/test_state.py`,
`tests/unit/ui_console/test_backend.py`,
`tests/unit/tools/test_console_tool.py`,
`tests/unit/test_prompts_console_flags.py`, and the end-to-end cells in
`tests/e2e/test_console_rpc.py` (a disposable peer speaking the real wire).

## Methods

The wire vocabulary is frozen by the UI half of the split (design §10.2): ten
`console_*` methods, one tool, one `method` parameter.
`list`, `status`, `read`, `screenshot` are read-tier; everything else is
`exec`-tier, because each of those writes into a pty, changes a grid, or ends a
process. The tier is recorded per call (`call_approval_tier`) and — this matters —
**it is not the protection**: the gate is one callback for both tiers and
`tool_approval_mode: auto` installs no gate at all.

| method | what it does |
|---|---|
| `list` | Surfaces in this session, including the ones the user opened (`origin`, `command`, `cwd`, grid, running/exited, last output). |
| `create` | Start a surface: `command`/`args`/`cwd`/`env`, `cols`/`rows` (default 100x30), `reveal`, `retain`. |
| `status` | `running`, `exit_code`, grid, `live`, `truncated`, `secure`, `retain`, time since last output. |
| `read` | Text: `viewport` (the visible screen) or a `scrollback` window (`start`/`count`). |
| `screenshot` | A PNG of the surface, written by the harness to a file whose path comes back in the result. |
| `input` | `text`, or a stored secret by `secret_ref`; `paste` asks for a bracketed paste. |
| `keys` | Named keys (`['ctrl+c']`, `['up']`, `['shift+tab']`, `['f5']`), in the encoder's own `+` spelling. Common synonyms are accepted and normalised before the call: `ctrl-c`, `CTRL+C`, `^c`, `shift-tab`, `esc`, `cr`, `pgup`, `page-up`. |
| `resize` | Change the grid. |
| `secure` | The user's do-not-capture span. |
| `close` | End the surface (`kill` to signal the process, `retain=false` to discard the output). |

stdout and stderr are **one stream** — that is what a pty is — and the tool says
so rather than pretending otherwise.

## Provenance (R18)

Surface handles are `con:<n>:<nonce>` and the prefix names the host: a terminal in
another window has no such handle, so "the user says they ran it in a terminal" is
answerable rather than guessable. `list` reports `origin` (`user` or `agent`) and
the surface's `session_id`, so a surface a person opened is discoverable and
readable by that session's agent, and the agent can be sure it is *this* console.
Inside the surface, the app sets `LOCAL_OPERATOR_CONSOLE_SURFACE=<handle>` and
`LOCAL_OPERATOR_CONSOLE_SESSION=<session_id>`, so a program can tell too — and
`local_operator/terminals.py`'s `is_local_operator_console` gives the Python side
the same answer, which is what lets `glyphs.py` draw Nerd Font glyphs in a console
surface (the app bundles a Nerd-patched face; the marker is a fact, not an
impersonation of ghostty, which would also flip the notification protocol for
every process in the surface).

## Secrets, sudo, and the limits

| path | allowed? | why |
|---|---|---|
| the human types it into the surface | **yes, recommended** | the app records no keystrokes, so an echo-off prompt's input never enters the byte log, the record, `read`, `status` or a screenshot |
| the agent types a literal credential via `input.text` | **forbidden by policy, not prevented by construction** | the harness cannot know a string is a credential; the description says so and the `secret_ref` alternative is easier to use |
| the agent passes `secret_ref` | **yes** | resolved from the encrypted store in the session, never returned, never echoed, and registered for redaction |

`secret_ref` in detail: the **Python tool** resolves the ref (`retrieve_secret`,
the same store `lop secret` and the `secret` tool use), puts the value in that one
call's RPC params, and registers it with `VariableStore.register_redaction`. The
model's argument is the ref, so the transcript's tool call says
`secret_ref: "SUDO_PASSWORD"`; the result says `{accepted: true, bytes: <count>}`;
and after the call the value is scrubbed from anything the session renders. The
value does travel session → app on the loopback wire for that one call — the app is
the only process that can write to the pty — over a 0600-key-authenticated socket
on the user's own machine. That disclosure is stated rather than glossed, and the
better end state (an app-side resolution path so the plaintext never leaves the
runtime) is recorded in the design's §20.2 as future work.

What is **not** claimed:

- **A program can echo a secret.** If something prints it (`echo`, `set -x`, an app
  logging its config), it is in the record and `read` returns it — as it should,
  because a person looking at that screen sees it too.
- **A screenshot is pixels**, and redaction cannot read them. The controls there
  are echo-off and the user's `secure` span.
- **`secure` is a typed refusal, not a wall.** It is enforced at the tool seam and
  the app refuses reads/screenshots while it is on; a session that ignores the
  refusal has no other lever, and the design says so.
- **Another terminal is not readable.** The tool reads the app's surfaces only.

Administrator commands are the `ask` path, not the approval tier: before running
something that needs root — or that permanently changes the user's machine — the
agent uses `ask` with the exact command and what it will change. Surfaces start as
the user's own shell (never root), so `sudo`'s prompt lands where the user can
answer it.

## Installing a program a task needs

The console is also where a missing program is installed, and **nothing installs
silently**: the agent asks first — the exact command, what it changes — and a
declined install is an answer, not an obstacle. The rule as the agent reads it,
and the per-platform paths, are in `guide://system-tools`.

What to expect, in order:

1. **The question.** `ask` names the package, what it is for, the exact command,
   roughly how big it is, and whether a password will be needed — for example
   *"I need FFmpeg to convert your video. It is not installed; I would install it
   with `brew install ffmpeg` (about 100 MB with its dependencies), removable
   later with `brew uninstall ffmpeg`. Homebrew is already set up, so no
   administrator password is needed. Shall I go ahead?"* The question is the
   authorisation for the change; the tool-call approval is not.
2. **The install, in a surface.** The package manager runs in a console surface —
   the only place that can host its progress output and its prompts. The pane can
   stay closed while it works and the surface keeps running.
3. **`sudo`, if it is needed.** The surface is the user's own shell, never root,
   so `sudo`'s prompt appears there and **the user types the password into it**.
   Keystrokes into a pty are not recorded, so a password typed at an echo-off
   prompt (as `sudo` uses) does not enter the surface's output, the transcript or
   the record. The agent never types it; if the user has stored one, the agent
   can pass it by name (`secret_ref`) and never see the value.
4. **Prompts the agent cannot answer.** macOS's Command Line Tools dialog and
   Windows's UAC consent dialog are drawn by the OS: the pty cannot see them and
   the agent cannot click them. An installer's own agreement or licence prompt is
   the user's consent too. The agent's job in all three cases is to say what is
   on the screen, hand it to the user, and wait.
5. **The receipt.** When it finishes the agent reports the package, the command
   that ran, the version now installed, and the undo line
   (`brew uninstall ffmpeg`, `sudo apt remove ffmpeg`, `winget uninstall …`) —
   and verifies by running the tool on the real task, not by finding the binary.

The full playbook is `guide://system-tools`: detect, decide, explain, `ask`, run,
verify — with the per-platform paths. Homebrew on macOS (including installing
Homebrew itself, which does need the administrator password, and can stall behind
the developer-tools dialog); the distribution's own manager on Linux, where root
is granted with `sudo` — or a static build into the user's home directory when it
is not; `winget` or `scoop` on Windows, where `scoop` needs no elevation and a
newly installed tool usually appears only in a **new** shell.

What the agent is told never to do: install anything without the `ask`, type or
pipe in a password, accept licences on the user's behalf, add package
repositories on its own initiative, make an install permanent in shell config
without asking, or install a terminal emulator to route around a machine that
has no console.

## Capture

`screenshot` returns `rendered: "displayed"` (the app photographed its own window,
cropped to the pane) or `"offscreen"` (a reconstruction from the surface's record,
with the DOM renderer pinned and a non-blank assert plus one retry), and the
result always says which. An offscreen frame is faithful to the record but it is
not a photograph of a live screen, and the field is what keeps that honest. macOS
`screencapture` is never used: it captures the frontmost window and needs exactly
the focus theft these surfaces avoid.

A build that has no offscreen capture view yet (design §17.1 splits the pane path
from the replay one) refuses a screenshot of a surface with no displayed pane with
`capture_unavailable` — a code of its own, not `console_capture_full`: that one is
the capture view being busy with another surface, which only a build WITH the view
can reach. The refusal names the condition and points at the pane or at `read`,
and it is an ordinary state of that build rather than an app fault or a version
skew, so nothing should tell the user to update.

## Refusals

Every mid-flight failure is a typed `ErrorCode`, so nothing substring-matches a
message: `unsupported_method` (update the app), `surface_unavailable`,
`surface_not_owned`, `process_exited`, `input_queue_full`, `unknown_key`,
`secure_input_active`, `invalid_grid`, `console_capture_full`,
`capture_unavailable`, `console_unavailable`, plus the pre-existing
`proto_mismatch` and the transport's
"not running"/"not answering". Absence is reported as absence: a session whose app
has quit is told the surfaces ended with it, immediately and without a socket,
rather than being left to hang.

## Where this half departs from the design document, and why

The design (`docs/design/ui-console-tab.md`) is the contract, and it is **landed
on `origin/main`** (`122c64c8`, "docs(design): the console tab …" #1336) as of the
round-1 review of this PR — so the few places this half needed the document to say
more are amended IN the document by this same PR rather than recorded here. What
follows are the places the Python half could not follow it literally, each stated
with its reason rather than silently diverged from. The PR that lands this tool
carries the same list.

1. **§19.3's `secret_ref` cell contradicts §11.3, and §11.3 wins.** §19.3 asks for
   an assertion that the value never appears "in the peer's received params as
   plaintext"; §11.3 says the value *does* travel session → app inside that one
   call's RPC params — the app is the only process that can write to a pty, and the
   better end state (an app-side resolution path) is recorded as future work in
   §20.2. The implemented model is §11.3's: the value reaches **the app**, and
   never the model's result, the arguments it emitted (the trace), a log line, or
   the transcript. The e2e cell asserts both halves, so neither claim can drift.
2. **The console's method names are NOT added to the bridge's `METHODS` tuple.**
   §10.2 names `protocol.py`'s `METHODS`; that tuple is the *extension/bridge* wire
   — `gen_ts` renders it into the extension's TypeScript union, and a test pins
   `set(METHODS) == set(COMMAND_TIMEOUTS)` with a handler behind every entry.
   Adding ten app-host methods there would advertise ten extension methods whose
   handler answers with a bare `internal`. The vocabulary is frozen by the UI half
   (row A's `protocol.ts` mirror); this half carries its own copy in
   `ui_console/backend.py` (`CONSOLE_METHODS`, `CONSOLE_TIMEOUTS`).
3. **The new `ErrorCode` values ARE added to the shared enum** (and the generated
   TS mirror regenerated, which restamps the vendored `extension/ui-vendor/*`
   copies). Without them a typed refusal from the app fails `Response` validation
   and the tool reports "unreadable answer" instead of the §15 copy. **Cross-PR
   contract:** the app half must emit exactly §10.6's names
   (`unsupported_method`, `surface_unavailable`, `surface_not_owned`,
   `process_exited`, `input_queue_full`, `unknown_key`, `secure_input_active`,
   `console_unavailable`, `invalid_grid`, `console_capture_full`,
   `capture_unavailable`).

   **The `data` key table, published so neither half has to guess one.** §10.6
   freezes the CODE names and §15 the sentences, but neither said which key inside
   `ErrorDetail.data` carries the specific — and a key chosen privately on one side
   degrades the sentence SILENTLY: the copy stays correct and generic while the
   handle, the clamp or the accepted byte count never appears, with nothing red in
   any gate. §10.6 now carries this table; it is the spelling both halves test
   against, and each row says whether the SESSION reads the value or the host only
   publishes it:

   | code | `data` keys |
   |---|---|
   | `surface_unavailable` | session reads `surface` (the handle asked for), `count` (how many exist) |
   | `surface_not_owned` | session reads `surface` |
   | `process_exited` | session reads `exit_code`, `retain` (absent means retained) |
   | `input_queue_full` | session reads `accepted` — the bytes the host TOOK, never the size of the refused payload |
   | `unknown_key` | session reads `accepted` (the names the encoder has), `key` (the one not found) |
   | `invalid_grid` | session reads `clamp: {cols, rows}`; `reason: "fixed"` for a surface that cannot be resized at all |
   | `console_unavailable` | session reads `reason` — `disabled` (the Settings toggle or the launch flag), `pty_unavailable` (the native module design §10.1 names), or the app's per-surface `spawn_failed`/`no_runtime`; an unknown value is named as it arrived |
   | `proto_mismatch` | session reads `proto` — the peer's revision, so both numbers can be named |
   | `console_capture_full` | host-facing, MADE UNREAD: the copy names no holder, and design §10.6's "names the surface whose capture holds the view" is implemented when a host emits this code — which no released host does, since it is PR B that has the capture view |
   | `capture_unavailable` | host-facing, MADE UNREAD: `rendered: null`, the app's own proof that there is no frame. The code IS the condition (§13.2), so a copy reading this key would be claiming a specific from a value that is always `null` |

   `unsupported_method` and `secure_input_active` carry no keys: the copy alone
   answers them. Two legacy spellings are accepted DEFENSIVELY and never read as
   `None` — `handle` for `surface`, and a flat `cols`/`rows` for `invalid_grid`'s
   nested `clamp` — so a host written against an earlier draft renders instead of
   reporting "the app did not say"; neither is a wire spelling any app may emit.

   **The frozen `cursor` shape, stated once.** §10.2's `console_status` and
   `console_read` results carry `cursor`, and the type is the emulator's own
   (§5.4): `{x, y}`, `x` the COLUMN and `y` the ROW, exactly as
   `@xterm/headless`'s `cursorX`/`cursorY` mean it. It is `null` in `status` when
   the app holds no grid for the surface (a restored one, §7.3). The renderer
   accepts `{row, col}` defensively — a renderer that required one spelling printed
   "row None, column None" for a healthy host that used the other — and prints an
   unrecognised shape as it arrived rather than as `None`.
4. **`console_list`'s result must be WRAPPED.** §10.2 shows an array; the shared
   `Response` envelope types `result` as a dict, so the app has to answer
   `{"surfaces": [...]}` (which the tool reads, and which the e2e peer sends).
5. **`console_status`'s idle field has ONE spelling, and this renderer's
   tolerance is defensive.** §0.4's revision-4 paragraph settles it: §10.2's table
   is the vocabulary the app implements and the only one any prose may use, so
   §11.1's `last_output_at` IS that row's `last_activity`, and the idle interval is
   the agent's own derivation rather than a returned field. Nothing is unsettled
   and there is nothing for the app half to choose. The renderer still accepts the
   legacy pair and prints what it is given — a host written before the settlement
   renders instead of reading as console-less — which is defensively accepting a
   legacy spelling, NOT a contract gap, and it invites no second wire spelling.
6. **`secret_ref` is resolved in the session and sent as `text`**, per §11.3 —
   `secret_ref` is not forwarded to the app. Two further wire params are not
   exposed as tool parameters because they earn no schema slot: `console_input`'s
   `bytes` (the model types text; `keys` covers control sequences) and
   `console_screenshot`'s `format` (one legal value today).
7. **No agent-seed change was needed for §14.6.** The four read-only roles
   (`architect`, `manager`, `reviewer`, `scout`) already omit `console` from their
   allowlists, exactly as they omit `browser`; the omission is now pinned by a
   test rather than assumed.
8. **The context-budget ceiling was raised** 28,000 → 29,950 for this tool, with
   the measured arithmetic in `scripts/bench_context_budget.py` (base 27,912 →
   head 29,904 billed: schema +1,697, `system.md` +292, inventory +4).
9. **Three seams were added to the shared `HostClient`**: `copy_for()` (so the
   console's transport-failure sentences live in the console's module rather than
   in a host-keyed table about a different capability on the same process),
   `timeout_for()` (so a console call cannot inherit the browser's 190-second
   human-prompt budget), and `unreadable_response()` (so a well-formed refusal
   whose `ErrorCode` this version does not model is answered as a NEWER app rather
   than as a broken one — the two are the same `Response.model_validate` failure,
   because `code` is typed on the shared enum, and only a client that knows its own
   vocabulary can tell them apart). All three default to the existing behaviour, so
   every current host is byte-identical.

## Not shipped by this half

The tool ships in `local-operator`; the app half is a separate split
(`local-operator-ui`): PR A is the console host (surface registry, the emulator of
record, the byte log, the `console_*` RPC methods, the record's console fields) and
PR B is the pane. **Until A lands, this tool exists only where an app publishes
`console: true` in its record** — which no shipped app does yet — so the e2e cells
in this repository drive a disposable peer that speaks the frozen wire rather than
the real app. `tests/e2e/test_console_rpc.py` names which cells those are.
