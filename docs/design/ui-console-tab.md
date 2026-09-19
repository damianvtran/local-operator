# A Console tab inside `local-operator-ui`, driven by lop agents

Status: **proposal (architect), revision 3** — the design authority for the
implementation that follows. Scope: two repos — `local-operator-ui` (the pty
host, the terminal of record, the pane, the theme, the notification) and
`local-operator` (the `console` tool, its prompt guidance, its docs and its
e2e cells). Four PRs. No `pyproject.toml` or `package.json` version bump — the
release owner handles that.

## 0. How to read this document

### 0.1 Bases, and a warning about the brief's line numbers

Everything below is re-derived from the trees named here, by reading them, not
carried forward:

| repo | ref | notes |
|---|---|---|
| `local-operator` | `e65548e994f2b573b94def0987bb927108ef7627` (`origin/main`) | read with `git -C ~/local-operator show <ref>:<path>`; the root checkout's *working tree* is the divergent thing below and is never the source of a number in this document |
| `local-operator-ui` | `db615bd46d55f9375911d2359af5b4feff27a1cd` (`origin/main`) | read with `git show <ref>:<path>` from `~/local-operator-ui`, which is checked out on `chore/release-0.25.11` at `0f76af9e1` and is **1,355 commits behind** `origin/main` (656 files differ) |

**Refs move, and this derivation has a date.** Both SHAs above are `origin/main`
as fetched on **2026-09-19**, which is when every `file:line` in this revision
was re-derived; the numbers below are true at *those* commits and are not meant
to survive the branches moving. The `local-operator` ref moved during review:
revision 1 cited `a8fe0c1d66896ff4843092a708adef9464eafed2` (the merge of PR
#1326), which is no longer `origin/main`, so a number that was right in revision
1 can be wrong here and vice versa. A reader re-checking this document later
must fetch `origin/main` and re-derive — the rule is to read the ref, never to
trust the number.

**Both checkouts on this machine are behind `origin/main`, and no implementation
branch may be cut from either.** Measured at the time of writing:
`~/local-operator`'s working tree is HEAD `0e287be0`, **295 changes** away from
`origin/main` by `git status --porcelain | wc -l` (including deleted modules —
`local_operator/classification/`, `local_operator/agent_shell.py`); and
`~/local-operator-ui` sits on a *release branch*, 1,355 commits behind its
`origin/main`. Every PR in §17.1 must branch from `origin/main` in a fresh
worktree — the same rule `AGENTS.md:593-682` states, and the reason this
document's citations are all resolved at a ref rather than from the disk.

**The recon line numbers in the design brief are stale, and not by a little.**
The brief cites `builtin.py:10775` for `build_browser_tool`; at `lop@e65548e9`
it is `local_operator/tools/builtin.py:12780`. It cites `protocol.py:342/354`
for the `Request`/`Response` envelopes; at that ref they are `:422`/`:434`. The
brief's numbers match neither ref: they match the *working tree* of
`~/local-operator`, the divergent checkout above. Every citation in this
document was therefore re-derived at the refs above, and §2 says where each one
moved. This is the same failure the browser design authority records about its
own revision 1 (`docs/design/ui-browser-tab.md:21-26`), and the same one
`AGENTS.md` names ("Read the committed ref, not the working tree",
`AGENTS.md:593-682`).

### 0.2 Honesty conventions (copied from the browser authority, and load-bearing)

Every claim is one of three kinds, and the kind is stated:

- **Verified** — read in the source tree at the ref above, cited `file:line`.
- **Measured** — a number produced by running something, with the method given.
- **Unverified** — stated as such, with the exact experiment that would settle
  it, named as a QA probe in §19. **No Electron behaviour is asserted here
  without a citation to the pinned runtime or a measurement.**
- **Not applicable** — where a requirement cannot be met as asked, this document
  says so in the requirement's own section rather than satisfying it on paper.

Three shapes of claim are deliberately absent: no "should work", no "Electron
supports this", and no line number I did not read at the ref in §0.1.

### 0.3 Requirements traceability

R1-R21 are the operator's asks. A reviewer checking this document should be able
to confirm from this table that none was dropped, and each row names where it is
satisfied and what it produces.

| # | the ask (verbatim intent) | satisfied in | the deliverable it produces |
|---|---|---|---|
| **R1** | A Console tab on any chat session, opened like the other session/sidebar tab buttons, with a console icon | **§6.1**, §14.4 | the fourth right-slot pane, its header control, its icon decision |
| **R2** | Fully functional console view: stored history, retention, per session, recalled when switching across sessions | **§6.2**, §7 | the byte log, the emulator of record, `console_read`, replay-on-reattach |
| **R3** | Multiplexed: multiple running console commands across all chat sessions at once | **§6.3** | the surface registry keyed by session, the cap, the cost arithmetic |
| **R4** | Terminal emulation "similar to ghostty"; review ghostty, verify MIT, copy the relevant functionality | **§5** | the three-way assessment (A/B/C), the licence findings, the verdict on vendoring Zig |
| **R5** | Robust stdout/stdin; nerd fonts; ANSI; 24-bit colour | **§5.4**, §9 | the emulator seam, the already-shipped Nerd Font, the colour model |
| **R6** | End-to-end testing of the feature | **§19** | the QA matrix surfaces, the rigs, the named probes |
| **R7** | Usable as a tool for agents for TUI + visual testing; a proper capture capability usable headlessly while the app is open, mirroring how agents detect the app for the browser | **§13**, §10 | `console_screenshot`/`console_read` with no view required, the detection record |
| **R8** | Differentiated from `bash` and DISCOURAGED except for TUI/interactive use | **§14** | the description's first clause, the prompt note, the subagent-allowlist enforcement |
| **R9** | Agents can open a terminal in the console tab through the live tool | **§10.4** | `console_create` with a `reveal` policy that cannot steal OS focus |
| **R10** | Validated e2e TUI usage: capture frames, send input, copilot between human and agent on one surface | **§19.4** | the co-pilot cell, the frame pair, the input path |
| **R11** | Reviewed by design and UX | **§16.3** | what the design round looks at (frames × 12 themes) and what the UX round walks |
| **R12** | Theme aware to the selected UI theme (all twelve palettes; roles only, never hex) | **§9** | `shared/themes/terminal-theme.ts`, the contrast block, the honest gap |
| **R13** | The terminal keeps running while the tab is closed — surfaces inside the session workspace; cmux as the inspiration | **§6.4**, §8.3 | the surface=pane separation, the no-resize-on-attach rule |
| **R14** | Blip on the console icon; a native clickable notification that opens the session with the surface focused; e2e exemptions | **§12** | the completion ladder, the click payload, the exemption path |
| **R15** | Implement end to end, then release once validated in a thorough QA matrix | **§18**, §19 | merge order, the matrix (§19.5), the probe list (§19.1) |
| **R16** | The tool appears in traces as `console` with a console icon; usable only when the UI is available | **§14.2-14.4** | the createIf gate, the label, the TUI glyph, the UI icon |
| **R17** | Self-describing methods: screenshot, read stdout/stderr, send input, send keys | **§10.2** | the method vocabulary, with the key set named |
| **R18** | If a user says they ran something in the console, the agent can read that surface; the surface must be distinguishable as the Local Operator console | **§6.5**, §13.4 | provenance (`owner`, `origin`), the `con:` handle prefix, the visible marker |
| **R19** | Sudo/admin commands with approval, requesting permission through `ask` | **§11.1** | the per-call approval tier, the copy standard, the shell-side affordances |
| **R20** | Password entry by agents AND by users; security is part of the design; nothing in the model context or a trace | **§11.2-11.5** | the exact-value path, the secure-input span, the redaction seam, the stated limits |
| **R21** | Keep the window-focus rule: every agent-driven launch names a window mode; headless for captures | **§16.1** | the mode table per rig, the assertion that a capture raises nothing |

### 0.4 Revision-2 changes, in one table

Revision 1 was written before the compatibility spike finished, so its D1
(emulator) decision was made from published packages rather than from runs. The
spike has since run on this machine (macOS 26.6.2 / Darwin 25.6.0, arm64, Electron
44.3.0 → node 24.20.0, ABI 149, N-API 10), and revision 2 folds it in. **Only D1
moved; every other decision, section and citation stands unless the table says
otherwise.**

**Revision 3 — this commit — re-derived every `file:line` at the refs now named
in §0.1 and answers the agent-review round 1. No decision moved.** Revision 2's
table below stands as written; what changed is the ref basis (the
`local-operator` ref moved, so every number whose file drifted moved with it),
the review's one-paragraph and one-line corrections (§10.6 gains
`console_capture_full`; §19.1's lead sentence now matches its own MEASURED rows;
§6.1 fixes the user-initiated create path and its `reveal` default; §10.4 states
it per actor; §12.3 and §11.4 define `exit_epoch` and qualify what the
secure-input span drops), and the dangling cross-references.

**Revision 4 — this commit — answers the agent-review round 2, and no decision
moved: four `file:line` corrections, the fifth citation the same sweep caught,
and one vocabulary fix.** The four (§14.6's `_reconcile_web_tools`, §6.6's
CSP line, §9.1's grounds line, §9.3's contrast sentence) now resolve to the line
that carries the thing they name, and Appendix A's `session.py` census follows
the first of them from `:14001` to `:14012`; §2.10's `TOOL_ICONS` citation sat
one line below the `const` it names and now reads `:54-72`. `exit_epoch` has one
owner and one spelling: §7.3's sidecar record, beside `exit_code`, is what keeps
the counter, §10.2's `console_status` row is what returns it, and §12.3 is what
keys its dedupe on it. §10.2's table is the vocabulary PR A implements, so it is
also the only spelling any prose may use — §11.1's `last_output_at` was the
table's `last_activity`, and the idle interval is the agent's own derivation
rather than a returned field. §0.5 and §17.1 are unchanged between revision 3
and this one.

**Revision 5 — this commit — shapes two things the table left to the reader,
and no decision moved.** §10.2's `console_status` and `console_read` rows name
`cursor` without saying what it is, while the only cursor type the document
defines is §5.4's emulator cursor; the rows now carry that spelling and §10.2
states it in one paragraph, because a wire field the frozen table does not shape
is a field two implementations can disagree about while both stay conformant.
§10.6 gains the `data` keys per code for the same reason one level down: its copy
rules promise the specifics (“the clamp it applied”, “the accepted byte count”)
without saying which key carries them, and a privately chosen spelling degrades
exactly the diagnosis §15 promises while every gate stays green. Both amendments
come from the agent-review round 1 on the tool half (`local-operator` #1338) and
are the contract its app half implements.

| area | revision 1 | revision 2 |
|---|---|---|
| D1 emulator | xterm.js, ghostty assessed from licences and packaging | **unchanged decision, now evidence-backed** — and the ghostty path is recorded as *measured-viable*, with its sha256, its minisign verification and a named switch trigger (§5.5) |
| the capture path | `capturePage()` offscreen; no renderer chosen | **the capture view pins the DOM renderer**, `canvas.toDataURL()` is forbidden on the WebGL canvas, and every capture asserts a non-blank frame and retries once (§13.2) — three measured traps |
| the pty | N-API prebuilds "expected" to work | **measured**: the same `pty.node` loads under ABI 147 and 149, with spawn/echo/24-bit round-trip/resize/exit verified (§5.3) |
| packaging | `asarUnpack` "sufficient" | **two measured traps**: the shipped `spawn-helper` is mode 0644 and nothing chmods it, and a packed asar fails to spawn until the prebuilds are actually unpacked (§5.3, §18 risk 1) |
| fonts (R5) | the app bundles a Nerd-patched face, so glyphs are available | **the pane is fine; the machine is not** — no Nerd Font is installed here at all, the app ships the face with **no font licence notice**, and the box-drawing join was measured as broken at that size (§6.6) |
| probes | P1-P12, all unrun | **P1 and the capture half of P4 are measured**; P3 stays open; two new probes (P13 packaging, P14 VT conformance against ghostty's own VT) (§19.1) |

### 0.5 Fixed interface decisions, and this revision's verdict on each

| fixed decision | verdict | where argued |
|---|---|---|
| exactly one emulator **of record**, in main; the pane is a mirror | **new, and the load-bearing one** — kept in revision 2, re-decided against the spike's measurements | §3, §4, §5.5 |
| the pty lives in the app's main process (`node-pty`) | **new** | §3 |
| one loopback host, one 0600 record, a `console_*` method namespace — not a second endpoint | **new** | §10.1 |
| the console tool is `createIf`-gated on the app, and is **not** hidden | **new** | §14.2 |
| the right slot gains a **fourth** pane | **new, following the third's own precedent** | §6.1 |
| capture is text from the record and pixels from a view fed by the record | **new**; revised in revision 2 for three measured traps (DOM renderer pinned, no canvas readback, assert-and-retry) | §13.2 |
| no detached (surviving-logout) surfaces in v1 | **new, with the cost stated** | §7.4 |
| cmux's *model* is adopted; its **code is not** (GPL-3.0-or-later) | **new** | §2.12, §4 |
| a new native dependency is a packaging decision, not an install | **new** | §5.3, §17 |

---

## 1. Problem and scope

### 1.1 The request

Add a **Console pane to the `local-operator-ui` desktop app** that hosts a real
pty, gives the user a fully functional terminal view with per-session history,
keeps surfaces running while the pane is closed, multiplexes surfaces across
every chat session, and exposes those surfaces to lop agents through a new
`console` tool with screenshot/read/input/keys methods — so that a TUI can be
driven and inspected end to end by an agent while the user watches, or without
them watching at all.

R1-R21 are the operator's asks (§0.3). The one that shapes the architecture
rather than decorating it is **R7**: an agent must be able to capture what the
terminal renders **while the app is open but the pane is not**. That single
requirement decides where the terminal state lives (§3), and every other
decision follows from it.

### 1.2 What is actually missing

Read at the refs of §0.1:

1. **There is no pty anywhere in either product.** In `local-operator`, `import
   pty` appears only in tests and one script
   (`tests/unit/secrets/test_cli.py:20`, `tests/unit/providers/test_login_cancel_cli.py:256`,
   `tests/unit/test_e2e_death_witnesses.py:32`,
   `tests/e2e/test_terminal_close_survives_e2e.py:61`,
   `tests/e2e/test_tui_boot_e2e.py:94`, `scripts/word_caret_pty.py:11`).
   `local_operator/terminals.py` is emulator **detection** from environment
   markers; `local_operator/ansi.py` strips control sequences; there is no
   terminal emulator in the product. In `local-operator-ui` there is no pty, no
   VT parser and no native dependency at all: the runtime `dependencies` map is
   six pure-JS packages (`package.json`).
2. **Today's "TUI emulation" is Textual's headless driver.** `tests/e2e/conftest.py:1-14`
   documents `App.run_test()` on a `HeadlessDriver` — "allocates no terminal,
   opens no window and never touches a display server" — with SVG export through
   `scripts/visual_capture.py:259 save_capture`. That is a *pytest fixture*, not
   a product capability: it needs the Python tree, needs the developer's
   terminal-less test run, and produces a Rich SVG rather than a real terminal
   grid.
3. **The one real pty in the tree is a test.** `tests/e2e/test_terminal_close_survives_e2e.py:166 _launch_tui`
   forks a pty around the product's own console script to prove the *runtime*
   survives the interface dying. It is the closest existing art and its
   constraints (drain the master or wedge the child; strip `CMUX_*`/`LOP_*`/
   `ELECTRON_*`-class variables; pin `LOP_SESSION_GRACE_S`) are inherited by
   §19's cells.
4. **There is no agent-callable capture tool.** The only agent-facing
   screenshot path is the `browser` tool's, which drives a *page*; it cannot
   enter a renderer (`docs/agent-driver.md:1-10` records exactly this:
   "An agent could read this app's JSX and could not see it run").
5. **The app cannot be captured from outside.** Python's own tests neutralize
   `local-operator-ui` on `PATH` (`tests/conftest.py`), so UI verification is a
   UI-repo rig (`scripts/renderer-driver.mjs`, `docs/agent-driver.md`) — a
   *developer-facing* harness, not a tool an agent in an arbitrary session can
   call.
6. **Nothing in either repo is terminal-shaped on the UI side either:** no
   `xterm`, no `node-pty`, no `libghostty` string appears anywhere under
   `local-operator-ui/src`, `scripts/` or its manifests at `ui@db615bd46`
   (`git grep -l -i -E 'xterm|node-pty|libghostty|ghostty-vt' origin/main` →
   only unrelated files: `scripts/fixtures/*.json`, `scripts/verify-*.mjs`,
   `src/main/backend/managed-python.ts`, all matching "xterm" inside bundled
   Python docs/strings).

### 1.3 Non-goals (v1), each with the reason

- **No detached surfaces.** A surface's process is a child of the app; quitting
  the app kills it. Surviving logout requires a supervisor process outside the
  app, a re-attach protocol, and a second lifecycle owner — §7.4 states what the
  user gets instead and what a later version would need.
- **No tmux-style panes, splits or window management.** One surface is one pty
  plus one grid. Splits are a multiplexer's job and cmux already exists; this is
  the app's own terminal, not a replacement for the user's.
- **No serial/SSH transport.** A surface runs an argv on this machine.
- **No shell integration installer.** §12.3 *detects* OSC 133 marks; it does
  not write to the user's dotfiles.
- **No Sixel/graphics protocols and no inline images in v1.** `@xterm/addon-image`
  exists and is MIT, and it is deliberately out of scope: it would put a second
  raster path beside `capturePage` and a second contrast surface beside the
  text grid.
- **No terminal-in-the-transcript.** The console is a pane in the window's right
  slot, not a row in the conversation. R10's co-pilot story is the *tool trace*
  naming the surface, not the terminal's pixels entering the transcript.
- **No GPU-parity claim.** "Similar to ghostty" is satisfied as *VT
  correctness*, not as *ghostty's renderer* — §5 argues why, and the three
  options are assessed rather than waved through.

---

## 2. What exists today (file:line evidence, all re-derived)

### 2.1 The `browser` tool is the precedent, and its gate is one line

- `local_operator/tools/registry.py:32 TOOL_BUILDERS` maps name → `createIf`
  factory; `browser` enters at `:60`; `DEFAULT_TOOL_NAMES` (`:70-97`) is an
  explicit list, "kept explicit (not `list(TOOL_BUILDERS)`) so the default
  surface is a deliberate decision" (`:66-69`).
- `create_tools(context, enabled=None)` (`registry.py:100-130`) resolves
  availability **at construction time, never at dispatch time** (`:107-108`),
  skips unknown names rather than crashing session startup (`:117-118`), and
  injects the `i` intent property at one choke point (`:121-128`) because the
  tool array rides in the prompt cache prefix.
- `local_operator/tools/builtin.py:12780 build_browser_tool` returns `None`
  unless a browsable host is reachable — one line, three probes:
  `if not (cmux_browser_available() or bridge_browser_advertisable() or ui_browser_advertisable()): return None`
  (`builtin.py:12822`), with the comment at `:12819-12822` naming the rule this
  document must not break: "Three checks on ONE `createIf` entry, rather than a
  second gating convention beside it".
- The returned `AgentTool` (`builtin.py:12825-12879`) sets `approval_tier="write"`
  with a **per-call** `call_approval_tier` that escalates `upload` to `exec`
  (`:12873-12875`), `concurrency="shared"`, `interruptible=False`, and a
  `description` whose first sentence says what the surface *is* — measured: a
  session that needed before/after screenshots wrote a playwright script and
  spent 23 s on `playwright install chromium` while the tool sat in its inventory
  unused, and then, told outright to use the cmux browser, still shelled the CLI
  through `bash` (`:12793-12802`).
- **The description is now deliberately SHORT, and that is a binding constraint
  on §14.3.** The `Footprint:` comment at `:12830-12837` says the per-action
  detail "lives in `guide://browser` … and in the parameter descriptions, and this
  string carries only what a model needs to CHOOSE the tool and call it
  correctly. The context-budget guard measures the whole surface and it is why
  the download/upload sentence is one clause rather than the four the design
  first drafted" — the guard being `scripts/bench_context_budget.py`, which CI
  runs for the `context-budget` scope (`scripts/ci_scope.py:178-182`).
- **The approval tier is not the protection, and the code says so.** At
  `:12864-12872`: "today the gate is ONE callback for both tiers and
  `tool_approval_mode: auto` / `--yolo` installs no gate at all, so the tier
  records intent and future-proofs a tier-sensitive host — it is NOT the
  protection." §14.6 is written against that fact rather than the convenient one.
- `local_operator/harness/types.py:1242 AgentTool` carries `name` (`:1258`),
  `approval_tier` (`:1262`), `concurrency` (`:1263`) and `resource_keys`
  (`:1266-1268`), `interruptible` (`:1269`), **`hidden`** (`:1270`), `execute`
  (`:1271`), `describe_approval` (`:1272`) and the per-call `call_approval_tier`
  hook (`:1277-1279`).
- `hidden` means "not listed in the prompt inventory", not "not callable":
  `local_operator/prompts_api.py:365` renders `- {name}` only for
  `not tool.hidden`, and the one shipped user of it is the structured-reply
  transport (`local_operator/harness/reply_channel.py:143`), "Kept out of any
  surface that lists callable tools to a user: this is transport, not a
  capability someone can invoke". **R16 requires the opposite** (§14.2).

### 2.2 Detection: a 0600 record, three states, one socket probe

- `local_operator/ui_browser/state.py:34-35` — `RUN_DIRNAME = "run/ui-browser"`,
  `STATE_FILENAME = "host.json"`; `:47 HOST = "ui"`; `:53-78 UiHostState`
  (`pid`, `port`, `session_key: str = Field(min_length=32)`, `proto`, `host`,
  `app_version`, `profile_dir`, `tabs`, `agent_tabs`), extending the bridge's
  `HeartbeatState` for `heartbeat_at`/`started_at`.
- `:87-89 state_path` is "Pure path arithmetic: creates NOTHING"; `:92-95 publish`
  is the test-side writer; `:103-105 read` uses the UI host's own model "so no
  field is dropped". The module deliberately has **no** `run_dir`/`remove`
  wrappers (`:108-112`).
- `:120-136 liveness` → `ABSENT` (no file, or dead pid) / `FRESH` (heartbeat
  inside `HEARTBEAT_TIMEOUT_S`) / `STALE`; `:145-146 available` is FRESH only;
  `:150-157 advertisable` is FRESH-or-STALE, because "advertising only promises
  the agent can ASK". The heartbeat constants are shared with the bridge
  (`:37-41`) rather than re-declared.
- `local_operator/ui_browser/backend.py:34 UiHostClient` is
  `browser_bridge.backend.HostClient` pointed at this record — "That inheritance
  is the whole design of this module" (`:3-8`); `:43-53 ui_browser_available`,
  `:56-67 ui_browser_advertisable`, `:70-77 ui_liveness` all never raise;
  `:80-100 ui_browser_reachable` buys exactly one bounded `/health` probe when
  the file says STALE.
- `:103-129 _health_ok` requires HTTP 200 **and** `body["host"] == "ui"` **and**
  `int(body["pid"]) == pid` — the pid check is the UI host's equivalent of the
  bridge's `extension_connected` requirement — and deliberately does **not**
  require `proto` to match: "a version-skewed host is a REAL host that can
  explain itself, so it stays reachable and the per-action path returns the
  typed `proto_mismatch`" (`:112-115`).

### 2.3 The wire: one POST, a closed method list, a version *window*

- `local_operator/browser_bridge/protocol.py` is "the protocol's only
  hand-written source" (`:1-7`), with a TypeScript mirror generated by
  `.gen_ts` so a Python-only edit fails CI.
- `:16 PROTO_VERSION = 1`, `:45 MIN_SUPPORTED_PROTO = 1`, and the change rules
  at `:20-45`, quoted because the console's method additions are governed by
  them: raising the floor is allowed only by a commit that *changes the meaning*
  of an existing frame or method, and such a commit **must** raise it in the same
  commit; "Additive optional fields, new `ErrorCode`s the peer only emits, and
  new events an old peer harmlessly drops are what keep the floor where it is".
- `:200 METHODS` (the closed list), `:288 COMMAND_TIMEOUTS`, `:349 ErrorCode`,
  `:422 Request`, `:434 Response`.
- `local_operator/browser_bridge/backend.py:718 HostClient` — "one host's
  authenticated loopback leg, over that host's discovery file… A subclass rather
  than a parameter so the extension's call sites keep their `BridgeClient()`
  spelling" (`:727-730`); `:679 client_timeout` sizes the HTTP budget as
  `base + ORIGIN_PROMPT_WINDOW_S + margin` so a human-prompted wait inside the
  host never reads as "unreachable" (`:531-553` at the older ref, same text);
  `:829 BridgeClient`.
- The failure taxonomy is already typed: `401` → `rejected_key`, a connect
  timeout → "not answering", a **read** timeout → "accepted but did not answer"
  with the likely human-prompt cause named, an unparseable body →
  `invalid_response` (all in `HostClient.call`).

### 2.4 The UI host: four safety rules, a closed dispatch, a 0600 writer

`local-operator-ui` at `ui@db615bd46`:

- `src/main/browser/rpc.ts:29-46` states the four rules that make the endpoint
  safe, and they are enforced, not documented: bind `127.0.0.1` only
  (`:68 LOOPBACK_HOST`), require `X-Bridge-Key` on **every** request with a
  constant-time compare (`:58 KEY_HEADER`, `keysMatch` from `state-file.ts`),
  send **no CORS headers** (the app renderer runs `webSecurity: true`, so a
  page inside the app cannot reach the port), and expose no CDP. `:50 RPC_PATH`,
  `:53 HEALTH_PATH`, `:62 MAX_BODY_BYTES = 1 << 20`.
- `src/main/browser/state-file.ts:31 RUN_DIRNAME`, `:32 STATE_FILENAME`,
  `:36-37 STATE_DIR_MODE 0o700 / STATE_FILE_MODE 0o600`, `:42-43` heartbeat
  15 s / timeout 45 s mirroring the Python side, `:48-60` the record shape.
  "`session_key` in a readable file IS the authorization — anything able to read
  it already owns the agent's browser" (`:24-27`).
- `src/main/browser/protocol.ts:53-78 METHODS` is a **closed** list and `:81
  isMethod` refuses anything else; `COMMAND_TIMEOUTS_S` mirrors
  `protocol.py`'s numbers "because inventing a second constant beside the one
  the Python side already publishes is how the two drift" (`:90-93`).
- `src/main/browser/host.ts:383` the dispatch switch; `:369` at the older ref,
  same shape.
- `src/main/browser/ipc.ts` — the per-feature renderer namespace, with its three
  rules (`:20-35`): every handler authorizes the sender; **this namespace is NOT
  routed through `desktop-request`** because "that vocabulary maps to backend
  HTTP paths"; and the renderer never receives a surface nonce. The channel
  list is one exported literal, "so a test can enumerate them" (`:56-78`).
- `src/main/browser/registry.ts` — the tab registry. `:116 MAX_AGENT_TABS = 8`;
  **`:120-125 BACKGROUND_VIEWPORT` (1280x720) with the rationale that matters
  most to this document**: "Background rendering must not depend on a foreground
  route's measurement. The fleet cap and bounded viewport bound raster memory
  without activating views." `:527 setContentRect`, `:546-557 applyLayout`
  (a non-active or invisible view takes `BACKGROUND_VIEWPORT` and
  `setVisible(false)`), `:609 forget`/`:664 destroy`.
- `src/main/browser/index.ts:182 startBrowserHost`; `:175-180 browserHostEnabled`
  reading `LOCAL_OPERATOR_UI_BROWSER_HOST` — on by default, "because the state
  file is how a session DISCOVERS the host", and present so an isolated run "can
  assert 'no host, no state file'". A hidden `WebContentsView` is also already
  used as a CDP target for cookie work and is "never attached to a window, so it
  is never laid out, painted or focused".
- `src/main/browser/vendor/` + `PROVENANCE.json` + `scripts/check-vendored.mjs`
  (read-only gate: hashes every listed file, refuses extras, pins the manifest's
  proto to the number the app's wire code imports) + `scripts/sync-vendored.mjs`
  — the cross-repo sharing mechanism, and its stated blind spot: "drift AGAINST
  lop… there is no lop checkout here".

### 2.5 The right slot already has three panes, and the third one is a browser

This is the single most useful piece of ground truth in the UI repo, because it
means R1 is *not* a new concept: it is the fourth occupant of a slot that has
already absorbed a third.

- `src/renderer/src/shared/store/ui-preferences-store.ts:380-390 claimRightSlot`
  is typed `pane: "isRunPanelOpen" | "isCanvasOpen" | "isBrowserPaneOpen"`, and
  the comment above it (`:370-379`) says it outright: "**The third pane arrived
  exactly as that comment predicted, and this is the whole of what it cost: one
  more name in the union and one more `===` below.** The browser pane
  (`docs/design/browser-approval-ux.md` 7.3) is the window's, not the
  conversation's, so it belongs in this rule rather than beside it."
- The three pane preferences are global window state, not per-conversation:
  `isCanvasOpen` (`:51`), `isRunPanelOpen` (`:72`), `isBrowserPaneOpen`
  (`:92-99`), and each is documented as surviving a conversation switch "because
  the pane is a property of the window's slot rather than of one conversation".
  Widths: `DEFAULT_CANVAS_WIDTH = 800` (`:413`),
  `DEFAULT_RUN_PANEL_WIDTH = 420` (`:422`, exported), `DEFAULT_BROWSER_PANEL_WIDTH = 640`
  (`:434`) with per-pane clamps set by each divider.
- `src/renderer/src/features/chat/components/chat-content.tsx:1206-1215` (canvas),
  `:1285-1300` (run panel), `:1383-1392` (browser pane) are the three
  `{isXOpen && (<> <ResizableDivider …/> <div …> … </div> </>)}` blocks, each
  with its own `label` for the divider ("Resize canvas" / "Resize run details" /
  "Resize browser") and its own `minWidth`.
- `src/renderer/src/features/chat/components/chat-header.tsx`: the action cluster
  is documented at `:289-303` as "THESE ARE THE RIGHT PANE'S THREE CHOICES, and
  they are mutually exclusive in the STORE rather than here: each setter clears
  the other two", with a measured note about the 12px the badge earns and the
  `gap-3`/`gap-2` switch that pays for it (`:303-338`, the switch at `:335`), and a focus-return effect
  when the pane closes (`browserPaneWasOpen`, `:199-207`).
- `src/renderer/src/features/browser/components/browser-pane.tsx` is the pane
  component to copy: `:86` the FC, `:117` `data-tour-tag="browser-pane"`, a
  **40px `bg-sunken` header** matching the slot's other two panes
  (`canvas/index.tsx:542`, `run-details/run-panel.tsx:688-695`), `bg-surface`
  as the pane ground "so the three occupants of this slot read as one slot with
  three modes", a close button, and a `*.stories.tsx` beside it with the
  store-driven open state.
- **The session panel remounts on switch, so nothing terminal-shaped may live in
  React state.** `chat-page.tsx:2231` renders `<SessionPanel key={identity} …>`;
  `browser-pane.tsx`'s own header comment records the defect this causes
  ("the pane is remounted when the conversation changes, so a `useState` here
  reverted the lens… while the pane itself stayed open at the width the user had
  dragged"), and the fix is that the *choice* lives in the store and only the
  content follows the session.

### 2.6 Three renderer↔main transports, and why a pty needs a fourth noun

- `desktop-request`: a zod discriminated union of ops mapped to backend HTTP
  paths (`src/shared/desktop-contract.ts:678` the union, `:2617 desktopEndpoint`),
  bounded at `MAX_DESKTOP_REQUEST_BYTES` (`:2173`) and documented as
  "value-sized" — it is a request/response channel to the *backend*, not a
  stream, and the browser's namespace exists precisely because it is not.
- The session event stream: `src/main/desktop-ipc.ts:297` subscribe /
  `:321` unsubscribe, relayed to the owned window only, with the renderer never
  seeing the bearer; `src/preload/index.ts:108-124` is the renderer half of
  `desktop-open-conversation` (its explicit `null` handling for a digest banner)
  and `:163-182` of `desktop-stream-event`.
- The per-feature namespace: `src/main/browser/ipc.ts` (§2.4).
- **A pty is a high-rate byte stream with a request/response control plane on
  top**, which is none of the three: `desktop-request` is the wrong size class
  and the wrong direction (it faces the backend), the event stream is per
  *session* rather than per *surface* and carries session semantics (frames,
  epochs, `seq`), and the browser namespace's ops are the right shape but its
  payloads are snapshots. §10.3 puts the console's control plane on the
  browser's namespace and its data plane on a new, surface-keyed push channel.

### 2.7 The app already ships a Nerd Font, and `--font-mono` is the second source of truth

- `src/renderer/src/assets/fonts/fonts.css:33-45` declares `font-family: "Geist Mono"`
  from **`GeistMonoNerdFontMono-Regular.otf`** and `-Bold.otf` (the file's own
  comment: "Geist Mono ships as a Nerd Font patch, so each weight is a 2.1 MB
  OTF. Only the two weights the app actually asks for are declared"), with
  `font-display: swap` and a recorded incident about what a blocking font did to
  the composer's session strip.
- `src/renderer/src/styles/index.css:130` — `--font-mono: "Geist Mono", ui-monospace, SFMono-Regular, Menlo, monospace;`
  with the comment above it (`:117-121`) explaining why the stack is bundled
  rather than named ("CSP is `font-src 'self' data:`, so a remote face cannot
  load — naming one is not a graceful degradation, it is a silent
  substitution").
- **Consequence for R5:** Nerd Font glyphs need no new asset and no new licence
  review; what they need is (a) the terminal rendering through `--font-mono` and
  (b) the *harness* telling the TUI it may draw them, because the Python side
  gates on terminal detection it cannot perform from inside the app (§9.5).

### 2.8 Notifications: the ladder, the click, and the kill switch already exist

- `src/main/desktop-notifier.ts:304 DesktopNotifier` is the app's one banner
  raiser. `:267-302 DesktopNotifierHost` is "what a click on a banner needs from
  the app that owns the windows" (`reopen(sessionId)`), with a `SILENT_HOST`
  default (`:205`) that "no click to serve". `:395 canNotify` is
  `Notification.isSupported()`. `:1155 private show(sessionId, title, status, body, isSnippet, isFailure)`
  is the single choke point every banner goes through — `legacyTurn` (`:707`),
  `composed` (`:721`), the gate path (`:1033`), `turn` (`:1086-1098`).
- The click handler (`:1174-1215`) **sends before it raises**, queues through
  `host.reopen(sessionId)` when there is no window, and files its raise under a
  `trigger` label (`:1210`). The whole path is a reviewed fix for a click that
  used to do nothing at all, and its constraint is stated in the code: only
  `normal` may activate the app; an `inactive` window is ordered without taking
  focus and a `headless` one is never shown (`:1195-1208`).
- The backend-composed contract is `docs/DESKTOP_API.md`'s notification frame:
  emitted iff a newly published unseen `completions` row is observed, live-only
  and never replayed, one banner per `completion_token` claimed through
  `POST .../{id}/notified`, and the eligibility ladder (`:1314-1342`) whose rung
  1 is "a surface is WATCHING the session — the predicate is the VISIBILITY
  one… NEVER `notification_surfaces()`".
- The click ladder is Python-side and desktop-first (`docs/DESKTOP_API.md:1344-1403`,
  `lop resume-click`), which is why R14's click can ride an existing path rather
  than a new one.
- **The e2e kill switch exists and is already wired:** `LOCAL_OPERATOR_NO_NOTIFICATIONS`
  is declared once (`src/main/backend/notification-launch.ts:75 NOTIFICATIONS_ENV`,
  "`.env: LOCAL_OPERATOR_NO_NOTIFICATIONS=0 -> backend child sees "0"`"), consumed
  by `scripts/notifications-off.mjs` (`:19`, `:88`) which wraps children, by
  `package.json`'s `app:headless`/`dev:headless` scripts, and by the Python side
  (`local_operator/tui/notify.py:115 _ENV_DISABLE`, `:404`). `pnpm test:desktop`
  injects it through `scripts/run-desktop-tests.mjs`. §12.4 reuses it rather
  than inventing a second switch.
- **`headless` suppresses native banners entirely** (`AGENTS.md:758-762`: "the
  notifier's delivery gate… a banner's own click handler is a path that raises a
  window"), and `headless` is the wrong mode for anything about read receipts or
  the notifier's focus gate (`AGENTS.md:749-754`). §19.2 turns that into a cell.

### 2.9 Themes: twelve palettes, roles with floors, and a scope claim to respect

- Twelve themes from eleven palette files plus the two brand entries
  (`src/renderer/src/shared/themes/index.ts:4-17`, `themes` at `:202`,
  `DEFAULT_THEME` at `:217`).
- `palette-contract.ts` owns the role set and the reason roles exist ("a palette
  that omits one no longer compiles"): grounds `canvas`/`surface`/`elevated`/`sunken`
  (`:186-211`), ink `ink`/`inkMuted`/`inkDim`/`inkDisabled` (`:429-453`, with
  `inkDisabled` the single exemption from the floors), lines `hairline`/`borderControl`
  (`:477`, `:494`), the accent ramp (`:520-522`), `chartBarHover` (`:719`), and
  the semantic triples `success`/`warning`/`danger`/`info` each with a `Wash` and
  a `Border` (`:776-787`), `overlayShadow`/`scrim` (`:796-798`).
- `scripts/contrast-contract.mjs`: `FLOOR = { strongText: 7.0, text: 4.5, nonText: 3.0 }`
  (`:69`), component **triples** not pairs (`:30-38`), and — the part that
  governs a terminal grid — the scope claim at `:40-56`: "It cannot see… a colour
  a **third-party widget** (ag-grid, CodeMirror, mermaid) picked for itself.
  Those need a human and a screenshot. Do not read a clean run as 'the app is
  accessible'." A terminal's 16 ANSI colours are exactly that territory unless
  the app derives them (§9).
- The precedent for deriving widget colours from roles is
  `shared/themes/code-mirror-theme.ts`. Its header records three decisions §9
  must not re-learn: `accent` appears in **no** syntax role because it measures
  ΔE00 **0.00** against `success` on monokai, against `info` on dune and against
  `ink` on obsidian; the four gated semantics carry keyword/string/number/function
  by **exhaustive search over 24 permutations maximising the worst pairwise
  ΔE00 across all twelve palettes (worst case 8.87, radient)**; and
  `inkDisabled` was measured and **rejected** for comments (fails 4.5:1 on the
  editor ground in all twelve, down to 1.91:1 on iceberg). Background/foreground
  ride `--color-sunken`/`--color-ink` through CSS variables, and the editor's own
  `#212121` focus ring is called out as something the app's gate cannot see.
- `pnpm check-themes` = `generate-theme-css.mjs --check` + `contrast-contract.mjs`
  (`package.json` scripts), and `AGENTS.md:61-65` states the rule for adding a
  component: "**Adding a component with its own fill and border means adding a
  row to `CONTROLS` in `scripts/contrast-contract.mjs`**".

### 2.10 Tool traces: one label table in the TUI, one icon table in the UI

- Python: `local_operator/tui/glyphs.py:82-112 NERD_TOOL_ICONS` (glyph per tool
  name, taken **exclusively from the Font Awesome block U+F000-U+F2E0**, "the
  one region present in BOTH Nerd Fonts v2 and v3"), `:116 NERD_ICON_MCP`,
  `:118 NERD_ICON_DEFAULT` (wrench), `:123-149 PLAIN_TOOL_ICONS` (the ASCII
  fallback: `bash` is `$`, "the shell prompt sigil"). `bash: "\uf120"` is
  `nf-fa-terminal`, i.e. **the terminal glyph is already taken**.
- `nerd_icons_enabled()` (`:228-241`) is TRI-STATE — env kill switch, then an
  explicit `display.nerd_icons` bool, then `_nerd_capable_terminal()` — because a
  renderer cannot be asked, only inferred, and "Every Nerd Font codepoint"
  detection cannot detect a *missing* glyph (`:168`).
- UI: `src/renderer/src/features/chat/components/trace/tool-glyphs.ts:54-72 TOOL_ICONS`
  maps the same names to lucide icons "of the same meaning" (`bash: Terminal`,
  `browser: Globe`, `web_search: Globe`, `peer: Inbox`), with two distinct
  fallbacks — unknown → `Wrench`, `mcp__*` → `Plug` — and case-insensitive
  lookup because "a tool name is MODEL-controlled". `SuccessGlyph`/`ErrorGlyph`/
  `InterruptedGlyph` must be distinguishable "with no colour at all" (`:79-96`).

### 2.11 Packaging, signing and the dependency allowlist (the cost of a native dep)

Read at `ui@db615bd46`:

- `package.json` `build`: `asar: true`; `files: ["out/**/*", "bin/**/*", …]` with
  eight explicit `!node_modules/...` exclusions; `mac.target` = `dmg` + `zip` for
  **arm64 and x64**, `hardenedRuntime: true`, `gatekeeperAssess: false`,
  `entitlements`/`entitlementsInherit: build/entitlements.mac.plist`,
  `afterSign: scripts/notarize.js`, `afterPack: scripts/after-pack.mjs`;
  `win.target` = `nsis` (x64+arm64); `linux.target` = `deb`/`AppImage`/`rpm`
  (**x64 only**). There is **no `asarUnpack` key at all today**.
- `scripts/check-runtime-deps.mjs` is a **manifest allowlist**, not a scan: "an
  allowlist fails on everything a human did not write down, so adding an entry is
  a reviewable decision rather than an accident" (`:14-18`), and it exists
  because at v0.19.6 the production closure put ~470 MB of unreachable
  node_modules into a 1.0 GB install. The allowlist is six packages today.
- `scripts/check-packaged-closure.mjs`, `scripts/check-vendored.mjs`, and the
  hand-maintained exclusion list in `package.json` (`files`) are the other three
  closure gates.
- `docs/CODE_SIGNING.md`: signing via `CSC_CONTENT`, app notarization through
  `afterSign` gated on `NOTARIZE=true` (`mac.notarize` stays `false` on purpose,
  "`:50-58`"), DMG signing and `scripts/notarize-artifacts.mjs` rewriting
  `latest-mac.yml`'s `sha512`/`size` after stapling.
- `pnpm-workspace.yaml` carries `onlyBuiltDependencies` (biome, core-js, electron,
  electron-winstaller, esbuild, sharp) — pnpm 10 does **not** run a dependency's
  install script unless it is listed there.
- `AGENTS.md:1096-1127` pins pnpm to **10.29.2** because 10.29.3+ drops dependency
  edges from the command electron-builder uses to build the asar, and the failure
  is silent until launch.

### 2.12 cmux: the model, and the licence that forbids the code

- The repository the operator named is `manaflow-ai/cmux`. Its `LICENSE` reads:
  "Except where a file or accompanying notice states otherwise, material
  contributed under the cmux project license is licensed under the GNU General
  Public License v3.0 or later (**GPL-3.0-or-later**)". **Verified** by fetching
  `https://raw.githubusercontent.com/manaflow-ai/cmux/main/LICENSE`.
- **Consequence, stated once and then assumed:** not one line of cmux may enter
  either repository. What may be adopted is its *model*, and this document does
  adopt it explicitly (§4): one authoritative cell grid per surface independent
  of any view ("attaching never resizes it"), surfaces accumulating output while
  invisible, views cropping/pinning rather than dictating, create params
  (`cwd`/`initial_command`/`initial_input`/`env`), `send_text` with bracketed
  paste keyed on the **live** DEC 2004 mode, `send_key` with named chords encoded
  through the terminal's own key encoder synced to surface modes, read-screen as
  the current viewport plus a separate read-scrollback, a typed failure taxonomy
  (`input_queue_full`, `surface_unavailable`, `process_exited`, `not-a-terminal`,
  `unknown key`), focus-neutral commands by default with an explicit
  focus-intent allowlist, attention without focus, and the honest durability
  boundary.

---

## 3. Candidate topologies: where the pty lives, and where the terminal lives

The two placement questions are independent in principle and entangled in
practice, because **R7 forces the terminal's *state* to be readable with no
view present**, and **R13 forces output to keep accumulating while the pane is
closed**.

### (a) pty in main + `@xterm/headless` terminal of record in main + pane mirror — recommended

- `node-pty` forks the pty in the app's main process. Every byte the pty emits is
  appended to that surface's **byte log** and written into a **`@xterm/headless`
  `Terminal`** instance — the terminal of record. Main owns `cols`/`rows`.
- The visible pane is React, in the app's own renderer, hosting `@xterm/xterm`
  (the same version as the headless core) and fed from the same byte log: it
  replays retained bytes on mount, then streams live bytes.
- `console_read` (both its `mode: "viewport"` and `mode: "scrollback"` forms, §10.2)
  answers from the **record**, with no view needed. `console_screenshot` captures
  pixels through a view fed by the record (§13.2).
- **Cost:** one pure-JS `Terminal` per surface in main (measured size on npm:
  `@xterm/headless@6.0.0` unpacked 1,957,834 B, no dependencies) — plus at most
  one live pane renderer at a time, because the right slot shows one pane.
- **Status:** this topology was chosen before the compatibility spike ran and is
  **unchanged by it**; §5.5 re-opens the emulator half on the spike's
  measurements and closes it again, and the two things the spike changed inside
  this topology are named there (the capture path, §13.2) and in §0.4.

### (b) The emulator lives in the app's renderer, one view per surface — rejected

Attractive because it collapses (a)'s two instances into one. Rejected on three
verifiable grounds:

1. **Cost per surface.** A `WebContentsView` is a renderer process. R3 asks for
   "multiple running console commands across all chat sessions at once"; eight
   agent surfaces like the browser cap (`registry.ts:116 MAX_AGENT_TABS = 8`)
   would be eight renderers holding a WebGL context each, whether or not anybody
   is looking.
2. **R7 is unsatisfiable without inventing the same state anyway.** An agent's
   `console_read` on a surface whose pane was never opened would have nothing to
   read unless main held a copy — which is (a)'s terminal of record, plus the
   view.
3. **The mirror would then be authoritative**, so `read` (from the copy) and
   `screenshot` (from the view) could disagree, with no rule saying which is
   right.

### (c) A Python-side pty in the session runtime — rejected

Rejected on ownership, not on capability: `local_operator` *can* fork a pty
(§1.2.3), but the surface must exist when the app is open and die with it (R13,
R16), the discovery record is the app's (§2.2), and the browser precedent puts
the surface in the app. A Python-owned pty would also give the surface a second
possible owner whenever a session runs without the app, which is exactly the
state the tool's gate says cannot happen. The one thing Python keeps is the
*harness* half: the tool, the redaction seam, the e2e cells.

### (d) `utilityProcess` helper — deferred, not rejected

Electron's `utilityProcess.fork` can host both node-pty and the headless core,
keeping a byte-flooding pty off the main thread. It is **deferred** rather than
chosen because it adds a second process to supervise, an extra hop for every
method, and a second lifecycle to reason about, in exchange for a risk (§17.3)
that has not yet been measured. The seam is designed so this is a swap: every
pty and emulator operation goes through one interface (`ConsoleHost`), whose only
implementation in v1 is in-process. §17.3 names the measurement that would force
the move.

### Comparison

| | cost per surface | R7 with no view | R13 while closed | one authority | packaging |
|---|---|---|---|---|---|
| **(a) main: node-pty + headless core, pane mirror** | one JS `Terminal` (~2 MB installed once) | **satisfied** | satisfied (the log + the core keep running) | **yes** (main) | one native dep (§5.3) — its pty half is now measured (§19.1 P1), its packing traps are §18 risk 1 |
| (b) emulator in the renderer, view per surface | one renderer process + GL context | **not satisfiable** | satisfied only if the view stays alive | no — the copy and the view can differ | none |
| (c) Python pty | n/a | no (app may be closed) | no | no — two owners | none in the app |
| (d) `utilityProcess` | one JS core + one helper process | satisfied | satisfied | yes | one native dep |

**Recommended: (a).** The consequence for the two questions the brief asks
directly:

- **R13 (the terminal keeps running while the tab is closed):** the *surface*
  (pty + log + core) is a property of the app, not of the pane. Closing the pane
  unmounts a React component; it sends **no** resize and **no** signal to the
  pty. Reopening replays the retained bytes. This is (a)'s cheapest property and
  the reason the pane may be treated as disposable chrome.
- **The app is closed:** there is no surface. The tool is not in the inventory
  at all (the `createIf` gate reads a file), and a session that had one gets a
  typed, honest answer from `console_*` — `ui_host_unavailable` with copy that
  names the app and says the surfaces ended with it. §15 gives every cell.

---

## 4. Capability matrix, and what "similar to ghostty" is being held to

`cmux`'s model, second column, is the yardstick; the third is this design's
verdict. "v1" means the PR that ships the surface; "later" means out of scope
with a reason, not a maybe.

| capability | cmux's model | this design | where |
|---|---|---|---|
| one authoritative grid per surface, independent of any view | yes — "attaching never resizes it" | **yes**: `cols`/`rows` are surface state owned by main; attach changes bounds only | §8 |
| output accumulates while invisible | yes | **yes**: the byte log and the record keep consuming | §7.1 |
| views crop/pin/scale, never dictate | yes | **yes**: the pane reports a rect, main decides the grid | §8.2 |
| create params: `working_directory`, `initial_command`, `initial_input`, `env` | yes | **yes** (`cwd`, `command`, `args`, `input`, `env` in §10.2) | §10.2 |
| send text with bracketed paste keyed on the LIVE mode | yes (DEC 2004) | **yes**: `console_input` reads the record's `modes.bracketedPasteMode` | §10.2 |
| send named keys via the terminal's own key encoder, synced to modes | yes | **yes, with a conformance test** against the renderer's encoder | §10.5 |
| read-screen = current viewport; read-scrollback = separate | yes | **yes**: two methods, and the text is `translateToString` off the record | §13.1 |
| typed failure taxonomy | yes | **yes**: §10.6, mapped onto the existing `ErrorCode` values where one fits | §10.6 |
| focus-neutral commands with an explicit focus-intent allowlist | yes | **yes**: `reveal` is per-call, and OS focus is never taken | §10.4 |
| attention without focus | yes | **yes**: the blip + the notification | §12 |
| durability boundary stated honestly | yes | **yes, and slightly better than cmux here**: we keep an exact byte log, so a closed surface's history is exact rather than "visually recoverable" | §7.3 |
| Sixel / inline images | yes | **later** (§1.3) | — |
| detached (survives app quit) | yes | **later**, with the cost named | §7.4 |
| splits / panes | yes | **no** — out of scope by §1.3 | — |

---

## 5. R4/R5 — the emulator, assessed against Ghostty rather than assumed

R4 asks for terminal emulation "similar to ghostty", asks that ghostty's licence
be verified, and asks that the relevant functionality be copied into
`local-operator-ui`. The licence is verified; the copying is where the
requirement needs interpreting, and the interpretation is stated rather than
implied.

### 5.1 Ghostty's licence, verified

**Verified** by API read at the time of writing: `ghostty-org/ghostty` reports
`spdx_id: MIT` (61,305 stars, default branch `main`, last push 2026-09-19).
Ghostty is first-party MIT, so **the licence permits vendoring or compiling its
VT core**. That is not the question; the question is what it costs to own.

**Also verified** (the rolling `tip` release's asset list): upstream publishes
`ghostty-vt.wasm` — **1,007,826 B** — with `ghostty-vt.wasm.minisig`, plus
`ghostty-vt-small.wasm` (747,017 B), `ghostty-vt.xcframework.zip` (14,147,171 B)
and `libghostty-vt-source.tar.gz` (4,005,248 B), each minisign-signed. The
existence of these artifacts is what makes option B below a real option rather
than a fantasy, and the minisign signatures are what make pinning one honest.

### 5.2 The three options, with what each actually buys

| | what it is | licence | packaging cost | what it cannot do |
|---|---|---|---|---|
| **A** | `@xterm/xterm@6.0.0` + `@xterm/headless@6.0.0` + addons (`serialize` 0.14, `webgl` 0.19, `unicode11` 0.9, `fit` 0.11, `search` 0.16, `image` 0.9 — all **MIT**, all zero-dependency) with `node-pty@1.1.0` (**MIT**) for the pty | MIT | one native dep (§5.3) | not ghostty's renderer; no Sixel without addon-image. **Measured (spike):** the browser bundle loads in a hidden `BrowserWindow`; WebGL gets a real ANGLE-Metal context; `serialize`/`search`/`unicode11` all work (unicode 11 proven discriminatingly: `cursorX` after an emoji is 1 under v6 and 2 under v11); cells readable with fg/bg colour **modes**. **Its capture is the weak half** (§13.2) |
| **B** | Ghostty's own VT compiled to WASM (`ghostty-vt.wasm`), wrapped by `ghostty-web@0.4.0` (**MIT**, `coder/ghostty-web`) or driven **directly, with no shim at all** | MIT | a 1,007,826 B WASM blob (or `ghostty-web`'s bundled 423,045 B one), a **pinned single-vendor** wrapper if the wrapper is taken, and its own renderer to audit | no shell integration with the app's CSS variables. **Measured (spike):** the raw blob is `wasm32-freestanding`, **0 imports / 189 exports**, sha256 `139c9617c9dfea4a51dfab7a87a72017163c472ad248c61cdea8d7e942c1f424`, minisign **VALID** against the key in Ghostty's own `PACKAGING.md`; it created a terminal, took bytes, resized to 100×30 and returned **plain text, re-emitted VT, HTML and styled cells** (wide/`SPACER_TAIL` flags on CJK and emoji tails, palette red → `[204,102,102]`) with **no C or Rust shim**; `ghostty_type_json()` hands over a 43,531-byte machine-readable ABI (159 types, struct sizes/offsets, enum values). `ghostty-web` renders its canvas **while unattached to the document** and `toDataURL()`s it correctly first attempt |
| **C** | `@coder/libghostty-vt-node@0.1.0-beta.0` (**MIT**, N-API, `engines.node >= 20.19`, "ABI-stable… terminal semantics") | MIT | a native dep **and** a beta ABI the project itself calls explicitly unstable | **no renderer at all** — it gives state and structured snapshots, so the pane still needs xterm or a hand-written renderer |

**And the option the brief invites a verdict on: vendoring ghostty's Zig
`src/terminal` graph.** Verdict — **do not**. Reasons, in order of weight:

1. **It buys the one thing we do not need and costs the three things we cannot
   carry.** Its output is a VT state machine; we already need a JS-side state
   machine *in main* (R7) and a renderer in the pane, and vendoring supplies
   neither the renderer nor a JS boundary — it supplies Zig that must be compiled
   and bound.
2. **A permanent build burden in a repo that has deliberately zero native
   dependencies** (§2.11): a Zig 0.16 toolchain on four CI platforms, one per
   macOS arch, one per Windows arch, and Linux x64, in a project whose whole
   current closure is six pure-JS packages.
3. **A permanent patch burden.** Ghostty's VT is developed with Ghostty's own
   priorities; a vendored fork means owning the divergence, and the repository's
   stated rule for vendored code (`check-vendored.mjs`) is that a *human* pins
   every byte and re-pins it deliberately. That is affordable for a few hundred
   lines of driver logic; it is not affordable for a Zig terminal library.
4. **The comparison is not favourable on failure modes either**: a WASM boundary
   (option B) is *reviewable* — one blob, one signature, one wrapper — where a
   vendored Zig build is a compiler version, a build graph and a patch stack.

**Recommended: A**, on the spike's evidence and not merely on availability. The
reasoning is in §5.5, which is the section the spike re-opened; the short version
is that A's weak half turned out to be **pixels** (three measured capture traps,
§13.2) and B's weak half is still **the pane's renderer** (2-D canvas only, one
vendor, 0.x) — and A's weak half is a font/renderer choice we control, while B's
would be a renderer we would have to write. What R4's "copy the relevant terminal
functionality" is therefore held to is **VT correctness and coverage**, not a
shared codebase: a corpus of escape sequences, the project's own `lop` interface
run inside a surface, and — new in revision 2 — a **differential conformance
probe against ghostty's own VT** (P14), which is only possible *because* the wasm
drives with no shim.

### 5.3 `node-pty` is the one native dependency, and it is a packaging decision

Measured from the published tarball (`node-pty@1.1.0`):

- **N-API.** It depends on `node-addon-api ^7.1.0` and ships prebuilt binaries
  under `prebuilds/<platform>-<arch>/pty.node` for **darwin-arm64, darwin-x64,
  win32-x64, win32-arm64** plus `conpty.dll`/`winpty.dll`/`OpenConsole.exe`/
  `winpty-agent.exe` and the macOS **`spawn-helper`** executable. N-API means the
  prebuilt binary is **ABI-stable across Node and Electron versions**, so no
  `electron-rebuild` step is expected — §19.1 makes it a probe rather than an
  assumption.
- **No Linux prebuild** ships in the tarball. Its `install` script is
  `node scripts/prebuild.js || node-gyp rebuild`; `prebuild.js` exits 1 when
  `prebuilds/<platform>-<arch>` is absent, so **on Linux the package compiles
  from source at install time** and needs a toolchain. The UI repo builds
  `deb`/`AppImage`/`rpm` for Linux x64 (`package.json` `build.linux.target`), so
  this is a CI-toolchain question, not a hypothetical.
- **`asar` is already handled by the library — and the rewrite is necessary but
  NOT sufficient.** `lib/unixTerminal.js` computes
  `helperPath = path.resolve(__dirname, native.dir + '/spawn-helper')` and then
  rewrites `app.asar` → `app.asar.unpacked` (and `node_modules.asar` likewise)
  before `pty.fork`. **Measured (spike):** in a fully-packed `app.asar`,
  `pty.node` resolves and the spawn still fails, because the rewritten path does
  not exist until the prebuilds are *actually* unpacked; with
  `--unpack-dir node_modules/node-pty/prebuilds` it works. So the requirement is
  that the tree **is** unpacked, not that the path is patched.
- **The shipped helper is not executable, and nothing in the install fixes it.**
  **Measured (spike):** the published tarball carries
  `prebuilds/darwin-arm64/spawn-helper` with mode **0644**, so the first spawn
  dies `FATAL Error: posix_spawnp failed. at new UnixTerminal (lib/unixTerminal.js:92)`.
  npm 11.17's install-script gate means `scripts/prebuild.js` does not run, and
  the UI repo's own gate is pnpm 10's `onlyBuiltDependencies` (§2.11) — the same
  class. **PR A therefore chmods the helper to 0755 in its packaging step and
  asserts the mode in the packaged tree**, because "the install script will do
  it" is measurably false in this fleet.
- **The prebuilds carry 58 MB the app does not need.** **Measured (spike):** the
  64 MB unpacked package is mostly `win32` prebuilds and their `.pdb` files; the
  `darwin-arm64` prebuilds are **135,976 B across 2 files**. PR A prunes the
  foreign-platform prebuilds for a platform-specific artifact and says so in the
  packaged-closure evidence.
- **pnpm 10 will not run its install script** unless `node-pty` joins
  `onlyBuiltDependencies` in `pnpm-workspace.yaml` (§2.11).
- **The closure gates must be edited deliberately**: `check-runtime-deps.mjs`'s
  allowlist gains `node-pty` (and its declared `node-addon-api` dependency
  reaches the closure check), and `package.json` gains an `asarUnpack` entry —
  which does not exist today — for the package directory so that both `pty.node`
  and the exec-bit `spawn-helper` land outside the archive.
- **Signing.** The app is signed, notarized and hardened (§2.11). A nested
  executable in `app.asar.unpacked` must be signed with the same identity, and
  `build/entitlements.mac.plist` already grants
  `com.apple.security.cs.allow-jit`, `allow-unsigned-executable-memory`,
  `disable-library-validation` and `network.client`, so the entitlements
  surface does not need to grow for a pty. §19.1 makes "a signed, notarized build
  launches and forks a pty" a release gate rather than a hope.

### 5.4 The emulator seam (R5, and the exit from option A)

The emulator is behind **one interface**, in one file, so 5.2's table is a
swappable decision rather than a rewrite:

```ts
// src/main/console/emulator.ts   (shape, not final API)
export interface ConsoleEmulator {
  write(bytes: Uint8Array): void;
  resize(cols: number, rows: number): void;
  text(mode: "viewport" | "scrollback"): string;
  readonly modes: Readonly<{
    bracketedPaste: boolean;
    applicationCursorKeys: boolean;
    mouseTracking: string;
  }>;
  readonly cursor: Readonly<{ x: number; y: number }>;
  dispose(): void;
}
```

`@xterm/headless` satisfies this today (its typing declares `readonly modes: IModes`
with `applicationCursorKeysMode`, `applicationKeypadMode`, `bracketedPasteMode`,
`insertMode`, `mouseTrackingMode`, `originMode`, `reverseWraparoundMode`,
`sendFocusMode`; `write`, `resize(columns, rows)`, `input(data, wasUserInput?)`,
`onBell`/`onLineFeed`/`onWriteParsed`/`onResize`/`onScroll`/`onTitleChange`; and
the buffer API `cursorX`/`viewportY`/`baseY`/`translateToString` with
`IBufferCell.getWidth()`/`getChars()`). **One caveat that shapes §7 and §13:**
the package's README says "⚠ This package is experimental" and "Currently no
official addons are packaged on npm" for the headless build — so **no
`addon-serialize` in main**. The design does not need it (§7.1), and this is the
reason it does not: *replay* replaces *serialize*.

---

### 5.5 D1 re-opened: the emulator decision, on the spike's measurements

The spike ran after revision 1 was written, so D1 was the one decision taken on
published facts rather than on runs. It is re-opened here, decided, and closed.
The measurements are in §5.2's table and §0.4; what follows is the argument.

**What changed in the evidence, in three lines.** (i) `node-pty`'s N-API prebuild
loads under this Electron with no rebuild, and a real pty was driven end to end —
so the *pty* half of option A is no longer a risk at all. (ii) xterm's half is
confirmed working in a hidden window **except for capture**: the WebGL canvas
cannot be read (`toDataURL()` returns blank white) and `capturePage()` on a hidden
window returns a stale/blank first frame, where the **DOM renderer captures
correctly first attempt**. (iii) The ghostty path is **real rather than
theoretical**: the raw `ghostty-vt.wasm` drives with no C or Rust shim, its
provenance verifies (sha256 + minisign against Ghostty's own published key), and
`ghostty-web` renders and captures offscreen first attempt — exactly where xterm's
WebGL path failed.

**The three options, against that evidence:**

| | what it means concretely | cost | does it satisfy R7 (capture with no pane)? |
|---|---|---|---|
| **(1) xterm.js only** — one emulator (`headless` in main + `xterm` in the pane), ghostty as a *behavioural* reference | keep revision 1's architecture; fix the capture path (§13.2: DOM renderer for the capture view, `capturePage` only, non-blank assert + one retry) | the capture-path work, which is now *known* work rather than presumed | **yes** |
| **(2) ghostty's VT wasm in MAIN as the authority** | main holds a `ghostty-vt.wasm` terminal; text/VT/HTML/styled cells come from its formatters. **The renderer is the item**: `ghostty-web` owns its own wasm instance and renders 2-D canvas only, so it cannot straightforwardly be used as a renderer *for someone else's* terminal — the choices are to take `ghostty-web` wholesale (which moves the authority into the renderer, i.e. it becomes option 3), or to write our own renderer over `GhosttyCell` rows | **a renderer, written by us** — the single largest item in this feature, and the one thing revision 1 refused on principle ("write no renderer") | **yes** (the authority is still in main) |
| **(3) `ghostty-web` in the renderer, authority there** | the pane owns the terminal state | cheapest of the three to reach a *screenshot*, and it makes the capture trap disappear | **no** — a surface whose pane was never opened has no state to read, which is R7's whole case, and it re-introduces the mirror problem in the worse direction (the authority would be the view) |

**Decision: (1) xterm.js only — revision 1's architecture stands, with the
capture path corrected.** The deciding asymmetry is that A's weak half is
*pixels* and B's two weak halves are a *renderer we would have to write* and, for
option 3, R7 itself. Three measured facts make (1) cheap to hold: the DOM
renderer captures correctly first attempt, so the offscreen path has a known-good
renderer; `capturePage()` (the compositor) is not the same mechanism as
`toDataURL()` (the canvas' own bitmap), so a WebGL pane can still be photographed
as part of the window — measured in the same harness with the WebGL addon loaded,
where `capturePage()` returned a correct 27,869 B frame on its second attempt;
and the failure mode when a first frame is stale is a *retry*, not a redesign.
*(The compositor-vs-canvas distinction is the mechanism I read into those two
numbers rather than something the spike isolated; §13.2's assert-and-retry does
not depend on that reading being right.)*

**R4, R7 and R10 under this decision, in one paragraph.** R4 is satisfied in
**behaviour, not in code**: the surface is a VT-correct terminal measured against
ghostty's own VT (P14) rather than a build of it, and the document says so plainly
instead of implying a shared codebase. R7 is satisfied *because* the authority is
in main — capture works for a surface nobody has opened, which is the requirement
that eliminated options (2)-without-a-renderer and (3). R10 is satisfied by the
record being the single source both actors read: the human sees the mirror, the
agent reads the record, and the frame the agent gets is either the live pane or a
DOM-rendered reconstruction from that same record, labelled `rendered`.

**The seam that keeps (1)↔(2) a swap rather than a rewrite** is §5.4's
`ConsoleEmulator` interface — `write`, `resize`, `text`, `modes`, `cursor`,
`dispose` — plus two rules that make a swap honest: the pane never reads the pty
and never answers a read (so the authority can move without touching the pane's
role), and the conformance tests (P5, P14) are written against the interface, not
against xterm's API.

**The upgrade path, recorded with its provenance.** If (1) ever fails a
correctness probe on a TUI the product ships, the replacement is
`ghostty-vt.wasm` — **1,007,826 B, sha256
`139c9617c9dfea4a51dfab7a87a72017163c472ad248c61cdea8d7e942c1f424`**, MIT
(Ghostty is first-party MIT, §5.1), GPG/minisign signature **verified in this
environment** against the key published in Ghostty's own `PACKAGING.md`, 0 imports
/ 189 exports, `wasm32-freestanding`, instantiating in both Node and the Electron
renderer. Two ABI gotchas a future implementer must not re-learn:
`CELLS_RAW` writes a **pointer** into `out` (not a copy, so the buffer must stay
alive and the read must be immediate), and a row iterator is **single-use**. Its
formatters return text, re-emitted VT, HTML and styled cells, and `ghostty_type_json()`
publishes the full ABI (43,531 bytes, 159 types) — so the type layer is
machine-checkable rather than guessed at, and `scripts/check-vendored.mjs`'s
byte-pinning discipline (§2.4) applies to the blob exactly as it does to the
browser driver modules.

## 6. R1/R2/R3/R13/R18 — the surface, the pane, and what owns what

### 6.1 R1 — the pane: the fourth occupant of the right slot

The console becomes the **fourth** right-slot pane, and the change is the same
shape as the third one's: one more name in `claimRightSlot`'s union, one more
`===`, one more `isConsolePaneOpen` field, one more width preference, one more
`{isOpen && …}` block in `chat-content.tsx` with its own divider label, and one
more control in the header cluster.

Specifics this document fixes, so the implementation is reviewable against them:

- **Store** (`ui-preferences-store.ts`): `isConsolePaneOpen: boolean` (global,
  persisted, one-at-a-time with its three siblings through `claimRightSlot`),
  `consolePanelWidth: number` with a default chosen so the pane's *default grid*
  is reachable (see below), and — because the pane needs a lens like the
  browser's scope switch — `consoleActiveSurface: string | null`, the surface
  the pane is showing when a session has more than one.
- **Pane width vs grid.** The browser's `640` is justified as "a page at 420px is
  not a page"; the console's justification is a character cell, and the cell is
  now **measured**: in the spike's harness the xterm DOM renderer reported
  **8.425 × 16 px per cell at `fontSize: 14`** (100 columns → 800×384 px), so a
  100-column pane needs **≈843px** plus the pane's horizontal padding at that
  step, and the same grid at a smaller step is proportionally narrower. (The
  spike's `ghostty-web` measured 9×15 with a 12px baseline at DPR 2 — the two
  stacks do not agree, which is one more reason the *grid* is main's and the
  *mirror* is replaceable.) Recommendation: pick the pane's font step, then set
  `consolePanelWidth`'s default from the measured advance of **the shipped
  Geist Mono** (not Menlo, which is what the spike's harness resolved to — see
  §6.6) so the default lands on the default grid below; clamp the divider at
  `minWidth={480}`; and leave the *surface's* 40-column floor to main, not to
  the divider. The PR must print the measured px-per-column it used.
- **Default grid: 100×30.** Rationale, all citable: the project's own visual
  evidence is produced at 100×30 and 150×40 (`AGENTS.md:1574-1582` names
  `scripts/ask_shot.py out.svg 100x30` etc.); the TUI's own regressions are
  recorded at 80×24 and 120×36 (`local_operator/tui/app.py:2878`, `:10794-10802`);
  and 100×30 is wide enough for the interface this feature exists to display
  while staying under the 120 columns that would force an unnecessarily wide
  panel. Floor **40×10**, ceiling **500×200** (the ceiling is a resource bound,
  not a UI one).
- **Header control**: a `variant="ghost" size="icon"` trigger in
  `chat-header.tsx`'s cluster, `aria-label="Open console"`, carrying the
  attention blip (§12.4) with the same 12px-earning discipline the browser
  button's badge already documents, and the same focus-return effect on close.
- **The user-initiated create path, fixed here rather than left to PR B.** The
  header control opens the *pane*; a human then needs a way to make an ordinary
  surface, and that is a second entry point with its own defaults. They are:
  - **Affordance**: a `+` button in the pane's own header (the pane's chrome, not
    the app's) beside the surface list, present in the empty state too, because
    the empty state is the one a first-run user actually meets.
  - **`command`**: the user's login shell with no arguments — `$SHELL`, else
    `zsh`, else `/bin/sh` — resolved as §6.5 describes for the marker text. The
    same object the agent's `console_create` makes, sized by the same rule (§8.4).
  - **`cwd`**: the session's own working directory, the same value the agent's
    create uses, so a user who opens a console to run what the agent just
    described lands where the agent was.
  - **`env`**: §6.6's environment, unchanged. A user-initiated surface does not
    get a second, thinner environment, and the three variables §6.6 says must
    survive must survive here too.
  - **Grid**: **100×30**, §6.1's default, from the same pure sizing function —
    the user's surface and the agent's differ only in who asked.
  - **`reveal`**: defaults to **`"open"`** for a user-initiated create, against
    `"none"` for an agent (§10.4). The human clicked a control inside the pane,
    so the pane is what they are already looking at; §10.4's rule still binds,
    so an `"open"` in an unfocused window opens the pane and does **not** raise,
    activate or focus the OS window.
- **Icon**: `SquareTerminal` in `tool-glyphs.ts`'s spirit — a terminal *inside a
  frame*, distinct from `bash`'s bare `Terminal` at 16px, which is the size this
  control is drawn at (the two must be told apart at a glance in one 56px bar).
  The TUI gets `\uf108` (`nf-fa-laptop`) in `NERD_TOOL_ICONS` and `>` in
  `PLAIN_TOOL_ICONS` — both **distinct from `bash`'s `\uf120`/`$`**, both inside
  the Font Awesome block the table's docstring restricts itself to, and both to
  be checked against the shipped GeistMono Nerd Font in PR B (§19.1, probe P9).
- **Empty, loading, error states** are the pane's, and are named in §19.5 as the
  Storybook set.

### 6.2 R2 — stored history, per session, recalled on switch

Concretely, "recalled when switching across sessions" means three different
things and this design answers all three:

| switching to… | what the user sees | mechanism |
|---|---|---|
| another session, pane open | that session's surfaces, each with its full retained scrollback | the pane re-subscribes; the **record** holds the text and the **byte log** holds the bytes (§7.1) |
| back to the first session | its surfaces, still running, with everything they printed while the pane was away | §7.1: the log and the core never stopped |
| a session whose surface was opened by an *agent* | the same surface, marked as agent-opened (§6.5) | the registry is keyed by session, not by pane |

### 6.3 R3 — multiplexing: what it costs and where it is capped

- Surfaces live in one registry in main, keyed by `session_id`, each with its own
  pty, byte log and headless `Terminal`. N sessions × M surfaces is a memory
  question, and the numbers are stated so the cap is arguable: a headless
  `Terminal` with a 5,000-line scrollback holds roughly the cells it has seen
  (order of a few MB at 100×30 with attribute runs), the byte log is capped at
  4 MiB per surface (§7.2), and nothing renders unless a pane is open.
- Caps, mirroring the browser's `MAX_AGENT_TABS = 8` (`registry.ts:116`) and
  its reasoning ("nothing else bounds how many an agent fleet can open"):
  **8 agent surfaces per session**, **32 surfaces per app**, user surfaces
  uncapped within the app total. `status`/`console_list` report the counts rather
  than inventing a second number, exactly as the browser's does.
- **Cross-session multiplexing is the default, not a feature**: the registry is
  app-global, and a session's pane filters by `session_id`. There is no per-session
  daemon to start.

### 6.4 R13 — what "keeps running" means precisely

Stated as invariants, because "keeps running while the tab is closed" is the
requirement most likely to be satisfied by accident and then broken silently:

1. **The pty's lifetime is the surface's, never the pane's.** No code path
   between "pane unmounts" and "pty receives a signal" exists.
2. **Closing the pane sends no resize.** The grid is surface state (§8); hiding
   a view never re-derives it. This is cmux's rule and it has a concrete
   consequence: a TUI inside a closed surface is not reflowed, so it neither
   redraws stale frames nor loses its own scrollback.
3. **Output continues to be consumed.** The log and the core are fed by main, so
   a surface that prints 50 MB while invisible does not block (bounded by §7.2's
   ring and §17.3's backpressure).
4. **The pane may be closed and the app left running for days.** The retention
   policy (§7.4) decides what happens to a 4 MiB log nobody has looked at in a
   week; "keeps running" never means "keeps growing without bound".

### 6.5 R18 — provenance: telling this console from any other terminal

An agent told "I ran it in the console" must be able to (a) find the surface and
(b) be sure it is *this* console rather than a shell in some other window.

- **The handle's own prefix.** Surface ids are `con:<n>:<nonce>`, following the
  browser's `ui:<tabId>:<nonce>` grammar (`registry.ts:134 SURFACE_PREFIX = "ui"`
  and the anchored `SURFACE_TOKEN` grammar below it). A handle therefore *names
  its host* in every trace, every error and every listing.
- **A provenance field on every listing.** `origin: "user" | "agent"` (who
  created it) and `session_id` (immutable, set at create) are both in
  `console_list`; a user-opened surface is discoverable and readable by that
  session's agent, which is exactly R18's case, and the *listing* says so rather
  than the agent having to guess.
- **A visible marker in the pane**, so the human side matches the agent's view:
  the pane's header carries the surface's command (or `zsh`, its argv[0]) and, on
  an agent-opened surface, the agent marker the browser's tab strip already uses
  for agent tabs. The point is that a person looking at the screen and an agent
  reading `console_list` describe the same object.
- **An environment marker inside the pty**, so a *program* can tell:
  `LOCAL_OPERATOR_CONSOLE_SURFACE=<surface_id>` and
  `LOCAL_OPERATOR_CONSOLE_SESSION=<session_id>` plus `TERM=xterm-256color` in the
  surface's environment by default (`env` may add to it; §6.6 says what may not
  be removed). The project has a precedent for env-marker detection
  (`local_operator/terminals.py`), and a marker is the only mechanism that works
  for a program that is not talking to us.

### 6.6 What a surface's environment is, and the three variables it must not lose

- Defaults: `TERM=xterm-256color`, `COLORTERM=truecolor`, the two markers above,
  and the inherited environment minus the families that must never cross into a
  driven surface (the `CMUX_*`/`LOP_*` strip `tests/e2e/test_terminal_close_survives_e2e.py`
  already performs, and the `CMUX_*` strip `docs/agent-driver.md` performs for
  the app).
- **`COLORTERM=truecolor` matters** and is not decoration: it is the standard
  signal a TUI reads to decide whether to emit 24-bit colour (R5). The project's
  own `--color-*`-using renderer (`local_operator/tui/`) reads colour support from
  the environment, and `tests/e2e/conftest.py` fixes `TERM`/`NO_COLOR` for the
  same class of reason.
- `LOP_*` is stripped from the *default* environment and may be set explicitly by
  the creator; `CMUX_*` is always stripped, because an inherited `CMUX_WORKSPACE_ID`
  has already renamed the operator's real cmux workspaces from a test run (the
  incident `docs/agent-driver.md` records).
- **Nerd Font glyphs (R5):** the app renders `--font-mono` = Geist Mono, which
  **is** the Nerd Font patch (§2.7), so the surface can draw them. But the
  Python side's gate cannot be talked into it from the environment:
  `glyphs.py:228-241 nerd_icons_enabled()` is tri-state (env kill switch →
  explicit `display.nerd_icons` bool → `_nerd_capable_terminal()`), and
  `_nerd_capable_terminal` (`:191-224`) answers True **only** for a positive
  ghostty/kitty/wezterm marker, deliberately not on `TERM` and not for emulators
  that bundle no symbol fallback ("when in doubt, plain beats tofu"), and it is
  documented as a judgement narrower than the notification stack's own list.
  **Therefore the auto-detection path is closed to us, and the design does not
  take it by impersonation.** Setting `GHOSTTY_RESOURCES_DIR` would be a lie
  about what is running *and* would flip `notify.detect_protocol` to ghostty's
  notification protocol for every process in the surface — a second, unrelated
  behaviour bought with the same forgery.
  What the design does instead: **the console gets its own marker and its own
  predicate.** `local_operator/terminals.py` gains
  `is_local_operator_console(env)` (the `LOCAL_OPERATOR_CONSOLE_SURFACE` marker
  §6.5 already sets), and `glyphs.py`'s `_nerd_capable_terminal` gains one clause
  for it, because the app is an emulator that *does* render PUA glyphs and the
  detector's own rule is "a positive emulator marker wins" (`:201-206`). That is
  a two-line Python change in PR C, it is testable with an injected env (the
  module already accepts one), and it leaves every existing decision intact.
  `console_status` reports the marker it set and the glyph decision, so the
  behaviour is observable rather than inferred.
- **And the machine itself has no Nerd Font at all — measured.** `~/Library/Fonts`
  is empty, `/Library/Fonts` holds only `Arial Unicode.ttf`, and the system
  fonts are Menlo/Monaco/Courier and friends; a
  `"JetBrainsMono Nerd Font Mono", Menlo, monospace` stack resolved to **Menlo**
  and every PUA codepoint tried (U+E0B0-E0B3, U+F015, U+F07B) rendered **tofu**.
  Three consequences, and they are the whole of R5's font story:
  1. **Inside the app this is already solved, and it is the app's job.** The pane
     renders `--font-mono` = the bundled `GeistMonoNerdFontMono` faces (§2.7), and
     CSP is `font-src 'self' data:` (`styles/index.css:117`) so no remote face
     can rescue a missing glyph — the bundled file is the only mechanism. The
     spike's tofu is what happens *outside* the app and in any harness that does
     not load the app's CSS.
  2. **The bundled face has no licence notice in the repository.** Read at
     `ui@db615bd46`: `LICENSE` is the project's MIT licence (Radient Inc.),
     `build/license_*.txt` is the installer EULA in ten languages and mentions no
     font, and there is no `THIRD_PARTY`/`OFL` file anywhere in the tree. The face
     is **Geist Mono, SIL OFL 1.1** — verified from `vercel/geist-font`'s own
     `OFL.txt`: "Copyright 2024 The Geist Project Authors
     (https://github.com/vercel/geist-font) … This Font Software is licensed under
     the SIL Open Font License, Version 1.1", with no Reserved Font Name declared
     — redistributed as a Nerd-patched derivative, and Nerd Fonts' own `LICENSE`
     states that "Nerd Fonts source fonts, patched fonts, and folders with
     explicit OFL SIL files are licensed under SIL OPEN FONT LICENSE Version 1.1".
     OFL's condition is that the copyright notice and the licence text travel with
     the font. **Action, owned by PR B and stated as a checklist item rather than
     a nicety:** ship the OFL notice and text for the face (and Nerd Fonts' MIT
     notice for the patch), or record in the PR why the existing distribution
     already satisfies it. This is pre-existing rather than caused by the console,
     but the console is the feature that *depends* on the font, so it is the
     feature that should not leave it undocumented. *(The licence is
     web-verified, not legal advice; if in doubt the OFL text and copyright line
     above are the two things to attach.)*
  3. **A design-round item, measured, not a stack defect:** Menlo's box-drawing
     glyphs **do not join** at the pane's size — a visible gap in `─` in both
     stacks the spike tried. The bundled Geist Mono must be checked for
     continuity at the chosen font size in the torture stream (§9.3), because a
     terminal whose box frames have gaps reads as broken even when it is
     rendering correctly. If the bundled face does not join either, the answer is
     the font step (and, if that fails, a different bundled face), not the
     emulator.
- **Tofu is not detectable in a canvas, and does not need to be here.** A missing
  glyph in a canvas renderer produces a blank cell with no signal to read, which
  is why the Python side has an env-marker gate rather than a runtime probe
  (`glyphs.py:168`). The pane does not need detection because the app owns the
  font and ships the face; what it *does* need is the design round's glyph sweep
  (§19.1's P9) and the licence notice above.

### 6.7 Surface identity and lifetime, stated as a table

| event | pty | byte log | record (text) | pane |
|---|---|---|---|---|
| pane opened | unchanged | unchanged | unchanged | mounts, replays the log, streams |
| pane closed | **unchanged** | keeps growing (bounded) | keeps updating | unmounts, sends nothing |
| session switched away | unchanged | unchanged | unchanged | re-subscribes to the new session |
| surface created by an agent with `reveal: "none"` | starts | starts | starts | **not** opened |
| session deleted | signalled (bounded grace, then `SIGKILL`) | dropped | dropped | closes if it was showing one |
| surface closed (`console_close kill:true`) | signalled | retained | retained, marked `exited` with the exit code | shows "ended" state |
| app quit | dies with the app | persisted, if the surface is retained | reconstructed at next launch, marked `live: false` | shows the recorded history |
| app relaunch | **nothing is running** | replayed into a fresh core | `console_read` returns `live: false` and the last-known grid | "this terminal has ended" |

---

## 7. R2/R13 — retention, storage, and the honest durability boundary

### 7.1 The three representations, and why there are three

| representation | who reads it | what it is exact for |
|---|---|---|
| **byte log** (append-only `Uint8Array` ring per surface, in main) | the pane (replay), persistence | every byte the pty emitted, in order — the only *lossless* record |
| **the record** (`@xterm/headless` `Terminal`, §5.4) | `console_read` (both modes), `console_status`, the key/paste mode decisions | the *rendered* grid and its scrollback, i.e. what a human would see |
| **the core's buffer as text** | the agent and the pane's initial paint | the current viewport and the scrollback, as plain text |

Two of these are load-bearing and the third is a projection. Keeping the byte
log is what lets this design be *more* exact than cmux's stated boundary: cmux
tells the truth that bytes emitted while no tap existed are "recoverable
visually from a snapshot but not as exact historical records"; here there is
always a tap, because main is the pty's only reader.

### 7.2 Caps, defaults and the reasons

- **Byte log: 4 MiB retained per surface, 16 MiB hard ceiling.** A ring, not a
  file: past the retained window the oldest bytes are dropped and the log reports
  `truncated: true` (so `console_read`'s `scrollback` mode can say "showing the
  last N lines; earlier output was dropped" rather than pretending completeness).
- **Scrollback: 5,000 lines** (`Terminal`'s `scrollback` option) per surface.
  This is 5× the addon default and is chosen against the grid: at 100 columns a
  line is ≤100 cells, so the worst case is ~500k cells of attributes.
- **Surfaces: 8 per session by an agent, 32 per app** (§6.3).
- **Persistence: opt-in per surface, default ON for a user surface and OFF for an
  agent surface.**

### 7.3 Where it is persisted, and what "recalled" means across a relaunch

- Location: `<config-dir>/run/ui-console/history/<session_id>/<surface_id>.log`,
  plus a `<surface_id>.json` sidecar holding the create parameters (argv, cwd,
  grid, `created_at`, `last_seen_at`, `exit_code`, `truncated`) and
  **`exit_epoch`** — the surface's exit *generation*, incremented once per
  `process_exited` (§10.6). The retention layer owns that counter rather than the
  live registry, because its job is to outlive the process: a retained surface is
  replayed under its own id (§6.7) and may be run again, and only a generation
  that survived can tell that second exit from the first. `console_status`
  returns it (§10.2) and the notifier keys on it (§12.3). The directory follows
  the discovery namespace's discipline (0700/0600, §2.2, §2.4) and the project's
  config-root conventions rather than inventing a second root.
- **A read never creates anything** — the rule the state modules state explicitly
  (`ui_browser/state.py:87-89`) — so a listing on a machine with no history does
  not leave a directory behind.
- **Reconstruction is replay, not serialization.** At relaunch a retained surface
  is shown by feeding its persisted bytes into a fresh core (§5.4's caveat: there
  is no headless `addon-serialize`, and this design does not need one). The cost
  is O(bytes) of parsing, bounded by the 4 MiB cap, and it is paid once per
  surface at open — not per read.
- **The honesty the operator asked for explicitly** (R2's "properly recalled",
  R13's cmux nod): after a relaunch, `console_status` and the pane both say
  **`live: false`** and the pane's header carries "this terminal has ended" with
  the exit code if it was observed. Nothing pretends to be a running terminal,
  and `console_read` on such a surface is typed as history rather than output.

### 7.4 The durability boundary, stated rather than implied

- **What survives app quit:** the byte log and the sidecar, for retained
  surfaces. History, argv, cwd, grid, exit code.
- **What does not:** the process, its children, and any state inside them (a
  shell's working directory, a REPL's variables, a TUI's in-memory buffer beyond
  the 4 MiB ring).
- **What this design deliberately does not offer:** detached surfaces that
  survive logout or app quit. The cost of doing it properly is a supervisor that
  is not the app (a launchd job or a small daemon), a re-attach protocol with
  its own authentication, and a second lifecycle owner for a pty — all of which
  duplicate what the user's own terminal multiplexer already does well. If this is
  wanted later, the boundary to design is `ConsoleHost` (§3(d)): the same
  interface, a different implementation location.
- **No compression and no rotation below the cap.** Below 4 MiB a surface is
  four orders of magnitude away from mattering; above it the ring truncates and
  says so. A GC pass removes history whose session is deleted, and history older
  than `console.history_days` (default 30) — and it says what it removed.

---

## 8. R13/D6 — sizing: one owner, a stable pure function, and no oscillation

### 8.1 The rule

**The grid (`cols` × `rows`) is surface state owned by the app's main process. A
view never sets it; a view reports what it is showing and main decides.** This is
the same split the browser already uses for a different quantity — "the renderer
owns layout and reports it" (`ipc.ts`'s `browser-set-content-rect`, `registry.ts:527
setContentRect`) while main owns the view's bounds — applied to the terminal's one
authoritative number.

### 8.2 How a *displayed* grid is decided

1. The pane measures its own content rect and its own cell metrics
   (`cellWidth`/`cellHeight` from the mirror's renderer) and reports
   `{surface, contentRect, cellWidth, cellHeight}` to main on mount, on resize
   (via `ResizeObserver`) and on a font/zoom change.
2. Main computes `cols = clamp(floor(width / cellWidth), 40, 500)` and
   `rows = clamp(floor(height / cellHeight), 10, 200)`, compares with the
   surface's current grid, and only then (a) resizes the record, (b) resizes the
   pty (one `TIOCSWINSZ`, i.e. one `SIGWINCH`), and (c) broadcasts the grid to
   every view of that surface.
3. The pane's mirror applies the grid **only** as the value main returned. It
   never calls `addon-fit`; if `addon-fit` is imported at all it is used only to
   *propose* a size, never to apply one.

### 8.3 Why a *hidden* or *undisplayed* surface cannot oscillate

- A surface created by an agent with `reveal: "none"`, or a surface whose pane
  closed, keeps the grid it had. Its view — if any — is unmounted, so nothing
  reports a rect, so nothing recomputes. There is no "background viewport" for
  the console, and that is a deliberate difference from the browser's
  `BACKGROUND_VIEWPORT` (`registry.ts:120-125`): the browser needs *some* view
  size for a page that keeps painting, whereas a terminal's size is its grid, and
  a synthetic 1280×720 would be a resize event the program inside would act on.
- The **oscillation** the brief warns about is therefore structurally impossible:
  grid changes require (i) a displayed pane, (ii) a rect that differs from the
  last one reported by *that same* pane, and (iii) main's clamps. A pane whose
  parent re-lays-out at 1px intervals produces at most one grid change per
  clamp-crossing, and never a change caused by the pane's own grid changing
  (there is no feedback path from the view's applied size back into the
  measurement — the mirror is a child of a fixed-size box in flex layout).

### 8.4 What an agent's create op takes

`console_create` carries `cols`/`rows` explicitly (defaults **100×30**, §6.1), so
a headless surface has a deterministic grid from birth. A create that omits them
does not inherit "whatever the pane happens to be" — it inherits the documented
default. `console_resize` is the only other writer, and it is refused on a
surface whose creator set `sizing: "fixed"` (a convenience for a test that needs
frame-exact output; §19 uses it).

### 8.5 When the pane and the grid disagree

Two cases, and both have a stated answer:

- **The pane is narrower than the grid's floor.** At ~7.8px per column the
  40-column floor is ≈312px, which is *below* the divider's proposed 480px
  minimum — so dragging the divider alone cannot reach it. Two other paths can:
  dragging the **window** narrower than the pane's own minimum (the pane is in a
  flex row with a 220px chat column floor), and a later decision to lower the
  divider's minimum. When it happens, main holds the grid at its floor and the
  pane **crops** (horizontal scroll) rather than shrinking the grid further — a
  terminal reflowed to 20 columns is less useful than one you scroll. The pane
  shows a horizontal affordance; the grid stays honest.
- **The pane is wider than the ceiling** (500 columns): main holds the ceiling
  and the pane letterboxes. A 500-column TUI is already past what the app's own
  interface uses.

---

## 9. R12/R5 — the theme-aware terminal colour model

### 9.1 What the emulator's theme is, and where it comes from

The mirror's `ITheme` is derived from **roles**, never hexes
(`AGENTS.md:32-35`: "Name roles, never colours… If a value maps to no role, the
system is missing one; add it to the contract rather than working around it"),
in a new `src/renderer/src/shared/themes/terminal-theme.ts` that follows
`code-mirror-theme.ts` exactly: a CSS-variable table consumed by both the light
and dark builds, re-evaluated by the browser at paint so a theme swap needs no
work from React.

| xterm key | role | why |
|---|---|---|
| `background` | `sunken` | the same ground the CodeMirror editor takes ("wells, tracks, code grounds", `palette-contract.ts:192`) |
| `foreground` | `ink` | primary text, 7:1 floor on all four grounds |
| `cursor` | `accent` | the accent's one non-text job in this pane is the caret, and the editor already does exactly this (`code-mirror-theme.ts`: `caretColor: var(--color-accent)`) |
| `selectionBackground` | `accentWash` | the "faintest accent tint: hover fills, active rows, focus washes" |
| `cursorAccent` | `surface` | ink that sits on the accent — the same relationship as `onAccent` |
| ANSI 1/2/3/6 (red/green/yellow/blue) | `danger`/`success`/`warning`/`info` | the four gated semantics, in the mapping the editor's exhaustive search already chose for keyword/string/number/function |
| ANSI 8/9/10/11/14/15 (bright…, magenta, cyan) | the same four hues plus `inkMuted`/`ink` for the greys | see §9.2's honest gap |

### 9.2 The honest gap, stated before someone finds it in a frame

Sixteen ANSI slots cannot be sourced one-for-one from a contract whose hue roles
number four. On some palettes two ANSI slots will therefore be *close* or equal —
cyan/magenta landing on `info`/`danger` is the likely shape. This design's
position, stated so it can be argued rather than discovered:

- The terminal's 16 colours are **the program's** vocabulary, not the app's. A
  TUI that prints red text is not the app making a colour decision, and the app's
  job is to render that vocabulary legibly on its own ground — which is what the
  §9.3 floors assert — not to invent sixteen hues per palette.
- Therefore: **no new palette roles in v1**, and **no `accent` in the ANSI set**
  (the editor's own measurement forbids it: ΔE00 0.00 against `success` on
  monokai, `info` on dune, `ink` on obsidian).
- **Trigger for revisiting**, named so this is not a permanent excuse: if the
  design round's frames (§19.5) show two ANSI slots that a *TUI the product ships*
  relies on being distinguishable — the project's own interface is the test case,
  since it is the one TUI this feature exists to display — then add the minimum
  new role(s) to `palette-contract.ts` and re-run `pnpm check-themes`. Adding a
  role is a twelve-file change and a mandatory field; it should be paid when
  evidence demands it, not pre-emptively.

### 9.3 What is asserted, and what is only looked at

- **Asserted (new block in `scripts/contrast-contract.mjs`, per
  `AGENTS.md:61-65`):** on the terminal ground (`sunken`), `ink` and `inkMuted`
  meet their floors; each of the four hue roles used for ANSI 1/2/3/6 meets the
  `nonText` 3:1 floor as a foreground on `sunken`; `cursor`/`accent` is
  distinguishable from the ground by ≥3:1; `selectionBackground`/`accentWash` is
  a *ground* under `ink` and is checked as one ("a colour used as a background is
  treated as a ground", `contrast-contract.mjs:34-35`). The block states its own
  bound the way that file's header does: it measures flat hexes, so it cannot see
  a program's own `\x1b[48;2;…m` truecolor, and it says so.
- **Looked at, by a human, in frames (the design round, §16.3):** the twelve
  palettes rendered against one **torture stream** — all 16 ANSI colours, the
  256-colour cube sampled, bold/dim/italic/underline/reverse/strikethrough, a
  box-drawing frame, a wide-character (CJK) sample, an emoji, a combining
  sequence, every Nerd Font glyph the app's own tables use, and a `--color-*`-free
  plain paragraph — plus the three states a terminal grid is worst at: full-screen
  alternate buffer (the product's own TUI), a scroll storm, and a 24-bit gradient.
  This is the third-party-widget territory `contrast-contract.mjs:40-56` names as
  human-and-screenshot work, and a terminal grid is precisely that.
- **24-bit passthrough:** xterm emits SGR 38;2/48;2 unchanged; nothing in this
  design quantises, downsamples or maps a truecolor sequence to the 16. The
  torture stream's gradient is the evidence.

### 9.4 The pane's own chrome is not the terminal's business

The pane header, the empty state, the ended state and the error copy are the
app's surfaces: they use the app's roles, they are covered by the app's existing
`CONTROLS` rows, and the pane follows the slot's grammar (40px `bg-sunken`
header, `bg-surface` ground) so the four panes read as one slot with four modes
(§6.1).

### 9.5 The one place the app *does* influence the program's colours — and its bound

Per §6.6: the surface sets `TERM=xterm-256color`, `COLORTERM=truecolor`, and its
own `LOCAL_OPERATOR_CONSOLE_SURFACE` marker, which the Python side's Nerd-Font
detector learns to read as a positive emulator marker. Two things this
deliberately is not: it is not `TERM_PROGRAM`-spoofing (a lie about what is
running, and a second unrelated behaviour — ghostty's notification protocol —
bought with the same forgery), and it is not a *global* setting such as writing
`display.nerd_icons: true` into the user's config, which would change every
other terminal the user runs. The bound: the marker describes the app's own
rendering capability, is reported in `console_status`, and never overrides a
value the creator supplied explicitly.

---

## 10. D2/D3 — detection, precedence and the method vocabulary

### 10.1 One endpoint, one record, one new namespace — not a second host

The console rides the **existing** loopback host, its **existing** 0600 record
and its **existing** session key, and adds a method namespace:

| option | verdict |
|---|---|
| a second endpoint + a second record | **rejected.** The record's *lifecycle* does not differ (same process, same heartbeat, same teardown), which is the only argument that justifies a second namespace (`ui_browser/state.py:8-14` argues the browser's separate namespace because a *different process* — the bridge daemon — owns that directory). A second listener would also mean a second key, a second heartbeat and a second set of the four safety rules to keep true. |
| **extend the existing host** | **recommended.** One key, one heartbeat, one `isMethod` list, one dispatch. The four rules at `rpc.ts:29-46` protect the console verbatim, and the Python side reuses `HostClient` unchanged (`backend.py:718`). |
| a second gating convention in the harness | **forbidden** by the design authority (`builtin.py:12818-12821`, `AGENTS.md`'s tool ladder: "Adding a second gating convention beside `createIf` is itself a footprint regression"). |

Three consequences, each a small explicit change:

1. **The record gains console fields** — `console: boolean` (the feature is up and
   node-pty loaded), `console_surfaces: number`, `console_agent_surfaces: number`,
   and `console_proto: number` if the console namespace ever needs its own floor.
   Additive fields with defaults, so an older `UiHostState` reader
   (`extra="ignore"`, `state.py:68`) is unaffected.
2. **The enable flag is split, additively.** `LOCAL_OPERATOR_UI_BROWSER_HOST`
   keeps its current meaning for the whole host — a rig that sets it to assert
   "no host, no state file" must keep working (`index.ts:175-180`) — and a new
   `LOCAL_OPERATOR_UI_CONSOLE_HOST` (default on) disables the console feature
   alone. Without the split, the browser host would be able to take the console
   down with it, and the console tool's gate would be a lie.
3. **`/health` reports the console capability** as well as `host`/`pid`, and the
   Python gate stays **file-only** — the same "no socket while constructing every
   session" rule the browser's `advertisable()` states
   (`ui_browser/backend.py:56-67`).

### 10.2 D3 — the method vocabulary, both sides

Wire shape: one JSON `Request` per call (`protocol.py:422`), `Response`
(`:434`), the closed `METHODS` list (`:200`) extended with the names below, and
each name also added to `src/main/browser/protocol.ts:53-78 METHODS` and its
`COMMAND_TIMEOUTS_S` mirror. Params are lower_snake to match the existing
vocabulary; ids are opaque strings.

| method | params | returns |
|---|---|---|
| `console_list` | `{session_id?: str}` | `[{surface, session_id, origin, command, argv_tail, cwd, cols, rows, running, exit_code, last_activity, live, agent_owned}]` |
| `console_create` | `{session_id, cwd?, command?, args?, input?, env?, cols?, rows?, reveal?, retain?}` | `{surface, cols, rows, pid, live, revealed}` |
| `console_status` | `{surface}` | `{running, exit_code, exit_epoch, cols, rows, live, truncated, modes, cursor ({x, y} — §5.4's emulator cursor), last_activity, retain, secure}` |
| `console_read` | `{surface, mode: "viewport"\|"scrollback", start?, count?}` | `{text, cols, rows, cursor ({x, y} — §5.4's emulator cursor), truncated, live, mode}` |
| `console_screenshot` | `{surface, format?: "png"}` | `{image_base64, cols, rows, rendered: "displayed"\|"offscreen", theme, live}` |
| `console_input` | `{surface, text?, bytes?, secret_ref?, paste?}` | `{accepted: true, bytes: <count>}` |
| `console_keys` | `{surface, keys: [str, …]}` | `{accepted: true, encoded: [str, …]}` |
| `console_resize` | `{surface, cols, rows}` | `{cols, rows}` |
| `console_secure` | `{surface, on}` | `{secure}` |
| `console_close` | `{surface, kill?: bool, retain?: bool}` | `{closed, exit_code?}` |

**`cursor` is the emulator's own shape, and that is the only one.** §5.4 fixes it
(`readonly cursor: Readonly<{ x: number; y: number }>`, i.e. `@xterm/headless`'s
`cursorX`/`cursorY`, where `x` is the COLUMN and `y` the ROW) and both rows above
carry that value unchanged, so a host emits `{x, y}` and nothing else. It is
`null` in `console_status` when the app has no grid for the surface — a restored,
never-runtime surface (§7.3) — which is a value rather than an omission, and the
session-side renderer prints a field it does not recognise as it arrived rather
than as `None`. The renderer also ACCEPTS `{row, col}` defensively for a host
written against an earlier draft; that tolerance is not a second wire spelling
and no app may emit it.

And the renderer-facing ops on the browser's IPC namespace shape (not RPC —
`ipc.ts:20-35`'s rule 2 forbids routing them through `desktop-request`):
`console-state`, `console-open-pane`, `console-close-pane`, `console-select-surface`,
`console-input` (human typing), `console-keys` (human special keys),
`console-content-rect`, `console-secure-toggle`, `console-install-shell-integration`
(§12.3).

**R17's self-description is satisfied by this list being a *description*, not by
the names alone**: the tool's description names what each method can do
(§14.3), because the browser's own description establishes that a verb list gave
the model no reason to use the tool.

### 10.3 The data plane: a new surface-keyed push channel

The control plane is request/response (§10.2). The pty's output is not: it is a
byte stream that must reach exactly one consumer (the pane, when mounted) without
being re-requested, and it must not travel on the session event stream (which
carries session frames with `seq`/`epoch` semantics the console has no use for).

- **A surface-keyed subscription on the renderer's IPC**: `console-subscribe
  {surface, from_byte}` → the renderer receives `console-output {surface, seq,
  bytes_base64}` frames, plus a terminal `console-exit {surface, exit_code}`.
  Delivery is scoped to the subscription and the surface, mirroring
  `desktop-stream.ts`'s discipline ("a late frame from a dead stream must not
  land on a new one's consumer", `preload/index.ts:150-160`).
- **Replay-then-stream, with one boundary.** `from_byte` names the offset the
  subscriber has already replayed, so the first frame after subscription is
  exactly the next byte the surface emits. A subscriber that reconnects (a pane
  remount mid-stream) replays from the log up to its own last offset and
  subscribes from there — the same two-step the log already supports.
- **Coalescing is a *view* concern.** Main may batch frames on a short timer to
  bound IPC traffic, but it must not batch the *log*: the log is append-per-read.

### 10.4 R9 — `reveal`, and the focus-intent allowlist

`console_create` takes `reveal`, one of:

- `"none"` (default for an agent) — the surface exists, the pane is untouched.
- `"session"` — the pane opens **only if the app is displaying that session**, so
  an agent cannot yank the user's viewport to another conversation.
- `"open"` — the pane is claimed in the right slot (closing whichever pane held
  it) and the console pane is focused, **if and only if the app's window is
  already focused**. This is also the default for a **user-initiated** create
  (§6.1), where the human is already inside the pane and `"none"` would make the
  button they pressed appear to do nothing.

**No value of `reveal` may raise, activate or focus the OS window.** The app's
only module allowed to call `show`/`showInactive`/`focus` on a window is
`src/main/window-raise.ts`, and `scripts/window-mode.test.mjs` asserts it
(`AGENTS.md:729-735`). A `reveal` that would have to raise a window is downgraded
to `"none"` and reported as `revealed: false`, which is the same rule the
notification click obeys (`desktop-notifier.ts:1195-1208`).

### 10.5 The key encoder, and the conformance test that pins it

Human typing needs no encoder we own: the mirror's `@xterm/xterm` DOM handler
produces bytes for a real keystroke. Agent keys do — and `@xterm/headless`'s
`input(data, wasUserInput?)` **does not encode**; its own typing says it fires
`onData` with what it is given. So main owns a named-key encoder (→ `\x03` for
ctrl-c, `\x1b[A` vs `\x1bOA` for up depending on `modes.applicationCursorKeysMode`,
`\t`, `\x1b[Z` for shift-tab, F1-F12, home/end/pgup/pgdn/ins/del, alt/meta
prefixes, and the DECCKM/DECKPWM/DECBKM interactions the mode flags expose).

**The invariant that keeps one encoder from becoming two:** a conformance test
drives each named key through the *mirror's* real encoder (a DOM `KeyboardEvent`
into a headless-adjacent xterm instance) and through main's encoder, and asserts
byte equality for the whole named set, in both application and normal modes. If it
fails, the fix is to change main's table, never to change the renderer's
behaviour.

`paste: true` wraps the payload in `\x1b[200~ … \x1b[201~` **only when the
record's live `modes.bracketedPasteMode` is true** — read at the moment of the
call, from the record (which is why this is a main-side decision and not a pane
one).

### 10.6 D11 — typed errors, mapped onto the existing vocabulary

The wire already has a typed taxonomy (`protocol.py:349 ErrorCode`). Where a value
fits, **reuse it** rather than inventing a parallel one; the additions are named
with the reason the existing value would be a lie:

| condition | code | copy rule |
|---|---|---|
| host not running | (transport) `BridgeUnreachable` → `copy.no_state` | the existing sentence; the Python client already produces it |
| key rejected | (transport) 401 | existing |
| proto skew | `proto_mismatch` | existing, with the UI variant's remedy |
| unknown method on an older host | `unsupported_method` (**new**) | "This app version has no console. Update Local Operator." — a *typed refusal*, which per `protocol.py:20-45` is what keeps the floor where it is |
| surface id not found / not this host's | `surface_unavailable` (**new**, cmux's `surface_unavailable`) | names the handle and the live set's size |
| surface belongs to another session | `surface_not_owned` (**new**) | the browser's `owner_refused` precedent: typed so a caller need not substring-match |
| surface's process has exited | `process_exited` (**new**, cmux's) | carries the exit code and whether the log is retained |
| queue full (input too fast) | `input_queue_full` (**new**, cmux's) | says how many bytes were accepted |
| named key unknown | `unknown_key` (**new**, cmux's) | lists the accepted names |
| secure input active | `secure_input_active` (**new**) | refused read/screenshot, with the reason |
| capability off / pty unavailable | `console_unavailable` (**new**) | names which of the three conditions failed (§10.1) |
| grid out of range | `invalid_grid` (**new**) | carries the clamp it applied |
| a second capture requested while the capture view is busy | `console_capture_full` (**new**) | names the surface whose capture holds the view; the caller's own surface is never the one named, because §13.3's one-at-a-time rule is what makes this reachable at all |

**The `data` keys, per code.** `ErrorDetail.data` is `dict[str, Any]`, so a copy
rule that names a specific — the handle, the accepted byte count, the clamp — is
unimplementable until the key carrying it is written down, and a spelling chosen
privately on one side fails *silently*: every gate stays green while the sentence
degrades to its generic form. These are the keys, one spelling each:

| code | `data` keys |
|---|---|
| `surface_unavailable` | `surface` (the handle asked for), `count` (how many surfaces exist) |
| `surface_not_owned` | `surface` (the handle that belongs to another session) |
| `process_exited` | `exit_code`, `retain` (absent means the log is retained) |
| `input_queue_full` | `accepted` — the bytes the host TOOK; the size of the refused payload is the host's own `message`, and a copy that read it as an accepted count would state the opposite of what happened |
| `unknown_key` | `accepted` (the names the encoder has), `key` (the one that was not found) |
| `invalid_grid` | `clamp: {cols, rows}` — the grid applied instead; `reason: "fixed"` for the surface that cannot be resized at all, where there is no clamp to carry |
| `console_capture_full` | none in this half: §13.3's busy-view copy names no holder yet, and the key is recorded here as one to add when a host emits this code (§10.6's rule that the caller's own surface is never the one named) |
| `console_unavailable` | `reason` (which of §10.1's conditions failed) |
| `proto_mismatch` | `proto` — the peer's revision, so the sentence can name both numbers |

No other code carries keys: `unsupported_method` and `secure_input_active` are
answered by the copy alone. The session side reads exactly these, with two
legacy spellings accepted defensively and never read as `None` — `handle` for
`surface`, and a flat `cols`/`rows` for `invalid_grid` — recorded here so the
renderer's tolerance is not mistaken for the contract.

**Proto rule this design follows:** every addition above is *additive* — new
methods, new `ErrorCode` values that an old peer never emits (and that a peer
which emits them never expects an old host to understand, because the host is the
one refusing), and new optional params. Therefore **`PROTO_VERSION` stays 1 and
`MIN_SUPPORTED_PROTO` stays 1**, per `protocol.py:20-45`'s rule: the floor moves
only for a commit that changes the *meaning* of an existing frame or method, and
these do not. The one thing that *would* move it is a change to an existing
method's semantics — e.g. if `console_read`'s default `mode` ever changed meaning
— and the rule then requires the bump in the same commit.

---

## 11. R19/R20 — approval, sudo, and the secrets model

Security is part of the design, so this section states the model *and its
limits*, and never claims a control it does not have.

### 11.1 R19 — admin commands and the two consents

Two different consents are involved and conflating them is how this feature
becomes either annoying or unsafe:

1. **The harness gate** — `console`'s `call_approval_tier` is `"read"` for
   `console_list`/`console_status`/`console_read`/`console_screenshot` and
   **`"exec"`** for `console_create`/`console_input`/`console_keys`/`console_resize`/
   `console_secure`/`console_close`. `bash` is `exec` unconditionally
   (`builtin.py:3281` and its `approval_tier="exec"`), so this is *not* a
   differentiation mechanism — §14's discouragement is.
   `describe_approval` is set (`types.py:1250-1251`: "without one the loop falls back
   to a JSON dump"), and it names the surface, the session and the exact bytes
   or keys about to be written.
2. **The user's consent for the privileged command itself** — this is R19's
   `ask`. The tool's description tells the agent the rule explicitly: *before
   running a command that needs administrator rights, use `ask` with the exact
   command and what it will change; do not attempt a password yourself.* The
   reason this is `ask`-based rather than approval-tier-based is scope: the
   harness gate authorises *the tool call*, while the user is being asked to
   authorise *a root-level change to their machine*, and those happen at
   different moments (the agent may type the command minutes after the call was
   approved).

Three shell-side affordances make the admin path usable rather than aspirational:

- The pane's new-surface control offers a **"Run with admin"** variant that
  starts the surface as the user's shell (never as root) so the `sudo` prompt
  arrives in the surface where the user can type it.
- **`sudo`'s password prompt is typed by the human into the surface, or supplied
  by the secret path (§11.3), and never by the agent's own text** — the tool
  description forbids it and §11.3 gives the structural alternative.
- `console_status` reports whether the surface's process is still waiting on
  input (`waiting_for_input`, derived from "the pty has emitted output and then
  received no input for N ms while the last line ends in a prompt-ish byte" —
  **no**, see below).

**That last one is a heuristic and it is rejected as one.** There is no reliable
in-band way for the pty layer to know a program is waiting for input; the
honest facts are "the process is running" and "the last output arrived at T".
`console_status` therefore reports exactly those two facts — `running` and
`last_activity`, §10.2's name for the second of them, the timestamp of the most
recent output — and *nothing derived* is reported: the idle interval is the
agent's to compute, and §14.3's description tells it that a TUI or a prompt
waiting for input looks like an idle surface. Naming the heuristic as rejected is
the point: this is exactly the class of control that looks helpful and produces a
wrong answer with confidence.

### 11.2 R20 — the three ways a value can reach a terminal, and which are allowed

| path | allowed? | why |
|---|---|---|
| the human types it into the surface | **yes, and it is the recommended path** | nothing records keystrokes (§11.4) |
| the agent types a **literal** credential via `console_input` | **forbidden by policy, not prevented by construction** | the harness cannot know a string is a credential; §11.5 states the residual and the mitigations |
| the agent passes a **`secret_ref`** naming a stored credential | **yes** | the value never enters the model's context or a trace (§11.3) |

### 11.3 The exact-value path: `secret_ref`

- `console_input {surface, secret_ref: "SUDO_PASSWORD"}`. The **Python tool**
  resolves the ref from the encrypted secret store (the same store the `secret`
  tool uses, `local_operator/tools/secret_tool.py:95 build_secret_tool`), which
  is where a value the user entrusted to the harness already lives.
- **The value is never returned and never echoed.** The tool's result is
  `{accepted: true, bytes: <count>}`; the value is not in the result, not in the
  error path, and not in any log line.
- **The value is registered with the session's existing redaction sink** —
  `VariableStore.register_redaction(value)` (`local_operator/variables.py:442`),
  the exact-value pass `redaction_shapes.py:1-20` describes as the layer that
  covers "a secret the session was handed". After the call, the value is
  contained for the rest of the session: any rendered trace, tool result or
  transcript write replaces it with `[redacted]` (`redaction_shapes.py:84
  REDACTION_MARKER`).
- **The argument the model emitted is the ref, not the value.** This is what
  keeps the *trace* clean: the tool call in the transcript says
  `secret_ref: "SUDO_PASSWORD"`, which is the whole point of the indirection.
- **The value travels session → app inside the RPC params** for that one call
  (the app is the only process that can write to the pty). That is a disclosure
  to the user's own desktop app over a loopback socket authenticated by a 0600
  key; it is stated here rather than glossed, because it is the one place the
  design puts a credential on a wire, and the alternative (the app reading the
  encrypted store itself) does not exist today — there is no desktop-contract op
  for it (`src/shared/desktop-contract.ts` has `sessions.variables.*` and
  `auth.*` only). §20.2 records that alternative as the better end state.

### 11.4 The human's password: why the agent's `console_read` cannot see it

This is the part most likely to be claimed and not proved, so it is stated as
mechanism:

1. **Keystrokes are not recorded.** The human's typing goes
   renderer → `console-input` IPC → main → the pty master. It is never appended
   to the byte log, never persisted, never logged, never captured. (The *byte
   log* is pty **output**, and its name says so.)
2. **The emulator's grid holds only what a program wrote.** With echo off — the
   default for `sudo`, `read -s`, `ssh`'s password prompt, and anything using
   `termios` — the tty does not echo, so the bytes never enter the pty's output
   stream, so they never enter the record, so `console_read`, `console_status`
   and `console_screenshot` cannot contain them. **The agent sees what a person
   looking at the screen sees**, which is the correct ceiling.
3. **The mirror cannot leak it either**: the mirror is fed only from the same
   record/log path (§10.3), and it holds no input state.
4. **The secure-input span**, for the cases where "echo off" is not enough
   because the *human* wants certainty: `console_secure {surface, on: true}`
   (togglable from the pane with a visible lock marker) makes
   `console_read` (either mode) and `console_screenshot` fail with
   `secure_input_active` for as long as it is on, stops appending to the byte log
   for that window (bytes already retained are **kept**, and §7.2's cap and GC
   still age them out; what the span suspends is *new* retention, not the
   history that is already on disk — otherwise turning the toggle on and off
   would be a way to erase the log, which is the opposite of what it is for),
   and suppresses the surface's completion notification. It is the
   design's explicit "do not capture this" switch, and its boundary is that it
   is **advisory against a cooperating agent**: a session that ignores the typed
   refusal has no other lever, because the app cannot distinguish "a read that
   should have been refused" from any other read. What makes it more than
   advisory is that the refusal is typed at the *tool* seam, so the model's own
   loop records the refusal in the transcript — and §11.5 says what that is worth.

### 11.5 The limits, stated as limits

- **A program can echo a secret.** If something prints the password (a script
  with `echo`, a `set -x` trace, an app that logs its own config), the value is in
  the record and `console_read` returns it — and it *should*, because a human
  looking at that screen also sees it. The app cannot know a printed string is a
  credential.
- **What is already in place covers a good part of that anyway**: the harness's
  shape pass (`local_operator/redaction_shapes.py`) sits on the *tool/result
  seam* — which is where `console_read`'s return value lands — and masks a value
  because of how it is spelled (a DSN, an `AWS_SECRET_ACCESS_KEY=` line, a PEM
  block, an issuer-prefixed token). Its stated residual is the honest one: "an
  opaque value with none of these spellings around it… a bare tenant id, a
  session cookie pasted without its header" is not caught.
- **A screenshot is pixels.** Redaction cannot read them. The control for a
  screenshot is §11.4's echo argument plus the secure span, not the redactor —
  and this document says so rather than implying coverage.
- **A literal credential the agent chose to type** is in the transcript before
  anything can redact it. The prevention is the tool description's rule plus the
  `secret_ref` alternative being *easier* than the literal (it is one parameter,
  and the model does not have to know the value).
- **The agent can read a file that contains the secret.** This is true of `bash`
  today and is not made worse by a terminal.

### 11.6 What the app must never do with a surface's content

Three prohibitions, because each is a plausible "helpful" feature:

1. **Never write terminal output into the app log.** `src/main/browser/log-capture.ts`
   exists for browser console output and takes a bounded ring for exactly this
   reason; the console's log lines are events (`surface created`, `exit 3`,
   `resized 100x30`), never content.
2. **Never put terminal content into a notification body** (§12) — the banner
   names the session and the surface, never the last line printed.
3. **Never persist keystrokes** (§11.4.1) — and the *test* for this is a
   byte-level assertion on the artifact (§19.3), not an inspection of the code.

---

## 12. R14 — completion, the blip, the notification and the click

### 12.1 The completion ladder

First match wins, and each rung says what it cannot see:

1. **The surface's process exited** — `node-pty`'s `onExit` in main. Always
   available, exactly once per surface, carries the exit code. This is the only
   signal that is always true.
2. **A command finished inside a persistent shell** — an **OSC 133** semantic
   mark parsed from the surface's own byte stream in main
   (`ESC ] 133 ; D ; <exit_status> ST`). This gives the per-command blip R14
   asks for, and its limits are named rather than hidden:
   - it requires the shell to emit prompt marks (bash/zsh need the integration,
     fish emits them natively, and whether the operator's shell does is probe
     **P7**);
   - the parse is a byte scan in the log writer, not an emulator feature — the
     headless build exposes no OSC-133 event;
   - a program can spoof the sequence, whose worst case is one spurious blip.
   The app ships a small, documented `console shell-integration` snippet the user
   may source (and never writes to their dotfiles, §1.3).
3. **Output quiescence — rejected.** "No output for 800 ms" is true of a `sleep`,
   of a TUI waiting on a keystroke, of a build between link steps, and of a
   process that has died. A completion signal that is wrong in the common case is
   worse than one that is silent, and the *measured* cost of guessing wrong here
   is a notification the user learns to ignore.

### 12.2 The blip (in-app)

- **What**: an attention mark on the console trigger in the chat header — a dot,
  not a count, following the canvas button's own discipline ("A dot, not a
  count… here the only job is to say 'there is something' before the user has
  opened it") — plus a brief pulse on the surface's row in the pane, and a bold
  dot on the surface's title in the pane's own list.
- **When cleared**: when the pane is displayed *and focused* on that surface, the
  same predicate family the notifier's rung 1 uses
  (`docs/DESKTOP_API.md:1314-1342`: the **visibility** predicate, never
  "could a banner reach them"). An uncleared blip survives a session switch and
  an app relaunch (it is a mark on recorded history, not on a live process).
- **Colour**: `inkMuted` for the resting dot, `accent` for the pulsing one — the
  canvas button's dot is `ink-muted` for the same reason ("nothing is unread, and
  the accent is spent on actions the app is asking for"); here something *is*
  unread, so the accent is earned.

### 12.3 R14's native notification, and the click that restores context

- **Raised by the app's existing `DesktopNotifier`**, through one new public
  entry on it that routes into the single `show(...)` choke point
  (`desktop-notifier.ts:1155`) rather than constructing its own `Notification`.
  Extending the one notifier is the same argument as extending the one browser
  host: a second banner raiser would duplicate the TTL dedupe map, the window
  state, the raise policy and the click path — four things this feature has no
  opinion about.
- **Eligibility**, in order, mirroring the backend's ladder so the two cannot
  disagree about the same user:
  1. the app is in `headless` → **nothing** (`desktop-notifier.ts`'s own delivery
     gate; `AGENTS.md:758-762`);
  2. the console pane is displayed on that surface and the window is focused →
     **no banner**, the blip is the whole signal;
  3. otherwise → the banner.
- **Body**: session name + surface name + `exited with code N` (or `finished`).
  Never terminal content (§11.6.2).
- **Click**: a click calls the existing `host.reopen(sessionId)` when there is no
  window (the reviewed fix at `desktop-notifier.ts:1189`), then delivers
  `{session_id, surface}` to the renderer. The payload gains **one additive
  field** on a channel that already exists (`desktop-open-conversation`,
  `preload/index.ts:108-124`), which already admits an explicit `null` for the
  digest case — so a new optional `surface` is additive in both skew directions.
  The renderer then: navigates to the session, claims the right slot for the
  console pane, and selects that surface. A surface that no longer exists (the
  app restarted) selects the session's console pane with its recorded history and
  the "ended" state (§7.3) — an honest landing, not a no-op.
- **One banner per completion**, claimed before delivery, in the notifier's own
  dedupe map (`:1101 claim(key)`), keyed on `(surface, exit_epoch)`. `exit_epoch`
  is the exit *generation* of the surface: the retention layer's counter, kept
  with the surface's record and persisted beside `exit_code` (§7.3), incremented
  once per `process_exited`, and returned to a caller by `console_status`
  (§10.2). This notifier is a *reader* of it, never its owner — so a retained
  surface that is replayed and run again under the same id gets a new epoch and
  can therefore banner again, while a re-notification for one exit cannot. It is
  the same "claim-then-deliver" discipline the backend's `/notified` uses, so the
  blip and the banner can never both be counted twice.

### 12.4 The e2e exemption path (no new switch)

- `LOCAL_OPERATOR_NO_NOTIFICATIONS=1` already reaches the app
  (`src/main/backend/notification-launch.ts:75`), the Python side
  (`tui/notify.py:115`, `:404`), and every child `scripts/run-desktop-tests.mjs`
  spawns. **The console path honours that one switch** — it adds no second one,
  and `scripts/notifications-off.test.mjs`'s enumeration of spawn sites is the
  test that will catch a new site that forgets.
- The console adds **one** test-only surface, which is not a switch but a state:
  a **blip must be assertable without a banner**. `console-state` exposes the
  attention mark as data, so an e2e cell asserts the blip while notifications are
  off — that is the point of separating them.
- In `headless` the banner is not delivered at all, so a capture run needs no
  exemption beyond the window mode it already names.

---

## 13. R7/R17/R10 — the capture contract

### 13.1 `console_read`: text, from the record, with no view

- `mode: "viewport"` returns the visible rows — `translateToString` over
  `buffer.active.viewportY..baseY` — which is what "the terminal's current screen"
  means and what an agent needs to decide where a TUI is.
- `mode: "scrollback"` returns a windowed slice (`start`, `count`) of
  `0..baseY`, so an agent can page history without pulling 5,000 lines. Both
  carry `cursor`, `cols`, `rows`, `truncated` and `live`.
- **stdout and stderr are one stream** (R17's "read output (stdout and stderr)"):
  that is what a pty *is*, and pretending otherwise would require a second
  channel the program would not use. The tool description says so, so the model
  does not look for a way to separate them.

### 13.2 `console_screenshot`: pixels, and which pixels

Three cases, and the result always names which one it was:

| case | mechanism | `rendered` |
|---|---|---|
| the pane is displayed on this surface | `webContents.capturePage()` of the app's own window, cropped to the pane's reported rect | `"displayed"` |
| the pane is closed / this is an agent-only surface | a **capture view**: a hidden renderer fed the surface's record (replay), at the surface's grid, with the **DOM renderer** pinned, photographed with `capturePage()` | `"offscreen"` |
| the record is `live: false` (post-relaunch) | the same capture view over the replayed history | `"offscreen"` |

**Three measured traps shape this path, and revision 1 did not know about any of
them** (all from the compatibility spike, §0.4):

1. **A WebGL canvas cannot be read back.** `canvas.toDataURL()` on the xterm
   WebGL canvas returned **blank white**, byte-identical at 26,791 B with
   `preserveDrawingBuffer` both `true` and `false` — so nothing in this feature
   may take pixels from a canvas' own bitmap. `capturePage()` is a **compositor**
   capture and is not the same mechanism, which is why the displayed-pane row
   above still works with a WebGL pane.
2. **`capturePage()` on a hidden window can return a stale first frame.** Measured
   on a hidden `BrowserWindow`: the first capture came back blank/stale at
   9,866 B, the next was correct at 27,869 B. **Every capture therefore asserts a
   non-blank frame and retries exactly once**, and the retry is part of the
   contract rather than a hopeful `setTimeout`.
3. **The DOM renderer captures correctly on the first attempt** — which is why the
   capture view pins it. It is the slower renderer, and for an offscreen
   single-shot reconstruction that is the right trade: the pane (interactive,
   watched) may use WebGL; the capture view (one frame, unattended) may not.

- `rendered` is what keeps this honest: an offscreen frame is a faithful
  *reconstruction from the record* and not a photograph of a live screen, and a
  consumer that cares about that difference can see it.
- **Never `screencapture`.** macOS `screencapture` photographs the frontmost
  window and requires the focus theft the window modes exist to remove
  (`AGENTS.md:810-816`); `capturePage` is the app photographing itself, which is
  also what `docs/agent-driver.md` builds on.
- **The capture view is the same page as the pane** (one component, §13.3), so
  the two paths cannot drift in font, theme or renderer.
- The PNG's magic is re-verified on the Python side before the tool returns it —
  the same rule the browser tool applies to its screenshots.

### 13.3 The capture view, and the one-at-a-time rule

- Created lazily, on the first offscreen capture for a surface, and reaped after
  a short idle (default 30 s) — a `WebContentsView` that is never attached to a
  window is never laid out, painted or focused, which is the property the
  cookie-jar target already relies on (`index.ts`, §2.4).
- **The capture view is a renderer process**, so its cost is real and bounded:
  at most **one** capture view at a time app-wide (`console_capture_full` is a
  typed refusal, mirroring cmux's `input_queue_full` honesty), sized to the
  surface's grid, and torn down by pid-exact teardown (§16.2).
- **The renderer is pinned to the DOM renderer, and the pin is asserted** — a
  renderer choice that could silently become WebGL would turn every offscreen
  frame blank, and a blank frame is exactly what an unbounded capture retry would
  hide. The capture view therefore reports which renderer it is using in the
  capture result's log line, and a test asserts the DOM one.
- **The frame is a function of `(record bytes, grid, theme, font, renderer)`**, all
  of which are pinned, so a frame is reproducible — which is what makes it
  evidence (§19.1's P10 requires the determinism check, because a frame that
  changes run to run cannot be a before/after). The retry above is compatible
  with that only because a retry re-captures the *same* settled state: the frame
  must be re-fed from the record, not re-read from a live surface.

### 13.4 R10/R18 — the co-pilot cell, and reading a surface the user opened

- The co-pilot flow is: the user opens a surface (or the agent creates one with
  `reveal: "session"`), the human types into it, the agent reads it with
  `console_read`, sends keys with `console_keys`, and captures with
  `console_screenshot` — **on the same surface, at the same time**, because
  there is exactly one pty and one record per surface. §19.4 makes this a QA cell
  with both actors interleaved, because "both can attach" is the claim most
  likely to be true in a demo and false under a race.
- R18's "the user says they ran something" flow: `console_list` shows the
  user-opened surface with `origin: "user"`, its `command`, `cwd` and
  `session_id`; the agent reads it. The `con:` prefix and the pane's visible
  marker (§6.5) are what let the agent be sure this is the Local Operator console
  and not another terminal on the machine — and the tool description says a
  *different* terminal is not readable by this tool, so the agent reports that
  instead of guessing.

---

## 14. R8/R16/R17 — the tool: gating, description, trace, discouragement

### 14.1 The gate, as one `createIf` entry

`registry.py:32 TOOL_BUILDERS` gains one row,
`"console": lambda _context: builtin.build_console_tool(_context)`, and
`DEFAULT_TOOL_NAMES` (`:70-97`) gains `"console"` in a position that keeps the
array's order stable for the prompt cache (`:121-128`). The builder returns
`None` unless **one** file-only predicate holds:

```python
def build_console_tool(context: ToolContext | None) -> AgentTool | None:
    if not ui_console_advertisable():      # FRESH-or-STALE, file only, never a socket
        return None
```

where `ui_console_advertisable` lives in `local_operator/ui_console/` (a sibling
of `ui_browser/`, same shape: `state.py` reading the shared record for the new
`console` capability field, `backend.py` subclassing `HostClient`). Three
properties, each inherited rather than invented:

- **File-only and synchronous**, because this runs while constructing every
  session and "a socket round-trip there would tax startup for every session on
  the machine" (`ui_browser/backend.py:43-49`).
- **The weaker `advertisable` commitment**, because "advertising only promises
  the agent can ASK" and a host whose heartbeat writer died must not hide the tool
  (`ui_browser/state.py:150-157`).
- **Never raises** — an unreadable record degrades to "no console", which is the
  honest answer and the one `createIf` exists for.

### 14.2 Should it be hidden? No — and this is a deliberate difference from `bash`'s neighbours

`hidden=True` means "not listed in the prompt inventory" (`prompts_api.py:365`)
and its one shipped use is a transport `AgentTool` (`reply_channel.py:143`).
R16 requires the opposite: the tool appears in traces, the model knows it exists,
and it is *discouraged* rather than concealed. A concealed tool cannot be
discouraged — it can only be absent, which would fail R7 (an agent asked to test
a TUI must know it can).

### 14.3 The description: short by policy, with the playbook in a guide

The description is code and is reviewed as code. Two rules govern its length and
its content, both read from the browser tool at `lop@e65548e9` rather than
invented here:

1. **The string carries only what a model needs to CHOOSE the tool and call it
   correctly** (`builtin.py:12830-12837`), and the per-method detail lives in
   **parameter descriptions** and in a `guide://console` playbook — the same
   split the browser tool now uses, and the reason its download/upload sentence
   is one clause "rather than the four the design first drafted".
2. **`scripts/bench_context_budget.py` measures the whole surface and CI runs it
   for the `context-budget` scope** (`scripts/ci_scope.py:178-182`), so the
   budget is enforced, not advisory. Probe P8 (§19.1) reports the delta.

Draft, at the length the convention implies:

```
Drive a real interactive terminal inside the Local Operator desktop app: a pty
running a command, with a real terminal grid, that keeps running and keeps its
output while its tab is closed. Use it for things `bash` cannot host — a
full-screen TUI, a REPL, an installer, an interactive prompt — and NOT for
ordinary commands: `bash` returns output directly, cannot wedge on a prompt, and
cannot leave a process running behind your turn. The surface handle names this
host (`con:`). `list` shows surfaces the USER opened too; read those rather than
asking them to repeat their output. Playbook: `guide://console`.
```

Five things that text is doing, so a reviewer can check them:

1. **Differentiation from `bash` is a sentence about consequences** ("returns
   output directly, cannot wedge, cannot leave a process running"), not a
   preference — and it is the *second* sentence, where a model choosing a tool
   will read it.
2. **It names the non-interactive case as `bash`'s**, which is R8's posture
   exactly (`web_search`/`web_fetch` before `browser`).
3. **It gives the one fact that stops the `playwright`-class mistake** — the
   handle names its host — so an agent that is asked about "the terminal" does
   not go looking for another one.
4. **It names `list`'s user-opened surfaces**, which is R18's whole case, in one
   clause, and defers the mechanism to the guide.
5. **It routes the rest to `guide://console`**, whose content is enumerated in
   PR C: the method table, the named key set, the stdout/stderr-is-one-stream
   fact, the `secret_ref` rule (§11.3), the `ask`-before-sudo rule (§11.1), and
   the two honest limits (a program that echoes a secret; a screenshot is
   pixels).

**Parameter descriptions carry the per-method detail**, exactly as the browser's
do, and three of them are load-bearing rather than descriptive:
`console_read`'s `mode` (viewport vs scrollback), `console_keys`'s `keys` (the
accepted names, so an `unknown_key` refusal is avoidable), and
`console_create`'s `reveal` (whose three values and their focus rule are §10.4).

### 14.4 R16 — the trace: label, icon, and the two glyph tables

- `AgentTool(name="console", label="Console", …)` — the label is what the TUI's
  tool card and the UI's trace row render.
- `local_operator/tui/glyphs.py`: `"console": "\uf108"` in `NERD_TOOL_ICONS` and
  `"console": ">"` in `PLAIN_TOOL_ICONS`, both distinct from `bash`'s
  `"\uf120"`/`"$"` — a *different noun* for a different thing (the app's console
  vs the shell), which is the distinction the table's own comments care about
  (`write`/`edit` share a pencil because they share a meaning).
- `src/renderer/src/features/chat/components/trace/tool-glyphs.ts`:
  `console: SquareTerminal` (a terminal *in a frame* — the app's surface, vs
  `bash`'s bare `Terminal`), with a story asserting the two are distinguishable
  at the rendered size.

### 14.5 The prompt note, and the three-state problem the browser note solved

The browser has a *pair* of notes because one string would have to assert
something false about the host (`prompts_api.py:379-420`: `_NO_BROWSER_NOTE` for
a host with no browser at all, `_ROLE_HAS_NO_BROWSER_NOTE` for a session whose
*role* was not granted it). The console needs the same pair, for the same reason:

- `system.md` gains `{{#if has_console}}` … `{{/if}}{{#if no_console}}` around a
  console paragraph, mirroring `system.md:264`/`:287`; `prompts_api.py` gains the
  answer for a session that has no console tool, and the flag is decided the same
  way `host_has_browser` is (`render_tool_inventory_block`, `:304-333`).
- The `no_console` text is a **three-line prohibition, not a setup playbook**: the
  console is not installable by an agent (it needs the app), so telling the model
  to go and set it up would be an invitation to the exact dead end the browser's
  playwright incident was.
- The `has_console` text carries the *discouragement*, so the posture reaches the
  model in the system prompt as well as in the tool description — belt and braces,
  because the description is read once per call and the prompt is read every
  turn.
- `prompts_api.py`'s existing flag-completing helper for the browser
  (`:185-246`, which exists because `{{#if}}` has no `else` and shipping
  `has_browser=True` with a defaulted `no_browser` shipped *both* sections)
  gains the console pair, including the "both true" refusal. This is the single
  most bug-prone part of the change and it is a copy of a solved problem rather
  than a new solution.

### 14.6 The discouragement's enforcement, stated honestly

Prose is not enforcement. Two levers actually change behaviour, and one that
looks like a lever does not:

1. **Subagent role seeds.** The role seeds (`local_operator/agent_seeds/architect.md:6`,
   `manager.md:6`, `reviewer.md:6`, `scout.md:6`, and `manifest.json`'s per-seed
   `tools` lists) constrain a child's tool set, and the browser is deliberately
   absent from all four read-only roles — a fact `prompts_api.py:489-493` records
   as a defect it had to design around. **Recommendation: `console` joins that
   exclusion set**, so an architect/reviewer/scout/manager gets no `console_*` at
   all unless a future need appears. This is enforcement rather than advice.
2. **The turn-boundary inventory** (`session.py:14012 _reconcile_web_tools` and
   `refresh_tools`, `:6120`) is where a *capability* appears and disappears, not
   where a preference is expressed — the tool is gated on presence of the app, and
   it must not be used to implement "discourage" (a tool that vanishes
   mid-session is worse than one that is explained).

**And the lever that is not one, stated so nobody counts on it:** the approval
tier records *intent*, and the browser tool's own comment says why that is not
protection — "today the gate is ONE callback for both tiers and
`tool_approval_mode: auto` / `--yolo` installs no gate at all, so the tier
records intent and future-proofs a tier-sensitive host — it is NOT the
protection" (`builtin.py:12864-12872`). A session in `auto` mode will not be
prompted for `console_create`, so this design does not claim an approval gate as
its discouragement, only as its tier bookkeeping.

---

## 15. D11 — failure and absence semantics, with the copy each actor reads

Every cell names (a) what the model reads, (b) what the user sees. A missing cell
is how a feature produces "it just didn't work".

| condition | model sees | user sees |
|---|---|---|
| app not running | **no `console` tool in the inventory**; the prompt carries the `no_console` note | nothing (there is no console to open); the pane control is absent, not disabled-with-a-tooltip |
| app running, console feature off (`LOCAL_OPERATOR_UI_CONSOLE_HOST=0`) | same as above — the tool is absent because `console: false` in the record | the pane is unavailable; Settings says why in one line |
| app running, `node-pty` failed to load | tool **absent**, and the app log carries one line naming the failure; `console: false` in the record | the pane shows the load failure with the remedy, because a broken native module is a packaging bug the user can report |
| proto skew | `proto_mismatch` with the update remedy (existing copy, UI variant) | the pane shows "update Local Operator" |
| old host, console methods unknown | `unsupported_method` naming the app | same |
| surface id unknown | `surface_unavailable`, naming the handle and how many exist | the pane's list is authoritative; a stale click lands on the empty state with a sentence |
| surface belongs to another session | `surface_not_owned` | the pane's list shows only this session's surfaces |
| the surface's process exited | `process_exited` with the exit code and `live: true/false`; `read` still works | the pane shows the ended state with the exit code and the history |
| app restarted | `console_read` works and returns `live: false`; `create` makes a new one | "this terminal has ended" over the recorded history (§7.3) |
| input queue full | `input_queue_full` with the accepted byte count; retry guidance | nothing (this is an agent-driven case) |
| unknown key name | `unknown_key` listing the accepted set | nothing |
| grid out of range | refused with `invalid_grid` and the clamp it would have applied | the divider's own clamp is the user-facing bound |
| read/screenshot while secure input is on | `secure_input_active` | the lock marker is visible, so the refusal is explained by the screen |
| capture view busy | `console_capture_full` | nothing |
| pane closed, agent captures anyway | **works** (§13.1/§13.2) — `rendered: "offscreen"` | nothing |
| app closed *while a surface was running* | `ui_host_unavailable` from the transport, with copy that names the app and says the surfaces ended with it | the pane is gone with the app; next launch shows the recorded history |

Two rules the table encodes and prose does not:

1. **Absence is reported as absence, never as failure.** The house rule is the
   browser tool's: a host that cannot do the thing offers no tool (`builtin.py:9067-9070`),
   because "advertising a tool whose every action errors is worse".
2. **Every mid-flight failure is typed**, so a caller never substring-matches a
   message (`protocol.py:377-383`'s own rationale for `OWNER_REFUSED`).

---

## 16. R21/R11 — window modes, evidence discipline, and the review gates

### 16.1 R21 — window modes, per rig

| rig | mode | why |
|---|---|---|
| `scripts/renderer-driver.mjs` scenes for the pane | `headless` | the supported driving path (`docs/agent-driver.md`) |
| the console host's own proof rig | `headless` | a launch that captures nothing needs no window |
| a frame that must show focus-dependent rendering (caret, `:focus-visible` ring on the close button) | `inactive` | `AGENTS.md:744-748` |
| any cell about the notification or the blip's cleared state | **`inactive`, never `headless`** | native banners are suppressed entirely in `headless` and the notifier's focus gate reads `visible && focused`, which a headless run never is (`AGENTS.md:749-762`) |
| the release-smoke of a packaged build | `headless` | packaging evidence, not a UI frame |

**And the assertion behind all of them:** nothing a capture does may raise the
app. The measurement is outside the app, by pid, over a dense sample, with the
control run in `normal` — the discipline and the numbers are already recorded
(`AGENTS.md:717-723`), and §19.2 makes it a cell for this feature.

### 16.2 Evidence rules this feature inherits, verbatim

- **Capture from inside the app** (`capturePage`/CDP), never `screencapture`
  (`AGENTS.md:810-816`).
- **When an Electron binary is launched by a rig, it gets an absolute app path
  and its process group is reaped by exact pid on exit** — the
  `app-tree-teardown.mjs` pattern — and a visible window is only ever produced by
  a single self-contained command that launches, captures and reaps before it
  returns.
- **Before/after frames, and a look at the settled frame** — a single "looks
  fine" hides what a pair makes obvious; consecutive frames when anything settles.
- **A rig's Chrome or Electron never reaches the keychain** and never prompts
  (the `--use-mock-keychain` / `GIT_CONFIG_SYSTEM` / `GH_CONFIG_DIR` rules).
- **Evidence goes on the PR, never into the repository**, except the committed
  evidence tree the repo already keeps (`docs/evidence/`, `pnpm check-evidence`).

### 16.3 R11 — what the design round looks at, and what the UX round walks

- **Design round (D-findings)**, on rendered artifacts: the pane in all four
  states (loading/empty/error/populated) at 100×30; the **12-theme torture grid**
  (§9.3); the pane header against its three siblings (the four panes must read as
  one slot); the blip's resting and pulsed states on the header control (and
  beside the browser badge, which is the wide case that earned the 12px); the
  ended state; the secure-input lock; and the two before/after pairs the change
  *is* (pane closed → open; surface running → exited). Numbers behind the frames:
  the pane's reported content rect vs the view's bounds, the grid main decided,
  whether a scrollbar appeared, and the divider's clamp behaviour at the window
  floor.
- **UX round (U-findings)**, walking the flow on the real app: creating a surface
  from the header control, typing in it, creating a second surface from the tool,
  switching conversations with one running, closing the pane and reopening it
  **and seeing the output from while it was closed**, the notification click
  landing on the right surface, and the secure toggle. The two flows most likely
  to be wrong and least likely to be caught by stills: *close-then-reopen* and
  *switch-away-then-back*.
- Both rounds are recorded on the PR in the standard form (`### Design review —
  round N`, `### Agent review — round N`), and a UI PR merges only when both are
  clean on the current head.

---

## 17. D12 — the PR split, the merge order, and what each PR freezes

### 17.1 The split

| PR | repo | content | what it freezes |
|---|---|---|---|
| **0** | `local-operator` | `docs(design): the console tab` — **this document**, alone | nothing in code; it is the citation target for A/B/C |
| **A** | `local-operator-ui` | `feat(console-host)`: `src/main/console/` (surface registry, `ConsoleHost`, the emulator seam, the byte log, retention/persistence, the OSC-133 scanner), the pty dependency + its packaging changes (`asarUnpack`/unpack-dir, the **chmod 0755 of `spawn-helper`** and its assertion in the packed tree, the foreign-platform prebuild prune, `pnpm-workspace.yaml`, `check-runtime-deps` allowlist), the `console_*` methods on the existing RPC host, the record's console fields, the `console_*` renderer IPC namespace, the push channel, main-process tests, and a host proof rig | the emulator seam, the record fields, the method vocabulary and its params, the failure taxonomy, the grid-ownership rule |
| **B** | `local-operator-ui` | `feat(console-pane)`: the fourth pane (component, header, divider, empty/ended/secure states), the mirror (`@xterm/xterm` + addons, replay-then-stream), `terminal-theme.ts` + the contrast block, **the DOM-renderer capture view with its non-blank assert and one retry**, **the bundled face's OFL notice and licence text (§6.6)**, the blip, the notification entry + click payload field, stories, designer + UX rounds (including the box-drawing join check) | the pane's chrome and states, the theme mapping, the blip/notification UX, the capture path |
| **C** | `local-operator` | `feat(console-tool)`: `local_operator/ui_console/` (state + client), `build_console_tool`, `registry.py` + `DEFAULT_TOOL_NAMES`, the description and its parameter descriptions (§14.3), the prompt notes and their flag completer, `secret_ref`, the redaction registration, the glyph tables **plus the console's own `terminals.py` predicate and its `glyphs.py` clause (§6.6)**, `scripts/bench_context_budget.py`'s measurement in the PR body, `docs/CONSOLE.md` **and the `guide://console` playbook**, the e2e cells | the tool's schema and description, the gate, the prompt notes, the harness contract |

**Merge order: 0 → A → B ∥ C.** The doc first because A/B/C cite it and because a
design doc that lands after the code it describes is documentation, not a
contract. A before B because B cannot be exercised without a surface and A is the
risky half (a native dependency, a packaging change, a new wire namespace). C can
land in parallel with B, because the tool's tests drive the wire and not the pane.

### 17.2 Why A and B are not one PR (the brief asked)

Three reasons, and the counter-argument is stated so it can be weighed:

1. **Different review gates.** A needs packaging, signing and native-dependency
   review (`check-packaged-closure`, `check-runtime-deps`, the entitlements
   question, the Linux toolchain) and a rig; B needs a designer round on twelve
   themes and a UX round on two flows. One PR would carry four review streams
   whose findings would then have to be batched into one remediation commit
   anyway — and a packaging defect would block a visual fix and vice versa.
2. **A is independently verifiable.** A's deliverable is observable with curl and
   a raw byte transcript: create a surface, read it, resize it, capture it
   offscreen. That is a complete story without a single pixel of chrome.
3. **Different rollback shapes.** If the native dependency turns out to be
   untenable on one platform (§18, risk 1), A is the PR that would have to change; B is
   unaffected either way.

The honest counter-argument, which the operator should hear: **A alone is not
useful to a human** — there is no UI to see, so "merging A" ships an invisible
capability and a new native dependency into the app for a feature nobody can
reach. That is why §19.6's gate for A includes a *proof rig* and why A's PR
description says plainly that the feature is not user-visible until B lands. If
the operator prefers a single user-visible PR, the fallback is to keep the split
but merge A and B in one window (§17.3's release mechanics) — **not** to make one
PR, which would put a native-dependency change behind a design round.

### 17.3 Effort, and where the cut list is

Order-of-magnitude, from the module sizes read in §2 and the browser's own
delivered sizes (`src/main/browser/*.ts` is 8,495 lines across 22 files for a
comparable surface).

| PR | estimate (touched lines) |
|---|---|
| 0 | ~1,500-2,000 (this document) |
| A | ~2,200-3,000 + packaging |
| B | ~1,600-2,200 + design/UX |
| C | ~900-1,400 + docs |

**Cut list, in this order if time is short:** (1) the capture view (offscreen
screenshots) — `console_read` covers text, and a *displayed* pane can still be
photographed; keep the `rendered` field so the gap is visible; (2) the
scrollback *paging* params (ship `viewport` + full scrollback); (3) the
shell-integration snippet and per-command OSC 133 blips (keep exit-based
completion); (4) persistence across relaunch (keep in-memory history).
**Do not cut:** the grid-ownership rule (§8), the byte log (§7.1 — it is what
makes closed-pane reopen and offscreen capture correct), the secret path (§11.3),
the e2e kill-switch reuse (§12.4), the `con:` provenance (§6.5), or the
window-mode rules (§16.1).

---

## 18. Risks and failure modes

Ordered by expected cost.

1. **The native dependency breaks packaging — and the failure modes are now
   measured rather than imagined.** Three traps, all from the spike:
   (a) the published tarball ships `prebuilds/darwin-arm64/spawn-helper` with mode
   **0644** and nothing chmods it, so the first spawn dies
   `FATAL Error: posix_spawnp failed. at new UnixTerminal (lib/unixTerminal.js:92)`;
   (b) a **fully-packed `app.asar`** resolves `pty.node` and still fails to spawn,
   because `helperPath` is rewritten to `app.asar.unpacked` which does not exist
   until the prebuilds are actually unpacked (`--unpack-dir
   node_modules/node-pty/prebuilds` works); and (c) **no install script will save
   us** — npm 11.17's install-script gate skips `scripts/prebuild.js`, and this
   repo's own gate is pnpm 10's `onlyBuiltDependencies` (§2.11), the same class.
   On top of those: `node-pty` ships **no Linux prebuild** while the repo builds
   Linux x64 artifacts, the tree carries **58 MB of unused win32 prebuilds and
   `.pdb` files**, and the app is signed, hardened and notarized — so the unpacked
   helper must be signed with the same identity.
   *Symptom if this is got wrong:* the app works in development and ships broken,
   or the Linux artifact fails weeks later at release time.
   *Mitigation:* all of it is PR A, reviewed as one set — chmod 0755 the helper and
   **assert the mode in the packed tree**, unpack the prebuilds, prune the foreign
   platforms, and prove it with §19.1's P11 (packed, signed, spawning) rather than
   with a dev run. *Accepted residual:* a Linux prebuild regression upstream is a
   future surprise; the mitigation is that the build fails loudly rather than
   silently.
2. **Two terminal instances diverge, so an agent's read and the human's screen
   disagree.** This is the risk the "exactly one emulator" requirement exists for,
   and the honest statement is that this design has **one authority and a
   mirror**.
   *Mitigations, all structural:* main is the only reader of the pty (the pane
   never opens it); main owns the grid (the mirror cannot reflow itself); the
   mirror is fed the same bytes from the same log; both instances are the same
   pinned version from the same lockfile; and the mirror answers no read.
   *Evidence:* a conformance test (§19.3) that feeds a recorded stream to both
   and compares scrollback text, cursor position and the full grid cell-for-cell
   after every chunk.
3. **A byte flood starves the app's main thread.** `yes` at 100 MB/s through a JS
   emulator on the main thread would make the window unresponsive.
   *Mitigation:* a bounded read per event-loop turn, a coalescing timer, and
   `input_queue_full`-style backpressure on the pty read side; §19.1's probe P3
   measures main-thread responsiveness under a flood. *Escape hatch:* §3(d)'s
   `utilityProcess` behind the same seam — this is the risk that would justify it.
4. **A credential reaches the model anyway.** §11.5 lists four ways; the design's
   controls are the `secret_ref` path, the existing shape/redaction seam, and the
   prohibition in the description.
   *Accepted residual, named:* a literal credential the agent chose to type is in
   the transcript; a program that echoes a secret puts it in the output; a
   screenshot is pixels.
5. **The pane's history is mistaken for the truth after a restart.** A user (or
   an agent) could assume a surface is still running. *Mitigation:* `live: false`
   in every read, the ended state in the pane, and the §7.3 rule that the copy
   says "ended" rather than staying silent.
6. **Notification noise.** A build loop's per-command completions could produce a
   banner per step. *Mitigation:* §12.3's eligibility (rung 2 suppresses while
   the pane is displayed and focused), one banner per completion via the existing
   dedupe map, and OSC-133 blips distinguished from process-exit banners.
7. **The emulator seam has to be swapped later** (a VT-correctness failure on a
   TUI, or a ghostty-parity requirement). *Mitigation:* §5.4's interface is the
   only thing the rest of the code sees; the swap is one file plus the mirror's
   import, and the tests in §19.3 are written against the interface so they
   survive it.
8. **Disk growth from retained history.** *Mitigation:* §7.2's caps, the 30-day
   default, and a GC pass that reports what it removed.
9. **The font is tofu, or the box frames do not join.** Measured: this machine has
   **no Nerd Font installed at all**, and PUA glyphs rendered as replacement boxes
   in every stack the spike tried; Menlo's box-drawing glyphs also visibly fail to
   join at terminal size. The pane is the one place this is solvable, because the
   app bundles a Nerd-patched face and CSP (`font-src 'self' data:`) forbids any
   other — so the mitigation is that the face is bundled, the design round checks
   the glyph table and box continuity in the torture stream (§9.3), and the PR
   ships the face's **OFL notice and licence text, which the repository currently
   does not carry** (§6.6). *Accepted residual:* outside the app, in the user's own
   terminal, this remains their environment and the TUI's tri-state gate is the
   only lever — nothing in this feature can repair a terminal's fonts.
10. **The tool's schema cost.** Every core tool ships its schema on every request
   (`AGENTS.md`'s ladder). *Mitigation:* the tool is `createIf`-gated, so a host
   without the app pays nothing; §19.1's probe P8 runs `/context` with and without
   it and puts the measured delta in the PR. **The honest expectation is that the
   gate does not save the operator anything on their own machine** — their app is
   usually running — so the number belongs in the PR rather than in an assumption.
   **Spike context, not a substitute measurement:** the whole xterm + addons
   *renderer* bundle is 523.31 kB JS (138.72 kB gzip) + 3.52 kB CSS, which says
   nothing about schema tokens — P8 measures those.

---

## 19. R6/R15 — the test, evidence and QA plan

### 19.1 The empirical probes, named as experiments

Each is a cell an independent QA pass can run and report PASS/FAIL/BLOCKED with
its actual output. **None has been run to completion**; they are the plan. Two
rows are partly done, and both were done by the compatibility spike rather than
by the harness or the emulator's own suite: **P1** was measured end to end, and
**P4** and **P9** are partly measured. Each of the three says so in its own row,
and §0.4 counts the same way — a reader who takes the lead sentence alone must
not conclude that §5.5's D1 decision was made without evidence, because it was
not.

| probe | question | exact experiment | settles |
|---|---|---|---|
| **P1** | Does `node-pty`'s prebuilt binary load under the pinned Electron (44.3.0) on this arm64 host, with no `electron-rebuild`? | **MEASURED — PASS** by the compatibility spike: the same 85,496 B `pty.node` loads under node 26.5.0 (ABI 147) and Electron 44.3.0 (ABI 149); `/bin/zsh -f -i` spawned, prompt read back, echo round-tripped, 24-bit/UTF-8 byte-exact, `resize(100,30)` reflected in `stty size`, `exit 42` → `exitCode 42` | §5.3's N-API claim; risk 1 |
| **P2** | Does `@xterm/headless` work in the main process of the packaged app (no DOM)? | feed a recorded ANSI stream, assert `translateToString("scrollback")` and `modes.bracketedPasteMode` | §3(a)'s core |
| **P3** | Does a byte flood starve the main thread? | `yes` into a surface for 30 s; sample main-thread responsiveness (an IPC echo's round trip) and the surface's dropped-byte count | risk 3; the `utilityProcess` trigger |
| **P4** | Does `capturePage()` on a **hidden, unattached** `WebContentsView` return a complete frame of a terminal grid? | **PARTLY MEASURED — by the compatibility spike**: on a hidden `BrowserWindow` the first capture was stale/blank (9,866 B) and the second correct (27,869 B) — hence §13.2's assert-and-retry-once — and the **DOM renderer captured correctly first attempt** while the WebGL canvas could not be read at all (`toDataURL()` blank white, identical 26,791 B either way). **Still open:** the *unattached `WebContentsView`* variant, and whether an unattached view paints at all under Chromium's visibility rules. Run the grid capture against the capture view itself and compare with the pane's capture of the same record | §13.2 and §13.3 |
| **P5** | Do the mirror and the record agree? | replay a 2 MB recorded stream into both, compare scrollback text, cursor, and grid cells per row/column | risk 2 |
| **P6** | Does the pane's first frame after a replay equal the record's viewport text? | the conformance harness of P5, extended to the paint path | §7.3's replay claim |
| **P7** | Does the operator's shell emit OSC 133 marks (and with which integration)? | run their shell in a surface, print the surface's raw byte log, grep for `\x1b]133;` | §12.1's rung 2 |
| **P8** | What does the console tool's schema cost? | `scripts/bench_context_budget.py` (the guard CI already runs for the `context-budget` scope, `scripts/ci_scope.py:178-182`) plus `/context` in the live session, with the tool present and absent | risk 10; the §14.3 description budget |
| **P9** | Does the shipped GeistMono Nerd Font cover the recommended glyphs (`\uf108`, and every glyph in `NERD_TOOL_ICONS`), and do its box-drawing glyphs join at the pane's size? | **PARTLY MEASURED — by the compatibility spike**: this machine has **no Nerd Font installed at all** (`~/Library/Fonts` empty, `/Library/Fonts` = `Arial Unicode.ttf`), so a `"…Nerd Font Mono", Menlo, monospace` stack resolved to Menlo and every PUA codepoint tried was **tofu**; Menlo's `─` also showed visible gaps at size. **Still open:** the same sweep *inside the pane*, where the bundled face is what renders — the glyph table, the UI trace icon, and the box-drawing continuity at the chosen font step, in a rendered frame | §6.1's icon choice; §6.6; risk 9 |
| **P10** | Is the frame reproducible? | capture the same surface twice, 30 s apart, with no output in between; assert byte equality | §13.3's evidence claim |
| **P11** | Does a signed, notarized, packaged build fork a pty and capture a frame — with the helper executable and the prebuilds unpacked? | `pnpm exec electron-builder --dir --arm64` → **assert `stat` mode 0755 on `app.asar.unpacked/**/spawn-helper`** → assert the packaged tree is inside `check-packaged-closure` → the signed smoke with a console scene, forking a real pty and capturing one frame | risk 1, release gate |
| **P12** | Does an agent-driven capture leave the app non-frontmost? | dense external sampling of the frontmost process by pid during a full console capture sequence, in `headless`, with a `normal` control | R21 |
| **P13** | Do the two measured packing traps stay fixed? | in `dist/mac-arm64/Local Operator.app`: `stat -f '%Sp %N' "Contents/Resources/app.asar.unpacked/node_modules/node-pty/prebuilds/darwin-arm64/spawn-helper"` (**expect `-rwxr-xr-x`**), `lsof`-free spawn of a real pty from the packaged app, and a size check that the foreign-platform prebuilds are absent | risk 1 |
| **P14** | Is xterm's VT behaviour the *same* as ghostty's, on the streams the product actually uses? | the differential probe the raw wasm makes possible: replay one recorded corpus (the torture stream, a `lop` TUI session, a vim/less session, a CJK+emoji frame) through `@xterm/headless` and through `ghostty-vt.wasm`'s formatters, and diff the plain-text snapshots row by row | R4's "similar to ghostty" claim and §5.5's switch trigger |

### 19.2 `local-operator-ui` — the gates, and what each can prove

- `pnpm check-types` (both projects), `pnpm lint`, `pnpm check-themes`,
  `pnpm check-vendored`, `pnpm check-evidence`, `pnpm test:desktop` (the
  hand-written file list in `package.json` — the console's main-process tests are
  added to it, and `scripts/run-desktop-tests.mjs` wraps them with
  `LOCAL_OPERATOR_NO_NOTIFICATIONS=1` for free).
- **New in-process tests**: the surface registry and its caps; the byte log
  (ring, truncation flag, offsets); the emulator seam against a recorded stream;
  the OSC-133 scanner (marks, exit codes, a spoof, a split-across-chunks mark —
  the last is the bug this class always has); the key encoder vs the mode table
  (§10.5); retention/GC; the record's console fields; the method dispatch for
  every name in §10.2 including the unknown-method and wrong-key paths; the pane's
  renderer IPC authorization; and the notifier entry (its key, its eligibility,
  and that its body never carries content).
- **A host proof rig** (`scripts/console-host-proof.mjs`, following
  `browser-host-proof.mjs`'s shape): boot the built app headless on a scratch
  profile, then drive the real RPC path with real requests — create, read, resize,
  keys, screenshot, close — capturing the app's own log lines and the raw
  responses, including the unauthorized (no key / wrong key), `unsupported_method`,
  `surface_unavailable` and `process_exited` cells.
- **Rendered evidence**: the frames and the geometry numbers of §16.3, produced
  with `scripts/renderer-driver.mjs` (its `browser-pane` scene is the precedent
  for a new `console-pane` scene) and the app's own `capturePage`.
- **Storybook**: the pane's four states plus the secure and ended states, and the
  header control with and without a blip (`chat-header-cluster.stories.tsx` is the
  precedent), captured through `scripts/capture-evidence.mjs` with
  `docs/evidence/manifest.json` re-stamped.
- **What none of this can prove**: that the terminal *looks right* in twelve
  themes (that is the design round's frames), that a TUI behaves (that is P5/P6
  and §19.4), or that the packaging works (P11).

### 19.3 `local-operator` — the gates, and the cells

Unit (`.venv/bin/python -m pytest tests/unit`, whole tree, exactly as CI):

- `tests/unit/ui_console/test_state.py` — the record's console fields, the
  three-state classifier, "a read creates nothing", and the 0600/0700 modes
  asserted with `stat` rather than with intent.
- `tests/unit/ui_console/test_backend.py` — the client against a real loopback
  listener: 200 round-trip, 401 on a wrong key, a typed `proto_mismatch`, a
  refused connection, and a read timeout producing "accepted but did not answer"
  rather than "unreachable" (mirroring the existing bridge client tests).
- `test_console_tool.py` — the gate matrix (record absent / present-but-console-false
  / present-and-true), the `call_approval_tier` per method, the parameter
  validation, the `secret_ref` path's redaction registration (assert the value is
  **absent** from the result and that `VariableStore.redact` masks it afterwards),
  and that the tool is **not** hidden.
- `test_prompts_console_flags.py` — the `has_console`/`no_console` pair, the
  "both true" refusal, and the three-state diagnosis (a `reviewer` child on a
  console-capable host must not be told the host lacks one) — the exact defect
  `prompts_api.py:489-493` records for the browser.
- `test_glyphs.py` (extended) — `console` in both tables, distinct from `bash`.

Then for real (`tests/e2e`, marker by location,
`env -u NO_COLOR TERM=xterm-256color .venv/bin/python -m pytest tests/e2e -m e2e -n0`):

- **`test_console_rpc.py`**, modelled on `tests/e2e/test_browser_ownership.py:54-104`:
  a disposable peer publishing a `ConsoleState` record, driven over real HTTP with
  the key header (`:111`) and asserting the `401` cell (`:148`), then the typed
  refusals of §10.6. This cell needs no app and no browser — it is the one that
  proves the Python half end to end.
- **A `secret_ref` cell** driving the real store against a disposable peer, with a
  byte-level assertion that the value never appears in the tool result, the
  transcript, or the peer's received params as plaintext — the §19.3 shape the
  brief asks for, asserted rather than argued.
- **The pty cells** inherit `tests/e2e/test_terminal_close_survives_e2e.py`'s
  layout: strip `CMUX_*`/`LOP_*`, pin `TERM`, drain the master, and assert on the
  runtime's own record rather than on a screenshot.
- **Quality gates before the PR, exactly as CI**: `flake8`, `black --check`
  (pinned), `isort --check-only`, `pyright` (through the bounded runner,
  `make type-check`, because pyright's node child outlives a killed wrapper), and
  the unit suite over the whole tree with `.venv/bin/python`.

### 19.4 R10's co-pilot cell, spelled out

One cell, two actors, one surface: the human types a command through the
renderer IPC path while the agent simultaneously (a) reads the viewport, (b)
sends a named key, (c) captures a frame, and (d) resizes. The assertions: every
read after a write reflects it; no read returns a stale grid; the frame's
`rendered` field matches whether the pane is displayed; and the exit is observed
once. This is the cell that catches a push-channel race, which is the defect this
architecture is most likely to have and the least likely to appear in a
single-actor test.

### 19.5 The QA matrix surfaces (what an independent pass must cover)

The pane (four states × two themes minimum, all twelve for the design round); the
header control with/without blip; the divider clamps; session switch with a
running surface; pane close/reopen with output in between; app-restart history;
the notification (delivered, suppressed while displayed, click); the secure
toggle; the capture view (offscreen frame + reuse + reap); the tool's refusals
(all of §15); the caps (9th agent surface refused with the browser's own
granularity); and the two platform cells (macOS arm64 signed build; Linux x64
build if the artifact is in scope this window).

### 19.6 What "done" means per PR

- **A**: probes P1-P5, P11 and **P13** pass; the RPC transcript; the rig; the closure gates;
  a PR description that says the feature is not yet user-visible.
- **B**: the frames, the design round (D-findings clean), the UX round (U-findings
  clean), the stories, `check-evidence` re-stamped, `check-themes` green.
- **C**: the unit cells, the e2e cells, the prompt-flag cells, the
  `bench_context_budget.py` delta (P8), `guide://console`, and the doc.

---

## 20. Open decisions for the operator

Each is stated with a recommendation, so "no answer" has a default.

### 20.1 The default grid, and therefore the pane's default width

**Recommendation: 100×30**, with the pane's default width chosen to render it
(§6.1). The alternative — matching the pane's default to something narrower
because the run panel's 420 is the smallest existing pane — would put a
80-column TUI one keystroke from a reflow. *What settles it:* a width sweep in
PR B showing the px-per-column of the shipped font at the chosen size, and a
frame of the product's own TUI at both grids.

### 20.2 Where the secret's plaintext goes (§11.3)

Today: session → app, inside the RPC params of the one call. **Recommendation:
accept it for v1** (the app is the user's own process and the transport is
key-authenticated), and record the better end state: an app-side credential
resolution path, so the plaintext never leaves the runtime — which needs a new
desktop-contract op that does not exist. *What settles it:* whether the operator
wants a `secrets.reveal`-class op added to the desktop API, which is a larger
surface than this feature.

### 20.3 Shell integration (§12.1)

Do we ship the OSC-133 snippet, and do we offer to source it? **Recommendation:
ship the snippet as documentation and a copyable command; never write to the
user's dotfiles.** Per-command blips are then opt-in, which is the honest
position for a control that depends on the shell.

### 20.4 Retaining history across a relaunch (§7.3)

**Recommendation: on for user-opened surfaces, off for agent-opened ones**, with
a Settings row. Rationale: a user-opened surface is the user's own work; an
agent's surface is usually scaffolding, and persisting it by default grows the
config root with outputs nobody asked to keep.

### 20.5 The Linux artifacts

If `node-pty` cannot be built in the Linux CI image without new toolchain work,
the choice is (a) add the toolchain, or (b) ship the console disabled on Linux
with the tool absent and the pane explaining why. **Recommendation: (a) if the
toolchain cost is one apt line; otherwise (b), stated in the Release notes** —
because a silently missing feature is worse than a documented one.

### 20.6 Whether PR 0 (this document) should exist

It costs one small PR and one review. **Recommendation: yes** — it is what A/B/C
cite, and the alternative (the doc landing inside C) means the UI half is
implemented against a document that is not yet in the repository.

---

## Appendix A: claims checked, and things I could not verify

**Verified in this session, from the trees.** Every `file:line` above was read at
the refs of §0.1 — `local-operator` at `e65548e9` and `local-operator-ui` at
`db615bd46`, both read with `git show <ref>:<path>` and never off the disk —
including the ones the brief's numbers miss. The two repos are listed separately
because a citation list that silently mixed them is what the round-1 review
caught; where a name is ambiguous between them (`AGENTS.md` exists in both) the
repo is named:

- **`local-operator` @ `e65548e9`:**
  `builtin.py:12780`/`:12822`/`:3281`/`:9056`/`:10004`/`:10018`/`:12166`;
  `registry.py:32`/`:60`/`:70`/`:100`; `harness/types.py:1242` and its fields
  (`:1258`-`:1279`);
  `ui_browser/state.py:34`/`:47`/`:53`/`:87`/`:120`/`:145`/`:150`;
  `ui_browser/backend.py:34`/`:43`/`:56`/`:80`/`:103`;
  `browser_bridge/protocol.py:16`/`:20-45`/`:45`/`:200`/`:288`/`:349`/`:422`/`:434`;
  `browser_bridge/backend.py:159`/`:679`/`:718`/`:829`;
  `prompts_api.py:184`/`:304`/`:333`/`:365`/`:379`/`:401`/`:411`/`:489-513`;
  `prompts_md/system.md:264`/`:287`; `session/session.py:6120`/`:14012`;
  `tui/glyphs.py:82`/`:116`/`:118`/`:123`/`:191-224`/`:228-241`; `tui/notify.py:115`/`:404`;
  `terminals.py`'s marker predicates (`is_kitty`/`is_ghostty`/`is_wezterm` at
  `:103`/`:115`/`:131` and their callers);
  `agent_seeds/{architect,manager,reviewer,scout}.md:6` and `manifest.json`'s
  `tools` lists;
  `tests/e2e/conftest.py:1-20`; `test_terminal_close_survives_e2e.py:166`;
  `test_browser_ownership.py:111`/`:148`; `scripts/visual_capture.py:259`;
  `variables.py:255`/`:442`/`:489`/`:523`; `redaction_shapes.py:1-60`/`:84`;
  `scripts/bench_context_budget.py`; `scripts/ci_scope.py:178-182`;
  `docs/DESKTOP_API.md:1314-1342`/`:1344-1403`; `docs/design/ui-browser-tab.md:21-26`.
  LO `AGENTS.md:593-682` ("Read the committed ref, not the working tree") and
  `AGENTS.md:1574-1582` (the `ask_shot.py out.svg 100x30` block).
- **`local-operator-ui` @ `db615bd46`:**
  `ui-preferences-store.ts:51`/`:72`/`:92`/`:198`/`:264`/`:380`/`:413`/`:422`/`:434`;
  `chat-content.tsx:1206`/`:1285`/`:1383`; `chat-header.tsx:199`/`:271-341`;
  `browser-pane.tsx:86`/`:117`; `chat-page.tsx:2231`;
  `main/browser/rpc.ts:29-46`/`:50`/`:53`/`:58`/`:68`; `protocol.ts:53`/`:81`;
  `state-file.ts:31`/`:36`/`:42`; `host.ts:383`; `ipc.ts:20-35`/`:56`;
  `registry.ts:96`/`:120`/`:473`/`:490`/`:527`/`:546-557`/`:609`/`:664`;
  `index.ts:175`/`:182`;
  `desktop-notifier.ts:205`/`:267`/`:304`/`:395`/`:707`/`:721`/`:1101`/`:1155`/`:1174`/`:1210`;
  `notification-launch.ts:75`; `preload/index.ts:108-182`;
  `desktop-ipc.ts:297`/`:321`;
  `desktop-contract.ts:678`/`:2173`/`:2617`; `tool-glyphs.ts:54-97`;
  `fonts.css:33-45`; `styles/index.css:117`/`:130`;
  `palette-contract.ts:186`/`:429`/`:477`/`:494`/`:520`/`:719`/`:776`/`:796` (all role
  keys); `themes/index.ts:4-17`/`:202`/`:217`;
  `code-mirror-theme.ts:20-70`; `contrast-contract.mjs:30-56`/`:69`;
  `package.json` (`build`, `dependencies`, `optionalDependencies`, the `test:desktop`
  list); `pnpm-workspace.yaml`; `scripts/check-runtime-deps.mjs:1-60`;
  `docs/CODE_SIGNING.md`; `docs/design/panel-views.md:1-120`;
  `docs/design/browser-approval-ux.md:1-80`/`:539-640`;
  `docs/agent-driver.md:1-200`; UI `AGENTS.md:61-65`/`:32-35`/`:717-723`/`:729-735`/
  `:744-748`/`:758-762`/`:810-816`/`:1096-1127`.

**One citation set is deliberately *not* claimed as re-derived:** the `file:line`
references in `docs/design/ui-console-tab.md` itself (this document's own internal
cross-references) are prose, not citations, and no number in this appendix refers
to them.


**Verified from the web, with the source named:** ghostty's `spdx_id: MIT`
(GitHub API); the `tip` release's asset list including
`ghostty-vt.wasm` = 1,007,826 B and `ghostty-vt-small.wasm` = 747,017 B with
minisign files (GitHub API); cmux's `LICENSE` = GPL-3.0-or-later
(`raw.githubusercontent.com/manaflow-ai/cmux/main/LICENSE`); and the npm registry
metadata for `@xterm/xterm` 6.0.0, `@xterm/headless` 6.0.0 (README: "experimental",
"no official addons are packaged on npm" for headless), the six `@xterm/addon-*`
packages (all MIT, no dependencies), `node-pty` 1.1.0 (MIT; prebuilds for
darwin-arm64/x64 and win32-x64/arm64 only; `install: node scripts/prebuild.js ||
node-gyp rebuild`; `node-addon-api ^7.1.0`; the `app.asar` → `app.asar.unpacked`
helper-path rewrite in `lib/unixTerminal.js`), `ghostty-web` 0.4.0 (MIT),
`@coder/libghostty-vt-node` 0.1.0-beta.0 (MIT, `engines.node >= 20.19`).

**Measured on this machine, by me:** the two refs and their divergence (§0.1 —
`~/local-operator`'s working tree is HEAD `0e287be0` with **295 changes** away
from `origin/main` by `git status --porcelain | wc -l`, and `~/local-operator-ui`
`chore/release-0.25.11` at `0f76af9e1`, **1,355 commits behind** its `origin/main`,
656 files differing); `node-pty`'s tarball contents (286 files, no Linux prebuild,
`prebuilds/darwin-arm64/spawn-helper`); `@xterm/headless`'s tarball size
(477,641 B compressed / 1,957,834 B unpacked, 7 files) and its declared API
surface (modes, buffer, `translateToString`, `getWidth`/`getChars`, the event
list); the UI repo's shipped font faces and the **absence** of any font licence
notice in it (`LICENSE`, `build/license_*.txt`, no `THIRD_PARTY`/`OFL`); `df` on
the working volume (78 GiB free).

**Measured by the compatibility spike, and relayed to me rather than re-run here
(it ran on the same machine: macOS 26.6.2 / Darwin 25.6.0, arm64, Electron 44.3.0
→ node 24.20.0, ABI 149, N-API 10).** Quoted because §5.5's decision rests on it:

- **`node-pty@1.1.0`** — the same `pty.node` (**85,496 B**) loads under node 26.5.0
  (ABI **147**) and Electron 44.3.0 (ABI **149**), so **no `electron-rebuild`**;
  `/bin/zsh -f -i` spawned, prompt read back, echo round-trip, 24-bit and UTF-8
  **byte-exact**, `resize(100,30)` visible in `stty size`, `exit 42` → 42. Two
  packing traps: `prebuilds/darwin-arm64/spawn-helper` ships **mode 0644** and
  nothing chmods it → `FATAL Error: posix_spawnp failed. at new UnixTerminal
  (lib/unixTerminal.js:92)`; and a fully-packed `app.asar` resolves `pty.node` yet
  still fails to spawn until the prebuilds are unpacked
  (`--unpack-dir node_modules/node-pty/prebuilds` works). npm 11.17's
  install-script gate means `scripts/prebuild.js` does not run. `darwin-arm64`
  prebuilds are **135,976 B / 2 files**; win32 prebuilds are **58 MB of the 64 MB**
  unpacked (mostly `.pdb`).
- **xterm 6.0.0 + addons** — Vite bundle **523.31 kB JS** (138.72 kB gzip) +
  3.52 kB CSS; loads in a hidden `BrowserWindow` (`show:false`, `isVisible()`
  false, `getFocusedWindow()` null); 100×24; WebGL got a real GPU context (ANGLE
  Metal); `serialize()` returned the text; `search.findNext()` true; `unicode.versions`
  `["6","11"]` with a discriminating proof (`cursorX` after an emoji = 1 under v6,
  2 under v11); cells read via `buffer.active.getLine(y).getCell(x)` with fg/bg
  **colour modes**. Capture: WebGL `toDataURL()` **blank white** (identical
  26,791 B with `preserveDrawingBuffer` both ways); `capturePage()` on a hidden
  window **stale first frame** (9,866 B) then correct (27,869 B); **DOM renderer
  correct first attempt**.
- **`ghostty-vt.wasm`** — **1,007,826 B**, sha256
  `139c9617c9dfea4a51dfab7a87a72017163c472ad248c61cdea8d7e942c1f424`; minisign
  **VALID** (file signature and trusted comment) against the key in Ghostty's own
  `PACKAGING.md`, verified with `node:crypto`; 0 imports, 189 exports,
  `wasm32-freestanding`; `ghostty_type_json()` = 43,531 B (159 types, struct
  sizes/offsets, enum values). With **no shim written**: a terminal was created,
  fed bytes, resized to 100×30, and read back as plain text, re-emitted VT and
  HTML; styled cells came back via the row-cells container and via the packed u64
  `GhosttyCell` (wide=1, `SPACER_TAIL`=2 on CJK/emoji tails, palette red →
  `[204,102,102]`). The same wasm instantiates in the Electron renderer. ABI
  gotchas: `CELLS_RAW` writes a **pointer** into `out`; a row iterator is
  **single-use**.
- **`ghostty-web@0.4.0`** — bundles a **423,045 B** wasm; API largely
  xterm-compatible (`open`/`write`/`resize`/`paste`/`input`/`buffer`/`getCell`/
  `onData`/`onResize`); **2-D canvas renderer only**; styles arrive as a `flags`
  bitfield; its canvas renders **and** `toDataURL()`s while **not attached to the
  document** (36,886 B, pixels verified: truecolor, CJK, emoji, box-drawing).
- **Fonts** — **no Nerd Font is installed on this machine** (`~/Library/Fonts`
  empty; `/Library/Fonts` = `Arial Unicode.ttf`); a `"JetBrainsMono Nerd Font
  Mono", Menlo, monospace` stack resolved to Menlo and every PUA codepoint tried
  (U+E0B0-E0B3, U+F015, U+F07B) was **tofu**. Cell metrics: xterm DOM
  **8.425 × 16 px at `fontSize: 14`** (100 columns → 800×384); `ghostty-web`
  9 × 15 with a 12px baseline at DPR 2. Menlo's box-drawing glyphs **do not join**
  at that size in either stack.

**NOT verified — every one of these needs running, and each names what would
settle it:**

1. ~~That `node-pty`'s prebuilt binary loads under Electron 44.3.0 with no
   rebuild.~~ **Settled by the spike — PASS** (P1), including spawn, echo, 24-bit
   round-trip, resize and exit code. What it did **not** settle is the packaged
   case, which is now P11/P13.
2. **`@coder/libghostty-vt-node`** — never loaded (option C is not chosen), and
   its beta ABI caveat is upstream's own statement rather than a measurement.
   The **other** two WASM options are now measured rather than judged (§5.2's B
   row, the appendix above), and `ghostty-vt.wasm`'s provenance was verified in
   this environment; what remains unmeasured is **the differential VT comparison**
   itself (P14), i.e. whether xterm and ghostty agree on the streams this product
   uses.
3. **`@xterm/headless`'s behaviour under Electron's main process** (P2), and its
   exact memory per surface at 100×30 with a 5,000-line scrollback — §6.3's cost
   figures are order-of-magnitude, not measured.
4. **`capturePage()` on a hidden, *unattached* `WebContentsView`** returning a
   complete frame of a terminal grid (P4) — the hidden-`BrowserWindow` case is
   measured (stale first frame, then correct), the unattached-view case is not.
   §13.2's assert-and-retry is designed for exactly the measured half.
5. **The mirror/record conformance test existing at all** (P5) — the design
   asserts the invariant and names the test; nothing was run.
6. **Whether the operator's shell emits OSC 133** (P7).
7. **The exact px-per-column of *the shipped Geist Mono* at the pane's text
   step**, and therefore whether 100×30 fits the default pane width. The spike
   measured **xterm's DOM renderer at 8.425 px/column with `fontSize: 14`** — but
   that run's stack resolved to **Menlo**, not to the bundled face, so the number
   the pane will actually get is still unmeasured (§6.1).
8. **That the shipped Nerd Font covers `\uf108`** and every glyph the tables use
   **inside the pane**, and that its box-drawing glyphs join at the chosen size
   (P9). What the spike measured is the *negative*: with no Nerd Font installed, a
   PUA sweep is tofu, and Menlo's `─` has gaps.
9. **A signed, notarized, packaged build** (P11), and the Linux build at all.
10. **The tool's schema cost** in tokens (P8), and its effect on the prompt cache.
11. **The `renderer-driver.mjs` scene for a pane whose content is a live pty** —
    the existing scenes drive stores and the hash router, and whether a pty-fed
    pane can be driven and captured in the same run is a PR-B question.
12. **Anything about Electron's `utilityProcess` hosting node-pty** (§3(d)) —
    deliberately not explored, since it is the deferred option.
13. **That the console's Nerd-Font predicate produces the TUI the design
    claims.** §6.6 adds a marker and a one-clause change to `glyphs.py`; that
    this yields the project's own interface drawn with Nerd glyphs in the pane
    (rather than tofu, or rather than a *different* set of glyphs than the TUI's
    tables were measured against) is unverified until P9 runs against the real
    TUI, not against a glyph sweep.
14. **Cross-platform behaviour of the resize→`SIGWINCH` path on Windows**
    (ConPTY's resize semantics differ), which §8 asserts only for the platforms
    the app builds; if Windows is in scope for the console in the same window,
    that is an unverified cell.
