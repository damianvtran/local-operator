# A browser tab inside `local-operator-ui`, driven by lop agents

Status: **proposal (architect), revision 2** — the design authority for the
implementation that follows. Scope: two repos — `local-operator` (the Python
seam, the shared policy modules, the protocol generator, the vendored bundle)
and `local-operator-ui` (the Electron host, the tab, its chrome, the profile,
the extensions directory). Seven PRs. No `pyproject.toml` or `package.json`
version bump — the release owner handles that.

## 0. How to read this document, and what changed in this revision

### 0.1 Bases, and a warning about every other document in this repo

Everything below is re-derived from the trees named here, not carried forward:

| repo | ref | notes |
|---|---|---|
| `local-operator` | `b97c646dc` (`origin/main`, the merge of PR #1098) | checked out at `~/local-operator-worktrees/ui-browser`, branch `feat/ui-browser-host` |
| `local-operator-ui` | `7f0511dd4` (`origin/main`, PR #153) | checked out at `~/local-operator-ui-worktrees/browser-tab`, branch `feat/browser-tab`; `version` `0.21.0` (`package.json:3`); Electron pinned **`44.3.0`** (`package.json:106`) |

**Revision 1 of this document was written ~160 commits behind and every
`file:line` in it is now wrong.** Symbols moved; the ownership layer gained a
concept (`BrowserResource.ownership_mode`) that revision 1 did not know existed;
the daemon grew from ~2,084 to 3,748 lines and landed multi-identity pairing.
This revision re-derives each citation. Where a number below disagrees with
revision 1, this document is the correct one.

Two sibling documents are cited by *rule*, never by line number, because their
line numbers are stale: `docs/design/browser-extension.md` still says "no code
exists yet" in its §1 and cites `builtin.py:5685` for `build_browser_tool`
(actually `:10119`); it is the contract's *rationale* source, not a current map.
`docs/design/browser-multi-identity-pairing.md` is a proposal that has since
been **implemented** (§2.8).

### 0.2 Honesty conventions (kept from revision 1, and load-bearing)

Every claim is one of three kinds, and the kind is stated:

- **Verified** — read in the source tree at the ref above, or quoted from the
  Electron v44.3.0 documentation in the `electron` repo. Cited.
- **Measured** — a number produced by running something, with the method given.
- **Unverified** — stated as such, with the exact experiment that would settle
  it, named as a QA probe. **No Electron behaviour is asserted in this document
  without a v44 citation or a measurement.**

The Electron docs site serves only `latest`, so the pinned tag is the
authoritative source for the runtime this app ships:
`https://raw.githubusercontent.com/electron/electron/v44.3.0/docs/...`.

### 0.3 Revision-2 changes, in one table

| area | revision 1 | revision 2 |
|---|---|---|
| Electron | 35.5.1, EOL concern | **44.3.0**; the 35-era EOL worry is void. Section §2.7, §11 |
| persistent profile | "its own partition", re-login assumed | requirement 1: **one shared `persist:` partition**, persistence specified, lifecycle specified. §5 |
| tabs | tab strip as a nice-to-have | requirement 2: full chrome + a **shared tab registry**, cross-actor driving, concurrency semantics. §6 |
| restore | explicitly a non-goal | requirement 3: **yes, restore**, via Electron 44's `navigationHistory.restore`. §7 |
| extensions | "no extension support in the view" | requirement 4: bounded, honest, measured capability. §8 |
| capability parity | claimed superset, not argued | requirement 5: superset **argued per row**, environmental gaps named. §4 |
| security | per-origin approval | requirement 6: the approval model is now the load-bearing control against an agent holding real sessions. §9 |
| sharing | npm considered | requirement: **no new npm pipeline**; vendored-and-generated from a pinned ref + drift checks. §12 |
| ownership seam | "the one structural change" | unchanged in shape, but now nine `BridgeClient()` sites and `ownership_mode`'s own discovery read must be host-selected. §10.5 |
| actions/methods count | "18 actions" | **17** actions (`builtin.py:7498-7529`), **20** wire methods (`protocol.py:162-209`) |

### 0.4 Requirements traceability

The six operator asks, each mapped to the section that satisfies it. A reviewer
checking this document should be able to confirm from this table that none was
dropped.

| # | the ask (verbatim intent) | satisfied in | the deliverable it produces |
|---|---|---|---|
| **R1** | Persistent, shared browser profile — "log in with my credentials once and then all browser windows across all agent sessions … should be able to use that even if I close and reopen" | **§5** | partition name, on-disk location, the persists/does-not-persist table, reset + update + uninstall lifecycle |
| **R2** | Basic browser controls — URL bar, back/forward, reload/stop, new/close tab, tab strip, and **cross-actor** tab driving | **§6** | the control list, the shared tab registry, tab identity across the two actors, concurrency semantics |
| **R3** | Tab restore across restarts ("similar to my personal browser") | **§7** | the decision (restore), where session state lives, how a restored tab meets a stale agent handle |
| **R4** | Extension support, **measured not asserted** | **§8** | the cited Electron 44 support matrix, the managed directory + load flow, the attempt-and-report UI, the 1Password verdict, QA probes |
| **R5** | Same capability as the extension host | **§4** | the capability matrix re-verified row by row, environmental vs capability gaps separated, parity probes named |
| **R6** | The login persistence changes the security model — design the guardrail | **§9** | the per-origin model, what the user sees, consent lifetime, revocation, and the honest default plus its limits |

Carried-forward fixed interface decisions, and this revision's verdict on each
(§10–§12 argue them; this table is the index):

| fixed decision | verdict | evidence |
|---|---|---|
| UI-owned loopback RPC speaking the existing session leg | **keep** | §3(a), §10 |
| discovery via a 0600 state file in its own namespace | **keep** | §10.1 |
| `ui:` surface prefix | **keep** | §10.4 |
| UI → bridge → cmux precedence on a fresh `open` | **keep** — R1 *strengthens* the case for it, see §10.4 | §10.4 |
| CDP through `webContents.debugger`, not `--remote-debugging-port` | **keep** | §3(c) |
| no new Python dependency | **keep** | §10.3 |
| no second browser engine | **keep** | §3(d) |
| one `WebContentsView` per tab (not `BrowserWindow` per tab) | **keep** — re-argued for tabs in §11.1 | §11.1 |
| one partition per host, `persist:local-operator-browser` | **keep the name, change the scope**: R1/R2 make it *shared across every tab and every agent surface*, and R4 makes it *mandatory* | §5.1 |
| **no new npm publishing pipeline in v1** | **new constraint, satisfied by §12** | §12 |

---

## 1. Problem and scope

### 1.1 The request

Add a **browser tab to the `local-operator-ui` desktop app** that lop agents
pilot through the existing `browser` tool, the way they pilot the cmux browser
panel today — lightweight, open source, on the standard Chromium
implementation, with no second browser engine. Preferred cascade for a fresh
`open`: UI browser tab → Local Operator browser extension → cmux → the typed
error.

To that, the operator added six requirements (R1–R6 in §0.4). They are additive
to the above but they change the *shape* of the feature rather than decorating
it: R1 and R2 together turn "a tab the agent can drive" into "a small browser
the user and the agent share", which is a different security posture (R6) and a
different persistence story (R1) from the one revision 1 designed.

### 1.2 What is actually missing

Nothing about the tool surface, and that is still the important observation.
`BROWSER_ACTIONS` (`builtin.py:7498-7529`), `BrowserParams`
(`builtin.py:7626-7689`), the `write` approval tier
(`builtin.py:10199`), the URL guard (`_validate_browser_url`,
`builtin.py:7922-7938`; `_BROWSER_URL_SCHEMES`, `builtin.py:7623`) and the
model-facing diagnostics (`format_error`, `backend.py:202-315`) are already
backend-agnostic: the tool was built so the model cannot tell which backend
answered. The extension work landed that claim as a first-class two-backend
seam.

What is missing is (a) a third **host process** that owns a Chromium surface,
(b) a place in the discovery/precedence/seam logic for it, (c) the UI-side app
work — and now also (d) the profile/tab/extensions surfaces of R1–R6, which are
the UI host's own responsibilities, not the tool's.

The single hardest constraint in revision 1 was that the seam is written as
"cmux, or the bridge daemon" in about a dozen places. That is still true, and §10.5
now counts it precisely: **nine `BridgeClient()` construction sites** and five
direct bridge state-file reads, all of which must become host-selected.

Two things bound the design, unchanged but now better informed:

- **The desktop app already is Chromium.** The UI is Electron **44.3.0**
  (`package.json:106`), so "no second engine" is free, and the browser tab is a
  `WebContentsView` in the app the operator is already running.
- **The UI has no browser view and no HTTP server today.** Verified at this ref:
  no `WebContentsView`, `BrowserView`, `<webview>` or `webviewTag` anywhere under
  `src/`, and no `createServer`/`.listen(` in `src/main/*.ts` (the only
  `createServer` hits in the repo are inside `scripts/*.test.mjs`, i.e. test
  harnesses). The app talks to the lop backend over `fetch`
  (`desktop-transport.ts`) and to its own renderer over `ipcMain.handle`
  (`desktop-ipc.ts:89-254`: `desktop-request` `:118`, `desktop-close-window`
  `:126`, `desktop-watch-heartbeat` `:134`, `desktop-media` `:161`,
  `desktop-open-authorization` `:167`, `desktop-stream-subscribe` `:217`,
  `desktop-stream-unsubscribe` `:249`). So this design still adds the first
  inbound network listener the app has ever had.

### 1.3 Non-goals (v1) — revised

Revision 1's list changes in three places, and each change is a deliberate
consequence of R1–R6 rather than an oversight:

- **Still a non-goal: a general-purpose browser.** No bookmarks, no download UI,
  no history *browser* (a restore exists — §7 — but there is no history page), no
  profiles manager, no multiple windows.
- **Now IN scope (bounded): extension support.** R4 asks for it, so it is
  specified in §8 as a managed directory of *unpacked* extensions with an honest
  supported/unsupported matrix. It is not "install anything from the store".
- **Now IN scope: tab restore.** Revision 1 listed "no history UI" as a
  non-goal; the operator asked for "similar to my personal browser", so §7
  restores open tabs (with their navigation stacks) and does nothing more.
- **Still a non-goal: driving the user's existing Chrome/Arc profile.** The app
  gets its own persistent jar (§5). **R1 changes** the *consequence* of that from
  revision 1's "so the thing the extension exists for is not what the UI offers"
  to "so the UI offers a real logged-in jar of its own, which removes most of the
  reason to prefer the extension". The extension remains the only way to reach
  device-trust / hardware-key / enterprise-conditional-access sessions bound to
  the real profile.
- **Still a non-goal: attaching to the app's own renderer.** §11.7.
- **Firefox and Safari** — unchanged from `docs/design/browser-extension.md` §10.
- **Remote access.** Loopback only, as the bridge daemon is (`daemon.py:3741`,
  the only loopback bind: `uvicorn.Server(uvicorn.Config(app,
  host="127.0.0.1", ...))`).
- **A second Chrome/Chromium engine or a bundled headless browser.** The repo
  refuses this in the tool builder's own docstring (`builtin.py:10125-10131`:
  "This repo ships no browser engine — playwright belongs to the pre-rewrite
  codebase and appears in no dependency group").
- **Restarting the app to recover a host-side wedge.** Out of v1; the UI host is
  a single long-lived process, so the extension's MV3 worker-churn hazards do not
  exist here. That simplification survives R1–R6.

---

## 2. What exists today (file:line evidence, all re-derived)

### 2.1 The tool and its single dispatcher

`build_browser_tool` (`builtin.py:10119`) is `createIf`-gated: it returns `None`
unless cmux *or* the extension bridge is advertisable (`builtin.py:10160`:

```python
if not cmux_browser_available() and not bridge_browser_advertisable():
    return None
```

), and its description names the surfaces the model can expect
(`builtin.py:10166-10195`). Its `approval_tier` is `"write"`
(`builtin.py:10199`) — it navigates and writes a screenshot file.

`execute_browser` (`builtin.py:9623`) is the host-owned serialization boundary.
For anything but a cmux surface it runs the action under the session's browser
resource lock, including the ownership lane (`builtin.py:9658-9817`).
`_execute_browser` (`builtin.py:9820`) is the transport chooser: it validates
params, checks availability, and dispatches per backend.

`tools/registry.py:60` holds the `createIf` entry
(`"browser": lambda _context: builtin.build_browser_tool(_context)`).

### 2.2 The cascade as it exists

Two backends, one precedence rule, on a *fresh* `open` only — at
`builtin.py:9912-9918`:

```python
if state.surface_id.startswith("bridge:"):
    return await _bridge_open(tool_call_id, state, params.url, context)
if state.surface_id.startswith("surface:"):
    return await _browser_open(tool_call_id, state, params.url)
if bridge_available:
    return await _bridge_open(tool_call_id, state, params.url, context)
return await _browser_open(tool_call_id, state, params.url)
```

A prefixed handle pins the transport; with no handle, a reachable bridge wins
over cmux. The comment (`builtin.py:9895-9911`) states why the extension wins:
it drives a real Chromium profile with the user's own logins and never steals
focus — its tab is created inactive and no action raises a window. **R1 changes
the weight of this argument and §10.4 revisits it.**

Availability is four separate questions, deliberately (all in `builtin.py` and
`browser_bridge/backend.py`):

- `cmux_browser_available()` (`builtin.py:7722-7759`) — a PATH lookup plus one
  env read; spawns nothing. `cmux browser-status` "costs a process spawn and a
  socket round trip and can hang when the socket is wedged — session start must
  never block on a terminal emulator" (`:7727-7730`).
- `bridge_browser_available()` (`builtin.py:8509`, → `backend.py:112-127`) —
  file-only, FRESH only.
- `bridge_browser_advertisable()` (`builtin.py:8518`, → `backend.py:129-142`) —
  file-only, FRESH or STALE-but-alive, used for **gating** because advertising
  is a weaker commitment than executing. The doctrine is in
  `state.py:206-234`: "a tool that explains why it cannot reach the bridge beats
  a tool that silently does not exist", and gating with the strict check "is a
  hole in the RC2 rescue".
- `bridge_browser_reachable()` (`builtin.py:8532`, → `backend.py:145-182`) — the
  browser-path check, which spends one bounded socket probe *only to acquit* a
  stale-but-alive daemon (`HEALTH_PROBE_TIMEOUT_S = 1.5`, `backend.py:109`).

Every new host must supply the same four answers, or the semantics above break.

### 2.3 Surface pinning

The transport is pinned in the surface token's prefix — `surface:<n>` for cmux,
`bridge:<tab>:<nonce>` for the extension — so a browser opening or closing
mid-session cannot move the agent to a different surface. The prefix is validated
by regex — `re.fullmatch(r"bridge:\d+:[A-Za-z0-9_-]+", pending)` at
`builtin.py:9053` and `re.match(r"^bridge:\d+:[A-Za-z0-9_-]+$", surface)` at
`builtin.py:9058` — branched on by `close_browser_surface`
(`builtin.py:9574`, `startswith("bridge:")` at `:9592`),
`retitle_browser_surface` (`builtin.py:9530`, `:9549`), `_bridge_open`'s resume
(`:9031`) and the dispatch sites listed in §2.2.

`BrowserSurface` (`harness/types.py:649-675`) holds `surface_id`, an optional
host `resource`, and `extension_update_notified`
(`__slots__` at `:664`). It is constructed by the **session** as the owner:
`self._browser = BrowserSurface(BrowserResource(transcript.directory,
self._session_id))` (`session/session.py:2503`) and injected into every rebuilt
`ToolContext` (`session/session.py:7599`, `browser=self._browser`). The
`BrowserSurfaceProtocol` (`harness/types.py:639-646`) is the narrow view the tool
layer takes.

### 2.4 The state file: liveness vs advertisable, and why it is cheap

`browser_bridge/state.py` is the pattern to mirror, and its comments are the
requirements:

- 0600 file under a 0700 directory, staged write + `os.replace`
  (`state.py:99-117`); `state_path()` is "pure path arithmetic: creates NOTHING"
  (`state.py:83-96`) because routing a reader through `run_dir()`'s `mkdir`
  turned an ENOSPC handler into a second `OSError` from inside the first
  (`state.py:86-88`).
- `BridgeState` (`state.py:28-72`) — `pid`, `port`,
  `session_key` (`min_length=32`), `proto`, `extension_connected`,
  `paired`, `extension_id`, `browser_name`, `extension_unresponsive`,
  `extension_version`, `extension_proto`, `extension_update_available`,
  `heartbeat_at`, `started_at` — with `model_config = ConfigDict(extra="ignore")`
  (`:29`).
- `HEARTBEAT_INTERVAL_S = 15.0` and `HEARTBEAT_TIMEOUT_S = 45.0` (`state.py:24-25`).
- The file heartbeat **lies in both directions**, so it yields three states, not
  two: `ABSENT`, `FRESH`, `STALE` (`state.py:152-176`). The caller decides how
  much an answer is worth paying for, and only `STALE` buys a socket probe.
- `session_key` in a 0600 file *is* the session-leg authorization
  (`state.py:3-5`): anything able to read it already owns the account.

**One change from revision 1 that matters for the UI host.** `liveness()` now
reads (`state.py:179-188`):

```python
current = read(root)
if current is None or not current.extension_connected or not pid_alive(current.pid):
    return Liveness.ABSENT, current
```

`extension_connected` is folded into the classification. That field has **no
analogue on the UI host** (the UI host's views are its own children; there is no
second process that can be detached). So the UI must **not** reuse `liveness()`
verbatim — §10.2 specifies what it reuses instead, and why reusing this function
would classify a perfectly healthy UI host as `ABSENT` forever.

### 2.5 The wire protocol and its codegen

`browser_bridge/protocol.py` is the single hand-written source:

- `PROTO_VERSION = 1` (`:16`), `MIN_SUPPORTED_PROTO = 1` (`:45`) — and these are
  a **window**, not an equality: `proto_supported(proto, low=..., high=...)`
  (`:48-49`). The invariant is written at `:20-45`.
- `METHODS` (`:162-209`) — **20** methods: open, goto, read, snapshot,
  screenshot, click, type, close, status, tabs, scroll, logs, request_access,
  await_access, cancel_access, retitle, owner_recover, owner_finish,
  owner_retain, owner_release. Each carries its rationale inline.
- `ORIGIN_PROMPT_TIMEOUT_MS = 60_000` (`:227`) and the timeout-chain invariant
  (`:217-226`): extension deny (60 s) < daemon prompt window (65 s) < session
  client timeout.
- `COMMAND_TIMEOUTS` (`:233-...`), per-method budgets the daemon enforces and the
  client derives its longer HTTP timeout from.
- `OWNERSHIP_MIN_EXTENSION_VERSION = "0.1.9"` (`:99`) and `extension_older`
  (`:135`).
- `ErrorCode` (`:283-333`) — including `ORIGIN_NOT_ALLOWED`, `TAB_LIMIT`,
  `TAB_AMBIGUOUS`, `ACCESS_QUEUE_FULL`, `OWNER_REFUSED`, `PROTO_MISMATCH`,
  `EXTENSION_UNRESPONSIVE`.
- `WireModel` with `extra="forbid"` (`:336-339`) so version skew fails at the
  boundary rather than later; `Request` (`:342`), `ErrorDetail` (`:348`),
  `Response` (`:354`), `Hello` (`:361`), `HelloAck` (`:369-393`), `Role` (`:396`).

`gen_ts.py` renders one target today — `TARGET = REPO_ROOT / "extension" /
"src" / "protocol.gen.ts"` (`:25`) — with `--check` (`:114`) that diffs
byte-exactly and exits non-zero when stale (`:121`). CI enforces it in the
paths-gated extension workflow and the main Python job. §12 extends this to two
targets.

The client on the session side is `BridgeClient.call` (`backend.py:341-409`): it
reads the state file (`:346`), POSTs one `Request` to
`http://127.0.0.1:<port>/rpc` with `X-Bridge-Key` (`:357-361`), and maps the
`Response` to a result or a typed `BridgeError`. `client_timeout()` derives the
HTTP budget from the daemon's worst case so the daemon's typed answer always wins
the race (`backend.py:316-338`); the error mapping is one table plus a few
data-discriminated branches (`backend.py:202-315`). This is the ~70 lines of
transport the UI host reuses, and every model-facing sentence in the tool comes
out of it.

### 2.6 The extension's CDP driver, and where the `chrome.*` coupling sits

**Re-measured on this ref** (non-comment lines, excluding `popup/` and
`options/`, block comments stripped then line comments dropped; a line counts as
`chrome.*` if it contains the literal `chrome.`):

| module | non-comment LOC | `chrome.*` lines | what it is |
|---|---|---|---|
| `worker.ts` | 393 | 20 | WS client, dispatch, reconnect, events |
| `approval-store.ts` | 339 | 9 | approval queue persistence (storage) |
| `commands/nav.ts` | 245 | 14 | open/goto/status/tabs/close (tabs, webNavigation) |
| `origins.ts` | 206 | 7 | origin gate + prompt wiring |
| `state.ts` | 190 | 20 | surfaces map, refs, admission cap (storage) |
| `tab-groups.ts` | 187 | 12 | tab groups |
| `access-grants.ts` | 179 | 22 | grant persistence (storage) |
| **`access-queue.ts`** | **161** | **0** | pure FIFO approval-queue rules |
| `ownership.ts` | 160 | 8 | allocation journal / owner scopes |
| `log-capture.ts` | 146 | 2 | console + runtime log ring buffer |
| **`origin-policy.ts`** | **143** | **0** | PSL/registrable domain, loopback rules, url guard |
| `commands/input.ts` | 136 | 12 | trusted click/type |
| `cdp.ts` | 127 | 11 | attach/detach, `sendCommand`, surface prune |
| `commands/scroll.ts` | 118 | 2 | scroll + more-below reporting |
| **`access-flow.ts`** | **101** | **0** | pure async approval state machine |
| **`protocol.gen.ts`** | **62** | **0** | generated |
| **`commands/access.ts`** | **61** | **0** | request/await/cancel handlers |
| `settle.ts` | 51 | 6 | `deadline()` + the ceiling table |
| **`ax-compact.ts`** | **50** | **0** | AX tree pruning/rendering |
| `action-surface.ts` | 42 | 5 | badge/tooltip notifications |
| `commands/snapshot.ts` | 33 | 2 | `Accessibility.getFullAXTree` |
| **`scroll-expressions.ts`** | **31** | **0** | the injected scroll JS |
| `commands/read.ts` | 28 | 4 | text extraction |
| `commands/logs.ts` | 17 | 2 | log read/filter |
| **`tab-lifecycle.ts`** | **15** | **0** | `onRemoved`/`onReplaced` reclamation |
| `reconnect.ts` | 12 | 0 | backoff/alarm policy |
| `commands/shot.ts` | 12 | 2 | `Page.captureScreenshot` |
| **`errors.ts`** | **9** | **0** | `BridgeCommandError` |
| **`psl.gen.ts`** | **3** | **0** | generated public-suffix data |

**Total 3,257 non-comment LOC; the eleven bold rows are the `chrome.*`-free set
and they sum to exactly 648** — which is the check a reviewer should run against
the table rather than take on trust. (Revision 1 measured 3,221/641; the set is
identical, the numbers moved.) Four of them say so in their own headers —
`access-flow.ts` ("Pure state machine for the async site-access flow, DOM- and
chrome-free for the pure-node test suite") and `access-queue.ts` ("Pure,
storage-independent approval queue rules") — and four more are host-free by
construction rather than by declaration (`origin-policy.ts`, which imports only
the generated PSL data; `ax-compact.ts`; `scroll-expressions.ts`;
`commands/access.ts`). The remaining four zero-chrome rows are generated or MV3
plumbing that does not survive the move. **The reusable half is the *policy*
half**, and §12 builds on that rather than on an imagined wholesale port.

Two behaviours in that driver the UI cannot copy literally, and which §4/§11 must
substitute:

- **`read` runs in the extension's isolated world.** `chrome.scripting.executeScript`
  defaults to `ISOLATED`, so page script cannot observe or shadow the extraction.
  Electron's `webContents.executeJavaScript` runs in the *main* world; the
  equivalent is `webContents.executeJavaScriptInIsolatedWorld(worldId, scripts)`
  (web-contents.md:1446-1455, which explicitly permits any integer world id —
  `0` is the default world, `999` is Electron's `contextIsolation` world).
  Using `executeJavaScript` would be a *weaker* port.
- **`chrome.tabs.get` is the liveness read for every action** (`cdp.ts`,
  `commands/nav.ts`, `commands/read.ts`, `commands/logs.ts`). Electron has no
  tab registry: the host's own view map is the registry, and
  `webContents.isDestroyed()` (web-contents.md:1170) plus a `destroyed` listener
  replace it.

And one capability the extension does **not** have, which R2 asks us to add:

- **The extension can only drive surfaces it created.** `tabs`
  (`commands/nav.ts:274-299`) enumerates `surfaces` — its own map — one
  `chrome.tabs.get` per stored surface; it never enumerates the user's own
  browser tabs. There is no verb for adopting a tab the user opened. So R2's
  "the agent must be able to drive tabs the user created" is a genuine
  **superset** over the extension host and needs its own guardrail (§6.3).

### 2.7 The UI app today, on Electron 44.3.0

- Main window: `new BrowserWindow({...})` at `src/main/index.ts:251`, with
  `webPreferences` at `:277-298` — `preload` `:278`, **`sandbox: false`** `:279`,
  `devTools: Boolean(process.env.ELECTRON_RENDERER_URL)` `:282`,
  `nodeIntegration: false` `:284`, `contextIsolation: true` `:285`,
  `webSecurity: true` `:286`, `allowRunningInsecureContent: false` `:287`,
  `backgroundThrottling: windowLaunch.backgroundThrottling` `:297`.
- `setWindowOpenHandler` already exists for the main window
  (`src/main/index.ts:348-428`): an auth-popup branch allowing a trusted-domain
  list via `overrideBrowserWindowOptions` (`:399-422`, the popup sub-window gets
  `sandbox: true` `:415`), and everything else denied to `shell.openExternal`
  (`:425-427`).
- Routes: `/chat`, `/chat/:agentId`, `/agents`, `/agents/:agentId`, `/settings`,
  `/agent-hub`, `/agent-hub/:agentId`, `/schedules`, plus a `/` → `/chat`
  redirect and a `*` → `/chat` catch-all (`src/renderer/src/app.tsx:192-206`).
  The shell is `<div className="relative flex h-screen overflow-hidden">`
  (`:157`) → `<SidebarNavigation />` (`:182`) + `<main className="flex grow
  flex-col overflow-hidden">` (`:184`). The rail is
  `RAIL_WIDTH = { expanded: "w-55", collapsed: "w-12" }`
  (`sidebar-navigation.tsx:95`, used at `:242`).
- Overlays that paint over the content area today (`app.tsx:158-180`):
  `CommandPalette`, `ModelsInitializer`, `OnboardingModal`, `ConnectivityBanner`,
  `BackendCompatibilityBanner`, `UpdateNotification`, `LowCreditsDialog`,
  `CreateAgentDialog`. These are exactly the elements §11.3 must hide the view
  for.
- The renderer's privilege boundary is real and narrow: `window.api.desktop.*`
  (`preload/index.ts:12-25`, exposed at `:338-339`) goes through a
  schema-validated allowlist in main, with a sender check
  (`trustedDesktopFrame`, `desktop-transport.ts:84-99`; used at
  `desktop-ipc.ts:108`). `docs/desktop-controls.md:1-32` is the contract.
- **The login/update banner path already exists**: `DesktopNotifier`
  (`desktop-notifier.ts:153`) with its body gate (`:131`).
- `sandbox: false` on the main window is worth noting loudly: the app's *own*
  renderer is the privileged one, which is exactly why the CDP handle must live
  in main and the driven views must be in a different session (§11.7).

**Linux/macOS focus discipline already exists as a module.** `window-raise.ts`
is "the only place in the main process that raises or focuses a window" (`:2-17`),
and `scripts/window-mode.test.mjs` asserts that. `window-mode.ts` resolves
`--window-mode=headless|inactive|normal` from the flag or
`LOCAL_OPERATOR_UI_WINDOW_MODE` (`:34-37`, `:98-100`) with
`backgroundThrottling` off in both non-normal modes (`:98-100`). §11.4 builds on
this rather than inventing a parallel rule.

### 2.8 The multi-identity pairing state, and whether it changes anything

Revision 1's parent asked whether `docs/design/browser-multi-identity-pairing.md`
changes this design. **It does not change the design, but its own header is now
wrong and that matters for anyone reading it as a map.**

- The document's header says "no implementation". **It has been implemented**:
  `26eaa938b feat(bridge): pair several extension identities at once`, with
  follow-up rounds up to `050ab1fb4`. `AGENTS.md:895-961` documents it as
  shipped: the pairing record now holds a legacy trio (the driver's record, kept
  verbatim) plus an `identities` list with **one token per identity**
  (`daemon.py:252-346` region: `_identity_ids` `:276`, `add_identity` `:464`,
  `revoke_identity` `:507`, `note_identity_seen` `:381`); two installs may be
  connected and only one **drives**, the other being a **standby**
  (`HelloAck.role` `protocol.py:390`, `authorized_count` `:393`, the `Role` event
  `:396`).
- **What it changes here: nothing in the topology, and it strengthens §3(b)'s
  rejection.** The machinery that landed is *extension-identity* machinery: an
  Origin pinned to a 32-char `chrome-extension://` id (`daemon.py:2216-2218`,
  close 4004 on no id), a lockstep proto window (`:2227-2244`), a wheel that
  promotes a standby (`_take_free_wheel` `:2186`, `attach` `:997`,
  `is_authoritative` `:1012`), and per-link wedging keyed on "the one
  authoritative socket" (`_drop_unproven_link` `:1797`, `_wire_loss` `:1754`,
  `LINK_SILENCE_TIMEOUT_S = 50.0` `:71`, `LINK_DROP_TTL_S = 60.0` `:90`). A UI
  host is not an extension identity, has no Origin, and — decisively — would have
  to be given a **third role** in a wheel whose whole job is to answer "which one
  socket may serve commands". The daemon is now 3,748 lines (`daemon.py`); the
  rewrite §3(b) declined is larger than it was, and the payoff (saving one small
  HTTP client) is unchanged.
- **One thing it makes possible, which §11.4 records:** the extension host can
  now be paired as a *standby* while something else drives. Nothing in this design
  depends on that, but it means "the extension is connected" no longer implies
  "the extension will answer", so the UI host must not infer bridge health from
  connection state. It already does not (§10.2 uses the same four availability
  judgements).

### 2.9 Documentation defects to fix on the way through

These are in-repo documents that a future agent will read and be misled by. They
are PR-2/PR-3 scope, not separate tickets.

| file | today | must become |
|---|---|---|
| `docs/BROWSER.md:4-5` | "It is **the only browser backend**, and it is advertised only when a cmux CLI can be reached" | three backends; the cmux-only claim is wrong in both halves now (the bridge has existed for a while) |
| `docs/design/browser-extension.md:1` status and §1 | "DESIGN — no code exists yet", `builtin.py:5685`/`:4731` for symbols now at `:10119`/`:7722` | mark the status as *landed* and re-point the symbols, or the doc keeps producing stale citations (revision 1 of this file was written from it) |
| `docs/design/browser-multi-identity-pairing.md` header | "DESIGN PROPOSAL — no implementation", written against `99e81a1c5` | shipped; the record is in `AGENTS.md:895-961` and the code |
| `local-operator-ui/AGENTS.md:212` and `:366` | "Measured on Electron 35.5.1 / macOS", `electron=35.5.1` | the pin is **44.3.0**; these are stale *measurements*, not a stale pin — they should say 35.5.1 *was* the version measured and either be re-measured or labelled |
| `local-operator-ui/AGENTS.md:190,197` | `--remote-debugging-port=9451 --user-data-dir=...` documented as the headless recipe | note that `--user-data-dir` is the *profile* root and that the browser feature adds a partition inside it (§5.3), so an evidence run's `--user-data-dir` must not be confused with the browser profile |

---

## 3. Candidate topologies

### (a) UI-owned loopback endpoint speaking the existing session leg — recommended

The `local-operator-ui` main process listens on `127.0.0.1:<ephemeral>`,
publishes a 0600 state file, and answers the existing `Request`/`Response`
envelope and `METHODS` catalog by driving `WebContentsView`s through
`webContents.debugger` (CDP 1.3).

Blast radius: bounded to `builtin.py`'s seam, a new `local_operator/ui_browser/`
package, and the UI app. Nothing in the extension changes except the import paths
of the modules §12 moves.

Coupling: the UI depends on the *protocol* (generated, versioned, vendored per
§12) and on nothing else in the session. Sessions depend on a state file and one
HTTP POST.

Capability: everything the extension serves except reaching the user's *real*
profile (§4.2).

Security: one new loopback listener with a 0600 key, no CORS headers, no renderer
reachability, no CDP surface exposed — narrower than the alternatives below. R1
raises the stakes of what sits behind it, which is why §9 is now a full section
rather than a subsection.

### (b) UI attached to the existing bridge daemon as a second host role — rejected

Attractive on paper: one session leg already exists, one daemon to install, one
pairing story, one per-tab lock. Rejected on the daemon's own design, and the
case is *stronger* than revision 1's because the daemon has since grown (§2.8).

- The handshake identifies the host by **Origin**, pinned to a 32-character
  `chrome-extension://` id (`daemon.py:2216`: `extension_id =
  self._origin_extension_id(websocket)`, refused with close 4004 at `:2218`). An
  Electron WebSocket client sends no such Origin; to pass it the UI would have to
  *present itself as an extension id*, i.e. impersonate a peer the daemon cannot
  distinguish from the real one — in a daemon that now decides *which identity
  drives*.
- "A later extension wins": a new handshake enters the wheel and can close the
  incumbent (`_take_free_wheel` `:2186`, the close-4000 sites), so a UI that
  dialled in would contend for the wheel with the user's extension. The requested
  cascade (UI *and* extension both available) is impossible without changing that
  rule.
- The wedging machinery is per-link and would have to be rewritten, not extended:
  the generation/wire fence (`is_authoritative` `:1012`, `_wire_loss` `:1754`),
  the silence gate and latch (`LINK_SILENCE_TIMEOUT_S = 50.0` `:71`,
  `LINK_DROP_TTL_S = 60.0` `:90`, `_drop_unproven_link` `:1797`), the
  solicited-ping corroboration, and the single `link.pending` future table
  (`:3094`). Each is keyed on "the one authoritative socket".
- The pairing model has no meaning for the UI: the pairing code exists because a
  *browser extension* is a semi-trusted third party the user must authorise
  (`browser-extension.md` §6.2). The desktop app is the user's own application.

So (b) is not "a second host role"; it is a rewrite of the daemon into a
multi-host broker, in exchange for saving one small HTTP client. The honest
verdict: (a)'s ~120 lines of Python client are cheaper and safer than reworking
the crux of a 3,748-line daemon whose current shape is the residue of five review
rounds.

### (c) UI exposing CDP remotely — rejected

`--remote-debugging-port` and `--remote-debugging-pipe` are process-wide switches
that must be set before `app` is ready
(command-line-switches.md, "Enables remote debugging over HTTP on the specified
`port`"), so neither can be scoped to "while a browser tab exists". Beyond that:

- The **port** has no token of any kind. Its only protection is the loopback
  bind, and the target list it publishes includes the app's *own* renderer — the
  one with `sandbox: false` (`src/main/index.ts:279`), the preload, and
  `window.api.desktop.request` behind it. Any local process could attach to it
  and reach IPC that `docs/desktop-controls.md:8-12` protects with a bearer
  capability.
- The **pipe** is the more defensible of the two (a pipe is not
  network-reachable), so I will not pretend the "no token" objection is decisive
  against it. It is rejected for a different reason: Python would have to receive
  the app's stdio descriptors and speak CDP itself, which moves the whole driver
  into Python, deletes the host (so there is no UI-side owner, no human takeover,
  no tab strip), and couples the session's lifetime to a spawned process's fds.
- Both debug *every* target in the browser process, so the blast radius is the
  entire app rather than the one view that was delegated.

What the UI must do instead: keep the CDP handle in the main process via
`webContents.debugger` (web-contents.md:2535-2537), which needs no listener, no
switch and no credentials, and expose only the RPC vocabulary of §3(a) to the
session.

### (d) Bundling another engine — rejected

Already refused by design (`builtin.py:10125-10131`), and the tool description
tells the model never to install one (`builtin.py:10192-10194`: "never install or
script a browser engine instead"). Worth restating only because "add a browser
tab" is the kind of request that invites a `puppeteer` dependency; it would be a
~150 MB download, a second Chromium, no cookies, and a direct contradiction of
the product's persistence-first rationale. **R1 makes this refusal stronger, not
weaker**: the whole point of R1 is that the jar persists and is reachable by hand,
which a throwaway engine cannot do.

### Comparison

| | blast radius | coupling | capability | security |
|---|---|---|---|---|
| (a) UI loopback endpoint | `builtin.py` seam + new package + UI app | protocol only | everything except the user's real profile | one keyed loopback listener, no CDP exposed |
| (b) UI as daemon host | `daemon.py` crux rewrite | daemon internals, pairing, the wheel | full, plus profile-adjacent | daemon impersonation; single-link semantics broken |
| (c) remote CDP | whole browser process | fds / open port | full, but no UI owner | exposes the privileged renderer |
| (d) another engine | new runtime + packaging | a second browser | no cookie jar | a second engine to patch |

**Recommended: (a)**, with the state-file and error-copy patterns of §10, the
view/tab/profile design of §5–§7 and §11, and the sharing split of §12.

---

## 4. Capability matrix

`BROWSER_ACTIONS` — **19** actions since PR #1318 (`download`, `upload`) — and
the 22 `METHODS` (`protocol.py`). The file counts and `file:line` references
below are the ones this document was written against; the live source is the
constants themselves. "UI" is the UI host's v1 verdict.

R5 asks whether the UI host really is a superset. The answer is **yes for the
wire, yes for the tool, and no for one environmental axis** — and the matrix
below is now argued per row rather than asserted.

| action | UI | mechanism / note |
|---|---|---|
| `open` | yes | creates a `WebContentsView` + tab; the active tab does not change (the extension's `active: false` equivalent, `commands/nav.ts:179`); a pinned `ui:` handle resumes its tab, and `tab_closed` re-creates it (same rule as `_bridge_open`, `builtin.py:9036-9050`). **New in rev 2:** accepts the optional `handle` param (R2 hand-over, §6.3) — the wire's existing `tab` field (`commands/nav.ts:126`, `requireSurface(params.tab)`) |
| `goto` | yes | `webContents.loadURL` (web-contents.md:1087) + settle on `did-finish-load`/`did-fail-load`/`did-navigate-in-page` |
| `read` | yes | `executeJavaScriptInIsolatedWorld` (web-contents.md:1446-1455) — **not** `executeJavaScript`, which would run in the main world |
| `snapshot` | yes | `Accessibility.getFullAXTree` over `webContents.debugger`, then the shared `ax-compact` pruner |
| `screenshot` | yes | `Page.captureScreenshot` over CDP → base64 → Python keeps path resolution, PNG validation (`PNG_MAGIC`) and write approval (`builtin.py:9497-9527`). **Not** `capturePage`: it has visibility semantics (web-contents.md:1815, "The page is considered visible when its browser window is hidden and the capturer count is non-zero") and would change the hidden-window question |
| `click` | yes | `Input.dispatchMouseEvent` at the ref's box, with the extension's ref-epoch rule (§6.4) |
| `type` | yes | `Input.insertText` + read-back comparison |
| `close` | yes | destroys the view, closes the webContents (web-contents.md:1174), removes the tab; `tab_ambiguous` when several are open and no handle was given (existing code, existing copy) |
| `scroll` | yes | the shared `scroll-expressions` JS + `moreBelow`/`moreRight` reporting |
| `logs` | yes | `Runtime.enable` + `Log.enable` + `Runtime.consoleAPICalled`/`Log.entryAdded`/`Runtime.exceptionThrown` into the same ring-buffer shape and level vocabulary |
| `tabs` | yes | the host's own registry (§6.2); handles redacted exactly as `commands/nav.ts:169` redacts them, "your own tab is marked `(yours)`" (`builtin.py:9358-9366`). **New in rev 2:** lists user-created tabs too, and shows a **full** handle only for a tab handed over to the *calling* session (§6.3) |
| `request_access` | yes | reuses the shared pure policy modules (§12); the prompt renders in UI chrome (§9) |
| `await_access` | yes | same bounded-slice polling contract (`await_access` budget 25 s, `protocol.py:263`) |
| `cancel_access` | yes | requester-bound, same semantics; requester identity from `_browser_identity_params` (`builtin.py:8943-8952`) |
| `recover` | yes | `owner_recover` against the UI host's in-process surface ledger; **unreachable on any host without a bridge today** — see §10.5 |
| `retain` / `release` | yes | `owner_retain`/`owner_release`; `release` also drives `owner_finish` when the scope is terminal (`builtin.py:9750-9755`). Same §10.5 gate caveat |
| `status` (method) | yes | the host's own state: proto, app version, profile path, tabs, per-surface url/title |
| `download` | **PR B** | the UI host is the ONLY host that can serve this: `will-download` + `item.setSavePath()` picks the destination before the bytes land, with no CDP primitive and no new permission (the extension's two candidate primitives are refused by Chrome — `docs/design/browser-file-transfer.md`). The harness already refuses it on a host whose record does not advertise it, so an old app answers with a typed `capability_unsupported` rather than a mystery failure |
| `upload` | yes | `DOM.setFileInputFiles` over `webContents.debugger`, plus the read-back comparison; the shared `driver/file-transfer-policy.ts` is vendored for the host-side name check, and Python's `browser_files.check_upload` remains the control |
| `retitle` (method) | **served as a no-op, never called** | the UI has no tab *groups*; Python should skip `ui:` the way it already skips cmux for this method (`retitle_browser_surface` returns early for anything not `bridge:`, `builtin.py:9549`). The UI tab label is the page title, which is strictly more informative than a session title |
| `owner_recover` / `owner_finish` / `owner_retain` / `owner_release` (methods) | yes | implemented against an in-process ledger keyed by `session_id` + `owner_proof` + `owner_generation`, with the same param contract (`resources.py:248-280`). Requires the Python-side host selection of §10.5 |

Nothing is outright refused at the wire level, which is a property worth naming:
unlike the cmux path, the UI host is a *superset* of the bridge's wire
capability. The UI's limitations are environmental, not protocol.

### 4.1 The cmux-degrade copy must change

`BRIDGE_ONLY_BROWSER_ACTIONS` (`builtin.py:7542-7545`) is
`{"scroll", "logs", "tabs", "request_access", "await_access", "cancel_access"}` —
really "cmux cannot serve these" — and its messages assert that the *extension*
is the only alternative. On a three-backend host those sentences are wrong. Exact
sites, re-derived:

| site | today | must become |
|---|---|---|
| `builtin.py:9943-9952` (`BRIDGE_ONLY_BROWSER_ACTIONS` degrade) | "use the Local Operator browser extension (run 'lop browser status' / 'lop browser install' …)" | name both non-cmux hosts and pick by availability |
| `builtin.py:9921-9934` (`tabs`) | "use the Local Operator browser extension …" | same |
| `builtin.py:9883-9892` (access actions) | "This action only exists for the Local Operator browser extension" | same |
| `builtin.py:9862-9869` (no backend) | "neither cmux nor a connected Local Operator browser extension is reachable. Run 'lop browser status' and 'lop browser install' to set up the bridge." | three-way; add the app's browser tab, and stop recommending `install` on a host that has the app |
| `builtin.py:8565-8602` `_bridge_absent_result` | "the bridge daemon … remembers this browser, but nothing is connected to it right now" | stays BRIDGE-ONLY, and no UI sibling is expected: the state it describes is a *paired* browser that is not attached, and the app's host has no equivalent — a tab is a child of the running app, so "remembered but not attached" cannot arise there (see §10.2). The daemon-only `extension_id` branch is bridge-only for the same reason. |
| `builtin.py:8605-8634` `_bridge_demotion_hint` | names `lop browser status --repair` | needs a host argument; the UI hint names the app |
| `builtin.py:7537-7541` docstring | "Actions that only the Local Operator browser extension can serve" | rewrite: cmux cannot serve them; both non-cmux hosts can |
| `builtin.py:7629-7642` `BrowserParams.action` description | "on the extension backend a fresh open creates a NEW tab" | describe the general rule, not one host; add the `handle` param's semantics |
| `builtin.py:10166-10195` tool description | names "a cmux browser panel or their paired Local Operator browser extension" | add the desktop app's browser tab |
| `backend.py:28-73` `ERROR_MESSAGES` | extension-specific copy: "ask the user to open their browser", "toggle the … extension OFF then ON in chrome://extensions" | split into per-host tables; shared codes keep shared copy |
| `backend.py:216-222` `origin_denied` / `format_error` | "ask the user to allow it from the extension popup" | parameterise the approval surface |
| `docs/BROWSER.md:4-5` | cmux is "the only browser backend" | rewrite for three backends |
| `guides/browser/GUIDE.md:15-32` | "The `browser` tool has three possible backends, in preference order: 1. **The Local Operator browser extension** (preferred) … This is what you should set up and use." (and the `description:` line above it) | three HOSTS with the app's browser tab first, the extension as the host for a session that lives only in the user's own profile, cmux below both and `bash`+curl last |
| `prompts_md/system.md:258-270` (`has_browser`) | "The preferred backend is the **Local Operator browser extension** … a cmux browser panel is the fallback where the extension is not installed" | the app's browser tab is preferred; the extension is the real-profile host; cmux is the fallback where neither is connected |
| `prompts_md/system.md:282-293` (`no_browser`) | "the host has neither backend connected … Only when the user declines the extension and no cmux panel exists" | three hosts, not two: name the app's browser tab as a setup path BESIDE the extension (`lop browser install` is not the only way in), and require declining both before falling back to `bash`+curl |

I would **keep the constant name** `BRIDGE_ONLY_BROWSER_ACTIONS` only if the
docstring is rewritten; the name is now actively misleading (the UI serves every
one of them) and tests reference it. Renaming to
`CMUX_UNSUPPORTED_BROWSER_ACTIONS` is a mechanical grep-and-rename in
`builtin.py` + `tests/unit/tools/test_browser_tool.py`; I recommend doing it in
PR 2 while the seam is being touched.

### 4.2 The genuinely environmental gaps (R5, answered)

Revision 1 said "the UI's limitations are environmental (no user profile, no tab
groups), not protocol". That is still right, but R1 removed the first one, so the
list is now one item and one asymmetry:

| gap | environmental or capability? | verdict |
|---|---|---|
| **The user's real browser profile** — their Chrome/Arc cookies, device trust, hardware keys, enterprise conditional access | **Environmental.** The UI drives its own persistent jar (§5); the extension drives their real one. Nothing in the protocol prevents the UI from driving a real profile — it is a policy choice that this app must not attach to the operator's banking profile | accepted, and now *much* smaller: R1 gives the UI jar real persistence, so the remaining gap is only profiles an *existing* device-bound session lives in |
| **Chrome tab groups** (`chrome.tabGroups`, `tab-groups.ts`, 187 LOC) | **Environmental.** Electron has no tab-group API at all. There is no equivalent to substitute | replaced by the tab strip. This is *presentation*, never capability: groups exist to label the agent's tabs in someone else's browser, and the UI's strip labels its own tabs directly |
| **Reaching tabs outside the app window** | **Capability, deliberately withheld.** `chrome.tabs.query` sees every tab in the user's browser; the UI host sees only its own views | refused on purpose (§11.7, and §6.3's hand-over exists precisely because the boundary is real) |
| `chrome.alarms`, `chrome.notifications`, `chrome.identity`, `chrome.cookies`, `chrome.webNavigation`, `chrome.debugger` | **Environmental substitutions**, each with a documented equivalent: alarms → not needed (no MV3 worker); `webNavigation` → `did-*` events; `chrome.debugger` → `webContents.debugger`; `cookies`/`identity`/`notifications` → the host's own main-process APIs (`ses.cookies`, `DesktopNotifier`) | none is a *capability* loss for the driver; §12's table records which module needs which substitution |

**So: no gap is a capability gap, and every environmental gap is either replaced
by a documented equivalent or is a policy refusal.** The parity claim is
*argued*, not measured; §15 names the probe that would measure it (drive both
hosts against the same page and diff the snapshots).

---

## 5. R1 — The persistent, shared browser profile

### 5.1 The partition: one name, shared by everything

**Decision: `persist:local-operator-browser`** — one persistent Electron session
shared by **every browser tab the user creates and every tab any agent session
opens**. This is the same name revision 1 chose; the change is its *scope*: it is
no longer per-tab or per-surface, it is the single jar for the whole feature.

Verified semantics (session.md:26-44): `session.fromPartition(partition)` "When
there is an existing `Session` with the same `partition`, it will be returned";
"if `partition` starts with `persist:`, the page will use a persistent session
available to all pages in the app with the same `partition`"; with no `persist:`
prefix it is in-memory. The `partition` webPreferences key carries the same rule
(web-preferences.md:34-39). `ses.isPersistent()` (session.md:1390-1395) confirms
it: "When creating a session from a partition, session prefixed with `persist:`
will be persistent, while others will be temporary."

**This shared partition is what makes R1 and R4 possible at all, and R4 requires
it.** Electron's extension docs say plainly: "loading extensions is only supported
in persistent sessions. Attempting to load an extension into an in-memory session
will throw an error." (extensions.md; repeated at extensions-api.md,
"Loading extensions into in-memory (non-persistent) sessions is not supported and
will throw an error.") So the profile requirement and the extensions requirement
converge on the same object, and neither is optional given the other.

How it is created and wired:

- The host resolves the session **once**, in main, after `app.whenReady()`:
  `const browserSession = session.fromPartition(BROWSER_PARTITION)`.
- Every tab's `WebContentsView` gets
  `new WebContentsView({ webPreferences: { partition: BROWSER_PARTITION, ... } })`
  (`WebContentsView` constructor options: web-contents-view.md:40-46).
- Because the session is shared, the **session-level** handlers are installed once,
  not per view: `setPermissionRequestHandler` / `setPermissionCheckHandler`
  (session.md:928, :1001), `will-download` (session.md:91), `setUserAgent`
  (session.md:1377). Only the per-`webContents` handlers are per view:
  `setWindowOpenHandler` (web-contents.md:1463), `will-navigate`,
  `will-redirect`, `did-*` (web-contents.md:120).
- The app's **own** renderer keeps the default session (it sets no `partition`,
  `src/main/index.ts:277-298`). So the app's own session and the browser jar are
  different persistent sessions, and §11.7 leans on that.

### 5.2 What persists, and what does not

Chromium's per-session profile is the mechanism; the honest table is:

| data | persists across app restart? | why / how to check |
|---|---|---|
| **Persistent cookies** (with an `Expires`) | **yes** | written to the profile's cookie store |
| **localStorage** | **yes** | part of the profile's DOM storage |
| **IndexedDB** | **yes** | part of the profile |
| **Service worker registrations and CacheStorage** | **yes** | written under the profile; `clearStorageData`'s storage list names `serviceworkers` and `cachestorage` as first-class, which is the evidence they are real stored state (session.md:709-712) |
| **HTTP cache** | **yes** | the profile's disk cache (`clearCache`, session.md:698) |
| **HTTP auth (server) credentials** | **unverified** — probe | Chromium keeps in-memory auth cache and has profile-level persistence for some schemes; Electron does not document it. *Probe P1 (§15).* |
| **Session cookies** (no `Expires`) | **unverified, and probably no** — probe | By Chromium's design session cookies are in-memory and die with the browser process; Electron performs no session restore, so the UI host does not get Chrome's "continue where you left off" behaviour that saves them. **This is the one honest caveat on R1 and it must be measured, not promised.** *Probe P2 (§15).* |
| **sessionStorage** | **no** | `sessionStorage` is per-tab by definition and dies with the tab; restoring a tab recreates a document, not its sessionStorage. Do not claim otherwise to the user |
| **Per-tab scroll position and form state** | **yes, if §7's restore is used** | `NavigationEntry.pageState` is "a base64 encoded data string containing Chromium page state including information like the current scroll position or form values" (structures/navigation-entry.md) |

**The R1 acceptance test the user will actually apply** is: "I log into
<site> inside the app, quit the app, reopen it, and I am still logged in."
Probe P2 is that exact experiment, and if it fails for a session-cookie-only site
the honest answer is that the site's own cookie policy is what fails, not the
profile — and the UI copy must say so rather than the feature quietly looking
broken.

### 5.3 Where it lands on disk

- Electron names the root: `ses.getStoragePath()` returns "The absolute file
  system path where data for this session is persisted on disk. For in memory
  sessions this returns `null`." (session.md:1712-1715; property form
  `ses.storagePath` at :1767-1770). **The host must call this and publish it**,
  both in its log and in the state file's `profile_dir` (§10.1), so that "where
  are my logins kept" has a one-command answer.
- That path is under `app.getPath('sessionData')`, which "by default … points to
  `userData`" (app.md:634-640); `userData` is "the `appData` directory appended
  with your app's name" (app.md:625-633). So on macOS the jar is under
  `~/Library/Application Support/Local Operator/` — and Electron itself
  recommends "store app-specific files within a subdirectory of `userData` …
  rather than directly in `userData` itself, to avoid naming conflicts with
  Chromium's own subdirectories (such as `Cache`, `GPUCache`, and `Local
  Storage`)".
- **The exact leaf directory name for a `persist:` partition is not documented by
  Electron.** *Probe P3 (§15):* after a cookie write in a scratch profile, list
  the tree under `getPath('userData')` and record the real path, then pin it in
  the doc. Until then the design uses `getStoragePath()` and never a
  hand-constructed path — that is the whole point of citing the API.
- **Do not put the approval store (§9) inside the profile directory.** The
  profile is the thing a user clears when they want to "log out of everything";
  the per-origin approvals are a *policy* record and belong beside the app's own
  settings, so that clearing data and revoking approvals are two separate,
  separately-explained actions. The tab-session file (§7) is policy too, and lives
  beside the approvals.

### 5.4 Lifecycle: reset, update, uninstall

**Reset — a "Clear browsing data" affordance, and it is required, not optional.**
R1 makes the app hold the operator's real sessions; there must be a way to
destroy them without deleting the app. Electron 44 offers two APIs and the
design uses the newer, broader one:

- `ses.clearData(options)` (session.md:1717-1752) — "Clears various different
  types of data… more thorough than the `clearStorageData` method", with
  `dataTypes` (`backgroundFetch`, `cache`, `cookies`, `downloads`, `fileSystems`,
  `indexedDB`, `localStorage`, `serviceWorkers`, `webSQL`), `origins`,
  `excludeOrigins`, `originMatchingMode`. It documents the cookie-scope surprise:
  "Cookies are stored at a broader scope than origins… clearing cookies for the
  origin `https://really.specific.origin.example.com/` will end up clearing all
  cookies for `example.com`" — so a per-origin "forget this site" must say it
  forgets the registrable domain, and the UI must not promise finer granularity
  than Chromium has.
- `ses.clearStorageData(options)` (session.md:704-714) — the older, narrower list
  (`cookies`, `filesystem`, `indexdb`, `localstorage`, `shadercache`,
  `serviceworkers`, `cachestorage`).
- `ses.clearCache()` (session.md:698) for the cache-only case.
- `ses.flushStorageData()` (session.md:716-718) — "Writes any unwritten DOMStorage
  data to disk." Worth calling before a `clearData`, and worth calling on quit so
  a hard kill does not lose just-written storage.

The affordance, in plain words, in the browser feature's own Settings section:
**"Clear cookies and site data"**, **"Clear cache"**, **"Clear everything"**, each
naming what it does and that it will sign the user out. And a separate
**"Revoke all site approvals"** (§9.4). Three buttons that each do one honest
thing beat one button whose behaviour nobody can predict.

**On app update: the profile is not touched.** It lives in `userData`, outside
the app bundle, and neither `electron-updater` nor the packaging path deletes it.
The honest risk is Chromium's own profile-format migration: a major Chromium step
forward migrates the profile automatically, and a *downgrade* can leave it
unreadable. Because the jar holds nothing durable but logins, the mitigation is
the reset affordance plus this sentence in the docs — not a migration shim.
*Probe P4 (§15):* update the app across an Electron major once in QA and confirm
the logged-in session survives.

**On uninstall: the profile survives, and that is correct.** On macOS, moving the
app to the Trash does not remove `~/Library/Application Support/<app>`, so the jar
(and therefore logins) outlives the app. The design **does not** delete it on
quit/uninstall: silently destroying a user's saved sessions is a destructive act
the user did not ask for, and it would break the ordinary "reinstall and I'm still
logged in" expectation. The doc and the app's Settings must say where it is and
offer the clear buttons.

**What the profile must never hold:** the operator's real-browser profile. The
app never attaches a `WebContentsView` to `session.defaultSession` and never
points a `partition` at another app's data directory.

---

## 6. R2 — Chrome, the tab registry, and two actors in one window

### 6.1 The controls

Rendered in the renderer's DOM, above the view rectangle (§11.2), all in the
brand's roles (§11.2 note on `docs/branding.md`):

| control | behaviour | Electron API |
|---|---|---|
| **tab strip** | one entry per tab: page title, agent/user ownership marker, active marker, close affordance; append-only ordering with the active tab highlighted; the strip already exists as a pattern in the app's rail and list panes | renderer DOM |
| **new tab** | creates a user-owned tab at `about:blank`, becomes active | `new WebContentsView(...)` → registry insert |
| **close tab** | closes the tab in the strip; closing an agent-owned tab is allowed and forces the agent onto its `tab_closed` → re-create path | `view.webContents.close()` (web-contents.md:1174) + `removeChildView` |
| **URL bar** | shows the **active tab's live URL**, updated from navigation events, never from what was requested; Enter navigates the active tab | `webContents.getURL()` (`:1152`), `getTitle()` (`:1166`), `loadURL` (`:1087`) |
| **back / forward** | enabled from history state | `contents.navigationHistory.canGoBack()` / `goBack()` / `canGoForward()` / `goForward()` (navigation-history.md:21-57). **Note:** the old `contents.canGoBack()`/`goBack()` are *deprecated* in 44 (`web-contents.md:1223-1293`, "Should use the new `contents.navigationHistory.…` API") — the design must not use them |
| **reload / stop** | reload toggles to stop while loading | `webContents.reload()` (`:1215`), `webContents.stop()` (`:1211`), `isLoading()` (`:1197`) |

**The honest behaviour of a typed URL vs the agent's navigation** — this is worth
stating precisely because it is an asymmetry, not a bug:

- A URL the **user types** navigates the **active tab**, as a browser does. It
  does not create or consume an agent handle, and it does **not** pass through the
  agent's per-origin approval gate: the user typing their own URL *is* the
  consent. Non-`http(s)` input is refused, mirroring `_BROWSER_URL_SCHEMES`
  (`builtin.py:7623`) and the extension's independent re-check.
- A URL the **agent** navigates goes through `open`/`goto` and **is** gated: a
  top-level navigation to an origin the user has not approved fails early with
  `origin_not_allowed` (`protocol.py:296`), and the agent runs the
  `request_access` → notify → `await_access` flow (§9).
- The consequence to document in the UI: an origin can be reachable by the user
  and still refused to the agent. That is the intended direction of the gate, and
  the consent bar should say so in words rather than leaving the user to infer it.

### 6.2 The tab registry, and how a tab is identified across the two actors

**The main process owns the registry.** One `Map<number, TabRecord>`, no parallel
truth in the renderer (the renderer gets a projection over IPC and sends
intents). Sketch of the record (a design sketch for the coder, not a schema
promise):

```ts
type TabOwner = "user" | "agent";

interface TabRecord {
  tabId: number;                 // host-minted, monotonic, stable for the tab's life
  view: WebContentsView;
  owner: TabOwner;
  sessionId: string | null;      // which lop session owns it, when owner === "agent"
  nonce: string | null;          // the capability; null for a user tab that has not been handed over
  handedTo: string | null;       // session id a user tab has been handed to (§6.3)
  restored: boolean;             // §7
  createdAt: number;
}
```

- **`tabId` is the host's own integer**, not a `webContents.id`, not a
  Chromium tab id. It is the `<n>` in the surface token and it is what both actors
  can be shown.
- **Identity across the two actors.** The **user** addresses a tab by its position
  in the strip (and its title); the renderer IPC carries `tabId`. The **agent**
  addresses a tab only by the surface token `ui:<tabId>:<nonce>`. **The nonce is
  the authority, not the `tabId`.** A token with the wrong nonce, or a stale one,
  is refused — that is what makes the handle a capability rather than a name, and
  it is the same discipline as `bridge:<tab>:<nonce>`
  (`builtin.py:9053-9058`) and the extension's `requireSurface(token)`
  (`commands/nav.ts:127`).
- **What the agent can see.** `tabs` lists every tab in the registry. Handles are
  redacted per the existing rule (`builtin.py:9358-9366`, and the extension's
  rationale at `commands/nav.ts:169`: "the full token is the drive capability …
  an error surface must not hand one session control of another's tab"), **except**
  for a tab this calling session may drive — its own, or one handed to it (§6.3).
  The requester identity needed to make that decision already exists:
  `_browser_identity_params` sends `requester` and `session_label`
  (`builtin.py:8943-8952`), derived from `_browser_requester` (`builtin.py:8637-8648`).
- **`status`** reports the registry's summary (proto, app version, profile path,
  tab count, per-surface url/title) for `lop browser status` and for QA.

### 6.3 Ownership, and driving a tab the *user* created (the R2 superset)

This is the one place the UI host goes beyond the extension, and it needs a
guardrail rather than a paragraph of good intentions. The extension never faces
this problem because it has no user tab to adopt (§2.6).

**The rule: the agent may drive a tab it created, or a tab the user has
explicitly handed over. Nothing else.**

- A tab the **agent creates** (`open`) gets `owner: "agent"`, `sessionId`, and a
  nonce at creation. This is exactly today's behaviour.
- A tab the **user creates** has no nonce. It is listed in `tabs` (awareness) with
  a redacted handle, exactly like another session's tab. The agent cannot drive
  it, and cannot guess its way in — the token requires a nonce it has never been
  given.
- **Hand-over** is one affordance in the tab strip: "Let the agent use this tab".
  It names the **session** (the UI knows the live sessions — it already speaks
  `sessions.list` through the desktop transport, `docs/desktop-controls.md:57-60`)
  and mints a nonce, setting `handedTo = sessionId`. From then on `tabs` shows
  that session the **full** handle with the marker `(handed to you)`.
- **Taking it:** the agent calls `open` with that handle. This is the one
  model-facing addition in this whole design, and it is the smallest possible:
  **one optional field on an existing tool's existing schema**, mapped onto a
  field the wire already carries (`open`'s `params.tab`,
  `commands/nav.ts:126-137`). Per the footprint ladder (`AGENTS.md:2194-2196`)
  this is rung 1 ("extend an existing tool… A new parameter or mode on a tool
  that already exists costs no new schema"), not a new tool and not a new gating
  convention. `_validate_browser_args` (`builtin.py:7982`) gains one clause
  refusing `handle` on every action except `open`, in the same spirit as every
  other refusal there.
- **Adoption is refused for an unhanded tab**, with a typed refusal (reuse
  `OWNER_REFUSED`'s shape in `protocol.py:317`, or a dedicated code — the coder
  should choose at implementation time and say which; do not overload
  `tab_closed`).
- **Revoking a hand-over** is a second affordance ("Stop letting the agent use
  this tab") which nulls the nonce; the agent's next action on that handle gets
  the refusal, and `open` with that handle fails the same way. A hand-over does
  **not** survive an app restart (§7): the nonce is not persisted.
- **The hand-over is not the origin approval.** Both must hold. Handing over a tab
  that sits on an unapproved origin lets the agent *see* the tab but not navigate
  it: the agent's `goto` still fails `origin_not_allowed` until the origin is
  approved (§9). Two independent gates, and the UI should say so once, in the
  hand-over confirmation — "the agent can use this tab, and can only reach sites
  you have approved".

### 6.4 Concurrency: the user clicks while an agent command is in flight

The extension's answer, and the one this design adopts, is **no lock, plus
ref-epoch discipline**:

- **There is no "agent busy" modal and no exclusive claim on a tab.** The same
  contract cmux's panel has, and for the same reason: the user is sovereign over
  their own window, and a modal would reproduce the failure class the async
  approval flow was built to remove.
- **Every action reports what is actually on screen, never what was requested** —
  `_page_line` (`builtin.py:8185-8192`) is already the tool's rule: "so a redirect,
  a login wall or a consent interstitial shows up in the transcript rather than
  hiding behind the model's own intent". So if the user navigates the tab first,
  the agent's next `read` reports the user's page, not the agent's intent.
- **Snapshots carry an epoch, and a navigation bumps it.** The extension does this
  on both `open`-resume and `goto` (`commands/nav.ts:135-138`, `:145`);
  `requireSurface` + the epoch gate is what makes "a pre-navigation ref pushed
  against a document that no longer exists" fail with `element_not_found` rather
  than clicking something else. **This is the mechanism that makes concurrent
  interaction safe rather than merely untested**, and PR 3 must implement it and
  PR 4's tests must exercise it (agent `snapshot` → user navigates → agent `click`
  → typed `element_not_found`).
- **Commands are serialised per tab in the host**, across sessions. The session
  side already serialises per session (`resource.lock`, `execute_browser`
  `builtin.py:9658`), but two sessions can target different tabs and one session
  can target one tab — the host owns the per-tab queue. A second command on a
  busy tab does not queue silently forever: it gets the existing `busy` code
  (`protocol.py:309`) with the existing "retry this action once" copy.
- **A user-initiated close of an agent tab** is the ordinary `tab_closed` path:
  the agent's next action gets the existing copy ("browser tab … is gone; dropped
  the handle. Use 'open' with a URL to get a new tab", `format_error`
  `backend.py:211-215`) and `open` re-creates. Already implemented at
  `builtin.py:9036-9050` for the bridge; PR 2 extends the same branch to `ui:`.

### 6.5 Caps

- **Agent-owned concurrent tabs: 8**, mirroring the extension's `MAX_SURFACES = 8`
  (`extension/src/state.ts:12`), returning the existing `TAB_LIMIT` code and
  copy (`protocol.py:303`; the copy at `commands/nav.ts:168`). The reasoning
  transfers unchanged: "an agent fleet could spray tabs into the user's real
  browser" — the same reasoning applies to the app.
- **User-owned tabs: no agent-facing cap.** Revision 1 recommended capping the
  whole strip at 8; that was written before there was a user-facing tab strip. A
  user opening a dozen tabs is their business, and inventing a second number with
  no evidence behind it is exactly what the old doc warned against. The honest
  bound on user tabs is *memory* (each is a real `WebContents`), so the host
  reports the live view count, `status` carries it, and §15 has a QA cell that
  measures RSS against tab count (probe P10). If that measurement shows a problem,
  a number gets chosen *from the numbers*.
- **`MAX_SURFACES` is agent-scoped in the UI host too**, which is a deliberate
  divergence from the extension's global counter: in the extension every surface
  is an agent tab, so the two are the same number.

---

## 7. R3 — Tab restore across restarts

**Decision: yes, restore.** The operator asked for "similar to my personal
browser", and Electron 44 provides a real mechanism rather than a hand-rolled
one — which is why this revision reverses revision 1's non-goal.

### 7.1 The mechanism, cited

Electron 44 has `webContents.navigationHistory` (web-contents.md:2518), whose
`restore(options)` is documented as (navigation-history.md:87-103):

> Restores navigation history and loads the given entry in the in stack. Will make
> a best effort to restore not just the navigation stack but also the state of the
> individual pages — for instance including HTML form values or the scroll
> position. **It's recommended to call this API before any navigation entries are
> created, so ideally before you call `loadURL()` or `loadFile()`** … `entries`:
> Result of a prior `getAllEntries()` call; `index`: Index of the stack that
> should be loaded.

`getAllEntries()` returns `NavigationEntry[]` where each entry is
`{ url, title, pageState? }` and `pageState` is "A base64 encoded data string
containing Chromium page state including information like the current scroll
position or form values" (structures/navigation-entry.md). `getActiveIndex()`
(navigation-history.md:39-41) gives the index to restore at.

So a faithful restore is: snapshot `getAllEntries()` + `getActiveIndex()` per tab,
write them down, and on launch create the view and call `restore({entries,
index})` **before any `loadURL`**.

### 7.2 Where the session state is stored

A private JSON file owned by the **UI host**, beside the approval store (§9), not
inside the browser profile and not in lop's config dir:

- `userData/browser/session.json`, 0600, staged write + `os.replace`/atomic
  rename, written on a debounce (tab create/close/navigate) and on `before-quit`.
- Shape: per tab — `tabId`, `owner`, `restored`, `active`, `entries[]`,
  `activeIndex`. **No nonce, ever** (§7.3), and **no hand-over** (§6.3).
- Writes go through the same discipline as `browser_bridge/state.py`
  (`state.py:99-117`): staged write, `os.replace`, 0600 under a 0700 dir. The
  pattern is the requirement, so it should be copied with its comment.
- `session.json` is *policy + convenience*, so it is neither cleared by "Clear
  cookies and site data" nor required for correctness. If it is corrupt or
  missing, the app opens one blank tab. **A failed restore must never block
  startup** — one `try`/`catch` around the whole read, and a log line, exactly as
  `_bridge_liveness` guards a diagnostic (`builtin.py:8555-8562`).

### 7.3 A restored tab and a stale agent handle — the part that must not be got wrong

**A nonce is never re-issued across a restart.** On restore, every restored tab
gets a **fresh** `tabId` and **no** nonce, and the tab is restored as
`owner: "user"` regardless of what it was before. Consequences, stated:

- An agent session that survives an app restart (lop sessions are independent of
  the UI process) holds `ui:<oldTabId>:<oldNonce>`. Its next action fails the
  handle check, and the existing recovery contract applies: the typed
  `tab_closed` copy, and `open` re-creates (`builtin.py:9036-9050`). **No
  special-case code is needed** — the pinning discipline already produces exactly
  the right behaviour, which is a good sign that the discipline is right.
- An agent does **not** silently reclaim a restored tab, and a restored tab is not
  automatically agent-owned even if an agent had opened it. That is deliberate:
  re-granting drive authority to whoever's token happens to be written in a file
  is precisely the fail-open the capability model exists to prevent, and
  `session.json` sits in `userData` where anything on the machine can read it.
  The user can hand the tab back (§6.3) — that is one click, and it is a *human*
  click, which is the property worth keeping.
- **URLs are restored, and the pages are re-fetched.** A restored tab is a fresh
  navigation to the same URL, not a revived process. So a page that has since
  logged out shows logged out, which is honest, and a page that needs a POST
  cannot be restored (there is no POST replay) — the restored entry is the GET
  the browser actually has in history.

*Probe P5 (§15):* with two tabs of a real site (one navigated twice), quit and
relaunch, confirm the tab count, order, per-tab history depth, scroll position and
form values, and confirm an agent's stale handle produces `tab_closed` and not a
drive of the restored tab.

---

## 8. R4 — Extensions, measured not asserted

### 8.1 What Electron 44 actually supports (cited, verbatim where it matters)

From `docs/api/extensions.md` at v44.3.0:

- "Electron supports a subset of the Chrome Extensions API… primarily to support
  DevTools extensions and Chromium-internal extensions, but it also happens to
  support some other extension capabilities."
- "**Electron does not support arbitrary Chrome extensions from the store, and it
  is a non-goal of the Electron project to be perfectly compatible with Chrome's
  implementation of Extensions.**"
- "Electron only supports loading unpacked extensions (i.e., `.crx` files do not
  work). Extensions are installed per-`session`."
- "Loaded extensions will **not** be automatically remembered across exits; if you
  do not call `loadExtension` when the app runs, the extension will not be
  loaded."
- "**Note that loading extensions is only supported in persistent sessions.
  Attempting to load an extension into an in-memory session will throw an
  error.**"

**Supported manifest keys** (the whole list): `name`, `version`, `author`,
`permissions`, `content_scripts`, `default_locale`, `devtools_page`, `short_name`,
`host_permissions` (Manifest V3), `manifest_version`, `background` (**Manifest
V2**), `minimum_chrome_version`.

Two things follow, and both matter for R4's honesty:

1. **MV3 is not categorically excluded.** `manifest_version` and
   `host_permissions` (Manifest V3) are supported keys, and `chrome.scripting` is
   listed as "All features of this API are supported". So an MV3 extension's
   content scripts and scripting calls are inside the supported surface.
2. **`background.service_worker` is not in the list — only MV2 `background`.** An
   MV3 extension whose entire logic lives in its service worker therefore has no
   *documented* runtime. **Whether Electron actually runs an MV3 background
   service worker is not answered by the docs in either direction**, so this
   design does not answer it either: it is *probe P6* (§15), and the UI reports
   what happened rather than predicting it. This is the single most important
   sentence in §8 and it is why §8.4 is written the way it is.

**Supported extension APIs** (the whole list, with the caveat that "Other APIs
may additionally be supported, but support for any APIs not listed here is
provisional and may be removed"): `chrome.devtools.inspectedWindow` (all),
`chrome.devtools.network` (all), `chrome.devtools.panels` (all),
`chrome.extension` (`lastError`, `getURL`, `getBackgroundPage`),
`chrome.management` (`getAll`, `get`, `getSelf`, `getPermissionWarningsById`,
`getPermissionWarningsByManifest`; events `onEnabled`, `onDisabled`),
`chrome.runtime` (`lastError`, `id`, `getBackgroundPage`, `getManifest`,
`getPlatformInfo`, `getURL`, `connect`, `sendMessage`, `reload`; events
`onStartup`, `onInstalled`, `onSuspend`, `onSuspendCanceled`, `onConnect`,
`onMessage`), `chrome.scripting` (all), `chrome.storage.local` only
(`sync` and `managed` are **not** supported), `chrome.tabs` (`sendMessage`,
`reload`, `executeScript`; `query` partial — `url`, `title`, `audible`, `active`,
`muted`; `update` partial — `url`, `muted`; **`-1` as a tab id raises**),
`chrome.webRequest` (all, with Electron's own `webRequest` taking precedence).

**Explicitly absent from that list:** `chrome.debugger`, `chrome.alarms`,
`chrome.notifications`, `chrome.cookies`, `chrome.webNavigation`,
`chrome.identity`, and native messaging. The operator's own survey of these is
confirmed by the docs, and this design treats the absence as a fact about the
supported surface rather than as a per-extension guess.

The **loading API** moved in 44 (extensions-api.md): `ses.extensions.loadExtension(path[, options])`
on the `Extensions` class reached via `session.extensions`, with
`options.allowFileAccess`; `resolveExtension`-adjacent helpers
`removeExtension(id)`, `getExtension(id)`, `getAllExtensions()`, and the events
`extension-loaded`, `extension-unloaded`, `extension-ready`. **`ses.loadExtension`
still exists but is `_Deprecated_`** (session.md:1626-1710, "Deprecated: Use the
new `ses.extensions.loadExtension` API"), so the design uses the new namespace —
this is a correction to any instruction that says "call `session.loadExtension`".
Other constraints from the same page: "This API cannot be called before the
`ready` event of the `app` module is emitted"; "If there are warnings when
installing the extension (e.g. if the extension requests an API that Electron
does not support) then they will be logged to the console."

### 8.2 The managed directory and the load flow

**A directory the user fills, and the app loads.** Not a store, not a `.crx`
installer, not a download-and-extract helper in v1.

```
<userData>/browser/extensions/
  <one-directory-per-extension>/      # an UNPACKED extension, as the user unzipped it
```

- **Placement:** `userData/browser/extensions/` — beside `session.json` and the
  approval store, i.e. app policy area, not inside the Chromium profile
  (`userData/Partitions/...`) and not inside `userData/Extensions/` (which is
  Chromium's own, for internally-installed extensions). A short
  `browser/extensions/README.txt` written on first launch states the rule in one
  sentence ("unpack each extension's folder here, then restart the app"), because
  the alternative is a support conversation.
- **Load, once per run, after `app.whenReady()` and before the first tab can be
  created** (§8.1: `loadExtension` cannot be called before `ready`):

  ```ts
  for (const dir of listUnpackedExtensionDirs(extensionsRoot)) {
    try {
      const ext = await browserSession.extensions.loadExtension(dir, { allowFileAccess: false });
      record({ dir, id: ext.id, name: ext.name, version: ext.version, ok: true });
    } catch (err) {
      record({ dir, ok: false, error: String(err) });
    }
  }
  ```
  `allowFileAccess: false` is the default and is stated explicitly because the
  only documented reason to set it true is DevTools-extension injection into
  `file://` pages, which this feature does not want.
- **Every run reloads.** §8.1's "will not be automatically remembered across
  exits" is the whole reason the load loop runs on every launch; there is no
  enable/disable state to persist, and the app must not pretend there is one.
  The UI's per-extension switch therefore means "do not load this one this run",
  which is a kept-value in the app's own settings and nothing more.
- **Reconcile with what actually loaded**: after the loop,
  `browserSession.extensions.getAllExtensions()` is the truth, and the UI shows
  that list, not the directory listing. An extension the directory lists but the
  session does not is shown as "failed to load" with its error.
- **The loaded extension runs in the browser session**, so its content scripts
  apply to the pages the *agent* drives — because they are the same session
  (§5.1). That is the point (a content-script extension can help on a driven page)
  and it is also a security statement, so it is in §9: **a loaded extension's
  content scripts can read and modify every page the agent and the user browse in
  this app**, and `chrome.scripting` being fully supported means it can inject
  deliberately. The UI must say so on the extensions screen.

### 8.3 The supported/unsupported matrix, computed and shown

The screen the operator asked for is concrete: for each extension in the
directory, show **what it declared**, **what Electron supports**, and **what is
therefore expected to work** — computed from the manifest, not narrated.

- Inputs available without guessing: the unpacked directory's `manifest.json`, and
  (once loaded) `Extension.manifest`, `.name`, `.version`, `.path`, `.url`
  (structures/extension.md).
- The check is a table lookup against §8.1's cited lists:
  - `manifest_version`: 2 or 3 → supported key either way; note which.
  - `background.service_worker` present → **"unsupported manifest key: the
    extension's background service worker is not in Electron's supported key
    list. Its background logic may not run — see the probe notes."**
  - `background.scripts`/`persistent` present → supported (`background`, MV2).
  - each entry of `permissions` and `host_permissions` → matched against the
    supported-API list; anything else is reported by name as **provisional**
    (Electron's own word: "support for any APIs not listed here is provisional and
    may be removed").
  - specifically called out when present, with the reason: `nativeMessaging`,
    `debugger`, `alarms`, `notifications`, `cookies`, `webNavigation`, `identity`.
- Output per extension: a one-line **expected verdict** (`expected to work` /
  `likely broken` / `unknown — needs a try`), the **specific reasons**, and the
  **actual** result after the attempt: loaded or the thrown error, plus whatever
  warnings Electron logged to the console during load (the app already forwards
  main-process console output, `src/main/index.ts:307-311` — the extension
  warnings land in the same place and must be captured alongside the attempt, not
  looked for by the user in a terminal).
- **The wording rule:** the UI never says "supported" about a behaviour the docs
  do not list, and never says "unsupported" about MV3 as a category. It says what
  is declared, what is documented, and what was observed.

### 8.4 The 1Password-class verdict, in three sentences

The stated goal was "we could install the 1Password extension for example, which
makes logins very simple", and the honest answer has three parts:

1. **Electron will show it and will not make it work.** 1Password is a Manifest
   V3 extension whose behaviour lives in a `background.service_worker`, and it
   reaches its desktop app over native messaging; `background.service_worker` is
   not in Electron's supported manifest-key list (only MV2 `background` is), and
   native messaging is not in the supported-API list at all — so even in the
   best case where the extension *loads*, the two paths that make it a password
   manager are outside the supported surface, and its popup is what the user will
   see working while nothing else does.
2. **What it will actually do, then, is appear.** It will load into the managed
   directory, list itself (name, version, and its declared permissions), and then
   either work in the narrow ways a content script can (reading and filling a page
   it is injected into, if its content script does not depend on the service
   worker) or visibly fail — and the app's job is to say *which*, per extension,
   from the manifest inspection of §8.3 plus the observed load result, rather than
   promising "1Password works".
3. **The operator's actual goal is served better by R1 than by R4.** "Logins very
   simple" is a persistence problem, and §5 solves it: sign in by hand **once**
   inside the app's browser tab — which the tool's own description already tells
   the user they can do (`builtin.py:10170-10172`, "the user can sign in by hand
   when you ask them to") — and every later tab, in every agent session, in every
   later run of the app, is already signed in. The extensions feature is then a
   genuine bonus for the extension-shaped things that *do* fit the supported
   surface, not the load-bearing path for logging in.

**Where I could not settle something, I say so and name the experiment** rather
than choosing a comfortable assumption — that governs the whole of §8:

| question | status | experiment that settles it |
|---|---|---|
| Does Electron run an MV3 `background.service_worker`? | **not settled by the docs in either direction** | **P6**: build a minimal MV3 extension whose service worker writes a timestamp via `chrome.storage.local`, load it, and read the value back |
| Does a real password-manager extension behave differently from that minimal one? | **not settled** | **P7**: unpack the operator's own 1Password MV3 build, load it, and record the load result, the console warnings, and what its popup does |
| Do unsupported-API warnings appear anywhere we can capture them? | docs say "logged to the console" | **P8**: capture main-process console during a load of an extension declaring `alarms`; confirm the warning text and that the app's existing console forwarder (`src/main/index.ts:307-311`) sees it |
| Can a content-script-only extension usefully help an agent-driven page? | plausible, unmeasured | **P9**: an extension that highlights elements; check injection on an agent-driven tab in the shared partition |

The UI must carry this table's *content* — not the table — in its "what this can
and cannot do" copy, so the user is never surprised by the boundary.

---

## 9. R6 — Approval, security and revocation under a persistent jar

### 9.1 The problem, stated exactly

R1 makes the app hold the operator's real, persistent, authenticated sessions. R2
makes the agent able to drive tabs. Together they mean **the agent can act as the
operator on authenticated sites**, for as long as the jar and the grant live. That
is a genuine change in the threat model versus revision 1, where the jar was
nominally fresh and the grants were transient — and it must change the guardrail,
not just the prose.

The extension's model is the right starting point and its reasoning transfers:
default-deny per origin, enforced **in the extension** rather than in the daemon,
because the prompt renders in browser UI that no local process can click — "so
'the agent opened the user's bank' always passed through a human click on that
machine's screen" (`browser-extension.md` §6, threat 3). The scope vocabulary is
`domain | site | once` (extension 0.1.8, `browser-extension.md` §6.3), the queue
is FIFO with a 10-minute async TTL and a 60 s in-command TTL (`access-queue.ts`),
and one-shot grants are requester-bound so a grant earned by one session cannot be
spent by another (`access-flow.ts`).

### 9.2 The model

- **Where it renders.** A **consent bar inside the browser tab's chrome band**,
  above the view — not a modal. Two reasons, both verified by the layout rule:
  a modal reproduces the failure the async flow was built to remove (the agent
  gets no turn to tell the user, prompts expire unseen); and *a native view paints
  above all DOM* (§11.3), so anything rendered over the page area is invisible
  unless the view is hidden. The chrome band is outside the view's rect and so is
  never occluded.
- **Blocking semantics are unchanged from the extension**, verbatim three-step
  dance: `open`/`goto` to an unapproved origin fails early with
  `origin_not_allowed` (`protocol.py:296`, raised "EARLY, before any prompt
  exists, so the agent can run the explicit request_access → notify user →
  await_access flow"); `request_access` raises the pending entry and returns
  immediately; `await_access` polls in bounded slices; `cancel_access` removes
  only the caller's exact-origin entry. **Reusing `access-queue.ts` and
  `access-flow.ts` verbatim is what makes this true by construction rather than by
  review**, and it is why §12 moves them rather than reimplements them.
- **Requester binding is preserved.** The requester is derived from the host's
  context, never from an argument (`_browser_requester` `builtin.py:8637-8648`,
  carried by `_browser_identity_params` `:8943-8952`), so a one-shot grant earned
  by session A cannot be spent by session B. This property is *more* important
  now, because the grant buys access to a persistent authenticated session.
- **Notification.** The bar alone is not enough when the window is on another
  Space or behind something. The app already has `DesktopNotifier`
  (`desktop-notifier.ts:153`), so raise one notification per consent bar and mark
  the tab. The primary channel still stays the model: `origin_not_allowed`'s copy
  already instructs the agent to notify the user through `ask`, and that
  instruction must be kept with only the surface name changed.
- **The model-facing vocabulary is unchanged.** `origin_prompt_pending`,
  `origin_denied` and `origin_not_allowed` keep their shapes
  (`protocol.py:290-296`); only the sentence naming where the user must click
  changes (§4.1).

### 9.3 Consent lifetime, and what the user sees

**Scopes: `once | session | site | domain | deny`.** The extension has
`once | site | domain | deny`; R1 adds `session`, because with a persistent jar a
"site" grant that dies with the process would make the user re-approve on every
launch, which is the friction that pushes people to the dangerous allow-all switch
(`docs/BROWSER.md:22-27`). So:

| scope | lifetime | stored where |
|---|---|---|
| `once` | one navigation, requester-bound, ≤10 min | in-memory only |
| `session` | until the app quits | in-memory |
| `site` (exact origin, scheme+port) | durable | the approval store |
| `domain` (registrable domain via the bundled PSL) | durable | the approval store |
| `deny` | durable negative, so the agent stops asking | the approval store |

- **The approval store** is a JSON file beside `session.json` and the extensions
  directory: `userData/browser/approvals.json`, 0600, atomic write, listing each
  granted origin/domain with the scope, when it was granted, and the requester
  that earned it. It is **not** inside the Chromium profile, so "clear browsing
  data" (§5.4) and "revoke approvals" (§9.4) stay separate actions with separate
  consequences.
- **Persistence is the R6-relevant change.** A persistent grant against a
  persistent jar means an agent can act as the operator on that origin *days*
  later with no further human involvement. That is the price of R1, and the design
  answers it with visibility rather than with a timer: the granted set is a
  first-class, always-reachable list (§9.4), every agent-owned tab shows which
  origins the agent may reach from it, and the tab strip marks agent tabs
  (`builtin.py:9358-9366`'s `(yours)` analogue).
- **What the user sees when an agent drives an authenticated tab:**
  - the tab is marked agent-owned in the strip, with the session's label;
  - the URL bar shows the live URL, so the user can see where the agent is;
  - when the agent navigates to an approved origin, no prompt — the grant was
    already given, and prompting again per action is the behaviour the extension
    deliberately removed;
  - when it navigates to an **unapproved** origin, the navigation fails, the bar
    appears naming the origin, and the strip marks the tab as waiting;
  - a persistent, always-visible "agents can drive tabs in this app" indicator in
    the browser feature, with the Settings toggle that turns the host off entirely
    (which removes the state file, so detection is honest — revision 1's §12.3,
    kept).
- **Second review's worth of honesty:** the user should be able to answer, from
  the UI alone, "which sites can an agent act on as me right now?" The answer is
  the approvals list plus the open agent tabs, and both must be reachable in one
  click from the browser feature.

### 9.4 Revocation

Revocation must be as easy as granting, because the grant is now durable:

- **Per-origin:** in the approvals list, one row per granted origin/domain with a
  revoke action; and a "forget this site" action in the browser chrome for the
  origin currently shown.
- **Bulk:** "Revoke all site approvals" beside "Clear cookies and site data",
  with copy that says what each does and does not do. Revoking approvals does not
  log the user out; clearing cookies does not restore the deny state (so the agent
  will ask again for an origin it was denied — say so, because a revoked-then-asked
  prompt is exactly the kind of thing that reads as a bug).
- **Per-tab hand-over revocation** (§6.3) and **per-tab close** are separate
  affordances, because "stop this agent" and "forget this site" are different
  user intentions.
- **Turning the host off entirely** (the Settings toggle) stops new agent tabs,
  tears down the state file so discovery is honest, and leaves the profile and the
  approvals intact and inspectable — with a sentence saying so.

### 9.5 The honest default, and its limits

**Default-deny, no exemptions, and no auto-approval of `file://`, `about:`,
`blob:` or loopback.** The Python guard already refuses anything but http(s)
before the wire (`_BROWSER_URL_SCHEMES` `builtin.py:7623`, called from
`_validate_browser_args` `:7993-7997`), and the UI host **must re-validate a
third time** (`will-navigate`, §11.6), because the RPC endpoint is a new trust
boundary: a local process able to read the key could otherwise ask the UI to load
`file:///Users/<me>/.ssh/id_ed25519`.

**And the limit, stated plainly because pretending otherwise would be the actual
security failure:** *this model is a consent gate, not a sandbox.* Once an origin
is approved, the agent holds the operator's authenticated session on that origin
for the whole life of the grant and can do anything the operator could do there —
there is no per-action confirmation, by design, and no read-only mode. The
properties that *are* real, and worth stating because they are what a reviewer
should check:

1. **The agent can only reach origins a human approved**, per-navigation, and a
   redirect into an unlisted origin pauses and prompts (the extension's per-hop
   rule, `browser-extension.md` §6.3, to be reused).
2. **The agent can only drive tabs it created or was handed** (§6.3), so a tab the
   user opened for themselves is not reachable by guessing.
3. **The jar is not the user's real browser profile** (§5), so the blast radius of
   a mistake is the sites the user logged into *inside this app*.
4. **Every prompt is a human click in this app's chrome**, and no local process can
   synthesize it (the extension's threat-3 property, preserved by putting the bar
   in chrome rather than behind an IPC any renderer can reach — §11.7).
5. **The grants are inspectable and revocable** (§9.3–§9.4), which is the property
   that makes a durable grant defensible at all.

What is *not* true, and must not be implied in the UI: that the agent is limited
to reading; that an approval is narrow to a task; that a denial is permanent (it
is durable, but the user can change it); that clearing the cache has any effect on
what the agent can reach.

---

## 10. Detection, precedence and error UX

### 10.1 The state file

`~/.local-operator/run/ui-browser/host.json`, 0600 under a 0700 directory, staged
write + `os.replace`, with:

```json
{
  "pid": 8127,
  "port": 52133,
  "session_key": "<32-byte urlsafe token>",
  "proto": 1,
  "host": "ui",
  "app_version": "0.21.0",
  "profile_dir": "/Users/<me>/Library/Application Support/Local Operator/…",
  "tabs": 2,
  "agent_tabs": 1,
  "heartbeat_at": 1774000000.0,
  "started_at": 1773990000.0
}
```

Decisions and why:

- **A separate directory, not a sibling of `run/browser/bridge.json`.** `lop
  browser status` reads the bridge's own path through
  `browser_state.read()` (`cli.py:1951`, `:1998`; `state_path()` at
  `state.py:83-96`), and the bridge's install/cleanup/health paths all assume that
  directory is the daemon's. A second process's record beside it invites exactly
  the confusion a future `lop browser cleanup` sweep would cause. Same reason, one
  level up, as `docs/design-daemon-discovery.md`'s "A new namespace, not the
  session one."
- **An ephemeral port, unlike the daemon's fixed `4099` (`daemon.py:54`).** The
  daemon's port must be stable because the extension cannot read files. Neither
  the UI nor Python has that limitation, so the UI reads its own bound port from
  `server.address()`, writes it down, and Python reads it from the file.
  Ephemeral also removes collisions with `4099`.
- **`proto` carries `PROTO_VERSION`.** The UI and lop are independent release
  lines, so a mismatch is likely rather than exotic, and it must be typed.
- **`profile_dir` is new in rev 2** and is diagnostic only: it is
  `ses.getStoragePath()` (§5.3), so `lop browser status` can answer "where are my
  logins" without the user reading this doc.
- **`agent_tabs` is diagnostic**, split from `tabs` so the cap (§6.5) is
  observable.
- **No `extension_connected` analogue.** The daemon needs that flag because the
  browser is a separate process the user can close while the daemon stays up. The
  UI host's views are its own children: if the process is alive and heartbeating,
  it can create a tab on demand. `tabs` is diagnostic and deliberately **not** part
  of the liveness predicate.

### 10.2 The predicate, exactly — and why it is not `liveness()`

Mirroring `state.py:152-234` clause by clause, but **not reusing `liveness()`**,
for the reason §2.4 identifies: that function now returns `ABSENT` unless
`extension_connected` is true (`state.py:184`), and the UI host has no such field.

```python
# local_operator/ui_browser/state.py
RUN_DIRNAME = "run/ui-browser"      # separate from the bridge's "run/browser"
STATE_FILENAME = "host.json"

def liveness(root=None, *, now=None) -> tuple[Liveness, UiHostState | None]:
    current = read(root)                       # one JSON read, no mkdir, no socket
    if current is None or not pid_alive(current.pid):
        return Liveness.ABSENT, current        # definite no; never probe
    if heartbeat_age(current, now=now) <= HEARTBEAT_TIMEOUT_S:
        return Liveness.FRESH, current         # definite yes; never probe
    return Liveness.STALE, current             # unknown from the file; caller may probe

def available(...)     -> liveness(...)[0] is FRESH
def advertisable(...)  -> liveness(...)[0] in (FRESH, STALE)
```

This is the one place the UI host's classification *must* differ from the
bridge's, and the difference is small and arguable from the code: the UI host's
"is my browser attached" question has no third party to fold in, so pid + heartbeat
is the whole predicate.

and the socket-confirming acquittal, mirroring `backend.py:145-194`:

```python
async def ui_browser_reachable(classified=None) -> bool:
    # FRESH -> True, no probe. ABSENT -> False, no probe.
    # STALE -> one bounded GET /health, ceiling 1.5 s, requiring
    #          {"host": "ui", "pid": <same pid>} — `proto` is REPORTED by
    #          `/health` but deliberately NOT required here (backend.py:112-127):
    #          a skewed host is a real host that can explain itself, so it stays
    #          reachable and the per-action path returns the typed
    #          `proto_mismatch` below (§10.6)
```

The probe must check **the pid it answered for**, not just HTTP 200. The bridge's
`_health_ok` (`backend.py:185-194`) requires `extension_connected` in the body for
the same reason a stale file whose port has been recycled must not be acquitted;
for the UI host `pid` in the JSON body plus a `/health` that reports `os.getpid()`
is the equivalent check.

**Proto mismatch is advertisable, not absent.** A version-skewed host is a real
host that can explain itself: `advertisable()` returns `True`, `reachable()`
returns `True` (the file *and* the socket both answer), and the per-action path
then fails with `PROTO_MISMATCH` (`protocol.py:310`) whose copy already says
"update Local Operator and the browser extension, then restart the bridge daemon"
(`backend.py:69-72`) — rewritten in PR 2 for the UI variant. This is the
`state.py:225-232` doctrine applied honestly: silently unadvertising leaves the
agent with no browser tool and no explanation for a host that is running.

### 10.3 Where the Python changes land

New package `local_operator/ui_browser/`:

```
local_operator/ui_browser/
  __init__.py
  state.py      # RUN_DIRNAME/STATE_FILENAME, UiHostState, liveness/available/advertisable
  backend.py    # UiHostClient + ui_browser_available/advertisable/reachable + copy
```

plus, in `local_operator/browser_bridge/state.py`, a **defaulted-parameter
generalisation** of the *primitives* so the UI package shares them instead of
copying them (not of `liveness()` — see §10.2):

```python
def run_dir(root=None, *, dirname: str = RUN_DIRNAME) -> Path: ...
def state_path(root=None, *, dirname: str = RUN_DIRNAME, filename: str = STATE_FILENAME) -> Path: ...
def publish(state, root=None, *, dirname=RUN_DIRNAME, filename=STATE_FILENAME) -> Path: ...
def read(root=None, *, dirname=RUN_DIRNAME, filename=STATE_FILENAME, model=BridgeState) -> ...: ...
def remove(root=None, *, dirname=RUN_DIRNAME, filename=STATE_FILENAME) -> None: ...
```

`pid_alive` (`state.py:127`), `heartbeat_age` (`state.py:141`), `Liveness`
(`state.py:152`) and `HEARTBEAT_INTERVAL_S`/`HEARTBEAT_TIMEOUT_S` (`state.py:24-25`)
are reused **unchanged**. Note that `read()`'s model must be selectable, or the
UI's extra fields (`host`, `app_version`, `profile_dir`, `agent_tabs`) are dropped
by `BridgeState`'s `model_config = ConfigDict(extra="ignore")` (`state.py:29`) —
harmless but blind. Pass the model class in and default it. Every parameter gains a
default equal to today's value, so no existing call site changes behaviour — the
ladder rule at `AGENTS.md:2217-2223` and the pattern
`docs/design-daemon-discovery.md` recommends.

Then `browser_bridge/backend.py` grows one seam so the transport is reused, not
duplicated. `BridgeClient` already takes `root` (`backend.py:342`), which is a
smaller change than revision 1 assumed:

```python
class HostClient:
    def __init__(self, store=state_store, root: Path | None = None) -> None: ...

class BridgeClient(HostClient):          # unchanged constructor signature
    def __init__(self, root: Path | None = None) -> None:
        super().__init__(state_store, root)
```

`HostClient.call` is `BridgeClient.call` (`backend.py:345-409`) with `self.store`
in place of the module-level `state_store` (used at `:346`) — a handful of lines
inside a 65-line method — and `format_error` (`:202`), `client_timeout` (`:316`),
`BridgeError` (`:92`), `BridgeUnreachable` (`:100`) are reused verbatim.
`format_error` gains a `host: str = "extension"` keyword for the host-specific
sentences in §4.1; every other branch is already host-neutral.

**No new Python dependency.** The UI package is stdlib + `pydantic` (already a
dependency) + the existing `httpx` client, and the `/health` probe reuses the
`httpx` pattern in `backend.py:185-194`. Nothing is added to `pyproject.toml`.

### 10.4 Precedence and the dispatcher

`_execute_browser` (`builtin.py:9820`) changes at four points:

1. **Availability** (`builtin.py:9839-9847`): add
   `ui_available = await ui_browser_reachable()` and classify all three hosts once,
   so the decision and the diagnostic cannot disagree — the reasoning is already
   written at `:9840-9842` ("Classify the daemon ONCE per action and reuse that
   answer for both the backend decision and the demotion diagnostic, so they can
   never describe different readings of a file that may change between two
   reads").
2. **No-backend diagnostic** (`builtin.py:9848-9869`): three-way, with the UI-host
   shape added.
3. **Fresh-`open` precedence** (`builtin.py:9912-9918`): **UI → bridge → cmux**,
   per the fixed interface decision, with the `backend` hint of §16.1.
4. **`close`/`tabs`/access degrade branches** (`builtin.py:9883-9892`,
   `:9921-9934`, `:9943-9952`): replace "not cmux, therefore extension" with
   "which host is serving this surface", branching on the token prefix.

Surface prefix: `ui:<n>:<nonce>`, where `<n>` is the host's own tab id (§6.2,
stable for the tab's life) and `<nonce>` a 32-hex capability minted per open. The
regexes at `builtin.py:9053` and `:9058` become a single alternation, and
`close_browser_surface` (`:9574`, `startswith("bridge:")` at `:9592`) and
`retitle_browser_surface` (`:9530`, `:9549`) branch on it.

**"Tool advertised?"** `build_browser_tool`'s gate (`builtin.py:10160`) becomes:

```python
if not (cmux_browser_available() or bridge_browser_advertisable()
        or ui_browser_advertisable()):
    return None
```

using the *advertisable* test for the same reason the bridge does
(`builtin.py:10152-10159`), and the description (`:10166-10195`) names three
surfaces. No change to `tools/registry.py:60` — the existing `createIf` entry
gains a third check, per `AGENTS.md:2222-2223` ("Adding a second gating convention
beside `createIf` is itself a footprint regression — extend the table, do not
invent a parallel mechanism").

**On the ordering itself, and what R1 changes.** Revision 1 worried that UI-first
would send agents to a logged-out view "on the machines that have the extension",
and asked for a `backend` escape hatch. **R1 removes most of that worry**: the UI
jar is now persistent *and* shared, so "log in once, stay logged in" makes the
UI-first default far less costly than it was — the re-login is one time, not one
per task. The remaining case for the extension is specific and real: *device trust,
hardware keys, and enterprise conditional access*, where the session exists only
in the user's real profile and cannot be re-created inside the app. So the
`backend` hint stays recommended (§16.1), and the ordering stays UI-first, but its
justification is now "the right default with an explicit escape hatch" rather than
"a trade-off the operator has to weigh". Say this in the tool description and the
docs so a future reader does not re-litigate it from revision 1's argument.

### 10.5 The one structural Python change: who owns a `ui:` surface

This is the finding I would most want a reviewer to check, because it is a real
defect that a third host exposes — and it is now **larger** than revision 1
thought, because the ownership layer gained a concept and more call sites.

The ownership lane in `execute_browser` (`builtin.py:9658-9817`) is supposed to
run before every action: `resource.initialize()` (`:9660`), `resource.recover()`
if not yet recovered (`:9661-9663`), the `recover` action (`:9675`),
`retain`/`release` (`:9683-9765`), `owner_allocate` on a fresh open (`:9782-9783`,
via `resource.allocate()`), and `resource.remember(...)` on the way out
(`:9809-9813`). Every one of those calls goes through `BridgeClient()`, which
reads `run/browser/bridge.json` (`backend.py:346`).

But the lane is entered through a gate that reads a field only the lane ever sets
(`builtin.py:9637`):

```python
if not resource.generation and not await bridge_browser_reachable():
    return _browser_update_note(
        state, await _execute_browser(tool_call_id, args, signal, on_update, context)
    )
```

`resource.generation` is `""` on a freshly constructed `BrowserResource`
(`resources.py:71`) and is only assigned inside `initialize()`
(`resources.py:149-151`) — and the call sites of `initialize()` outside the lane
are themselves guarded on `resource.generation` (`_browser_identity_params` reads
`resource.generation` at `builtin.py:8950`). So on a host where
`bridge_browser_reachable()` is false, this is a **closed loop that never opens**:
the lane is skipped on the first action and therefore on every action, and none of
`initialize`, `recover`, `allocate`, `remember`, `retain` or `release` — nor the
stored-`surface_id` adoption at `:9662-9663` — ever runs.

Three distinct consequences now, and the third is new:

1. **Silent loss of the ownership contract on a UI-only host.** `recover` (the
   action) is dispatched only inside the lane (`builtin.py:9675-9682`), so on a
   UI-only host `browser action=recover` falls through to the tail of
   `_execute_browser` — where the last branch is a screenshot
   (`_browser_screenshot`, `builtin.py:8296`). It reports "Screenshot of … saved
   to …". The same is true of `retain`/`release` (`:9683-9765` is the only
   handler). And because `resource.remember` never runs, a `ui:` surface is never
   recorded, so an interrupted session strands the tab with no reclaim path — the
   exact leak `resources.py:1-6` exists to prevent.

2. **A hard failure once the lane *is* entered.** If `resource.generation` is ever
   truthy — a session that had a reachable bridge at least once, a resumed
   execution whose stored record is adopted, or simply a future change that sets it
   eagerly — then `resource.recover()` at `builtin.py:9662` sends `owner_recover`
   to the *bridge* for a `ui:` surface, and on a host with no daemon it raises
   `BridgeUnreachable`, returning the recovery sentence ("browser bridge
   unreachable: no live daemon state…") before the browser is ever touched.

3. **NEW in rev 2: the ownership *mode* is learned from the bridge discovery file.**
   `BrowserResource.ownership_mode()` (returning `bool | None`,
   `resources.py:310-328`) is driven by `_peer_identity()`
   (`resources.py:292-308`), which calls `state_store.read()` and requires
   `current.extension_connected`, returning `(current.extension_version,
   current.extension_proto)`. On a UI-only host that read finds no bridge file
   (or a stale one with `extension_connected=False`), so `_peer_identity()` returns
   `None`, `ownership_mode()` returns `None` ("must ask", per its own docstring
   `:321-322`), and `recover()` proceeds to `BridgeClient().call("owner_recover",
   ...)` (`resources.py:352`) — i.e. the capability-only path
   (`_degraded_recover`, `resources.py:482`) is *never* taken for the UI host, and
   the degraded path that exists precisely to handle "this peer cannot reconcile
   ownership" is unreachable. The version-skew logic at
   `resources.py:330-342` (`_peer_is_pre_ownership`) has the same problem: with no
   bridge file it returns `False` by design ("cannot tell must not be read as
   old"), so the UI host is treated as a *current, ownership-aware* peer when it is
   neither.

The full inventory of what must become host-selected — re-derived, because
revision 1's list was wrong:

| site | today |
|---|---|
| `builtin.py:8996` `_bridge_call` | `BridgeClient().call(action, params)` |
| `builtin.py:9746` | `BridgeClient().call(f"owner_{action}", …)` |
| `builtin.py:9751` | `BridgeClient().call("owner_finish", …)` |
| `builtin.py:9795` | `BridgeClient().call("owner_release", …)` |
| `resources.py:352` `recover` | `BridgeClient().call("owner_recover", …)` |
| `resources.py:450` | `BridgeClient().call("owner_release", …)` |
| `resources.py:560` `finish` | `BridgeClient().call("owner_finish", …)` |
| `resources.py:563` `finish` | `BridgeClient().call(...)` (legacy close) |
| `resources.py:621` `finish_degraded` | `BridgeClient().call("close", …)` |
| `builtin.py:7788` `_browser_update_note` | `state_store.read()` — the extension-update advisory |
| `builtin.py:8557-8560` `_bridge_liveness` | `state_store.liveness()` |
| `builtin.py:8623-8627` `_bridge_demotion_hint` | `state_store.Liveness` / `heartbeat_age` |
| `resources.py:19` + `:303` `_peer_identity` | `state_store.read()` — the ownership-mode input |

So the change is not "add an availability check". It is: **the ownership lane must
be selected per host, and so must the discovery reads that feed it.** Recommended,
smallest coherent shape:

- `BrowserResource`'s sidecar record (`.browser-resource.json`,
  `resources.py:27`) gains `"host": "" | "ui" | "bridge"`, written on the first
  successful `open` alongside `surface_id` (next to the existing
  `resource.remember(...)` call, `builtin.py:9809-9813`).
- `resources.py` stops constructing `BridgeClient()` at its five sites and takes a
  **client factory**, and stops calling `state_store.read()` directly in
  `_peer_identity` and takes a **discovery store** the same way; the `host` field
  selects both, defaulting to the bridge for a record written before this change
  (fail-safe: an old record behaves exactly as it does today).
- `execute_browser`'s own four `BridgeClient()` sites use the same selection.
- `_browser_update_note`, `_bridge_liveness` and `_bridge_demotion_hint` take the
  selected store, so a UI-only host gets a UI-shaped advisory/hint rather than a
  bridge read that returns nothing.
- The **gate at `builtin.py:9637` becomes host-based**: the lane runs whenever a
  browsable host is available for this session, not only when the bridge is. That
  is the fix for consequence 1, and it should be asserted by a test that fails if
  the lane did not run.
- The UI host implements the four `owner_*` methods against an in-process ledger
  (§4), and answers the *discovery* question the ownership mode needs with its own
  peer facts (proto + app version rather than extension version), so
  `ownership_mode()` has a meaningful answer on both hosts rather than `None`
  meaning two different things (consequence 3).

**An adjacent defect, same edit.** Consequence 1 is not UI-specific: on a
**cmux-only** host the same gate skips the lane, so `recover`, `retain` and
`release` already fall through to `_browser_screenshot` today. It is rarely hit
because the tool's description steers the model to those actions only after an
interruption, and because a bridge is usually present — but it is wrong, it is
cheap to fix while this function is being touched (an explicit refusal for the
ownership actions when no ownership host is available, rather than a silent
fall-through), and leaving it would mean this design knowingly shipped a third
host on top of it. Fix it in PR 2 and record it in the PR body as an existing
defect found on the way through, with the repro (`browser action=retain` against a
host with no bridge) in the testing evidence.

The alternative — have the UI host answer `owner_*` with no-ops and leave the
resource talking to the bridge — is rejected: it makes the ownership lane require
a daemon the surface does not use, which is exactly the coupling that produces all
three consequences above.

### 10.6 What the model reads in each failure

| condition | tool result |
|---|---|
| no UI host, no bridge, no cmux | tool not advertised (`createIf`). Honest absence |
| UI host up, proto mismatch (nothing else available) | advertised; every action: "the Local Operator desktop app's browser host speaks bridge protocol N; this Local Operator speaks M. Update the desktop app or Local Operator and retry." (typed `proto_mismatch`, `protocol.py:310`) |
| UI host up, protocol matches, cmux also present, UI host broken | `_bridge_demotion_hint`'s UI sibling: names the app, the measured fact, and the one-line repair |
| pinned `ui:` handle, user closed the tab | `tab_closed` → the existing copy ("browser tab … is gone; dropped the handle. Use 'open' …", `backend.py:211-215`) and the existing `open`-resumes-by-recreating path (`builtin.py:9036-9050`) |
| pinned `ui:` handle, agent's tab was restored as a user tab (§7.3) | same `tab_closed` path — the nonce is gone, so this is the right answer and no special case is needed |
| pinned `ui:` handle, UI quit | `unreachable` copy, UI variant: "the Local Operator desktop app is no longer answering at 127.0.0.1:<port>; if you quit it, re-open it and retry." Must **not** say "run 'lop browser install'" |
| UI host up, no browser tab open | not an error: `open` creates one. `tabs` returns "no browser tabs are open" with the existing footer (`builtin.py:9350-9357`) |
| site not approved | `origin_not_allowed`, sentence names the app's consent bar (not the extension popup) and keeps the instruct-the-agent-to-notify clause |
| prompt unanswered / denied | `origin_prompt_pending` / `origin_denied` (`protocol.py:290-291`), host-parameterised origin name, "do not retry the same origin" |
| agent asked to drive a tab it was not given (§6.3) | typed refusal, not `tab_closed`; copy names the hand-over affordance |
| agent-owned tab cap reached | `tab_limit` (`protocol.py:303`) with the existing "close one … before opening another" copy |
| two commands on one tab at once | `busy` (`protocol.py:309`), "retry this action once" |
| tab crashed / view destroyed mid-command | `internal` + `tab_crashed` (host-neutral, `backend.py:266`) |
| DevTools attached to the view | `debugger_conflict` (`protocol.py:297`). In Electron this manifests as a **`detach` after a successful attach** rather than a refused attach: the Debugger docs say the `detach` event fires "when `webContents` is closed or DevTools is invoked for the attached `webContents`" (debugger.md:39-47), so the host must translate a mid-command detach into this code rather than mapping `attach` failures only. In production the app's own DevTools are gated off (`src/main/index.ts:282`), so this is mostly a dev-mode case |
| RPC answered 401 | "the desktop app rejected the state-file key; restart the app." (mirrors `backend.py:394-397`) |

---

## 11. UI-side design

### 11.1 The view: `WebContentsView`, one per tab — re-argued for tabs

Recommended: **`WebContentsView`** (`web-contents-view.md:1`, "A View that
displays a WebContents"), one per tab, added to `win.contentView`
(`View.addChildView`, view.md:50; `win.contentView`, base-window.md:429).

Revision 1 chose this with one view in mind; R2 makes it *many* views, so the
choice is worth re-making rather than inheriting.

**Why not a `BrowserWindow` per tab.** It is the obvious alternative once tabs
exist — real windows, real tab tear-off — and it is wrong here for four reasons:

1. **It would take focus.** `window-raise.ts` is the only module that raises a
   window (`window-raise.ts:2-17`) and `scripts/window-mode.test.mjs` asserts it.
   A per-tab window either goes through that gate (so it can never show without a
   human asking, which is not a browser) or breaks the invariant that makes every
   agent run safe.
2. **The tab strip and URL bar would have to live in every window** rather than in
   the app's own renderer, duplicating the shell.
3. **The bounds problem does not go away**, it becomes OS window management, which
   is strictly more platform surface to be wrong on.
4. **Nothing about the feature needs it.** R2's "tabs" are a strip the user can use
   plus a registry the agent can drive; neither needs OS windows.

**Why not `<webview>`.** Electron's own pinned documentation recommends against
it: the tag "is based on Chromium's `webview`, which is undergoing dramatic
architectural changes… We currently recommend to not use the `webview` tag and to
consider alternatives, like `iframe`, a `WebContentsView`, or an architecture that
avoids embedded content altogether" (webview-tag.md:3-10). It is disabled by
default and must be enabled with `webviewTag: true` **on the host
`BrowserWindow`'s `webPreferences`** (web-preferences.md:117-124, "Defaults to
`false`", plus the note that a `<webview>`'s preload "will have node integration
enabled"). Enabling it therefore widens what the app's *own* renderer — the
`sandbox: false` one carrying the preload and the desktop IPC surface
(`src/main/index.ts:277-298`) — can instantiate. That is the wrong direction for a
feature whose whole trust story is "one remotely-loaded view". The renderer-side
lifecycle would also put the CDP handle on the far side of the IPC boundary from
where it must live (main).

**Why `WebContentsView` is nonetheless a real cost, and what it costs with N of
them:**

- Bounds are manual and there is no auto-resize (`View.setBounds`, view.md:65), so
  the renderer must report the content rect (§11.2) — **once, not per tab**: only
  the active tab occupies the rect; every other tab is `setVisible(false)`
  (view.md:127) and keeps its own bounds or none. One rect authority, N views.
- `BaseWindow`'s resource-management note applies: "When you add a
  `WebContentsView` to a `BaseWindow` and the `BaseWindow` is closed, the
  `webContents` of the `WebContentsView` are not destroyed automatically… Unlike
  with a `BrowserWindow`, if you don't explicitly close the `webContents`, you'll
  encounter memory leaks" (base-window.md:67-90). So **every** tab close must
  `view.webContents.close()` (`web-contents.md:1174`) and `removeChildView`
  (view.md:59), and app quit must iterate the registry. A test must assert the
  `webContents.getAllWebContents()` count returns to the app's own renderers
  (probe P10, §15).
- With N tabs the *inactive* ones are hidden views, which is where §11.5's
  throttling question lands (and where the app already has a policy, §11.4).

### 11.2 Layout: who owns the rectangle

The renderer owns layout (one authority, not two). The browser route renders its
chrome — tab strip, URL bar and navigation buttons, consent bar, extensions/about
surfaces — in ordinary DOM, and measures the content area with a `ResizeObserver`
on a placeholder element, reporting `{x, y, width, height}` to main over IPC. Main
calls `view.setBounds(rect)` for the **active** tab only. Sources of rect change
that must be handled: window resize, the rail collapsing (`RAIL_WIDTH`,
`sidebar-navigation.tsx:95`), route changes away from and back to `/browser`, and
devicePixelRatio changes on a monitor switch.

Throttle the reports (one per animation frame at most) and coalesce in main: a
`setBounds` storm during a live window resize is visible as tearing.

Branding applies as it does everywhere (`docs/branding.md`, roles not colours,
`cn` from `@shared/lib/utils`, twelve themes, contrast floors via
`pnpm check-themes`) — and specifically: the tab strip and consent bar are
*controls*, so their boundaries are `border-control` with the 3:1 floor, not
`hairline`; the agent-owned marker is a role in the semantic or ink scale, not a
bespoke colour; and the consent bar's primary action is the most prominent thing
in the band, per the agent-output hierarchy (`branding.md` §7).

### 11.3 The layering consequence — the single most important UI constraint

**A native `WebContentsView` paints above all DOM.** It is a sibling of the
window's own webContents in the view tree, so no `z-index`, portal, or overlay in
the renderer can draw over it. Everything the app currently paints over the content
area therefore becomes invisible while the browser tab is showing — and §2.7 lists
the eight overlays that do that today (`app.tsx:158-180`).

Two rules follow, and they should be in the code as comments because they look
like bugs to anyone who does not know the cause:

1. **Full-window overlays hide the view.** One derived boolean in the renderer
   (the existing store flags, plus a per-overlay "does it cover the content area"
   set) → one IPC call → `view.setVisible(false)` on open, `setVisible(true)` on
   close. The tab strip and URL bar are outside the view rect and stay painted, so
   the screen never looks empty; the area under the view shows the route's canvas
   plus the page title and a one-line "paused while a dialog is open" note. The
   exact visual treatment is a designer decision — hand it to the design round with
   before/after frames.
2. **Consent UI lives in the chrome band, never over the page.** This is why §9.2
   puts the approval bar above the view rather than in a modal.

I want this flagged as a genuine unknown to be measured rather than assumed:
whether `setVisible(false)`/`(true)` around a modal produces a visible flash on
macOS at Electron 44 is exactly the kind of thing a still frame hides. QA should
capture consecutive frames across a palette open/close with the browser tab active
(§15, probe P11).

### 11.4 Focus: the contract is "never steal", and it must be testable

The product has a documented focus-safety rule with a recorded incident:
headful Chrome launched by a test harness "took focus from the operator repeatedly
while he was working in another application", and "never stealing focus is the
general rule for every browser this project starts, in any context". The UI repo
already has the enforcement module (`window-raise.ts:2-17`) and the test
(`scripts/window-mode.test.mjs`), and extension parity is `commands/nav.ts:179`'s
`chrome.tabs.create({ active: false, … })`.

The UI equivalents, now stated as *additions to an existing rule* rather than a
parallel one:

- Creating a `WebContentsView` and calling `setBounds` does not focus anything.
  The forbidden calls are `view.webContents.focus()`, `win.show()`, `win.focus()`,
  `win.moveTop()`, `app.focus()`, and `app.dock.bounce()` — none may appear
  anywhere on the browser path. The existing grep-based test over `src/main/`
  (`scripts/window-mode.test.mjs`) extends to `src/main/browser/`.
- An agent `open` must **not** switch the active tab: the new tab is appended and
  marked, the user's current tab stays active. This is `active: false`'s
  equivalent and it must be tested, not assumed. **New in rev 2:** the same applies
  to a restored tab (§7) — restoring must not steal the active tab from whatever
  the user then clicks.
- `capturePage` is deliberately not used (§4) partly because it has
  visibility-forcing semantics (web-contents.md:1815; Electron 44 adds a
  `stayHidden` option at `:1809`, which is a *new* fact worth noting but does not
  change the decision — CDP capture sidesteps the question entirely).
- Test: record the frontmost process before and after a full agent interaction and
  assert it is unchanged (probe P12).

**And one consequence of the multi-identity work (§2.8) worth recording:**
"the extension is connected" no longer implies "the extension will answer" — it may
be a standby. Nothing here depends on that, but it means the UI host must not
infer bridge health from the extension's connection state, and it already does not.

### 11.5 Session and partition wiring

The view's `webPreferences`: `partition: "persist:local-operator-browser"` (§5.1),
`contextIsolation: true`, `nodeIntegration: false`, `webSecurity: true`,
`sandbox: true`, and **no preload at all**.

- `sandbox: true` on the driven views is deliberate and is the mirror of the auth
  popup's existing `sandbox: true` (`src/main/index.ts:415`). The app's own
  renderer is `sandbox: false` because it carries the preload and the desktop IPC
  surface (`:279`); the driven views carry nothing and must not gain it.
- Session-level handlers are installed once on the browser session (§5.1).
- `webSecurity: true` and `allowRunningInsecureContent: false` (security items 6,
  8 — the same settings the main window already uses, `src/main/index.ts:286-287`).
- **User agent**: set once on the browser session (`ses.setUserAgent`,
  session.md:1377-1388) so the jar presents a stable identity, and note the
  documented limit — "This doesn't affect existing `WebContents`" — so it must be
  set before any view is created. `app.userAgentFallback` (app.md:1871-1878) is
  the app-wide alternative; the design uses the session-level one so the app's own
  renderer keeps Electron's default. Which UA string, and whether to strip the
  `Electron/` token, is a UX call with a real hazard (site compatibility) and is
  §16.4.

**Background-tab throttling, and the trap in it.** With several tabs, an agent
acting on a hidden tab may be on a throttled page. Two facts, both needed:

- Electron's own breaking-change note is still present in 44 and appears twice:
  `contents.setBackgroundThrottling(false)` / `contents.backgroundThrottling`
  "affects all `WebContents` in the host `BrowserWindow`"
  (web-contents.md:2381-2395, history block at `:2383-2389` and `:2541-2547`).
  So it is **not** a per-tab switch in practice — setting it off on one view is a
  window-wide decision.
- The app already makes that decision per launch: `window-mode.ts:98-100` sets
  `backgroundThrottling: true` only in `normal`, so in `headless` and `inactive`
  it is already `false` for the whole window — which will therefore already
  include any browser view. **Consequence for evidence:** a throttling difference
  cannot be observed from a `headless` or `inactive` run, because those modes have
  already disabled it. Any A/B about throttling must be run in `normal` on the
  operator's desk, or measured by reading the property rather than by timing.

So: **leave Electron's default in v1** (i.e. keep the window-mode policy, which is
a deliberate existing decision), and have QA measure the agent-on-an-inactive-tab
case the honest way (probe P13, §15) rather than asserting the default is fine.

### 11.6 Navigation, popups, permissions, downloads

- **`setWindowOpenHandler` on every view** (web-contents.md:1463): deny, and for
  http(s) only, offer the URL to the user rather than auto-opening
  (`shell.openExternal` on untrusted content is security item 15; window creation
  is item 14). The app's existing main-window handler
  (`src/main/index.ts:348-428`) is the pattern but must **not** be copied
  wholesale: its trusted-auth-domain allowlist is for the app's own OAuth popup,
  and a browser view's `window.open` is arbitrary web content. The browser view's
  handler denies everything; an auth redirect inside a driven page navigates the
  same tab, which is what a browser does.
- **`will-navigate` / `will-redirect`**: allow `http`/`https` only, matching
  `_BROWSER_URL_SCHEMES` (`builtin.py:7623`) and the extension's independent
  re-check (`origin-policy.ts`). Refuse `file:`, `javascript:`, `data:`, `blob:`,
  `chrome:`, `devtools:`, `about:` other than `about:blank` (which `open` uses as
  the pre-navigation document — the extension does the same,
  `commands/nav.ts:179`). Security item 13 is explicit that a `startsWith`
  comparison is not a check: parse with `new URL` and compare origins.
- **`will-attach-webview`**: `event.preventDefault()` always
  (web-contents.md:981).
- **Permissions**: `setPermissionRequestHandler` **and**
  `setPermissionCheckHandler` on the browser session, default-deny, surfacing the
  request in the consent bar. This is not optional: "By default, Electron will
  automatically approve all permission requests unless the developer has manually
  configured a custom handler", and "you must also implement
  `setPermissionCheckHandler` to get complete permission handling… Most web APIs
  do a permission check and then make a permission request if the check is denied"
  (session.md:985-987, :1066-1068). Both handlers are needed; one is a hole.
- **Downloads**: `session.on('will-download')` (session.md:91-99) →
  `event.preventDefault()` and tell the user; the extension lists downloads as a
  non-goal too, and a download UI is a separate feature with its own security
  surface.
  **SUPERSEDED IN PR #1318** — not deleted, because the reasoning above is still
  why the destination is a quarantined directory rather than the user's
  `~/Downloads`. `will-download` becomes ARM-GATED policy instead of
  `preventDefault`: a `download` call arms the tab, the handler takes the item,
  sets a save path the HARNESS composed (`item.setSavePath`, so the app host does
  not need the CDP primitives Chrome refuses to an extension), and Python
  classifies what landed. The refusal above stays the behaviour for an UNARMED
  download — a file the user did not ask for still does not land in their
  Downloads folder. See `docs/design/browser-file-transfer.md` (and PR B for the
  app-host half, including the consent/reveal surface §16.4 argues for).

### 11.7 The trust boundary, and why the renderer must not reach it

The RPC listener is the first inbound network surface this app has had. Rules,
unchanged from revision 1 except where marked:

- Bind `127.0.0.1` only; never `0.0.0.0`, never IPv6-any.
- Require the key header on every request and compare in constant time, exactly as
  `daemon.py:2669-2670` does. Answer 401 otherwise, never a partial response.
- **Send no CORS headers.** The renderer runs with `webSecurity: true`
  (`src/main/index.ts:286`), so a `fetch` from the app's own page to the loopback
  port fails CORS preflight without them. That is the mechanism that keeps the
  *page* from using the agent's browser — deliberately, and it must be commented as
  load-bearing, not left as an accident of defaults.
- Validate the envelope with the generated schema (`WireModel`, `protocol.py:336`)
  and refuse unknown methods with a typed error.
- Never expose CDP on any socket. The `webContents.debugger` handle stays in main.
- The driven view gets no preload and `sandbox: true` (§11.5), so page script
  cannot reach `window.api.*` even if a future refactor adds an IPC channel to the
  browser namespace.
- Renderer→main IPC uses a **new namespace** (`window.api.browser.*`), validated
  in main with the same sender check the desktop transport uses
  (`trustedDesktopFrame`, `desktop-transport.ts:84-99`; security item 17). It must
  **not** be routed through `desktop-request`'s allowlist: that vocabulary maps to
  backend HTTP paths (`docs/desktop-controls.md:1-6`), and mixing the two would
  give the renderer a way to name browser operations through a channel designed
  for backend operations.
- **NEW: the app's own renderer is not in the browser partition** (§5.1), so app
  UI code cannot read the jar or the driven pages' storage directly. That is a
  second, structural barrier behind the no-CORS one, and it is why the partition
  choice is stated in §5.1 as "the app's own session is different".
- **NEW: the hand-over affordance (§6.3) is the only renderer→browser authority
  transfer**, it mints a capability rather than exposing one, and the renderer
  never sees a nonce. The nonce goes to the *session* through `tabs`.

### 11.8 Human takeover

Every tab in the strip is the user's: they can click it (main re-orders the child
views / moves bounds so the active one owns the content rect), type a URL, reload,
go back, and interact with the page. When the user navigates a tab the agent owns,
the agent's next action simply reads the live page — `_page_line` reports the live
url and title (`builtin.py:8185-8192`) — and the "report what is actually on
screen" rule is already documented. There is no lock, no modal claim, and no
"agent busy" state: the same contract cmux's panel has, with §6.4's ref-epoch
discipline as the safety mechanism.

The strip must visibly mark agent-created tabs (the extension's analogue is the tab
group labelled with the session title) and the consent bar is the place a human
grants new origins. `retitle` is *not* used to name the UI tab (§4): the tab shows
the page title.

### 11.9 A route now, a panel later

The UI repo has a `feat/panel-views` worktree at exactly `origin/main` — i.e.
**unstarted** — so any panel system is unlanded and must be treated as such. The
browser feature therefore ships as **a route** (`/browser`) with a rail item, in
the existing `SidebarNavigation` (`sidebar-navigation.tsx:95`) and
`<Routes>` (`app.tsx:192-206`) structure, and it must slot into a rail/panel system
later **without a rewrite**. The design constraints that make that true, and that
the coder should honour now:

- The browser's state lives in **main** (registry, session, approvals, RPC) and is
  reachable over an IPC namespace (`window.api.browser.*`) that does not assume a
  route. So a panel and a route are two presentations of the same API.
- The chrome is a **self-contained component tree** (strip + bar + consent band)
  that takes the content rect from its parent rather than assuming full-window.
  A route gives it the whole `<main>`; a panel would give it a pane.
- **One thing a panel makes harder, and it should be said now:** a native view
  paints above all DOM (§11.3), so a browser *panel* beside other panels means the
  view must be confined to its pane's rect — which `setBounds` supports — and every
  overlay in the app must still hide it. That is a bounds-and-visibility problem,
  not a redesign, which is exactly the property worth preserving by keeping layout
  authority in the renderer.

---

## 12. Cross-repo artifact sharing, under "no new npm publishing pipeline in v1"

### 12.1 What is shareable, and what is not

The measurement in §2.6 is the argument. **648 of 3,257 non-comment lines contain
no `chrome.*` call**, and the four modules that matter most there
(`origin-policy`, `access-queue`, `access-flow`, `ax-compact`/`scroll-expressions`)
were written `chrome`-free on purpose, for the pure-node test suite. That half is
reusable **unchanged**.

The other ~2,600 lines are not reusable as written, and an adapter interface does
not change that, because the *semantics* differ rather than the names:

| extension | Electron | consequence |
|---|---|---|
| `chrome.debugger.attach({tabId}, "1.3")` — per tab | `webContents.debugger.attach("1.3")` — per WebContents (debugger.md:66) | one view per tab; the attach state set becomes the view registry |
| DevTools conflict **refuses the attach** | DevTools **detaches** an existing session (debugger.md:39-47) | a differently shaped error path (§10.6) |
| `chrome.tabs.get` as the per-action liveness read | `webContents.isDestroyed()` + `destroyed` (web-contents.md:1170) | different failure timing |
| `chrome.webNavigation.onCompleted` / `onBeforeNavigate` / `onHistoryStateUpdated` | `did-finish-load` / `did-fail-load` (web-contents.md:120) / `will-navigate` / `did-navigate` / `did-navigate-in-page` | the settle logic must be rewritten against real events; `did-fail-load` fires for subframes and needs an `isMainFrame` filter, and `did-finish-load` is not the same signal as `onCompleted` |
| `chrome.scripting.executeScript` (ISOLATED world) | `executeJavaScriptInIsolatedWorld` (web-contents.md:1446) | `executeJavaScript` would be a *weaker* port |
| `chrome.storage.session/local` | in-process state, or a JSON file | the whole deadline-vs-storage hazard analysis in `settle.ts` becomes irrelevant — a genuine simplification |
| `chrome.alarms` (MV3 keepalive) | none needed | the reconnect machinery (`reconnect.ts`, 12 LOC) is deleted, not ported |
| `chrome.tabGroups` | none | `tab-groups.ts` (187 LOC) is deleted; tab-strip labelling takes its place |

### 12.2 The mechanism

**Vendored-and-generated from a pinned `local-operator` ref, with a drift check on
both sides.** No npm package, no submodule, no relative path dependency, no
publish pipeline in v1.

1. **Shared, moved verbatim** into `extension/src/driver/` **in the
   `local-operator` repo**: `origin-policy.ts`, `psl.gen.ts`, `access-queue.ts`,
   `access-flow.ts`, `ax-compact.ts`, `scroll-expressions.ts`, `errors.ts`, and
   `settle.ts`'s `deadline()` + ceiling table. Modules keep their import paths via
   re-export shims so the extension diff stays small (the pattern `cdp.ts` already
   uses for `BridgeCommandError`). ≈648 non-comment lines moved, ~15 import sites
   updated.
2. **Not shared**: the `chrome`-coupled driver. The UI implements its own
   `src/main/browser/` against Electron, borrowing the *algorithms* (settle grace,
   ref epochs, log ring shape, surface cap) with the extension's rationale comments
   carried over where the behaviour is deliberately identical.
3. **Protocol + driver, generated into a vendored bundle.** `gen_ts.py`'s single
   `TARGET` (`gen_ts.py:25`) becomes a list of *targets*, and a `render_bundle()`
   produces the UI's copy from the same inputs. The emitted header carries
   `PROTO_VERSION` and a `sha256` over the generator's **inputs**
   (`protocol.py` + `gen_ts.py` + each `driver/*.ts`), never a git SHA, or every
   unrelated commit would make `--check` red — the same rule revision 1 stated and
   that still holds.

Where the files land:

```
local-operator/
  extension/src/driver/                    # NEW — host-free, moved verbatim (~648 LOC)
    origin-policy.ts  psl.gen.ts  access-queue.ts  access-flow.ts
    ax-compact.ts     scroll-expressions.ts  errors.ts  deadline.ts
  extension/src/*.ts                       # existing chrome-coupled modules, shims to driver/
  local_operator/browser_bridge/gen_ts.py  # + UI bundle target, + input-hash stamp, + --check
  local_operator/ui_browser/               # NEW — state.py, backend.py (~330 LOC)

local-operator-ui/
  src/main/browser/vendor/
    protocol.gen.ts                        # generated
    driver/*.ts                            # generated copy of driver/**
    PROVENANCE.json                        # pinned lop ref + per-file sha256 + input hash
  scripts/sync-vendored.mjs                # the ONLY writer
  scripts/check-vendored.mjs               # CI gate, read-only
```

**`PROVENANCE.json`** is the pin, and it is the whole mechanism:

```json
{
  "source_repo": "damianvtran/local-operator",
  "source_ref": "<a commit SHA, never a branch>",
  "proto_version": 1,
  "inputs_sha256": "<sha256 over protocol.py + gen_ts.py + driver sources>",
  "files": { "protocol.gen.ts": "<sha256>", "driver/origin-policy.ts": "<sha256>", "…": "…" }
}
```

**`scripts/sync-vendored.mjs`** is the only thing that writes `vendor/`. It
requires an explicit `--from <ref>` (so the pin is always a deliberate value,
never "whatever main is now"), regenerates into the target, rewrites
`PROVENANCE.json`, and refuses to run with uncommitted changes in `vendor/` unless
`--force` is passed. It is run by the implementer when bumping the pin, and its
output is committed — which means **the diff of a pin bump is reviewable**: a
reviewer sees the ref change plus the exact vendored delta.

**`scripts/check-vendored.mjs`** is the CI gate and is read-only. It verifies:
(i) every file listed in `PROVENANCE.json` exists and its sha256 matches;
(ii) no file exists in `vendor/` that is not listed (so a hand-added file fails);
(iii) `vendor/protocol.gen.ts`'s declared `PROTO_VERSION` equals the number the UI
code imports. It shares its shape with the existing `scripts/generate-theme-css.mjs
--check` gate (`package.json`, `check-themes`), which is the established in-repo
pattern for "generated artifact must be fresh".

### 12.3 Three layers of staleness protection, honestly labelled

| layer | catches | airtight? |
|---|---|---|
| `gen_ts.py --check` inside `local-operator` CI (already wired for the extension target; extended to the bundle) | the lop-side generated file is stale relative to its source | **yes**, within this repo |
| `check-vendored.mjs` inside `local-operator-ui` CI | a hand-edit of a vendored file, a stale `PROVENANCE.json`, an unlisted file | **yes**, within that repo — but it **cannot** detect drift against lop, because it has no lop checkout |
| runtime `PROTO_VERSION` check (`proto_supported`, `protocol.py:49`; typed `PROTO_MISMATCH` `:310`) | an actually-mismatched pair in the field | **yes, for the user**, and it is the layer that matters: a mismatched host gets a typed error, never a mystery timeout |
| an optional scheduled workflow in `local-operator` that checks out the UI repo at the ref in its `PROVENANCE.json` and diffs a fresh regeneration | cross-repo drift before it ships | **no checkout needed for either repo's own CI, and not built in v1** — named as a follow-up, with the runtime check as the reason it is not urgent |

**What happens when the two disagree, concretely:**
- A **hand-edited** vendored file: `check-vendored.mjs` fails the UI build. Nothing
  ships.
- A **stale pin** (lop's protocol moved and nobody bumped the UI): lop's own CI is
  green and the UI's own CI is green. The mismatch is caught at **runtime** by
  `PROTO_VERSION` — the session gets `proto_mismatch`, which is advertisable and
  typed (§10.2), so the agent reads an actionable sentence instead of hanging. The
  scheduled workflow would have caught it earlier; until it exists, this is the
  honest answer and it is the same answer the extension design chose for its own
  cross-boundary version skew.
- **`PROTO_VERSION` is not bumped by this work.** Nothing here adds a wire method,
  changes an envelope or adds an `ErrorCode` the extension could emit, so the window
  invariant at `protocol.py:20-45` is respected. `AGENTS.md:929-935` is explicit
  that a bump closes the released store build with 4001, whose popup reads as an
  unfixable "update needed" card — so a bump here would be a defect. The `ui:`
  surface prefix and the `handle` param are *additive to the conversation with a
  host that is not a released extension at all*.

### 12.4 Why not the alternatives

- **An npm package** (`local-operator-ui-browser-protocol` or similar): rejected,
  and now by fiat rather than by argument — the operator fixed "no new npm
  publishing pipeline in v1". It remains the right *long* answer if a third
  consumer appears; the vendored bundle is a subset of it with a manual sync step.
- **A git submodule or a relative path dependency**: rejected, unchanged. The UI
  repo is a standalone public package and must not depend on a sibling clone's
  presence. A consumer that clones only `local-operator-ui` must build.
- **Copy-paste with no check**: rejected — that is the failure the manifest gate
  exists for, and it is the realistic one (someone edits the vendored file to make
  a test pass).

---

## 13. Effort estimate, split into PRs

Line counts are order-of-magnitude, from the measured module sizes in §2.6 and the
file lists in §12.2, and they are the honest kind (touched, not "net new").

| PR | repo | content | estimate |
|---|---|---|---|
| **1** | local-operator | `refactor(browser-driver)`: move the host-free modules to `extension/src/driver/`, re-export shims, update imports, add the drift test that the moved modules still have no `chrome.*` reference | ~648 moved + ~30 new, ~15 files touched |
| **2** | local-operator | `feat(browser)`: `local_operator/ui_browser/` (state + client + copy), the `browser_bridge/state.py` primitives generalisation, `HostClient`, the `ui:` prefix, three-way precedence, availability helpers, the per-host ownership **and discovery** selection of §10.5, the `handle` param on `open`, the copy changes of §4.1, `gen_ts.py` bundle target, `docs/BROWSER.md` + the doc defects of §2.9 | ~900-1,200 |
| **3** | local-operator-ui | `feat(browser-host)`: `WebContentsView` tab manager + registry, loopback RPC + state file + heartbeat, CDP driver (with ref epochs), the shared persistent partition + session handlers, security handlers (navigation, popups, permissions, downloads, UA), vendored protocol/driver + `sync-vendored`/`check-vendored`, preload namespace, main-process tests | ~1,600-2,100 |
| **4** | local-operator-ui | `feat(browser-tab)`: route + rail item, tab strip, URL bar, back/forward/reload/stop, bounds reporting, overlay-hides-view policy, agent-tab markers + the hand-over affordance, designer round | ~800-1,000 + design |
| **5** | local-operator-ui | `feat(browser-profile)`: the profile lifecycle — clear browsing data (three honest buttons), the profile path surfaced in Settings and in `status`, tab restore (`session.json` + `navigationHistory.restore`), Settings > Browser | ~450-650 |
| **6** | local-operator-ui | `feat(browser-extensions)`: the managed directory + README, load-once-per-run, the manifest-vs-supported compute, the attempt-and-report UI, the constraint copy | ~450-650 |
| **7** | local-operator-ui | `feat(browser-consent)`: the consent bar, the durable approval store + scopes, revocation UI, notification wiring, designer + `ux-reviewer` rounds | ~450-650 + design/UX |

**Total: roughly 5,300-6,900 touched lines across 7 PRs**, of which ~648 are a move
and ~700 are generated/vendored (so ~4,000 are hand-written).

**The defensible first merge.** PR 1 alone is a pure refactor with no behaviour
change and is independently reviewable — land it first, because it makes PR 3's
generated copy trivial. PRs 2 and 3 are the minimum pair that produces anything
observable, and I would not merge PR 2 without PR 3 in flight, because a detection
path with no host to detect is dead code with a maintenance cost and nothing to
test it against.

**Phases, and what each proves:**

1. **Seam** (PR 1 + PR 2). Nothing user-visible. Verified by: the dispatcher
   selection matrix (ui-only / bridge-only / cmux-only / both / all / none), the
   state-file freshness matrix, and a faked UI host answering the wire.
2. **Host** (PR 3). The app can be driven end to end by a real `lop` session, with
   no UI beyond a single always-on view. Verified by curl transcripts and a real
   click/read/screenshot on a real site.
3. **Chrome** (PR 4). Tabs, URL bar, consent placeholder, overlay policy.
   Verified by rendered frames and a designer round.
4. **Profile** (PR 5). R1 observable: log in once, quit, relaunch, still logged in.
5. **Consent** (PR 7) — deliberately *after* chrome, because the bar lives in the
   chrome band and needs the layout to exist; and *before* extensions, because R6
   is the load-bearing control while R4 is the optional one.
6. **Extensions** (PR 6) — last, and the first thing to cut.

**Cut list, in this order if time is short:** (1) PR 6 entire — given R1 solves the
login problem, the extensions feature is a bonus, not a need; (2) PR 5's
clear-browsing-data trio, keeping the partition and the restore; (3) the
notification channel (§9.2). **Do not cut:** the consent bar (the alternative is
auto-approved permissions, which session.md:985 says is Electron's default), the
restore's stale-handle rule (§7.3), the ref-epoch discipline (§6.4), or the focus
test (§11.4).

---

## 14. Risks and failure modes

Ordered by expected cost, not by likelihood alone.

1. **The consent model is the whole ballgame now, and it is a speed bump, not a
   sandbox** (§9.5). A persistent jar plus a durable grant means an agent can act
   as the operator on that origin days later. Symptom if this is got wrong: an
   approval granted casually for a demo becomes standing authority over a real
   account, and nobody notices because everything looks normal. Mitigation: §9 in
   full — default-deny, per-origin scopes, a durable and inspectable grant list,
   revocation as easy as granting, and no copy anywhere that implies the agent is
   read-only.
2. **The ownership lane is gated on the bridge, not on a host, and its discovery
   reads are bridge-coupled too** (§10.5). Three shapes, the third new: on a
   UI-only host the lane never runs (`recover`/`retain`/`release` fall through to a
   screenshot, and a `ui:` surface is never recorded for reclamation); once the
   lane does engage, `owner_*` goes to a daemon that is not there; and
   `ownership_mode()` learns its verdict from `bridge.json`, so on a UI-only host
   the degraded capability-only path is unreachable and the UI host is mistaken for
   a current ownership-aware peer. This is why PR 2 must not be scoped as "add an
   availability check".
3. **The persistent profile is the wrong default for a machine that must not hold
   the operator's sessions in an agent-drivable jar.** R1 asks for convenience and
   §9.5 says what the convenience costs. Symptom: the operator's real logins are
   reachable by an agent after one approval, on a machine they intended to keep
   agents away from. Mitigation: the host-off switch (§9.3) that tears down the
   state file, the clear-data affordances (§5.4), and — the honest one — the
   extension-backed path remains available for anyone who wants agents to reach
   their *real* profile and can then reason about one boundary instead of two.
4. **Native-view layering** (§11.3). Cheap to hit (any modal over the browser
   area), easy to miss in review, expensive to retrofit if the answer turns out to
   require restructuring the route. Mitigation: §11.3's two rules stated in code,
   plus a design round on the paused state, plus §11.9's bounds-not-redesign
   property.
5. **Focus theft.** One `focus()` call, or a `show()` on a `ready-to-show`-style
   path, reproduces the recorded incident. Mitigation: the existing single-module
   rule (`window-raise.ts:2-17`), the forbidden-call list extended to
   `src/main/browser/`, and the frontmost-process assertion in QA (probe P12).
6. **Session cookies may not survive a restart** (§5.2), which would make the R1
   promise visibly false on sites that only issue session cookies. This is the
   single most likely way R1 disappoints. Mitigation: probe P2 measures it before
   the feature ships, and the UI copy is written from the measurement rather than
   from this document's expectation.
7. **Two release lines.** The app and lop version independently; a protocol
   mismatch must never present as a hang. Mitigated by advertisable-on-mismatch
   (§10.2), the `PROTO_MISMATCH` copy, and `check-vendored.mjs` catching the
   hand-edit case — but the *first* real mismatch will be found in the field, so it
   should be exercised deliberately in QA (probe P14).
8. **Config-dir divergence.** The UI main process never receives
   `LOCAL_OPERATOR_CONFIG_DIR` today — it passes only `LOCAL_OPERATOR_DESKTOP_TOKEN`
   (`backend-service.ts:67`, `:290`). Python resolves the state path through
   `config_dir()` (`paths.py:56-66`: the env override, else `~/.local-operator`).
   So a user who sets the override in their shell and launches the app from the Dock
   gets UI-browser state in one place and detection looking in another — a silent
   no-host. **This risk is higher in rev 2 than in rev 1** because there is now more
   than one file affected (the state file, and the profile/approvals if anyone is
   tempted to co-locate them). Mitigation: honour the variable in the UI, log the
   resolved path at startup, have `lop browser status` print the path it looked for
   when detection fails, and keep the profile/approvals out of the config dir
   entirely (§5.3, §9.3) so only the state file can drift.
9. **`WebContents` lifecycle leaks at N tabs** (§11.1, base-window.md:67-90).
   Mitigation: close-and-remove on every tab close, iterate on quit, and a test that
   asserts `webContents.getAllWebContents().length` returns to the app's own
   renderer count (probe P10).
10. **Unbounded user tabs.** §6.5 deliberately sets no agent-facing cap on user
    tabs, which means the only bound is memory. Mitigation: probe P10 measures RSS
    against tab count; if the measurement shows a problem, choose a number from the
    numbers.
11. **Background-tab throttling is unobservable from a headless run** (§11.5): the
    app's window-mode policy already sets `backgroundThrottling: false` outside
    `normal` for the whole window, which includes any child view. Symptom: a QA
    cell that "proves" nothing is throttled. Mitigation: measure it in `normal` or
    read the property, and say which was done.
12. **Site compatibility.** UA sniffing, and sites that refuse embedded
    environments. Mitigation: the explicit session UA of §11.5; residual risk is
    real and should be sampled on the sites the operator actually uses (§16.4).
13. **The new loopback listener** (§11.7). A keyed, no-CORS, loopback-only RPC
    surface is a narrow boundary, but it is the app's first, and its default-deny
    behaviour on permissions is not Electron's default. Mitigation: §11.7 in full,
    plus tests for 401 / malformed envelope / unknown method / oversize body.
14. **An extension loaded into the browser session can read and modify every page
    the agent and the user browse there** (§8.2), because `chrome.scripting` is
    fully supported and the session is shared. Mitigation: the extensions screen
    states it, the directory is user-managed (nothing is installed automatically),
    and the load is per-run and visible.
15. **cmux users.** A host with cmux *and* the UI still resolves by precedence; any
    change to that order will surprise somebody. Precedence is per fresh `open` and
    visible in the tool result, which is the mitigation today (§16.1).
16. **The paused-state flash** (§11.3) — cosmetic, but the kind of thing that gets
    reported as "flickering" and then costs a day. Measure it, do not assume it
    (probe P11).

---

## 15. Verification and QA plan

Every QA run obeys the isolation rules, which are stated here once because they are
the difference between evidence and an incident:

- `HOME` **and** `LOCAL_OPERATOR_CONFIG_DIR` both redirected (`AGENTS.md:290-304`),
  because `LOCAL_OPERATOR_CONFIG_DIR` alone leaves the cache and hardcoded home
  roots in the real home.
- **Unset every `CMUX_*` and `LOP_*` variable** in anything that boots a TUI or a
  fork (`AGENTS.md:315-319`). An inherited `CMUX_WORKSPACE_ID` has already renamed
  the operator's real cmux workspaces; an inherited `LOP_*` provider/model leaks a
  provider the fixture never chose.
- **Never touch the operator's live sessions or the installed bridge daemon.** Run
  any daemon as a plain subprocess on a non-default port
  (`python -m local_operator.browser_bridge.daemon --port <port>`), never through
  `install` — but note that the supervisor-name fix (`AGENTS.md:340-370`) is why
  this is now "cheaper" rather than "the only safe way": verify the reported
  supervisor name if `install` is ever called.
- The UI app is launched with a **window mode**, never `normal`
  (`AGENTS.md:169-207`), and its `--user-data-dir` (the Electron profile root) is a
  scratch path that must not be confused with the browser feature's partition
  inside it (§2.9).
- **Verify where writes actually go, not just that reads are redirected**
  (`AGENTS.md:328-338`): the analytics backfill wrote 612 rows into the live
  `~/.local-operator/analytics.db` from a sandboxed run. For this feature that means
  checking the resolved `getStoragePath()` and the state file's real location.

### 15.1 local-operator

Unit (`.venv/bin/python -m pytest tests/unit`, whole tree, exactly as CI):

- `tests/unit/ui_browser/test_state.py` — the freshness matrix: missing file,
  malformed JSON, dead pid, stale heartbeat, fresh, proto mismatch; plus "a read
  creates nothing" (the `state.py:83-96` rule) and "the write is 0600 under 0700"
  asserted with `stat`, not with intent.
- `tests/unit/ui_browser/test_backend.py` — the client against a real loopback
  listener: 200 round-trip, 401 on a wrong key, 422 on a malformed envelope, typed
  `proto_mismatch`, a refused connection, and a read timeout producing the "accepted
  but did not answer" sentence rather than "unreachable". Mirrors the existing
  `tests/unit/browser_bridge/test_backend.py` shape.
- `test_tool_selection.py` (existing file, extended) — the dispatcher matrix over
  all **eight** availability combinations (ui/bridge/cmux, present or not),
  asserting the chosen prefix and the *exact* copy for each degrade, with a fixture
  per host. This is where the §4.1 table is enforced.
- `test_ownership_host.py` (new) — the §10.5 selection and the gate: a record with
  `host: "ui"` sends `owner_*` to the UI client **and** resolves
  `ownership_mode()` from the UI's discovery store rather than `bridge.json`; a
  legacy record with no host still goes to the bridge; a UI-only host never
  constructs a bridge client; the lane **runs** on a UI-only host (assert it via
  `resource.initialize()` having been called, not via the resulting text);
  `recover`/`retain`/`release` are refused rather than silently dispatched when no
  ownership host is available — the cmux fall-through repro belongs here too.
  The selection claims are asserted THROUGH THE GATE (`execute_browser`), not
  against `_lane()` in isolation, and with both hosts faked so the test can tell
  "the right host" from "the only host": a resumed record naming the host the
  availability order would not pick must reach that host's wire, in BOTH
  directions, and a legacy record must reach the bridge with the app up.
- `test_browser_tool.py` (existing, extended) — the `handle` param: accepted on
  `open`, refused on every other action by `_validate_browser_args`; adoption of an
  unhanded tab refused with the typed code.

Then, for real (this is the part that counts):

- Boot the **real desktop app** against an isolated config dir, and drive a real
  `lop` session against it from an isolated `HOME` + `LOCAL_OPERATOR_CONFIG_DIR`
  (§10.3), with the window mode set.
- `curl` transcripts of `/rpc` for: no key, wrong key, unknown method, malformed
  body, and a valid call — the pattern
  `docs/design/browser-extension-evidence.md` already establishes.
- A real navigation, `read`, `snapshot`, `click`, `type`, `scroll`, `logs`,
  `screenshot` against a real site, with the screenshot's PNG magic re-verified by
  Python (`builtin.py:9511`'s `PNG_MAGIC` check is the same rule).
- Kill the UI process mid-session and read the typed error; restart it and re-`open`.
- A deliberate `PROTO_VERSION` skew (bump one side in a scratch build) and the typed
  mismatch.
- Quality gates before the PR, exactly as CI: `flake8`, `black --check` (pinned),
  `isort --check-only`, `pyright`, and the unit suite over the whole tree with
  `.venv/bin/python`.

### 15.2 local-operator-ui

- `pnpm check-types` (both projects), `pnpm lint`, `pnpm check-themes`,
  `pnpm check-vendored` (new, §12.2), `pnpm check-evidence`, `pnpm test:desktop`.
- **Real execution, not a green suite.** Stand the app up (`pnpm app:headless` for
  most cells, `inactive` where focus-dependent rendering matters) and drive a real
  `lop` session through the real RPC path with the real view. Cover the
  unauthorized / wrong-key / invalid-input cases with the actual responses, not with
  a mocked fetch.
- **Rendered evidence**, per the standing rules and the UI repo's
  `AGENTS.md:169-282`: before/after frames for the new route, the tab strip, the URL
  bar, the consent bar, the paused-with-overlay state, the profile settings and the
  extensions screen; consecutive frames across opening a tab, an agent `open` on a
  background tab, a palette open/close over the browser tab, and a restore; the four
  states required of any surface (loading, empty, error, populated); and the
  geometry numbers behind the frames (reported content rect vs actual view bounds,
  window size vs content box, whether a scrollbar appeared). The still shows the
  symptom; the numbers show the cause.
- **Focus assertions**: frontmost process unchanged before/after a full agent
  interaction, and the active tab unchanged after an agent `open` and after a
  restore.
- **Leak assertion**: `webContents.getAllWebContents().length` returns to the app's
  own renderer count after closing every tab and after app quit.
- UI review gates: a designer round on the rendered artifacts (D-findings) and a
  `ux-reviewer` round on the consent + hand-over flows, which change the interaction
  model — budget for both.

### 15.3 The empirical probes, named as experiments

Each is a cell an independent QA pass can run and report PASS/FAIL/BLOCKED with the
actual output, per the QA gate.

| probe | question | exact experiment | settles |
|---|---|---|---|
| **P1** | Do server (HTTP auth) credentials persist in the partition? | log into an HTTP-Basic-protected test endpoint in a tab, quit, relaunch, request it again, record whether credentials are re-prompted; also inspect the profile dir for a network-state file | §5.2 row 6 |
| **P2** | **Do session cookies survive an app restart?** (the R1 acceptance test) | pick a site that issues only session cookies; sign in inside the app's tab; quit the app *cleanly* and also by `SIGKILL` (two runs); relaunch; record the signed-in state both times | §5.2 row 7, and the R1 copy |
| **P3** | Where exactly does the persistent partition live on disk? | in a scratch `--user-data-dir`, write a cookie in the browser tab, call `ses.getStoragePath()`, then `find` the tree under `getPath('userData')` and record the real leaf path | §5.3, and the doc's pinned path |
| **P4** | Does the profile survive an Electron major-version update? | with a logged-in profile, install the next Electron major (or the previous one, to test downgrade), launch, and record whether the session survives and whether Chromium logs a profile migration | §5.4 |
| **P5** | Does tab restore reproduce count, order, per-tab history, scroll and form state, and what happens to a stale agent handle? | two tabs of a real site, one navigated twice with a scrolled long page and a half-typed form; quit; relaunch; record tab count/order/active, per-tab `navigationHistory.length()`, scroll offset, form value; then have a `lop` session act on its pre-restart `ui:` handle and record the typed error | §7.3 |
| **P6** | **Does Electron run an MV3 `background.service_worker`?** | a minimal MV3 extension whose service worker writes `Date.now()` via `chrome.storage.local`; load it; read the value back; record whether the key exists | §8.1, §8.4 |
| **P7** | What does a real 1Password-class extension actually do here? | unpack the operator's own MV3 build into the managed directory, restart, record: load success/error, the console warnings emitted during load, whether its popup responds, and whether its content script injects on a driven page | §8.4 |
| **P8** | Are unsupported-API warnings capturable? | an extension declaring `alarms`; capture main-process console during load; confirm the warning text and that the app's forwarder (`src/main/index.ts:307-311`) sees it | §8.3 |
| **P9** | Can a content-script-only extension help an agent-driven page? | an extension that highlights all `a` elements; check injection on an agent-driven tab in the shared partition | §8.3 |
| **P10** | Do views leak, and what does a tab cost? | open N=1,4,8 user tabs and 8 agent tabs, record `webContents.getAllWebContents().length` and tree RSS at each step, close all, record the count again after close and after quit | §11.1, §14.9, §14.10 |
| **P11** | Does `setVisible(false)/(true)` around an overlay flash on macOS? | consecutive-frame capture (≥5 frames at ~16 ms) across a command-palette open and close with the browser tab active, recording every frame's hash and the first frame that differs | §11.3, §14.16 |
| **P12** | Is focus ever stolen? | frontmost-process sample before/during/after a full agent interaction, ≥8 samples, as the UI repo's own evidence method already does (`AGENTS.md:225-230`) | §11.4 |
| **P13** | Does a hidden agent tab behave differently? | drive `read`/`snapshot`/`screenshot`/`scroll` on an *inactive* agent tab vs an active one, **in `normal` mode** (see §11.5), and read `backgroundThrottling` on both views; then repeat in `headless` and record that the property is already false there | §11.5, §14.11 |
| **P14** | What does a real protocol mismatch look like? | build the UI with `PROTO_VERSION` bumped in a scratch copy, run a real `lop` session, record the typed error and the tool text | §10.2, §14.7 |
| **P15** | Do the two hosts agree on the same page? | drive the extension host and the UI host against the same page and diff the `snapshot`, `read` and screenshot outputs | §4.2's parity claim (revision 1's open question, carried forward) |

### 15.4 The written test matrix QA must produce

Per the QA gate: a matrix file with columns (surface, command, actual output,
PASS/FAIL/BLOCKED), a repro per FAIL, and a verdict posted as
`### QA report — round <N>` with Q-findings, answered by
`### QA remediation — round <N>` until no FAIL remains on the head. The matrix must
cover, at minimum: the surfaces the diff touches, the likely-regressing neighbours
(`lop browser status`, the existing bridge-only degrade copy, the cmux path, the
app's own auth popup), and every probe above that the round's scope includes.
Round 2+ focuses on the remediation delta, not the whole matrix.

---

## 16. Open decisions for the operator

Each with my recommendation. Only the first two would change the shape of the work.

### 16.1 The `backend` hint on `open` — I still recommend taking it, for a different reason than before

Revision 1 argued for an optional `backend` argument (`"" | "ui" | "extension" |
"cmux"`) on `open`, refused on every other action by `_validate_browser_args`
(`builtin.py:7982`). **R1 weakens the original argument** (the UI jar is now a real
persistent jar, so the re-login cost that motivated it is mostly gone) **and
strengthens the surviving one**: device trust, hardware keys and enterprise
conditional access are exactly the cases where the session exists only in the real
profile, and the agent is the only party that knows whether a task needs it.

**Recommendation: take it.** Cost: one optional param on an already-large schema —
and `AGENTS.md:2194-2196` is explicit that rung 1 is "extend an existing tool",
while the footprint ladder's concern is *new tools*, not new params. Benefit: the
agent can say "this task needs your real logins" instead of being pinned to the
wrong jar for the surface's life (the surface prefix pins the transport —
`builtin.py:9908-9911`). Echo the choice in the `open` result so the model can tell
the user which jar it is in.

### 16.2 The `handle` param on `open` (R2)

**Recommendation: take it**, per §6.3, as the mechanism that lets an agent drive a
tab the user handed over. It is one optional field mapped onto a field the wire
already carries (`open`'s `params.tab`, `commands/nav.ts:126`). The alternative —
a new `adopt` action — would add a tool action, a wire method and a copy branch for
something `open`-with-a-handle expresses. This is the only model-facing schema
change in the design; if you would rather have none, the capability R2 asks for
becomes "the user can drive the agent's tabs", not both ways, and §6.3's hand-over
disappears.

### 16.3 Where the prompt renders

**Recommendation: the chrome band above the view**, per §9.2, with a system
notification as secondary. The alternative — a centred modal that hides the view —
is prettier and strictly worse (§9.2, §11.3). Your call only because it is a visual
decision, and the designer round should own the final treatment.

### 16.4 The user agent string

**Recommendation: set it once on the browser session, and decide deliberately
whether to strip the `Electron/` token.** Sites sniff; a jar that persists logins
is worse than useless if a bank refuses it. The honest options: (a) keep Electron's
default (most honest, most refusals), (b) present a plain Chrome UA (best
compatibility, mild deception), (c) present a Chrome UA without the
`Electron/...` suffix but with a `LocalOperator/` product token (a middle path that
still identifies the app). This is a UX/legal-flavoured call rather than a
technical one and I would take (c). *Settled by:* probing the operator's own sites
(§15.3, P2's site list doubles for this).

### 16.5 State-file directory name

**Recommendation: `run/ui-browser/host.json`, ephemeral port, per §10.1.** The only
decision needing input is whether the *directory* should carry the product rather
than the protocol (`ui-browser` vs `desktop`). I chose `ui-browser` because the
owner is the UI's browser host and it matches the `run/browser` / `run/mobile` /
`run/serve` pattern by *function*.

### 16.6 The hand-over's session picker

**Recommendation: name the session, the way §6.3 describes.** The alternative — a
hand-over that makes the tab available to *any* session — is simpler to build and
strictly worse: it turns a human act into a broadcast capability, and a second
agent session would inherit authority it was never given. If naming a session feels
heavy in practice, the fallback is "hand it to the session that most recently
drove a tab in this window", shown in the confirmation — weaker, but still a human
decision with a named target.

### 16.7 Extensions: how much of R4 to build

**Recommendation: build PR 6 as specified but keep it last in the order** (§13),
because §8.4's third sentence is the real answer to "install 1Password for easy
logins" and R1 delivers it. If R4's *exploratory* value is the point for you — you
want to see the honest matrix in the UI rather than to log in with fewer steps —
then PR 6 earns its place ahead of PR 5. That is the one ordering judgement in this
document I would genuinely hand back.

---

## Appendix A: claims checked, and things I could not verify

**Verified in this session, from the trees.** Every `local-operator` and
`local-operator-ui` `file:line` above was read at the refs in §0.1, not recalled —
including the ones that moved: `BROWSER_ACTIONS` `builtin.py:7498`,
`BRIDGE_ONLY_BROWSER_ACTIONS` `:7542`, `cmux_browser_available` `:7722`,
`_browser_state` `:7808`, `bridge_browser_advertisable` `:8518`,
`execute_browser` `:9623`, `_execute_browser` `:9820`, `build_browser_tool`
`:10119`; `state.py` `liveness` `:179`, `advertisable` `:206`; `backend.py`
`BridgeClient` `:341`, `_health_ok` `:185`; `protocol.py` `METHODS` `:162`,
`ErrorCode` `:283`, `HelloAck` `:369`; `resources.py` `ownership_mode` `:310`,
`_peer_identity` `:292`; `daemon.py` `DEFAULT_PORT` `:54`, the handshake `:2207`,
the loopback bind `:3741`; the nine `BridgeClient()` sites and five bridge state
reads listed in §10.5.

**Verified from the Electron v44.3.0 docs**, cited inline: extensions.md
(unpacked-only, per-session, not remembered across exits, persistent-sessions-only,
the supported manifest keys and API list), extensions-api.md
(`extensions.loadExtension` + its `ready`/in-memory constraints, `allowFileAccess`,
the deprecation of `ses.loadExtension` at session.md:1626-1710), session.md
(`fromPartition` `:26`, `clearCache` `:698`, `clearStorageData` `:704`,
`flushStorageData` `:716`, `setPermissionRequestHandler` `:928`,
`setPermissionCheckHandler` `:1001`, `setUserAgent` `:1377`, `isPersistent` `:1390`,
`getStoragePath` `:1712`, `clearData` `:1717`, `cookies` `:1774`, `will-download`
`:91`), web-contents.md (`did-fail-load` `:120`, `will-attach-webview` `:981`,
`loadURL` `:1087`, `isDestroyed` `:1170`, `close` `:1174`, `stop` `:1211`,
`reload` `:1215`, the deprecated navigation methods `:1223-1293`,
`executeJavaScriptInIsolatedWorld` `:1446`, `setWindowOpenHandler` `:1463`,
`capturePage` `:1805`, `setBackgroundThrottling` `:2381`, `navigationHistory`
`:2518`, `debugger` `:2535`), web-contents-view.md, view.md
(`addChildView` `:50`, `removeChildView` `:59`, `setBounds` `:65`, `setVisible`
`:127`), base-window.md (`:67-90` resource management, `:429` `contentView`),
navigation-history.md (including `restore` `:87`), structures/navigation-entry.md,
structures/extension.md, debugger.md (`:39-47`), webview-tag.md (`:3-10`),
structures/web-preferences.md (`partition` `:34`, `webviewTag` `:117`), app.md
(`getPath` `:616`, `setPath` `:679`, `userAgentFallback` `:1871`),
command-line-switches.md.

**Measured on this machine.** The extension's non-comment LOC and `chrome.*`
distribution (§2.6: 3,257 total, 648 chrome-free over eleven files, set identical
to revision 1's); the absence of `WebContentsView`/`<webview>`/`webviewTag` and of
any HTTP listener under `local-operator-ui/src`; the pinned Electron 44.3.0 and UI
version 0.21.0; the nine `BridgeClient()` sites and five bridge state reads; that
`feat/panel-views` sits at exactly `origin/main`; that multi-identity pairing has
landed (`26eaa938b`); that `docs/BROWSER.md:4-5`, `browser-extension.md`'s header
and `browser-multi-identity-pairing.md`'s header are stale; that the UI main process
receives only `LOCAL_OPERATOR_DESKTOP_TOKEN`.

**Not verified — and what would settle each.** The probes of §15.3, in full:
P1 session-cookie persistence, P2 the R1 acceptance test, P3 the on-disk partition
path, P4 profile survival across an Electron major, P5 restore fidelity and the
stale-handle path, P6 whether Electron runs an MV3 service worker (the single most
important unknown in this document), P7 a real 1Password-class extension's
behaviour, P8 warning capture, P9 content-script utility, P10 leak and memory at N
tabs, P11 the paused-state flash, P12 focus, P13 throttling (unobservable outside
`normal`), P14 a real protocol mismatch, P15 host parity on the same page.

Two more, which are not feature probes but should be settled before the relevant PR
is claimed done:

- Whether an inherited **`ELECTRON_RUN_AS_NODE`** or a stray `ELECTRON_*` variable
  can change how a driven view behaves under a harness. *Settled by:* running the
  QA matrix once with each `ELECTRON_*` variable present and once stripped.
- Whether `ses.getStoragePath()` returns a usable path **before** the session has
  written anything (it is documented to return `null` for in-memory sessions, but
  the persistent-session ordering is not stated). *Settled by:* calling it
  immediately after `session.fromPartition(PERSIST)` in a scratch run and logging
  the value; if it can be `null`, the host must publish the path lazily.
