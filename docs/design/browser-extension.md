# The Local Operator browser extension: driving the user's real browser without cmux

Status: DESIGN — no code exists yet.

> **Naming.** The store-facing product is the **Local Operator browser
> extension** (store listing: "Local Operator", like the Claude and ChatGPT
> extensions — product-named rather than capability-named, since it will grow
> features beyond this bridge). The local piece is the **browser bridge**: the
> Python package `local_operator/browser_bridge/`, the **bridge daemon** it
> runs, and the `lop browser` CLI namespace (`lop browser` is unclaimed —
> cli.py's top-level subcommands are credential/config/agents/teams/serve/
> mobile/update/exec/login/logout/mcp and friends, none named `browser`; the
> `browser` *tool* name is a tool, not a CLI subcommand, so there is no
> collision). The extension source lives in `extension/` at the repo root.

## 1. Problem

The `browser` tool is the only sanctioned way for a session to drive a
browser, and today it has exactly one backend: the cmux browser panel.
`build_browser_tool` (tools/builtin.py:5685) returns `None` unless
`cmux_browser_available()` (builtin.py:4731) resolves a cmux CLI, so on any
host without cmux the tool does not exist at all and the agent is told to fall
back to curl. The repo deliberately ships **no** browser engine — the builder's
own docstring (builtin.py:5691-5697) records that playwright belongs to the
pre-rewrite codebase and that the deciding feature of this tool is
*persistence*: it drives the browser the user is looking at, so cookies and
logins survive between calls and sessions, and the user can sign in by hand
when asked.

Most users do not run cmux. They run Chrome, Edge, Brave, or Arc. All of those
install Chrome extensions and expose the same `chrome.debugger`/CDP surface.
A small browser extension plus a loopback bridge daemon gives every one of
those users the same capability cmux users have today: the agent drives a
dedicated tab in the user's real browser, with the user's real cookie jar,
with no bundled engine and no headless anything.

Kimi's browser-extension bridge proves the shape works; ours adapts that shape
to this codebase's conventions (createIf gating, the mobile daemon's run-dir
discovery pattern, the one-surface-per-session rule).

### Non-goals (v1)

- Firefox and Safari (section 10).
- Driving the user's **active** tab. The bridge owns a dedicated tab, exactly
  as cmux owns a dedicated surface (`_cmux_new_surface`, builtin.py:4854
  documents why the one-surface rule exists). An explicit
  "attach to my current tab" affordance is future work.
- Multi-tab sessions, downloads, file uploads, iframes-as-first-class targets.
- Any remote (non-loopback) access.

## 2. What the existing code fixes in place

These are constraints, not choices — the extension backend must slot in
underneath all of them unchanged:

- **Tool schema.** `BrowserParams` (builtin.py:4683) and `BROWSER_ACTIONS`
  (builtin.py:4600: open/goto/read/snapshot/screenshot/click/type/close) are
  what the model, the prompts, and the skills know. The bridge backend
  presents the **same** actions with the same parameter names. No new tool,
  no new schema — the model must not be able to tell which backend answered.
- **createIf gating.** `TOOL_BUILDERS["browser"]` (tools/registry.py:50) calls
  a builder that returns `None` when no backend is reachable. Detection runs
  at session start on every session, so it must stay a file/env read that
  spawns nothing and opens no socket — the documented reason
  `cmux_browser_available` refuses to shell out (builtin.py:4734-4739).
- **URL scheme guard.** `_BROWSER_URL_SCHEMES` (builtin.py:4680) and
  `_validate_browser_url` (builtin.py:4885) refuse anything but http(s)
  *before* it reaches any backend. The bridge inherits this: the guard runs in
  `_validate_browser_args` (builtin.py:4945) ahead of dispatch, so
  `chrome://`, `file://`, `data:` and flag-shaped values never reach the
  extension. The extension additionally re-checks (section 6.3) because the
  daemon, not the Python tool, is the last trusted hop before the cookie jar.
- **Approval surface.** `_describe_browser_approval` (builtin.py:678) and the
  `write` approval tier (builtin.py:5727-5729) are backend-agnostic already —
  they read only the args. Unchanged.
- **Surface lifecycle.** The session owns a `BrowserSurface`
  (harness/types.py:410, held at session.py:1321, injected at session.py:3532)
  precisely so the handle outlives the per-turn ToolContext, and teardown
  closes it via `close_browser_surface` (session.py:5702-5729,
  builtin.py:5438). The bridge tab must ride the same holder and the same
  teardown path.
- **Truth-checking discipline.** The cmux backend never trusts an exit code:
  navigation is confirmed by document state (`_await_navigation`,
  builtin.py:5054), screenshots are verified by PNG magic (builtin.py:5263),
  fills are read back and compared (builtin.py:5413-5429), stale handles are
  probed per action (`_stale_surface_error`, builtin.py:5009). The bridge
  backend keeps every one of these behaviours; most become *cheaper* because
  the extension has real events where cmux only had polling.

## 3. Topology: a small standalone daemon, not a route on `lop serve`

Three candidate homes for the WebSocket the extension connects to:

**(a) A WS endpoint on the existing FastAPI server** (`local_operator/server/`,
`lop serve`). Rejected. That server is the desktop-UI API facade; TUI users —
the population this feature exists for — do not run it, its port is
user-configured per invocation (cli.py:361-365), and its lifecycle is bound to
one serve process, not to the set of live sessions. Bolting a
must-always-be-up, must-have-a-stable-port service onto a
sometimes-up, any-port server gives the extension nothing stable to dial.

**(b) In-session server: the first session that needs the browser binds the
port.** Rejected. Sessions come and go; the extension would reconnect-churn
across session boundaries, port ownership would have to be handed off between
dying and starting sessions, and two concurrent sessions racing for the bind
is exactly the coordination problem a broker exists to remove.

**(c) A standalone loopback daemon — recommended.** One small process owns:

- the extension leg: a WebSocket server the extension keeps open,
- the session leg: a trivial local RPC surface any number of lop sessions
  call,
- pairing state and the mux between them.

This is the mobile daemon's exact shape (`lop mobile serve`,
mobile/daemon.py:1-27: one process, loopback-only, Starlette, launchd-managed
via `lop mobile install`, discovery through 0600 files under
`~/.local-operator/run/mobile/`). We copy that pattern rather than inventing a
second one:

- **Stable port** for the extension: default `4099` (adjacent to mobile's
  `DEFAULT_PORT = 4098`, daemon.py:70), configurable, and echoed into the
  state file. The extension cannot read files, so the port must be stable by
  convention; the extension's options page allows overriding it.
- **Sessions come and go** without the extension noticing: the daemon holds
  the one extension socket; sessions are stateless callers.
- **Multiple concurrent sessions share one browser connection**: the daemon
  serializes commands per tab behind a per-tab lock so two sessions cannot
  interleave commands on the one surface.

> **v1 scope: a single active browser surface.** The extension owns exactly one
> dedicated tab, and a full per-session→tab table is deliberately deferred
> (section 12 / non-goals). Two concurrent `lop` sessions therefore SHARE that
> one tab rather than each getting an isolated surface: the daemon's per-tab
> lock keeps their commands from interleaving, and a second `open` degrades to
> the existing tab exactly as the cmux backend's one-surface rule does. This is
> the honest v1 contract; multi-surface isolation (a real session→tab table) is
> future work, not a silent gap.

### 3.1 The session leg is plain HTTP, not a second WebSocket

The extension leg must be WS (the daemon pushes commands *to* the extension —
the extension is the client, chrome extensions cannot listen). The session leg
does not need push in either direction: every browser tool call is strictly
request/response. So sessions speak `POST http://127.0.0.1:<port>/rpc` with
the bridge key in a header, one command per request, connection per request.

Why this instead of a session-side WS client:

- **No connection lifecycle on the session side.** A persistent WS from the
  session would have to live somewhere that survives the per-turn ToolContext
  rebuild (session.py:3516-3522 documents why per-turn storage loses state),
  which means growing `BrowserSurface` or adding a session-owned client with
  its own teardown. A per-call POST needs none of that: `BrowserSurface.
  surface_id` stays the only session-held state, exactly as today.
- **Testable with curl**, which matters for the testing-evidence bar.
- Loopback HTTP overhead is sub-millisecond against actions that take
  hundreds of ms in the browser.

Long-running commands (navigation settle, the origin-permission prompt) simply
hold the POST open; the daemon applies per-command deadlines (section 5.4) so
a hung extension turns into a bounded, typed error, not a hung tool call.

### 3.2 State file: how detection stays cheap

The daemon maintains `~/.local-operator/run/browser/bridge.json` (0600 under
0700, staged-write + `os.replace`, mirroring mobile/registry.py:39-57):

```json
{
  "pid": 4123,
  "port": 4099,
  "session_key": "<32-byte urlsafe token>",
  "proto": 1,
  "extension_connected": true,
  "extension_id": "abcdefghijklmnopabcdefghijklmnop",
  "browser_name": "Chrome",
  "heartbeat_at": 1774000000.0,
  "started_at": 1773990000.0
}
```

The daemon rewrites `heartbeat_at` every 15 s and flips `extension_connected`
on WS connect/disconnect. `bridge_browser_available()` is then: read one JSON
file, check pid liveness + heartbeat freshness (45 s timeout — the
`HEARTBEAT_INTERVAL_S`/`HEARTBEAT_TIMEOUT_S` pair from mobile/types.py:202-204)
and `extension_connected`. No socket, no subprocess — the same budget as
`cmux_browser_available`'s PATH lookup.

`session_key` in a 0600 file **is** the session-leg authorization, the same
argument mobile/registry.py:1-16 makes: anything that can read the file is
already the owning account. The extension-leg trust model is stronger and
separate (section 6).

Trade-off accepted: `extension_connected` can be up to one heartbeat stale
(browser quit an instant ago). The per-action error path covers it — the POST
fails or times out and the model gets the "extension not connected" string
(section 8) — so staleness costs one clear error, never a hang. The reverse
case (tool not advertised though the user opens the browser mid-session) is
the same accepted behaviour cmux has for a mid-session cmux install; a fresh
session picks it up.

### 3.3 Daemon lifecycle

`lop browser` CLI namespace, mirroring `lop mobile` (cli.py:384-401):

- `lop browser install` — write the LaunchAgent
  (`com.local-operator.browser.plist` on macOS, matching the fixed
  `com.local-operator.mobile` label convention in mobile/install.py:34; a systemd user
  unit on Linux), start it, print the pairing instructions.
- `lop browser serve [--port]` — foreground daemon.
- `lop browser status` — state file + a live `/health` probe + pairing state.
- `lop browser pair` — begin/complete pairing (section 6.2).
- `lop browser logs`, `lop browser uninstall` — as mobile.

Auto-spawning the daemon from a session's first browser call is deliberately
**out** of v1: a tool call that forks a daemon is a surprising side effect,
and the failure modes (half-started daemon, two racers) are the kind of thing
the mobile install path already solved with launchd. The error string for
"bridge not running" tells the user the one command to run (section 8).

## 4. Wire protocol

### 4.1 Single source of truth

`local_operator/browser_bridge/protocol.py` — Pydantic models for every message,
plus:

```python
PROTO_VERSION = 1            # bump = breaking; daemon refuses mismatched hellos
class ErrorCode(StrEnum): ...
```

A generator, `python -m local_operator.browser_bridge.gen_ts`, emits
`extension/src/protocol.gen.ts` (discriminated unions + the ErrorCode enum +
`PROTO_VERSION`) from the Pydantic models' JSON schema. `--check` mode diffs
the would-be output against the checked-in file and exits non-zero when stale
— the exact contract `generate-theme-css.mjs --check` already implements for
the mobile SPA (mobile/web/scripts/generate-theme-css.mjs:6-9), and CI runs it
the same way (section 9.3). The generated file is checked in so the extension
builds without a Python toolchain present.

### 4.2 Envelope

JSON text frames (extension leg) / JSON bodies (session leg). Request:

```json
{"id": "r-7f3a", "method": "goto", "params": {"url": "https://example.com"}}
```

Response, exactly one per id:

```json
{"id": "r-7f3a", "ok": true, "result": {"url": "...", "title": "..."}}
{"id": "r-7f3a", "ok": false, "error": {"code": "nav_failed", "message": "...", "data": {}}}
```

Extension→daemon unsolicited frames are `{"event": ...}` (no id):
`hello`, `tab_closed`, `origin_decision`, `pong`. The daemon never persists
events; they update in-memory state and the state file.

Handshake, first frame after WS connect, extension→daemon:

```json
{"event": "hello", "proto": 1, "token": "<pairing token or ''>",
 "extension_version": "0.1.0", "browser": "Chrome/126"}
```

Daemon replies `{"event": "hello_ack", "proto": 1, "paired": true}` or closes
with a WS close code from a small reserved range (4001 proto mismatch, 4003
unpaired, 4004 bad origin) so the extension can render the right popup state
without parsing a close reason string.

### 4.3 Command catalog (daemon→extension = the session-leg methods, relayed)

Every command carries `tab` (the daemon-issued surface token, section 5.1)
except `open` and `status`.

| method | params | result | notes |
|---|---|---|---|
| `open` | `url` | `tab, url, title` | Creates the dedicated tab (inactive, `active:false`), attaches debugger, waits for load. If a live bridge tab exists, degrades to `goto` — mirroring `_browser_open`'s open-degrades-to-goto rule (builtin.py:5145-5152). |
| `goto` | `tab, url` | `url, title` | Navigate + settle (section 5.3). |
| `read` | `tab, selector?` | `text, url, title` | `chrome.scripting.executeScript` running the same textContent extraction as `_dom_text_js` (builtin.py:4644) — that JS moves to a shared constant both backends use, keeping the innerText-vs-textContent lesson (builtin.py:4647-4652) in one place. |
| `snapshot` | `tab, selector?` | `snapshot, url, title` | Compact accessibility tree with `[e<N>]` click refs (section 5.5). |
| `screenshot` | `tab` | `data` (base64 PNG), `url, title` | `Page.captureScreenshot`. The **Python** side writes the file and keeps the PNG-magic check (builtin.py:5263-5275) and path resolution/approval (builtin.py:5231-5246) — the extension never sees a filesystem path. |
| `click` | `tab, ref \| selector` | `navigated, url, title` | Trusted input via `Input.dispatchMouseEvent` at the element's centre after `DOM.scrollIntoViewIfNeeded`. `navigated` from real navigation events replaces the cmux two-signal heuristic (`_navigation_started`, builtin.py:5307). |
| `type` | `tab, ref \| selector, text` | `value, url, title` | Focus + replace-not-append (the cmux fill-vs-type lesson, builtin.py:5388-5392): select-all then `Input.insertText`. Returns the field's post-fill value; **Python** keeps the compare-not-echo check (builtin.py:5413-5429). |
| `close` | `tab` | `{}` | `chrome.tabs.remove`; idempotent — an already-gone tab is success. |
| `status` | — | `tab?, url?, title?, origin_mode` | Liveness probe; the bridge analogue of `_cmux_url_probe`-as-liveness (builtin.py:4998-5006). |

Session-leg-only methods (answered by the daemon itself, never relayed):
`ping` (health), used by `lop browser status`.

### 4.4 Error taxonomy

`ErrorCode` values, each mapping to one actionable model-facing string
(section 8):

| code | meaning |
|---|---|
| `extension_disconnected` | Daemon up, no extension WS (browser closed, extension disabled). |
| `not_paired` | Extension connected but failed the token check. |
| `tab_closed` | Surface token names a tab that no longer exists. |
| `nav_failed` | Navigation errored (DNS, net::ERR_*, HTTP-level abort). |
| `nav_timeout` | Navigation started but never settled within the deadline. |
| `element_not_found` | Selector/ref resolved to nothing (stale ref after nav included). |
| `origin_denied` | The user denied this origin, or the prompt timed out. |
| `origin_prompt_pending` | Reserved for a future non-blocking flow; v1 never emits it. |
| `debugger_conflict` | Another debugger (DevTools open on the bridge tab) holds the target. |
| `busy` | A command is already in flight for this tab (daemon serializes; only returned if the queue is full). |
| `proto_mismatch` | Handshake version disagreement. |
| `internal` | Anything unclassified; message carries the detail. |

Timeout is expressed as `nav_timeout`/`internal` with `data.timeout_s` rather
than a transport-level silence: the daemon's per-command deadline (section
5.4) guarantees every request gets a typed response.

## 5. Extension architecture (Manifest V3)

### 5.1 Components and state

```
extension/
  manifest.json
  src/
    worker.ts          # service worker: WS client, command dispatch
    cdp.ts             # chrome.debugger attach/detach + typed CDP call helper
    commands/
      nav.ts           # open/goto/status, settle logic
      read.ts          # executeScript text extraction
      snapshot.ts      # AX tree -> compact refs
      input.ts         # click/type via Input.dispatch*
      shot.ts          # Page.captureScreenshot
    origins.ts         # allowlist store + prompt orchestration
    state.ts           # chrome.storage.session/local persistence
    protocol.gen.ts    # GENERATED from protocol.py — do not edit
    popup/
      popup.html popup.ts popup.css   # vanilla TS, no framework
    options/
      options.html options.ts        # port override, allowlist management
```

The **surface token** the daemon hands to Python (and Python stores in
`BrowserSurface.surface_id`) is `bridge:<tabId>:<nonce>` — the nonce is
minted per `open` and stored with the tab id in `chrome.storage.session`.
Chromium reuses tab ids rarely but not never; the nonce makes a recycled id
fail as `tab_closed` instead of silently driving an unrelated tab — the same
class of bug `_stale_surface_error` exists to stop on cmux
(builtin.py:5015-5035), closed here by construction rather than by probing.

Persistent state:

- `chrome.storage.local`: pairing token, daemon port override, per-origin
  allowlist (`{origin: "allow" | "deny"}`).
- `chrome.storage.session`: current tab id + nonce, current snapshot ref map
  (`e<N>` → CDP backendNodeId, keyed by a navigation epoch), pending origin
  prompt. Survives worker death, dies with the browser — which is correct for
  all three.

### 5.2 Worker lifetime

MV3 service workers are killed after ~30 s of inactivity. Design for death,
not against it:

- **Keepalive while connected:** the daemon pings over the WS every 20 s and
  the worker pongs. Since Chrome 116, active WebSocket traffic extends the
  worker's lifetime, so a connected worker stays alive indefinitely on ping
  traffic alone. An attached `chrome.debugger` session additionally pins the
  worker while a command runs.
- **Reconnect after death:** a `chrome.alarms` alarm every 30 s (the MV3
  minimum period is fine — reconnect latency of ≤30 s when idle is
  acceptable; a command arriving meanwhile fails fast with
  `extension_disconnected` and the user-visible state is honest) plus
  `chrome.runtime.onStartup`/`onInstalled` wake the worker; on wake it
  re-reads `chrome.storage`, re-dials the daemon with exponential backoff
  (1 s → 30 s cap) inside its awake window, and re-arms the alarm.
- **Reattach after death:** `chrome.debugger.attach` does not survive the
  worker; on the first command after a reconnect, `cdp.ts` re-attaches to the
  stored tab id (verifying the nonce) lazily. The ref map's navigation epoch
  makes any pre-death snapshot refs fail closed as `element_not_found` with a
  "re-snapshot" hint if the page changed while nobody was attached.

The `chrome.debugger` infobar ("… is debugging this browser") is **accepted
as a feature**: it is the user-visible truth that an agent can drive a tab,
and it disappears when the debugger detaches. We detach on `close` and when
the bridge tab is gone, so the banner tracks reality.

### 5.3 Navigation settle

Where the cmux backend had to poll two disagreeing views of the URL
(`_await_navigation`, builtin.py:5054-5112, and the two-signal click
heuristic, builtin.py:5307-5347), the extension has real events:

- `goto`/`open`: `chrome.tabs.update(tabId, {url})`, then wait for
  `webNavigation.onCompleted` (or `onErrorOccurred` → `nav_failed`) for that
  tab's main frame, then read `tab.url`/`tab.title`. Redirects are covered
  because `onCompleted` fires for the final document.
- `click`: dispatch the trusted click, then race a short grace window
  (1.5 s, matching `BROWSER_CLICK_GRACE_S`'s reasoning, builtin.py:4622-4626)
  for `webNavigation.onBeforeNavigate`; if navigation started, wait for
  settle as above and return `navigated: true`. Same-document SPA route
  changes surface via `onHistoryStateUpdated` and report
  `navigated: true` with the new URL — a case the cmux backend could only
  approximate with its document marker.

Python keeps its own outer timeout (section 5.4) but no longer re-implements
settle detection: the extension's answer is authoritative, and the result
always carries the **live** `url`/`title` so `_page_line`'s
report-what-is-actually-on-screen rule (builtin.py:5115-5122) holds.

### 5.4 Deadlines

Per-command budgets enforced by the **daemon** (so a dead worker cannot hang
a session): `open`/`goto` 30 s, `click`/`type` 25 s, `read`/`snapshot`/
`screenshot` 20 s, origin prompt 60 s (section 6.3). Python's HTTP client uses
budget + 5 s so the typed daemon error always wins the race. Every timeout
yields a typed response, never a dropped id.

### 5.5 Snapshot and click refs

`Accessibility.getFullAXTree` (scoped by `DOM.querySelector` when a selector
is given), pruned to interesting roles (roughly: focusable, clickable, named,
or landmark) and rendered as an indented compact tree:

```
- link "Sign in" [e3]
- textbox "Email" [e5]
- button "Continue" [e6]
```

Each `e<N>` maps to a CDP `backendNodeId`, stored in the ref map with the
current navigation epoch. `click`/`type` accept either a ref or a CSS
selector; refs resolve via `DOM.pushNodesByBackendIdsToFrontend` →
`DOM.getBoxModel`, selectors via `DOM.querySelector`. A ref from a previous
epoch answers `element_not_found` with "the page has navigated since that
snapshot; take a new snapshot" — matching the format the model already knows
from cmux's `snapshot`/`[e5]` refs (the `BrowserParams.selector` description,
builtin.py:4693-4697, names `e5` explicitly and is unchanged).

## 6. Security and pairing

Threat model, in decreasing order of concern:

1. A **web page** reaching the daemon (WebSocket connections are not blocked
   by CORS): defeated by the Origin check — the daemon accepts the extension
   leg only when `Origin` is exactly `chrome-extension://<paired id>`; pages
   send `https://…` origins and are closed with 4004 before the handshake.
2. A **rogue extension**: defeated by pairing — the token is issued only after
   a human confirms a short code, and the Origin's extension id is pinned at
   pairing time.
3. A **compromised local process** (already able to read the state file and
   thus the session key): cannot be fully stopped from *invoking* the bridge
   — it owns the account — but is stopped from silently reaching sensitive
   sites by the **in-extension** per-origin allowlist (6.3): the prompt
   renders in browser UI that no local process can click, so "the agent opened
   the user's bank" always passed through a human click on that machine's
   screen. This is why the allowlist lives in the extension and not the
   daemon.

Binds are loopback-only (`127.0.0.1`), as the mobile daemon's are
(daemon.py:69-70); remote access is explicitly a non-goal.

### 6.1 Legs and credentials

- **Session leg** (`POST /rpc`): `X-Bridge-Key: <session_key>` from the 0600
  state file. File permissions are the authz model (mobile/registry.py:4-8).
- **Extension leg** (`WS /extension`): Origin pin + pairing token. The token
  is a 32-byte urlsafe secret held in `chrome.storage.local`; the daemon
  stores only its SHA-256 in `~/.local-operator/browser/pairing.json` (0600),
  beside the pinned extension id. Config dir conventions follow
  `local_operator/paths.py` (`config_dir()`, paths.py:56) — nothing hardcodes
  `~/.local-operator`.

### 6.2 First-connect pairing

1. Extension connects with an empty token → daemon replies `hello_ack
   {paired: false}` and mints a 6-digit code with a 2-minute TTL, single
   attempt counter (5 tries then a new code).
2. The daemon **displays** the code via `lop browser pair` (and `install`
   prints it at the end of its run). The user **types it into the extension
   popup**. Direction matters: the secret flows terminal→browser, so a rogue
   extension cannot learn the code by merely connecting — it has to present
   what only the machine's terminal showed.
3. On a correct code the daemon issues the long-lived token, records
   `{extension_id, token_sha256, paired_at}`, and the popup flips to the
   paired state. Re-pairing (new browser profile, token wipe) is the same
   flow; `lop browser pair --reset` revokes the stored hash first.

### 6.3 Per-origin allowlist (extension-enforced)

Default-deny. On `open`/`goto`/click-navigation to an origin not in the
allowlist, the extension:

1. Holds the command, sets the toolbar badge (`!`), and shows the pending
   request in the popup.
2. The user's choice resolves the command: allow → proceed (and persist on
   a standing grant); deny or 60 s of silence → `origin_denied` with the
   origin named.

**2026-09-05 (extension 0.1.8):** the prompt is now a scope dropdown
(domain / site / once) with Allow and Deny buttons, replacing the three
fixed buttons above. See the PR for the scoped-grant design and the
dangerous allow-all setting.

Redirect handling: the allowlist is checked against the **final** origin too
(via `webNavigation.onBeforeNavigate` per hop); a redirect into an unlisted
origin pauses at that hop and prompts. Subresource origins are *not* gated —
this is a navigation gate, not a network filter; the browser's own sandbox
governs subresources — and the doc for the popup says so.

The extension re-validates scheme (http/https only) independently of the
Python guard, so a compromised daemon still cannot send the tab to
`chrome://settings` or `file://`.

## 7. lop-side integration

### 7.1 Backend seam in the tool layer

The smallest change that keeps one tool over two transports: extract the
transport behind an internal interface, keep every model-facing behaviour in
the shared dispatcher.

New module `local_operator/browser_bridge/backend.py` implements the session-leg
client (read state file → POST /rpc → typed result), and
`tools/builtin.py` gains a thin dispatch:

- `bridge_browser_available() -> bool` — the state-file check (3.2), imported
  lazily so the CLI startup path stays import-light (the rule
  mobile/types.py:8-10 records).
- `execute_browser` (builtin.py:5476) keeps its validation prefix unchanged
  (params → action check → availability → `_validate_browser_args` →
  `_browser_state`) and then dispatches per-backend. Backend choice is made
  **once per surface**, recorded in the surface token's prefix
  (`surface:<n>` = cmux, `bridge:<tab>:<nonce>` = extension bridge), so a session
  that opened on one backend never silently continues on the other. With no
  surface open, `open` picks: cmux if `cmux_browser_available()` (preferred —
  inside cmux the panel is the surface the user is looking at), else bridge,
  else the existing terminal error.
- `build_browser_tool` (builtin.py:5685) advertises when **either** backend
  is available. The description keeps its persistence-first wording
  (builtin.py:5716-5724) — it already describes both backends accurately once
  "the CMUX browser panel in their terminal" is generalized to name both
  surfaces.
- `close_browser_surface` (builtin.py:5438) branches on the token prefix so
  session teardown (session.py:5702) closes a bridge tab exactly as it closes
  a cmux surface today. Same drop-the-handle-regardless rule.

Unchanged: `BrowserParams`, `BROWSER_ACTIONS`, `_describe_browser_approval`,
the `write` approval tier, `registry.py` (the existing `browser` entry's
builder just gains the second availability check — no new table entry, per
the AGENTS.md rule that a second gating convention is itself a regression,
AGENTS.md "footprint ladder").

Shared helpers hoisted rather than duplicated: the text-extraction JS
(`_dom_text_js`) and `_page_line` move to (or are imported by) the bridge
path so both backends report pages the same way.

### 7.2 What each Python check becomes on the bridge

| cmux behaviour | bridge equivalent |
|---|---|
| `_stale_surface_error` per-action probe (builtin.py:5009) | Not needed as a probe: the nonce in the surface token fails closed (`tab_closed`) — the error string mirrors the cmux one ("use 'open' to get a new tab"). |
| `_await_navigation` polling (builtin.py:5054) | Extension-side event wait; Python trusts the typed result. |
| PNG magic check (builtin.py:5263) | Kept in Python: decode base64, check magic, write file, report size — identical model-facing message. |
| fill read-back compare (builtin.py:5413) | Kept in Python against the `value` the extension returns. |
| open-degrades-to-goto (builtin.py:5145) | Same rule, implemented in the extension's `open`. |

## 8. Failure UX: what the model reads

Every state maps to one actionable string (the cmux backend's standard —
e.g. builtin.py:5495-5503). No path may hang: daemon deadlines (5.4) bound
every wait.

| condition | tool result (is_error=true unless noted) |
|---|---|
| No cmux, no bridge state file / stale heartbeat | Tool **not advertised** (createIf) — the honest absence the current builder documents. |
| Bridge daemon up, extension never connected / browser closed | `browser extension not connected: the bridge daemon is running but no browser is attached. Ask the user to open their browser (the extension reconnects automatically), or check the extension is enabled.` |
| Daemon up, extension unpaired | `browser bridge not paired: run 'lop browser pair' and enter the code in the extension popup, then retry.` |
| Daemon died between availability check and call (POST refused) | `browser bridge unreachable: the daemon at 127.0.0.1:<port> is not answering. Run 'lop browser status'; 'lop browser install' starts it.` |
| Origin denied / prompt timeout | `navigation to <origin> was denied by the user (or the permission prompt went unanswered). Do not retry the same origin; ask the user to allow it from the extension popup if it is needed.` |
| Tab closed by user | `browser tab <token> is gone; dropped the handle. Use 'open' with a URL to get a new tab.` (mirrors builtin.py:5045-5051) |
| Tab crashed / debugger detached mid-command | `the browser tab crashed while <action> was running. 'open' the URL again to recover.` |
| DevTools open on the bridge tab | `cannot drive the tab: DevTools (or another debugger) is attached to it. Ask the user to close DevTools on that tab.` |
| Navigation failed / timed out | Same shapes as today's cmux strings (builtin.py:5129-5132, 5177-5182), with the extension's `net::ERR_*` detail interpolated. |

## 9. Repository layout, build, CI, packaging

### 9.1 Files to create

```
local_operator/browser_bridge/
  __init__.py
  protocol.py        # Pydantic messages, ErrorCode, PROTO_VERSION (4.1)
  state.py           # run-dir state file read/write (3.2; mirrors mobile/registry.py)
  daemon.py          # Starlette app: WS /extension, POST /rpc, /health; mux; pairing
  backend.py         # session-leg client + the bridge implementations of the 8 actions
  gen_ts.py          # protocol.gen.ts generator, --check mode
  install.py         # LaunchAgent/systemd install-uninstall (mirrors mobile/install.py)
extension/           # the Local Operator browser extension — tree per 5.1
docs/design/browser-extension.md   # this document
tests/unit/browser_bridge/ # protocol round-trip, state-file, backend-against-fake-daemon
```

### 9.2 Files to modify

- `local_operator/tools/builtin.py` — backend seam (7.1); hoist shared JS.
- `local_operator/session/session.py` — none expected: teardown already calls
  `close_browser_surface`, which becomes backend-aware in builtin.py.
- `local_operator/cli.py` — `lop browser` subparser, lazy imports (the
  mobile rule, cli.py:384-385).
- `.github/workflows/ci.yml` (or a sibling workflow) — section 9.3.
- `pyproject.toml` — **no packaging change needed**:
  `[tool.setuptools.packages.find] include = ["local_operator*"]`
  (pyproject.toml:146-147) already scopes the wheel to the Python package, so
  a repo-root `extension/` can never leak into it. The `browser_bridge` package
  rides the existing include glob. The daemon's deps (starlette/uvicorn/
  websockets) are already in the base dependency set (pyproject.toml, base
  deps list). Verify in CI the way publish.yml already greps the wheel for
  the mobile bundle (publish.yml:56-58) — assert `extension/` is absent.

### 9.3 Extension build and CI

- **esbuild, not vite, no framework.** The worker and popup are a few KB of
  vanilla TS with no HMR need and no SPA; one `build.mjs` calling esbuild
  (bundle worker.ts, popup.ts, options.ts; copy static HTML/manifest) is the
  lightest thing that works. vite stays the right call for the mobile SPA;
  adopting it here would import a dev-server we would never start. pnpm as
  the package manager for consistency with mobile/web.
- **CI**: a new workflow `extension.yml` triggered on
  `paths: [extension/**, local_operator/browser_bridge/protocol.py, local_operator/browser_bridge/gen_ts.py]`
  running: `pnpm install --frozen-lockfile`, `tsc --noEmit`, `node build.mjs`,
  and `python -m local_operator.browser_bridge.gen_ts --check`. The gen-check
  **also** runs in the main python CI job (it is pure Python and cheap), so a
  protocol.py edit that forgets to regenerate fails even when no extension
  file changed — the failure mode a paths-gated job alone would miss.
- The extension is **not** part of the wheel or the PyPI release. Its release
  artifact is the store submission zip (`node build.mjs --zip`), versioned
  independently in its own manifest; `PROTO_VERSION` is the compatibility
  contract between the two release lines, and the handshake enforces it.

### 9.4 Manifest permissions (with store-justification bullets)

| permission | justification (as submitted) |
|---|---|
| `debugger` | Screenshots (`Page.captureScreenshot`), accessibility snapshots, and trusted click/type input for the one tab the user delegated to their local agent. This is the core capability; no remote party is involved. |
| `tabs` | Create/read/close the single agent-owned tab; read its URL/title to report what page an action landed on. |
| `scripting` | Cheap page-text extraction (`executeScript` returning `textContent`) without a debugger round trip. |
| `storage` | Pairing token, daemon port, per-origin allowlist, current tab handle. Nothing leaves the machine. |
| `alarms` | Reconnect timer so the MV3 worker re-dials the local daemon after being suspended. |
| `webNavigation` | Navigation-completion events (replaces polling) and the redirect-hop origin gate. |
| `host_permissions: <all_urls>` | The agent may be asked to open any site the **user approves per origin** in the extension's own prompt; the extension enforces default-deny itself. Required for `scripting`/`debugger` on user-approved origins. |

Store submission additionally needs, from the implementation: screenshots of
the four popup states (unpaired with code entry; paired-idle/connected;
origin-permission prompt; disconnected/reconnecting), a short demo of the
pairing flow, a privacy statement ("all traffic is between this extension and
a daemon on 127.0.0.1 on your own machine; no analytics; no remote servers"),
and the single-purpose description. Budget review lead time: extensions
requesting `debugger` get slower, stricter review.

### 9.5 Test plan (executable by the coder)

1. **Unit (Python)**: protocol model round-trips + gen_ts determinism
   (`--check` green after generate); state-file freshness matrix (missing,
   stale heartbeat, dead pid, connected flag); backend action functions
   against an in-process fake daemon (Starlette TestClient) covering every
   ErrorCode; dispatcher backend-selection matrix (cmux only / bridge only /
   both / neither) and token-prefix routing incl. teardown.
2. **Unit (TS)**: snapshot pruning/ref-map epoch logic and origin-gate
   decisions as pure functions (extracted precisely so they are testable
   without a browser); `tsc --noEmit` as the type gate.
3. **Integration (manual, recorded as PR testing evidence per house rules)**:
   real Chrome + real daemon — pairing flow end to end; each of the 8 actions
   against a live site; sign-in persistence (log into a site by hand,
   restart the *session*, `read` shows the logged-in page); origin deny;
   kill the browser mid-session and read the typed error; kill the worker
   (chrome://serviceworker-internals) and observe alarm-driven reconnect;
   two concurrent lop sessions sharing the daemon; `curl` transcripts of
   `/rpc` for the unauthorized/wrong-key/invalid-input cases.
4. **Regression**: the full existing suite — the cmux path must be
   byte-identical in behaviour when cmux is present.

## 10. Browser compatibility

Chrome, Edge, Brave, and Arc are all Chromium and expose identical
`chrome.debugger`/CDP, MV3 worker semantics, and extension install flows
(Edge sideloads Chrome-store extensions; Brave/Arc use the Chrome store
directly). One build serves all four; the only per-browser difference worth
recording is cosmetic (where the toolbar pin lives).

Out of scope v1, and why:

- **Firefox**: MV3 exists but there is **no `chrome.debugger` equivalent**
  (no CDP; Firefox's remote protocol is not exposed to extensions), so
  screenshots, AX snapshots, and trusted input have no implementation path.
  A degraded content-script-only backend is possible later but would be a
  different, weaker capability and should not silently share the tool's name.
- **Safari**: Safari Web Extensions require a native app wrapper, App Store
  distribution, and again expose no debugger API. Different distribution
  pipeline, different capability ceiling — its own project if ever.

## 11. Risks to watch during rollout

- **MV3 worker churn**: the keepalive/reconnect design (5.2) rests on
  Chrome ≥116 WS-activity lifetime extension. Watch for users on older
  Chromium forks; the failure mode is benign (≤30 s reconnect latency) but
  should be measured, not assumed.
- **Store review**: `debugger` + `<all_urls>` is the highest-scrutiny
  combination. Mitigation: the per-origin default-deny prompt is exactly the
  story reviewers want; submit early, and keep a sideload path documented
  (`chrome://extensions` → developer mode) so the feature is usable before
  approval.
- **Origin-prompt fatigue**: default-deny with exact-origin persistence is the
  right trade. The popup shows the full authority, including nondefault ports.
  Only literal `localhost`, `127.0.0.1`, and `[::1]` prompts offer an explicit
  same-scheme, same-host all-port grant; trailing dots, subdomains, shorthand
  IPv4, mapped IPv6, and names that merely resolve to loopback remain exact-
  origin only. Watch real transcripts for prompt-storms (redirect chains
  through consent/SSO domains). The standing-grant affordances plus gating
  navigations only (not subresources) should keep it to one prompt per new
  site.
- **Two release lines**: extension and Python versions drift by design;
  `PROTO_VERSION` mismatches must show up as the popup's "update needed"
  state and the daemon's 4001 close, not as mystery timeouts. Test the
  mismatch path explicitly before the first protocol bump, not after.
- **Backend precedence surprises**: a cmux user with the extension installed
  gets cmux (7.1). If real usage shows people wanting to prefer the bridge
  inside cmux, add an explicit config knob then — do not guess now.

## 12. Summary of recommendations

1. Standalone loopback daemon (`lop browser` namespace) on a stable default
   port, mirroring the mobile daemon's run-dir/state-file/launchd patterns —
   not a route on `lop serve`, not an in-session server.
2. WS only on the extension leg; plain authenticated HTTP on the session leg;
   stateless sessions, all mux state in the daemon.
3. One protocol source of truth in Pydantic, generated TS checked in, `--check`
   in CI on both the paths-gated extension workflow and the main Python job.
4. Same tool, same schema, backend chosen per `open` and pinned in the surface
   token prefix; every cmux truth-check (PNG magic, fill compare, live
   URL/title reporting) retained on the Python side.
5. Security = loopback bind + Origin pin + human-confirmed pairing
   (terminal→browser code direction) + extension-enforced site-access grants
   (exact-origin by default, explicit loopback all-port only) and default-deny;
   the debugger infobar embraced as visibility.
6. esbuild + vanilla TS extension, repo-root directory excluded from the wheel
   by the existing packages.find include, released to the store on its own
   version line with `PROTO_VERSION` as the compatibility contract.
