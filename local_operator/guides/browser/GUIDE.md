---
name: browser
description: Set up and use the Local Operator browser extension so the browser tool can drive the user's real Chromium browser (Chrome, Edge, Arc, Brave). Covers install, pairing, per-site permissions, what to ask the user for, focus safety, and the cmux/playwright fallbacks.
---

# Browser: drive the user's real browser with the Local Operator extension

Read this guide before doing browser work when the `browser` tool is missing,
when the user asks to set up browser access, or when a browser action returns a
"not connected / not paired / unreachable" error. It is the operational
playbook; the design contract lives in `docs/design/browser-extension.md`.

## The one thing to know

The `browser` tool has three possible backends, in preference order:

1. **The Local Operator browser extension** (preferred) — a real Chromium
   profile (Chrome, Edge, Arc, Brave, any Chromium) paired to lop over a
   loopback bridge daemon. It carries the user's real cookies and logins, so it
   reaches authenticated pages, and the user can sign in by hand and you carry
   on. This is what you should set up and use.
2. **A cmux browser panel** (fallback) — used automatically when lop runs
   inside cmux and no extension is connected.
3. **`bash` + curl for static pages** (last resort) — only when the user
   declines the extension and there is no cmux panel. Never a downloaded
   browser engine.

**Never** run `playwright install`, puppeteer, or download Chromium to load a
page or take a screenshot. A throwaway headless browser cannot hold the user's
logins, cannot be reached by the user to sign in, and adds a ~150 MB dependency
this project deliberately does not ship. If the extension truly cannot be set
up and no cmux panel exists, say a rendered screenshot is unavailable and why —
do not build a second browser stack.

## Focus safety — never steal the user's focus

The agent often browses in the background while the user works in another
window. The extension is built to respect that and you must keep it that way:

- The extension's tab is created **inactive** (`active: false`) and no action
  ever activates it, raises a window, or brings the browser to the foreground.
- Screenshots use CDP `Page.captureScreenshot`, which works on a background
  tab — you never need the tab focused to capture it.
- Do **not** ask the user to switch to the browser or leave a tab focused for
  actions to work. If you ever find yourself wanting to activate a tab, don't —
  the whole point is that the user keeps doing their own thing.

Chrome shows a small "Local Operator is debugging this browser" banner on the
tab the agent controls. That is intentional and good: it is how the user can
always see which tab the agent has. Tell them it is expected, not a warning.

## When the `browser` tool is missing: set the extension up

Treat the tool's absence as a one-minute setup step, not a dead end. Do the
work with the user — do not dump commands and wait.

### 1. Check status

```sh
lop browser status
```

It reports: `installed`, `daemon healthy`, `extension connected`, `paired`,
the `port` (default 4099, loopback only), and the log path. This tells you
which step is missing.

### 2. Install and start the bridge daemon

```sh
lop browser install
```

This installs and starts the loopback daemon (a LaunchAgent on macOS /
systemd user unit on Linux, so it survives restarts) and prints the pairing
code at the end. The daemon binds **127.0.0.1 only** — never widen it.

### 3. Get the extension into the user's browser

The extension must be loaded once per browser profile. **Ask the user which
Chromium browser they want the agent to use** (Chrome, Edge, Arc, or Brave),
then:

- **If published to the Chrome Web Store:** send them the store listing and ask
  them to click **Add to <browser>**. This is the normal path once the
  extension is live.
- **If loading unpacked (development / pre-store):** ask the user to:
  1. Open `chrome://extensions` (or `edge://extensions`, `brave://extensions`,
     Arc's extensions page).
  2. Turn on **Developer mode** (top-right toggle).
  3. Click **Load unpacked** and select the built `extension/dist` directory.

  You cannot click these buttons for them — enabling Developer mode and loading
  an unpacked extension is a deliberate user action the browser requires. Give
  them the exact directory path and wait for them to confirm it loaded.

### 4. Pair the extension to this machine

```sh
lop browser pair
```

This **displays** a 6-digit code (2-minute TTL, five tries then a fresh code).
Ask the user to **type it into the extension popup** (click the toolbar icon →
enter the code). The secret flows terminal → browser on purpose, so pairing
proves the extension is talking to *this* user's lop and nothing else.

Re-pairing (new browser profile, wiped token): same flow. `lop browser pair
--reset` revokes the current pairing first.

### 5. Confirm and use

Re-run `lop browser status`; you want `extension connected: yes` and
`paired: yes`. The `browser` tool now appears — use it.

## What to ask the user for

Only the things the browser genuinely requires a human for. Ask concisely, and
batch them:

- **Which browser** to use (Chrome / Edge / Arc / Brave) — before loading the
  extension.
- **To load the extension** (store "Add", or Developer-mode "Load unpacked" of
  `extension/dist`) — a required manual browser action.
- **To type the pairing code** into the extension popup.
- **To allow a site** when a per-site permission prompt appears (see below).
- **To sign in** to a site by hand when the agent hits a login wall — then you
  continue in the same authenticated session.
- **To close DevTools** on the agent's tab if you get a "another debugger is
  attached" error (Chrome only lets one debugger attach at a time).

Never ask for passwords, tokens, or session cookies — the extension uses the
browser's existing login state; the user signs in themselves.

## Per-site permissions (default-deny)

The extension gates navigation by origin, enforced in the browser where no
local process can click for it. The **first** time the agent opens a new site,
the extension prompts the user in the popup: **Allow once / Always allow /
Deny**. A redirect into a new origin prompts again at that hop.

What this means for you:

- The first `open`/`goto` to a new site may pause until the user answers. If a
  navigation returns `origin_denied` (or the prompt timed out), **do not retry
  the same origin** — tell the user the site needs approval and ask them to
  Allow it from the popup, then retry once.
- If you know you will visit several sites, tell the user up front so the
  prompts are expected.
- Subresources are not gated (this is a navigation gate, not a network filter).
- `http`/`https` only; the extension refuses `chrome://`, `file://`, etc.

## Using the tool

Same `browser` tool, one surface at a time. Actions:

- `open` (start a surface at an http/https URL) · `goto` (navigate the surface)
- `read` (page text; `selector` scopes it) · `snapshot` (accessibility tree
  with click refs like `e5`)
- `screenshot` (writes a PNG; works on the background tab)
- `click` / `type` (by CSS selector or a snapshot ref; trusted input events)
- `scroll` (by direction, pixel delta, or scroll an element into view)
- `logs` (the tab's console output and uncaught exceptions since it opened —
  for debugging web apps)
- `close` (drop the surface and its tab)

Prefer `snapshot` to discover click targets (it returns stable refs), `read`
for content, `screenshot` for visual verification, `logs` when a page
misbehaves. Capture before/after screenshots for any visual change.

## Troubleshooting (what each error means)

Every failure is one actionable string; act on it rather than retrying blindly:

- **"extension not connected: the bridge daemon is running but no browser is
  attached"** — the user's browser is closed or the extension is disabled. Ask
  them to open the browser (it reconnects automatically) or enable the
  extension at `chrome://extensions`.
- **"browser bridge not paired"** — run `lop browser pair` and have the user
  enter the code.
- **"browser bridge unreachable: the daemon … is not answering"** — run
  `lop browser status`; `lop browser install` (re)starts it. Logs:
  `lop browser logs`.
- **"navigation to <origin> was denied"** — per-site permission; ask the user
  to Allow it in the popup. Do not retry the same origin unprompted.
- **"tab is gone" / "tab crashed"** — the user closed or crashed it; `open` the
  URL again to get a fresh surface.
- **"another debugger is attached"** — ask the user to close DevTools on that
  tab.

## Cleanup / removal

```sh
lop browser uninstall           # stop and remove the daemon
lop browser uninstall --purge   # also delete pairing state
```

The extension is removed by the user from their browser's extensions page.

## Browser compatibility

Works identically on any Chromium browser that installs Chrome extensions and
supports the `chrome.debugger` API: **Chrome, Edge, Brave, Arc**. Firefox and
Safari are out of scope (different extension/debugger models). If the user is
on Firefox/Safari and inside cmux, the cmux panel is the fallback; otherwise
static-page reading via `bash`/curl is the honest limit.
