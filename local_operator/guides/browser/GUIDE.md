---
name: browser
description: Drive the user's real browser via the Local Operator extension — setup, pairing, launching a closed browser, async site approvals, multi-tab surfaces, focus safety.
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

### 0. Make sure the paired browser is actually open

The most common "broken bridge" is simply that the paired browser is not
running — especially when it is not the user's daily browser (they live in
Arc, the extension is in Chrome). Check first, silently:

```sh
pgrep -x "Google Chrome"
```

If it is not running, ask the user before starting it the FIRST time (a
standing "yes, launch Chrome when you need it" is enough forever after),
then launch it backgrounded:

```sh
open -g -a "Google Chrome"
```

`open -g` launches it in the background without stealing focus or raising a
window — the extension connects to the daemon within seconds of the browser
starting. Never launch with
`--remote-debugging-port` or developer flags: the extension is the debugger,
and a debug-port browser on a real logged-in profile is an open door for any
local process.

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
systemd user unit on Linux, so it survives restarts). The daemon binds
**127.0.0.1 only** — never widen it. Once the extension is loaded and connected
you pair it (step 4); until then `install` just gets the daemon healthy.

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

### 4. Pair the extension to this machine — YOU run it and show the code

Pairing is **one-time**: once it succeeds, the extension stores the token and
the daemon stores its hash, and every later reconnect (daemon restart, browser
restart, reboot, a brand-new lop session) happens silently with no code and no
prompt. The user does this exactly once, ever.

Run the command yourself and read the code out of its output:

```sh
lop browser pair
```

It prints a 6-digit code (2-minute TTL, five tries then a fresh code). **You
MUST show that code to the user in your reply and ask them to type it into the
extension popup** (click the toolbar icon → enter the code). Do not just tell
them to "run `lop browser pair`": most users start from the TUI and never see
terminal output, so if you don't surface the code in the conversation they have
no way to get it. Showing it is correct and safe here — unlike a durable
password, this is a single-use code with a 2-minute TTL whose entire purpose is
to be shown to the user; it authorizes nothing on its own and expires in
minutes. (This is the one browser secret you DO print. Contrast the mobile
portal password, which must never enter the transcript.)

Concretely, say something like: *"Your one-time pairing code is **NNNNNN**.
Click the Local Operator extension icon in your browser toolbar and type it in.
This is the only time you'll need to do this."* If the code expires before they
enter it, just run `lop browser pair` again and show the fresh one.

The code flows terminal → browser (you → the user → the popup) on purpose: that
direction is the security property that stops a rogue local process from
pairing itself. Do not try to script the code into the popup on the user's
normal browser — you have no debug access to it, and that is intentional.

Re-pairing (new browser profile, wiped token, uninstalled extension): same
flow. `lop browser pair --reset` revokes the current pairing first.

### 5. Confirm and use

Re-run `lop browser status`; you want `extension connected: yes` and
`paired: yes`. The `browser` tool now appears — use it. From here on the user
does nothing: pairing persists across restarts, so future sessions just work.

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

## Per-site permissions (default-deny) — the three-step approval dance

The extension gates navigation by origin, enforced in the browser where no
local process can click for it. Access is denied by default, and the approval
flow is ASYNC and agent-legible — never a silent long block:

1. **`open`/`goto` to a not-yet-allowed origin fails in milliseconds** with a
   typed `origin_not_allowed` error naming the origin. No prompt exists yet.
   This fast failure is deliberate: it returns your turn so you can notify
   the user properly instead of timing out against an unseen popup.
2. **`request_access` (url)** appends a requester-bound approval to the extension
   popup's FIFO queue (badge count + one best-effort Chrome notification) and
   returns immediately with `state: pending`, `position`, `pending_count`, and
   `expires_at`. An identical live requester+origin request deduplicates without
   extending its ~10-minute lifetime. The queue holds 16 live requests; a typed
   `access_queue_full` response tells you to wait, approve, or cancel before retrying.
   **Then NOTIFY THE USER YOURSELF — this step is mandatory.** Use the `ask`
   tool or a message: the harness notification is the RELIABLE channel.
   Chrome's own notification frequently never reaches the user (macOS
   Notification Center authorization is commonly missing), and the popup
   badge is invisible unless they happen to click the toolbar icon. Tell
   them the origin and that the buttons are in the extension popup
   (toolbar icon → **Allow once** / **Always allow** / **Deny**).
3. **`await_access` (url, timeout_s — default 120, max 240)** waits for the
   decision and returns a terminal or pending state:
   - `allowed` — proceed; your next navigation to the origin succeeds.
   - `denied` — final; do not re-request unless the user changes their mind.
   - `pending` — still undecided; message the user again and call
     `await_access` again (the ~10-minute request survives between calls).
   - `cancelled` — this requester cancelled its own exact-origin request.
   - `superseded` — legacy receipt from a pre-queue extension; new queued
     requests are never superseded.
   - `none` — no live request for you (expired unanswered, or never raised).
     `request_access` again and re-notify the user.
4. **`cancel_access` (url)** removes only your own pending exact-origin request
   and returns `cancelled`; it never dismisses another session's request.

After `allowed`, the next `open`/`goto` to that origin succeeds. **Allow
once** mints a single-use grant bound to YOUR session — another session
cannot spend it, it covers exactly one navigation, and it expires if left
unspent for ~10 minutes (navigate promptly after approval). **Always allow**
persists forever (revocable in the extension's Settings → Allowed sites).

Other rules:

- `denied` is final for the task at hand: **do not retry or re-request the
  same origin** unless the user says they've changed their mind.
- A redirect INTO a new origin mid-navigation still prompts synchronously at
  that hop (a running command cannot fail early); if a navigation returns
  `origin_denied` after a redirect, explain which hop needed approval.
- If you know you will visit several sites, `request_access` them and tell
  the user up front in ONE message so they can approve in a batch.
- Subresources are not gated (this is a navigation gate, not a network filter).
- `http`/`https` only; the extension refuses `chrome://`, `file://`, etc.

## Using the tool

Same `browser` tool. Actions:

- `open` (start a surface at an http/https URL) · `goto` (navigate the surface)
- `read` (page text; `selector` scopes it) · `snapshot` (accessibility tree
  with click refs like `e5`)
- `screenshot` (writes a PNG; works on the background tab)
- `click` / `type` (by CSS selector or a snapshot ref; trusted input events)
- `scroll` (by direction, pixel delta, or scroll an element into view)
- `logs` (the tab's console output and uncaught exceptions since it opened —
  for debugging web apps)
- `tabs` (list every live extension-owned tab — see multi-tab below)
- `request_access` / `await_access` / `cancel_access` (queued site approval — see above)
- `close` (drop the surface and its tab)

Prefer `snapshot` to discover click targets (it returns stable refs), `read`
for content, `screenshot` for visual verification, `logs` when a page
misbehaves. Capture before/after screenshots for any visual change.

### Multi-tab: parallel sessions each own a tab

The extension drives MULTIPLE tabs concurrently (capped at 8), so parallel
agents/sessions never fight over one surface:

- **A fresh `open` (no existing surface) always creates a NEW background
  tab** — it never hijacks another session's tab. Your session then pins that
  surface and every later action drives it.
- **Your surface persists**: `open` again navigates YOUR tab (resume), it
  does not spawn more. One session = one tab unless you deliberately need
  more.
- **`tabs` lists all live extension-owned tabs** — URL, title, created/last
  used — with handles REDACTED (`bridge:123:abc…`). Listings are awareness
  only: you can see what other sessions are doing but cannot drive their tabs
  (driving requires the full handle only the owning session holds).
- **`close` when you are done.** Tabs you leave behind sit in the user's
  browser and count against the cap; a `tab_limit` error means the fleet
  should close finished tabs, not that the bridge is broken.
- cmux backend: `tabs`, `request_access`, `await_access`, `cancel_access`, `scroll`, and
  `logs` are extension-only; on cmux they return a typed not-supported error.

## Troubleshooting (what each error means)

Every failure is one actionable string; act on it rather than retrying blindly:

- **"extension not connected: the bridge daemon is running but no browser is
  attached"** — the paired browser is closed, the extension is disabled, or
  the MV3 service worker is suspended. Work through this checklist IN ORDER,
  and don't silently churn — after the first failed check, tell the user
  what's wrong and what you're doing about it:
  1. **Is the paired browser even running?** Check before asking the user
     anything: `pgrep -x "Google Chrome"` (or the browser they paired). This
     matters especially when the paired browser is NOT the user's primary one
     — a user who lives in Arc but paired Chrome will forget Chrome exists.
     If it isn't running, ask the user for permission to start it, then
     launch it BACKGROUNDED so it never steals focus:
     `open -g -a "Google Chrome"` (macOS). Never add `--remote-debugging-port`
     or other debug flags — the extension IS the debugger; a debug-port
     browser on a real profile is a security hole.
  2. **Browser running but still disconnected?** The service worker may be
     idle-suspended. A periodic reconnect alarm rewakes and reconnects it on
     its own — up to ~1 minute in a packed/released build (Chrome clamps
     sub-30s alarm periods; a developer/unpacked load fires sooner), but
     Chrome may delay alarms arbitrarily, so when it matters now open any
     page in that browser:
     `open -g -a "Google Chrome" "https://example.com"` — or ask the user to
     click the extension's toolbar icon (opening the popup wakes the worker
     instantly). Reconnection then happens within seconds.
  3. **Still nothing?** The extension may be disabled or removed — ask the
     user to check `chrome://extensions`.
- **"browser bridge not paired"** — run `lop browser pair` and have the user
  enter the code.
- **"browser bridge unreachable: the daemon … is not answering"** — run
  `lop browser status`; `lop browser install` (re)starts it. Logs:
  `lop browser logs`.
- **"site <origin> is not allowed yet" (`origin_not_allowed`)** — normal, not
  an error to fight: run the three-step approval dance (`request_access` →
  notify the user via `ask`/message → `await_access`). See the permissions
  section above.
- **"navigation to <origin> was denied"** — the user denied it (or a redirect
  hop's synchronous prompt expired). Do not retry the same origin unprompted;
  if it was a redirect hop, explain which origin needs approval.
- **"the browser bridge accepted '<method>' but did not answer within Ns…"**
  — the daemon is fine; the command is likely stuck in the browser, e.g.
  waiting on a site-permission decision. Point the user at the extension
  popup rather than restarting anything (the error text says exactly this).
- **"tab is gone" / "tab crashed"** — the user closed or crashed it; `open` the
  URL again to get a fresh surface.
- **"another debugger is attached"** — ask the user to close DevTools on that
  tab.
- **"The extensions gallery cannot be scripted"** — Chrome forbids the debugger
  API on the Chrome Web Store domains (`chrome.google.com/webstore`,
  `chromewebstore.google.com`) as a platform security rule, and once the
  extension's tab touches that domain the whole tab is poisoned for the session.
  This is not a bug and affects every CDP-based browser agent. You cannot
  automate the Web Store (install other extensions, fill the developer console,
  etc.) through the extension. Drive the user there and have them click, or use
  a separate tab for the rest of the task.

## Pages the extension cannot drive

The `chrome.debugger` API is blocked on a few origins by Chrome itself, so the
tool cannot navigate to or act on them:

- **The Chrome Web Store** (`chromewebstore.google.com`,
  `chrome.google.com/webstore`) — see the error above.
- **`chrome://` / `edge://` / browser-internal pages**, `file://`, and other
  non-http(s) schemes — the URL guard refuses these before they reach the
  browser.

For these, the human drives. Everything else (normal http/https sites,
including authenticated app dashboards, Google Workspace, GitHub, etc.) works.

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
