# Chrome Web Store privacy-tab declarations

Paste-ready copy for the Chrome Web Store dashboard. These declarations cover
the Manifest V3 permissions specified by docs/design/browser-extension.md
§9.4. Before submission, compare them with the **built** `manifest.json`; if
implementation changed the permission list, update this file rather than
submitting stale copy.

## Single purpose

> Connects the Local Operator app on this computer to this browser so its agent can browse on the user's behalf.

The extension has no independent chatbot, advertising, analytics, search,
shopping, or content-modification purpose. It provides the browser side of a
local connection to the free Local Operator app.

## Permission justifications

### `debugger`

> Local Operator uses `chrome.debugger` only on the single tab created for the user's agent. The permission is required to call Chrome DevTools Protocol methods that capture a screenshot, read the accessibility tree, resolve page elements, and dispatch trusted clicks and keystrokes. These are the extension's core browsing functions and cannot be provided with ordinary content scripts alone. The extension attaches only while handling a user-approved Local Operator action, detaches when the tab closes or the session ends, and leaves Chrome's debugging banner visible. Commands arrive as structured data from the user's own bridge daemon on `127.0.0.1`; the extension does not download or execute remote code, and no remote party controls the debugger connection.

Why this is reviewable: it names the exact CDP uses, the single delegated tab,
attach lifetime, user-visible infobar, local command source, and remote-code
boundary. Do not shorten it to “browser automation.”

### `tabs`

> Local Operator uses `chrome.tabs` to create, navigate, inspect, and close one tab dedicated to the user's agent. Reading that tab's URL and title lets the extension confirm the page actually reached after navigation or a redirect. The extension does not read, modify, or close the user's other tabs.

### `scripting`

> Local Operator uses `chrome.scripting.executeScript` in the agent-owned tab to return that page's rendered text content to the agent when the user asks it to read a page. The injected function is bundled with the extension, performs text extraction only, and neither loads remote code nor persists on the page.

### `storage`

> Local Operator uses `chrome.storage.local` to store the pairing token for the user's local bridge, an optional local port override, and the user's per-site allow or deny choices. It uses `chrome.storage.session` for the current agent tab handle, a nonce that prevents reuse of a stale tab ID, temporary element references, and a pending site prompt. This state stays in the user's browser profile and is not sent off the device.

### `alarms`

> Manifest V3 may suspend the extension service worker while it is idle. Local Operator uses one periodic alarm to wake the worker and reconnect it to the user's bridge daemon on `127.0.0.1` after suspension or a local restart. The alarm performs no background browsing, tracking, or remote network request.

### `webNavigation`

> Local Operator uses `chrome.webNavigation` events for the agent-owned tab to detect when a navigation completes or fails and to pause redirects that reach a site the user has not allowed. This replaces polling and ensures the agent reports the final live URL. It is not used to build browser history or observe navigation in the user's other tabs.

### `notifications`

> When the agent asks to open a site the user has not yet allowed, the extension raises a system notification naming the site so the user notices the pending decision even if the popup is closed. Without it, an approval the agent is blocked on could go unseen until it times out. Notifications are only shown for a pending site-permission decision; the extension does not send marketing or background notifications.

### Host permission: `<all_urls>`

> Users may ask their Local Operator agent to browse any HTTP or HTTPS site, so the extension cannot declare a fixed site list. Access is denied by default: before the agent-owned tab enters a new origin, the extension displays an in-browser prompt with Allow once, Always allow, and Deny choices, and it repeats the check for redirect destinations. `<all_urls>` is required for `chrome.scripting` and debugger-backed actions on the specific origins the user approves. The extension independently rejects non-HTTP(S) schemes and does not use this permission on the user's other tabs.

**Submission check (finding N2):** before uploading, diff this file against the
**built** `extension/dist/manifest.json`, not the source list. The built name is
`Local Operator`, and the permission set that must match here is
`debugger, tabs, scripting, storage, alarms, webNavigation, notifications` plus
host `<all_urls>`. Any implementation change to the manifest updates this file,
not the reverse.

**Implementation assumption to verify:** `<all_urls>` in Chrome includes schemes
beyond HTTP(S), while the design requires independent runtime rejection of
`chrome:`, `file:`, and `data:` URLs. If the implemented manifest can meet all
API requirements with `http://*/*` and `https://*/*` instead, narrow the host
patterns and update this heading; least-privilege scope is easier to review.

## Remote code declaration

Dashboard answer: **No, the extension does not use remote code.**

Paste-ready explanation:

> No remote code is loaded or executed. All JavaScript and TypeScript is compiled and packaged inside the submitted extension. The extension does not use `eval`, remote scripts, remote modules, WebAssembly downloaded at runtime, or executable strings. Agent instructions arrive as typed data over a WebSocket connection to the user's own bridge daemon on `127.0.0.1`. The extension maps those data messages to a fixed, bundled set of browser actions; neither instructions nor page content are treated as code. In particular, use of `chrome.debugger` executes only fixed Chrome DevTools Protocol methods selected by packaged extension code. It does not evaluate model-provided JavaScript.

This framing matters for `debugger`: an agent choosing among bundled commands
is data-driven behavior, not remote-code execution. The implementation must
preserve it. A generic CDP “evaluate arbitrary JavaScript” command exposed to
the bridge would invalidate this declaration.

## Data-use disclosure checklist

CWS dashboard wording changes occasionally; map the current checkboxes to the
actual behavior rather than copying labels blindly.

- **Website content:** disclose as handled. The extension reads page text,
  accessibility information, field values needed to verify typing, and
  screenshots from the user-approved agent tab.
- **Web history:** **assumption pending dashboard interpretation.** The
  extension reads the current and final URL of its own dedicated tab but does
  not collect or retain a user's browsing history. If CWS defines any URL
  access as “web history,” disclose it conservatively and explain this scope.
- **Authentication information:** the extension uses the browser's existing
  login session but does not read or collect passwords, session cookies, or
  auth tokens. Do not select this category unless CWS's current field guidance
  treats authenticated page content itself as authentication information.
- **User activity:** action results and page interactions are processed only
  to perform the requested agent task. If the current CWS form treats clicks
  and navigation as user activity even when generated in a delegated tab,
  disclose it conservatively.
- **Personally identifiable, health, financial/payment, personal
  communications, location:** the extension has broad site capability and may
  encounter these categories as website content. The implementation does not
  collect them as defined datasets. Complete the dashboard using Google's
  current definitions and describe all selected categories as local,
  task-scoped website content.

For every category selected, declare:

- data is used only for the extension's single purpose;
- data is not sold to third parties;
- data is not used or transferred for unrelated purposes;
- data is not used or transferred for creditworthiness or lending; and
- extension-originated traffic remains on loopback. The separate Local
  Operator app may send conversation text to the user's chosen model under
  the app's own policy.

## Reviewer test notes

Include these in the dashboard's reviewer-instructions field if offered:

> This extension requires the free Local Operator desktop/CLI app on the same computer. Start its browser bridge with `lop browser install`, then run `lop browser pair`; enter the displayed six-digit code in the extension popup. The extension connects only to `127.0.0.1` (default port 4099). Ask Local Operator to open an HTTP(S) URL. The extension will create one dedicated tab and prompt before entering an unapproved site. Choose Allow once. Chrome's debugger notice will appear while the extension drives that tab. Suggested review actions: open, read, snapshot, click, type, and screenshot. Contact [ASSUMPTION: insert monitored review-support email] if the review environment cannot complete pairing.

**Assumption:** final CLI commands and default port come from the approved
design. Confirm them against the shipped build before pasting.
