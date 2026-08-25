# Browser extension: live end-to-end validation

Real-Chrome, real-daemon exercise of the Local Operator browser extension and
every `browser` tool action, including the two new ones (`scroll`, `logs`).
This is the "prove the real path" evidence for the guidance/precedence change
and the scroll/logs capability. It is a manual validation record, not a CI
gate; the automated suites cover the units.

## Setup exercised

- Chrome 151.0.7922.174 (a Chromium browser), dedicated profile.
- `lop browser install` → daemon healthy on `127.0.0.1:4099`.
- Extension loaded unpacked via CDP `Extensions.loadUnpacked` (Chrome 151 no
  longer honours `--load-extension`; documented in the design doc's rollout
  risks). Service worker `chrome-extension://…/worker.js` connected.
- Paired by typing the `lop browser pair` code into the **real popup form**
  (`#pair-code` + real submit) → `paired: yes`, `extension connected: yes`.

## Actions driven through `BridgeClient` (the exact path the `browser` tool uses)

All 14 checks passed against the live extension + a local test page:

| Check | Result |
|---|---|
| open + **real default-deny consent** (clicked the popup's "Always allow") | surface `bridge:…`, title "LOP E2E Test" |
| read (DOM text) | 94 chars, page body returned |
| snapshot (accessibility tree, click refs) | `- RootWebArea "LOP E2E Test" [e1]` |
| screenshot | real 1440×813 PNG (magic-byte verified, 17 KB) on a **background** tab |
| click `#btn` | DOM mutated: `#para` → "clicked!" |
| type `#field` "hello lop" | field value read back independently = "hello lop" |
| scroll direction=bottom | `scrollY=2463, moreBelow=false` (reached end) |
| scroll selector=`#top` | `scrollY=0` (element scrolled back into view) |
| logs (all) | 7 entries — log/info/warning/error console output, the per-keystroke input logs, AND the uncaught `Error: boom uncaught` (source=exception) |
| logs level=error | 2 entries, only the deliberate error + the exception |
| goto page two | title "Page Two" |
| read after goto | "Page Two Loaded" |
| screenshot page two | real 13 KB PNG |
| close | surface + tab dropped |

## Focus safety (the standing requirement) — verified, not asserted

With a user tab active in front, the agent opened a new tab, navigated it, and
screenshotted it. Checked via `chrome.tabs.query({active:true})`: the agent's
tab was **never the active tab** — it stayed in the background throughout.
Confirms the code guarantees (`chrome.tabs.create({active:false})`, no
`activate`/`windows.update`/`captureVisibleTab`, CDP `Page.captureScreenshot`
which works headless-of-tab).

## Persistence and consent

- After "Always allow", the origin was persisted in `chrome.storage.local`
  (`{ "http://127.0.0.1:8799": "allow" }`); a second open did not re-prompt.
- Clearing the allowlist re-armed the default-deny prompt on the next open —
  the gate is genuinely default-deny, enforced in the extension.

## New actions — wire shapes (for the guide and the tool schema)

- **scroll** — precedence `selector` > `x`/`y` deltas > `direction`
  (`top|bottom|up|down|left|right`) > default one viewport down. Returns
  `scrollX`, `scrollY`, `moreBelow`, `moreRight` so the agent knows if content
  remains. Drives the background tab; never activates it.
- **logs** — returns buffered console output + uncaught exceptions since the
  surface opened, newest-last, each with `level`, `text`, `source`, `url`,
  `line`, `timestamp`. Optional `level` (`error|warning|info|log|all`) and
  `limit`. Ring-buffered in the extension (cap 200 entries / 256 KB).

Both are bridge-only; on a cmux surface they return an actionable
"not supported on the cmux backend — use the Local Operator browser extension".

Captured frames: `docs/evidence/browser-extension/e2e-01-page-top.png`,
`e2e-02-page-two.png`.
