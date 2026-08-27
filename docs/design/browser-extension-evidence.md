# Browser extension bridge: execution evidence

## Loopback all-port grants and visible authorities

Captured from the built extension v0.1.4 loaded unpacked into throwaway Google
Chrome through CDP `Extensions.loadUnpacked`. Chrome ran off-screen in a fresh
profile and was never focused. The rendered frames were inspected after capture:

- `popup-origin-prompt.png`: ordinary `example.com` retains the existing three
  controls and exact-origin **Always allow** choice.
- `popup-loopback-all-ports.png`: `127.0.0.1:5173` is visible in the SITE trough;
  **Always this port** stays beside **Allow once**, while the broader explicit
  **Always all ports** action has its own secondary row.
- `popup-loopback-ipv6.png`: `[::1]:8000` uses the URL parser's bracketed
  authority and the same dedicated all-port row.
- `options-empty.png` and `options-populated.png`: the empty state remains clear;
  populated Settings independently labels and revokes `http://localhost · all
  ports` and `http://localhost:5173 · this port`.

The before-frame on current `origin/main` was the prior
`popup-origin-prompt.png`: the SITE trough hid nondefault ports because it used
`URL.hostname`, and the only standing action read **Always allow**. The new
ordinary-site frame preserves its layout while loopback prompts add the broader
choice explicitly. No manifest permission or privacy category changed.

## Round 3 remediation: origin-prompt host quarantined + reconnect hardening

Design round 3 flagged one MAJOR (D11) and a nit (D12); code round 3 was clean
with two minors (A11, A12). All four fixed.

- **D11 (MAJOR):** the per-site allow prompt no longer renders the host in the
  serif display heading. The heading is fixed prose ("Let the agent use this
  site?") and the origin is quarantined to a monospace `.trough` (`--sunken`,
  1px `--hairline-strong`, radius 2px, `overflow-wrap: anywhere`) under a "SITE"
  label — the same treatment `callback_page.py` gives every URL/host and the
  same trough the connected-state driven URL already used. Re-captured
  `popup-origin-prompt.png` / `-dark.png` with a long host
  (`accounts.corp.internal.staging.example.com`): it wraps at a token boundary
  inside the trough (reads as a URL), the heading stays prose, no mid-word break.
- **D12 (nit):** `white-space: nowrap` on `.sub code` keeps `lop browser status`
  a single unbroken chip; re-captured `popup-disconnected.png` / `-dark.png`.
- **A11 (minor):** the service-worker dial is wrapped in try/catch so a
  `chrome.storage` await rejection resets `connecting=false` and reschedules the
  reconnect instead of wedging the guard.
- **A12 (minor):** an explicit 10s dial timeout force-closes a socket that never
  fires `onopen`/`onerror`/`onclose`, cleared on any real settle, so a dead
  handshake can no longer deadlock the connecting state.

Frames re-captured from the real extension in Chrome 151 (CDP `loadUnpacked`)
against the live daemon, both ramps; the allow-site store screenshot was
regenerated from the new origin-prompt frame.

## Design-language pass: popup + options redrawn (real Chrome, both ramps)

The popup and options pages were redrawn in Local Operator's own design system —
the same one the OAuth/MCP auth-callback page uses (`local_operator/
callback_page.py`): warm paper, one accent spent only on real semantics,
hairlines instead of shadows, an old-style serif display over a humanist sans,
a monospace label/wordmark register, and a full dark ramp. The token blocks
(light `:root` and `prefers-color-scheme: dark`) are copied verbatim with their
names kept; the header identity is the shared inline SVG mark stroked in
`--ink`, never state-tinted; the 2px top status rule is `--success` for
connected, `--danger` for error/incompatible, and a neutral hairline for the
transitional states.

Every state was re-captured from the real extension loaded into Chrome 151 (via
CDP `Extensions.loadUnpacked`) against the live daemon, in BOTH ramps, under
`docs/evidence/browser-extension/` (`*-dark.png` for the dark variants):
`popup-connected` (with the live driven URL in a monospace trough),
`popup-origin-prompt` (long hostname wraps inside 300px — D1/N4 preserved),
`popup-pairing`, `popup-pairing-error` (danger rule + `role="alert"` — D7),
`popup-disconnected` (neutral rule, `lop browser status` chip — U4),
`popup-incompatible` (danger rule, update-needed copy — D2),
`options-empty`, `options-populated` (masked dot field, live pairing status,
styled Remove actions, long origins wrapping — D8). The store screenshots and
promo tiles were regenerated from the new frames with
`docs/store/assets/build_assets.py`. Both ramps were eyeballed frame by frame.

**Reconnect-storm bug found and fixed during capture.** Driving the real
extension surfaced a genuine defect: a `chrome.alarms` tick firing in the window
between `new WebSocket()` and its `onopen` started a second socket, the daemon's
"later-connection-wins" rule closed the first, and the resulting teardown →
reconnect cascaded into a tight loop (measured 79 WS accepts in seconds, and
commands hung because each RPC raced the churn). Fixed with a synchronous
`connecting` guard in the service worker so only one dial is ever in flight;
after the fix the same sequence converges to a single stable socket (1–2
accepts) and all eight actions drive cleanly again.

## Round 1 remediation: real Chrome, real extension, real daemon (A9/D4 closed)

The round-1 reviews flagged that the earlier `--load-extension` was ignored by
recent Chrome. Resolved this round: the built `extension/dist` was loaded into a
real Google Chrome 151 via the CDP `Extensions.loadUnpacked` command (the
`--load-extension` switch is removed in current Chrome; the
`--disable-features=DisableLoadExtensionCommandLineSwitch` flag no longer exists
either, so CDP is the working route on this host). The extension's MV3 service
worker connected to a real bridge daemon on loopback, paired with the real
6-digit code, and drove a real tab.

**Connection proof** (`GET /health`, live):

```json
{"status":"ok","proto":1,"extension_connected":true,"paired":true,
 "browser":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36"}
```

### All eight actions exercised end to end through the extension

Against a local test page, every browser action ran through the real RPC ->
WS -> extension -> CDP path and returned the real result (full transcript:
`/tmp/lop-actions-evidence.txt`, summarized):

| action | request | real response |
|---|---|---|
| `open` | `http://127.0.0.1:8791/lop-testpage.html` | `ok`, `tab bridge:1512442283:833e…`, title `Bridge Test Page` |
| `read` | full page | `"Local Operator bridge test\nHello from the test page.\nGo to page two\n\nClick me"` |
| `snapshot` | — | `- RootWebArea "Bridge Test Page" [e1]` |
| `type` | `#box` = `hello bridge` | `value: "hello bridge"` (A2 fix: read-back correct, no false failure) |
| `click` | `#btn` (no nav) | `navigated: false`; a following `read #para` returned **`"clicked"`** — the onclick side effect fired |
| `click` | `#link` (nav) | `navigated: true`, url advanced to `…/page2.html` (A7 fix: navigation detected) |
| `goto` | `…/lop-page2.html` | `url …/lop-page2.html`, title `Page Two` |
| `screenshot` | — | 29,339-byte PNG of Page Two (`docs/evidence/browser-extension/action-screenshot-page2.png`) |
| `status` | — | live `url`/`title`, `origin_mode: default-deny` |
| `close` | — | `ok`, tab removed |

**Background-tab input defect found and fixed (A9/A7).** Real-Chrome testing
revealed that current Chrome drops CDP `Input.dispatchMouseEvent` on a hidden
(inactive) tab entirely — the agent tab is intentionally inactive, and the
compositor never delivered the synthetic press, so `#btn`'s onclick did not fire
and `#link` did not navigate. Confirmed directly: `document.hidden === true`, a
native `.click()` set `#para` to `clicked`, but a CDP mouse sequence at the
button's center left it unchanged. Fixed by driving click and type on the
resolved DOM node through the debugger's own `Runtime.callFunctionOn` (a full
pointer/mouse event sequence and value-setter + input/change events), which
fires handlers and default actions reliably on a background tab. Re-verified
above: `#btn` → `clicked`, `#link` → `navigated: true`. This is exactly the class
of defect the reviewers predicted a real run would catch.

### Rendered popup and options states (real extension, live daemon)

Captured from the actual extension pages backed by the live daemon (not static
HTML), under `docs/evidence/browser-extension/`:

- `popup-connected.png` — **Connected.**, and **“Driving: Page Two — http://127.0.0.1:8791/lop-page2.html”** (U3: the driven site is shown; N4: the URL wraps within 300 px).
- `popup-origin-prompt.png` — the three consent choices **Allow once / Always allow / Deny** as one coherent decision, fitting the 300 px popup even with the long hostname `accounts.corp.internal.example.com` (D1; D5/N3).
- `popup-incompatible.png` — the **Update needed.** state with product-level recovery copy (D2).
- `popup-pairing.png`, `popup-pairing-error.png` — pairing entry and the live-region error (D7/U8).
- `popup-disconnected.png` — the corrected “Local Operator isn't reachable… (`lop browser status`)” copy (U4).
- `options-empty.png`, `options-populated.png` — the settings page showing live pairing status, the empty-state line, and long origins wrapping with the Remove action intact (D8).
- `icons-on-light.png`, `icons-on-dark.png` — all four icon sizes on light and dark chrome, now legible on dark surfaces via the rounded backplate (D3).

### Security fixes verified against the running daemon

- **A1** (pairing lockout): 5 wrong guesses rotate the code and the original stops working — covered by `test_a1_pairing_locks_out_and_rotates_after_max_attempts`, reproducing the reviewer's scenario.
- **A5/U1** (revoke severs live session): an out-of-process `reset_pairing` makes the very next authenticated RPC fail `not_paired` and flips `link.paired` false; the options-page `unpair` event drops the live socket. Covered by `test_a5_u1_out_of_process_revoke_severs_live_rpc` and `test_a5_u1_unpair_event_drops_live_socket`.
- **A3** (approval budget): a command blocked on a human decision extends its deadline past the base command timeout while `awaiting_origin` holds, and still times out otherwise. Covered by `test_a3_wait_extends_while_awaiting_origin` / `test_a3_wait_times_out_without_awaiting_flag`.

---

# Browser extension bridge: execution evidence (round 0)

Captured on macOS with Local Operator 0.34.0 source from `dev-webbridge`.
Secrets below are redacted; the actual requests used the daemon-generated key.

## Extension build

```console
$ cd extension && pnpm typecheck && pnpm test && pnpm build
✔ origin policy only permits stored HTTP origins
✔ AX compaction assigns epoch-scoped click refs
ℹ pass 2
/private/tmp/lop-webbridge/extension/dist
```

The built manifest, worker, popup, options page, source maps and all four PNG
icons were present under `extension/dist/`. Google Chrome 151 was started with
`--load-extension=/tmp/lop-webbridge/extension/dist` and a fresh temporary
profile. Chrome itself started and exposed CDP (`Chrome/151.0.7922.174`), but
its current automation policy did not register the command-line-loaded unpacked
extension in that fresh profile. No claim is made that extension APIs or browser actions were manually exercised
in Chrome in this run. The built popup was served as a static page and inspected
in a real browser. The first frame exposed a blank-state defect because every
section started hidden; after fixing the disconnected state as the safe first
paint, `/tmp/lop-webbridge-popup-after.png` showed the icon, “Not connected.”,
a reconnect instruction, Retry, and Settings without reflow. This validates the
rendered disconnected frame, not Chrome extension API integration. The extension
build, TypeScript gate, and pure origin/AX tests also ran.

## Real daemon, HTTP authorization, and invalid input

Started with:

```console
$ LOCAL_OPERATOR_CONFIG_DIR=/tmp/lop-bridge-evidence \
    .venv/bin/python -m local_operator.browser_bridge.daemon --port 4109
```

Observed responses:

```console
$ curl http://127.0.0.1:4109/health
{"status":"ok","proto":1,"extension_connected":false,"paired":false,"browser":""}

$ POST /rpc with bad token
{"error":"unauthorized"}
HTTP 401

$ POST /rpc with valid key but missing request id
{"error":"invalid_request","detail":"1 validation error for Request ... Field required ..."}
HTTP 422

$ POST /rpc while extension disconnected
{"id":"r-open","ok":false,"error":{"code":"extension_disconnected","message":"extension not connected","data":{}}}
HTTP 200
```

## Real WebSocket handshake and pairing

A real `websockets` client connected with Origin
`chrome-extension://aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa`; the 6-digit code was
read through the same safe `pairing_status()` path used by `lop browser pair`.
The issued long-lived token was redacted before capture.

```console
$ extension hello
{"event":"hello_ack","proto":1,"paired":false}

$ extension pairing response
{"event":"pair_result","ok":true,"token":"<redacted>","message":""}
```

The daemon stored only the SHA-256 token digest and the pinned extension id.

## Unpaired and denied-origin failure paths

With no saved pairing, a connected extension and an authenticated local RPC
caller produced:

```console
$ unpaired extension hello
{"event":"hello_ack","proto":1,"paired":false}

$ RPC while unpaired
{"id":"r-unpaired","ok":false,"error":{"code":"not_paired","message":"extension is not paired","data":{}}}
```

After pairing, the daemon relayed a real RPC command over the live WebSocket.
The client returned the extension's denied-origin wire response:

```console
$ daemon relayed command
{"id":"r-deny","method":"goto","params":{"tab":"bridge:1:test","url":"https://bank.example"}}

$ RPC origin denied response
{"id":"r-deny","ok":false,"error":{"code":"origin_denied","message":"site permission was denied","data":{"origin":"https://bank.example"}}}
```

The TypeScript extension enforces this before main-document network requests by
pausing `Document` requests through CDP `Fetch.requestPaused`; redirects and
click navigation use the same gate.

## Daemon-down failure path

A fresh state file pointing at a closed loopback port exercised the actual
session client:

```console
$ bridge client with daemon down
browser bridge unreachable: the daemon at 127.0.0.1:4199 is not answering. Run 'lop browser status'; 'lop browser install' starts it.
```

## Quality gates

The branch was checked with the repository's complete gate commands:

```console
$ uvx --from flake8==7.1.0 flake8 .
$ uvx --from black==26.1.0 black --check .
524 files would be left unchanged.
$ uvx isort==5.13.2 --check .
$ .venv/bin/python -m pyright --pythonpath .venv/bin/python .
0 errors, 0 warnings, 0 informations
$ env -u NO_COLOR TERM=xterm-256color .venv/bin/python -m pytest tests/unit -q
6755 passed, 2 failed, 7 skipped
```

Both failures were existing timing-sensitive tests under the full xdist load,
not browser-bridge paths: conversation-name persistence and mobile registrant
discovery. Re-running those exact two node IDs immediately in isolation passed
`2 passed`. A focused final regression suite covering the bridge, all existing
browser-tool tests, CLI style, and import boundaries passed `145 passed`.
A serial full-suite rerun to remove xdist scheduling pressure also finished
`6755 passed, 2 failed, 7 skipped`, but with two different unrelated failures:
background bash process-group cancellation and subagent-view scroll anchoring.
Those exact two node IDs immediately passed in isolation (`2 passed`). Across
both full runs the browser bridge tests remained green; the changing failures
are recorded honestly as pre-existing timing-sensitive full-suite instability.
