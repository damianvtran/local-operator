# Browser extension bridge: execution evidence

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
