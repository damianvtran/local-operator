<!-- Moved verbatim from the project README (2026-11-08 README refresh); the README now links here. -->

# Drive Your Own Browser (Browser Extension)

The `browser` tool can drive a [cmux](https://cmux.com/) panel, a macOS
terminal built for running agents side by side. On any Chromium
browser (Chrome, Edge, Arc, Brave) the free **Local Operator browser extension**
gives the agent the same capability against the browser you already use, with
your real logins, entirely on your machine. A small loopback **bridge daemon**
connects the extension to your `lop` sessions; nothing leaves your computer, and
the agent can only open sites you approve.

**You can ask Local Operator to set this up for you**, or do it by hand:

```bash
lop browser install     # install and start the loopback bridge daemon
lop browser status      # daemon health, whether a browser is attached, pairing state
lop browser pair        # print the 6-digit code to type into the extension popup
```

Then install the extension from the
[Chrome Web Store](https://chromewebstore.google.com/detail/local-operator/omibaecbjdhgbbcedbnnnmjpmopfheof)
(or build it from `extension/` in this repo with `pnpm -C extension build` and
load `extension/dist` unpacked at `chrome://extensions` with Developer mode
on), click its toolbar icon, and enter the pairing code. Once paired, `browser`
tool calls drive a dedicated tab in your real browser. The first time the
agent wants a new site, the published store build asks **Allow
once**, **Always allow**, or **Deny**; you stay in control of which sites it
can reach, and can revoke the whole browser any time from the extension's
Settings or with `lop browser pair --reset`.

The bridge daemon binds **loopback only** and is pinned to your extension by the
pairing code; a compromised page cannot reach it, and a revoke drops any live
connection within seconds. Each store release is recorded in
[docs/store/release-record.md](../../docs/store/release-record.md).
