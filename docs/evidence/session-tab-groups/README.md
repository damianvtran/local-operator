# Session tab groups evidence

Captured 2026-08-27 from the built 0.1.4 extension loaded with
`Extensions.loadUnpacked` into a temporary Google Chrome profile. The bridge ran
on port 4199 with `LOCAL_OPERATOR_CONFIG_DIR=/tmp/lop-tab-groups-e2e/config`, so
neither the paired browser daemon nor its state was touched.

`e2e-output.txt` records real bridge responses for five separately identified
sessions, duplicate labels, rename, explicit resume, close/prune, and the
redacted multi-tab listing. `native-state.json` is Chrome's own `tabs.query`,
`tabGroups.get`, and `windows.getAll` response after those calls. It proves the
cyan native groups and titles, inactive tabs, and `focused: false` window.

A native browser-chrome screenshot was attempted with non-activating
`screencapture -x` after moving the test window via `chrome.windows.update`
without changing `focused:false`. macOS denied Screen Recording to this process:
`could not create image from display`. CoreGraphics window enumeration was also
empty for the same privacy gate. No synthetic tab-strip image was substituted.
The JSON is retained as truthful rendered-surface state; a still requires the
operator to grant Screen Recording and rerun the capture.
