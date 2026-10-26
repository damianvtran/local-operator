# Desktop Team Bridge account avatar relay

The desktop may request an image using its **desktop transport authentication**:

```http
GET /v1/desktop/team-bridge/accounts/{target_account_id}/avatar?v=0123456789abcdef&size=64&expected_account_id={viewer_account_id}
```

The target and viewer must satisfy the shared account-ID grammar (including
Radient account UUIDs). Exactly three query parameters are required, each once:
`v` is 16 lowercase hexadecimal characters; `size` is exactly `64` or `256`;
`expected_account_id` is the signed-in viewer captured by the renderer. Extra,
missing, malformed and duplicate query arguments are rejected with 422 before
upstream dispatch. URLs are never accepted in place of account IDs.

Core resolves the viewer's Radient bearer from AuthStore and calls the fixed
upstream `/v1/accounts/{target_account_id}/avatar?v=...&size=...`. Membership and
image access remain upstream decisions; this route offers no account search or
enumeration and does not weaken upstream refusal. Redirects are not followed.
No Radient bearer, upstream headers, or upstream URL is returned to the renderer.

The required viewer receipt and the exact credential row are checked before and
after the read. AuthStore's durable `native_login_incarnation` is captured before
OAuth refresh, checked again before dispatch and after all image bytes arrive.
This distinguishes A→B→A re-login from unchanged A, and survives process boundaries.
Unknown incarnation fails closed. Ordinary token refresh does not change this
login identity. A stale viewer/row/incarnation returns 409 and no image bytes.
No available login follows the existing Team Bridge sign-in refusal contract.

Successful replies are nonempty PNG or JPEG with an inclusive **1 MiB** ceiling.
The existing bounded streaming transport stops an oversized answer without
buffering the rest. It requests identity encoding; avatar responses that ignore
that request are refused before decompression, preventing compressed-body
allocation from evading the limit. Non-200/non-image/empty replies fail closed.
Upstream error statuses use the existing refusal mapper. Access is refreshed
before dispatch as usual; this optional image read deliberately does **not**
force-refresh/retry on upstream 401 (the renderer can fall back to initials).

Downstream success headers:

- `Content-Type: image/png` or `image/jpeg`
- `X-Content-Type-Options: nosniff`
- `Content-Security-Policy: default-src 'none'; sandbox`
- `Cache-Control: private, no-store`

There is no core image cache, disk write, feature flag or deployment flag. The
renderer owns its bounded, viewer-scoped blob cache and fallback for old runtimes
that return 404. It must clear that cache on account transitions.

## Targeted evidence

`tests/unit/server/test_desktop_team_bridge_avatar.py` uses the existing real
loopback HTTP fixture: a desktop-authenticated Uvicorn route calls a separate
HTTP hub fixture through the production OAuth client and bounded transport.
It checks exact upstream URL, returned bytes/headers, query closure, desktop and
viewer auth, MIME/size refusals and upstream 404. This is relay proof, not a claim
about production Radient membership or the renderer.

`tests/unit/team_bridge/test_avatar.py` covers incremental streams, signout,
account switch, A→B→A, missing epoch, URL/path injection, redirects, encoded-body
rejection before reading, exact-limit PNG/JPEG, and actual AuthStore SQLite login
incarnation changes. The initial two tests failed on the absent `fetch_avatar`
method before implementation. Run these and the existing attachment/client tests
in an isolated HOME/config with the primary checkout's Python and worktree
`PYTHONPATH`; no dependency tree is installed in this worktree.

Observed on the implementation worktree (2026-10-26):

```text
python -m pytest tests/unit/team_bridge/test_avatar.py \
  tests/unit/team_bridge/test_client.py \
  tests/unit/server/test_desktop_team_bridge_avatar.py \
  tests/unit/server/test_desktop_team_bridge_attachments.py \
  -q -n 0 --no-cov --basetemp=<isolated scratch directory>
150 passed, 9 warnings in 23.29s
```

The loopback assertions observed `200 image/png` with exact image bytes,
`private, no-store` and `nosniff`; malformed/duplicate queries returned 422 with
zero hub calls; wrong viewer returned 409 with zero hub calls; bad desktop bearer
returned 401 with zero hub calls; empty/wrong-MIME/oversized upstream bodies
returned 502; upstream 404 remained 404. Existing Pydantic deprecation and
sync-test asyncio-marker warnings are unrelated. Cached offline Black, isort,
flake8 and `git diff --check` also passed. No production hub, browser, Electron,
or full-suite validation is claimed.
