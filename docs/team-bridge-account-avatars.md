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

## HTTP revalidation and memory layers

The renderer owns its bounded, viewer-scoped blob cache and fallback for old
runtimes returning 404; it clears/revokes blob URLs on account transitions.
Downstream HTTP remains `private, no-store` because browser/disk caches cannot
perform the core login-incarnation fence.

Core separately retains **HTTP entities** (bytes, MIME and upstream ETag) in a
process-wide LRU, bounded globally to **64 entries and 8 MiB of image bytes**.
It survives renderer recreation and the route's per-request client instances.
Core process restart is a cold cache; there are no disk writes. Keys include
resolved configuration root, upstream origin/base, credential-row ID, viewer,
durable login incarnation, target, version hash and size. No bearer is retained.

**Freshness lifetime is zero, with mandatory revalidation and no stale-if-error.**
Every cache hit still resolves current credentials and sends an upstream GET
with `If-None-Match`. The hub must perform membership authorization before
answering 304. Only that admitted request's cached body may satisfy 304, with an
absent or matching ETag, and only after the post-read owner/incarnation fence.
Unsolicited 304 or a changed ETag is refused with 502. This preserves membership
revocation while eliminating repeat image transfer across renderer cache loss.
The desktop receives a normal 200 image from validated cached bytes, not 304.

A nonempty valid 200 with a bounded syntactically valid ETag replaces the entry;
missing/invalid ETags cold-fetch next time. Upstream `Cache-Control: no-store`
prevents retention. HTTP refusals, transport failures, invalid representations
and invalid 304s evict the affected entry, never serve stale. LRU eviction forces
an unconditional GET. A changed/missing owner purges that configuration/upstream
scope when observed. After an unobserved signout, physical bytes can remain until
observation or bounded eviction, but the pre/post incarnation admission makes
them inaccessible: A→B→A receives a new login incarnation and cannot reuse the
old validator or body. This is not a claim of synchronous physical core purge.

There are no feature or deployment flags. Upstream ETag/conditional-GET support
is required for transfer savings; without it the relay safely cold-fetches.

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

### R1 HTTP-cache remediation evidence

The real loopback test first failed before remediation: upstream statuses were
`[200, 200]` instead of `[200, 304]`. After remediation it observed
`If-None-Match: "avatar-one"` on the second upstream request, statuses
`[200, 304]`, and identical PNG bytes in both downstream 200 responses. Each
request constructs a new core client, proving retention beyond renderer/client
memory. Additional real HTTP cases cover upstream refusal eviction, A→B→A,
observed signout, entry eviction, unsolicited 304, changed ETag, and signout or
re-login while a conditional response is in flight. Store tests cover the global
byte ceiling, replacement accounting and LRU ordering; client tests cover
transport-failure eviction, missing/invalid ETag and upstream no-store.

The same isolated serial command above, adding
`tests/unit/team_bridge/test_avatar_cache.py`, produced:

```text
166 passed, 9 warnings in 28.91s
```

Cached offline Black/isort/flake8 and `git diff --check` passed. These are relay
and HTTP-fixture observations, not evidence of production-hub ETag support.
