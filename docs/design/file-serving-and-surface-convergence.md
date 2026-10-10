# Design: one file server for every surface — unclamped on loopback, phased auth, surface convergence

Status: proposal v2 (architect) — manager rulings of 2026-10-10 folded in (see §11). **Proposed final path: `docs/design/file-serving-and-surface-convergence.md`.**
Scope: core (server app, static routes, CLI, registry, settings), local-operator-ui (bridge, previews), mobile relay, docs. No `pyproject.toml` bump — the release owner handles that.
Base: core `origin/main` @ `f8bc3ba3ba` (v0.68.24), UI `origin/main` @ `b53efe973ea`. Every file:line below was read at those refs, not recalled. Claims that are inference rather than reading are marked **[spec]**.

**The operator's direction (fixed requirement, quoted).** *"I'm not sure if clamping down to a server's served roots is the right way here, the server needs to be able to serve from downloads, documents, etc. Because the TUI for example would access through that server. If we're just going to be accessing local files anyways, it doesn't make sense to lock that down. As long as we host on 127.0.0.1 and not 0.0.0.0 and the server is not made discoverable outside of the local device and any explicit tunnels, it's ok to have the server be able to serve. If needs be, we can have some sort of auth schema where surfaces generate a key … completely abstracted and automatic and not require the user to do anything, and shouldn't block in these next few releases … it's better to find a security solution that is general (and backward compatible) where all surfaces access through the same file server instead of redundantly implementing their own filesystem access … Worth noting that any public surfaces like the mobile relay and mobile app ONLY use the app over an authenticated tunnel in the first place."*

**What this RFC supersedes (provenance).** Core #2134 (`8c698ff7ba1c`, v0.68.23) shipped the served-roots clamp as the bound on an unauthenticated route; its *rationale* is replaced by the loopback condition below. Its hardening (response policy, Host check, uniform refusals, regular-file predicate) is carried forward and becomes MORE load-bearing. UI #942 (`bce26ad89777`, v0.34.2) stays as the interim desktop reading path; the server becomes the single semantic home going forward. The five follow-ups recorded on #2134 are dispositioned in §7.

## 1. The problem as found in the code

### 1.1 The file routes today

`/v1/static/{images,videos,audio,html}` are the only routes that serve user files (`local_operator/server/routes/static.py:110-319`; mime allowlists `:26-76`). They are **unauthenticated and outside every gate**: not in `_LEGACY_CONTROL_PATHS` (`server/app.py:602-616`), not under `_LEGACY_GATED_PREFIXES` (`app.py:639`), not in the desktop plane's sensitive prefixes (`app.py:751`). Since #2134 the only bound is the served-root allowlist in `server/utils/static_roots.py` — `resolve_servable` (`:427-479`) requires `path` to resolve inside one of: agent home, `<config>/sessions`, `<config>/uploads`, a live session's cwd below `$HOME`, or explicit `static.roots` / `LOCAL_OPERATOR_STATIC_ROOTS` (`build_roots`, `:374-392`).

That bound is wrong under the operator's direction: `~/Downloads`, `~/Documents`, `/tmp`, `~/Pictures`, agent output anywhere — none of it is servable unless the operator edits `static.roots`, which is exactly the "suddenly stop working" class the operator wants removed. The served-roots clamp's own docstring states the user-facing cost in full (`static_roots.py:94-104`).

### 1.2 The exposure that justified it, and what actually contains it

The probe points that need to hold for an unclamped route:

- **Loopback-only hosting**: `lop serve` defaults to `127.0.0.1:1111` (`cli.py:685-702`) but `--host` accepts *any* address, including `0.0.0.0`, with only a help-text warning (`cli.py:688-696`). A wildcard bind also **disables** the static Host check by design (`static_roots.py:525-529`; QA measured: "bind `--host 0.0.0.0` + rebinding Host → 200 — wildcard bind disables the check").
- **DNS rebinding**: a rebound page is same-origin with the daemon, so no CORS grant is needed; the one thing it cannot forge is the `Host` header. The check (`host_is_acceptable`, `static_roots.py:506-546`, wired at `app.py:976-985`) refuses every DNS name except `localhost`/the announced bind; IP literals pass (a rebinding page cannot present one). **With the roots gone, this check is the sole control that stands between a rebinding page and whole-disk reads.**
- **Cross-origin pages**: the static middleware strips the CORS grant for non-admitted origins on every static response (`app.py:986-994`), so a foreign page cannot read a response body; it can still trigger requests and observe image existence/dimensions via `<img>` (`static_roots.py:22-34`, S-4b).
- **Same-user processes**: can read any file directly; no route rule changes that. Not a boundary to defend.
- **Specials**: directories, FIFOs, sockets, devices are refused by the regular-file predicate (`static_roots.py:467-476`) — keep; a FIFO read hangs a worker.

### 1.3 The surfaces, and their divergent file paths today (summary; full audit in the appendix)

| Surface | Reads files today via | Divergence already visible |
|---|---|---|
| TUI | in-process Python (`tui/app.py:35873-35923`) | none vs itself; not reachable remotely for files |
| UI renderer | IPC bridge `read-file-bytes`/`probe-files` (`UI src/main/index.ts:2684-2799`), blob URLs; `video-preview` route-first (`video-preview.tsx:117`); `html-preview` route-only (`html-preview.tsx:123`) | 64 MiB IPC cap vs unlimited HTTP; HEIC converted on the route, typed `image/heic` over IPC (the #942 R6/D1 glyph class); mtime-keyed blobs vs per-request reads |
| Mobile portal | transcript-store images only (`mobile/daemon.py:5295-5331`); no general file route | the phone cannot see the session's other files at all |
| Mesh peers | session text only; attachments deliberately degrade (`network/relay.py:7230-7235`) | by design |

## 2. Decisions (the shape of the change)

**D1 — The general file server is unclamped, and its predicate is: any absolute path that resolves to a readable regular file, with the per-route mime allowlist unchanged.**
No root allowlist in the default mode. Keep: raw `..` refusal, `expanduser`/resolve failures → one uniform 403, missing → 404, non-regular → 400, unreadable → 403 (`static_roots.py:435-479` structure). **Drop the dot-component refusal in this mode** (§3.3 says why; the code keeps it for rooted mode). **[spec]** Hardlinks, symlinks and case-insensitivity stop being boundary questions; symlinks stay followed exactly as today.

**D2 — "Loopback" is enforced twice: the bind is refused, and the predicate itself is per-connection gated.**
(a) `lop serve` refuses **every** non-loopback bind — `0.0.0.0`, `::`, LAN/other non-loopback IPs, and names that do not resolve exclusively to loopback — with exit 1 and **no override path** (Manager ruling 2026-10-10, strict). (b) Independently of (a), the static family applies the **unclamped predicate only when the connection's local socket address is loopback** (`request.scope["server"]` — the address the kernel actually accepted on). Everything else gets the *rooted* predicate (the v0.68.23 behaviour: `static.roots` + built-ins). So the invariant is not "we asked people not to bind wide"; it is **"unclamped reads are only ever answered on a loopback-accepted connection"** — which an SSH port-forward or a local tunnel deliberately provides, and a LAN client never does.

**D3 — The Host check stays and is promoted to the load-bearing rebinding control on these routes; the wildcard-bind bypass is DELETED and a wildcard/empty announced host is fail-closed (IP literals and `localhost` only; DNS names refused).** Parser hardening (`:520-536`) kept. Extending the check beyond `/v1/static/*` (follow-up d) is §3.6.

**D4 — The response policy is untouched**: CSP (media/HTML), `frame-ancestors`, `nosniff`, CORS-grant strip — every static response, errors included (`static_roots.py:152-196, 482-490, 554-565`; `app.py:941-994`).

**D5 — Route families are explicit, because the clamp's removal must not widen what an executing document can reach** (supplements lane constraint, verbatim requirement): *"a DISTINCT route family for executable generated documents, not reuse of the general file route"* and *"nothing we ship weakens the sandbox on a page that executes."* The family map:
   - `/v1/static/images|videos|audio` — bytes embedded or streamed by the app (media CSP). **These never serve `text/html` or any document type** (mime allowlists already enforce it; pinned by test).
   - `/v1/static/html` — **the only family that serves user documents that can run script**, with the document CSP + `frame-ancestors` and the UI-side `sandbox="allow-scripts"` unchanged. It is not "a permissive rule under the general route": it keeps its family, its policy, and its `MEDIA_CSP`-vs-`HTML_CSP` selection (`response_policy`, `:554-565`).
   - **Documents-that-execute for supplements (C2 lane) ride their own family only** — content-addressed blobs served from the store (digest-addressed, no caller path), carrying the §4.1 document policy, `nosniff`, `Cache-Control: no-store`, as their design already requires. This RFC changes none of that; the general media family cannot reach an executing frame's capability set because it serves no executable document type at all.
   - No new "general bytes" route ships in P1. If a later phase adds one, it serves `application/octet-stream` + `Content-Disposition: attachment` + `nosniff`, and §4's token gate — decided in that phase, not now.

**D6 — Phased, abstracted auth that is permissive first** (§4): a per-boot token published only in the daemon's `0600` serve record (the `claim_key` precedent, `server/desktop.py:10-24`), accepted as a header or a signed query URL, **not required for several releases**; enforcement becomes opt-in, then default, gated on the compat matrix, never on user action.

**D7 — Surfaces converge on this server, not the reverse** (§6): the UI IPC bridge is kept for now (it is the shipped fix for real breakage and the daemon-down path) but loses source-of-truth status — new file features go through the server, bridge changes are maintenance-only, and its retirement/narrowing per surface is scheduled after P2. `html-preview` keeps the route permanently (route+CSP pairing, `html-preview.tsx:44-57`). The mobile relay gains a proxy path so one session shows the same files on the phone.

**D8 — The mesh relay is out of this condition** (§3.5): it is not a file server (no file bytes flow; attachments degrade, `relay.py:7230-7235`), and its wide default is the feature's topology requirement (`relay.py:56-71`). It must stay a distinct story in docs so "loopback-only" is never read as fleet-wide.

## 3. The loopback condition — enforcement design

### 3.1 Bind refusal (CLI) — strict, no override (Manager ruling 2026-10-10)

- `cli.py` `serve_command` (`:9534`) gains a bind-policy check before `_bind_serve_socket` (`:9406`):
  - **Accepted**: `127.0.0.1`, `::1`, `localhost`, and only names/addresses that resolve exclusively to loopback. The exact predicate set is pinned in code review and tests (both address families).
  - **Refused, exit 1**: `0.0.0.0`, `::`, any LAN/non-loopback IP, and any non-loopback-resolving name — **with no override flag**. The `_refuse_serve_bind`-shaped message (`:9472-9490`) names the address and the remedy: bind loopback and reach it through an explicit tunnel or an SSH port-forward (the operator's "explicit tunnels" path).
- Help text (`cli.py:688-696`) is rewritten: the API is loopback-only by policy; file serving is unclamped on loopback connections; public reachability is tunnels/port-forwards.
- The serve record (`server/registry.py`) is unchanged; a refused bind publishes nothing (the record is only written by an announced boot, `registry.py:185-216`), so "no record for a refused bind" is a testable property.

### 3.2 Per-connection gating (the actual enforcement)

- In `routes/static.py` `_servable_path` (`:79-107`), compute the mode from `request.scope.get("server")` — the `(host, port)` uvicorn sets from the accepted socket — via a new helper in `static_roots.py`:
  - `is_loopback_host(host)` → true for `127.0.0.0/8`, `::1`, `localhost`.
  - `mode = "general" if connection-is-loopback else "rooted"` (missing/unknown `scope["server"]` → **rooted**, fail-closed).
- `resolve_servable(raw, roots, *, general: bool)` (`:427`) branches:
  - shared prefix: empty/NUL → 400; raw `..` → 403 (keep `:435-440`); resolve failures → the uniform 403 (keep `:446-452`); `stat` errno classification (keep `:461-472`); `S_ISREG` (keep `:473-476`); `os.access` (keep `:477-478`).
  - `general=True`: return the realpath (no root check, no dot rule).
  - `general=False`: the existing root check + dot rule + live-session arm (`:454-459`) — untouched.
- Why both layers: bind refusal alone would be defeated by a future composition path (a bare `uvicorn local_operator.server.app:app`, an unannounced embed) that never sees the CLI flag; `scope["server"]` is derived from the socket the kernel actually accepted on, so it cannot be influenced by a header or path. **[spec]** Verify during implementation that uvicorn's `scope["server"]` for a wildcard bind reports the per-connection local address (or `0.0.0.0`, which is non-loopback → rooted) — either way fail-closed; confirm on macOS + Linux CI, note Windows.

### 3.3 Why the dot rule is dropped in general mode (and kept in rooted mode)

The rule (`:457-459`) was written as an inner bound of the root list. Under no roots it cannot be a boundary (any same-user process reads `.ssh` directly; hardlinks beat it), and keeping it **breaks real preview paths**: session scratchpads live under `~/.local-operator/sessions/<id>/scratchpad` — a dot-component path — and they are among the most common agent-output preview subjects (the desktop's mentioned-files/probe pipeline routinely carries them). A "dot components below `$HOME` are refused" variant refuses exactly those; an exemption list for the product's own directories is more machinery than the rule is worth. Rooted mode keeps the rule unchanged — it is now the fail-closed posture a non-loopback connection reaches only through embedding, never a user-selectable bind (Manager ruling: strict bind refusal, §3.1). Accepted residual: a hostile page's `<img>` observation now covers dot-directories' image files (existence/decodability/dimensions only; not bytes). Documented, not hidden.

### 3.4 What this does to the old claims

- **The wildcard-bind Host bypass is deleted** (`:525-529`): with no wide binds possible through the product, `host_is_acceptable` becomes fail-closed for a wildcard/empty announced host — it admits only IP literals and `localhost` and refuses DNS names (there is no name to compare against, so a name fails closed). If a wide bind ever appears through embedding, the per-connection gate (§3.2) keeps its file serving rooted **and** the Host check still applies.
- **SSH port-forward / local tunnel = unclamped: decided yes** (the operator's explicit-tunnel case; Manager ruling). The connection is loopback-accepted, which is exactly the condition.
- The uniform-403 body (`OUTSIDE_ROOTS_DETAIL`, `:187-196`) is rewritten per mode; in general mode the remaining refusal bodies are: invalid path, `..`, uniform resolve/stat-failure 403, not-a-file 400, unreadable 403. The "add the directory to static.roots" remedy text moves to rooted mode only.

### 3.5 The mesh relay — explicitly out, with reasons

1. It serves no file bytes: no static routes in `network/relay.py`; session history drops attachments (`:7230-7235`); `net_stream` carries viewer frames, not files (`_op_stream`, `:7885`).
2. Its 0.0.0.0 default is a product requirement — "the mesh's PRIMARY topology is one reachable device and one that is not" (`relay.py:56-71`) — and it has its own identity/pairing/capability auth with a pre-auth connection cap (`:75-98`).
3. Consequence, stated for docs: **the mesh relay remains the one LAN-listening component**; the file server must never be reachable *through* it — no `net_*` op may carry file bytes without its own design (this RFC's change set adds none).

### 3.6 Host-check scope (follow-up d)

Extend `host_is_acceptable`-style validation beyond `/v1/static/*` in a **separate, later change** (P1.5), because it touches tunnels and remote-configured apps: the mobile tunnel gateway forwards a *checked public host* to the relay harness port, which is not `lop serve` (`tunnels/gateway.py:526-546`), so the relay path is unaffected — but a remote-configured desktop backend dialed by DNS name would need that name admitted (the announced-bind rule already admits it). Fallback if disruptive: keep static-only, and make the docs say so loudly (as today).

### 3.7 Tests that prove the condition (P1 gates)

Unit + integration (isolated `HOME`, `env -i`, loopback ephemeral port; patterns from `tests/unit/test_cli_serve.py:80-114`):
1. `--host 0.0.0.0` / `::` / a LAN IP / a non-loopback-resolving name → exit 1, message names the remedy; **no listener**, **no serve record**; and a row asserting **no override flag exists** (the strict policy is the product's, not a default).
2. Default boot → listener address is `127.0.0.1`; record host is loopback.
3. Real server, general mode: `~/Downloads/x.png` (outside every old root) → 200; dot-file `~/.dot/pic.png` → 200; FIFO/dir → 400; missing → 404; over-long component → uniform 403 (not 500). (A PDF has no route in P1 — no general bytes route, D5 — so every row uses an allowlisted media mime.)
4. Same requests against a **rooted-mode** server — reached only by fabricating a non-loopback `scope["server"]` (embedding path; there is no CLI route to one) → 403 uniform for out-of-root; dot rule enforced. Plus: a name-bearing `Host` against a wildcard/empty announced host is refused (fail-closed Host check, §3.4).
5. Per-connection gate: drive the app with a fabricated non-loopback `scope["server"]` → rooted even when the announced bind is loopback (the predicate does not trust announcements).
6. Host matrix unchanged from #2134 (rebind name → 403; IP literal/localhost → pass; no-Host passes), plus: general mode never disables the check.
7. Response policy on every status (CSP variants, nosniff, no ACAO for foreign origin) re-run from the `test_server_static.py` matrix; `turn-supplements`' copy of the attack matrix is the reference.
8. A test pins that `/v1/static/images|videos|audio` cannot serve `text/html` or `*/*` bytes (D5's "no executable document under the general family").
9. Docs tests (if the repo has any for DESKTOP_API/turn-supplements status) — else manual review checklist.

## 4. The auth design (phased, abstracted, automatic)

### 4.1 Primitives chosen (no new deps, macOS + Windows)

- **Channel: the serve record.** `run/serve/<pid>.json` under the config root, `0700` dir / `0600` staged write — the exact machinery already trusted for `claim_key` (`server/registry.py:413-455`; `session/runtime/registry.py`'s shared publisher). Windows: same publish path the desktop `claim_key` already relies on (ACL form); no new platform code.
- **Minting:** at boot, the daemon draws a 256-bit seed (`secrets.token_bytes(32)`, stdlib, CSPRNG). Two HKDF-SHA256 derivations (`hmac`/`hashlib`, stdlib): the **access token** clients present, and a server-side **signing key** that never leaves the process. New record fields, additive by contract (unknown keys are dropped by old readers — `ServeRecord.from_json`, `registry.py:359-369`): the access token and a per-boot generation counter. The seed and the signing key are NOT published. [spec] The final field set is an implementation detail; the property to keep: the record carries exactly what a local surface needs to read files, and nothing that grants more than file reads.
- **Automatic, zero user action:** every surface reads the record the way the desktop app already discovers the daemon (`backend-service.ts`, `serveRecord`), presents the token, and on 401 re-reads once. A daemon restart bumps the generation; surfaces refresh.
- **For `src=` loads (the S-6 constraint: routes load by `src`, no bearer header possible):** an authenticated mint endpoint, `POST /v1/static/sign` (name proposal), auth = the access token or the desktop bearer, body `{route, path, ttl_s}` → `{url, exp}`. The URL carries the route, the original `path`, `exp` (unix seconds) and `sig = HMAC-SHA256(signing_key, route + newline + raw path string + newline + exp)`; the server verifies the signature **before any resolution** (no oracle) and refuses expired ones. TTL default 10 minutes, **approved** for previews (Manager ruling); the `<video>` Range/seek lifetime (longer TTL vs re-sign) is documented in P2 and pinned by QA. The UI re-signs on version changes, as it already re-keys on mtime/version — `html-preview.tsx:137`.
- **Why not cookies:** the renderer runs at `file://` (opaque origin `"null"`, `desktop.py:44-52`) and the TUI/mobile are not browsers; signed query URLs + header tokens cover both classes without a cookie/session concept.
- **Why HKDF:** one root, purpose-separated keys, stdlib; lets a future "read-only file token" split without re-minting; the manager's suggestion, honored.

### 4.2 Server behaviour, phase by phase

- **Phase A (ships next; permissive).** Requests are served when the loopback condition holds, **regardless of token presence**. A presented token is validated and, if valid, recorded (so the path is exercised); **an invalid or stale token is treated as absent — never a refusal in phase A** (no mixed-version cliff, no oracle). Log once per generation mismatch. The sign endpoint exists and works.
- **Phase B (opt-in enforcement).** A settings key (e.g. `static.require_token`, default **false**) makes static GETs require a valid token; absence/invalid → 401 with a stable code (`files_token_required`) so clients re-read the record and retry once. Ships after **≥2 release windows** in which every first-party surface supports Phase A (Manager ruling).
- **Phase C (default enforcement).** Flip the default after the compat audit (§4.3) holds for **≥2 further release windows** past Phase B (Manager ruling). Never touches tunnels (§4.4). Operator-overridable back to A at any time.
- **Rooted mode is unaffected** by token enforcement in P1/P2 (it is now the fail-closed posture a non-loopback connection reaches only through embedding, not a supported CLI config; a later phase may require tokens there first — it is the more exposed posture).

### 4.3 Mixed-version matrix (what each combination does)

| Server | Surface | Result |
|---|---|---|
| old (≤ v0.68.24) | new | Surface tries sign/record; 404/no field → falls back to plain URL → old clamp decides (status quo). No breakage beyond today. |
| new (Phase A) | old | Plain URLs, no token → served (permissive). No breakage. |
| new (Phase A) | new | Token + signed URLs work; token invalid/missing also served. |
| new (Phase B/C) | old | **Only combination that breaks.** Gated by opt-in + release windows; the operator flips only when their surfaces are updated; docs name it. |
| new (rooted mode) | any | v0.68.23 semantics; tokens not required to *loosen*. |

### 4.4 Tunnels keep their own auth (unchanged)

Public reachability remains: phone → Radient Worker auth → Cloudflare tunnel → gateway (≤30 s request-bound assertion; owner/tunnel/harness/body-hash checks; replay rejected; `docs/tunnels.md:306-322`) → loopback harness port (relay 4098, `tunnels/gateway.py:526-546`). The file token adds an **inner** local hop only (relay → core over loopback); it never becomes the public auth. Nothing in this design is exposed on a tunnel: `lop serve` itself is not a tunnel harness.

## 5. Surface convergence

### 5.1 Target architecture

One file-serving semantics, owned by core: predicate + mime policy + size policy + token auth. Every surface either (a) speaks the HTTP family with a token, or (b) is a local process that proxies (a) — no surface re-implements path rules.

### 5.2 Per surface

- **Desktop UI renderer**: phase 1 (with P2) gets tokens + signed URLs; `video-preview`'s route-first path keeps streaming (Range) and gains a token; `html-preview` keeps the route **permanently** (CSP pairing) and gets signed URLs. The IPC bridge stays as the interim path until the server path reaches parity, then each surface moves back: **decided (Manager ruling): keep the bridge as the fast-path / daemon-down path with policy constants shared with the server** (same 64 MiB → one constant; HEIC handled in one place); maintenance-only; revisit in P5. No big-bang.
- **TUI**: today in-process; the operator's "TUI would access through that server" becomes (Manager ruling): **new** TUI file features (browsing/previews for a session, including remote/mesh-attached ones) use the server API; existing in-process previews are unchanged. A small client helper (token from the record) lives in core so no surface re-derives it.
- **Mobile relay/app**: add a relay-side proxy (`/api/sessions/{id}/file?path=…`) that calls core's family with the local token, so the phone shows the same files as the desktop for the same session. Today's `/api/sessions/{id}/image` (transcript store) stays for history/attachments. Same-session-same-files becomes true for the live session; durable-history file references stay transcript-scoped. **[spec]** scope/ownership rules for the proxy mirror `api_session_image`'s (session resolution + `gate()`).
- **Mesh peers**: unchanged; no file bytes; keep the degradation placeholder.
- **Browser bridge**: unchanged (not a session-file surface).

### 5.3 Order of migration (mixed versions)

1. Core P1 (unclamp + loopback condition). No surface changes required; UI benefits immediately for out-of-root files via existing routes; IPC path unaffected.
2. Core P2 + UI P2 in the same window (tokens accepted; UI adopts where cheap: signed URLs for the two route users; bridge unchanged).
3. Compat window (≥2 releases): all surfaces on Phase A-capable builds.
4. Core Phase B opt-in; then Phase C default.
5. Surface moves: mobile proxy; UI per-surface bridge narrowing; TUI helper adoption. Each its own reviewed change.

## 6. The general route predicate — exact change set (P1)

| File | Change |
|---|---|
| `local_operator/server/utils/static_roots.py` | `resolve_servable(raw, roots, *, general)`; new `connection_mode(host)`/`is_loopback_host`; dot rule + root branch moved under `general=False`; `OUTSIDE_ROOTS_DETAIL` split (general vs rooted bodies); docstring rewritten (provenance: supersedes #2134's rationale; states what still stands). `host_is_acceptable`: keep, **except** the wildcard/empty-announced-host branch becomes fail-closed (IP literals + `localhost` only; DNS names refused — §3.4). `response_policy`, `frame_ancestors`, mime-independent helpers unchanged. |
| `local_operator/server/routes/static.py` | `_servable_path` computes mode from `request.scope["server"]`; passes through. No mime-list changes. |
| `local_operator/server/app.py` | None required for P1 beyond docstring updates; Host-check wiring unchanged (`:976-994`). (Whole-surface Host extension = P1.5.) |
| `local_operator/cli.py` | Bind policy in `serve_command`: loopback-only allowlist (both address families), refusal message naming the remedy, **no override flag**; help text rewritten. `_bind_serve_socket`/adopt path untouched. |
| `local_operator/settings_io.py` | `static.roots` row help/warning reworded: it applies only to the rooted fallback posture (the daemon itself can no longer be bound beyond loopback). `network.listen_address` untouched. New `static.require_token` (Phase B, default false) added then. |
| docs | `docs/DESKTOP_API.md:167-211, 2051-2052`; `docs/design/turn-supplements.md` status note (`:1031-1040`); `docs/API_FILESYSTEM_BOUNDARIES.md:52-70` (trust paragraph: loopback condition, unclamped serving); this RFC's status header when merged. |
| tests | §3.7 + update `tests/unit/server/test_static_roots.py` (root cases become rooted-mode cases), `test_server_static.py`, `tests/unit/test_cli_serve.py` (bind policy). `tests/conftest.py` `_AMBIENT_VARS` keeps `LOCAL_OPERATOR_STATIC_ROOTS`. |
| UI repo | **No P1 change.** (Phase 2: signed-URL adoption for `video-preview`/`html-preview`.) |

## 7. Disposition of the five #2134 follow-ups (required)

| # | Follow-up | Disposition |
|---|---|---|
| a | Token-authenticated static URLs (`src` loads) | **In scope — Phase 2** (§4.1): header token + signed query URL + mint endpoint; permissive first. The `src`-loads constraint is the design's core case, not an afterthought. |
| b | TOCTOU check→open (~1-in-4 wins) | **Impact converted.** With no roots there is no "outside" to win toward; remaining risks are robustness (swap-to-FIFO hang, device reads) and the rooted-mode boundary. Fix shape unchanged (`O_NOFOLLOW` open + `fstat` + stream-from-fd). **Scheduled Phase 3, not blocking**; rooted mode inherits the same fix. |
| c | Hardlinks inside a root | **Moot** for confidentiality (same-user reads directly; pages cannot create links; rooted-mode users of a shared root remain the only real case). Note only. |
| d | Rebinding-Host coverage only `/v1/static/*` | **P1.5**, separate change (§3.6); fallback = keep static-only + loud docs. |
| e | `<img>` existence/dimension signal | **Accepted residual** (now covering the disk rather than the roots): documented in the module docstring, `DESKTOP_API.md`, and this RFC. Not fixable without requiring tokens on `<img>` loads; Phase C shrinks it for token-holding surfaces, not for hostile pages. |

## 8. Phasing, checkpoints, risks

### Phasing
- **P0** — this RFC: supplements-lane review (sandbox half) + manager read. No code.
- **P1** — core unclamp + loopback condition + docs + tests (§3.7, §6). One core PR; standard gates (reviewer + QA + security round; no design round — no UI pixels). Release: patch/minor per materiality.
- **P1.5** — Host-check scope extension (d). Separate PR.
- **P2** — Phase A auth (record field, acceptance, sign endpoint) + UI signed-URL adoption in the same window.
- **P3** — Phase B opt-in enforcement + (b) TOCTOU hardening; rooted mode may require tokens first.
- **P4** — Phase C default; **P5** — surface moves (mobile proxy; UI bridge narrowing; TUI helper).

### Release checkpoints
- After P1: assert the QA matrix posts a live-server proof (loopback 200 for `~/Downloads`, refused wide bind, no record, rebinding 403); verify no surface regressed (UI #942 flows unaffected — it never used the roots).
- After P2: run the mixed-version matrix (§4.3) against an old and new daemon; snapshots of the record fields; token acceptance with/without.
- Before Phase C: audit every shipped surface's release notes for token support ≥2 windows; keep the flip a single-line default change with the revert documented.

### Risks
1. **Rebinding now reads the whole disk if the Host check fails.** Tests + security round mandatory; consider (later) randomizing the port or a per-boot path prefix as belt (not proposed now — noted as an alternative).
2. **Two postures (general/rooted) in one listener** doubles the test matrix; mitigated by deriving mode from the socket, not configuration.
3. **`scope["server"]` semantics** must be verified on the pinned uvicorn and CI (macOS + Linux; **Windows posture is [spec] with CI verification**) — fail-closed default keeps this safe; SSH-forward = unclamped is decided (yes, §3.4).
4. **Supplements contract**: if their review reads the unclamped `/v1/static/html` as weakening (an executing frame can *load* any html, though it cannot read bytes or gain origin), fallback = the html family stays rooted until Phase C tokens gate it; the media families' unclamping is unaffected.
5. **Documentation drift**: three docs + settings copy state the old bound; P1 ships their updates in the same PR (the #2134 R4 lesson).
6. **macOS TCC (documented, not a mystery)**: reads of `~/Downloads`, `~/Documents` etc. may still be refused by OS privacy controls depending on the daemon's launch context; the predicate's errno classification (`static_roots.py:461-472`) already maps that to the uniform 403. Say so in the docs and the module docstring.

## 9. Open items after the rulings (no product decisions outstanding)

1. **Supplements-lane review** of the sandbox half (their P0 gate before merge): the one reading that could change P1 is if they object to the unclamped `/v1/static/html`; fallback documented in §8 Risk 4 and §11.
2. **[spec] implementation verifications**: `scope["server"]` wildcard semantics on the pinned uvicorn (macOS + Linux + Windows CI); the loopback-resolution predicate for `--host` names; the final record field set (naming decided at implementation).
3. **P2 documentation item**: the `<video>` Range/seek re-sign approach (longer TTL vs re-sign), pinned by QA.

## 10. Appendix — condensed exposure and surface audit

(The full audit, with per-row citations, is held for the PR comment; this section is the committed summary.)

### 10.1 Listeners

| Listener | Default bind | Can widen? | Auth | Advertisement |
|---|---|---|---|---|
| `lop serve` (core API) | `127.0.0.1:1111` | **No** after this RFC (strict refusal) | none on `/v1/static/*` (tokens phased, §4); desktop plane token on control families | serve record, 0600 under 0700 — local file only |
| Mobile relay (`lop mobile serve`) | `127.0.0.1:4098` (hardcoded) | No (`--port` only) | portal password cookie + device tier; public only via tunnel | session records, local |
| Browser bridge (`lop browser serve`) | `127.0.0.1:4099` (hardcoded) | No | pairing code → hashed token in a private record | pairing record, local |
| Mesh relay (`lop network`) | **`0.0.0.0:4097`** (deliberate, §3.5) | `--listen-address` / settings | device identity + SAS pairing + capability grants; pre-auth cap 8 | none (no mDNS); the one LAN listener |
| Tunnel gateway | `127.0.0.1:<gateway_port>` | No | Radient Worker auth + <=30 s request-bound assertion + local cookies | the operator-created tunnel URL |
| OAuth callback (MCP login) | ephemeral loopback | refused off-loopback | one-shot state | none |
| `cloudflared` connector | outbound only | — | per-tunnel token file | — |
| Secrets broker | unix socket | — | filesystem perms + peer check | local |

### 10.2 Surfaces and their file paths

| Surface | Today | Converged (P2+) |
|---|---|---|
| TUI | in-process Python reads | server API for new file features; shared client helper in core |
| UI renderer | IPC bridge; `video-preview` route-first; `html-preview` route-only | tokens + signed URLs for route users; bridge as fast-path/daemon-down |
| Mobile portal | transcript-store images only | relay-side proxy → core (same files as the desktop for a live session) |
| Mesh peers | session text only; attachments degrade | unchanged |

## 11. Manager rulings (2026-10-10)

Folded into v2; recorded here so reviewers see dispositions without re-reading the thread.

| # | Item | Ruling |
|---|---|---|
| 1 | Wide-bind compatibility | **Strict — decided.** `lop serve` refuses every non-loopback bind; no override flag; Host check fail-closed for wildcard/empty announced hosts (§3.1, §3.4). |
| 2 | SSH port-forward / local tunnel | **Decided:** unclamped (the explicit-tunnel case) (§3.2, §3.4). |
| 3 | `/v1/static/html` unclamp | **Decided to proceed**, policy unchanged; supplements-lane sandbox read is the P0 gate before merge; fallback documented (§8 Risk 4). |
| 4 | Dot rule | **Decided:** dropped in general mode, kept in rooted mode; `<img>` observe expansion documented (§3.3). |
| 5 | Auth window counts | **Decided:** >=2 release windows of surface adoption before Phase B; >=2 more before Phase C (§4.2). |
| 6 | UI bridge end-state | **Decided:** keep as fast-path/daemon-down with shared policy constants; maintenance-only; P5 revisit (§5.2). |
| 7 | Host scope (follow-up d) | **Decided:** P1.5, separate change, not blocking P1 (§3.6). |
| 8 | Naming | `--allow-non-loopback-api` **gone** with ruling 1; `static.require_token` and record field names decided at implementation, keeping the stated property (§4.1). |
| — | Signed-URL TTL (open Q2) | 10-min default **approved** for previews; `<video>` Range/seek lifetime documented in P2, QA pins it (§4.1). |
| — | TUI adoption (open Q7) | **Decided:** new features first; existing in-process previews unchanged (§5.2). |
