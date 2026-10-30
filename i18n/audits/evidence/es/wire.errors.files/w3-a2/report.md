# Spanish W3 audit — wire.errors.files

- **Audit:** `w3-a2`, round 1, 2026-10-29
- **Auditor:** independent `translation-auditor` / session `600fd4a56564`
- **Actual model:** `kimi/k3.1-pro` (provider-quota fallback from `anthropic/claude-sonnet-5-5`, through `openai/gpt-6-astra`)
- **Verdict:** **PASSED — clean**
- **Read:** 11/11 English and Spanish keys
- **Findings:** 0 blocker, 0 major, 0 minor, 0 nit

## Input binding

Worktree `/Users/damian/lo-i18n-es`, branch `feat/i18n-es-wire-errors`, HEAD `66da866475f3e74553196a3167c18c6125141844`. The Spanish target is **UNTRACKED** (`git status` reports `??`), so this verdict binds its exact on-disk bytes:

- English SHA-256: `ccfeb1ea55e0fed62213846cb669f28075b20e91e361f380456772d433983f64`
- English blob: `479e05332cbde315e887645fceeebf81bf4e73c7`
- Spanish SHA-256: `99d5fc6b856cc3501640e67de115915a7f7c75545e08eb7db34d367d9facb8bd`

**No English context sidecar exists.** No role, tone or width contract was invented. Register was compared with the Spanish style guide and shipped `wire.errors.desktop`, `wire.errors.settings`, `wire.errors.notifications` and `wire.exec` copy.

## What was verified

Each English and Spanish value was read side by side from the exact bytes. Audio, HTML, image and video messages remain **type** failures, with `{mime_type}` intact; none becomes a size, permission or missing-file error. `heic_conversion` remains a failed **conversion**, not a failed load or save. `html_utf8` remains an invalid UTF-8 text diagnosis. `internal_error` remains an internal server error with its `{error}` detail.

The signing errors preserve their exact causes and remedies: `sign_auth` requires either a local access token or desktop bearer; `sign_path` requires a non-empty `path`; `sign_route` restricts `route` to `{routes}`. `sign_ttl` is exactly `ttl_s debe estar entre 1 y {maximum}.`, preserving the integer field and inclusive `1` through `{maximum}` range without changing either bound or inventing fractional support.

No failure reads as success, and no refusal is presented as an outage or retryable operation. All 11 keys have exact placeholder parity. `HEIC/HEIF`, `UTF-8`, `token`, `bearer`, `path`, `route`, `ttl_s`, `{mime_type}`, `{routes}`, `{maximum}` and `{error}` survive byte-identically. No ICU plural branches occur. Terminology and sentence case match the guide and shipped Spanish error register.

Relative character lengths were measured; the largest expansions remain idiomatic and no width contract exists. They are not findings or a rendered-fit claim.

## Mechanical evidence

`python3 /Users/damian/.local-operator/sessions/600fd4a56564/scratchpad/audit_es_w3a2.py` exited 0. Complete output is in `audit.log`: 11/11 key parity, 11/11 placeholder parity, zero plural nodes, no disallowed categories, dropped-placeholder negative control detected, `FAILURES=[]`.

The prescribed worktree-local `read_public_copy` route returned the 11-key English source after refusing the invalid aggregate key `all`. Full source and untracked-target bytes were read directly and hashed. The reader cannot read the untracked target or nonexistent sidecar; no reader success is claimed for those. No display-side redaction was used for a judgement.

No catalogue, sidecar, ledger, allowance, generated file or git state was edited. No repo-wide tests were run.
