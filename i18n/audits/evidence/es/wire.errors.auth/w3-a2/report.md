# Spanish W3 audit — wire.errors.auth

- **Audit:** `w3-a2`, round 1, 2026-10-29
- **Auditor:** independent `translation-auditor` / session `600fd4a56564`
- **Actual model:** `kimi/k3.1-pro` (provider-quota fallback from `anthropic/claude-sonnet-5-5`, through `openai/gpt-6-astra`)
- **Verdict:** **PASSED — clean**
- **Read:** 12/12 English and Spanish keys
- **Findings:** 0 blocker, 0 major, 0 minor, 0 nit

## Input binding

Worktree `/Users/damian/lo-i18n-es`, branch `feat/i18n-es-wire-errors`, HEAD `66da866475f3e74553196a3167c18c6125141844`. The Spanish target is **UNTRACKED** (`git status` reports `??`), so this verdict binds its exact on-disk bytes:

- English SHA-256: `9d48edaa61269239e132201224c1c7d6cf89ef198598f0ec6c99c1f3ade542f6`
- English blob: `d6781d2567c65a7cc51e797883eaeea3fde1be68`
- Spanish SHA-256: `5e23604abad61cc486b8d7764f09cfd882f9afd2a32248ea1efbf5d28d9f7ad3`

**No English context sidecar exists.** No role, tone or width contract was invented. Register was compared with the Spanish style guide and shipped `wire.errors.desktop`, `wire.errors.settings`, `wire.errors.notifications` and `wire.exec` copy.

## What was verified

Each English and Spanish value was read side by side from the exact bytes. Missing credentials remain missing credentials; an unaccepted API-key mechanism remains distinct from a provider-rejected key; an absent provider account, absent server address and unreadable credential store remain distinct causes. `local_server_only` preserves the local-only testing restriction. `sign_in_missing` preserves both permanent unavailability and the restart remedy; `sign_in_not_waiting` does not claim that sign-in failed or expired. `input_invalid` preserves the exact non-empty and 32768-character constraints.

`auth_refusal` and `provider_refusal` are pure `{error}` passthroughs on both sides and are **correct**, not missing translations. No failure reads as success; no retryable/permanent inversion or new promise was found.

All 12 keys have exact placeholder parity. No ICU plural branches occur. Machine terms and the numeric bound are unchanged. Neutral **tú** (`Introduce`, `Empieza`) matches the guide; no usted/vosotros drift. Credential/provider terminology is consistent with the glossary and shipped copy.

Relative character lengths were measured, including `input_invalid` (52 English / 78 Spanish) and `server_address_missing` (44 / 65). With no sidecar width budget and clear, idiomatic phrasing, these measurements are not findings or a rendered-fit claim.

## Mechanical evidence

`python3 /Users/damian/.local-operator/sessions/600fd4a56564/scratchpad/audit_es_w3a2.py` exited 0. Complete output is in `audit.log`: 12/12 key parity, 12/12 placeholder parity, zero plural nodes, no disallowed categories, dropped-placeholder negative control detected, `FAILURES=[]`.

The prescribed worktree-local `read_public_copy` route returned `account_not_found`, `auth_refusal` and `key_not_accepted`; some individual-key invocations timed out. Full source and untracked-target bytes were read directly and hashed. The reader cannot read the untracked target or nonexistent sidecar; no reader success is claimed for those. No display-side redaction was used for a judgement.

No catalogue, sidecar, ledger, allowance, generated file or git state was edited. No repo-wide tests were run.
