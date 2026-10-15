# Operator authoring of a signed future-login permission (`credential inherit`)

Scope: **CLI only**, on top of the merged consent foundation (#2412). No new permission
class, schema, signature format or expiry policy; no UI repository change.

## What it is
`lop network credential inherit <provider> --to <device> [--account A] [--org O]
--request-approval | --approval <id> | --status` authors the signed
`credential_inheritance` approval that nothing previously authored. Code:
`local_operator/network/credentials/inheritance.py`; seams patched: the generic-runner
guard (`onboard_approvals.begin_run`), `approvals run`, `approvals approve` disclosure,
`approvals show/list` rendering, the desktop approve route, and the reconcile
precondition (`reconcile.transport_gap`, one owner for the broker-row words).

## Decisions (each a constraint, not an omission)
* **Lifetime is the foundation's: until revoked.** The 1 h figure is the request window of
  the CARD; every sentence about it says so, and no sentence calls the permission expiring.
  No automatic expiry, no wildcard (`eligible-owner-native`), no MCP selector is authored.
* **Exact scope.** One explicit provider id with `future_logins=true`, `mcp.urls=[]`,
  origin `owner-native-login`, bound to the exact native account (and optional org) read from
  this device's own inventory and to the signing operator key. Copied, borrowed, device-bound,
  static-key and account-less rows are never a source.
* **`broker_credential` is a precondition, never an output.** Missing on either member row:
  refused before any write with `reconcile.transport_gap`'s exact remedy.
* **Sticky withdrawal wins.** A recipient with a withdrawal (`credential revoke`) or a retained
  finite-recovery fence for that provider cannot be re-granted here; a second standing
  permission that would REPLACE the first is refused rather than silently narrowed.
* **No self-approval, as strong as the key.** The request is filed by the CLI; only
  `approvals approve` signs. With a presence-gated operator key (Secure Enclave / Touch ID)
  that needs the human's gesture; with a `file-only` key any process running as the same user
  can sign, so an agent on that machine could approve its own request (`lop operator status`
  names the level). The Touch ID path is not exercised here. The desktop route refuses to sign
  it until the app can state what is signed, and names the command; deny is unaffected.
* **One-shot apply.** `begin_run -> provision_from_run -> mark_connected`; the card ends
  `connected`, not a `connecting` that folds to `expired` after an hour. A failed apply is
  `failed` (retry-eligible), not a stuck run.
* **Status is permission, not delivery.**

## Evidence
Software-signer, isolated roots. **Touch ID / a live second host is not exercised and is not
claimed.** Real-path cells: `tests/unit/network/test_credential_inherit.py` (CLI entrypoints
request -> approve -> apply -> status; 40 cells including every negative) and
`tests/unit/network/test_credential_inherit_link.py` (3 cells; two real relays + rotating IdP: the
authored permission, then a same-account re-login, heals a RUNNING session on the 15 s
cadence with one IdP POST and no `credential share`/pull/restart; a new account costs zero
IdP POSTs and reads `notready:scope_mismatch`).

## Not in this slice
Pair-time authoring, MCP/`approved-owner-servers`, `eligible-owner-native`, supersede of a
revoke, permission expiry, a desktop approval that states what is signed (the UI follow-up
consumes the optional `summary`/`details`/`can_approve_here`/`approve_remedy` row keys),
live delivery-stuck reporting (needs a new local op), `docs/openapi.json` regeneration
(already stale on main for ~50 unrelated schemas).
