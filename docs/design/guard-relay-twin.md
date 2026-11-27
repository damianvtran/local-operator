# Design: Radient Guard on the phone relay (`/api/guard`)

Status: contract for slice L2d (Guard RFC v0.9, C3 under the RFC approved 2026-11-27).
Client: the phone (local-operator-mobile PR #402, `docs/guard.md`, mock
`tools/mock-relay/guard.ts`). Host code: `local_operator/mobile/guard_relay.py`, mounted by
`mobile/daemon.build_app`. Models: `server/models/desktop_guard.py` (shared with L2c; the
three consent models at the bottom are relay-only). Stacked on L2c
(`docs/design/guard-desktop-routes.md`): the routes are the same engine under a second
transport.

## 0. What this is, and what it is not

The phone **shows** the host's account of Guard and **relays** the member's answer. It never
grants: the host signs the member's consent with its own key (RFC §5.2.2) and sends the one
grant to Radient. These routes are thin skins over the functions the desktop routes call
(`server/utils/desktop_guard.py`, `network/credentials/desktop_approval.py`); there is no second
implementation of any Guard rule.

**There is no grant route.** The only path to a grant is `POST /api/guard/consents/{id}/accept`,
which is the typed `approve-operation` for the pending `guard_consent` card, with the operation
fixed to `guard_consent` by the host. It needs the card's own `request_digest` and
`disclosure_digest` (closed body, no scope field), recomputes both from the record on disk and
refuses a mismatch **before** its signer runs. The generic bodiless `approve` stays refused for
that card (`guard_consent_typed_approval_required`); the relay never calls it. Nothing here
raises a notification, push or badge.

## 1. Auth, capability, errors

* **Auth:** every route is behind the relay's own `gate`: the paired-device login
  (the relay session cookie), plus the exact-Origin mutation check. A request without it is
  answered 401 before any engine call. This is how `/api/asks` and `/api/notifications` are
  authenticated.
* **Capability:** `capabilities.features.guard` = `1`, republished by the relay **only while
  the twin is mounted** (`daemon._GUARD_TWIN_MOUNTED`). `guard` is no longer in
  `DESKTOP_ONLY_FEATURES`; the key means "this plane's Guard routes exist" (`/v1/desktop/guard`
  on the desktop plane, `/api/guard` here). A relay that failed to load the twin withholds the
  key and the routes 404.
* **Bodies:** bare models (no `CRUDResponse` envelope; the relay's convention).
* **Errors:** `{"error": <sentence>, "code": <token>}`.

| Status | `code` | When |
|---|---|---|
| 401 | (relay) | no/invalid paired-device cookie |
| 403 | (relay) | cross-origin mutation |
| 404 | `guard_unavailable` | any write and any consent route when Guard is not available here: non-Team account, or no org has turned Guard on |
| 404 | `guard_not_applicable` | `non-team-card` for an account with a Team org |
| 404 | `guard_org_unknown` | `tenant` is not an org this computer knows (or is not shaped like one) |
| 404 | `approval_not_found` | no such card, **a card of another kind**, or a malformed id: one answer for all three, so the relay is not an approvals oracle |
| 409 | `guard_not_sharing` / `guard_not_stopped` | revoke / share-again for an org in the wrong state |
| 409 | `guard_answer_not_admitted` | share-again where this daemon was STARTED agent-driven (`can_answer: false`) |
| 409 | `guard_consent_expected_mismatch` | a digest that is not the card's (the engine's `credential_request_changed` / `credential_disclosure_changed`, renamed to the code the phone already reads; the engine's own code rides in `engine_code`) |
| 409 | engine code | the typed engine or consent engine refused: `approval_decision_conflict` (card already answered), `approval_expired`, `credential_presence_required`, `credential_sign_rate_limited`, `guard_consent_not_granted` (signed, grant did not land: re-POST the same body), ... |
| 422 | `invalid_body` | `accept` body is not exactly the two digests (`sha256:<64 hex>`) |

## 2. Routes

| Route | Answer |
|---|---|
| `GET /api/guard` | `GuardStatus` (L2c, unchanged). Local facts only: zero network for every account. |
| `GET /api/guard/non-team-card` | `GuardNonTeamCard` (L2c). The one read that leaves the machine (`/v1/team-trial/eligibility`, never `/v1/guard`). |
| `POST /api/guard/orgs/{tenant}/revoke` | `GuardRevokeResult` (L2c). No body. |
| `POST /api/guard/orgs/{tenant}/share-again` | `GuardShareAgainResult` (L2c). No body. Person-gated. |
| `GET /api/guard/consents/{approval_id}` | `GuardConsentCard`. |
| `POST /api/guard/consents/{approval_id}/accept` | Body `GuardConsentAcceptRequest` `{request_digest, disclosure_digest}` → `{approval_id, state:"granted", tenant}`. |
| `POST /api/guard/consents/{approval_id}/decline` | No body → `{approval_id, state:"denied", tenant}`. |

`GuardConsentCard { approval_id, state, tenant, org, reason, widening, asked_scopes[],
request_digest, disclosure_digest, signable, refusal? }`. The digests are the engine's own
(`desktop_approval.row_fields`, the same two values the desktop's `typed_approval` carries);
`signable`/`refusal` are the engine's policy verdict (assurance level, card state).

**Answers are stamped `mobile`.** The host signs, but the person answered on the phone, so
the queue's `decided_via` (rendered "Answered on the phone app", and the `answered_via` the status
reads) and the credential audit rows (`credential.desktop_signed`, `_applied`, `_sign_refused`)
carry `mobile` (`desktop_approval.answering_as`). `decline` folds the sticky "no" immediately with
`mobile`, rather than leaving it to the next admission point (which would stamp `desktop`).

**Signed but not applied.** If the grant does not reach Radient, `accept` answers 409 with the
engine's code; the card stays signed and its prompt stays pending. Re-POSTing the same body retries
the apply (the desktop's `apply-operation`): it signs nothing new and re-checks both digests. A card
whose grant landed, or whose prompt moved on, is refused `approval_decision_conflict`.

**The person-only gate.** `share-again` is refused wherever the desktop plane refuses it, from
the same capture (`surface.served_host_can_answer`): `lop mobile` started inside an agent's shell
or a console pane. `lop mobile` does not strip its own markers, so the value is read once when the
app is built. It is per daemon process, not per caller: what keeps a stranger out is the paired
cookie, the same residual as every other relay route. It only re-raises the prompt; it grants
nothing.

## 3. Rollout / rollback

Additive routes plus one additive feature key on the relay. A phone that predates them never calls
them; a host that predates them does not advertise `features.guard`. Rollback = revert the PR:
the routes disappear, `features.guard` disappears from the relay, and the phone returns to its
"unavailable" line. The three small engine changes (`accept_signed(source=)`,
`desktop_approval.answering_as`, the audit surface) default to `desktop`, so the desktop plane and
the TUI are unchanged byte for byte. Nothing is migrated.

## 4. Not in this contract

* `GuardNonTeamCard.role` (the phone's optional §14' owner/admin/member variants): the host
  has no source for a non-Team account's role. Absent, the phone shows the core TUI's card.
* Report confirm, notice-shown, notice-due: the phone shows no receipts or reports. A receipt
  the status lists stays unshown until the TUI or desktop shows it.
* Lifting `guard.consent.UNMEASURED_SCOPES` (unchanged from L2c).
