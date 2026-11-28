# Design: Radient Guard on the desktop HTTP plane (`/v1/desktop/guard`)

Status: contract for slice L2c (Guard RFC v0.9, C3 under the RFC approved
2026-11-27). Clients: the desktop app (U1) and the phone (M1). Machine form:
`local_operator/server/models/desktop_guard.py` (typed Pydantic models; the two
change together). Capability key: `features.guard` = `1` in
`GET /v1/capabilities` (desktop-only: the phone relay does not republish it).

## 0. What this is, and what it is not

The TUI's `/guard` screen is built from in-process calls
(`guard/surface.py`, `guard/consent.py`, `guard/reports.py`). Until now the only
Guard surface a desktop client could reach was the pending `guard_consent` card
in `GET /v1/desktop/approvals`. These routes expose the REST of the TUI's
screen — status/indicator, stop sharing, share again, non-Team card, receipts —
by calling the **same engine functions** the TUI calls. There is no second
implementation: the routes are thin skins over `server/utils/desktop_guard.py`,
which composes `surface.org_states`, `surface.non_team_card`,
`GuardConsentHost.revoke/share_again`, `GuardReports.drafts/discard`.

**There is no grant route.** A first grant or a widening is answered ONLY by the
typed `guard_consent` card (accept = typed `approve-operation` carrying
`request_digest` + `disclosure_digest`; decline = `deny`). The generic
`approve` on that card keeps refusing (`guard_consent_typed_approval_required`).
Nothing here accepts a scope, a digest or a signature in a body, and request
models forbid extra keys. **Report confirmation is also not a route** (it sends
data off this computer and needs a person-gated answer the desktop plane does
not yet have a typed card for); only *discard*, the safe direction, is.

## 1. Auth, origin, errors

Every route sits behind `require_desktop` (the app-wide desktop bearer + Origin
allowlist), exactly like `desktop_approvals`. Refusals are the plane's
`{code, params, text}` envelope under `detail`.

| Status | `code` | When |
|---|---|---|
| 401/503/403 | (plane) | no/incorrect bearer; plane closed; origin not allowed |
| 404 | `guard_unavailable` | any Guard **write** (and the notice/report writes) when Guard is not available here: non-Team account, or no org has turned Guard on (no consent store). `GET /v1/desktop/guard` and `GET .../non-team-card` never use it: they ARE the answer for an account where Guard is unavailable. |
| 404 | `guard_not_applicable` | `non-team-card` for an account that has a Team org |
| 404 | `guard_org_unknown` | `tenant` is not an org this computer knows |
| 404 | `guard_report_unknown`, `guard_notice_unknown` | no such open draft / no receipt with that id **that this daemon has listed** (see `notices/{id}/shown`) |
| 409 | `guard_not_sharing` | revoke for an org with nothing to stop |
| 409 | `guard_not_stopped` | share-again for an org that is not stopped |
| 409 | `guard_answer_not_admitted` | share-again where this backend was STARTED agent-driven (`can_answer: false`; see "The person-only gate") |
| 409 | engine code (`guard_*`) | the consent engine refused (damaged/newer store, ...) |
| 422 | (FastAPI) | malformed body / path |

## 2. Routes

### `GET /v1/desktop/guard` → `GuardStatus`

Local facts only. **Zero `/v1/guard/*` calls and zero network of any kind**, for
every account; for a non-Team account it also opens no Guard file and creates
none (pinned by test).

```
GuardStatus {
  mode: "none" | "orgs" | "non_team" | "unavailable"
  badge: "" | "waiting" | "report_waiting" | "sharing_pending" | "sharing"
  orgs: GuardOrg[]            # one per Team org that raised Guard for this member
  reports: GuardReportDraft[] # agent-drafted, enum-only, awaiting a person; `created_at` floored to the minute
  notices: GuardNotice[]      # unshown grant/widen receipts ("Not you? Stop sharing"); `at` floored to the minute
  insights: GuardInsightsLink[]  # Console link for owner/admin orgs (advertising only)
  non_team: GuardNonTeamCard | null  # mode == "non_team": url only (see below)
  backup_days: int | null     # RFC §5.4 "X" = 31 (G2 receipt, 2026-11-27)
  this_computer: string
  can_answer: bool            # surface.host_can_answer() as captured when this daemon started
  team_url: string
}
GuardOrg { tenant, org, role, state, reason, since, answered_via,
           scopes_shared[], prompt: GuardPrompt | null, can_revoke, can_share_again }
state: sharing | sharing_pending | waiting | not_sharing | paused | will_ask
reason (paused): entitlement_lapsed | org_off | membership_ended | scope_revoked |
                 revoke_pending | console_revoked   (unknown value => plain "paused")
GuardPrompt { approval_id | null, collection_rev, reason, widening,
              asked_scopes[], grant_scopes[] }   # display only; answer on the card
```

* `mode`: `non_team` = no Team org and not an insights admin (the §14' card);
  `none` = Team account, no org raised Guard (TUI "Guard is off for you");
  `orgs` = at least one `orgs` row; `unavailable` = a damaged/newer Guard store
  (read-only; show "unavailable").
* `sharing_pending` is "still sharing, new items waiting" — **never** render it
  as paused. `waiting` is "first grant pending; nothing is shared yet".
* `prompt.approval_id` names the card in `GET /v1/desktop/approvals`. `null`
  means this computer has no operator key to file a card against; the person
  answers in the terminal `/guard`.
* `can_revoke` / `can_share_again` are the TUI's own button predicates
  (`surface.can_revoke`, `surface.can_share_again`); `can_share_again` is also
  `false` when `can_answer` is `false`.
* The indicator sentences are the client's keyed copy (§15.4' C9); the backend
  ships enums only. Text in a response is the org name (one line, capped).
* **Timestamps are minutes.** `notices[].at` and `reports[].created_at` are epoch seconds
  floored to the minute (`surface.to_minute`), the resolution the TUI shows (`%H:%M`), so
  no client (or later phone-relay twin) receives finer timing than the screen it mirrors.
* `backup_days` is `surface.BACKUP_DAYS`. If `null`, do not render the
  trace-detail revoke sentence. (Today no consent can hold `trace_detail`: the
  engine still refuses that scope, `guard.consent.UNMEASURED_SCOPES`.)

### `GET /v1/desktop/guard/non-team-card` → `GuardNonTeamCard`

`{trial_days: int|null, org: string|null, url}`. The ONE network read in this
family, and it is `GET /v1/team-trial/eligibility` as the signed-in person, never
a `/v1/guard/*` call; failure or signed-out degrades to
`{trial_days:null, org:null, url}` ("See Radient Team"), never an error. Answers
404 `guard_not_applicable` only for an account that HAS a Team org (the card is
not for it).

### `POST /v1/desktop/guard/orgs/{tenant}/revoke` → `GuardRevokeResult`

"Stop sharing" (the C2 confirm is the client's job). No body. Always allowed while
this computer lists the org (a lapsed `revoke_pending` is retried by the admission
point); once the cached Team membership is empty the route answers 404
`guard_unavailable` exactly as the TUI hides its screen then, so a lapsed-but-live
consent has no stop path on this plane (an RFC §5.4 question for the owners, not
decided here). The local stop is committed **before** the request to Radient is
sent, so `status: "revoke_pending"` (Radient not reached) is a 200: this install is
not sharing. `org` is the fresh `GuardOrg`. The outbox of already-queued items is
emptied by the producers tick (first ~30 s after serve start, then every 900 s),
not inside the route, so a client must not promise "deleted now".

### `POST /v1/desktop/guard/orgs/{tenant}/share-again` → `GuardShareAgainResult`

Lifts the sticky "declined/revoked" answer and then does one best-effort
admission-point read so the org's prompt can come back (TUI `_ask_again_now`).
It grants nothing: the returned `org.prompt`, if any, still needs the signed
card. No body. 409 `guard_not_stopped` unless `can_share_again` was true.

**The person-only gate.** Refused with 409 `guard_answer_not_admitted` (and
`can_answer`/`can_share_again` read `false`) when this backend was started inside an
agent's shell (`LOCAL_OPERATOR_AGENT_SHELL`) or a desktop console pane
(`LOCAL_OPERATOR_CONSOLE_SURFACE`). That is a fact about how the process was
**started**, and `lop serve` strips the agent-shell markers from its own environment
at startup, so it is read once, *before* that drop (`surface.capture_host_can_answer`,
called first in `cli.serve_command`), published as `app.state.guard_host_can_answer`
by the lifespan, and read from there by the routes. A child of a refused daemon (a
`--reload` worker, a re-exec) inherits the refusal through
`LOCAL_OPERATOR_GUARD_HOST_CAN_ANSWER=0`; the variable is only ever written as `0`
and only ever read to refuse, so setting it to anything else widens nothing.
**Scope of the claim:** the gate is per daemon *process*, not per *caller*. What keeps
a caller in an agent's shell out of a person-started daemon is the desktop bearer
(withheld from agent shells), the same residual as every other desktop route; it is not
a proof that a person is present. A caller holding the bearer can still lift a sticky
"no" on a person-started daemon, which only re-raises the prompt (card fatigue plus one
admission-point read): it never grants, because the grant needs the signed card.

### `POST /v1/desktop/guard/reports/{id}/discard` → `{id, state:"discarded"}`

The safe direction; no admission.

### `POST /v1/desktop/guard/notices/{id}/shown` → `{id, shown:true}`

Marks a receipt shown (shared with the TUI: shown once, anywhere). It is the
acknowledgement a client makes **after it rendered** the receipt, so only an id that
`GET /v1/desktop/guard` returned from *this daemon* is accepted (404
`guard_notice_unknown` otherwise): the "Not you? Stop sharing" receipt is the detective
control after a grant, and a bearer holder cannot dismiss one no surface here displayed
by guessing small integers. Idempotent: a repeat for a receipt already shown is the same
200 with no second write. The receipt stays in the store; the route only flips `shown`.

### `POST /v1/desktop/guard/notice-due` body `{surface: "desktop"}` → `{due: bool}`

The §15.4' C6 once-per-local-day host notice (`surface.should_show_notice`).
**Asking consumes the day.** Call it only when about to show the notice.
The plane has one shared bearer and **no per-caller identity**, so the surface is bound
per capability instead: this router is desktop-only (`features.guard` is in
`DESKTOP_ONLY_FEATURES`), so `desktop` is the only accepted value (`mobile` is 422) and
no caller can consume another surface's day. Residual: any holder of the bearer can still
spend the *desktop* day (suppressing the host notice for that day); the bearer holder is
the desktop app. A phone-relay twin carries its own authenticated surface (§4).

## 3. Rollout / rollback

Additive routes plus one additive feature key: a client that predates it never
calls them. Rollback = revert the PR; `features.guard` disappears, and a client
reading it returns to "Guard only via the approvals card". Nothing is migrated.

## 4. Not in this contract

* A phone-relay twin of these routes (the relay serves its own `/api/...`
  surface; M1 decides whether to proxy these or add relay twins).
* Report confirm, and any scope change/grant (typed approvals card only).
* Lifting `guard.consent.UNMEASURED_SCOPES` (`trace_detail`, `admin_share`). X is measured
  (31, `surface.BACKUP_DAYS`) but offering those scopes is a separate decision.
* "Other computers": the server owns that list; this computer reports only itself.
