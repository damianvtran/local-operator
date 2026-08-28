/* Pure state machine for the async site-access flow, DOM- and chrome-free for
 * the pure-node test suite — the same pattern as pair-flow.ts/origin-flow.ts.
 *
 * Why this flow exists at all: the original design answered a first visit to a
 * new origin by BLOCKING the open/goto RPC inside the 60 s popup prompt. The
 * agent never got a turn to tell the user a prompt was up, so prompts expired
 * unseen (auto-deny), the daemon meanwhile extended the command's deadline,
 * and the session client timed out with a misleading "bridge unreachable" — a
 * real session burned an hour misdiagnosing the bridge as broken. The fix is
 * to make approval a first-class three-step dance the agent can narrate:
 * open fails early → request_access raises the prompt and returns → the agent
 * TELLS the user → await_access waits for the decision.
 *
 * Multi-surface constraint (round-1 review, B1): parallel sessions now each
 * drive their OWN tab, and this state is extension-global. Everything that
 * carries authority — a pending request, a once-grant — is therefore bound to
 * the REQUESTER (the daemon's per-command request id, which one session owns
 * and another cannot guess). A grant minted for session A's flow can never be
 * spent by session B's navigation: consumption fails CLOSED when the caller
 * cannot prove it asked. A shared-grant model was considered and rejected:
 * "Allow once" is the user's answer to "should THIS agent open THIS site",
 * and letting a different agent ride it turns one approval into a fleet-wide
 * permission — indistinguishable from "always" for the other sessions. */

export type OriginDecision = "once" | "always" | "all_ports" | "deny";

/** What request_access/await_access report back to the agent. "none" means no
 * live request exists FOR THE CALLER (never raised, expired, or superseded);
 * "superseded" means a DIFFERENT requester's prompt replaced this origin's
 * pending request — an explicit verdict so the displaced requester learns
 * what happened instead of timing out into the generic "none" (round-1 B1b).
 * The recovery for both is the same: call request_access again. */
export type AccessState = "allowed" | "denied" | "pending" | "superseded" | "none";

/** How long an async access request stays answerable. The in-command prompt
 * gave the user 60 s because someone was assumed to be watching the popup;
 * here the user has been notified ASYNCHRONOUSLY (badge + best-effort system
 * notification + the agent messaging them through the harness) and may
 * reasonably take minutes to come back to the machine, so the window is 10
 * minutes. It is still bounded because an unanswered prompt left open forever
 * would let a long-forgotten Allow click grant a navigation nobody remembers
 * requesting. */
export const ACCESS_REQUEST_TTL_MS = 10 * 60_000;

/** How long an unconsumed "Allow once" grant from the async flow survives.
 * In the in-command flow "once" covered the navigation that was already in
 * flight; here the navigation happens on the agent's NEXT open/goto, which
 * may be a turn or two away, so the grant must outlive the approval moment —
 * but not the browser session, and not long enough to surprise the user with
 * a navigation they approved an afternoon ago. Same 10-minute bound as the
 * request itself, consumed by the first navigation to the origin. */
export const ONCE_GRANT_TTL_MS = 10 * 60_000;

/** The persisted async request record. Lives in chrome.storage.session (not
 * worker memory) because MV3 kills the service worker between the user's
 * decision and the agent's next await_access poll; the record must survive
 * that death or a granted approval would read as "none". */
export interface AccessRequest {
  origin: string;
  hostname: string;
  /** The daemon request id that raised this request; also minted onto any
   * "once" grant the decision produces, binding the grant to this requester. */
  requester: string;
  requestedAt: number;
  expiresAt: number;
  /** Set once the user decided; absent while the prompt is still open. */
  decision?: OriginDecision;
}

/** A live "Allow once" grant, bound to the requester the user approved FOR
 * (see the multi-surface constraint at the top): only a navigation carrying
 * that requester may consume it. `handoff` records the command id that raised
 * the request, so a raw-RPC caller (no session identity) can still spend its
 * own grant when its navigation reuses that id. */
export interface OnceGrant {
  expiresAt: number;
  requester: string;
  handoff?: string;
}

/** Origin -> grant, one per origin at most. */
export type OnceGrants = Record<string, OnceGrant>;

/** The live prompt slot holds ONE record (the popup's constraint), so a
 * replace-don't-queue overwrite would otherwise erase the displaced request
 * silently — the displaced requester's next poll would read "none",
 * indistinguishable from expiry, and its agent would nag the user by
 * re-raising the prompt in a steal loop (round-1 B1b). The tombstone map
 * remembers RECENT replacements (bounded, same TTL discipline) so that
 * requester learns "superseded" instead. Not a queue: the new prompt still
 * replaces the old one immediately; this is only the receipt. */
export interface SupersededTombstone {
  origin: string;
  requester: string;
  /** Matches the displaced record's expiry: the receipt lives no longer
   * than the request it stands for. */
  expiresAt: number;
}

/** Keyed by origin AND requester (see receiptKey): an A→B→C chain on one
 * origin displaces two DIFFERENT requesters, and a per-origin key made C's
 * receipt overwrite A's so A read "none" instead of "superseded" (round-2
 * M1, reproduced by review). Each displaced requester owns its own receipt. */
export type AccessTombstones = Record<string, SupersededTombstone>;

export function receiptKey(origin: string, requester: string): string {
  // \n cannot appear in an origin or a requester id, so the composite is
  // collision-free without a structured key.
  return `${origin}\n${requester}`;
}

/** Cap on retained tombstones: a churny fleet must not grow the map without
 * bound; the oldest beyond the cap are dropped (a tombstone that old is
 * telling its owner about a prompt the user has long forgotten anyway). */
export const TOMBSTONE_CAP = 8;

/** A record past its TTL reads as absent everywhere: await_access answers
 * "none" and request_access raises a fresh prompt. Expiry is computed on read
 * rather than by a trusted timer because worker death cancels timers. */
export function activeRequest(
  record: AccessRequest | undefined,
  now: number,
): AccessRequest | undefined {
  if (!record) return undefined;
  return now < record.expiresAt ? record : undefined;
}

/** A live grant this requester may consume. Fail-CLOSED: a grant whose
 * requester does not match the caller — or a caller with no requester at all —
 * is not this caller's grant, even if the origin matches. */
export function consumableGrant(
  grants: OnceGrants | undefined,
  origin: string,
  requester: string,
  now: number,
): OnceGrant | undefined {
  const grant = grants?.[origin];
  if (!grant || now >= grant.expiresAt) return undefined;
  if (!requester || grant.requester !== requester) return undefined;
  return grant;
}

/** Whether ANY live grant exists for the origin (regardless of requester).
 * Used for display/diagnostics only — never as an admission decision. */
export function anyGrantLive(
  grants: OnceGrants | undefined,
  origin: string,
  now: number,
): boolean {
  const grant = grants?.[origin];
  return !!grant && now < grant.expiresAt;
}

/** What request_access should do. Single-slot semantics: the popup and the
 * pendingOrigin session record only ever show ONE origin, and the in-command
 * flow already lets the last writer win that slot, so a request for a
 * DIFFERENT origin REPLACES the live one rather than queueing — a queue
 * would show the user prompts in an order the agents no longer care about.
 * The displaced requester learns about it: their next request_access or
 * await_access answers "superseded" (their record is tombstoned below), not
 * a silent "none" they could misread as expiry (round-1 B1b).
 * A repeat request for the SAME pending origin BY THE SAME requester is
 * idempotent ("pending", original TTL kept — resetting it would let a polling
 * agent extend the window forever). The same origin re-requested by a
 * DIFFERENT requester also replaces: two sessions genuinely racing one site
 * get the newest prompt, and the earlier requester reads "superseded". A fresh
 * "deny" answers "denied" without re-prompting: the user just said no, and
 * re-raising the prompt on every agent retry is a nag; the record's TTL is
 * the cool-down after which a deliberate re-ask is allowed. */
export function requestVerdict(
  record: AccessRequest | undefined,
  storedAllowed: boolean,
  onceGrant: boolean,
  origin: string,
  requester: string,
  now: number,
): "allowed" | "pending" | "denied" | "raise" {
  if (storedAllowed || onceGrant) return "allowed";
  const live = activeRequest(record, now);
  if (!live || live.origin !== origin) return "raise";
  // A resolved record only answers ITS OWN requester (round-2 M1: B reading
  // A's "once" as allowed told B it could navigate a grant it cannot spend,
  // and B reading A's deny as denied hid that B never asked). Another
  // requester's resolved record is displaced by a fresh raise; an undecided
  // one is also displaced (replace-don't-queue), leaving a receipt.
  if (live.requester !== requester) return "raise";
  if (live.decision === "deny") return "denied";
  if (live.decision) return "allowed";
  return "pending";
}

/** What await_access answers WITHOUT waiting. "pending" is the only state the
 * caller should then block on; everything else is final for this poll. A
 * tombstone only reads as "superseded" to the requester it displaced — a
 * third session asking about the origin gets the neutral "none". */
export function accessState(
  record: AccessRequest | undefined,
  tombstones: AccessTombstones | undefined,
  storedAllowed: boolean,
  onceGrant: boolean,
  origin: string,
  requester: string,
  now: number,
): AccessState {
  if (storedAllowed || onceGrant) return "allowed";
  const live = activeRequest(record, now);
  // Only the record's OWN requester reads its live state (round-2 M1): after
  // B replaces A on the same origin, A must read its receipt (superseded),
  // not B's pending; after A's once-decision, B must not read allowed for a
  // grant only A can spend.
  if (live && live.origin === origin && live.requester === requester) {
    if (!live.decision) return "pending";
    return live.decision === "deny" ? "denied" : "allowed";
  }
  const tomb = tombstones?.[receiptKey(origin, requester)];
  if (tomb && now < tomb.expiresAt) return "superseded";
  return "none";
}

export function newRequest(
  origin: string,
  hostname: string,
  requester: string,
  now: number,
): AccessRequest {
  return { origin, hostname, requester, requestedAt: now, expiresAt: now + ACCESS_REQUEST_TTL_MS };
}

/** The tombstone a replacement leaves for the displaced requester. */
export function tombstoneFor(record: AccessRequest): SupersededTombstone {
  return { origin: record.origin, requester: record.requester, expiresAt: record.expiresAt };
}
