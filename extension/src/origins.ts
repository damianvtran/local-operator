import {
  activeRequest,
  consumableGrant,
  newRequest,
  ONCE_GRANT_TTL_MS,
  receiptKey,
  TOMBSTONE_CAP,
  tombstoneFor,
  type AccessRequest,
  type OriginDecision,
} from "./access-flow";
import { BridgeCommandError, cdp } from "./cdp";
import { safeHttpUrl, storedOriginAllowed } from "./origin-policy";
import { ORIGIN_PROMPT_TIMEOUT_MS } from "./protocol.gen";
import { getLocal, getSession, withSessionMutation } from "./state";

export { safeHttpUrl } from "./origin-policy";
export { ORIGIN_PROMPT_TIMEOUT_MS } from "./protocol.gen";
export type { OriginDecision } from "./access-flow";

// The popup resolves a decision by ORIGIN, not by command id: a single command
// can pause on several origins in a redirect chain, and keying the resolver by
// command id let the second origin overwrite the first's resolver so only the
// last was answerable (finding A6). Keying by origin lets each hop resolve
// independently, and the popup only ever shows one origin at a time anyway.
const waiting = new Map<string, (decision: OriginDecision) => void>();

// A hook the worker installs so a pending decision can raise an ambient signal
// (a system notification) the user sees without already having the popup open
// (finding U2). Kept injectable so origins.ts stays free of worker wiring.
let onPendingChange: ((pending: { origin: string; hostname: string } | null) => void) | null = null;
export function setPendingObserver(
  observer: (pending: { origin: string; hostname: string } | null) => void,
): void {
  onPendingChange = observer;
}

export async function originAllowed(url: URL): Promise<boolean> {
  const { origins = {} } = await getLocal();
  return storedOriginAllowed(origins, url);
}

/** The outcome of the ONE admission decision a top-level navigation makes.
 * {allowed:true} covers both the persistent allowlist and a consumed
 * once-grant; navigate() trusts it and never re-consults the grant map, so
 * admission and consumption cannot drift apart across the TTL (round-1 M1). */
export interface OriginAdmission {
  allowed: boolean;
  /** True when admission consumed a once-grant — kept for diagnostics and
   * tests; navigate() only needs `allowed`. */
  viaOnceGrant: boolean;
}

/** Consume an unconsumed async "Allow once" grant for this origin, if the
 * CALLER may. Consumed (deleted) on use because "once" means one navigation.
 * Fails CLOSED on a requester mismatch: a grant minted for session A's flow
 * is not session B's to spend (see the multi-surface constraint in
 * access-flow.ts). */
export function consumeOnceGrant(url: URL, requester: string): Promise<boolean> {
  // Atomic under the session-mutation queue: read-check-delete as one unit,
  // or two concurrent navigations could both read the grant before either
  // delete landed and double-spend a one-shot approval (round-2 B2).
  return withSessionMutation(async () => {
    const { onceGrants = {} } = await getSession();
    const grant = consumableGrant(onceGrants, url.origin, requester, Date.now());
    if (!grant) return false;
    const remaining = { ...onceGrants };
    delete remaining[url.origin];
    await chrome.storage.session.set({ onceGrants: remaining });
    await clearConsumedRequest(url.origin, grant.requester);
    return true;
  });
}

/** Whether a top-level open/goto to this origin may START, making the
 * admission decision EXACTLY ONCE: stored allowlist → caller's once-grant
 * (consumed here, so it cannot lapse into a prompt mid-command) → typed
 * early error teaching the agent the async flow. The returned admission
 * travels to navigate() unchanged; the whole point of the early error is
 * that the old behaviour (block the navigation RPC inside the popup prompt)
 * expired unseen and made sessions misread the bridge as broken. Redirect
 * hops inside a running navigation still take the synchronous askOrigin
 * prompt: the command is already in flight there, so an early fail is
 * impossible by construction.
 *
 * Grant consumption matches EITHER the session identity the tool attached to
 * this command (same session that asked, normal path) OR the command-id
 * handoff recorded on the grant at decision time (raw-RPC callers reusing
 * the request id). A navigation carrying NEITHER is another session trying
 * to ride a grant it did not earn — refused (fail-closed). */
export async function ensureTopLevelAccess(
  url: URL,
  requestId: string,
  sessionRequester: string = "",
): Promise<OriginAdmission> {
  if (await originAllowed(url)) return { allowed: true, viaOnceGrant: false };
  if (await consumeGrantFor(url, sessionRequester, requestId)) {
    return { allowed: true, viaOnceGrant: true };
  }
  throw new BridgeCommandError("origin_not_allowed", `site ${url.origin} is not allowed yet`, {
    origin: url.origin,
    url: url.href,
  });
}

/** Consume a grant on behalf of a NAVIGATION command: the caller's session
 * identity must equal the grant's requester, or the command's own id must
 * equal the grant's handoff (see recordAccessDecision). */
function consumeGrantFor(
  url: URL,
  sessionRequester: string,
  commandId: string,
): Promise<boolean> {
  // Same atomicity as consumeOnceGrant (round-2 B2): #321 gives different
  // tabs different daemon locks and the worker dispatches frames
  // concurrently, so two same-session navigations genuinely race here.
  // Exactly one caller may win the delete.
  return withSessionMutation(async () => {
    const { onceGrants = {} } = await getSession();
    const grant = onceGrants[url.origin];
    if (!grant || Date.now() >= grant.expiresAt) return false;
    const ownsBySession = sessionRequester !== "" && grant.requester === sessionRequester;
    const ownsByHandoff = !!grant.handoff && grant.handoff === commandId;
    if (!ownsBySession && !ownsByHandoff) return false;
    const remaining = { ...onceGrants };
    delete remaining[url.origin];
    await chrome.storage.session.set({ onceGrants: remaining });
    await clearConsumedRequest(url.origin, grant.requester);
    return true;
  });
}

/** A resolved "once" request is the receipt await_access reads while its grant
 * waits to be spent. Once the grant is consumed, remove that matching receipt:
 * otherwise a later request_access to the same origin sees decision="once"
 * and answers allowed even though no grant remains — silently turning "once"
 * into a second admission. A newer request is preserved by the full match. */
async function clearConsumedRequest(origin: string, requester: string): Promise<void> {
  const { accessRequest } = await getSession();
  if (
    accessRequest?.origin === origin &&
    accessRequest.decision === "once" &&
    accessRequest.requester === requester
  ) {
    await chrome.storage.session.remove("accessRequest");
  }
}

/** Raise the pending-approval surfaces: session record (popup prompt), badge,
 * and — via the worker's observer — the system notification and the daemon's
 * awaiting_origin event. Shared by the in-command prompt and the async
 * request_access path so the popup renders both identically. */
async function raisePrompt(url: URL, requestId: string): Promise<string> {
  // Every (re)raise mints a fresh prompt generation. The popup binds its
  // rendered view AND its decision message to this id, and resolveOrigin
  // rejects a decision carrying a stale one — the guard that stops a popup
  // still showing origin A from approving whatever origin B replaced the
  // slot with (round-2 B1).
  const promptId = crypto.randomUUID().replaceAll("-", "");
  await chrome.storage.session.set({
    pendingOrigin: { origin: url.origin, hostname: url.hostname, requestId, promptId },
  });
  await chrome.action.setBadgeBackgroundColor({ color: "#e96042" });
  await chrome.action.setBadgeText({ text: "!" });
  await chrome.action.setTitle({ title: `Local Operator wants to open ${url.hostname}` });
  onPendingChange?.({ origin: url.origin, hostname: url.hostname });
  return promptId;
}

async function clearPrompt(): Promise<void> {
  await chrome.storage.session.remove("pendingOrigin");
  await chrome.action.setBadgeText({ text: "" });
  await chrome.action.setTitle({ title: "Local Operator" });
  onPendingChange?.(null);
}

export { clearPrompt, raisePrompt };

export async function askOrigin(url: URL, requestId: string): Promise<boolean> {
  if (await originAllowed(url)) return true;
  // A live async grant minted for THIS command's request id covers exactly
  // this navigation. Redirect hops carry the top-level command's id, which
  // is NOT the async requester, so they correctly miss and fall through to
  // the prompt — a hop must never ride a grant another flow earned.
  if (await consumeOnceGrant(url, requestId)) return true;
  // Record the pending decision for the popup and the ambient observer. Keyed
  // by origin so concurrent hops do not clobber each other.
  await raisePrompt(url, requestId);
  const decision = await new Promise<OriginDecision>((resolve) => {
    waiting.set(url.origin, resolve);
    setTimeout(() => {
      if (waiting.delete(url.origin)) resolve("deny");
    }, ORIGIN_PROMPT_TIMEOUT_MS);
  });
  await clearPrompt();
  if (decision === "always") {
    const { origins = {} } = await getLocal();
    await chrome.storage.local.set({ origins: { ...origins, [url.origin]: "allow" } });
  }
  return decision !== "deny";
}

/** Fold the user's decision into a live async access request — the
 * NON-LOCKING body (suffix `Locked`: caller already holds the session
 * mutation queue). Splitting the old self-locking helper is what makes the
 * round-3 B1 fix possible: resolveOrigin's validation and this mutation now
 * share one critical section, so a queued replacement cannot slip between
 * them. Returns false when there is nothing left to apply (no live record,
 * wrong origin, or already decided — the duplicate-delivery idempotence).
 * This is the second half of the decoupling: the popup's one decision message
 * must land on BOTH an in-command wait (resolveOrigin's map) and the async
 * record, because the agent may be waiting either way and the popup cannot
 * tell which. Persisting "always" and minting the "once" grant happen here
 * for the async path — there is no askOrigin continuation to do it. */
async function recordAccessDecisionLocked(
  origin: string,
  decision: OriginDecision,
  session: { accessRequest?: AccessRequest; onceGrants?: import("./access-flow").OnceGrants; pendingOrigin?: { requestId?: string } },
): Promise<boolean> {
  const now = Date.now();
  const live = activeRequest(session.accessRequest, now);
  if (!live || live.origin !== origin || live.decision) return false;
  await chrome.storage.session.set({ accessRequest: { ...live, decision } });
  if (decision === "once") {
    // The grant is bound to the REQUEST's requester AND to the command that
    // raised the request: an async "Allow once" is earned by the flow that
    // asked, so the handoff records BOTH identities. A navigation spends it
    // if it carries EITHER — the session identity (normal path: the same
    // session's next open/goto) or the exact command id (a raw-RPC caller
    // whose navigation reuses the request_access id). A third session
    // carries neither and is refused (fail-closed; see access-flow.ts).
    const grants = session.onceGrants ?? {};
    await chrome.storage.session.set({
      onceGrants: {
        ...grants,
        [origin]: {
          expiresAt: now + ONCE_GRANT_TTL_MS,
          requester: live.requester,
          handoff: session.pendingOrigin?.requestId ?? "",
        },
      },
    });
  }
  if (decision === "always") {
    const { origins = {} } = await getLocal();
    await chrome.storage.local.set({ origins: { ...origins, [origin]: "allow" } });
  }
  // The async prompt has no waiting command to clean up after itself; clear
  // the badge/notification here so the "!" does not outlive the decision.
  await clearPrompt();
  return true;
}

/** Apply the popup's decision — IF it names the live prompt generation.
 *
 * Returns a promise that settles only when the decision is durably recorded
 * (record + grant + allowlist + prompt teardown), so the worker's message
 * listener can keep the MV3 event alive until persistence completes instead
 * of racing worker suspension (round-2 M2). Resolves true when applied,
 * false when rejected as stale.
 *
 * Validation AND persistence run in ONE withSessionMutation critical section
 * (round-3 B1): validating before entering the queue was a TOCTOU — a
 * same-origin replacement queued behind the stale decision's validation
 * installed its fresh generation after the check passed but before
 * recordAccessDecision ran, so the stale click applied to (and minted a grant
 * for) the NEW request. Inside the queue the pending prompt is re-read
 * atomically with the mutation that consumes it; the in-command waiter is
 * resolved only after that atomic validation succeeds. A duplicate delivery
 * of the CURRENT generation is idempotent: the record already carries the
 * decision, so the re-read sees `live.decision` set and nothing re-applies
 * (`applied:false` — the click already landed once; a second landing must be
 * a no-op, and the caller distinguishes "this delivery did something" from
 * "the decision exists" via the record, not the return value). */
export function resolveOrigin(
  origin: string,
  decision: OriginDecision,
  promptId: string = "",
): Promise<boolean> {
  return withSessionMutation(async () => {
    // Stale rejection (round-2 B1): the popup binds its click to the promptId
    // it RENDERED. If another request replaced the slot after that render,
    // nothing resolves, no grant is minted — the user's click was an answer
    // to a question that is no longer being asked. A missing promptId
    // (legacy popup / health-fallback render) still requires the ORIGIN to
    // match the live prompt, the pre-generation guard.
    const session = await getSession();
    const pendingOrigin = session.pendingOrigin;
    if (!pendingOrigin || pendingOrigin.origin !== origin) return false;
    if (promptId && pendingOrigin.promptId && promptId !== pendingOrigin.promptId) return false;
    const applied = await recordAccessDecisionLocked(origin, decision, session);
    if (!applied) return false;
    // In-command wait AFTER atomic validation: the waiter must never be
    // resolved by a decision the queue just rejected as stale.
    const resolve = waiting.get(origin);
    if (resolve) {
      waiting.delete(origin);
      resolve(decision);
    }
    return true;
  });
}

/** Lazy TTL cleanup, called from the worker's expiry alarm and on every
 * access read: an expired request must drop its prompt surfaces too, or the
 * popup would keep offering three buttons whose clicks resolve nothing.
 * Returns the record it swept so callers can re-arm the expiry alarm for
 * whatever is next (round-1 m2: a replace-don't-queue overwrite used to drop
 * the OLD request's alarm entirely). */
export function expireAccessRequest(): Promise<void> {
  // Under the mutation queue: the sweep read-modify-writes the same record
  // the decision and consume paths do.
  return withSessionMutation(async () => {
    const session = await getSession();
    const record = session.accessRequest;
    if (!record || activeRequest(record, Date.now())) return;
    await chrome.storage.session.remove("accessRequest");
    const { pendingOrigin } = session;
    // Only clear the prompt if it is OURS — an in-command prompt for another
    // origin may be live and owns its own cleanup.
    if (pendingOrigin && pendingOrigin.origin === record.origin) await clearPrompt();
  });
}

/** Raise a fresh async request, tombstoning any live one it replaces.
 * The displaced requester gets a receipt (see AccessTombstones in
 * access-flow.ts) so its next poll learns "superseded" instead of reading a
 * silent absence as expiry — without it, two sessions could steal the single
 * prompt slot back and forth forever (round-1 B1b). The new prompt's surfaces
 * are raised immediately, overwriting the replaced one's badge/record
 * atomically (round-1 m2: a dead prompt's badge must not outlive it). */
export function raiseAccessRequest(url: URL, requester: string): Promise<AccessRequest> {
  return withSessionMutation(async () => {
  const now = Date.now();
  const session = await getSession();
  const previous = activeRequest(session.accessRequest, now);
  const record = newRequest(url.origin, url.hostname, requester, now);
  const tombs = { ...(session.accessTombstones ?? {}) };
  // A deliberate fresh request by the requester that was displaced consumes
  // its old supersession receipt. Leaving it behind made this NEW request's
  // eventual TTL expiry read as "superseded" instead of "none" — the stale
  // receipt outlived the event it described.
  delete tombs[receiptKey(record.origin, requester)];
  if (previous && (previous.origin !== record.origin || previous.requester !== record.requester)) {
    // Different origin OR same origin from a different requester: both
    // displace the pending prompt, both leave a receipt. Keyed per
    // origin+requester so an A→B→C chain preserves EVERY displaced
    // requester's receipt instead of overwriting A's with B's (round-2 M1).
    tombs[receiptKey(previous.origin, previous.requester)] = tombstoneFor(previous);
    // Drop expired/oldest beyond the cap so a churny fleet cannot grow the
    // map unbounded — a receipt that old describes a prompt nobody remembers.
    const entries = Object.entries(tombs)
      .filter(([, t]) => now < t.expiresAt)
      .sort((a, b) => b[1].expiresAt - a[1].expiresAt)
      .slice(0, TOMBSTONE_CAP);
    await chrome.storage.session.set({ accessTombstones: Object.fromEntries(entries) });
  } else {
    // Persist the consumed stale receipt even when this request displaced
    // nothing live.
    await chrome.storage.session.set({ accessTombstones: tombs });
  }
  await chrome.storage.session.set({ accessRequest: record });
  await raisePrompt(url, requester);
  // The expiry alarm is armed by the CALLER (access.ts) with this record's
  // own expiry; chrome.alarms.create replaces a same-named alarm, so the
  // displaced request's alarm is intentionally dropped — the new record's
  // alarm covers the only live prompt now, and expireAccessRequest's lazy
  // sweep handles anything older.
  return record;
  });
}

interface PausedRequest {
  requestId: string;
  request: { url: string };
  resourceType: string;
}

export async function withOriginGate<T>(
  tabId: number,
  commandId: string,
  operation: () => Promise<T>,
  alreadyAllowed: string[] = [],
): Promise<T> {
  // Fetch interception is the security boundary for redirects and trusted
  // clicks. webNavigation fires after Chrome has begun a request; pausing the
  // main document here means an unapproved origin receives no cookies or page
  // request before the browser-owned popup decision.
  const once = new Set(alreadyAllowed);
  const decisions: Promise<void>[] = [];
  let denied: BridgeCommandError | undefined;
  const listener = (
    source: chrome.debugger.Debuggee,
    method: string,
    raw: object | undefined,
  ): void => {
    if (source.tabId !== tabId || method !== "Fetch.requestPaused") return;
    const event = raw as PausedRequest;
    const decision = (async () => {
      let url: URL;
      try {
        url = safeHttpUrl(event.request.url);
      } catch (error) {
        denied = new BridgeCommandError("origin_denied", String(error));
        await cdp(tabId, "Fetch.failRequest", { requestId: event.requestId, errorReason: "BlockedByClient" });
        return;
      }
      if (!once.has(url.origin) && !(await originAllowed(url))) {
        if (!(await askOrigin(url, commandId))) {
          denied = new BridgeCommandError("origin_denied", "site permission was denied", { origin: url.origin });
          await cdp(tabId, "Fetch.failRequest", { requestId: event.requestId, errorReason: "BlockedByClient" });
          return;
        }
        once.add(url.origin);
      }
      await cdp(tabId, "Fetch.continueRequest", { requestId: event.requestId });
    })();
    decisions.push(decision);
  };
  await cdp(tabId, "Fetch.enable", { patterns: [{ resourceType: "Document", requestStage: "Request" }] });
  chrome.debugger.onEvent.addListener(listener);
  try {
    const result = await operation();
    await Promise.all(decisions);
    if (denied) throw denied;
    return result;
  } finally {
    chrome.debugger.onEvent.removeListener(listener);
    try { await cdp(tabId, "Fetch.disable"); } catch { /* a closed tab needs no cleanup */ }
  }
}
