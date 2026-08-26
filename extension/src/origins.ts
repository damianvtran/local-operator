import {
  activeRequest,
  anyGrantLive,
  consumableGrant,
  newRequest,
  ONCE_GRANT_TTL_MS,
  supersedeRequest,
  type OriginDecision,
} from "./access-flow";
import { BridgeCommandError, cdp } from "./cdp";
import { safeHttpUrl, storedOriginAllowed } from "./origin-policy";
import { ORIGIN_PROMPT_TIMEOUT_MS } from "./protocol.gen";
import { getLocal, getSession } from "./state";

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
export async function consumeOnceGrant(url: URL, requester: string): Promise<boolean> {
  const { onceGrants = {} } = await getSession();
  if (!consumableGrant(onceGrants, url.origin, requester, Date.now())) return false;
  const remaining = { ...onceGrants };
  delete remaining[url.origin];
  await chrome.storage.session.set({ onceGrants: remaining });
  return true;
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
 * impossible by construction. */
export async function ensureTopLevelAccess(url: URL, requester: string): Promise<OriginAdmission> {
  if (await originAllowed(url)) return { allowed: true, viaOnceGrant: false };
  if (await consumeOnceGrant(url, requester)) return { allowed: true, viaOnceGrant: true };
  throw new BridgeCommandError("origin_not_allowed", `site ${url.origin} is not allowed yet`, {
    origin: url.origin,
    url: url.href,
  });
}

/** Raise the pending-approval surfaces: session record (popup prompt), badge,
 * and — via the worker's observer — the system notification and the daemon's
 * awaiting_origin event. Shared by the in-command prompt and the async
 * request_access path so the popup renders both identically. */
async function raisePrompt(url: URL, requestId: string): Promise<void> {
  await chrome.storage.session.set({
    pendingOrigin: { origin: url.origin, hostname: url.hostname, requestId },
  });
  await chrome.action.setBadgeBackgroundColor({ color: "#e96042" });
  await chrome.action.setBadgeText({ text: "!" });
  await chrome.action.setTitle({ title: `Local Operator wants to open ${url.hostname}` });
  onPendingChange?.({ origin: url.origin, hostname: url.hostname });
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

/** Fold the user's decision into a live async access request, if one matches.
 * This is the second half of the decoupling: the popup's one decision message
 * must land on BOTH an in-command wait (resolveOrigin's map) and the async
 * record, because the agent may be waiting either way and the popup cannot
 * tell which. Persisting "always" and minting the "once" grant happen here
 * for the async path — there is no askOrigin continuation to do it. */
async function recordAccessDecision(origin: string, decision: OriginDecision): Promise<void> {
  const now = Date.now();
  const session = await getSession();
  const live = activeRequest(session.accessRequest, now);
  if (!live || live.origin !== origin || live.decision) return;
  await chrome.storage.session.set({ accessRequest: { ...live, decision } });
  if (decision === "once") {
    // The grant is bound to the REQUEST's requester: "Allow once" is the
    // user's answer to this agent's ask, so only a command carrying that
    // requester may spend it (see access-flow.ts).
    const grants = session.onceGrants ?? {};
    await chrome.storage.session.set({
      onceGrants: {
        ...grants,
        [origin]: { expiresAt: now + ONCE_GRANT_TTL_MS, requester: live.requester },
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
}

export function resolveOrigin(origin: string, decision: OriginDecision): void {
  const resolve = waiting.get(origin);
  if (resolve) {
    waiting.delete(origin);
    resolve(decision);
  }
  // Fire-and-forget: the in-command resolution above must stay synchronous
  // (the worker's onMessage listener is not awaited), and the async record
  // update has no caller to report to.
  void recordAccessDecision(origin, decision);
}

/** Lazy TTL cleanup, called from the worker's expiry alarm and on every
 * access read: an expired request must drop its prompt surfaces too, or the
 * popup would keep offering three buttons whose clicks resolve nothing.
 * Returns the record it swept so callers can re-arm the expiry alarm for
 * whatever is next (round-1 m2: a replace-don't-queue overwrite used to drop
 * the OLD request's alarm entirely). */
export async function expireAccessRequest(): Promise<void> {
  const session = await getSession();
  const record = session.accessRequest;
  if (!record || activeRequest(record, Date.now())) return;
  await chrome.storage.session.remove("accessRequest");
  const { pendingOrigin } = session;
  // Only clear the prompt if it is OURS — an in-command prompt for another
  // origin may be live and owns its own cleanup.
  if (pendingOrigin && pendingOrigin.origin === record.origin) await clearPrompt();
}

/** Raise a fresh async request, tombstoning any live one it replaces.
 * The displaced request is NOT deleted: its record stays with the
 * "superseded" sentinel so the displaced requester's next poll learns its
 * prompt was taken over (round-1 B1b) instead of reading a silent absence as
 * expiry. The replaced prompt surfaces are cleared immediately — waiting for
 * the lazy sweep let a dead prompt's badge outlive it (round-1 m2). */
export async function raiseAccessRequest(url: URL, requester: string): Promise<AccessRequest> {
  const now = Date.now();
  const session = await getSession();
  const record = newRequest(url.origin, url.hostname, requester, now);
  const previous = activeRequest(session.accessRequest, now);
  const replacedOwnPrompt =
    previous &&
    previous.origin !== record.origin &&
    session.pendingOrigin?.origin === previous.origin;
  if (previous && previous.origin !== record.origin) {
    await chrome.storage.session.set({ accessRequest: supersedeRequest(previous) });
  }
  await chrome.storage.session.set({ accessRequest: record });
  if (replacedOwnPrompt) {
    // The old prompt record is stale the moment the new one lands; raise
    // the new one over it, which overwrites badge/title atomically.
  }
  await raisePrompt(url, requester);
  // The expiry alarm is armed by the CALLER (access.ts) with this record's
  // own expiry; chrome.alarms.create replaces a same-named alarm, so the
  // displaced request's alarm is intentionally dropped — the new record's
  // alarm covers the only live prompt now, and expireAccessRequest's lazy
  // sweep handles anything older.
  return record;
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
