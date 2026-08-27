import { requesterOriginKey, type AccessQueueEntry, type OriginDecision } from "./access-queue";
import {
  cancelAccess,
  decideAccess,
  enqueueAccess,
  setQueueObserver,
  sweepQueue,
  type QueueSnapshot,
} from "./approval-store";
import { BridgeCommandError, cdp } from "./cdp";
import { safeHttpUrl, storedOriginAllowed } from "./origin-policy";
import { getLocal, getSession, withSessionMutation } from "./state";

export { safeHttpUrl } from "./origin-policy";
export type { OriginDecision } from "./access-queue";

// In-command continuations are keyed by immutable queue generation, never by
// origin: two commands paused on the same redirect must resume independently.
const waiting = new Map<string, (decision: OriginDecision) => void>();
let onPendingChange: ((snapshot: QueueSnapshot) => void) | null = null;

function reconcileWaiters(snapshot: QueueSnapshot): void {
  for (const [entryId, resolve] of waiting) {
    if (snapshot.queue.some((entry) => entry.entryId === entryId)) continue;
    const receipt = Object.values(snapshot.results).find((candidate) => candidate.entryId === entryId);
    if (!receipt) continue;
    waiting.delete(entryId);
    resolve(receipt.state === "allowed" ? "always" : "deny");
  }
}

// Decisions, cancellation, and TTL expiry all fold through durable receipts.
// Install reconciliation when this module loads, not when worker UI wiring is
// attached, so storage integration and restarted-worker paths have the same
// completion guarantee.
setQueueObserver((snapshot) => {
  reconcileWaiters(snapshot);
  onPendingChange?.(snapshot);
});

export function setPendingObserver(observer: (snapshot: QueueSnapshot) => void): void {
  onPendingChange = observer;
}

export async function originAllowed(url: URL): Promise<boolean> {
  const { origins = {} } = await getLocal();
  return storedOriginAllowed(origins, url);
}

export interface OriginAdmission {
  allowed: boolean;
  viaOnceGrant: boolean;
}

export async function consumeOnceGrant(url: URL, requester: string): Promise<boolean> {
  const consumed = await withSessionMutation(async () => {
    const session = await getSession();
    const grants = { ...(session.onceGrants ?? {}) };
    const key = requesterOriginKey(url.origin, requester);
    const grant = grants[key];
    if (!grant || Date.now() >= grant.expiresAt || grant.requester !== requester) return false;
    delete grants[key];
    await chrome.storage.session.set({ onceGrants: grants });
    return true;
  });
  if (consumed) await sweepQueue();
  return consumed;
}

async function consumeGrantFor(url: URL, sessionRequester: string, commandId: string): Promise<boolean> {
  const consumed = await withSessionMutation(async () => {
    const session = await getSession();
    const grants = { ...(session.onceGrants ?? {}) };
    const directKey = requesterOriginKey(url.origin, sessionRequester);
    let key = sessionRequester ? directKey : "";
    let grant = key ? grants[key] : undefined;
    if (!grant) {
      const match = Object.entries(grants).find(
        ([, candidate]) => candidate.origin === url.origin && candidate.handoff === commandId,
      );
      if (match) [key, grant] = match;
    }
    if (!grant || Date.now() >= grant.expiresAt) return false;
    delete grants[key];
    await chrome.storage.session.set({ onceGrants: grants });
    return true;
  });
  if (consumed) await sweepQueue();
  return consumed;
}

export async function ensureTopLevelAccess(
  url: URL,
  requestId: string,
  sessionRequester: string = "",
): Promise<OriginAdmission> {
  if (await originAllowed(url)) return { allowed: true, viaOnceGrant: false };
  if (await consumeGrantFor(url, sessionRequester, requestId)) return { allowed: true, viaOnceGrant: true };
  throw new BridgeCommandError("origin_not_allowed", `site ${url.origin} is not allowed yet`, {
    origin: url.origin,
    url: url.href,
  });
}

async function updatePromptSurfaces(snapshot: QueueSnapshot): Promise<void> {
  const count = snapshot.queue.length;
  await chrome.action.setBadgeBackgroundColor({ color: "#e96042" });
  await chrome.action.setBadgeText({ text: count ? (count > 9 ? "9+" : String(count)) : "" });
  await chrome.action.setTitle({
    title: count === 0 ? "Local Operator" : count === 1 ? "1 site request waiting" : `${count} site requests waiting`,
  });
  onPendingChange?.(snapshot);
}

export async function restoreAccessQueue(): Promise<void> {
  await updatePromptSurfaces(await sweepQueue());
}

export async function askOrigin(url: URL, requestId: string): Promise<boolean> {
  if (await originAllowed(url)) return true;
  if (await consumeOnceGrant(url, requestId)) return true;
  let resolveWaiter!: (decision: OriginDecision) => void;
  const decision = new Promise<OriginDecision>((resolve) => { resolveWaiter = resolve; });
  const queued = await enqueueAccess(url, requestId, "in_command", requestId, (entry) => {
    // enqueueAccess invokes this callback under the same mutation lock that
    // publishes the entry. No decision or expiry sweep can interleave between
    // visibility and registration.
    waiting.set(entry.entryId, resolveWaiter);
  });
  if (!queued.entry) return false;
  await updatePromptSurfaces(await sweepQueue());
  return (await decision) !== "deny";
}

export async function resolveOrigin(
  _origin: string,
  decision: OriginDecision,
  entryId: string = "",
): Promise<boolean> {
  if (!entryId) return false;
  const result = await decideAccess(entryId, decision);
  if (!result.applied) return false;
  // decideAccess persisted receipts before releasing the mutation lock; the
  // queue observer resolves every matching waiter from those receipts.
  reconcileWaiters(result.snapshot);
  await updatePromptSurfaces(result.snapshot);
  return true;
}

// Compatibility exports keep an older worker/popup path from crashing during
// a rolling extension update; the queue itself is authoritative.
export async function clearPrompt(): Promise<void> { await updatePromptSurfaces(await sweepQueue()); }
export async function expireAccessRequest(): Promise<void> { await restoreAccessQueue(); }
export async function raiseAccessRequest(url: URL, requester: string): Promise<AccessQueueEntry> {
  const result = await enqueueAccess(url, requester, "async");
  if (!result.entry) throw new BridgeCommandError("access_queue_full", "site approval queue is full", { pending_count: result.pending_count ?? 16 });
  await updatePromptSurfaces(await sweepQueue());
  return result.entry;
}
export { cancelAccess, sweepQueue };

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
