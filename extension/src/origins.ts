import { reconcileActionSurface } from "./action-surface";
import { liveQueue, requesterOriginKey, type AccessQueueEntry, type OriginDecision } from "./access-queue";
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
// Each generation can hold SEVERAL resolvers: enqueueAccess dedupes a retried
// same-command navigation onto the existing entry (approval-store), and every
// paused document waiting on that entry registered its own resolver here — a
// single-slot map would overwrite the first and strand its navigation.
const waiting = new Map<string, Set<(decision: OriginDecision) => void>>();
let onPendingChange: ((snapshot: QueueSnapshot) => void) | null = null;

function reconcileWaiters(snapshot: QueueSnapshot): void {
  for (const [entryId, resolvers] of waiting) {
    if (snapshot.queue.some((entry) => entry.entryId === entryId)) continue;
    const receipt = Object.values(snapshot.results).find((candidate) => candidate.entryId === entryId);
    if (!receipt) continue;
    waiting.delete(entryId);
    const decision: OriginDecision = receipt.state === "allowed" ? "always" : "deny";
    for (const resolve of resolvers) resolve(decision);
  }
}

// Decisions, cancellation, and TTL expiry all fold through durable receipts.
// Install reconciliation when this module loads, not when worker UI wiring is
// attached, so storage integration and restarted-worker paths have the same
// completion guarantee.
setQueueObserver((snapshot) => {
  reconcileWaiters(snapshot);
  // The queue store deliberately does not await observers while holding its
  // mutation lock. Catch here so a Chrome API rejection cannot become an
  // unhandled worker promise; reconcileActionSurface logs per-operation detail.
  void updatePromptSurfaces(snapshot).catch((error) =>
    console.warn("approval action surface reconciliation failed", error),
  );
});

export function setPendingObserver(observer: (snapshot: QueueSnapshot) => void): void {
  onPendingChange = observer;
}

export async function originAllowed(url: URL): Promise<boolean> {
  const { origins = {}, hostGrants } = await getLocal();
  return storedOriginAllowed(origins, url, hostGrants);
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

export async function updatePromptSurfaces(snapshot: QueueSnapshot): Promise<void> {
  await reconcileActionSurface(snapshot, onPendingChange ?? undefined);
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
    // visibility and registration. The callback also fires for a DEDUPED
    // enqueue (retried same-command navigation), so several resolvers can
    // attach to one generation — see the `waiting` map's comment.
    const resolvers = waiting.get(entry.entryId) ?? new Set<(decision: OriginDecision) => void>();
    resolvers.add(resolveWaiter);
    waiting.set(entry.entryId, resolvers);
  });
  if (!queued.entry) return false;
  await updatePromptSurfaces(await sweepQueue());
  return (await decision) !== "deny";
}

export async function resolveOrigin(
  origin: string,
  decision: OriginDecision,
  entryId: string = "",
): Promise<boolean> {
  if (!["once", "always", "all_ports", "deny"].includes(decision)) return false;
  let targetId = entryId;
  const session = await getSession();
  const queue = liveQueue(session.accessQueue, Date.now());
  if (!targetId || !queue.some((entry) => entry.entryId === targetId)) {
    // Generation miss: a /health-fallback render carries no entry id (queue
    // storage empty after a worker restart, or the entry expired while the
    // daemon still echoes it), and a stale id means the entry was replaced.
    // The click still names the ORIGIN the user looked at, so honour it
    // against that origin's single live entry — the mismatch came from a
    // fallback render, not a replaced prompt. Zero or several matches stay
    // REJECTED: with several live entries for the origin the click cannot say
    // which paused navigation the user meant, so round-2 B1's consent hole
    // stays closed.
    const match = queue.filter((entry) => entry.origin === origin);
    if (match.length !== 1) return false;
    targetId = match[0]?.entryId ?? "";
    if (!targetId) return false;
  }
  const result = await decideAccess(targetId, decision);
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
