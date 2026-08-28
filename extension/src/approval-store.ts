import {
  ACCESS_QUEUE_CAP,
  ACCESS_QUEUE_VERSION,
  ONCE_GRANT_TTL_MS,
  cleanResults,
  findPending,
  liveQueue,
  newEntry,
  policyCovers,
  queuePosition,
  receiptFor,
  receiptForRequester,
  requesterOriginKey,
  resultKey,
  type AccessKind,
  type AccessQueueEntry,
  type AccessReceipt,
  type AccessResults,
  type AccessState,
  type OnceGrants,
  type OriginDecision,
} from "./access-queue";
import { grantExactOriginLocked, grantLoopbackHostLocked } from "./access-grants";
import { isLoopbackHost } from "./origin-policy";
import { getLocal, getSession, withSessionMutation, type SessionState } from "./state";

export interface QueueSnapshot {
  queue: AccessQueueEntry[];
  results: AccessResults;
  onceGrants: OnceGrants;
}

export interface QueueStatus {
  origin: string;
  state: AccessState;
  entryId?: string;
  position?: number;
  pending_count?: number;
  expires_at?: number;
}

export interface EnqueueResult extends QueueStatus {
  entry?: AccessQueueEntry;
  full?: boolean;
}

export const ACCESS_EXPIRY_ALARM = "lop-access-expiry";

export type QueueObserver = (snapshot: QueueSnapshot) => void;
let observer: QueueObserver | null = null;
export function setQueueObserver(next: QueueObserver): void {
  observer = next;
}

/** One alarm owns every ephemeral approval deadline. Recomputing from durable
 * state after each mutation prevents a later 10-minute request from replacing
 * an earlier 60-second redirect deadline. */
export function nextAccessExpiry(snapshot: QueueSnapshot): number | undefined {
  const deadlines = [
    ...snapshot.queue.map((entry) => entry.expiresAt),
    ...Object.values(snapshot.results).map((receipt) => receipt.expiresAt),
    ...Object.values(snapshot.onceGrants).map((grant) => grant.expiresAt),
  ];
  return deadlines.length ? Math.min(...deadlines) : undefined;
}

export async function armNextExpiry(snapshot: QueueSnapshot): Promise<void> {
  const when = nextAccessExpiry(snapshot);
  if (when === undefined) {
    await chrome.alarms.clear(ACCESS_EXPIRY_ALARM);
  } else {
    chrome.alarms.create(ACCESS_EXPIRY_ALARM, { when });
  }
}

function legacyRequester(record: { requester?: string; requestId?: string } | undefined): string {
  return record?.requester || record?.requestId || "legacy";
}

/** Import #329's single slot inside the same mutation that first reads the new
 * queue. Removing the old keys makes the conversion exactly-once even when two
 * commands wake a restarted worker together. */
async function normalizedLocked(now: number): Promise<QueueSnapshot> {
  const session = await getSession();
  let queue = liveQueue(session.accessQueue, now);
  let results = cleanResults(session.accessResults, now);
  let onceGrants = Object.fromEntries(
    Object.entries(session.onceGrants ?? {}).filter(([, grant]) => now < grant.expiresAt),
  ) as OnceGrants;
  const needsMigration = session.accessQueueVersion !== ACCESS_QUEUE_VERSION;
  if (needsMigration) {
    const legacy = session.accessRequest;
    const pending = session.pendingOrigin;
    const legacyLive = !!legacy && now < legacy.expiresAt;
    const legacyOwner = legacyRequester(legacy);
    const pendingOwner = legacyRequester(pending);
    // The old async receipt and popup slot were independently writable. Merge
    // only when both authority keys match; otherwise preserve both generations
    // so an A async request plus a B redirect cannot corrupt or erase either.
    const matching =
      legacyLive &&
      !!pending &&
      legacy!.origin === pending.origin &&
      legacyOwner === pendingOwner;
    let sequence = Math.max(0, ...queue.map((entry) => entry.sequence));
    if (legacyLive) {
      const entry = newEntry(
        legacy!.origin,
        legacy!.hostname,
        legacyOwner,
        "async",
        legacy!.requestedAt,
        ++sequence,
        matching ? pending?.requestId : undefined,
        matching ? pending?.promptId : undefined,
      );
      entry.expiresAt = legacy!.expiresAt;
      if (legacy!.decision) {
        const state = legacy!.decision === "deny" ? "denied" : "allowed";
        const receipt = receiptFor(entry, state, now);
        results[resultKey(entry.entryId, legacyOwner)] = receipt;
      } else queue.push(entry);
    }
    if (pending && !matching) {
      queue.push(
        newEntry(
          pending.origin,
          pending.authority,
          pendingOwner,
          "in_command",
          now,
          ++sequence,
          pending.requestId,
          pending.promptId,
        ),
      );
    }
    // Legacy grants were origin-keyed. Re-key without broadening them: the
    // embedded requester remains the authority boundary.
    for (const [key, grant] of Object.entries(onceGrants)) {
      if (!key.includes("\n")) {
        delete onceGrants[key];
        onceGrants[requesterOriginKey(key, grant.requester)] = { ...grant, origin: key };
      }
    }
  }
  queue.sort((a, b) => a.sequence - b.sequence);
  await chrome.storage.session.set({
    accessQueueVersion: ACCESS_QUEUE_VERSION,
    accessQueue: queue,
    accessResults: cleanResults(results, now),
    onceGrants,
  });
  if (needsMigration) await chrome.storage.session.remove(["accessRequest", "pendingOrigin"]);
  const snapshot = { queue, results: cleanResults(results, now), onceGrants };
  await armNextExpiry(snapshot);
  return snapshot;
}

async function persistLocked(snapshot: QueueSnapshot): Promise<void> {
  await chrome.storage.session.set({
    accessQueueVersion: ACCESS_QUEUE_VERSION,
    accessQueue: snapshot.queue,
    accessResults: snapshot.results,
    onceGrants: snapshot.onceGrants,
  });
  await armNextExpiry(snapshot);
}

export function sweepQueue(now: number = Date.now()): Promise<QueueSnapshot> {
  return withSessionMutation(async () => {
    // Capture expired generations before normalization filters them. Their
    // receipts are what turn a timed-out paused navigation into a typed denial
    // rather than making it vanish as if it never existed.
    const raw = await getSession();
    const expired = raw.accessQueue?.filter((entry) => now >= entry.expiresAt) ?? [];
    const snapshot = await normalizedLocked(now);
    for (const entry of expired) {
      const receipt = receiptFor(entry, entry.kind === "in_command" ? "denied" : "cancelled", now);
      snapshot.results[resultKey(entry.entryId, entry.requester)] = receipt;
    }
    snapshot.results = cleanResults(snapshot.results, now);
    await persistLocked(snapshot);
    observer?.(snapshot);
    return snapshot;
  });
}

export function enqueueAccess(
  url: URL,
  requester: string,
  kind: AccessKind,
  commandId?: string,
  onAdmitted?: (entry: AccessQueueEntry) => void,
): Promise<EnqueueResult> {
  return withSessionMutation(async () => {
    const now = Date.now();
    const snapshot = await normalizedLocked(now);
    // Async requests deduplicate on origin+requester: one narrated question is
    // asked once. In-command requests deduplicate ONLY on
    // origin+requester+commandId — a retried navigation re-asks the identical
    // question and must not stack a twin the user has to answer twice
    // (approving one used to leave the other looping the popup on
    // "Request changed."). Two paused documents under DIFFERENT commands keep
    // separate generations so each command resumes on its own consent; a
    // same-command twin shares the one generation the user is looking at, and
    // one Allow resumes every waiter attached to it — exactly the consent
    // given, never a broader grant.
    const existing =
      kind === "async"
        ? findPending(snapshot.queue, url.origin, requester, kind, now)
        : snapshot.queue.find(
            (entry) =>
              entry.kind === "in_command" &&
              entry.origin === url.origin &&
              entry.requester === requester &&
              !!commandId &&
              entry.commandId === commandId,
          );
    if (existing) {
      // A deduplicated generation must carry the new caller's waiter too:
      // onAdmitted otherwise fires only on first admission, and an in-command
      // caller whose waiter never registers would await a promise nothing can
      // resolve (the paused document frozen forever).
      onAdmitted?.(existing);
      await armNextExpiry(snapshot);
      return {
        origin: url.origin,
        state: "pending",
        entryId: existing.entryId,
        position: queuePosition(snapshot.queue, existing.entryId)!,
        pending_count: snapshot.queue.length,
        expires_at: existing.expiresAt,
        entry: existing,
      };
    }
    if (snapshot.queue.length >= ACCESS_QUEUE_CAP) {
      return { origin: url.origin, state: "none", pending_count: snapshot.queue.length, full: true };
    }
    const nextSequence = Math.max(0, ...snapshot.queue.map((entry) => entry.sequence)) + 1;
    const entry = newEntry(url.origin, url.host, requester, kind, now, nextSequence, commandId);
    onAdmitted?.(entry);
    snapshot.queue.push(entry);
    await persistLocked(snapshot);
    observer?.(snapshot);
    return {
      origin: url.origin,
      state: "pending",
      entryId: entry.entryId,
      position: snapshot.queue.length,
      pending_count: snapshot.queue.length,
      expires_at: entry.expiresAt,
      entry,
    };
  });
}

export async function accessStatus(origin: string, requester: string): Promise<QueueStatus> {
  const snapshot = await sweepQueue();
  const entry = findPending(snapshot.queue, origin, requester);
  if (entry) {
    return {
      origin,
      state: "pending",
      entryId: entry.entryId,
      position: queuePosition(snapshot.queue, entry.entryId)!,
      pending_count: snapshot.queue.length,
      expires_at: entry.expiresAt,
    };
  }
  const receipt = receiptForRequester(snapshot.results, origin, requester, Date.now());
  if (receipt) return { origin, state: receipt.state, entryId: receipt.entryId, pending_count: snapshot.queue.length };
  // Keep old-daemon/new-extension upgrades intelligible until the old
  // supersession receipt ages out.
  const session = await getSession();
  const legacy = session.accessTombstones?.[`${origin}\n${requester}`];
  if (legacy && Date.now() < legacy.expiresAt) return { origin, state: "superseded" };
  return { origin, state: "none", pending_count: snapshot.queue.length };
}

export function cancelAccess(origin: string, requester: string): Promise<QueueStatus> {
  return withSessionMutation(async () => {
    const now = Date.now();
    const snapshot = await normalizedLocked(now);
    const entry = findPending(snapshot.queue, origin, requester, undefined, now);
    if (!entry) return { origin, state: "none", pending_count: snapshot.queue.length };
    snapshot.queue = snapshot.queue.filter((candidate) => candidate.entryId !== entry.entryId);
    const receipt = receiptFor(entry, "cancelled", now);
    snapshot.results[resultKey(entry.entryId, requester)] = receipt;
    snapshot.results = cleanResults(snapshot.results, now);
    await persistLocked(snapshot);
    observer?.(snapshot);
    return { origin, state: "cancelled", entryId: entry.entryId, pending_count: snapshot.queue.length };
  });
}

export interface DecisionResult {
  applied: boolean;
  decided: AccessQueueEntry[];
  snapshot: QueueSnapshot;
}

export function decideAccess(entryId: string, decision: OriginDecision): Promise<DecisionResult> {
  return withSessionMutation(async () => {
    const now = Date.now();
    const snapshot = await normalizedLocked(now);
    const selected = snapshot.queue.find((entry) => entry.entryId === entryId);
    if (!selected) return { applied: false, decided: [], snapshot };
    let decided = [selected];
    if (decision === "always") {
      // Commit persistent policy first. If MV3 stops before queue receipts are
      // reconciled, a restarted worker still admits every covered request.
      await grantExactOriginLocked(selected.origin);
      decided = snapshot.queue.filter((entry) => policyCovers(selected.origin, entry.origin));
    } else if (decision === "all_ports") {
      const selectedUrl = new URL(selected.origin);
      if (!isLoopbackHost(selectedUrl) || !(await grantLoopbackHostLocked(selectedUrl))) {
        return { applied: false, decided: [], snapshot };
      }
      decided = snapshot.queue.filter((entry) =>
        policyCovers(selected.origin, entry.origin, "loopback_all_ports"),
      );
    }
    const state: AccessReceipt["state"] = decision === "deny" ? "denied" : "allowed";
    const ids = new Set(decided.map((entry) => entry.entryId));
    snapshot.queue = snapshot.queue.filter((entry) => !ids.has(entry.entryId));
    for (const entry of decided) {
      const receipt = receiptFor(entry, state, now);
      snapshot.results[resultKey(entry.entryId, entry.requester)] = receipt;
    }
    if (decision === "once" && selected.kind === "async") {
      // An in-command entry spends its one use by resuming that paused document
      // immediately. Minting a stored grant as well would silently authorize a
      // second future navigation.
      snapshot.onceGrants[requesterOriginKey(selected.origin, selected.requester)] = {
        origin: selected.origin,
        requester: selected.requester,
        expiresAt: now + ONCE_GRANT_TTL_MS,
        ...(selected.commandId ? { handoff: selected.commandId } : {}),
      };
    }
    snapshot.results = cleanResults(snapshot.results, now);
    await persistLocked(snapshot);
    observer?.(snapshot);
    return { applied: true, decided, snapshot };
  });
}
