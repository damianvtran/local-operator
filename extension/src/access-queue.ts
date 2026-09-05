/* Pure, storage-independent approval queue rules.
 *
 * The queue is FIFO across async and already-running navigation requests. An
 * in-command request receives a shorter lifetime because its navigation is
 * paused, but it appends rather than displacing the request the user is
 * currently reviewing. Keeping ordering boring is intentional: it makes two
 * popup contexts and a restarted worker converge on the same next entry. */

import { registrableDomain, type BroadGrant } from "./origin-policy";

/** The popup's answer to a prompt, defined ONCE here and `import type`d by
 * access-flow, origin-flow and origins. `site` is the exact-origin grant
 * (scheme, host and port; the old "always"), `domain` the registrable-domain
 * or loopback-host grant carried by the entry's `broad` field, `once` a
 * single navigation. Deny is never persisted. */
export type OriginDecision = "once" | "site" | "domain" | "deny";
export type AccessKind = "async" | "in_command";
export type AccessState = "allowed" | "denied" | "pending" | "cancelled" | "superseded" | "none";

export const ACCESS_QUEUE_VERSION = 1;
export const ACCESS_QUEUE_CAP = 16;
export const ACCESS_RESULT_CAP = 32;
export const ACCESS_REQUEST_TTL_MS = 10 * 60_000;
export const IN_COMMAND_TTL_MS = 60_000;
export const ACCESS_RESULT_TTL_MS = 15 * 60_000;
export const ONCE_GRANT_TTL_MS = 10 * 60_000;

export interface AccessQueueEntry {
  entryId: string;
  origin: string;
  displayAuthority: string;
  /** Authority boundary only. Never render or include in ambient notifications. */
  requester: string;
  kind: AccessKind;
  requestedAt: number;
  expiresAt: number;
  sequence: number;
  commandId?: string;
  /** The broad grant the popup may offer, computed by the worker at enqueue.
   * Absent on entries persisted by 0.1.7 and on /health-only renders, in
   * which case the popup omits the domain option (fail closed). */
  broad?: BroadGrant;
}

export interface AccessReceipt {
  entryId: string;
  origin: string;
  requester: string;
  state: Exclude<AccessState, "pending" | "none">;
  decidedAt: number;
  expiresAt: number;
}

export interface OnceGrant {
  expiresAt: number;
  requester: string;
  origin: string;
  handoff?: string;
}

export type AccessResults = Record<string, AccessReceipt>;
export type OnceGrants = Record<string, OnceGrant>;

export function requesterOriginKey(origin: string, requester: string): string {
  return `${origin}\n${requester}`;
}

export function resultKey(entryId: string, requester: string): string {
  return `${entryId}\n${requester}`;
}

export function liveQueue(queue: AccessQueueEntry[] | undefined, now: number): AccessQueueEntry[] {
  return (queue ?? []).filter((entry) => now < entry.expiresAt).sort((a, b) => a.sequence - b.sequence);
}

export function cleanResults(results: AccessResults | undefined, now: number): AccessResults {
  return Object.fromEntries(
    Object.entries(results ?? {})
      .filter(([, receipt]) => now < receipt.expiresAt)
      .sort((a, b) => b[1].decidedAt - a[1].decidedAt)
      .slice(0, ACCESS_RESULT_CAP),
  );
}

export function findPending(
  queue: AccessQueueEntry[] | undefined,
  origin: string,
  requester: string,
  kind?: AccessKind,
  now: number = Date.now(),
): AccessQueueEntry | undefined {
  return liveQueue(queue, now).find(
    (entry) =>
      entry.origin === origin && entry.requester === requester && (!kind || entry.kind === kind),
  );
}

export function queuePosition(queue: AccessQueueEntry[], entryId: string): number | undefined {
  const index = queue.findIndex((entry) => entry.entryId === entryId);
  return index < 0 ? undefined : index + 1;
}

export function selectEntry(
  queue: AccessQueueEntry[],
  selectedEntryId: string | undefined,
): AccessQueueEntry | undefined {
  return queue.find((entry) => entry.entryId === selectedEntryId) ?? queue[0];
}

export function adjacentEntryId(
  queue: AccessQueueEntry[],
  selectedEntryId: string | undefined,
  delta: -1 | 1,
): string | undefined {
  if (!queue.length) return undefined;
  const current = Math.max(0, queue.findIndex((entry) => entry.entryId === selectedEntryId));
  return queue[Math.max(0, Math.min(queue.length - 1, current + delta))]?.entryId;
}

export function newEntry(
  origin: string,
  displayAuthority: string,
  requester: string,
  kind: AccessKind,
  now: number,
  sequence: number,
  commandId?: string,
  entryId: string = crypto.randomUUID().replaceAll("-", ""),
  broad?: BroadGrant | null,
): AccessQueueEntry {
  return {
    entryId,
    origin,
    displayAuthority,
    requester,
    kind,
    requestedAt: now,
    expiresAt: now + (kind === "async" ? ACCESS_REQUEST_TTL_MS : IN_COMMAND_TTL_MS),
    sequence,
    ...(commandId ? { commandId } : {}),
    ...(broad ? { broad } : {}),
  };
}

export function receiptFor(
  entry: AccessQueueEntry,
  state: AccessReceipt["state"],
  now: number,
): AccessReceipt {
  return {
    entryId: entry.entryId,
    origin: entry.origin,
    requester: entry.requester,
    state,
    decidedAt: now,
    expiresAt: now + ACCESS_RESULT_TTL_MS,
  };
}

export function receiptForRequester(
  results: AccessResults | undefined,
  origin: string,
  requester: string,
  now: number,
): AccessReceipt | undefined {
  return Object.values(cleanResults(results, now))
    .filter((receipt) => receipt.origin === origin && receipt.requester === requester)
    .sort((a, b) => b.decidedAt - a.decidedAt)[0];
}

const LOOPBACK_HOSTS = ["localhost", "127.0.0.1", "[::1]"];

/** The one reconciliation matcher for durable grants. Exact grants retain
 * origin semantics. `domain` covers every hostname with the same registrable
 * domain and `host` every port on the same literal loopback hostname, both
 * regardless of scheme: the trust the user asserts with a broad grant is in
 * the site operator, not the transport, and http-to-https redirect chains
 * are gated per hop, so a scheme-bound grant would re-prompt on the first
 * hop of nearly every `open http://` (the fatigue these scopes remove).
 * Legacy loopback all-port grants keep their same-scheme semantics. */
export function policyCovers(
  grantedOrigin: string,
  candidateOrigin: string,
  scope: "origin" | "domain" | "host" | "loopback_all_ports" = "origin",
): boolean {
  if (scope === "origin") return grantedOrigin === candidateOrigin;
  try {
    const granted = new URL(grantedOrigin);
    const candidate = new URL(candidateOrigin);
    if (scope === "domain") {
      const domain = registrableDomain(granted);
      return domain !== null && registrableDomain(candidate) === domain;
    }
    if (scope === "host") {
      return granted.hostname === candidate.hostname && LOOPBACK_HOSTS.includes(granted.hostname);
    }
    return (
      granted.protocol === candidate.protocol &&
      granted.hostname === candidate.hostname &&
      LOOPBACK_HOSTS.includes(granted.hostname)
    );
  } catch {
    return false;
  }
}
