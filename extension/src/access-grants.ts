import {
  loopbackHostGrantKey,
  loopbackHostGrantLabel,
  matchingGrantScope,
  type HostGrant,
  type HostGrantsState,
  validHostGrantSchema,
} from "./origin-policy";
import { withSessionMutation } from "./state";

/** Persistent grants and the session approval queue share one worker-owned
 * mutation chain. The `Locked` helpers exist for approval-store, which already
 * owns that chain while committing the durable grant before queue receipts. */

function validGrant(value: unknown): value is HostGrant {
  return !!value && typeof value === "object" && !Array.isArray(value) &&
    (value as { scope?: unknown }).scope === "all_ports" &&
    typeof (value as { createdAt?: unknown }).createdAt === "number" &&
    Number.isFinite((value as { createdAt: number }).createdAt);
}

export function normalizedHostGrants(value: unknown): HostGrantsState | null {
  if (!validHostGrantSchema(value)) return null;
  const entries = Object.entries(value.grants);
  // A malformed current-version record may belong to a newer writer using a
  // shape this build cannot preserve losslessly. Refuse the whole mutation.
  if (entries.some(([key, grant]) => loopbackHostGrantLabel(key) === null || !validGrant(grant))) {
    return null;
  }
  return { version: 1, grants: Object.fromEntries(entries) };
}

export async function grantExactOriginLocked(origin: string): Promise<boolean> {
  const url = new URL(origin);
  const { origins = {}, hostGrants } = await chrome.storage.local.get([
    "origins",
    "hostGrants",
  ]);
  // A broad grant already covers this exact origin. Do not recreate a hidden
  // redundant row after the broad approval compacted it away.
  if (matchingGrantScope({}, hostGrants, url) === "loopback_all_ports") return true;
  await chrome.storage.local.set({ origins: { ...origins, [origin]: "allow" } });
  return true;
}

export function grantExactOrigin(origin: string): Promise<boolean> {
  return withSessionMutation(() => grantExactOriginLocked(origin));
}

export function revokeExactOrigin(origin: string): Promise<boolean> {
  return withSessionMutation(async () => {
    const { origins = {} } = await chrome.storage.local.get(["origins"]);
    const next = { ...origins };
    delete next[origin];
    await chrome.storage.local.set({ origins: next });
    return true;
  });
}

export async function grantLoopbackHostLocked(url: URL): Promise<boolean> {
  const canonicalKey = loopbackHostGrantKey(url);
  if (!canonicalKey) return false;
  const { hostGrants, origins = {} } = await chrome.storage.local.get([
    "hostGrants",
    "origins",
  ]);
  if (hostGrants !== undefined && !normalizedHostGrants(hostGrants)) return false;
  const current = normalizedHostGrants(hostGrants) ?? { version: 1 as const, grants: {} };
  const remainingOrigins = Object.fromEntries(
    Object.entries(origins).filter(([origin, verdict]) => {
      if (verdict !== "allow") return true;
      try {
        const exact = new URL(origin);
        return exact.protocol !== url.protocol || exact.hostname !== url.hostname;
      } catch {
        // Unknown legacy entries are not ours to discard.
        return true;
      }
    }),
  );
  // One multi-key storage write is the durable transaction: failure cannot
  // report success after cleaning exact grants without storing the broad one.
  await chrome.storage.local.set({
    origins: remainingOrigins,
    hostGrants: {
      version: 1,
      grants: {
        ...current.grants,
        [canonicalKey]: { scope: "all_ports", createdAt: Date.now() },
      },
    },
  });
  return true;
}

export function grantLoopbackHost(url: URL): Promise<boolean> {
  return withSessionMutation(() => grantLoopbackHostLocked(url));
}

export function revokeLoopbackHost(canonicalKey: string): Promise<boolean> {
  return withSessionMutation(async () => {
    const { hostGrants } = await chrome.storage.local.get(["hostGrants"]);
    const current = normalizedHostGrants(hostGrants);
    if (!current) return false;
    const grants = { ...current.grants };
    delete grants[canonicalKey];
    await chrome.storage.local.set({ hostGrants: { version: 1, grants } });
    return true;
  });
}

export function clearAllAccessGrants(): Promise<boolean> {
  return withSessionMutation(async () => {
    await chrome.storage.local.remove(["token", "origins", "hostGrants"]);
    return true;
  });
}
