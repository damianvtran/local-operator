import {
  loopbackHostGrantKey,
  loopbackHostGrantLabel,
  type HostGrant,
  type HostGrantsState,
  validHostGrantSchema,
} from "./origin-policy";

/** Every persistent access mutation executes in the one MV3 worker context.
 * Popup approval and Settings messages both enter this queue, so exact and
 * broad grants cannot race revoke/unpair through separate storage snapshots. */
let mutationQueue: Promise<unknown> = Promise.resolve();

function enqueue<T>(operation: () => Promise<T>): Promise<T> {
  const run = mutationQueue.catch(() => {}).then(operation);
  mutationQueue = run;
  return run;
}

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

export function grantExactOrigin(origin: string): Promise<boolean> {
  return enqueue(async () => {
    const { origins = {} } = await chrome.storage.local.get(["origins"]);
    await chrome.storage.local.set({ origins: { ...origins, [origin]: "allow" } });
    return true;
  });
}

export function revokeExactOrigin(origin: string): Promise<boolean> {
  return enqueue(async () => {
    const { origins = {} } = await chrome.storage.local.get(["origins"]);
    const next = { ...origins };
    delete next[origin];
    await chrome.storage.local.set({ origins: next });
    return true;
  });
}

export function grantLoopbackHost(url: URL): Promise<boolean> {
  const canonicalKey = loopbackHostGrantKey(url);
  if (!canonicalKey) return Promise.resolve(false);
  return enqueue(async () => {
    const { hostGrants } = await chrome.storage.local.get(["hostGrants"]);
    if (hostGrants !== undefined && !normalizedHostGrants(hostGrants)) return false;
    const current = normalizedHostGrants(hostGrants) ?? { version: 1 as const, grants: {} };
    await chrome.storage.local.set({
      hostGrants: {
        version: 1,
        grants: {
          ...current.grants,
          [canonicalKey]: { scope: "all_ports", createdAt: Date.now() },
        },
      },
    });
    return true;
  });
}

export function revokeLoopbackHost(canonicalKey: string): Promise<boolean> {
  return enqueue(async () => {
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
  return enqueue(async () => {
    await chrome.storage.local.remove(["token", "origins", "hostGrants"]);
    return true;
  });
}
