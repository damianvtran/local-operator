import {
  loopbackHostGrantKey,
  type HostGrant,
  type HostGrantsState,
  validHostGrantSchema,
} from "./origin-policy";

/** All broad-grant mutations execute in the single MV3 worker context. Options
 * sends runtime messages instead of touching the snapshot, and popup decisions
 * already arrive here. Chrome runs one service-worker instance, so this queue
 * serializes every read-modify-write; suspension cannot overlap two instances. */
let mutationQueue: Promise<unknown> = Promise.resolve();

function enqueue<T>(operation: () => Promise<T>): Promise<T> {
  const run = mutationQueue.catch(() => {}).then(operation);
  mutationQueue = run;
  return run;
}

function validGrant(value: unknown): value is HostGrant {
  return !!value && typeof value === "object" && !Array.isArray(value) &&
    (value as { scope?: unknown }).scope === "all_ports" &&
    typeof (value as { createdAt?: unknown }).createdAt === "number";
}

export function normalizedHostGrants(value: unknown): HostGrantsState | null {
  if (!validHostGrantSchema(value)) return null;
  const grants = Object.fromEntries(
    Object.entries(value.grants).filter(([, grant]) => validGrant(grant)),
  );
  return { version: 1, grants };
}

export function grantLoopbackHost(url: URL): Promise<boolean> {
  const canonicalKey = loopbackHostGrantKey(url);
  if (!canonicalKey) return Promise.resolve(false);
  return enqueue(async () => {
    const { hostGrants } = await chrome.storage.local.get(["hostGrants"]);
    if (hostGrants !== undefined && !validHostGrantSchema(hostGrants)) return false;
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
