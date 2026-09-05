import {
  broadGrantFor,
  isLoopbackHost,
  loopbackHostGrantLabel,
  matchingGrantScope,
  registrableDomain,
  type HostGrant,
  type HostGrantsState,
  type SiteGrant,
  type SiteGrantsState,
  validHostGrantSchema,
  validSiteGrantSchema,
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

/** Site-grant keys are validated per scope: a `domain` key must be the
 * registrable domain of itself (so `co.uk` or `www.x.com` cannot be keys)
 * and a `host` key a literal loopback hostname. Both scheme-less. */
function validSiteGrantKey(key: string, scope: SiteGrant["scope"]): boolean {
  try {
    const url = new URL(`http://${key}/`);
    if (url.hostname !== key) return false;
    return scope === "host" ? isLoopbackHost(url) : registrableDomain(url) === key;
  } catch {
    return false;
  }
}

function validSiteGrant(value: unknown): value is SiteGrant {
  return !!value && typeof value === "object" && !Array.isArray(value) &&
    ((value as { scope?: unknown }).scope === "domain" || (value as { scope?: unknown }).scope === "host") &&
    typeof (value as { createdAt?: unknown }).createdAt === "number" &&
    Number.isFinite((value as { createdAt: number }).createdAt);
}

export function normalizedSiteGrants(value: unknown): SiteGrantsState | null {
  if (!validSiteGrantSchema(value)) return null;
  const entries = Object.entries(value.grants);
  // Same rule as normalizedHostGrants: one malformed entry refuses the whole
  // mutation rather than silently dropping a record this build cannot read.
  if (entries.some(([key, grant]) => !validSiteGrant(grant) || !validSiteGrantKey(key, grant.scope))) {
    return null;
  }
  return { version: 1, grants: Object.fromEntries(entries) };
}

export async function grantExactOriginLocked(origin: string): Promise<boolean> {
  const url = new URL(origin);
  const { origins = {}, hostGrants, siteGrants } = await chrome.storage.local.get([
    "origins",
    "hostGrants",
    "siteGrants",
  ]);
  // A broad grant already covers this exact origin. Do not recreate a hidden
  // redundant row after the broad approval compacted it away.
  const covering = matchingGrantScope({}, hostGrants, url, siteGrants);
  if (covering === "loopback_all_ports" || covering === "domain" || covering === "host") return true;
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

/** Write the broad grant for `url` (registrable domain, or loopback host)
 * and compact the exact-origin rows it now covers in the same multi-key
 * write. Returns false, and writes nothing, when the URL has no broad grant
 * to offer or the stored record is one this build cannot preserve. */
export async function grantSiteLocked(url: URL): Promise<boolean> {
  const broad = broadGrantFor(url);
  if (!broad) return false;
  const { siteGrants, origins = {} } = await chrome.storage.local.get(["siteGrants", "origins"]);
  if (siteGrants !== undefined && !normalizedSiteGrants(siteGrants)) return false;
  const current = normalizedSiteGrants(siteGrants) ?? { version: 1 as const, grants: {} };
  const remainingOrigins = Object.fromEntries(
    Object.entries(origins).filter(([origin, verdict]) => {
      if (verdict !== "allow") return true;
      try {
        const exact = new URL(origin);
        return broad.scope === "host"
          ? exact.hostname !== broad.key
          : registrableDomain(exact) !== broad.key;
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
    siteGrants: {
      version: 1,
      grants: { ...current.grants, [broad.key]: { scope: broad.scope, createdAt: Date.now() } },
    },
  });
  return true;
}

export function grantSite(url: URL): Promise<boolean> {
  return withSessionMutation(() => grantSiteLocked(url));
}

export function revokeSiteGrant(key: string): Promise<boolean> {
  return withSessionMutation(async () => {
    const { siteGrants } = await chrome.storage.local.get(["siteGrants"]);
    const current = normalizedSiteGrants(siteGrants);
    if (!current) return false;
    const grants = { ...current.grants };
    delete grants[key];
    await chrome.storage.local.set({ siteGrants: { version: 1, grants } });
    return true;
  });
}

/** Revoke a LEGACY (0.1.4 to 0.1.7) same-scheme loopback all-port grant.
 * Nothing writes these any more; this stays so the Settings row for an
 * existing one can still be removed. */
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
    // Unpairing also drops the all-sites bypass: a browser that is no longer
    // trusted must not come back pre-opened when it is paired again.
    await chrome.storage.local.remove(["token", "origins", "hostGrants", "siteGrants", "allowAllSites"]);
    return true;
  });
}
