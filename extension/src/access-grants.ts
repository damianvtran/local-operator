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
/** Is `key` one this build would itself mint for `scope`?
 *
 * Enforced at WRITE time only (grantSiteLocked). A key is minted from
 * `broadGrantFor(url)`, so this is a belt-and-braces assertion that the
 * derivation and the stored shape agree; it must NOT gate reads or
 * revocation, because the PSL moves under stored keys over time (A2). */
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

/** Normalize a stored `siteGrants` record for a MUTATION.
 *
 * Validates the record's SHAPE only. Key derivability is deliberately not
 * checked here: keys are written under the PSL bundled at grant time, and the
 * generator refreshes that list every release, so a domain that later becomes
 * a public suffix (blogspot.com is the worked example) would turn its own
 * stored key invalid. Re-deriving here made one such key refuse the whole
 * mutation, which bricked every future grant AND the Remove button for the
 * very row the user was trying to delete, while the read path kept honouring
 * the other entries: access stayed granted and the off-switch disappeared
 * (A2). Losing the ability to revoke is strictly worse than carrying a key
 * this build cannot re-derive, and carrying it is safe because
 * `matchingGrantScope` looks grants up by a key derived from the URL under
 * TODAY's list, so an unrecognised key can only fail to match, never widen.
 *
 * One structurally malformed entry still refuses the whole mutation, matching
 * normalizedHostGrants: a record whose shape we cannot read is one we cannot
 * safely rewrite. */
export function normalizedSiteGrants(value: unknown): SiteGrantsState | null {
  if (!validSiteGrantSchema(value)) return null;
  const entries = Object.entries(value.grants);
  if (entries.some(([, grant]) => !validSiteGrant(grant))) return null;
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
  // The key this build is about to mint must be one it would recognise. A
  // stored key that no longer derives is tolerated (A2); a NEW one that does
  // not derive would be a bug in broadGrantFor, and is refused here.
  if (!validSiteGrantKey(broad.key, broad.scope)) return false;
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
    // Report failure rather than a false "Removed …" for a key that was never
    // there, so the Settings receipt describes what actually happened (A5).
    if (!(key in current.grants)) return false;
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
