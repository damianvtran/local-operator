/* Stored per-site decisions on a HEADLESS NODE: list them, remember a signed
 * Deny, and remove either.
 *
 * WHY THIS EXISTS. On a node nobody can open the extension's Settings page (the
 * Chrome is headless), so the two things Settings does for a person — show what
 * is allowed and take a grant back — need a bridge door (`list_access`,
 * `revoke_access`) that `lop browser sites` drives on the node itself (PR #3938
 * review D2/U4: the card promises "until removed", so the removal must exist).
 *
 * And the operator's D3 rule is "ask me once per site": after a SIGNED Deny the
 * node must not raise another card for the same origin. The ordinary queue keeps
 * a deny only as a short-lived receipt (a popup Deny is deliberately not
 * sticky), so a node keeps its own list here, written ONLY after a signature
 * from the operator anchor decided "deny" (access.ts). `request_access` for a
 * listed origin answers `denied` without queueing anything (Q4).
 *
 * CONSTRAINTS. Removal is permission-REDUCING for a grant and only re-enables
 * ASKING for a denial — never approval — so neither door needs a signature: an
 * agent holding the bridge token can at worst make the operator see a card
 * again. Grants themselves stay where the popup path writes them (`origins`,
 * exact origin → "allow"); this module never writes a grant.
 *
 * Pure over a `StorageLike` so the node test runner can drive it without
 * Chrome; access.ts supplies `chrome.storage.local` behind `deadline`. */

export const NODE_DENIALS_KEY = "nodeSiteDenials";

export interface StorageLike {
  get(keys: string[]): Promise<Record<string, unknown>>;
  set(items: Record<string, unknown>): Promise<void>;
}

export interface SiteDecisions {
  allowed: string[];
  denied: string[];
  /** Broad grants (registrable domain / host) a desktop popup may have written
   * before this browser became a node; listed so the operator is not told an
   * origin is gone while one of these still covers it. */
  broad: string[];
  allow_all: boolean;
}

export interface RevokeResult {
  origin: string;
  removed: Array<"allowed" | "denied">;
}

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
}

/** `{origin: deniedAtMs}`, dropping anything malformed rather than failing. */
export function normalizedDenials(value: unknown): Record<string, number> {
  const out: Record<string, number> = {};
  for (const [origin, at] of Object.entries(record(value))) {
    if (typeof at === "number" && Number.isFinite(at)) out[origin] = at;
  }
  return out;
}

export async function recordNodeDenial(storage: StorageLike, origin: string, now: number): Promise<void> {
  const current = normalizedDenials((await storage.get([NODE_DENIALS_KEY]))[NODE_DENIALS_KEY]);
  await storage.set({ [NODE_DENIALS_KEY]: { ...current, [origin]: now } });
}

export async function nodeDenied(storage: StorageLike, origin: string): Promise<boolean> {
  const current = normalizedDenials((await storage.get([NODE_DENIALS_KEY]))[NODE_DENIALS_KEY]);
  return Object.prototype.hasOwnProperty.call(current, origin);
}

export async function listSiteDecisions(storage: StorageLike): Promise<SiteDecisions> {
  const stored = await storage.get(["origins", "siteGrants", "allowAllSites", NODE_DENIALS_KEY]);
  const origins = record(stored.origins);
  const grants = record(record(stored.siteGrants).grants);
  return {
    allowed: Object.keys(origins).filter((origin) => origins[origin] === "allow").sort(),
    denied: Object.keys(normalizedDenials(stored[NODE_DENIALS_KEY])).sort(),
    broad: Object.keys(grants).sort(),
    allow_all: stored.allowAllSites === true,
  };
}

/** Remove the exact origin's stored grant AND its node denial. Reports what was
 * there, so the CLI can say "nothing was stored for X" instead of a false
 * "removed". */
export async function revokeSiteDecision(storage: StorageLike, origin: string): Promise<RevokeResult> {
  const stored = await storage.get(["origins", NODE_DENIALS_KEY]);
  const origins = { ...record(stored.origins) };
  const denials = normalizedDenials(stored[NODE_DENIALS_KEY]);
  const removed: RevokeResult["removed"] = [];
  const writes: Record<string, unknown> = {};
  if (Object.prototype.hasOwnProperty.call(origins, origin)) {
    if (origins[origin] === "allow") removed.push("allowed");
    delete origins[origin];
    writes.origins = origins;
  }
  if (Object.prototype.hasOwnProperty.call(denials, origin)) {
    removed.push("denied");
    delete denials[origin];
    writes[NODE_DENIALS_KEY] = denials;
  }
  // One multi-key write: a grant and a denial for one origin go together.
  if (Object.keys(writes).length) await storage.set(writes);
  return { origin, removed };
}
