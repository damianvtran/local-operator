export const DEFAULT_PORT = 4099;

// Hard ceiling on concurrently-driven tabs. Parallel sessions each open their
// own surface now, and nothing else bounds how many an agent fleet can spray
// into the user's real browser — each one costs a debugger banner, a log
// buffer and the user's attention to close by hand if a session dies without
// cleanup. Eight comfortably covers a parallel batch while keeping the worst
// case something a human can recover from in seconds. The refusal is a typed
// tab_limit error telling the agent to close one, not a silent reuse.
export const MAX_SURFACES = 8;

export interface StoredSurface {
  tabId: number;
  nonce: string;
  epoch: number;
  // For the `tabs` listing: when the surface was created and last driven. The
  // live URL/title are deliberately NOT stored — they are fetched from
  // chrome.tabs at list time so the listing can never show a stale page.
  createdAt: number;
  lastUsedAt: number;
  // Optional for seamless upgrade from 0.1.3 session storage. Identity and the
  // stable collision ordinal persist; native group ids do not survive Chrome.
  ownerKey?: string;
  groupBaseLabel?: string;
  groupOrdinal?: number;
  groupAppliedLabel?: string;
  // Advisory ownership proof for the current browser lifetime. Group IDs are
  // ephemeral, so a mismatch is never repaired on an ordinary command; only
  // explicit open/resume may establish and persist a fresh LO-owned group.
  appliedGroupId?: number;
}

export interface SnapshotRef {
  backendNodeId: number;
  epoch: number;
}

export interface PendingOrigin {
  origin: string;
  hostname: string;
  requestId: string;
  // Immutable per-PROMPT generation token, minted every time the prompt slot
  // is (re)written. The popup binds its rendered view and its decision message
  // to this id, and the worker rejects a decision whose id is no longer the
  // live one — without it, a popup still showing origin A could click through
  // to whatever origin B had replaced the slot with (round-2 B1: a consent UI
  // must never approve something the user was not looking at).
  promptId: string;
}

// Async access-request state (see access-flow.ts for the machine and the
// incident that motivated it). Session-scoped on purpose: like `origins`
// "once" grants, an approval should not outlive the browser session, and
// session storage survives MV3 worker death, which worker memory does not —
// a decision made between two await_access polls must still be readable.
export type { AccessRequest, OnceGrants } from "./access-flow";

export interface LocalState {
  token?: string;
  port?: number;
  origins?: Record<string, "allow" | "deny">;
}

export interface SessionState {
  // Keyed by surface token (bridge:<tabId>:<nonce>) so each parallel session's
  // tab has its own nonce/epoch and one session cannot address another's tab
  // without knowing its nonce. Replaces the old single `surface` slot, whose
  // open-time reuse let a second session silently steal the first's tab.
  surfaces?: Record<string, StoredSurface>;
  // Snapshot refs are per-surface for the same reason: one tab's snapshot must
  // not overwrite another's click targets mid-interaction.
  refs?: Record<string, Record<string, SnapshotRef>>;
  pendingOrigin?: PendingOrigin;
  accessRequest?: import("./access-flow").AccessRequest;
  onceGrants?: import("./access-flow").OnceGrants;
  accessTombstones?: import("./access-flow").AccessTombstones;
}

export async function getLocal(): Promise<LocalState> {
  return chrome.storage.local.get(["token", "port", "origins"]);
}

export async function getSession(): Promise<SessionState> {
  return chrome.storage.session.get([
    "surfaces",
    "refs",
    "pendingOrigin",
    "accessRequest",
    "onceGrants",
    "accessTombstones",
  ]);
}

export async function getSurfaces(): Promise<Record<string, StoredSurface>> {
  const { surfaces = {} } = (await chrome.storage.session.get(["surfaces"])) as SessionState;
  return surfaces;
}

// Serializes read-modify-write mutations of the surfaces/refs maps. The daemon
// serializes commands per TAB, so two different tabs' commands run truly
// concurrently in this worker — two interleaved map writes would lose one
// tab's update (the same lost-update shape the old single-surface model never
// had to face). Same promise-chain pattern as snapshot's axQueue: mutations
// are rare and tiny, and each link swallows its predecessor's failure so the
// chain cannot poison later calls.
let storeQueue: Promise<unknown> = Promise.resolve();
function withStore<T>(op: () => Promise<T>): Promise<T> {
  const run = storeQueue.catch(() => {}).then(op);
  storeQueue = run;
  return run;
}

/** The same serialized-mutation queue, exported for the access-flow state
 * (grants/requests/receipts in chrome.storage.session). Grant consumption is
 * a read-check-delete; two concurrent navigations (different tabs = different
 * daemon locks, and the worker dispatches frames concurrently) could both
 * read the grant before either delete landed and DOUBLE-SPEND a one-shot
 * approval (round-2 B2, reproduced by review). One queue for every session
 * mutation makes each read-modify-write atomic with respect to the others;
 * the ops are tiny, so a single queue beats a per-key lock map that would
 * need its own eviction story. */
export function withSessionMutation<T>(op: () => Promise<T>): Promise<T> {
  return withStore(op);
}

export function putSurface(surface: StoredSurface): Promise<void> {
  return withStore(async () => {
    const surfaces = await getSurfaces();
    surfaces[surfaceToken(surface)] = surface;
    await chrome.storage.session.set({ surfaces });
  });
}

/** Remove one surface and its snapshot refs; other surfaces are untouched. */
export function removeSurface(token: string): Promise<void> {
  return withStore(async () => {
    const { surfaces = {}, refs = {} } = (await chrome.storage.session.get([
      "surfaces",
      "refs",
    ])) as SessionState;
    delete surfaces[token];
    delete refs[token];
    await chrome.storage.session.set({ surfaces, refs });
  });
}

/**
 * Refresh a surface's ``lastUsedAt`` — only if it is still in the map.
 *
 * An unconditional put could resurrect an entry a concurrent prune (tabs /
 * status run under a different daemon lock key) just removed, leaving a dead
 * surface counting toward the cap until the next prune (review finding m5).
 * The presence check runs inside the store queue, so it cannot interleave
 * with the prune's own read-modify-write.
 */
export function touchSurface(token: string, at: number): Promise<void> {
  return withStore(async () => {
    const surfaces = await getSurfaces();
    const surface = surfaces[token];
    if (!surface) return;
    surface.lastUsedAt = at;
    await chrome.storage.session.set({ surfaces });
  });
}

export function setRefs(token: string, forSurface: Record<string, SnapshotRef>): Promise<void> {
  return withStore(async () => {
    const { refs = {} } = (await chrome.storage.session.get(["refs"])) as SessionState;
    refs[token] = forSurface;
    await chrome.storage.session.set({ refs });
  });
}

export async function getRefs(token: string): Promise<Record<string, SnapshotRef>> {
  const { refs = {} } = (await chrome.storage.session.get(["refs"])) as SessionState;
  return refs[token] ?? {};
}

export function surfaceToken(surface: StoredSurface): string {
  return `bridge:${surface.tabId}:${surface.nonce}`;
}

// How many nonce characters a redacted handle shows. Enough to prefix-match a
// session's own full token against a listing entry, far too few to reconstruct
// the 32-hex-char nonce (26 chars of entropy stay hidden).
const REDACTED_NONCE_CHARS = 6;

/**
 * A display-safe form of a surface handle: `bridge:<tabId>:<nonce[0..6]>…`.
 *
 * Every surface that leaves the extension OTHER than the caller's own `open`
 * response uses this — the `tabs` listing, the tab_limit refusal, the
 * ambiguous-close refusal, the no-handle `status`. The full token IS the
 * drive capability (review finding M1): listing it would hand every session
 * control of every tab and make the "listing does not grant control" claim
 * false. The redacted prefix still lets a caller recognise its OWN tab
 * (prefix-match against the full token it received at open) without being
 * able to drive anyone else's.
 */
export function redactToken(token: string): string {
  const parsed = parseSurface(token);
  if (!parsed) return token;
  return `bridge:${parsed.tabId}:${parsed.nonce.slice(0, REDACTED_NONCE_CHARS)}…`;
}

/** Whether ``fullToken`` (a caller's own handle) names the surface a redacted
 * listing entry describes. The trailing ellipsis marks redaction; matching is
 * a plain prefix test on the un-ellipsised part. */
export function ownsRedacted(fullToken: string, redacted: string): boolean {
  if (!redacted.endsWith("…")) return fullToken === redacted;
  return fullToken.startsWith(redacted.slice(0, -1));
}

export function parseSurface(token: unknown): { tabId: number; nonce: string } | undefined {
  if (typeof token !== "string") return undefined;
  const match = /^bridge:(\d+):([a-z0-9_-]+)$/i.exec(token);
  if (!match?.[1] || !match[2]) return undefined;
  return { tabId: Number(match[1]), nonce: match[2] };
}

/**
 * Resolve a caller-supplied handle against the surfaces map.
 *
 * The map is keyed by the FULL token, nonce included, so this is an exact
 * lookup: a caller that knows only a tab id (or guesses a nonce) simply
 * misses, which is the whole anti-guessing property the nonce exists for —
 * unchanged from the single-surface model, now enforced per surface.
 */
export function resolveSurfaceToken(
  token: unknown,
  surfaces: Record<string, StoredSurface>,
): StoredSurface | undefined {
  if (!parseSurface(token)) return undefined;
  return surfaces[token as string];
}

export function atSurfaceCap(surfaces: Record<string, StoredSurface>): boolean {
  return Object.keys(surfaces).length >= MAX_SURFACES;
}
