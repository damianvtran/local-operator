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
}

export interface SnapshotRef {
  backendNodeId: number;
  epoch: number;
}

export interface PendingOrigin {
  origin: string;
  hostname: string;
  requestId: string;
}

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
}

export async function getLocal(): Promise<LocalState> {
  return chrome.storage.local.get(["token", "port", "origins"]);
}

export async function getSession(): Promise<SessionState> {
  return chrome.storage.session.get(["surfaces", "refs", "pendingOrigin"]);
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
