export const DEFAULT_PORT = 4099;

export interface StoredSurface {
  tabId: number;
  nonce: string;
  epoch: number;
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
  surface?: StoredSurface;
  refs?: Record<string, SnapshotRef>;
  pendingOrigin?: PendingOrigin;
}

export async function getLocal(): Promise<LocalState> {
  return chrome.storage.local.get(["token", "port", "origins"]);
}

export async function getSession(): Promise<SessionState> {
  return chrome.storage.session.get(["surface", "refs", "pendingOrigin"]);
}

export async function setSurface(surface?: StoredSurface): Promise<void> {
  if (surface) await chrome.storage.session.set({ surface });
  else await chrome.storage.session.remove(["surface", "refs"]);
}

export async function setRefs(refs: Record<string, SnapshotRef>): Promise<void> {
  await chrome.storage.session.set({ refs });
}

export function surfaceToken(surface: StoredSurface): string {
  return `bridge:${surface.tabId}:${surface.nonce}`;
}

export function parseSurface(token: unknown): StoredSurface | undefined {
  if (typeof token !== "string") return undefined;
  const match = /^bridge:(\d+):([a-z0-9_-]+)$/i.exec(token);
  if (!match?.[1] || !match[2]) return undefined;
  return { tabId: Number(match[1]), nonce: match[2], epoch: 0 };
}
