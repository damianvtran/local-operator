import { getSession, parseSurface, type StoredSurface } from "./state";

const attached = new Set<number>();

export class BridgeCommandError extends Error {
  constructor(
    public readonly code: string,
    message: string,
    public readonly data: Record<string, unknown> = {},
  ) {
    super(message);
  }
}

export async function requireSurface(token: unknown): Promise<StoredSurface> {
  const parsed = parseSurface(token);
  const { surface } = await getSession();
  if (!parsed || !surface || parsed.tabId !== surface.tabId || parsed.nonce !== surface.nonce) {
    throw new BridgeCommandError("tab_closed", "the browser tab handle is stale");
  }
  try {
    await chrome.tabs.get(surface.tabId);
  } catch {
    throw new BridgeCommandError("tab_closed", "the browser tab was closed");
  }
  return surface;
}

export async function attach(tabId: number): Promise<void> {
  if (attached.has(tabId)) return;
  try {
    await chrome.debugger.attach({ tabId }, "1.3");
    attached.add(tabId);
  } catch (error) {
    const message = String(error);
    if (message.includes("Another debugger") || message.includes("already attached")) {
      throw new BridgeCommandError("debugger_conflict", message);
    }
    throw new BridgeCommandError("internal", message);
  }
}

export async function detach(tabId: number): Promise<void> {
  if (!attached.has(tabId)) return;
  try {
    await chrome.debugger.detach({ tabId });
  } catch {
    // Close is idempotent; a browser-initiated detach already achieved it.
  }
  attached.delete(tabId);
}

export async function cdp<T>(
  tabId: number,
  method: string,
  params: Record<string, unknown> = {},
): Promise<T> {
  await attach(tabId);
  try {
    return (await chrome.debugger.sendCommand({ tabId }, method, params)) as T;
  } catch (error) {
    throw new BridgeCommandError("internal", String(error));
  }
}

chrome.debugger.onDetach.addListener((source) => {
  if (source.tabId !== undefined) attached.delete(source.tabId);
});
