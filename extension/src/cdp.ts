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
      // The module-global `attached` set does not survive service-worker death,
      // so after MV3 churn a tab OUR debugger is still attached to throws here
      // on the first post-restart command. That is the exact reconnect path the
      // bridge claims to handle, and mapping it to debugger_conflict told the
      // user to "close DevTools" that was never open (finding A8). Reconcile:
      // if the still-attached debugger is ours, adopt it silently; only a
      // FOREIGN attachment (DevTools) is a real conflict.
      if (await ownAttachment(tabId)) {
        attached.add(tabId);
        return;
      }
      throw new BridgeCommandError("debugger_conflict", message);
    }
    throw new BridgeCommandError("internal", message);
  }
}

async function ownAttachment(tabId: number): Promise<boolean> {
  // A tab we can still drive answers a trivial CDP command; a tab held by a
  // foreign debugger rejects it. This distinguishes our surviving attachment
  // from DevTools without a fragile string match on Chrome's error text.
  try {
    await chrome.debugger.sendCommand({ tabId }, "Runtime.evaluate", { expression: "1" });
    return true;
  } catch {
    return false;
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
