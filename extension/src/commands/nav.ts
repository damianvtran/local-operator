import { attach, BridgeCommandError, detach, requireSurface } from "../cdp";
import { askOrigin, safeHttpUrl, withOriginGate } from "../origins";
import { getSession, setSurface, surfaceToken, type StoredSurface } from "../state";

async function page(tabId: number): Promise<{ url: string; title: string }> {
  const tab = await chrome.tabs.get(tabId);
  return { url: tab.url ?? "", title: tab.title ?? "" };
}

function settle(tabId: number, timeout = 30_000): Promise<void> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => done(new BridgeCommandError("nav_timeout", "navigation timed out")), timeout);
    const complete = (details: chrome.webNavigation.WebNavigationFramedCallbackDetails) => {
      if (details.tabId === tabId && details.frameId === 0) done();
    };
    const failed = (details: chrome.webNavigation.WebNavigationFramedErrorCallbackDetails) => {
      if (details.tabId === tabId && details.frameId === 0) done(new BridgeCommandError("nav_failed", details.error));
    };
    function done(error?: Error): void {
      clearTimeout(timer);
      chrome.webNavigation.onCompleted.removeListener(complete);
      chrome.webNavigation.onErrorOccurred.removeListener(failed);
      if (error) reject(error); else resolve();
    }
    chrome.webNavigation.onCompleted.addListener(complete);
    chrome.webNavigation.onErrorOccurred.addListener(failed);
  });
}

async function navigate(tabId: number, url: URL, requestId: string): Promise<{ url: string; title: string }> {
  if (!(await askOrigin(url, requestId))) {
    throw new BridgeCommandError("origin_denied", "site permission was denied", { origin: url.origin });
  }
  return withOriginGate(
    tabId,
    requestId,
    async () => {
      const waiting = settle(tabId);
      await chrome.tabs.update(tabId, { url: url.href });
      await waiting;
      return page(tabId);
    },
    [url.origin],
  );
}

export async function open(params: Record<string, unknown>, requestId: string): Promise<Record<string, unknown>> {
  const url = safeHttpUrl(params.url);
  const previous = (await getSession()).surface;
  if (previous) {
    try {
      await chrome.tabs.get(previous.tabId);
      const result = await navigate(previous.tabId, url, requestId);
      return { tab: surfaceToken(previous), ...result };
    } catch (error) {
      if (!(error instanceof BridgeCommandError && error.code === "tab_closed")) throw error;
    }
  }
  if (!(await askOrigin(url, requestId))) {
    throw new BridgeCommandError("origin_denied", "site permission was denied", { origin: url.origin });
  }
  // Create about:blank first. Creating directly at the destination starts its
  // redirect chain before a debugger can attach, leaving a race where a second
  // origin could receive cookies before the permission gate exists.
  const tab = await chrome.tabs.create({ active: false, url: "about:blank" });
  if (tab.id === undefined) throw new BridgeCommandError("internal", "Chrome created no tab id");
  const surface: StoredSurface = {
    tabId: tab.id,
    nonce: crypto.randomUUID().replaceAll("-", ""),
    epoch: 1,
  };
  await setSurface(surface);
  await attach(tab.id);
  const live = await navigate(tab.id, url, requestId);
  return { tab: surfaceToken(surface), ...live };
}

export async function goto(params: Record<string, unknown>, requestId: string): Promise<Record<string, unknown>> {
  const surface = await requireSurface(params.tab);
  const result = await navigate(surface.tabId, safeHttpUrl(params.url), requestId);
  surface.epoch += 1;
  await setSurface(surface);
  return result;
}

export async function status(params: Record<string, unknown>): Promise<Record<string, unknown>> {
  const surface = params.tab ? await requireSurface(params.tab) : (await getSession()).surface;
  if (!surface) return { origin_mode: "default-deny" };
  return { tab: surfaceToken(surface), ...(await page(surface.tabId)), origin_mode: "default-deny" };
}

export async function close(params: Record<string, unknown>): Promise<Record<string, unknown>> {
  const surface = await requireSurface(params.tab);
  await detach(surface.tabId);
  try { await chrome.tabs.remove(surface.tabId); } catch { /* already gone is success */ }
  await setSurface();
  return {};
}
