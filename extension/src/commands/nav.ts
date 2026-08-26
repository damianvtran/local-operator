import { attach, BridgeCommandError, cdp, detach, requireSurface } from "../cdp";
import { dropLogCapture, startLogCapture } from "../log-capture";
import { askOrigin, safeHttpUrl, withOriginGate } from "../origins";
import { settle } from "../settle";
import {
  atSurfaceCap,
  getSurfaces,
  MAX_SURFACES,
  putSurface,
  removeSurface,
  surfaceToken,
  type StoredSurface,
} from "../state";

async function page(tabId: number): Promise<{ url: string; title: string }> {
  const tab = await chrome.tabs.get(tabId);
  return { url: tab.url ?? "", title: tab.title ?? "" };
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

/**
 * Open a surface. Two explicit modes, decided by the caller's `tab` param:
 *
 * - no `tab` → a brand-NEW tab and surface. Never reuses an existing one:
 *   the old "reuse whatever surface exists" behaviour meant a second
 *   session's open silently stole the first session's tab mid-task (the
 *   parallelism-breaking hijack this multi-surface model exists to fix).
 * - `tab` set → navigate THAT surface (the resume path). Same effect the old
 *   reuse had, but explicit and scoped to the caller's own handle, so it can
 *   only ever re-drive a tab the caller already owns.
 */
export async function open(params: Record<string, unknown>, requestId: string): Promise<Record<string, unknown>> {
  const url = safeHttpUrl(params.url);
  if (params.tab !== undefined && params.tab !== null && params.tab !== "") {
    const surface = await requireSurface(params.tab);
    // Re-arm log capture on the resumed surface: after a worker restart the
    // ring buffer was lost and the domains may need re-enabling, and
    // startLogCapture is idempotent when they are already on.
    await startLogCapture(surface.tabId, cdp);
    const result = await navigate(surface.tabId, url, requestId);
    return { tab: surfaceToken(surface), ...result };
  }
  const surfaces = await getSurfaces();
  if (atSurfaceCap(surfaces)) {
    // Typed refusal, not silent reuse: each parallel session opens its own
    // tab now, so an unbounded map would let an agent fleet spray tabs into
    // the user's real browser. See MAX_SURFACES for the cap rationale.
    throw new BridgeCommandError(
      "tab_limit",
      `already driving ${MAX_SURFACES} tabs — close one (action 'close' with its handle) before opening another`,
      { limit: MAX_SURFACES, tabs: Object.keys(surfaces) },
    );
  }
  if (!(await askOrigin(url, requestId))) {
    throw new BridgeCommandError("origin_denied", "site permission was denied", { origin: url.origin });
  }
  // Create about:blank first. Creating directly at the destination starts its
  // redirect chain before a debugger can attach, leaving a race where a second
  // origin could receive cookies before the permission gate exists.
  const tab = await chrome.tabs.create({ active: false, url: "about:blank" });
  if (tab.id === undefined) throw new BridgeCommandError("internal", "Chrome created no tab id");
  const now = Date.now();
  const surface: StoredSurface = {
    tabId: tab.id,
    nonce: crypto.randomUUID().replaceAll("-", ""),
    epoch: 1,
    createdAt: now,
    lastUsedAt: now,
  };
  await putSurface(surface);
  await attach(tab.id);
  // Start buffering console/runtime logs immediately after attach and BEFORE
  // navigating, so the `logs` command captures output from the destination
  // page's very first script (finding: logs must be "since the surface opened").
  await startLogCapture(tab.id, cdp);
  const live = await navigate(tab.id, url, requestId);
  return { tab: surfaceToken(surface), ...live };
}

export async function goto(params: Record<string, unknown>, requestId: string): Promise<Record<string, unknown>> {
  const surface = await requireSurface(params.tab);
  const result = await navigate(surface.tabId, safeHttpUrl(params.url), requestId);
  surface.epoch += 1;
  await putSurface(surface);
  return result;
}

export async function status(params: Record<string, unknown>): Promise<Record<string, unknown>> {
  if (params.tab) {
    const surface = await requireSurface(params.tab);
    return { tab: surfaceToken(surface), ...(await page(surface.tabId)), origin_mode: "default-deny" };
  }
  // No handle: report the most recently driven live surface, if any. With
  // several sessions each owning a tab there is no single "the" surface any
  // more, so recency is the most honest one-line answer.
  const surfaces = Object.values(await getSurfaces()).sort((a, b) => b.lastUsedAt - a.lastUsedAt);
  for (const surface of surfaces) {
    try {
      return { tab: surfaceToken(surface), ...(await page(surface.tabId)), origin_mode: "default-deny" };
    } catch {
      await removeSurface(surfaceToken(surface));
    }
  }
  return { origin_mode: "default-deny" };
}

/**
 * List every live extension-owned surface, pruning entries whose Chrome tab
 * is gone (detaching leftover debugger sessions best-effort as we go).
 *
 * URL/title come from chrome.tabs at call time, never from storage, so the
 * listing cannot show a page the tab has since left. This is the discovery
 * verb for parallel sessions: read-only awareness of ALL surfaces, including
 * other sessions' — driving one still requires its exact token (nonce and
 * all), so listing does not grant control.
 */
export async function tabs(_params: Record<string, unknown>): Promise<Record<string, unknown>> {
  const surfaces = await getSurfaces();
  const live: Record<string, unknown>[] = [];
  for (const [token, surface] of Object.entries(surfaces)) {
    try {
      const tab = await chrome.tabs.get(surface.tabId);
      live.push({
        tab: token,
        url: tab.url ?? "",
        title: tab.title ?? "",
        createdAt: surface.createdAt,
        lastUsedAt: surface.lastUsedAt,
      });
    } catch {
      // Tab gone (user closed it, browser restarted): prune so the map cannot
      // fill with dead entries that count toward the surface cap.
      dropLogCapture(surface.tabId);
      await detach(surface.tabId);
      await removeSurface(token);
    }
  }
  live.sort((a, b) => Number(b.lastUsedAt) - Number(a.lastUsedAt));
  return { tabs: live, limit: MAX_SURFACES };
}

async function closeSurface(token: string, surface: StoredSurface): Promise<void> {
  dropLogCapture(surface.tabId);
  await detach(surface.tabId);
  try { await chrome.tabs.remove(surface.tabId); } catch { /* already gone is success */ }
  await removeSurface(token);
}

/**
 * Close ONE surface. With a `tab` param, exactly that surface (each session
 * closes its own tab when done). Without one — the pre-multi-tab call shape —
 * close the sole surface if exactly one exists; with several, refuse and name
 * the live handles, because guessing which tab another session still needs is
 * precisely the hijack this model removes.
 */
export async function close(params: Record<string, unknown>): Promise<Record<string, unknown>> {
  if (params.tab !== undefined && params.tab !== null && params.tab !== "") {
    const surface = await requireSurface(params.tab);
    await closeSurface(surfaceToken(surface), surface);
    return {};
  }
  const surfaces = await getSurfaces();
  const entries = Object.entries(surfaces);
  if (entries.length === 0) {
    throw new BridgeCommandError("tab_closed", "no browser tab is open");
  }
  if (entries.length > 1) {
    throw new BridgeCommandError(
      "internal",
      `several tabs are open (${entries.map(([token]) => token).join(", ")}) — pass the handle of the one to close`,
      { tabs: entries.map(([token]) => token) },
    );
  }
  const [token, surface] = entries[0]!;
  await closeSurface(token, surface);
  return {};
}
