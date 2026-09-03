import { attach, BridgeCommandError, cdp, detach, pruneSurface, requireSurface } from "../cdp";
import { dropLogCapture, startLogCapture } from "../log-capture";
import {
  askOrigin,
  ensureTopLevelAccess,
  safeHttpUrl,
  withOriginGate,
  type OriginAdmission,
} from "../origins";
import { settle } from "../settle";
import { reconcileTabGroup } from "../tab-groups";
import {
  atSurfaceCap,
  getSurfaces,
  MAX_SURFACES,
  putSurface,
  redactToken,
  removeSurface,
  surfaceToken,
  type StoredSurface,
} from "../state";

/**
 * The surfaces map with dead entries pruned (full cleanup via pruneSurface).
 *
 * Called before the cap check and by `tabs`: counting stale entries against
 * the cap refused fresh opens after a browser restart until something else
 * happened to prune (review finding m2), and every prune site must do the
 * same buffer/debugger cleanup (finding m1).
 */
async function liveSurfaces(): Promise<Record<string, StoredSurface>> {
  const surfaces = await getSurfaces();
  const live: Record<string, StoredSurface> = {};
  for (const [token, surface] of Object.entries(surfaces)) {
    try {
      await chrome.tabs.get(surface.tabId);
      live[token] = surface;
    } catch {
      await pruneSurface(token, surface.tabId);
    }
  }
  return live;
}

/** The session-scoped requester the tool attaches (session:<id>); absent or
 * foreign-shaped values yield "" so admission falls back to the command-id
 * handoff only — never trusting a caller-invented identity. */
function sessionRequester(params: Record<string, unknown>): string {
  const value = typeof params.requester === "string" ? params.requester.trim() : "";
  return value.startsWith("session:") ? value : "";
}

async function page(tabId: number): Promise<{ url: string; title: string }> {
  const tab = await chrome.tabs.get(tabId);
  return { url: tab.url ?? "", title: tab.title ?? "" };
}

async function navigate(tabId: number, url: URL, requestId: string, admission: OriginAdmission): Promise<{ url: string; title: string }> {
  // The admission decision was made ONCE at command entry
  // (ensureTopLevelAccess): a grant consumed there is already spent, so this
  // function must NOT consult the grant map again — doing so would let the
  // 10-min TTL lapse between entry and here and drop a granted navigation
  // into the old 60 s prompt, the exact "prompt after the agent thinks it
  // has access" surprise the async flow exists to kill (round-1 M1).
  if (!admission.allowed) {
    if (!(await askOrigin(url, requestId))) {
      throw new BridgeCommandError("origin_denied", "site permission was denied", { origin: url.origin });
    }
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
  // Fail EARLY on a not-yet-allowed top-level origin, before any tab exists
  // and before any prompt is raised: the old behaviour (block this RPC inside
  // the popup prompt) expired unseen because the agent never got a turn to
  // point the user at the popup, and the resulting client timeout read as
  // "bridge unreachable". The typed error teaches the agent the explicit
  // request_access -> tell the user -> await_access flow instead. Redirect
  // hops inside the navigation still take the synchronous prompt — the
  // command is already running there, so an early fail is impossible. Runs
  // BEFORE the cap check so a not-allowed open answers the actionable error
  // even on a full browser, and admission CONSUMES the grant (see
  // ensureTopLevelAccess) so the decision cannot lapse into the old 60 s
  // prompt before navigate() runs.
  const admission = await ensureTopLevelAccess(url, requestId, sessionRequester(params));
  if (params.tab !== undefined && params.tab !== null && params.tab !== "") {
    const surface = await requireSurface(params.tab);
    // Explicit resume refreshes trusted metadata and is the one command that
    // may rejoin a tab the user manually ungrouped.
    await reconcileTabGroup(surface, params, true);
    // Re-arm log capture on the resumed surface: after a worker restart the
    // ring buffer was lost and the domains may need re-enabling, and
    // startLogCapture is idempotent when they are already on.
    await startLogCapture(surface.tabId, cdp);
    const result = await navigate(surface.tabId, url, requestId, admission);
    // Same epoch bump as `goto`: resume navigates to a new document, so
    // pre-resume snapshot refs must fail the epoch gate rather than being
    // pushed against nodes that no longer exist (review finding m3).
    surface.epoch += 1;
    await putSurface(surface);
    return { tab: surfaceToken(surface), ...result };
  }
  const surfaces = await liveSurfaces();
  if (atSurfaceCap(surfaces)) {
    // Typed refusal, not silent reuse: each parallel session opens its own
    // tab now, so an unbounded map would let an agent fleet spray tabs into
    // the user's real browser. See MAX_SURFACES for the cap rationale.
    // Handles are REDACTED: the full token is the drive capability, and an
    // error surface must not hand one session control of another's tab
    // (finding M1).
    throw new BridgeCommandError(
      "tab_limit",
      `already driving ${MAX_SURFACES} tabs — close one (action 'close' with its handle) before opening another`,
      { limit: MAX_SURFACES, tabs: Object.keys(surfaces).map(redactToken) },
    );
  }
  // No separate pre-create gate: ensureTopLevelAccess above already refused a
  // not-allowed origin, and gating again here would double-consume a once
  // grant before navigate() (the single consumption point) ran.
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
  // Security-sensitive ordering stays intact: blank tab persisted, debugger
  // attached, and log capture armed before presentation, then navigation.
  await reconcileTabGroup(surface, params, true);
  const live = await navigate(tab.id, url, requestId, admission);
  return { tab: surfaceToken(surface), ...live };
}

export async function goto(params: Record<string, unknown>, requestId: string): Promise<Record<string, unknown>> {
  const surface = await requireSurface(params.tab);
  await reconcileTabGroup(surface, params, false);
  const url = safeHttpUrl(params.url);
  // Same early refusal as open — see the comment there. Consumption likewise
  // happens HERE, once, bound to this command's request id.
  const admission = await ensureTopLevelAccess(url, requestId, sessionRequester(params));
  const result = await navigate(surface.tabId, url, requestId, admission);
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
  // more, so recency is the most honest one-line answer. The handle is
  // REDACTED: without a tab param the caller has not proven ownership, and a
  // full token here would hand it control of another session's tab (M1).
  const surfaces = Object.entries(await liveSurfaces()).sort(
    (a, b) => b[1].lastUsedAt - a[1].lastUsedAt,
  );
  const first = surfaces[0];
  if (first) {
    return { tab: redactToken(first[0]), ...(await page(first[1].tabId)), origin_mode: "default-deny" };
  }
  return { origin_mode: "default-deny" };
}

/**
 * List every live extension-owned surface, pruning entries whose Chrome tab
 * is gone (full cleanup via pruneSurface).
 *
 * URL/title come from chrome.tabs at call time, never from storage, so the
 * listing cannot show a page the tab has since left. This is the discovery
 * verb for parallel sessions: read-only awareness of ALL surfaces, including
 * other sessions'. Handles are REDACTED (nonce truncated) because the full
 * token IS the drive capability — listing it would grant every session
 * control of every tab (review finding M1). A caller recognises its own tab
 * by prefix-matching the full token it received at open; driving any tab
 * still requires that full token.
 */
export async function tabs(_params: Record<string, unknown>): Promise<Record<string, unknown>> {
  const surfaces = await liveSurfaces();
  const live: Record<string, unknown>[] = [];
  for (const [token, surface] of Object.entries(surfaces)) {
    try {
      const tab = await chrome.tabs.get(surface.tabId);
      live.push({
        tab: redactToken(token),
        url: tab.url ?? "",
        title: tab.title ?? "",
        createdAt: surface.createdAt,
        lastUsedAt: surface.lastUsedAt,
      });
    } catch {
      // Closed between the liveness pass and this read — rare; prune now.
      await pruneSurface(token, surface.tabId);
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
  // The closed handle is RETURNED (`closed`, not `tab` — `tab` on a result is
  // the handle of a tab that still exists, and worker.ts reads it as such) so
  // the worker can name the surface in its `tab_closed` announcement even in
  // the no-param call shape. Without it that shape announced an empty handle,
  // which the daemon can only read as "blank every driven record" — including
  // other sessions' live tabs. The daemon knows the token here; not sending it
  // was the only reason the clear-all path was reachable from a routine close.
  if (params.tab !== undefined && params.tab !== null && params.tab !== "") {
    const surface = await requireSurface(params.tab);
    const token = surfaceToken(surface);
    await closeSurface(token, surface);
    return { closed: token };
  }
  const surfaces = await liveSurfaces();
  const entries = Object.entries(surfaces);
  if (entries.length === 0) {
    throw new BridgeCommandError("tab_closed", "no browser tab is open");
  }
  if (entries.length > 1) {
    // Typed as tab_ambiguous (not internal — finding n1): this is the caller
    // holding an under-specified request, not a bridge fault. Handles are
    // redacted for the same reason as the listing: naming full tokens here
    // would let any session close (or adopt) any tab (M1).
    const redacted = entries.map(([token]) => redactToken(token));
    throw new BridgeCommandError(
      "tab_ambiguous",
      `several tabs are open (${redacted.join(", ")}) — pass the handle of the one to close`,
      { tabs: redacted },
    );
  }
  const [token, surface] = entries[0]!;
  await closeSurface(token, surface);
  return { closed: token };
}
