import { BridgeCommandError } from "./errors";
import { dropLogCapture } from "./log-capture";
import { CHROME_API_DEADLINE_MS, CDP_ATTACH_DEADLINE_MS, CDP_DEADLINE_MS, deadline } from "./settle";
import { getSurfaces, removeSurface, resolveSurfaceToken, touchSurface, type StoredSurface } from "./state";

// Re-exported so the many existing `import { BridgeCommandError } from
// "./cdp"` sites stay valid; see errors.ts for why the value's home moved.
export { BridgeCommandError } from "./errors";

const attached = new Set<number>();

/**
 * Whether an error is one of OUR OWN per-call deadlines (`settle.deadline`)
 * rather than a refusal from Chrome.
 *
 * Every `catch` that reads a failure as "the tab is gone, prune the surface"
 * must discriminate on this first. A stall is a fact about this worker's event
 * loop, not about the tab: pruning a live surface because one `chrome.tabs.get`
 * was slow would tell the session its tab had been closed — the same class of
 * confident misdiagnosis that made the reported wedge expensive.
 */
export function isStalled(error: unknown): boolean {
  return error instanceof BridgeCommandError && Boolean(error.data.stalled);
}

export async function requireSurface(token: unknown): Promise<StoredSurface> {
  // Exact-token lookup in the surfaces map: the nonce is part of the key, so a
  // handle from another session (or a guessed tab id) resolves to nothing and
  // keeps the same tab_closed shape callers already handle. This is what lets
  // parallel sessions share the map without being able to drive each other's
  // tabs by accident.
  const surface = resolveSurfaceToken(token, await getSurfaces());
  if (!surface) {
    throw new BridgeCommandError("tab_closed", "the browser tab handle is stale");
  }
  try {
    await deadline(
      chrome.tabs.get(surface.tabId),
      CHROME_API_DEADLINE_MS,
      `chrome.tabs.get(${surface.tabId})`,
    );
  } catch (error) {
    // Our own deadline is not a missing tab (see isStalled).
    if (isStalled(error)) throw error;
    // The Chrome tab is gone: full prune, identical to the `tabs` prune site
    // (review finding m1) — dropping only the map entry leaked the log ring
    // buffer and left the dead tabId in the `attached` set for the worker's
    // lifetime.
    await pruneSurface(String(token), surface.tabId);
    throw new BridgeCommandError("tab_closed", "the browser tab was closed");
  }
  // Recency for the `tabs` listing; best-effort, never on the command's
  // critical path for correctness. Conditional on the entry still existing so
  // it cannot resurrect a concurrently-pruned surface (finding m5).
  surface.lastUsedAt = Date.now();
  await touchSurface(String(token), surface.lastUsedAt);
  return surface;
}

/**
 * The ONE dead-surface cleanup: map entry, log ring buffer, debugger session.
 * Every prune site (requireSurface, status, tabs) must go through this — the
 * three previously diverged and two of them leaked buffers/attachments
 * (finding m1).
 */
export async function pruneSurface(token: string, tabId: number): Promise<void> {
  dropLogCapture(tabId);
  await detach(tabId);
  await removeSurface(token);
}

/**
 * Prune every surface bound to one Chrome tab id.
 *
 * `attach` is addressed by numeric tab id (that is how CDP works) but a surface
 * can only be removed by its TOKEN, so a failure that has to retire the surface
 * must go through the map. Best-effort by construction: this runs on the way to
 * an error that has already been decided, and a storage failure here must not
 * replace it with a different one.
 */
async function pruneSurfaceByTabId(tabId: number): Promise<void> {
  try {
    const surfaces = await getSurfaces();
    for (const token of Object.keys(surfaces)) {
      if (surfaces[token]?.tabId === tabId) await pruneSurface(token, tabId);
    }
  } catch (error) {
    console.warn(`could not prune surfaces for tab ${tabId}`, error);
  }
}

/** Chrome's refusal to debug a page belonging to a DIFFERENT extension. */
const DIFFERENT_EXTENSION = "chrome-extension:// URL of different extension";

export async function attach(tabId: number): Promise<void> {
  if (attached.has(tabId)) return;
  try {
    await deadline(
      chrome.debugger.attach({ tabId }, "1.3"),
      CDP_ATTACH_DEADLINE_MS,
      `chrome.debugger.attach(${tabId})`,
    );
    // Reached only when the attach RESOLVED. A deadline rejection lands in the
    // catch below WITHOUT registering the tab here, and the abandoned attach can
    // still complete in Chrome afterwards — leaving a live debugger session (and
    // its "…is debugging this browser" infobar) that a later `detach()` skips
    // because `attached.has(tabId)` is false. That residue is deliberately
    // tolerated rather than tracked: it reconciles on the next `cdp()` for this
    // tab, where the re-attach is refused with "already attached" and
    // `ownAttachment` adopts the surviving session, and until then nothing can
    // drive the tab anyway (review R1-4). `onDetach` would not cover it even if
    // we wanted it to, because it does not fire for an attach that never
    // registered locally.
    attached.add(tabId);
  } catch (error) {
    // A deadline here is our own bound firing, not a refusal: rethrow it so the
    // `stalled` discriminator survives instead of being flattened into a
    // generic internal error by the fallback below.
    if (isStalled(error)) throw error;
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
    if (message.includes(DIFFERENT_EXTENSION)) {
      // The tab is (or redirected to) another extension's page, which Chrome
      // refuses to debug. It is alive but can NEVER be driven, and unlike a
      // failed `chrome.tabs.get` this refusal used to leave the surface in the
      // map — so every later command for it failed the same way permanently,
      // including a fresh `open` (reported from a live session: a `scroll`
      // timeout, then this message on the retry, then the same on `open`).
      // Retire the surface so the session's next `open` starts from a clean
      // map, and say what happened and what to do about it: `internal` plus a
      // data discriminator is this protocol's pattern for exactly this shape
      // (`tab_crashed`), and a NEW wire code would be silently dropped by any
      // already-released daemon that does not know it.
      await pruneSurfaceByTabId(tabId);
      throw new BridgeCommandError(
        "internal",
        "Chrome refused to debug this tab: it is another extension's page",
        { undrivable_tab: "different_extension" },
      );
    }
    throw new BridgeCommandError("internal", message);
  }
}

async function ownAttachment(tabId: number): Promise<boolean> {
  // A tab we can still drive answers a trivial CDP command; a tab held by a
  // foreign debugger rejects it. This distinguishes our surviving attachment
  // from DevTools without a fragile string match on Chrome's error text.
  //
  // Bounded, and deliberately still catch-all: a probe that does not answer is
  // a session we cannot drive, so `false` (⇒ debugger_conflict) is the answer
  // this function exists to give. A per-call deadline that turned a REAL
  // conflict into a generic stall would be the worse failure.
  try {
    await deadline(
      chrome.debugger.sendCommand({ tabId }, "Runtime.evaluate", { expression: "1" }),
      CDP_ATTACH_DEADLINE_MS,
      `chrome.debugger.sendCommand(Runtime.evaluate probe on ${tabId})`,
    );
    return true;
  } catch {
    return false;
  }
}

export async function detach(tabId: number): Promise<void> {
  if (!attached.has(tabId)) return;
  try {
    await deadline(
      chrome.debugger.detach({ tabId }),
      CDP_ATTACH_DEADLINE_MS,
      `chrome.debugger.detach(${tabId})`,
    );
  } catch {
    // Close is idempotent; a browser-initiated detach already achieved it. A
    // deadline lands here too, on purpose: a detach we cannot complete must not
    // park pruneSurface (and through it the store queue) forever.
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
    return (await deadline(
      chrome.debugger.sendCommand({ tabId }, method, params),
      CDP_DEADLINE_MS,
      `chrome.debugger.sendCommand(${method})`,
    )) as T;
  } catch (error) {
    if (isStalled(error)) throw error;
    throw new BridgeCommandError("internal", String(error));
  }
}

chrome.debugger.onDetach.addListener((source) => {
  if (source.tabId !== undefined) attached.delete(source.tabId);
});
