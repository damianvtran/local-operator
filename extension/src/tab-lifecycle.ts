import { pruneSurface } from "./cdp";
import { getSurfaces } from "./state";

/**
 * Reconciliation for tabs that go away WITHOUT a `close` command.
 *
 * WHY THIS EXISTS: the daemon advertises what it is "driving" so the popup and
 * `lop browser status` can show the human which page an agent is on. That
 * record was only ever cleared on disconnect or on a `tab_closed` event — and
 * nothing in this extension ever SENT `tab_closed`. There was no
 * `chrome.tabs.onRemoved` listener at all, so a tab closed by the user, by the
 * agent's own `close`, or by a crash left the daemon advertising it forever.
 * A user who closed every tab in Chrome still saw `driving: <dead url>`, which
 * read as a phantom lock nobody could clear.
 *
 * The surface map was equally affected: entries were reclaimed only lazily,
 * when some later command happened to touch them (`requireSurface`, the `tabs`
 * listing). Until then a dead tab still counted against MAX_SURFACES, so
 * sessions that died without closing could exhaust the cap with ghosts and
 * deny `open` to everyone with a `tab_limit`.
 *
 * MV3 CONSTRAINT: the listener that calls this MUST be registered at the top
 * level of the worker (see worker.ts), not inside a callback or after an
 * await. A service worker is torn down when idle and re-instantiated on an
 * event; only listeners registered during that synchronous first evaluation
 * are wired up to wake it. A listener added later simply does not exist for
 * the events that would have woken the worker — the same reasoning reconnect.ts
 * applies to timers vs alarms.
 */

/** Whether a removed tab id belongs to a surface we own, and its handle. */
export async function ownedSurfaceFor(tabId: number): Promise<string | undefined> {
  const surfaces = await getSurfaces();
  for (const [token, surface] of Object.entries(surfaces)) {
    if (surface.tabId === tabId) return token;
  }
  return undefined;
}

/**
 * Reclaim one removed tab and report the handle to announce as `tab_closed`.
 *
 * Returns `undefined` for a tab we do not own — the user's own tabs are none
 * of our business, and announcing them would blank a driven record that is
 * still live. Cleanup goes through `pruneSurface`, the ONE dead-surface path
 * (map entry + log ring buffer + debugger attachment), so this cannot become
 * the fourth site that leaks one of the three (cdp.ts finding m1).
 */
export async function reclaimRemovedTab(tabId: number): Promise<string | undefined> {
  const token = await ownedSurfaceFor(tabId);
  if (!token) return undefined;
  await pruneSurface(token, tabId);
  return token;
}
