import { requireSurface } from "../cdp";
import { loggedOrigins, readLogs } from "../log-capture";
import { originAdmitted } from "../surface-origin";
import { CHROME_API_DEADLINE_MS, deadline } from "../settle";

// Levels the tool filters on. "all" (the default) keeps everything; anything
// else must match one of these normalized levels, so a typo'd filter returns an
// empty set predictably rather than silently keeping all.
const LEVELS = new Set(["error", "warning", "info", "log", "all"]);

/**
 * Return the buffered console/runtime logs for the driven tab, newest-last.
 *
 * The entries were captured from the moment the surface opened (see
 * log-capture.ts, wired in `open`). This handler is a read of that ring
 * buffer, filtered to admitted capture origins, plus optional level/limit
 * filtering — no CDP round trip, so it stays cheap and cannot fail on a slow
 * page.
 */
export async function logs(params: Record<string, unknown>): Promise<Record<string, unknown>> {
  const surface = await requireSurface(params.tab);
  const rawLevel = typeof params.level === "string" ? params.level.trim().toLowerCase() : "all";
  const level = LEVELS.has(rawLevel) ? rawLevel : "all";
  // limit 0/absent means "no cap"; a positive value keeps the most recent n.
  const limit = typeof params.limit === "number" && params.limit > 0 ? Math.floor(params.limit) : 0;
  // Only entries captured on a page this tab may expose NOW: the buffer spans
  // navigations, and an unapproved site the tab visited on its own must not
  // reach the agent through its console output (log-capture.ts ORIGIN TAGS).
  // Output from iframes is judged by the TOP page it ran under, like
  // read/snapshot: an approved page's embedded widgets still log here.
  // One admission check per distinct origin, not per entry.
  const admitted = new Set<string>();
  for (const origin of loggedOrigins(surface.tabId)) {
    if (await originAdmitted(surface, origin)) admitted.add(origin);
  }
  const entries = readLogs(surface.tabId, level, limit, admitted);
  const tab = await deadline(
    chrome.tabs.get(surface.tabId),
    CHROME_API_DEADLINE_MS,
    `chrome.tabs.get(${surface.tabId})`,
  );
  return { entries, level, url: tab.url ?? "", title: tab.title ?? "" };
}
