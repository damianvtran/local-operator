import { requireSurface } from "../cdp";
import { readLogs } from "../log-capture";

// Levels the tool filters on. "all" (the default) keeps everything; anything
// else must match one of these normalized levels, so a typo'd filter returns an
// empty set predictably rather than silently keeping all.
const LEVELS = new Set(["error", "warning", "info", "log", "all"]);

/**
 * Return the buffered console/runtime logs for the driven tab, newest-last.
 *
 * The entries were captured from the moment the surface opened (see
 * log-capture.ts, wired in `open`). This handler is a pure read of that ring
 * buffer plus optional level/limit filtering — no CDP round trip, so it stays
 * cheap and cannot fail on a slow page.
 */
export async function logs(params: Record<string, unknown>): Promise<Record<string, unknown>> {
  const surface = await requireSurface(params.tab);
  const rawLevel = typeof params.level === "string" ? params.level.trim().toLowerCase() : "all";
  const level = LEVELS.has(rawLevel) ? rawLevel : "all";
  // limit 0/absent means "no cap"; a positive value keeps the most recent n.
  const limit = typeof params.limit === "number" && params.limit > 0 ? Math.floor(params.limit) : 0;
  const entries = readLogs(surface.tabId, level, limit);
  const tab = await chrome.tabs.get(surface.tabId);
  return { entries, level, url: tab.url ?? "", title: tab.title ?? "" };
}
