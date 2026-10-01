import { BridgeCommandError, requireSurface } from "../cdp";
import { readStyles } from "../driver/geometry-read";
import { SCRIPTING_DEADLINE_MS, deadline } from "../settle";

/**
 * Read the bounding rect + computed styles of up to 5 elements matching
 * `selector`, plus each element's own inline custom properties (`--*`).
 *
 * The shared driver function (driver/geometry-read.ts) runs IN the page's
 * isolated world and is the authority for every bound (5 matches, 30
 * properties, 120-char class names, 200-char values); this handler only
 * validates the wire arguments' shapes, because they cross a process boundary
 * (daemon -> worker) before reaching the page, and maps the function's `null`
 * (selector matched nothing) to the typed `element_not_found` the tool expects
 * — the same shape `read` uses for the same condition.
 */
export async function styles(params: Record<string, unknown>): Promise<Record<string, unknown>> {
  const surface = await requireSurface(params.tab);
  const selector = typeof params.selector === "string" ? params.selector.trim() : "";
  if (!selector) throw new BridgeCommandError("element_not_found", "selector is required");
  // Wire-shape guard only: keep strings, cap the requested extras at 20. The
  // page function dedupes them against its defaults and caps the combined list
  // at 30, so an over-long list is clipped by the source of truth, not here.
  const requested = Array.isArray(params.properties) ? params.properties : [];
  const properties = requested
    .filter((entry): entry is string => typeof entry === "string")
    .slice(0, 20);
  // Runs page script, so it shares the CDP risk profile and `read`'s 20 s
  // daemon budget (settle.ts's deadline table).
  const results = await deadline(chrome.scripting.executeScript({
    target: { tabId: surface.tabId },
    func: readStyles,
    args: [selector, properties],
  }), SCRIPTING_DEADLINE_MS, `chrome.scripting.executeScript(${surface.tabId})`);
  const value = results[0]?.result;
  // `null` is the function's matched-nothing signal; `undefined` means the
  // frame produced no result at all (e.g. the frame went away mid-call), which
  // is a not-found for the caller's purposes rather than a silent `{}`.
  if (value === null || value === undefined) {
    throw new BridgeCommandError("element_not_found", `selector ${selector} matched nothing`);
  }
  return value as Record<string, unknown>;
}
