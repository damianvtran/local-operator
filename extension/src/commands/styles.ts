import { BridgeCommandError, requireSurface } from "../cdp";
import { readStyles } from "../driver/geometry-read";
import { runGeometry } from "./geometry-run";

/**
 * Read the bounding rect + computed styles of up to 5 elements matching
 * `selector`, plus each element's own inline custom properties (`--*`).
 *
 * The shared driver function (driver/geometry-read.ts) runs IN the page's
 * isolated world and is the authority for every bound (5 matches, 30
 * properties, 120-char class names, 200-char values); this handler only
 * validates the wire arguments' shapes, because they cross a process boundary
 * (daemon -> worker) before reaching the page, and hands execution + error
 * mapping to `runGeometry` — the helper that also gives a bad selector the
 * app host's exact refusal (m1 fix) instead of a bare `internal`.
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
  // daemon budget (settle.ts's deadline table). `null` is the function's
  // matched-nothing signal; `undefined` means the frame produced no result at
  // all (e.g. the frame went away mid-call) — both are a not-found for the
  // caller's purposes rather than a silent `{}`.
  return runGeometry(surface, readStyles, [selector, properties], {
    selector,
    missing: `selector ${selector} matched nothing`,
  });
}
