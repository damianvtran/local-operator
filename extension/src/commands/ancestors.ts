import { BridgeCommandError, requireSurface } from "../cdp";
import { ancestors as ancestorsInPage } from "../driver/geometry-read";
import { runGeometry } from "./geometry-run";

/** The chain bound the page function applies when `depth` is absent, mirrored
 *  here so the value sent over the wire is already sane. The page function
 *  re-applies both the default and the cap, because it (not this handler) is
 *  what bounds a hostile peer's arguments. */
const DEFAULT_DEPTH = 12;
const MAX_DEPTH = 16;

/**
 * The ancestor chain of the first element matching `selector`, from the element
 * itself upward to and including `document.documentElement`.
 *
 * `depth` bounds the walk (default 12, hard cap 16). The shared driver function
 * (driver/geometry-read.ts) runs in the page's isolated world and owns the
 * cap; this handler validates + pre-clamps the number and hands execution to
 * `runGeometry`, which maps the function's `null` (no match) to the typed
 * `element_not_found` — and a selector that is not valid CSS to the app host's
 * exact "is not valid" copy.
 */
export async function ancestors(params: Record<string, unknown>): Promise<Record<string, unknown>> {
  const surface = await requireSurface(params.tab);
  const selector = typeof params.selector === "string" ? params.selector.trim() : "";
  if (!selector) throw new BridgeCommandError("element_not_found", "selector is required");
  const raw =
    typeof params.depth === "number" && Number.isFinite(params.depth)
      ? Math.floor(params.depth)
      : DEFAULT_DEPTH;
  const depth = Math.max(1, Math.min(MAX_DEPTH, raw));
  return runGeometry(surface, ancestorsInPage, [selector, depth], {
    selector,
    missing: `selector ${selector} matched nothing`,
  });
}
