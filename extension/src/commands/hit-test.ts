import { BridgeCommandError, requireSurface } from "../cdp";
import { hitTest as hitTestInPage } from "../driver/geometry-read";
import { runGeometry } from "./geometry-run";

/**
 * Hit-test a viewport point: the stack of elements under (x, y), topmost first.
 *
 * `x`/`y` are viewport coordinates in CSS pixels — the same space mouse events
 * and `document.elementsFromPoint` use, NOT the scroll deltas the `scroll`
 * action reads from the same parameter names. The shared driver function
 * (driver/geometry-read.ts) runs in the page's isolated world and enforces the
 * 8-element cap; this handler validates the numbers and hands execution to
 * `runGeometry`, which maps the function's `null` (nothing at that point, e.g.
 * off-viewport) to the typed `element_not_found`. No selector on this action:
 * the helper's invalid-selector mapping has no subject to name here, and the
 * numbers cannot produce one.
 */
export async function hitTest(params: Record<string, unknown>): Promise<Record<string, unknown>> {
  const surface = await requireSurface(params.tab);
  const x = params.x;
  const y = params.y;
  if (
    typeof x !== "number" ||
    !Number.isFinite(x) ||
    typeof y !== "number" ||
    !Number.isFinite(y)
  ) {
    // The Python tool refuses this before the wire; this is the last gate for a
    // frame from a peer whose own validation is missing or older.
    throw new BridgeCommandError(
      "internal",
      "'hit_test' needs numeric x and y (viewport coordinates)",
    );
  }
  return runGeometry(surface, hitTestInPage, [x, y], {
    missing: `no element at point (${x}, ${y})`,
  });
}
