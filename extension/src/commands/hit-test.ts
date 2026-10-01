import { BridgeCommandError, requireSurface } from "../cdp";
import { hitTest as hitTestInPage } from "../driver/geometry-read";
import { SCRIPTING_DEADLINE_MS, deadline } from "../settle";

/**
 * Hit-test a viewport point: the stack of elements under (x, y), topmost first.
 *
 * `x`/`y` are viewport coordinates in CSS pixels — the same space mouse events
 * and `document.elementsFromPoint` use, NOT the scroll deltas the `scroll`
 * action reads from the same parameter names. The shared driver function
 * (driver/geometry-read.ts) runs in the page's isolated world and enforces the
 * 8-element cap; this handler validates the numbers and maps the function's
 * `null` (nothing at that point, e.g. off-viewport) to the typed
 * `element_not_found`.
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
  const results = await deadline(chrome.scripting.executeScript({
    target: { tabId: surface.tabId },
    func: hitTestInPage,
    args: [x, y],
  }), SCRIPTING_DEADLINE_MS, `chrome.scripting.executeScript(${surface.tabId})`);
  const value = results[0]?.result;
  if (value === null || value === undefined) {
    throw new BridgeCommandError("element_not_found", `no element at point (${x}, ${y})`);
  }
  return value as Record<string, unknown>;
}
