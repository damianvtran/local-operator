import { BridgeCommandError, requireSurface } from "../cdp";
import { queryElements } from "../driver/element-query";
import { runGeometry } from "./geometry-run";

/**
 * Read-only element query: CSS in, bounded descriptors out.
 *
 * The one question that would have unblocked the measured incident in a single
 * call — "what is the id/name/role of the checkbox next to 'Active'?" — had no
 * route: `read` returns text without attributes, `snapshot` returns an AX tree,
 * and there is no `eval`. This answers it without any write, any navigation or
 * any caller-supplied code: the selector is an ARGUMENT to
 * `document.querySelectorAll`, and the page function (driver/element-query.ts)
 * is the authority for every cap.
 *
 * It rides `runGeometry` rather than a private `executeScript` call so a
 * selector that is not valid CSS gets the SAME typed `element_not_found`
 * ("selector X is not valid") the app host and the other page-script reads
 * answer, instead of the worker's last-gate `internal`.
 *
 * `limit` is pre-clamped here because it crosses a process boundary from a peer
 * whose own validation may be older; the page function clamps it again (1..50)
 * and that copy is the one that bounds the result.
 */
export async function query(params: Record<string, unknown>): Promise<Record<string, unknown>> {
  const surface = await requireSurface(params.tab);
  const selector = typeof params.selector === "string" ? params.selector.trim() : "";
  if (!selector) throw new BridgeCommandError("element_not_found", "selector is required");
  const requested = typeof params.limit === "number" ? params.limit : Number(params.limit);
  const limit = Number.isFinite(requested) && requested > 0 ? Math.floor(requested) : 20;
  // A caller that asked for a selector which matches nothing gets `count: 0`
  // and an empty list, NOT an error: "this selector matches nothing" is the
  // answer to the question asked. `runGeometry`'s null branch exists for the
  // empty-selector case, which the guard above already refused.
  // `missing` is `runGeometry`'s generic fallback and is unreachable here — an
  // empty selector is refused above and `queryElements` reports every other
  // input as a result — but a shared helper's contract is not this command's to
  // narrow (agent review round 1, R1-2).
  const result = await runGeometry(surface, queryElements, [selector, limit], {
    selector,
    missing: `selector ${selector} matched nothing`,
  });
  // The page function REPORTS a bad selector as a value rather than throwing,
  // because the isolated world's rejection message does not survive Electron's
  // wrapper (measured: "Script failed to execute"). This is where that value
  // becomes the caller's typed refusal, in the words the other reads use.
  if ((result as { error?: string }).error === "invalid_selector") {
    throw new BridgeCommandError("element_not_found", `selector ${selector} is not valid`);
  }
  return result;
}
