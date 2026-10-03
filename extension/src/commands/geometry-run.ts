import { BridgeCommandError } from "../cdp";
import { SCRIPTING_DEADLINE_MS, deadline } from "../settle";

/* One page-execution shape for the three geometry reads, so their error
 * mappings cannot drift apart. They did drift: with each handler inlining
 * `chrome.scripting.executeScript`, a selector that is not valid CSS reaches
 * this worker as the isolated world's REJECTION (the page function throws
 * SyntaxError before producing any value; `deadline` passes rejections through
 * unchanged), and an inlined handler let it fall to the worker's last gate —
 * `internal` — while the desktop app's host answers the same mistake with a
 * typed `element_not_found` ("selector X is not valid", the app's
 * `INVALID_SELECTOR` at page.ts). The two hosts must answer identically for
 * the same caller input, so the recognition and the wording live here, once. */

/** The message a selector that is not valid CSS produces inside the page's
 *  isolated world. Deliberately the SAME pattern as the app host's
 *  `INVALID_SELECTOR` — recognising the same rejection is what keeps the two
 *  answers identical; module scope because there is no reason to compile it
 *  per call. */
const INVALID_SELECTOR = /SyntaxError|not a valid selector/i;

/**
 * Run one of the fixed geometry functions in the page's isolated world and
 * return its value.
 *
 * Errors, in the order the caller can meet them:
 *
 * 1. A page-side rejection whose message carries a syntax error is the
 *    CALLER's selector, not a host fault — mapped to the typed
 *    `element_not_found` with the app host's exact wording. Only when a
 *    `selector` is given: hit_test's inputs are numbers, so a rejection there
 *    has no selector to name and rethrows to the worker's last gate rather
 *    than inventing one.
 * 2. `null`/`undefined` from the frame — the function's matched-nothing
 *    signal, or no result frame at all — becomes the caller's `missing`
 *    message (each command names its own subject) as a typed
 *    `element_not_found`.
 */
export async function runGeometry<Value>(
  surface: { tabId: number },
  func: (...args: any[]) => Value | null | undefined,
  args: unknown[],
  copy: { selector?: string; missing: string },
): Promise<NonNullable<Value>> {
  let results: Array<{ result?: Value | null | undefined }>;
  try {
    // Runs page script, so it shares the CDP risk profile and `read`'s 20 s
    // daemon budget (settle.ts's deadline table). The cast narrows the lib's
    // `InjectionResult<Awaited<Result>>` — whose `Awaited` cannot resolve back
    // through this function's unresolved `Value` — to the one field read here.
    results = (await deadline(
      chrome.scripting.executeScript({
        target: { tabId: surface.tabId },
        func,
        args,
      }),
      SCRIPTING_DEADLINE_MS,
      `chrome.scripting.executeScript(${surface.tabId})`,
    )) as Array<{ result?: Value | null | undefined }>;
  } catch (error) {
    if (
      copy.selector !== undefined &&
      error instanceof Error &&
      INVALID_SELECTOR.test(error.message)
    ) {
      throw new BridgeCommandError("element_not_found", `selector ${copy.selector} is not valid`);
    }
    throw error;
  }
  const value = results[0]?.result;
  if (value === null || value === undefined) {
    throw new BridgeCommandError("element_not_found", copy.missing);
  }
  return value as NonNullable<Value>;
}
