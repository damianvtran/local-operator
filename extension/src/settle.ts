/* The host-coupled half of the old `settle.ts`.
 *
 * `settle.ts` held two unrelated things: the `deadline()` helper with its
 * ceiling table — pure, because the numbers are BUDGETS rather than calls, and
 * the one piece a second host wants verbatim — and `settle()` here, which waits
 * on `chrome.webNavigation` events and so cannot leave the extension. The pure
 * half moved to `driver/deadline.ts` (the module the UI vendors); this file
 * keeps `settle()` and re-exports the deadline helpers, which is what keeps the
 * existing `from "./settle"` import sites valid unchanged — the same re-export
 * pattern `cdp.ts` already uses to keep `BridgeCommandError` reachable from
 * `./cdp`.
 *
 * The re-export is a shim, so it must not become the place a host-bound helper
 * is added back: anything that touches `chrome.*` belongs on THIS side of the
 * boundary (outside `driver/`), where tests/driver-host-free.test.mjs cannot
 * see it, not behind the same path the vendored copy is generated from. */
import { BridgeCommandError } from "./driver/errors";

export {
  CDP_DEADLINE_MS,
  CDP_ATTACH_DEADLINE_MS,
  CHROME_API_DEADLINE_MS,
  SCRIPTING_DEADLINE_MS,
  deadline,
} from "./driver/deadline";

/**
 * Resolve when the tab's main frame finishes its next navigation.
 *
 * Shared by nav (goto/open) and input (click-navigation) so both report the
 * page that actually arrived rather than the one being left. Rejects with a
 * typed nav_failed/nav_timeout so the daemon can map it to a model-facing
 * string instead of a silent hang.
 */
export function settle(tabId: number, timeout = 30_000): Promise<void> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(
      () => done(new BridgeCommandError("nav_timeout", "navigation timed out")),
      timeout,
    );
    const complete = (details: chrome.webNavigation.WebNavigationFramedCallbackDetails) => {
      if (details.tabId === tabId && details.frameId === 0) done();
    };
    const failed = (details: chrome.webNavigation.WebNavigationFramedErrorCallbackDetails) => {
      if (details.tabId === tabId && details.frameId === 0) {
        done(new BridgeCommandError("nav_failed", details.error));
      }
    };
    function done(error?: Error): void {
      clearTimeout(timer);
      chrome.webNavigation.onCompleted.removeListener(complete);
      chrome.webNavigation.onErrorOccurred.removeListener(failed);
      if (error) reject(error);
      else resolve();
    }
    chrome.webNavigation.onCompleted.addListener(complete);
    chrome.webNavigation.onErrorOccurred.addListener(failed);
  });
}
