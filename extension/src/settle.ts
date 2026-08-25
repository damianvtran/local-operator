import { BridgeCommandError } from "./cdp";

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
