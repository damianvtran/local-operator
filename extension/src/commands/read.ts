import { BridgeCommandError, requireSurface } from "../cdp";
import { assertDocumentAdmitted } from "../surface-origin";
import { CHROME_API_DEADLINE_MS, SCRIPTING_DEADLINE_MS, deadline } from "../settle";

export async function readPage(params: Record<string, unknown>): Promise<Record<string, unknown>> {
  const surface = await requireSurface(params.tab);
  const selector = typeof params.selector === "string" && params.selector ? params.selector : "body";
  // Runs page script, so it shares the CDP risk profile and `read`'s 20 s
  // daemon budget (settle.ts's deadline table).
  const results = await deadline(chrome.scripting.executeScript({
    target: { tabId: surface.tabId },
    func: (query: string) => {
      // The document's own URL travels with the text: the dispatch gate
      // checked the tab's URL BEFORE this ran, and a page can navigate in
      // between, so the result is only trusted for the document it came from.
      const href = location.href;
      const element = document.querySelector(query);
      if (!element) return { href, text: null };
      const clone = element.cloneNode(true) as Element;
      clone.querySelectorAll("script,style,noscript,template").forEach((node) => node.remove());
      const text = (clone.textContent ?? "")
        .replace(/[ \t\u00a0]+/g, " ")
        .replace(/\n\s*\n\s*\n+/g, "\n\n")
        .trim();
      return { href, text };
    },
    args: [selector],
  }), SCRIPTING_DEADLINE_MS, `chrome.scripting.executeScript(${surface.tabId})`);
  const frame = results[0]?.result;
  // Confirm-only re-check (no once grant is spent here): text read from a
  // document whose origin is not admitted is discarded, never returned.
  await assertDocumentAdmitted(surface, String(frame?.href ?? ""), "", "", false);
  const text = frame?.text;
  if (text === null) throw new BridgeCommandError("element_not_found", `selector ${selector} matched nothing`);
  const tab = await deadline(
    chrome.tabs.get(surface.tabId),
    CHROME_API_DEADLINE_MS,
    `chrome.tabs.get(${surface.tabId})`,
  );
  return { text: String(text ?? ""), url: tab.url ?? "", title: tab.title ?? "" };
}
