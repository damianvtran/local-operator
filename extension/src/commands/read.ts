import { BridgeCommandError, requireSurface } from "../cdp";
import { CHROME_API_DEADLINE_MS, SCRIPTING_DEADLINE_MS, deadline } from "../settle";

export async function readPage(params: Record<string, unknown>): Promise<Record<string, unknown>> {
  const surface = await requireSurface(params.tab);
  const selector = typeof params.selector === "string" && params.selector ? params.selector : "body";
  // Runs page script, so it shares the CDP risk profile and `read`'s 20 s
  // daemon budget (settle.ts's deadline table).
  const results = await deadline(chrome.scripting.executeScript({
    target: { tabId: surface.tabId },
    func: (query: string) => {
      const element = document.querySelector(query);
      if (!element) return null;
      const clone = element.cloneNode(true) as Element;
      clone.querySelectorAll("script,style,noscript,template").forEach((node) => node.remove());
      return (clone.textContent ?? "")
        .replace(/[ \t\u00a0]+/g, " ")
        .replace(/\n\s*\n\s*\n+/g, "\n\n")
        .trim();
    },
    args: [selector],
  }), SCRIPTING_DEADLINE_MS, `chrome.scripting.executeScript(${surface.tabId})`);
  const text = results[0]?.result;
  if (text === null) throw new BridgeCommandError("element_not_found", `selector ${selector} matched nothing`);
  const tab = await deadline(
    chrome.tabs.get(surface.tabId),
    CHROME_API_DEADLINE_MS,
    `chrome.tabs.get(${surface.tabId})`,
  );
  return { text: String(text ?? ""), url: tab.url ?? "", title: tab.title ?? "" };
}
