import { cdp, requireSurface } from "../cdp";
import { CHROME_API_DEADLINE_MS, deadline } from "../settle";

export async function screenshot(params: Record<string, unknown>): Promise<Record<string, unknown>> {
  const surface = await requireSurface(params.tab);
  const shot = await cdp<{ data: string }>(surface.tabId, "Page.captureScreenshot", { format: "png" });
  const tab = await deadline(
    chrome.tabs.get(surface.tabId),
    CHROME_API_DEADLINE_MS,
    `chrome.tabs.get(${surface.tabId})`,
  );
  return { data: shot.data, url: tab.url ?? "", title: tab.title ?? "" };
}
