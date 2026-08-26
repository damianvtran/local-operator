import { cdp, requireSurface } from "../cdp";

export async function screenshot(params: Record<string, unknown>): Promise<Record<string, unknown>> {
  const surface = await requireSurface(params.tab);
  const shot = await cdp<{ data: string }>(surface.tabId, "Page.captureScreenshot", { format: "png" });
  const tab = await chrome.tabs.get(surface.tabId);
  return { data: shot.data, url: tab.url ?? "", title: tab.title ?? "" };
}
