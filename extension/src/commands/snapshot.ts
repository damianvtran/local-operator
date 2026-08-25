import { compactAX, type AXNode } from "../ax-compact";
import { cdp, requireSurface } from "../cdp";
import { setRefs } from "../state";

export async function snapshot(params: Record<string, unknown>): Promise<Record<string, unknown>> {
  const surface = await requireSurface(params.tab);
  const result = await cdp<{ nodes: AXNode[] }>(surface.tabId, "Accessibility.getFullAXTree");
  const rendered = compactAX(result.nodes, surface.epoch);
  await setRefs(rendered.refs);
  const tab = await chrome.tabs.get(surface.tabId);
  return { snapshot: rendered.snapshot, url: tab.url ?? "", title: tab.title ?? "" };
}
