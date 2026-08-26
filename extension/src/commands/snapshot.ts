import { compactAX, type AXNode } from "../ax-compact";
import { cdp, requireSurface } from "../cdp";
import { setRefs } from "../state";

export async function snapshot(params: Record<string, unknown>): Promise<Record<string, unknown>> {
  const surface = await requireSurface(params.tab);
  // Accessibility.getFullAXTree only returns a populated tree once
  // Accessibility.enable has run on THIS debugger session; without it Chrome
  // answers with just the root node (observed live: one-line snapshots). We
  // disable again after the read so the a11y engine is not kept hot on the tab
  // between snapshots — the stored refs are backendDOMNodeIds, which belong to
  // the DOM (not the AX tree) and remain resolvable after disable.
  await cdp(surface.tabId, "Accessibility.enable");
  let result: { nodes: AXNode[] };
  try {
    result = await cdp<{ nodes: AXNode[] }>(surface.tabId, "Accessibility.getFullAXTree");
  } finally {
    // Best-effort: a failed disable (tab closing mid-command) must not mask
    // the snapshot result or the original error.
    await cdp(surface.tabId, "Accessibility.disable").catch(() => {});
  }
  const rendered = compactAX(result.nodes, surface.epoch);
  await setRefs(rendered.refs);
  const tab = await chrome.tabs.get(surface.tabId);
  return { snapshot: rendered.snapshot, url: tab.url ?? "", title: tab.title ?? "" };
}
