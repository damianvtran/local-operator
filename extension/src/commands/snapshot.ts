import { compactAX, type AXNode } from "../ax-compact";
import { cdp, requireSurface } from "../cdp";
import { setRefs } from "../state";

// Serializes the enable→read→disable window below. The worker dispatches
// daemon frames fire-and-forget (worker.ts onmessage), so two concurrent
// snapshots could otherwise interleave as enable/enable/read/DISABLE/read —
// the second read then hits a disabled a11y engine, the exact one-node bug
// this file fixes. The daemon pipelines commands per tab today, so this is
// latent, but the extension should not depend on that scheduling promise.
// A promise chain (not a refcount) keeps it simple: snapshots are rare and
// heavy, so serializing them costs nothing observable. Failures are swallowed
// by each link's own try/finally, so the chain can never poison later calls.
let axQueue: Promise<unknown> = Promise.resolve();

export async function snapshot(params: Record<string, unknown>): Promise<Record<string, unknown>> {
  const surface = await requireSurface(params.tab);
  // Accessibility.getFullAXTree only returns a populated tree once
  // Accessibility.enable has run on THIS debugger session; without it Chrome
  // answers with just the root node (observed live: one-line snapshots). We
  // disable again after the read so the a11y engine is not kept hot on the tab
  // between snapshots — the stored refs are backendDOMNodeIds, which belong to
  // the DOM (not the AX tree) and remain resolvable after disable.
  const run = async (): Promise<{ nodes: AXNode[] }> => {
    await cdp(surface.tabId, "Accessibility.enable");
    try {
      return await cdp<{ nodes: AXNode[] }>(surface.tabId, "Accessibility.getFullAXTree");
    } finally {
      // Best-effort: a failed disable (tab closing mid-command) must not mask
      // the snapshot result or the original error.
      await cdp(surface.tabId, "Accessibility.disable").catch(() => {});
    }
  };
  const result = await (axQueue = axQueue.catch(() => {}).then(run)) as { nodes: AXNode[] };
  const rendered = compactAX(result.nodes, surface.epoch);
  await setRefs(rendered.refs);
  const tab = await chrome.tabs.get(surface.tabId);
  return { snapshot: rendered.snapshot, url: tab.url ?? "", title: tab.title ?? "" };
}
