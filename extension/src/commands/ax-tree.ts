import type { AXNode } from "../driver/ax-compact";
import { BridgeCommandError, cdp, requireSurface } from "../cdp";
import type { StoredSurface } from "../state";

/* ONE reader for the page's accessibility tree, shared by `snapshot` and the
 * locator resolver in `input.ts`.
 *
 * The per-tab queue is the reason this is a module rather than a copy in each
 * command. `snapshot` serializes enable→read→disable PER TARGET TAB because the
 * worker dispatches frames fire-and-forget: two concurrent reads of the same
 * tab could interleave as enable/enable/read/DISABLE/read, and the second read
 * then hits a disabled a11y engine and risks a degraded tree. A locator
 * resolution is a second reader of the same engine, so it belongs under the
 * same queue — a private copy in `input.ts` would reintroduce exactly the
 * interleaving the queue exists to prevent, and it would do it on the path an
 * agent uses most (click by role/name).
 *
 * Per TAB, not module-global: the CDP `Accessibility` domain is per debuggee,
 * so serializing across tabs protects nothing while spending another owner's
 * budget (the measurement is in snapshot.ts's history and design §5.6).
 */
const axQueues = new Map<number, Promise<unknown>>();

export async function readAXTree(surface: StoredSurface): Promise<AXNode[]> {
  const run = async (): Promise<AXNode[]> => {
    // Enable for the read window as documented CDP hygiene; the live one-line
    // snapshots blamed on a missing enable were actually compactAX pruning
    // ignored wrappers (ax-compact.ts). Kept because it is cheap, matches the
    // documented contract, and the disable keeps the engine cold between calls.
    await cdp(surface.tabId, "Accessibility.enable");
    try {
      const tree = await cdp<{ nodes?: AXNode[] }>(surface.tabId, "Accessibility.getFullAXTree");
      return Array.isArray(tree?.nodes) ? tree.nodes : [];
    } finally {
      // Best-effort: a failed disable (tab closing mid-command) must not mask
      // the read's result or its original error.
      await cdp(surface.tabId, "Accessibility.disable").catch(() => {});
    }
  };
  const previous = axQueues.get(surface.tabId) ?? Promise.resolve();
  // Swallow a predecessor's REJECTION so the chain cannot poison later calls; a
  // HANG cannot park it either, because every cdp is bounded by settle.ts.
  const chain = previous.catch(() => {}).then(run);
  axQueues.set(surface.tabId, chain);
  void chain
    .finally(() => {
      if (axQueues.get(surface.tabId) === chain) axQueues.delete(surface.tabId);
    })
    .catch(() => {});
  return (await chain) as AXNode[];
}
