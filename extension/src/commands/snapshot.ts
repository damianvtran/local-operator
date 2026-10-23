import { compactAX, windowSnapshot, type AXNode } from "../driver/ax-compact";
import { BridgeCommandError, cdp, requireSurface } from "../cdp";
import { CHROME_API_DEADLINE_MS, deadline } from "../settle";
import { setRefs, surfaceToken } from "../state";
import { backendNodeIdForSelector } from "./input";

// Serializes the enable→read→disable window below, PER TARGET TAB. The worker
// dispatches daemon frames fire-and-forget (worker.ts onmessage), so two
// concurrent snapshots of the SAME tab could otherwise interleave as
// enable/enable/read/DISABLE/read — the second read then hits a disabled a11y
// engine and risks a degraded tree.
//
// Per TAB, not module-global (audit A3). The hazard above is one a11y engine,
// and the CDP `Accessibility` domain is per debuggee — two tabs have two
// engines, so serializing across them protects nothing while costing a
// DIFFERENT owner its budget: with a global lane, owner A's `getFullAXTree`
// stalling its full deadline plus its `finally` disable stalling another left
// owner B's snapshot unable to issue even `Accessibility.enable` until both
// drained, against B's own 20 s daemon budget. A per-tab map cannot express that
// wait, and no smaller global ceiling fixes it either: a global lane is only
// safe if three ceilings fit inside the tightest budget (~5 s), and a healthy
// heavy `screenshot` was measured at 10.91 s (settle.ts), so the two
// constraints are incompatible.
//
// The eviction idiom (identity check, then delete, on drain) is copied from the
// per-owner lane in ownership.ts — a second eviction idiom beside that one
// would be the anti-pattern, and that one is already proven against the "the
// head may have been replaced while we awaited" race. An idle tab holds no
// entry, so a closed tab leaves nothing behind and no listener is needed;
// growth is bounded by that eviction plus MAX_SURFACES (state.ts), since only a
// driven surface reaches here (requireSurface).
const axQueues = new Map<number, Promise<unknown>>();

/** `start`/`limit` are the window; the daemon validates them too, and this is
 *  the second guard rather than a duplicate: a wire client that predates the
 *  tool's own checks (or a hand-written frame) must not be able to ask for
 *  line -4 of a tree. `start` is clamped to 1, `limit` to 0 (everything). */
function positiveInt(value: unknown, fallback: number): number {
  const parsed = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  const whole = Math.floor(parsed);
  return whole > 0 ? whole : fallback;
}

export async function snapshot(params: Record<string, unknown>): Promise<Record<string, unknown>> {
  const surface = await requireSurface(params.tab);
  const scopeSelector = typeof params.selector === "string" ? params.selector.trim() : "";
  // Resolve BEFORE reading the tree: a selector that matches nothing must fail
  // as a typed element_not_found naming the selector, not return an empty
  // snapshot that reads like "this section has no controls".
  const rootBackendNodeId = scopeSelector
    ? await backendNodeIdForSelector(surface, scopeSelector)
    : undefined;
  // Enable the a11y domain for the read window as documented CDP hygiene.
  // NOTE: the live one-line snapshots previously blamed on a missing enable
  // were actually compactAX pruning the subtrees of ignored wrapper nodes
  // (see ax-compact.ts) — verified in headful Chrome 151/145 where bare
  // getFullAXTree on a hidden tab returns a full tree even without enable.
  // We keep the enable→read→disable window anyway: it is cheap, matches the
  // documented contract, and disable ensures the a11y engine is not kept hot
  // on the tab between snapshots — the stored refs are backendDOMNodeIds,
  // which belong to the DOM (not the AX tree) and remain resolvable after.
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
  const previous = axQueues.get(surface.tabId) ?? Promise.resolve();
  // `.catch(() => {})` swallows a predecessor's REJECTION so the chain cannot
  // poison later calls; a HANG cannot park it either, because the cdps inside
  // `run` are each bounded by settle.ts's deadline, so every link settles.
  const chain = previous.catch(() => {}).then(run);
  axQueues.set(surface.tabId, chain);
  void chain
    .finally(() => {
      if (axQueues.get(surface.tabId) === chain) axQueues.delete(surface.tabId);
    })
    .catch(() => {});
  const result = (await chain) as { nodes: AXNode[] };
  const rendered = compactAX(result.nodes, surface.epoch, { rootBackendNodeId });
  // The DOM element exists but contributes no accessible node (display:none,
  // aria-hidden, inside a closed <details>). Say so in those terms: the old
  // "selector matched nothing" wording would be a lie here — it matched, it is
  // just not in the accessibility tree.
  if (rootBackendNodeId !== undefined && !rendered.scoped) {
    throw new BridgeCommandError(
      "element_not_found",
      `selector ${scopeSelector} matched an element with no accessibility subtree ` +
        "(hidden, aria-hidden, or inside a collapsed section); expand it first, then snapshot again",
    );
  }
  // Refs are stored under THIS surface's token: with several tabs driven in
  // parallel, a global ref map would let one tab's snapshot silently repoint
  // another tab's click targets.
  await setRefs(surfaceToken(surface), rendered.refs);
  const page = windowSnapshot(rendered.snapshot, {
    search: typeof params.search === "string" ? params.search : "",
    start: positiveInt(params.start, 1),
    limit: Number.isFinite(Number(params.limit)) && Number(params.limit) > 0 ? Math.floor(Number(params.limit)) : 0,
  });
  const tab = await deadline(
    chrome.tabs.get(surface.tabId),
    CHROME_API_DEADLINE_MS,
    `chrome.tabs.get(${surface.tabId})`,
  );
  return {
    snapshot: page.text,
    url: tab.url ?? "",
    title: tab.title ?? "",
    // The window's numbers, additive to the old shape: an agent that asked for
    // a page needs to know whether there is a next one, and a caller that asked
    // for a scope needs to tell a scoping host from one that ignored the param.
    scoped: rendered.scoped,
    total_lines: page.totalLines,
    matched_lines: page.matchedLines,
    returned_lines: page.returnedLines,
    start: page.start,
    limit: page.limit,
    search: page.search,
  };
}
