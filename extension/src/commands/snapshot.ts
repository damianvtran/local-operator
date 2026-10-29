import { compactAX, windowSnapshot } from "../driver/ax-compact";
import { BridgeCommandError, requireSurface } from "../cdp";
import { CHROME_API_DEADLINE_MS, deadline } from "../settle";
import { setRefs, surfaceToken } from "../state";
import { readAXTree } from "./ax-tree";
import { backendNodeIdForSelector } from "./input";

/* The a11y read (enable→read→disable under a per-tab queue) lives in
 * `ax-tree.ts`, shared with the locator resolver: two readers of one engine
 * must queue against each other, and a private copy per command is how that
 * stops being true. */
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
  const nodes = await readAXTree(surface);
  const rendered = compactAX(nodes, surface.epoch, { rootBackendNodeId });
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
