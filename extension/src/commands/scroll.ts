import { BridgeCommandError, cdp, requireSurface } from "../cdp";
import {
  defaultScrollExpression,
  deltaScrollExpression,
  SCROLL_INTO_VIEW_FN,
  scrollExpressionFor,
} from "../scroll-expressions";
import { getSession } from "../state";

interface CallResult { result: { value?: unknown } }
interface PushResult { nodeIds: number[] }
interface DocumentResult { root: { nodeId: number } }
interface QueryResult { nodeId?: number }
interface ResolvedNode { object: { objectId: string } }

interface Metrics {
  scrollX: number;
  scrollY: number;
  moreBelow: boolean;
  moreRight: boolean;
}

// Direction keywords the tool accepts. "top"/"bottom" jump to the extremes;
// the rest move one page-sized step in that direction. Kept here (not the page)
// so the set is fixed and no page-provided string is ever executed.
const DIRECTIONS = new Set(["top", "bottom", "up", "down", "left", "right"]);

/**
 * Read the scroll position and whether more content remains past the viewport.
 *
 * This is a FIXED expression evaluated through our own debugger session (not
 * injected page script, not a page-provided string), so it is same-origin,
 * agent-scoped, and safe. `moreBelow`/`moreRight` let the agent know it reached
 * the end instead of paging forever. A 1px slack absorbs sub-pixel/zoom
 * rounding so a fully-scrolled page does not falsely report more content.
 */
async function readMetrics(tabId: number): Promise<Metrics> {
  const out = await cdp<CallResult>(tabId, "Runtime.evaluate", {
    expression: `(() => {
      const de = document.scrollingElement || document.documentElement;
      const x = window.scrollX, y = window.scrollY;
      return {
        scrollX: Math.round(x),
        scrollY: Math.round(y),
        moreBelow: (de.scrollHeight - (y + window.innerHeight)) > 1,
        moreRight: (de.scrollWidth - (x + window.innerWidth)) > 1,
      };
    })()`,
    returnByValue: true,
  });
  const value = (out.result?.value ?? {}) as Partial<Metrics>;
  return {
    scrollX: Number(value.scrollX ?? 0),
    scrollY: Number(value.scrollY ?? 0),
    moreBelow: Boolean(value.moreBelow),
    moreRight: Boolean(value.moreRight),
  };
}

async function nodeObjectId(tabId: number, selector: string, epoch: number): Promise<string> {
  // Resolve a snapshot ref (e5) exactly as click/type do so the two ways of
  // naming an element stay consistent; otherwise treat it as a CSS selector.
  const { refs = {} } = await getSession();
  const ref = /^e\d+$/.test(selector) ? refs[selector] : undefined;
  let nodeId: number;
  if (ref) {
    if (ref.epoch !== epoch) {
      throw new BridgeCommandError("element_not_found", "the page navigated since that snapshot");
    }
    // Same constraint as input.ts nodeIdFor: pushNodesByBackendIdsToFrontend
    // rejects with -32000 "Document needs to be requested first" unless this
    // debugger session has requested the document. See input.ts for the full
    // rationale (found live once #319 made refs reachable at all).
    await cdp<DocumentResult>(tabId, "DOM.getDocument", { depth: 0 });
    const pushed = await cdp<PushResult>(tabId, "DOM.pushNodesByBackendIdsToFrontend", {
      backendNodeIds: [ref.backendNodeId],
    });
    const node = pushed.nodeIds[0];
    if (!node) throw new BridgeCommandError("element_not_found", "snapshot ref is stale");
    nodeId = node;
  } else {
    const document = await cdp<DocumentResult>(tabId, "DOM.getDocument", { depth: 0 });
    const queried = await cdp<QueryResult>(tabId, "DOM.querySelector", {
      nodeId: document.root.nodeId,
      selector,
    });
    if (!queried.nodeId) {
      throw new BridgeCommandError("element_not_found", `selector ${selector} matched nothing`);
    }
    nodeId = queried.nodeId;
  }
  const resolved = await cdp<ResolvedNode>(tabId, "DOM.resolveNode", { nodeId });
  return resolved.object.objectId;
}

/**
 * Scroll the driven tab.
 *
 * Modes, in priority order: `selector` (scroll that element into view) →
 * explicit `x`/`y` pixel deltas → `direction` keyword → default (one viewport
 * down). All movement is driven through our debugger session with fixed CDP
 * calls; nothing the page supplied is evaluated.
 *
 * CONSTRAINT — background tab: the surface is intentionally inactive and current
 * Chrome drops compositor input (Input.dispatchMouseEvent mouseWheel) on a
 * hidden tab, the same limitation click/type hit (finding A9). So we scroll by
 * calling the FIXED `window.scrollBy`/`scrollTo` / `Element.scrollIntoView`
 * DOM methods via Runtime, which run regardless of tab visibility and never
 * activate the tab — preserving the no-focus-steal guarantee.
 *
 * CONSTRAINT — behavior:'instant' everywhere: those DOM methods honour the
 * page's CSS `scroll-behavior: smooth`, which never progresses in a hidden tab
 * because Chrome throttles requestAnimationFrame to zero there. All scroll
 * expressions live in scroll-expressions.ts, which documents and enforces the
 * instant override (and keeps them loadable by the pure tests).
 */
export async function scroll(params: Record<string, unknown>): Promise<Record<string, unknown>> {
  const surface = await requireSurface(params.tab);
  const tabId = surface.tabId;

  const selector = typeof params.selector === "string" ? params.selector.trim() : "";
  const direction = typeof params.direction === "string" ? params.direction.trim().toLowerCase() : "";
  const hasX = typeof params.x === "number" && Number.isFinite(params.x);
  const hasY = typeof params.y === "number" && Number.isFinite(params.y);

  if (selector) {
    // scrollIntoView on the resolved node, centered so the element is usable
    // after the scroll rather than jammed against the viewport edge.
    const objectId = await nodeObjectId(tabId, selector, surface.epoch);
    await cdp(tabId, "Runtime.callFunctionOn", {
      objectId,
      functionDeclaration: SCROLL_INTO_VIEW_FN,
      returnByValue: true,
    });
  } else if (hasX || hasY) {
    const dx = hasX ? Number(params.x) : 0;
    const dy = hasY ? Number(params.y) : 0;
    await cdp(tabId, "Runtime.evaluate", {
      expression: deltaScrollExpression(dx, dy),
      returnByValue: true,
    });
  } else if (direction) {
    if (!DIRECTIONS.has(direction)) {
      throw new BridgeCommandError(
        "internal",
        `unknown scroll direction: ${direction} (top/bottom/up/down/left/right)`,
      );
    }
    // Page-sized deltas computed from the LIVE viewport, with top/bottom jumping
    // to the document extremes. All literals/fixed expression; `direction` only
    // selects a branch, it is never interpolated into code.
    const expr = scrollExpressionFor(direction);
    await cdp(tabId, "Runtime.evaluate", { expression: expr, returnByValue: true });
  } else {
    // Default: one viewport down (minus overlap), the "read more of this page"
    // gesture the agent reaches for most.
    await cdp(tabId, "Runtime.evaluate", {
      expression: defaultScrollExpression(),
      returnByValue: true,
    });
  }

  // Scrolls are instant now, but scroll-linked effects (lazy loading, sticky
  // headers, IntersectionObserver reveals) may still reflow the page right
  // after; a short settle keeps the reported position and moreBelow honest.
  await new Promise((resolve) => setTimeout(resolve, 150));
  const metrics = await readMetrics(tabId);
  const tab = await chrome.tabs.get(tabId);
  return { ...metrics, url: tab.url ?? "", title: tab.title ?? "" };
}
