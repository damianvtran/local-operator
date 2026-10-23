import { locatorSuggestions, parseLocator, resolveLocator } from "../driver/ax-locator";
import { BridgeCommandError, cdp, requireSurface } from "../cdp";
import { withOriginGate } from "../origins";
import { CHROME_API_DEADLINE_MS, deadline, settle } from "../settle";
import { getRefs, surfaceToken, type StoredSurface } from "../state";
import { readAXTree } from "./ax-tree";

interface QueryResult { nodeId?: number }
interface PushResult { nodeIds: number[] }
interface DocumentResult { root: { nodeId: number } }
interface ResolvedNode { object: { objectId: string } }
interface CallResult { result: { value?: unknown } }
interface DescribedNode { node: { backendNodeId?: number } }

async function objectIdFor(tabId: number, nodeId: number): Promise<string> {
  const resolved = await cdp<ResolvedNode>(tabId, "DOM.resolveNode", { nodeId });
  return resolved.object.objectId;
}

async function callOnNode<T>(
  tabId: number,
  objectId: string,
  functionDeclaration: string,
  args: unknown[] = [],
): Promise<T> {
  const out = await cdp<CallResult>(tabId, "Runtime.callFunctionOn", {
    objectId,
    functionDeclaration,
    arguments: args.map((value) => ({ value })),
    returnByValue: true,
  });
  return out.result?.value as T;
}

/* Exported because `upload` resolves its file input through it: a second copy of
 * the ref/selector resolution would be a second place for the
 * `pushNodesByBackendIdsToFrontend` document-request fixup to be forgotten. */
/** What resolved a click/type target, and which node it resolved to. */
export interface ResolvedTarget {
  nodeId: number;
  by: "ref" | "css" | "locator";
  role?: string;
  name?: string;
  /** How many AX nodes matched a locator; the first is the one returned. */
  matched?: number;
}

/* An accessibility locator resolved against the page's real a11y tree.
 *
 * The tree is read through `readAXTree`, i.e. under the SAME per-tab queue a
 * snapshot uses: the resolution is a second reader of one debuggee's a11y
 * engine, and interleaving it with a snapshot's enable/disable window is how a
 * page's tree comes back degraded for the other caller.
 *
 * A miss throws with the locator, the count, and up to five CANDIDATES (slice
 * 4's diagnostics live in `locatorSuggestions`): in the measured incident each
 * wrong guess cost a whole tool call and taught the agent nothing, so nine
 * guesses happened. Naming the nearest real elements turns the same failure
 * into one correction.
 */
async function resolveLocatorTarget(
  surface: StoredSurface,
  locator: NonNullable<ReturnType<typeof parseLocator>>,
  source: string,
): Promise<ResolvedTarget> {
  const nodes = await readAXTree(surface);
  const hit = resolveLocator(nodes, locator);
  if (!hit) {
    throw new BridgeCommandError("element_not_found", `no element matches ${source}`, {
      candidates: locatorSuggestions(nodes, source),
      matched: 0,
    });
  }
  await cdp<DocumentResult>(surface.tabId, "DOM.getDocument", { depth: 0 });
  const pushed = await cdp<PushResult>(surface.tabId, "DOM.pushNodesByBackendIdsToFrontend", {
    backendNodeIds: [hit.backendNodeId],
  });
  const nodeId = pushed.nodeIds[0];
  if (!nodeId) {
    throw new BridgeCommandError("element_not_found", `${source} is no longer in the document`);
  }
  return { nodeId, by: "locator", role: hit.role, name: hit.name, matched: hit.matched };
}

export async function resolveTarget(surface: StoredSurface, selector: unknown): Promise<ResolvedTarget> {
  const tabId = surface.tabId;
  if (typeof selector !== "string" || !selector) throw new BridgeCommandError("element_not_found", "selector is required");
  // A locator is tried BEFORE CSS: `role=…`/`label=…`/`text=…` are not valid
  // CSS, so parsing first cannot shadow a selector, and it keeps the
  // framework-generated-id case out of DOM.querySelector entirely.
  const locator = parseLocator(selector);
  if (locator) return resolveLocatorTarget(surface, locator, selector);
  // Refs are read from THIS surface's own map: a snapshot taken on another
  // concurrently-driven tab must never supply the click target here.
  const refs = await getRefs(surfaceToken(surface));
  const ref = /^e\d+$/.test(selector) ? refs[selector] : undefined;
  if (ref) {
    if (ref.epoch !== surface.epoch) throw new BridgeCommandError("element_not_found", "the page navigated since that snapshot");
    // DOM.pushNodesByBackendIdsToFrontend requires the DOM agent to hold the
    // document for this session first, or Chrome rejects with -32000
    // "Document needs to be requested first". The CSS-selector branch below
    // gets this for free from its own getDocument call; the ref branch must
    // request it explicitly. Found live: every ref-based click failed while
    // selector-based clicks worked (the pre-#319 snapshot bug meant refs were
    // never exercised before). depth:0 keeps the payload minimal.
    await cdp<DocumentResult>(tabId, "DOM.getDocument", { depth: 0 });
    const pushed = await cdp<PushResult>(tabId, "DOM.pushNodesByBackendIdsToFrontend", { backendNodeIds: [ref.backendNodeId] });
    const node = pushed.nodeIds[0];
    if (!node) throw new BridgeCommandError("element_not_found", "snapshot ref is stale");
    return { nodeId: node, by: "ref" };
  }
  const document = await cdp<DocumentResult>(tabId, "DOM.getDocument", { depth: 0 });
  const queried = await cdp<QueryResult>(tabId, "DOM.querySelector", { nodeId: document.root.nodeId, selector });
  if (!queried.nodeId) throw new BridgeCommandError("element_not_found", `selector ${selector} matched nothing`);
  return { nodeId: queried.nodeId, by: "css" };
}

/* `nodeIdFor` is the narrow contract `upload` and the selector hops use; every
 * caller that wants to REPORT what it resolved (click/type) uses
 * `resolveTarget` directly. */
export async function nodeIdFor(surface: StoredSurface, selector: unknown): Promise<number> {
  return (await resolveTarget(surface, selector)).nodeId;
}

/* A CSS selector's element, named by the id the ACCESSIBILITY tree uses.
 *
 * `snapshot` scoping needs this rather than `nodeIdFor`: the AX tree indexes
 * nodes by `backendDOMNodeId`, while every other DOM command speaks frontend
 * node ids that belong to one session's document and change between them.
 * The describe hop is what translates the queried node into the stable id, and
 * `depth: 0` keeps it to the element itself. Exported from here (rather than a
 * second file) so the document-request dance and the typed miss stay in ONE
 * place; a snapshot that cannot find its scope must fail exactly like a click
 * that cannot find its target.
 */
export async function backendNodeIdForSelector(surface: StoredSurface, selector: string): Promise<number> {
  const tabId = surface.tabId;
  if (!selector) throw new BridgeCommandError("element_not_found", "selector is required");
  const document = await cdp<DocumentResult>(tabId, "DOM.getDocument", { depth: 0 });
  const queried = await cdp<QueryResult>(tabId, "DOM.querySelector", { nodeId: document.root.nodeId, selector });
  if (!queried.nodeId) throw new BridgeCommandError("element_not_found", `selector ${selector} matched nothing`);
  const described = await cdp<DescribedNode>(tabId, "DOM.describeNode", { nodeId: queried.nodeId, depth: 0 });
  const backendNodeId = described.node?.backendNodeId;
  if (backendNodeId === undefined) {
    throw new BridgeCommandError("element_not_found", `selector ${selector} has no backend node id`);
  }
  return backendNodeId;
}

// The click/type paths drive the resolved node directly rather than computing
// a viewport point for Input.dispatchMouseEvent: the agent tab is inactive, and
// current Chrome drops synthetic compositor input on a hidden tab, so a
// coordinate-based click never fired (finding A9). Node-level dispatch through
// our own debugger session is what actually works there.

export async function click(
  params: Record<string, unknown>,
  requestId: string,
): Promise<Record<string, unknown>> {
  const surface = await requireSurface(params.tab);
  const before = (
    await deadline(
      chrome.tabs.get(surface.tabId),
      CHROME_API_DEADLINE_MS,
      `chrome.tabs.get(${surface.tabId})`,
    )
  ).url ?? "";
  // `resolved` is what lets the tool say WHICH element it clicked — a locator
  // answer that reads back as the locator is verifiable by the caller, while
  // "clicked e57" sends it back to the snapshot to find out what e57 was.
  const target = await resolveTarget(surface, params.selector ?? params.ref);
  const nodeId = target.nodeId;
  await cdp(surface.tabId, "DOM.scrollIntoViewIfNeeded", { nodeId });
  const objectId = await objectIdFor(surface.tabId, nodeId);
  // Race a short grace window for a navigation the click may start. A click is
  // page-initiated and asynchronous, so hardcoding navigated:false mislabelled
  // real navigations even though the returned url updated (finding A7).
  let navigationSeen = false;
  const onBefore = (details: chrome.webNavigation.WebNavigationParentedCallbackDetails) => {
    if (details.tabId === surface.tabId && details.frameId === 0) navigationSeen = true;
  };
  const onHistory = (details: chrome.webNavigation.WebNavigationTransitionCallbackDetails) => {
    if (details.tabId === surface.tabId && details.frameId === 0) navigationSeen = true;
  };
  chrome.webNavigation.onBeforeNavigate.addListener(onBefore);
  chrome.webNavigation.onHistoryStateUpdated.addListener(onHistory);
  try {
    await withOriginGate(surface.tabId, requestId, async () => {
      // The agent tab is intentionally INACTIVE (background), and current
      // Chrome drops CDP Input.dispatchMouseEvent on a hidden tab entirely:
      // real-Chrome testing showed the compositor never delivered the press
      // and no handler fired (finding A9/A7). So the click is driven on the
      // resolved node through the debugger's own Runtime — a full mouse event
      // sequence dispatched at the element, which reliably fires handlers and
      // default actions (link navigation, form submit) on a background tab.
      // This runs in the page via OUR debugger session, not injected page
      // script, keeping it same-origin and agent-scoped.
      await callOnNode<void>(
        surface.tabId,
        objectId,
        `function(){
          const r=this.getBoundingClientRect();
          const cx=r.left+r.width/2, cy=r.top+r.height/2;
          const opts={bubbles:true,cancelable:true,view:window,clientX:cx,clientY:cy,button:0};
          for(const type of ['pointerover','pointerenter','pointerdown','mousedown','pointerup','mouseup','click']){
            const Ctor=type.startsWith('pointer')?PointerEvent:MouseEvent;
            this.dispatchEvent(new Ctor(type,opts));
          }
          if(typeof this.focus==='function') this.focus();
        }`,
      );
      await new Promise((resolve) => setTimeout(resolve, 1200));
    });
    if (navigationSeen) {
      // Let the started navigation settle so the returned url/title describe
      // the page that actually arrived, not the one being left.
      await settle(surface.tabId, 10_000).catch(() => undefined);
    }
  } finally {
    chrome.webNavigation.onBeforeNavigate.removeListener(onBefore);
    chrome.webNavigation.onHistoryStateUpdated.removeListener(onHistory);
  }
  const tab = await deadline(
    chrome.tabs.get(surface.tabId),
    CHROME_API_DEADLINE_MS,
    `chrome.tabs.get(${surface.tabId})`,
  );
  const navigated = navigationSeen || (tab.url ?? "") !== before;
  return { navigated, url: tab.url ?? "", title: tab.title ?? "", resolved: target };
}

export async function typeText(params: Record<string, unknown>): Promise<Record<string, unknown>> {
  const surface = await requireSurface(params.tab);
  const selector = params.selector ?? params.ref;
  const target = await resolveTarget(surface, selector);
  const nodeId = target.nodeId;
  await cdp(surface.tabId, "DOM.scrollIntoViewIfNeeded", { nodeId });
  const objectId = await objectIdFor(surface.tabId, nodeId);
  // Replace-not-append (the cmux fill-vs-type lesson): focus, set the value on
  // the node, and dispatch the input/change events a framework-controlled
  // field listens for. Driving the node directly is what makes fills reliable
  // on the intentionally-inactive tab, where key-event dispatch is unreliable
  // (same background-tab constraint as click; finding A9). The read-back comes
  // from the SAME node so a snapshot ref ("e5", not a CSS selector) verifies
  // correctly rather than reporting a false "fill did not take" (finding A2).
  const value = await callOnNode<string>(
    surface.tabId,
    objectId,
    `function(text){
      if(typeof this.focus==='function') this.focus();
      if('value' in this){
        const setter=Object.getOwnPropertyDescriptor(Object.getPrototypeOf(this),'value');
        if(setter&&setter.set){setter.set.call(this,text);} else {this.value=text;}
        this.dispatchEvent(new Event('input',{bubbles:true}));
        this.dispatchEvent(new Event('change',{bubbles:true}));
        return this.value;
      }
      if(this.isContentEditable){
        this.textContent=text;
        this.dispatchEvent(new Event('input',{bubbles:true}));
        return this.textContent;
      }
      return this.textContent||'';
    }`,
    [String(params.text ?? "")],
  );
  const tab = await deadline(
    chrome.tabs.get(surface.tabId),
    CHROME_API_DEADLINE_MS,
    `chrome.tabs.get(${surface.tabId})`,
  );
  return { value: value ?? "", url: tab.url ?? "", title: tab.title ?? "", resolved: target };
}
