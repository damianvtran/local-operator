import { BridgeCommandError, cdp, requireSurface } from "../cdp";
import { withOriginGate } from "../origins";
import { getSession } from "../state";

interface BoxModel { model: { content: number[] } }
interface QueryResult { nodeId?: number }
interface PushResult { nodeIds: number[] }
interface DocumentResult { root: { nodeId: number } }

async function nodeIdFor(tabId: number, selector: unknown, epoch: number): Promise<number> {
  if (typeof selector !== "string" || !selector) throw new BridgeCommandError("element_not_found", "selector is required");
  const { refs = {} } = await getSession();
  const ref = /^e\d+$/.test(selector) ? refs[selector] : undefined;
  if (ref) {
    if (ref.epoch !== epoch) throw new BridgeCommandError("element_not_found", "the page navigated since that snapshot");
    const pushed = await cdp<PushResult>(tabId, "DOM.pushNodesByBackendIdsToFrontend", { backendNodeIds: [ref.backendNodeId] });
    const node = pushed.nodeIds[0];
    if (!node) throw new BridgeCommandError("element_not_found", "snapshot ref is stale");
    return node;
  }
  const document = await cdp<DocumentResult>(tabId, "DOM.getDocument", { depth: 0 });
  const queried = await cdp<QueryResult>(tabId, "DOM.querySelector", { nodeId: document.root.nodeId, selector });
  if (!queried.nodeId) throw new BridgeCommandError("element_not_found", `selector ${selector} matched nothing`);
  return queried.nodeId;
}

async function centre(tabId: number, selector: unknown, epoch: number): Promise<{ nodeId: number; x: number; y: number }> {
  const nodeId = await nodeIdFor(tabId, selector, epoch);
  await cdp(tabId, "DOM.scrollIntoViewIfNeeded", { nodeId });
  const { model } = await cdp<BoxModel>(tabId, "DOM.getBoxModel", { nodeId });
  const xs = [model.content[0] ?? 0, model.content[2] ?? 0, model.content[4] ?? 0, model.content[6] ?? 0];
  const ys = [model.content[1] ?? 0, model.content[3] ?? 0, model.content[5] ?? 0, model.content[7] ?? 0];
  return { nodeId, x: xs.reduce((a, b) => a + b, 0) / 4, y: ys.reduce((a, b) => a + b, 0) / 4 };
}

export async function click(
  params: Record<string, unknown>,
  requestId: string,
): Promise<Record<string, unknown>> {
  const surface = await requireSurface(params.tab);
  const target = await centre(surface.tabId, params.selector ?? params.ref, surface.epoch);
  await withOriginGate(surface.tabId, requestId, async () => {
    await cdp(surface.tabId, "Input.dispatchMouseEvent", { type: "mousePressed", x: target.x, y: target.y, button: "left", clickCount: 1 });
    await cdp(surface.tabId, "Input.dispatchMouseEvent", { type: "mouseReleased", x: target.x, y: target.y, button: "left", clickCount: 1 });
    await new Promise((resolve) => setTimeout(resolve, 1500));
  });
  const tab = await chrome.tabs.get(surface.tabId);
  return { navigated: false, url: tab.url ?? "", title: tab.title ?? "" };
}

export async function typeText(params: Record<string, unknown>): Promise<Record<string, unknown>> {
  const surface = await requireSurface(params.tab);
  const selector = params.selector ?? params.ref;
  const target = await centre(surface.tabId, selector, surface.epoch);
  await cdp(surface.tabId, "DOM.focus", { nodeId: target.nodeId });
  await cdp(surface.tabId, "Input.dispatchKeyEvent", { type: "keyDown", key: "a", code: "KeyA", modifiers: 2 });
  await cdp(surface.tabId, "Input.dispatchKeyEvent", { type: "keyUp", key: "a", code: "KeyA", modifiers: 2 });
  await cdp(surface.tabId, "Input.insertText", { text: String(params.text ?? "") });
  const value = await chrome.scripting.executeScript({
    target: { tabId: surface.tabId },
    func: (query: string) => {
      const element = document.querySelector(query) as HTMLInputElement | HTMLTextAreaElement | null;
      return element?.value ?? element?.textContent ?? "";
    },
    args: [String(selector)],
  });
  const tab = await chrome.tabs.get(surface.tabId);
  return { value: value[0]?.result ?? "", url: tab.url ?? "", title: tab.title ?? "" };
}
