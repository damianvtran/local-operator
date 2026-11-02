import { BridgeCommandError, cdp, requireSurface } from "../cdp";
import { withOriginGate } from "../origins";
import { resolveTarget } from "./input";
import { selectNative } from "./select-native";

/** Select without opening the OS popup or requiring hidden-tab Input.* delivery.
 * Reuse the same AX/ref/CSS resolver and origin gate as click, rather than
 * inventing a second selector language or letting change handlers bypass consent.
 */
export async function selectOptions(params: Record<string, unknown>, requestId: string): Promise<Record<string, unknown>> {
  const requested = params.options;
  if (!Array.isArray(requested) || requested.some((v) => typeof v !== "string" && !(typeof v === "number" && Number.isInteger(v) && v >= 0))) {
    throw new BridgeCommandError("internal", "options must be labels/values or non-negative integer indexes", { reason: "invalid_arguments" });
  }
  const surface = await requireSurface(params.tab);
  const resolved = await resolveTarget(surface, params.selector);
  const node = await cdp<{ object: { objectId: string } }>(surface.tabId, "DOM.resolveNode", { nodeId: resolved.nodeId });
  const result = await withOriginGate(surface.tabId, requestId, async () => {
    const response = await cdp<{ result?: { value?: Record<string, unknown> }; exceptionDetails?: unknown }>(surface.tabId, "Runtime.callFunctionOn", {
      objectId: node.object.objectId,
      functionDeclaration: selectNative.toString(),
      arguments: [{ value: requested }],
      returnByValue: true,
    });
    if (response.exceptionDetails || !response.result?.value) {
      throw new BridgeCommandError("internal", "select did not return a readable result; check the page before retrying", { reason: "selection_unverified" });
    }
    const payload = response.result.value;
    if (payload.error) {
      const error = payload.error as { code: string; message: string; data: Record<string, unknown> };
      throw new BridgeCommandError(error.code, error.message, error.data);
    }
    return payload;
  });
  const tab = await chrome.tabs.get(surface.tabId);
  return { ...result, resolved, url: tab.url ?? "", title: tab.title ?? "" };
}
