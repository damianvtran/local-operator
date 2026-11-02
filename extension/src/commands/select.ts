import { BridgeCommandError, cdp, requireSurface } from "../cdp";
import { withOriginGate } from "../origins";
import { resolveTarget } from "./input";
import { readNativeSelection, selectNative } from "./select-native";

/** Select without opening the OS popup or requiring hidden-tab Input.* delivery.
 * Reuse the same AX/ref/CSS resolver and origin gate as click, rather than
 * inventing a second selector language or letting change handlers bypass consent.
 */
export async function selectOptions(params: Record<string, unknown>, requestId: string): Promise<Record<string, unknown>> {
  const requested = params.options;
  if (!Array.isArray(requested) || requested.length === 0 || requested.some((v) => typeof v !== "string" && !(typeof v === "number" && Number.isInteger(v) && v >= 0))) {
    throw new BridgeCommandError("internal", "options must be a non-empty list of labels/values or non-negative integer indexes", { reason: "invalid_arguments" });
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
    // Framework reconciliation can replace the target, synchronously or on a
    // queued render. Give queued work a bounded 150ms grace (no rAF: hidden tabs
    // may throttle it indefinitely), then resolve the caller's selector AGAIN.
    // The gate stays armed across the grace/read so change-triggered navigation
    // cannot bypass the origin policy. Verification is the live sampled state,
    // not a promise that a page will never change its selection later.
    await new Promise((resolve) => setTimeout(resolve, 150));
    try {
      const live = await resolveTarget(surface, params.selector);
      const liveNode = await cdp<{ object: { objectId: string } }>(surface.tabId, "DOM.resolveNode", { nodeId: live.nodeId });
      const read = await cdp<{ result?: { value?: Record<string, unknown> }; exceptionDetails?: unknown }>(surface.tabId, "Runtime.callFunctionOn", {
        objectId: liveNode.object.objectId,
        functionDeclaration: readNativeSelection.toString(),
        arguments: [{ value: payload.expected }],
        returnByValue: true,
      });
      if (!read.exceptionDetails && read.result?.value) return read.result.value;
    } catch {
      // Selection was already attempted: a disappeared/ref-detached control or
      // failed read must not prompt an implicit retry or report detached state.
    }
    return { kind: "native", selected: [], verified: false };
  });
  const tab = await chrome.tabs.get(surface.tabId);
  return { ...result, resolved, url: tab.url ?? "", title: tab.title ?? "" };
}
