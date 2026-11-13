import { requesterOriginKey } from "../access-queue";
import { accessStatus, cancelAccess, decideAccess, enqueueAccess, readQueueSnapshot } from "../approval-store";
import { BridgeCommandError } from "../cdp";
import { originAllowed, safeHttpUrl } from "../origins";
import { getSession } from "../state";
import { SITE_SCOPE, readNodeApproval, submitSignedDecisionWith } from "../signed-decision";

/* The managed-policy area, or undefined where Chrome offers none (tests, old
 * builds). Read per call, never cached: setup rewrites the anchor in a running
 * Chrome, and a revoked anchor must stop verifying at once. */
function managedArea(): Pick<chrome.storage.StorageArea, "get"> | undefined {
  return typeof chrome !== "undefined" ? chrome.storage?.managed : undefined;
}

/* Approval RPCs are surface-free: they are the recovery path after open/goto
 * refused a new origin. Each poll reads only the caller's requester-bound entry
 * or receipt, so a busy queue never turns into a global lock. */
export const AWAIT_SLICE_MS = 20_000;
const POLL_MS = 300;

function requesterOf(params: Record<string, unknown>, requestId: string): string {
  const supplied = typeof params.requester === "string" ? params.requester.trim() : "";
  return supplied || requestId;
}

async function callerAlreadyAllowed(url: URL, requester: string): Promise<boolean> {
  if (await originAllowed(url)) return true;
  const { onceGrants = {} } = await getSession();
  const grant = onceGrants[requesterOriginKey(url.origin, requester)];
  return !!grant && Date.now() < grant.expiresAt;
}

export async function requestAccess(
  params: Record<string, unknown>,
  requestId: string,
): Promise<Record<string, unknown>> {
  const url = safeHttpUrl(params.url);
  const requester = requesterOf(params, requestId);
  if (await callerAlreadyAllowed(url, requester)) return { origin: url.origin, state: "allowed" };
  // HEADLESS NODE (signed-decision.ts): nobody can click the popup, so the
  // decision has to come from the operator's signature. With no usable anchor
  // there is nobody who can decide at all — say so now (`approval:
  // "no_approver"`, an ok result, never a new error code) instead of queueing a
  // prompt that expires unseen ten minutes later.
  const node = await readNodeApproval(managedArea());
  if (node.headless && !node.anchor) {
    return { origin: url.origin, state: "none", approval: "no_approver" };
  }
  const result = await enqueueAccess(url, requester, "async");
  if (result.full) {
    throw new BridgeCommandError(
      "access_queue_full",
      `site approval queue is full with ${result.pending_count} pending requests`,
      { pending_count: result.pending_count },
    );
  }
  // enqueueAccess centrally arms the earliest durable deadline. Arming this
  // request's own expiry here would let a later request delay an earlier one.
  const { entry: _entry, full: _full, ...response } = result;
  if (node.anchor && response.state === "pending" && response.entryId) {
    // Everything the session needs to raise the operator's card and bind it to
    // THIS entry: the question id is derived from these, and the extension
    // re-derives it from its own entry when the signed answer comes back.
    return {
      ...response,
      approval: "signed",
      scope: SITE_SCOPE,
      node_device_id: node.nodeDeviceId,
    };
  }
  return { ...response };
}

/** `submit_signed_decision`: the operator's signed answer to a node site ask.
 * Requester-free on purpose — the signature, not the caller, is the authority,
 * and it names its entry through the question id. */
export async function submitSignedDecision(params: Record<string, unknown>): Promise<Record<string, unknown>> {
  const approval = await readNodeApproval(managedArea());
  const result = await submitSignedDecisionWith(params, {
    approval,
    liveEntries: async () => (await readQueueSnapshot()).queue,
    decide: async (entryId, decision) => (await decideAccess(entryId, decision)).applied,
  });
  return { ...result };
}

export async function awaitAccess(
  params: Record<string, unknown>,
  requestId: string,
): Promise<Record<string, unknown>> {
  const url = safeHttpUrl(params.url);
  const requester = requesterOf(params, requestId);
  const requested = Number(params.timeout_ms);
  const sliceMs = Math.min(
    Number.isFinite(requested) && requested > 0 ? requested : AWAIT_SLICE_MS,
    AWAIT_SLICE_MS,
  );
  const deadline = Date.now() + sliceMs;
  for (;;) {
    if (await callerAlreadyAllowed(url, requester)) return { origin: url.origin, state: "allowed" };
    const current = await accessStatus(url.origin, requester);
    if (current.state !== "pending" || Date.now() >= deadline) return { ...current };
    await new Promise((resolve) => setTimeout(resolve, POLL_MS));
  }
}

export async function cancelAccessCommand(
  params: Record<string, unknown>,
  requestId: string,
): Promise<Record<string, unknown>> {
  const url = safeHttpUrl(params.url);
  return { ...(await cancelAccess(url.origin, requesterOf(params, requestId))) };
}
