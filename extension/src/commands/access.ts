import { requesterOriginKey } from "../access-queue";
import { accessStatus, cancelAccess, enqueueAccess } from "../approval-store";
import { BridgeCommandError } from "../cdp";
import { originAllowed, safeHttpUrl } from "../origins";
import { getSession } from "../state";

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
  const result = await enqueueAccess(url, requester, "async");
  if (result.full) {
    throw new BridgeCommandError(
      "access_queue_full",
      `site approval queue is full with ${result.pending_count} pending requests`,
      { pending_count: result.pending_count },
    );
  }
  chrome.alarms.create("lop-access-expiry", { when: result.expires_at });
  const { entry: _entry, full: _full, ...response } = result;
  return { ...response };
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
