import {
  accessState,
  activeRequest,
  consumableGrant,
  newRequest,
  requestVerdict,
  type AccessRequest,
} from "../access-flow";
import {
  expireAccessRequest,
  originAllowed,
  raiseAccessRequest,
  safeHttpUrl,
} from "../origins";
import { getSession } from "../state";

/* Async site-access commands — the agent-legible half of the approval flow.
 *
 * request_access raises the approval prompt and returns IMMEDIATELY;
 * await_access waits for the decision in bounded slices. The wait is
 * implemented as a SHORT in-worker poll with the session-side tool looping
 * slices, rather than a daemon-side long-poll with an extended timeout entry,
 * because (a) every entry in the daemon's COMMAND_TIMEOUTS stays an honest
 * per-RPC bound instead of one method needing a special multi-minute carve-out
 * — the exact mechanism (awaiting_origin deadline extension) that made the old
 * flow's failures unreadable; and (b) the MV3 worker never has to keep a
 * multi-minute promise alive across a service-worker death — each slice is
 * short, and the decision itself lives in session storage, which survives.
 *
 * Every verdict is computed against a REQUESTER identity: params.requester
 * when the calling tool supplies one (the session-scoped identity — see
 * access-flow.ts), else the daemon's per-command request id as the fallback
 * for raw-RPC callers. */

/** One await_access slice. Below the daemon's 25 s command timeout with margin
 * so the RPC always answers before the daemon gives up on it. */
export const AWAIT_SLICE_MS = 20_000;

/** How often a slice re-reads the decision record. Storage reads are cheap
 * (session storage is in-memory) and the human is the latency floor anyway. */
const POLL_MS = 300;

/** The requester identity for this command: the tool passes a session-scoped
 * value (session:<id>) so all of one session's commands share it — a
 * per-command request id could never match between request_access and the
 * later open/goto, which is what would make grants unspendable. Raw-RPC
 * callers (no session concept) fall back to the per-command id and get
 * per-command binding, the fail-closed default. */
function requesterOf(params: Record<string, unknown>, requestId: string): string {
  const supplied = typeof params.requester === "string" ? params.requester.trim() : "";
  return supplied || requestId;
}

/** Re-read the pieces an access verdict needs in ONE getSession round-trip,
 * sweeping TTL only when it can actually fire. expireAccessRequest does two
 * storage reads and writes on every call; polling at 300 ms makes that the
 * dominant cost of a 20 s slice, so the poll path skips it unless the record
 * is past its TTL right now (round-1 m3). Same correctness: expiry is
 * computed on read, and the sweep's side effects (prompt teardown) only
 * matter when something IS expired. */
async function pollSnapshot(requestId: string, origin: string): Promise<{ origin: string; state: string }> {
  const now = Date.now();
  let session = await getSession();
  const record = session.accessRequest;
  if (record && now >= record.expiresAt) {
    await expireAccessRequest();
    session = await getSession();
  }
  const url = new URL(origin + "/");
  const state = accessState(
    session.accessRequest,
    session.accessTombstones,
    await originAllowed(url),
    !!consumableGrant(session.onceGrants, origin, requestId, now),
    origin,
    requestId,
    now,
  );
  return { origin, state };
}

export async function requestAccess(
  params: Record<string, unknown>,
  requestId: string,
): Promise<Record<string, unknown>> {
  const url = safeHttpUrl(params.url);
  const requester = requesterOf(params, requestId);
  await expireAccessRequest();
  const now = Date.now();
  const session = await getSession();
  const verdict = requestVerdict(
    session.accessRequest,
    await originAllowed(url),
    !!consumableGrant(session.onceGrants, url.origin, requester, now),
    url.origin,
    requester,
    now,
  );
  if (verdict !== "raise") return { origin: url.origin, state: verdict };
  const record = await raiseAccessRequest(url, requester);
  // Arm the TTL sweep for THIS record: chrome.alarms survives MV3 worker
  // death, a setTimeout does not. Creating a same-named alarm replaces the
  // prior one, which is intended — the record this alarm covers is now the
  // only live prompt (see raiseAccessRequest).
  chrome.alarms.create("lop-access-expiry", { when: record.expiresAt });
  return { origin: url.origin, state: "pending" };
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
    const current = await pollSnapshot(requester, url.origin);
    if (current.state !== "pending" || Date.now() >= deadline) return current;
    await new Promise((resolve) => setTimeout(resolve, POLL_MS));
  }
}
