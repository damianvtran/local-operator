import { BridgeCommandError } from "./cdp";
import { getSurfaces, withSessionMutation } from "./state";

/** A private proof is intentionally separate from ownerKey (tab-group copy).
 * Records live with surfaces in storage.session: worker churn preserves them;
 * a full browser restart clears both, so a reused numeric tab id is never proof.
 */
export interface Scope {
  session: string;
  generation: string;
  terminal?: string;
  retention?: string;
  allocations: Record<string, { tab?: string | undefined; state: string }>;
  /** Tabs Chrome may have created for this owner whose handle never reached
   * the journal (worker death between create and persist). Reported by
   * diagnostics and counted against the pool; NEVER closed by inference. */
  unknownReservations?: number;
}
type Params = Record<string, unknown>;
const queues = new Map<string, Promise<unknown>>();
let allocations: Promise<unknown> = Promise.resolve();

async function scopes(): Promise<Record<string, Scope>> {
  return (await chrome.storage.session.get(["ownerScopes"])).ownerScopes ?? {};
}
function identity(params: Params): [string, string, string] {
  const proof = String(params.owner_proof ?? "");
  const session = String(params.requester ?? "");
  const generation = String(params.owner_generation ?? "");
  if (!/^[a-zA-Z0-9_-]{32,}$/.test(proof) || !session.startsWith("session:") || !generation) {
    throw new BridgeCommandError("owner_refused", "missing private browser ownership proof");
  }
  return [proof, session, generation];
}
async function mutate<T>(params: Params, fn: (scope: Scope) => T, create = false): Promise<T> {
  return withSessionMutation(async () => {
    const [proof, session, generation] = identity(params);
    const all = await scopes();
    let scope = all[proof];
    if (!scope && create) scope = all[proof] = { session, generation, allocations: {} };
    if (!scope || scope.session !== session || scope.generation !== generation) {
      throw new BridgeCommandError("owner_refused", "browser owner generation is stale or unresolved");
    }
    const result = fn(scope);
    await chrome.storage.session.set({ ownerScopes: all });
    return result;
  });
}

export async function recordAllocation(params: Params, tab: string, state: string): Promise<void> {
  if (!params.owner_proof) return; // legacy clients retain capability-only behavior
  await mutate(params, scope => {
    const allocation = String(params.allocation_id ?? "");
    if (!allocation || scope.terminal) throw new BridgeCommandError("owner_refused", "browser scope ended");
    scope.allocations[allocation] = { tab: tab || undefined, state };
  });
}

/** Serialize owner commands through the entire side effect, not just the map
 * write. A finish cannot overtake an in-flight allocation and then let its late
 * navigation resurrect the tab. The daemon's deadlines remain bounded; a lost
 * response is replayed by allocation id rather than creating a second tab.
 */
export function withOwnership(
  method: string, params: Params, handler: () => Promise<Record<string, unknown>>,
  close: (params: Params) => Promise<Record<string, unknown>>,
): Promise<Record<string, unknown>> {
  const key = String(params.owner_proof ?? "legacy");
  const previous = queues.get(key) ?? Promise.resolve();
  const operate = async (): Promise<Record<string, unknown>> => {
    if (!params.owner_proof) {
      // A legacy capability cannot bypass fencing for a modern allocation.
      if (params.tab) {
        const surface = (await getSurfaces())[String(params.tab)];
        if (surface?.allocationId) throw new BridgeCommandError("owner_refused", "owner-aware client required");
      }
      return handler();
    }
    const [proof, session, generation] = identity(params);
    if (method === "owner_recover") {
      return withSessionMutation(async () => {
        const all = await scopes();
        const scope = all[proof];
        if (!scope) return { ownership_version: 1, state: "unresolved" };
        const predecessors = Array.isArray(params.previous_generations) ? params.previous_generations : [];
        if (scope.session !== session || (scope.generation !== generation && scope.generation !== params.previous_generation && !predecessors.includes(scope.generation))) {
          throw new BridgeCommandError("owner_refused", "browser owner generation is stale");
        }
        // A resume retires the ended scope. Keying this on the generation
        // CHANGING was unreachable for an in-process child: that path
        // deliberately reuses one generation per session (so a second live
        // instance can never fence the incumbent), so the clear never fired
        // and `open` below refused forever — the owner was told to resume and
        // then refused for not having resumed. `resumed_scope` is the owner's
        // own statement that this is a later run over a settled scope, and it
        // is only reachable here past the proof/session/generation check
        // above, so a foreign caller cannot use it to clear someone else's.
        if (scope.generation !== generation || params.resumed_scope === true) delete scope.terminal;
        scope.generation = generation;
        await chrome.storage.session.set({ ownerScopes: all });
        const allocation = scope.allocations[String(params.allocation_id ?? "")];
        const live = allocation?.tab && (await getSurfaces())[allocation.tab];
        const unresolved = allocation && ["allocating", "allocated", "cleanup_pending"].includes(allocation.state);
        return { ownership_version: 1, state: live ? (live.cleanupPending ? "cleanup_pending" : allocation.state) : unresolved ? "allocating" : "closed", tab: live ? allocation.tab : "", retention: scope.retention ?? "", terminal: scope.terminal ?? "", unknown_reservations: scope.unknownReservations ?? 0 };
      });
    }
    if (method === "open") {
      const existing = await mutate(params, scope => {
        if (scope.terminal) throw new BridgeCommandError("owner_refused", "browser scope ended; resume a new generation first");
        const id = String(params.allocation_id ?? "");
        if (!id) throw new BridgeCommandError("owner_refused", "missing browser allocation id");
        return scope.allocations[id];
      }, true);
      if (params.tab) {
        if (existing?.tab !== params.tab) throw new BridgeCommandError("owner_refused", "tab does not belong to allocation");
        return handler();
      }
      if (existing?.tab && (await getSurfaces())[existing.tab]) {
        // A response may have been lost after successful navigation. Return
        // the recorded capability without navigating/submitting a second time.
        const surface = (await getSurfaces())[existing.tab];
        if (!surface) throw new BridgeCommandError("tab_closed", "browser tab disappeared");
        const tab = await chrome.tabs.get(surface.tabId);
        return { tab: existing.tab, url: tab.url ?? "", title: tab.title ?? "", state: existing.state };
      }
      if (existing && existing.state !== "closed") {
        // The journal says this allocation began but no live tab carries it:
        // the worker died between chrome.tabs.create and the handle write.
        // That tab is UNKNOWN — counted and reported, never closed by
        // inference — but it must not lock the owner out of browsing, so the
        // allocation is retried under a fresh intent.
        await mutate(params, scope => {
          scope.unknownReservations = (scope.unknownReservations ?? 0) + 1;
        });
      }
      await recordAllocation(params, "", "allocating");
      return handler();
    }
    const scope = await mutate(params, value => value,
      ["tabs", "status", "request_access", "await_access", "cancel_access", "owner_retain", "owner_finish"].includes(method));
    if (method === "owner_retain") {
      const reason = String(params.reason ?? "").trim();
      if (!reason || reason.length > 500) throw new BridgeCommandError("owner_refused", "retention requires a bounded reason");
      await mutate(params, value => { value.retention = reason; });
      return { state: "retained" };
    }
    if (method === "owner_release") {
      await mutate(params, value => { delete value.retention; });
      return { state: "released", terminal: scope.terminal ?? "" };
    }
    if (method === "owner_finish") {
      await mutate(params, value => { value.terminal = String(params.outcome ?? "completed"); });
      if (scope.retention) return { state: "retained" };
      let pending = false;
      for (const allocation of Object.values(scope.allocations)) {
        if (!allocation.tab || !(await getSurfaces())[allocation.tab]) {
          // A worker may die after Chrome creates a tab but before its id is
          // journaled. That reservation is unknown, not invented free capacity.
          if (["allocating", "allocated", "cleanup_pending"].includes(allocation.state)) pending = true;
          continue;
        }
        try { await close({ ...params, tab: allocation.tab }); }
        catch { pending = true; }
      }
      return { state: pending ? "pending" : "closed" };
    }
    if (scope.terminal && method !== "close" && method !== "tabs") throw new BridgeCommandError("owner_refused", "browser scope ended");
    if (params.tab && !Object.values(scope.allocations).some(a => a.tab === params.tab)) {
      throw new BridgeCommandError("owner_refused", "tab does not belong to this browser owner");
    }
    return handler();
  };
  const run = previous.catch(() => {}).then(() => {
    if (method !== "open") return operate();
    // Pool admission spans create+persist, including legacy callers. Sharing
    // the allocation lane avoids two owners both observing the last free slot.
    const allocation = allocations.catch(() => {}).then(operate);
    allocations = allocation;
    return allocation;
  });
  queues.set(key, run);
  void run.finally(() => { if (queues.get(key) === run) queues.delete(key); }).catch(() => {});
  return run;
}
