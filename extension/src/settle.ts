import { BridgeCommandError } from "./errors";

// Per-call deadlines for chrome API awaits. The numbers are CATASTROPHE
// CEILINGS ("this operation is milliseconds when healthy"), not measured
// distributions, and they are deliberately collected here rather than spelled
// at each call site so the timeout-chain invariant stays readable as one table:
//
//   extension per-call deadline (5-15 s)
//     < extension nav settle (30 s, below - unchanged)
//     < daemon COMMAND_TIMEOUTS (20-30 s, protocol.py)
//     < daemon awaiting_origin extension (+65 s)
//     < session client timeout (base + 65 + 5)
//
// The innermost deadline must fire first so the daemon always receives a TYPED
// answer instead of timing out blind.

/** One `chrome.debugger.sendCommand`. Must fit under the tightest daemon budget
 * that reaches CDP: read/snapshot/screenshot are 20 s (protocol.py).
 *
 * 15 s, not 10 s. The record originally set this from the "milliseconds when
 * healthy" premise and flagged it as the weakest number in the table, with the
 * rule "raise it if real measurements land within 2× of it". They did: on a
 * real-Chrome rig a legitimate (not SIGSTOPped) `read` took **7.49 s** and a
 * heavy `screenshot` **10.91 s** — the second EXCEEDS the old bound, so a
 * 10 s ceiling kills work the daemon's own 20 s budget would have accepted
 * (QA A8), and the load A/B showed the fixed head 9/16 where the pre-fix
 * extension, which had no inner bound at all, was 14/16 over the same sequence
 * in the same load band (both 16/16 at normal load, so this is insurance for a
 * starved host, not a fix for a normal-load bug).
 *
 * 15 s is the most the nesting invariant allows for the tightest budget it
 * must stay inside: 20 s (read/snapshot/screenshot) minus 5 s for the handler
 * to build and send its answer. It still fires strictly before the daemon's
 * deadline, and it still makes a HUNG call settle — the property the whole
 * helper exists for. Growth is bounded by the invariant, not by taste: raise
 * this only with a fresh measurement and re-derive the table below. */
export const CDP_DEADLINE_MS = 15_000;
/** `chrome.debugger.attach`/`detach`, and the trivial `Runtime.evaluate` probe
 * `ownAttachment` uses to tell our surviving attachment from DevTools. Attach is
 * a local browser operation with no page involvement: milliseconds or broken.
 * If the PROBE does not answer in this window the session is dead, which is
 * exactly the answer `ownAttachment` wants. */
export const CDP_ATTACH_DEADLINE_MS = 5_000;
/** `chrome.tabs.*`, `chrome.tabGroups.*`, and `chrome.storage.*`: browser-process
 * IPC with no page script on the path (for `session` storage, no disk either).
 * Seconds here means the browser or the storage layer is wedged. */
export const CHROME_API_DEADLINE_MS = 5_000;
/** `chrome.scripting.executeScript` runs page script, so it shares the CDP
 * risk profile and the same 20 s `read` budget - hence the same bound and the
 * same reason to raise it from 10 s (see `CDP_DEADLINE_MS`; QA measured a real
 * `read` at 7.49 s). */
export const SCRIPTING_DEADLINE_MS = 15_000;

/**
 * Bound one chrome API await.
 *
 * A HANG IS NOT AN ERROR, and that single sentence is the whole reason this
 * exists. Every serialized queue in this extension is built as
 * `queue.catch(() => {}).then(op)` so that "each link swallows its
 * predecessor's failure so the chain cannot poison later calls". That is true
 * for a REJECTION and false for a HANG: `.catch()` never runs on a promise that
 * never settles, so one stuck op parks every later command behind it — for
 * every session, because the queues are module-global, not per-session. This is
 * the only thing that makes those chains self-draining: the op SETTLES (with a
 * typed, retryable error) so the chain's `.catch` moves on to the next link.
 *
 * Note the op is ABANDONED, not cancelled — `chrome.debugger.sendCommand` has
 * no cancellation — so a stuck call may still complete later into nothing. That
 * is safe: its result is discarded and the attach state is reconciled by
 * `chrome.debugger.onDetach` and by pruneSurface. One residue is NOT covered by
 * either, and is reconciled by `attach`'s own adopt path instead: a timed-out
 * `chrome.debugger.attach` rejects before it registers the tab in the local
 * `attached` set, so Chrome can still complete the attach afterwards, and a
 * later `detach()` then returns early on `!attached.has(tabId)` and leaves a
 * real debugger session behind. It self-heals on the next `cdp()` for that tab
 * (the re-attach is refused with "already attached" and `ownAttachment` adopts
 * it) — see `attach`.
 *
 * THE POLICY, stated once because it is what every call site is judged against
 * (contention-scoping addendum D4): a deadline guarantees that the OP SETTLES
 * — so the queue link drains and the next command runs — and guarantees NOTHING
 * about the mutation, which may land afterwards. The corollary is why this
 * paragraph is longer than one sentence: every `chrome.storage.*.set` inside a
 * `deadline()` is a write that may commit after its caller gave up, and since
 * each set is a whole-key overwrite of a value read before the stall, a late set
 * can revert a newer write. Per call class, that is either already reconciled —
 * with the mechanism named — or it is not and the cost is stated:
 *
 *   - surfaces map (`state.ts` putSurface/removeSurface/touchSurface): a
 *     resurrected dead entry is pruned by `liveSurfaces` on the next
 *     open/tabs/status; a lost entry stops resolving, which is a typed
 *     `tab_closed`, and the tab stays journaled so `owner_recover` reports it.
 *   - `ownerScopes` (`ownership.ts` `mutate`): every read validates session +
 *     generation and refuses, so a stale write surfaces as a typed
 *     `owner_refused` rather than being honoured.
 *   - access queue/receipts (`approval-store.ts`): liveness is computed on read
 *     (`expiresAt` re-tested) and re-derived by the sweep, so a revived entry
 *     reads as absent.
 *   - snapshot refs (`state.ts` setRefs): already closed — refs carry an `epoch`
 *     and consumers refuse a mismatch. This is the model the others are measured
 *     against.
 *   - log buffers (`log-capture.ts`): nothing to do — in-memory only, no storage
 *     write in the path, and the enables are idempotent behind a tabId guard.
 *   - CDP commands and `attach`: covered above.
 *   - one-shot grants (`origins.ts` consumeOnceGrant/consumeGrantFor): the ONE
 *     genuine residue, and it is deliberately NOT closed here. Two existing
 *     properties bound it — the grant is requester-bound and TTL-bounded (10
 *     min) — so the worst case is one requester spending its own grant twice
 *     inside its own TTL, not a cross-owner consent hole. Deferred because the
 *     closing move (a monotonic per-key spend counter, or a read-back verify
 *     inside `withSessionMutation`) is a consent-surface change that wants its
 *     own review round and its own repro. What WOULD promote it into scope: a
 *     repro where a late set restores a grant for a DIFFERENT requester or a
 *     different origin.
 *
 * The related ordering hypothesis — a delayed `storage.set` letting a stale
 * write overwrite a newer one (the surfaces-map row above) — is a mechanically
 * plausible consequence of the whole-key overwrite, and it is UNPROVEN: nothing
 * in this change measures it, in a fault-injected or in a native-Chrome run, so
 * nothing here claims native storage reordering. The fix it would justify
 * (epoch-guarded map writes) is out of scope. The fixture that would settle it
 * is named in the addendum: a whole-map write from a stalled caller that
 * resolves after a peer's set has already committed.
 *
 * Do NOT wrap the queue itself with this. Timing out a chain head while the op
 * still holds a half-read copy of `surfaces` is a lost update or a double-spent
 * one-shot grant (state.ts documents the reproduced double-spend); bound the
 * awaits INSIDE the ops instead, which is what this does.
 *
 * Rejects with code `internal` and `data.stalled = <what>`, deliberately not a
 * new wire code: an already-released daemon validates `ErrorDetail.code`
 * against its own ErrorCode enum, so an unknown value fails
 * `Response.model_validate` and the frame is SILENTLY DROPPED — the command
 * then times out, strictly worse than an `internal` it understands.
 */
export function deadline<T>(op: Promise<T>, ms: number, what: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => {
      reject(
        new BridgeCommandError("internal", `${what} did not respond within ${ms}ms`, {
          stalled: what,
        }),
      );
    }, ms);
    // Both arms clear the timer, and both are attached immediately: the
    // abandoned op's eventual rejection must be HANDLED here or it would
    // surface as an uncaught rejection in the worker.
    op.then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      (error) => {
        clearTimeout(timer);
        reject(error);
      },
    );
  });
}

/**
 * Resolve when the tab's main frame finishes its next navigation.
 *
 * Shared by nav (goto/open) and input (click-navigation) so both report the
 * page that actually arrived rather than the one being left. Rejects with a
 * typed nav_failed/nav_timeout so the daemon can map it to a model-facing
 * string instead of a silent hang.
 */
export function settle(tabId: number, timeout = 30_000): Promise<void> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(
      () => done(new BridgeCommandError("nav_timeout", "navigation timed out")),
      timeout,
    );
    const complete = (details: chrome.webNavigation.WebNavigationFramedCallbackDetails) => {
      if (details.tabId === tabId && details.frameId === 0) done();
    };
    const failed = (details: chrome.webNavigation.WebNavigationFramedErrorCallbackDetails) => {
      if (details.tabId === tabId && details.frameId === 0) {
        done(new BridgeCommandError("nav_failed", details.error));
      }
    };
    function done(error?: Error): void {
      clearTimeout(timer);
      chrome.webNavigation.onCompleted.removeListener(complete);
      chrome.webNavigation.onErrorOccurred.removeListener(failed);
      if (error) reject(error);
      else resolve();
    }
    chrome.webNavigation.onCompleted.addListener(complete);
    chrome.webNavigation.onErrorOccurred.addListener(failed);
  });
}
