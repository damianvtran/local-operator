// Two-tier reconnection timing for the MV3 service worker (finding A13).
//
// WHY two tiers, and why the split matters:
//
// An MV3 service worker is torn down after ~30s idle. A pending `setTimeout`
// does NOT survive that teardown and does NOT wake a suspended worker — the
// timer is simply lost when the worker dies (Chrome MV3 lifecycle). So any
// reconnection that bottoms out on `setTimeout` never happens once the worker
// has idle-suspended: the socket stays down until a real event (a page load,
// the toolbar icon) happens to wake the worker. That is the observed defect —
// `extension connected: no` sticking for minutes with no automatic recovery.
//
// `chrome.alarms` is the ONLY timer Chrome will use to wake a suspended worker,
// so reconnection after suspension MUST bottom out on an alarm, never on a
// `setTimeout`. The two tiers:
//
//   - ALARM (guaranteed floor): a periodic alarm re-dials whenever the socket
//     is down. It survives suspension and wakes the worker. Chrome clamps alarm
//     periods to a 30s minimum and, per the chrome.alarms docs, "may delay them
//     an arbitrary amount more"; periods below 0.5 min "will not be honored and
//     will cause a warning". The previous 0.5-min period sat exactly on that
//     clamp edge, where Chrome delays or drops the tick — which is why the
//     automatic rewake never fired. One minute is comfortably above the clamp
//     where the alarm reliably fires, at the cost of up to ~1 min to rewake
//     after idle suspension. That worst case is acceptable for a bridge that is
//     idle by definition when suspended; instant rewake still happens on the
//     real events that already work (page load, toolbar click).
//
//   - setTimeout (fast path, best-effort): while the worker is still ALIVE, a
//     transient mid-session socket drop should reconnect in ~seconds, not wait
//     up to a minute for the next alarm. An exponential backoff `setTimeout`
//     handles that. It is best-effort ONLY: if the worker suspends before it
//     fires the timer dies with the worker, and the alarm floor picks
//     reconnection back up on the next tick. Nothing may depend on it running.

export const RECONNECT_ALARM_NAME = "lop-bridge-reconnect";

// Guaranteed-wake period. Kept at/above Chrome's 30s alarm clamp so the tick
// actually fires on a suspended worker (see module header). Do not lower below
// 1 without evidence the shorter period is honored in RELEASED Chrome — the
// unpacked-dev build has no floor and will mislead a local test.
export const RECONNECT_ALARM_PERIOD_MINUTES = 1;

// Ceiling on the alive-only fast-path backoff. A capped exponential keeps a
// dead daemon from being hammered while still recovering a live socket quickly.
export const MAX_BACKOFF_MS = 30_000;

// The FIRST delay (attempt 0 = 1000ms) is load-bearing beyond backoff, and must
// not be shortened to make pairing feel faster.
//
// Submitting a pairing code opens the popup's own socket, and the daemon's rule
// is later-connection-wins: that socket evicts the worker's (close 4000). The
// worker then re-dials after this delay. Shortening it so the "Paired." card
// hands over to "Connected." sooner inverts the race — the worker's re-dial
// arrives while the popup's pairing socket is still open and EVICTS IT, and
// pairing breaks outright (measured: 4/4 stuck on "Paired." with the daemon
// reporting paired: false).
//
// So the ~1s dwell on the success card is not merely honest latency waiting for
// /health to catch up; it is the window that lets the popup finish pairing on
// its own socket before the worker takes the connection back. Treat it as a
// serialisation constraint, not as a UX cost to be optimised away.
export function backoffDelayMs(attempt: number): number {
  return Math.min(MAX_BACKOFF_MS, 1_000 * 2 ** attempt);
}

// Whether to arm the best-effort `setTimeout` fast path. Only while the worker
// is ALIVE and no fast-path timer is already pending: a suspended worker cannot
// run a `setTimeout` at all (the alarm floor covers that case), and a second
// concurrent timer would just double-dial into the connecting guard.
export function shouldArmFastPath(input: { alive: boolean; fastPathPending: boolean }): boolean {
  return input.alive && !input.fastPathPending;
}

// Whether an alarm tick (the guaranteed wake) should dial. On a cold wake after
// suspension the globals have reset to false, so this returns true and
// reconnection proceeds without any page interaction — the whole point of the
// fix. When the worker is alive and already connected or mid-dial it stays a
// no-op; connect()'s own guard is authoritative, this just avoids a pointless
// call (and a redundant one right after the top-level `void connect()` that a
// cold wake already kicked off, which will have set connecting=true).
export function shouldDialOnAlarm(input: { connected: boolean; connecting: boolean }): boolean {
  return !input.connected && !input.connecting;
}
