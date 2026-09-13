import { awaitAccess, cancelAccessCommand, requestAccess } from "./commands/access";
import { close, goto, open, status, tabs } from "./commands/nav";
import { click, typeText } from "./commands/input";
import { readPage } from "./commands/read";
import { screenshot } from "./commands/shot";
import { snapshot } from "./commands/snapshot";
import { scroll } from "./commands/scroll";
import { logs } from "./commands/logs";
import { BridgeCommandError } from "./cdp";
import { clearAllAccessGrants, revokeExactOrigin, revokeLoopbackHost, revokeSiteGrant } from "./access-grants";
import { ACCESS_EXPIRY_ALARM, allowAllPending } from "./approval-store";
import { expireAccessRequest, resolveOrigin, restoreAccessQueue, setPendingObserver } from "./origins";
import { DEFAULT_PORT, getLocal, isRedactedToken } from "./state";
import { reconcileCommandTab, retitle } from "./tab-groups";
import { reclaimRemovedTab } from "./tab-lifecycle";
import { withOwnership } from "./ownership";
import {
  RECONNECT_ALARM_NAME,
  RECONNECT_ALARM_PERIOD_MINUTES,
  backoffDelayMs,
  shouldArmFastPath,
  shouldDialOnAlarm,
} from "./reconnect";
import { ErrorCode, type DaemonMessage, type ExtensionEvent, type Response } from "./protocol.gen";

const HANDLERS: Record<
  string,
  (params: Record<string, unknown>, requestId: string) => Promise<Record<string, unknown>>
> = {
  // The daemon's request id is also the origin-prompt correlation id. Minting
  // a private id here made the popup's decision impossible to match back to
  // the command the daemon is holding.
  open,
  goto,
  read: readPage,
  snapshot,
  screenshot,
  click,
  type: typeText,
  close,
  status,
  tabs,
  scroll,
  logs,
  // Async site-approval flow: request returns immediately after raising the
  // prompt; await polls the stored decision in bounded slices (access.ts
  // explains why slices, not a daemon long-poll).
  request_access: requestAccess,
  await_access: awaitAccess,
  cancel_access: cancelAccessCommand,
  // Presentation-only rename of an already-open tab's group, for a session
  // title that arrived after the tab was created. It drives no tab and reads
  // no page, so unlike every other handler it needs no surface admission.
  retitle,
  owner_recover: async () => ({}),
  owner_finish: async () => ({}),
  owner_retain: async () => ({}),
  owner_release: async () => ({}),
};

// How long a dial may sit unresolved before we force it closed and retry. A
// live loopback WS opens in milliseconds; this only bounds a pathological
// handshake that neither opens nor errors (finding A12).
const DIAL_TIMEOUT_MS = 10_000;

let socket: WebSocket | undefined;
let paired = false;
let attempt = 0;
let connected = false;
let connecting = false;
let alive = false;
// Best-effort fast-path timer (see reconnect.ts). Only meaningful while the
// worker is alive; it dies with a suspending worker and the alarm floor takes
// over — nothing may depend on it firing.
let fastPathTimer: ReturnType<typeof setTimeout> | undefined;

//: Monotonic id of the CURRENT wire, bumped by every dial.
//
// The socket alone cannot fence anything here, because `dispatch` is fired
// fire-and-forget: a handler that finishes after a reconnect would otherwise
// write its response — and its `tab_update` — to `socket`, which by then is the
// REPLACEMENT connection. That is not a cosmetic misdelivery: the daemon
// matches a response to the request IT sent on the OLD socket, so the new socket
// receives a frame it never asked for and the daemon's own
// "nothing is replayed on a new socket" contract (see `respond`) is broken from
// the other side. Reproduced by the concurrency audit by bundling this file,
// gating one handler, replacing the wire through the reconnect alarm and then
// releasing the handler — the NEW wire received
// `oldResponseOnNewWire=[tab_update bridge:1:old, {id: old-request, ok:true}]`.
//
// So every send that belongs to a REQUEST carries the generation of the wire the
// request arrived on, and anything that would land on a different wire is
// dropped with a log instead of being delivered.
let wireGeneration = 0;

function send(frame: object, generation: number = wireGeneration): void {
  if (generation !== wireGeneration) {
    // An event produced by a superseded request. Dropping it is the point: the
    // daemon that asked has already failed that request's future, and the live
    // connection has its own records to keep.
    console.warn("dropped an event from a superseded connection", frame);
    return;
  }
  if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(frame));
}

/**
 * Fire-and-forget chrome API call whose rejection nobody can act on.
 *
 * A floating promise that rejects surfaces in an MV3 worker as
 * `Uncaught (in promise) Error: …`, captured from the operator's own console
 * as "Could not establish connection. Receiving end does not exist." from
 * `chrome.runtime.sendMessage` when no popup was open to receive the frame.
 * None of these call sites has a caller who could act on a failure, so the only
 * correct handling is to record it and move on — but a fire-and-forget that can
 * surface as an uncaught error is a defect whether or not the rejection is
 * expected, and an uncaught error in the worker is indistinguishable from a
 * crash when someone is reading the console to diagnose a bridge fault.
 */
function fireAndForget(op: Promise<unknown> | undefined | void, what: string): void {
  void Promise.resolve(op).catch((error) => console.warn(`${what} failed`, error));
}

/**
 * Fire-and-forget an async function whose SYNCHRONOUS throw is as fatal as its
 * rejection.
 *
 * `fireAndForget` above takes a promise, so the call that produces it has
 * already run by the time containment is applied: an `async function` that
 * throws before its first `await` still rejects (safe), but a plain function
 * that throws synchronously escapes into the caller. That caller here is always
 * a Chrome event handler — an alarm tick, a runtime lifecycle event, the socket
 * `onmessage` — and an exception thrown out of one of those is an uncaught
 * error in the worker, which is the state this PR exists to eliminate: Chrome's
 * MV3 worker is poisoned by exactly that, and the operator's dead toolbar
 * clicks (2026-09-11 20:23) coincided with a worker that never dialled again.
 *
 * Taking a THUNK rather than a promise is what closes that window — the call
 * itself happens inside the try.
 */
function guarded(op: () => Promise<unknown> | unknown, what: string): void {
  try {
    fireAndForget(Promise.resolve(op()), what);
  } catch (error) {
    console.warn(`${what} failed`, error);
  }
}

// Raise a system notification when a site decision is pending (finding U2).
// BEST-EFFORT ONLY: on macOS this banner frequently never reaches the user —
// Chrome needs its own Notification Center authorization (System Settings →
// Notifications → Google Chrome) and on machines without it the notification
// only renders inside Chrome's extensions menu, invisible in practice
// (confirmed on a real machine). The PRIMARY signal is therefore the AGENT:
// the origin_not_allowed error and the request_access result text both tell
// it to message the user through the harness, which notifies reliably. This
// banner stays because it costs nothing and helps the machines where it works.
const PENDING_NOTIFICATION_ID = "lop-origin-pending";
let notificationQueueKey = "unreconciled";

// Command ids the worker has announced to the daemon as awaiting a human
// origin decision. Module-level so it survives across observer invocations
// but resets with the worker — a restarted worker re-announces its live
// entries on the first snapshot, and the daemon's own record resets on
// disconnect, so neither side carries stale ids across a restart.
const announcedAwaitingIds = new Set<string>();

/** Chrome's notification typings retain callback overloads across versions,
 * while MV3 implementations return promises. Normalize the runtime result so
 * the action-surface allSettled boundary always owns rejection handling. */
async function notificationResult(result: Promise<unknown> | unknown): Promise<void> {
  await Promise.resolve(result);
}

setPendingObserver(async (snapshot) => {
  // Command ids announced as awaiting a human decision. The daemon pops its
  // record when the RPC's response arrives or the RPC times out, but a queue
  // entry that is decided, cancelled, or expired on the extension side
  // produces neither when the worker loses the command (suspension, restart),
  // so /health kept echoing a prompt the popup could never resolve — the
  // stale echo that looped the approval popup. Announce clearances for ids
  // that leave the queue so the echo is bounded by the queue's lifetime.
  const liveCommandIds = new Set<string>();
  for (const entry of snapshot.queue) {
    if (entry.kind === "in_command" && entry.commandId) {
      liveCommandIds.add(entry.commandId);
      send({ event: "awaiting_origin", id: entry.commandId, origin: entry.origin });
      announcedAwaitingIds.add(entry.commandId);
    }
  }
  for (const id of announcedAwaitingIds) {
    if (liveCommandIds.has(id)) continue;
    announcedAwaitingIds.delete(id);
    send({ event: "awaiting_origin_cleared", id });
  }
  const count = snapshot.queue.length;
  // Badge/title reconciliation may legitimately repeat during a sweep, but an
  // identical aggregate OS notification is noise. Entry generations form a
  // stable key that changes on enqueue, decision, cancellation, and expiry.
  const nextNotificationKey = snapshot.queue.map((entry) => entry.entryId).join("\n");
  if (nextNotificationKey === notificationQueueKey) return;
  if (count) {
    const first = snapshot.queue[0]!;
    await notificationResult(chrome.notifications?.create(PENDING_NOTIFICATION_ID, {
      type: "basic",
      iconUrl: chrome.runtime.getURL("icons/icon-128.png"),
      title: "Local Operator needs your OK",
      message:
        count === 1
          ? `Allow the agent to open ${first.displayAuthority}? Click the extension icon in the toolbar to decide.`
          : `${count} site requests are waiting. Click the extension icon in the toolbar to decide.`,
      priority: 2,
    }));
  } else {
    await notificationResult(chrome.notifications?.clear(PENDING_NOTIFICATION_ID));
  }
  // Commit dedup state only after Chrome accepts create/clear. A transient
  // rejection leaves the previous key intact so the next identical sweep
  // retries instead of suppressing recovery.
  notificationQueueKey = nextNotificationKey;
});

// Clicking the banner opens the consent popup directly — one click instead of
// "find the toolbar icon". openPopup() historically demands a user gesture
// and Chrome has moved the boundary between versions; a notification click
// may or may not count as one, so failure is swallowed and the notification
// text keeps pointing at the toolbar icon as the fallback path.
chrome.notifications?.onClicked.addListener((id) => {
  if (id !== PENDING_NOTIFICATION_ID) return;
  try {
    const opened = chrome.action.openPopup() as Promise<void> | undefined;
    void opened?.catch(() => {});
  } catch {
    // No gesture credit for this click on this Chrome version — the toolbar
    // icon named in the notification text remains the way in.
  }
});

async function daemonPort(): Promise<number> {
  const { port } = await getLocal();
  return port ?? DEFAULT_PORT;
}

async function respond(response: Response, generation: number): Promise<void> {
  if (generation !== wireGeneration) {
    // The request this answers arrived on a connection that has since been
    // replaced. `worker.ts`'s own contract (see the note below) is that a stale
    // result must never be replayed onto a new socket — and the daemon has
    // already failed this request's future when it accepted the replacement, so
    // there is nothing left to answer. Say so rather than misdelivering it.
    console.warn(
      `dropped response for ${response.id}: it belongs to a superseded connection`,
    );
    return;
  }
  if (socket?.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify(response));
    return;
  }
  // A completed command whose socket is no longer OPEN: the daemon evicted it
  // ("later connection wins") mid-command, and this is the extension's side of
  // the resulting `RuntimeError('extension disconnected')` the daemon logs.
  // There is no correct place to DELIVER the frame — the socket is gone, the
  // daemon has already failed every pending future, and a response queue would
  // replay a stale result onto a NEW socket that never asked for it. So do not
  // pretend it was sent: say so, which is what makes those daemon-side
  // disconnects attributable to a dropped answer rather than a mystery timeout.
  console.warn(`dropped response for ${response.id}: the extension socket is not open`);
}

async function dispatch(
  request: { id: string; method: string; params: Record<string, unknown> },
  generation: number,
): Promise<void> {
  const handler = HANDLERS[request.method];
  if (!handler) {
    await respond({ id: request.id, ok: false, error: { code: ErrorCode.INTERNAL, message: `unknown method ${request.method}`, data: {} } }, generation);
    return;
  }
  try {
    // Rename propagation is presentation-only and deliberately precedes every
    // owned-tab command; failures never mask the command's real result.
    // `retitle` is excluded because reconciling IS its handler — riding this
    // hook too would run the same serialized reconcile twice per push.
    if (request.method !== "open" && request.method !== "retitle") {
      await reconcileCommandTab(request.params);
    }
    const result = await withOwnership(request.method, request.params,
      () => handler(request.params, request.id), close);
    // Push the driven page so the daemon (and the Connected popup) can show
    // the human what the agent is on (finding U3).
    if (typeof result.url === "string" && result.url) {
      // Carry the surface handle so the daemon tracks this tab individually.
      // Without it the daemon kept ONE global "driving" slot that the most
      // recent command overwrote, which framed a multi-tab world as a single
      // binding. `goto` returns no handle, so echo the request's own tab.
      //
      // Only a FULL handle may key a driven record. A handle-less `status`
      // answers with a REDACTED token (an unproven caller must not receive the
      // drive capability), and forwarding that string created a second key for
      // a tab already tracked under its full token — a duplicate that survived
      // the real close and advertised a dead URL forever, which is the phantom
      // this change removes. A redacted token is therefore reported as no
      // handle at all, so the daemon refreshes the most recent record instead
      // of forking one.
      const raw = typeof result.tab === "string" ? result.tab : String(request.params.tab ?? "");
      const handle = isRedactedToken(raw) ? "" : raw;
      send({ event: "tab_update", tab: handle, url: result.url, title: String(result.title ?? "") }, generation);
    }
    // An explicit `close` retires the surface, so tell the daemon now rather
    // than relying on the onRemoved listener: chrome.tabs.remove fires
    // onRemoved too, but announcing here keeps the driven record accurate even
    // if the worker is torn down before that event is delivered.
    // Prefer the handle the command RESOLVED (`closed`) over the one the
    // request carried: `close` legitimately accepts no `tab` param (the
    // pre-multi-tab shape that closes the sole surface), and announcing ""
    // there tells the daemon to blank EVERY driven record, other sessions'
    // live tabs included. The command knows exactly which surface it retired,
    // so it says so and the daemon drops precisely that one.
    if (request.method === "close") {
      const closed = typeof result.closed === "string" ? result.closed : String(request.params.tab ?? "");
      send({ event: "tab_closed", tab: isRedactedToken(closed) ? "" : closed }, generation);
    }
    await respond({ id: request.id, ok: true, result }, generation);
  } catch (error) {
    if (error instanceof BridgeCommandError) {
      await respond({ id: request.id, ok: false, error: { code: codeFor(error.code), message: error.message, data: error.data } }, generation);
    } else {
      await respond({ id: request.id, ok: false, error: { code: ErrorCode.INTERNAL, message: String(error), data: {} } }, generation);
    }
  }
}

function codeFor(code: string): ErrorCode {
  const values = Object.values(ErrorCode) as string[];
  return (values.includes(code) ? code : ErrorCode.INTERNAL) as ErrorCode;
}

async function connect(): Promise<void> {
  // `connecting` guards the window between `new WebSocket()` and `onopen`, when
  // `connected` is still false: without it, a `chrome.alarms` tick (or a wake
  // event) firing in that window starts a SECOND socket, the daemon's
  // "later-connection-wins" rule closes the first, and the resulting
  // teardown→reconnect cascades into a tight reconnect storm. The flag is set
  // SYNCHRONOUSLY, before the first `await` below: two `connect()` calls that
  // both reached the first await before either set it would each open a socket,
  // so the guard has to close that window too. One in-flight dial at a time
  // makes reconnection converge on a single stable socket.
  if (connected || connecting || !alive) return;
  connecting = true;

  let token: string | undefined;
  let port: number;
  try {
    ({ token } = await getLocal());
    port = await daemonPort();
  } catch {
    // A transient chrome.storage rejection must not leave `connecting` stuck
    // true — that would wedge every later dial at the guard above until the
    // next worker suspend reset the globals (finding A11). Reset and retry.
    connecting = false;
    scheduleReconnect();
    return;
  }

  // `new WebSocket()` THROWS synchronously on a malformed or blocked URL — and
  // `port` comes from chrome.storage, so a corrupted value reaches this line as
  // a constructor argument. Every caller of connect() is `void connect()` from
  // an event handler, so an escape here is an uncaught worker error rather than
  // a failed dial. Contain it and let the ordinary backoff retry: a bad stored
  // port is fixed by re-pairing, not by crashing the worker in between.
  let wire: WebSocket;
  try {
    wire = new WebSocket(`ws://127.0.0.1:${port}/extension`);
  } catch (error) {
    console.warn("extension dial failed to open a socket", error);
    connecting = false;
    scheduleReconnect();
    return;
  }
  socket = wire;
  // This dial's identity. Everything asynchronous that belongs to it — the
  // handshake writes, the frames it carries, and every response a handler it
  // dispatched eventually produces — is scoped to this number, so a later dial
  // cannot be written to by an earlier one (see `wireGeneration`).
  const generation = ++wireGeneration;

  // Explicit dial deadline: if a dead loopback handshake neither opens nor
  // fires onerror/onclose, `connecting` would otherwise stay true forever and
  // deadlock reconnection (finding A12). Force the socket closed after the
  // deadline; close() surfaces as onclose→teardown, which clears the guard and
  // reschedules. Cleared on any real settle below so a live socket is untouched.
  let dialTimer: ReturnType<typeof setTimeout> | undefined = setTimeout(() => {
    dialTimer = undefined;
    if (wire.readyState !== WebSocket.OPEN) {
      connecting = false;
      try {
        wire.close();
      } catch {
        // Already closing; teardown/scheduleReconnect still run below.
      }
      if (socket === wire) socket = undefined;
      scheduleReconnect();
    }
  }, DIAL_TIMEOUT_MS);
  const clearDialTimer = () => {
    if (dialTimer !== undefined) {
      clearTimeout(dialTimer);
      dialTimer = undefined;
    }
  };

  wire.onopen = () => {
    clearDialTimer();
    if (socket !== wire) {
      // A later dial already owns the worker; this one must not claim
      // `connected`, must not send its own `hello` (two handshakes race for the
      // daemon's single authority), and must not be left dangling.
      try {
        wire.close();
      } catch {
        // Already closing.
      }
      return;
    }
    connected = true;
    connecting = false;
    attempt = 0;
    const hello: ExtensionEvent = { event: "hello", proto: 1, token: token ?? "", extension_version: chrome.runtime.getManifest().version, browser: navigator.userAgent };
    wire.send(JSON.stringify(hello));
  };
  wire.onmessage = (message) => {
    if (socket !== wire) return; // a superseded socket's frames are not ours
    // A frame that does not parse is a DAEMON-side defect (a truncated write, a
    // future protocol version, a proxy injecting something), and it used to
    // throw straight out of this handler — an uncaught error in the worker for
    // one bad byte on the wire, with every later frame on a healthy socket
    // still pending. Drop the frame, keep the socket: the daemon retries or the
    // dial deadline reaps it, and the console says which one it was.
    let frame: DaemonMessage;
    try {
      const parsed: unknown = JSON.parse(String(message.data));
      // PARSING IS ONLY HALF THE GUARD. `null`, `2`, `"x"`, `true` and `[]` are
      // all VALID JSON, so they clear `JSON.parse` and then reach the `"method"
      // in frame` test below — and `in` throws `TypeError` on any non-object.
      // That is the same uncaught-throw-in-an-event-handler this whole block
      // exists to remove, reached by the same class of input (a truncated or
      // garbled daemon write) that motivated the parse guard, so the shape
      // check has to live inside the same guard rather than trusting the cast.
      // Arrays are rejected too: `"method" in []` is legal but an array is not
      // a frame, and letting one through would hand `dispatch` a bad request.
      if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
        console.warn("dropped an unparseable frame from the daemon", parsed);
        return;
      }
      frame = parsed as DaemonMessage;
    } catch (error) {
      console.warn("dropped an unparseable frame from the daemon", error);
      return;
    }
    // `dispatch` already answers its own failures through `respond`, but the
    // guard covers the path where dispatch itself cannot start (a throw before
    // its first await), which would otherwise land as an uncaught rejection.
    if ("method" in frame) guarded(() => dispatch(frame as { id: string; method: string; params: Record<string, unknown> }, generation), `dispatch ${String((frame as { method: string }).method)}`);
    // `send` on a socket that raced into CLOSING throws InvalidStateError. The
    // pong is the daemon's liveness probe, so losing one costs a link teardown
    // — but throwing here costs the whole worker.
    else if (frame.event === "ping") guarded(() => wire.send(JSON.stringify({ event: "pong" })), "pong");
    else if (frame.event === "hello_ack") {
      paired = frame.paired;
      fireAndForget(
        chrome.storage.session.set({ connState: frame.paired ? "connected" : "pairing" }),
        "connState write",
      );
    } else if (frame.event === "pair_result" && frame.ok) {
      fireAndForget(chrome.storage.local.set({ token: frame.token }), "token write");
    }
  };
  const teardown = (event?: CloseEvent) => {
    clearDialTimer();
    if (socket !== wire) {
      // A superseded socket finishing its close, long after a replacement dial
      // took over. Resetting the live connection's flags here — or publishing a
      // `connState` from it — would tear down the socket that is actually up:
      // the delayed-onclose half of the same defect class as a stale response.
      return;
    }
    connected = false;
    connecting = false;
    paired = false;
    socket = undefined;
    // Preserve the close code so the popup can distinguish a protocol mismatch
    // (4001 — "update needed", which pairing cannot fix) from an ordinary
    // disconnect (finding D2). 4003 is an unpair/revoke.
    if (event?.code === 4001) fireAndForget(chrome.storage.session.set({ connState: "incompatible" }), "connState write");
    else if (event?.code === 4003) fireAndForget(chrome.storage.session.set({ connState: "pairing" }), "connState write");
    // 4000 is the daemon's later-connection-wins eviction (daemon.py), which the
    // POPUP's own pairing socket triggers on every pair attempt. It is not a
    // loss of connectivity, so publishing "disconnected" here drove the popup's
    // render BACKWARDS mid-pair: the storage write re-enters render() through
    // chrome.storage.onChanged, and the card painted an extra transition
    // (pairing -> connected -> paired -> connected) as the user submitted.
    //
    // Only the STORAGE WRITE is suppressed. The socket really is gone, so the
    // local connected/connecting/paired resets above and the scheduleReconnect()
    // below must still run — the worker has to re-dial with the new token.
    // Drive authority is enforced daemon-side by link.paired (a second socket
    // arrives paired:false and its RPCs are refused not_paired), never by this
    // storage key, so suppressing the write grants nothing.
    else if (event?.code !== 4000) fireAndForget(chrome.storage.session.set({ connState: "disconnected" }), "connState write");
    scheduleReconnect();
  };
  wire.onclose = (event) => teardown(event);
  wire.onerror = () => {
    // onerror without a prior onopen still needs the connecting guard cleared,
    // else a failed dial wedges the worker as permanently "connecting". close()
    // triggers onclose→teardown which clears it.
    clearDialTimer();
    connecting = false;
    wire.close();
  };
}

// FAST PATH ONLY — see reconnect.ts for the two-tier design. This recovers a
// transient socket drop in seconds WHILE THE WORKER IS ALIVE. It is best-effort:
// a `setTimeout` does not survive worker suspension, so if the worker suspends
// before it fires the timer is lost and the reconnect alarm (the guaranteed
// floor) rewakes the worker and re-dials instead. Reconnection therefore never
// DEPENDS on this — it only makes the alive case faster than the ~1-min alarm.
function scheduleReconnect(): void {
  if (!shouldArmFastPath({ alive, fastPathPending: fastPathTimer !== undefined })) return;
  const delay = backoffDelayMs(attempt);
  attempt += 1;
  fastPathTimer = setTimeout(() => {
    fastPathTimer = undefined;
    guarded(connect, "fast-path dial");
  }, delay);
}

// GUARANTEED WAKE — the alarm is the only timer Chrome uses to wake a suspended
// MV3 worker, so it is the reconnection FLOOR. Period is kept at Chrome's
// reliable minimum (see RECONNECT_ALARM_PERIOD_MINUTES); the old 0.5-min period
// sat on the clamp edge where Chrome delayed or dropped the tick, which is why
// the automatic rewake never fired after idle suspension. `create` re-arms an
// existing alarm idempotently, so calling it at every worker start (top level +
// onStartup + onInstalled) guards against a lost alarm without stacking copies.
function ensureReconnectAlarm(): void {
  chrome.alarms.create(RECONNECT_ALARM_NAME, { periodInMinutes: RECONNECT_ALARM_PERIOD_MINUTES });
}
ensureReconnectAlarm();
// Every dial below is `guarded` rather than `void`d. These are the RECOVERY
// paths — the alarm floor is the only thing that rewakes a suspended worker —
// so a rejection escaping one of them is both an uncaught worker error and the
// loss of the tick that was meant to heal the connection.
chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === RECONNECT_ALARM_NAME && shouldDialOnAlarm({ connected, connecting })) {
    guarded(connect, "alarm dial");
  }
  // The sweep persists receipts, resolves in-command waiters through the queue
  // observer, and centrally re-arms the earliest remaining queue/result/grant
  // deadline. No caller owns this alarm independently.
  if (alarm.name === ACCESS_EXPIRY_ALARM) guarded(expireAccessRequest, "access expiry sweep");
});
chrome.runtime.onStartup.addListener(() => {
  alive = true;
  ensureReconnectAlarm();
  guarded(connect, "startup dial");
});
chrome.runtime.onInstalled.addListener(() => {
  alive = true;
  ensureReconnectAlarm();
  guarded(connect, "install dial");
});
// Cold-start convergence: on every worker start (including a rewake from
// suspension, when the globals have reset to their false initializers) the
// top-level dial runs immediately, the alarm is (re)armed as the floor, and
// both funnel through connect()'s connecting/connected guard onto ONE stable
// socket. `connecting` is never persisted, so a suspend can never leave it
// wedged true across a restart — a fresh worker always starts able to dial.
alive = true;
// Observer registration above precedes restoration. Chain startup so persisted
// queue state reconciles its global badge/title before connection work, and
// contain failure so MV3 never reports an unhandled top-level rejection.
void restoreAccessQueue()
  .catch((error) => console.warn("approval queue restore failed", error))
  .finally(() => guarded(connect, "cold-start dial"));

// TOP-LEVEL REGISTRATION IS LOAD-BEARING (MV3): a service worker is torn down
// when idle and re-instantiated by an event, and only listeners registered
// during that synchronous first evaluation are wired up to wake it. Registered
// inside a callback or after an await, this listener would silently not exist
// for the very events it must catch — the same lifecycle reasoning reconnect.ts
// applies to timers. It is deliberately not behind a connection check: a tab
// closed while the socket is down must still be reclaimed locally, and the
// daemon clears its own driven records on disconnect anyway.
//
// This is what makes "the user closed the tab" reach the daemon at all. Before
// it, nothing ever sent tab_closed and `status` advertised a dead tab forever.
chrome.tabs.onRemoved.addListener((tabId) => {
  void reclaimRemovedTab(tabId)
    .then((token) => {
      // Only OUR surfaces are announced: the user's own tabs are not ours to
      // report, and announcing one would clear a driven record still in use.
      if (token) send({ event: "tab_closed", tab: token });
    })
    .catch((error) => console.warn("tab close reclaim failed", error));
});

// A tab REPLACED (prerender activation, or a crashed tab restored under a new
// id) keeps the page but retires the old tab id, so the surface bound to that
// id is dead exactly as if it had been removed. Chrome fires no onRemoved for
// the replaced id, so without this the surface would linger as a ghost against
// the cap until some later command happened to notice.
chrome.tabs.onReplaced.addListener((_addedTabId, removedTabId) => {
  void reclaimRemovedTab(removedTabId)
    .then((token) => {
      if (token) send({ event: "tab_closed", tab: token });
    })
    .catch((error) => console.warn("tab replace reclaim failed", error));
});

chrome.storage.onChanged.addListener((changes, area) => {
  if (area === "session" && changes.accessQueue) {
    // With no popup open there is no receiver for this message, and MV3 rejects
    // the send with "Could not establish connection. Receiving end does not
    // exist." — an EXPECTED outcome for a background-only delivery attempt, not
    // an error, so it must not surface as an uncaught rejection in the worker
    // console (captured live from the operator's own hands).
    fireAndForget(
      chrome.runtime.sendMessage({ event: "origin_prompt", queue: changes.accessQueue.newValue }),
      "origin_prompt broadcast",
    );
  }
  // The all-sites switch is written by the options page directly (never by
  // a message to this worker or a daemon RPC: there is deliberately no such
  // path). This listener only reacts to it turning ON, resolving prompts
  // that are now moot so paused navigations resume instead of timing out.
  if (area === "local" && changes.allowAllSites && changes.allowAllSites.newValue === true) {
    void allowAllPending().catch((error) => console.warn("allow-all queue resolution failed", error));
  }
});
chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  // Decisions are keyed by ORIGIN (finding A6) and carry the prompt
  // GENERATION the popup rendered (round-2 B1): resolveOrigin rejects a
  // decision for a prompt that was replaced after the popup drew it.
  if (message?.event === "origin_grant_revoke") {
    revokeExactOrigin(String(message.origin))
      .then((applied) => sendResponse({ applied }))
      .catch(() => sendResponse({ applied: false }));
    return true;
  }
  if (message?.event === "host_grant_revoke") {
    revokeLoopbackHost(String(message.canonicalKey))
      .then((applied) => sendResponse({ applied }))
      .catch(() => sendResponse({ applied: false }));
    return true;
  }
  if (message?.event === "site_grant_revoke") {
    revokeSiteGrant(String(message.key))
      .then((applied) => sendResponse({ applied }))
      .catch(() => sendResponse({ applied: false }));
    return true;
  }
  if (message?.event === "clear_access_grants") {
    clearAllAccessGrants()
      .then((applied) => sendResponse({ applied }))
      .catch(() => sendResponse({ applied: false }));
    return true;
  }
  if (message?.event === "origin_decision") {
    // sendResponse + `return true` keeps the MV3 event alive until the
    // decision is DURABLY recorded (record, grant, allowlist, prompt
    // teardown). A fire-and-forget here let Chrome settle the popup's
    // sendMessage and suspend this worker mid-persistence, losing the
    // user's approval (round-2 M2).
    resolveOrigin(String(message.origin), message.decision, String(message.entryId ?? message.promptId ?? ""))
      .then((applied) => sendResponse({ applied }))
      .catch(() => sendResponse({ applied: false }));
    return true;
  }
  return undefined;
});
