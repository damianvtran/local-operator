import { awaitAccess, requestAccess } from "./commands/access";
import { close, goto, open, status, tabs } from "./commands/nav";
import { click, typeText } from "./commands/input";
import { readPage } from "./commands/read";
import { screenshot } from "./commands/shot";
import { snapshot } from "./commands/snapshot";
import { scroll } from "./commands/scroll";
import { logs } from "./commands/logs";
import { BridgeCommandError } from "./cdp";
import { expireAccessRequest, resolveOrigin, setPendingObserver } from "./origins";
import { DEFAULT_PORT, getLocal } from "./state";
import { reconcileCommandTab } from "./tab-groups";
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
// The request id currently being handled, so an origin pause can tell the
// daemon WHICH command to keep alive past the base timeout (finding A3).
let activeRequestId = "";

function send(frame: object): void {
  if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(frame));
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
setPendingObserver((pending) => {
  if (pending) {
    send({ event: "awaiting_origin", id: activeRequestId, origin: pending.origin });
    chrome.notifications?.create(PENDING_NOTIFICATION_ID, {
      type: "basic",
      iconUrl: chrome.runtime.getURL("icons/icon-128.png"),
      title: "Local Operator needs your OK",
      message: `Allow the agent to open ${pending.authority}? Click the extension icon in the toolbar to decide.`,
      priority: 2,
    });
  } else {
    chrome.notifications?.clear(PENDING_NOTIFICATION_ID);
  }
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

async function respond(response: Response): Promise<void> {
  if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(response));
}

async function dispatch(request: { id: string; method: string; params: Record<string, unknown> }): Promise<void> {
  const handler = HANDLERS[request.method];
  if (!handler) {
    await respond({ id: request.id, ok: false, error: { code: ErrorCode.INTERNAL, message: `unknown method ${request.method}`, data: {} } });
    return;
  }
  activeRequestId = request.id;
  try {
    // Rename propagation is presentation-only and deliberately precedes every
    // owned-tab command; failures never mask the command's real result.
    if (request.method !== "open") await reconcileCommandTab(request.params);
    const result = await handler(request.params, request.id);
    // Push the driven page so the daemon (and the Connected popup) can show
    // the human what the agent is on (finding U3).
    if (typeof result.url === "string" && result.url) {
      send({ event: "tab_update", url: result.url, title: String(result.title ?? "") });
    }
    await respond({ id: request.id, ok: true, result });
  } catch (error) {
    if (error instanceof BridgeCommandError) {
      await respond({ id: request.id, ok: false, error: { code: codeFor(error.code), message: error.message, data: error.data } });
    } else {
      await respond({ id: request.id, ok: false, error: { code: ErrorCode.INTERNAL, message: String(error), data: {} } });
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

  const wire = new WebSocket(`ws://127.0.0.1:${port}/extension`);
  socket = wire;

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
    connected = true;
    connecting = false;
    attempt = 0;
    const hello: ExtensionEvent = { event: "hello", proto: 1, token: token ?? "", extension_version: chrome.runtime.getManifest().version, browser: navigator.userAgent };
    wire.send(JSON.stringify(hello));
  };
  wire.onmessage = (message) => {
    const frame = JSON.parse(String(message.data)) as DaemonMessage;
    if ("method" in frame) void dispatch(frame);
    else if (frame.event === "ping") wire.send(JSON.stringify({ event: "pong" }));
    else if (frame.event === "hello_ack") {
      paired = frame.paired;
      void chrome.storage.session.set({ connState: frame.paired ? "connected" : "pairing" });
    } else if (frame.event === "pair_result" && frame.ok) chrome.storage.local.set({ token: frame.token });
  };
  const teardown = (event?: CloseEvent) => {
    clearDialTimer();
    connected = false;
    connecting = false;
    paired = false;
    socket = undefined;
    // Preserve the close code so the popup can distinguish a protocol mismatch
    // (4001 — "update needed", which pairing cannot fix) from an ordinary
    // disconnect (finding D2). 4003 is an unpair/revoke.
    if (event?.code === 4001) void chrome.storage.session.set({ connState: "incompatible" });
    else if (event?.code === 4003) void chrome.storage.session.set({ connState: "pairing" });
    else void chrome.storage.session.set({ connState: "disconnected" });
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
    void connect();
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
chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === RECONNECT_ALARM_NAME && shouldDialOnAlarm({ connected, connecting })) void connect();
  // TTL sweep for an async access request nobody is polling: without it an
  // abandoned request would leave the "!" badge and popup prompt up until the
  // next access RPC happened to run the lazy sweep. The alarm is created by
  // request_access with the record's own expiry time.
  if (alarm.name === "lop-access-expiry") void expireAccessRequest();
});
chrome.runtime.onStartup.addListener(() => {
  alive = true;
  ensureReconnectAlarm();
  void connect();
});
chrome.runtime.onInstalled.addListener(() => {
  alive = true;
  ensureReconnectAlarm();
  void connect();
});
// Cold-start convergence: on every worker start (including a rewake from
// suspension, when the globals have reset to their false initializers) the
// top-level dial runs immediately, the alarm is (re)armed as the floor, and
// both funnel through connect()'s connecting/connected guard onto ONE stable
// socket. `connecting` is never persisted, so a suspend can never leave it
// wedged true across a restart — a fresh worker always starts able to dial.
alive = true;
void connect();

chrome.storage.onChanged.addListener((changes, area) => {
  if (area === "local" && changes.pendingOrigin) void chrome.runtime.sendMessage({ event: "origin_prompt", pending: changes.pendingOrigin.newValue });
});
chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  // Decisions are keyed by ORIGIN (finding A6) and carry the prompt
  // GENERATION the popup rendered (round-2 B1): resolveOrigin rejects a
  // decision for a prompt that was replaced after the popup drew it.
  if (message?.event === "origin_decision") {
    // The worker treats popup messages as hostile input. resolveOrigin validates
    // both the decision vocabulary and loopback eligibility before persistence.
    // sendResponse + `return true` keeps the MV3 event alive until the
    // decision is DURABLY recorded (record, grant, allowlist, prompt
    // teardown). A fire-and-forget here let Chrome settle the popup's
    // sendMessage and suspend this worker mid-persistence, losing the
    // user's approval (round-2 M2).
    resolveOrigin(String(message.origin), message.decision, String(message.promptId ?? ""))
      .then((applied) => sendResponse({ applied }))
      .catch(() => sendResponse({ applied: false }));
    return true;
  }
  return undefined;
});
