import { close, goto, open, status } from "./commands/nav";
import { click, typeText } from "./commands/input";
import { readPage } from "./commands/read";
import { screenshot } from "./commands/shot";
import { snapshot } from "./commands/snapshot";
import { scroll } from "./commands/scroll";
import { logs } from "./commands/logs";
import { BridgeCommandError } from "./cdp";
import { resolveOrigin, setPendingObserver } from "./origins";
import { DEFAULT_PORT, getLocal } from "./state";
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
  scroll,
  logs,
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
let reconnectTimer: ReturnType<typeof setTimeout> | undefined;
// The request id currently being handled, so an origin pause can tell the
// daemon WHICH command to keep alive past the base timeout (finding A3).
let activeRequestId = "";

function send(frame: object): void {
  if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(frame));
}

// Raise a system notification when a site decision is pending, so a user who
// does not already have the popup open still sees that the agent is blocked on
// them rather than the command silently stalling toward a deny (finding U2).
const PENDING_NOTIFICATION_ID = "lop-origin-pending";
setPendingObserver((pending) => {
  if (pending) {
    send({ event: "awaiting_origin", id: activeRequestId, origin: pending.origin });
    chrome.notifications?.create(PENDING_NOTIFICATION_ID, {
      type: "basic",
      iconUrl: chrome.runtime.getURL("icons/icon-128.png"),
      title: "Local Operator needs your OK",
      message: `Allow the agent to open ${pending.hostname}? Click the extension to decide.`,
      priority: 2,
    });
  } else {
    chrome.notifications?.clear(PENDING_NOTIFICATION_ID);
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

function scheduleReconnect(): void {
  if (!alive || reconnectTimer) return;
  const delay = Math.min(30_000, 1_000 * 2 ** attempt);
  attempt += 1;
  reconnectTimer = setTimeout(() => {
    reconnectTimer = undefined;
    void connect();
  }, delay);
}

chrome.alarms.create("lop-bridge-reconnect", { periodInMinutes: 0.5 });
chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === "lop-bridge-reconnect" && !connected) void connect();
});
chrome.runtime.onStartup.addListener(() => {
  alive = true;
  void connect();
});
chrome.runtime.onInstalled.addListener(() => {
  alive = true;
  void connect();
});
alive = true;
void connect();

chrome.storage.onChanged.addListener((changes, area) => {
  if (area === "local" && changes.pendingOrigin) void chrome.runtime.sendMessage({ event: "origin_prompt", pending: changes.pendingOrigin.newValue });
});
chrome.runtime.onMessage.addListener((message) => {
  // Decisions are keyed by ORIGIN now (finding A6): one command can pause on
  // several origins in a redirect chain, and each resolves independently.
  if (message?.event === "origin_decision") resolveOrigin(String(message.origin), message.decision);
});
