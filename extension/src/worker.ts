import { close, goto, open, status } from "./commands/nav";
import { click, typeText } from "./commands/input";
import { readPage } from "./commands/read";
import { screenshot } from "./commands/shot";
import { snapshot } from "./commands/snapshot";
import { BridgeCommandError } from "./cdp";
import { resolveOrigin } from "./origins";
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
};

let socket: WebSocket | undefined;
let paired = false;
let attempt = 0;
let connected = false;
let alive = false;
let reconnectTimer: ReturnType<typeof setTimeout> | undefined;

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
  try {
    const result = await handler(request.params, request.id);
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
  if (connected || !alive) return;
  const { token } = await getLocal();
  const port = await daemonPort();
  const wire = new WebSocket(`ws://127.0.0.1:${port}/extension`);
  socket = wire;
  wire.onopen = () => {
    connected = true;
    attempt = 0;
    const hello: ExtensionEvent = { event: "hello", proto: 1, token: token ?? "", extension_version: chrome.runtime.getManifest().version, browser: navigator.userAgent };
    wire.send(JSON.stringify(hello));
  };
  wire.onmessage = (message) => {
    const frame = JSON.parse(String(message.data)) as DaemonMessage;
    if ("method" in frame) void dispatch(frame);
    else if (frame.event === "ping") wire.send(JSON.stringify({ event: "pong" }));
    else if (frame.event === "hello_ack") paired = frame.paired;
    else if (frame.event === "pair_result" && frame.ok) chrome.storage.local.set({ token: frame.token });
  };
  const teardown = () => {
    connected = false;
    paired = false;
    socket = undefined;
    scheduleReconnect();
  };
  wire.onclose = teardown;
  wire.onerror = () => wire.close();
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
  if (message?.event === "origin_decision") resolveOrigin(String(message.requestId), message.decision);
});
