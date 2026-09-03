/* What the worker tells the daemon it is DRIVING, and what it says went away.
 *
 * These drive real `status`/`close` commands through the real worker dispatch
 * over a stub socket, because both defects they guard live in the seam between
 * a command's RESULT and the event the worker derives from it — reading either
 * side alone shows nothing wrong.
 *
 * 1. A handle-less `status` answers with a REDACTED handle (an unproven caller
 *    must not receive the drive capability). Forwarded as a `tab_update`
 *    handle it keyed a SECOND driven record for one tab, and that duplicate
 *    survived the real close, advertising a dead URL forever — the phantom
 *    this release exists to remove, reintroduced one layer down.
 * 2. `close` legitimately takes no `tab` param (the pre-multi-tab shape that
 *    closes the sole surface). Announcing `tab: ""` there tells the daemon to
 *    blank EVERY driven record, other sessions' live tabs included, even
 *    though the extension knew exactly which surface it had just retired.
 */
import assert from "node:assert/strict";
import test from "node:test";
import { build } from "esbuild";
import { pathToFileURL } from "node:url";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

const NONCE = "abcdef0123456789abcdef0123456789";
const TOKEN = `bridge:42:${NONCE}`;

/** Chrome surface big enough to evaluate worker.ts and serve one live tab. */
function installChrome(surfaces) {
  const areas = { session: new Map(Object.entries({ surfaces })), local: new Map() };
  const removed = [];
  const makeArea = (name) => ({
    get: async (keys) => {
      const out = {};
      for (const key of Array.isArray(keys) ? keys : [keys]) {
        if (areas[name].has(key)) out[key] = areas[name].get(key);
      }
      return out;
    },
    set: async (obj) => { for (const [k, v] of Object.entries(obj)) areas[name].set(k, v); },
    remove: async (keys) => { for (const k of Array.isArray(keys) ? keys : [keys]) areas[name].delete(k); },
  });
  globalThis.chrome = {
    storage: { session: makeArea("session"), local: makeArea("local"), onChanged: { addListener: () => {} } },
    alarms: { create: () => {}, clear: async () => {}, onAlarm: { addListener: () => {} } },
    action: Object.fromEntries(
      ["setBadgeBackgroundColor", "setBadgeTextColor", "setBadgeText", "setTitle"].map((m) => [m, async () => {}]),
    ),
    debugger: {
      attach: async () => {}, detach: async () => {}, sendCommand: async () => ({}),
      onEvent: { addListener: () => {} }, onDetach: { addListener: () => {} },
    },
    tabs: {
      get: async (tabId) => ({ id: tabId, url: "https://example.com/live", title: "Live" }),
      remove: async (tabId) => { removed.push(tabId); },
      onRemoved: { addListener: () => {} }, onReplaced: { addListener: () => {} },
      onUpdated: { addListener: () => {} },
    },
    tabGroups: undefined,
    notifications: { create: async () => {}, clear: async () => {}, onClicked: { addListener: () => {} } },
    windows: { get: async () => ({ id: 1 }), getCurrent: async () => ({ id: 1 }) },
    runtime: {
      getURL: (p) => `chrome-extension://test/${p}`,
      getManifest: () => ({ version: "0.1.6" }),
      onStartup: { addListener: () => {} }, onInstalled: { addListener: () => {} },
      onMessage: { addListener: () => {} }, sendMessage: async () => {},
    },
  };
  return {
    removed,
    surfaces: () => areas.session.get("surfaces"),
    restore: () => {
      delete globalThis.chrome;
      delete globalThis.WebSocket;
      Reflect.deleteProperty(globalThis, "navigator");
    },
  };
}

/** Load worker.ts with a socket that records frames and can inject requests. */
async function loadWorker(surfaces) {
  const chromeState = installChrome(surfaces);
  const frames = [];
  let socket;
  // `navigator` is a getter-only global on modern Node, so it must be
  // redefined rather than assigned (a plain assignment throws).
  Object.defineProperty(globalThis, "navigator", {
    value: { userAgent: "node-test" },
    configurable: true,
    writable: true,
  });
  globalThis.WebSocket = class {
    static OPEN = 1;
    readyState = 1;
    constructor() { socket = this; queueMicrotask(() => this.onopen?.()); }
    send(data) { frames.push(JSON.parse(String(data))); }
    close() {}
  };
  const dir = await mkdtemp(join(tmpdir(), "lop-driven-handles-it-"));
  const outfile = join(dir, "module.mjs");
  await build({ entryPoints: ["src/worker.ts"], bundle: true, platform: "node", format: "esm", outfile });
  await import(pathToFileURL(outfile) + `?${Date.now()}`);
  for (let i = 0; i < 40 && !frames.some((f) => f.event === "hello"); i++) await tick();
  assert.ok(frames.some((f) => f.event === "hello"), "worker connected");
  return {
    frames,
    chromeState,
    /** Feed a daemon->extension request in and wait for its response. */
    async request(method, params) {
      const id = `req-${frames.length}`;
      socket.onmessage?.({ data: JSON.stringify({ id, method, params }) });
      for (let i = 0; i < 200 && !frames.some((f) => f.id === id); i++) await tick();
      const response = frames.find((f) => f.id === id);
      assert.ok(response, `no response to ${method}`);
      return response;
    },
    close: () => rm(dir, { recursive: true, force: true }),
  };
}

const surface = { tabId: 42, nonce: NONCE, epoch: 1, createdAt: 1, lastUsedAt: 2 };

test("a handle-less status never reports a redacted handle as a driven tab", async () => {
  const worker = await loadWorker({ [TOKEN]: surface });
  try {
    const response = await worker.request("status", {});
    assert.equal(response.ok, true);
    // The command still redacts, which is the security property (M1).
    assert.equal(response.result.tab, "bridge:42:abcdef…");
    const updates = worker.frames.filter((f) => f.event === "tab_update");
    assert.equal(updates.length, 1, "one navigation, one update");
    // ...but the EVENT carries no handle, so the daemon refreshes the record
    // it already has instead of forking a second key for the same tab.
    assert.equal(updates[0].tab, "", "redacted handle must not key a driven record");
    assert.equal(updates[0].url, "https://example.com/live");
  } finally { await worker.close(); worker.chromeState.restore(); }
});

test("a keyed status reports the real handle, so per-tab tracking still works", async () => {
  const worker = await loadWorker({ [TOKEN]: surface });
  try {
    const response = await worker.request("status", { tab: TOKEN });
    assert.equal(response.ok, true);
    const updates = worker.frames.filter((f) => f.event === "tab_update");
    assert.equal(updates.at(-1).tab, TOKEN, "a proven caller's own handle is forwarded intact");
  } finally { await worker.close(); worker.chromeState.restore(); }
});

test("close with no tab param announces the surface it actually retired", async () => {
  const worker = await loadWorker({ [TOKEN]: surface });
  try {
    const response = await worker.request("close", {});
    assert.equal(response.ok, true);
    const closes = worker.frames.filter((f) => f.event === "tab_closed");
    assert.equal(closes.length, 1);
    // Without this the frame carried "", which the daemon can only read as
    // "blank every driven record" — including other sessions' live tabs.
    assert.equal(closes[0].tab, TOKEN, "the sole-surface close shape names its own handle");
    assert.deepEqual(worker.chromeState.removed, [42], "the tab really was closed");
  } finally { await worker.close(); worker.chromeState.restore(); }
});

test("close with an explicit tab param announces that same handle", async () => {
  const worker = await loadWorker({ [TOKEN]: surface });
  try {
    const response = await worker.request("close", { tab: TOKEN });
    assert.equal(response.ok, true);
    const closes = worker.frames.filter((f) => f.event === "tab_closed");
    assert.equal(closes.at(-1).tab, TOKEN);
  } finally { await worker.close(); worker.chromeState.restore(); }
});
