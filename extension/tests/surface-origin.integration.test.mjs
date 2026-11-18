/* Page-touching commands re-check the tab's CURRENT origin against the site
 * grants before they run.
 *
 * Origin approval used to be enforced only while a navigating command
 * (open/goto/click/select) held Fetch interception. A page the user approved
 * can navigate ITSELF afterwards (a timer, a meta refresh, a script) to a site
 * the user never approved, and the next `read`/`snapshot`/`screenshot`/... then
 * handed that site's content to the agent with no check at all. These tests
 * drive the real worker dispatch over a stub socket — the same harness as
 * driven-handles.integration.test.mjs — and run the real page functions
 * against a fake `document`/`location`, so they observe what the daemon would
 * actually receive.
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
const APPROVED = "https://approved.example";
const OTHER = "https://other.example";
const SECRET = "content-the-user-never-approved";

/**
 * Chrome stub whose tab URL and in-page `location` are independently
 * controllable: `tabUrl` is what chrome.tabs.get reports, `docUrl` is what the
 * page script sees when it runs (they differ only in the TOCTOU test).
 */
function installChrome({ surfaces, origins = {}, tabUrl, docUrl = tabUrl, session = {} }) {
  const areas = {
    session: new Map(Object.entries({ surfaces, ...session })),
    local: new Map(Object.entries({ origins })),
  };
  const page = { tabUrl, docUrl };
  const calls = { scripts: 0, cdp: [] };
  const makeArea = (name) => ({
    get: async (keys) => {
      const out = {};
      for (const key of Array.isArray(keys) ? keys : [keys]) {
        if (areas[name].has(key)) out[key] = structuredClone(areas[name].get(key));
      }
      return out;
    },
    set: async (obj) => { for (const [k, v] of Object.entries(obj)) areas[name].set(k, structuredClone(v)); },
    remove: async (keys) => { for (const k of Array.isArray(keys) ? keys : [keys]) areas[name].delete(k); },
  });
  globalThis.chrome = {
    storage: { session: makeArea("session"), local: makeArea("local"), onChanged: { addListener: () => {} } },
    alarms: { create: () => {}, clear: async () => {}, onAlarm: { addListener: () => {} } },
    action: Object.fromEntries(
      ["setBadgeBackgroundColor", "setBadgeTextColor", "setBadgeText", "setTitle"].map((m) => [m, async () => {}]),
    ),
    debugger: {
      attach: async () => {}, detach: async () => {},
      sendCommand: async (_target, method) => {
        calls.cdp.push(method);
        if (method === "Page.captureScreenshot") return { data: "PNGDATA" };
        return {};
      },
      onEvent: { addListener: () => {}, removeListener: () => {} }, onDetach: { addListener: () => {} },
    },
    scripting: {
      // Run the REAL injected function against a fake page, so the test sees
      // exactly what the command would return from a document at `docUrl`.
      executeScript: async ({ func, args = [] }) => {
        calls.scripts += 1;
        const loc = new URL(page.docUrl);
        const saved = { document: globalThis.document, location: globalThis.location };
        globalThis.location = { href: loc.href, origin: loc.origin };
        globalThis.document = {
          querySelector: () => ({
            cloneNode: () => ({ querySelectorAll: () => [], textContent: `${SECRET} @ ${loc.origin}` }),
          }),
        };
        try {
          return [{ result: func(...args) }];
        } finally {
          globalThis.document = saved.document;
          globalThis.location = saved.location;
        }
      },
    },
    tabs: {
      get: async (tabId) => ({ id: tabId, url: page.tabUrl, title: "Page" }),
      update: async () => ({}),
      remove: async () => {},
      onRemoved: { addListener: () => {} }, onReplaced: { addListener: () => {} },
      onUpdated: { addListener: () => {} },
    },
    webNavigation: { onCommitted: { addListener: () => {} } },
    tabGroups: undefined,
    notifications: { create: async () => {}, clear: async () => {}, onClicked: { addListener: () => {} } },
    windows: { get: async () => ({ id: 1 }), getCurrent: async () => ({ id: 1 }) },
    runtime: {
      getURL: (p) => `chrome-extension://test/${p}`,
      getManifest: () => ({ version: "0.1.7" }),
      onStartup: { addListener: () => {} }, onInstalled: { addListener: () => {} },
      onMessage: { addListener: () => {} }, sendMessage: async () => {},
    },
  };
  return {
    page,
    calls,
    surfaces: () => areas.session.get("surfaces"),
    restore: () => {
      delete globalThis.chrome;
      delete globalThis.WebSocket;
      Reflect.deleteProperty(globalThis, "navigator");
    },
  };
}

// One bundle for the file, built before any test and removed after all of
// them: each test re-imports it under a fresh query to get a fresh worker.
const bundleDir = await mkdtemp(join(tmpdir(), "lop-surface-origin-it-"));
const bundlePath = join(bundleDir, "module.mjs");
await build({ entryPoints: ["src/worker.ts"], bundle: true, platform: "node", format: "esm", outfile: bundlePath });
test.after(() => rm(bundleDir, { recursive: true, force: true }));

let loads = 0;
async function loadWorker(options) {
  const chromeState = installChrome(options);
  const frames = [];
  let socket;
  Object.defineProperty(globalThis, "navigator", {
    value: { userAgent: "node-test" }, configurable: true, writable: true,
  });
  globalThis.WebSocket = class {
    static OPEN = 1;
    readyState = 1;
    constructor() { socket = this; queueMicrotask(() => this.onopen?.()); }
    send(data) { frames.push(JSON.parse(String(data))); }
    close() {}
  };
  await import(pathToFileURL(bundlePath) + `?load=${++loads}`);
  for (let i = 0; i < 40 && !frames.some((f) => f.event === "hello"); i++) await tick();
  assert.ok(frames.some((f) => f.event === "hello"), "worker connected");
  return {
    frames,
    chromeState,
    async request(method, params) {
      const id = `req-${frames.length}-${method}`;
      socket.onmessage?.({ data: JSON.stringify({ id, method, params }) });
      for (let i = 0; i < 400 && !frames.some((f) => f.id === id); i++) await tick();
      const response = frames.find((f) => f.id === id);
      assert.ok(response, `no response to ${method}`);
      return response;
    },
  };
}

const surface = () => ({ tabId: 42, nonce: NONCE, epoch: 1, createdAt: 1, lastUsedAt: 2 });
const approvedOnly = { [APPROVED]: "allow" };

test("read refuses a page that navigated itself to an unapproved site", async () => {
  const worker = await loadWorker({ surfaces: { [TOKEN]: surface() }, origins: approvedOnly, tabUrl: `${OTHER}/landing` });
  try {
    const response = await worker.request("read", { tab: TOKEN });
    assert.equal(response.ok, false, `read must be refused, got ${JSON.stringify(response.result)}`);
    assert.equal(response.error.code, "origin_not_allowed");
    assert.equal(response.error.data.origin, OTHER);
    assert.equal(worker.chromeState.calls.scripts, 0, "no page script may run on the unapproved document");
    assert.ok(!JSON.stringify(worker.frames).includes(SECRET), "no page content reaches the daemon");
  } finally { worker.chromeState.restore(); }
});

test("read on an approved page still returns its text unchanged", async () => {
  const worker = await loadWorker({ surfaces: { [TOKEN]: surface() }, origins: approvedOnly, tabUrl: `${APPROVED}/docs` });
  try {
    const response = await worker.request("read", { tab: TOKEN });
    assert.equal(response.ok, true, JSON.stringify(response.error));
    assert.equal(response.result.text, `${SECRET} @ ${APPROVED}`);
    assert.equal(response.result.url, `${APPROVED}/docs`);
  } finally { worker.chromeState.restore(); }
});

test("an unknown handle still answers tab_closed", async () => {
  const worker = await loadWorker({ surfaces: { [TOKEN]: surface() }, origins: approvedOnly, tabUrl: `${APPROVED}/docs` });
  try {
    const response = await worker.request("read", { tab: `bridge:42:${"0".repeat(32)}` });
    assert.equal(response.ok, false);
    assert.equal(response.error.code, "tab_closed");
  } finally { worker.chromeState.restore(); }
});

test("read discards a result whose document left the checked origin mid-command", async () => {
  // chrome.tabs.get still reports the approved URL while the document the
  // script actually ran in is already the other site.
  const worker = await loadWorker({
    surfaces: { [TOKEN]: surface() }, origins: approvedOnly,
    tabUrl: `${APPROVED}/docs`, docUrl: `${OTHER}/landing`,
  });
  try {
    const response = await worker.request("read", { tab: TOKEN });
    assert.equal(response.ok, false, `stale-origin result must be discarded, got ${JSON.stringify(response.result)}`);
    assert.equal(response.error.code, "origin_not_allowed");
    assert.ok(!JSON.stringify(worker.frames).includes(SECRET));
  } finally { worker.chromeState.restore(); }
});

// Every page-touching command shares the gate; a table so a command added to
// the extractor family later is one row, not a new test.
const PAGE_COMMANDS = [
  ["snapshot", {}],
  ["screenshot", {}],
  ["query", { selector: "a" }],
  ["styles", { selector: "a" }],
  ["hit_test", { x: 1, y: 1 }],
  ["ancestors", { selector: "a" }],
  ["scroll", { direction: "down" }],
  ["logs", {}],
  ["click", { selector: "a" }],
  ["type", { selector: "input", text: "x" }],
  ["select", { selector: "select", options: ["a"] }],
];

for (const [method, params] of PAGE_COMMANDS) {
  test(`${method} refuses an unapproved current origin before touching the page`, async () => {
    const worker = await loadWorker({ surfaces: { [TOKEN]: surface() }, origins: approvedOnly, tabUrl: `${OTHER}/landing` });
    try {
      const response = await worker.request(method, { tab: TOKEN, ...params });
      assert.equal(response.ok, false, `${method} must be refused`);
      assert.equal(response.error.code, "origin_not_allowed", JSON.stringify(response.error));
      assert.equal(worker.chromeState.calls.scripts, 0, `${method} ran page script`);
      assert.ok(!worker.chromeState.calls.cdp.includes("Page.captureScreenshot"), `${method} captured the page`);
    } finally { worker.chromeState.restore(); }
  });
}

test("screenshot of an approved page still works", async () => {
  const worker = await loadWorker({ surfaces: { [TOKEN]: surface() }, origins: approvedOnly, tabUrl: `${APPROVED}/docs` });
  try {
    const response = await worker.request("screenshot", { tab: TOKEN });
    assert.equal(response.ok, true, JSON.stringify(response.error));
    assert.equal(response.result.data, "PNGDATA");
  } finally { worker.chromeState.restore(); }
});

test("status and close still work on an unapproved page, so the agent can recover", async () => {
  const worker = await loadWorker({ surfaces: { [TOKEN]: surface() }, origins: approvedOnly, tabUrl: `${OTHER}/landing` });
  try {
    const status = await worker.request("status", { tab: TOKEN });
    assert.equal(status.ok, true, JSON.stringify(status.error));
    const closed = await worker.request("close", { tab: TOKEN });
    assert.equal(closed.ok, true, JSON.stringify(closed.error));
  } finally { worker.chromeState.restore(); }
});

test("an origin this tab was admitted to by an allow-once decision stays readable", async () => {
  // Allow-once grants are consumed by the navigation that used them, so the
  // persistent grant set never names the origin; the surface remembers it.
  const worker = await loadWorker({
    surfaces: { [TOKEN]: { ...surface(), admittedOrigins: [OTHER] } },
    origins: approvedOnly, tabUrl: `${OTHER}/landing`,
  });
  try {
    const response = await worker.request("read", { tab: TOKEN });
    assert.equal(response.ok, true, JSON.stringify(response.error));
    assert.equal(response.result.text, `${SECRET} @ ${OTHER}`);
  } finally { worker.chromeState.restore(); }
});

test("an allow-once grant raised for the refused read admits the retry and is remembered", async () => {
  // The refusal teaches request_access; the user's "allow once" lands as a
  // requester-bound once grant, which the retried read consumes exactly like
  // a retried goto would.
  const requester = "session:abc";
  const worker = await loadWorker({
    surfaces: { [TOKEN]: surface() }, origins: approvedOnly, tabUrl: `${OTHER}/landing`,
    session: {
      onceGrants: {
        [`${OTHER}\n${requester}`]: { origin: OTHER, requester, expiresAt: Date.now() + 60_000 },
      },
    },
  });
  try {
    const first = await worker.request("read", { tab: TOKEN, requester });
    assert.equal(first.ok, true, JSON.stringify(first.error));
    const again = await worker.request("read", { tab: TOKEN, requester });
    assert.equal(again.ok, true, "the admitted origin stays readable after the grant is spent");
    assert.deepEqual(worker.chromeState.surfaces()[TOKEN].admittedOrigins, [OTHER]);
  } finally { worker.chromeState.restore(); }
});

test("about:blank (a fresh tab before its first navigation) stays readable", async () => {
  const worker = await loadWorker({ surfaces: { [TOKEN]: surface() }, origins: approvedOnly, tabUrl: "about:blank", docUrl: `${APPROVED}/` });
  try {
    const response = await worker.request("screenshot", { tab: TOKEN });
    assert.equal(response.ok, true, JSON.stringify(response.error));
  } finally { worker.chromeState.restore(); }
});
