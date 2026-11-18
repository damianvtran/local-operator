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
  // `intercept`: whether a navigation emits Fetch.requestPaused for its
  // document (false = a load the interceptor never sees, e.g. one a service
  // worker serves). `onCdp` lets a test move the tab mid-command.
  const page = { tabUrl, docUrl, intercept: true, onCdp: null };
  const calls = { scripts: 0, cdp: [], tabsGet: 0 };
  const event = () => {
    const listeners = new Set();
    return {
      listeners,
      addListener: (fn) => listeners.add(fn),
      removeListener: (fn) => listeners.delete(fn),
      fire: (...args) => { for (const fn of [...listeners]) fn(...args); },
    };
  };
  const cdpEvents = event();
  const nav = { onCommitted: event(), onCompleted: event(), onErrorOccurred: event() };
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
        page.onCdp?.(method);
        if (method === "Page.captureScreenshot") return { data: "PNGDATA" };
        return {};
      },
      onEvent: cdpEvents, onDetach: { addListener: () => {} },
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
      get: async (tabId) => { calls.tabsGet += 1; return { id: tabId, url: page.tabUrl, title: "Page" }; },
      // A navigation: Fetch pause (when intercepted), then the top-frame
      // commit, then completion — the order Chrome reports them in.
      update: async (tabId, { url }) => {
        setTimeout(async () => {
          if (page.intercept) {
            cdpEvents.fire({ tabId }, "Fetch.requestPaused", { requestId: "doc-1", request: { url } });
            for (let i = 0; i < 20; i++) await tick();
          }
          page.tabUrl = url;
          page.docUrl = url;
          nav.onCommitted.fire({ tabId, frameId: 0, url });
          for (let i = 0; i < 20; i++) await tick();
          nav.onCompleted.fire({ tabId, frameId: 0, url });
        }, 0);
        return {};
      },
      remove: async () => {},
      onRemoved: { addListener: () => {} }, onReplaced: { addListener: () => {} },
      onUpdated: { addListener: () => {} },
    },
    webNavigation: nav,
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
    cdpEvents,
    nav,
    session: () => areas.session,
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

// ---------------------------------------------------------------------------
// Remediation round 1 (R1-R4, Q1, Q2): captured state is scoped to the origin
// the tab is admitted to, and an allow-once admission is recorded even when
// the navigation is never intercepted.

const requester = "session:abc";
const onceFor = (origin) => ({
  onceGrants: { [`${origin}\n${requester}`]: { origin, requester, expiresAt: Date.now() + 60_000 } },
});
const settleTicks = async (n = 60) => { for (let i = 0; i < n; i++) await tick(); };

test("logs keeps console output an unapproved page wrote out of an approved page's logs (R1)", async () => {
  const worker = await loadWorker({ surfaces: { [TOKEN]: surface() }, origins: approvedOnly, tabUrl: `${APPROVED}/docs` });
  try {
    // Resume arms log capture on the surface (startLogCapture).
    const resumed = await worker.request("open", { tab: TOKEN, url: `${APPROVED}/docs` });
    assert.equal(resumed.ok, true, JSON.stringify(resumed.error));
    const fire = (method, params) => worker.chromeState.cdpEvents.fire({ tabId: 42 }, method, params);
    const context = (id, origin) => fire("Runtime.executionContextCreated", {
      context: { id, origin, auxData: { isDefault: true, frameId: "main" } },
    });
    const say = (id, text) => fire("Runtime.consoleAPICalled", {
      type: "log", executionContextId: id, args: [{ type: "string", value: text }],
    });
    context(1, APPROVED);
    say(1, "A-OWN-LOG");
    // The approved page navigates itself to the other site, which logs...
    fire("Runtime.executionContextsCleared", {});
    context(2, OTHER);
    say(2, "B-ORIGIN-SECRET-MARKER");
    fire("Runtime.exceptionThrown", { exceptionDetails: { executionContextId: 2, text: "B-EXCEPTION" } });
    fire("Log.entryAdded", { entry: { level: "error", text: "B-NETWORK-ENTRY", url: `${OTHER}/api?token=x` } });
    // ...and comes back.
    fire("Runtime.executionContextsCleared", {});
    context(3, APPROVED);
    say(3, "A-SECOND-LOG");
    fire("Log.entryAdded", { entry: { level: "warning", text: "A-NETWORK-ENTRY" } });

    const response = await worker.request("logs", { tab: TOKEN });
    assert.equal(response.ok, true, JSON.stringify(response.error));
    assert.deepEqual(
      response.result.entries.map((entry) => entry.text),
      ["A-OWN-LOG", "A-SECOND-LOG", "A-NETWORK-ENTRY"],
      "the approved page's own logs are returned; the other site's are not",
    );
    assert.ok(!JSON.stringify(worker.frames).includes("B-"), "no entry from the unapproved page reaches the daemon");
    // Entry shape on the wire is unchanged (the origin tag is not serialized).
    assert.deepEqual(Object.keys(response.result.entries[0]).sort(), ["level", "line", "source", "text", "timestamp", "url"]);
  } finally { worker.chromeState.restore(); }
});

for (const intercept of [false, true]) {
  const label = intercept ? "with Fetch interception (S12 control)" : "without Fetch interception (S11)";
  test(`goto on an allow-once grant keeps the page readable ${label} (Q1)`, async () => {
    const worker = await loadWorker({
      surfaces: { [TOKEN]: surface() }, origins: approvedOnly, tabUrl: `${APPROVED}/docs`, session: onceFor(OTHER),
    });
    worker.chromeState.page.intercept = intercept;
    try {
      const moved = await worker.request("goto", { tab: TOKEN, url: `${OTHER}/landing`, requester });
      assert.equal(moved.ok, true, JSON.stringify(moved.error));
      await settleTicks();
      const read = await worker.request("read", { tab: TOKEN, requester });
      assert.equal(read.ok, true, `the page the user just allowed must stay readable: ${JSON.stringify(read.error)}`);
      assert.equal(read.result.text, `${SECRET} @ ${OTHER}`);
      assert.deepEqual(worker.chromeState.session().get("onceGrants"), {}, "the grant is spent exactly once");
      assert.deepEqual(worker.chromeState.surfaces()[TOKEN].admittedOrigins, [OTHER]);
      assert.equal(worker.chromeState.calls.cdp.includes("Fetch.continueRequest"), intercept);
    } finally { worker.chromeState.restore(); }
  });
}

test("an allow-once admission ends when the tab commits another origin (R3)", async () => {
  const worker = await loadWorker({
    surfaces: { [TOKEN]: surface() }, origins: approvedOnly, tabUrl: `${APPROVED}/docs`, session: onceFor(OTHER),
  });
  try {
    const moved = await worker.request("goto", { tab: TOKEN, url: `${OTHER}/landing`, requester });
    assert.equal(moved.ok, true, JSON.stringify(moved.error));
    await settleTicks();
    // The page leaves on its own and later comes back.
    const { page, nav } = worker.chromeState;
    page.tabUrl = page.docUrl = `${APPROVED}/next`;
    nav.onCommitted.fire({ tabId: 42, frameId: 0, url: page.tabUrl });
    await settleTicks();
    // A subframe commit is not a document change for the tab.
    nav.onCommitted.fire({ tabId: 42, frameId: 7, url: `${OTHER}/frame` });
    page.tabUrl = page.docUrl = `${OTHER}/again`;
    nav.onCommitted.fire({ tabId: 42, frameId: 0, url: page.tabUrl });
    await settleTicks();
    const read = await worker.request("read", { tab: TOKEN, requester });
    assert.equal(read.ok, false, `returning needs the normal grant again, got ${JSON.stringify(read.result)}`);
    assert.equal(read.error.code, "origin_not_allowed");
    assert.equal(worker.chromeState.surfaces()[TOKEN].admittedOrigins, undefined);
  } finally { worker.chromeState.restore(); }
});

test("a saved Deny outranks an earlier allow-once admission (R3)", async () => {
  const worker = await loadWorker({
    surfaces: { [TOKEN]: { ...surface(), admittedOrigins: [OTHER] } },
    origins: { ...approvedOnly, [OTHER]: "deny" }, tabUrl: `${OTHER}/landing`,
  });
  try {
    const read = await worker.request("read", { tab: TOKEN });
    assert.equal(read.ok, false, `denied origin must be refused, got ${JSON.stringify(read.result)}`);
    assert.equal(read.error.code, "origin_not_allowed");
    assert.equal(worker.chromeState.calls.scripts, 0);
  } finally { worker.chromeState.restore(); }
});

test("revoke_access ends an allow-once admission a live tab holds (R3)", async () => {
  const worker = await loadWorker({
    surfaces: { [TOKEN]: { ...surface(), admittedOrigins: [OTHER] } }, origins: approvedOnly, tabUrl: `${OTHER}/landing`,
  });
  try {
    const revoked = await worker.request("revoke_access", { url: `${OTHER}/` });
    assert.equal(revoked.ok, true, JSON.stringify(revoked.error));
    const read = await worker.request("read", { tab: TOKEN });
    assert.equal(read.ok, false, `revoked origin must be refused, got ${JSON.stringify(read.result)}`);
    assert.equal(read.error.code, "origin_not_allowed");
  } finally { worker.chromeState.restore(); }
});

test("a screenshot whose tab committed an unapproved origin mid-capture is discarded (Q2/R2)", async () => {
  const worker = await loadWorker({ surfaces: { [TOKEN]: surface() }, origins: approvedOnly, tabUrl: `${APPROVED}/docs` });
  const { page } = worker.chromeState;
  page.onCdp = (method) => { if (method === "Page.captureScreenshot") page.tabUrl = `${OTHER}/landing`; };
  try {
    const response = await worker.request("screenshot", { tab: TOKEN });
    assert.equal(response.ok, false, `result from the other site must be discarded, got ${JSON.stringify(response.result)}`);
    assert.equal(response.error.code, "origin_not_allowed");
    assert.deepEqual(response.error.data, { origin: OTHER });
    assert.ok(!JSON.stringify(worker.frames).includes("PNGDATA"), "the captured pixels never reach the daemon");
  } finally { worker.chromeState.restore(); }
});

test("a screenshot whose tab moved within approved pages is returned (Q2 control)", async () => {
  const worker = await loadWorker({ surfaces: { [TOKEN]: surface() }, origins: approvedOnly, tabUrl: `${APPROVED}/docs` });
  const { page } = worker.chromeState;
  page.onCdp = (method) => { if (method === "Page.captureScreenshot") page.tabUrl = `${APPROVED}/other-path`; };
  try {
    const response = await worker.request("screenshot", { tab: TOKEN });
    assert.equal(response.ok, true, JSON.stringify(response.error));
    assert.equal(response.result.data, "PNGDATA");
  } finally { worker.chromeState.restore(); }
});

test("the refusal carries the page's origin, not its full URL (R4)", async () => {
  const worker = await loadWorker({
    surfaces: { [TOKEN]: surface() }, origins: approvedOnly, tabUrl: `${OTHER}/account?session_token=abc123`,
  });
  try {
    const response = await worker.request("read", { tab: TOKEN });
    assert.equal(response.ok, false);
    assert.deepEqual(response.error.data, { origin: OTHER });
    assert.ok(!JSON.stringify(worker.frames).includes("abc123"), "the page's path/query never reaches the daemon");
  } finally { worker.chromeState.restore(); }
});
