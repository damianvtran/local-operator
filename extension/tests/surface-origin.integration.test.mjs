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
import { copyFile, mkdtemp, rm } from "node:fs/promises";
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
function installChrome({ surfaces, origins = {}, tabUrl, docUrl = tabUrl, session = {}, attachError, retirementFails = false }) {
  const areas = {
    session: new Map(Object.entries({ surfaces, ...session })),
    local: new Map(Object.entries({ origins })),
  };
  // `intercept`: whether a navigation emits Fetch.requestPaused for its
  // document (false = a load the interceptor never sees, e.g. one a service
  // worker serves). `onCdp` lets a test move the tab mid-command.
  // `redirectTo`: where a navigation actually lands (a server redirect the
  // interceptor does not see), instead of the requested URL.
  const page = { tabUrl, docUrl, intercept: true, onCdp: null, redirectTo: null };
  const calls = {
    scripts: 0, cdp: [], tabsGet: 0, attach: 0, removals: 0,
    detach: 0, targets: 0, updates: [], creates: 0, windowUpdates: 0, frames: 0,
  };
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
  const events = { removed: event(), replaced: event(), updated: event(), detached: event() };
  const nav = { onCommitted: event(), onCompleted: event(), onErrorOccurred: event() };
  const makeArea = (name) => ({
    get: async (keys) => {
      const out = {};
      for (const key of Array.isArray(keys) ? keys : [keys]) {
        if (areas[name].has(key)) out[key] = structuredClone(areas[name].get(key));
      }
      return out;
    },
    set: async (obj) => {
      if (retirementFails && name === "session" && obj.surfaces && !Object.keys(obj.surfaces).length) {
        throw new Error("fixture storage rejected retirement");
      }
      for (const [k, v] of Object.entries(obj)) areas[name].set(k, structuredClone(v));
    },
    remove: async (keys) => { for (const k of Array.isArray(keys) ? keys : [keys]) areas[name].delete(k); },
  });
  globalThis.chrome = {
    storage: { session: makeArea("session"), local: makeArea("local"), onChanged: { addListener: () => {} } },
    alarms: { create: () => {}, clear: async () => {}, onAlarm: { addListener: () => {} } },
    action: Object.fromEntries(
      ["setBadgeBackgroundColor", "setBadgeTextColor", "setBadgeText", "setTitle"].map((m) => [m, async () => {}]),
    ),
    debugger: {
      attach: async () => {
        calls.attach += 1;
        if (attachError) throw Error(attachError);
      },
      detach: async () => { calls.detach += 1; },
      getTargets: async () => { calls.targets += 1; return []; },
      sendCommand: async (_target, method) => {
        calls.cdp.push(method);
        page.onCdp?.(method);
        if (method === "Page.captureScreenshot") return { data: "PNGDATA" };
        if (method === "Page.getFrameTree") return { frameTree: { frame: { id: "main" } } };
        return {};
      },
      onEvent: cdpEvents, onDetach: events.detached,
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
      update: async (tabId, options) => {
        calls.updates.push(options);
        const { url } = options;
        setTimeout(async () => {
          if (page.intercept) {
            cdpEvents.fire({ tabId }, "Fetch.requestPaused", { requestId: "doc-1", request: { url } });
            for (let i = 0; i < 20; i++) await tick();
          }
          const landed = page.redirectTo ?? url;
          page.tabUrl = landed;
          page.docUrl = landed;
          nav.onCommitted.fire({ tabId, frameId: 0, url: landed });
          for (let i = 0; i < 20; i++) await tick();
          nav.onCompleted.fire({ tabId, frameId: 0, url: landed });
        }, 0);
        return {};
      },
      create: async () => { calls.creates += 1; throw Error("unexpected new tab"); },
      remove: async () => { calls.removals += 1; },
      onRemoved: events.removed, onReplaced: events.replaced, onUpdated: events.updated,
    },
    webNavigation: { ...nav, getAllFrames: async () => { calls.frames += 1; return []; } },
    tabGroups: undefined,
    notifications: { create: async () => {}, clear: async () => {}, onClicked: { addListener: () => {} } },
    windows: {
      get: async () => ({ id: 1 }), getCurrent: async () => ({ id: 1 }),
      update: async () => { calls.windowUpdates += 1; },
    },
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
    events,
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
// Targeted smoke runs can exercise the exact artifact from node build.mjs;
// the regular tests build the same worker entry point without needing dist.
if (process.env.LOCAL_OPERATOR_TEST_BRIDGE_WORKER) {
  await copyFile(process.env.LOCAL_OPERATOR_TEST_BRIDGE_WORKER, bundlePath);
} else {
  await build({ entryPoints: ["src/worker.ts"], bundle: true, platform: "node", format: "esm", outfile: bundlePath });
}
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

const CHROME_ATTACH_REFUSAL = "Cannot access a chrome-extension:// URL of different extension";

for (const retirementFails of [false, true]) {
  test(`attach refusal from an approved HTTPS tab, cleanup ${retirementFails ? "unconfirmed" : "confirmed"}`, async () => {
    const worker = await loadWorker({
      surfaces: { [TOKEN]: surface() }, origins: approvedOnly, tabUrl: `${APPROVED}/docs`,
      attachError: CHROME_ATTACH_REFUSAL, retirementFails,
    });
    try {
      const result = await worker.request("screenshot", { tab: TOKEN });
      assert.equal(result.ok, false);
      assert.deepEqual(result.error, {
        code: "internal", message: "Chrome refused debugger attachment to this tab",
        data: { undrivable_tab: "different_extension", surface_retired: !retirementFails },
      });
      const { calls } = worker.chromeState;
      assert.equal(calls.attach, 1, "only the requested ordinary attach ran");
      assert.deepEqual(calls.cdp, [], "no debugger command, including an ownership probe, ran");
      assert.equal(calls.scripts, 0);
      assert.equal(calls.removals, 0);
      assert.equal(worker.frames.some((frame) => frame.event === "tab_closed"), false);
      assert.equal(Object.hasOwn(worker.chromeState.surfaces(), TOKEN), retirementFails);
      if (!retirementFails) {
        const stale = await worker.request("status", { tab: TOKEN });
        assert.equal(stale.error.code, "tab_closed");
        assert.equal(calls.attach, 1, "the retired token cannot drive another attach");
      }
    } finally { worker.chromeState.restore(); }
  });
}

for (const modern of [true, false]) {
  test(`resumed open keeps the first attach refusal receipt (${modern ? "modern" : "legacy request"})`, async () => {
    const proof = "a".repeat(40);
    const owner = {
      owner_proof: proof, requester: "session:synthetic", owner_generation: "g1", allocation_id: "a1",
    };
    const unrelatedNonce = "0".repeat(32);
    const unrelatedToken = `bridge:43:${unrelatedNonce}`;
    const unrelated = { ...surface(), tabId: 43, nonce: unrelatedNonce };
    const worker = await loadWorker({
      surfaces: { [TOKEN]: surface(), [unrelatedToken]: unrelated },
      origins: approvedOnly, tabUrl: `${APPROVED}/docs`, attachError: CHROME_ATTACH_REFUSAL,
      session: modern ? {
        ownerScopes: {
          [proof]: {
            session: owner.requester, generation: owner.owner_generation,
            allocations: { a1: { tab: TOKEN, state: "allocated" } },
          },
        },
      } : {},
    });
    try {
      const scoped = modern ? owner : {};
      const result = await worker.request("open", { ...scoped, tab: TOKEN, url: `${APPROVED}/next` });
      assert.equal(result.ok, false);
      const { tabsGet, ...sideEffects } = worker.chromeState.calls;
      assert.deepEqual(sideEffects, {
        scripts: 0, cdp: [], attach: 1, removals: 0, detach: 0, targets: 0,
        updates: [], creates: 0, windowUpdates: 0, frames: 0,
      });
      assert.deepEqual(result.error, {
        code: "internal", message: "Chrome refused debugger attachment to this tab",
        data: { undrivable_tab: "different_extension", surface_retired: true },
      });
      assert.deepEqual(worker.chromeState.surfaces(), { [unrelatedToken]: unrelated });
      assert.equal(worker.frames.some((frame) => frame.event === "tab_closed"), false);
      assert.equal((await worker.request("status", { ...scoped, tab: TOKEN })).error.code, "tab_closed");
      assert.equal(worker.chromeState.calls.attach, 1);
    } finally { worker.chromeState.restore(); }
  });
}

test("update and detach do not retire a handle; removal and replacement do", async () => {
  for (const lifecycle of ["removed", "replaced"]) {
    const worker = await loadWorker({ surfaces: { [TOKEN]: surface() }, origins: approvedOnly, tabUrl: `${APPROVED}/docs` });
    const { events, calls } = worker.chromeState;
    try {
      events.updated.fire(42, { status: "loading" }, { id: 42 });
      events.detached.fire({ tabId: 42 }, "target_closed");
      assert.equal((await worker.request("status", { tab: TOKEN })).ok, true);
      if (lifecycle === "removed") events.removed.fire(42, { windowId: 1 });
      else events.replaced.fire(43, 42);
      for (let i = 0; i < 400 && !worker.frames.some((f) => f.event === "tab_closed"); i++) await tick();
      assert.ok(worker.frames.some((f) => f.event === "tab_closed" && f.tab === TOKEN));
      assert.deepEqual(worker.chromeState.surfaces(), {}, "do not transfer ownership to a replacement");
      assert.equal((await worker.request("status", { tab: TOKEN })).error.code, "tab_closed");
      assert.equal(calls.attach, 0);
    } finally { worker.chromeState.restore(); }
  }
});

test("protected top-level page is refused before debugger attach", async () => {
  const worker = await loadWorker({
    surfaces: { [TOKEN]: surface() }, origins: approvedOnly, tabUrl: "chrome-extension://fixture/index.html",
  });
  try {
    const result = await worker.request("screenshot", { tab: TOKEN });
    assert.equal(result.error.code, "origin_denied");
    assert.equal(worker.chromeState.calls.attach, 0);
    assert.deepEqual(worker.chromeState.calls.cdp, []);
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

// Round 2 log-tag fixes (m1, m2, m3/Q4, Q3/n1). `logWorker` resumes a surface
// on `tabUrl` (which arms capture) and returns CDP event helpers.
async function logWorker(tabUrl, { runtimeAlreadyOn = false, replayOrigin = null } = {}) {
  const worker = await loadWorker({ surfaces: { [TOKEN]: surface() }, origins: approvedOnly, tabUrl });
  const fire = (method, params) => worker.chromeState.cdpEvents.fire({ tabId: 42 }, method, params);
  const context = (id, origin, frameId = "main", isDefault = true) => fire("Runtime.executionContextCreated", {
    context: { id, origin, auxData: { isDefault, frameId } },
  });
  const say = (id, text) => fire("Runtime.consoleAPICalled", {
    type: "log", executionContextId: id, args: [{ type: "string", value: text }],
  });
  // V8 semantics: Runtime.enable replays executionContextCreated only when
  // the domain was off; a repeat enable returns without replay.
  let runtimeOn = runtimeAlreadyOn;
  worker.chromeState.page.onCdp = (method) => {
    if (method === "Runtime.disable") runtimeOn = false;
    if (method === "Runtime.enable" && !runtimeOn) {
      runtimeOn = true;
      if (replayOrigin) context(1, replayOrigin);
    }
  };
  const resumed = await worker.request("open", { tab: TOKEN, url: tabUrl });
  assert.equal(resumed.ok, true, JSON.stringify(resumed.error));
  const texts = async () => {
    const response = await worker.request("logs", { tab: TOKEN });
    assert.equal(response.ok, true, JSON.stringify(response.error));
    return response.result.entries.map((entry) => entry.text);
  };
  return { worker, fire, context, say, texts };
}

test("Log.entryAdded is labelled by the MAIN frame's context, not an iframe's that came first (m1)", async () => {
  const { worker, fire, context, texts } = await logWorker(`${APPROVED}/docs`);
  try {
    // The page moves to the unapproved site; a script-less main document
    // embeds an approved-origin iframe whose context is created first.
    fire("Runtime.executionContextsCleared", {});
    context(5, APPROVED, "child-frame");
    fire("Log.entryAdded", { entry: { level: "error", text: "B-NETWORK-SECRET" } });
    context(6, OTHER, "main");
    assert.deepEqual(await texts(), [], "the unapproved page's browser messages are not returned");
  } finally { worker.chromeState.restore(); }
});

test("console output from a context the capture does not know is never returned (m2)", async () => {
  const { worker, fire, context, say, texts } = await logWorker(`${APPROVED}/docs`);
  try {
    fire("Runtime.executionContextsCleared", {});
    context(3, APPROVED);
    say(3, "A-OWN");
    say(99, "UNKNOWN-CONTEXT-SECRET");
    assert.deepEqual(await texts(), ["A-OWN"]);
  } finally { worker.chromeState.restore(); }
});

test("an iframe's console output on an approved page is returned, tagged with the top page (Q3/n1)", async () => {
  const { worker, fire, context, say, texts } = await logWorker(`${APPROVED}/docs`);
  try {
    fire("Runtime.executionContextsCleared", {});
    context(1, APPROVED, "main");
    context(2, OTHER, "widget-frame");
    say(1, "A-MAIN");
    say(2, "C-IFRAME");
    // The same iframe under an UNAPPROVED top page stays hidden with it.
    fire("Runtime.executionContextsCleared", {});
    context(3, OTHER, "main");
    context(4, APPROVED, "widget-frame");
    say(4, "IFRAME-UNDER-B-SECRET");
    assert.deepEqual(await texts(), ["A-MAIN", "C-IFRAME"]);
  } finally { worker.chromeState.restore(); }
});

test("re-arming capture on a session whose Runtime was already on replays contexts, so logs return (m3/Q4)", async () => {
  const { worker, say, texts } = await logWorker(`${APPROVED}/docs`, { runtimeAlreadyOn: true, replayOrigin: APPROVED });
  try {
    const cdp = worker.chromeState.calls.cdp;
    assert.ok(cdp.indexOf("Runtime.disable") > -1 && cdp.indexOf("Runtime.disable") < cdp.indexOf("Runtime.enable"),
      `Runtime.disable must precede Runtime.enable: ${cdp.join(",")}`);
    say(1, "A-AFTER-RESUME");
    assert.deepEqual(await texts(), ["A-AFTER-RESUME"]);
  } finally { worker.chromeState.restore(); }
});

test("an opaque or non-web origin is never recorded as an admission (Q5)", async () => {
  const chromeState = installChrome({ surfaces: { [TOKEN]: surface() }, tabUrl: "about:blank" });
  try {
    const statePath = join(bundleDir, "state.mjs");
    await build({ entryPoints: ["src/state.ts"], bundle: true, platform: "node", format: "esm", outfile: statePath });
    const { noteAdmittedOrigin } = await import(pathToFileURL(statePath) + `?q5=${++loads}`);
    for (const origin of ["null", "", "about:blank", "about:"]) await noteAdmittedOrigin(42, origin);
    assert.equal(chromeState.surfaces()[TOKEN].admittedOrigins, undefined);
    await noteAdmittedOrigin(42, OTHER);
    assert.deepEqual(chromeState.surfaces()[TOKEN].admittedOrigins, [OTHER], "a web origin is still recorded");
  } finally { chromeState.restore(); }
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

// Round 2 M1: an allow-once admission lasts for the surface, so a round trip
// off the site and back (SSO/OAuth, payment hand-off, link out + Back) never
// raises a second approval for the site the user just allowed.
const IDP = "https://idp.example";
for (const idpApproved of [true, false]) {
  test(`allow-once A -> ${idpApproved ? "approved" : "unapproved"} SSO provider -> back to A stays readable with no second request (M1)`, async () => {
    const worker = await loadWorker({
      surfaces: { [TOKEN]: surface() },
      origins: idpApproved ? { ...approvedOnly, [IDP]: "allow" } : approvedOnly,
      tabUrl: `${APPROVED}/docs`, session: onceFor(OTHER),
    });
    try {
      const moved = await worker.request("goto", { tab: TOKEN, url: `${OTHER}/app`, requester });
      assert.equal(moved.ok, true, JSON.stringify(moved.error));
      await settleTicks();
      const { page, nav } = worker.chromeState;
      // "Sign in with ..." takes the tab to the provider; it bounces back
      // after the click's interception window has closed.
      page.tabUrl = page.docUrl = `${IDP}/authorize?client=a`;
      nav.onCommitted.fire({ tabId: 42, frameId: 0, url: page.tabUrl });
      await settleTicks();
      if (idpApproved) {
        const atIdp = await worker.request("read", { tab: TOKEN, requester });
        assert.equal(atIdp.ok, true, JSON.stringify(atIdp.error));
      }
      page.tabUrl = page.docUrl = `${OTHER}/callback?code=x`;
      nav.onCommitted.fire({ tabId: 42, frameId: 0, url: page.tabUrl });
      await settleTicks();
      const read = await worker.request("read", { tab: TOKEN, requester });
      assert.equal(read.ok, true, `back on the allowed site must stay readable: ${JSON.stringify(read.error)}`);
      assert.equal(read.result.text, `${SECRET} @ ${OTHER}`);
      assert.deepEqual(worker.chromeState.surfaces()[TOKEN].admittedOrigins, [OTHER]);
      const session = worker.chromeState.session();
      assert.deepEqual(session.get("accessQueue") ?? [], [], "no second access request was raised");
      assert.equal(session.get("pendingOrigin"), undefined);
    } finally { worker.chromeState.restore(); }
  });
}

test("a goto on an allow-once grant that redirects elsewhere admits neither origin (M1)", async () => {
  const worker = await loadWorker({
    surfaces: { [TOKEN]: surface() }, origins: approvedOnly, tabUrl: `${APPROVED}/docs`, session: onceFor(OTHER),
  });
  // A load the interceptor never pauses, redirected server-side to C.
  worker.chromeState.page.intercept = false;
  worker.chromeState.page.redirectTo = `${IDP}/landing`;
  try {
    await worker.request("goto", { tab: TOKEN, url: `${OTHER}/landing`, requester });
    await settleTicks();
    assert.equal(worker.chromeState.surfaces()[TOKEN].admittedOrigins, undefined, "neither A nor C is admitted");
    const read = await worker.request("read", { tab: TOKEN, requester });
    assert.equal(read.ok, false, `the redirect target was never approved, got ${JSON.stringify(read.result)}`);
    assert.equal(read.error.code, "origin_not_allowed");
    assert.equal(worker.chromeState.calls.scripts, 0);
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
