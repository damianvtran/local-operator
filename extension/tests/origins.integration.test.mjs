/* Storage integration tests for the requester-bound approval queue. These run
 * the real approval-store/state modules against an honest in-memory chrome
 * storage shape; no browser profile or bridge is touched. */
import assert from "node:assert/strict";
import test from "node:test";
import { build } from "esbuild";
import { pathToFileURL } from "node:url";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

function installChromeStub(actionFailures = new Set(), notificationFailures = new Map()) {
  const areas = { session: new Map(), local: new Map() };
  const listeners = [];
  const alarmListeners = [];
  const alarms = new Map();
  const actionCalls = [];
  const notificationCalls = [];
  const makeArea = (name) => ({
    get: async (keys) => {
      const out = {};
      for (const key of Array.isArray(keys) ? keys : [keys]) {
        if (areas[name].has(key)) out[key] = areas[name].get(key);
      }
      return out;
    },
    set: async (obj) => {
      const changes = {};
      for (const [key, value] of Object.entries(obj)) {
        changes[key] = { oldValue: areas[name].get(key), newValue: value };
        areas[name].set(key, value);
      }
      for (const listener of listeners) listener(changes, name);
    },
    remove: async (keys) => {
      for (const key of Array.isArray(keys) ? keys : [keys]) areas[name].delete(key);
    },
  });
  globalThis.chrome = {
    storage: {
      session: makeArea("session"),
      local: makeArea("local"),
      onChanged: { addListener: (fn) => listeners.push(fn) },
    },
    alarms: {
      create: (name, info) => alarms.set(name, info),
      clear: async (name) => alarms.delete(name),
      onAlarm: { addListener: (listener) => alarmListeners.push(listener) },
    },
    action: Object.fromEntries(
      ["setBadgeBackgroundColor", "setBadgeTextColor", "setBadgeText", "setTitle"].map((method) => [
        method,
        async (params) => {
          actionCalls.push([method, params]);
          if (actionFailures.has(method)) throw new Error(`${method} rejected`);
        },
      ]),
    ),
    debugger: {
      onEvent: { addListener: () => {}, removeListener: () => {} },
      onDetach: { addListener: () => {} }, sendCommand: async () => ({}),
    },
    // The worker registers tab-lifecycle listeners at TOP LEVEL (MV3 requires
    // it: only listeners present during the first synchronous evaluation wake
    // a suspended worker), so they are wired up merely by importing it.
    tabs: {
      get: async () => ({}), remove: async () => {},
      onRemoved: { addListener: () => {} }, onReplaced: { addListener: () => {} },
    },
    notifications: {
      create: async (...args) => {
        notificationCalls.push(["create", args]);
        const remaining = notificationFailures.get("create") ?? 0;
        if (remaining > 0) {
          notificationFailures.set("create", remaining - 1);
          throw new Error("create rejected");
        }
      },
      clear: async (...args) => {
        notificationCalls.push(["clear", args]);
        const remaining = notificationFailures.get("clear") ?? 0;
        if (remaining > 0) {
          notificationFailures.set("clear", remaining - 1);
          throw new Error("clear rejected");
        }
      },
      onClicked: { addListener: () => {} },
    },
    runtime: {
      getURL: (path) => `chrome-extension://test/${path}`,
      getManifest: () => ({ version: "0.1.7" }),
      onStartup: { addListener: () => {} }, onInstalled: { addListener: () => {} },
      onMessage: { addListener: () => {} }, sendMessage: async () => {},
    },
  };
  return {
    session: (key) => areas.session.get(key),
    local: (key) => areas.local.get(key),
    setSession: (key, value) => areas.session.set(key, value),
    alarm: (name) => alarms.get(name),
    actionCalls,
    notificationCalls,
    fireAlarm: (name) => Promise.all(alarmListeners.map((listener) => listener({ name }))),
    restore: () => {
      delete globalThis.chrome;
      delete globalThis.WebSocket;
      delete globalThis.navigator;
    },
  };
}

async function loadModule(entry = "src/approval-store.ts") {
  const dir = await mkdtemp(join(tmpdir(), "lop-queue-it-"));
  const outfile = join(dir, "module.mjs");
  await build({ entryPoints: [entry], bundle: true, platform: "node", format: "esm", outfile });
  return {
    import: (tag = "") => import(pathToFileURL(outfile) + (tag ? `?${tag}` : "")),
    close: () => rm(dir, { recursive: true, force: true }),
  };
}

const loadStore = () => loadModule();
const url = (origin) => new URL(origin + "/page");

const tick = () => new Promise((resolve) => setTimeout(resolve, 0));
const lastAction = (chrome, method) => chrome.actionCalls.filter(([name]) => name === method).at(-1)?.[1];

test("queue mutations reconcile global numbered badge and exact tooltip", async () => {
  const chrome = installChromeStub();
  const bundle = await loadModule("src/origins.ts");
  try {
    const origins = await bundle.import();
    const first = await origins.raiseAccessRequest(url("https://one.example"), "session:one");
    await tick();
    assert.deepEqual(lastAction(chrome, "setBadgeText"), { text: "1" });
    assert.deepEqual(lastAction(chrome, "setTitle"), { title: "1 site request waiting" });
    const second = await origins.raiseAccessRequest(url("https://two.example"), "session:two");
    await tick();
    assert.deepEqual(lastAction(chrome, "setBadgeText"), { text: "2" });
    assert.deepEqual(lastAction(chrome, "setTitle"), { title: "2 site requests waiting" });
    await origins.resolveOrigin(first.origin, "deny", first.entryId);
    assert.deepEqual(lastAction(chrome, "setBadgeText"), { text: "1" });
    assert.deepEqual(lastAction(chrome, "setTitle"), { title: "1 site request waiting" });
    await origins.resolveOrigin(second.origin, "deny", second.entryId);
    assert.deepEqual(lastAction(chrome, "setBadgeText"), { text: "" });
    assert.deepEqual(lastAction(chrome, "setTitle"), { title: "Local Operator" });
    assert.ok(chrome.actionCalls.every(([, params]) => !("tabId" in params) && !("windowId" in params)));
  } finally { await bundle.close(); chrome.restore(); }
});

test("worker cold start restores persisted badge without unhandled rejection", async () => {
  const chrome = installChromeStub();
  const bundle = await loadModule("src/worker.ts");
  try {
    const now = Date.now();
    chrome.setSession("accessQueueVersion", 1);
    chrome.setSession("accessQueue", [
      { entryId: "a", origin: "https://a.example", displayAuthority: "a.example", requester: "A", kind: "async", requestedAt: now, expiresAt: now + 60_000, sequence: 1 },
      { entryId: "b", origin: "https://b.example", displayAuthority: "b.example", requester: "B", kind: "async", requestedAt: now, expiresAt: now + 60_000, sequence: 2 },
    ]);
    globalThis.navigator = { userAgent: "node-test" };
    globalThis.WebSocket = class {
      static OPEN = 1;
      readyState = 1;
      constructor() { queueMicrotask(() => this.onopen?.()); }
      send() {}
      close() {}
    };
    const unhandled = [];
    const onUnhandled = (reason) => unhandled.push(reason);
    process.on("unhandledRejection", onUnhandled);
    try {
      await bundle.import();
      for (let attempts = 0; attempts < 20 && lastAction(chrome, "setBadgeText")?.text !== "2"; attempts++) await tick();
      assert.deepEqual(lastAction(chrome, "setBadgeText"), { text: "2" });
      assert.deepEqual(lastAction(chrome, "setTitle"), { title: "2 site requests waiting" });
      assert.deepEqual(unhandled, []);
    } finally { process.off("unhandledRejection", onUnhandled); }
  } finally { await bundle.close(); chrome.restore(); }
});

for (const method of ["create", "clear"]) {
  test(`notification ${method} rejection is contained and identical reconciliation retries`, async () => {
    const failures = new Map([[method, 1]]);
    const chrome = installChromeStub(new Set(), failures);
    const bundle = await loadModule("src/worker.ts");
    const warnings = [];
    const originalWarn = console.warn;
    console.warn = (...args) => warnings.push(args);
    try {
      const now = Date.now();
      chrome.setSession("accessQueueVersion", 1);
      chrome.setSession("accessQueue", method === "create" ? [
        { entryId: "retry", origin: "https://retry.example", displayAuthority: "retry.example", requester: "R", kind: "async", requestedAt: now, expiresAt: now + 60_000, sequence: 1 },
      ] : []);
      globalThis.navigator = { userAgent: "node-test" };
      globalThis.WebSocket = class {
        static OPEN = 1;
        readyState = 1;
        constructor() { queueMicrotask(() => this.onopen?.()); }
        send() {}
        close() {}
      };
      const unhandled = [];
      const onUnhandled = (reason) => unhandled.push(reason);
      process.on("unhandledRejection", onUnhandled);
      try {
        await bundle.import(method);
        for (let attempts = 0; attempts < 20 && chrome.notificationCalls.length < 1; attempts++) await tick();
        // A queue sweep emits the identical snapshot. Failed notification work
        // must retry because its generation key was not committed.
        await chrome.fireAlarm("lop-access-expiry");
        for (let attempts = 0; attempts < 20 && chrome.notificationCalls.length < 2; attempts++) await tick();
        assert.equal(chrome.notificationCalls.filter(([name]) => name === method).length, 2);
        await chrome.fireAlarm("lop-access-expiry");
        await tick();
        assert.equal(chrome.notificationCalls.filter(([name]) => name === method).length, 2, "successful retry deduplicates");
        assert.deepEqual(unhandled, []);
        assert.ok(chrome.actionCalls.some(([name]) => name === "setBadgeText"));
        assert.ok(chrome.actionCalls.some(([name]) => name === "setTitle"));
        assert.ok(warnings.some((args) => args.some((value) => String(value).includes("pending observer"))));
      } finally { process.off("unhandledRejection", onUnhandled); }
    } finally {
      console.warn = originalWarn;
      await bundle.close(); chrome.restore();
    }
  });
}

test("cold-start restore reconciles a persisted two-entry queue", async () => {
  const chrome = installChromeStub();
  const bundle = await loadModule("src/origins.ts");
  try {
    const now = Date.now();
    chrome.setSession("accessQueueVersion", 1);
    chrome.setSession("accessQueue", [
      { entryId: "a", origin: "https://a.example", displayAuthority: "a.example", requester: "A", kind: "async", requestedAt: now, expiresAt: now + 60_000, sequence: 1 },
      { entryId: "b", origin: "https://b.example", displayAuthority: "b.example", requester: "B", kind: "async", requestedAt: now, expiresAt: now + 60_000, sequence: 2 },
    ]);
    const origins = await bundle.import();
    await origins.restoreAccessQueue();
    assert.deepEqual(lastAction(chrome, "setBadgeText"), { text: "2" });
    assert.deepEqual(lastAction(chrome, "setTitle"), { title: "2 site requests waiting" });
  } finally { await bundle.close(); chrome.restore(); }
});

for (const failedMethod of ["setBadgeBackgroundColor", "setBadgeText"]) {
  test(`${failedMethod} rejection does not abort later action surfaces`, async () => {
    const chrome = installChromeStub(new Set([failedMethod]));
    const bundle = await loadModule("src/origins.ts");
    const warnings = [];
    const originalWarn = console.warn;
    console.warn = (...args) => warnings.push(args);
    try {
      const origins = await bundle.import(failedMethod);
      let observed = 0;
      origins.setPendingObserver(() => { observed += 1; });
      await origins.raiseAccessRequest(url("https://failure.example"), "session:failure");
      assert.ok(chrome.actionCalls.some(([method]) => method === "setBadgeText"));
      assert.deepEqual(lastAction(chrome, "setTitle"), { title: "1 site request waiting" });
      assert.ok(observed > 0, "notification observer must still run");
      assert.ok(warnings.some((args) => args.some((value) => String(value).includes(failedMethod))));
    } finally {
      console.warn = originalWarn;
      await bundle.close(); chrome.restore();
    }
  });
}

test("concurrent enqueue preserves FIFO and separates same-origin requesters", async () => {
  const chrome = installChromeStub();
  const bundle = await loadStore();
  try {
    const store = await bundle.import();
    const [a, b, c] = await Promise.all([
      store.enqueueAccess(url("https://a.example"), "session:A", "async"),
      store.enqueueAccess(url("https://a.example"), "session:B", "async"),
      store.enqueueAccess(url("https://c.example"), "session:C", "async"),
    ]);
    assert.deepEqual([a.position, b.position, c.position], [1, 2, 3]);
    assert.deepEqual(chrome.session("accessQueue").map((entry) => entry.requester), ["session:A", "session:B", "session:C"]);
    const repeat = await store.enqueueAccess(url("https://a.example"), "session:A", "async");
    assert.equal(repeat.entryId, a.entryId);
    assert.equal(repeat.expires_at, a.expires_at, "dedup must not extend TTL");
  } finally { await bundle.close(); chrome.restore(); }
});

test("cancel is requester-bound and a stale generation cannot decide", async () => {
  const chrome = installChromeStub();
  const bundle = await loadStore();
  try {
    const store = await bundle.import();
    const a = await store.enqueueAccess(url("https://same.example"), "session:A", "async");
    const b = await store.enqueueAccess(url("https://same.example"), "session:B", "async");
    assert.equal((await store.cancelAccess("https://same.example", "session:A")).state, "cancelled");
    assert.equal((await store.accessStatus("https://same.example", "session:B")).state, "pending");
    assert.equal((await store.decideAccess(a.entryId, "once")).applied, false);
    assert.equal((await store.decideAccess(b.entryId, "deny")).applied, true);
    assert.equal((await store.decideAccess(b.entryId, "deny")).applied, false, "duplicate decision is idempotent");
  } finally { await bundle.close(); chrome.restore(); }
});

test("allow once grants only one requester while site reconciles exact-origin entries", async () => {
  const chrome = installChromeStub();
  const bundle = await loadStore();
  try {
    const store = await bundle.import();
    const a = await store.enqueueAccess(url("https://same.example"), "session:A", "async");
    await store.enqueueAccess(url("https://same.example"), "session:B", "async");
    await store.decideAccess(a.entryId, "once");
    assert.ok(chrome.session("onceGrants")["https://same.example\nsession:A"]);
    assert.equal(chrome.session("onceGrants")["https://same.example\nsession:B"], undefined);
    const a2 = await store.enqueueAccess(url("https://same.example"), "session:A2", "async");
    await store.decideAccess(a2.entryId, "site");
    assert.equal(chrome.local("origins")["https://same.example"], "allow");
    assert.equal(chrome.session("accessQueue").length, 0);
    assert.equal((await store.accessStatus("https://same.example", "session:B")).state, "allowed");
  } finally { await bundle.close(); chrome.restore(); }
});

test("loopback domain decision reconciles every port on both schemes and literal host", async () => {
  const chrome = installChromeStub();
  const bundle = await loadStore();
  try {
    const store = await bundle.import();
    await globalThis.chrome.storage.local.set({
      origins: {
        "http://localhost:3000": "allow",
        "http://localhost:5173": "allow",
        "https://localhost:5173": "allow",
        "http://127.0.0.1:5173": "allow",
        "http://localhost:9999": "deny",
      },
    });
    const selected = await store.enqueueAccess(url("http://localhost:3000"), "session:A", "async");
    await store.enqueueAccess(url("http://localhost:5173"), "session:B", "async");
    await store.enqueueAccess(url("https://localhost:5173"), "session:C", "async");
    await store.enqueueAccess(url("http://127.0.0.1:5173"), "session:D", "async");
    const decided = await store.decideAccess(selected.entryId, "domain");
    // Host scope covers every scheme and port on the literal loopback hostname.
    assert.deepEqual(decided.decided.map((entry) => entry.requester), ["session:A", "session:B", "session:C"]);
    assert.deepEqual(chrome.session("accessQueue").map((entry) => entry.requester), ["session:D"]);
    assert.equal((await store.accessStatus("http://localhost:5173", "session:B")).state, "allowed");
    assert.ok(chrome.local("siteGrants").grants["localhost"]);
    assert.equal(chrome.local("siteGrants").grants["localhost"].scope, "host");
    // Exact rows on the granted host are compacted away; other hosts keep theirs.
    assert.deepEqual(chrome.local("origins"), {
      "http://127.0.0.1:5173": "allow",
      "http://localhost:9999": "deny",
    });
    assert.equal(chrome.local("hostGrants"), undefined, "no legacy hostGrants are written");
  } finally { await bundle.close(); chrome.restore(); }
});

test("covered exact approval does not recreate a compacted Settings row", async () => {
  const chrome = installChromeStub();
  const bundle = await loadModule("src/access-grants.ts");
  try {
    const grants = await bundle.import();
    assert.equal(await grants.grantSite(url("http://localhost:3000")), true);
    assert.equal(await grants.grantExactOrigin("http://localhost:5173"), true);
    assert.deepEqual(chrome.local("origins"), {});
    assert.equal(Object.keys(chrome.local("siteGrants").grants).length, 1);
    // A second exact approval on a covered origin is a no-op.
    assert.equal(await grants.grantExactOrigin("http://localhost:3000"), true);
    assert.deepEqual(chrome.local("origins"), {});
  } finally { await bundle.close(); chrome.restore(); }
});

test("site persistent decision reconciles only the selected origin", async () => {
  const chrome = installChromeStub();
  const bundle = await loadStore();
  try {
    const store = await bundle.import();
    const selected = await store.enqueueAccess(url("http://localhost:3000"), "session:A", "async");
    await store.enqueueAccess(url("http://localhost:3000"), "session:B", "async");
    await store.enqueueAccess(url("http://localhost:5173"), "session:C", "async");
    await store.decideAccess(selected.entryId, "site");
    assert.deepEqual(chrome.session("accessQueue").map((entry) => entry.requester), ["session:C"]);
  } finally { await bundle.close(); chrome.restore(); }
});

test("persisted loopback host grant survives module restart and admits a second port", async () => {
  const chrome = installChromeStub();
  const bundle = await loadStore();
  const originsBundle = await loadModule("src/origins.ts");
  try {
    const store = await bundle.import();
    const selected = await store.enqueueAccess(url("http://localhost:3000"), "session:A", "async");
    await store.decideAccess(selected.entryId, "domain");
    const origins = await originsBundle.import("restart");
    assert.equal(await origins.originAllowed(url("http://localhost:5173")), true);
    assert.equal(await origins.originAllowed(url("https://localhost:8443")), true);
  } finally {
    await bundle.close();
    await originsBundle.close();
    chrome.restore();
  }
});

test("a stored key the current PSL cannot re-derive stays revocable and does not block writes", async () => {
  const chrome = installChromeStub();
  const bundle = await loadModule("src/access-grants.ts");
  try {
    const grants = await bundle.import();
    // blogspot.com is in the PRIVATE section today, so registrableDomain()
    // returns null for it: a grant written before that entry existed has a key
    // this build can no longer derive. Re-deriving it during a mutation
    // refused the WHOLE record, so every later grant failed and the Remove
    // button for this row reported "try again" forever, while the read path
    // kept honouring the other entries (A2).
    await globalThis.chrome.storage.local.set({
      siteGrants: { version: 1, grants: {
        "blogspot.com": { scope: "domain", createdAt: 1 },
        "gominerva.com": { scope: "domain", createdAt: 1 },
      } },
    });
    assert.equal(await grants.revokeSiteGrant("blogspot.com"), true, "the stale row must be removable");
    assert.equal(await grants.revokeSiteGrant("gominerva.com"), true, "an unrelated row must be removable");
    assert.equal(await grants.grantSite(url("https://app.example.com")), true, "later grants must still write");
    assert.deepEqual(Object.keys(chrome.local("siteGrants").grants), ["example.com"]);
    // A5: removing a key that is not there is not a success.
    assert.equal(await grants.revokeSiteGrant("never-granted.example"), false);
  } finally { await bundle.close(); chrome.restore(); }
});

test("an unreadable stored record still refuses the mutation, and clearing recovers", async () => {
  const chrome = installChromeStub();
  const bundle = await loadModule("src/access-grants.ts");
  try {
    const grants = await bundle.import();
    // Structurally malformed (no createdAt) is different from "key I cannot
    // re-derive": a record whose SHAPE is unreadable is one we cannot safely
    // rewrite, so it still refuses wholesale.
    await globalThis.chrome.storage.local.set({
      siteGrants: { version: 1, grants: { "gominerva.com": { scope: "domain" } } },
    });
    assert.equal(await grants.grantSite(url("https://app.example.com")), false);
    assert.equal(await grants.clearAllAccessGrants(), true);
    assert.equal(chrome.local("siteGrants"), undefined);
    assert.equal(await grants.grantSite(url("https://app.example.com")), true);
  } finally { await bundle.close(); chrome.restore(); }
});

test("revoking a loopback host grant makes a later port prompt again", async () => {
  const chrome = installChromeStub();
  const storeBundle = await loadStore();
  const originsBundle = await loadModule("src/origins.ts");
  const grantsBundle = await loadModule("src/access-grants.ts");
  try {
    const store = await storeBundle.import();
    const selected = await store.enqueueAccess(url("http://localhost:3000"), "session:A", "async");
    await store.decideAccess(selected.entryId, "domain");
    const grants = await grantsBundle.import();
    assert.equal(await grants.revokeSiteGrant("localhost"), true);
    const origins = await originsBundle.import("after-revoke");
    assert.equal(await origins.originAllowed(url("http://localhost:5173")), false);
    const queued = await store.enqueueAccess(url("http://localhost:5173"), "session:B", "async");
    assert.equal(queued.state, "pending");
  } finally {
    await storeBundle.close();
    await originsBundle.close();
    await grantsBundle.close();
    chrome.restore();
  }
});

test("decision between enqueue publication and return cannot miss the waiter", async () => {
  const chrome = installChromeStub();
  const bundle = await loadModule("src/origins.ts");
  try {
    const origins = await bundle.import();
    origins.setPendingObserver((snapshot) => {
      const entry = snapshot.queue[0];
      if (entry) void origins.resolveOrigin(entry.origin, "once", entry.entryId);
    });
    assert.equal(await origins.askOrigin(url("https://fast.example"), "cmd-fast"), true);
  } finally { await bundle.close(); chrome.restore(); }
});

test("same-command in-command admissions dedupe onto one generation", async () => {
  const chrome = installChromeStub();
  const bundle = await loadModule("src/origins.ts");
  try {
    const origins = await bundle.import();
    const first = origins.askOrigin(url("https://same-hop.example"), "same-command");
    const second = origins.askOrigin(url("https://same-hop.example"), "same-command");
    // A retried same-command navigation re-asks the identical question, so it
    // dedupes onto the ONE generation the user is looking at instead of
    // stacking a twin. One decision therefore settles every paused document
    // attached to that command.
    while ((chrome.session("accessQueue")?.length ?? 0) !== 1) await new Promise((resolve) => setTimeout(resolve, 0));
    const [entry] = chrome.session("accessQueue");
    await origins.resolveOrigin(entry.origin, "deny", entry.entryId);
    assert.deepEqual(await Promise.all([first, second]), [false, false]);
    assert.equal(chrome.session("accessQueue").length, 0, "the deduped generation is decided");
  } finally { await bundle.close(); chrome.restore(); }
});

test("site allow reconciles every identical in-command waiter", async () => {
  const chrome = installChromeStub();
  const bundle = await loadModule("src/origins.ts");
  try {
    const origins = await bundle.import();
    const first = origins.askOrigin(url("https://fleet-hop.example"), "same-command");
    const second = origins.askOrigin(url("https://fleet-hop.example"), "same-command");
    while ((chrome.session("accessQueue")?.length ?? 0) !== 1) await new Promise((resolve) => setTimeout(resolve, 0));
    const selected = chrome.session("accessQueue")[0];
    await origins.resolveOrigin(selected.origin, "site", selected.entryId);
    assert.deepEqual(await Promise.all([first, second]), [true, true]);
    assert.equal(chrome.session("accessQueue").length, 0);
  } finally { await bundle.close(); chrome.restore(); }
});

test("TTL sweep settles every identical in-command waiter", async () => {
  const chrome = installChromeStub();
  const bundle = await loadModule("src/origins.ts");
  try {
    const origins = await bundle.import();
    const first = origins.askOrigin(url("https://expire-both.example"), "same-command");
    const second = origins.askOrigin(url("https://expire-both.example"), "same-command");
    while ((chrome.session("accessQueue")?.length ?? 0) !== 1) await new Promise((resolve) => setTimeout(resolve, 0));
    const expiresAt = Math.max(...chrome.session("accessQueue").map((entry) => entry.expiresAt));
    await origins.sweepQueue(expiresAt);
    assert.deepEqual(await Promise.all([first, second]), [false, false]);
  } finally { await bundle.close(); chrome.restore(); }
});

test("TTL sweep resolves an in-command waiter as denied without popup action", async () => {
  const chrome = installChromeStub();
  const bundle = await loadModule("src/origins.ts");
  try {
    const origins = await bundle.import();
    const pending = origins.askOrigin(url("https://expire.example"), "cmd-expire");
    while (!chrome.session("accessQueue")?.length) await new Promise((resolve) => setTimeout(resolve, 0));
    const expiresAt = chrome.session("accessQueue")[0].expiresAt;
    await origins.sweepQueue(expiresAt);
    assert.equal(await pending, false);
  } finally { await bundle.close(); chrome.restore(); }
});

test("migration preserves distinct async and in-command legacy records", async () => {
  const chrome = installChromeStub();
  const bundle = await loadStore();
  try {
    const store = await bundle.import();
    const now = Date.now();
    chrome.setSession("accessRequest", {
      origin: "https://a.example", hostname: "a.example", requester: "session:A",
      requestedAt: now, expiresAt: now + 600_000,
    });
    chrome.setSession("pendingOrigin", {
      origin: "https://b.example", hostname: "b.example", requestId: "cmd-B", promptId: "generation-B",
    });
    await store.sweepQueue(now);
    const queue = chrome.session("accessQueue");
    assert.deepEqual(queue.map((entry) => [entry.origin, entry.kind, entry.entryId]), [
      ["https://a.example", "async", queue[0].entryId],
      ["https://b.example", "in_command", "generation-B"],
    ]);
    await store.sweepQueue(now + 1);
    assert.equal(chrome.session("accessQueue").length, 2, "migration is idempotent");
  } finally { await bundle.close(); chrome.restore(); }
});

test("migration handles each equal, request-only, and pending-only combination", async () => {
  for (const mode of ["equal", "request-only", "pending-only"]) {
    const chrome = installChromeStub();
    const bundle = await loadStore();
    try {
      const store = await bundle.import(mode);
      const now = Date.now();
      if (mode !== "pending-only") chrome.setSession("accessRequest", {
        origin: "https://legacy.example", authority: "legacy.example", requester: "session:L",
        requestedAt: now, expiresAt: now + 60_000,
      });
      if (mode !== "request-only") chrome.setSession("pendingOrigin", {
        origin: "https://legacy.example", authority: "legacy.example", requestId: "session:L", promptId: "legacy-generation",
      });
      await store.sweepQueue(now);
      const queue = chrome.session("accessQueue");
      assert.equal(queue.length, 1, mode);
      assert.equal(queue[0].kind, mode === "pending-only" ? "in_command" : "async");
      if (mode === "equal") assert.equal(queue[0].entryId, "legacy-generation");
    } finally { await bundle.close(); chrome.restore(); }
  }
});

test("earliest expiry alarm survives later async enqueue and advances after sweep", async () => {
  const chrome = installChromeStub();
  const bundle = await loadStore();
  try {
    const store = await bundle.import();
    const originalNow = Date.now;
    let now = 1_000_000;
    Date.now = () => now;
    try {
      const short = await store.enqueueAccess(url("https://short.example"), "cmd-short", "in_command", "cmd-short");
      assert.equal(chrome.alarm(store.ACCESS_EXPIRY_ALARM).when, short.expires_at);
      now += 1_000;
      const long = await store.enqueueAccess(url("https://long.example"), "session:long", "async");
      assert.equal(chrome.alarm(store.ACCESS_EXPIRY_ALARM).when, short.expires_at, "later enqueue cannot delay earlier expiry");
      await store.sweepQueue(short.expires_at);
      assert.equal(chrome.alarm(store.ACCESS_EXPIRY_ALARM).when, long.expires_at);
    } finally { Date.now = originalNow; }
  } finally { await bundle.close(); chrome.restore(); }
});

test("queue cap rejects without loss and migration imports legacy generation once", async () => {
  const chrome = installChromeStub();
  const bundle = await loadStore();
  try {
    const store = await bundle.import();
    const now = Date.now();
    chrome.setSession("accessRequest", {
      origin: "https://legacy.example", authority: "legacy.example", requester: "session:L",
      requestedAt: now, expiresAt: now + 60_000,
    });
    chrome.setSession("pendingOrigin", {
      origin: "https://legacy.example", authority: "legacy.example", requestId: "session:L", promptId: "legacy-generation",
    });
    await store.sweepQueue(now);
    await store.sweepQueue(now + 1);
    assert.equal(chrome.session("accessQueue").length, 1);
    assert.equal(chrome.session("accessQueue")[0].entryId, "legacy-generation");
    for (let index = 1; index < 16; index++) {
      await store.enqueueAccess(url(`https://${index}.example`), `session:${index}`, "async");
    }
    const overflow = await store.enqueueAccess(url("https://overflow.example"), "session:overflow", "async");
    assert.equal(overflow.full, true);
    assert.equal(overflow.pending_count, 16);
    assert.equal(chrome.session("accessQueue").length, 16);
  } finally { await bundle.close(); chrome.restore(); }
});

test("origin-fallback decision applies to a single live entry and rejects ambiguity", async () => {
  const chrome = installChromeStub();
  const bundle = await loadModule("src/origins.ts");
  try {
    const origins = await bundle.import();
    const now = Date.now();
    chrome.setSession("accessQueueVersion", 1);
    // ONE live entry for the origin: an empty-generation (health-fallback)
    // decision has no id to aim at, but the user is looking at exactly this
    // origin, so the click must land on it.
    chrome.setSession("accessQueue", [
      { entryId: "solo", origin: "https://solo.example", displayAuthority: "solo.example", requester: "cmd-solo", kind: "in_command", requestedAt: now, expiresAt: now + 60_000, sequence: 1, commandId: "cmd-solo" },
    ]);
    assert.equal(await origins.resolveOrigin("https://solo.example", "once", ""), true, "single match applies");
    assert.equal(chrome.session("accessQueue").length, 0, "single entry was decided");

    // TWO live entries share the origin: the fallback must not guess which
    // paused navigation the user meant (B1 stays closed).
    chrome.setSession("accessQueue", [
      { entryId: "twin-a", origin: "https://twin.example", displayAuthority: "twin.example", requester: "cmd-a", kind: "in_command", requestedAt: now, expiresAt: now + 60_000, sequence: 1, commandId: "cmd-a" },
      { entryId: "twin-b", origin: "https://twin.example", displayAuthority: "twin.example", requester: "cmd-b", kind: "in_command", requestedAt: now, expiresAt: now + 60_000, sequence: 2, commandId: "cmd-b" },
    ]);
    assert.equal(await origins.resolveOrigin("https://twin.example", "once", ""), false, "multi-match rejects");
    assert.equal(chrome.session("accessQueue").length, 2, "both twin entries survive the rejection");

    // No live entry for the origin: nothing to apply the decision to.
    assert.equal(await origins.resolveOrigin("https://absent.example", "once", ""), false, "zero-match rejects");
  } finally { await bundle.close(); chrome.restore(); }
});

test("in-command enqueue dedupes a retried same-command navigation but keeps distinct commands", async () => {
  const chrome = installChromeStub();
  const bundle = await loadStore();
  try {
    const store = await bundle.import();
    const first = await store.enqueueAccess(url("https://dup.example"), "cmd-1", "in_command", "cmd-1");
    const retry = await store.enqueueAccess(url("https://dup.example"), "cmd-1", "in_command", "cmd-1");
    assert.equal(retry.entryId, first.entryId, "same command dedupes onto one generation");
    assert.equal(chrome.session("accessQueue").length, 1, "no twin stacks for one command");
    const other = await store.enqueueAccess(url("https://dup.example"), "cmd-1", "in_command", "cmd-2");
    assert.notEqual(other.entryId, first.entryId, "a different command keeps its own generation");
    assert.equal(chrome.session("accessQueue").length, 2);
  } finally { await bundle.close(); chrome.restore(); }
});

test("a deduped same-command generation resolves every attached waiter", async () => {
  const chrome = installChromeStub();
  const bundle = await loadModule("src/origins.ts");
  try {
    const origins = await bundle.import();
    // Two paused documents under the SAME command share one generation after
    // dedupe; both registered resolvers must be released by one decision, or
    // one navigation would wait on a promise nothing can resolve.
    const p1 = origins.askOrigin(url("https://wait.example"), "cmd-w");
    const p2 = origins.askOrigin(url("https://wait.example"), "cmd-w");
    for (let i = 0; i < 40 && (chrome.session("accessQueue")?.length ?? 0) !== 1; i++) await tick();
    await tick();
    assert.equal(chrome.session("accessQueue").length, 1, "twins dedupe onto one generation");
    const entryId = chrome.session("accessQueue")[0].entryId;
    assert.equal(await origins.resolveOrigin("https://wait.example", "once", entryId), true);
    const [r1, r2] = await Promise.all([p1, p2]);
    assert.equal(r1, true, "first paused document resumes");
    assert.equal(r2, true, "deduped twin also resumes");
  } finally { await bundle.close(); chrome.restore(); }
});

test("worker observer announces an origin block and clears it when the entry leaves the queue", async () => {
  const chrome = installChromeStub();
  const frames = [];
  globalThis.navigator = { userAgent: "node-test" };
  globalThis.WebSocket = class {
    static OPEN = 1;
    readyState = 1;
    constructor() { queueMicrotask(() => this.onopen?.()); }
    send(data) { frames.push(JSON.parse(String(data))); }
    close() {}
  };
  const bundle = await loadModule("src/worker.ts");
  const originalNow = Date.now;
  let now = Date.now();
  try {
    chrome.setSession("accessQueueVersion", 1);
    chrome.setSession("accessQueue", [
      { entryId: "e1", origin: "https://x.example", displayAuthority: "x.example", requester: "cmd-1", kind: "in_command", requestedAt: now, expiresAt: now + 60_000, sequence: 1, commandId: "cmd-1" },
    ]);
    await bundle.import();
    // Wait for the socket to open (hello frame) so observer sends are captured.
    for (let i = 0; i < 40 && !frames.some((f) => f.event === "hello"); i++) await tick();
    assert.ok(frames.some((f) => f.event === "hello"), "worker connected");
    // Re-sweep with the entry still live: the announcement repeats, and no
    // clearance is emitted for an entry that is still in the queue.
    await chrome.fireAlarm("lop-access-expiry");
    for (let i = 0; i < 40 && !frames.some((f) => f.event === "awaiting_origin"); i++) await tick();
    assert.ok(frames.some((f) => f.event === "awaiting_origin" && f.id === "cmd-1"), "block announced");
    assert.ok(!frames.some((f) => f.event === "awaiting_origin_cleared"), "no clearance while the entry is live");
    // Expire the entry and sweep: the observer must announce the clearance so
    // the daemon stops echoing a prompt the popup can no longer resolve.
    now += 120_000;
    Date.now = () => now;
    await chrome.fireAlarm("lop-access-expiry");
    for (let i = 0; i < 40 && !frames.some((f) => f.event === "awaiting_origin_cleared"); i++) await tick();
    assert.ok(frames.some((f) => f.event === "awaiting_origin_cleared" && f.id === "cmd-1"), "clearance announced");
  } finally {
    Date.now = originalNow;
    await bundle.close(); chrome.restore();
  }
});
