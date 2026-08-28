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

function installChromeStub() {
  const areas = { session: new Map(), local: new Map() };
  const listeners = [];
  const alarms = new Map();
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
    },
    action: {
      setBadgeBackgroundColor: async () => {}, setBadgeText: async () => {}, setTitle: async () => {},
    },
    debugger: {
      onEvent: { addListener: () => {}, removeListener: () => {} },
      onDetach: { addListener: () => {} }, sendCommand: async () => ({}),
    },
  };
  return {
    session: (key) => areas.session.get(key),
    local: (key) => areas.local.get(key),
    setSession: (key, value) => areas.session.set(key, value),
    alarm: (name) => alarms.get(name),
    restore: () => { delete globalThis.chrome; },
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

test("allow once grants only one requester while always reconciles exact-origin entries", async () => {
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
    await store.decideAccess(a2.entryId, "always");
    assert.equal(chrome.local("origins")["https://same.example"], "allow");
    assert.equal(chrome.session("accessQueue").length, 0);
    assert.equal((await store.accessStatus("https://same.example", "session:B")).state, "allowed");
  } finally { await bundle.close(); chrome.restore(); }
});

test("loopback all-port decision reconciles only same scheme and literal host", async () => {
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
    const decided = await store.decideAccess(selected.entryId, "all_ports");
    assert.deepEqual(decided.decided.map((entry) => entry.requester), ["session:A", "session:B"]);
    assert.deepEqual(chrome.session("accessQueue").map((entry) => entry.requester), ["session:C", "session:D"]);
    assert.equal((await store.accessStatus("http://localhost:5173", "session:B")).state, "allowed");
    assert.ok(chrome.local("hostGrants").grants['["http:","localhost"]']);
    assert.deepEqual(chrome.local("origins"), {
      "https://localhost:5173": "allow",
      "http://127.0.0.1:5173": "allow",
      "http://localhost:9999": "deny",
    });
  } finally { await bundle.close(); chrome.restore(); }
});

test("covered exact approval does not recreate a compacted Settings row", async () => {
  const chrome = installChromeStub();
  const bundle = await loadModule("src/access-grants.ts");
  try {
    const grants = await bundle.import();
    assert.equal(await grants.grantLoopbackHost(url("http://localhost:3000")), true);
    assert.equal(await grants.grantExactOrigin("http://localhost:5173"), true);
    assert.deepEqual(chrome.local("origins"), {});
    assert.equal(Object.keys(chrome.local("hostGrants").grants).length, 1);
  } finally { await bundle.close(); chrome.restore(); }
});

test("exact persistent decision reconciles only the selected origin", async () => {
  const chrome = installChromeStub();
  const bundle = await loadStore();
  try {
    const store = await bundle.import();
    const selected = await store.enqueueAccess(url("http://localhost:3000"), "session:A", "async");
    await store.enqueueAccess(url("http://localhost:3000"), "session:B", "async");
    await store.enqueueAccess(url("http://localhost:5173"), "session:C", "async");
    await store.decideAccess(selected.entryId, "always");
    assert.deepEqual(chrome.session("accessQueue").map((entry) => entry.requester), ["session:C"]);
  } finally { await bundle.close(); chrome.restore(); }
});

test("persisted loopback grant survives module restart and admits a second port", async () => {
  const chrome = installChromeStub();
  const bundle = await loadStore();
  const originsBundle = await loadModule("src/origins.ts");
  try {
    const store = await bundle.import();
    const selected = await store.enqueueAccess(url("http://localhost:3000"), "session:A", "async");
    await store.decideAccess(selected.entryId, "all_ports");
    const origins = await originsBundle.import("restart");
    assert.equal(await origins.originAllowed(url("http://localhost:5173")), true);
  } finally {
    await bundle.close();
    await originsBundle.close();
    chrome.restore();
  }
});

test("revoking a loopback all-port grant makes a later port prompt again", async () => {
  const chrome = installChromeStub();
  const storeBundle = await loadStore();
  const originsBundle = await loadModule("src/origins.ts");
  const grantsBundle = await loadModule("src/access-grants.ts");
  try {
    const store = await storeBundle.import();
    const selected = await store.enqueueAccess(url("http://localhost:3000"), "session:A", "async");
    await store.decideAccess(selected.entryId, "all_ports");
    const grants = await grantsBundle.import();
    assert.equal(await grants.revokeLoopbackHost('["http:","localhost"]'), true);
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

test("identical in-command admissions are independently visible and decidable", async () => {
  const chrome = installChromeStub();
  const bundle = await loadModule("src/origins.ts");
  try {
    const origins = await bundle.import();
    const first = origins.askOrigin(url("https://same-hop.example"), "same-command");
    const second = origins.askOrigin(url("https://same-hop.example"), "same-command");
    while (chrome.session("accessQueue")?.length !== 2) await new Promise((resolve) => setTimeout(resolve, 0));
    const [a, b] = chrome.session("accessQueue");
    assert.notEqual(a.entryId, b.entryId);
    await origins.resolveOrigin(a.origin, "deny", a.entryId);
    assert.equal(await first, false);
    assert.equal(chrome.session("accessQueue").length, 1, "deny removes only the selected navigation");
    await origins.resolveOrigin(b.origin, "once", b.entryId);
    assert.equal(await second, true);
    assert.equal(chrome.session("onceGrants")?.["https://same-hop.example\nsame-command"], undefined);
  } finally { await bundle.close(); chrome.restore(); }
});

test("always allow reconciles every identical in-command waiter", async () => {
  const chrome = installChromeStub();
  const bundle = await loadModule("src/origins.ts");
  try {
    const origins = await bundle.import();
    const first = origins.askOrigin(url("https://fleet-hop.example"), "same-command");
    const second = origins.askOrigin(url("https://fleet-hop.example"), "same-command");
    while (chrome.session("accessQueue")?.length !== 2) await new Promise((resolve) => setTimeout(resolve, 0));
    const selected = chrome.session("accessQueue")[0];
    await origins.resolveOrigin(selected.origin, "always", selected.entryId);
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
    while (chrome.session("accessQueue")?.length !== 2) await new Promise((resolve) => setTimeout(resolve, 0));
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
