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
  };
  return {
    session: (key) => areas.session.get(key),
    local: (key) => areas.local.get(key),
    setSession: (key, value) => areas.session.set(key, value),
    restore: () => { delete globalThis.chrome; },
  };
}

async function loadStore() {
  const dir = await mkdtemp(join(tmpdir(), "lop-queue-it-"));
  const outfile = join(dir, "module.mjs");
  await build({ entryPoints: ["src/approval-store.ts"], bundle: true, platform: "node", format: "esm", outfile });
  return {
    import: (tag = "") => import(pathToFileURL(outfile) + (tag ? `?${tag}` : "")),
    close: () => rm(dir, { recursive: true, force: true }),
  };
}

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

test("queue cap rejects without loss and migration imports legacy generation once", async () => {
  const chrome = installChromeStub();
  const bundle = await loadStore();
  try {
    const store = await bundle.import();
    const now = Date.now();
    chrome.setSession("accessRequest", {
      origin: "https://legacy.example", hostname: "legacy.example", requester: "session:L",
      requestedAt: now, expiresAt: now + 60_000,
    });
    chrome.setSession("pendingOrigin", {
      origin: "https://legacy.example", hostname: "legacy.example", requestId: "session:L", promptId: "legacy-generation",
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
