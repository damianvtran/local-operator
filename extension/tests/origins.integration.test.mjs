/* Integration tests at the origins.ts seam: the dual-resolve decision path
 * (popup → resolveOrigin → BOTH the in-command `waiting` map and the async
 * record), exactly-once requester-bound grant consumption, deny persistence
 * across a simulated worker restart, and supersession tombstones — driven
 * against the REAL origins.ts/state.ts/access-flow.ts modules with a stubbed
 * chrome.* (round-1 M2: the pure helpers were tested, but nothing exercised
 * their composition; the Python tool tests mock the wire, so a regression
 * here would ship with all suites green).
 *
 * The chrome stub is deliberately minimal and HONEST: session/local storage
 * are in-memory Maps with the real API's get/set/remove shape, so a "worker
 * restart" re-import evaluates module-level state fresh against the SAME
 * storage — MV3 kills the worker, not the storage, which is the seam's most
 * failure-prone property. */

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
      const changes = {};
      for (const key of Array.isArray(keys) ? keys : [keys]) {
        changes[key] = { oldValue: areas[name].get(key), newValue: undefined };
        areas[name].delete(key);
      }
      for (const listener of listeners) listener(changes, name);
    },
  });
  globalThis.chrome = {
    storage: {
      session: makeArea("session"),
      local: makeArea("local"),
      onChanged: { addListener: (fn) => listeners.push(fn) },
    },
    action: {
      setBadgeBackgroundColor: async () => {},
      setBadgeText: async () => {},
      setTitle: async () => {},
    },
    alarms: { create: () => {} },
    debugger: {
      onEvent: { addListener: () => {}, removeListener: () => {} },
      // cdp.ts registers an onDetach listener at MODULE level to drop its
      // attached-tab set when the user detaches the debugger; the stub needs
      // it for the import to evaluate at all.
      onDetach: { addListener: () => {} },
      sendCommand: async () => ({}),
    },
  };
  const session = (key) => areas.session.get(key);
  return { session, areas, restore: () => { delete globalThis.chrome; } };
}

async function loadOrigins() {
  const dir = await mkdtemp(join(tmpdir(), "lop-origins-it-"));
  const outfile = join(dir, "module.mjs");
  await build({
    entryPoints: ["src/origins.ts"],
    bundle: true,
    platform: "node",
    format: "esm",
    outfile,
  });
  return {
    import: (tag = "") => import(pathToFileURL(outfile) + (tag ? `?${tag}` : "")),
    close: () => rm(dir, { recursive: true, force: true }),
  };
}

const flush = () => new Promise((resolve) => setTimeout(resolve, 0));
const ORIGIN = "https://example.com";
const urlOf = (href = ORIGIN + "/page") => new URL(href);

test("one popup decision resolves the in-command wait AND the async record", async () => {
  const chrome = installChromeStub();
  const bundle = await loadOrigins();
  try {
    const origins = await bundle.import();
    await origins.raiseAccessRequest(urlOf(), "req-A");
    // An in-command redirect hop for the SAME origin is parked on `waiting`.
    let inCommand = null;
    const hop = origins.askOrigin(urlOf(), "cmd-9").then((ok) => { inCommand = ok; });
    await flush();
    origins.resolveOrigin(ORIGIN, "once");
    await hop;
    assert.equal(inCommand, true, "the in-command wait must resolve from the same click");
    await flush();
    const record = chrome.session("accessRequest");
    assert.equal(record.decision, "once");
    assert.equal(record.requester, "req-A");
    const grants = chrome.session("onceGrants");
    assert.equal(grants[ORIGIN].requester, "req-A", "grant bound to the asking requester");
  } finally {
    await bundle.close();
    chrome.restore();
  }
});

test("a once-grant is consumed exactly once, and only by its requester", async () => {
  const chrome = installChromeStub();
  const bundle = await loadOrigins();
  try {
    const origins = await bundle.import();
    await origins.raiseAccessRequest(urlOf(), "req-A");
    origins.resolveOrigin(ORIGIN, "once");
    await flush();
    // Another session cannot spend it (fail-closed, not fail-shared).
    assert.equal(await origins.consumeOnceGrant(urlOf(), "req-B"), false);
    // An anonymous caller cannot spend it either.
    assert.equal(await origins.consumeOnceGrant(urlOf(), ""), false);
    // Refusals did not consume: the owner still can.
    assert.equal(await origins.consumeOnceGrant(urlOf(), "req-A"), true);
    // Exactly once.
    assert.equal(await origins.consumeOnceGrant(urlOf(), "req-A"), false);
  } finally {
    await bundle.close();
    chrome.restore();
  }
});

test("admission consumes the grant once and navigate never re-consults it (M1)", async () => {
  const chrome = installChromeStub();
  const bundle = await loadOrigins();
  try {
    const origins = await bundle.import();
    await origins.raiseAccessRequest(urlOf(), "session:A");
    origins.resolveOrigin(ORIGIN, "once");
    await flush();
    // Session A's own navigation (session identity matches the grant).
    const admission = await origins.ensureTopLevelAccess(urlOf(), "cmd-nav-1", "session:A");
    assert.deepEqual(admission, { allowed: true, viaOnceGrant: true });
    // The grant is spent AT ADMISSION: a second admission for the same
    // session must now take the early-fail path, not find a stale grant.
    await assert.rejects(
      origins.ensureTopLevelAccess(urlOf(), "cmd-nav-2", "session:A"),
      (error) => error.code === "origin_not_allowed",
    );
  } finally {
    await bundle.close();
    chrome.restore();
  }
});

test("a parallel session's navigation cannot spend the grant (B1a)", async () => {
  const chrome = installChromeStub();
  const bundle = await loadOrigins();
  try {
    const origins = await bundle.import();
    await origins.raiseAccessRequest(urlOf(), "session:A");
    origins.resolveOrigin(ORIGIN, "once");
    await flush();
    // Session B's open carries B's identity and its own command id: neither
    // the grant's requester nor its handoff — refused, and the grant is NOT
    // consumed by the attempt (fail-closed, not fail-shared).
    await assert.rejects(
      origins.ensureTopLevelAccess(urlOf(), "cmd-B-1", "session:B"),
      (error) => error.code === "origin_not_allowed",
    );
    const admission = await origins.ensureTopLevelAccess(urlOf(), "cmd-nav-1", "session:A");
    assert.equal(admission.allowed, true, "A's grant survives B's refused attempt");
  } finally {
    await bundle.close();
    chrome.restore();
  }
});

test("a raw-RPC caller spends its grant via the command-id handoff", async () => {
  const chrome = installChromeStub();
  const bundle = await loadOrigins();
  try {
    const origins = await bundle.import();
    // No session identity (raw caller): requester IS the command id.
    await origins.raiseAccessRequest(urlOf(), "r-abc123");
    origins.resolveOrigin(ORIGIN, "once");
    await flush();
    // The SAME command id navigating (the handoff) is admitted; a different
    // id with no session identity is not.
    const admission = await origins.ensureTopLevelAccess(urlOf(), "r-abc123");
    assert.deepEqual(admission, { allowed: true, viaOnceGrant: true });
  } finally {
    await bundle.close();
    chrome.restore();
  }
});

test("deny persists across a worker restart via session storage", async () => {
  const chrome = installChromeStub();
  const bundle = await loadOrigins();
  try {
    let origins = await bundle.import();
    await origins.raiseAccessRequest(urlOf(), "req-A");
    origins.resolveOrigin(ORIGIN, "deny");
    await flush();
    // "Worker restart": re-import resets module-level state (the `waiting`
    // map) but the record must still gate — the whole reason the record
    // lives in session storage, not memory.
    origins = await bundle.import("restart");
    const record = chrome.session("accessRequest");
    assert.equal(record.decision, "deny");
    // A fresh admission for a denied origin still fails early: the deny is a
    // cool-down, not an allow.
    await assert.rejects(
      origins.ensureTopLevelAccess(urlOf(), "req-A"),
      (error) => error.code === "origin_not_allowed",
    );
  } finally {
    await bundle.close();
    chrome.restore();
  }
});

test("a replaced request leaves a tombstone its owner reads as superseded", async () => {
  const chrome = installChromeStub();
  const bundle = await loadOrigins();
  try {
    const origins = await bundle.import();
    await origins.raiseAccessRequest(urlOf(), "req-A");
    // Session B's request for a DIFFERENT origin takes the single prompt slot.
    await origins.raiseAccessRequest(urlOf("https://other.example/x"), "req-B");
    await flush();
    const record = chrome.session("accessRequest");
    assert.equal(record.origin, "https://other.example");
    const tombs = chrome.session("accessTombstones");
    assert.equal(tombs[ORIGIN].requester, "req-A", "the receipt names the displaced requester");
  } finally {
    await bundle.close();
    chrome.restore();
  }
});
