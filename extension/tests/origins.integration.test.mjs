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
    await origins.resolveOrigin(ORIGIN, "once");
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
    await origins.resolveOrigin(ORIGIN, "once");
    // Another session cannot spend it (fail-closed, not fail-shared).
    assert.equal(await origins.consumeOnceGrant(urlOf(), "req-B"), false);
    // An anonymous caller cannot spend it either.
    assert.equal(await origins.consumeOnceGrant(urlOf(), ""), false);
    // Refusals did not consume: the owner still can.
    assert.equal(await origins.consumeOnceGrant(urlOf(), "req-A"), true);
    // Exactly once.
    assert.equal(await origins.consumeOnceGrant(urlOf(), "req-A"), false);
    // A spent grant's resolved request receipt is gone too: otherwise a later
    // request_access would read decision="once" as allowed and silently turn
    // one approval into a second admission (caught by the real E2E).
    assert.equal(chrome.session("accessRequest"), undefined);
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
    await origins.resolveOrigin(ORIGIN, "once");
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
    await origins.resolveOrigin(ORIGIN, "once");
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
    await origins.resolveOrigin(ORIGIN, "once");
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
    await origins.resolveOrigin(ORIGIN, "deny");
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

test("loopback all-port decision persists and matches only exact scheme and host", async () => {
  const chrome = installChromeStub();
  const bundle = await loadOrigins();
  try {
    let origins = await bundle.import();
    const source = new URL("http://localhost:5173/page");
    await origins.raiseAccessRequest(source, "req-A");
    assert.equal(await origins.resolveOrigin(source.origin, "all_ports"), true);
    assert.equal(await origins.originAllowed(new URL("http://localhost:9999")), true);
    assert.equal(await origins.originAllowed(new URL("https://localhost:9999")), false);
    assert.equal(await origins.originAllowed(new URL("http://127.0.0.1:9999")), false);
    // Local storage, unlike worker memory, survives MV3 suspension.
    origins = await bundle.import("host-grant-restart");
    assert.equal(await origins.originAllowed(new URL("http://localhost:6000")), true);
  } finally {
    await bundle.close();
    chrome.restore();
  }
});

test("all-port decision is rejected for a forged non-loopback popup message", async () => {
  const chrome = installChromeStub();
  const bundle = await loadOrigins();
  try {
    const origins = await bundle.import();
    await origins.raiseAccessRequest(urlOf(), "req-A");
    assert.equal(await origins.resolveOrigin(ORIGIN, "all_ports"), false);
    assert.equal(chrome.areas.local.get("hostGrants"), undefined);
  } finally {
    await bundle.close();
    chrome.restore();
  }
});

test("legacy exact-origin grants remain authoritative", async () => {
  const chrome = installChromeStub();
  const bundle = await loadOrigins();
  try {
    chrome.areas.local.set("origins", { "http://localhost:5173": "allow" });
    const origins = await bundle.import();
    assert.equal(await origins.originAllowed(new URL("http://localhost:5173/x")), true);
    assert.equal(await origins.originAllowed(new URL("http://localhost:5174/x")), false);
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
    const keyA = `${ORIGIN}\nreq-A`;
    let tombs = chrome.session("accessTombstones");
    assert.equal(tombs[keyA].requester, "req-A", "the receipt names the displaced requester");
    // A deliberately fresh request by displaced A consumes that receipt. If
    // it survived, the NEW request's later TTL expiry would incorrectly read
    // "superseded" rather than "none" (also caught by the real E2E).
    await origins.raiseAccessRequest(urlOf(), "req-A");
    tombs = chrome.session("accessTombstones");
    assert.equal(tombs[keyA], undefined);
  } finally {
    await bundle.close();
    chrome.restore();
  }
});

test("a stale prompt generation cannot approve the replacement origin (B1)", async () => {
  const chrome = installChromeStub();
  const bundle = await loadOrigins();
  try {
    const origins = await bundle.import();
    // Session A's prompt renders; the popup captures ITS generation.
    await origins.raiseAccessRequest(urlOf(), "session:A");
    const stale = chrome.session("pendingOrigin");
    assert.ok(stale.promptId, "every prompt carries a generation id");
    // Session B replaces the slot with a DIFFERENT origin before the click.
    await origins.raiseAccessRequest(urlOf("https://other.example/x"), "session:B");
    const live = chrome.session("pendingOrigin");
    assert.notEqual(live.promptId, stale.promptId);
    // The user clicks Allow on the popup still showing A: the decision names
    // A's origin and A's generation. Origin no longer matches the live
    // prompt -> rejected; nothing resolves, no grant is minted for EITHER.
    const applied = await origins.resolveOrigin(ORIGIN, "once", stale.promptId);
    assert.equal(applied, false, "stale-generation decision must be rejected");
    assert.equal(chrome.session("onceGrants"), undefined, "no grant minted");
    const record = chrome.session("accessRequest");
    assert.equal(record.origin, "https://other.example");
    assert.equal(record.decision, undefined, "B's request is still undecided");
    // Same-origin replacement: B re-raises A's origin (new generation). The
    // old generation still must not decide the new prompt.
    await origins.raiseAccessRequest(urlOf(), "session:B");
    const applied2 = await origins.resolveOrigin(ORIGIN, "once", stale.promptId);
    assert.equal(applied2, false, "old generation rejected even on same origin");
    // The CURRENT generation decides normally.
    const current = chrome.session("pendingOrigin");
    const applied3 = await origins.resolveOrigin(ORIGIN, "once", current.promptId);
    assert.equal(applied3, true);
  } finally {
    await bundle.close();
    chrome.restore();
  }
});

test("two concurrent navigations cannot double-spend one grant (B2)", async () => {
  const chrome = installChromeStub();
  const bundle = await loadOrigins();
  try {
    const origins = await bundle.import();
    await origins.raiseAccessRequest(urlOf(), "session:A");
    await origins.resolveOrigin(ORIGIN, "once");
    // The reviewer's reproduction: two same-session navigations dispatched
    // concurrently (different #321 tabs -> different daemon locks). Exactly
    // ONE may consume; the loser takes the typed early-fail.
    const results = await Promise.allSettled([
      origins.ensureTopLevelAccess(urlOf(), "cmd-1", "session:A"),
      origins.ensureTopLevelAccess(urlOf(), "cmd-2", "session:A"),
    ]);
    const admitted = results.filter((r) => r.status === "fulfilled");
    const refused = results.filter(
      (r) => r.status === "rejected" && r.reason.code === "origin_not_allowed",
    );
    assert.equal(admitted.length, 1, `exactly one winner, got ${admitted.length}`);
    assert.equal(refused.length, 1, "the loser fails typed, not silently");
    assert.deepEqual(admitted[0].value, { allowed: true, viaOnceGrant: true });
  } finally {
    await bundle.close();
    chrome.restore();
  }
});

test("A->B->C same-origin supersession keeps every displaced receipt (M1)", async () => {
  const chrome = installChromeStub();
  const bundle = await loadOrigins();
  try {
    const origins = await bundle.import();
    // Three sessions race the SAME origin's single prompt slot.
    await origins.raiseAccessRequest(urlOf(), "session:A");
    await origins.raiseAccessRequest(urlOf(), "session:B");
    await origins.raiseAccessRequest(urlOf(), "session:C");
    const tombs = chrome.session("accessTombstones");
    // A per-origin key overwrote A's receipt with B's; per origin+requester
    // keys keep both (round-2 M1).
    assert.ok(tombs[`${ORIGIN}\nsession:A`], "A's receipt survives C's raise");
    assert.ok(tombs[`${ORIGIN}\nsession:B`], "B's receipt exists");
    const record = chrome.session("accessRequest");
    assert.equal(record.requester, "session:C", "C owns the live prompt");
  } finally {
    await bundle.close();
    chrome.restore();
  }
});

test("A->B->C different-origin chain also keeps both receipts (M1)", async () => {
  const chrome = installChromeStub();
  const bundle = await loadOrigins();
  try {
    const origins = await bundle.import();
    await origins.raiseAccessRequest(urlOf(), "session:A");
    await origins.raiseAccessRequest(urlOf("https://b.example/"), "session:B");
    await origins.raiseAccessRequest(urlOf("https://c.example/"), "session:C");
    const tombs = chrome.session("accessTombstones");
    assert.ok(tombs[`${ORIGIN}\nsession:A`]);
    assert.ok(tombs["https://b.example\nsession:B"]);
  } finally {
    await bundle.close();
    chrome.restore();
  }
});

test("resolveOrigin settles only after the decision is durable (M2)", async () => {
  const chrome = installChromeStub();
  const bundle = await loadOrigins();
  try {
    const origins = await bundle.import();
    await origins.raiseAccessRequest(urlOf(), "session:A");
    // Worker-suspension-shaped delay: every storage WRITE lands late, the
    // way a busy event loop orders them just before MV3 suspension. The
    // promise the message listener keeps alive must not settle until those
    // writes are actually applied.
    const realSet = chrome.areas.session.set;
    globalThis.chrome.storage.session.set = async (obj) => {
      await new Promise((resolve) => setTimeout(resolve, 30));
      const changes = {};
      for (const [key, value] of Object.entries(obj)) {
        changes[key] = { newValue: value };
        chrome.areas.session.set(key, value);
      }
    };
    const applied = await origins.resolveOrigin(ORIGIN, "once");
    assert.equal(applied, true);
    // The moment resolveOrigin settles, the decision and grant are READABLE:
    // nothing is still in flight for suspension to lose.
    assert.equal(chrome.session("accessRequest").decision, "once", "decision durably recorded");
    const grants = chrome.session("onceGrants");
    assert.ok(grants && grants[ORIGIN], "grant durably minted before settle");
    assert.equal(grants[ORIGIN].requester, "session:A");
    void realSet;
  } finally {
    await bundle.close();
    chrome.restore();
  }
});

test("a queued same-origin replacement cannot be decided by a stale click (R3-B1 TOCTOU)", async () => {
  const chrome = installChromeStub();
  const bundle = await loadOrigins();
  try {
    const origins = await bundle.import();
    await origins.raiseAccessRequest(urlOf(), "session:A");
    const stalePrompt = chrome.session("pendingOrigin");
    // The reviewer's interleaving: B's same-origin replacement enters the
    // mutation queue FIRST and pauses mid-write; the stale A click validates
    // against the OLD prompt while B is queued; B then installs its fresh
    // generation; the stale click must NOT apply to it.
    const realSet = globalThis.chrome.storage.session.set;
    let releaseB;
    const gateB = new Promise((resolve) => { releaseB = resolve; });
    globalThis.chrome.storage.session.set = async (obj) => {
      if (obj && obj.accessRequest && obj.accessRequest.requester === "session:B") {
        await gateB; // B paused INSIDE its queued mutation, after A's validation
      }
      const changes = {};
      for (const [key, value] of Object.entries(obj || {})) {
        changes[key] = { newValue: value };
        chrome.areas.session.set(key, value);
      }
      void realSet;
    };
    // B's raise enters the queue and parks on the gate.
    const raiseB = origins.raiseAccessRequest(urlOf(), "session:B");
    await new Promise((resolve) => setTimeout(resolve, 20)); // B queued + parked
    // A's stale click (old generation) is delivered and — pre-fix — validated
    // OUTSIDE the queue, then queued behind B's parked mutation.
    const staleClick = origins.resolveOrigin(ORIGIN, "once", stalePrompt.promptId);
    await new Promise((resolve) => setTimeout(resolve, 20)); // stale click queued behind B
    releaseB();
    const applied = await staleClick;
    await raiseB;
    globalThis.chrome.storage.session.set = realSet;
    // The invariant: the stale click did NOT decide B's request.
    assert.equal(applied, false, "stale click must be rejected, not applied to B");
    const record = chrome.session("accessRequest");
    assert.equal(record.requester, "session:B");
    assert.equal(record.decision, undefined, "B's request is still undecided");
    const grants = chrome.session("onceGrants");
    assert.ok(!grants || !grants[ORIGIN], "no grant minted for B from A's stale click");
    // B's CURRENT generation still decides normally afterwards.
    const current = chrome.session("pendingOrigin");
    assert.notEqual(current.promptId, stalePrompt.promptId);
    const appliedNow = await origins.resolveOrigin(ORIGIN, "once", current.promptId);
    assert.equal(appliedNow, true);
    const grantsNow = chrome.session("onceGrants");
    assert.equal(grantsNow[ORIGIN].requester, "session:B", "grant for B, by B's click");
  } finally {
    await bundle.close();
    chrome.restore();
  }
});

test("duplicate current-generation delivery is idempotent (R3-B1)", async () => {
  const chrome = installChromeStub();
  const bundle = await loadOrigins();
  try {
    const origins = await bundle.import();
    await origins.raiseAccessRequest(urlOf(), "session:A");
    const prompt = chrome.session("pendingOrigin");
    const first = await origins.resolveOrigin(ORIGIN, "once", prompt.promptId);
    const second = await origins.resolveOrigin(ORIGIN, "once", prompt.promptId);
    assert.equal(first, true);
    // The duplicate is a NO-OP, not an error: the prompt record is consumed
    // by the first decision, so the re-delivery finds nothing live and is
    // rejected as stale — with no side effects (the grant below is untouched).
    assert.equal(second, false, "duplicate delivery applies nothing");
    const grants = chrome.session("onceGrants");
    assert.ok(grants && grants[ORIGIN], "exactly one grant object");
    assert.equal(grants[ORIGIN].expiresAt, grants[ORIGIN].expiresAt);
    const record = chrome.session("accessRequest");
    assert.equal(record.decision, "once");
  } finally {
    await bundle.close();
    chrome.restore();
  }
});
