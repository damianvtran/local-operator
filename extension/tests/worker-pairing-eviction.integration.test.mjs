/* Coverage for the worker's close-code handling during PAIRING (J4).
 *
 * When the user submits a pairing code, the POPUP opens its own socket to the
 * daemon. The daemon's "later connection wins" rule evicts the worker's socket
 * with close code 4000 — an ordinary, expected part of every pair, not a loss
 * of connectivity. The worker nonetheless published `connState: "disconnected"`
 * from its teardown, and because the popup re-enters render() on any connState
 * write (chrome.storage.onChanged), the card painted an extra transition
 * mid-pair. Measured in real Chrome against a real daemon: 4/15 baseline runs
 * showed pairing -> connected -> paired -> connected or
 * paired -> connected -> pairing -> connected.
 *
 * The fix suppresses ONLY the storage write for 4000. This file pins both
 * halves: the write is gone, and the reconnect the worker still owes is not.
 * Reverting the `!== 4000` guard turns the first test red.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { build } from "esbuild";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const tick = (n = 4) => new Promise((r) => setTimeout(r, n));

/** Load worker.ts against a socket whose close code the test controls, and
 * record every chrome.storage.session write it makes. */
async function loadWorker() {
  const session = new Map();
  const local = new Map([["token", "tok"], ["port", 4099]]);
  const writes = [];
  let socket;
  let alarmsCreated = 0;
  let socketsOpened = 0;

  Object.defineProperty(globalThis, "navigator", {
    value: { userAgent: "node-test" },
    configurable: true,
    writable: true,
  });
  globalThis.WebSocket = class {
    static OPEN = 1;
    readyState = 1;
    constructor() {
      socket = this;
      socketsOpened++;
      queueMicrotask(() => this.onopen?.());
    }
    send() {}
    close() {}
  };
  const area = (map, name) => ({
    get: async (keys) => {
      const out = {};
      for (const k of Array.isArray(keys) ? keys : [keys]) if (map.has(k)) out[k] = map.get(k);
      return out;
    },
    set: async (obj) => {
      if (name === "session") writes.push({ ...obj });
      for (const [k, v] of Object.entries(obj)) map.set(k, v);
    },
    remove: async (keys) => {
      for (const k of Array.isArray(keys) ? keys : [keys]) map.delete(k);
    },
  });
  // The chrome surface worker.ts evaluates against, mirroring the one in
  // driven-handles.integration.test.mjs. Only storage.session is instrumented:
  // that is the channel the popup renders off, and so the one this defect
  // travels down.
  globalThis.chrome = {
    storage: {
      session: area(session, "session"),
      local: area(local, "local"),
      onChanged: { addListener: () => {} },
    },
    alarms: {
      // The re-dial the worker still owes after an eviction is scheduled here,
      // so counting these is how the test proves the fix did not suppress it.
      create: () => {
        alarmsCreated++;
      },
      clear: async () => {},
      onAlarm: { addListener: () => {} },
    },
    action: Object.fromEntries(
      ["setBadgeBackgroundColor", "setBadgeTextColor", "setBadgeText", "setTitle"].map((m) => [
        m,
        async () => {},
      ]),
    ),
    debugger: {
      attach: async () => {},
      detach: async () => {},
      sendCommand: async () => ({}),
      onEvent: { addListener: () => {} },
      onDetach: { addListener: () => {} },
    },
    tabs: {
      get: async (tabId) => ({ id: tabId, url: "https://example.com/live", title: "Live" }),
      remove: async () => {},
      query: async () => [],
      onRemoved: { addListener: () => {} },
      onReplaced: { addListener: () => {} },
      onUpdated: { addListener: () => {} },
    },
    tabGroups: undefined,
    notifications: { create: async () => {}, clear: async () => {}, onClicked: { addListener: () => {} } },
    windows: { get: async () => ({ id: 1 }), getCurrent: async () => ({ id: 1 }) },
    runtime: {
      getURL: (p) => `chrome-extension://test/${p}`,
      getManifest: () => ({ version: "0.1.9" }),
      onStartup: { addListener: () => {} },
      onInstalled: { addListener: () => {} },
      onMessage: { addListener: () => {} },
      sendMessage: async () => {},
    },
  };

  const dir = await mkdtemp(join(tmpdir(), "lop-worker-evict-"));
  const outfile = join(dir, "worker.mjs");
  await build({
    entryPoints: [join(HERE, "..", "src", "worker.ts")],
    bundle: true,
    platform: "node",
    format: "esm",
    outfile,
  });
  await import(pathToFileURL(outfile) + `?${Date.now()}`);
  for (let i = 0; i < 60 && !socket; i++) await tick();
  assert.ok(socket, "the worker opened a socket");
  await tick(20);

  return {
    get writes() {
      return writes;
    },
    get alarmsCreated() {
      return alarmsCreated;
    },
    /** Close the worker's socket the way the daemon does, then wait out the
     * reconnect fast path (backoffDelayMs(0)) and report whether the worker
     * actually dialled again. scheduleReconnect uses setTimeout, not the
     * alarm, so the re-dial is observed as a NEW SOCKET rather than inferred
     * from chrome.alarms. */
    evict: async (code) => {
      const before = socketsOpened;
      socket.onclose?.({ code });
      await tick(1400);
      return { reconnectScheduled: socketsOpened > before };
    },
    connState: () => session.get("connState"),
    close: () => rm(dir, { recursive: true, force: true }),
  };
}

test("the pairing socket's 4000 eviction does not publish a disconnect (J4)", async () => {
  const worker = await loadWorker();
  try {
    const stateBefore = worker.connState();
    const { reconnectScheduled } = await worker.evict(4000);

    const disconnects = worker.writes.filter((w) => w.connState === "disconnected");
    assert.deepEqual(
      disconnects,
      [],
      "a 4000 eviction is the popup's own pairing socket winning, not a lost connection",
    );
    assert.equal(
      worker.connState(),
      stateBefore,
      "the published state must not move: a render driven off this write repaints the card mid-pair",
    );
    // The other half. The socket really is gone, so suppressing the write must
    // not suppress the re-dial — without it the worker never reconnects with
    // the new token and pairing hangs on "Connecting…" forever.
    assert.equal(reconnectScheduled, true, "the worker must still re-dial after a 4000 eviction");
  } finally {
    await worker.close();
  }
});

test("an ordinary disconnect still publishes disconnected (J4 regression guard)", async () => {
  // The suppression is scoped to 4000 alone. A real disconnect must still tell
  // the popup, or the fix has traded a flicker for a silent dead connection.
  const worker = await loadWorker();
  try {
    const { reconnectScheduled } = await worker.evict(1006);
    assert.equal(
      worker.connState(),
      "disconnected",
      "a genuine transport failure must still surface to the popup",
    );
    assert.equal(reconnectScheduled, true, "and must still re-dial");
  } finally {
    await worker.close();
  }
});

test("protocol and unpair close codes keep their own states (J4 regression guard)", async () => {
  for (const [code, expected] of [
    [4001, "incompatible"],
    [4003, "pairing"],
  ]) {
    const worker = await loadWorker();
    try {
      await worker.evict(code);
      assert.equal(worker.connState(), expected, `close ${code} must still publish "${expected}"`);
    } finally {
      await worker.close();
    }
  }
});

/** Load one source module on its own, for the pure helpers. */
async function loadModule(relative) {
  const dir = await mkdtemp(join(tmpdir(), "lop-worker-evict-mod-"));
  const outfile = join(dir, "module.mjs");
  await build({
    entryPoints: [join(HERE, "..", relative)],
    bundle: true,
    platform: "node",
    format: "esm",
    outfile,
  });
  return import(pathToFileURL(outfile) + `?${Date.now()}`);
}

test("a post-onopen 4000 close re-dials on the attempt-0 fast path, not the alarm floor", async () => {
  // The design's §4.2 claim — "the replacement registered in ~1 s" — pinned as
  // a STRUCTURAL fact about the delay the worker ASKS for, not as a measurement
  // of how long anything took (AGENTS.md: wait on the event, never on the
  // clock). `onopen` resets `attempt` to 0, so a close that arrives after a
  // successful open must arm `backoffDelayMs(0)`. If the reset were dropped,
  // the requested delay would be 2000 and this fails.
  const reconnect = await loadModule("src/reconnect.ts");
  assert.equal(reconnect.backoffDelayMs(0), 1_000, "attempt 0 is the ~1 s fast path");
  assert.equal(reconnect.backoffDelayMs(1), 2_000, "and attempt 1 doubles it");

  const worker = await loadWorker();
  const realSetTimeout = globalThis.setTimeout;
  const requested = [];
  globalThis.setTimeout = (fn, delay, ...rest) => {
    requested.push(delay);
    return realSetTimeout(fn, delay, ...rest);
  };
  try {
    const first = await worker.evict(4000);
    assert.equal(first.reconnectScheduled, true, "the worker must still re-dial after a 4000 eviction");
    assert.ok(
      requested.includes(1_000),
      `expected the attempt-0 fast path, got delays ${JSON.stringify(requested)}`,
    );
    assert.ok(!requested.includes(2_000), "a post-onopen close must not be armed as attempt 1");

    // Round 2 is what makes the RESET observable. The first close cannot
    // distinguish "onopen reset attempt" from "attempt was never incremented",
    // because it starts at 0 — so close a SECOND time, after a successful
    // re-dial whose onopen should have reset the backoff. Without that reset
    // the second close is armed as backoffDelayMs(1) = 2000.
    requested.length = 0;
    const second = await worker.evict(4000);
    assert.equal(second.reconnectScheduled, true, "the re-dial must be repeatable");
    assert.ok(
      requested.includes(1_000),
      `expected attempt-0 again after a successful re-dial, got ${JSON.stringify(requested)}`,
    );
    assert.ok(
      !requested.includes(2_000),
      "the backoff was not reset by the successful re-dial",
    );
  } finally {
    globalThis.setTimeout = realSetTimeout;
    await worker.close();
  }
});
