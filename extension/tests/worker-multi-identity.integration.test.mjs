/* Multi-identity coverage for the extension (design note §9.2, rows E1-E4).
 *
 * The daemon half is tested in Python. These rows are the extension half of the
 * same rule: the daemon may answer `role: "standby"`, and a standby install must
 * (a) behave exactly as before when a released daemon sends no role at all,
 * (b) HAND BACK everything it holds when it is told it is a standby, and
 * (c) come back to life on a live promotion without a reconnect.
 *
 * (b) is the user-visible half of the whole change: a standby receives no
 * Request, so any surface it kept would be a tab the agent can never close
 * again, wearing a "Local Operator is debugging this browser" banner. Reverting
 * `releaseAllSurfaces` from `applyRole` turns E2 red.
 *
 * Loads the REAL worker against a recordable fake WebSocket and a chrome stub,
 * the same technique worker-wire-generation.integration.test.mjs uses.
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

const TOKEN = "bridge:7:abc123def456";

async function loadWorker() {
  const session = new Map([
    [
      "surfaces",
      {
        [TOKEN]: {
          tabId: 7,
          nonce: "abc123def456",
          epoch: 1,
          createdAt: 1,
          lastUsedAt: 1,
        },
      },
    ],
  ]);
  const local = new Map([
    ["token", "tok"],
    ["port", 4099],
  ]);
  const writes = [];
  const attached = [];
  const detached = [];
  const wires = [];

  Object.defineProperty(globalThis, "navigator", {
    value: { userAgent: "node-test" },
    configurable: true,
    writable: true,
  });
  globalThis.WebSocket = class {
    static OPEN = 1;
    static CLOSED = 3;
    readyState = 1;
    constructor() {
      wires.push(this);
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

  globalThis.chrome = {
    storage: {
      session: area(session, "session"),
      local: area(local, "local"),
      onChanged: { addListener: () => {} },
    },
    alarms: {
      create: () => {},
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
      attach: async (target) => {
        attached.push(target?.tabId);
      },
      detach: async (target) => {
        detached.push(target?.tabId);
      },
      sendCommand: async () => ({ data: "" }),
      onEvent: { addListener: () => {} },
      onDetach: { addListener: () => {} },
    },
    scripting: {
      executeScript: async () => [{ result: "page text" }],
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
      getManifest: () => ({ version: "0.1.13" }),
      id: "jbadjeaodkoboanppmpjiifpconegdcj",
      onStartup: { addListener: () => {} },
      onInstalled: { addListener: () => {} },
      onMessage: { addListener: () => {} },
      sendMessage: async () => {},
    },
  };

  const dir = await mkdtemp(join(tmpdir(), "lop-worker-multi-"));
  const outfile = join(dir, "worker.mjs");
  await build({
    entryPoints: [join(HERE, "..", "src", "worker.ts")],
    bundle: true,
    platform: "node",
    format: "esm",
    outfile,
  });
  await import(pathToFileURL(outfile) + `?${Date.now()}`);
  for (let i = 0; i < 60 && wires.length === 0; i++) await tick();
  assert.ok(wires[0], "the worker dialled");
  await tick(20);

  return {
    get wire() {
      return wires[0];
    },
    wires,
    attached,
    detached,
    surfaces: () => session.get("surfaces") ?? {},
    /** Seed the surface map the way a fresh `open` would. */
    setSurfaces: (value) => session.set("surfaces", value),
    connState: () => session.get("connState"),
    /** Deliver a frame to the worker exactly as the daemon's socket would. */
    deliver: async (frame) => {
      wires[0].onmessage?.({ data: JSON.stringify(frame) });
      await tick(30);
    },
    /** Wait until `predicate` holds, or fail after ~1 s. */
    settle: async (predicate, what) => {
      for (let i = 0; i < 100; i++) {
        if (predicate()) return;
        await tick(10);
      }
      assert.ok(predicate(), `timed out waiting for ${what}`);
    },
    close: () => rm(dir, { recursive: true, force: true }),
  };
}

test("E1: a hello_ack with NO role behaves exactly as before (old daemon)", async () => {
  // The compat half that has to work without a PROTO_VERSION bump: a released
  // daemon never sends `role`, and reading an absent role as "standby" would
  // make this build hand its tabs to nobody on every existing install.
  const worker = await loadWorker();
  try {
    await worker.deliver({ event: "hello_ack", proto: 1, paired: true });
    assert.equal(worker.connState(), "connected");
    assert.deepEqual(worker.detached, [], "an absent role is a DRIVER, not a demotion");
    assert.deepEqual(Object.keys(worker.surfaces()), [TOKEN]);
    // And an explicit driver role is the same answer.
    await worker.deliver({ event: "role", role: "driver" });
    assert.equal(worker.connState(), "connected");
    assert.deepEqual(worker.detached, []);
  } finally {
    await worker.close();
  }
});

test("E2: role standby hands back every debugger session and surface", async () => {
  const worker = await loadWorker();
  try {
    await worker.deliver({ event: "hello_ack", proto: 1, paired: true, role: "driver" });

    // Drive a real command so this worker OWNS a debugger session: the point of
    // the row is that a demotion releases sessions it actually holds, so a test
    // that never attached anything would prove nothing.
    await worker.deliver({ id: "r-1", method: "screenshot", params: { tab: TOKEN } });
    await worker.settle(() => worker.attached.includes(7), "the driver to attach to tab 7");
    assert.ok(worker.surfaces()[TOKEN], "precondition: the surface is held by the driver");

    await worker.deliver({ event: "hello_ack", proto: 1, paired: true, role: "standby" });

    await worker.settle(() => worker.detached.includes(7), "the demotion to release tab 7");
    assert.deepEqual(
      Object.keys(worker.surfaces()),
      [],
      "a standby must hold no surfaces: nothing can close them once it stops driving",
    );
    assert.equal(worker.connState(), "standby");

    // And the sweep is idempotent: an ack repeats on every dial, and re-running
    // it must not churn storage or re-detach what is already gone.
    const detaches = worker.detached.length;
    await worker.deliver({ event: "hello_ack", proto: 1, paired: true, role: "standby" });
    assert.equal(worker.detached.length, detaches, "a repeated standby ack re-detached a session");
  } finally {
    await worker.close();
  }
});

test("E1b: an UNPAIRED standby lands on the pairing form, not the standby card", async () => {
  // A second install's first dial is exactly this: paired: false and standby,
  // because the first install holds the wheel. Reporting "standby" there showed
  // the user a card claiming a pairing that did not exist yet, with no way to
  // enter the code — and pairing is the only thing that link can still do.
  const worker = await loadWorker();
  try {
    await worker.deliver({ event: "hello_ack", proto: 1, paired: false, role: "standby" });
    assert.equal(worker.connState(), "pairing");
    // ... and once it IS paired, the same role reads as standby.
    await worker.deliver({ event: "hello_ack", proto: 1, paired: true, role: "standby" });
    assert.equal(worker.connState(), "standby");
  } finally {
    await worker.close();
  }
});

test("E3: a live promotion re-arms the worker without a reconnect", async () => {
  const worker = await loadWorker();
  try {
    await worker.deliver({ event: "hello_ack", proto: 1, paired: true, role: "standby" });
    assert.equal(worker.connState(), "standby");
    const dials = worker.wires.length;

    // The daemon promotes this link in place (a failover, or `lop browser
    // drive`). No new socket: waiting for the alarm floor would leave the
    // wheel with nobody for up to a minute.
    await worker.deliver({ event: "role", role: "driver" });

    assert.equal(worker.connState(), "connected", "the promotion must re-arm this install");
    assert.equal(worker.wires.length, dials, "the promotion must not need a re-dial");

    // The re-armed install serves a FRESH surface on the SAME socket. A fresh
    // handle rather than the old one on purpose: the demotion released the
    // old surface (and the daemon of the install that minted it owns it), so
    // the session re-opens and the newly promoted install picks the new tab up
    // — which is the documented cost of a failover, exercised end to end here.
    const fresh = "bridge:8:newhandle";
    worker.setSurfaces({
      [fresh]: { tabId: 8, nonce: "newhandle", epoch: 1, createdAt: 1, lastUsedAt: 1 },
    });
    await worker.deliver({ id: "r-2", method: "screenshot", params: { tab: fresh } });
    await worker.settle(() => worker.attached.includes(8), "the promoted install to serve a command");
    assert.ok(
      worker.wires.every((wire) => wire === worker.wire),
      "the promotion must not have dialled a new socket",
    );
  } finally {
    await worker.close();
  }
});
