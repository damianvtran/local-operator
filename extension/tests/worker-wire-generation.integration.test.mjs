/* Coverage for the worker's connection GENERATION (audit A1, extension half).
 *
 * `dispatch` is fired fire-and-forget from the socket's onmessage handler, so a
 * handler that finishes after a reconnect writes its response — and the
 * `tab_update` that precedes it — to the module-global `socket`, which by then
 * is the REPLACEMENT connection. The daemon matches a response to the request it
 * sent on the OLD socket, so the new socket receives a frame it never asked for,
 * and the worker's own promise ("nothing is replayed on a new socket") is broken
 * from the worker's side. Reproduced by the concurrency audit by bundling this
 * file, gating one handler, replacing the wire, then releasing the handler:
 *
 *   oldResponseOnNewWire=[tab_update bridge:1:old,
 *                         {id: old-request, ok: true, tab: bridge:1:old}]
 *
 * Two rows: the stale response must be dropped, and — the other half, because a
 * fence that dropped everything would also pass the first — a request that
 * arrives on the CURRENT wire must still be answered.
 *
 * Separate file from worker-pairing-eviction.integration.test.mjs on purpose:
 * this row needs a gated handler (via a `tab-groups` alias, the same technique
 * bridge-wedge.integration.test.mjs uses) and per-socket frame injection, which
 * that file's harness deliberately does not have.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { build } from "esbuild";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const SRC = resolve(HERE, "..", "src");
const tick = (n = 4) => new Promise((r) => setTimeout(r, n));

/** Load worker.ts with `tab-groups` aliased, a recordable fake WebSocket, and a
 * controllable `retitle` gate. */
async function loadWorker() {
  const session = new Map();
  const local = new Map([
    ["token", "tok"],
    ["port", 4099],
  ]);
  const wires = [];
  const warnings = [];
  let release = () => {};
  const gate = new Promise((r) => {
    release = r;
  });

  Object.defineProperty(globalThis, "navigator", {
    value: { userAgent: "node-test" },
    configurable: true,
    writable: true,
  });
  globalThis.WebSocket = class {
    static OPEN = 1;
    static CLOSED = 3;
    readyState = 1;
    sent = [];
    constructor() {
      wires.push(this);
      queueMicrotask(() => this.onopen?.());
    }
    send(data) {
      this.sent.push(JSON.parse(String(data)));
    }
    close() {
      this.readyState = 3;
      queueMicrotask(() => this.onclose?.({ code: 1000 }));
    }
  };
  const realWarn = console.warn;
  console.warn = (...args) => {
    warnings.push(args.map(String).join(" "));
  };
  const area = (map) => ({
    get: async (keys) => {
      const out = {};
      for (const k of Array.isArray(keys) ? keys : [keys]) if (map.has(k)) out[k] = map.get(k);
      return out;
    },
    set: async (obj) => {
      for (const [k, v] of Object.entries(obj)) map.set(k, v);
    },
    remove: async (keys) => {
      for (const k of Array.isArray(keys) ? keys : [keys]) map.delete(k);
    },
  });
  globalThis.chrome = {
    storage: { session: area(session), local: area(local), onChanged: { addListener: () => {} } },
    alarms: { create: () => {}, clear: async () => {}, onAlarm: { addListener: () => {} } },
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
    notifications: {
      create: async () => {},
      clear: async () => {},
      onClicked: { addListener: () => {} },
    },
    windows: { get: async () => ({ id: 1 }), getCurrent: async () => ({ id: 1 }) },
    runtime: {
      getURL: (p) => `chrome-extension://test/${p}`,
      getManifest: () => ({ version: "0.1.12" }),
      onStartup: { addListener: () => {} },
      onInstalled: { addListener: () => {} },
      onMessage: { addListener: () => {} },
      sendMessage: async () => {},
    },
  };

  const dir = await mkdtemp(join(tmpdir(), "lop-worker-gen-"));
  const fixture = join(dir, "tab-groups.mjs");
  // The gate IS the test's instrument: `retitle` reaches this fixture's promise
  // and nothing resolves it until the test says so, which is how a handler is
  // held open across the reconnect.
  await writeFile(
    fixture,
    [
      "export const reconcileCommandTab = async () => undefined;",
      "export const retitle = async () => {",
      "  await gatePromise;",
      "  return { tab: 'bridge:1:old', url: 'https://old-request.test', title: 'OLD' };",
      "};",
    ].join("\n"),
  );
  const outfile = join(dir, "worker.mjs");
  await build({
    entryPoints: [join(SRC, "worker.ts")],
    bundle: true,
    platform: "node",
    format: "esm",
    outfile,
    plugins: [
      {
        name: "gated-tab-groups",
        setup(b) {
          b.onResolve({ filter: /^\.\/tab-groups$/ }, () => ({ path: "tab-groups", namespace: "g" }));
          b.onLoad({ filter: /.*/, namespace: "g" }, () => ({
            contents: `const gatePromise = globalThis.__lopGate;\nexport const reconcileCommandTab = async () => undefined;\nexport const retitle = async () => {\n  await gatePromise;\n  return { tab: "bridge:1:old", url: "https://old-request.test", title: "OLD" };\n};\n`,
            loader: "js",
          }));
        },
      },
    ],
  });
  globalThis.__lopGate = gate;
  await import(pathToFileURL(outfile) + `?${Date.now()}`);
  for (let i = 0; i < 60 && !wires.length; i++) await tick();
  assert.ok(wires.length, "the worker opened a socket");
  await tick(20);

  return {
    gate: { release },
    wires,
    warnings,
    /** Deliver a daemon frame to a specific socket, as that socket's peer. */
    deliver: (wire, frame) => wire.onmessage?.({ data: JSON.stringify(frame) }),
    /** Wait for a NEW socket to appear — the reconnect fast path (~1 s). */
    waitForDial: async (count, ms = 4000) => {
      const started = Date.now();
      while (wires.length < count && Date.now() - started < ms) await tick(20);
      assert.equal(wires.length >= count, true, `the worker re-dialled (saw ${wires.length})`);
      await tick(20);
    },
    close: async () => {
      console.warn = realWarn;
      delete globalThis.__lopGate;
      await rm(dir, { recursive: true, force: true });
    },
  };
}

test("a response to a request from a superseded connection is not replayed (A1)", async () => {
  const worker = await loadWorker();
  try {
    const [first] = worker.wires;
    worker.deliver(first, { id: "old-request", method: "retitle", params: { title: "OLD" } });
    await tick(20);

    // The wire goes away while the handler is still parked — exactly what a
    // reconnect does to an in-flight command.
    first.onclose?.({ code: 1006 });
    await worker.waitForDial(2);
    const second = worker.wires[1];
    assert.notEqual(second, first, "a new socket is the live one");
    const secondSentBefore = second.sent.length;

    worker.gate.release();
    await tick(40);

    const replayed = second.sent
      .slice(secondSentBefore)
      .filter((f) => f.id === "old-request" || f.event === "tab_update");
    assert.deepEqual(
      replayed,
      [],
      "the old request's response and tab_update must not land on the new wire",
    );
    // The drop is observable, not silent: an agent reading the worker console
    // should be able to see WHY a response never went out.
    assert.ok(
      worker.warnings.some((w) => w.includes("old-request")),
      `the dropped response was logged (warnings: ${JSON.stringify(worker.warnings)})`,
    );
    assert.equal(first.sent.filter((f) => f.id === "old-request").length, 0);
  } finally {
    await worker.close();
  }
});

test("a request on the current wire is still answered, and still pushes its tab (A1, inverse)", async () => {
  const worker = await loadWorker();
  try {
    const [first] = worker.wires;
    worker.gate.release();
    worker.deliver(first, { id: "fresh-request", method: "retitle", params: { title: "NEW" } });
    await tick(40);

    const response = first.sent.find((f) => f.id === "fresh-request");
    assert.ok(response, "a live request must still be answered");
    assert.equal(response.ok, true);
    assert.ok(
      first.sent.some((f) => f.event === "tab_update" && f.tab === "bridge:1:old"),
      "a live request must still push its driven tab",
    );
    assert.deepEqual(worker.warnings, [], "nothing was dropped on a live wire");
  } finally {
    await worker.close();
  }
});