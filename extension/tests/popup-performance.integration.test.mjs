import { test } from "node:test";
import assert from "node:assert/strict";
import { build } from "esbuild";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { installPopupDom } from "./fixtures/popup-dom.mjs";
import { installPopupPerformance } from "./fixtures/popup-performance.mjs";

const source = resolve(process.env.POPUP_SOURCE || "src");
const html = await readFile(resolve(source, "popup/popup.html"), "utf8");
const { outputFiles } = await build({ entryPoints: [resolve(source, "popup/popup.ts")], bundle: true, format: "esm", platform: "node", write: false });
let sequence = 0;
const flush = () => new Promise(resolve => setImmediate(resolve));
const barrier = () => { let release; const promise = new Promise(resolve => { release = resolve; }); return { promise, release }; };
async function load(options = {}) {
  let host;
  const fixture = installPopupPerformance(options);
  host = installPopupDom(html, () => { fixture.metrics.mutations++; });
  if (options.warm) localStorage.setItem("lop:pin-hint", "148px");
  options.beforeImport?.();
  await import(`data:text/javascript;base64,${Buffer.from(outputFiles[0].text + `\n//# sourceURL=popup-performance-${sequence++}.mjs`).toString("base64")}`);
  return { ...fixture, ...host };
}

test("first render reads only the two projections concurrently; health does not wait for session", async () => {
  const session = barrier(), local = barrier();
  const f = await load({ large: true, gate: read => read.area === "session" ? session.promise : local.promise });
  const started = f.metrics.reads.map(r => r.area).sort();
  // Release even if the baseline assertion fails: no pending test work may
  // leak into the next fixture's globals.
  local.release(); await flush();
  const healthWhileSessionBlocked = f.metrics.health.length;
  const visibleWhileBlocked = f.visible();
  session.release(); await flush();
  assert.deepEqual(started, ["local", "session"]);
  assert.equal(healthWhileSessionBlocked, 1);
  assert.deepEqual(visibleWhileBlocked, ["pending"]);
  assert.deepEqual(f.visible(), ["connected"]);
  assert.equal(f.metrics.reads.length, 2);
  assert.equal(f.metrics.reads.reduce((n, r) => n + r.bytes, 0), 125);
  assert.equal(f.metrics.messages.length, 0);
  assert.equal(f.metrics.writes.length, 0);
});

test("slow health never paints cached connected authority, even with a warm pin", async () => {
  const health = barrier();
  const f = await load({ warm: true, healthFetch: async () => { await health.promise; return { ok: true, json: async () => ({ paired: false, extension_connected: false, protocol_version: 1 }) }; } });
  await flush();
  assert.deepEqual(f.visible(), ["pending"]);
  health.release(); await flush();
  assert.deepEqual(f.visible(), ["pairing"]);
});

test("burst coalesces into one queued render, and irrelevant ref events do not paint", async () => {
  const first = barrier(), second = barrier();
  let calls = 0;
  const f = await load({ healthFetch: async () => { const call = ++calls; await (call === 1 ? first.promise : second.promise); return { ok: true, json: async () => ({ paired: true, extension_connected: true, protocol_version: 1 }) }; } });
  await flush();
  for (let i = 0; i < 30; i++) f.emit({ connState: { newValue: "connected" } });
  await flush();
  assert.equal(calls, 1, "no parallel health probes/painters");
  first.release(); await flush();
  assert.equal(calls, 2, "exactly one queued rerender");
  second.release(); await flush();
  const mutations = f.metrics.mutations;
  for (let i = 0; i < 30; i++) f.emit({ refs: { newValue: { synthetic: {} } } });
  await flush();
  assert.equal(calls, 2);
  assert.equal(f.metrics.mutations, mutations);
});

for (const area of ["session", "local"]) {
  test(`${area} storage stall reaches Retry, releases latch, and ignores late completion`, async t => {
    t.mock.timers.enable({ apis: ["setTimeout"] });
    const blocked = barrier();
    let stall = true;
    const f = await load({ gate: read => stall && read.area === area ? blocked.promise : undefined });
    await flush();
    assert.deepEqual(f.visible(), ["pending"]);
    t.mock.timers.tick(5000); await flush();
    assert.deepEqual(f.visible(), ["disconnected"]);
    stall = false;
    f.nodes.get("retry").click(); await flush();
    assert.deepEqual(f.visible(), ["connected"]);
    f.health.extension_connected = false;
    const mutations = f.metrics.mutations;
    blocked.release(); await flush();
    assert.equal(f.metrics.mutations, mutations, "timed-out read cannot repaint");
    t.mock.timers.reset();
  });
}

test("read rejection never discards unknown consent state to show connected", async () => {
  const f = await load({ state: "consent", gate: read => { if (read.area === "session") throw new Error("synthetic storage failure"); } });
  await flush();
  assert.deepEqual(f.visible(), ["disconnected"]);
});

for (const area of ["session", "local"]) {
  test(`${area} synchronous Chrome failure also leaves a usable Retry`, async t => {
    const warnings = t.mock.method(console, "warn", () => {});
    const f = await load({ beforeImport() { chrome.storage[area].get = () => { throw new Error("Synthetic invalidated context"); }; } });
    await flush();
    assert.deepEqual(f.visible(), ["disconnected"]);
    assert.equal(warnings.mock.callCount(), 1);
  });
}

for (const [state, expected] of [["healthy", "connected"], ["standby", "standby"], ["consent", "origin"], ["unreachable", "disconnected"]]) {
  test(`${state} uses the same projected state with large refs`, async () => {
    const f = await load({ state, large: true }); await flush();
    assert.deepEqual(f.visible(), [expected]);
    assert.equal(f.metrics.reads.length, 2);
    assert.ok(f.metrics.reads.every(r => r.bytes < 500));
  });
}
