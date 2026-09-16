/* The shipped popup under synthetic Chrome storage and health. Proves paint
 * ORDER and IPC shape (reads, bytes, mutations) — never geometry, which
 * popup-performance-server.mjs renders from the real markup instead.
 *
 * LOCAL_OPERATOR_TEST_POPUP_SOURCE is a REVIEW-ONLY hook: it points the module
 * and markup under test at another tree (the pinned base, for a before/after
 * comparison) and is read by nothing in src/ or the build, so it cannot become a
 * production seam. It is deliberately named outside the `CMUX_*`/`LOP_*`
 * families the isolation wrappers scrub — a base-comparison run that inherited
 * that scrub would silently test the checked-out tree and read as a pass. */
import { test } from "node:test";
import assert from "node:assert/strict";
import { build } from "esbuild";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { installPopupDom } from "./fixtures/popup-dom.mjs";
import { installPopupPerformance } from "./fixtures/popup-performance.mjs";

const source = resolve(process.env.LOCAL_OPERATOR_TEST_POPUP_SOURCE || "src");
const html = await readFile(resolve(source, "popup/popup.html"), "utf8");
const { outputFiles } = await build({ entryPoints: [resolve(source, "popup/popup.ts")], bundle: true, format: "esm", platform: "node", write: false });
let sequence = 0;
// A BOUND, not the fixture's exact byte total: the point of the assertion is
// that the projection is narrow (the base's is 2,875,834 bytes on the large
// fixture, and its first read alone carries `refs` and the grant map), and a
// fixture edit that changes the payload should not fail this row opaquely. The
// measured figure for the healthy row is 125 bytes — 35 from the local read
// ({"port":4099,"allowAllSites":false}) plus 90 from the session read's own
// projection — and it is recorded as evidence in the PR, not asserted here.
const POPUP_STATE_BYTES_MAX = 500;
const flush = () => new Promise(resolve => setImmediate(resolve));
const barrier = () => { let release; const promise = new Promise(resolve => { release = resolve; }); return { promise, release }; };

/* The #disconnected card is the one card reached by TWO different failures, and
 * it carries a copy channel per failure (popup.html). These read it the way the
 * popup writes it. Both ids are looked up by name, so a tree without the second
 * copy channel fails HERE rather than on a string comparison somewhere else. */
const copyShown = (f, id) => {
  const node = f.nodes.get(id);
  assert.ok(node, `popup.html must carry #${id}`);
  return !node.classList.contains("hidden");
};
const assertUncheckedCopy = (f) => {
  assert.ok(copyShown(f, "disconnected-title-unchecked"), "the unchecked title must be the one shown");
  assert.ok(copyShown(f, "disconnected-sub-unchecked"), "the unchecked sentence must be the one shown");
  assert.ok(!copyShown(f, "disconnected-title-measured"), "a failed state read must not assert the diagnosis");
  assert.ok(!copyShown(f, "disconnected-sub-measured"), "a failed state read must not assert the diagnosis");
};
const assertMeasuredCopy = (f) => {
  assert.ok(copyShown(f, "disconnected-title-measured"), "a probe that answered nothing may diagnose");
  assert.ok(copyShown(f, "disconnected-sub-measured"));
  assert.ok(!copyShown(f, "disconnected-title-unchecked"), "the not-measured title must not survive the next render");
  assert.ok(!copyShown(f, "disconnected-sub-unchecked"));
};

/** The paragraph's rendered text, read out of the shipped markup. */
function copyText(id) {
  const at = html.indexOf(`id="${id}"`);
  assert.ok(at > -1, `popup.html must carry #${id}`);
  const end = html.indexOf("</p>", at);
  return html.slice(at, end).replace(/<[^>]*>/g, " ").replace(/\s+/g, " ").trim();
}
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
  const total = f.metrics.reads.reduce((n, r) => n + r.bytes, 0);
  assert.ok(total < POPUP_STATE_BYTES_MAX, `expected the two-read projection, read ${total} bytes`);
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
    assertUncheckedCopy(f);
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
  assertUncheckedCopy(f);
});

test("a probe that answered nothing keeps the measured diagnosis after a failed read did not", async t => {
  // The swap must not be one-way, and this is the sequence that would expose it:
  // a failed read paints the not-measured copy, the read recovers, and the card
  // is now the MEASURED one. A fix that only ever wrote the honest sentence
  // would replace a false diagnosis with a false "can't tell" on the most common
  // real state (no daemon answering) — so the diagnosis has to come back.
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const blocked = barrier();
  let stall = true;
  const f = await load({ state: "unreachable", gate: read => stall && read.area === "session" ? blocked.promise : undefined });
  await flush();
  t.mock.timers.tick(5000); await flush();
  assert.deepEqual(f.visible(), ["disconnected"]);
  assertUncheckedCopy(f);
  stall = false;
  f.nodes.get("retry").click(); await flush();
  assert.deepEqual(f.visible(), ["disconnected"], "the retried read succeeded and the probe answered nothing");
  assertMeasuredCopy(f);
  blocked.release(); await flush();
  t.mock.timers.reset();
});

for (const id of ["disconnected-title-unchecked", "disconnected-sub-unchecked"]) {
  test(`#${id} states what failed without diagnosing the daemon`, () => {
    const copy = copyText(id);
    assert.doesNotMatch(copy, /isn'?t reachable|make sure it'?s running|not reachable/i,
      "a render that failed to gather its own state never measured reachability");
    assert.match(copy, /couldn'?t (check|complete)/i, "it must say what did not happen");
    if (id.endsWith("sub-unchecked")) {
      assert.match(copy, /lop browser status/, "the CLI affordance must survive the swap");
      assert.match(copy, /retry/i, "and so must the instruction to use the button below");
    }
  });
}

test("the diagnostic copy is still shipped for the measured path", () => {
  const measured = copyText("disconnected-sub-measured");
  assert.match(measured, /lop browser status/, "#996's affordance is not what D1 removed");
  assert.match(measured, /reachable/i);
});

test("the healthy card shows the driven URL the daemon reports", async () => {
  // The popup reads `health.current_url`; a fixture carrying some other key
  // renders the empty-trough variant and hides the URL-populated card from every
  // capture (design D2). Pinned to the fixture's own payload, not to a literal.
  const f = await load();
  await flush();
  assert.deepEqual(f.visible(), ["connected"]);
  assert.equal(f.nodes.get("connected-detail").textContent, f.health.current_url);
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
