/* Per-command session-storage accounting for the shipped command path.
 *
 * Why this exists: every command used to pay a whole-map `get(surfaces)` AND a
 * whole-map `set(surfaces)` for a recency stamp nobody branches on (the `tabs`
 * listing's sort), on top of a full `ownerScopes` read plus a no-op write for
 * every owned command. All of those mutations are serialized through ONE
 * module-global lane (`state.ts` `withStore`), so the cost showed up as a
 * staircase proportional to how many sessions were talking at once — measured
 * 603/1004/1405/1807 ms for four concurrent commands against a 100 ms
 * artificial per-op delay, i.e. the multi-second-to-20 s stall that arrives as
 * "the bridge stopped answering". These rows pin the arithmetic that fixes:
 * three session round trips is now the ceiling for a drove-this-tab command,
 * and a no-op `ownerScopes` write must not happen at all.
 *
 * Chrome is a synthetic host with counters (`tests/fixtures/worker-performance.mjs`,
 * the fixture the popup work added): no browser, no tabs, no credentials.
 * Transport is never touched — the command modules are bundled and called
 * directly, exactly as the worker's dispatch calls them, so what is counted is
 * the command's own IPC, not a socket's.
 *
 * LOCAL_OPERATOR_TEST_BRIDGE_SOURCE is the SAME review-only hook
 * `bridge-wedge.integration.test.mjs` reads: it points the module under test at
 * another tree (the pinned base, for the before/after comparison below) and is
 * read by nothing in `src/` or the build, so it cannot become a production
 * seam.
 */
import assert from "node:assert/strict";
import test from "node:test";
import { build } from "esbuild";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { installWorkerPerformance } from "./fixtures/worker-performance.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const SRC = process.env.LOCAL_OPERATOR_TEST_BRIDGE_SOURCE || resolve(HERE, "..", "src");

const TAB = "bridge:100:abcdef";
/** A real owner proof shape: 32+ chars of [A-Za-z0-9_-] (ownership.identity). */
const PROOF = "proof-proof-proof-proof-proof-proof-01";
const OWNER = { requester: "session:s1", owner_proof: PROOF, owner_generation: "g1" };

/** The counting fixture, plus the three APIs a DRIVEN command needs.
 *
 * `installWorkerPerformance` counts storage and action calls and deliberately
 * has no tabs ("No real tabs in fixture"), because the popup never drives one.
 * The command path does, so this layers the minimum on top rather than
 * duplicating the counters: a tab that answers, a debugger that answers, and a
 * page script that returns text. Nothing here models timing or geometry. */
function installCommandHost() {
  const fixture = installWorkerPerformance({ port: 4099, id: "test", token: "test-token" });
  const chrome = globalThis.chrome;
  chrome.tabs.get = async (id) => ({
    id, windowId: 1, groupId: -1, url: "https://example.test/page", title: "P", active: false,
  });
  chrome.tabs.query = async () => [];
  chrome.debugger.sendCommand = async (_target, method) =>
    method === "Accessibility.getFullAXTree" ? { nodes: [] } : {};
  chrome.scripting.executeScript = async () => [{ result: "page text" }];
  return fixture;
}

/** The commands under test, bundled out of the tree under measurement. */
async function loadCommands() {
  const dir = await mkdtemp(join(tmpdir(), "lop-ipc-budget-"));
  const entry = join(dir, "entry.mjs");
  await writeFile(
    entry,
    [
      `export { readPage } from ${JSON.stringify(join(SRC, "commands", "read.ts"))};`,
      `export { screenshot } from ${JSON.stringify(join(SRC, "commands", "shot.ts"))};`,
      `export { snapshot } from ${JSON.stringify(join(SRC, "commands", "snapshot.ts"))};`,
      `export { status, tabs, close } from ${JSON.stringify(join(SRC, "commands", "nav.ts"))};`,
      `export { touchSurface } from ${JSON.stringify(join(SRC, "state.ts"))};`,
      `export { requireSurface } from ${JSON.stringify(join(SRC, "cdp.ts"))};`,
      `export { withOwnership, recordAllocation } from ${JSON.stringify(join(SRC, "ownership.ts"))};`,
    ].join("\n"),
  );
  const outfile = join(dir, "bundle.mjs");
  await build({
    entryPoints: [entry], bundle: true, platform: "node", format: "esm", outfile,
  });
  return { loaded: await import(pathToFileURL(outfile) + `?${Date.now()}`), dir };
}

/** The throttle's own interval, bundled SEPARATELY from the command entry.
 *
 * It cannot ride in the entry above: the pinned BASE tree has no
 * `TOUCH_INTERVAL_MS` at all (its stamp was unthrottled), and ONE unresolvable
 * named export fails the whole esbuild bundle — so the before/after table below
 * could not be measured at all, and every row failed before counting anything
 * (review R2). Only the throttle row needs the constant, so only that row pays
 * for a second bundle, and a tree that does not export it answers `undefined`
 * here instead of taking the file down with it. */
async function loadTouchInterval() {
  const dir = await mkdtemp(join(tmpdir(), "lop-ipc-budget-interval-"));
  try {
    const entry = join(dir, "entry.mjs");
    await writeFile(
      entry,
      `export { TOUCH_INTERVAL_MS } from ${JSON.stringify(join(SRC, "state.ts"))};`,
    );
    const outfile = join(dir, "bundle.mjs");
    try {
      // `logLevel: "silent"` because the BASE tree answers this one with a
      // missing-export error that is the expected outcome there: esbuild still
      // throws, but the before/after run should not print a red error block in
      // the middle of the numbers it is there to produce.
      await build({
        entryPoints: [entry],
        bundle: true,
        platform: "node",
        format: "esm",
        outfile,
        logLevel: "silent",
      });
    } catch {
      return undefined;
    }
    const mod = await import(pathToFileURL(outfile) + `?${Date.now()}`);
    return mod.TOUCH_INTERVAL_MS;
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}

/** Session-storage round trips for ONE command — what the single serialized
 * store lane costs, which is the whole quantity under test. */
function sessionTrips(metrics, reads, writes) {
  return [
    ...metrics.reads.slice(reads).map((r) => `session.get(${r.keys})`),
    ...metrics.writes.slice(writes).map((w) => `session.set(${w.keys})`),
  ].filter((call) => call.startsWith("session."));
}
const ownerScopesWrites = (metrics, before) =>
  metrics.writes.slice(before).filter((w) => w.keys.join(",") === "ownerScopes");

/** Seed one driven surface (and, for owned rows, the matching owner scope). */
function seed(fixture, { owned = false } = {}) {
  fixture.session.surfaces = {
    [TAB]: {
      tabId: 100, nonce: "abcdef", epoch: 0, createdAt: 0, lastUsedAt: 0,
      ownerKey: "session:s1", groupBaseLabel: "LO · 1", groupOrdinal: 1,
      ...(owned ? { allocationId: "alloc-1" } : {}),
    },
  };
  fixture.session.refs = {};
  fixture.session.ownerScopes = owned
    ? { [PROOF]: { session: "session:s1", generation: "g1", allocations: { "alloc-1": { tab: TAB, state: "allocated" } } } }
    : {};
}

/** Drive one command the way the worker's dispatch does: through the ownership
 * lane for an owned caller, straight through for a legacy one. Returns the
 * command's session round trips and any `ownerScopes` writes it made. */
async function drive(loaded, fixture, method, { owned = false } = {}) {
  seed(fixture, { owned });
  const params = owned ? { ...OWNER, tab: TAB, allocation_id: "alloc-1" } : { tab: TAB };
  const reads = fixture.metrics.reads.length;
  const writes = fixture.metrics.writes.length;
  const handler = () => loaded[method === "read" ? "readPage" : method](params);
  const result = owned
    ? await loaded.withOwnership(method, params, handler, loaded.close)
    : await handler();
  return {
    result,
    trips: sessionTrips(fixture.metrics, reads, writes),
    ownerScopesWrites: ownerScopesWrites(fixture.metrics, writes),
  };
}

// [label, handler, owned, ceiling on session round trips]
//
// The ceiling is 3 for a command that merely DRIVES a tab: one read of the
// surface map is the only storage the command itself needs. `snapshot` is the
// documented exception at 4, because it PUBLISHES its refs and that map is a
// shared read-modify-write of its own (`refs`: get + set) — real work, not
// overhead. Measured with THIS file against the pinned base (05e959a00), i.e. a
// plain `LOCAL_OPERATOR_TEST_BRIDGE_SOURCE=<base>/extension/src` run of it: the
// counting row prints all ten numbers before it asserts anything, so the before
// column below IS that run's own `t.diagnostic` line, not a separate probe:
//
//   read            3 -> 1     screenshot        3 -> 1
//   owned read      5 -> 2     owned screenshot  5 -> 2
//   snapshot        5 -> 3     owned snapshot    7 -> 4
//   status          3 -> 1     owned status      5 -> 2
//   tabs            1 -> 1     owned tabs        3 -> 2
//
// (`screenshot`'s before is **3**, not the 5 this table first claimed — it
// resolves no handle, so its recency write was the whole of its overhead, and
// the head column, 1, was right — review R2.) Rows other than the counting one
// are GUARDS, not measurements: the `requireSurface stamps nothing` row and the
// throttle row are both expected to fail on the base tree, which is exactly what
// they are for, and the throttle row is skipped there outright because the base
// exports no interval to throttle to.
//
// `tabs` is the one row with nothing to remove: it lists surfaces from a single
// read and never resolved a handle, so it never paid the recency write. It is
// kept as the control — a command whose count must NOT move.
const COMMANDS = [
  ["read", "read", false, 3],
  ["owned read", "read", true, 3],
  ["screenshot", "screenshot", false, 3],
  ["owned screenshot", "screenshot", true, 3],
  ["snapshot", "snapshot", false, 4],
  ["owned snapshot", "snapshot", true, 4],
  ["status", "status", false, 3],
  ["owned status", "status", true, 3],
  ["tabs", "tabs", false, 3],
  ["owned tabs", "tabs", true, 3],
];

test("a drove-this-tab command costs at most three session round trips", async (t) => {
  const fixture = installCommandHost();
  const { loaded, dir } = await loadCommands();
  try {
    // Measure ALL ten rows before asserting any of them: the numbers ARE the
    // before/after evidence this file carries, and a base run stops at the first
    // ceiling it exceeds — asserting inside the measuring loop printed only the
    // rows that happened to pass, leaving the table underivable (review R2).
    const measured = {};
    for (const [label, method, owned] of COMMANDS) {
      const { result, trips } = await drive(loaded, fixture, method, { owned });
      assert.ok(result, `${label} must answer`);
      measured[label] = trips;
    }
    t.diagnostic(`session round trips per command: ${JSON.stringify(measured)}`);
    for (const [label, , , ceiling] of COMMANDS) {
      assert.ok(
        measured[label].length <= ceiling,
        `${label} made ${measured[label].length} session round trips (ceiling ${ceiling}): ${measured[label].join(", ")}`,
      );
    }
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test("requireSurface stamps nothing: the recency write left the command path", async () => {
  // The sharpest form of the first fix. `requireSurface` is on EVERY command's
  // critical path, and its stamp used to be a whole-map get AND set — two of the
  // three session trips above, on the single lane every other session queues
  // behind. Pre-fix this row sees a set; post-fix it must see the read alone.
  const host = installCommandHost();
  const { loaded, dir } = await loadCommands();
  try {
    seed(host);
    const reads = host.metrics.reads.length;
    const writes = host.metrics.writes.length;
    const surface = await loaded.requireSurface(TAB);
    assert.equal(surface.tabId, 100);
    assert.deepEqual(
      host.metrics.reads.slice(reads).map((r) => [r.area, r.keys]),
      [["session", ["surfaces"]]],
    );
    assert.deepEqual(host.metrics.writes.slice(writes), [], "a handle lookup wrote storage");
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test("the recency stamp is throttled to one write per interval", async (t) => {
  const interval = await loadTouchInterval();
  if (interval === undefined) {
    // The pinned base has no interval: its stamp ran on every command, which is
    // the "before" half of the first fix rather than a failure of this file.
    t.skip("the tree under measurement exports no TOUCH_INTERVAL_MS (unthrottled stamp)");
    return;
  }
  const host = installCommandHost();
  const { loaded, dir } = await loadCommands();
  try {
    seed(host);
    const writes = () => host.metrics.writes.filter((w) => w.keys.join(",") === "surfaces").length;
    const start = 1_000_000;
    await loaded.touchSurface(TAB, start);
    assert.equal(writes(), 1, "the first stamp must land");
    await loaded.touchSurface(TAB, start + 1);
    await loaded.touchSurface(TAB, start + interval - 1);
    assert.equal(writes(), 1, "stamps inside the interval must be skipped");
    await loaded.touchSurface(TAB, start + interval);
    assert.equal(writes(), 2, "the next interval must stamp again");
    // …and it still honours the presence check: a pruned surface is never
    // resurrected by a late stamp (the m5 property the write itself must keep),
    // which here means the write does not run at all.
    delete host.session.surfaces[TAB];
    await loaded.touchSurface(TAB, start + 2 * interval);
    assert.equal(writes(), 2, "a stamp for a pruned surface must not write");
    assert.deepEqual(host.session.surfaces, {});
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test("no owned command performs a no-op ownerScopes write", async (t) => {
  // The second fix. Every owned command funnels through
  // `mutate(params, value => value, …)`, a pure validation read, and that read
  // re-wrote the whole ownerScopes map: a get plus a no-op set per command, per
  // session, on the same shared lane. The map on disk must come out
  // byte-identical and untouched — while a command that DOES change it must
  // still write, which the row below pins so this guard cannot pass by never
  // writing at all.
  const fixture = installCommandHost();
  const { loaded, dir } = await loadCommands();
  try {
    const unwritten = [];
    for (const [label, method, owned] of COMMANDS.filter((row) => row[2])) {
      seed(fixture, { owned: true });
      const seeded = JSON.stringify(fixture.session.ownerScopes);
      const writes = fixture.metrics.writes.length;
      const { ownerScopesWrites: wrote } = await drive(loaded, fixture, method, { owned });
      assert.deepEqual(wrote, [], `${label} re-wrote ownerScopes without changing it`);
      assert.equal(JSON.stringify(fixture.session.ownerScopes), seeded);
      unwritten.push(label);
    }
    t.diagnostic(`owned commands that wrote nothing: ${unwritten.join(", ")}`);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test("a command that DOES change the owner scope still writes it", async () => {
  const host = installCommandHost();
  const { loaded, dir } = await loadCommands();
  try {
    seed(host, { owned: true });
    const writes = host.metrics.writes.length;
    await loaded.recordAllocation(
      { ...OWNER, allocation_id: "alloc-1" },
      TAB,
      "cleanup_pending",
    );
    const scopes = ownerScopesWrites(host.metrics, writes);
    assert.equal(scopes.length, 1, "a real mutation must be persisted");
    assert.equal(
      host.session.ownerScopes[PROOF].allocations["alloc-1"].state,
      "cleanup_pending",
    );
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});
