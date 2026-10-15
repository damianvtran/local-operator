/* Actual source module, disposable Chrome only. No installed extension proof. */
import assert from "node:assert/strict";
import test from "node:test";
import { fixture } from "./fixtures/ownership.mjs";

test("read-only snapshots distinguish current/previous/ancestry without changing any bytes", async () => {
  const f = await fixture();
  try {
    await f.call("open");
    const before = JSON.stringify(f.store());
    const original = globalThis.chrome.storage.session.set;
    globalThis.chrome.storage.session.set = async () => { throw Error("inspection tried storage write"); };
    for (const [overrides, expected] of [
      [{}, [true, false, false]],
      [{ owner_generation: "new", previous_generation: "g1" }, [false, true, false]],
      [{ owner_generation: "new", previous_generations: ["g1"] }, [false, false, true]],
      [{ owner_generation: "new", previous_generations: ["other"] }, [false, false, false]],
    ]) {
      const result = await f.call("owner_inspect", { ...f.owner, ...overrides });
      assert.deepEqual([result.generation_current, result.generation_previous, result.generation_ancestry], expected);
      assert.equal(result.session_match, true);
      assert.equal(result.scope_missing, false);
      assert.equal(JSON.stringify(f.store()), before);
    }
    globalThis.chrome.storage.session.set = original;
  } finally { await f.close(); }
});

test("foreign, unknown and legacy callers cannot reach handler or foreign scope fields", async () => {
  const f = await fixture();
  try {
    await f.call("open");
    const before = JSON.stringify(f.store());
    assert.deepEqual(await f.call("owner_inspect", { ...f.owner, requester: "session:foreign" }), { version: 1, session_match: false });
    assert.deepEqual(await f.call("owner_inspect", { ...f.owner, owner_proof: "z".repeat(40) }), { version: 1, scope_missing: true, session_match: false });
    let called = false;
    for (const overrides of [
      { owner_proof: undefined }, { owner_proof: "short" }, { requester: {} },
      { owner_generation: "x".repeat(129) }, { owner_proof: "x".repeat(129) },
      { previous_generations: Array(17).fill("g1") }, { previous_generations: [null] },
      { previous_generations: "g1" }, { previous_generation: {} },
    ]) {
      await assert.rejects(() => f.loaded.withOwnership("owner_inspect", { ...f.owner, ...overrides }, async () => { called = true; return {}; }, async () => { called = true; return {}; }), error => error.code === "owner_refused");
    }
    assert.equal(called, false);
    assert.equal(JSON.stringify(f.store()), before);
  } finally { await f.close(); }
});

test("parallel inspection neither queues behind an owner operation nor changes its stale fence", async () => {
  const f = await fixture();
  try {
    await f.call("open");
    const before = JSON.stringify(f.store());
    let release;
    const gate = new Promise(resolve => { release = resolve; });
    let entered;
    const started = new Promise(resolve => { entered = resolve; });
    const pending = f.loaded.withOwnership("read", f.owner, async () => { entered(); await gate; return {}; }, async () => ({}));
    await started;
    try {
      const results = await Promise.all(Array.from({ length: 20 }, () => f.call("owner_inspect", { ...f.owner, owner_generation: "stale" })));
      assert.ok(results.every(result => result.generation_current === false));
      assert.equal(JSON.stringify(f.store()), before);
    } finally { release(); await pending; }
    await assert.rejects(() => f.call("read", { ...f.owner, owner_generation: "stale" }), error => error.code === "owner_refused");
    assert.equal(JSON.stringify(f.store()), before);
  } finally { await f.close(); }
});


test("a stalled read hits the actual shipped storage deadline without any write", { timeout: 8000 }, async () => {
  const f = await fixture({ realDeadline: true });
  try {
    await f.call("open");
    const before = JSON.stringify(f.store());
    globalThis.chrome.storage.session.get = () => new Promise(() => {});
    globalThis.chrome.storage.session.set = async () => { throw Error("unexpected write"); };
    await assert.rejects(() => f.call("owner_inspect"), error => error.code === "internal" && error.data.stalled === "chrome.storage.session.get(ownerScopes)");
    assert.equal(JSON.stringify(f.store()), before);
  } finally { await f.close(); }
});
