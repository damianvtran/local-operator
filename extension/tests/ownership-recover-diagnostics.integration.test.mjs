/* A refused `owner_recover` must say WHICH half of the fence refused, in numbers.
 *
 * The extension used to answer a bare "browser owner generation is stale" for two
 * different facts: the stored scope belongs to another SESSION, or it belongs to
 * this session but at a generation the caller never listed (its own current,
 * previous or replay chain). From the caller's side those need opposite next steps
 * and the message could not tell them apart, so a field failure could not be
 * classified without reading the peer's private state.
 *
 * What these tests pin, against the ACTUAL `withOwnership` dispatcher (not a
 * reimplementation of it):
 *  - the refusal carries a typed `refusal` plus a small structured `diagnostic`
 *    made only of booleans and counts;
 *  - nothing identifying leaks: no proof, session, generation, allocation id, tab
 *    handle or URL appears anywhere in the error;
 *  - a refusal mutates nothing, so diagnosing is never a repair, a rotation or a
 *    proof rebind;
 *  - a different session holding the proof learns nothing about the scope it does
 *    not own (no counts), and a forged proof reaches no scope at all;
 *  - the accepting paths (current, predecessor, ancestry member) are unchanged.
 *
 * Honest limit: this drives the module the extension ships from source with the
 * fixture's disposable Chrome stand-in. It proves what THIS source does, not what
 * an extension already installed in a browser does; that needs the extension to
 * be updated and reloaded by the operator.
 */
import assert from "node:assert/strict";
import test from "node:test";
import { fixture } from "./fixtures/ownership.mjs";

const OLD = "old-generation-0123456789abcdefghijkl";
const MID = "mid-generation-0123456789abcdefghijkl";

/** Open a tab under generation `g1` so the scope has one live allocation. */
async function seeded() {
  const f = await fixture();
  await f.call("open");
  return f;
}

const recover = (f, overrides) => f.call("owner_recover", { ...f.owner, ...overrides });

test("a stale refusal reports the ancestry the caller supplied and the scope's allocation counts", async () => {
  const f = await seeded();
  try {
    await assert.rejects(
      () => recover(f, { owner_generation: OLD, previous_generation: "", previous_generations: [MID, OLD] }),
      (err) => {
        assert.equal(err.code, "owner_refused");
        assert.equal(err.data.refusal, "stale");
        assert.deepEqual(err.data.diagnostic, {
          version: 1,
          session_match: true,
          request_ancestry_count: 2,
          request_previous_supplied: false,
          allocations_total: 1,
          allocations_live: 1,
          allocations_pending: 0,
          unknown_reservations: 0,
        });
        return true;
      },
    );
  } finally { await f.close(); }
});

test("an empty ancestry is distinguishable from a long one that still misses", async () => {
  const f = await seeded();
  try {
    const empty = await recover(f, { owner_generation: OLD, previous_generation: "", previous_generations: [] })
      .catch((e) => e);
    const long = await recover(f, { owner_generation: OLD, previous_generation: MID, previous_generations: [MID, OLD, "x".repeat(40)] })
      .catch((e) => e);
    assert.equal(empty.data.diagnostic.request_ancestry_count, 0);
    assert.equal(empty.data.diagnostic.request_previous_supplied, false);
    assert.equal(long.data.diagnostic.request_ancestry_count, 3);
    assert.equal(long.data.diagnostic.request_previous_supplied, true);
  } finally { await f.close(); }
});

test("a foreign-session refusal says only that the session differs, with no scope counts", async () => {
  const f = await seeded();
  try {
    await assert.rejects(
      () => recover(f, { requester: "session:someone-else" }),
      (err) => {
        assert.equal(err.data.refusal, "foreign");
        assert.deepEqual(err.data.diagnostic, { version: 1, session_match: false });
        return true;
      },
    );
  } finally { await f.close(); }
});

test("a refusal never carries a proof, session, generation, allocation id, tab or URL", async () => {
  const f = await seeded();
  try {
    const sentinelSession = "session:sentinel-session-id";
    const errors = [
      await recover(f, { requester: sentinelSession }).catch((e) => e),
      await recover(f, { owner_generation: OLD, previous_generations: [MID] }).catch((e) => e),
    ];
    for (const err of errors) {
      const wire = JSON.stringify({ message: err.message, data: err.data });
      for (const secret of [f.owner.owner_proof, f.owner.requester, sentinelSession, f.owner.owner_generation, OLD, MID,
        f.owner.allocation_id, f.owner.url, "bridge:", "example.test"]) {
        assert.ok(!wire.includes(secret), `refusal leaked ${JSON.stringify(secret)}: ${wire}`);
      }
      for (const value of Object.values(err.data.diagnostic)) {
        assert.ok(typeof value === "number" || typeof value === "boolean", "diagnostic is booleans and counts only");
      }
    }
  } finally { await f.close(); }
});

test("a refused recover mutates nothing: storage is byte-identical and the real owner still recovers", async () => {
  const f = await seeded();
  try {
    const before = JSON.stringify(f.store());
    await recover(f, { requester: "session:someone-else" }).catch(() => {});
    await recover(f, { owner_generation: OLD, previous_generations: [MID] }).catch(() => {});
    assert.equal(JSON.stringify(f.store()), before, "diagnosing must not rotate, rebind or write anything");
    const owned = await recover(f, {});
    assert.equal(owned.ownership_version, 1);
    assert.equal(owned.state, "owned", "the legitimate owner is unaffected by the refused attempts");
  } finally { await f.close(); }
});

test("a forged proof cannot introspect another scope", async () => {
  const f = await seeded();
  try {
    const forged = await recover(f, { owner_proof: "b".repeat(40) });
    assert.deepEqual(forged, { ownership_version: 1, state: "unresolved" }, "an unknown proof reaches no scope and learns no counts");
    const malformed = await recover(f, { owner_proof: "short" }).catch((e) => e);
    assert.equal(malformed.data.refusal, "foreign");
    assert.equal(malformed.data.diagnostic, undefined, "a shape refusal reads no scope, so it reports none");
  } finally { await f.close(); }
});

test("pending and unknown allocations are counted, not named", async () => {
  const f = await seeded();
  try {
    const all = f.store().ownerScopes;
    const scope = Object.values(all)[0];
    scope.allocations["pending-allocation"] = { tab: undefined, state: "allocating" };
    scope.unknownReservations = 2;
    await assert.rejects(
      () => recover(f, { owner_generation: OLD, previous_generations: [] }),
      (err) => {
        assert.equal(err.data.diagnostic.allocations_total, 2);
        assert.equal(err.data.diagnostic.allocations_live, 1);
        assert.equal(err.data.diagnostic.allocations_pending, 1);
        assert.equal(err.data.diagnostic.unknown_reservations, 2);
        assert.ok(!JSON.stringify(err.data).includes("pending-allocation"));
        return true;
      },
    );
  } finally { await f.close(); }
});

test("the accepting paths are unchanged: current, predecessor and ancestry member all recover", async () => {
  const f = await seeded();
  try {
    assert.equal((await recover(f, {})).ownership_version, 1);
    // Move the stored generation to OLD, then reach it again through each declared route.
    await recover(f, { owner_generation: OLD, previous_generation: f.owner.owner_generation, previous_generations: [] });
    const viaPrevious = await recover(f, { owner_generation: MID, previous_generation: OLD, previous_generations: [] });
    assert.equal(viaPrevious.ownership_version, 1);
    const viaAncestry = await recover(f, { owner_generation: f.owner.owner_generation, previous_generation: "", previous_generations: [MID] });
    assert.equal(viaAncestry.ownership_version, 1);
  } finally { await f.close(); }
});
