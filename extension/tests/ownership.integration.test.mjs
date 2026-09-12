import assert from "node:assert/strict";
import test from "node:test";
import { fixture } from "./fixtures/ownership.mjs";

for (const fault of ["navigation", "attach", "capture"]) {
  test(`new allocation rolls back ${fault} failure`, async () => {
    const f = await fixture();
    try { f.faults[fault]=true; await assert.rejects(f.call("open")); assert.equal(f.tabs.size,0); assert.equal(Object.keys(f.store().surfaces).length,0); assert.equal(f.removed.length,1); }
    finally { await f.close(); }
  });
}

test("failed rollback remains recoverable and cleanup retries exactly one owner", async () => {
  const f = await fixture();
  try {
    f.faults.navigation=true; f.faults.remove=true;
    await assert.rejects(f.call("open"),e=>e.data.cleanup_pending===true);
    const recovered=await f.call("owner_recover"); assert.ok(recovered.tab); assert.equal(recovered.state,"cleanup_pending");
    assert.equal((await f.call("owner_finish",{...f.owner,outcome:"failed"})).state,"pending");
    assert.equal(f.tabs.size,1); f.faults.remove=false;
    assert.equal((await f.call("owner_finish",{...f.owner,outcome:"failed"})).state,"closed"); assert.equal(f.tabs.size,0);
  } finally { await f.close(); }
});

test("lost response replays allocation and failed existing navigation never closes", async () => {
  const f=await fixture();
  try {
    const first=await f.call("open"); const replay=await f.call("open"); assert.equal(first.tab,replay.tab); assert.equal(f.tabs.size,1);
    f.faults.navigation=true; await assert.rejects(f.call("open",{...f.owner,tab:first.tab})); assert.equal(f.tabs.size,1); assert.equal(f.removed.length,0);
  } finally { await f.close(); }
});

test("new generation fences stale finalizers and foreign proof", async () => {
  const f=await fixture();
  try {
    const first=await f.call("open");
    const resumed={...f.owner,owner_generation:"g2",previous_generation:"g1"};
    assert.equal((await f.call("owner_recover",resumed)).tab,first.tab);
    await assert.rejects(f.call("owner_finish"),/generation is stale/);
    await assert.rejects(f.call("close",{...resumed,owner_proof:"b".repeat(40),tab:first.tab}));
    await assert.rejects(f.call("close",{tab:first.tab}),/owner-aware client required/);
    await assert.rejects(f.call("close",{}),/owner-aware client required/);
    assert.equal(f.tabs.size,1);
  } finally { await f.close(); }
});

test("legacy clients still close their own sole legacy allocation", async () => {
  const f=await fixture();
  try {
    await f.call("open",{url:"https://example.test/"});
    await f.call("close",{});
    assert.equal(f.tabs.size,0);
  } finally { await f.close(); }
});

test("retention survives completion and release enables exact cleanup", async () => {
  const f=await fixture();
  try {
    await f.call("open"); await f.call("owner_retain",{...f.owner,reason:"pending login"});
    assert.equal((await f.call("owner_finish",{...f.owner,outcome:"completed"})).state,"retained"); assert.equal(f.tabs.size,1);
    await f.call("owner_release"); assert.equal((await f.call("owner_finish")).state,"closed"); assert.equal(f.tabs.size,0);
  } finally { await f.close(); }
});

test("concurrent owners honor eight-tab capacity without hijacking live owners", async () => {
  const f=await fixture();
  try {
    const results=await Promise.allSettled(Array.from({length:10},(_,i)=>f.call("open",{...f.owner,owner_proof:String(i).repeat(40),requester:`session:synthetic-${i}`})));
    assert.equal(results.filter(r=>r.status==="fulfilled").length,8); assert.equal(f.tabs.size,8); assert.equal(f.removed.length,0);
  } finally { await f.close(); }
});

test("worker restart recovers exact allocation; browser restart never adopts numeric IDs", async () => {
  const f=await fixture();
  try {
    const first=await f.call("open"); await f.restart();
    assert.equal((await f.call("owner_recover")).tab,first.tab);
    f.resetStorage(); await f.restart();
    assert.equal((await f.call("owner_recover")).state,"unresolved");
    assert.equal(f.tabs.size,1); assert.equal(f.removed.length,0);
  } finally { await f.close(); }
});

test("normal close and user-closed stale pool release only confirmed missing tabs", async () => {
  const f=await fixture();
  try {
    const first=await f.call("open"); await f.call("close",{...f.owner,tab:first.tab});
    assert.equal(f.tabs.size,0);
    for(let i=0;i<8;i++) await f.call("open",{...f.owner,owner_proof:String(i).repeat(40),requester:`session:synthetic-${i}`});
    f.tabs.delete(101); // user closes one owned tab; listing performs stale reconciliation
    await f.call("tabs",f.owner);
    await f.call("open",{...f.owner,allocation_id:"replacement"});
    assert.equal(f.tabs.size,8);
  } finally { await f.close(); }
});

test("completion queued during allocation leaves no late tab", async () => {
  const f=await fixture();
  try { const opening=f.call("open"); const finishing=f.call("owner_finish",{...f.owner,outcome:"completed"}); await opening; assert.equal((await finishing).state,"closed"); assert.equal(f.tabs.size,0); }
  finally { await f.close(); }
});

// --- Audit A4 / scoping D3: the admission window -----------------------------

/** Wait until `predicate` holds, without asserting a duration. */
async function until(predicate, ms = 1000) {
  const started = Date.now();
  while (Date.now() - started < ms) {
    if (predicate()) return true;
    await new Promise((r) => setTimeout(r, 5));
  }
  return predicate();
}

test("the cap read and the slot write are one admission window (D3)", async () => {
  // The defect is the classic read-then-write race: two owners at 7 surfaces
  // both read "one slot free" and both take it. The lane that used to prevent
  // it spanned the WHOLE open handler (and with it another owner's navigation);
  // it now spans exactly cap-read -> create -> putSurface.
  const f = await fixture();
  try {
    for (let i = 0; i < 7; i++) {
      await f.call("open", {
        ...f.owner,
        owner_proof: String(i).repeat(40),
        requester: `session:synthetic-${i}`,
      });
    }
    assert.equal(Object.keys(f.store().surfaces).length, 7, "precondition: one slot left");

    const results = await Promise.allSettled([
      f.call("open", { ...f.owner, owner_proof: "x".repeat(40), requester: "session:x" }),
      f.call("open", { ...f.owner, owner_proof: "y".repeat(40), requester: "session:y" }),
    ]);

    assert.equal(results.filter((r) => r.status === "fulfilled").length, 1, "exactly one open wins");
    assert.equal(
      results.filter((r) => r.status === "rejected" && r.reason.code === "tab_limit").length,
      1,
      "the loser gets a typed refusal, not a silent ninth tab",
    );
    assert.equal(Object.keys(f.store().surfaces).length, 8);
    assert.equal(f.tabs.size, 8);
  } finally {
    await f.close();
  }
});

test("a parked owner's open does not block another owner's admission (D3)", async () => {
  // The audit's repro, inverted: owner A is parked INSIDE its navigation (a
  // slow page or a human origin prompt), which under the old whole-handler
  // global lane also held admission. Owner B must complete while A is still
  // parked, which cannot happen if B is queued behind A.
  const f = await fixture();
  try {
    let release;
    f.faults.navigationGate = new Promise((r) => {
      release = r;
    });
    const parked = f.call("open");
    assert.ok(
      await until(() => Object.keys(f.store().surfaces ?? {}).length === 1),
      "owner A consumed its slot and is now parked in navigate",
    );

    const other = await Promise.race([
      f.call("open", { ...f.owner, owner_proof: "b".repeat(40), requester: "session:b" }),
      new Promise((_, reject) =>
        setTimeout(() => reject(new Error("owner B's open is parked behind owner A's navigation")), 750),
      ),
    ]);
    assert.ok(other.tab, "owner B was admitted and finished while A was parked");

    release();
    await parked;
    assert.equal(f.tabs.size, 2);
  } finally {
    f.faults.navigationGate = null;
    await f.close();
  }
});

test("a same-owner finish cannot overtake its own parked open (D3)", async () => {
  // The per-owner lane must survive the narrowing: `owner_finish` racing its own
  // in-flight `open` would otherwise let a late navigation resurrect a tab the
  // session has already retired.
  const f = await fixture();
  try {
    let release;
    f.faults.navigationGate = new Promise((r) => {
      release = r;
    });
    const opening = f.call("open");
    assert.ok(await until(() => Object.keys(f.store().surfaces ?? {}).length === 1));

    const finishing = f.call("owner_finish", { ...f.owner, outcome: "completed" });
    const overtook = await Promise.race([
      finishing.then(() => true),
      new Promise((r) => setTimeout(() => r(false), 150)),
    ]);
    assert.equal(overtook, false, "owner_finish overtook its own open");

    release();
    await opening;
    assert.equal((await finishing).state, "closed");
    assert.equal(f.tabs.size, 0);
  } finally {
    f.faults.navigationGate = null;
    await f.close();
  }
});
