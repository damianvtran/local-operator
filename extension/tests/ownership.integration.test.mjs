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
