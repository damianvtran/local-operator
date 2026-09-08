/* Retiring an ended scope is a TWO-SIDED fence, and this side was the one left up.
 *
 * The session's durable record and the extension's `scope.terminal` both have
 * to be retired for a resumed owner to browse again. The extension used to
 * clear its half only when the generation string CHANGED, which an in-process
 * child never does — that path deliberately reuses one generation per session
 * so a second live instance cannot fence the incumbent — so `open` was refused
 * forever while `owner_recover` reported the scope settled and ready.
 */
import assert from "node:assert/strict";
import test from "node:test";
import { build } from "esbuild";
import { pathToFileURL } from "node:url";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

async function loadModule() {
  const session = new Map();
  globalThis.chrome = {
    storage: { session: {
      get: async (keys) => {
        const out = {};
        for (const key of Array.isArray(keys) ? keys : [keys]) {
          if (session.has(key)) out[key] = session.get(key);
        }
        return out;
      },
      set: async (obj) => { for (const [k, v] of Object.entries(obj)) session.set(k, v); },
    } },
    tabs: {
      get: async (id) => ({ id, url: "https://example.com/", title: "T" }),
      onRemoved: { addListener: () => {} }, onReplaced: { addListener: () => {} },
      onUpdated: { addListener: () => {} },
    },
    debugger: {
      attach: async () => {}, detach: async () => {}, sendCommand: async () => ({}),
      onEvent: { addListener: () => {} }, onDetach: { addListener: () => {} },
    },
    runtime: { getURL: (p) => p, getManifest: () => ({ version: "0.1.9" }), onMessage: { addListener: () => {} } },
  };
  const dir = await mkdtemp(join(tmpdir(), "lop-ownership-resume-"));
  const outfile = join(dir, "module.mjs");
  await build({ entryPoints: ["src/ownership.ts"], bundle: true, platform: "node", format: "esm", outfile });
  return {
    loaded: await import(pathToFileURL(outfile) + `?${Date.now()}`),
    close: () => rm(dir, { recursive: true, force: true }),
  };
}

const GEN = "reused-generation-0123456789abcdefgh";
const OWNER = {
  owner_proof: "proof-abcdefghijklmnopqrstuvwxyz0123456789",
  requester: "session:child",
  owner_generation: GEN,
  previous_generation: "",
  previous_generations: [GEN],
  allocation_id: "alloc-1",
};

/** Run one owner command against the real dispatcher. */
const run = (mod, method, params) =>
  mod.withOwnership(method, params,
    async () => ({ tab: "bridge:7:nonce", url: "", title: "", state: "owned" }),
    async () => ({ state: "closed" }));

async function settle(mod) {
  await run(mod, "open", OWNER);
  await run(mod, "owner_finish", { ...OWNER, outcome: "completed" });
}

test("a resumed owner may open again on the SAME generation", async (t) => {
  const { loaded, close } = await loadModule();
  t.after(close);
  await settle(loaded);

  const recovered = await run(loaded, "owner_recover", { ...OWNER, resumed_scope: true });
  assert.equal(recovered.terminal, "", "recover must retire the ended scope");
  const reopened = await run(loaded, "open", OWNER);
  assert.equal(reopened.tab, "bridge:7:nonce");
});

test("an owner that is NOT resuming keeps its terminal intent", async (t) => {
  const { loaded, close } = await loadModule();
  t.after(close);
  await settle(loaded);

  const recovered = await run(loaded, "owner_recover", { ...OWNER });
  assert.equal(recovered.terminal, "completed");
  await assert.rejects(() => run(loaded, "open", OWNER), /browser scope ended/);
});

test("a foreign session cannot clear someone else's terminal by claiming a resume", async (t) => {
  const { loaded, close } = await loadModule();
  t.after(close);
  await settle(loaded);

  // The flag is only reachable PAST the proof/session/generation check, so it
  // grants a stranger nothing: this is refused before the clear is considered.
  await assert.rejects(
    () => run(loaded, "owner_recover", { ...OWNER, requester: "session:attacker", resumed_scope: true }),
    /stale/,
  );
  const stale = "stale-generation-0123456789abcdefgh";
  await assert.rejects(
    () => run(loaded, "owner_recover", {
      ...OWNER, owner_generation: stale, previous_generations: [stale], resumed_scope: true,
    }),
    /stale/,
  );
  const recovered = await run(loaded, "owner_recover", { ...OWNER });
  assert.equal(recovered.terminal, "completed", "the real owner's intent survived");
});
