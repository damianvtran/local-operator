/* Headless-node pairing from Chrome managed policy (src/managed-pairing.ts).
 *
 * Pins the contract `lop browser setup --node` relies on: a code in the
 * managed area is submitted through the SAME submit path the popup and the
 * pairing tab use, its token is stored like theirs, an answered code is never
 * re-submitted, an unanswered one is retried, and with no policy nothing
 * happens (a user's machine). Also pins the manifest/schema wiring Chrome
 * needs before it delivers the key at all. */
import { test } from "node:test";
import assert from "node:assert/strict";
import { build } from "esbuild";
import { mkdtemp, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));

async function load() {
  const dir = await mkdtemp(join(tmpdir(), "lop-managed-pair-"));
  const outfile = join(dir, "managed.mjs");
  await build({
    entryPoints: [join(HERE, "..", "src", "managed-pairing.ts")],
    bundle: true,
    platform: "node",
    format: "esm",
    outfile,
  });
  return import(pathToFileURL(outfile) + `?${Date.now()}`);
}

function deps(policy, outcome, overrides = {}) {
  const calls = { submits: [], tokens: [] };
  return {
    calls,
    deps: {
      managed: policy === undefined ? undefined : { get: async () => policy },
      readLocal: async () => ({ token: "", port: 4123 }),
      writeToken: async (token) => calls.tokens.push(token),
      submit: async (input) => {
        calls.submits.push(input);
        return typeof outcome === "function" ? outcome(calls.submits.length) : outcome;
      },
      defaultPort: 4099,
      ...overrides,
    },
  };
}

test("a managed code is submitted once on the daemon port and its token stored", async () => {
  const mod = await load();
  const { calls, deps: d } = deps({ pairingCode: "123456" }, { ok: true, token: "tok-1" });
  assert.equal(await mod.pairFromManagedPolicy(d), "paired");
  assert.deepEqual(calls.submits, [{ code: "123456", port: 4123, token: "" }]);
  assert.deepEqual(calls.tokens, ["tok-1"]);
  // The same code again (alarm tick, storage event) is not re-submitted.
  assert.equal(await mod.pairFromManagedPolicy(d), "already_tried");
  assert.equal(calls.submits.length, 1);
});

test("no managed area, no key, or an empty key does nothing (a user's machine)", async () => {
  const mod = await load();
  for (const policy of [undefined, {}, { pairingCode: "" }, { pairingCode: 42 }]) {
    const { calls, deps: d } = deps(policy, { ok: true, token: "x" });
    assert.equal(await mod.pairFromManagedPolicy(d), "no_policy");
    assert.equal(calls.submits.length, 0);
  }
  const { calls, deps: d } = deps({}, { ok: true, token: "x" }, {
    managed: { get: async () => Promise.reject(new Error("no schema")) },
  });
  assert.equal(await mod.pairFromManagedPolicy(d), "no_policy");
  assert.equal(calls.submits.length, 0);
});

test("a refused code is not retried; a fresh code re-arms", async () => {
  const mod = await load();
  let policy = { pairingCode: "111111" };
  const { calls, deps: d } = deps(undefined, (n) =>
    n === 1 ? { ok: false, kind: "rejected", message: "expired" } : { ok: true, token: "tok-2" },
  );
  d.managed = { get: async () => policy };
  assert.equal(await mod.pairFromManagedPolicy(d), "rejected");
  assert.equal(await mod.pairFromManagedPolicy(d), "already_tried");
  policy = { pairingCode: "222222" };
  assert.equal(await mod.pairFromManagedPolicy(d), "paired");
  assert.deepEqual(calls.submits.map((s) => s.code), ["111111", "222222"]);
});

test("a code that never reached the daemon is retried on the next trigger", async () => {
  const mod = await load();
  const { calls, deps: d } = deps({ pairingCode: "333333" }, (n) =>
    n === 1 ? { ok: false, kind: "unreachable" } : { ok: true, token: "tok-3" },
  );
  assert.equal(await mod.pairFromManagedPolicy(d), "unreachable");
  assert.equal(await mod.pairFromManagedPolicy(d), "paired");
  assert.equal(calls.submits.length, 2);
});

test("concurrent triggers submit once (two sockets would evict each other)", async () => {
  const mod = await load();
  let release;
  const gate = new Promise((r) => (release = r));
  const { calls, deps: d } = deps({ pairingCode: "444444" }, { ok: true, token: "t" });
  const slow = d.submit;
  d.submit = async (input) => {
    await gate;
    return slow(input);
  };
  const first = mod.pairFromManagedPolicy(d);
  assert.equal(await mod.pairFromManagedPolicy(d), "busy");
  release();
  assert.equal(await first, "paired");
  assert.equal(calls.submits.length, 1);
});

test("the manifest declares the managed schema and the schema carries the key", async () => {
  const manifest = JSON.parse(await readFile(new URL("../manifest.json", import.meta.url), "utf8"));
  assert.equal(manifest.storage?.managed_schema, "managed_schema.json");
  const schema = JSON.parse(await readFile(new URL("../managed_schema.json", import.meta.url), "utf8"));
  const mod = await load();
  assert.equal(schema.properties?.[mod.MANAGED_PAIRING_KEY]?.type, "string");
  const files = await readFile(new URL("../store-package-files.txt", import.meta.url), "utf8");
  assert.ok(files.split("\n").includes("managed_schema.json"), "the store zip ships the schema");
});
