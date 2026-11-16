/* Headless-node stored site decisions (src/node-site-decisions.ts).
 *
 * Pins the two PR #3938 remediation contracts that live in the extension:
 * a signed Deny is remembered so the node does not raise a second card for the
 * same origin (Q4 / D3 "ask me once per site"), and `revoke_access` removes an
 * exact-origin grant or denial — the removal the card promises (D2 / U4) —
 * reporting what was actually there. */
import { test } from "node:test";
import assert from "node:assert/strict";
import { build } from "esbuild";
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));

async function load() {
  const dir = await mkdtemp(join(tmpdir(), "lop-node-site-decisions-"));
  const outfile = join(dir, "decisions.mjs");
  await build({
    entryPoints: [join(HERE, "..", "src", "node-site-decisions.ts")],
    bundle: true,
    platform: "node",
    format: "esm",
    outfile,
  });
  return import(pathToFileURL(outfile) + `?${Date.now()}`);
}

const mod = await load();

function memoryStorage(initial = {}) {
  const data = structuredClone(initial);
  const writes = [];
  return {
    data,
    writes,
    async get(keys) {
      return Object.fromEntries(keys.filter((key) => key in data).map((key) => [key, structuredClone(data[key])]));
    },
    async set(items) {
      writes.push(Object.keys(items).sort());
      Object.assign(data, structuredClone(items));
    },
  };
}

test("a signed denial is remembered per exact origin", async () => {
  const storage = memoryStorage();
  assert.equal(await mod.nodeDenied(storage, "https://a.example"), false);
  await mod.recordNodeDenial(storage, "https://a.example", 1000);
  assert.equal(await mod.nodeDenied(storage, "https://a.example"), true);
  // Exact origin only: another port or scheme is a different site.
  assert.equal(await mod.nodeDenied(storage, "https://a.example:8443"), false);
  assert.equal(await mod.nodeDenied(storage, "http://a.example"), false);
});

test("list reports grants, denials, broad grants and the all-sites switch", async () => {
  const storage = memoryStorage({
    origins: { "https://b.example": "allow", "https://a.example": "allow", "https://legacy.example": "deny" },
    siteGrants: { version: 1, grants: { "example.org": { scope: "domain", createdAt: 1 } } },
    nodeSiteDenials: { "https://c.example": 5, "https://bad.example": "nope" },
  });
  assert.deepEqual(await mod.listSiteDecisions(storage), {
    allowed: ["https://a.example", "https://b.example"],
    denied: ["https://c.example"],
    broad: ["example.org"],
    allow_all: false,
  });
});

test("revoke removes a grant and a denial for that origin in one write, and says what was there", async () => {
  const storage = memoryStorage({
    origins: { "https://a.example": "allow", "https://keep.example": "allow" },
    nodeSiteDenials: { "https://a.example": 7, "https://keep-denied.example": 8 },
  });
  const result = await mod.revokeSiteDecision(storage, "https://a.example");
  assert.deepEqual(result, { origin: "https://a.example", removed: ["allowed", "denied"] });
  assert.deepEqual(storage.writes, [["nodeSiteDenials", "origins"]]);
  assert.deepEqual(storage.data.origins, { "https://keep.example": "allow" });
  assert.deepEqual(storage.data.nodeSiteDenials, { "https://keep-denied.example": 8 });
});

test("revoke of an origin with nothing stored writes nothing and reports nothing removed", async () => {
  const storage = memoryStorage({ origins: { "https://keep.example": "allow" } });
  assert.deepEqual(await mod.revokeSiteDecision(storage, "https://nothing.example"), {
    origin: "https://nothing.example",
    removed: [],
  });
  assert.deepEqual(storage.writes, []);
});
