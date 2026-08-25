import assert from "node:assert/strict";
import test from "node:test";
import { build } from "esbuild";
import { pathToFileURL } from "node:url";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

async function load(entry) {
  const dir = await mkdtemp(join(tmpdir(), "lop-extension-test-"));
  const outfile = join(dir, "module.mjs");
  await build({ entryPoints: [entry], bundle: true, platform: "node", format: "esm", outfile });
  const loaded = await import(pathToFileURL(outfile));
  return { loaded, close: () => rm(dir, { recursive: true, force: true }) };
}

test("origin policy only permits stored HTTP origins", async () => {
  const module = await load("src/origin-policy.ts");
  try {
    const url = module.loaded.safeHttpUrl("https://example.com/path");
    assert.equal(module.loaded.storedOriginAllowed({ "https://example.com": "allow" }, url), true);
    assert.equal(module.loaded.storedOriginAllowed({}, url), false);
    assert.throws(() => module.loaded.safeHttpUrl("chrome://settings"));
  } finally { await module.close(); }
});

test("AX compaction assigns epoch-scoped click refs", async () => {
  const module = await load("src/ax-compact.ts");
  try {
    const rendered = module.loaded.compactAX([
      { nodeId: "1", role: { value: "main" }, name: { value: "Content" }, childIds: ["2"] },
      { nodeId: "2", role: { value: "button" }, name: { value: "Continue" }, backendDOMNodeId: 42 },
    ], 7);
    assert.match(rendered.snapshot, /button "Continue" \[e1\]/);
    assert.deepEqual(rendered.refs.e1, { backendNodeId: 42, epoch: 7 });
  } finally { await module.close(); }
});
