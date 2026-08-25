#!/usr/bin/env node
/** Build the store-ready extension with no dev server or runtime framework. */
import { build } from "esbuild";
import { cp, mkdir, rm } from "node:fs/promises";
import { execFileSync } from "node:child_process";
import { resolve } from "node:path";

const root = import.meta.dirname;
const dist = resolve(root, "dist");
// A store build (`--zip`) ships NO source maps: they roughly double the archive,
// expose the original TypeScript, and are dead weight in a published artifact
// that reviewers and the runtime never step through. The plain `dist` build
// keeps maps so local debugging (Load unpacked + DevTools) resolves to source.
const isStore = process.argv.includes("--zip");
await rm(dist, { recursive: true, force: true });
await mkdir(dist, { recursive: true });
await build({
  absWorkingDir: root,
  entryPoints: {
    worker: "src/worker.ts",
    "popup/popup": "src/popup/popup.ts",
    "options/options": "src/options/options.ts",
  },
  bundle: true,
  format: "esm",
  target: "chrome116",
  outdir: dist,
  sourcemap: !isStore,
});
await cp(resolve(root, "manifest.json"), resolve(dist, "manifest.json"));
await cp(resolve(root, "src/popup/popup.html"), resolve(dist, "popup/popup.html"));
await cp(resolve(root, "src/popup/popup.css"), resolve(dist, "popup/popup.css"));
await cp(resolve(root, "src/options/options.html"), resolve(dist, "options/options.html"));
await cp(resolve(root, "src/options/options.css"), resolve(dist, "options/options.css"));
await cp(resolve(root, "icons"), resolve(dist, "icons"), { recursive: true });
if (isStore) {
  const target = resolve(root, "local-operator-extension.zip");
  await rm(target, { force: true });
  // -x '*.map' is belt-and-suspenders: the store build already emits no maps,
  // but this guarantees none slip into the published archive even if a stray
  // map exists in dist from a prior plain build.
  execFileSync("zip", ["-qr", target, ".", "-x", "*.map"], { cwd: dist });
  console.log(target);
} else {
  console.log(dist);
}
