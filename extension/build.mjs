#!/usr/bin/env node
/** Build the store-ready extension with no dev server or runtime framework. */
import { build } from "esbuild";
import { cp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
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
// A DEV-ONLY manifest overlay, merged into the plain (non-store) build.
//
// Why it exists: a Chromium unpacked extension's id is derived from its
// DIRECTORY PATH, so one build loaded from two worktrees is two identities —
// three worktrees, three ids, and the operator re-pairs on every switch. A
// manifest `key` pins the id to a keypair instead, so every local build is ONE
// identity (manifest.dev.json holds the public half).
//
// Why it is NOT in manifest.json: that file is the STORE artifact's identity.
// The Web Store assigns the published id, so a `key` there would either
// conflict with it or silently change which id ships to users.
//
// Why a committed PUBLIC key is not a secret: `key` is base64 of an RSA public
// key. The private half is not in this repository (it was generated once and
// deliberately discarded — it only signs a .crx, which this project never
// distributes) and is not needed to Load unpacked. Anyone may copy it and build
// an extension claiming the dev id, and it buys them nothing: pairing still
// requires the 6-digit code that only the operator's terminal prints
// (docs/design/browser-extension.md §6.2). NEVER auto-trust this id daemon-side
// — a well-known id accepted without pairing is what would turn a public
// identity into a credential. AGENTS.md carries the same note.
if (!isStore) {
  // OPTIONAL on purpose: this file is a developer convenience, not a
  // correctness property, and scripts that copy a SUBSET of the extension
  // source (the real-Chrome rig copies `src/`, `icons/`, `manifest.json`,
  // `build.mjs`, `package.json` — see scripts/bridge_rig.py) would otherwise
  // break on a file they never needed. Builds without it get exactly the
  // pre-overlay behaviour: a path-derived id.
  const overlayPath = resolve(root, "manifest.dev.json");
  if (existsSync(overlayPath)) {
    const overlay = JSON.parse(await readFile(overlayPath, "utf8"));
    const target = resolve(dist, "manifest.json");
    const manifest = JSON.parse(await readFile(target, "utf8"));
    await writeFile(target, `${JSON.stringify({ ...manifest, ...overlay }, null, 2)}\n`);
    console.log("dev manifest key applied: this build's id is pinned to the dev keypair");
  } else {
    // Loud, because the silent version of this is the operator re-pairing on
    // every worktree and nobody knowing why.
    console.warn(
      "no manifest.dev.json beside this manifest: this build's extension id is " +
        "PATH-DERIVED and will change with the directory",
    );
  }
}
await cp(resolve(root, "src/popup/popup.html"), resolve(dist, "popup/popup.html"));
// Copied, never bundled: esbuild would emit it as a module entry and the
// deferral is exactly what it exists to avoid. See src/popup/first-paint.js.
await cp(resolve(root, "src/popup/first-paint.js"), resolve(dist, "popup/first-paint.js"));
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
