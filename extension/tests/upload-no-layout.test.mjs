/* `upload` must never need the file input to have a LAYOUT.
 *
 * WHY THIS TEST EXISTS. The overwhelmingly common file input is hidden
 * (`display:none`, triggered by a styled button: Slack, GitHub, every React
 * dropzone). Chrome answers `DOM.scrollIntoViewIfNeeded`, `DOM.getBoxModel` and
 * `DOM.getContentQuads` for such a node with "Node does not have a layout
 * object", so a single one of those calls on the upload path turns every
 * hidden-input upload into an error (measured in the Slack app-icon incident).
 * `DOM.setFileInputFiles` takes a node id and needs no layout at all, which is
 * why the attach itself works on a hidden input.
 *
 * The assertion is on WHICH CDP methods cross, not on a mocked outcome, and the
 * fake debugger REFUSES the layout methods the way Chrome does for a hidden node:
 * if a future change adds a scroll or a box-model step "to make sure the input is
 * visible", the upload fails here before it fails on a real page.
 */

import assert from "node:assert/strict";
import test from "node:test";
import { build } from "esbuild";
import { pathToFileURL } from "node:url";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

const TAB = 31;
const SURFACE = "bridge:31:nonce31";
const LAYOUT_METHODS = ["DOM.scrollIntoViewIfNeeded", "DOM.getBoxModel", "DOM.getContentQuads"];

const TREE = [
  { nodeId: "1", role: { value: "RootWebArea" }, name: { value: "App" }, childIds: ["2"] },
  { nodeId: "2", backendDOMNodeId: 21, role: { value: "button" }, name: { value: "Add App Icon" }, childIds: [] },
];

async function loadUpload() {
  const dir = await mkdtemp(join(tmpdir(), "lop-upload-no-layout-"));
  const outfile = join(dir, "module.mjs");
  await build({ entryPoints: ["src/commands/upload.ts"], bundle: true, platform: "node", format: "esm", outfile });

  const calls = [];
  globalThis.chrome = {
    storage: {
      session: {
        get: async (keys) => {
          const wanted = Array.isArray(keys) ? keys : [keys];
          const state = {
            surfaces: { [SURFACE]: { tabId: TAB, nonce: "nonce31", epoch: 1, createdAt: 0, lastUsedAt: 0 } },
            refs: { [SURFACE]: { e7: { backendNodeId: 21, epoch: 1 } } },
          };
          return Object.fromEntries(wanted.filter((key) => key in state).map((key) => [key, state[key]]));
        },
        set: async () => {},
      },
      // The operator's `allowUploads` switch is ON: consent is not what is under test.
      local: { get: async () => ({ allowUploads: true }), set: async () => {} },
    },
    permissions: { contains: async () => true },
    tabs: { get: async () => ({ url: "https://app.example.test/", title: "App" }) },
    debugger: {
      attach: async () => {},
      detach: async () => {},
      onDetach: { addListener: () => {} },
      sendCommand: async (_target, method, params) => {
        calls.push({ method, params });
        // A display:none input: every layout-dependent method fails like Chrome's.
        if (LAYOUT_METHODS.includes(method)) throw new Error("Node does not have a layout object");
        if (method === "Accessibility.getFullAXTree") return { nodes: TREE };
        if (method === "DOM.getDocument") return { root: { nodeId: 1 } };
        if (method === "DOM.querySelector") return { nodeId: 77 };
        if (method === "DOM.pushNodesByBackendIdsToFrontend") return { nodeIds: [88] };
        if (method === "DOM.resolveNode") return { object: { objectId: "object:1" } };
        if (method === "Runtime.callFunctionOn") {
          return {
            result: {
              value: JSON.stringify({ count: 1, names: ["slack-icon-1024.png"], sizes: [71571], accept: "image/*" }),
            },
          };
        }
        return {};
      },
    },
  };
  const loaded = await import(pathToFileURL(outfile));
  return { loaded, calls, close: () => rm(dir, { recursive: true, force: true }) };
}

const PATH = "/Users/me/workspace/icons/slack-icon-1024.png";

for (const selector of ["input[type=file]", "e7"]) {
  test(`upload to ${selector} never calls a layout method and still attaches`, async () => {
    const { loaded, calls, close } = await loadUpload();
    try {
      const out = await loaded.upload({ tab: SURFACE, selector, paths: [PATH] });
      const methods = calls.map((call) => call.method);
      for (const forbidden of LAYOUT_METHODS) {
        assert.ok(!methods.includes(forbidden), `upload called ${forbidden}: ${methods.join(",")}`);
      }
      const attach = calls.find((call) => call.method === "DOM.setFileInputFiles");
      assert.ok(attach, `no DOM.setFileInputFiles in ${methods.join(",")}`);
      assert.deepEqual(attach.params.files, [PATH]);
      assert.equal(out.refused.length, 0);
      assert.equal(out.accepted.length, 1);
      assert.equal(out.readback, "");
    } finally {
      await close();
    }
  });
}
