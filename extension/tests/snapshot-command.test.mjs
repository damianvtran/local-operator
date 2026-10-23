/* The `snapshot` COMMAND's wiring: a CSS scope resolved to a backend id, the
 * a11y read window, and the line window applied to what comes back.
 *
 * WHAT THIS LAYER OWNS vs THE DRIVER'S TESTS. `ax-window.test.mjs` proves what
 * compactAX/windowSnapshot do with a tree; this file proves the command asks
 * the browser for the right things in the right order — that a selector is
 * resolved against the DOM before the a11y tree is read, that a miss is a typed
 * `element_not_found` naming the selector (never an empty tree that reads like
 * an empty section), and that the wire result carries the numbers a footer
 * needs.
 *
 * The fake is deliberately a fake: `sendCommand` answers from a script, so the
 * assertions are about WHICH CDP calls cross and with what params. A real-page
 * exercise belongs to the host proof, not here.
 */

import assert from "node:assert/strict";
import test from "node:test";
import { build } from "esbuild";
import { pathToFileURL } from "node:url";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

const TAB = 11;
const SURFACE = "bridge:11:nonce11";

const TREE = [
  { nodeId: "1", role: { value: "RootWebArea" }, name: { value: "Create App" }, childIds: ["2", "8"] },
  { nodeId: "2", ignored: true, childIds: ["3"] },
  { nodeId: "3", backendDOMNodeId: 300, role: { value: "form" }, name: { value: "" }, childIds: ["4", "5", "6"] },
  { nodeId: "4", backendDOMNodeId: 401, role: { value: "textbox" }, name: { value: "GitHub App name" }, childIds: [] },
  { nodeId: "5", backendDOMNodeId: 402, role: { value: "checkbox" }, name: { value: "Active" },
    properties: [{ name: "checked", value: { value: true } }, { name: "focusable", value: { value: true } }],
    childIds: [] },
  { nodeId: "6", backendDOMNodeId: 403, role: { value: "DisclosureTriangle" }, name: { value: "Webhook" },
    properties: [{ name: "expanded", value: { value: false } }], childIds: [] },
  // OUTSIDE the form: a button that must never acquire a ref from a scoped call.
  { nodeId: "8", backendDOMNodeId: 900, role: { value: "button" }, name: { value: "Delete app" },
    properties: [{ name: "focusable", value: { value: true } }], childIds: [] },
];

/** Load `commands/snapshot.ts` with a scripted chrome + CDP. */
async function loadWith({ selectorNodeId = 7, backendNodeId = 300, tree = TREE } = {}) {
  const dir = await mkdtemp(join(tmpdir(), "lop-snapshot-command-"));
  const outfile = join(dir, "module.mjs");
  await build({ entryPoints: ["src/commands/snapshot.ts"], bundle: true, platform: "node", format: "esm", outfile });

  const calls = [];
  const written = [];
  globalThis.chrome = {
    storage: {
      session: {
        get: async (keys) => {
          const wanted = Array.isArray(keys) ? keys : [keys];
          const state = {
            surfaces: { [SURFACE]: { tabId: TAB, nonce: "nonce11", epoch: 2, createdAt: 0, lastUsedAt: 0 } },
            refs: {},
          };
          return Object.fromEntries(wanted.filter((key) => key in state).map((key) => [key, state[key]]));
        },
        set: async (value) => { written.push(value); },
      },
      local: { get: async () => ({}), set: async () => {} },
    },
    tabs: {
      get: async (tabId) => {
        assert.equal(tabId, TAB, "the command must only ever look at its own tab");
        return { url: "https://example.test/new", title: "Create App" };
      },
    },
    debugger: {
      attach: async () => {},
      detach: async () => {},
      onDetach: { addListener: () => {} },
      sendCommand: async (_target, method, params) => {
        calls.push({ method, params });
        if (method === "DOM.getDocument") return { root: { nodeId: 1 } };
        if (method === "DOM.querySelector") return { nodeId: selectorNodeId };
        if (method === "DOM.describeNode") return { node: { backendNodeId } };
        if (method === "Accessibility.getFullAXTree") return { nodes: tree };
        return {};
      },
    },
  };
  const loaded = await import(pathToFileURL(outfile));
  return { loaded, calls, written, close: () => rm(dir, { recursive: true, force: true }) };
}

test("a selector is resolved before the tree is read, and scopes the walk", async () => {
  const { loaded, calls, written, close } = await loadWith();
  try {
    const out = await loaded.snapshot({ tab: SURFACE, selector: "form#new_app" });
    const methods = calls.map((call) => call.method);
    // Order is the contract: the scope must be resolved BEFORE the a11y read,
    // or a miss would still pay for a full tree and fail afterwards.
    assert.deepEqual(methods, [
      "DOM.getDocument",
      "DOM.querySelector",
      "DOM.describeNode",
      "Accessibility.enable",
      "Accessibility.getFullAXTree",
      "Accessibility.disable",
    ]);
    assert.deepEqual(calls[1].params, { nodeId: 1, selector: "form#new_app" });
    assert.equal(out.scoped, true);
    assert.ok(out.snapshot.startsWith("- form"), out.snapshot);
    assert.ok(!out.snapshot.includes("RootWebArea"));
    assert.ok(out.snapshot.includes('textbox "GitHub App name" [e1]'));
    assert.ok(out.snapshot.includes('checkbox "Active" [e2] [checked]'));
    assert.ok(out.snapshot.includes("[collapsed]"));
    // Refs were stored for the SURFACE, not globally.
    const refsWrite = written.find((value) => value.refs);
    assert.ok(refsWrite, "the snapshot must persist its refs");
    assert.equal(refsWrite.refs[SURFACE].e1.backendNodeId, 401);
    assert.equal(refsWrite.refs[SURFACE].e1.epoch, 2);
  } finally {
    await close();
  }
});

test("a selector that matches nothing fails typed, before any a11y read", async () => {
  const { loaded, calls, close } = await loadWith({ selectorNodeId: 0 });
  try {
    await assert.rejects(
      () => loaded.snapshot({ tab: SURFACE, selector: "#hook_active" }),
      (error) => error.code === "element_not_found" && /#hook_active/.test(error.message),
    );
    assert.deepEqual(calls.map((call) => call.method), ["DOM.getDocument", "DOM.querySelector"]);
  } finally {
    await close();
  }
});

test("a matched element with no accessible node says so instead of returning an empty tree", async () => {
  const { loaded, close } = await loadWith({ backendNodeId: 999 });
  try {
    await assert.rejects(
      () => loaded.snapshot({ tab: SURFACE, selector: "#inside-a-closed-details" }),
      (error) =>
        error.code === "element_not_found" &&
        /no accessibility subtree/.test(error.message) &&
        /collapsed/.test(error.message),
    );
  } finally {
    await close();
  }
});

test("start/limit/search window the rendered tree and report the numbers", async () => {
  const { loaded, close } = await loadWith();
  try {
    const all = await loaded.snapshot({ tab: SURFACE });
    const page = await loaded.snapshot({ tab: SURFACE, start: 2, limit: 2 });
    assert.equal(page.total_lines, all.total_lines);
    assert.equal(page.returned_lines, 2);
    assert.equal(page.start, 2);
    assert.deepEqual(page.snapshot.split("\n"), all.snapshot.split("\n").slice(1, 3));

    const search = await loaded.snapshot({ tab: SURFACE, search: "active" });
    assert.equal(search.matched_lines, 1);
    assert.ok(search.snapshot.includes('checkbox "Active"'));
  } finally {
    await close();
  }
});

test("a scoped snapshot renders only the scope but mints refs for the whole page", async () => {
  // Agent review round 1, R1-1. This test used to assert the opposite — that a
  // scoped snapshot withheld refs for the elided part — and that is precisely
  // how a remembered `e1` came to name a different node: ref names restarted
  // inside the scope at the same epoch, so the next `click(e1)` hit whatever the
  // scope had put first. The contract now: the RENDERED text is the scope, the
  // REF TABLE is the whole tree, so a name means one node for the epoch.
  const { loaded, written, close } = await loadWith();
  try {
    const scoped = await loaded.snapshot({ tab: SURFACE, selector: "form", limit: 1 });
    const refsWrite = written.find((value) => value.refs);
    const refs = Object.values(refsWrite.refs[SURFACE]);
    assert.ok(refs.length > 1, "the ref table covers the whole tree");
    // The Delete button below the form keeps its name even though the scope
    // does not render it: clicking a remembered ref lands on the same element.
    assert.ok(
      refs.some((ref) => ref.backendNodeId === 900),
      "a ref minted before the scope must keep naming its node",
    );
    assert.ok(scoped.returned_lines < scoped.total_lines, "the text really was scoped");
    assert.ok(!scoped.snapshot.includes("Delete app"), "the elided part is not rendered");
  } finally {
    await close();
  }
});
