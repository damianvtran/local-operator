/* `click`/`type` target resolution: locators first, CSS and refs unchanged.
 *
 * The assertion that matters is WHICH CDP calls cross: a locator must read the
 * accessibility tree and then push the matched BACKEND id into the frontend,
 * while a CSS selector must never pay for an a11y read at all. A miss must be
 * typed and carry candidates, because the measured failure spent nine whole
 * tool calls on guesses that each said only "element not found".
 */

import assert from "node:assert/strict";
import test from "node:test";
import { build } from "esbuild";
import { pathToFileURL } from "node:url";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

const TAB = 21;
const SURFACE = "bridge:21:nonce21";

const TREE = [
  { nodeId: "1", role: { value: "RootWebArea" }, name: { value: "Create App" }, childIds: ["2"] },
  { nodeId: "2", backendDOMNodeId: 11, role: { value: "textbox" }, name: { value: "Homepage URL" }, childIds: [] },
  { nodeId: "3", backendDOMNodeId: 12, role: { value: "checkbox" }, name: { value: "Active" },
    properties: [{ name: "checked", value: { value: "true" } }, { name: "focusable", value: { value: true } }], childIds: [] },
];

async function loadWith({ nodes = TREE, queryNodeId = 77 } = {}) {
  const dir = await mkdtemp(join(tmpdir(), "lop-input-locator-"));
  const outfile = join(dir, "module.mjs");
  await build({ entryPoints: ["src/commands/input.ts"], bundle: true, platform: "node", format: "esm", outfile });

  const calls = [];
  globalThis.chrome = {
    storage: {
      session: {
        get: async (keys) => {
          const wanted = Array.isArray(keys) ? keys : [keys];
          const state = {
            surfaces: { [SURFACE]: { tabId: TAB, nonce: "nonce21", epoch: 1, createdAt: 0, lastUsedAt: 0 } },
            refs: {},
          };
          return Object.fromEntries(wanted.filter((key) => key in state).map((key) => [key, state[key]]));
        },
        set: async () => {},
      },
      local: { get: async () => ({}), set: async () => {} },
    },
    tabs: {
      get: async (tabId) => {
        assert.equal(tabId, TAB);
        return { url: "https://example.test/new", title: "Create App" };
      },
    },
    debugger: {
      attach: async () => {},
      detach: async () => {},
      onDetach: { addListener: () => {} },
      sendCommand: async (_target, method, params) => {
        calls.push({ method, params });
        if (method === "Accessibility.getFullAXTree") return { nodes };
        if (method === "DOM.getDocument") return { root: { nodeId: 1 } };
        if (method === "DOM.querySelector") return { nodeId: queryNodeId };
        if (method === "DOM.pushNodesByBackendIdsToFrontend") return { nodeIds: [88] };
        if (method === "DOM.resolveNode") return { object: { objectId: "object:1" } };
        if (method === "Runtime.callFunctionOn") return { result: { value: "hello" } };
        return {};
      },
    },
  };
  const loaded = await import(pathToFileURL(outfile));
  return { loaded, calls, close: () => rm(dir, { recursive: true, force: true }) };
}

test("label= reads the a11y tree and pushes the matched backend id", async () => {
  const { loaded, calls, close } = await loadWith();
  try {
    const out = await loaded.typeText({ tab: SURFACE, selector: 'label="Homepage URL"', text: "hello" });
    const methods = calls.map((call) => call.method);
    assert.ok(methods.includes("Accessibility.getFullAXTree"), methods.join(","));
    const push = calls.find((call) => call.method === "DOM.pushNodesByBackendIdsToFrontend");
    assert.deepEqual(push.params, { backendNodeIds: [11] });
    // No CSS query at all on the locator path: the whole point is that the id
    // is irrelevant.
    assert.ok(!methods.includes("DOM.querySelector"));
    assert.equal(out.value, "hello");
    assert.equal(out.resolved.by, "locator");
    assert.equal(out.resolved.role, "textbox");
    assert.equal(out.resolved.name, "Homepage URL");
    assert.equal(out.resolved.matched, 1);
  } finally {
    await close();
  }
});

test("role= resolves the first match and reports how many matched", async () => {
  const { loaded, calls, close } = await loadWith({
    nodes: [...TREE, { nodeId: "9", backendDOMNodeId: 13, role: { value: "checkbox" }, name: { value: "Active" }, childIds: [] }],
  });
  try {
    const out = await loaded.typeText({ tab: SURFACE, selector: 'role=checkbox[name="Active"]', text: "hello" });
    assert.deepEqual(calls.find((call) => call.method === "DOM.pushNodesByBackendIdsToFrontend").params, {
      backendNodeIds: [12],
    });
    assert.equal(out.resolved.matched, 2);
  } finally {
    await close();
  }
});

test("a CSS selector never pays for an a11y read", async () => {
  const { loaded, calls, close } = await loadWith();
  try {
    await loaded.typeText({ tab: SURFACE, selector: "#homepage", text: "hello" });
    const methods = calls.map((call) => call.method);
    assert.ok(methods.includes("DOM.querySelector"), methods.join(","));
    assert.ok(!methods.includes("Accessibility.getFullAXTree"));
  } finally {
    await close();
  }
});

test("a locator miss is typed and carries candidates", async () => {
  const { loaded, close } = await loadWith();
  try {
    await assert.rejects(
      () => loaded.typeText({ tab: SURFACE, selector: 'role=checkbox[name="Actve"]', text: "hello" }),
      (error) => {
        assert.equal(error.code, "element_not_found");
        assert.match(error.message, /Actve/);
        assert.ok(Array.isArray(error.data.candidates), JSON.stringify(error.data));
        assert.ok(error.data.candidates.length > 0);
        assert.ok(error.data.candidates.some((entry) => entry.name === "Active"));
        return true;
      },
    );
  } finally {
    await close();
  }
});

test("an unknown ref falls through to CSS exactly as before, with no a11y read", async () => {
  // `e99` is not in this surface's ref table, so it is treated as a CSS type
  // selector — the pre-existing (and deliberate) fall-through. What must not
  // change is the COST: no a11y read for a selector-shaped value.
  const { loaded, calls, close } = await loadWith({ queryNodeId: 0 });
  try {
    await assert.rejects(
      () => loaded.typeText({ tab: SURFACE, selector: "e99", text: "hello" }),
      (error) => error.code === "element_not_found" && /matched nothing/.test(error.message),
    );
    assert.deepEqual(
      calls.filter((call) => call.method === "DOM.querySelector").map((call) => call.params.selector),
      ["e99"],
    );
    assert.ok(!calls.some((call) => call.method === "Accessibility.getFullAXTree"));
  } finally {
    await close();
  }
});
