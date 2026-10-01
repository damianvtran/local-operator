/* The three read commands' handler layer, with a SCRIPTED chrome.
 *
 * WHAT THIS LAYER OWNS vs THE DRIVER'S TESTS. `geometry-read.test.mjs` proves
 * the page-side functions' bounds and serialization contract; this file proves
 * the commands wire them correctly: the right SELF-CONTAINED function crosses
 * `chrome.scripting.executeScript` (not a wrapper that would serialize without
 * the page logic), the target is only ever the surface's own tab, the arguments
 * are the shapes the driver expects (depth pre-clamped, properties filtered),
 * and a `null` from the page becomes the typed `element_not_found` rather than
 * a successful empty result.
 *
 * The fake is deliberately a fake: `executeScript` records its injection and
 * answers with a scripted value, so the assertions can be about WHICH function
 * and arguments were sent, not about what a browser would have rendered. A
 * real-browser exercise belongs to the e2e rig.
 */

import assert from "node:assert/strict";
import test from "node:test";
import { build } from "esbuild";
import { pathToFileURL } from "node:url";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

const TAB = 7;
const SURFACE = "bridge:7:abc123";

/** Load a command module with a scripted `chrome` installed first: `../cdp`
 *  registers a `chrome.debugger.onDetach` listener at module scope, so the
 *  global must exist before the import. Returns the recorded injections. */
async function loadWith(entry, scriptedResult) {
  const dir = await mkdtemp(join(tmpdir(), "lop-geometry-command-"));
  const outfile = join(dir, "module.mjs");
  await build({ entryPoints: [entry], bundle: true, platform: "node", format: "esm", outfile });

  const injections = [];
  globalThis.chrome = {
    storage: {
      local: { get: async () => ({}) },
      session: {
        get: async () => ({
          surfaces: {
            [SURFACE]: { tabId: TAB, nonce: "abc123", epoch: 1, createdAt: 0, lastUsedAt: 0 },
          },
        }),
      },
    },
    tabs: {
      get: async (tabId) => {
        assert.equal(tabId, TAB, "the command must only ever look at its own tab");
        return { url: "https://example.test/page", title: "Example" };
      },
    },
    // Registered at module scope by `cdp.ts`; nothing in these commands should
    // need the debugger, which is itself part of what the injections pin.
    debugger: { onDetach: { addListener: () => undefined } },
    scripting: {
      executeScript: async (injection) => {
        injections.push(injection);
        return [{ result: scriptedResult }];
      },
    },
  };
  const loaded = await import(pathToFileURL(outfile));
  return { loaded, injections, close: () => rm(dir, { recursive: true, force: true }) };
}

test("styles sends the driver function itself, with source-visible caps", async () => {
  const result = {
    count: 1,
    truncated: false,
    matches: [
      {
        tag: "div",
        id: "card",
        role: "",
        className: "a b",
        rect: { x: 1, y: 2, top: 2, right: 3, bottom: 4, width: 2, height: 2 },
        styles: { display: "block" },
        inline: { "--brand": "#f00" },
      },
    ],
  };
  const { loaded, injections, close } = await loadWith("src/commands/styles.ts", result);
  try {
    const out = await loaded.styles({
      tab: SURFACE,
      selector: ".card",
      properties: ["background-color", 42, null], // non-strings dropped at the boundary
    });
    assert.deepEqual(out, result, "the page result is returned as-is");

    assert.equal(injections.length, 1);
    const injection = injections[0];
    assert.equal(injection.target.tabId, TAB);
    assert.equal(injection.func.name, "readStyles", "the driver function crosses, not a wrapper");
    assert.match(injection.func.toString(), /MAX_MATCHES/, "the caps live in the serialized body");
    assert.deepEqual(injection.args, [".card", ["background-color"]]);

    // null from the page (no match) -> typed element_not_found.
    injections.length = 0;
    globalThis.chrome.scripting.executeScript = async () => [{ result: null }];
    await assert.rejects(
      () => loaded.styles({ tab: SURFACE, selector: ".none" }),
      (error) => error.code === "element_not_found",
    );

    // A missing selector is refused before any page script runs.
    await assert.rejects(
      () => loaded.styles({ tab: SURFACE }),
      (error) => error.code === "element_not_found",
    );

    // Wire-shape guard: more than 20 extras clip to 20 before they reach the page.
    globalThis.chrome.scripting.executeScript = async (injection) => {
      injections.push(injection);
      return [{ result }];
    };
    await loaded.styles({ tab: SURFACE, selector: ".card", properties: Array.from({ length: 30 }, (_, i) => `p${i}`) });
    assert.equal(injections.at(-1).args[1].length, 20);
  } finally {
    await close();
  }
});

test("hit_test passes viewport coordinates and refuses non-numbers", async () => {
  const result = { count: 1, elements: [] };
  const { loaded, injections, close } = await loadWith("src/commands/hit-test.ts", result);
  try {
    await loaded.hitTest({ tab: SURFACE, x: 640, y: 48.5 });
    const injection = injections[0];
    assert.equal(injection.func.name, "hitTest");
    assert.match(injection.func.toString(), /elementsFromPoint/, "the hit-test body crosses");
    assert.deepEqual(injection.args, [640, 48.5]);

    await assert.rejects(
      () => loaded.hitTest({ tab: SURFACE, x: "640", y: 48 }),
      (error) => error.code === "internal",
    );
    await assert.rejects(
      () => loaded.hitTest({ tab: SURFACE, x: Number.NaN, y: 48 }),
      (error) => error.code === "internal",
    );

    // An empty stack from the page (off-viewport) -> typed element_not_found.
    globalThis.chrome.scripting.executeScript = async () => [{ result: null }];
    await assert.rejects(
      () => loaded.hitTest({ tab: SURFACE, x: 0, y: 0 }),
      (error) => error.code === "element_not_found",
    );
  } finally {
    await close();
  }
});

test("ancestors pre-clamps depth and maps no-match to element_not_found", async () => {
  const result = { count: 2, chain: [] };
  const { loaded, injections, close } = await loadWith("src/commands/ancestors.ts", result);
  try {
    await loaded.ancestors({ tab: SURFACE, selector: "#card" });
    assert.equal(injections[0].func.name, "ancestors");
    assert.match(injections[0].func.toString(), /documentElement/, "the ancestor walk crosses");
    assert.deepEqual(injections[0].args, ["#card", 12], "omitted depth sends the documented default");

    await loaded.ancestors({ tab: SURFACE, selector: "#card", depth: 999 });
    assert.deepEqual(injections[1].args, ["#card", 16], "clamped at the hard cap");

    await loaded.ancestors({ tab: SURFACE, selector: "#card", depth: 0 });
    assert.deepEqual(injections[2].args, ["#card", 1], "clamped at one");

    await loaded.ancestors({ tab: SURFACE, selector: "#card", depth: 6.9 });
    assert.deepEqual(injections[3].args, ["#card", 6], "floored to an integer");

    globalThis.chrome.scripting.executeScript = async () => [{ result: null }];
    await assert.rejects(
      () => loaded.ancestors({ tab: SURFACE, selector: ".none" }),
      (error) => error.code === "element_not_found",
    );
    await assert.rejects(
      () => loaded.ancestors({ tab: SURFACE }),
      (error) => error.code === "element_not_found",
    );
  } finally {
    await close();
  }
});
