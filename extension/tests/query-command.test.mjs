/* The `query` command's handler layer: which function crosses into the page,
 * with which arguments, and what a bad selector becomes.
 *
 * `element-query.test.mjs` proves the page function's caps and its
 * password-value rule; this proves the command injects THAT function (not a
 * wrapper that would serialize without the page logic), on the surface's own
 * tab, with the limit pre-clamped — and that "not valid CSS" is the typed
 * `element_not_found` every other page-script read answers, not `internal`.
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

async function loadWith(scripted) {
  const dir = await mkdtemp(join(tmpdir(), "lop-query-command-"));
  const outfile = join(dir, "module.mjs");
  await build({ entryPoints: ["src/commands/query.ts"], bundle: true, platform: "node", format: "esm", outfile });

  const injections = [];
  globalThis.chrome = {
    storage: {
      session: {
        get: async (keys) => {
          const wanted = Array.isArray(keys) ? keys : [keys];
          const state = {
            surfaces: { [SURFACE]: { tabId: TAB, nonce: "nonce31", epoch: 1, createdAt: 0, lastUsedAt: 0 } },
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
        assert.equal(tabId, TAB, "the command must only ever look at its own tab");
        return { url: "https://example.test/", title: "Create App" };
      },
    },
    debugger: { onDetach: { addListener: () => undefined } },
    scripting: {
      executeScript: async (injection) => {
        injections.push(injection);
        if (scripted === "invalid") throw new Error("SyntaxError: '###' is not a valid selector");
        return [{ result: scripted }];
      },
    },
  };
  const loaded = await import(pathToFileURL(outfile));
  return { loaded, injections, close: () => rm(dir, { recursive: true, force: true }) };
}

const RESULT = { count: 1, truncated: false, matches: [{ tag: "input", id: "hook_active_55a1" }] };

test("query sends the driver function itself, on the surface's tab", async () => {
  const { loaded, injections, close } = await loadWith(RESULT);
  try {
    const out = await loaded.query({ tab: SURFACE, selector: "input[type=checkbox]", limit: 5 });
    assert.deepEqual(out, RESULT);
    assert.equal(injections.length, 1);
    assert.equal(injections[0].target.tabId, TAB);
    assert.equal(injections[0].func.name, "queryElements", "the driver function crosses, not a wrapper");
    assert.match(injections[0].func.toString(), /MAX_ELEMENTS/, "the caps live in the serialized body");
    assert.deepEqual(injections[0].args, ["input[type=checkbox]", 5]);
  } finally {
    await close();
  }
});

test("query defaults the limit and clamps a bad one before the wire", async () => {
  const { loaded, injections, close } = await loadWith(RESULT);
  try {
    await loaded.query({ tab: SURFACE, selector: "input" });
    assert.deepEqual(injections.at(-1).args, ["input", 20], "the default is 20 descriptors");
    await loaded.query({ tab: SURFACE, selector: "input", limit: -3 });
    assert.deepEqual(injections.at(-1).args, ["input", 20], "a negative limit falls back to the default");
    await loaded.query({ tab: SURFACE, selector: "input", limit: "7" });
    assert.deepEqual(injections.at(-1).args, ["input", 7], "a numeric string is accepted");
  } finally {
    await close();
  }
});

test("an invalid selector is a typed element_not_found, not internal", async () => {
  // The page REPORTS the bad selector as a value (a thrown error does not survive
  // the isolated world's transport), and the command turns it into the refusal.
  const { loaded, close } = await loadWith({ count: 0, truncated: false, matches: [], error: "invalid_selector" });
  try {
    await assert.rejects(
      () => loaded.query({ tab: SURFACE, selector: "###" }),
      (error) => error.code === "element_not_found" && /not valid/.test(error.message),
    );
  } finally {
    await close();
  }
});

test("a missing selector is refused before any page script runs", async () => {
  const { loaded, injections, close } = await loadWith(RESULT);
  try {
    await assert.rejects(
      () => loaded.query({ tab: SURFACE }),
      (error) => error.code === "element_not_found" && /selector is required/.test(error.message),
    );
    assert.equal(injections.length, 0);
  } finally {
    await close();
  }
});

test("zero matches is a successful answer, not a refusal", async () => {
  const { loaded, close } = await loadWith({ count: 0, truncated: false, matches: [] });
  try {
    const out = await loaded.query({ tab: SURFACE, selector: "#nope" });
    assert.equal(out.count, 0);
    assert.deepEqual(out.matches, []);
  } finally {
    await close();
  }
});
