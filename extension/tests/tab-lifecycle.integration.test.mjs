import assert from "node:assert/strict";
import test from "node:test";
import { build } from "esbuild";
import { pathToFileURL } from "node:url";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

// The bundle pulls in cdp.ts -> log-capture.ts, which touches chrome APIs at
// module scope, so the stub has to be installed BEFORE the dynamic import
// rather than after it.
async function loadModule(initialSurfaces = {}) {
  const chromeState = installChrome(initialSurfaces);
  const dir = await mkdtemp(join(tmpdir(), "lop-tab-lifecycle-it-"));
  const outfile = join(dir, "module.mjs");
  await build({ entryPoints: ["src/tab-lifecycle.ts"], bundle: true, platform: "node", format: "esm", outfile });
  return {
    loaded: await import(pathToFileURL(outfile) + `?${Date.now()}`),
    chromeState,
    close: () => rm(dir, { recursive: true, force: true }),
  };
}

/** Minimal chrome surface: the surfaces map, tabs, and a debugger that records
 * detach calls so the "prune reclaims everything" property is observable. */
function installChrome(initialSurfaces = {}) {
  let surfaces = structuredClone(initialSurfaces);
  const detached = [];
  globalThis.chrome = {
    storage: {
      session: {
        get: async (keys) => {
          const all = { surfaces, refs: {} };
          if (Array.isArray(keys)) return Object.fromEntries(keys.map((k) => [k, all[k]]));
          return all;
        },
        set: async (value) => {
          if (value.surfaces) surfaces = structuredClone(value.surfaces);
        },
      },
    },
    debugger: {
      attach: async () => {},
      detach: async ({ tabId }) => { detached.push(tabId); },
      sendCommand: async () => ({}),
      // log-capture.ts subscribes at module scope, so these must exist before
      // the bundle is imported or evaluation throws.
      onEvent: { addListener: () => {} },
      onDetach: { addListener: () => {} },
    },
    tabs: { get: async () => ({}), onRemoved: { addListener: () => {} } },
  };
  return { detached, surfaces: () => surfaces };
}

test("a removed tab we own is reclaimed and reported by handle", async () => {
  const token = "bridge:42:abc123";
  const module = await loadModule({ [token]: { tabId: 42, nonce: "abc123", epoch: 1, createdAt: 1, lastUsedAt: 2 } });
  const chromeState = module.chromeState;
  try {
    const announced = await module.loaded.reclaimRemovedTab(42);
    // The handle is what the daemon needs: it clears exactly this tab rather
    // than blanking every session's driven record.
    assert.equal(announced, token);
    // Full reclaim, so a dead tab cannot hold a slot against MAX_SURFACES.
    assert.deepEqual(chromeState.surfaces(), {});
  } finally {
    await module.close();
  }
});

test("a tab we do not own is ignored", async () => {
  const token = "bridge:42:abc123";
  const module = await loadModule({ [token]: { tabId: 42, nonce: "abc123", epoch: 1, createdAt: 1, lastUsedAt: 2 } });
  const chromeState = module.chromeState;
  try {
    // The user closing one of their OWN tabs must not announce anything: a
    // tab_closed here would clear a driven record that is still live.
    assert.equal(await module.loaded.reclaimRemovedTab(999), undefined);
    assert.deepEqual(Object.keys(chromeState.surfaces()), [token]);
  } finally {
    await module.close();
  }
});

test("reclaiming is idempotent so a double event cannot throw", async () => {
  const token = "bridge:7:dead99";
  const module = await loadModule({ [token]: { tabId: 7, nonce: "dead99", epoch: 1, createdAt: 1, lastUsedAt: 2 } });
  try {
    assert.equal(await module.loaded.reclaimRemovedTab(7), token);
    // onRemoved after an explicit close (chrome.tabs.remove fires it too).
    assert.equal(await module.loaded.reclaimRemovedTab(7), undefined);
  } finally {
    await module.close();
  }
});

test("ownedSurfaceFor picks the right surface among several", async () => {
  const module = await loadModule({
    "bridge:1:aaa": { tabId: 1, nonce: "aaa", epoch: 1, createdAt: 1, lastUsedAt: 1 },
    "bridge:2:bbb": { tabId: 2, nonce: "bbb", epoch: 1, createdAt: 1, lastUsedAt: 2 },
  });
  try {
    assert.equal(await module.loaded.ownedSurfaceFor(2), "bridge:2:bbb");
    assert.equal(await module.loaded.ownedSurfaceFor(3), undefined);
  } finally {
    await module.close();
  }
});
