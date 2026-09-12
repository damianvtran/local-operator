/* Coverage for the browser-bridge wedge on the EXTENSION side (design record
 * §8.2), plus two smaller live defects from the same family.
 *
 * The wedge: every module-level serialized queue in the worker is built as
 * `queue.catch(() => {}).then(op)` so "each link swallows its predecessor's
 * failure so the chain cannot poison later calls". That is true for a
 * REJECTION and false for a HANG — `.catch()` never runs on a promise that
 * never settles — so one stuck chrome/CDP call parked every later command for
 * every session. Reproduced on real hardware: a driven tab whose renderer
 * stopped answering made `read` and then `tabs` time out at 20.0 s each, with
 * the poison held in the per-OWNER lane, and `/health` reporting "connected"
 * for all 42 samples of the window.
 *
 * Every test here therefore asserts the STRUCTURAL consequence the design
 * rests on — the chain DRAINED and the next command answered — never a
 * duration (AGENTS.md, "Timing, flakes, and how to assert that something is
 * fast"). Deadlines are shortened through a fixture alias so a bounded failure
 * arrives in tens of milliseconds; the real values live in `src/settle.ts` and
 * are pinned there as one table.
 */
import assert from "node:assert/strict";
import test from "node:test";
import { build } from "esbuild";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const SRC = resolve(HERE, "..", "src");
const REAL_SETTLE = join(SRC, "settle.ts");
const REAL_TAB_GROUPS = join(SRC, "tab-groups.ts");

const tick = (n = 4) => new Promise((r) => setTimeout(r, n));
const clone = (value) => structuredClone(value);

/** Fail rather than hang when a chain is parked. The budget is far above the
 * fixture's 40 ms deadline and far below "forever", so a parked chain produces
 * a failure that NAMES the defect instead of a node:test timeout. */
function within(promise, ms, what) {
  let timer;
  const guard = new Promise((_, reject) => {
    timer = setTimeout(
      () => reject(new Error(`${what}: chain is parked, not slow (no settle in ${ms}ms)`)),
      ms,
    );
    timer.unref?.();
  });
  return Promise.race([promise, guard]).finally(() => clearTimeout(timer));
}

/** Bundle a synthetic entry module against isolated chrome fixtures. */
async function load(entrySource, { aliasTabGroups = null } = {}) {
  const dir = await mkdtemp(join(tmpdir(), "lop-bridge-wedge-it-"));
  const entry = join(dir, "entry.mjs");
  await writeFile(entry, entrySource);
  const fixture = {
    // The real helper with the real per-call semantics, and CATASTROPHE
    // CEILINGS shrunk from seconds to tens of milliseconds so a bounded
    // failure is observable without a suite that sleeps.
    settle: `export { settle, deadline } from ${JSON.stringify(REAL_SETTLE)};
      export const CDP_DEADLINE_MS = 40;
      export const CDP_ATTACH_DEADLINE_MS = 40;
      export const CHROME_API_DEADLINE_MS = 40;
      export const SCRIPTING_DEADLINE_MS = 40;`,
  };
  if (aliasTabGroups) fixture["tab-groups"] = aliasTabGroups;
  const outfile = join(dir, "bundle.mjs");
  await build({
    entryPoints: [entry],
    bundle: true,
    platform: "node",
    format: "esm",
    outfile,
    plugins: [
      {
        name: "isolated-transport",
        setup(b) {
          b.onResolve({ filter: /^\.\.?\/(settle|tab-groups)$/ }, (a) => {
            const name = a.path.split("/").at(-1);
            return fixture[name] ? { path: name, namespace: "fixture" } : undefined;
          });
          b.onLoad({ filter: /.*/, namespace: "fixture" }, (a) => ({
            contents: fixture[a.path],
            loader: "js",
            resolveDir: SRC,
          }));
        },
      },
    ],
  });
  const loaded = await import(pathToFileURL(outfile) + `?${Date.now()}`);
  return { loaded, close: () => rm(dir, { recursive: true, force: true }) };
}

/** Isolated chrome surface. `overrides` replace individual APIs by namespace;
 * `hangFirstSessionGet` parks the first storage read, which is how the
 * pre-deadline wedge is simulated. */
function installChrome({ overrides = {}, hangFirstSessionGet = false } = {}) {
  const store = {};
  const log = { tabsGet: [], group: [], groupsUpdate: [], attach: [], sendCommand: [], set: [] };
  const area = (target) => ({
    get: async (keys) => {
      const out = {};
      for (const key of Array.isArray(keys) ? keys : [keys]) {
        if (key in target) out[key] = clone(target[key]);
      }
      return out;
    },
    set: async (value) => {
      log.set.push(clone(value));
      Object.assign(target, clone(value));
    },
    remove: async (keys) => {
      for (const key of Array.isArray(keys) ? keys : [keys]) delete target[key];
    },
    onChanged: { addListener: () => {} },
  });
  const base = {
    storage: { session: area(store), local: area({}), onChanged: { addListener: () => {} } },
    tabs: {
      get: async (tabId) => {
        log.tabsGet.push(tabId);
        return { id: tabId, windowId: 1, groupId: -1, url: "https://example.test/", title: "T" };
      },
      group: async () => {
        log.group.push(1);
        return 10;
      },
      remove: async () => {},
      create: async () => ({ id: 1 }),
      update: async () => {},
      query: async () => [],
      onRemoved: { addListener: () => {} },
      onReplaced: { addListener: () => {} },
      onUpdated: { addListener: () => {} },
    },
    tabGroups: {
      get: async (groupId) => ({ id: groupId, title: "LO · A", collapsed: false }),
      update: async (...args) => {
        log.groupsUpdate.push(args);
      },
    },
    debugger: {
      attach: async (...args) => {
        log.attach.push(args);
      },
      detach: async () => {},
      sendCommand: async (...args) => {
        log.sendCommand.push(args);
        return {};
      },
      onEvent: { addListener: () => {} },
      onDetach: { addListener: () => {} },
    },
    scripting: { executeScript: async () => [] },
    webNavigation: {
      onCompleted: { addListener: () => {}, removeListener: () => {} },
      onErrorOccurred: { addListener: () => {}, removeListener: () => {} },
      onBeforeNavigate: { addListener: () => {}, removeListener: () => {} },
      onHistoryStateUpdated: { addListener: () => {}, removeListener: () => {} },
    },
    alarms: { create: () => {}, clear: async () => {}, onAlarm: { addListener: () => {} } },
    notifications: { create: async () => {}, clear: async () => {}, onClicked: { addListener: () => {} } },
    windows: { get: async () => ({ id: 1 }), getCurrent: async () => ({ id: 1 }) },
    action: Object.fromEntries(
      ["setBadgeBackgroundColor", "setBadgeTextColor", "setBadgeText", "setTitle"].map((m) => [
        m,
        async () => {},
      ]),
    ),
    runtime: {
      getURL: (path) => `chrome-extension://test/${path}`,
      getManifest: () => ({ version: "0.1.10" }),
      onStartup: { addListener: () => {} },
      onInstalled: { addListener: () => {} },
      onMessage: { addListener: () => {} },
      sendMessage: async () => {},
    },
  };
  for (const [namespace, patch] of Object.entries(overrides)) {
    base[namespace] = { ...base[namespace], ...patch };
  }
  if (hangFirstSessionGet) {
    let calls = 0;
    const realGet = base.storage.session.get;
    base.storage.session.get = async (keys) => {
      calls += 1;
      if (calls === 1) return new Promise(() => {});
      return realGet(keys);
    };
  }
  globalThis.chrome = base;
  return { store, log, chrome: base };
}

const SURFACE_A = {
  tabId: 101,
  nonce: "aaaa",
  epoch: 0,
  createdAt: 0,
  lastUsedAt: 0,
  ownerKey: "session:s1",
  groupBaseLabel: "LO · A",
  groupOrdinal: 1,
};
const SURFACE_B = { ...SURFACE_A, tabId: 102, nonce: "bbbb", ownerKey: "session:s2" };

// --- E1: the store queue drains -------------------------------------------

test("E1 deadline drains the store queue", async () => {
  // The first storage read never settles. `withStore` chains the SECOND op onto
  // it with `.catch(() => {}).then(op)`, and `.catch` never runs on a promise
  // that never settles — so pre-fix the second putSurface never runs at all.
  const { store } = installChrome({ hangFirstSessionGet: true });
  const state = await load(`export * from ${JSON.stringify(join(SRC, "state.ts"))};`);
  try {
    const stalled = state.loaded.putSurface(SURFACE_A);
    const stalledSettled = assert.rejects(
      stalled,
      (error) => error.code === "internal" && String(error.data.stalled).includes("surfaces"),
    );
    await within(state.loaded.putSurface(SURFACE_B), 2_000, "the second putSurface");
    await stalledSettled;
    assert.ok(
      store.surfaces?.["bridge:102:bbbb"],
      "the second putSurface really ran rather than resolving without writing",
    );
  } finally {
    await state.close();
    delete globalThis.chrome;
  }
});

// --- E2: the group queue drains, across surfaces --------------------------

test("E2 deadline drains the group queue, and one session's stuck tab does not park another's", async () => {
  const { store, log } = installChrome({
    overrides: {
      tabs: {
        get: async (tabId) => {
          log.tabsGet.push(tabId);
          // The reported incident's exact shape: one driven tab whose renderer
          // stopped answering. Everything else is healthy.
          if (tabId === SURFACE_A.tabId) return new Promise(() => {});
          return { id: tabId, windowId: 1, groupId: -1, url: "https://example.test/", title: "T" };
        },
      },
    },
  });
  store.surfaces = {
    "bridge:101:aaaa": clone(SURFACE_A),
    "bridge:102:bbbb": clone(SURFACE_B),
  };
  const groups = await load(`export * from ${JSON.stringify(join(SRC, "tab-groups.ts"))};`);
  try {
    const owner = (label) => ({ requester: "session:s1", session_label: label });
    const first = groups.loaded.reconcileCommandTab({ tab: "bridge:101:aaaa", ...owner("A") });
    const second = groups.loaded.reconcileCommandTab({ tab: "bridge:102:bbbb", ...owner("B") });
    await within(second, 2_000, "session B's grouping behind session A's stuck tab");
    assert.ok(
      log.tabsGet.includes(SURFACE_B.tabId),
      "session B's op entered and ran — it was not queued behind the wedge",
    );
    // The first one is bounded too: it settles (as a rejection or a bail-out),
    // rather than parking the module-global queue for the worker's lifetime.
    await within(Promise.allSettled([first]), 2_000, "session A's own op");
  } finally {
    await groups.close();
    delete globalThis.chrome;
  }
});

// --- E3: the per-owner lane drains ---------------------------------------

test("E3 deadline drains the ownership lane so owner_recover still answers", async () => {
  // The reporting session's own path: an op parked in owner 1's lane, and
  // `owner_recover` — the one command whose job is recovery — queued behind it
  // and reported as a version mismatch. The hang is a chrome await INSIDE the
  // op (the replay path's tabs.get), which is what the per-call deadline
  // bounds: the op settles, the lane drains, the next owner command answers.
  const proof = "a".repeat(40);
  const params = {
    owner_proof: proof,
    requester: "session:s1",
    owner_generation: "g1",
    allocation_id: "a1",
  };
  let firstTabsGet = true;
  const { store } = installChrome({
    overrides: {
      tabs: {
        get: async (tabId) => {
          if (firstTabsGet) {
            firstTabsGet = false;
            return new Promise(() => {});
          }
          return { id: tabId, windowId: 1, groupId: -1, url: "https://example.test/", title: "T" };
        },
      },
    },
  });
  store.ownerScopes = {
    [proof]: {
      session: "session:s1",
      generation: "g1",
      allocations: { a1: { tab: "bridge:101:aaaa", state: "allocated" } },
    },
  };
  store.surfaces = { "bridge:101:aaaa": clone(SURFACE_A) };
  const ownership = await load(`export * from ${JSON.stringify(join(SRC, "ownership.ts"))};`);
  try {
    // `open` WITHOUT a tab param replays the journaled allocation, which reads
    // the live tab: that read is the one that hangs.
    const parked = ownership.loaded.withOwnership(
      "open",
      params,
      async () => ({ tab: "bridge:101:aaaa", url: "u", title: "t" }),
      async () => ({}),
    );
    const parkedSettled = parked.then(
      () => "resolved",
      (error) => (error?.data?.stalled ? "bounded" : "rejected"),
    );
    const recovered = await within(
      ownership.loaded.withOwnership(
        "owner_recover",
        params,
        async () => ({}),
        async () => ({}),
      ),
      2_000,
      "owner_recover behind a parked op on the SAME owner proof",
    );
    assert.equal(await parkedSettled, "bounded", "the parked op settled on its deadline");
    assert.equal(recovered.state, "allocated");
    assert.equal(recovered.tab, "bridge:101:aaaa");
  } finally {
    await ownership.close();
    delete globalThis.chrome;
  }
});

// --- E4 / E5 / X1: CDP ------------------------------------------------------

test("E4 a CDP command deadline is typed and retryable", async () => {
  const { log } = installChrome({
    overrides: {
      debugger: {
        sendCommand: async (_target, method) => {
          log.sendCommand.push(method);
          if (method === "Page.captureScreenshot") return new Promise(() => {});
          return { ok: true };
        },
      },
    },
  });
  const cdp = await load(`export * from ${JSON.stringify(join(SRC, "cdp.ts"))};`);
  try {
    await assert.rejects(
      within(
        cdp.loaded.cdp(101, "Page.captureScreenshot", { format: "png" }),
        2_000,
        "the CDP command",
      ),
      (error) =>
        error.code === "internal" &&
        error.data.stalled === "chrome.debugger.sendCommand(Page.captureScreenshot)",
    );
    // Nothing global was poisoned: a different CDP call on the same worker
    // still answers.
    assert.deepEqual(await cdp.loaded.cdp(101, "Runtime.evaluate", {}), { ok: true });
  } finally {
    await cdp.close();
    delete globalThis.chrome;
  }
});

test("E5 the attach deadline does not mask a real debugger conflict", async () => {
  // "Verify it still refuses where it should": the 5 s bound on
  // `ownAttachment`'s trivial probe must not convert DevTools' real conflict
  // into a generic stall.
  installChrome({
    overrides: {
      debugger: {
        attach: async () => {
          throw new Error("Another debugger is already attached to the tab with id: 101.");
        },
        sendCommand: async () => new Promise(() => {}),
      },
    },
  });
  const cdp = await load(`export * from ${JSON.stringify(join(SRC, "cdp.ts"))};`);
  try {
    await assert.rejects(cdp.loaded.attach(101), (error) => error.code === "debugger_conflict");
  } finally {
    await cdp.close();
    delete globalThis.chrome;
  }
});

test("X1 an undrivable page is a typed, actionable failure AND prunes the surface", async () => {
  // Live sighting from the reporter's browser (NOT reproduced in our own rig —
  // Chrome 153 let a second extension's attach succeed alongside ours, so this
  // path is untested there rather than refuted): Chrome refuses to debug
  // another extension's page, and — unlike a failed `chrome.tabs.get` — the
  // refusal used to leave the surface in the map, so every later command for
  // it failed the same way, including a fresh `open`.
  const { store } = installChrome({
    overrides: {
      debugger: {
        attach: async () => {
          throw new Error(
            "Cannot access a chrome-extension:// URL of different extension",
          );
        },
      },
    },
  });
  store.surfaces = { "bridge:101:aaaa": clone(SURFACE_A) };
  const cdp = await load(
    `export * from ${JSON.stringify(join(SRC, "cdp.ts"))};
     export * from ${JSON.stringify(join(SRC, "state.ts"))};`,
  );
  try {
    await assert.rejects(
      cdp.loaded.attach(SURFACE_A.tabId),
      (error) =>
        error.code === "internal" && error.data.undrivable_tab === "different_extension",
    );
    assert.deepEqual(
      Object.keys(store.surfaces ?? {}),
      [],
      "the surface must be retired so the session's next open starts clean",
    );
  } finally {
    await cdp.close();
    delete globalThis.chrome;
  }
});

// --- Worker-level rows: E6, E8, X2 ----------------------------------------

/** Load worker.ts against a fake socket the test can drive. */
async function loadWorker({ sendMessage, aliasTabGroups = null } = {}) {
  const handles = installChrome({
    overrides: {
      runtime: {
        getURL: (path) => `chrome-extension://test/${path}`,
        getManifest: () => ({ version: "0.1.10" }),
        onStartup: { addListener: () => {} },
        onInstalled: { addListener: () => {} },
        onMessage: { addListener: () => {} },
        sendMessage: sendMessage ?? (async () => {}),
      },
    },
  });
  Object.defineProperty(globalThis, "navigator", {
    value: { userAgent: "node-test" },
    configurable: true,
    writable: true,
  });
  const sent = [];
  const storageListeners = [];
  handles.chrome.storage.onChanged.addListener = (fn) => storageListeners.push(fn);
  let socket;
  globalThis.WebSocket = class {
    static OPEN = 1;
    readyState = 1;
    constructor() {
      socket = this;
      queueMicrotask(() => this.onopen?.());
    }
    send(data) {
      sent.push(JSON.parse(String(data)));
    }
    close() {}
  };
  const worker = await load(
    `export * from ${JSON.stringify(join(SRC, "worker.ts"))};`,
    { aliasTabGroups },
  );
  for (let i = 0; i < 60 && !socket; i++) await tick();
  assert.ok(socket, "the worker opened a socket");
  await tick(20);
  return {
    sent,
    storageListeners,
    handles,
    get socket() {
      return socket;
    },
    deliver: (frame) => socket.onmessage?.({ data: JSON.stringify(frame) }),
    close: worker.close,
  };
}

test("E6 a healthy worker pongs without entering any queue", async () => {
  // The structural fact the whole daemon-side detector rests on: the pong is
  // answered in the socket's own onmessage handler, off every serialized
  // queue. If a future refactor routes `ping` through a queue, a quiet-but-busy
  // worker stops ponging and the daemon tears down a perfectly healthy link.
  //
  // CHARACTERISATION, not a pre-fix guard: this is already true today. Proving
  // it can fail means moving the ping branch behind the queue and watching the
  // assertion go red (recorded in the PR).
  const worker = await loadWorker({
    // Park EVERY command inside the module-global group queue — the wedge.
    aliasTabGroups: `export * from ${JSON.stringify(REAL_TAB_GROUPS)};
      export const reconcileCommandTab = () => new Promise(() => {});`,
  });
  try {
    worker.deliver({ id: "r-parked", method: "status", params: {} });
    await tick(20);
    worker.deliver({ event: "ping" });
    await tick(20);
    assert.deepEqual(
      worker.sent.filter((frame) => frame.event === "pong"),
      [{ event: "pong" }],
      "a worker with a command parked in a queue must still pong",
    );
  } finally {
    await worker.close();
    delete globalThis.chrome;
  }
});

test("E8 a response the socket can no longer carry is logged, not silently dropped", async () => {
  const worker = await loadWorker();
  const warnings = [];
  const realWarn = console.warn;
  console.warn = (...args) => warnings.push(args.map(String).join(" "));
  try {
    // The daemon evicted this socket mid-command ("later connection wins",
    // close 4000): the answer is computed and has nowhere to go. There is no
    // correct place to deliver it, so the fix is observability — the 23
    // `extension disconnected` events in the operator's live log are this path.
    worker.socket.readyState = 3;
    worker.deliver({ id: "r-dropped", method: "retitle", params: { tab: "bridge:1:n" } });
    await tick(40);
    assert.ok(
      warnings.some((line) => line.includes("dropped response for r-dropped")),
      `expected a drop warning, got ${JSON.stringify(warnings)}`,
    );
  } finally {
    console.warn = realWarn;
    await worker.close();
    delete globalThis.chrome;
  }
});

test("X2 a runtime.sendMessage with no receiver cannot surface as an uncaught error", async () => {
  // Captured from the operator's hands: with no popup open there is no
  // receiver, so the origin_prompt broadcast rejects with "Could not establish
  // connection. Receiving end does not exist." and the worker console shows
  // `Uncaught (in promise)` — which is indistinguishable from a crash to
  // anyone reading the console to diagnose a bridge fault.
  const rejections = [];
  const onRejection = (error) => rejections.push(error);
  process.on("unhandledRejection", onRejection);
  const worker = await loadWorker({
    sendMessage: async () => {
      throw new Error("Could not establish connection. Receiving end does not exist.");
    },
  });
  const warnings = [];
  const realWarn = console.warn;
  console.warn = (...args) => warnings.push(args.map(String).join(" "));
  try {
    assert.equal(worker.storageListeners.length, 1, "the worker registers one storage listener");
    worker.storageListeners[0](
      { accessQueue: { newValue: { queue: [], results: {}, onceGrants: {} } } },
      "session",
    );
    await tick(60);
    assert.deepEqual(rejections, [], "the rejection escaped as an unhandled rejection");
    assert.ok(
      warnings.some((line) => line.includes("origin_prompt broadcast")),
      `expected the rejection to be recorded, got ${JSON.stringify(warnings)}`,
    );
  } finally {
    process.off("unhandledRejection", onRejection);
    console.warn = realWarn;
    await worker.close();
    delete globalThis.chrome;
  }
});

// --- E7: the grant lane is bounded too (review R1-3) ----------------------

test("E7 a hung chrome.storage.LOCAL read in the grant lane drains like any other", async () => {
  // The popup's Allow path — worker dispatch → origins.resolveOrigin →
  // approval-store.decideAccess → access-grants.grantExactOriginLocked — runs
  // inside `withSessionMutation`, which IS `withStore` (state.ts). One bare
  // `chrome.storage.local.get` here therefore parks EVERY command of EVERY
  // session: the incident's exact shape, reached by clicking a button rather
  // than by a stalled page. Eleven such awaits lived in this file; this pins the
  // class so the grep in state.ts's comment stays true.
  let reads = 0;
  const local = {};
  const localArea = {
    get: async (keys) => {
      reads += 1;
      if (reads === 1) return new Promise(() => {});
      const out = {};
      for (const key of Array.isArray(keys) ? keys : [keys]) {
        if (key in local) out[key] = clone(local[key]);
      }
      return out;
    },
    set: async (value) => Object.assign(local, clone(value)),
    remove: async (keys) => {
      for (const key of Array.isArray(keys) ? keys : [keys]) delete local[key];
    },
    onChanged: { addListener: () => {} },
  };
  installChrome({ overrides: { storage: { local: localArea } } });
  const grants = await load(
    `export { grantExactOrigin } from ${JSON.stringify(join(SRC, "access-grants.ts"))};`,
  );
  try {
    const parked = grants.loaded.grantExactOrigin("https://first.test/");
    const parkedSettled = assert.rejects(
      parked,
      (error) =>
        error.code === "internal" &&
        String(error.data.stalled).includes("chrome.storage.local.get"),
    );
    await within(
      grants.loaded.grantExactOrigin("https://second.test/"),
      2_000,
      "the grant after the parked one",
    );
    await parkedSettled;
    assert.equal(
      local.origins?.["https://second.test/"],
      "allow",
      "the next grant really ran rather than resolving without writing",
    );
  } finally {
    await grants.close();
    delete globalThis.chrome;
  }
});

// --- Audit A3 / scoping D1: the AX lane is per target -----------------------

test("A3 a snapshot stalled on one tab does not hold another tab's lane (D1)", async () => {
  // The lane exists because two snapshots of the SAME tab could interleave an
  // enable/read/disable window against one a11y engine. It was module-global, so
  // a stalled `getFullAXTree` (plus the `finally` disable, both inside `run`)
  // also held a DIFFERENT owner's unrelated tab: that owner could not issue even
  // `Accessibility.enable` before its own 20 s daemon budget expired. The hazard
  // is one engine, and the CDP `Accessibility` domain is per debuggee (`cdp()` is
  // addressed by tabId), so per-target is the correct scope — not merely a
  // smaller one.
  //
  // The assertion is ORDERING, not a duration: with a per-tab lane owner B's
  // first CDP call must fall INSIDE owner A's stalled window (A's enable and its
  // `finally` disable bracket one another around B's work); with a global lane
  // every one of A's calls completes first and B's can only start after A's
  // chain drains. Neither a timeout nor a stopwatch is involved, so the row is
  // not sensitive to how long the aliased ceilings are.
  const calls = [];
  const { store } = installChrome({
    overrides: {
      debugger: {
        sendCommand: async (target, method) => {
          calls.push([target.tabId, method]);
          // Owner A's AX READ answers nothing, so A's chain is stalled inside
          // its window: `Accessibility.getFullAXTree` never resolves and the
          // `finally` disable is therefore still ahead of it. (Parking the
          // enable instead would abort `run` before the window even opens, and
          // then there would be no window for B to interleave with — which is
          // exactly what made an earlier version of this row pass against the
          // defective build.)
          if (target.tabId === SURFACE_A.tabId && method === "Accessibility.getFullAXTree") {
            await new Promise(() => {});
          }
          return { nodes: [] };
        },
      },
    },
  });
  store.surfaces = {
    "bridge:101:aaaa": clone(SURFACE_A),
    "bridge:102:bbbb": clone(SURFACE_B),
  };
  const snap = await load(`export * from ${JSON.stringify(join(SRC, "commands", "snapshot.ts"))};`);
  try {
    const parked = snap.loaded.snapshot({ tab: "bridge:101:aaaa" });
    const parkedSettled = parked.then(
      () => "resolved",
      () => "stalled",
    );
    for (let i = 0; i < 40 && !calls.some(([tab, method]) => tab === SURFACE_A.tabId && method === "Accessibility.getFullAXTree"); i++) {
      await tick(5);
    }
    assert.ok(
      calls.some(([tab, method]) => tab === SURFACE_A.tabId && method === "Accessibility.getFullAXTree"),
      "precondition: owner A's AX read is in flight",
    );

    const other = await within(
      snap.loaded.snapshot({ tab: "bridge:102:bbbb" }),
      2_000,
      "owner B's snapshot behind owner A's stalled one",
    );
    assert.ok(other.snapshot !== undefined, "owner B's snapshot answered");
    // Snapshot the CDP trace at the INSTANT B answered. Under a global lane B's
    // chain cannot start until A's whole chain drains — and A's chain includes
    // its `finally` `Accessibility.disable` — so A's disable appears here only
    // if B waited for it. No stopwatch: the presence of that one call IS the
    // ordering.
    const whileBAnswered = [...calls];
    assert.ok(
      whileBAnswered.some(
        ([tab, method]) => tab === SURFACE_A.tabId && method === "Accessibility.getFullAXTree",
      ),
      "precondition: owner A's stalled AX read is still in flight while B answers",
    );
    assert.equal(
      whileBAnswered.some(([tab, method]) => tab === SURFACE_A.tabId && method === "Accessibility.disable"),
      false,
      `owner B's CDP work must interleave with owner A's stalled window, not queue behind its chain (calls: ${JSON.stringify(whileBAnswered)})`,
    );
    // A's own snapshot is bounded and typed — the stall settles rather than
    // parking its lane forever, which is what lets the LAST assertion below
    // finish.
    assert.equal(await parkedSettled, "stalled");

    // A drained lane leaves nothing behind: a fresh snapshot runs.
    const third = await within(
      snap.loaded.snapshot({ tab: "bridge:102:bbbb" }),
      2_000,
      "a later snapshot after the stalled chain drained",
    );
    assert.ok(third.snapshot !== undefined);
  } finally {
    await snap.close();
    delete globalThis.chrome;
  }
});

// --- X3/X4: the worker's remaining uncaught paths (popup-stale-worker lane) --
//
// X2 above covers the fire-and-forget `sendMessage` broadcast. These two cover
// the paths that were still able to throw OUT of a Chrome event handler on that
// fix's head — which is the same defect with a different origin: an uncaught
// error in an MV3 worker is the state Chrome's worker is poisoned into, and a
// poisoned worker is what made the operator's toolbar clicks do nothing.

test("X3 an unparseable daemon frame does not throw out of the socket handler", async () => {
  const rejections = [];
  const uncaught = [];
  const onRejection = (error) => rejections.push(error);
  const onUncaught = (error) => uncaught.push(error);
  process.on("unhandledRejection", onRejection);
  process.on("uncaughtException", onUncaught);
  const worker = await loadWorker();
  const warnings = [];
  const realWarn = console.warn;
  console.warn = (...args) => warnings.push(args.map(String).join(" "));
  try {
    // A truncated write, a proxy injecting a body, a future protocol version:
    // whatever the cause, it reaches onmessage as a string JSON.parse rejects.
    // The old handler let that throw, taking the whole event handler with it.
    worker.socket.onmessage?.({ data: "{not json" });
    await tick(40);

    // The observation is the harness's, not a string grep: nothing escaped.
    assert.deepEqual(rejections, [], "a bad frame escaped as an unhandled rejection");
    assert.deepEqual(uncaught, [], "a bad frame escaped as an uncaught exception");
    assert.ok(
      warnings.some((line) => line.includes("unparseable frame")),
      `expected the drop to be recorded, got ${JSON.stringify(warnings)}`,
    );

    // And the SOCKET survives it: the next good frame is still answered, which
    // is the whole point of dropping the frame rather than the connection.
    worker.deliver({ event: "ping" });
    await tick(40);
    assert.deepEqual(
      worker.sent.filter((frame) => frame.event === "pong"),
      [{ event: "pong" }],
      "one bad byte on the wire must not cost the live socket",
    );
  } finally {
    process.off("unhandledRejection", onRejection);
    process.off("uncaughtException", onUncaught);
    console.warn = realWarn;
    await worker.close();
    delete globalThis.chrome;
  }
});

test("X4 a socket constructor that throws leaves the worker able to dial again", async () => {
  // `new WebSocket()` throws synchronously on a malformed URL, and the port it
  // is built from comes out of chrome.storage — so a corrupted stored value
  // reaches that constructor. Every caller is a fire-and-forget from an event
  // handler, so the throw surfaced as an uncaught worker error instead of a
  // failed dial.
  const rejections = [];
  const uncaught = [];
  const onRejection = (error) => rejections.push(error);
  const onUncaught = (error) => uncaught.push(error);
  process.on("unhandledRejection", onRejection);
  process.on("uncaughtException", onUncaught);
  installChrome({
    overrides: {
      runtime: {
        getURL: (path) => `chrome-extension://test/${path}`,
        getManifest: () => ({ version: "0.1.12" }),
        onStartup: { addListener: () => {} },
        onInstalled: { addListener: () => {} },
        onMessage: { addListener: () => {} },
        sendMessage: async () => {},
      },
    },
  });
  Object.defineProperty(globalThis, "navigator", {
    value: { userAgent: "node-test" },
    configurable: true,
    writable: true,
  });
  let constructed = 0;
  globalThis.WebSocket = class {
    static OPEN = 1;
    readyState = 1;
    constructor() {
      constructed += 1;
      // Only the FIRST dial throws. A worker that could never dial again would
      // pass a "no uncaught error" assertion while being exactly as dead as the
      // one this fix is about, so recovery is what is actually asserted below.
      if (constructed === 1) throw new SyntaxError("'ws://127.0.0.1:NaN/extension' is invalid.");
      queueMicrotask(() => this.onopen?.());
    }
    send() {}
    close() {}
  };
  const warnings = [];
  const realWarn = console.warn;
  console.warn = (...args) => warnings.push(args.map(String).join(" "));
  let worker;
  try {
    worker = await load(`export * from ${JSON.stringify(join(SRC, "worker.ts"))};`);
    await tick(40);
    assert.deepEqual(rejections, [], "the constructor throw escaped as an unhandled rejection");
    assert.deepEqual(uncaught, [], "the constructor throw escaped as an uncaught exception");
    assert.ok(
      warnings.some((line) => line.includes("dial failed to open a socket")),
      `expected the failed dial to be recorded, got ${JSON.stringify(warnings)}`,
    );
    assert.equal(constructed, 1, "precondition: exactly the first dial threw");

    // RECOVERY is the assertion that matters. A worker that swallowed the throw
    // and then never dialled again would satisfy every check above while being
    // exactly as dead as the wedge this PR is about. The fast-path backoff for
    // attempt 0 is 1s (reconnect.ts), so wait past it and require a SECOND
    // socket — which also drains the armed timer, leaving no live handle behind
    // for the next test in this file.
    for (let i = 0; i < 40 && constructed < 2; i++) await tick(50);
    assert.equal(constructed, 2, "a failed dial must be retried, not abandoned");
  } finally {
    process.off("unhandledRejection", onRejection);
    process.off("uncaughtException", onUncaught);
    console.warn = realWarn;
    if (worker) await worker.close();
    delete globalThis.chrome;
  }
});
