import assert from "node:assert/strict";
import test from "node:test";
import { build } from "esbuild";
import { pathToFileURL } from "node:url";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

async function loadModule() {
  const dir = await mkdtemp(join(tmpdir(), "lop-tab-groups-it-"));
  const outfile = join(dir, "module.mjs");
  await build({ entryPoints: ["src/tab-groups.ts"], bundle: true, platform: "node", format: "esm", outfile });
  return { loaded: await import(pathToFileURL(outfile) + `?${Date.now()}`), close: () => rm(dir, { recursive: true, force: true }) };
}

function installChrome({ groupReject = false, updateReject = false, APIs = true } = {}) {
  let surfaces = {};
  const tabs = new Map();
  const groups = new Map();
  const groupCalls = [];
  const updateCalls = [];
  let nextGroup = 10;
  globalThis.chrome = {
    storage: { session: {
      get: async () => ({ surfaces }),
      set: async (value) => { if (value.surfaces) surfaces = structuredClone(value.surfaces); },
    } },
    tabs: {
      get: async (id) => { if (!tabs.has(id)) throw new Error("dead"); return { ...tabs.get(id) }; },
      ...(APIs ? { group: async (options) => {
        groupCalls.push(structuredClone(options));
        if (groupReject) throw new Error("unsupported by policy");
        const id = options.groupId ?? nextGroup++;
        for (const tabId of options.tabIds) tabs.get(tabId).groupId = id;
        groups.set(id, groups.get(id) ?? { id, collapsed: false });
        return id;
      } } : {}),
    },
    ...(APIs ? { tabGroups: {
      get: async (id) => { if (!groups.has(id)) throw new Error("dead group"); return { ...groups.get(id) }; },
      update: async (id, options) => {
        updateCalls.push([id, structuredClone(options)]);
        if (updateReject) throw new Error("managed browser rejected title");
        groups.set(id, { ...(groups.get(id) ?? { id }), ...options });
        return groups.get(id);
      },
    } } : {}),
  };
  return {
    tabs, groups, groupCalls, updateCalls,
    seed: (surface, tab) => { surfaces[`bridge:${surface.tabId}:${surface.nonce}`] = structuredClone(surface); tabs.set(tab.id, { groupId: -1, ...tab }); },
    surface: (tabId) => Object.values(surfaces).find((surface) => surface.tabId === tabId),
    ungroup: (tabId) => { tabs.get(tabId).groupId = -1; },
    moveToGroup: (tabId, groupId, group = { title: "Personal", color: "red", collapsed: true }) => {
      tabs.get(tabId).groupId = groupId;
      groups.set(groupId, { id: groupId, ...group });
    },
    restore: () => { delete globalThis.chrome; },
  };
}

const surface = (tabId, extra = {}) => ({ tabId, nonce: `nonce${tabId}`, epoch: 1, createdAt: tabId, lastUsedAt: tabId, ...extra });
const params = (owner, label) => ({ requester: `session:${owner}`, session_label: label });

for (const options of [{ APIs: false }, { groupReject: true }, { updateReject: true }]) {
  test(`group API failure stays best-effort ${JSON.stringify(options)}`, async () => {
    const chrome = installChrome(options);
    const bundle = await loadModule();
    try {
      const owned = surface(1); chrome.seed(owned, { id: 1, windowId: 1, active: false });
      await assert.doesNotReject(bundle.loaded.reconcileTabGroup(owned, params("A", "Planning"), true));
      assert.equal((await chrome.tabs.get(1)).active, false); // grouping never activates the tab
    } finally { await bundle.close(); chrome.restore(); }
  });
}

test("same owner joins per-window groups without expanding existing groups", async () => {
  const chrome = installChrome();
  const bundle = await loadModule();
  try {
    const one = surface(1); const two = surface(2); const otherWindow = surface(3);
    chrome.seed(one, { id: 1, windowId: 7, active: false });
    chrome.seed(two, { id: 2, windowId: 7, active: false });
    chrome.seed(otherWindow, { id: 3, windowId: 8, active: false });
    await bundle.loaded.reconcileTabGroup(one, params("A", "Planning"), true);
    chrome.groups.get(10).collapsed = true;
    await bundle.loaded.reconcileTabGroup(two, params("A", "Planning"), true);
    await bundle.loaded.reconcileTabGroup(otherWindow, params("A", "Planning"), true);
    assert.deepEqual(chrome.groupCalls, [
      { tabIds: [1] },
      { groupId: 10, tabIds: [2] },
      { tabIds: [3] },
    ]);
    assert.equal(chrome.groups.get(10).collapsed, true);
    assert.equal(chrome.groups.get(11).title, "LO · Planning");
    assert.equal(chrome.updateCalls[1][1].collapsed, undefined, "joining must preserve collapsed state");
  } finally { await bundle.close(); chrome.restore(); }
});

test("duplicate labels get stable owner ordinals and rename updates in place", async () => {
  const chrome = installChrome();
  const bundle = await loadModule();
  try {
    const a = surface(1); const b = surface(2);
    chrome.seed(a, { id: 1, windowId: 1 }); chrome.seed(b, { id: 2, windowId: 1 });
    await bundle.loaded.reconcileTabGroup(a, params("A", "Same"), true);
    await bundle.loaded.reconcileTabGroup(b, params("B", "Same"), true);
    assert.equal(chrome.groups.get(10).title, "LO · Same");
    assert.equal(chrome.groups.get(11).title, "LO · Same (2)");
    assert.equal(chrome.surface(2).groupOrdinal, 2);
    await bundle.loaded.reconcileTabGroup(b, params("B", "Renamed"), false);
    assert.equal(chrome.groups.get(11).title, "LO · Renamed");
    assert.equal(chrome.groupCalls.length, 2, "rename must not create/rejoin a group");
  } finally { await bundle.close(); chrome.restore(); }
});

test("old stored surfaces backfill metadata and ignore dead siblings", async () => {
  const chrome = installChrome();
  const bundle = await loadModule();
  try {
    const old = surface(1); // exact 0.1.3 shape: no owner/group fields
    const dead = surface(99, { ownerKey: "session:A", groupBaseLabel: "LO · Name", groupOrdinal: 1 });
    chrome.seed(old, { id: 1, windowId: 1 });
    chrome.seed(dead, { id: 99, windowId: 1 });
    chrome.tabs.delete(99);
    await bundle.loaded.reconcileTabGroup(old, params("A", "Name"), true);
    assert.equal(chrome.surface(1).ownerKey, "session:A");
    assert.equal(chrome.surface(1).groupAppliedLabel, "LO · Name");
    assert.deepEqual(chrome.groupCalls, [{ tabIds: [1] }]);
  } finally { await bundle.close(); chrome.restore(); }
});

test("defensive sanitizer never exposes requester or invisible controls", async () => {
  const chrome = installChrome();
  const bundle = await loadModule();
  try {
    const owned = surface(1); chrome.seed(owned, { id: 1, windowId: 1 });
    await bundle.loaded.reconcileTabGroup(
      owned,
      { requester: "session:full-secret-uuid", session_label: " \u202e\u200b  " },
      true,
    );
    assert.equal(chrome.groups.get(10).title, "LO · Session");
    assert.doesNotMatch(chrome.groups.get(10).title, /full-secret-uuid/);
  } finally { await bundle.close(); chrome.restore(); }
});

test("ordinary rename never mutates a personal group after manual regrouping", async () => {
  const chrome = installChrome();
  const bundle = await loadModule();
  try {
    const owned = surface(1); chrome.seed(owned, { id: 1, windowId: 1 });
    await bundle.loaded.reconcileTabGroup(owned, params("A", "Name"), true);
    chrome.moveToGroup(1, 77);
    const updatesBefore = chrome.updateCalls.length;

    await bundle.loaded.reconcileTabGroup(chrome.surface(1), params("A", "Renamed"), false);

    assert.equal(chrome.updateCalls.length, updatesBefore, "ordinary command must not update personal group");
    assert.deepEqual(chrome.groups.get(77), {
      id: 77,
      title: "Personal",
      color: "red",
      collapsed: true,
    });
  } finally { await bundle.close(); chrome.restore(); }
});

test("explicit resume rejoins an LO group without mutating the personal group", async () => {
  const chrome = installChrome();
  const bundle = await loadModule();
  try {
    const owned = surface(1); chrome.seed(owned, { id: 1, windowId: 1 });
    await bundle.loaded.reconcileTabGroup(owned, params("A", "Name"), true);
    chrome.moveToGroup(1, 77);
    const updatesBefore = chrome.updateCalls.length;

    await bundle.loaded.reconcileTabGroup(chrome.surface(1), params("A", "Renamed"), true);

    assert.deepEqual(chrome.groups.get(77), {
      id: 77,
      title: "Personal",
      color: "red",
      collapsed: true,
    });
    assert.equal(chrome.updateCalls.length, updatesBefore + 1);
    assert.deepEqual(chrome.groupCalls.at(-1), { tabIds: [1] });
    assert.equal(chrome.groups.get(11).title, "LO · Renamed");
    assert.equal(chrome.surface(1).appliedGroupId, 11);
  } finally { await bundle.close(); chrome.restore(); }
});

test("recycled advisory group id with personal metadata fails safe", async () => {
  const chrome = installChrome();
  const bundle = await loadModule();
  try {
    const stale = surface(1, {
      ownerKey: "session:A",
      groupBaseLabel: "LO · Name",
      groupOrdinal: 1,
      groupAppliedLabel: "LO · Name",
      appliedGroupId: 88,
    });
    chrome.seed(stale, { id: 1, windowId: 1, groupId: 88 });
    chrome.moveToGroup(1, 88);

    await bundle.loaded.reconcileTabGroup(stale, params("A", "Renamed"), false);

    assert.equal(chrome.updateCalls.length, 0);
    assert.equal(chrome.groups.get(88).title, "Personal");
  } finally { await bundle.close(); chrome.restore(); }
});

test("stale advisory group id fails safe until explicit resume", async () => {
  const chrome = installChrome();
  const bundle = await loadModule();
  try {
    const stale = surface(1, {
      ownerKey: "session:A",
      groupBaseLabel: "LO · Name",
      groupOrdinal: 1,
      groupAppliedLabel: "LO · Name",
      appliedGroupId: 404,
    });
    chrome.seed(stale, { id: 1, windowId: 1, groupId: 88 });
    chrome.moveToGroup(1, 88);

    await bundle.loaded.reconcileTabGroup(stale, params("A", "Renamed"), false);
    assert.equal(chrome.updateCalls.length, 0);
    assert.equal(chrome.groups.get(88).title, "Personal");

    await bundle.loaded.reconcileTabGroup(chrome.surface(1), params("A", "Renamed"), true);
    assert.deepEqual(chrome.groupCalls, [{ tabIds: [1] }]);
    assert.equal(chrome.groups.get(10).title, "LO · Renamed");
  } finally { await bundle.close(); chrome.restore(); }
});

test("ordinary command respects manual ungrouping while resume rejoins", async () => {
  const chrome = installChrome();
  const bundle = await loadModule();
  try {
    const owned = surface(1); chrome.seed(owned, { id: 1, windowId: 1 });
    await bundle.loaded.reconcileTabGroup(owned, params("A", "Name"), true);
    chrome.ungroup(1);
    await bundle.loaded.reconcileTabGroup(chrome.surface(1), params("A", "Renamed"), false);
    assert.equal(chrome.groupCalls.length, 1, "ordinary commands respect manual ungrouping");
    await bundle.loaded.reconcileTabGroup(chrome.surface(1), params("A", "Renamed"), true);
    assert.equal(chrome.groupCalls.length, 2, "explicit resume is the rejoin point");
  } finally { await bundle.close(); chrome.restore(); }
});
