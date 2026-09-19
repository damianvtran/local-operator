/* The download handler's decisions, with a SCRIPTED `chrome.downloads`.
 *
 * WHAT THIS IS AND WHAT IT IS NOT. It is not the end-to-end evidence: that is a
 * real headless Chrome driving the built extension through the bridge, and it is
 * the one thing that can prove a file actually leaves the user's download folder.
 * It is the layer below — the handler's own decisions about a transfer it did NOT
 * start (no selector: the page starts it), which no other executable test covers:
 * when to cancel, what to report, and what the report must not claim.
 *
 * Why that layer needs a test even so: these three decisions are all about a
 * transfer that goes WRONG (over the ceiling, never finishing, arriving twice),
 * and every one of them ends with a file in the user's own folder that only the
 * harness can remove. A regression here is invisible in the happy path — a
 * handler that never cancelled would still save small files correctly.
 *
 * The fake is deliberately a fake and is named as one: `chrome.downloads` is
 * scripted per id, so a 256 MiB ceiling can be crossed without writing 256 MiB.
 * The handler reads `bytesReceived` from `search`, exactly as it does against
 * Chrome, so the decision under test is the real one.
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

async function load(entry) {
  const dir = await mkdtemp(join(tmpdir(), "lop-download-test-"));
  const outfile = join(dir, "module.mjs");
  await build({ entryPoints: [entry], bundle: true, platform: "node", format: "esm", outfile });
  const loaded = await import(pathToFileURL(outfile));
  return { loaded, close: () => rm(dir, { recursive: true, force: true }) };
}

function item(id, overrides = {}) {
  return {
    id,
    filename: `/Users/someone/Downloads/receipt.pdf`,
    state: "in_progress",
    bytesReceived: 10,
    totalBytes: 100,
    mime: "application/pdf",
    danger: "safe",
    exists: true,
    paused: false,
    error: "",
    ...overrides,
  };
}

/** A scripted `chrome`: each `search({})` advances every id's script by one step
 *  and clamps at the last entry, so a test states "this is what Chrome will say
 *  next" rather than timing anything. */
function installChrome(scripts, { consented = true } = {}) {
  const steps = new Map(
    Object.entries(scripts).map(([id, list]) => [Number(id), { list, index: 0 }]),
  );
  const cancels = [];
  const cancelled = new Set();
  const listeners = new Map();
  let polls = 0;
  globalThis.chrome = {
    storage: {
      // The operator's switch, as the options page would have written it. The
      // handler is only reached at all when this is true AND the permission is
      // held — the gate is tested separately below, and having to set it here is
      // itself a statement about the default.
      local: { get: async () => ({ allowDownloads: consented }), set: async () => undefined },
      session: {
        get: async () => ({
          surfaces: { [SURFACE]: { tabId: TAB, nonce: "abc123", epoch: 1, createdAt: 0, lastUsedAt: 0 } },
        }),
      },
    },
    permissions: { contains: async () => consented },
    tabs: {
      get: async (tabId) => {
        assert.equal(tabId, TAB, "the handler must only ever look at its own tab");
        return { url: "https://example.test/export" };
      },
    },
    // `cdp.ts` registers a detach listener at module scope; the handler never
    // uses the debugger (a page-initiated download needs no click), which is
    // exactly the property this test is pinning.
    debugger: { onDetach: { addListener: () => undefined } },
    downloads: {
      search: async (query = {}) => {
        // A script entry may be a value or a thunk: a thunk lets a test vary an
        // item per poll (bytes growing, state advancing) without keeping a mutable
        // object the handler could accidentally be handed twice.
        const resolve = (entry) => {
          const value = entry.list[entry.index];
          const item = typeof value === "function" ? value() : value;
          // A cancelled download goes to `interrupted` on the next read, as
          // Chrome's does — without this the scripted transfer would stay
          // `in_progress` and the handler would (correctly) keep waiting for the
          // deadline, which is 20 s of CI spent proving the fake's own staleness.
          return cancelled.has(item.id) ? { ...item, state: "interrupted" } : item;
        };
        if (query.id !== undefined) {
          const entry = steps.get(query.id);
          return entry ? [resolve(entry)] : [];
        }
        // The FIRST unqualified read is the handler's BASELINE ("what did Chrome
        // already know about?"), and it must be empty: a download the page has not
        // started yet does not exist. Returning scripted items there would make
        // every one of them look pre-existing — which is exactly what the handler
        // must ignore, since moving a file the user asked for by hand is the worst
        // failure it could have.
        polls += 1;
        if (polls === 1) return [];
        const out = [];
        for (const entry of steps.values()) {
          out.push(resolve(entry));
          if (entry.index < entry.list.length - 1) entry.index += 1;
        }
        return out;
      },
      cancel: async (id) => {
        cancels.push(id);
        cancelled.add(id);
      },
      onChanged: {
        addListener: (fn) => listeners.set("changed", fn),
        removeListener: (fn) => {
          if (listeners.get("changed") === fn) listeners.delete("changed");
        },
      },
    },
  };
  return { cancels, listeners };
}

test("the handler refuses a download while the operator's switch is off", async () => {
  // The LAST gate, and the one that matters when the advertisement is stale: a
  // daemon that predates the capability advertisement sends the command anyway,
  // and a switch flipped off between the advertisement and the command must still
  // mean off. The refusal names the SWITCH rather than the build — the capability
  // is present and only its consent is missing — and it happens before the tab is
  // touched at all, so nothing about the page is read.
  installChrome({}, { consented: false });
  const module = await load("src/commands/download.ts");
  try {
    await assert.rejects(
      () => module.loaded.download({ tab: SURFACE, timeout_s: 5 }, "r-0"),
      (error) => {
        assert.equal(error.code, "capability_unsupported");
        assert.match(error.message, /Allow downloads/);
        assert.match(error.message, /switched off/);
        return true;
      },
    );
  } finally {
    await module.close();
  }
});

test("a page-initiated download is reported with the absolute path Chrome wrote", async () => {
  // The happy path, in the one shape the E2E cannot easily produce on demand:
  // the page starts its own download (no selector), so the handler's whole job is
  // to observe it. `filename` is the ABSOLUTE path — the fact the whole
  // harness-side relocation depends on, and the reason this handler exists rather
  // than a CDP download primitive that could have chosen a destination.
  installChrome({
    1: [
      () => item(1),
      () => item(1, { state: "complete", bytesReceived: 100, exists: true }),
    ],
  });
  const module = await load("src/commands/download.ts");
  try {
    const report = await module.loaded.download({ tab: SURFACE, timeout_s: 20 }, "r-1");
    assert.equal(report.armed, true);
    assert.equal(report.url, "https://example.test/export");
    assert.deepEqual(
      report.files.map((file) => [file.name, file.state, file.cancelled, file.path]),
      [["receipt.pdf", "complete", "", "/Users/someone/Downloads/receipt.pdf"]],
    );
    assert.equal(report.note, undefined, "a completed transfer needs no warning");
  } finally {
    await module.close();
  }
});

test("the ceiling cancels the transfer instead of letting it land", async () => {
  // The decision rule 4 describes: a ceiling checked only after a file completes
  // has already written the file. The script crosses the ceiling while the
  // transfer is still in progress, and the assertion is on BOTH halves — the
  // cancel was asked for, and the report tells the harness why a partial file is
  // about to be found in the user's folder.
  const fake = installChrome({
    42: [
      () => item(42),
      () => item(42, { bytesReceived: 300 * 1024 * 1024, totalBytes: -1 }),
    ],
  });
  const module = await load("src/commands/download.ts");
  try {
    const report = await module.loaded.download({ tab: SURFACE, timeout_s: 20 }, "r-2");
    assert.deepEqual(fake.cancels, [42], "the transfer must be cancelled, not awaited");
    const [file] = report.files;
    assert.equal(file.cancelled, "over_cap");
    assert.equal(file.totalBytes, -1, "an unknown total is reported as unknown, never as a size");
    assert.match(report.note ?? "", /over_cap/);
  } finally {
    await module.close();
  }
});

test("a transfer that never finishes is cancelled and reported for cleanup", async () => {
  // The case that leaves a partial file behind with nothing watching it: the
  // page's server stalls, the command's own time runs out, and the file sits in
  // the user's download folder under the page's name. The deadline must cancel it
  // AND report the path, because nothing later in the pipeline ever looks outside
  // the quarantine root.
  const fake = installChrome({ 9: [() => item(9, { bytesReceived: 16, totalBytes: 4096 })] });
  const module = await load("src/commands/download.ts");
  try {
    const report = await module.loaded.download({ tab: SURFACE, timeout_s: 1 }, "r-3");
    assert.deepEqual(fake.cancels, [9]);
    const [file] = report.files;
    assert.equal(file.cancelled, "unfinished");
    // NOT `complete`: Chrome's own terminal state after a cancel is `interrupted`,
    // and the assertion is deliberately on the invariant rather than on that word
    // — what the harness must never see from this path is a transfer reported as
    // finished.
    assert.notEqual(file.state, "complete");
    assert.match(report.note ?? "", /unfinished/);
    // The listener is removed on the way out: a listener left behind would police
    // the NEXT call's downloads against a ceiling this one never enforced.
    assert.equal(fake.listeners.has("changed"), false);
  } finally {
    await module.close();
  }
});

test("several downloads in one gesture are all reported, at their own paths", async () => {
  // A multi-file export creates its downloads in sequence. Returning on the first
  // `complete` would report one file and silently drop the rest — and the ones it
  // dropped would stay in the user's folder, which is the outcome the operator's
  // decision rules out.
  installChrome({
    11: [
      () => item(11, { state: "complete", filename: "/Users/someone/Downloads/report.pdf" }),
    ],
    12: [
      () => item(12),
      () => item(12, { state: "complete", filename: "/Users/someone/Downloads/report (1).pdf" }),
    ],
  });
  const module = await load("src/commands/download.ts");
  try {
    const report = await module.loaded.download({ tab: SURFACE, timeout_s: 20 }, "r-4");
    const names = report.files.map((file) => file.name).sort();
    // The COLLIDING name is Chrome's own uniquifying, reported as it is: the
    // handler must never "fix" a name, because the path it reports is what the
    // harness moves and the file on disk is the only authority.
    assert.deepEqual(names, ["report (1).pdf", "report.pdf"]);
    assert.deepEqual(
      report.files.map((file) => file.state),
      ["complete", "complete"],
    );
  } finally {
    await module.close();
  }
});
