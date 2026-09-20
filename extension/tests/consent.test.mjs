/* The operator's consent switches: what each capability is, and what a call sees.
 *
 * WHY THIS FILE EXISTS. Every other gate in this extension is enforced by
 * something the extension cannot argue with (a Chrome permission, an origin
 * grant, the daemon's advertisement). This one is a SETTING, which means the
 * failure modes are all about state rather than about access: a flag that
 * outlives the permission that made it meaningful, a capability reported as
 * available while its grant is gone, a refusal that names a switch the user
 * cannot find. Each of those has a test below, because each of them is silent —
 * a switch that reads ON while the API is unavailable is exactly the defect the
 * design names, and no other test in this suite would catch it.
 *
 * Loaded through esbuild (as the policy conformance test is) because the module
 * is TypeScript importing the generated protocol table, and bundled so the test
 * exercises the same module graph the worker does rather than a copy of it.
 */

import assert from "node:assert/strict";
import test from "node:test";
import { build } from "esbuild";
import { pathToFileURL } from "node:url";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

async function load(entry) {
  const dir = await mkdtemp(join(tmpdir(), "lop-consent-test-"));
  const outfile = join(dir, "module.mjs");
  await build({ entryPoints: [entry], bundle: true, platform: "node", format: "esm", outfile });
  const loaded = await import(pathToFileURL(outfile));
  return { loaded, close: () => rm(dir, { recursive: true, force: true }) };
}

/** The minimum `chrome` surface `consent.ts` touches, with the flags and grants
 *  a test sets. Nothing is resolved from the real browser: this suite runs under
 *  `node --test`, and a shared stub is what makes "the permission was revoked"
 *  expressible at all. */
function installChrome({ stored = {}, held = [], onRequest = null, onRemove = null } = {}) {
  const writes = [];
  const request = { permissions: [] };
  globalThis.chrome = {
    storage: {
      local: {
        get: async (keys) => {
          const out = {};
          for (const key of keys) if (key in stored) out[key] = stored[key];
          return out;
        },
        set: async (values) => {
          writes.push(values);
          Object.assign(stored, values);
        },
      },
    },
    permissions: {
      contains: async ({ permissions }) => permissions.every((name) => held.includes(name)),
      request: async ({ permissions }) => {
        request.permissions.push(...permissions);
        const granted = onRequest ? await onRequest(permissions) : true;
        if (granted) held.push(...permissions);
        return granted;
      },
      remove: async ({ permissions }) => {
        const removed = onRemove ? await onRemove(permissions) : true;
        if (removed) {
          for (const name of permissions) {
            const index = held.indexOf(name);
            if (index >= 0) held.splice(index, 1);
          }
        }
        return removed;
      },
    },
  };
  return { writes, request, held, stored };
}

test("both capabilities are off until a human turns them on", async () => {
  installChrome();
  const module = await load("src/consent.ts");
  try {
    const { capabilityEnabled, disabledCapabilities, requireConsent } = module.loaded;
    assert.equal(await capabilityEnabled("download"), false);
    assert.equal(await capabilityEnabled("upload"), false);
    // Reported as DISABLED rather than as absent: the methods are served by this
    // build and only their consent is missing, and that difference is what the
    // harness turns into "turn the switch on" instead of "update the extension".
    assert.deepEqual(await disabledCapabilities(), ["download", "upload"]);
    await assert.rejects(() => requireConsent("upload"), /switched off/);
    await assert.rejects(() => requireConsent("download"), /switched off/);
  } finally {
    await module.close();
  }
});

test("a granted download switch is enabled, and the record stops calling it disabled", async () => {
  installChrome({ stored: { allowDownloads: true }, held: ["downloads"] });
  const module = await load("src/consent.ts");
  try {
    const { capabilityEnabled, disabledCapabilities, requireConsent } = module.loaded;
    assert.equal(await capabilityEnabled("download"), true);
    await requireConsent("download"); // resolves: the whole gate opens
    // Upload is untouched by the download switch: they are independent, and a
    // shared flag would let one capability's consent speak for the other.
    assert.equal(await capabilityEnabled("upload"), false);
    assert.deepEqual(await disabledCapabilities(), ["upload"]);
  } finally {
    await module.close();
  }
});

test("the stored flag does NOT survive the permission being taken back", async () => {
  // The defect this module exists to design out, in its exact shape: the user
  // turned the switch on (so the flag is true), then revoked `downloads` in
  // chrome://extensions. A switch that read ON here would claim a capability the
  // very next call would fail.
  installChrome({ stored: { allowDownloads: true }, held: [] });
  const module = await load("src/consent.ts");
  try {
    const { capabilityEnabled, disabledCapabilities } = module.loaded;
    assert.equal(await capabilityEnabled("download"), false);
    assert.deepEqual(await disabledCapabilities(), ["download", "upload"]);
  } finally {
    await module.close();
  }
});

test("a missing permissions API is 'not held', never 'held'", async () => {
  // The conservative direction matters: the only claim this module may never make
  // is that an unverifiable capability is available. The cost of the wrong answer
  // here is a switch reading ON for a call that would fail with Chrome's error.
  installChrome();
  delete globalThis.chrome.permissions;
  const module = await load("src/consent.ts");
  try {
    const { capabilityEnabled, permissionHeld } = module.loaded;
    assert.equal(await permissionHeld("downloads"), false);
    assert.equal(await capabilityEnabled("download"), false);
    // …while a capability that needs NO permission is unaffected, which is why
    // the same helper is not used to gate uploads.
    assert.equal(await permissionHeld(""), true);
  } finally {
    await module.close();
  }
});

test("writeSwitch persists under the key the worker reads, and only that key", async () => {
  const chrome = installChrome();
  const module = await load("src/consent.ts");
  try {
    const { writeSwitch, storedSwitch } = module.loaded;
    await writeSwitch("upload", true);
    assert.deepEqual(chrome.writes, [{ allowUploads: true }]);
    assert.equal(await storedSwitch("upload"), true);
    assert.equal(await storedSwitch("download"), false);
    // An unknown method is a programming error, not a silent no-op: a typo that
    // wrote nothing would leave a user believing they had consented.
    await assert.rejects(() => writeSwitch("delete", true), /no consent switch/);
  } finally {
    await module.close();
  }
});

test("the labels and permissions come from the generated table, not from prose", async () => {
  installChrome();
  const module = await load("src/consent.ts");
  try {
    const { switchLabel, switchPermission, consentOffMessage, hasConsentSwitch } = module.loaded;
    // Pinned to Python's own constants through the generator: these words are
    // what the refusal and the options page both show, and a rename in one
    // language must not be able to leave the other stale.
    assert.equal(switchLabel("download"), "Allow downloads");
    assert.equal(switchLabel("upload"), "Allow uploads");
    assert.equal(switchPermission("download"), "downloads");
    assert.equal(switchPermission("upload"), "", "upload rides the debugger grant");
    assert.equal(hasConsentSwitch("screenshot"), false);
    const copy = consentOffMessage("download");
    assert.match(copy, /"Allow downloads"/);
    assert.match(copy, /options page/);
    assert.match(copy, /'downloads' permission/, "the grant is named, because the switch asks for it");
  } finally {
    await module.close();
  }
});
