/* The policy port's conformance with Python, and the hostile names it exists for.
 *
 * WHY THIS TEST IS THE POINT. `driver/file-transfer-policy.ts` is a SECOND
 * implementation of rules that live in `local_operator/browser_files.py`, in a
 * second language, vendored into a second repository. The thing that keeps the
 * two honest is the shared fixture: Python's generator refuses to emit the tables
 * unless its own classifier reproduces every hand-written expectation, and this
 * file replays the same expectations against the TypeScript sanitiser. A
 * divergence in either language fails one of the two gates.
 *
 * What it CANNOT catch is a logic difference on a case the fixture does not
 * cover — which is stated in the design (§10.4) and is why adding a case is how a
 * bug becomes a gate.
 */

import assert from "node:assert/strict";
import test from "node:test";
import { build } from "esbuild";
import { pathToFileURL } from "node:url";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

async function load(entry) {
  const dir = await mkdtemp(join(tmpdir(), "lop-file-transfer-test-"));
  const outfile = join(dir, "module.mjs");
  await build({ entryPoints: [entry], bundle: true, platform: "node", format: "esm", outfile });
  const loaded = await import(pathToFileURL(outfile));
  return { loaded, close: () => rm(dir, { recursive: true, force: true }) };
}

test("the sanitiser and the name check reproduce every shared conformance case", async () => {
  const module = await load("src/driver/file-transfer-policy.ts");
  const tables = await load("src/driver/file-transfer.tables.gen.ts");
  try {
    const { safeName, executableName } = module.loaded;
    const cases = tables.loaded.CONFORMANCE_CASES;
    assert.ok(cases.length >= 20, `expected a real fixture, got ${cases.length} cases`);
    for (const item of cases) {
      assert.equal(
        safeName(item.name, item.safeNameSniffedExt),
        item.safeName,
        `safeName(${JSON.stringify(item.name)}, ${JSON.stringify(item.safeNameSniffedExt)})`,
      );
      assert.equal(
        executableName(item.name),
        item.nameIsDenyListed,
        `executableName(${JSON.stringify(item.name)})`,
      );
    }
  } finally {
    await module.close();
    await tables.close();
  }
});

test("the sanitiser removes every hostile-name class the design enumerates", async () => {
  const module = await load("src/driver/file-transfer-policy.ts");
  try {
    const { safeName, fallbackName } = module.loaded;
    // A separator in either flavour truncates rather than round-trips.
    assert.equal(safeName("../../etc/passwd"), "passwd");
    assert.equal(safeName("..\\..\\Windows\\evil.txt"), "evil.txt");
    assert.equal(safeName("/absolute/path/report.pdf"), "report.pdf");
    // NUL and the C0/C1 range: a terminal escape in a filename is an injection
    // into the operator's next approval card.
    assert.equal(safeName("re\u0000port\u001b[31m.pdf"), "report[31m.pdf");
    // RTL and zero-width overrides, which exist to make `evil.exe` DISPLAY as a
    // harmless name.
    assert.equal(safeName("photo\u202ejpg.exe"), "photojpg.exe");
    assert.equal(safeName("pay\u2066load.exe.pdf"), "payload.exe.pdf");
    // Trailing dots and spaces, which Windows silently drops.
    assert.equal(safeName("report.pdf "), "report.pdf");
    assert.equal(safeName("report.pdf..."), "report.pdf");
    // `.` and `..` and the nothing-left case fall back to a generated name.
    assert.equal(safeName("."), fallbackName("."));
    assert.equal(safeName(".."), fallbackName(".."));
    assert.equal(safeName("   "), fallbackName("   "));
    // Windows reserved stems, tested on the STEM.
    assert.equal(safeName("CON.txt"), fallbackName("CON.txt"));
    assert.equal(safeName("lpt1"), fallbackName("lpt1"));
    // A 300-character name is cut to the 200-byte cap with its extension kept.
    const long = `${"a".repeat(300)}.pdf`;
    const cut = safeName(long);
    assert.ok(new TextEncoder().encode(cut).length <= 200, cut.length);
    assert.ok(cut.endsWith(".pdf"), cut);
    // A sniffed extension corrects the name, which is the one case where the
    // name is CHANGED rather than cleaned.
    assert.equal(safeName("invoice.zip", "pdf"), "invoice.pdf");
    assert.equal(safeName("handout", "zip"), "handout.zip");
  } finally {
    await module.close();
  }
});

test("credential paths are refused by basename and by component", async () => {
  const module = await load("src/driver/file-transfer-policy.ts");
  try {
    const { credentialRefusal } = module.loaded;
    for (const path of [
      "/Users/x/.ssh/id_rsa",
      "/Users/x/.ssh/id_rsa.pub",
      "/Users/x/keys/server.pem",
      "/Users/x/keys/server.key",
      "/Users/x/.env",
      "/Users/x/.env.production",
      "/Users/x/credentials.json",
      "/Users/x/service-account-prod.json",
      "/Users/x/.aws/config",
      "/Users/x/.kube/config",
      "/Users/x/Library/Keychains/login.keychain-db",
      "/Users/x/work/secrets/notes.txt",
      "/Users/x/.git-credentials",
      "/Users/x/project/.npmrc",
    ]) {
      assert.notEqual(credentialRefusal(path), "", path);
    }
    // The long tail the operator actually asked for must stay attachable: a deny
    // list, never an allow list.
    for (const path of [
      "/Users/x/Documents/quarterly deck.pptx",
      "/Users/x/Downloads/receipt-2026-09.pdf",
      "/Users/x/work/diagram.drawio",
      "/Users/x/work/model.stl",
      "/Users/x/work/mail.msg",
      "/Users/x/work/notes.txt",
      "/Users/x/work/archive.zip",
    ]) {
      assert.equal(credentialRefusal(path), "", path);
    }
  } finally {
    await module.close();
  }
});
