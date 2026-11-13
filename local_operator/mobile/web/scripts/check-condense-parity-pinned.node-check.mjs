import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { createServer } from "node:http";
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync, rmSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";
import test from "node:test";
import { checkPinned } from "./check-condense-parity-pinned.mjs";

const hash = (s) => createHash("sha256").update(s).digest("hex");
const HERE = dirname(fileURLToPath(import.meta.url));
const revision = "a".repeat(40);
const sourcePath = "src/model.ts";
const source = "export const model = 1;\n";
const sourceHash = hash(`${sourcePath}\0${source}\0`);
const fixture = JSON.stringify({ desktop: { sourceHash, modelSources: [sourcePath] }, cases: [] });
const manifest = JSON.stringify({ file: "fixtures/condense.parity.json", sha256: hash(fixture), sourceHash });

// Real HTTP, a moving main ref, and the production legacy comparison. No mocks
// of fetch or hash: the negative controls must traverse the same reader as CI.
test("moving main is reproduced; immutable provenance is stable and fails closed", async (t) => {
	assert.ok(process.env.LOCAL_OPERATOR_SCRATCHPAD, "scratchpad root required for isolated fixtures");
	const root = mkdtempSync(join(process.env.LOCAL_OPERATOR_SCRATCHPAD, "condense-pin-"));
	t.after(() => rmSync(root, { recursive: true, force: true }));
	const lib = join(root, "lib");
	mkdirSync(lib);
	const pin = { repo: "damianvtran/local-operator-ui", revision, path: "scripts/fixtures/condense.parity.json", manifest: "scripts/fixtures/condense.parity.manifest.json", sha256: hash(fixture), sourceHash };
	const save = (entry = pin) => writeFileSync(join(lib, "parity-sources.json"), JSON.stringify({ "condense.parity.json": entry }));
	writeFileSync(join(lib, "condense.parity.json"), fixture);
	save();
	const original = new Map([[pin.path, fixture], [pin.manifest, manifest], [sourcePath, source]]);
	let pinned = new Map(original);
	let main = new Map(original);
	const requests = [];
	const server = createServer((request, response) => {
		requests.push(request.url);
		const [, , , ref, ...parts] = request.url.split("/");
		const body = (ref === revision ? pinned : ref === "main" ? main : new Map()).get(parts.join("/"));
		response.writeHead(body === undefined ? 404 : 200);
		response.end(body);
	});
	await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
	t.after(() => new Promise((resolve) => server.close(resolve)));
	const baseUrl = `http://127.0.0.1:${server.address().port}`;
	const options = { lib, baseUrl };
	await t.test("declared revision succeeds before and after main moves", async () => {
		assert.match(await checkPinned(options), /immutable/);
		main.set(pin.path, "new UI main fixture");
		assert.match(await checkPinned(options), /immutable/);
		assert.ok(requests.every((url) => url.includes(`/${revision}/`)));
	});
	await t.test("legacy moving-main gate changes pass to fail with unchanged relay bytes", async () => {
		// Legacy checker resolves its LIB beside itself: stage only the script and
		// synthetic pins, never copy a tree or import live configuration.
		const staged = join(root, "legacy");
		mkdirSync(join(staged, "scripts"), { recursive: true });
		mkdirSync(join(staged, "src/lib"), { recursive: true });
		writeFileSync(join(staged, "scripts/check.mjs"), readFileSync(join(HERE, "check-condense-parity-latest.mjs")));
		writeFileSync(join(staged, "src/lib/parity-sources.json"), JSON.stringify({ "condense.parity.json": pin }));
		const sibling = join(root, "desktop");
		mkdirSync(join(sibling, "scripts/fixtures"), { recursive: true });
		const run = async () => {
			for (const path of [pin.path, pin.manifest]) {
				const bytes = await (await fetch(`${baseUrl}/${pin.repo}/main/${path}`)).text();
				writeFileSync(join(sibling, path), bytes);
			}
			return spawnSync(process.execPath, [join(staged, "scripts/check.mjs"), "--latest", "--require-sibling", "--sibling", sibling], { encoding: "utf8" });
		};
		main = new Map(original);
		assert.equal((await run()).status, 0);
		main.set(pin.path, JSON.stringify({ desktop: { sourceHash: "b".repeat(64) } }));
		const result = await run();
		assert.equal(result.status, 1);
		assert.match(result.stderr, /manifest does not match its own fixture/);
		console.log("REPRO legacy main: PASS before move -> FAIL after move; immutable revision: PASS unchanged");
	});
	await t.test("missing, unordered and unsafe source inventories fail closed", async () => {
		for (const modelSources of [undefined, [], ["../outside"], ["z.ts", "a.ts"], [sourcePath, sourcePath]]) {
			const invalidFixture = JSON.stringify({ desktop: { sourceHash, modelSources } });
			const alteredPin = { ...pin, sha256: hash(invalidFixture) };
			pinned = new Map(original);
			pinned.set(pin.path, invalidFixture);
			pinned.set(pin.manifest, JSON.stringify({ ...JSON.parse(manifest), sha256: alteredPin.sha256 }));
			writeFileSync(join(lib, "condense.parity.json"), invalidFixture);
			save(alteredPin);
			await assert.rejects(checkPinned(options), /source inventory/);
		}
	});
	for (const [label, mutate, error] of [
		["wrong fixture SHA", () => pinned.set(pin.path, "wrong"), /fixture SHA mismatch/],
		["wrong manifest", () => pinned.set(pin.manifest, JSON.stringify({ ...JSON.parse(manifest), sha256: "b".repeat(64) })), /identity mismatch/],
		["wrong manifest source", () => pinned.set(pin.manifest, JSON.stringify({ ...JSON.parse(manifest), sourceHash: "b".repeat(64) })), /identity mismatch/],
		["wrong source bytes", () => pinned.set(sourcePath, "changed model"), /model source hash mismatch/],
		["missing fixture", () => pinned.delete(pin.path), /HTTP 404/],
		["missing manifest", () => pinned.delete(pin.manifest), /HTTP 404/],
		["missing source", () => pinned.delete(sourcePath), /HTTP 404/],
		["missing revision", () => save({ ...pin, revision: undefined }), /immutable revision/],
		["mutable revision", () => save({ ...pin, revision: "main" }), /immutable revision/],
		["unavailable revision", () => save({ ...pin, revision: "b".repeat(40) }), /HTTP 404/],
		["wrong local fixture", () => writeFileSync(join(lib, "condense.parity.json"), "wrong"), /local fixture/],
	]) {
		await t.test(`${label} fails closed`, async () => {
			pinned = new Map(original); save(); writeFileSync(join(lib, "condense.parity.json"), fixture);
			mutate();
			await assert.rejects(checkPinned(options), error);
		});
	}
});
