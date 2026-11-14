#!/usr/bin/env node
/**
 * Required parity is a statement about this branch's declared desktop revision,
 * not whatever UI main happens to contain when a runner starts. The external
 * commit is a retrieval address, NOT a generation-time self-SHA: authenticate
 * fixture, manifest and the generator's source inventory against our content pins.
 * Keep latest-main freshness in its own workflow so drift stays visible without
 * changing the outcome for unchanged branch bytes.
 */
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const LIB = join(dirname(fileURLToPath(import.meta.url)), "../src/lib");
const hash = (bytes) => createHash("sha256").update(bytes).digest("hex");
const sha = /^[a-f0-9]{40}$/;
const digest = /^[a-f0-9]{64}$/;
const safePath = (path) => typeof path === "string" && /^[a-zA-Z0-9_./-]+$/.test(path)
	&& !path.split("/").some((part) => part === ".." || part === "." || part === "");

/** baseUrl is injectable for controlled HTTP tests; production always uses GitHub. */
export async function checkPinned({ lib = LIB, baseUrl = "https://raw.githubusercontent.com" } = {}) {
	const entries = JSON.parse(readFileSync(join(lib, "parity-sources.json"), "utf8"));
	for (const [file, pin] of Object.entries(entries).filter(([key]) => !key.startsWith("_"))) {
		if (!digest.test(pin.sha256) || hash(readFileSync(join(lib, file))) !== pin.sha256)
			throw new Error(`${file}: local fixture does not match its content pin`);
	}
	const pin = entries["condense.parity.json"];
	if (pin.repo !== "damianvtran/local-operator-ui" || !sha.test(pin.revision)
		|| !digest.test(pin.sourceHash) || pin.path !== "scripts/fixtures/condense.parity.json"
		|| pin.manifest !== "scripts/fixtures/condense.parity.manifest.json")
		throw new Error("condense: missing or invalid immutable revision/provenance");
	const get = async (path) => {
		const response = await fetch(`${baseUrl}/${pin.repo}/${pin.revision}/${path}`);
		if (!response.ok) throw new Error(`${path}: HTTP ${response.status} at ${pin.revision}`);
		return Buffer.from(await response.arrayBuffer());
	};
	const bytes = await get(pin.path);
	if (hash(bytes) !== pin.sha256) throw new Error("condense: revision fixture SHA mismatch");
	const fixture = JSON.parse(bytes);
	const manifest = JSON.parse(await get(pin.manifest));
	if (manifest.file !== "fixtures/condense.parity.json" || manifest.sha256 !== pin.sha256
		|| manifest.sourceHash !== pin.sourceHash || fixture.desktop?.sourceHash !== pin.sourceHash)
		throw new Error("condense: manifest/fixture source identity mismatch");
	const sources = fixture.desktop.modelSources;
	if (!Array.isArray(sources) || sources.length === 0 || sources.some((file) => !safePath(file))
		|| JSON.stringify(sources) !== JSON.stringify([...new Set(sources)].sort()))
		throw new Error("condense: missing or invalid generated model source inventory");
	// Match UI modelSourceHash exactly: sorted repository path, NUL, bytes, NUL.
	// The inventory itself is authenticated by the fixture's committed SHA above;
	// do not accept a server-provided substitute inventory or trust manifest alone.
	const sourceHash = createHash("sha256");
	for (const file of sources) sourceHash.update(file).update("\0").update(await get(file)).update("\0");
	if (sourceHash.digest("hex") !== pin.sourceHash)
		throw new Error("condense: revision model source hash mismatch");
	return `check-condense-parity: immutable ${pin.repo}@${pin.revision}; fixture, manifest and ${sources.length} model sources match`;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
	try { console.log(await checkPinned()); }
	catch (error) { console.error(`check-condense-parity: FAILED - ${error.message}`); process.exitCode = 1; }
}
