#!/usr/bin/env node
/**
 * Fail the build when the emitted bundle is degenerate.
 *
 * A SILENT success is the expensive failure here. With no candidate sources,
 * Tailwind v4 emits a stylesheet carrying only its base layer — measured 31 kB
 * against 50 kB for these sources — vite exits 0, and the phone renders
 * unstyled with nothing in any log to explain it. That happened where the
 * sources were named least obviously: an INSTALLED tree (site-packages, or a
 * `lop-update` generation under /var/folders) has no `.git`/`.gitignore` above
 * it for automatic detection to resolve a project root from, which is also why
 * `src/styles/index.css` now names its sources explicitly.
 *
 * Runs as npm's `postbuild`, so it covers every `pnpm build`: this checkout,
 * CI (mobile-web.yml), the release (publish.yml) and an installed tree's own
 * self-heal (`lop mobile install`) — the path that served the unstyled phone.
 * `local_operator/mobile/install.py::_verify_bundle` runs this same file again
 * after its own build, for a snapshot whose package.json predates this hook.
 */
import { existsSync, readFileSync, statSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");

/**
 * Class selectors the published 0.59.11 bundle carries, and the floor below
 * which a stylesheet cannot be describing this app. The two numbers are far
 * apart on purpose — 235 for these sources, 19 for a stylesheet whose
 * candidate sources were never found — so the floor fails the degenerate
 * build and cannot fire on a stylesheet that merely lost a utility or two.
 */
const REFERENCE_CLASS_SELECTORS = 235;
const MIN_CLASS_SELECTORS = 100;

function fail(message) {
	console.error(`error: ${message}`);
	process.exit(1);
}

const args = process.argv.slice(2);
const distFlag = args.indexOf("--dist");
const dist = distFlag === -1 ? join(ROOT, "dist") : args[distFlag + 1];
if (!dist) fail("--dist needs the directory to check");

const indexPath = join(dist, "index.html");
if (!existsSync(indexPath)) fail(`no ${indexPath}; the build produced nothing to serve`);

const html = readFileSync(indexPath, "utf8");
const referenced = [...html.matchAll(/["'](\.?\/?assets\/[^"']+)["']/g)].map((match) => match[1]);
const scripts = referenced.filter((path) => path.endsWith(".js"));
const styles = referenced.filter((path) => path.endsWith(".css"));
if (!scripts.length || !styles.length) {
	fail(
		`${indexPath} links ${styles.length} stylesheet(s) and ${scripts.length} script(s); ` +
			"the app cannot load without one of each",
	);
}
const onDisk = (relative) => join(dist, relative.replace(/^\.?\//, ""));
for (const relative of [...scripts, ...styles]) {
	const file = onDisk(relative);
	if (!existsSync(file) || statSync(file).size === 0) fail(`${file} is missing or empty`);
}

const css = styles.map((relative) => readFileSync(onDisk(relative), "utf8")).join("\n");
const classes = new Set(
	[...css.matchAll(/\.([A-Za-z0-9_\\:-]+)(?=[\s,{:>~+])/g)].map((match) => match[1]),
);
if (classes.size < MIN_CLASS_SELECTORS) {
	fail(
		`the stylesheet names ${classes.size} classes; this bundle has ${REFERENCE_CLASS_SELECTORS}. ` +
			"Tailwind found no candidate sources, so every utility is missing and the app would " +
			"render unstyled. Check the @source globs in src/styles/index.css.",
	);
}

console.log(
	`bundle ok: ${classes.size} classes, ${scripts.length} script(s), ${styles.length} stylesheet(s)`,
);
