#!/usr/bin/env node
/**
 * Fail the build when the emitted bundle cannot style the app.
 *
 * A SILENT success is the expensive failure here. With no candidate sources,
 * Tailwind v4 emits a stylesheet carrying only its base layer, vite exits 0,
 * and the phone renders unstyled with nothing in any log to explain it. That
 * happened where the sources were named least obviously: an INSTALLED tree
 * (site-packages, or a `lop-update` generation under /var/folders) has no
 * `.git`/`.gitignore` above it for automatic detection to resolve a project
 * root from, which is also why `src/styles/index.css` names its sources
 * explicitly.
 *
 * HOW BADLY IT DEGRADES IS LAYOUT-DEPENDENT, so no size and no class count is
 * pinned here as "the" broken figure: in the uv-tool layout this was found in,
 * the stylesheet lost most of its selectors, while a layout whose scan wandered
 * into node_modules measured MORE than the reference and still carried every
 * reference selector. Both are wrong, in opposite directions, which is why the
 * checks below are BOTH per token and bounded. The layout-independent property
 * — the one this change is verified against — is that the bundle built from
 * unchanged sources is byte-identical to the published one.
 *
 * Two directions, because both were measured on this defect:
 *
 *   * UNDER-INCLUSION — a stylesheet that is missing the classes the app
 *     renders with. Checked per token: every class token a `className`
 *     expression actually names must have a selector in the emitted CSS
 *     (escaped or not, variant prefixes included). This is what a dropped
 *     utility, a partial scan, or a `source(none)` with a broken glob looks
 *     like — and it caught a 62-class stylesheet that the flat "≥100 classes"
 *     floor this check used to apply had passed.
 *   * OVER-INCLUSION — a scan that found the WRONG tree. A polluted build
 *     (measured: 695 and 852 class selectors, where these sources produce
 *     287) has every real class present, so a flat floor and even the
 *     per-token check pass it. Worse, extra selectors mean the bundle is
 *     styling things the app never asked for. So the selector count is also
 *     bounded, relative to the token count rather than to an absolute number
 *     that would rot.
 *
 * Runs as npm's `postbuild`, so it covers every `pnpm build`: this checkout,
 * CI (mobile-web.yml), the release (publish.yml) and an installed tree's own
 * self-heal (`lop mobile install`) — the path that served the unstyled phone.
 * `local_operator/mobile/install.py::_verify_bundle` runs this same file again
 * after its own build, for a snapshot whose package.json predates this hook.
 */
import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import { dirname, join, relative } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");

/**
 * Class tokens a source names that legitimately have NO rule of their own.
 * Every entry is an argued exception, not a convenience: the check exists to
 * notice a class the app renders with that the stylesheet lost, so adding a
 * real utility here disarms it. `lo-loadbar` is a hook — `src/styles/index.css`
 * defines `@keyframes lo-loadbar` and styles the child `.lo-loadbar-fill`
 * (measured: the published 0.59.11 bundle carries `.lo-loadbar-fill` and no
 * `.lo-loadbar` rule, with the class still on the element in transcript.tsx).
 */
const HOOK_CLASSES = new Set(["lo-loadbar"]);

/**
 * Upper bound on emitted class selectors, as `RATIO * tokens + SLACK`. Wide on
 * both sides of the measured reference (287 selectors from 225 tokens, ratio
 * 1.3) so ordinary style churn never approaches it, and far below a scan that
 * wandered into node_modules (695-852 selectors for the same sources).
 */
const SELECTOR_RATIO = 2;
const SELECTOR_SLACK = 100;

const SOURCE_EXTENSIONS = [".ts", ".tsx"];

function fail(message) {
	console.error(`error: ${message}`);
	process.exit(1);
}

const args = process.argv.slice(2);
const option = (name, fallback) => {
	const at = args.indexOf(name);
	return at === -1 ? fallback : args[at + 1];
};
const dist = option("--dist", join(ROOT, "dist"));
const web = option("--web", ROOT);
if (!dist || !web) fail("--dist and --web each need a directory");

// ---------------------------------------------------------------------------
// The emitted bundle
// ---------------------------------------------------------------------------
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
const onDisk = (assetPath) => join(dist, assetPath.replace(/^\.?\//, ""));
for (const assetPath of [...scripts, ...styles]) {
	const file = onDisk(assetPath);
	if (!existsSync(file) || statSync(file).size === 0) fail(`${file} is missing or empty`);
}

/**
 * Every class selector the stylesheet defines, in UNESCAPED form — Tailwind
 * escapes the characters a class name cannot carry, so a fraction-bearing
 * width and a variant-prefixed utility are written with backslashes in the CSS
 * and without them in the source. Matching on the unescaped form is what lets
 * the comparison below use each token as the source wrote it.
 *
 * NOTE FOR WHOEVER EDITS THIS FILE: keep its prose free of anything shaped like
 * a utility class. `src/styles/index.css` scans `scripts/*.mjs` (for the theme
 * generator's comment that contributes a background utility the markup never
 * names), so a class spelled out in this file lands in the SHIPPED stylesheet —
 * an earlier revision of this very comment added two selectors to the bundle,
 * and the only reason that was caught is that this change tracks the emitted
 * bytes against the published ones. A sentence is not a build input; do not let
 * it become one.
 */
const css = styles.map((assetPath) => readFileSync(onDisk(assetPath), "utf8")).join("\n");
const SELECTOR = /\.((?:\\.|[^\s,{}:;>~+()"'])+)/g;
const selectors = new Set(
	[...css.matchAll(SELECTOR)].map((match) => match[1].replace(/\\(.)/g, "$1")),
);

// ---------------------------------------------------------------------------
// The classes the sources name
// ---------------------------------------------------------------------------
/** The expression a `className=` introduces: a literal, or one balanced `{...}`. */
function classNameExpression(text, at) {
	let index = at;
	while (index < text.length && " \n\t".includes(text[index])) index += 1;
	const quote = text[index];
	if (quote === '"' || quote === "'" || quote === "`") {
		let end = index + 1;
		while (end < text.length && !(text[end] === quote && text[end - 1] !== "\\")) end += 1;
		return text.slice(index, end + 1);
	}
	if (quote === "{") {
		let depth = 0;
		for (let end = index; end < text.length; end += 1) {
			if (text[end] === "{") depth += 1;
			else if (text[end] === "}") {
				depth -= 1;
				if (depth === 0) return text.slice(index, end + 1);
			}
		}
	}
	return "";
}

const LITERAL = /"((?:[^"\\]|\\.)*)"|'((?:[^'\\]|\\.)*)'|`((?:[^`\\$]|\\.)*)`/g;
/** A literal compared against something is a VALUE (`state === "error"`), not a class. */
const COMPARISON = /(?:===|!==|==|!=|case)\s*$/;
/** `source()` kept in comments is documentation; and a token must look like a class. */
const COMMENT = /\/\*[\s\S]*?\*\/|\/\/[^\n]*/g;
const TOKEN = /^[a-z][A-Za-z0-9:/_.[\]()%#-]*$/;

const tokens = new Map();
const addFrom = (text, file) => {
	const expressions = file.endsWith(".html")
		? [...text.matchAll(/class="([^"]*)"/g)].map((match) => `"${match[1]}"`)
		: [...text.matchAll(/className=/g)].map((match) =>
				classNameExpression(text, match.index + "className=".length),
			);
	for (const expression of expressions) {
		const code = expression.replace(COMMENT, " ");
		for (const literal of code.matchAll(LITERAL)) {
			if (COMPARISON.test(code.slice(0, literal.index))) continue;
			const content = literal[1] ?? literal[2] ?? literal[3] ?? "";
			for (const token of content.split(/\s+/)) {
				if (TOKEN.test(token) && !token.endsWith(":")) {
					if (!tokens.has(token)) tokens.set(token, file);
				}
			}
		}
	}
};

const collect = (directory) => {
	for (const entry of readdirSync(directory, { withFileTypes: true })) {
		const full = join(directory, entry.name);
		if (entry.isDirectory()) collect(full);
		else if (SOURCE_EXTENSIONS.includes(entry.name.slice(entry.name.lastIndexOf(".")))) {
			addFrom(readFileSync(full, "utf8"), relative(web, full));
		}
	}
};
const sourceHtml = join(web, "index.html");
if (existsSync(sourceHtml)) addFrom(readFileSync(sourceHtml, "utf8"), "index.html");
const sourceDir = join(web, "src");
if (existsSync(sourceDir)) collect(sourceDir);

if (tokens.size === 0) {
	fail(
		`no class tokens found under ${join(web, "src")}; the sources this check derives ` +
			"from are not where it looked, so it would pass anything",
	);
}

// ---------------------------------------------------------------------------
// Both directions
// ---------------------------------------------------------------------------
const missing = [...tokens.keys()].filter(
	(token) => !selectors.has(token) && !HOOK_CLASSES.has(token),
);
if (missing.length) {
	const [first] = missing;
	fail(
		`the stylesheet has no rule for ${missing.length} class(es) the app renders with: ` +
			`${missing.slice(0, 12).join(", ")}${missing.length > 12 ? ", …" : ""} ` +
			`(first named in ${tokens.get(first)}). Tailwind did not see them, so they would ` +
			"render unstyled — check the @source globs in src/styles/index.css and that the " +
			"web tree's .gitignore lets the scanner walk src/.",
	);
}

const ceiling = SELECTOR_RATIO * tokens.size + SELECTOR_SLACK;
if (selectors.size > ceiling) {
	fail(
		`the stylesheet defines ${selectors.size} classes for ${tokens.size} tokens (ceiling ` +
			`${ceiling}); the scan found a tree the build does not own, most likely ` +
			"node_modules. Check the @source globs in src/styles/index.css.",
	);
}

console.log(
	`bundle ok: ${selectors.size} classes for ${tokens.size} tokens, ` +
		`${scripts.length} script(s), ${styles.length} stylesheet(s)`,
);
