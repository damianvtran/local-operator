#!/usr/bin/env node
/**
 * Generates `src/styles/themes.generated.css` from the palette
 * files.
 *
 *     node scripts/generate-theme-css.mjs           # write
 *     node scripts/generate-theme-css.mjs --check   # verify up to date, CI
 *
 * ## Why generate rather than hand-write
 *
 * Two systems need the same colours during this migration: MUI (still styling
 * most of the app) and Tailwind (styling everything already ported). If those
 * two read from different places they drift, and the drift shows up as a card
 * that is one shade off from the card beside it — the exact defect that makes
 * a half-migrated app look broken rather than in progress.
 *
 * So the palette TypeScript is the single source. MUI consumes it directly as
 * hex values, which matters because ~299 `alpha()` call sites need a real
 * colour and cannot take a `var()`. Tailwind consumes the CSS variables this
 * script emits. Same numbers, two consumers, no hand-copying.
 *
 * ## Why static CSS rather than setting variables at runtime
 *
 * The theme provider could set these on `document.documentElement` at boot.
 * Emitting them as static CSS instead means the correct palette is present in
 * the stylesheet before first paint, so switching themes is a class change
 * rather than a scripted restyle, and a cold start has no flash of the wrong
 * ramp.
 */

import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { loadPalettes } from "./palette-source.mjs";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const PALETTE_DIR = join(ROOT, "src/themes");
const OUT = join(ROOT, "src/styles/themes.generated.css");
const THEME_CSS = join(ROOT, "src/styles/index.css");

/** camelCase role -> kebab-case custom property suffix. */
const kebab = (s) => s.replace(/[A-Z]/g, (c) => `-${c.toLowerCase()}`);

const palettes = loadPalettes();
if (palettes.length === 0) {
	console.error(`No palettes found in ${PALETTE_DIR}`);
	process.exit(1);
}

const block = ({ id, palette }) => {
	const lines = Object.entries(palette)
		.filter(([role]) => role !== "mode")
		.map(([role, value]) => `\t--lo-${kebab(role)}: ${value};`)
		.join("\n");
	/* `color-scheme` makes the platform paint native scrollbars, form controls
	   and the caret to match the palette. Without it a dark theme keeps a white
	   scrollbar track, which is the single most visible "unthemed" artefact in
	   an Electron app. */
	return `[data-theme="${id}"] {\n\tcolor-scheme: ${palette.mode};\n${lines}\n}`;
};

/**
 * Re-declare Tailwind's role variables inside any scoped theme subtree.
 *
 * THE DEFECT THIS FIXES. `@theme` emits `--color-canvas: var(--lo-canvas)` on
 * `:root`. A `var()` inside a custom property is substituted where the property
 * is DECLARED, not where it is used, so `--color-canvas` is frozen to the root
 * palette's value and every descendant inherits that already-resolved colour.
 * Putting `data-theme="dracula"` on a wrapper therefore rebinds `--lo-*` for
 * the subtree while every Tailwind role utility inside it keeps painting the
 * ACTIVE theme.
 *
 * Measured before this existed, with root dark and a light-scoped subtree:
 * `--lo-canvas` was #f7f4ee (the light canvas AT THE TIME - that value has
 * since moved, and the number is kept as the historical reading rather than
 * refreshed, because a measurement is of a moment) while `--color-canvas` was
 * #16130e and `bg-canvas` painted rgb(22,19,14). The twelve previews in the
 * theme picker were all showing the active theme.
 *
 * The pairs are parsed out of index.css rather than restated here, so adding a
 * role to `@theme` cannot silently leave scoped previews stale.
 */
/**
 * Strip CSS comments, preserving string literals.
 *
 * Order matters and cost a reviewer a reproduction to find: this MUST run
 * before the brace matcher, not after. Matching braces on un-stripped source
 * means a single unbalanced `}` inside a comment ends the block early, the
 * harvest returns the roles above it and silently drops the rest, and the gate
 * stays green because a short harvest is indistinguishable from a short block.
 * Reproduced at 29 -> 9, 29 -> 11 and 29 -> 15 roles.
 */
const stripCssComments = (src) => {
	let out = "";
	let i = 0;
	while (i < src.length) {
		if (src[i] === '"' || src[i] === "'") {
			const q = src[i];
			out += src[i++];
			while (i < src.length) {
				out += src[i];
				if (src[i] === "\\" && i + 1 < src.length) {
					out += src[++i];
					i++;
					continue;
				}
				if (src[i] === q) {
					i++;
					break;
				}
				i++;
			}
			continue;
		}
		if (src[i] === "/" && src[i + 1] === "*") {
			i += 2;
			while (i < src.length && !(src[i] === "*" && src[i + 1] === "/")) i++;
			i += 2;
			continue;
		}
		out += src[i++];
	}
	return out;
};

const scopedRoleBlock = () => {
	/* Comments first, then braces - see stripCssComments. */
	const css = stripCssComments(readFileSync(THEME_CSS, "utf8"));

	const open = css.indexOf("@theme {");
	if (open === -1) throw new Error("no @theme block in styles/index.css");
	if (css.indexOf("@theme", open + 6) !== -1) {
		/* Two @theme blocks would mean half the roles live outside the one this
		   function harvests, which is the same silent-truncation failure wearing
		   a different hat. */
		throw new Error("more than one @theme block in styles/index.css");
	}

	/* Skip braces inside string literals.
	 *
	 * `stripCssComments` already removes comment braces, but a `}` inside a
	 * quoted value - a `content:` string, a font name - is still counted as
	 * structure and ends the block early. And an UNTERMINATED string makes the
	 * comment stripper copy the remainder of the file verbatim, which brings
	 * the comment-brace truncation straight back. Both produce a short harvest
	 * with every gate green, which is the one failure shape this function
	 * exists to make impossible, so the matcher has to understand quotes for
	 * the same reason the stripper does. */
	let depth = 0;
	let close = -1;
	let quote = null;
	for (let i = css.indexOf("{", open); i < css.length; i++) {
		const c = css[i];
		if (quote) {
			if (c === "\\") i++;
			else if (c === quote) quote = null;
			continue;
		}
		if (c === '"' || c === "'") quote = c;
		else if (c === "{") depth++;
		else if (c === "}") {
			depth--;
			if (depth === 0) {
				close = i;
				break;
			}
		}
	}
	if (close === -1) throw new Error("unterminated @theme block");

	const block = css.slice(open, close);

	/* Any custom property whose value is a single --lo-* reference, not just
	   --color-*. `--shadow-overlay: var(--lo-overlay-shadow)` is exactly such a
	   pair and was being dropped, so scoped subtrees kept the root's shadow. */
	const pairs = [...block.matchAll(/(--[\w-]+):\s*var\((--lo-[\w-]+)\)\s*;/g)];

	/* Completeness against the PALETTES, not against index.css.
	   
	   Checking "every role index.css mentions was harvested" sounds right and
	   catches nothing that matters: the dangerous case is a role added to the
	   palettes with no `@theme` pair at all, and such a role is never mentioned
	   in index.css either, so it passes a self-referential check trivially.
	   Comparing against the palette role set is what makes it loud - that role
	   would otherwise reach `:root` and silently never reach a scoped subtree,
	   which is the round-1 `--shadow-overlay` defect exactly. A short harvest
	   from a truncated block fails here too, since the missing pairs are missing
	   from both sides. */
	const harvested = new Set(pairs.map(([, , src]) => src));
	const paletteRoles = new Set(
		loadPalettes().flatMap(({ palette }) =>
			Object.keys(palette)
				.filter((role) => role !== "mode")
				.map((role) => `--lo-${kebab(role)}`),
		),
	);
	const missing = [...paletteRoles].filter((r) => !harvested.has(r));
	if (missing.length > 0) {
		throw new Error(
			`@theme is missing ${missing.length} role pair(s): ${missing.join(", ")}. ` +
				`Every palette role needs a \`--x: var(--lo-x);\` line inside @theme, ` +
				`or it reaches :root and never reaches a [data-theme] subtree.`,
		);
	}

	/* And the converse, in two parts: every pair must name a real role, and no
	 * two pairs may name the SAME role.
	 *
	 * Without these the guard only asserts paletteRoles subset-of harvested,
	 * which is blind to a pair whose `--lo-*` source another pair already
	 * supplies. Today the two sets are a bijection - 29 roles, 29 pairs - and
	 * it is that coincidence, not the guard, that makes a short harvest loud:
	 * add one alias and a truncated block passes every gate with output
	 * byte-identical to the committed file, which is the round-1
	 * `--shadow-overlay` defect restored. The duplicate check is the half that
	 * closes it, because an alias is exactly a duplicate source. Together they
	 * make the bijection an invariant this script enforces rather than a
	 * property it happens to have. */
	const unknown = [...harvested].filter((r) => !paletteRoles.has(r));
	if (unknown.length > 0) {
		throw new Error(
			`@theme references ${unknown.length} role(s) no palette defines: ` +
				`${unknown.join(", ")}. Check the spelling against the palette keys.`,
		);
	}
	if (pairs.length !== harvested.size) {
		const seen = new Set();
		const dupes = [
			...new Set(
				pairs
					.map(([, , src]) => src)
					.filter((r) => (seen.has(r) ? true : (seen.add(r), false))),
			),
		];
		throw new Error(
			`@theme maps ${dupes.join(", ")} more than once. Aliases are not ` +
				`allowed here: two properties sharing one role break the one-to-one ` +
				`mapping that is what makes a truncated harvest detectable at all.`,
		);
	}

	const lines = pairs
		.map(([, name, src]) => `\t${name}: var(${src});`)
		.join("\n");
	return [
		"/* Scoped theme subtrees: see scripts/generate-theme-css.mjs for why this",
		"   re-declaration is required rather than redundant. */",
		`[data-theme]:not(:root) {\n${lines}\n}`,
	].join("\n");
};

const body = [
	"/*",
	" * GENERATED FILE — do not edit.",
	" *",
	" * Regenerate with `node scripts/generate-theme-css.mjs` after changing any",
	" * palette in src/themes/. `pnpm check-themes`",
	" * fails if this file is stale.",
	" *",
	" * Source of truth: the ThemePalette objects in that directory. The floors",
	" * these values must satisfy live in docs/branding.md § 3 and are enforced",
	" * by scripts/contrast-contract.mjs.",
	" */",
	"",
	...palettes.map(block),
	"",
	scopedRoleBlock(),
	"",
].join("\n");

if (process.argv.includes("--check")) {
	let current = "";
	try {
		current = readFileSync(OUT, "utf8");
	} catch {
		console.error(
			`${OUT} does not exist. Run: node scripts/generate-theme-css.mjs`,
		);
		process.exit(1);
	}
	if (current !== body) {
		console.error(
			"themes.generated.css is stale. Run: node scripts/generate-theme-css.mjs",
		);
		process.exit(1);
	}
	console.log(
		`themes.generated.css is up to date (${palettes.length} themes).`,
	);
} else {
	writeFileSync(OUT, body);
	console.log(
		`Wrote ${OUT} — ${palettes.length} themes, ${Object.keys(palettes[0].palette).length - 1} roles each.`,
	);
}
