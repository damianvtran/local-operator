/**
 * The one place that reads palette values out of the palette TypeScript.
 *
 * Two scripts need these values - `contrast-contract.mjs` asserts over them and
 * `generate-theme-css.mjs` emits CSS from them - and until this file existed
 * they each carried their own copy of the parser. That is the shape of bug this
 * module exists to prevent: two parsers that agree with each other are
 * indistinguishable from two parsers that are both correct, right up until one
 * is fixed and the other is not. A gate that reads the source differently from
 * the generator it is gating is not a gate.
 *
 * Why parse text at all rather than importing the modules: the palettes are
 * TypeScript and these scripts deliberately have no build step, so they cannot
 * `import` them. That constraint is real and worth stating - it means a palette
 * assembled at runtime, or spread from another object, is invisible here. The
 * completeness check in `palette-contract.ts` is what keeps the parse honest:
 * every role must be written out literally in each palette file, and a palette
 * that omits one fails rather than being silently skipped.
 */

import { readFileSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
export const PALETTE_DIR = join(
	ROOT,
	"src/themes",
);

/**
 * Remove comments before harvesting.
 *
 * Without this the harvester reads `// was "#ff0000"` as a real declaration.
 * Combined with first-match-wins that is actively dangerous rather than merely
 * untidy: a commented-out old value sitting ABOVE the live one wins, so the
 * generated CSS paints the colour someone deliberately replaced while the
 * palette file plainly shows the new one. Nothing downstream can detect it,
 * because every consumer reads the same wrong answer.
 *
 * String literals are preserved - a `//` inside a quoted value is content, not
 * a comment - which is why this walks the source rather than running a regex
 * over it.
 */
const stripComments = (src) => {
	let out = "";
	let i = 0;
	while (i < src.length) {
		const c = src[i];
		const next = src[i + 1];
		if (c === '"' || c === "'" || c === "`") {
			// Copy the whole literal verbatim, honouring backslash escapes.
			const quote = c;
			out += c;
			i++;
			while (i < src.length) {
				out += src[i];
				if (src[i] === "\\") {
					if (i + 1 < src.length) out += src[++i];
					i++;
					continue;
				}
				if (src[i] === quote) {
					i++;
					break;
				}
				i++;
			}
			continue;
		}
		if (c === "/" && next === "/") {
			while (i < src.length && src[i] !== "\n") i++;
			continue;
		}
		if (c === "/" && next === "*") {
			i += 2;
			while (i < src.length && !(src[i] === "*" && src[i + 1] === "/")) i++;
			i += 2;
			continue;
		}
		out += c;
		i++;
	}
	return out;
};

/**
 * Read every palette as `{ id, file, palette }`, sorted by id.
 *
 * Throws on a duplicate role within one palette. The previous first-match-wins
 * behaviour silently picked one of two conflicting values, which is the worst
 * available option: the file says two things, and the build quietly believes
 * one of them. Refusing is the only honest response.
 */
export const loadPalettes = () => {
	const out = [];
	for (const file of readdirSync(PALETTE_DIR).filter((f) =>
		f.endsWith(".ts"),
	)) {
		const src = stripComments(readFileSync(join(PALETTE_DIR, file), "utf8"));
		// Each exported ThemeDefinition opens with `id: "..."` and carries one
		// `palette: { ... }` block. Split on the id so multi-theme files (the
		// brand pair lives in one) yield one entry each.
		for (const chunk of src.split(/\bid:\s*"/).slice(1)) {
			const id = chunk.slice(0, chunk.indexOf('"'));
			const pStart = chunk.indexOf("palette: {");
			if (pStart === -1) continue;
			const palette = {};
			for (const m of chunk.slice(pStart).matchAll(/(\w+):\s*"([^"]+)"/g)) {
				if (m[1] in palette && palette[m[1]] !== m[2]) {
					throw new Error(
						`${file}: palette "${id}" declares role "${m[1]}" twice, as ` +
							`"${palette[m[1]]}" and "${m[2]}". Remove one; the parser ` +
							`cannot know which you meant.`,
					);
				}
				palette[m[1]] = m[2];
			}
			out.push({ id, file, palette });
		}
	}
	return out.sort((a, b) => a.id.localeCompare(b.id));
};
