/**
 * NO SIDEWAYS SCROLL, ENFORCED IN SOURCE (operator finding B, review round 1).
 *
 * "The view should always be fixed to the phone width and not scroll sideways;
 * anything should either wrap or be safely truncated." A rendered audit catches
 * this only on the states someone thought to capture, so the classes that
 * CREATE a horizontal scroller are refused here, on every file, before render:
 *
 * - `overflow-x-auto` / `overflow-x-scroll`: an inline-axis scroller.
 * - bare `whitespace-pre` / `whitespace-nowrap` on a block that holds free text
 *   is how a code block or a long path grows past the column. `-pre-wrap` and
 *   `-pre-line` wrap, so they are fine. `whitespace-nowrap` is NOT refused:
 *   it is the right tool on a short count inside a `min-w-0` row (the cells
 *   that must not break mid-number), and refusing it would push those cells to
 *   truncate a number, which is the false reading D1 removed.
 *
 * Keep exceptions local to the individual rail. The inline data grid is
 * focusable and independent of the viewport: the 390px browser repro moved
 * the last column 40px with ArrowRight without moving the document. Otherwise
 * wide named columns would truncate exact values or lose their association.
 */
import { readdirSync, readFileSync } from "node:fs";
import { join, relative } from "node:path";
import { describe, expect, it } from "vitest";

const SRC = join(__dirname);
const REFUSED = /\boverflow-x-(?:auto|scroll)\b|\bwhitespace-pre(?![-\w])/g;
const INLINE_GRID_RAIL = '<div className={wide ? "overflow-x-auto rounded-md focus-visible:outline-accent" : undefined} {...(wide ? {';

function sources(directory: string): string[] {
	return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
		const path = join(directory, entry.name);
		if (entry.isDirectory()) return sources(path);
		return /\.tsx?$/.test(entry.name) && !/\.test\.tsx?$/.test(entry.name) ? [path] : [];
	});
}

/** A line with its comments removed. `//` inside a URL is NOT a comment (code
    review round 2, C2-1) — the first cut stripped from any `//`, so an offender
    sharing a line with `https://…` was invisible to this guard. A line comment
    starts at `//` that is either at the start of the line or preceded by
    whitespace AND not part of `scheme://`. */
export function stripComments(line: string): string {
	let code = line.replace(/^\s*\*.*$/, "");
	const block = /\/\*.*?\*\//g;
	code = code.replace(block, "");
	for (let at = code.indexOf("//"); at >= 0; at = code.indexOf("//", at + 2)) {
		const before = code.slice(0, at);
		if (!/^\s*$/.test(before) && !/\s$/.test(before)) continue; // part of a URL
		return before;
	}
	return code;
}

describe("the phone never scrolls sideways", () => {
	it("no source file creates a horizontal scroller or an unwrapped text block", () => {
		const offenders: string[] = [];
		for (const path of sources(SRC)) {
			const name = relative(SRC, path).replaceAll("\\", "/");
			readFileSync(path, "utf8")
				.split("\n")
				.forEach((line, index) => {
					// Prose in comments may NAME a class (this file's own note does).
					const code = stripComments(line);
					for (const match of code.matchAll(REFUSED)) {
						if (name === "components/inline-ui.tsx" && code.trim() === INLINE_GRID_RAIL && match[0] === "overflow-x-auto") continue;
						offenders.push(`${name}:${index + 1}: ${match[0]}`);
					}
				});
		}
		expect(
			offenders,
			"A phone view must wrap or truncate, never scroll sideways (operator finding B). " +
				"Use whitespace-pre-wrap / flex-wrap / a priority fit with an overflow sheet; " +
				"if a rail is genuinely the best option, add it to ALLOWED with the measured reason.",
		).toEqual([]);
	});

	it("an offender on a line carrying a URL is caught (negative control)", () => {
		/* The stripper, not just the regex, has to survive this line: a real
		   mistake can sit beside a link. Code review round 2, C2-1 reproduced the
		   gap: the old stripper returned "" here, so the guard passed. */
		const line = '\t\t<a href="https://example.invalid/x" className="overflow-x-auto">link</a>';
		expect([...stripComments(line).matchAll(REFUSED)].map((match) => match[0])).toEqual(["overflow-x-auto"]);
		expect([...stripComments('\tconst url = "https://example.invalid/x"; // trailing note overflow-x-auto')
			.matchAll(REFUSED)]).toEqual([]);
	});

	it("a refused class is actually caught (negative control)", () => {
		const sample = 'className="lo-scroll overflow-x-auto whitespace-pre p-2"';
		expect([...sample.matchAll(REFUSED)].map((match) => match[0])).toEqual([
			"overflow-x-auto",
			"whitespace-pre",
		]);
		expect([..."whitespace-pre-wrap whitespace-pre-line".matchAll(REFUSED)]).toEqual([]);
	});
});
