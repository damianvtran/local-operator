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
 * A file that genuinely needs one of these goes in ALLOWED with the measured
 * reason. It is empty: every former rail was replaced (filter strip and status
 * tray fit by priority, the library categories and markdown wrap).
 */
import { readdirSync, readFileSync } from "node:fs";
import { join, relative } from "node:path";
import { describe, expect, it } from "vitest";

const SRC = join(__dirname);
const REFUSED = /\boverflow-x-(?:auto|scroll)\b|\bwhitespace-pre(?![-\w])/g;
/** `path` → reason. Empty on purpose; see the module note. */
const ALLOWED: Record<string, string> = {};

function sources(directory: string): string[] {
	return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
		const path = join(directory, entry.name);
		if (entry.isDirectory()) return sources(path);
		return /\.tsx?$/.test(entry.name) && !/\.test\.tsx?$/.test(entry.name) ? [path] : [];
	});
}

describe("the phone never scrolls sideways", () => {
	it("no source file creates a horizontal scroller or an unwrapped text block", () => {
		const offenders: string[] = [];
		for (const path of sources(SRC)) {
			const name = relative(SRC, path).replaceAll("\\", "/");
			if (name in ALLOWED) continue;
			readFileSync(path, "utf8")
				.split("\n")
				.forEach((line, index) => {
					// Prose in comments may NAME a class (this file's own note does).
					const code = line.replace(/\/\/.*$|\/\*.*?\*\/|^\s*\*.*$/g, "");
					for (const match of code.matchAll(REFUSED)) offenders.push(`${name}:${index + 1}: ${match[0]}`);
				});
		}
		expect(
			offenders,
			"A phone view must wrap or truncate, never scroll sideways (operator finding B). " +
				"Use whitespace-pre-wrap / flex-wrap / a priority fit with an overflow sheet; " +
				"if a rail is genuinely the best option, add it to ALLOWED with the measured reason.",
		).toEqual([]);
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
