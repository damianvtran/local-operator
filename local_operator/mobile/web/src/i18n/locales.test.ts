/** The runtime's locale contract and the shipped manifest must be ONE list.
 *
 * `isResolvedLocale` gates both `registerBundle` and the wire loader, and its
 * tags are literals in `runtime.ts` while `i18n/data/shipped.json` is generated
 * from ledger completeness. Emitting a locale's copy into the bundle while the
 * runtime refuses its tag at registration fails silently in the worst way: every
 * catalogue audited, the bytes shipped, and the surface still English — the
 * failure the completeness audits cannot see because nothing is incomplete.
 */
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { directionFor, isResolvedLocale } from "./runtime";

/** The committed manifest, read the way `generate.py` reads it — not a list
 * repeated here, so a new language reaches this test by being shipped. */
function shippedLocales(): string[] {
	const path = join(
		dirname(fileURLToPath(import.meta.url)),
		"../../../../i18n/data/shipped.json",
	);
	return JSON.parse(readFileSync(path, "utf-8")).locales;
}

describe("shipped locale contract", () => {
	it("accepts every shipped locale, so a complete bundle can register", () => {
		const locales = shippedLocales();
		expect(locales.length).toBeGreaterThan(0);
		for (const locale of locales) expect([locale, isResolvedLocale(locale)]).toEqual([locale, true]);
	});

	it("still refuses tags the manifest does not ship", () => {
		const shippedSet = new Set(shippedLocales());
		// Regional, cased and pseudo tags: the shapes a config write or a header
		// can carry. None may be registered, and none may select a wire catalogue.
		for (const refused of ["auto", "en-GB", "fr-CA", "EN", "zh-CN-x", ""]) {
			if (shippedSet.has(refused)) continue;
			expect([refused, isResolvedLocale(refused)]).toEqual([refused, false]);
		}
	});

	it("gives Urdu the RTL direction and every other shipped locale LTR", () => {
		for (const locale of shippedLocales()) {
			expect([locale, directionFor(locale)]).toEqual([locale, locale === "ur" ? "rtl" : "ltr"]);
		}
	});
});
