import { describe, expect, it } from "vitest";
import { contentDir } from "./content-direction";

describe("dominant-script truncation direction", () => {
	it.each([
		["تحقیقاتی منصوبے کی پیش رفت اور اگلے مرحلے کی تفصیلی گفتگو", "rtl"],
		["Identifying Latin title", "ltr"],
		["API منصوبے کی پیش رفت اور اگلے مرحلے کی تفصیلی گفتگو", "rtl"],
		// The caller passes the name, never the generated [fork] or chat: marker.
		["شناختیابتدا اردو گفتگو کی شناخت اور تفصیلی جائزہ", "rtl"],
		["Latin majority with اردو", "ltr"],
		["éÀčŁœ ß", "ltr"],
		["2026-11-03 1234567890", undefined],
		["", undefined],
		[" ✨ [!?] ١٢٣ ", undefined],
		["aا", undefined],
		// Arabic-block punctuation/digits are not letters and must not cast votes.
		["A؟،١٢٣", "ltr"],
		["پ \u0750 \u08a0 \ufb50 \ufe8d", "rtl"],
	])("resolves %j to %s", (text, expected) => {
		expect(contentDir(text)).toBe(expected);
	});
});
