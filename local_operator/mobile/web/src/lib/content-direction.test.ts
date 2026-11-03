import { describe, expect, it } from "vitest";
import { contentDir } from "./content-direction";

const urdu = "منصوبے کی پیش رفت اور اگلے مرحلے کی تفصیلی گفتگو";

describe("first-substantive-run truncation direction", () => {
	it.each([
		[`Investigate production database migration — ${urdu.repeat(3)}`, "ltr"],
		[`${urdu} — Investigate production database migration and a much longer English explanation`.repeat(2), "rtl"],
		[`API ${urdu}`, "rtl"],
		[`API SDK ${urdu}`, "rtl"],
		[`GLM-5.2 ${urdu}`, "ltr"],
		["GLM-5.2 (Token Plan)", "ltr"],
		[`V4 Flash ${urdu}`, "ltr"],
		["DeepSeek V4 Flash (2026-07-31)", "ltr"],
		// A vendor identifier wins even when its Urdu explanation is longer;
		// the existing title/long-press affordance exposes the clipped remainder.
		[`DeepSeek ${urdu}`, "ltr"],
		[urdu, "rtl"],
		["Identifying Latin title", "ltr"],
		["éÀčŁœ ß", "ltr"],
		["2026-11-03 1234567890", undefined],
		["١٢٣ ۱۲۳", undefined],
		["✨ 👩‍💻", undefined],
		["", undefined],
		[" ✨ [!?] ١٢٣ ", undefined],
		["\u0301 \u200d", undefined],
		// The first lowercase token is substantive even in a one-letter tie.
		["aا", "ltr"],
		["A؟،١٢٣", "ltr"],
		["API SDK", "ltr"],
		["پ \u0750 \u08a0 \ufb50 \ufe8d", "rtl"],
		[`short ${urdu}`, "ltr"],
		[`two words ${urdu}`, "ltr"],
		[`abcdefghijk ${urdu}`, "ltr"],
		[`abcdefghijkl ${urdu}`, "ltr"],
		[`ABCDE 12345 ${urdu}`, "rtl"],
		[`ABCDEF GHI ${urdu}`, "ltr"],
		[`one-thing ${urdu}`, "ltr"],
		[`API ✨ SDK ${urdu}`, "rtl"],
		[`API 1234567890 ${urdu}`, "rtl"],
		[`API !!! ${urdu}`, "rtl"],
	])("resolves %j to %s", (text, expected) => {
		expect(contentDir(text)).toBe(expected);
	});
});
