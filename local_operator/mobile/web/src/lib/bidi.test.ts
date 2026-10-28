import { describe, expect, it } from "vitest";
import { isolate, isolateTechnicalRuns, textDirection } from "./bidi";

describe("bidi isolation at the render boundary", () => {
	it("takes an element's direction from its first strong character", () => {
		expect(textDirection("Files a conversation creates will appear here.")).toBe("ltr");
		expect(textDirection("فائلیں یہاں ظاہر ہوں گی۔")).toBe("rtl");
		// Weak-only strings must not flip a paragraph: numbers and separators are
		// exactly the content that reads wrong under a guessed direction.
		expect(textDirection("1 · 2")).toBe("ltr");
		expect(textDirection("")).toBe("ltr");
	});

	it("isolates path and URL runs without touching the rest of the sentence", () => {
		expect(isolate("Message…")).toBe("\u2068Message…\u2069");
		expect(isolateTechnicalRuns("see /Users/damian/local-operator for logs")).toBe(
			"see \u2068/Users/damian/local-operator\u2069 for logs",
		);
		expect(isolateTechnicalRuns("open https://example.test/a?b=c")).toBe(
			"open \u2068https://example.test/a?b=c\u2069",
		);
	});

	it("leaves prose with word-internal slashes and copy without paths alone", () => {
		expect(isolateTechnicalRuns("and/or 3/4")).toBe("and/or 3/4");
		expect(isolateTechnicalRuns("سیشن سے منسلک ہو رہا ہے…")).toBe("سیشن سے منسلک ہو رہا ہے…");
	});
});
