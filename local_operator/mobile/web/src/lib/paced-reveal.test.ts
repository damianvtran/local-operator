import { describe, expect, it } from "vitest";
import { MAX_LAG_MS, PacedReveal, TICK_MS, unitEnd } from "./paced-reveal";

/** Drive a reveal the way the hook does: ticks every TICK_MS until idle. */
function drain(reveal: PacedReveal, from: number, limitMs = 5000): { at: number; shown: string }[] {
	const steps: { at: number; shown: string }[] = [];
	for (let now = from + TICK_MS; now <= from + limitMs && !reveal.idle(); now += TICK_MS) {
		if (reveal.tick(now)) steps.push({ at: now, shown: reveal.shown() });
	}
	return steps;
}

describe("unitEnd", () => {
	it("ends a word only once the whitespace after it has arrived", () => {
		expect(unitEnd("hello", 0)).toBe(-1);
		expect(unitEnd("hello ", 0)).toBe(6);
		expect(unitEnd("hello  world", 0)).toBe(7);
		expect(unitEnd("hello world", 6)).toBe(-1);
	});

	it("merges markup-only units with the word they open", () => {
		// "## " alone renders an empty heading; it travels with its first word.
		expect(unitEnd("## Title more", 0)).toBe(9);
		expect(unitEnd("## ", 0)).toBe(-1);
		expect(unitEnd("## Ti", 0)).toBe(-1);
		expect(unitEnd("- item x", 0)).toBe(7);
		expect(unitEnd("| --- | --- |\nCell ", 0)).toBe(19);
	});

	it("treats each ideograph as its own unit", () => {
		expect(unitEnd("你好世界", 0)).toBe(1);
		expect(unitEnd("你好世界", 1)).toBe(2);
	});

	it("never ends a unit before a combining mark or joiner", () => {
		// "e" + U+0301 arriving as separate deltas must not split the grapheme.
		expect(unitEnd("e\u0301 x", 0)).toBe(3);
		expect(unitEnd("你\u0301", 0)).toBe(-1);
	});
});

describe("PacedReveal", () => {
	it("shows the first word immediately, and all of a partial first word", () => {
		const first = new PacedReveal("Hello world and more", 0, true);
		expect(first.shown()).toBe("Hello ");
		expect(new PacedReveal("Hel", 0, true).shown()).toBe("Hel");
	});

	it("shows a first word that includes the heading markup it opens", () => {
		expect(new PacedReveal("## Heading\n\nBody text ", 0, true).shown()).toBe("## Heading\n\n");
		// Markup with no word yet: show what arrived rather than nothing.
		expect(new PacedReveal("## ", 0, true).shown()).toBe("## ");
	});

	it("does not hold back a row that was already on screen", () => {
		const seed = new PacedReveal("already shown text", 0, false);
		expect(seed.shown()).toBe("already shown text");
		seed.push("already shown text and a burst of new words", 10);
		expect(seed.shown()).toBe("already shown text");
	});

	it("spreads a burst over several ticks and never shows a partial word", () => {
		const words = Array.from({ length: 60 }, (_, i) => `word${i}`).join(" ");
		const reveal = new PacedReveal("Intro ", 0, true);
		reveal.push(`Intro ${words} `, 0);
		const steps = drain(reveal, 0);
		expect(steps.length).toBeGreaterThan(3);
		for (const { shown } of steps) expect(shown.endsWith(" ")).toBe(true);
		// Every step is a strict prefix of the next: nothing reordered or dropped.
		steps.forEach(({ shown }, i) => {
			if (i > 0) expect(shown.startsWith(steps[i - 1].shown)).toBe(true);
		});
		expect(reveal.shown()).toBe(`Intro ${words} `);
	});

	it("bounds the lag: a word stuck mid-token is shown MAX_LAG_MS after it arrived", () => {
		const reveal = new PacedReveal("Hi ", 0, true);
		reveal.push("Hi partia", 100);
		expect(reveal.tick(100 + MAX_LAG_MS - 1)).toBe(false);
		expect(reveal.tick(100 + MAX_LAG_MS)).toBe(true);
		expect(reveal.shown()).toBe("Hi partia");
	});

	it("shows a rewrite at once instead of replaying it", () => {
		const reveal = new PacedReveal("The first draft ", 0, false);
		reveal.push("The first draft is long enough to pace ", 10);
		expect(reveal.shown()).toBe("The first draft ");
		reveal.push("A completely different answer that was corrected ", 20);
		expect(reveal.shown()).toBe("A completely different answer that was corrected ");
		expect(reveal.idle()).toBe(true);
	});

	it("ends on exactly the received text after flush", () => {
		const text = `${"alpha beta gamma ".repeat(40)}tail`;
		const reveal = new PacedReveal("alpha ", 0, true);
		reveal.push(text, 0);
		reveal.tick(TICK_MS);
		expect(reveal.shown().length).toBeLessThan(text.length);
		reveal.flush();
		expect(reveal.shown()).toBe(text);
		expect(reveal.idle()).toBe(true);
	});

	it("is idempotent for equal text and never moves backwards", () => {
		const reveal = new PacedReveal("one two three four ", 0, true);
		reveal.push("one two three four ", 5);
		reveal.tick(TICK_MS);
		const before = reveal.shown();
		reveal.push("one two three four ", 40);
		expect(reveal.shown()).toBe(before);
	});
});
