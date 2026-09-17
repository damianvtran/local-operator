// @vitest-environment happy-dom
//
// The band's clock: withheld when no start is known, reserved when it is.
//
// Two design round 1 findings live in this one span (D4, D2) and both are about
// the same thing — a number the server sends as `activity_started_s` is the ONLY
// thing the band has to go on, and it carries 0.0 for two different states:
// "this phase began this instant" and "no anchor was stated at all" (the fold's
// third step, its own arrival). The band must not turn the second into a
// fabricated `0.0s` while the tool rows in the same frame withhold theirs
// (`entry.elapsed_s > 0`), and it must not let the number's own width reflow the
// label beside it as the format changes.
//
// What jsdom can prove is the STRUCTURE: which span exists, what it says, and
// that the slot is reserved by a width class rather than measured at runtime.
// What it cannot prove is the widget's geometry, so the pixel evidence for both
// findings is the before/after browser frames recorded on the PR (`320x568`,
// `360x780`, `390x844`) with the band's slot box and the label's clip point in
// characters read back from the live DOM.
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { WorkingLine } from "./working-line";

afterEach(cleanup);

describe("WorkingLine", () => {
	it("withholds the clock when the projection states no start", () => {
		const { container } = render(<WorkingLine activity="responding" startedS={0} />);
		expect(container.textContent).toContain("responding");
		// No digits-and-seconds anywhere: the fabricated `0.0s` this change removes.
		expect(container.textContent).not.toMatch(/\d+(\.\d+)?s/);
	});

	it("prints the TUI's bounded wording once it has an age", () => {
		const { container } = render(<WorkingLine activity="responding" startedS={135} />);
		expect(screen.getByText("2m15s")).toBeTruthy();
		expect(container.textContent).not.toContain("2m 15s");
	});

	it("reserves the clock's slot rather than measuring it", () => {
		// `w-[6ch]` is the width of the widest form the formatter can produce, so
		// the label's clip point cannot move as the number changes form — and an
		// empty slot is the same width as a full one, which is what makes the
		// withhold above free. The TUI reserves `WorkingBlock._CLOCK_COL` for the
		// same reason.
		for (const startedS of [0, 45, 3599, 3602400]) {
			const { container } = render(<WorkingLine activity="responding" startedS={startedS} />);
			const clock = container.querySelector(".w-\\[6ch\\]");
			expect(clock, `startedS=${startedS}`).toBeTruthy();
			expect(clock?.className).toContain("text-right");
			expect(clock?.className).toContain("shrink-0");
			cleanup();
		}
	});
});
