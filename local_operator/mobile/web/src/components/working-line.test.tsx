// @vitest-environment happy-dom
//
// The band's clock: withheld only when the server says it has no instant.
//
// Two design round 1 findings live in this one span (D4, D2), and review round 2
// found that the fix for D4 had over-reached. The wire carries ONE nullable
// number, `activity_started_s`, and it distinguishes exactly two states:
//
//   * `null` — the fold has no instant it can honestly date the phase from (a
//     label it joined mid-flight whose producer stated none). WITHHOLD the
//     digits, keep the reserved cells.
//   * `0.0` — a KNOWN zero: the phase edge the server watched begin. Render
//     `0s` from the first frame and count up, exactly as the TUI's working block
//     does. Every phase edge publishes 0.0, so gating on `> 0` (the previous
//     head) deleted the clock for the whole life of any phase the phone watched
//     begin — a responding run, a dictation batch, a single running tool call
//     (review round 2, MAJOR 1).
//
// What jsdom can prove is the STRUCTURE: which span exists, what it says, and
// that the slot is reserved by a width class rather than measured at runtime.
// What it cannot prove is the widget's geometry, so the pixel evidence for the
// band's slot is the before/after browser frames recorded on the PR (`320x568`,
// `360x780`, `390x844`) with the slot box and the label's clip point in
// characters read back from the live DOM.
import { cleanup, render, screen } from "@testing-library/react";
import { act } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { WorkingLine } from "./working-line";

afterEach(() => {
	cleanup();
	vi.useRealTimers();
});

describe("WorkingLine", () => {
	it("withholds the clock when the server states no instant", () => {
		const { container } = render(<WorkingLine activity="responding" startedS={null} />);
		expect(container.textContent).toContain("responding");
		// No digits-and-seconds anywhere: the fabricated `0.0s` D4 removed.
		expect(container.textContent).not.toMatch(/\d+(\.\d+)?s/);
	});

	it("paints a known zero and ticks up from it", () => {
		// The MAJOR 1 state: a phase edge the server watched begin. `0s`, and the
		// local tick carries it forward between the server's repaints.
		vi.useFakeTimers();
		const { container } = render(<WorkingLine activity="responding" startedS={0} />);
		expect(screen.getByText("0s")).toBeTruthy();

		act(() => {
			vi.advanceTimersByTime(1000);
		});
		expect(screen.getByText("1s")).toBeTruthy();
		expect(container.textContent).toMatch(/1s/);

		act(() => {
			vi.advanceTimersByTime(1000);
		});
		expect(screen.getByText("2s")).toBeTruthy();
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
		// same reason. Both states this PR distinguishes are covered: withheld
		// (`null`) and known-zero (`0`).
		for (const startedS of [null, 0, 45, 3599, 3602400] as (number | null)[]) {
			const { container } = render(<WorkingLine activity="responding" startedS={startedS} />);
			const clock = container.querySelector(".w-\\[6ch\\]");
			expect(clock, `startedS=${startedS}`).toBeTruthy();
			expect(clock?.className).toContain("text-right");
			expect(clock?.className).toContain("shrink-0");
			cleanup();
		}
	});
});
