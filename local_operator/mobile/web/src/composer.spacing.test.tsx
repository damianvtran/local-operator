// @vitest-environment happy-dom
//
// THE COMPOSER'S SEPARATION RULES (operator finding A).
//
// The reported defect was geometric: the stop and send discs touched each other
// and the send disc sat flush against the composer's foot, so two 44 px controls
// read as one clipped blob. happy-dom does no layout, so the pixels themselves
// are browser-measured (390x844: stop->steer gap 2 px, every disc 8 px above the
// shell's foot; 320x568 and 360x780: attach->workspace gap 2 px, 8 px below).
//
// What is pinnable HERE is the structure that made those numbers true, and the
// one thing a later edit can silently break: the composer and the first-paint
// frame must share the SAME constants, or the reserve stops matching the real
// box and every conversation opens with a layout shift.
import { cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { ComposerFrame } from "./components/composer";

afterEach(cleanup);

describe("the composer's spacing constants", () => {
	it("keeps the shell's side and foot padding symmetrical, so no disc sits flush", () => {
		const { container } = render(<ComposerFrame />);
		const shell = container.querySelector(".lo-composer");
		expect(shell).toBeTruthy();
		// px-2 = 8 px at the sides; pb-2 = 8 px below the discs. One constant pair
		// for both, and the measured clearance is that same 8 px.
		expect(shell?.className).toContain("px-2");
		expect(shell?.className).toContain("pb-2");
	});

	it("puts every control in the row 2 px apart, and groups the right-hand pair", () => {
		const { container } = render(<ComposerFrame />);
		const row = container.querySelector(".lo-composer > div");
		expect(row?.className).toContain("gap-x-0.5");
		const cluster = row?.lastElementChild;
		expect(cluster?.className).toContain("gap-0.5");
		// The cluster holds the submit disc; the real composer puts stop beside it
		// in the same cluster, which is where the 2 px between them comes from.
		expect(cluster?.children.length).toBe(1);
		// `ms-auto` pins the pair to the inline end now that the meta facts sit
		// on their own line below it.
		expect(cluster?.className).toContain("ms-auto");
	});

	it("gives the meta facts their own full-width line, in the first frame too", () => {
		/* Operator finding B / D1: on the action line the model label got 44-58 px
		   and read "Claude Op…"; on its own line it measured 140 px at 320 and
		   210 px at 390 (have = need). The frame and the composer share this
		   constant, so the reserved first frame has the same three lines. */
		const { container } = render(<ComposerFrame />);
		const row = container.querySelector(".lo-composer > div");
		const meta = [...(row?.children ?? [])].find((child) => child.className.includes("order-last"));
		expect(meta?.className).toContain("basis-full");
		expect(meta?.className).toContain("min-w-0");
	});
});
