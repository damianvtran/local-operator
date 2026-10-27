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
		/* ONE OPTICAL INSET ON ALL FOUR SIDES (operator finding, round 2). The
		   shell's own `pt-2` (8 px) plus the field box's `py-1.5` (6 px) is the
		   14-15 px the placeholder sits at; the foot's `pb-0` plus the meta row's
		   own centring of its 17.4 px line inside a 44 px band is the 15.2 px under
		   the meta text. Both come from these shared constants, so the first-paint
		   frame reserves the same box. Measured in the browser, not here: happy-dom
		   does no layout. */
		expect(shell?.className).toContain("px-2");
		expect(shell?.className).toContain("pt-2");
		expect(shell?.className).toContain("pb-0.5");
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

	it("lets the meta facts take only the LEFTOVER width in a short viewport (QA round 3, O1)", () => {
		/* O1: with the keyboard open (max-height 480) the meta facts move inline
		   beside attach and the stop/send pair. As `basis-auto` their flex basis
		   was their max-content (cwd 44 + the model label + the 96 px reading,
		   248 px [redacted] at 320), so the line breaker pushed the 90 px action
		   pair onto a SECOND footer line: composer 331.4 px against 287.4, and the
		   above-composer slot shrank to 20.6 px. `basis-0` gives the meta facts
		   only what is left, and the model label, the elastic element, truncates
		   instead of the controls wrapping. */
		const { container } = render(<ComposerFrame />);
		const row = container.querySelector(".lo-composer > div");
		const meta = [...(row?.children ?? [])].find((child) => child.className.includes("order-last"));
		const cls = meta?.className.split(/\s+/) ?? [];
		expect(cls).toContain("short-column:basis-0");
		expect(cls).not.toContain("short-column:basis-auto");
		// It never wraps on its own minimum either: the minimum is made to fit.
		expect(cls).toContain("min-w-0");
		expect(cls).toContain("overflow-x-clip");
	});
});
