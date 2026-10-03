// @vitest-environment happy-dom
//
// THE SHEET BELONGS TO THE PHONE'S VIEWPORT, NOT TO WHICHEVER ANCESTOR HAPPENS
// TO BE POSITIONED. This is the regression the operator hit on the approvals
// sheet.
//
// WHAT WENT WRONG. The overlay was `absolute inset-0`, which resolves against
// the nearest POSITIONED ancestor — and in the session view that is not the
// column. The header hangs inside a `relative` div (so the session-health ladder
// can anchor to that div's bottom) and `GateSheet` is mounted inside the header,
// so the dialog's box resolved to 390x53 — the header — and the bottom-anchored
// panel landed at top=-263 / bottom=53 on a 390x844 screen: a 53px sliver of the
// panel's own tail at the top of the display, cut mid-sentence, with the buttons
// and the ✕ 263px above a screen that cannot scroll up to them.
//
// The unit half is the same lesson `session-view.overflow.test` records for the
// ask card, and the sheet was the one bounded region still exempt from it:
// `max-h-[85dvh]` follows the DYNAMIC viewport while the column is pinned to
// `visualViewport.height`, and a virtual keyboard shrinks only the second. With
// the column at a 480px visual viewport the asks sheet's panel rendered 717px
// tall (85dvh of an 844px layout viewport), 237px of it above the screen.
//
// WHAT THIS LAYER CAN PROVE, AND WHAT IT CANNOT. happy-dom does no layout — every
// box is 0x0 — so "the panel is on screen" is unanswerable here, and it carries
// no viewport units either. What it CAN resolve is the panel's `max-height`, which
// is the cap that has to track the column; that is asserted below, and it fails
// if the cap goes back to a `dvh` class or to any other source than the pin.
// The pixels are asserted on the real bundle: the capture rig measured the gate
// sheet's panel at top=-263/bottom=53 before the fix and top=528/bottom=844 after
// it, and the asks sheet's at 717px tall clipped in a 480px pinned column before
// it and 408px after — the frames and the numbers are on the PR.
import { cleanup, render, screen } from "@testing-library/react";
import type { CSSProperties } from "react";
import { afterEach, describe, expect, it } from "vitest";
import { COLUMN_HEIGHT_VAR, COLUMN_TOP_VAR } from "../../lib/column";
import { Sheet } from "./sheet";

afterEach(cleanup);

/** The REAL mount shape: a column that publishes the pin, the `relative` header
 *  wrapper the ladder hangs off, the header itself, and the sheet inside it —
 *  the arrangement that put the approvals panel 263px above a 390x844 screen. */
function mountInHeader(pinPx: number, offsetPx = 0) {
	const pin = {
		[COLUMN_HEIGHT_VAR]: `${pinPx}px`,
		[COLUMN_TOP_VAR]: `${offsetPx}px`,
	} as CSSProperties;
	const { container } = render(
		<div className="h-dvh" style={pin}>
			<div className="relative">
				<header data-testid="column-header">
					<Sheet open onClose={() => {}} title="Approvals in this session">
						<p>Ask parks every gated tool call until you answer its card.</p>
					</Sheet>
				</header>
			</div>
		</div>,
	);
	const dialog = screen.getByRole("dialog");
	const panel = dialog.querySelector<HTMLElement>(".lo-sheet-panel");
	if (!panel) throw new Error("no sheet panel");
	return { container, dialog, panel };
}

describe("the sheet overlay's box", () => {
	it("reads the panel's cap from the column's pin, so it cannot be taller than the column", () => {
		const { panel } = mountInHeader(480);
		// Column units: the cap resolves against the pinned visual viewport the
		// column was pinned to, fraction and all. `85dvh` resolves to nothing
		// here, which is exactly how the divergence shipped.
		expect(getComputedStyle(panel).maxHeight).toContain("480px");
		expect(getComputedStyle(panel).maxHeight).toContain("0.85");
	});

	it("tightens with the pin instead of holding a second source of truth", () => {
		const tall = mountInHeader(844);
		expect(getComputedStyle(tall.panel).maxHeight).toContain("844px");
		cleanup();
		// A keyboard-sized visual viewport: the same fraction of a smaller box.
		const short = mountInHeader(300);
		expect(getComputedStyle(short.panel).maxHeight).toContain("300px");
	});

	it("stays in the phone column instead of becoming a body portal", () => {
		// The cmux screenshot surface IS this column, and a body portal paints
		// outside it — the reason the in-flow overlay exists at all. Whatever
		// anchors the overlay to the viewport, it must not be bought with a
		// portal: the node stays where the caller mounted it.
		const { container, dialog } = mountInHeader(844);
		expect(screen.getByTestId("column-header").contains(dialog)).toBe(true);
		expect(container.firstElementChild?.contains(dialog)).toBe(true);
		expect(document.body.querySelectorAll('[role="dialog"]').length).toBe(1);
	});

	it("keeps the panel's content scrollable inside whatever the cap allows", () => {
		// A cap is only a bound: the tail has to stay reachable through the
		// panel's own scroller, which is also what `min-h-0` is for — a flex
		// child defaults to `min-height: auto` and would refuse to shrink.
		const { panel } = mountInHeader(480);
		const scroller = panel.querySelector<HTMLElement>(".lo-scroll");
		expect(scroller).not.toBeNull();
		expect(scroller?.className).toContain("overflow-y-auto");
		expect(scroller?.className).toContain("min-h-0");
		expect(panel.contains(scroller)).toBe(true);
	});
});
