// @vitest-environment happy-dom
//
// THE TABLE-FIT DECISION (code review round 3, C3-2). The code comment and the
// round-2 reply both claimed these helpers were unit-tested; they were not, and
// the missing test is exactly what would have caught C3-1's wrong-font verdict.
//
// Covered here:
//   - `markdownTableText` strips inline syntax and leaves the visible text;
//   - `minContentWidths` measures the longest WORD, adds the cell padding, and
//     sizes to the widest column count;
//   - `tableFitsIn` at the boundary, including the 13 px-vs-14 px band C3-1 was
//     about (a verdict that flips with the font is the defect);
//   - the component renders ONE form and keeps it across repeated observer
//     callbacks at an unchanged width (the flip-flop C3-1 described);
//   - both forms render the same number of cells for a row wider than its header
//     (C3-5), and the stacked form's header is part of the cell (C3-4).
import { act, cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type * as MarkdownModule from "./components/markdown";

/* ONE FRESH MODULE PER TEST. The measured-font probe is a module-level canvas
   (C3-6), so a cached context from an earlier test would answer for a later one.
   `vi.resetModules()` plus a dynamic import is the honest way to test a module
   that deliberately holds state; nothing test-only is exported from product code. */
let md: typeof MarkdownModule;
beforeEach(async () => {
	vi.resetModules();
	md = await import("./components/markdown");
});

function restoreMeasure() {
	vi.unstubAllGlobals();
	vi.restoreAllMocks();
	for (const proto of [HTMLElement.prototype, Element.prototype]) {
		Reflect.deleteProperty(proto, "clientWidth");
	}
}

afterEach(() => {
	cleanup();
	restoreMeasure();
});

/* A measure standing in for canvas `measureText`: 0.55 em per character at the
   given font size, which is close to the system-ui advance the phone actually
   gets. The absolute number does not matter; the RATIO between 13 px and 14 px
   does, because that 7.7 % is what made C3-1's verdict wrong. */
const measureAt = (fontPx: number) => (text: string) => text.length * fontPx * 0.55;

describe("markdownTableText", () => {
	it("keeps the visible text and drops the inline syntax", () => {
		expect(md.markdownTableText("**bold**")).toBe("bold");
		expect(md.markdownTableText("`code`")).toBe("code");
		expect(md.markdownTableText("~~gone~~")).toBe("gone");
		expect(md.markdownTableText("[label](https://example.invalid/x)")).toBe("label");
		expect(md.markdownTableText("  spaced  ")).toBe("spaced");
	});
});

describe("minContentWidths", () => {
	it("measures the longest WORD, not the longest cell", () => {
		// "conversation" (12) is the longest word; the whole cell is 24 characters.
		const widths = md.minContentWidths(["screen"], [["conversation list"]], measureAt(10));
		expect(widths[0]).toBe(Math.ceil(12 * 10 * 0.55) + 16);
	});

	it("sizes to the widest COLUMN COUNT, not to the header", () => {
		/* A row with more cells than the header must still report a width for the
		   extra column (C3-5): the stacked form renders those cells too. */
		const widths = md.minContentWidths(["a"], [["a", "bbbb"]], measureAt(10));
		expect(widths).toHaveLength(2);
		expect(widths[1]).toBe(Math.ceil(4 * 10 * 0.55) + 16);
	});

	it("adds the cell padding to every column, including an empty one", () => {
		expect(md.minContentWidths(["", ""], [], measureAt(10))).toEqual([16, 16]);
	});

	it("reads a column's inline overhead off a cell's computed style, rounded up (Q4-1)", () => {
		expect(md.cellInlineOverflow({ paddingLeft: "8px", paddingRight: "8px",
			borderLeftWidth: "0px", borderRightWidth: "0px" } as CSSStyleDeclaration)).toBe(16);
		expect(md.cellInlineOverflow({ paddingLeft: "8px", paddingRight: "8px",
			borderLeftWidth: "0.5px", borderRightWidth: "0.5px" } as CSSStyleDeclaration)).toBe(17);
		expect(md.cellInlineOverflow({ paddingLeft: "", paddingRight: "",
			borderLeftWidth: "", borderRightWidth: "" } as CSSStyleDeclaration)).toBe(0);
	});
});

describe("tableFitsIn", () => {
	it("fits when the columns plus rules equal the available width, not when over by one", () => {
		const widths = [100, 100];
		expect(md.tableFitsIn(widths, 202)).toBe(true);
		expect(md.tableFitsIn(widths, 201)).toBe(false);
	});

	it("gives ONE verdict inside the 13 px-vs-14 px band (C3-1)", () => {
		/* The band: a table whose min-content total is <= the column at the cell
		   font (13 px) but > it at the wrapper's inherited 14 px. Measuring at the
		   drawn font is what makes this a single verdict. */
		const cell = md.minContentWidths(["chrome"], [["conversation"]], measureAt(13));
		const inherited = md.minContentWidths(["chrome"], [["conversation"]], measureAt(14));
		const available = (cell[0] as number) + 2;
		expect(md.tableFitsIn(cell, available)).toBe(true);
		expect(inherited.some((width) => width > (cell[0] as number))).toBe(true);
		// At the drawn font it fits; judging by the larger font would have stacked it.
		expect(md.tableFitsIn(inherited, available)).toBe(false);
	});

	it("measures the header row at the HEADER font, so a fitting table is not stacked (Q4-1)", () => {
		/* The fixture table's real numbers, measured in the browser (rig probe):
		   header draws at 600, body at 400, both 13 px; per-column overhead 16 px.
		   The header words advance 135 / 157 px at that weight, so the honest total
		   is 135+16 + 157+16 + 2 rules = 326 — while the ENGINE's rendered
		   min-content for the same table is 323.3 px. Measured at the BODY font the
		   total was 312, i.e. 11 px BELOW what the engine needs, which is why a
		   315–322 px container chose `table` and clipped its right edge. */
		const head = ["counterreconciliation", "acknowledgementledger"];
		const rows = [["settled", "confirmed"]];
		const ADVANCE: Record<string, [number, number]> = {
			// [at the header's 600, at the body's 400] — the probe's own numbers
			counterreconciliation: [135, 128],
			acknowledgementledger: [157, 150],
		};
		const advance = (text: string, isHeader: boolean) =>
			ADVANCE[text]?.[isHeader ? 0 : 1] ?? text.length * (isHeader ? 6.4 : 6.0);
		const right = md.minContentWidths(head, rows, advance, 16);
		const bodyOnly = md.minContentWidths(head, rows, (text: string) => advance(text, false), 16);
		expect(right).toEqual([151, 173]);
		expect(bodyOnly).toEqual([144, 166]);
		// The body-font total (312) sat BELOW the engine's 323.3 px — the clip.
		expect((bodyOnly[0] as number) + (bodyOnly[1] as number) + 2).toBeLessThan(323.3);

		const total = (right[0] as number) + (right[1] as number) + 2;
		// Exactly the width the columns need: it fits.
		expect(md.tableFitsIn(right, total)).toBe(true);
		// One pixel narrower than the columns need: it does not.
		expect(md.tableFitsIn(right, total - 1)).toBe(false);
		// AND the helper is on the SAFE side of the engine: its total is never below
		// the rendered min-content (323.3 px here), which is the property that stops
		// a fitting verdict from clipping inside `overflow: hidden`.
		expect(total).toBeGreaterThanOrEqual(323.3);
	});

	it("treats one very long token as that token's own width", () => {
		const token = "a".repeat(200);
		const widths = md.minContentWidths(["path"], [[token]], measureAt(13));
		expect(widths[0]).toBe(Math.ceil(200 * 13 * 0.55) + 16);
		expect(md.tableFitsIn(widths, 300)).toBe(false);
	});
});

describe("MarkdownTable", () => {
	/** Force the measurement path: a real width and a real canvas substitute.
	    happy-dom defines no `clientWidth` at all, so it is defined rather than
	    spied on; `restoreAllMocks` plus the descriptor delete below put the
	    prototype back exactly as it was. */
	function forceMeasure(width: number) {
		/* happy-dom carries `clientWidth` as a getter on the HTML element
		   prototypes (not on `Element.prototype`, and not as an own property), so
		   it has to be replaced where it actually lives. A holder rather than a
		   closure value: the observer tests change the width mid-test. */
		const box = { width };
		for (const proto of [HTMLElement.prototype, Element.prototype]) {
			Object.defineProperty(proto, "clientWidth", {
				configurable: true,
				get: () => box.width,
			});
		}
		const measureText = vi.fn((text: string) => ({ width: text.length * 13 * 0.55 }));
		vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue({
			measureText,
			font: "",
		} as unknown as CanvasRenderingContext2D);
		return { measureText, setWidth: (next: number) => { box.width = next; } };
	}

	/* A ResizeObserver that records its callback so a test can invoke it exactly as
	   the engine would. `window.dispatchEvent(new Event("resize"))` never reached
	   happy-dom's observer, so the previous version of the no-flip-flop test passed
	   even with the width guard deleted (code review round 4, C4-1). */
	class StubResizeObserver {
		static instances: StubResizeObserver[] = [];
		callback: () => void;
		constructor(callback: () => void) {
			this.callback = callback;
			StubResizeObserver.instances.push(this);
		}
		observe() {}
		unobserve() {}
		disconnect() {}
	}
	function stubResizeObserver() {
		StubResizeObserver.instances = [];
		vi.stubGlobal("ResizeObserver", StubResizeObserver);
		return StubResizeObserver;
	}

	it("pins the measured font to the drawn font on BOTH forms (C3-1)", () => {
		const { container } = render(<md.Markdown text={"| a | b |\n|---|---|\n| 1 | 2 |"} />);
		const wrapper = container.querySelector(`.${md.TABLE_FONT_CLASS.split(" ")[0]}`);
		expect(wrapper).toBeTruthy();
		// The class carries the cell size, so the branch cannot change the metric.
		expect(md.TABLE_FONT_CLASS).toContain("text-body-sm");
	});

	it("pins the header weight to the class the header cell renders with (Q4-1)", () => {
		// The header font is BUILT from TABLE_HEAD_WEIGHT, so the constant and the
		// header cell's class must name the same weight: `font-semibold` is 600.
		expect(md.TABLE_HEAD_WEIGHT).toBe("600");
		const { container } = render(<md.Markdown text={"| a | b |\n|---|---|\n| 1 | 2 |"} />);
		expect(container.querySelector("th")?.className).toContain("font-semibold");
	});

	it("re-decides ONLY when the width changes — same width, no re-measure (C4-1)", () => {
		const observer = stubResizeObserver();
		const { measureText, setWidth } = forceMeasure(120);
		render(<md.Markdown text={"| chrome | content |\n|---|---|\n| 96 px | 472 px |"} />);
		expect((screen.getByRole("table") as HTMLElement).tagName).toBe("DIV");

		/* The observer firing for a HEIGHT change — which is exactly what switching
		   forms causes — must not re-measure: the verdict is width-derived. */
		const afterFirstDecide = measureText.mock.calls.length;
		expect(afterFirstDecide).toBeGreaterThan(0);
		for (let index = 0; index < 5; index += 1) act(() => observer.instances[0]?.callback());
		expect(measureText.mock.calls.length).toBe(afterFirstDecide);
		expect((screen.getByRole("table") as HTMLElement).tagName).toBe("DIV");

		/* A real width change DOES re-decide: wide enough now, so the table returns. */
		setWidth(400);
		act(() => observer.instances[0]?.callback());
		expect(measureText.mock.calls.length).toBeGreaterThan(afterFirstDecide);
		expect((screen.getByRole("table") as HTMLElement).tagName).toBe("TABLE");
	});

	it("keeps the header as part of the cell, so the value is not orphaned (C3-4)", () => {
		forceMeasure(120);
		render(<md.Markdown text={"| chrome | content |\n|---|---|\n| 96 px | 472 px |"} />);
		const cells = screen.getAllByRole("cell");
		expect(cells[0]?.textContent).toContain("chrome");
		expect(cells[0]?.textContent).toContain("96 px");
	});

	it("renders the same number of cells in both forms when a row is wider than its header (C3-5)", () => {
		/* Narrow enough that the columns cannot fit, so the STACKED form renders
		   (a row with 3 cells under a 1-cell header). The table form must agree. */
		forceMeasure(60);
		const { container } = render(<md.Markdown text={"| a |\n|---|\n| 1 | 2 | 3 |"} />);
		expect(container.querySelectorAll('[role="cell"]')).toHaveLength(3);
		/* No layout engine now: `clientWidth` is 0, the table form is kept, and it
		   must render the same three cells rather than dropping two. */
		for (const proto of [HTMLElement.prototype, Element.prototype]) {
			Object.defineProperty(proto, "clientWidth", { configurable: true, get: () => 0 });
		}
		const { container: tableForm } = render(<md.Markdown text={"| a |\n|---|\n| 1 | 2 | 3 |"} />);
		expect(tableForm.querySelectorAll("td")).toHaveLength(3);
		expect(tableForm.querySelectorAll("th")).toHaveLength(3);
	});
});
