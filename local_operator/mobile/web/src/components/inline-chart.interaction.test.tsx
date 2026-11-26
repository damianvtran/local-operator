// @vitest-environment happy-dom
/**
 * Interaction contract of the inline charts (inline-chart spec §5/§6): one
 * tab stop, arrow/Home/End/Escape driving one active mark with a live-region
 * announcement, hover and tap tooltips, the donut fold, and the collapsed
 * "Chart data" disclosure. happy-dom has no layout, so the chart lays out at
 * the 360px phone fallback width — the relay's common case.
 */
import { act, cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { InlineUi } from "../inline-ui-parts";
import { AssistantContent } from "./inline-ui";

afterEach(cleanup);

const show = (value: InlineUi) => render(<AssistantContent text="" overlay={[{ kind: "ui", value }]} />);
const spend: InlineUi = { type: "line", title: "Spend", unit: "USD", data: [["Nov 17", 84.04], ["Nov 18 · rollout", 140.61], ["Nov 19", 153.66]] };
const live = (chart: HTMLElement) => chart.querySelector("[aria-live]")!.textContent;
const plotOf = (container: HTMLElement) => container.querySelector("[data-inline-plot]")!.parentElement as HTMLElement;

describe("inline chart interaction", () => {
	it("collapses the exact values behind a Chart data disclosure", () => {
		show(spend);
		const toggle = screen.getByRole("button", { name: "Chart data" });
		expect(toggle.getAttribute("aria-expanded")).toBe("false");
		expect(screen.queryByRole("table")).toBeNull();
		const chart = screen.getByRole("group", { name: "Spend" });
		// The hidden table still describes the chart for screen readers.
		expect(chart.getAttribute("aria-describedby")).toBe(toggle.getAttribute("aria-controls"));
		fireEvent.click(toggle);
		expect(toggle.getAttribute("aria-expanded")).toBe("true");
		const table = screen.getByRole("table", { name: "Spend" });
		expect(within(table).getByRole("columnheader", { name: "USD" })).toBeTruthy();
		expect(within(table).getByRole("rowheader", { name: "Nov 18 · rollout" })).toBeTruthy();
		expect(within(table).getByRole("cell", { name: "153.66" })).toBeTruthy();
	});

	it("is one tab stop whose arrows, Home, End and Escape move the active mark", () => {
		const { container } = show(spend);
		const chart = screen.getByRole("group", { name: "Spend" });
		expect(chart.getAttribute("tabindex")).toBe("0");
		expect(chart.getAttribute("aria-roledescription")).toBe("chart");
		// The only other focusable thing is the disclosure button, outside the group.
		expect(chart.querySelectorAll("[tabindex], button, a")).toHaveLength(0);
		act(() => chart.focus());
		fireEvent.keyDown(chart, { key: "ArrowRight" });
		// The axis shows the label before " · "; the announcement keeps it whole.
		expect(live(chart)).toBe("Nov 18 · rollout, 140.61 USD, 2 of 3");
		expect(container.querySelector("[data-chart-tooltip]")?.textContent).toContain("Nov 18 · rollout");
		fireEvent.keyDown(chart, { key: "End" });
		expect(live(chart)).toBe("Nov 19, 153.66 USD, 3 of 3");
		fireEvent.keyDown(chart, { key: "Home" });
		expect(live(chart)).toBe("Nov 17, 84.04 USD, 1 of 3");
		fireEvent.keyDown(chart, { key: "Escape" });
		expect(live(chart)).toBe("");
		expect(container.querySelector("[data-chart-tooltip]")).toBeNull();
		const axis = [...container.querySelectorAll("[data-inline-plot] text")].map((node) => node.textContent);
		expect(axis).toContain("Nov 18");
		expect(axis).not.toContain("Nov 18 · rollout");
	});

	it("walks series with up/down in a grouped chart", () => {
		show({ type: "series", title: "Mix", mode: "grouped", unit: "h", labels: ["Mon", "Tue"],
			series: [{ label: "Build", values: [3, 4] }, { label: "Test", values: [1, 2] }] });
		const chart = screen.getByRole("group", { name: "Mix" });
		act(() => chart.focus());
		fireEvent.keyDown(chart, { key: "ArrowRight" });
		fireEvent.keyDown(chart, { key: "ArrowDown" });
		expect(live(chart)).toBe("Tue, Test, 2 h, 2 of 2");
		fireEvent.keyDown(chart, { key: "ArrowUp" });
		expect(live(chart)).toBe("Tue, Build, 4 h, 2 of 2");
	});

	it("shows a borderless, pointer-transparent tooltip on hover and dims the rest", () => {
		const { container } = show({ type: "bar", title: "Q", unit: "USD", data: [["A", 12000], ["B", 3], ["C", 7]] });
		const plot = plotOf(container);
		// Slot 0 of three equal columns sits at the far left of the plot.
		fireEvent.pointerMove(plot, { clientX: 60, clientY: 100, pointerType: "mouse" });
		const tooltip = container.querySelector<HTMLElement>("[data-chart-tooltip]")!;
		expect(tooltip.textContent).toContain("A");
		// Ticks compact from 10 000; the tooltip keeps the full grouped value.
		expect(tooltip.textContent).toContain("12,000 USD");
		expect(container.querySelector("[data-inline-plot]")!.textContent).toContain("15K");
		expect(tooltip.className).toContain("pointer-events-none");
		expect(tooltip.className).not.toMatch(/\bborder\b/);
		const bars = container.querySelectorAll("[data-inline-plot] path");
		expect(bars[0].getAttribute("class")).not.toContain("opacity-35");
		expect(bars[1].getAttribute("class")).toContain("opacity-35");
		fireEvent.pointerLeave(plot, { pointerType: "mouse" });
		expect(container.querySelector("[data-chart-tooltip]")).toBeNull();
	});

	it("pins a tooltip on tap, keeps vertical scroll, and dismisses on a second tap", () => {
		const { container } = show(spend);
		const plot = plotOf(container);
		expect(plot.style.touchAction).toBe("pan-y");
		const tap = (id: number, x = 2, y = 100) => {
			fireEvent.pointerDown(plot, { clientX: x, clientY: y, pointerType: "touch", pointerId: id });
			fireEvent.pointerUp(plot, { clientX: x + 3, clientY: y + 2, pointerType: "touch", pointerId: id });
		};
		// Decided on release (X7): a press alone shows nothing yet.
		fireEvent.pointerDown(plot, { clientX: 2, clientY: 100, pointerType: "touch", pointerId: 1 });
		expect(container.querySelector("[data-chart-tooltip]")).toBeNull();
		fireEvent.pointerUp(plot, { clientX: 4, clientY: 101, pointerType: "touch", pointerId: 1 });
		expect(container.querySelector("[data-chart-tooltip]")?.textContent).toContain("Nov 17");
		tap(2);
		expect(container.querySelector("[data-chart-tooltip]")).toBeNull();
		// Tapping outside the chart dismisses a pinned tooltip.
		tap(3);
		expect(container.querySelector("[data-chart-tooltip]")).not.toBeNull();
		fireEvent.pointerDown(document.body, { pointerType: "touch", pointerId: 4 });
		expect(container.querySelector("[data-chart-tooltip]")).toBeNull();
	});

	it("folds a 12-slice pie into 6 + Other (6) while the table keeps all 12", () => {
		const data = Array.from({ length: 12 }, (_, i) => [`Item ${i + 1}`, 12 - i] as [string, number]);
		const { container } = show({ type: "pie", title: "Share", data });
		expect(container.querySelectorAll('[data-inline-plot="pie"] path')).toHaveLength(7);
		const chart = screen.getByRole("group", { name: "Share" });
		expect(within(chart).getByText("Other (6)")).toBeTruthy();
		// Narrow: the centre readout is the tooltip (no floating box under a finger).
		fireEvent.keyDown(chart, { key: "End" });
		expect(container.querySelector("[data-chart-tooltip]")).toBeNull();
		expect(container.querySelector('[data-inline-plot="pie"]')!.textContent).toContain("Other (6)");
		expect(live(chart)).toContain("7 of 7");
		fireEvent.click(screen.getByRole("button", { name: "Chart data" }));
		expect(within(screen.getByRole("table", { name: "Share" })).getAllByRole("rowheader")).toHaveLength(12);
	});

	it("draws the SVG at its real pixel size with the 180px phone plot", () => {
		const { container } = show(spend);
		const svg = container.querySelector("[data-inline-plot]")!;
		expect(svg.getAttribute("viewBox")).toBe(`0 0 ${svg.getAttribute("width")} ${svg.getAttribute("height")}`);
		expect(svg.getAttribute("preserveAspectRatio")).toBeNull();
		expect(Number(svg.getAttribute("height"))).toBe(12 + 180 + 24);
	});
});

/**
 * Round-1 remediation regressions (PR #4631). Each block names its ledger ID.
 * Layout stubs stand in for what happy-dom lacks: a chart width (the chart
 * block's getBoundingClientRect), a canvas whose measureText the renderer
 * really calls, and a tooltip box size.
 */
describe("inline chart remediation round 1", () => {
	const CHAR = 7; // px per glyph from the stubbed canvas
	const measured = (text: string) => Array.from(text).length * CHAR;
	let width = 360;
	let measureCalls = 0;
	const restore: (() => void)[] = [];
	beforeEach(() => {
		width = 360;
		measureCalls = 0;
		const rect = vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockImplementation(() => (
			{ width, height: 0, left: 0, top: 0, right: width, bottom: 0, x: 0, y: 0, toJSON: () => ({}) } as DOMRect));
		const context = {
			font: "",
			measureText: (text: string) => { measureCalls++; return { width: measured(text) } as TextMetrics; },
		};
		const canvas = vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockImplementation((() => context) as never);
		const tipSize = (size: number) => function (this: HTMLElement) { return this.hasAttribute("data-chart-tooltip") ? size : 0; };
		const ow = vi.spyOn(HTMLElement.prototype, "offsetWidth", "get").mockImplementation(tipSize(120));
		const oh = vi.spyOn(HTMLElement.prototype, "offsetHeight", "get").mockImplementation(tipSize(60));
		restore.push(() => { rect.mockRestore(); canvas.mockRestore(); ow.mockRestore(); oh.mockRestore(); });
	});
	afterEach(() => {
		while (restore.length) restore.pop()!();
		document.documentElement.removeAttribute("dir");
	});
	const tooltip = (container: HTMLElement) => container.querySelector<HTMLElement>("[data-chart-tooltip]");
	const px = (value: string | null | undefined) => Number.parseFloat(value ?? "NaN");

	it("R3: measures label widths with the canvas after mount", () => {
		const { container } = show({ type: "line", title: "Spend", unit: "USD", data: [["Nov 16", 153.66], ["Nov 20", 101.2], ["Nov 24", 84.04]] });
		expect(measureCalls).toBeGreaterThan(0);
		// Widest tick "160" = 3 × 7 = 21px → gutter max(28, 21 + 8) = 29 → ticks at x 21.
		// The 6.6 px/char estimate would give gutter 28 and x 20.
		const ticks = [...container.querySelectorAll('[data-inline-plot] text[text-anchor="end"]:not([data-x-label])')];
		expect(ticks.map((node) => node.textContent)).toEqual(["80", "100", "120", "140", "160"]);
		expect(ticks.every((node) => node.getAttribute("x") === "21")).toBe(true);
	});

	it.each([360, 760])("X2: 32 long x labels at %ipx render disjoint, first and last shown", (w) => {
		width = w;
		for (const make of [(i: number) => `Week ${i + 1} of the quarterly reporting period`, (i: number) => `2025-11-16 ${String(i % 24).padStart(2, "0")}:00 UTC`, (i: number) => `Nov ${i + 1}`]) {
			const { container, unmount } = show({ type: "line", title: "Long", data: Array.from({ length: 32 }, (_, i) => [make(i), i * 3 + 10] as [string, number]) });
			const labels = [...container.querySelectorAll<SVGTextElement>("[data-x-label]")];
			const boxes = labels.map((node) => {
				const x = px(node.getAttribute("x"));
				const w2 = measured(node.textContent ?? "");
				const anchor = node.getAttribute("text-anchor");
				const left = anchor === "start" ? x : anchor === "end" ? x - w2 : x - w2 / 2;
				return [left, left + w2];
			});
			expect(labels[0].getAttribute("data-x-label")).toBe("0");
			expect(labels[labels.length - 1].getAttribute("data-x-label")).toBe("31");
			for (const [left, right] of boxes) {
				expect(left).toBeGreaterThanOrEqual(-0.5);
				expect(right).toBeLessThanOrEqual(w + 0.5);
			}
			for (let i = 1; i < boxes.length; i++) expect(boxes[i][0] - boxes[i - 1][1]).toBeGreaterThanOrEqual(4 - 1e-6);
			unmount();
		}
	});

	it("R1: up/down reach every series of a dual chart, right axis included", () => {
		// Dual is the relay schema's only line-like multi-series mode.
		show({ type: "series", title: "Dual", mode: "dual", unit: "req", rightUnit: "ms", labels: ["a", "b"],
			series: [{ label: "Req", values: [100, 200], axis: "left" }, { label: "Errors", values: [3, 4], axis: "left" },
				{ label: "Latency", values: [250, 260], axis: "right" }] });
		const dual = screen.getByRole("group", { name: "Dual" });
		act(() => dual.focus());
		fireEvent.keyDown(dual, { key: "ArrowRight" });
		expect(live(dual)).toBe("b, Req, 200 req, 2 of 2");
		fireEvent.keyDown(dual, { key: "ArrowDown" });
		expect(live(dual)).toBe("b, Errors, 4 req, 2 of 2");
		fireEvent.keyDown(dual, { key: "ArrowDown" });
		expect(live(dual)).toBe("b, Latency · right axis, 260 ms, 2 of 2");
		expect(dual.querySelector("[data-active-series]")).not.toBeNull();
		fireEvent.keyDown(dual, { key: "ArrowUp" });
		expect(live(dual)).toBe("b, Errors, 4 req, 2 of 2");
	});

	it("X8: keeps the chart LTR under an RTL document", () => {
		document.documentElement.setAttribute("dir", "rtl");
		const { container } = show(spend);
		const chart = screen.getByRole("group", { name: "Spend" });
		expect(chart.getAttribute("dir")).toBe("ltr");
		expect(chart.style.direction).toBe("ltr");
		const svg = container.querySelector("[data-inline-plot]")!;
		expect(svg.getAttribute("direction")).toBe("ltr");
		expect(getComputedStyle(svg).direction).toBe("ltr");
	});

	it("X7: a press that turns into a scroll, or is cancelled, leaves nothing pinned or dimmed", () => {
		const { container } = show({ type: "bar", title: "Q", data: [["A", 4], ["B", 3], ["C", 7]] });
		const plot = plotOf(container);
		const dimmed = () => container.querySelectorAll("[data-inline-plot] .opacity-35").length;
		fireEvent.pointerDown(plot, { clientX: 60, clientY: 100, pointerType: "touch", pointerId: 1 });
		fireEvent.pointerCancel(plot, { pointerType: "touch", pointerId: 1 });
		expect(tooltip(container)).toBeNull();
		expect(dimmed()).toBe(0);
		fireEvent.pointerDown(plot, { clientX: 60, clientY: 100, pointerType: "touch", pointerId: 2 });
		fireEvent.pointerMove(plot, { clientX: 61, clientY: 120, pointerType: "touch", pointerId: 2 });
		fireEvent.pointerUp(plot, { clientX: 61, clientY: 140, pointerType: "touch", pointerId: 2 });
		expect(tooltip(container)).toBeNull();
		// A pinned tooltip is cleared when the next gesture becomes a scroll.
		fireEvent.pointerDown(plot, { clientX: 60, clientY: 100, pointerType: "touch", pointerId: 3 });
		fireEvent.pointerUp(plot, { clientX: 60, clientY: 100, pointerType: "touch", pointerId: 3 });
		expect(tooltip(container)).not.toBeNull();
		expect(dimmed()).toBeGreaterThan(0);
		fireEvent.pointerDown(plot, { clientX: 60, clientY: 100, pointerType: "touch", pointerId: 4 });
		fireEvent.pointerCancel(plot, { pointerType: "touch", pointerId: 4 });
		expect(tooltip(container)).toBeNull();
		expect(dimmed()).toBe(0);
	});

	it("X7: a horizontal drag on a line scrubs and stays pinned after release", () => {
		const { container } = show(spend);
		const plot = plotOf(container);
		fireEvent.pointerDown(plot, { clientX: 30, clientY: 100, pointerType: "touch", pointerId: 1 });
		fireEvent.pointerMove(plot, { clientX: 350, clientY: 102, pointerType: "touch", pointerId: 1 });
		expect(tooltip(container)?.textContent).toContain("Nov 19");
		fireEvent.pointerUp(plot, { clientX: 350, clientY: 102, pointerType: "touch", pointerId: 1 });
		expect(tooltip(container)?.textContent).toContain("Nov 19");
	});

	it("U3: after Escape, the first arrow lands on mark 1 of n", () => {
		show(spend);
		const chart = screen.getByRole("group", { name: "Spend" });
		act(() => chart.focus());
		fireEvent.keyDown(chart, { key: "End" });
		fireEvent.keyDown(chart, { key: "Escape" });
		fireEvent.keyDown(chart, { key: "ArrowRight" });
		expect(live(chart)).toBe("Nov 17, 84.04 USD, 1 of 3");
	});

	it("U2 + X9: tapping a narrow legend row pins its slice into the centre readout, whole percents and <1%", () => {
		const { container } = show({ type: "pie", title: "Mix", data: [["A", 700], ["B", 296], ["C", 4]] });
		const pie = container.querySelector('[data-inline-plot="pie"]')!;
		const rows = [...container.querySelectorAll("[data-pie-legend]")];
		expect(rows.map((row) => row.lastElementChild?.textContent)).toEqual(["70%", "30%", "<1%"]);
		fireEvent.click(rows[2]);
		expect(pie.textContent).toContain("C");
		expect(pie.textContent).toContain("<1%");
		const chart = screen.getByRole("group", { name: "Mix" });
		expect(live(chart)).toBe("C, 4, <1%, 3 of 3");
		// The finger lifting fires pointerleave: the pin must survive it.
		fireEvent.pointerLeave(rows[2], { pointerType: "touch" });
		expect(live(chart)).toBe("C, 4, <1%, 3 of 3");
		fireEvent.click(rows[2]);
		expect(pie.textContent).toContain("Total");
	});

	it("X9 + U2: wide donut labels and the Also: list print whole percents; an Also item pins", () => {
		width = 760;
		const { container } = show({ type: "pie", title: "Wide", data: [["A", 700], ["B", 296], ["C", 4]] });
		const pie = container.querySelector('[data-inline-plot="pie"]')!;
		expect(pie.textContent).toContain("70%");
		expect(pie.textContent).not.toMatch(/\d\.\d%/);
		const also = container.querySelector("[data-pie-also]")!;
		expect(also.textContent).toContain("<1%");
		fireEvent.click(also);
		expect(live(screen.getByRole("group", { name: "Wide" }))).toBe("C, 4, <1%, 3 of 3");
	});

	it("X5: a wide donut's tooltip sits beside the ring on the slice's side", () => {
		width = 760;
		const { container } = show({ type: "pie", title: "Side", data: [["A", 50], ["B", 30], ["C", 20]] });
		const chart = screen.getByRole("group", { name: "Side" });
		act(() => chart.focus()); // slice A, centred on 3 o'clock
		const cx = 380;
		const R = 100;
		expect(px(tooltip(container)!.style.left)).toBeGreaterThanOrEqual(cx + R + 8);
		fireEvent.keyDown(chart, { key: "End" }); // slice C, left half
		const tip = tooltip(container)!;
		expect(px(tip.style.left) + 120).toBeLessThanOrEqual(cx - R - 8);
		const top = px(tip.style.top);
		expect(top).toBeGreaterThanOrEqual(0);
		expect(top + 60).toBeLessThanOrEqual(200 + 24);
	});

	it("X4: stacked and percent tooltips stay inside the plot, above the x labels", () => {
		for (const mode of ["percent", "stacked"] as const) {
			const { container, unmount } = show({ type: "series", title: mode, mode, labels: ["Q1", "Q2"],
				series: [{ label: "A", values: [30, 10] }, { label: "B", values: [50, 70] }, { label: "C", values: [20, 20] }] });
			const chart = screen.getByRole("group", { name: mode });
			act(() => chart.focus());
			for (let s = 0; s < 3; s++) {
				const top = px(tooltip(container)!.style.top);
				expect(top).toBeGreaterThanOrEqual(0);
				expect(top + 60).toBeLessThanOrEqual(12 + 180);
				fireEvent.keyDown(chart, { key: "ArrowDown" });
			}
			unmount();
		}
	});

	it("X6: a mixed-sign stack keeps Total but drops the share; a same-sign stack keeps both", () => {
		const { container } = show({ type: "series", title: "Net", mode: "stacked", unit: "USD", labels: ["Q1", "Q2"],
			series: [{ label: "Revenue", values: [70, 40] }, { label: "Refunds", values: [-18, 12] }] });
		const chart = screen.getByRole("group", { name: "Net" });
		act(() => chart.focus());
		expect(tooltip(container)!.textContent).toContain("Total");
		expect(tooltip(container)!.textContent).not.toContain("of total");
		fireEvent.keyDown(chart, { key: "ArrowRight" });
		expect(tooltip(container)!.textContent).toContain("Total");
		expect(tooltip(container)!.textContent).toContain("76.9% of total");
	});

	it("X10: the heatmap's lowest cell and legend start at the shared 28% floor", () => {
		const { container } = show({ type: "heatmap", title: "Heat", columns: ["Mon", "Tue"], rowLabels: ["AM"], rows: [[0, 10]] } as InlineUi);
		const cells = [...container.querySelectorAll<SVGRectElement>('[data-inline-plot="heatmap"] rect')];
		expect(cells[0].style.fill).toContain("var(--lo-accent) 28%");
		expect(cells[1].style.fill).toContain("var(--lo-accent) 100%");
	});

	it("X11 + R7: sub-1e-20 values render unique scientific ticks without duplicate keys", () => {
		const errors = vi.spyOn(console, "error").mockImplementation(() => {});
		try {
			const { container } = show({ type: "line", title: "Tiny", data: [["a", 1e-21], ["b", 2e-21], ["c", 3e-21]] });
			const ticks = [...container.querySelectorAll('[data-inline-plot] text[text-anchor="end"]:not([data-x-label])')].map((node) => node.textContent);
			expect(new Set(ticks).size).toBe(ticks.length);
			expect(ticks.every((text) => /E-2\d$/.test(text ?? ""))).toBe(true);
			expect(errors.mock.calls.flat().join(" ")).not.toMatch(/same key/);
		} finally {
			errors.mockRestore();
		}
	});

	it("D5: horizontal-bar value ticks sit one per ~96px", () => {
		// 360px, 16 bars → horizontal; plot ≈ 291px → target 3 → 0, 100, 200
		// (the round-1 64px rule targeted 4 → 0, 50, …, 200 at ~73px apart).
		const { container } = show({ type: "bar", title: "Many", data: Array.from({ length: 16 }, (_, i) => [`Item ${i + 1}`, (i + 1) * 10] as [string, number]) });
		const xs = [...container.querySelectorAll('[data-inline-plot] text[text-anchor="middle"]')].map((node) => px(node.getAttribute("x")));
		expect(xs.length).toBeGreaterThanOrEqual(3);
		for (let i = 1; i < xs.length; i++) expect(xs[i] - xs[i - 1]).toBeGreaterThanOrEqual(95);
	});

	it.each([328, 360, 760])("C2: a 24-column hourly heatmap at %ipx keeps both edge labels whole inside the chart", (w) => {
		width = w;
		const columns = Array.from({ length: 24 }, (_, h) => `${String(h).padStart(2, "0")}:00`);
		const { container } = show({ type: "heatmap", title: "Hourly", columns, rowLabels: ["Mon", "Wednesday"], rows: [columns.map((_, i) => i), columns.map(() => null)] } as InlineUi);
		const labels = [...container.querySelectorAll<SVGTextElement>("[data-x-label]")];
		expect(labels[0].textContent).toBe("00:00");
		expect(labels.at(-1)!.textContent).toBe("23:00");
		for (const node of labels) {
			const text = node.textContent ?? "";
			expect(text === "…" || (text.endsWith("…") && Array.from(text).length < 4)).toBe(false);
			const x = px(node.getAttribute("x"));
			const half = measured(text) / 2;
			expect(node.getAttribute("text-anchor")).toBe("middle");
			expect(x - half).toBeGreaterThanOrEqual(-0.5);
			expect(x + half).toBeLessThanOrEqual(w + 0.5);
		}
	});

	it("R2-1: the Chart data table fits the column, and takes a focusable rail only when its whole numbers overflow", () => {
		// happy-dom has no table layout: stand in the two widths the frame reads.
		let tableWidth = 300;
		const ow = vi.spyOn(HTMLElement.prototype, "offsetWidth", "get").mockImplementation(function (this: HTMLElement) {
			return this.tagName === "TABLE" ? tableWidth : 0;
		});
		const cw = vi.spyOn(HTMLElement.prototype, "clientWidth", "get").mockImplementation(() => width);
		restore.push(() => { ow.mockRestore(); cw.mockRestore(); });
		const open = () => fireEvent.click(screen.getAllByRole("button", { name: "Chart data" }).at(-1)!);
		const { unmount } = show({ type: "bar", title: "Fit", data: [["a", 12], ["b", 1_234_567]] });
		open();
		const fit = document.getElementById(screen.getByRole("button", { name: "Chart data" }).getAttribute("aria-controls")!)!;
		expect(fit.className).not.toContain("overflow-x-auto");
		expect(fit.getAttribute("tabindex")).toBeNull();
		// Numbers never break mid-value (a wrapped `12` reads as `1` and `2`); text may.
		expect(screen.getByRole("cell", { name: "1,234,567" }).className).toContain("whitespace-nowrap");
		expect(screen.getByRole("rowheader", { name: "b" }).className).toContain("wrap-anywhere");
		unmount();
		tableWidth = 552;
		show({ type: "heatmap", title: "Wide", columns: Array.from({ length: 24 }, (_, h) => `${h}:00`), rowLabels: ["Mon"], rows: [Array.from({ length: 24 }, (_, h) => h * 10)] } as InlineUi);
		open();
		const rail = screen.getByRole("region", { name: "Wide" });
		expect(rail.className).toContain("overflow-x-auto");
		expect(rail.getAttribute("tabindex")).toBe("0");
	});

	it("U5 + U6: a finger on a legend row or Also item does not preview; a mouse does; both read as tappable", () => {
		const { container } = show({ type: "pie", title: "Mix", data: [["A", 700], ["B", 296], ["C", 4]] });
		const chart = screen.getByRole("group", { name: "Mix" });
		const row = container.querySelector<HTMLElement>('[data-pie-legend="2"]')!;
		fireEvent.pointerEnter(row, { pointerType: "touch" });
		expect(live(chart)).toBe("");
		expect(container.querySelectorAll(".opacity-35").length).toBe(0);
		fireEvent.pointerLeave(row, { pointerType: "touch" });
		fireEvent.pointerEnter(row, { pointerType: "mouse" });
		expect(live(chart)).toBe("C, 4, <1%, 3 of 3");
		fireEvent.pointerLeave(row, { pointerType: "mouse" });
		// ≥24px hit area: 17.4px row + py-1 (8px); the grid gap moved into the padding.
		expect(row.className).toMatch(/\bcursor-pointer\b.*\bpy-1\b|\bpy-1\b.*\bcursor-pointer\b/);
		expect(row.parentElement!.className).not.toContain("gap-y-2");
		cleanup();
		width = 760;
		const wide = show({ type: "pie", title: "Wide", data: [["A", 700], ["B", 296], ["C", 4]] });
		const also = wide.container.querySelector<HTMLElement>("[data-pie-also]")!;
		const wideChart = screen.getByRole("group", { name: "Wide" });
		fireEvent.pointerEnter(also, { pointerType: "touch" });
		expect(live(wideChart)).toBe("");
		fireEvent.click(also);
		expect(live(wideChart)).toBe("C, 4, <1%, 3 of 3");
		expect(also.className).toContain("cursor-pointer");
		expect(also.className).toContain("py-[5px]");
	});
});
