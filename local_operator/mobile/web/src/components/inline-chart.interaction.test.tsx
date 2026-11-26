// @vitest-environment happy-dom
/**
 * Interaction contract of the inline charts (inline-chart spec §5/§6): one
 * tab stop, arrow/Home/End/Escape driving one active mark with a live-region
 * announcement, hover and tap tooltips, the donut fold, and the collapsed
 * "Chart data" disclosure. happy-dom has no layout, so the chart lays out at
 * the 360px phone fallback width — the relay's common case.
 */
import { act, cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
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
		fireEvent.pointerDown(plot, { clientX: 2, clientY: 100, pointerType: "touch", pointerId: 1 });
		fireEvent.pointerUp(plot, { pointerType: "touch", pointerId: 1 });
		expect(container.querySelector("[data-chart-tooltip]")?.textContent).toContain("Nov 17");
		fireEvent.pointerDown(plot, { clientX: 2, clientY: 100, pointerType: "touch", pointerId: 2 });
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
