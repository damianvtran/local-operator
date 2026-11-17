// @vitest-environment happy-dom
// A mermaid fence through the real Markdown renderer and the real Pele engine
// (lazy-imported, as in the app). Layout and pixels are the browser rig's job;
// this file pins the fence rules, the swap from code to drawing, the error
// fallback, the controls and that hostile labels never become elements.
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Markdown } from "./components/markdown";
import { DIAGRAM_SAMPLES } from "./fixtures/diagram-samples";
import { resetDiagramCache } from "./lib/diagram-pele";

const fenced = (source: string) => "```mermaid\n" + source + "\n```";

/* happy-dom has no layout: give the wrapper a column width so the width-keyed draw runs. */
beforeEach(() => {
	resetDiagramCache();
	document.documentElement.dataset.theme = "localOperatorDark";
	vi.spyOn(HTMLElement.prototype, "clientWidth", "get").mockReturnValue(358);
});
afterEach(() => {
	cleanup();
	vi.restoreAllMocks();
});

const drawn = (container: HTMLElement) => container.querySelector('[data-diagram-state="drawn"]');

describe("Markdown: mermaid fences", () => {
	it("keeps an OPEN fence (a reply still streaming) as the plain code block", async () => {
		const { container } = render(<Markdown text={"Here:\n\n```mermaid\nflowchart TD\n  A-->"} />);
		await act(async () => {});
		expect(container.querySelector("pre")?.textContent).toContain("A-->");
		expect(container.querySelector("[data-diagram-state]")).toBeNull();
		expect(container.querySelector("svg")).toBeNull();
	});

	it("draws a closed fence: code block first, then an SVG with the controls", async () => {
		const { container } = render(<Markdown text={fenced(DIAGRAM_SAMPLES.flowchart)} />);
		expect(container.querySelector('[data-diagram-state="pending"] pre')).not.toBeNull();
		await waitFor(() => expect(drawn(container)).not.toBeNull());
		const svg = container.querySelector("figure svg");
		expect(svg).not.toBeNull();
		expect(svg?.getAttribute("class")).toContain("pele-flowchart");
		expect(container.querySelector("figure")?.getAttribute("aria-label")).toBe("Diagram");
		expect(screen.getByTestId("diagram-copy").textContent).toContain("Copy source");
		expect(screen.getByTestId("diagram-save").textContent).toBe("Save SVG");
		expect(screen.getByTestId("diagram-full-size").textContent).toBe("Full size");
		/* the code block it replaced is gone, not stacked underneath */
		expect(container.querySelector("pre")).toBeNull();
	});

	it("themes the drawing with --pele-* variables that point at the app's variables", async () => {
		const { container } = render(<Markdown text={fenced(DIAGRAM_SAMPLES.sequence)} />);
		await waitFor(() => expect(drawn(container)).not.toBeNull());
		const style = (container.querySelector("figure") as HTMLElement).getAttribute("style") ?? "";
		expect(style).toContain("--pele-bg: var(--lo-surface)");
		expect(style).toContain("--pele-fg: var(--lo-ink)");
		expect(style).toContain("--pele-font: var(--font-sans)");
		/* colours are variables in the SVG too, so a theme flip needs no redraw */
		expect(container.querySelector("figure svg")?.getAttribute("style")).toContain("var(--pele-bg");
	});

	it("falls back to the code block with the engine's message visible when the source is invalid", async () => {
		const bad = "flowchart TD\n    A[Start --> B";
		const { container } = render(<Markdown text={fenced(bad)} />);
		await waitFor(() => expect(container.querySelector('[data-diagram-state="error"]')).not.toBeNull());
		const alert = screen.getByRole("alert");
		expect(alert.textContent).toContain("Couldn't draw this diagram");
		expect(alert.textContent).toMatch(/Parse error|Lexical error/i);
		expect(container.querySelector("pre:not([role])")?.textContent).toContain("A[Start --> B");
		expect(container.querySelector("svg")).toBeNull();
	});

	it("treats a hostile label as text: no img, script or handler ever reaches the DOM", async () => {
		const source = 'flowchart LR\n A["<img src=x onerror=alert(1)>"] --> B["<script>alert(2)</script>"]\n click A "javascript:alert(3)"';
		const { container } = render(<Markdown text={fenced(source)} />);
		await waitFor(() => expect(drawn(container)).not.toBeNull());
		expect(container.querySelector("img, script, iframe, foreignObject, a, image")).toBeNull();
		for (const el of Array.from(container.querySelectorAll("*"))) {
			for (const attr of Array.from(el.attributes)) expect(attr.name).not.toMatch(/^on|href/i);
		}
		/* Pele drops the tags from a label and keeps what is left as text. */
		expect(container.querySelector("svg")?.textContent).toContain("alert(2)");
	});

	it("does not redraw the diagram when the answer around it streams on", async () => {
		const head = fenced(DIAGRAM_SAMPLES.state);
		const { container, rerender } = render(<Markdown text={head} />);
		await waitFor(() => expect(drawn(container)).not.toBeNull());
		const svg = container.querySelector("figure svg");
		rerender(<Markdown text={head + "\n\nAnd then some more words."} />);
		expect(container.querySelector("figure svg")).toBe(svg);
	});

	it("draws two identical diagrams in one message separately", async () => {
		const one = fenced(DIAGRAM_SAMPLES.pie);
		const { container } = render(<Markdown text={`${one}\n\n${one}`} />);
		await waitFor(() => expect(container.querySelectorAll('[data-diagram-state="drawn"]').length).toBe(2));
	});

	it("leaves other code blocks alone", () => {
		const { container } = render(<Markdown text={"```js\nconst a = 1;\n```"} />);
		expect(container.querySelector("pre")?.textContent).toBe("const a = 1;");
		expect(container.querySelector("[data-diagram-state]")).toBeNull();
	});
});

describe("diagram controls", () => {
	it("copies the fence source, then says so and goes back", async () => {
		const writeText = vi.fn().mockResolvedValue(undefined);
		Object.defineProperty(navigator, "clipboard", { value: { writeText }, configurable: true });
		const { container } = render(<Markdown text={fenced(DIAGRAM_SAMPLES.pie)} />);
		await waitFor(() => expect(drawn(container)).not.toBeNull());
		await act(async () => fireEvent.click(screen.getByTestId("diagram-copy")));
		expect(writeText).toHaveBeenCalledWith(DIAGRAM_SAMPLES.pie);
		await waitFor(() => expect(screen.getByTestId("diagram-copy").textContent).toContain("Copied"));
	});

	it("reports a refused clipboard instead of pretending", async () => {
		Object.defineProperty(navigator, "clipboard", { value: { writeText: vi.fn().mockRejectedValue(new Error("no")) }, configurable: true });
		const { container } = render(<Markdown text={fenced(DIAGRAM_SAMPLES.pie)} />);
		await waitFor(() => expect(drawn(container)).not.toBeNull());
		await act(async () => fireEvent.click(screen.getByTestId("diagram-copy")));
		await waitFor(() => expect(screen.getByTestId("diagram-copy").textContent).toContain("Couldn't copy"));
	});

	it("opens a full-size viewer (modal, closable) on Full size and on a tap of the drawing", async () => {
		const { container } = render(<Markdown text={fenced(DIAGRAM_SAMPLES.flowchart)} />);
		await waitFor(() => expect(drawn(container)).not.toBeNull());
		for (const open of [() => fireEvent.click(screen.getByTestId("diagram-full-size")), () => fireEvent.click(screen.getByTestId("diagram-tap"))]) {
			await act(async () => open());
			const dialog = await screen.findByRole("dialog");
			expect(dialog.getAttribute("aria-modal")).toBe("true");
			await waitFor(() => expect(dialog.querySelector("svg")).not.toBeNull());
			/* full size keeps the author's pixel size: no shrink-to-fit cap */
			expect(document.body.contains(dialog)).toBe(true);
			await act(async () => fireEvent.click(screen.getByRole("button", { name: "Close diagram" })));
			expect(screen.queryByRole("dialog")).toBeNull();
		}
	});

	it("never captures a drag: only click handlers sit on the drawing", async () => {
		const { container } = render(<Markdown text={fenced(DIAGRAM_SAMPLES.flowchart)} />);
		await waitFor(() => expect(drawn(container)).not.toBeNull());
		const tap = screen.getByTestId("diagram-tap");
		const html = tap.outerHTML;
		expect(html).not.toMatch(/touch-none|touch-action|overscroll/);
		const scrolled = fireEvent.touchMove(tap);
		expect(scrolled).toBe(true); /* not preventDefault-ed */
		const wheel = fireEvent.wheel(tap, { deltaY: 120 });
		expect(wheel).toBe(true);
	});
});
