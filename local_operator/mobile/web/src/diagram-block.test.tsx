// @vitest-environment happy-dom
// A mermaid fence through the real Markdown renderer and the real Pele engine
// (lazy-imported, as in the app). Layout and pixels are the browser rig's job;
// this file pins the fence rules, the swap from code to drawing, the error
// fallback, the controls and that hostile labels never become elements.
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
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
	it("retains the open duplicate diagram and its controls after preceding prose is corrected", async () => {
		const source = fenced(DIAGRAM_SAMPLES.flowchart);
		const text = (opening: string) => `${opening}\n\n${source}\n\n${source}`;
		const writeText = vi.fn().mockResolvedValue(undefined);
		Object.defineProperty(navigator, "clipboard", { value: { writeText }, configurable: true });
		const { container, rerender } = render(<Markdown text={text("Opening.")} />);
		await waitFor(() => expect(container.querySelectorAll('[data-diagram-state="drawn"]')).toHaveLength(2));
		const controls = screen.getAllByTestId("diagram-full-size");
		controls[1].focus();
		fireEvent.click(controls[1]);
		const dialog = await screen.findByRole("dialog");
		await waitFor(() => expect(dialog.querySelector("svg")).not.toBeNull());
		const focused = document.activeElement;

		rerender(<Markdown text={text("An unrelated corrected opening.")} />);

		expect(screen.getByRole("dialog")).toBe(dialog);
		expect(document.activeElement).toBe(focused);
		expect(screen.getAllByTestId("diagram-full-size")).toEqual(controls);
		fireEvent.keyDown(dialog, { key: "Escape" });
		await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
		const copies = screen.getAllByTestId("diagram-copy");
		await act(async () => fireEvent.click(copies[1]));
		expect(writeText).toHaveBeenCalledWith(DIAGRAM_SAMPLES.flowchart);
		expect(copies[1].querySelector("span span:not([aria-hidden])")?.textContent).toBe("Copied");
		expect(copies[0].querySelector("span span:not([aria-hidden])")?.textContent).toBe("Copy source");
	});

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

	it("never cancels a touch, pointer or wheel gesture on the drawing, so the transcript keeps scrolling", async () => {
		const { container } = render(<Markdown text={fenced(DIAGRAM_SAMPLES.flowchart)} />);
		await waitFor(() => expect(drawn(container)).not.toBeNull());
		const tap = screen.getByTestId("diagram-tap");
		const svg = container.querySelector("figure svg")!;
		for (const target of [tap, svg]) {
			for (const type of ["touchstart", "touchmove", "touchend", "pointerdown", "pointermove", "wheel", "mousedown", "mousemove"]) {
				const event = new Event(type, { bubbles: true, cancelable: true });
				target.dispatchEvent(event);
				expect(event.defaultPrevented, `${type} on ${target.localName}`).toBe(false);
			}
		}
		expect(tap.outerHTML).not.toMatch(/touch-none|touch-action|overscroll/);
	});

	it("opens the viewer at the author's pixel size: no shrink-to-fit cap on the drawing", async () => {
		const { container } = render(<Markdown text={fenced(DIAGRAM_SAMPLES.flowchart)} />);
		await waitFor(() => expect(drawn(container)).not.toBeNull());
		await act(async () => fireEvent.click(screen.getByTestId("diagram-full-size")));
		const dialog = await screen.findByRole("dialog");
		await waitFor(() => expect(dialog.querySelector("svg")).not.toBeNull());
		const host = dialog.querySelector("svg")!.parentElement!;
		expect(host.className).toContain("max-w-none");
		expect(host.className).toContain("w-max");
		expect(dialog.querySelector('[data-testid="diagram-viewer-scroll"]')!.className).toContain("overflow-auto");
	});
});

describe("diagram block edge cases", () => {
	it("draws nothing while the column is hidden (0 wide), then draws when it shows", async () => {
		vi.spyOn(HTMLElement.prototype, "clientWidth", "get").mockReturnValue(0);
		const { container } = render(<Markdown text={fenced(DIAGRAM_SAMPLES.pie)} />);
		await act(async () => {});
		expect(container.querySelector('[data-diagram-state="pending"]')).not.toBeNull();
		expect(container.querySelector("svg")).toBeNull();
	});

	it("keeps the code block, with no error card, when the renderer chunk cannot be fetched", async () => {
		vi.resetModules();
		vi.doMock("pele/lazy", () => {
			throw new Error("Failed to fetch dynamically imported module");
		});
		const { Markdown: Fresh } = await import("./components/markdown");
		const { container } = render(<Fresh text={fenced(DIAGRAM_SAMPLES.pie)} />);
		await act(async () => {});
		await act(async () => {});
		expect(container.querySelector('[data-diagram-state="pending"] pre')).not.toBeNull();
		expect(screen.queryByRole("alert")).toBeNull();
		vi.doUnmock("pele/lazy");
		vi.resetModules();
	});

	it("shortens a very long engine message in the error card", async () => {
		const bad = `flowchart TD\n    A[${"x".repeat(3000)} --> B`;
		const { container } = render(<Markdown text={fenced(bad)} />);
		await waitFor(() => expect(container.querySelector('[data-diagram-state="error"]')).not.toBeNull());
		const message = screen.getByRole("alert").querySelector("pre")!.textContent!;
		expect(message.length).toBeLessThanOrEqual(401);
	});

	it("says the row changed height again when a new width redraws it (R3 F2)", async () => {
		const observers: ResizeObserverCallback[] = [];
		vi.stubGlobal(
			"ResizeObserver",
			class {
				constructor(cb: ResizeObserverCallback) {
					observers.push(cb);
				}
				observe() {}
				unobserve() {}
				disconnect() {}
			},
		);
		const heard = vi.fn();
		const { container } = render(
			<div ref={(el) => el?.addEventListener("lo:content-resized", heard)}>
				<Markdown text={fenced(DIAGRAM_SAMPLES.flowchart)} />
			</div>,
		);
		await waitFor(() => expect(drawn(container)).not.toBeNull());
		expect(heard).toHaveBeenCalledTimes(1);
		/* a rotation: the column narrows into another width bucket */
		vi.spyOn(HTMLElement.prototype, "clientWidth", "get").mockReturnValue(250);
		await act(async () => {
			for (const cb of [...observers]) cb([], {} as ResizeObserver);
		});
		await waitFor(() => expect(heard).toHaveBeenCalledTimes(2));
		vi.unstubAllGlobals();
	});

	it("titles an unusable drawing with the localized heading alone, no app-authored English (R3 F5)", async () => {
		/* The engine's SVG parses to something that is not an <svg> root. */
		const real = DOMParser.prototype.parseFromString;
		vi.spyOn(DOMParser.prototype, "parseFromString").mockImplementation(function (this: DOMParser, _s: string, type: DOMParserSupportedType) {
			return real.call(this, "<html/>", type === "image/svg+xml" ? "text/html" : type);
		});
		const { container } = render(<Markdown text={fenced(DIAGRAM_SAMPLES.pie)} />);
		await waitFor(() => expect(container.querySelector('[data-diagram-state="error"]')).not.toBeNull());
		const alert = screen.getByRole("alert");
		expect(alert.textContent).toBe("Couldn't draw this diagram");
		expect(alert.querySelector("pre")).toBeNull();
	});

	it("says the row changed height, once, when the drawing replaces the code block", async () => {
		const heard = vi.fn();
		const { container } = render(
			<div onClick={undefined} ref={(el) => el?.addEventListener("lo:content-resized", heard)}>
				<Markdown text={fenced(DIAGRAM_SAMPLES.state)} />
			</div>,
		);
		await waitFor(() => expect(drawn(container)).not.toBeNull());
		expect(heard).toHaveBeenCalledTimes(1);
	});

	it("reserves room for the longest copy label so the row does not reflow when copying fails", async () => {
		const { container } = render(<Markdown text={fenced(DIAGRAM_SAMPLES.pie)} />);
		await waitFor(() => expect(drawn(container)).not.toBeNull());
		const cell = screen.getByTestId("diagram-copy").querySelector(".grid")!;
		const hidden = Array.from(cell.querySelectorAll('[aria-hidden="true"]')).map((n) => n.textContent);
		expect(hidden.sort()).toEqual(["Copied", "Couldn't copy"]);
	});
});

describe("full-size viewer while its own draw is pending or fails (R3 F3)", () => {
	it("shows the source while pending, then the error card, never an empty pane", async () => {
		let settle: (r: unknown) => void = () => {};
		vi.resetModules();
		vi.doMock("./lib/diagram-pele", async (importOriginal) => {
			const real = await importOriginal<typeof import("./lib/diagram-pele")>();
			return {
				...real,
				peekDrawing: (source: string, mode: { kind: string }) =>
					mode.kind === "full" ? undefined : real.peekDrawing(source, mode as never),
				drawDiagram: (source: string, mode: { kind: string }) =>
					mode.kind === "full"
						? new Promise((resolve) => {
								settle = resolve;
							})
						: real.drawDiagram(source, mode as never),
			};
		});
		const { Markdown: Fresh } = await import("./components/markdown");
		const { container } = render(<Fresh text={fenced(DIAGRAM_SAMPLES.flowchart)} />);
		await waitFor(() => expect(drawn(container)).not.toBeNull());
		await act(async () => fireEvent.click(screen.getByTestId("diagram-full-size")));
		const pane = within(await screen.findByRole("dialog")).getByTestId("diagram-viewer-scroll");
		expect(pane.querySelector('[data-diagram-state="pending"] pre')?.textContent).toContain("flowchart");
		await act(async () => settle({ ok: false, message: "output limit exceeded", retryable: false }));
		expect(within(pane).getByRole("alert").textContent).toContain("output limit exceeded");
		expect(pane.querySelector("pre")).not.toBeNull();
		vi.doUnmock("./lib/diagram-pele");
		vi.resetModules();
	});
});

describe("viewer layout and focus rings (design R2 D2, D3)", () => {
	it("insets the focus ring so the figure and the viewport edge cannot clip it", async () => {
		const { container } = render(<Markdown text={fenced(DIAGRAM_SAMPLES.pie)} />);
		await waitFor(() => expect(drawn(container)).not.toBeNull());
		expect(screen.getByTestId("diagram-full-size").className).toContain("focus-visible:-outline-offset-2!");
		await act(async () => fireEvent.click(screen.getByTestId("diagram-full-size")));
		expect(screen.getByRole("button", { name: "Close diagram" }).className).toContain("focus-visible:-outline-offset-2!");
	});

	it("centres a short drawing vertically in the viewer without capping a tall one", async () => {
		const { container } = render(<Markdown text={fenced(DIAGRAM_SAMPLES.pie)} />);
		await waitFor(() => expect(drawn(container)).not.toBeNull());
		await act(async () => fireEvent.click(screen.getByTestId("diagram-full-size")));
		const dialog = await screen.findByRole("dialog");
		await waitFor(() => expect(dialog.querySelector("svg")).not.toBeNull());
		const host = dialog.querySelector("svg")!.parentElement!;
		expect(host.className).toContain("items-center");
		expect(host.className).toContain("min-h-full");
		expect(host.className).not.toMatch(/(^|\s)h-full/);
	});
});

describe("inline height clamp", () => {
	/* happy-dom has no layout: the drawing's natural height is stubbed, and the
	   clip box's clientHeight is what the browser would give it, the natural height
	   held to the inline max-height the component set (16 px root font). */
	function stubLayout(naturalPx: number) {
		vi.spyOn(HTMLElement.prototype, "scrollHeight", "get").mockReturnValue(naturalPx);
		vi.spyOn(HTMLElement.prototype, "clientHeight", "get").mockImplementation(function (this: HTMLElement) {
			const max = parseFloat(this.style.maxHeight) * 16;
			return Number.isNaN(max) ? naturalPx : Math.min(naturalPx, max);
		});
	}

	it("shows a small phone-width flowchart (542 px with padding) whole: no clip, no fade, no hint", async () => {
		stubLayout(542);
		const { container } = render(<Markdown text={fenced(DIAGRAM_SAMPLES.flowchart)} />);
		await waitFor(() => expect(drawn(container)).not.toBeNull());
		const tap = screen.getByTestId("diagram-tap");
		expect(parseFloat(tap.style.maxHeight) * 16).toBeGreaterThanOrEqual(542);
		expect(screen.queryByTestId("diagram-fade")).toBeNull();
		expect(screen.queryByTestId("diagram-cut-hint")).toBeNull();
	});

	it("holds a genuinely tall drawing at the cap behind the fade, with a labelled hint, and still opens full size", async () => {
		stubLayout(3000);
		const { container } = render(<Markdown text={fenced(DIAGRAM_SAMPLES.flowchart)} />);
		await waitFor(() => expect(drawn(container)).not.toBeNull());
		expect(screen.getByTestId("diagram-tap").style.maxHeight).toBe("40rem");
		expect(screen.getByTestId("diagram-fade")).not.toBeNull();
		const hint = screen.getByTestId("diagram-cut-hint");
		expect(hint.textContent).toBe("Full size");
		/* decorative: opening stays with the tap area and the footer button */
		expect(hint.closest("button")).toBeNull();
		await act(async () => fireEvent.click(screen.getByTestId("diagram-tap")));
		expect(await screen.findByRole("dialog")).not.toBeNull();
	});
});
