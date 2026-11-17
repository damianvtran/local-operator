// @vitest-environment happy-dom
import { beforeEach, describe, expect, it, vi } from "vitest";
import { DIAGRAM_SAMPLES } from "../fixtures/diagram-samples";
import { WIDTH_STEP, bucketWidth, drawDiagram, peekDrawing, resetDiagramCache } from "./diagram-pele";

beforeEach(resetDiagramCache);

describe("drawDiagram", () => {
	it("draws a flowchart with Pele and caches it", async () => {
		const mode = { kind: "fit", width: 358 } as const;
		expect(peekDrawing(DIAGRAM_SAMPLES.flowchart, mode)).toBeUndefined();
		const first = await drawDiagram(DIAGRAM_SAMPLES.flowchart, mode);
		expect(first.ok && first.svg.startsWith("<svg")).toBe(true);
		expect(peekDrawing(DIAGRAM_SAMPLES.flowchart, mode)).toBe(first);
	});

	it("does not depend on the theme: a theme flip needs no redraw", async () => {
		const mode = { kind: "fit", width: 358 } as const;
		document.documentElement.dataset.theme = "localOperatorDark";
		const dark = await drawDiagram(DIAGRAM_SAMPLES.sequence, mode);
		resetDiagramCache();
		document.documentElement.dataset.theme = "localOperatorLight";
		const light = await drawDiagram(DIAGRAM_SAMPLES.sequence, mode);
		expect(light).toEqual(dark);
		expect(dark.ok && dark.svg).toContain("var(--pele-bg");
		expect(dark.ok && dark.svg).not.toMatch(/#(?:101010|181d27)/i);
	});

	it("buckets the width, and a wider bucket is a separate drawing", async () => {
		expect(bucketWidth(358)).toBe(352);
		expect(bucketWidth(359)).toBe(352);
		expect(bucketWidth(3)).toBe(WIDTH_STEP);
		await drawDiagram(DIAGRAM_SAMPLES.flowchart, { kind: "fit", width: 358 });
		expect(peekDrawing(DIAGRAM_SAMPLES.flowchart, { kind: "fit", width: 355 })).toBeDefined();
		expect(peekDrawing(DIAGRAM_SAMPLES.flowchart, { kind: "fit", width: 420 })).toBeUndefined();
		expect(peekDrawing(DIAGRAM_SAMPLES.flowchart, { kind: "full" })).toBeUndefined();
	});

	it("reflows a wide left-to-right chart to a vertical one on a phone and keeps it at full size", async () => {
		const wide = "flowchart LR\n A[Alpha service] --> B[Bravo service] --> C[Charlie service] --> D[Delta service] --> E[Echo service]";
		const fit = await drawDiagram(wide, { kind: "fit", width: 352 });
		const full = await drawDiagram(wide, { kind: "full" });
		expect(fit.ok && full.ok).toBe(true);
		if (fit.ok && full.ok) {
			expect(fit.width).toBeLessThanOrEqual(352);
			expect(full.width).toBeGreaterThan(fit.width);
			expect(full.width / full.height).toBeGreaterThan(fit.width / fit.height);
		}
	});

	it("reports a parse error as a result, caches it, and never throws", async () => {
		const bad = "flowchart TD\n A[Start --> B";
		const result = await drawDiagram(bad, { kind: "fit", width: 352 });
		expect(result.ok).toBe(false);
		if (!result.ok) {
			expect(result.message).toMatch(/error/i);
			expect(result.retryable).toBe(false);
		}
		expect(peekDrawing(bad, { kind: "fit", width: 352 })).toBe(result);
	});

	it("keeps links and images off", async () => {
		const src = 'flowchart LR\n A --> B\n click A "https://example.com" "t"\n C@{ img: "https://evil.example/p.png", label: "x", w: 40, h: 40 }';
		const result = await drawDiagram(src, { kind: "full" });
		expect(result.ok).toBe(true);
		if (result.ok) expect(result.svg).not.toMatch(/<a\b|<image\b|https:\/\/evil/);
	});

	it("keeps shrink-to-fit even when the source asks for a fixed size", async () => {
		const src = '%%{init: {"flowchart": {"useMaxWidth": false}}}%%\nflowchart TD\n A --> B';
		const result = await drawDiagram(src, { kind: "fit", width: 352 });
		expect(result.ok && result.svg).toContain("max-width:100%");
	});

	it("shares one render between identical requests", async () => {
		const [a, b] = await Promise.all([
			drawDiagram(DIAGRAM_SAMPLES.er, { kind: "fit", width: 352 }),
			drawDiagram(DIAGRAM_SAMPLES.er, { kind: "fit", width: 352 }),
		]);
		expect(a).toBe(b);
	});

	it("reports a renderer that cannot load as retryable and does not cache it", async () => {
		vi.resetModules();
		vi.doMock("pele/lazy", () => {
			throw new Error("Failed to fetch dynamically imported module");
		});
		const fresh = await import("./diagram-pele");
		const first = await fresh.drawDiagram("flowchart TD\n A --> B", { kind: "full" });
		expect(first).toMatchObject({ ok: false, retryable: true });
		expect(fresh.peekDrawing("flowchart TD\n A --> B", { kind: "full" })).toBeUndefined();
		vi.doUnmock("pele/lazy");
		vi.resetModules();
	});

	it("gives distinct drawings distinct SVG ids", async () => {
		const a = await drawDiagram(DIAGRAM_SAMPLES.sankey, { kind: "full" });
		const b = await drawDiagram(DIAGRAM_SAMPLES.sankey, { kind: "fit", width: 352 });
		const ids = (r: typeof a) => (r.ok ? Array.from(r.svg.matchAll(/ id="([^"]+)"/g)).map((m) => m[1]) : []);
		expect(ids(a).length).toBeGreaterThan(0);
		expect(ids(a).filter((id) => ids(b).includes(id))).toEqual([]);
	});
});
