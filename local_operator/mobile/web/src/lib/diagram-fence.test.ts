import { describe, expect, it } from "vitest";
import { diagramKey, shouldDrawDiagram } from "./diagram-fence";

const fence = (over: Partial<Parameters<typeof shouldDrawDiagram>[0]> = {}) => ({
	lang: "mermaid",
	closed: true,
	source: "flowchart TD\n  A-->B",
	...over,
});

describe("shouldDrawDiagram", () => {
	it("draws a closed mermaid fence", () => {
		expect(shouldDrawDiagram(fence())).toBe(true);
		expect(shouldDrawDiagram(fence({ lang: "Mermaid" }))).toBe(true);
	});
	it("keeps an OPEN fence (a reply still streaming) as code", () => {
		expect(shouldDrawDiagram(fence({ closed: false }))).toBe(false);
	});
	it("keeps other languages, no language and an empty body as code", () => {
		expect(shouldDrawDiagram(fence({ lang: "js" }))).toBe(false);
		expect(shouldDrawDiagram(fence({ lang: undefined }))).toBe(false);
		expect(shouldDrawDiagram(fence({ source: "  \n " }))).toBe(false);
	});
});

describe("diagramKey", () => {
	it("is stable for one source and differs by source and by repeat", () => {
		expect(diagramKey("a", 0)).toBe(diagramKey("a", 0));
		expect(diagramKey("a", 0)).not.toBe(diagramKey("b", 0));
		expect(diagramKey("a", 0)).not.toBe(diagramKey("a", 1));
	});
});
