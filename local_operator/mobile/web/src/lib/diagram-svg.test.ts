// @vitest-environment happy-dom
// The SVG allow-list, run against Pele's real output. Pele is the producer, so
// the test for "does the sanitiser drop real drawing" and "does hostile source
// survive it" both start from `render()`, not from hand-written markup.
import { render } from "pele";
import { describe, expect, it } from "vitest";
import { DIAGRAM_SAMPLES } from "../fixtures/diagram-samples";
import { cleanStyle, sanitizeSvg, serializeSvg } from "./diagram-svg";

const OPTIONS = { links: false, images: false, idPrefix: "t" } as const;

/** Every element and attribute in a tree, as `tag` and `tag@attr=value` strings. */
function inventory(root: Element): { tags: string[]; attrs: string[] } {
	const tags: string[] = [];
	const attrs: string[] = [];
	for (const el of [root, ...Array.from(root.querySelectorAll("*"))]) {
		tags.push(el.localName);
		for (const a of Array.from(el.attributes)) attrs.push(`${el.localName}@${a.name}=${a.value}`);
	}
	return { tags, attrs };
}

function parse(svg: string): Element {
	return new DOMParser().parseFromString(svg, "image/svg+xml").documentElement;
}

const FORBIDDEN_TAGS = ["script", "foreignObject", "iframe", "image", "a", "style", "use", "object", "embed", "set", "animate"];

function assertInert(node: Element, label: string): void {
	const { tags, attrs } = inventory(node);
	for (const tag of FORBIDDEN_TAGS) expect(tags, `${label}: <${tag}>`).not.toContain(tag);
	for (const attr of attrs) {
		const [, name, value] = /^[^@]+@([^=]+)=([\s\S]*)$/.exec(attr)!;
		expect(name, `${label}: ${attr}`).not.toMatch(/^on/i);
		expect(name, `${label}: ${attr}`).not.toMatch(/href|src/i);
		expect(value, `${label}: ${attr}`).not.toMatch(/javascript:|vbscript:|data:/i);
		expect(value.replace(/url\(\s*["']?#[\w.:-]+["']?\)/g, ""), `${label}: ${attr}`).not.toMatch(/url\(/i);
	}
}

describe("sanitizeSvg on every diagram kind", () => {
	for (const [kind, source] of Object.entries(DIAGRAM_SAMPLES)) {
		it(`${kind}: keeps every element and every attribute value Pele wrote`, () => {
			const svg = render(source, OPTIONS).svg;
			const before = inventory(parse(svg));
			const node = sanitizeSvg(svg);
			expect(node).not.toBeNull();
			const after = inventory(node!);
			expect(after.tags).toEqual(before.tags);
			/* `xmlns` is re-created by createElementNS, and the style attribute is re-joined
			   (same declarations, no spaces) so it is compared as a set of declarations. */
			const plain = (list: string[]) => list.filter((a) => !/@(xmlns|style)=/.test(a));
			expect(plain(after.attrs)).toEqual(plain(before.attrs));
			const styles = (list: string[]) => list.filter((a) => /@style=/.test(a)).map((a) => a.replace(/\s/g, ""));
			expect(styles(after.attrs)).toEqual(styles(before.attrs));
		});
	}
});

/* ~30 hostile sources. Pele refuses some outright (a syntax error is a pass:
   nothing is drawn); every one it does draw must come through inert. */
const HOSTILE: Record<string, string> = {
	imgOnerror: 'flowchart LR\n A["<img src=x onerror=alert(1)>"] --> B',
	scriptTag: 'flowchart LR\n A["<script>alert(1)</script>"] --> B',
	svgOnload: 'flowchart LR\n A["</text><svg onload=alert(1)>"] --> B',
	jsLinkClick: 'flowchart LR\n A --> B\n click A "javascript:alert(1)" "t"',
	jsLinkHref: 'flowchart LR\n A --> B\n click A href "JaVaScRiPt:alert(1)"',
	dataLink: 'flowchart LR\n A --> B\n click A "data:text/html,<script>alert(1)</script>"',
	callback: 'flowchart LR\n A --> B\n click A callback "tip"\n click B call alert(1)',
	protoRelative: 'flowchart LR\n A --> B\n click A "//evil.example/x"',
	quoteBreak: "flowchart LR\n A[\"x' onmouseover='alert(1)\"] --> B",
	foreignObject: 'flowchart LR\n A["<foreignObject><body onload=alert(1)></foreignObject>"] --> B',
	markdownLabel: 'flowchart LR\n A["`<div onclick=alert(1)>x</div>`"] --> B',
	remoteImage: 'flowchart LR\n A@{ img: "https://evil.example/pixel.png", label: "x", pos: "t", w: 60, h: 60, constraint: "on" } --> B',
	icon: 'flowchart LR\n A@{ icon: "fa:fa-car", form: "square" } --> B',
	sequenceNames: 'sequenceDiagram\n participant A as <img src=x onerror=alert(1)>\n A->>B: <script>alert(1)</script>\n Note over A: " onload="alert(1)',
	sequenceLink: 'sequenceDiagram\n participant A\n link A: Dash @ javascript:alert(1)\n links A: {"Dash":"javascript:alert(1)"}',
	stateNote: 'stateDiagram-v2\n [*] --> S\n S: "onmouseover=alert(1) x"\n note right of S: <b onclick=alert(1)>x</b>',
	classLabel: 'classDiagram\n class A["<img src=x onerror=alert(1)>"]\n A : +x() <svg/onload=alert(1)>\n click A href "javascript:alert(1)"',
	frontmatter: "---\ntitle: <img src=x onerror=alert(1)>\n---\nflowchart LR\n A-->B",
	directive: '%%{init: {"securityLevel":"loose","themeVariables":{"primaryColor":"red\\" onload=\\"alert(1)"}}}%%\nflowchart LR\n A-->B',
	entity: 'flowchart LR\n A["&lt;img src=x onerror=alert(1)&gt; #quot;"] --> B',
	cdata: 'flowchart LR\n A["]]><svg onload=alert(1)>"] --> B',
	pie: 'pie title <img src=x onerror=alert(1)>\n "a\\" onload=\\"x" : 1\n "b" : 2',
	mindmap: 'mindmap\n root(("<img src=x onerror=alert(1)>"))\n  child["javascript:alert(1)"]',
	gantt: 'gantt\n title x\n section s\n t :done, a, 2024-01-01, 1d\n click a href "javascript:alert(1)"\n click a call alert(1)',
	styleUrl: "flowchart LR\n A --> B\n style A fill:url(https://evil.test/x)",
	styleFixed: "flowchart LR\n A --> B\n style A position:fixed,z-index:9999,display:none",
	linkStyleUrl: "flowchart LR\n A --> B\n linkStyle 0 stroke:url(https://evil/x)",
	classDefInject: 'flowchart LR\n A --> B\n classDef c fill:#f00" onload="alert(1)\n class A c',
	longLabel: `flowchart LR\n A["${"x".repeat(5000)}"] --> B`,
	rtl: 'flowchart LR\n A["\u202Eevil\u202C"] --> B',
};

describe("hostile sources", () => {
	let drawn = 0;
	let refused = 0;
	for (const [name, source] of Object.entries(HOSTILE)) {
		it(`${name}: refused by Pele, or drawn inert after sanitising`, () => {
			let svg: string;
			try {
				svg = render(source, OPTIONS).svg;
			} catch (error) {
				expect((error as Error).name).toBe("PeleError");
				refused++;
				return;
			}
			const node = sanitizeSvg(svg);
			expect(node).not.toBeNull();
			assertInert(node!, name);
			drawn++;
		});
	}
	it("exercised both paths", () => {
		expect(drawn).toBeGreaterThan(10);
		expect(refused).toBeGreaterThan(0);
	});
});

describe("sanitizeSvg on hand-written markup Pele would never emit", () => {
	const NS = 'xmlns="http://www.w3.org/2000/svg"';
	const run = (inner: string, rootAttrs = "") => sanitizeSvg(`<svg ${NS} ${rootAttrs}>${inner}</svg>`)!;

	it("drops script, foreignObject, style, a, image, use and animation elements, keeping siblings", () => {
		const node = run(
			'<script>alert(1)</script><foreignObject><div/></foreignObject><style>*{}</style><a href="javascript:x"><text>kept?</text></a>' +
				'<image href="https://e/x.png"/><use href="#a"/><animate attributeName="x"/><rect id="ok" width="1" height="1"/>',
		);
		expect(inventory(node).tags).toEqual(["svg", "rect"]);
	});

	it("drops event handlers, hrefs and unknown attributes but keeps aria and data", () => {
		const node = run('<rect onclick="x()" onload="x()" href="javascript:x" xlink:href="#a" src="x" bogus="1" aria-label="L" data-id="n1" width="3"/>'.replace("xlink:href", "xmlns:xlink='http://www.w3.org/1999/xlink' xlink:href"));
		const rect = node.querySelector("rect")!;
		expect(Array.from(rect.attributes).map((a) => a.name).sort()).toEqual(["aria-label", "data-id", "width"]);
	});

	it("allows url() only to a fragment", () => {
		const node = run('<rect fill="url(#g1)" stroke="url(https://evil/x)" clip-path="url(&quot;#c&quot;)"/>');
		const rect = node.querySelector("rect")!;
		expect(rect.getAttribute("fill")).toBe("url(#g1)");
		expect(rect.hasAttribute("stroke")).toBe(false);
	});

	it("filters a style attribute down to allow-listed properties and safe values", () => {
		expect(cleanStyle("color:red;position:fixed;fill:url(https://e/x);--_bg:var(--pele-bg,#fff);--evil:1;font-family:a\\62")).toBe(
			"color:red;--_bg:var(--pele-bg,#fff)",
		);
		expect(cleanStyle("background:url(javascript:x)")).toBe("");
		expect(cleanStyle("fill:#fff;stroke:expression(alert(1))")).toBe("fill:#fff");
	});

	it("keeps text as text, never as markup", () => {
		const node = run("<text>&lt;img src=x onerror=alert(1)&gt;</text>");
		expect(node.querySelector("img")).toBeNull();
		expect(node.querySelector("text")!.textContent).toBe("<img src=x onerror=alert(1)>");
	});

	it("returns null for text that is not an SVG document", () => {
		expect(sanitizeSvg("<svg")).toBeNull();
		expect(sanitizeSvg("<html><body>x</body></html>")).toBeNull();
		expect(sanitizeSvg('<svg xmlns="http://www.w3.org/1999/xhtml"/>')).toBeNull();
		expect(sanitizeSvg("")).toBeNull();
	});
});

describe("serializeSvg", () => {
	it("writes the theme into the root style so a saved file keeps its colours", () => {
		const node = sanitizeSvg(render(DIAGRAM_SAMPLES.flowchart, OPTIONS).svg)!;
		const text = serializeSvg(node, [["--pele-bg", "#101010"]]);
		expect(text).toContain('xmlns="http://www.w3.org/2000/svg"');
		expect(text).toMatch(/<svg[^>]* style="--pele-bg:#101010;/);
		expect(node.getAttribute("style")).not.toContain("#101010");
		expect(sanitizeSvg(text)).not.toBeNull();
	});
});
