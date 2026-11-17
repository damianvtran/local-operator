/**
 * Pele's SVG string -> a DOM subtree the page can safely adopt.
 *
 * The transcript renders model-written text and the markdown renderer has no
 * HTML sink at all (`markdown.tsx`: React elements only, no
 * dangerouslySetInnerHTML), so the diagram must not be the first one. Pele
 * escapes diagram text and filters links and styles itself, but this layer does
 * not depend on that: the string is PARSED as an inert XML document (no script
 * runs, nothing is fetched while parsing) and then COPIED, element by element,
 * into fresh nodes made with `createElementNS`. Anything not on an allow-list
 * is simply not copied:
 *
 *   - elements: shapes, text, gradients, markers. No script, style, foreignObject,
 *     a, image, use, animation or iframe.
 *   - attributes: a fixed presentation set plus `aria-*` and `data-*`. No `on*`,
 *     no `href`/`xlink:href`, no `src`. `class` keeps only Pele's own `pele-*` names.
 *   - values: no `javascript:`/`vbscript:`/`data:`; a `url(...)` may only point
 *     at a fragment (`url(#gradient)`); no CSS escapes, `@import` or
 *     `expression(`. A `<` or `>` in a value is data (`data-id="A->B"`), set with
 *     `setAttribute` and escaped again by the serialiser; it is not markup.
 *   - `style`: only allow-listed properties, each value checked the same way.
 *
 * A document that does not parse yields `null` and the caller keeps the code
 * block. Nothing here assigns `innerHTML` or parses HTML.
 */

const SVG_NS = "http://www.w3.org/2000/svg";

const ELEMENTS = new Set([
	"svg", "g", "defs", "title", "desc", "path", "rect", "circle", "ellipse", "line", "polyline", "polygon",
	"text", "tspan", "linearGradient", "radialGradient", "stop", "clipPath", "marker", "pattern", "mask", "symbol",
]);

const ATTRIBUTES = new Set([
	"class", "id", "viewBox", "preserveAspectRatio", "width", "height", "x", "y", "x1", "y1", "x2", "y2", "cx", "cy",
	"r", "rx", "ry", "d", "points", "transform", "fill", "fill-opacity", "fill-rule", "clip-rule", "stroke",
	"stroke-width", "stroke-linecap", "stroke-linejoin", "stroke-dasharray", "stroke-dashoffset", "stroke-opacity",
	"stroke-miterlimit", "opacity", "offset", "stop-color", "stop-opacity", "gradientUnits", "gradientTransform",
	"spreadMethod", "patternUnits", "patternTransform", "clipPathUnits", "maskUnits", "markerWidth", "markerHeight",
	"markerUnits", "refX", "refY", "orient", "clip-path", "marker-start", "marker-mid", "marker-end", "mask",
	"text-anchor", "dominant-baseline", "alignment-baseline", "font-size", "font-weight", "font-style",
	"font-family", "letter-spacing", "dx", "dy", "textLength", "lengthAdjust", "role", "style", "xml:space",
	"xmlns", "visibility",
]);

const ARIA_OR_DATA = /^(aria-[a-z]+|data-[a-z0-9_-]+)$/;

const STYLE_PROPERTIES = new Set([
	"color", "direction", "font", "font-family", "font-size", "font-weight", "font-style", "height", "max-width",
	"letter-spacing", "word-spacing", "text-transform", "text-decoration", "rx", "fill", "fill-opacity", "stroke",
	"stroke-width", "stroke-dasharray", "stroke-opacity", "opacity",
]);

const CUSTOM_PROPERTY = /^--(_[a-z0-9-]*|pele-[a-z0-9-]+)$/;

/** Values no attribute or declaration may carry, whatever its name. */
const FORBIDDEN_VALUE = /javascript:|vbscript:|data:|expression\s*\(|@import|\\|\/\*/i;
/**
 * The ONLY `url(...)` a value may carry: a fragment of this document. Every
 * quantifier is bounded, so removing them is linear in the value's length (an
 * unbounded `url\(\s*([^)]*)\)` was quadratic on `url(` plus tens of thousands
 * of spaces: 4.5 s for a 45k-character fence, measured). Whatever `url(` is left
 * afterwards, closed or not, is refused.
 */
const FRAGMENT_URL = /url\(\s{0,8}["']?#[A-Za-z0-9_.:-]{1,200}["']?\s{0,8}\)/gi;
const ANY_URL = /url\s*\(/i;

function safeValue(value: string): boolean {
	if (FORBIDDEN_VALUE.test(value)) return false;
	return !ANY_URL.test(value.replace(FRAGMENT_URL, ""));
}

/** Splits `a:b;c:var(--x,#fff)` into declarations without breaking inside parentheses. */
function declarations(style: string): Array<[string, string]> {
	const out: Array<[string, string]> = [];
	let depth = 0;
	let start = 0;
	const push = (end: number) => {
		const part = style.slice(start, end);
		const colon = part.indexOf(":");
		if (colon > 0) out.push([part.slice(0, colon).trim().toLowerCase(), part.slice(colon + 1).trim()]);
	};
	for (let i = 0; i < style.length; i++) {
		const ch = style[i];
		if (ch === "(") depth++;
		else if (ch === ")") depth = Math.max(0, depth - 1);
		else if (ch === ";" && depth === 0) {
			push(i);
			start = i + 1;
		}
	}
	push(style.length);
	return out;
}

/** The declarations of a `style` attribute that survive the allow-list, re-joined. */
export function cleanStyle(style: string): string {
	return declarations(style)
		.filter(([name, value]) => (STYLE_PROPERTIES.has(name) || CUSTOM_PROPERTY.test(name)) && value !== "" && safeValue(value))
		.map(([name, value]) => `${name}:${value}`)
		.join(";");
}

/** Class names Pele writes. An author-chosen one (`:::hidden`) would meet the app's own utilities. */
const PELE_CLASS = /^pele(?:-[A-Za-z0-9]+)*$/;

function copyAttributes(from: Element, to: Element): void {
	for (const attr of Array.from(from.attributes)) {
		const name = attr.name;
		const inert = ARIA_OR_DATA.test(name);
		if (!(ATTRIBUTES.has(name) || inert)) continue;
		if (name === "xmlns") continue; // createElementNS already set the namespace
		let value = attr.value;
		if (name === "style") {
			value = cleanStyle(value);
			if (value === "") continue;
		} else if (name === "class") {
			value = value.split(/\s+/).filter((token) => PELE_CLASS.test(token)).join(" ");
			if (value === "") continue;
		} else if (!inert && !safeValue(value)) {
			/* aria-* and data-* are never interpreted as a URL or a style, so a label
			   that happens to say "data:" is kept as the text it is. */
			continue;
		}
		if (name === "xml:space") to.setAttributeNS("http://www.w3.org/XML/1998/namespace", name, value);
		else to.setAttribute(name, value);
	}
}

function copyNode(from: Element, into: Element, doc: Document): void {
	for (const child of Array.from(from.childNodes)) {
		if (child.nodeType === 3 /* text */ || child.nodeType === 4 /* CDATA */) {
			into.appendChild(doc.createTextNode(child.nodeValue ?? ""));
		} else if (child.nodeType === 1) {
			const el = child as Element;
			if (el.namespaceURI !== SVG_NS || !ELEMENTS.has(el.localName)) continue;
			const copy = doc.createElementNS(SVG_NS, el.localName);
			copyAttributes(el, copy);
			copyNode(el, copy, doc);
			into.appendChild(copy);
		}
	}
}

/**
 * Parse and copy. Returns a detached `<svg>` owned by `doc`, or null when the
 * text is not a well-formed SVG document whose root is `<svg>`.
 */
export function sanitizeSvg(svg: string, doc: Document = document): SVGSVGElement | null {
	let parsed: Document;
	try {
		parsed = new DOMParser().parseFromString(svg, "image/svg+xml");
	} catch {
		return null;
	}
	const root = parsed.documentElement;
	if (!root || root.localName !== "svg" || root.namespaceURI !== SVG_NS) return null;
	if (parsed.getElementsByTagName("parsererror").length > 0) return null;
	const out = doc.createElementNS(SVG_NS, "svg");
	copyAttributes(root, out);
	copyNode(root, out, doc);
	return out as SVGSVGElement;
}

/** Serialises a sanitised tree, adding `declarations` to the root's style so a saved file keeps its colours. */
export function serializeSvg(node: SVGSVGElement, extraRootStyle: Array<[string, string]> = []): string {
	const copy = node.cloneNode(true) as SVGSVGElement;
	copy.setAttribute("xmlns", SVG_NS);
	if (extraRootStyle.length > 0) {
		const prefix = extraRootStyle.map(([name, value]) => `${name}:${value}`).join(";");
		const existing = copy.getAttribute("style");
		copy.setAttribute("style", existing ? `${prefix};${existing}` : prefix);
	}
	return new XMLSerializer().serializeToString(copy);
}
