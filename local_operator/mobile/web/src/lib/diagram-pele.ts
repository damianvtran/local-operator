/**
 * Draws Mermaid text with Pele, loaded on demand.
 *
 * `pele/lazy` is imported dynamically, so the renderer and each diagram kind
 * are separate chunks fetched when the first diagram appears: none of it is in
 * the entry chunk the phone downloads to paint the session list. A failed fetch
 * clears the memo and comes back as `retryable`, so the block keeps its code block
 * and the next mount tries again.
 *
 * WHAT A DRAWING DEPENDS ON. The SVG is a pure function of the source, the
 * width it may use, the fonts and the mode. Colours are not in that list:
 * Pele paints with CSS variables (`diagram-theme.ts`), so a theme flip never
 * needs a redraw. That is the invariant the cache below rests on, and
 * `diagram-pele.test.ts` pins it by drawing the same source twice and comparing.
 *
 * LINKS AND IMAGES ARE OFF. A diagram link would be one more thing a tap could
 * navigate to from model-written text, and a diagram image is a request the
 * phone makes to a host the model chose. Both are refused at the source, on top
 * of the allow-list in `diagram-svg.ts`.
 */
import { PELE_FONT_SIZE, diagramFonts } from "./diagram-theme";

type PeleModule = typeof import("pele/lazy");

export type DiagramMode =
	/** Inline in the transcript: fit the column, let Pele reflow narrow diagrams. */
	| { kind: "fit"; width: number }
	/** Full size: the author's direction, fixed pixel size, scrolled by the viewer. */
	| { kind: "full" };

export type DrawResult =
	| { ok: true; svg: string; width: number; height: number }
	/** `retryable`: the engine could not be loaded (a chunk fetch failed), which says nothing about the source. */
	| { ok: false; message: string; retryable: boolean };

let peleModule: Promise<PeleModule> | null = null;

export function loadPele(): Promise<PeleModule> {
	if (!peleModule) {
		peleModule = import("pele/lazy").catch((error: unknown) => {
			peleModule = null;
			throw error;
		});
	}
	return peleModule;
}

/** Widths are bucketed so a column that moves by a pixel does not redraw. */
export const WIDTH_STEP = 8;
export function bucketWidth(width: number): number {
	return Math.max(WIDTH_STEP, Math.floor(width / WIDTH_STEP) * WIDTH_STEP);
}

const CACHE_LIMIT = 48;
const cache = new Map<string, DrawResult>();
/** Draws in flight, so two identical fences in one message share one render. */
const inflight = new Map<string, Promise<DrawResult>>();

function hash(text: string): string {
	let h = 0x811c9dc5;
	for (let i = 0; i < text.length; i++) {
		h ^= text.charCodeAt(i);
		h = Math.imul(h, 0x01000193) >>> 0;
	}
	return h.toString(36);
}

function cacheKey(source: string, mode: DiagramMode, fonts: { sans: string; mono: string }): string {
	const shape = mode.kind === "fit" ? `fit:${bucketWidth(mode.width)}` : "full";
	return `${shape}\u0000${fonts.sans}\u0000${fonts.mono}\u0000${source}`;
}

function remember(key: string, value: DrawResult): void {
	cache.delete(key);
	cache.set(key, value);
	if (cache.size > CACHE_LIMIT) {
		const oldest = cache.keys().next().value;
		if (oldest !== undefined) cache.delete(oldest);
	}
}

/** A finished drawing already in memory, if any: lets a remounted row paint without a flash. */
export function peekDrawing(source: string, mode: DiagramMode): DrawResult | undefined {
	return cache.get(cacheKey(source, mode, diagramFonts()));
}

/** Test seam: forget every drawing and the loaded module. */
export function resetDiagramCache(): void {
	cache.clear();
	inflight.clear();
	peleModule = null;
}

function messageOf(error: unknown): string {
	return error instanceof Error && error.message ? error.message : String(error);
}

export function drawDiagram(source: string, mode: DiagramMode): Promise<DrawResult> {
	const fonts = diagramFonts();
	const key = cacheKey(source, mode, fonts);
	const hit = cache.get(key);
	if (hit) return Promise.resolve(hit);
	let pending = inflight.get(key);
	if (!pending) {
		pending = draw(source, mode, fonts, key).finally(() => inflight.delete(key));
		inflight.set(key, pending);
	}
	return pending;
}

async function draw(
	source: string,
	mode: DiagramMode,
	fonts: { sans: string; mono: string },
	key: string,
): Promise<DrawResult> {
	let pele: PeleModule;
	try {
		pele = await loadPele();
	} catch (error) {
		return { ok: false, message: messageOf(error), retryable: true };
	}
	try {
		await document.fonts?.ready;
		const { svg, width, height } = await pele.renderAsync(source, {
			fontFamily: fonts.sans,
			fontFamilyMono: fonts.mono,
			fontSize: PELE_FONT_SIZE,
			idPrefix: `pele-${hash(key)}`,
			links: false,
			images: false,
			...(mode.kind === "fit"
				? /* `responsive` is set, not left to the source: an author's
				     `%%{init: {"flowchart": {"useMaxWidth": false}}}%%` would otherwise drop the
				     shrink-to-fit style. */
				  { responsive: true, maxWidth: bucketWidth(mode.width) }
				: { responsive: false, autoDirection: false }),
		});
		const result: DrawResult = { ok: true, svg, width, height };
		remember(key, result);
		return result;
	} catch (error) {
		/* A parse failure is a fact about the source and is cached. */
		const result: DrawResult = { ok: false, message: messageOf(error), retryable: false };
		remember(key, result);
		return result;
	}
}
