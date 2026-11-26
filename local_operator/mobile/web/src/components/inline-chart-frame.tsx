/**
 * The interactive shell every inline chart shares: width measurement, the one
 * tab stop, pointer/touch/keyboard driving ONE active-mark state, the floating
 * tooltip, the screen-reader live region and the collapsed "Chart data"
 * disclosure (inline-chart spec §5 and §6).
 *
 * Kept apart from the per-type geometry in `inline-ui-charts.tsx` so each
 * chart type only answers four questions — what is under this point, what
 * does mark (c, s) say, where is it, and how do I draw it — and cannot
 * diverge on the interaction model, which the operator asked to be the same
 * across every chart ("hover on each data point and see what specific value
 * it represents … should be more interactive").
 */
import {
	type KeyboardEvent, type PointerEvent, type ReactNode,
	useCallback, useEffect, useId, useLayoutEffect, useMemo, useRef, useState,
} from "react";
import { estimateWidth, stepIndex } from "../lib/chart-geometry";
import { messageRuntime } from "../store";

/** Column-major mark address: c = category/slice/column, s = series/row. */
export type Mark = { c: number; s: number };

export type TipRow = { swatch?: string; label?: string; value: string };
export type Tip = { heading?: string; rows: TipRow[]; notes?: string[] };

/**
 * Tooltip anchor in plot px: `y` is the mark's top, `below` its bottom.
 * With `side`, the tooltip sits beside `x` instead (its near edge at `x`),
 * vertically centred on `y` — the wide donut's placement (X5).
 */
export type Anchor = { x: number; y: number; below?: number; side?: "left" | "right" };

/** Every series/slice colour, as Tailwind classes so they are emitted. */
export const FILL = ["fill-chart-1", "fill-chart-2", "fill-chart-3", "fill-chart-4", "fill-chart-5", "fill-chart-6", "fill-chart-7"];
export const STROKE = ["stroke-chart-1", "stroke-chart-2", "stroke-chart-3", "stroke-chart-4", "stroke-chart-5", "stroke-chart-6", "stroke-chart-7"];
export const SWATCH = ["bg-chart-1", "bg-chart-2", "bg-chart-3", "bg-chart-4", "bg-chart-5", "bg-chart-6", "bg-chart-7"];
/** Pie "Other": ink-dim, deliberately outside the categorical ramp (spec §4). */
export const OTHER_FILL = "fill-ink-dim";
export const OTHER_SWATCH = "bg-ink-dim";

/** Highlight: everything but the active mark drops to 35% (spec §5). */
export const dim = (on: boolean) =>
	`transition-opacity duration-120 motion-reduce:transition-none ${on ? "opacity-35" : ""}`;

/** Widest-first fallback when a phone has no measurable font yet. */
const FALLBACK_WIDTH = 360;

/**
 * Measure the chart's content width (spec §8 "Relay web": same geometry as the
 * desktop at its viewport). Layout effect, so the first painted frame already
 * has the real width and the chart never reflows after mount.
 */
function useWidth() {
	const ref = useRef<HTMLDivElement>(null);
	const [width, setWidth] = useState(0);
	useLayoutEffect(() => {
		const node = ref.current;
		if (!node) return;
		const read = () => setWidth(Math.round(node.getBoundingClientRect().width) || FALLBACK_WIDTH);
		read();
		if (typeof ResizeObserver === "undefined") return;
		const observer = new ResizeObserver(read);
		observer.observe(node);
		return () => observer.disconnect();
	}, []);
	return [ref, width] as const;
}

/** One shared canvas for every chart's text measurement; contexts are cheap
 * to reuse and a canvas per chart would be one more allocation per render. */
let sharedContext: CanvasRenderingContext2D | null = null;
function measuringContext(): CanvasRenderingContext2D | null {
	// Only a real context is cached: a null (no canvas yet, or a test engine
	// without one) is retried by the next chart instead of pinning every chart
	// to the estimate for the page's lifetime.
	if (sharedContext) return sharedContext;
	try {
		sharedContext = typeof document === "undefined" ? null : document.createElement("canvas").getContext("2d");
	} catch {
		sharedContext = null;
	}
	return sharedContext;
}

/** Builds a cached measurer for one font; the estimate when there is no canvas. */
export function textMeasurer(font: string | null, context: CanvasRenderingContext2D | null = measuringContext()) {
	const cache = new Map<string, number>();
	return (text: string) => {
		let w = cache.get(text);
		if (w === undefined) {
			let measured: number | undefined;
			if (font && context?.measureText) {
				context.font = font;
				measured = context.measureText(text)?.width;
			}
			w = measured !== undefined && Number.isFinite(measured) && (measured > 0 || text === "") ? measured : estimateWidth(text);
			cache.set(text, w);
		}
		return w;
	};
}

/**
 * Text width at the label font. Real glyph widths via canvas (spec §2: label
 * thinning must use measured widths so 32 labels on a 360px phone never
 * overlap); the 6.6 px/char estimate until then, or where there is no canvas.
 *
 * Why a layout effect and a font version (R3): the round-1 hook read
 * `ref.current` inside `useMemo` during the first render, before the probe's
 * ref was attached, so the canvas path never ran and zh-CN labels (≈1.6× the
 * estimate) overlapped. The effect runs after the ref attaches and before
 * paint, so the first painted frame already uses canvas widths; it bumps again
 * when web fonts finish loading, since a fallback face measures differently.
 */
export function useMeasure(ref: React.RefObject<HTMLElement | null>, width: number) {
	const [font, setFont] = useState<string | null>(null);
	const [fontsVersion, setFontsVersion] = useState(0);
	useLayoutEffect(() => {
		const node = ref.current;
		if (!node || typeof getComputedStyle === "undefined") return;
		const style = getComputedStyle(node);
		const next = `${style.fontWeight || "400"} 12px ${style.fontFamily || "sans-serif"}`;
		setFont((prev) => (prev === next ? prev : next));
	}, [ref, width, fontsVersion]);
	useEffect(() => {
		const fonts = typeof document === "undefined" ? undefined : document.fonts;
		if (!fonts?.ready) return;
		let live = true;
		void fonts.ready.then(() => { if (live) setFontsVersion((v) => v + 1); });
		return () => { live = false; };
	}, []);
	// A new measurer per font (and per fonts-ready pass) drops widths cached
	// against the fallback face.
	return useMemo(() => textMeasurer(font), [font, fontsVersion]);
}

export type DataTable = {
	/** Header of the category column (often empty). */
	corner?: string;
	columns: string[];
	rows: { label?: string; cells: string[] }[];
};

/**
 * Title, measured body and the collapsed "Chart data" disclosure (spec §6).
 * The table stays in the DOM while collapsed (`hidden`) because the chart
 * root's `aria-describedby` points at it: hidden referenced content still
 * contributes the description, so a screen reader hears every value.
 */
export function ChartBlock({ title, table, children }: {
	title: string;
	table: DataTable;
	children: (args: { width: number; titleId: string; tableId: string }) => ReactNode;
}) {
	const titleId = useId();
	const tableId = useId();
	const [open, setOpen] = useState(false);
	const [ref, width] = useWidth();
	const tableRef = useRef<HTMLTableElement>(null);
	const [wide, setWide] = useState(false);
	// The table fits the phone column whenever it can: text cells break
	// anywhere and numbers stay whole and end-aligned (R2-1, operator finding
	// B). Only a table whose whole numbers cannot fit at all — measured by the
	// browser, not guessed; a 3-row × 24-hour heatmap with 4-digit values lays
	// out 1,286px wide at 328px — gets the keyboard-reachable rail the inline
	// data grid uses. Breaking `12` into `1`/`2` would be a false
	// reading, and truncating it loses the exact value the table exists for.
	// On the rail the table takes its max-content width, so labels stop
	// breaking letter by letter beside numbers that already scroll. Its width
	// exceeds the box only while its content does, so a rotate or resize wide
	// enough for every cell drops the rail again.
	useLayoutEffect(() => {
		const node = tableRef.current;
		if (!open || !node?.parentElement) return;
		const next = node.offsetWidth > node.parentElement.clientWidth + 0.5;
		setWide((prev) => (prev === next ? prev : next));
	}, [open, width, table]);
	return (
		// 16px margin + the transcript's 8px gap = the 24px block rhythm. No card,
		// border or fill: the chart sits on the message ground (spec §1), so the
		// desktop's translucent window ground shows through on the relay too.
		<section className="my-4 min-w-0 text-ink">
			<div id={titleId} className="text-start text-body-sm font-medium wrap-anywhere"><bdi>{title}</bdi></div>
			<div ref={ref} className="mt-1 min-w-0">{width > 0 && children({ width, titleId, tableId })}</div>
			<button type="button" aria-expanded={open} aria-controls={tableId} onClick={() => setOpen(!open)}
				className="mt-3 inline-flex items-center gap-1 rounded-sm text-meta text-ink-muted">
				{messageRuntime.t("relay.chart.data_toggle")}
				<svg aria-hidden="true" viewBox="0 0 12 12" className={`size-3 transition-transform duration-120 motion-reduce:transition-none ${open ? "rotate-180" : ""}`}>
					<path d="M3 4.5 6 7.5 9 4.5" className="fill-none stroke-current" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round" />
				</svg>
			</button>
			{/* Fit first (see `wide` above): headers and row labels break
			    anywhere, so the auto layout shrinks them before any number. */}
			<div id={tableId} hidden={!open} className={wide ? "mt-2 overflow-x-auto rounded-sm focus-visible:outline-accent" : "mt-2 min-w-0"} {...(wide ? {
				role: "region", "aria-labelledby": titleId, tabIndex: 0,
			} : {})}>
				<table ref={tableRef} aria-labelledby={titleId} className={`${wide ? "w-max min-w-full" : "w-full"} border-collapse text-meta`}>
					<thead className="text-ink-muted">
						<tr className="h-7">
							{table.rows.some((row) => row.label !== undefined) && <th scope="col" className="px-2 text-start font-medium wrap-anywhere"><bdi>{table.corner ?? ""}</bdi></th>}
							{table.columns.map((column, i) => <th key={i} scope="col" className="px-2 text-end font-medium wrap-anywhere"><bdi>{column}</bdi></th>)}
						</tr>
					</thead>
					<tbody>
						{table.rows.map((row, i) => (
							<tr key={i} className="h-7 border-t border-hairline">
								{row.label !== undefined && <th scope="row" className="px-2 text-start font-normal text-ink-muted wrap-anywhere"><bdi>{row.label}</bdi></th>}
								{row.cells.map((cell, j) => <td key={j} className="px-2 text-end tabular-nums whitespace-nowrap"><bdi>{cell}</bdi></td>)}
							</tr>
						))}
					</tbody>
				</table>
			</div>
		</section>
	);
}

/** One data row of the tooltip: swatch · label · exact value. */
function TipLine({ row }: { row: TipRow }) {
	return (
		<div className="flex items-center gap-2">
			{row.swatch && <span aria-hidden="true" className={`size-2 shrink-0 rounded-[2px] ${row.swatch}`} />}
			{row.label && <span className="min-w-0 truncate text-ink-muted"><bdi>{row.label}</bdi></span>}
			<span className="ms-auto font-semibold whitespace-nowrap text-ink tabular-nums"><bdi>{row.value}</bdi></span>
		</div>
	);
}

/**
 * Floating tooltip (spec §5 "Tooltip style"/"Placement"): elevated ground,
 * overlay shadow, no border, pointer-events none, positioned in the chart's
 * relative container so it never moves layout. Placement is written straight
 * to the node in a layout effect so there is no second render and no frame
 * where it sits at the wrong spot:
 * - default: 8px above the mark, flipped below at the top edge, clamped 4px
 *   inside the chart horizontally and inside [0, limit] vertically, where
 *   `limit` is the plot's bottom — so it never parks over the x-axis labels
 *   or the "Chart data" button below them (X4; round 1 put a percent-column
 *   tooltip below the SVG, QA Q2).
 * - `anchor.side` (wide donut, X5): beside the ring on the slice's side,
 *   vertically centred on the slice, its width capped to the room outside
 *   the ring so it never covers the ring or the centre readout (design D1).
 */
function Tooltip({ tip, anchor, width, limit }: { tip: Tip; anchor: Anchor; width: number; limit: number }) {
	const ref = useRef<HTMLDivElement>(null);
	useLayoutEffect(() => {
		const node = ref.current;
		if (!node) return;
		const side = anchor.side;
		node.style.maxWidth = side
			? `${Math.max(96, Math.min(256, side === "right" ? width - anchor.x - 4 : anchor.x - 4))}px`
			: "";
		const w = node.offsetWidth;
		const h = node.offsetHeight;
		let left: number;
		let top: number;
		if (side) {
			left = side === "right" ? anchor.x : anchor.x - w;
			top = anchor.y - h / 2;
		} else {
			left = anchor.x - w / 2;
			top = anchor.y - 8 - h;
			if (top < 0) top = (anchor.below ?? anchor.y) + 8;
		}
		left = Math.max(4, Math.min(left, width - w - 4));
		top = Math.max(0, Math.min(top, limit - h));
		node.style.left = `${left}px`;
		node.style.top = `${top}px`;
	});
	return (
		<div ref={ref} data-chart-tooltip="" aria-hidden="true"
			className="pointer-events-none absolute top-0 left-0 z-10 max-w-64 min-w-24 rounded-sm bg-elevated px-2 py-1 text-meta text-ink shadow-overlay">
			{tip.heading && <div className="text-ink-muted wrap-anywhere"><bdi>{tip.heading}</bdi></div>}
			{tip.rows.map((row, i) => <TipLine key={i} row={row} />)}
			{tip.notes?.map((note, i) => <div key={i} className="text-ink-muted"><bdi>{note}</bdi></div>)}
		</div>
	);
}

/**
 * What chart drawings and legends get besides the active mark. `hover`
 * previews a mark (mouse enter/leave) and is ignored while something is
 * pinned; `toggle` pins a mark or, if it is already pinned, dismisses it —
 * the touch path for legend rows and "Also:" items (U2), where hover never
 * happens.
 */
export type ChartControls = { hover: (mark: Mark | null) => void; toggle: (mark: Mark) => void };

export type SurfaceProps = {
	title: string;
	tableId: string;
	width: number;
	height: number;
	/**
	 * Bottom of the plotted marks in px (top of the x-axis labels). The
	 * tooltip stays above it so it never covers the axis or the "Chart data"
	 * button (X4). Defaults to `height`.
	 */
	plotBottom?: number;
	/** `data-inline-plot`, the chart kind tests and the fixture key off. */
	kind: string;
	/** Navigation grid: columns (←/→) by rows (↑/↓). */
	columns: number;
	rows: number;
	/** Whether (c, s) is a real mark; scatter/pie have one row. */
	exists?: (mark: Mark) => boolean;
	hit: (x: number, y: number) => Mark | null;
	tip: (mark: Mark) => { tip: Tip; anchor: Anchor } | null;
	announce: (mark: Mark) => string;
	/** Line-like charts scrub horizontally under a dragging finger. */
	scrub?: boolean;
	header?: ReactNode;
	children: (active: Mark | null, controls: ChartControls) => ReactNode;
	after?: (active: Mark | null, controls: ChartControls) => ReactNode;
	/** Class for the svg; height is always the real pixel height. */
	svgClass?: string;
	svgWidth?: number;
};

/** Touch slop in px: a press that moves less than this is a tap (X7, native's 8dp). */
const TAP_SLOP = 8;

type Gesture = { id: number; x: number; y: number; scrubbing: boolean };

/**
 * One chart: ONE tab stop with `role="group"` / `aria-roledescription` (spec
 * §5 "Keyboard"), the SVG at its real pixel size (viewBox == rendered size,
 * so circles stay round — the stretched 320×100 viewBox was the operator's
 * "axes are not proper, the scale is off"), and the shared active mark.
 *
 * The root is forced LTR (X8): plot geometry is physical — the y axis is on
 * the left, x grows rightward — and SVG `text-anchor` resolves against the
 * inherited `direction`, so under `<html dir="rtl">` (Urdu) every end-anchored
 * tick label rendered INTO the plot (review R2). Label content keeps `<bdi>`
 * isolation, so Urdu strings still shape right-to-left inside their boxes.
 */
export function ChartSurface(props: SurfaceProps) {
	const { title, tableId, width, height, plotBottom = height, kind, columns, rows, exists, hit, tip, announce, scrub, header, children, after, svgClass, svgWidth = width } = props;
	const [active, setActiveState] = useState<Mark | null>(null);
	const activeRef = useRef(active);
	activeRef.current = active;
	const pinned = useRef(false);
	const gesture = useRef<Gesture | null>(null);
	const root = useRef<HTMLDivElement>(null);
	const setActive = useCallback((mark: Mark | null) => {
		setActiveState((prev) => (prev?.c === mark?.c && prev?.s === mark?.s ? prev : mark));
	}, []);
	const valid = useCallback((mark: Mark) => !exists || exists(mark), [exists]);
	const clear = useCallback(() => {
		pinned.current = false;
		setActive(null);
	}, [setActive]);
	const controls = useMemo<ChartControls>(() => ({
		hover: (mark) => { if (!pinned.current) setActive(mark); },
		toggle: (mark) => {
			// Read the committed mark from a ref, not inside a state updater:
			// updaters must stay pure (StrictMode runs them twice), and this one
			// also flips `pinned`.
			const prev = activeRef.current;
			const same = pinned.current && prev?.c === mark.c && prev?.s === mark.s;
			pinned.current = !same;
			setActive(same ? null : mark);
		},
	}), [setActive]);

	// A pinned (tapped) tooltip dismisses on a tap anywhere outside the chart.
	useEffect(() => {
		const dismiss = (event: globalThis.PointerEvent) => {
			if (pinned.current && !root.current?.contains(event.target as Node)) clear();
		};
		document.addEventListener("pointerdown", dismiss);
		return () => document.removeEventListener("pointerdown", dismiss);
	}, [clear]);

	const locate = (event: PointerEvent<HTMLDivElement>) => {
		const rect = event.currentTarget.getBoundingClientRect();
		return hit(event.clientX - rect.left, event.clientY - rect.top);
	};
	// Touch (X7, review R4 / UX U1): nothing activates on pointerdown. A press
	// is a TAP only if it lifts within TAP_SLOP; a horizontal drag on a
	// line-like chart becomes a scrub; anything else is the page scrolling.
	// Round 1 pinned on pointerdown, so every chart a thumb scrolled across
	// flashed a tooltip and the last one kept it with the rest dimmed. When the
	// browser takes the gesture for `pan-y` it sends pointercancel, which now
	// clears pending AND pinned state.
	const onPointerDown = (event: PointerEvent<HTMLDivElement>) => {
		if (event.pointerType !== "touch") return;
		gesture.current = { id: event.pointerId, x: event.clientX, y: event.clientY, scrubbing: false };
	};
	const onPointerMove = (event: PointerEvent<HTMLDivElement>) => {
		if (event.pointerType !== "touch") {
			// A mouse hovering the plot takes over from any pin (a clicked legend
			// row): hover is the mouse's way to read a mark.
			pinned.current = false;
			setActive(locate(event));
			return;
		}
		const g = gesture.current;
		if (!g || g.id !== event.pointerId) return;
		const dx = event.clientX - g.x;
		const dy = event.clientY - g.y;
		if (!g.scrubbing) {
			if (Math.hypot(dx, dy) <= TAP_SLOP) return;
			if (!scrub || Math.abs(dy) >= Math.abs(dx)) {
				// The page is scrolling under this finger: not a tap, not ours.
				gesture.current = null;
				clear();
				return;
			}
			g.scrubbing = true;
			pinned.current = true;
		}
		setActive(locate(event));
	};
	const onPointerUp = (event: PointerEvent<HTMLDivElement>) => {
		const g = gesture.current;
		if (event.pointerType !== "touch" || !g || g.id !== event.pointerId) return;
		gesture.current = null;
		if (g.scrubbing) return; // a scrub leaves its last mark pinned
		if (Math.hypot(event.clientX - g.x, event.clientY - g.y) > TAP_SLOP) return;
		// Tap to inspect, tap the same mark again (or empty plot) to dismiss.
		const mark = locate(event);
		if (mark) controls.toggle(mark);
		else clear();
	};
	const onPointerCancel = () => {
		gesture.current = null;
		clear();
	};
	const onPointerLeave = (event: PointerEvent<HTMLDivElement>) => {
		if (event.pointerType !== "touch" && !pinned.current) setActive(null);
	};

	const first = (): Mark | null => {
		for (let s = 0; s < rows; s++) for (let c = 0; c < columns; c++) if (valid({ c, s })) return { c, s };
		return null;
	};
	const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
		if (event.key === "Escape") {
			if (active) event.stopPropagation();
			clear();
			return;
		}
		const start = first();
		if (!start) return;
		const vertical = event.key === "ArrowUp" || event.key === "ArrowDown";
		const arrow = vertical || event.key === "ArrowLeft" || event.key === "ArrowRight";
		if (!active) {
			// From rest (after Escape, or a click-focus) the first arrow lands ON
			// mark 0 instead of stepping past it, as desktop and native do (U3).
			if (arrow) {
				event.preventDefault();
				setActive(start);
				return;
			}
		}
		const current = active ?? start;
		const next = { ...current };
		if (vertical) {
			if (rows < 2) return;
			next.s = stepIndex(event.key, current.s, rows) ?? current.s;
		} else {
			const c = stepIndex(event.key, current.c, columns);
			if (c === null) return;
			next.c = c;
		}
		event.preventDefault();
		if (valid(next)) setActive(next);
	};
	const onFocus = () => {
		// Keyboard focus activates mark 0; a mouse click that focuses the root
		// does not, or every click would pin a tooltip on the first bar.
		let visible = true;
		try { visible = root.current?.matches(":focus-visible") ?? true; } catch { /* old engines */ }
		if (visible && !active) setActive(first());
	};
	const onBlur = () => { if (!pinned.current) setActive(null); };

	const shown = active && valid(active) ? tip(active) : null;
	return (
		<div ref={root} role="group" tabIndex={0} aria-roledescription={messageRuntime.t("relay.chart.role")}
			aria-label={title} aria-describedby={tableId} onKeyDown={onKeyDown} onFocus={onFocus} onBlur={onBlur}
			dir="ltr" style={{ direction: "ltr" }} className="relative rounded-sm text-meta outline-offset-4">
			{header}
			<div className="relative" style={{ touchAction: "pan-y" }}
				onPointerMove={onPointerMove} onPointerDown={onPointerDown} onPointerUp={onPointerUp}
				onPointerCancel={onPointerCancel} onPointerLeave={onPointerLeave}>
				<svg aria-hidden="true" data-inline-plot={kind} width={svgWidth} height={height} direction="ltr"
					viewBox={`0 0 ${svgWidth} ${height}`} className={`block max-w-full overflow-visible ${svgClass ?? ""}`}>
					<g className="lo-chart-in">{children(active, controls)}</g>
				</svg>
				{shown && <Tooltip tip={shown.tip} anchor={shown.anchor} width={width} limit={plotBottom} />}
			</div>
			{after?.(active, controls)}
			<div className="sr-only" aria-live="polite" dir="auto">{active && valid(active) ? announce(active) : ""}</div>
		</div>
	);
}

/** Legend/unit row, 16px tall, 8px above the plot (spec §1 block rhythm). */
export function HeaderRow({ start, end, children }: { start?: ReactNode; end?: ReactNode; children?: ReactNode }) {
	return (
		<div className="mb-2 flex min-h-4 flex-wrap items-center gap-x-3 gap-y-1 text-meta leading-4 text-ink-muted">
			{start && <span className="min-w-0 truncate"><bdi>{start}</bdi></span>}
			{children}
			{end && <span className="ms-auto min-w-0 truncate"><bdi>{end}</bdi></span>}
		</div>
	);
}

/** One interactive legend entry; hovering it highlights its series (spec §5). */
export function LegendItem({ swatch, stroke, label, dashed, active, onEnter, onLeave }: {
	/** `bg-chart-N` for the square swatch; `stroke-chart-N` for the dashed one.
	 * Both spelled in full at the call site: the bundle check (and Tailwind's
	 * scanner) only see literal class names, never `swatch.replace("bg-", …)`. */
	swatch: string; stroke: string; label: string; dashed?: boolean; active?: boolean;
	onEnter: () => void; onLeave: () => void;
}) {
	return (
		<span className={`inline-flex min-w-0 items-center gap-1.5 ${dim(active === false)}`}
			onPointerEnter={onEnter} onPointerLeave={onLeave}>
			{dashed
				? <svg aria-hidden="true" width={12} height={2} className="shrink-0 overflow-visible"><line x1={0} x2={12} y1={1} y2={1} strokeWidth={2} strokeDasharray="4 2" className={stroke} /></svg>
				: <span aria-hidden="true" className={`size-2.5 shrink-0 rounded-[2px] ${swatch}`} />}
			<span className="truncate text-ink"><bdi>{label}</bdi></span>
		</span>
	);
}
