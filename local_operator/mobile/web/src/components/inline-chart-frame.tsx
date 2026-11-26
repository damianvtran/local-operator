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

/** Tooltip anchor in plot px: `y` is the mark's top, `below` its bottom. */
export type Anchor = { x: number; y: number; below?: number };

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

/**
 * Text width at the label font. Real glyph widths via canvas where the browser
 * offers one (spec §2: label thinning must use measured widths so 32 labels on
 * a 360px phone never overlap); the 6.6 px/char estimate otherwise.
 */
export function useMeasure(ref: React.RefObject<HTMLElement | null>, width: number) {
	return useMemo(() => {
		let context: CanvasRenderingContext2D | null = null;
		try {
			const node = ref.current;
			if (node && typeof document !== "undefined") {
				context = document.createElement("canvas").getContext("2d");
				const style = getComputedStyle(node);
				if (context) context.font = `${style.fontWeight} 12px ${style.fontFamily || "sans-serif"}`;
			}
		} catch {
			context = null;
		}
		const cache = new Map<string, number>();
		return (text: string) => {
			let w = cache.get(text);
			if (w === undefined) {
				const measured = context?.measureText?.(text)?.width;
				w = measured && Number.isFinite(measured) ? measured : estimateWidth(text);
				cache.set(text, w);
			}
			return w;
		};
		// Re-measure on resize only; the font does not change per render.
	}, [width]);
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
			<div id={tableId} hidden={!open} className="mt-2 overflow-x-auto">
				<table aria-labelledby={titleId} className="w-full border-collapse text-meta">
					<thead className="text-ink-muted">
						<tr className="h-7">
							{table.rows.some((row) => row.label !== undefined) && <th scope="col" className="px-2 text-start font-medium"><bdi>{table.corner ?? ""}</bdi></th>}
							{table.columns.map((column, i) => <th key={i} scope="col" className="px-2 text-end font-medium"><bdi>{column}</bdi></th>)}
						</tr>
					</thead>
					<tbody>
						{table.rows.map((row, i) => (
							<tr key={i} className="h-7 border-t border-hairline">
								{row.label !== undefined && <th scope="row" className="px-2 text-start font-normal text-ink-muted wrap-anywhere"><bdi>{row.label}</bdi></th>}
								{row.cells.map((cell, j) => <td key={j} className="px-2 text-end tabular-nums"><bdi>{cell}</bdi></td>)}
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
 * to the node in a layout effect — 8px above the mark, flipped below at the
 * top edge, clamped 4px inside the chart — so there is no second render and
 * no frame where it sits at the wrong spot.
 */
function Tooltip({ tip, anchor, width }: { tip: Tip; anchor: Anchor; width: number }) {
	const ref = useRef<HTMLDivElement>(null);
	useLayoutEffect(() => {
		const node = ref.current;
		if (!node) return;
		const w = node.offsetWidth;
		const h = node.offsetHeight;
		const left = Math.max(4, Math.min(anchor.x - w / 2, width - w - 4));
		let top = anchor.y - 8 - h;
		if (top < 0) top = (anchor.below ?? anchor.y) + 8;
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

export type SurfaceProps = {
	title: string;
	tableId: string;
	width: number;
	height: number;
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
	children: (active: Mark | null, setActive: (mark: Mark | null) => void) => ReactNode;
	after?: (active: Mark | null, setActive: (mark: Mark | null) => void) => ReactNode;
	/** Class for the svg; height is always the real pixel height. */
	svgClass?: string;
	svgWidth?: number;
};

/**
 * One chart: ONE tab stop with `role="group"` / `aria-roledescription` (spec
 * §5 "Keyboard"), the SVG at its real pixel size (viewBox == rendered size,
 * so circles stay round — the stretched 320×100 viewBox was the operator's
 * "axes are not proper, the scale is off"), and the shared active mark.
 */
export function ChartSurface(props: SurfaceProps) {
	const { title, tableId, width, height, kind, columns, rows, exists, hit, tip, announce, scrub, header, children, after, svgClass, svgWidth = width } = props;
	const [active, setActiveState] = useState<Mark | null>(null);
	const pinned = useRef(false);
	const dragging = useRef<number | null>(null);
	const root = useRef<HTMLDivElement>(null);
	const setActive = useCallback((mark: Mark | null) => {
		setActiveState((prev) => (prev?.c === mark?.c && prev?.s === mark?.s ? prev : mark));
	}, []);
	const valid = useCallback((mark: Mark) => !exists || exists(mark), [exists]);

	// A pinned (tapped) tooltip dismisses on a tap anywhere outside the chart.
	useEffect(() => {
		const dismiss = (event: globalThis.PointerEvent) => {
			if (pinned.current && !root.current?.contains(event.target as Node)) {
				pinned.current = false;
				setActive(null);
			}
		};
		document.addEventListener("pointerdown", dismiss);
		return () => document.removeEventListener("pointerdown", dismiss);
	}, [setActive]);

	const locate = (event: PointerEvent<HTMLDivElement>) => {
		const rect = event.currentTarget.getBoundingClientRect();
		return hit(event.clientX - rect.left, event.clientY - rect.top);
	};
	const onPointerMove = (event: PointerEvent<HTMLDivElement>) => {
		if (event.pointerType === "touch") {
			if (scrub && dragging.current === event.pointerId) setActive(locate(event));
			return;
		}
		setActive(locate(event));
	};
	const onPointerDown = (event: PointerEvent<HTMLDivElement>) => {
		if (event.pointerType !== "touch") return;
		// Tap to inspect, tap the same mark again to dismiss (spec §5 "Touch").
		const mark = locate(event);
		const same = pinned.current && mark && active && mark.c === active.c && mark.s === active.s;
		pinned.current = Boolean(mark) && !same;
		setActive(same ? null : mark);
		dragging.current = scrub ? event.pointerId : null;
	};
	const endDrag = () => { dragging.current = null; };
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
			pinned.current = false;
			setActive(null);
			return;
		}
		const current = active ?? first();
		if (!current) return;
		const vertical = event.key === "ArrowUp" || event.key === "ArrowDown";
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
			className="relative rounded-sm text-meta outline-offset-4">
			{header}
			<div className="relative" style={{ touchAction: "pan-y" }}
				onPointerMove={onPointerMove} onPointerDown={onPointerDown} onPointerUp={endDrag}
				onPointerCancel={endDrag} onPointerLeave={onPointerLeave}>
				<svg aria-hidden="true" data-inline-plot={kind} width={svgWidth} height={height}
					viewBox={`0 0 ${svgWidth} ${height}`} className={`block max-w-full overflow-visible ${svgClass ?? ""}`}>
					<g className="lo-chart-in">{children(active, setActive)}</g>
				</svg>
				{shown && <Tooltip tip={shown.tip} anchor={shown.anchor} width={width} />}
			</div>
			{after?.(active, setActive)}
			<div className="sr-only" aria-live="polite">{active && valid(active) ? announce(active) : ""}</div>
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
export function LegendItem({ swatch, label, dashed, active, onEnter, onLeave }: {
	swatch: string; label: string; dashed?: boolean; active?: boolean;
	onEnter: () => void; onLeave: () => void;
}) {
	return (
		<span className={`inline-flex min-w-0 items-center gap-1.5 ${dim(active === false)}`}
			onPointerEnter={onEnter} onPointerLeave={onLeave}>
			{dashed
				? <svg aria-hidden="true" width={12} height={2} className="shrink-0 overflow-visible"><line x1={0} x2={12} y1={1} y2={1} strokeWidth={2} strokeDasharray="4 2" className={swatch.replace("bg-", "stroke-")} /></svg>
				: <span aria-hidden="true" className={`size-2.5 shrink-0 rounded-[2px] ${swatch}`} />}
			<span className="truncate text-ink"><bdi>{label}</bdi></span>
		</span>
	);
}
