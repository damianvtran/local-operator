/**
 * A closed ```mermaid fence, drawn with Pele.
 *
 * STATES. Until the drawing exists the block shows the plain code block it
 * replaces, so a slow or failed chunk fetch costs nothing the page did not
 * already do. A drawing replaces it; a parse failure puts an error card above
 * it with the engine's own line, because a diagram that cannot be drawn is still
 * something the reader can read.
 *
 * FIT, THEN FULL SIZE. The transcript has no horizontal play (operator finding
 * B), so the inline drawing is laid out for the column's width (Pele reflows a
 * wide left-to-right chart top-to-bottom under 640 px) and its height is
 * clamped, with a fade and the labelled "Full size" button when it is. Full size
 * is a viewer portaled to <body> like the attachment viewer: the author's own
 * layout at pixel size, panned in both directions by the viewer's own scroll
 * area. Nothing here captures a one-finger drag: the inline drawing only reacts
 * to a TAP, so vertical scrolling of the transcript is untouched.
 *
 * THEME. Colours are CSS variables on the wrapper (`diagram-theme.ts`); a theme
 * change recolours the drawing without redrawing it.
 *
 * SAFETY. The SVG is copied into fresh DOM nodes through an allow-list
 * (`diagram-svg.ts`); it is never parsed as HTML.
 */
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { ReactNode, RefObject } from "react";
import { createPortal } from "react-dom";
import { cn } from "../lib/cn";
import {
	WIDTH_STEP,
	bucketWidth,
	drawDiagram,
	peekDrawing,
	type DiagramMode,
	type DrawResult,
} from "../lib/diagram-pele";
import { sanitizeSvg, serializeSvg } from "../lib/diagram-svg";
import { PELE_THEME_STYLE, resolvedThemeDeclarations } from "../lib/diagram-theme";
import { messageRuntime } from "../store";
import { useModalLayer } from "./ui/use-modal-layer";

/** The inline drawing's height cap. Taller drawings clip behind a fade and open full size. */
const INLINE_MAX_HEIGHT = "32rem";

/** How long "Copied" stays up. */
const COPIED_MS = 2000;

/** The nearest ancestor that scrolls vertically: the transcript's scroller. */
function scrollerOf(el: HTMLElement | null): HTMLElement | null {
	for (let node = el?.parentElement ?? null; node; node = node.parentElement) {
		const overflow = getComputedStyle(node).overflowY;
		/* Not "is scrollable yet": a transcript shorter than its box does not scroll
		   until this very drawing makes it taller, and that is the case to catch. */
		if (overflow === "auto" || overflow === "scroll") return node;
	}
	return null;
}

/** Within this many px of the end counts as "following the tail" (the transcript's own threshold). */
const TAIL_PX = 48;

/** The sanitised tree for a drawing, or null when the text was not a usable SVG. */
function useSvgNode(result: DrawResult | undefined): SVGSVGElement | null {
	return useMemo(() => (result?.ok ? sanitizeSvg(result.svg) : null), [result]);
}

/** Puts one detached node into the document and takes it out again. */
function SvgHost({ node, className }: { node: SVGSVGElement; className?: string }) {
	const host = useRef<HTMLDivElement>(null);
	useLayoutEffect(() => {
		const el = host.current;
		el?.replaceChildren(node);
		return () => {
			node.remove();
		};
	}, [node]);
	return <div ref={host} className={className} />;
}

export function DiagramErrorCard({ message, source }: { message: string; source: ReactNode }) {
	return (
		<div data-diagram-state="error" className="flex min-w-0 flex-col gap-1.5">
			<div role="alert" className="rounded-sm border border-danger-border bg-danger-wash px-3 py-2">
				<div className="text-body-sm font-semibold text-danger">{messageRuntime.t("relay.diagram.error.title")}</div>
				{/* The engine's own message, as TEXT: it quotes the diagram, and it is
				    the one line a reader can act on. English by design (a library's
				    diagnostic, not product copy). */}
				<pre dir="ltr" className="mt-1 font-mono text-mono-sm whitespace-pre-wrap text-ink-muted [overflow-wrap:anywhere]">
					{message}
				</pre>
			</div>
			{source}
		</div>
	);
}

type Copy = "idle" | "copied" | "failed";

const BUTTON =
	"flex min-h-11 min-w-fit grow items-center justify-center px-3 text-body-sm text-ink-muted select-none active:bg-elevated";

function useCopy(text: string): [Copy, () => void] {
	const [state, setState] = useState<Copy>("idle");
	const timer = useRef<number | undefined>(undefined);
	useEffect(() => () => window.clearTimeout(timer.current), []);
	const copy = useCallback(() => {
		const done = (next: Copy) => {
			setState(next);
			window.clearTimeout(timer.current);
			timer.current = window.setTimeout(() => setState("idle"), COPIED_MS);
		};
		void navigator.clipboard
			?.writeText(text)
			.then(() => done("copied"), () => done("failed"));
		if (!navigator.clipboard) done("failed");
	}, [text]);
	return [state, copy];
}

function CopyButton({ source }: { source: string }) {
	const [state, copy] = useCopy(source);
	const label = (s: Copy) =>
		s === "copied"
			? messageRuntime.t("relay.diagram.copied")
			: s === "failed"
				? messageRuntime.t("relay.diagram.copy_failed")
				: messageRuntime.t("relay.diagram.copy");
	const other = state === "idle" ? messageRuntime.t("relay.diagram.copied") : messageRuntime.t("relay.diagram.copy");
	return (
		<button type="button" data-testid="diagram-copy" onClick={copy} aria-live="polite" className={BUTTON}>
			{/* One grid cell holds the live label and the longest other one, so the
			    row never reflows when the label swaps (same trick as the ask card). */}
			<span className="grid">
				<span className="col-start-1 row-start-1">{label(state)}</span>
				<span aria-hidden="true" className="invisible col-start-1 row-start-1">
					{other}
				</span>
			</span>
		</button>
	);
}

/** Saves the drawing as an .svg file with the theme's current colours written in. */
function saveSvg(node: SVGSVGElement): void {
	const text = serializeSvg(node, resolvedThemeDeclarations());
	const url = URL.createObjectURL(new Blob([text], { type: "image/svg+xml" }));
	const link = document.createElement("a");
	link.href = url;
	link.download = "diagram.svg";
	document.body.appendChild(link);
	link.click();
	link.remove();
	window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** The full-size viewer: the author's own layout at pixel size, scrolled both ways. */
function DiagramViewer({
	source,
	onClose,
	returnFocusRef,
}: {
	source: string;
	onClose: () => void;
	returnFocusRef: RefObject<HTMLElement | null>;
}) {
	const { dialogRef, closeRef } = useModalLayer({ open: true, onClose, returnFocusRef });
	const mode: DiagramMode = { kind: "full" };
	const [result, setResult] = useState<DrawResult | undefined>(() => peekDrawing(source, mode));
	useEffect(() => {
		let live = true;
		void drawDiagram(source, { kind: "full" }).then((r) => {
			if (live) setResult(r);
		});
		return () => {
			live = false;
		};
	}, [source]);
	const node = useSvgNode(result);
	return createPortal(
		<div
			ref={dialogRef}
			role="dialog"
			aria-modal="true"
			aria-label={messageRuntime.t("relay.diagram.view")}
			data-diagram-viewer
			className="fixed inset-0 z-[100] flex flex-col bg-surface"
			style={PELE_THEME_STYLE}
		>
			<div
				className="flex shrink-0 items-center justify-between border-b border-hairline ps-3"
				style={{ paddingTop: "env(safe-area-inset-top)" }}
			>
				<span className="text-heading font-semibold text-ink">{messageRuntime.t("relay.diagram.view")}</span>
				<button
					ref={closeRef}
					type="button"
					onClick={onClose}
					aria-label={messageRuntime.t("relay.diagram.close")}
					className="flex min-h-11 min-w-11 items-center justify-center text-ink-muted active:bg-elevated"
				>
					✕
				</button>
			</div>
			{/* `touch-action` is left at its default: one finger pans this area in both
			    directions and two fingers still pinch-zoom the page. */}
			<div dir="ltr" data-testid="diagram-viewer-scroll" className="min-h-0 flex-1 overflow-auto p-3">
				{node ? (
					<SvgHost node={node} className="w-max min-w-full [&_svg]:max-w-none! [&_svg]:h-auto" />
				) : null}
			</div>
		</div>,
		document.body,
	);
}

export function DiagramBlock({ source, fallback }: { source: string; fallback: ReactNode }) {
	const box = useRef<HTMLDivElement>(null);
	const clip = useRef<HTMLDivElement>(null);
	const openRef = useRef<HTMLButtonElement>(null);
	const [width, setWidth] = useState(0);
	const [result, setResult] = useState<DrawResult | undefined>(undefined);
	const [clipped, setClipped] = useState(false);
	const [open, setOpen] = useState(false);

	/* A drawing replaces a code block of a different height, after the transcript
	   has already settled. A reader who was at the tail (the usual case: the answer
	   just arrived) would be left above the fold with the diagram cut off, so the
	   swap re-pins them. A reader scrolled up into history is never moved. */
	const wasAtTail = useRef(false);
	const commit = (next: DrawResult) => {
		const scroller = scrollerOf(box.current);
		wasAtTail.current = scroller !== null && scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < TAIL_PX;
		setResult(next);
	};
	useLayoutEffect(() => {
		if (!wasAtTail.current) return;
		wasAtTail.current = false;
		const scroller = scrollerOf(box.current);
		if (scroller) scroller.scrollTop = scroller.scrollHeight;
	}, [result]);

	/* The column's width, bucketed so a pixel of movement never redraws. */
	useLayoutEffect(() => {
		const el = box.current;
		if (!el) return;
		const measure = () => setWidth(bucketWidth(el.clientWidth));
		measure();
		if (typeof ResizeObserver === "undefined") return;
		const observer = new ResizeObserver(measure);
		observer.observe(el);
		return () => observer.disconnect();
	}, []);

	useLayoutEffect(() => {
		if (width < WIDTH_STEP) return;
		const mode: DiagramMode = { kind: "fit", width };
		const hit = peekDrawing(source, mode);
		if (hit) {
			commit(hit);
			return;
		}
		let live = true;
		void drawDiagram(source, mode).then((r) => {
			if (live) commit(r);
		});
		return () => {
			live = false;
		};
	}, [source, width]);

	const node = useSvgNode(result);
	const drawn = result?.ok === true && node !== null;

	useLayoutEffect(() => {
		const el = clip.current;
		if (!el) {
			setClipped(false);
			return;
		}
		const check = () => setClipped(el.scrollHeight > el.clientHeight + 1);
		check();
		if (typeof ResizeObserver === "undefined") return;
		const observer = new ResizeObserver(check);
		observer.observe(el);
		return () => observer.disconnect();
	}, [node]);

	const explicitFailure = result && !result.ok ? result.message : null;
	const unusable = result?.ok && node === null ? "invalid SVG" : null;
	const failure = explicitFailure ?? unusable;

	let body: ReactNode;
	let state: "pending" | "drawn" | "error";
	if (failure !== null) {
		state = "error";
		body = <DiagramErrorCard message={failure} source={fallback} />;
	} else if (drawn && node) {
		state = "drawn";
		body = (
			<figure
				role="group"
				aria-label={messageRuntime.t("relay.diagram.view")}
				className="min-w-0 overflow-hidden rounded-sm border border-hairline bg-surface"
				style={PELE_THEME_STYLE}
			>
				<div
					ref={clip}
					dir="ltr"
					data-testid="diagram-tap"
					onClick={() => setOpen(true)}
					className="relative cursor-zoom-in overflow-hidden p-2"
					style={{ maxHeight: INLINE_MAX_HEIGHT }}
				>
					<SvgHost node={node} className="[&_svg]:mx-auto [&_svg]:block" />
					{clipped ? (
						<div
							aria-hidden="true"
							data-testid="diagram-fade"
							className="pointer-events-none absolute inset-x-0 bottom-0 h-12 bg-linear-to-t from-surface to-transparent"
						/>
					) : null}
				</div>
				<div className="flex flex-wrap border-t border-hairline">
					<CopyButton source={source} />
					<button
						type="button"
						data-testid="diagram-save"
						onClick={() => saveSvg(node)}
						className={cn(BUTTON, "border-s border-hairline")}
					>
						{messageRuntime.t("relay.diagram.save")}
					</button>
					<button
						ref={openRef}
						type="button"
						data-testid="diagram-full-size"
						onClick={() => setOpen(true)}
						className={cn(BUTTON, "border-s border-hairline")}
					>
						{messageRuntime.t("relay.diagram.full_size")}
					</button>
				</div>
			</figure>
		);
	} else {
		state = "pending";
		body = fallback;
	}

	return (
		<div ref={box} data-diagram-state={state} className="min-w-0">
			{body}
			{open && drawn ? <DiagramViewer source={source} onClose={() => setOpen(false)} returnFocusRef={openRef} /> : null}
		</div>
	);
}
