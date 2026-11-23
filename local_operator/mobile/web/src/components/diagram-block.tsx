/**
 * A closed ```mermaid fence, drawn with Pele.
 *
 * STATES. Until the drawing exists the block shows the plain code block it
 * replaces, and it keeps showing it if the renderer's chunk could not be fetched
 * (a tab left open across a deploy): that says nothing about the diagram, so no
 * error card, and the next mount tries again. A drawing replaces the code block; a
 * parse failure puts an error card above it with the engine's own line, because a
 * diagram that cannot be drawn is still something the reader can read.
 *
 * FIT, THEN FULL SIZE. The transcript has no horizontal play (operator finding
 * B), so the inline drawing is laid out for the column's width (Pele reflows a
 * wide left-to-right chart top-to-bottom under 640 px). Its height is clamped
 * only for a genuinely tall drawing (`diagram-clip.ts`: a cap plus a tolerance, so
 * a small chart is never sliced); a clamped drawing gets a fade AND a visible
 * "Full size" pill over it, besides the footer button. Full size
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
import { CONTENT_RESIZED_EVENT } from "../lib/content-resized";
import {
	WIDTH_STEP,
	bucketWidth,
	drawDiagram,
	peekDrawing,
	type DiagramMode,
	type DrawResult,
} from "../lib/diagram-pele";
import { inlineMaxHeight, mustClamp } from "../lib/diagram-clip";
import { sanitizeSvg, serializeSvg } from "../lib/diagram-svg";
import { PELE_THEME_STYLE, resolvedThemeDeclarations } from "../lib/diagram-theme";
import { messageRuntime } from "../store";
import { useModalLayer } from "./ui/use-modal-layer";

/** How long "Copied" stays up. */
const COPIED_MS = 2000;

/** Border (1px x 2) and padding (8px x 2) around the drawing inside the card. */
const CARD_CHROME_PX = 18;

/** The error card shows the engine's message as text; a parse error can quote tens of thousands of characters. */
const MESSAGE_MAX = 400;

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

/* `message` is null when there is no engine text to quote (the drawing came back
   as something that is not an SVG): the localized title then stands alone,
   rather than an app-authored English sentence in every locale (R3 F5). */
export function DiagramErrorCard({ message, source }: { message: string | null; source: ReactNode }) {
	return (
		<div data-diagram-state="error" className="flex min-w-0 flex-col gap-1.5">
			<div role="alert" className="rounded-sm border border-danger-border bg-danger-wash px-3 py-2">
				<div className="text-body-sm font-semibold text-danger">{messageRuntime.t("relay.diagram.error.title")}</div>
				{/* The engine's own message, as TEXT: it quotes the diagram, and it is
				    the one line a reader can act on. English by design (a library's
				    diagnostic, not product copy). */}
				{message === null ? null : (
					<pre dir="ltr" className="mt-1 font-mono text-mono-sm whitespace-pre-wrap text-ink-muted [overflow-wrap:anywhere]">
						{message.length > MESSAGE_MAX ? `${message.slice(0, MESSAGE_MAX)}…` : message}
					</pre>
				)}
			</div>
			{source}
		</div>
	);
}

type Copy = "idle" | "copied" | "failed";

const BUTTON =
	"flex min-h-11 min-w-fit grow items-center justify-center px-3 text-body-sm text-ink-muted select-none active:bg-elevated focus-visible:-outline-offset-2!";
/* The `!` is load-bearing: the unlayered `html :focus-visible` rule in
   styles/index.css sets `outline-offset: 2px` and beats any layered utility, so
   without it the ring sat OUTSIDE the button and the figure's `overflow-hidden`
   (or the viewport edge, for the viewer's ✕) clipped it to one stroke (D2). */
const INSET_RING = "focus-visible:-outline-offset-2!";

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
	const labels: Record<Copy, string> = {
		idle: messageRuntime.t("relay.diagram.copy"),
		copied: messageRuntime.t("relay.diagram.copied"),
		failed: messageRuntime.t("relay.diagram.copy_failed"),
	};
	return (
		<button type="button" data-testid="diagram-copy" onClick={copy} aria-live="polite" className={BUTTON}>
			{/* One grid cell holds the live label and the other two, invisible, so the
			    button is as wide as the longest of the three in this locale and the row
			    never reflows when the label swaps (the ask card's trick; "Couldn't copy"
			    is the longest in seven of eight locales). */}
			<span className="grid">
				<span className="col-start-1 row-start-1">{labels[state]}</span>
				{(Object.keys(labels) as Copy[])
					.filter((key) => key !== state)
					.map((key) => (
						<span key={key} aria-hidden="true" className="invisible col-start-1 row-start-1">
							{labels[key]}
						</span>
					))}
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
	fallback,
	onClose,
	returnFocusRef,
}: {
	source: string;
	fallback: ReactNode;
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

	/* More drawing past the right or bottom edge? Said with a fade on that edge,
	   the cue the relay uses for clipped scroll content, so a wide drawing that
	   pans sideways does not look like it simply ends. */
	const scroller = useRef<HTMLDivElement>(null);
	const [more, setMore] = useState({ right: false, below: false });
	useLayoutEffect(() => {
		const el = scroller.current;
		if (!el) return;
		const measure = () =>
			setMore({
				right: el.scrollWidth - el.clientWidth - el.scrollLeft > 1,
				below: el.scrollHeight - el.clientHeight - el.scrollTop > 1,
			});
		measure();
		el.addEventListener("scroll", measure, { passive: true });
		const resize = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(measure);
		resize?.observe(el);
		return () => {
			el.removeEventListener("scroll", measure);
			resize?.disconnect();
		};
	}, [node]);

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
				style={{
					paddingTop: "env(safe-area-inset-top)",
					paddingInlineStart: "max(0.75rem, env(safe-area-inset-left))",
					paddingInlineEnd: "env(safe-area-inset-right)",
				}}
			>
				<span className="text-heading font-semibold text-ink">{messageRuntime.t("relay.diagram.view")}</span>
				<button
					ref={closeRef}
					type="button"
					onClick={onClose}
					aria-label={messageRuntime.t("relay.diagram.close")}
					className={cn("flex min-h-11 min-w-11 items-center justify-center text-ink-muted active:bg-elevated", INSET_RING)}
				>
					✕
				</button>
			</div>
			<div className="relative min-h-0 flex-1">
				{/* `touch-action` is left at its default: one finger pans this area in both
				    directions and two fingers still pinch-zoom the page. */}
				<div
					ref={scroller}
					dir="ltr"
					data-testid="diagram-viewer-scroll"
					className="h-full overflow-auto p-3"
					style={{
						paddingBottom: "max(0.75rem, env(safe-area-inset-bottom))",
						paddingLeft: "max(0.75rem, env(safe-area-inset-left))",
						paddingRight: "max(0.75rem, env(safe-area-inset-right))",
					}}
				>
					{/* The `full` layout is unbounded in width, so it can fail (e.g. Pele's
					    outputLimit) where the inline `fit` draw succeeded, and it is async on
					    a cold cache: say so, or show the source, never an empty pane (R3 F3).
					    A drawing shorter than the pane is centred vertically rather than
					    pinned to the top under 90% empty space (D3); `min-h-full` (not
					    `h-full`) lets a tall drawing grow the box and scroll, never clip. */}
					{node ? (
						<SvgHost
							node={node}
							className="flex min-h-full w-max min-w-full items-center [&_svg]:max-w-none! [&_svg]:h-auto"
						/>
					) : result === undefined ? (
						<div data-diagram-state="pending">{fallback}</div>
					) : (
						<DiagramErrorCard message={result.ok ? null : result.message} source={fallback} />
					)}
				</div>
				{more.right ? (
					<div
						aria-hidden="true"
						data-testid="diagram-viewer-fade-right"
						className="pointer-events-none absolute inset-y-0 right-0 w-10 bg-linear-to-l from-surface to-transparent"
					/>
				) : null}
				{more.below ? (
					<div
						aria-hidden="true"
						data-testid="diagram-viewer-fade-below"
						className="pointer-events-none absolute inset-x-0 bottom-0 h-10 bg-linear-to-t from-surface to-transparent"
					/>
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
	const [clamped, setClamped] = useState(false);
	const [open, setOpen] = useState(false);

	/* A drawing replaces a code block of a different height AFTER the transcript
	   has settled. Whether the reader is held or kept at the tail is the
	   transcript's call (it owns the pin state): this only says the row changed.
	   EVERY new result, not just the first: a rotation changes the width bucket,
	   the scroller's ResizeObserver re-pins at once, and the redraw at the new
	   width lands later with a new height that must be followed too (R3 F2). A
	   cache hit hands back the same object, so an unchanged drawing is silent. */
	useLayoutEffect(() => {
		if (result === undefined) return;
		box.current?.dispatchEvent(new CustomEvent(CONTENT_RESIZED_EVENT, { bubbles: true }));
	}, [result]);

	/* The column's width, bucketed so a pixel of movement never redraws. */
	useLayoutEffect(() => {
		const el = box.current;
		if (!el) return;
		/* A hidden or collapsed pane is 0 wide: draw nothing yet rather than a 8 px
		   drawing that is replaced the moment the pane shows. */
		const measure = () => setWidth(el.clientWidth < WIDTH_STEP ? 0 : bucketWidth(el.clientWidth - CARD_CHROME_PX));
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
			setResult(hit);
			return;
		}
		let live = true;
		void drawDiagram(source, mode).then((r) => {
			if (live) setResult(r);
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
			setClamped(false);
			return;
		}
		/* scrollHeight is the content's natural height whatever max-height is
		   set, so deciding to clamp from it cannot feed back into itself. */
		const check = () => {
			const remPx = parseFloat(getComputedStyle(document.documentElement).fontSize) || 16;
			const clamp = mustClamp(el.scrollHeight, remPx);
			setClamped(clamp);
			setClipped(clamp && el.scrollHeight > el.clientHeight + 1);
		};
		check();
		if (typeof ResizeObserver === "undefined") return;
		const observer = new ResizeObserver(check);
		observer.observe(el);
		return () => observer.disconnect();
	}, [node]);

	/* A chunk that would not load is not a verdict on the diagram: keep the code block. */
	const explicitFailure = result && !result.ok && !result.retryable ? result.message : null;
	const unusable = result?.ok === true && node === null;
	const failure = explicitFailure;

	let body: ReactNode;
	let state: "pending" | "drawn" | "error";
	if (unusable) {
		state = "error";
		body = <DiagramErrorCard message={null} source={fallback} />;
	} else if (failure !== null) {
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
					style={{ maxHeight: inlineMaxHeight(clamped) }}
				>
					<SvgHost node={node} className="[&_svg]:mx-auto [&_svg]:block" />
					{clipped ? (
						<div
							aria-hidden="true"
							data-testid="diagram-fade"
							className="pointer-events-none absolute inset-x-0 bottom-0 flex h-16 items-end justify-center bg-linear-to-t from-surface via-surface/80 to-transparent pb-2"
						>
							{/* The fade alone read as polish, not as "this is cut". The pill
							    names the way to the rest; it is decorative (the tap area above
							    and the footer button do the opening, so no second control or
							    duplicate accessible name) and reuses the footer's own label. */}
							<span
								data-testid="diagram-cut-hint"
								className="rounded-full border border-hairline bg-surface px-3 py-1 text-body-sm text-ink-muted"
							>
								{messageRuntime.t("relay.diagram.full_size")} ⤢
							</span>
						</div>
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
			{open && drawn ? <DiagramViewer source={source} fallback={fallback} onClose={() => setOpen(false)} returnFocusRef={openRef} /> : null}
		</div>
	);
}
