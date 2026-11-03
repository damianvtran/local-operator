/**
 * One inline attachment thumbnail with designed loading and failure states.
 *
 * A phone loads images over a flaky link, so the two off-happy-path states are
 * common, not edge cases, and both are designed rather than left to the
 * browser: a bare <img> would paint the native broken-image glyph on a 404
 * (which reads as a bug) and would reflow the bubble taller the instant bytes
 * decode (motion branding §7 rules out). Both are avoided by reserving a
 * fixed box up front and swapping a muted placeholder in on error.
 *
 * Lives in its own module because TWO paths render attachments now — the
 * transcript's user turn and the image-generation card's finished artifact —
 * and the frozen attachment contract says a finished image renders through
 * the surface's EXISTING image path. One component is that path; a copy in
 * the card would be a second implementation of the loading/error states
 * drifting from this one.
 *
 * The bytes are fetched lazily from the transcript keyed by the entry id
 * plus the image-only index (see `imageUrl` in api.ts) — never carried in
 * the projection. `pid` is the route param (the session id, despite the
 * name it keeps from the daemon's pid-keyed routes).
 */
import { useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useModalLayer } from "./ui/use-modal-layer";
import { imageUrl } from "../api";
import { cn } from "../lib/cn";

export function AttachmentImage({
	pid,
	entryId,
	index,
}: {
	pid: string;
	entryId: string;
	index: number;
}) {
	const [state, setState] = useState<"loading" | "loaded" | "error">("loading");
	/* Tap-to-enlarge, matching the desktop UI: the thumbnail is a 160px box, so
	   a phone user otherwise has no way to read a screenshot they sent. The
	   overlay is the same bytes at full size (the image route is immutable and
	   cached, so opening costs no second download). */
	const [open, setOpen] = useState(false);
	const thumbRef = useRef<HTMLSpanElement>(null);
	/* The reserved frame: a fixed height so the row never jumps when the
	   bytes arrive, capped width so a wide image cannot push the bubble past
	   the viewport. object-contain keeps aspect within the frame. */
	if (state === "error") {
		return (
			<div className="flex h-40 w-40 flex-col items-center justify-center gap-1 rounded-sm border border-hairline bg-sunken text-ink-dim">
				<span aria-hidden className="text-body">
					⊘
				</span>
				<span className="text-meta">image unavailable</span>
			</div>
		);
	}
	return (
		<>
		{open ? <Viewer pid={pid} entryId={entryId} index={index} onClose={() => setOpen(false)} returnFocusRef={thumbRef} /> : null}
		<span
			ref={thumbRef}
			role={state === "loaded" ? "button" : undefined}
			tabIndex={state === "loaded" ? 0 : undefined}
			aria-label={state === "loaded" ? "open image attachment full screen" : undefined}
			onClick={() => state === "loaded" && setOpen(true)}
			onKeyDown={(e) => {
				if (state === "loaded" && (e.key === "Enter" || e.key === " ")) {
					e.preventDefault();
					setOpen(true);
				}
			}}
			className={cn(
				"relative block h-40 overflow-hidden rounded-sm border border-hairline",
				state === "loading" && "w-40 bg-sunken",
				state === "loaded" && "cursor-zoom-in",
			)}
		>
			{state === "loading" ? (
				<span
					aria-hidden
					className="absolute inset-0 flex items-center justify-center text-meta text-ink-dim"
				>
					loading…
				</span>
			) : null}
			<img
				src={imageUrl(pid, entryId, index)}
				alt="attachment"
				onLoad={() => setState("loaded")}
				onError={() => setState("error")}
				className={cn(
					"h-40 max-w-full rounded-sm object-contain",
					state === "loading" && "invisible",
				)}
			/>
		</span>
		</>
	);
}

/* The full-screen viewer.

   PORTALED TO <body>, deliberately: the thumbnail lives inside the transcript's
   scroller, whose fade mask and overflow clip a `fixed` child to the message
   band (measured on the real build: the overlay covered only the area between
   header and composer, and the ✕ landed under the header's session menu).
   Unlike Sheet — which stays in the phone column so the screenshot surface
   contains it — a photo viewer wants the whole screen.

   `useModalLayer` is the app's one modal contract: it makes every sibling of
   the dialog's ancestors inert (with the portal that is the whole app root),
   moves focus to the ✕, traps Tab, closes on Escape and returns focus to the
   thumbnail. */
function Viewer({ pid, entryId, index, onClose, returnFocusRef }: {
	pid: string; entryId: string; index: number; onClose: () => void;
	returnFocusRef: React.RefObject<HTMLElement | null>;
}) {
	const { dialogRef, closeRef } = useModalLayer({ open: true, onClose, returnFocusRef });
	return createPortal(
		<div
			ref={dialogRef}
			role="dialog"
			aria-modal="true"
			aria-label="Image attachment"
			data-attachment-viewer
			onClick={onClose}
			className="fixed inset-0 z-[100] flex items-center justify-center bg-black/90 p-2"
		>
			<img src={imageUrl(pid, entryId, index)} alt="attachment full size" className="max-h-full max-w-full object-contain" />
			<button
				ref={closeRef}
				type="button"
				aria-label="close"
				onClick={onClose}
				style={{ top: "max(0.75rem, env(safe-area-inset-top))" }}
				className="absolute end-3 flex h-11 w-11 items-center justify-center rounded-full bg-black/60 text-white"
			>
				✕
			</button>
		</div>,
		document.body,
	);
}
