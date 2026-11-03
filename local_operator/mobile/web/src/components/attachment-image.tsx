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
import { useEffect, useState } from "react";
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
	useEffect(() => {
		if (!open) return;
		const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
		window.addEventListener("keydown", onKey);
		return () => window.removeEventListener("keydown", onKey);
	}, [open]);
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
		{open ? (
			<div
				role="dialog"
				aria-modal="true"
				aria-label="attachment"
				data-attachment-viewer
				onClick={() => setOpen(false)}
				className="fixed inset-0 z-50 flex items-center justify-center bg-black/90 p-2"
			>
				<img src={imageUrl(pid, entryId, index)} alt="attachment full size" className="max-h-full max-w-full object-contain" />
				<button
					type="button"
					aria-label="close"
					onClick={() => setOpen(false)}
					className="absolute end-3 top-3 flex h-11 w-11 items-center justify-center rounded-full bg-black/60 text-white"
				>
					✕
				</button>
			</div>
		) : null}
		<span
			role="button"
			tabIndex={0}
			aria-label="open attachment full screen"
			onClick={() => state === "loaded" && setOpen(true)}
			onKeyDown={(e) => {
				if (state === "loaded" && (e.key === "Enter" || e.key === " ")) {
					e.preventDefault();
					setOpen(true);
				}
			}}
			className={cn(
				"relative block h-40 cursor-zoom-in overflow-hidden rounded-sm border border-hairline",
				state === "loading" && "w-40 bg-sunken",
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
