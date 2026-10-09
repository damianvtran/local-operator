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
import { useState } from "react";
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
		<span
			className={cn(
				"relative block h-40 overflow-hidden rounded-sm border border-hairline",
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
	);
}
