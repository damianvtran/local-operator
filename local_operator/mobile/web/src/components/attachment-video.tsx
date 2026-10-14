/**
 * One attachment video with designed resolving and failure states — the
 * player half of the attachment contract's follow-up design (§3.2), and the
 * first mount that needs the RANGE-capable attachment route.
 *
 * WHY A RESERVED BOX AND NOT A SPINNER. The surface's no-spinner rule: the
 * bytes are fetched by the BROWSER's media stack (preload="metadata"), not
 * by this component, and a spinner would be this surface claiming motion it
 * does not own. The reserved 16:9 box IS the resolving state. The box is one
 * fixed shape across resolving → ready → missing, so the card never reflows
 * when the bytes land or fail (the image path's reserved-frame rule, kept).
 *
 * MOUNTING IS THE FETCH. Nothing here requests bytes: the <video> element
 * renders when the component mounts (only the card's done body mounts it),
 * its preload is metadata, and the element then issues its own Range
 * requests against the attachment route — the one path this surface has for
 * video bytes (no second transport, no inlined base64). A miss degrades to
 * the same ⊘ + state line treatment `attachment-image.tsx` uses for
 * `image unavailable`, with the material's own word: `video unavailable`.
 */
import { useState } from "react";
import { attachmentUrl } from "../api";
import { cn } from "../lib/cn";

export function AttachmentVideo({
	pid,
	entryId,
	index,
}: {
	pid: string;
	entryId: string;
	index: number;
}) {
	const [state, setState] = useState<"resolving" | "ready" | "error">(
		"resolving",
	);
	/* The reserved frame: 16:9 on the sunken ground, width-capped so a wide
	   clip lettersboxes inside it (object-contain) instead of pushing the
	   card past the viewport. The test id + data-state are shipped markup,
	   not harness-only hooks — the capture rig's geometry dump reads them to
	   tell the three states apart in a still (same reason the hold's test id
	   exists). */
	const frame =
		"relative block aspect-video w-full max-w-80 overflow-hidden rounded-sm border border-hairline bg-sunken";
	if (state === "error") {
		return (
			<div
				data-testid="attachment-video"
				data-state="error"
				className={cn(frame, "flex flex-col items-center justify-center gap-1")}
			>
				<span aria-hidden className="text-body text-ink-dim">
					⊘
				</span>
				<span className="text-meta text-ink-dim">video unavailable</span>
			</div>
		);
	}
	return (
		<span data-testid="attachment-video" data-state={state} className={frame}>
			<video
				src={attachmentUrl(pid, entryId, index, "video")}
				aria-label="attachment"
				controls
				preload="metadata"
				onLoadedMetadata={() => setState("ready")}
				onError={() => setState("error")}
				className={cn(
					"h-full w-full object-contain",
					/* THE RESOLVING STATE IS THE BOX, NOT THE ELEMENT: the media
					   stack draws its OWN load UI into a buffering <video> (Chrome
					   paints a spinner + dead controls), which would be a spinner
					   on this surface's resolving face — the exact thing the
					   no-spinner rule excludes. Hidden (never display:none, which
					   can cancel the load) until metadata lands: the component's
					   sibling does the same while an image's bytes are in flight. */
					state === "resolving" && "invisible",
				)}
			/>
		</span>
	);
}
