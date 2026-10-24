/**
 * One Library tile.
 *
 * Two decisions worth naming:
 *
 * 1. **The thumbnail mounts lazily.** A catalog can hold thousands of assets and
 *    the list only paints a window of them; firing every `/thumbnail` on first
 *    paint would be a request storm against a relay whose own list is budgeted.
 *    The IntersectionObserver gate is the transcript's own pattern
 *    (`supplement-block.tsx`), one viewer per card, disconnected on unmount.
 * 2. **A failed thumbnail is not a failed card.** The bytes may legitimately be
 *    gone (a moved file, a digest never materialised) and the row's metadata is
 *    still true. The image falls back to the kind glyph, never a broken-image
 *    icon, and the card stays tappable.
 *
 * Everything visual is a `@theme` role token — no hex, no palette assumption —
 * so the same tile renders on both themes without a conditional.
 */
import { useEffect, useRef, useState, type RefObject } from "react";
import { libraryBytesUrl, libraryThumbUrl, libraryWaveformUrl } from "../../api";
import { cn } from "../../lib/cn";
import { formatBytes, kindGlyph, kindLabel, originLabel, sourceTitle } from "../../lib/library";
import { SoundPlayer, useWaveformPeaks } from "../sound-player";
import type { LibraryAsset } from "../../types";

/** Mount `children` only after the element approaches the viewport. Falls back
    to "visible" when IntersectionObserver is absent (an old WebView, a test
    host), so a cell never stays blank because a browser API is missing. */
function useNearViewport<T extends Element>(): [RefObject<T | null>, boolean] {
	const ref = useRef<T | null>(null);
	const [near, setNear] = useState(false);
	useEffect(() => {
		if (!("IntersectionObserver" in globalThis)) {
			setNear(true);
			return;
		}
		const node = ref.current;
		if (!node) return;
		const observer = new IntersectionObserver(
			(entries) => {
				if (entries.some((entry) => entry.isIntersecting)) {
					setNear(true);
					observer.disconnect();
				}
			},
			{ rootMargin: "240px" },
		);
		observer.observe(node);
		return () => observer.disconnect();
	}, []);
	return [ref, near];
}

const ORIGIN_CLASS: Record<string, string> = {
	generated: "border-info-border bg-info-wash text-info",
	created: "border-accent-wash bg-accent-wash text-accent",
	mentioned: "border-hairline bg-sunken text-ink-muted",
};

export function AssetCard({
	asset,
	onOpen,
	onMenu,
}: {
	asset: LibraryAsset;
	onOpen: () => void;
	onMenu: () => void;
}) {
	const [thumbRef, near] = useNearViewport<HTMLDivElement>();
	const [thumbFailed, setThumbFailed] = useState(false);
	const isImage = asset.kind === "image" && !thumbFailed;
	const isAudio = asset.kind === "audio";
	/* Lazy like the thumbnail: a catalog can hold thousands of rows, and a sound's
	   peaks are a request per mounted tile. `near` only ever turns on. */
	const peaks = useWaveformPeaks(libraryWaveformUrl(asset.id), isAudio && near);
	const subtitle = [kindLabel(asset.kind), formatBytes(asset.size_bytes)].filter(Boolean).join(" · ");
	const source = sourceTitle(asset);
	const shownTags = asset.tags.slice(0, 2);
	const extraTags = asset.tags.length - shownTags.length;

	return (
		<article className="relative flex min-w-0 flex-col overflow-hidden rounded-md border border-hairline bg-surface">
			<div ref={thumbRef} className="relative aspect-[4/3] w-full bg-sunken">
				{isAudio ? (
					/* THE ONE MOBILE PLAYER, seated in the tile's own 4:3 well instead
					   of an audio glyph. It is a SIBLING ABOVE the inset-0 open button
					   (z-10): a press on play must play, never open the asset, while a
					   press anywhere else in the well still opens it. Mounted only once
					   the tile approaches the viewport — same gate as the thumbnail. */
					<div
						className="absolute inset-x-2 top-1/2 z-10 -translate-y-1/2"
						data-sound-seat=""
					>
						{near ? (
							<SoundPlayer
								variant="compact"
								src={libraryBytesUrl(asset.id)}
								peaks={peaks}
								label={asset.name}
								durationHint={asset.duration_s ?? 0}
							/>
						) : null}
					</div>
				) : isImage && near ? (
					<img
						src={libraryThumbUrl(asset.id)}
						alt=""
						loading="lazy"
						decoding="async"
						className="size-full object-cover"
						onError={() => setThumbFailed(true)}
					/>
				) : (
					<span
						className="flex size-full items-center justify-center font-mono text-mono text-ink-dim"
						aria-hidden
					>
						{kindGlyph(asset.kind)}
					</span>
				)}
				{asset.highlighted ? (
					<span
						className="absolute left-1 top-1 rounded-xs bg-canvas/80 px-1 text-meta text-accent"
						aria-label="highlighted"
					>
						★
					</span>
				) : null}
				<span
					className={cn(
						"absolute right-1 top-1 rounded-xs border px-1 text-meta",
						ORIGIN_CLASS[asset.origin_group] ?? ORIGIN_CLASS.mentioned,
					)}
				>
					{originLabel(asset.origin_group)}
				</span>
				<button
					type="button"
					onClick={onOpen}
					aria-label={`open ${asset.name}`}
					className="absolute inset-0 rounded-md focus-visible:outline-offset-[-4px]"
				/>
			</div>
			<div className="flex min-w-0 flex-col gap-1 px-2 py-1.5">
				<p className="line-clamp-2 text-body-sm font-medium text-ink">{asset.name}</p>
				<p className="truncate text-meta text-ink-muted">{subtitle}</p>
				{source ? (
					<p className="truncate pr-8 text-meta text-ink-dim">
						<span aria-hidden>↳ </span>
						{source}
					</p>
				) : null}
				{shownTags.length ? (
					<div className="flex min-w-0 flex-wrap gap-1">
						{shownTags.map((tag) => (
							<span
								key={tag.id}
								className="max-w-full truncate rounded-xs bg-elevated px-1 text-meta text-ink-muted"
							>
								{tag.name}
							</span>
						))}
						{extraTags > 0 ? <span className="text-meta text-ink-dim">+{extraTags}</span> : null}
					</div>
				) : null}
			</div>
			<button
				type="button"
				onClick={onMenu}
				aria-label={`actions for ${asset.name}`}
				className="absolute bottom-0 right-0 flex min-h-11 min-w-11 items-center justify-center rounded-sm text-body text-ink-muted active:bg-elevated"
			>
				⋯
			</button>
		</article>
	);
}
