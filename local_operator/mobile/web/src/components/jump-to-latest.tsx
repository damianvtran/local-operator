/**
 * "JUMP TO LATEST" — the way back to the tail of a transcript the reader has
 * scrolled away from.
 *
 * WHY IT IS OUTSIDE THE SCROLLER, NOT A ROW INSIDE IT. Two reasons, both
 * structural. A row in the flow would change `scrollHeight` when it appeared,
 * so the transcript's anchor hold (`lib/scroll-follow`) would measure a delta
 * the reader never asked for, and the chip itself would scroll away with the
 * content it is meant to leave. The chip is therefore absolutely positioned in
 * a wrapper around the scroller: showing and hiding it moves nothing.
 *
 * WHY THE ABSOLUTE BOX IS A SEPARATE, EMPTY-TAP WRAPPER. `.lo-hit` gives a
 * 32 px pill its 44 px target with `position: relative` (`styles/index.css`),
 * and that rule is unlayered CSS, so it BEATS Tailwind's `absolute` utility
 * whatever order the classes are written in — measured on the real bundle: a
 * `lo-hit absolute` button computed `position: relative`, sat in the flex
 * flow, pushed the working line down and rendered full-width at the left edge.
 * So the positioning lives on a `pointer-events-none` wrapper, the button stays
 * a normal in-flow chip inside it, and the wrapper's missing pointer events are
 * what keeps a tap OUTSIDE the button's own box going to the transcript.
 *
 * WHERE IT SITS: the conversation viewport's bottom-right (logical `end`, so
 * Urdu mirrors it), 12 px in — the transcript's own `px-3` — and 8 px above its
 * bottom edge. `bg-elevated` + `shadow-overlay` is the floating ground the
 * snackbar already uses; over prose, a plate that is only `canvas` would
 * dissolve into it.
 *
 * NARROW FALLBACK: at 380 px and below the label shortens to one word. The
 * accessible name stays the full sentence from `aria-label` (the shortened
 * visible word is its suffix, so label-in-name still holds).
 */
import { messageRuntime } from "../store";

export function JumpToLatest({ onJump }: { onJump: () => void }) {
	return (
		<div className="pointer-events-none absolute inset-x-0 bottom-2 z-10 flex justify-end px-3">
			<button
				type="button"
				aria-label={messageRuntime.t("relay.transcript.jump.latest")}
				onClick={onJump}
				className="lo-hit pointer-events-auto inline-flex h-8 max-w-full min-w-0 select-none items-center gap-1 rounded-full bg-elevated px-3 text-body-sm text-ink shadow-overlay active:bg-sunken"
			>
				<span aria-hidden>↓</span>
				<span className="min-w-0 truncate max-[380px]:hidden">{messageRuntime.t("relay.transcript.jump.latest")}</span>
				<span className="hidden min-w-0 truncate max-[380px]:inline">{messageRuntime.t("relay.transcript.jump.latest_short")}</span>
			</button>
		</div>
	);
}
