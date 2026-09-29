/**
 * Session status glance — spend and context, read-only.
 *
 * The phone's counterpart to the desktop strip's two readings (that repo's
 * `features/chat/session-status/session-cost.ts` + `session-context.ts`),
 * adapted to a 390-px touch layout. Phase 1 of the mobile parity program:
 * nothing here is interactive, so it adds no tap targets and no flow.
 *
 * PLACEMENT. The desktop tucks its readings into the composer's button row;
 * on a phone that row is a touch surface (attach / model / mic / send), so the
 * readings would either crowd a control strip or invite taps. The header is
 * also width-starved — the name truncates between four controls. This row sits
 * under the header, styled like the gate receipt's line (`text-meta` scale, a
 * hairline bottom border), which is the smallest always-visible footprint: it
 * costs one text line, and it disappears ENTIRELY when there is nothing to
 * say. A fresh session shows neither `$0.0000` nor `0%` — both spellings are
 * refusals the Python sources already make (`_spend_text`'s zero policy and
 * `context_spelling`'s empty), so "no row" here is the same decision, not a
 * second one.
 *
 * COLOUR AND WEIGHT. The context reading warms with its rung
 * (`contextSemanticColor`, a union of the absolute and proportional ladders),
 * the way the TUI band's one coloured segment and the desktop ring do — it is
 * the single reading whose colour carries information, and it is the only
 * cell that also carries a non-colour step: the warm rungs are weighted with
 * the band's own colour-vision remedy (see `RUNG_CLASS`). The base rung stays
 * muted; `label` takes the accent; `danger` the danger role. Spend stays
 * muted: it is a fact, not an alarm.
 *
 * The estimate marker is the WORD `estimate`, rendered as its own dim segment
 * beside the reading — the desktop strip's convention, per the design round's
 * D2 ruling. See `contextEstimateMarker` for both precedents and the ruling.
 */
import { cn } from "../lib/cn";
import {
	contextEstimateMarker,
	contextReading,
	sessionSpend,
	type ContextRung,
} from "../lib/spend-context";
import type { SessionProjection } from "../types";

/**
 * The TUI's semantic rungs mapped onto this surface's roles — each entry
 * carries the non-colour half of the rung as well.
 *
 * WEIGHT IS THE COLOUR-VISION CARRIER, ported from the band's own rule
 * (`status_line.py`): the context reading is bold on every rung EXCEPT the
 * base one (`bold=semantic != CONTEXT_COLOR_BASE`), because hue alone cannot
 * carry the step — for the commonest colour-vision deficiency the label→danger
 * step collapses, so without a second carrier the warm rungs do not exist for
 * those readers. Weight is orthogonal to hue and costs no cells; the base rung
 * stays regular so "warm" remains the marked state rather than the default.
 * 600, not 700, because it is this app's emphasis step (the type ramp's
 * display/title/heading weights are all 600) and it resolves to a real heavier
 * face on the OS mono stack — measured in the re-captured frames.
 */
const RUNG_CLASS: Record<ContextRung, string> = {
	signal: "text-ink-muted",
	label: "text-accent font-semibold",
	danger: "text-danger font-semibold",
};

export function SessionStatus({ projection }: { projection: SessionProjection }) {
	const spend = sessionSpend(projection, projection.usage);
	const context = contextReading(projection);
	const estimateMarker = contextEstimateMarker(context);
	if (!spend.text && !context.spelling) return null;
	return (
		<div
			data-testid="session-status"
			className="flex items-center gap-2 border-b border-hairline px-3 py-1 font-mono text-mono-sm tabular-nums"
		>
			{/* Reading order is the one both references use — the TUI band's right
			    group and the desktop strip both read context before spend — and
			    `ml-auto` sits on the SECOND cell rather than on `justify-between`,
			    so a lone reading keeps a fixed edge and the row does not reflow as
			    the other appears or vanishes. */}
			{context.spelling ? (
				<span
					data-testid="session-status-context"
					className={cn("shrink-0", RUNG_CLASS[context.rung])}
				>
					{context.spelling}
					{/* The estimate marker: the desktop's dim WORD, not a glyph —
					    D2's ruling; `contextEstimateMarker` names both precedents. */}
					{estimateMarker ? (
						<span className="ml-1 text-ink-dim">{estimateMarker}</span>
					) : null}
				</span>
			) : null}
			{spend.text ? (
				<span
					data-testid="session-status-spend"
					className="ml-auto min-w-0 truncate text-ink-muted"
				>
					{spend.text}
				</span>
			) : null}
		</div>
	);
}
