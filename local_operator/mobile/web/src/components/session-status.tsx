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
 * COLOUR. The context reading warms with its rung (`contextSemanticColor`, a
 * union of the absolute and proportional ladders), the way the TUI band's one
 * coloured segment and the desktop ring do — it is the single reading whose
 * colour carries information. The base rung stays muted; `label` takes the
 * accent; `danger` the danger role. Spend stays muted: it is a fact, not an
 * alarm.
 *
 * The estimate marker (`~`) is composed by `contextDisplaySpelling`, not
 * baked into the ported spelling — see that module for why.
 */
import { cn } from "../lib/cn";
import {
	contextDisplaySpelling,
	contextReading,
	sessionSpend,
	type ContextRung,
} from "../lib/spend-context";
import type { SessionProjection } from "../types";

/** The TUI's semantic rungs mapped onto this surface's roles. */
const RUNG_CLASS: Record<ContextRung, string> = {
	signal: "text-ink-muted",
	label: "text-accent",
	danger: "text-danger",
};

export function SessionStatus({ projection }: { projection: SessionProjection }) {
	const spend = sessionSpend(projection, projection.usage);
	const context = contextReading(projection);
	const contextText = contextDisplaySpelling(context);
	if (!spend.text && !contextText) return null;
	return (
		<div
			data-testid="session-status"
			className="flex items-center gap-2 border-b border-hairline px-3 py-1 font-mono text-mono-sm tabular-nums"
		>
			{/* `ml-auto` on the context cell rather than `justify-between`: with
			    only one reading present, each keeps its own edge and the row
			    does not reflow as the other appears or vanishes. */}
			{spend.text ? (
				<span data-testid="session-status-spend" className="min-w-0 truncate text-ink-muted">
					{spend.text}
				</span>
			) : null}
			{contextText ? (
				<span
					data-testid="session-status-context"
					className={cn("ml-auto shrink-0", RUNG_CLASS[context.rung])}
				>
					{contextText}
				</span>
			) : null}
		</div>
	);
}
