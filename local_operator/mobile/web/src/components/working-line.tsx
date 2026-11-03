/**
 * Working line — the ONE aggregate in-progress indicator, the phone's
 * counterpart to the TUI's WorkingBlock (branding §7 / D25: a single working
 * message, never a per-row spinner). It carries three things and no more:
 *
 *   - the braille spinner the TUI uses everywhere "running" is said (the
 *     status band, the subagent panel) — plain Unicode, no patched font;
 *   - the ACTIVITY label the model is doing ("thinking", "responding", or a
 *     running tool's intent), folded server-side from live events so the
 *     phone never invents one;
 *   - a clock: seconds since that phase began, ticking locally between the
 *     projection repaints that re-seed it.
 *
 * The shimmer sweep rides the label, not the row — motion says alive, the
 * clock says how long, and neither is a spinner beside a state line for the
 * same thing (which §7 forbids).
 */
import { cn } from "../lib/cn";
import { contentDir } from "../lib/content-direction";
import { useEffect, useState } from "react";
import { formatElapsed } from "../lib/format";
import { stopTurn, useTurnStopping } from "../lib/turn-stop";

/* The exact tuple the TUI's WorkingBlock and status band cycle. */
const SPINNER = ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"];

/* The turn's Stop on the line that says the turn is running. The composer's
 * primary disc is Stop only while the box is EMPTY (typing turns it into Steer),
 * and a phone has no Escape, so without this a user who has typed could not
 * stop the turn at all. It is shown whenever the line is, at the line's own
 * fixed 44 px height, so it neither moves nor appears with the draft. It is NOT
 * under the composer's arrival guard: it sits in a slot no other role ever
 * occupies, so no tap aimed elsewhere can land on it. A repeat tap while the
 * stop is pending is the shared latch's no-op (`lib/turn-stop`). */
function WorkingStop({ pid }: { pid: string }) {
	const stopping = useTurnStopping(pid, true);
	return (
		<button
			type="button"
			data-working-stop
			aria-label="stop the turn"
			aria-busy={stopping || undefined}
			disabled={stopping}
			onClick={() => void stopTurn(pid)}
			className="-me-1 flex min-h-11 min-w-11 shrink-0 items-center justify-center rounded-sm px-2 text-body-sm text-danger active:bg-danger-wash disabled:opacity-60"
		>
			{/* The label never changes: a longer "stopping…" widened this control 44 -> 79
			    px and pushed the clock (measured). The pressed state is aria-busy plus
			    the dimmed face. */}
			stop
		</button>
	);
}

export function WorkingLine({
	activity,
	startedS,
	stopPid,
}: {
	activity: string;
	startedS: number | null;
	/** The session whose turn the line offers to stop. Absent (a subagent's own
	    line) draws no control: the abort op addresses a session's turn. */
	stopPid?: string;
}) {
	const [frame, setFrame] = useState(0);
	const [elapsed, setElapsed] = useState(startedS ?? 0);
	/* The wire says whether an instant EXISTS, and that is the whole gate: `null`
	 * is a phase the server has no honest instant for (a label joined mid-flight
	 * whose producer stated none) and renders no digits, while `0.0` is a KNOWN
	 * zero — the phase edge the server watched begin — and renders `0s` from the
	 * first frame and counts up, exactly as the TUI's working block does
	 * (`transcript.py` `_clock = self._clock_text() if self._clock_known else ""`,
	 * with the zero being the phase's own start).
	 *
	 * The previous head gated on `startedS > 0` instead, which conflated the two:
	 * every phase edge publishes 0.0, so the band went blank for the whole life of
	 * any phase the phone watched begin — including a single running tool call,
	 * the "is this stuck?" reading the clock exists for (review round 2, MAJOR 1).
	 * Withholding is now exactly "the server said it has no instant".
	 *
	 * Either way the SLOT stays reserved by the width class below, so withholding
	 * cannot reflow the label beside it — the same cells the TUI keeps reserved
	 * when its own clock is withheld. */
	const hasClock = startedS !== null;

	/* Re-seed the clock when the server sends a new phase or a fresh age. */
	useEffect(() => {
		setElapsed(startedS ?? 0);
	}, [activity, startedS]);

	useEffect(() => {
		const spin = window.setInterval(() => setFrame((f) => f + 1), 80);
		const tick = window.setInterval(
			() => setElapsed((s) => Math.round((s + 1) * 10) / 10),
			1000,
		);
		return () => {
			window.clearInterval(spin);
			window.clearInterval(tick);
		};
	}, []);

	/* With a Stop the line is drawn even before a label exists: "a turn is alive"
	 * is the condition that must always have its Stop in reach. */
	if (!activity && !stopPid) return null;
	return (
		<div
			className={cn(
				"flex items-center gap-2 px-3 text-body-sm",
				stopPid ? "min-h-11" : "py-1.5",
			)}
			aria-live="polite"
			aria-busy="true"
		>
			<span
				className="shrink-0 font-mono text-mono text-accent"
				aria-hidden
			>
				{SPINNER[frame % SPINNER.length]}
			</span>
			<span dir={contentDir(activity)} className="lo-shimmer min-w-0 flex-1 truncate text-ink-muted">
				{activity}
			</span>
			{/* The clock's SLOT is reserved, not measured: ``w-[6ch]`` is the width of
			 * the widest form the formatter can produce (``59m59s``/``41d16h``/
			 * ``100d+``), end-aligned so the digits grow leftward into their own
			 * space. The TUI reserves ``WorkingBlock._CLOCK_COL`` for exactly this
			 * reason — an unreserved clock re-clips the label as the number changes
			 * form, so at 320px the label read 38 characters at ``1h`` and 28 at
			 * ``1000h 40m`` and jumped 5 characters in the single tick across the
			 * ``59m 59s`` → ``1h`` crossing (design round 1 D2). Reserved cells also
			 * make withholding free: an empty slot is the same width as a full one. */}
			<span className="w-[6ch] shrink-0 text-end font-mono text-mono-sm text-ink-dim tabular-nums">
				{hasClock ? formatElapsed(elapsed) : ""}
			</span>
			{stopPid ? <WorkingStop pid={stopPid} /> : null}
		</div>
	);
}
