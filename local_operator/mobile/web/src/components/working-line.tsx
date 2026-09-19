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
import { useEffect, useState } from "react";
import { formatElapsed } from "../lib/format";

/* The exact tuple the TUI's WorkingBlock and status band cycle. */
const SPINNER = ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"];

export function WorkingLine({
	activity,
	startedS,
}: {
	activity: string;
	startedS: number | null;
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

	if (!activity) return null;
	return (
		<div
			className="flex items-center gap-2 px-3 py-1.5 text-body-sm"
			aria-live="polite"
			aria-busy="true"
		>
			<span
				className="shrink-0 font-mono text-mono text-accent"
				aria-hidden
			>
				{SPINNER[frame % SPINNER.length]}
			</span>
			<span className="lo-shimmer min-w-0 flex-1 truncate text-ink-muted">
				{activity}
			</span>
			{/* The clock's SLOT is reserved, not measured: ``w-[6ch]`` is the width of
			 * the widest form the formatter can produce (``59m59s``/``41d16h``/
			 * ``100d+``), right-aligned so the digits grow leftward into their own
			 * space. The TUI reserves ``WorkingBlock._CLOCK_COL`` for exactly this
			 * reason — an unreserved clock re-clips the label as the number changes
			 * form, so at 320px the label read 38 characters at ``1h`` and 28 at
			 * ``1000h 40m`` and jumped 5 characters in the single tick across the
			 * ``59m 59s`` → ``1h`` crossing (design round 1 D2). Reserved cells also
			 * make withholding free: an empty slot is the same width as a full one. */}
			<span className="w-[6ch] shrink-0 text-right font-mono text-mono-sm text-ink-dim tabular-nums">
				{hasClock ? formatElapsed(elapsed) : ""}
			</span>
		</div>
	);
}
