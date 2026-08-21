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
	startedS: number;
}) {
	const [frame, setFrame] = useState(0);
	const [elapsed, setElapsed] = useState(startedS);

	/* Re-seed the clock when the server sends a new phase or a fresh age. */
	useEffect(() => {
		setElapsed(startedS);
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
			<span className="shrink-0 font-mono text-mono-sm text-ink-dim tabular-nums">
				{formatElapsed(elapsed)}
			</span>
		</div>
	);
}
