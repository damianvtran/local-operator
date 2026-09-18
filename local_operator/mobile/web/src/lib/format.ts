/**
 * The two formatters the UI needs, hand-rolled (no date library):
 * elapsed durations for tool rows / subagents, and a relative timestamp
 * for the past-sessions list.
 */

/** Active processing time: `9s`, `41m1s`, `1h2m`, `4d5h` — the TUI's port.
 *
 * A port of `local_operator/tui/widgets/tool_card.py::format_duration`, branch
 * for branch, because both surfaces print the same elapsed time for the same
 * work and two spellings of one number is two answers to one question (design
 * round 1 on the phone clock, D3/D5). Units are dropped once they stop carrying
 * information: past an hour the seconds are noise, and a whole minute renders
 * as `5m` rather than `5m0s`. Sub-second work renders as `0s` rather than
 * vanishing, so a finished turn always leaves a mark.
 *
 * BOUNDED AT SIX CELLS over the whole domain, which the callers reserve room
 * for rather than measure: the widest strings are `59m59s`, `23h59m` and
 * `99d23h`, and from 100 days it is `100d+`. It used to compute
 * `hours = floor(minutes / 60)` with no ceiling and no days branch, so
 * `100h 40m` was 8 characters and `1000h 40m` was 9 — which is what put the
 * phone's clock past the edge of a narrow row and pushed the band's label
 * `1000h 40m` wide enough to re-clip 5 characters in one tick. The days branch
 * is also the more readable answer at that magnitude, and the TUI's reason for
 * keeping it a branch rather than a clamp holds here identically: clipping
 * `100h40m` to fit renders `100h4…`, and `100h4m`, `100h40m` and `100h45m` all
 * collapse to that same string. Prose survives truncation because the reader
 * reconstructs it; a duration does not — and this number is load-bearing
 * exactly when it is largest.
 *
 * The non-finite/negative guard is the phone's own (a wire value is untrusted;
 * the TUI's callers filter with `parse_duration` before it gets there), and an
 * empty string is the same "no duration to state" the callers already render
 * as nothing.
 */
export function formatElapsed(seconds: number): string {
	if (!Number.isFinite(seconds) || seconds < 0) return "";
	const total = Math.trunc(seconds);
	if (total < 60) return `${total}s`;
	if (total < 3600) {
		const minutes = Math.floor(total / 60);
		const secs = total % 60;
		return secs ? `${minutes}m${secs}s` : `${minutes}m`;
	}
	if (total < 86400) {
		const hours = Math.floor(total / 3600);
		const minutes = Math.floor((total % 3600) / 60);
		return minutes ? `${hours}h${minutes}m` : `${hours}h`;
	}
	const days = Math.floor(total / 86400);
	if (days > 99) return "100d+";
	const hours = Math.floor((total % 86400) / 3600);
	return hours ? `${days}d${hours}h` : `${days}d`;
}

/** Epoch seconds → "just now" / "4m ago" / "2h ago" / "3d ago" / "mar 3". */
export function formatRelative(epochSeconds: number): string {
	const delta = Date.now() / 1000 - epochSeconds;
	if (delta < 45) return "just now";
	if (delta < 3600) return `${Math.max(1, Math.round(delta / 60))}m ago`;
	if (delta < 86400) return `${Math.round(delta / 3600)}h ago`;
	if (delta < 86400 * 7) return `${Math.round(delta / 86400)}d ago`;
	const d = new Date(epochSeconds * 1000);
	return d
		.toLocaleDateString(undefined, { month: "short", day: "numeric" })
		.toLowerCase();
}

/** Spell out a subagent's launch tier for display.
 *
 * A child job records the tier it was launched at abbreviated (`lo`/`med`/`hi`
 * — see `AsyncJob.effort`), while the session footer shows the model's resolved
 * reasoning effort as a full word (`low`/`medium`/`high`). Rendering the raw
 * `hi` beside a footer that says `high` is two vocabularies for one concept
 * (design D3), so subagent surfaces spell the tier out. Any value that is not a
 * known abbreviation (e.g. an already-resolved effort word) passes through. */
export function formatEffort(effort: string): string {
	switch (effort) {
		case "lo":
			return "low";
		case "med":
			return "medium";
		case "hi":
			return "high";
		default:
			return effort;
	}
}

/** "/Users/damian/projects/foo" → "foo". Trailing slashes tolerated. */
export function basename(path: string): string {
	const trimmed = path.replace(/\/+$/, "");
	const i = trimmed.lastIndexOf("/");
	return i === -1 ? trimmed : trimmed.slice(i + 1) || trimmed;
}

/** "/Users/damian/projects/foo" → "~/projects/foo" for display. */
export function shortenHome(path: string, home: string): string {
	if (home && (path === home || path.startsWith(home + "/"))) {
		return "~" + path.slice(home.length);
	}
	return path;
}
