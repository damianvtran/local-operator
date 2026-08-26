/**
 * The two formatters the UI needs, hand-rolled (no date library):
 * elapsed durations for tool rows / subagents, and a relative timestamp
 * for the past-sessions list.
 */

/** 83.4 → "1m 23s"; 0.9 → "0.9s"; 3661 → "1h 1m". */
export function formatElapsed(seconds: number): string {
	if (!Number.isFinite(seconds) || seconds < 0) return "";
	if (seconds < 10) return `${seconds.toFixed(1)}s`;
	if (seconds < 60) return `${Math.round(seconds)}s`;
	const minutes = Math.floor(seconds / 60);
	if (minutes < 60) {
		const rem = Math.round(seconds - minutes * 60);
		return rem > 0 ? `${minutes}m ${rem}s` : `${minutes}m`;
	}
	const hours = Math.floor(minutes / 60);
	const remM = minutes - hours * 60;
	return remM > 0 ? `${hours}h ${remM}m` : `${hours}h`;
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
