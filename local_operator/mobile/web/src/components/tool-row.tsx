/**
 * Tool row — ONE LINE per action (branding §7.3): state glyph, monospace
 * tool name, truncated summary, diff counts, elapsed. Everything else —
 * intent, args, output, diff — sits behind the row's own disclosure
 * (§7.4: available in one tap, never shown by default).
 *
 * The row itself is the state indicator: composing/running glyphs pulse,
 * there is no separate spinner.
 */
import { useState } from "react";
import { cn } from "../lib/cn";
import { formatElapsed } from "../lib/format";
import type { TranscriptEntry } from "../types";

const GLYPH: Record<TranscriptEntry["tool_state"], string> = {
	composing: "⟳",
	/* Not a spinner: nothing is turning. The elision says "still to come", which
	   is the honest thing for a call waiting its turn to execute. */
	queued: "⋯",
	running: "⟳",
	done: "✓",
	failed: "✗",
	interrupted: "–",
};

/** Normalize a details field to display lines. The fold emits args as a dict
    and diff as a list of lines (the shapes the tools produce), while output
    and partial are plain strings. Rendering must handle ALL of them: calling
    .split() on a non-string throws a TypeError, React unmounts the tree, and
    the user reads it as "tap → whole screen goes blank". */
function toLines(value: unknown): string[] {
	if (value == null) return [];
	if (typeof value === "string") return value.split("\n");
	if (Array.isArray(value)) return value.map((v) => String(v));
	if (typeof value === "object") {
		/* args dict — one "key: value" line per entry. */
		return Object.entries(value as Record<string, unknown>).map(
			([k, v]) => `${k}: ${typeof v === "string" ? v : JSON.stringify(v)}`,
		);
	}
	return [String(value)];
}

function DiffBlock({ diff }: { diff: string | string[] }) {
	/* span rows, never <div> inside <pre>: <div> is not phrasing content, so
	   the HTML parser hoists it out of the <pre> and the expansion repaints
	   as a broken, layout-filling block — read on the phone as "the whole
	   page went solid". whitespace-pre-wrap on the container plus block
	   spans gives the same monospace, per-line-tinted result legally.
	   toLines accepts the fold's list-of-lines form AND a pre-joined string. */
	return (
		<div className="lo-scroll max-h-64 overflow-auto rounded-sm bg-sunken p-2 font-mono text-mono-sm leading-snug whitespace-pre-wrap">
			{toLines(diff).map((line, i) => (
				<span
					key={i}
					className={cn(
						"block",
						line.startsWith("+") &&
							!line.startsWith("+++") &&
							"text-success",
						line.startsWith("-") &&
							!line.startsWith("---") &&
							"text-danger",
						line.startsWith("@@") && "text-info",
					)}
				>
					{line}
				</span>
			))}
		</div>
	);
}

/** Tools whose expansion is the rendered diff, mirroring the TUI: a
    write/edit carries its whole new content in args, and showing that
    payload next to the diff says the same thing twice. For these the args
    block is dropped and only the diff (+ any output) shows. */
const DIFF_FIRST_TOOLS = new Set(["write", "edit", "apply_patch", "patch"]);

export function ToolRow({ entry }: { entry: TranscriptEntry }) {
	/* Bang-mode (`! cmd`) opens EXPANDED, matching the TUI: the user typed
	   this command themselves and is waiting to read its output, so making
	   them tap to see it asks for a gesture to reveal the thing they asked
	   for. Every other card stays collapsed (§7.4 — one line per action,
	   details in one tap).

	   Tracked as an OVERRIDE rather than as initial state: a live bang row
	   is mounted while still running and only learns `user_run` when its
	   result settles, and a `useState` initializer runs once at mount, so
	   seeding it there would leave the live card shut. `null` means "the
	   user has not touched this row", in which case `user_run` decides. */
	const [override, setOverride] = useState<boolean | null>(null);
	const open = override ?? entry.details.user_run === true;
	const setOpen = setOverride;
	const running =
		entry.tool_state === "running" || entry.tool_state === "composing";
	/* A queued row is live — the call has been announced and may still execute —
	   so it keeps the raised background of a live row, but it does NOT pulse:
	   the pulse is the "work is happening" signal (branding §7.3), and nothing
	   is happening while the call waits behind a sibling. Its glyph is dim like
	   composing's for the same reason. */
	const queued = entry.tool_state === "queued";
	const isDiffFirst =
		DIFF_FIRST_TOOLS.has(entry.tool_name.toLowerCase()) &&
		entry.details.diff != null;
	const hasDetails =
		entry.intent ||
		entry.details.output ||
		entry.details.diff ||
		entry.error ||
		(!isDiffFirst && entry.details.args);

	return (
		<div
			className={cn(
				"rounded-sm px-1.5",
				(running || queued) && "bg-elevated",
				entry.tool_state === "failed" && "bg-danger-wash",
				entry.tool_state === "done" && "bg-surface",
			)}
		>
			<button
				type="button"
				onClick={() => hasDetails && setOpen(!open)}
				className="flex min-h-11 w-full items-center gap-1.5 text-left select-none"
			>
				<span
					className={cn(
						"w-4 shrink-0 text-center font-mono text-mono-sm",
						entry.tool_state === "failed" && "text-danger",
						entry.tool_state === "done" && "text-success",
						(entry.tool_state === "interrupted" ||
							entry.tool_state === "composing" ||
							queued) &&
							"text-ink-dim",
						running && "lo-pulse text-accent",
					)}
					aria-hidden
				>
					{GLYPH[entry.tool_state]}
				</span>
				{/* The NAME yields, so the clock is the last thing lost rather than the
				 * first. It used to be ``shrink-0``: on a row with a long MCP-name
				 * (``mcp__local-operator__subagent_dispatch``) it could not give up a
				 * pixel, so the whole deficit landed on the summary and then on the
				 * clock — which the transcript scroller CLIPS with no ellipsis
				 * (``overflow-x: hidden``, measured ``clientW 320 / scrollW 377``), so
				 * ``1000h 40m`` read as ``1000h`` and ``59m 59s`` as ``59m``: shorter
				 * strings that are themselves valid durations, with nothing to tell the
				 * reader the number is short (design round 1 D1). ``min-w-0 truncate``
				 * lets the name share the squeeze and ellipsize honestly; the clock
				 * span below keeps ``shrink-0``, so it always fits. */}
				<span className="min-w-0 shrink truncate font-mono text-mono-sm text-ink-muted">
					{entry.tool_name}
				</span>
				<span className="min-w-0 flex-1 truncate text-body-sm text-ink-dim">
					{entry.summary}
				</span>
				{entry.diff_added > 0 || entry.diff_removed > 0 ? (
					<span className="shrink-0 font-mono text-mono-sm">
						{entry.diff_added > 0 ? (
							<span className="text-success">
								+{entry.diff_added}
							</span>
						) : null}{" "}
						{entry.diff_removed > 0 ? (
							<span className="text-danger">
								−{entry.diff_removed}
							</span>
						) : null}
					</span>
				) : null}
				{entry.elapsed_s > 0 ? (
					<span className="shrink-0 font-mono text-mono-sm text-ink-dim">
						{formatElapsed(entry.elapsed_s)}
					</span>
				) : null}
				{hasDetails ? (
					<span className="shrink-0 text-ink-dim" aria-hidden>
						{open ? "▾" : "▸"}
					</span>
				) : null}
			</button>
			{open && hasDetails ? (
				/* Cap the WHOLE expansion, not just its blocks: intent + error +
				   args + diff + output stack, and unbounded they could still
				   fill the viewport. The expansion scrolls as one region. */
				<div className="lo-scroll flex max-h-96 flex-col gap-1.5 overflow-y-auto pb-1 pl-6">
					{entry.intent ? (
						<p className="text-body-sm text-ink-muted">
							{entry.intent}
						</p>
					) : null}
					{entry.error ? (
						<p className="text-body-sm text-danger">
							{entry.error}
						</p>
					) : null}
					{entry.details.args && !isDiffFirst ? (
						/* Hidden for write/edit: their args ARE the content the diff
						   already shows, so the payload would be redundant (the
						   TUI expands these to the diff alone). max-h + scroll:
						   an UNBOUNDED args block renders its full height and
						   fills the screen with the sunken ground. */
						<div className="lo-scroll max-h-40 overflow-y-auto rounded-sm bg-sunken p-2">
							{toLines(entry.details.args).map((line, i) => {
								const sep = line.indexOf(":");
								return (
									<div
										key={i}
										className="font-mono text-mono-sm"
									>
										{sep > 0 ? (
											<>
												<span className="text-ink-dim">
													{line.slice(0, sep)}:
												</span>
												<span className="text-ink-muted">
													{line.slice(sep + 1)}
												</span>
											</>
										) : (
											<span className="text-ink-muted">
												{line}
											</span>
										)}
									</div>
								);
							})}
						</div>
					) : null}
					{entry.details.diff ? (
						<DiffBlock diff={entry.details.diff} />
					) : null}
					{entry.details.output ? (
						<pre className="lo-scroll max-h-48 overflow-auto rounded-sm bg-sunken p-2 font-mono text-mono-sm whitespace-pre-wrap text-ink-muted">
							{toLines(entry.details.output).join("\n")}
						</pre>
					) : null}
				</div>
			) : null}
		</div>
	);
}
