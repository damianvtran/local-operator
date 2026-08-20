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

export function ToolRow({ entry }: { entry: TranscriptEntry }) {
	const [open, setOpen] = useState(false);
	const running =
		entry.tool_state === "running" || entry.tool_state === "composing";
	const hasDetails =
		entry.intent ||
		entry.details.args ||
		entry.details.output ||
		entry.details.diff ||
		entry.error;

	return (
		<div
			className={cn(
				"rounded-sm px-1.5",
				running && "bg-elevated",
				entry.tool_state === "failed" && "bg-danger-wash",
				entry.tool_state === "done" && "bg-surface",
			)}
		>
			<button
				type="button"
				onClick={() => hasDetails && setOpen(!open)}
				className="flex min-h-8 w-full items-center gap-1.5 text-left select-none"
			>
				<span
					className={cn(
						"w-4 shrink-0 text-center font-mono text-mono-sm",
						entry.tool_state === "failed" && "text-danger",
						entry.tool_state === "done" && "text-success",
						(entry.tool_state === "interrupted" ||
							entry.tool_state === "composing") &&
							"text-ink-dim",
						running && "lo-pulse text-accent",
					)}
					aria-hidden
				>
					{GLYPH[entry.tool_state]}
				</span>
				<span className="shrink-0 font-mono text-mono-sm text-ink-muted">
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
					{entry.details.args ? (
						/* max-h + scroll like the diff and output blocks: an UNBOUNDED
						   args block renders its full height, and a big payload
						   (a long command, a large write) fills the screen with
						   the sunken ground — the "solid background" the tap
						   produced. */
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
