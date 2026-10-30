/**
 * Tool row — ONE LINE per action (branding §7.3): state glyph, monospace
 * tool name, truncated summary, diff counts, elapsed. Everything else —
 * intent, args, output, diff — sits behind the row's own disclosure
 * (§7.4: available in one tap, never shown by default).
 *
 * The row itself is the state indicator: composing/running glyphs pulse,
 * there is no separate spinner.
 */
import { useState, type ReactNode } from "react";
import { cn } from "../lib/cn";
import { formatElapsed } from "../lib/format";
import type { BridgeRowCopy, BridgeRowFacts, TranscriptEntry } from "../types";
import { Markdown } from "./markdown";

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
		<div dir="ltr" className="lo-scroll max-h-64 overflow-auto rounded-sm bg-sunken p-2 font-mono text-mono-sm leading-snug whitespace-pre-wrap">
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

/**
 * THE TEAM BRIDGE ROW. One wire name (`bridge`) covers posting to a channel,
 * a DM, a mailbox send and a dozen reads, and the generic row painted all of
 * them as `bridge  op=channel_post` with an args dump whose most visible token
 * was a `bcv_…` conversation id. The facts here are composed server-side by the
 * shared `harness/rows.py::bridge_row` (the same decision the desktop and TUI
 * draw, lane c733bb5b44de) and ride `details.bridge_row`; this file only paints
 * them, so the three surfaces cannot disagree about what a call did.
 *
 * The OUTCOME is the part that matters most and is easiest to get wrong: an
 * `inbox_only` send was accepted into a peer's inbox and is NOT in any
 * conversation, so it must never read as a success (operator report
 * 2026-10-28: agents and people both mistook a pair send for a DM). It takes
 * warning ink, a refusal takes danger ink with its typed code, and the two
 * "not yet" states stay quiet words.
 */
const BRIDGE_TOOL = "bridge";

/* Opaque Team Bridge ids. The headline never prints one: a person reads a
   channel name, not `bcv_01J…`. The server already composes a label; this is
   the renderer's own floor for a summary or label that still carries one. */
/* The tail takes `_` and `-`: real ids include `acc_remote_9` and
   `ep_01J-ABC`, and an alphanumeric-only tail left their second half visible
   (review round 1, R3). Same rule as `projection._BRIDGE_ID_RE`. */
const BRIDGE_ID = /\b(?:bcv|bcmsg|acc|ep)_[A-Za-z0-9_-]+/g;

function scrubBridgeIds(text: string): string {
	return text.replace(BRIDGE_ID, "…").replace(/\s{2,}/g, " ").trim();
}

/** lucide icons (ISC), inlined the way `ui/chrome-icon.tsx` inlines its
    glyphs: the SPA ships no icon dependency, and three small icons are not a
    reason to add one. Paths copied from lucide-react 0.507.0, the version the
    desktop renders, so the two surfaces draw the same marks. */
const LUCIDE: Record<"messages-square" | "message-square-off" | "hourglass", string[]> = {
	"messages-square": [
		"M14 9a2 2 0 0 1-2 2H6l-4 4V4a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2z",
		"M18 9h2a2 2 0 0 1 2 2v11l-4-4h-6a2 2 0 0 1-2-2v-1",
	],
	/* Delivery marks, mirroring the desktop row (local-operator-ui #1529):
	   inbox-only = a message that did not reach a conversation; awaiting
	   acceptance = waiting on the other side. */
	"message-square-off": [
		"M21 15V5a2 2 0 0 0-2-2H9",
		"m2 2 20 20",
		"M3.6 3.6c-.4.3-.6.8-.6 1.4v16l4-4h10",
	],
	hourglass: [
		"M5 22h14",
		"M5 2h14",
		"M17 22v-4.172a2 2 0 0 0-.586-1.414L12 12l-4.414 4.414A2 2 0 0 0 7 17.828V22",
		"M7 2v4.172a2 2 0 0 0 .586 1.414L12 12l4.414-4.414A2 2 0 0 0 17 6.172V2",
	],
};

function LucideIcon({ name, size = 14 }: { name: keyof typeof LUCIDE; size?: number }) {
	return (
		<svg
			data-icon={name}
			width={size}
			height={size}
			viewBox="0 0 24 24"
			fill="none"
			stroke="currentColor"
			strokeWidth="2"
			strokeLinecap="round"
			strokeLinejoin="round"
			aria-hidden="true"
			focusable="false"
			className="pointer-events-none shrink-0"
		>
			{LUCIDE[name].map((d) => (
				<path key={d} d={d} />
			))}
		</svg>
	);
}

function bridgeFacts(entry: TranscriptEntry): BridgeRowFacts | null {
	if (entry.tool_name.toLowerCase() !== BRIDGE_TOOL) return null;
	const facts = entry.details.bridge_row;
	return facts && typeof facts === "object" ? facts : null;
}

/**
 * The bridge row's WORDS come rendered from the daemon (`details.bridge_copy`,
 * `harness/bridge_copy.py` over `wire.tool_rows.bridge`): the same strings the
 * TUI card paints, in the daemon's locale. The relay mints no copy of its own
 * for this row - a second translation of one decision is the drift class the
 * shared copy exists to remove. Absent copy (an older daemon, or details shed
 * under frame pressure) paints no chip and no sentence rather than English
 * guessed here.
 */
function bridgeCopy(entry: TranscriptEntry): BridgeRowCopy {
	const copy = entry.details.bridge_copy;
	return copy && typeof copy === "object" ? copy : {};
}

type OutcomeTone = "warning" | "danger" | "pending";

/* Ink per outcome. `inbox_only` is warning, never success: the send was
   accepted into a peer inbox and appears in NO conversation. */
const OUTCOME_TONE: Record<NonNullable<BridgeRowFacts["outcome"]>, OutcomeTone> = {
	inbox_only: "warning",
	refused: "danger",
	failed: "danger",
	/* Not yet / maybe not: the TUI's amber `warn` tier and the desktop's
	   `partial` ink (design round 1, D6). `unconfirmed` means "may not have
	   landed - don't re-post blindly", not a quiet fact. Only `inbox_only`
	   keeps the wash, so it stays the loudest of the three. */
	unconfirmed: "pending",
	awaiting_acceptance: "pending",
};

const OUTCOME_MARK: Partial<Record<NonNullable<BridgeRowFacts["outcome"]>, keyof typeof LUCIDE>> = {
	inbox_only: "message-square-off",
	awaiting_acceptance: "hourglass",
};

/** The chip's word, ink and mark, or null for a plain landed call (no chip:
    an ordinary post is the row's ✓ glyph, not a second success badge). */
function bridgeOutcome(
	facts: BridgeRowFacts | null,
	copy: BridgeRowCopy,
): { text: string; word: string; code: string; tone: OutcomeTone; mark?: keyof typeof LUCIDE } | null {
	const outcome = facts?.outcome;
	if (!outcome || !Object.hasOwn(OUTCOME_TONE, outcome)) return null;
	const text = copy.outcome ? scrubBridgeIds(copy.outcome) : "";
	if (!text) return null;
	/* `refused · not_member`: the WORD stays at every width, the code is the
	   first thing a narrow row sheds (it is repeated in the expansion's
	   sentence) - design round 1, D3(c). */
	const sep = text.indexOf(" · ");
	const word = sep > 0 ? text.slice(0, sep) : text;
	const code = sep > 0 ? text.slice(sep) : "";
	return { text, word, code, tone: OUTCOME_TONE[outcome], mark: OUTCOME_MARK[outcome] };
}

/** The row's summary, id-free, without the outcome tail the chip shows.

    The daemon's one-line summary (`bridge_row_line`) ends with ` · <outcome>`
    because the TUI-shaped single cell has no status cap. The phone row DOES
    (the chip), so leaving the tail would print the caveat twice - and the
    copy in the summary is the one that truncates away first at 320px. */
function bridgeSummary(summary: string, outcome: { text: string; word: string } | null): string {
	const text = scrubBridgeIds(summary);
	if (!outcome) return text;
	/* The daemon's line ends ` · <outcome>` for capless cells, and its cap may
	   cut that tail anywhere. With the chip beside it the caveat must be said
	   once (design D5 / QA Q3), so strip it in this order:
	   1. the FULL outcome, found by TEXT not by the last separator: a refusal's
	      outcome is itself two segments (`refused · not_member`), and splitting
	      on the last ` · ` compared `not_member` against it and missed (review
	      round 2, R7 / D9 / Q5);
	   2. a capped tail: the last ` · ` segment(s) ending `…` that are a prefix
	      of the outcome (`· awaiting…`, `· refused · not_m…`), including a bare
	      `· …` where the cap left nothing of it (QA O2). */
	const full = ` · ${outcome.text}`;
	if (text.endsWith(full)) return text.slice(0, -full.length);
	if (!text.endsWith("…")) return text;
	const capped = text.slice(0, -1);
	/* A cap that left nothing of the tail: `… · …` (QA O2). */
	if (capped.endsWith(" · ")) return text.slice(0, -" · …".length);
	for (let sep = capped.lastIndexOf(" · "); sep >= 0; sep = capped.lastIndexOf(" · ", sep - 1)) {
		const tail = capped.slice(sep + 3);
		if (tail === "" || outcome.text.startsWith(tail)) return text.slice(0, sep);
		/* Only the trailing segments can be the outcome; stop once the tail is
		   longer than the outcome itself. */
		if (tail.length > outcome.text.length) break;
	}
	return text;
}

const OUTCOME_INK: Record<OutcomeTone, string> = {
	warning: "text-warning",
	danger: "text-danger",
	pending: "text-warning",
};

/* Args the bridge expansion already says in words (the body as markdown, the
   op and target in the headline) - the generic args block keeps only what is
   left, so the expansion does not repeat itself and no raw id leads it. */
const BRIDGE_SPOKEN_ARGS = new Set([
	"body",
	"text",
	"op",
	"endpoint_id",
	"conversation_id",
	"channel_id",
	"message_id",
	"recipient",
	"to",
]);

function bridgeBody(entry: TranscriptEntry): string {
	const args = entry.details.args;
	if (!args || typeof args !== "object") return "";
	const body = args.body ?? args.text;
	return typeof body === "string" ? body : "";
}

function bridgeRestArgs(entry: TranscriptEntry): Record<string, unknown> | null {
	const args = entry.details.args;
	if (!args || typeof args !== "object") return null;
	const rest = Object.fromEntries(
		Object.entries(args).filter(([k]) => !BRIDGE_SPOKEN_ARGS.has(k)),
	);
	return Object.keys(rest).length > 0 ? rest : null;
}

/** Headline (who/where, in words), the markdown body, then the outcome as the
    daemon's full sentence in the chip's ink. */
function BridgeExpansion({ entry, facts }: { entry: TranscriptEntry; facts: BridgeRowFacts | null }) {
	const copy = bridgeCopy(entry);
	const body = bridgeBody(entry);
	const headline = copy.headline ? scrubBridgeIds(copy.headline) : "";
	const sentence = copy.outcome_sentence ? scrubBridgeIds(copy.outcome_sentence) : "";
	const tone = facts?.outcome && Object.hasOwn(OUTCOME_TONE, facts.outcome) ? OUTCOME_TONE[facts.outcome] : null;
	return (
		<div className="flex min-w-0 flex-col gap-1.5">
			{headline ? (
				<p data-bridge-headline className="min-w-0 break-words text-body-sm text-ink-muted">
					{headline}
				</p>
			) : null}
			{body ? (
				<div className="min-w-0 break-words rounded-sm bg-sunken p-2 text-body-sm text-ink">
					<Markdown text={body} />
				</div>
			) : null}
			{sentence ? (
				<p
					data-bridge-outcome-sentence
					className={cn("min-w-0 break-words text-body-sm", tone ? OUTCOME_INK[tone] : "text-ink-dim")}
				>
					{sentence}
				</p>
			) : null}
		</div>
	);
}

export function ToolRow({
	entry,
	children,
}: {
	entry: TranscriptEntry;
	/* Extra content rendered inside the row's own container, below the row
	   button and above any expansion. Exactly one caller has anything to put
	   here — the image-generation card, whose progress figure must sit on the
	   SAME ground and border as its row so the pair reads as one object — and
	   doing it through the container rather than beside it keeps the
	   state-to-background mapping (raised while live, danger-wash on failure)
	   in this one place. Absent for every other tool, so their DOM is
	   unchanged. */
	children?: ReactNode;
}) {
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
	   so it keeps the DIM pending glyph of a live row, but it does NOT pulse: the
	   pulse is the "work is happening" signal (branding §7.3), and nothing is
	   happening while the call waits behind a sibling. The redesign flattened the
	   tool rows onto the transcript's ground (§7.1), so the queued state is
	   carried by glyph ink alone; the glyph is dim like composing's for the same
	   reason. */
	const queued = entry.tool_state === "queued";
	const isDiffFirst =
		DIFF_FIRST_TOOLS.has(entry.tool_name.toLowerCase()) &&
		entry.details.diff != null;
	const bridge = entry.tool_name.toLowerCase() === BRIDGE_TOOL;
	const facts = bridgeFacts(entry);
	const outcome = bridge ? bridgeOutcome(facts, bridgeCopy(entry)) : null;
	/* The bridge row swaps the full args dict for the args it does not already
	   say in words; every other tool keeps its args block exactly as before. */
	const shownArgs = bridge ? bridgeRestArgs(entry) : entry.details.args;
	const hasDetails =
		(bridge && (facts != null || bridgeBody(entry) !== "")) ||
		entry.intent ||
		entry.details.output ||
		entry.details.diff ||
		entry.error ||
		(!isDiffFirst && shownArgs);

	return (
		<div
			className={cn(
				"rounded-sm px-1.5",
				
				
				
			)}
		>
			<button
				type="button"
				onClick={() => hasDetails && setOpen(!open)}
				className={cn(
					"flex min-h-11 w-full items-center gap-1.5 text-start select-none",
					/* A container only for the bridge row's narrow-width shed (D3). */
					bridge && "@container",
				)}
			>
				<span
					className={cn(
						"w-4 shrink-0 text-center font-mono text-mono-sm",
						entry.tool_state === "failed" && "text-danger",
						/* A bridge call that landed with a caveat (inbox-only, awaiting,
						   unconfirmed) keeps the ✓ SHAPE - nothing failed - but never the
						   success ink: the leading glyph is what a reader scans first
						   (design round 1, D1; the TUI's amber `warn` tier). */
						entry.tool_state === "done" && !(outcome && outcome.tone !== "danger") && "text-success",
						entry.tool_state === "done" && outcome && outcome.tone !== "danger" && "text-warning",
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
				{bridge ? (
					<span className="shrink-0 text-ink-dim">
						<LucideIcon name="messages-square" />
					</span>
				) : null}
				{/* NARROW-ROW PRIORITY for a bridge row (design round 1, D3; the
				    desktop's container shed in #1529). Shed in order: (1) the
				    `bridge` name below a 22rem row - 360/390 phones measure 302/332px
				    and the name still took ~45px the summary needed (design round 2,
				    D10), while the messages-square mark already says which tool;
				    then below 18rem (a 320px phone measures 262px): (2) the
				    inbox/awaiting chip's words - its mark and ink stay, the word
				    stays for screen readers; (3) a refusal's `· code` - the word
				    stays, the code is in the expansion's sentence. */}
				<span
					dir="ltr"
					className={cn(
						"min-w-0 shrink truncate text-body-sm text-ink-muted",
						bridge && "@max-[22rem]:hidden",
					)}
				>
					{entry.tool_name}
				</span>
				{/* A bridge row with an outcome chip holds a floor for the summary
				   (it names the op and who), so at 320px the chip and the name
				   share the squeeze instead of crushing the summary to one glyph
				   ("bridge o." measured before this floor). */}
				<span
					className={cn(
						"min-w-0 flex-1 truncate text-body-sm text-ink-dim",
						bridge && outcome && "min-w-[4.5rem]",
					)}
				>
					{bridge ? bridgeSummary(entry.summary, outcome) : entry.summary}
				</span>
				{outcome ? (
					/* The outcome chip may truncate (its whole sentence is in the
					   expansion); it shrinks AFTER the flex-1 summary has given up to
					   its floor, so a long summary cannot squeeze the caveat to
					   nothing. Wash chips carry a hairline in their own border token
					   so they read as a chip on the light ground, where the wash
					   alone measures ~1.03:1 against canvas (design D7). */
					<span
						data-bridge-outcome={facts?.outcome ?? undefined}
						className={cn(
							"min-w-0 max-w-[55%] shrink truncate rounded-sm px-1 text-mono-sm",
							outcome.tone === "warning" && "border border-warning-border/40 bg-warning-wash text-warning",
							outcome.tone === "danger" && "border border-danger-border/40 bg-danger-wash text-danger",
							outcome.tone === "pending" && "text-warning",
						)}
					>
						{outcome.mark ? (
							<span className="me-0.5 inline-block align-[-2px] @max-[18rem]:me-0">
								<LucideIcon name={outcome.mark} size={12} />
							</span>
						) : null}
						{/* A marked chip may collapse to its mark; the word stays in
						    the accessibility tree (sr-only), never dropped. */}
						<span className={cn(outcome.mark && "@max-[18rem]:sr-only")}>{outcome.word}</span>
						{outcome.code ? <span className="@max-[18rem]:hidden">{outcome.code}</span> : null}
					</span>
				) : null}
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
					<span className="lo-directional shrink-0 text-ink-dim" aria-hidden>
						{open ? "▾" : "▸"}
					</span>
				) : null}
			</button>
			{children}
			{open && hasDetails ? (
				/* Cap the WHOLE expansion, not just its blocks: intent + error +
				   args + diff + output stack, and unbounded they could still
				   fill the viewport. The expansion scrolls as one region. */
				<div className="lo-scroll flex max-h-96 flex-col gap-1.5 overflow-y-auto pb-1 ps-6">
					{entry.intent ? (
						<p className="text-body-sm text-ink-muted">
							{entry.intent}
						</p>
					) : null}
					{entry.error && !(bridge && bridgeCopy(entry).outcome_sentence) ? (
						/* A bridge row whose daemon sentence states the refusal does
						   not repeat it in a raw red line above - that line led the
						   panel with a `bcv_` id (design D2 / QA Q2). Without a
						   sentence the error still shows, id-scrubbed. */
						<p className="text-body-sm text-danger">
							{bridge ? scrubBridgeIds(entry.error) : entry.error}
						</p>
					) : null}
					{bridge ? <BridgeExpansion entry={entry} facts={facts} /> : null}
					{shownArgs && !isDiffFirst ? (
						/* Hidden for write/edit: their args ARE the content the diff
						   already shows, so the payload would be redundant (the
						   TUI expands these to the diff alone). max-h + scroll:
						   an UNBOUNDED args block renders its full height and
						   fills the screen with the sunken ground. */
						<div dir="ltr" className="lo-scroll max-h-40 overflow-y-auto rounded-sm bg-sunken p-2">
							{toLines(shownArgs).map((line, i) => {
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
					{entry.details.output && !(bridge && bridgeCopy(entry).outcome_sentence) ? (
						/* Same rule for the raw output on a bridge row: the tool's
						   text is the refusal/receipt the sentence already states,
						   and it interpolates conversation ids. Shown (scrubbed) only
						   when there is no sentence. */
						<pre dir="ltr" className="lo-scroll max-h-48 overflow-auto rounded-sm bg-sunken p-2 font-mono text-mono-sm whitespace-pre-wrap text-ink-muted">
							{bridge
								? toLines(entry.details.output).join("\n").replace(BRIDGE_ID, "…")
								: toLines(entry.details.output).join("\n")}
						</pre>
					) : null}
				</div>
			) : null}
		</div>
	);
}
