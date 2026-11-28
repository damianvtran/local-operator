/**
 * A minimal, safe markdown renderer for assistant rows.
 *
 * Everything renders through React elements — there is no
 * dangerouslySetInnerHTML anywhere in this tree, so escaping is the
 * default, not a discipline. Supported: headings (#–####), bold, italic,
 * strikethrough, inline code, fenced code blocks, unordered/ordered lists,
 * blockquotes, links, horizontal rules, paragraphs. That is the envelope
 * assistant prose actually uses; anything more exotic renders as literal
 * text, which fails safe.
 *
 * Links are forced https? and open in a new tab: this is a remote-control
 * surface, and a javascript: URL in agent output must never be clickable.
 */
import { Component, Fragment, memo, useLayoutEffect, useReducer, useRef, useState } from "react";
import type { ReactNode } from "react";
import { cn } from "../lib/cn";
import { isolateTechnicalRuns } from "../lib/bidi";
import { diagramKey, shouldDrawDiagram } from "../lib/diagram-fence";
import { DiagramBlock } from "./diagram-block";
import { AskIdChip } from "./ask-id";
import { askIdSegments, isAskId } from "../lib/ask-chip";
import { usePacedText } from "../lib/use-paced-text";
import { allowsReveal, revealText, settleReveals } from "../lib/stream-reveal";

/* ------------------------------------------------------------------ */
/* Inline                                                             */
/* ------------------------------------------------------------------ */

/**
 * Inline parsing is a longest-match-first token walk. Order matters: code
 * spans first (nothing nests inside them), then links, then bold, then
 * strike, then italic.
 */
const INLINE_RE =
	/(`[^`]+`)|\[([^\]]+)\]\(([^)\s]+)\)|\*\*([^*]+)\*\*|__([^_]+)__|~~([^~]+)~~|\*([^*\n]+)\*|_([^_\n]+)_/g;

type ProseReveal = { node: HTMLSpanElement | null; settled: boolean; attach: (node: HTMLSpanElement | null) => void };
type ProsePart = { start: number; reveal?: ProseReveal };

function selects(node: Node): boolean {
	const selection = node.ownerDocument?.getSelection();
	if (!selection || selection.isCollapsed) return false;
	for (let i = 0; i < selection.rangeCount; i++) {
		if (selection.getRangeAt(i).intersectsNode(node)) return true;
	}
	return false;
}

/** Don't split a grapheme, a shaping word, or an FSI/PDI technical isolate.
 * Deltas inside a word are visible immediately; the next safe word boundary
 * starts the fade, never a wait-for-a-word buffer. */
function proseBoundary(text: string, at: number): number | undefined {
	if (/[\u2066-\u2069]/u.test(text)) return;
	const suffix = text.slice(at);
	if ((at === 0 || /[ \t]/.test(text[at - 1])) && /^[\p{L}\p{N}\p{P}\p{S}]/u.test(suffix)) return at;
	const gap = /[ \t]+(?=[\p{L}\p{N}\p{P}\p{S}])/u.exec(suffix);
	// Keep a leading separator on the new span so an ordinary word append
	// does not even write to the existing prefix text node.
	return gap ? at + gap.index : undefined;
}

const Prose = memo(function Prose({ text, from }: { text: string; from?: number }) {
	const committed = useRef<{ text: string; parts: ProsePart[] } | null>(null);
	const [, retire] = useReducer((version: number) => version + 1, 0);
	const before = committed.current;
	const append = before && text.startsWith(before.text);
	const cut = (append || !before) && from !== undefined
		? proseBoundary(text, Math.max(from, before?.text.length ?? 0)) : undefined;
	const parts: ProsePart[] = [];
	const selected: Node[] = [];
	let compacting: Node | null = null;
	const keep = (part: ProsePart, next: ProsePart | undefined) => {
		const reveal = part.reveal;
		if (reveal?.settled) {
			const node = reveal.node;
			if (node && selects(node)) {
				selected.push(node);
			} else {
				// Retirement merges to the left. The following plain part loses
				// its text node even if the range has already left this span.
				// Only the next part of *this* prose run can be removed here;
				// a wrapped neighbor decides whether to retire on its own.
				const right = next && !next.reveal ? node?.nextSibling : null;
				if (right?.nodeType === Node.TEXT_NODE && selects(right)) {
					selected.push(right);
				} else {
					compacting ??= node?.parentNode ?? null;
					part = { start: part.start };
				}
			}
		}
		// The rest is a steady plain-text run, not one element per frame.
		// Keep wrappers only for live fades or to protect a range at the seam.
		if (part.reveal || parts.at(-1)?.reveal || !parts.length) parts.push(part);
	};
	if (append) for (const [i, part] of before.parts.entries()) keep(part, before.parts[i + 1]);
	if (!parts.length && (cut ?? text.length) > 0) parts.push({ start: 0 });
	if (cut !== undefined) {
		const reveal: ProseReveal = {
			node: null, settled: false,
			attach(node) {
				reveal.node = node;
				if (node) revealText(node, () => {
					if (reveal.node !== node) return;
					reveal.settled = true;
					retire();
				});
			},
		};
		parts.push({ start: cut, reveal });
	}
	useLayoutEffect(() => { committed.current = { text, parts }; });
	useLayoutEffect(() => {
		const doc = selected[0]?.ownerDocument;
		if (!doc) return;
		// No polling and no listener for an unrelated selection/resting caret.
		// Once the range leaves the affected node, retry the deferred merge.
		const changed = () => { if (selected.some(node => !selects(node))) retire(); };
		doc.addEventListener("selectionchange", changed);
		return () => doc.removeEventListener("selectionchange", changed);
	});
	return <SelectionGuard scope={() => compacting}>
		{parts.map(({ start, reveal }, i) => {
			const piece = text.slice(start, parts[i + 1]?.start);
			return reveal ? <span key={start} ref={reveal.attach}>{piece}</span> : <Fragment key={start}>{piece}</Fragment>;
		})}
	</SelectionGuard>;
});

function renderInline(text: string, from?: number, askChips = false): ReactNode[] {
	const out: ReactNode[] = [];
	let last = 0;
	const prose = (start: number, end: number, isolate: boolean) => {
		const raw = text.slice(start, end);
		const value = isolate ? isolateTechnicalRuns(raw) : raw;
		const fromHere =
			value === raw && from !== undefined && from < end ? Math.max(0, from - start) : undefined;
		/* THE CHIP SITS IN THE PROSE, not beside it (spec §2): an id the agent
		   wrote stays in its sentence, so the sentence still reads when copied. The
		   run is split BEFORE isolation — an isolate wrapped around the token would
		   put a control character between `a-` and its hex and the token would stop
		   being one — and every plain piece keeps its own reveal offset so a
		   streaming answer still fades from the right place. */
		if (!askChips) return <Prose key={`p${start}`} text={value} from={fromHere} />;
		const segments = askIdSegments(raw);
		if (segments.length === 1 && segments[0].kind === "text") {
			return <Prose key={`p${start}`} text={value} from={fromHere} />;
		}
		let cursor = 0;
		return segments.map((segment) => {
			const at = cursor;
			cursor += segment.kind === "text" ? segment.text.length : segment.id.length;
			if (segment.kind === "ask") return <AskIdChip key={`a${start + at}`} id={segment.id} />;
			const piece = isolate ? isolateTechnicalRuns(segment.text) : segment.text;
			const local =
				piece === segment.text && fromHere !== undefined && fromHere >= at && fromHere < cursor
					? fromHere - at
					: undefined;
			return <Prose key={`p${start + at}`} text={piece} from={local} />;
		});
	};
	for (const m of text.matchAll(INLINE_RE)) {
		const idx = m.index ?? 0;
		if (idx > last) {
			/* Plain prose: paths and URLs get an isolate so an RTL paragraph cannot
			   detach their leading "/" or trailing punctuation (QA Q-6). */
			out.push(prose(last, idx, true));
		}
		const [full, code, linkText, href, bold1, bold2, strike, italic1, italic2] = m;
		const ref = from !== undefined && idx >= from ? revealText : undefined;
		if (code !== undefined) {
			const inner = code.slice(1, -1);
			/* ONE ID AND NOTHING ELSE, the desktop's rule: a span that also holds a
			   command stays a code span, because the reader was shown a command. */
			if (askChips && isAskId(inner)) out.push(<AskIdChip key={idx} id={inner} />);
			else out.push(
				<code
					key={idx} ref={ref}
					dir="ltr"
					className="rounded-xs bg-sunken px-0.5 font-mono text-mono-sm"
				>
					{inner}
				</code>,
			);
		} else if (linkText !== undefined && href !== undefined) {
			const safe = /^https?:\/\//i.test(href);
			if (safe) {
				out.push(
					<a
						key={idx} ref={ref}
						href={href}
						target="_blank"
						rel="noopener noreferrer"
						className="text-info underline underline-offset-2"
					>
						{linkText}
					</a>,
				);
			} else {
				/* Unsafe scheme: render the visible text, drop the target. */
				out.push(<span key={idx} ref={ref}>{linkText}</span>);
			}
		} else if (bold1 !== undefined || bold2 !== undefined) {
			out.push(
				<strong key={idx} ref={ref} className="font-semibold">
					{renderInline(bold1 ?? bold2 ?? "", undefined, askChips)}
				</strong>,
			);
		} else if (strike !== undefined) {
			out.push(
				<s key={idx} ref={ref} className="text-ink-muted">
					{strike}
				</s>,
			);
		} else if (italic1 !== undefined || italic2 !== undefined) {
			/* CHIPS INSIDE EMPHASIS, BUT NOTHING ELSE CHANGES when chips are off:
			   parsing the italic body recursively turned backticks and links inside
			   `*...*` into code/links for every Markdown consumer, which is a
			   behaviour change this feature has no business making (review m3). */
			out.push(
				<em key={idx} ref={ref}>
					{askChips ? renderInline(italic1 ?? italic2 ?? "", undefined, askChips) : (italic1 ?? italic2)}
				</em>,
			);
		} else {
			out.push(full);
		}
		last = idx + full.length;
	}
	if (last < text.length) out.push(prose(last, text.length, false));
	return out;
}

/* ------------------------------------------------------------------ */
/* Block                                                              */
/* ------------------------------------------------------------------ */

interface Block {
	kind: "code" | "heading" | "ul" | "ol" | "quote" | "hr" | "table" | "para";
	level?: number;
	lang?: string;
	/** A fence reached its closing line; an open one is a reply still streaming in. */
	closed?: boolean;
	lines: string[];
	/** Source positions, even where the renderer strips a heading/list marker. */
	start: number;
	lineStarts: number[];
}

/* A GFM table row: leading/trailing pipes, at least one inner pipe. */
const TABLE_ROW = /^\s*\|.*\|\s*$/;
/* The delimiter row under the header: |---|:--|--:| etc. */
const TABLE_DELIM = /^\s*\|?[\s:|-]+\|[\s:|-]*$/;

function splitTableRow(line: string): string[] {
	/* Strip the outer pipes, then split; a \| escape is honoured as a
	   literal pipe inside a cell. */
	const inner = line.trim().replace(/^\|/, "").replace(/\|$/, "");
	const cells: string[] = [];
	let cur = "";
	for (let i = 0; i < inner.length; i++) {
		const ch = inner[i];
		if (ch === "\\" && inner[i + 1] === "|") {
			cur += "|";
			i++;
		} else if (ch === "|") {
			cells.push(cur.trim());
			cur = "";
		} else {
			cur += ch;
		}
	}
	cells.push(cur.trim());
	return cells;
}

function splitBlocks(text: string, start = 0): Block[] {
	const blocks: Block[] = [];
	const lines = text.slice(start).split("\n");
	let offset = start;
	const offsets = lines.map((line) => {
		const position = offset;
		offset += line.length + 1;
		return position;
	});
	let i = 0;
	while (i < lines.length) {
		const head = i;
		const line = lines[i];
		const push = (block: Omit<Block, "start" | "lineStarts">, first = head) => {
			blocks.push({ ...block, start: offsets[head], lineStarts: block.lines.map((content, n) =>
				offsets[first + n] + lines[first + n].length - content.length) });
		};

		const fence = line.match(/^```(\w*)\s*$/);
		if (fence) {
			const buf: string[] = [];
			i++;
			while (i < lines.length && !/^```\s*$/.test(lines[i])) {
				buf.push(lines[i]);
				i++;
			}
			const closed = i < lines.length;
			i++; /* consume the closing fence (or EOF, which ends the block) */
			push({ kind: "code", lang: fence[1] || undefined, closed, lines: buf }, head + 1);
			continue;
		}

		const heading = line.match(/^(#{1,4})\s+(.*)$/);
		if (heading) {
			push({
				kind: "heading",
				level: heading[1].length,
				lines: [heading[2]],
			});
			i++;
			continue;
		}

		if (/^\s*(-{3,}|\*{3,})\s*$/.test(line)) {
			push({ kind: "hr", lines: [] });
			i++;
			continue;
		}

		if (/^\s*[-*+]\s+/.test(line)) {
			const buf: string[] = [];
			while (i < lines.length && /^\s*[-*+]\s+/.test(lines[i])) {
				buf.push(lines[i].replace(/^\s*[-*+]\s+/, ""));
				i++;
			}
			push({ kind: "ul", lines: buf });
			continue;
		}

		if (/^\s*\d+[.)]\s+/.test(line)) {
			const buf: string[] = [];
			while (i < lines.length && /^\s*\d+[.)]\s+/.test(lines[i])) {
				buf.push(lines[i].replace(/^\s*\d+[.)]\s+/, ""));
				i++;
			}
			push({ kind: "ol", lines: buf });
			continue;
		}

		if (/^\s*>\s?/.test(line)) {
			const buf: string[] = [];
			while (i < lines.length && /^\s*>\s?/.test(lines[i])) {
				buf.push(lines[i].replace(/^\s*>\s?/, ""));
				i++;
			}
			push({ kind: "quote", lines: buf });
			continue;
		}

		/* GFM table: a pipe row whose NEXT line is the delimiter. Without the
		   delimiter check a single prose line that happens to contain pipes
		   would be misread as a table. */
		if (
			TABLE_ROW.test(line) &&
			i + 1 < lines.length &&
			TABLE_DELIM.test(lines[i + 1])
		) {
			const buf: string[] = [line];
			i += 2; /* header + delimiter consumed; delimiter is structural */
			while (i < lines.length && TABLE_ROW.test(lines[i])) {
				buf.push(lines[i]);
				i++;
			}
			push({ kind: "table", lines: buf });
			continue;
		}

		/* Paragraph: consume until a blank line or a construct that starts
		   another block. A pipe row whose next line is a delimiter starts a
		   table, so the scanner stops there too. */
		const buf: string[] = [];
		while (
			i < lines.length &&
			lines[i].trim() !== "" &&
			!/^```/.test(lines[i]) &&
			!/^(#{1,4})\s/.test(lines[i]) &&
			!/^\s*[-*+]\s+/.test(lines[i]) &&
			!/^\s*\d+[.)]\s+/.test(lines[i]) &&
			!/^\s*>\s?/.test(lines[i]) &&
			!/^\s*(-{3,}|\*{3,})\s*$/.test(lines[i]) &&
			!(
				TABLE_ROW.test(lines[i]) &&
				i + 1 < lines.length &&
				TABLE_DELIM.test(lines[i + 1])
			)
		) {
			buf.push(lines[i]);
			i++;
		}
		if (buf.length > 0) {
			push({ kind: "para", lines: buf });
		} else {
			i++; /* blank line */
		}
	}
	return blocks;
}

const HEADING_CLASSES = [
	"text-heading font-semibold",
	"text-heading font-semibold",
	"text-body font-semibold",
	"text-body-sm font-semibold",
];

/* A CELL'S OWN BOX: `px-2` on each side, plus the table's own rule. */
/* A table cell's inline box around its text: `px-2` on each side. Only the
   FALLBACK when no cell has been rendered yet (the unit tree); a mounted table
   reads the real number off a cell (see `cellInlineOverflow`). */
const CELL_PADDING = 16;
/* The collapsed border between two columns: one 1 px rule per boundary, and
   `border-collapse` draws it once. */
const TABLE_RULE = 2;

/**
 * THE ONE FONT BOTH FORMS ARE MEASURED AND DRAWN IN (code review round 3, C3-1).
 *
 * The fit verdict is a text-measurement, so the font it is measured in must be
 * the font the cells are drawn in — and it must not depend on which branch is
 * currently mounted, or the same table gets two answers and can oscillate. One
 * class, on both wrappers, read back by the measurement.
 */
export const TABLE_FONT_CLASS = "lo-table-font text-body-sm";

/* The header cells' `font-semibold`, as a weight rather than a measurement fudge:
   the header font is built from the wrapper's size and family plus this weight,
   which is the same class the `<th>` renders with (a unit test pins the two
   together, so they cannot drift). */
export const TABLE_HEAD_WEIGHT = "600";

/** A column's inline overhead from a rendered cell's computed style: its two
 *  paddings plus any inline borders it draws. Read, never assumed — a class change
 *  to the cell's padding then cannot silently break the fit verdict. */
export function cellInlineOverflow(style: Pick<CSSStyleDeclaration, "paddingLeft" | "paddingRight" | "borderLeftWidth" | "borderRightWidth">): number {
	const px = (value: string) => Number.parseFloat(value) || 0;
	return Math.ceil(px(style.paddingLeft) + px(style.paddingRight) + px(style.borderLeftWidth) + px(style.borderRightWidth));
}

/** ONE canvas for every table's measurement (code review round 3, C3-6): a new
 *  context per `decide()` put an allocation on a path that runs per streamed
 *  frame and per observed resize. */
let tableProbe: CanvasRenderingContext2D | null | undefined;
function tableContext(): CanvasRenderingContext2D | null {
	if (tableProbe !== undefined) return tableProbe;
	tableProbe = typeof document === "undefined" ? null : document.createElement("canvas").getContext("2d");
	return tableProbe;
}

/** A cell's VISIBLE text: inline-markdown markers are syntax, not content. */
export function markdownTableText(cell: string): string {
	return cell
		.replace(/\[([^\]]*)\]\([^)]*\)/g, "$1")
		.replace(/[*_~`]/g, "")
		.trim();
}

/**
 * How narrow each column can honestly be: its longest WORD, not its longest
 * cell. `measure` is the engine (canvas `measureText` with the block's own font),
 * so these are real glyph advances rather than character counts.
 */
export function minContentWidths(
	head: readonly string[],
	rows: readonly (readonly string[])[],
	measure: (text: string, isHeader: boolean) => number,
	overhead: number = CELL_PADDING,
): number[] {
	const columns = Math.max(head.length, ...rows.map((row) => row.length), 0);
	const widths = Array.from({ length: columns }, () => 0);
	const widestWord = (text: string, index: number, isHeader: boolean) => {
		for (const word of markdownTableText(text).split(/\s+/)) {
			if (!word) continue;
			const width = measure(word, isHeader);
			if (width > (widths[index] ?? 0)) widths[index] = width;
		}
	};
	/* THE HEADER IS DRAWN SEMIBOLD, THE BODY IS NOT (QA round 4, Q4-1). Measuring
	   every word at the body font under-measured each column by the header's extra
	   advance — 13 px over two columns on the fixture table — so containers just
	   wide enough chose `table` and clipped the right edge inside `overflow:
	   hidden`. The header row is measured at the header's own font. */
	head.forEach((cell, index) => widestWord(cell, index, true));
	for (const row of rows) row.forEach((cell, index) => widestWord(cell, index, false));
	return Array.from(widths, (width) => Math.ceil(width) + overhead);
}

/**
 * A table is only honest AS a table when every column can hold its longest word.
 *
 * IT WAS NOT (design review round 2, D4; UX U4). `table-fixed` gave every column
 * an equal share of the phone's width, so cells broke mid-word — "claude/op",
 * "Claude / Opus" — at 360 and 390, where a table is the format an agent most
 * often needs. This is the honest test: do the longest words fit side by side?
 */
export function tableFitsIn(widths: readonly number[], available: number): boolean {
	return widths.reduce((total, width) => total + width, TABLE_RULE) <= available;
}

/**
 * One markdown table, in whichever of its two forms fits the column.
 *
 * FRAME 0 IS THE FINAL FORM (design D4): `available` and the word widths are read
 * in `useLayoutEffect`, so the stacked/table switch is committed before the
 * browser paints. `overflow-hidden` on the wrapper means even that measuring
 * pass cannot scroll the page sideways.
 *
 * The stacked form keeps TABLE SEMANTICS (`role="table"/"row"/"cell"`), because
 * the data is still a table to assistive tech even when the phone shows it as
 * `header: value` lines.
 */
function MarkdownTable({ lines, askChips = false }: { lines: readonly string[]; askChips?: boolean }) {
	const [header, ...rest] = lines;
	const head = splitTableRow(header ?? "");
	const rows = rest.map((row) => splitTableRow(row));
	/* BOTH FORMS RENDER THE SAME CELLS (code review round 3, C3-5). The stacked
	   branch used to iterate the header while the table branch iterated the row, so
	   a row with more cells than its header silently lost the extras in the form a
	   phone actually shows. One column count, decided once, for both. */
	const columns = Math.max(head.length, ...rows.map((row) => row.length), 0);
	const box = useRef<HTMLDivElement>(null);
	/* The width this table was last judged at: an unchanged width cannot change a
	   width-based verdict, so the observer's other box changes are ignored. */
	const lastWidth = useRef(0);
	const [fits, setFits] = useState(true);
	/* The table's CONTENT, as one string: a streamed table whose cells grow must
	   re-decide, while the arrays themselves are new on every render and cannot
	   be compared by identity. */
	const signature = [head.join("\u0000"), ...rows.map((row) => row.join("\u0000"))].join("\u0001");

	useLayoutEffect(() => {
		const decide = () => {
			const node = box.current;
			if (!node) return;
			const available = node.clientWidth;
			/* A host with no layout engine (the vitest tree) reports 0. Keeping the
			   table form there is the honest default: "not measured" is not evidence
			   that the columns do not fit, and the pure helpers carry that coverage. */
			/* `!(available > 0)` rather than `<= 0`: an engine that reports the
			   property as absent (undefined) must take the same "not measured" path
			   as one that reports 0, instead of falling through to a comparison
			   against undefined — which is silently false and stacks every table. */
			if (!(available > 0)) {
				setFits(true);
				return;
			}
			/* THE WIDTH IS THE ONLY TRIGGER THAT CAN CHANGE THE ANSWER (code review
			   round 3, C3-1). The old code re-decided on ANY observed box change,
			   and switching forms changes the wrapper's HEIGHT — so a table near the
			   boundary could oscillate table → stacked → table, one observer callback
			   per switch. A decision that depends on width must not be re-run because
			   of a change the decision itself caused; measured widths are remembered
			   and an unchanged width short-circuits. */
			if (available === lastWidth.current) return;
			lastWidth.current = available;
			/* MEASURE THE FONT THE CELLS ARE DRAWN IN (C3-1). `getComputedStyle(node)`
			   read the WRAPPER's font, which differed by branch (14 px inheriting in
			   the table branch vs the stacked branch's own 13 px) and from the cells'
			   own `text-body-sm`. The verdict was ~7.7 % wide in the table branch —
			   over-stacking tables that fit — and not idempotent. `TABLE_FONT_CLASS`
			   is applied to both wrappers AND read here, so the metric cannot drift
			   from the drawn text again. */
			const probe = tableContext();
			if (!probe) {
				setFits(true);
				return;
			}
			/* BOTH WRAPPERS CARRY `TABLE_FONT_CLASS` THEMSELVES (round 4, C4-2), so
			   this reads the wrapper — no descendant query and no dead fallback. The
			   header's weight is the one thing the wrapper does not carry: header
			   cells are `font-semibold` and the body is not, so the two row groups are
			   measured at the fonts they are actually drawn in. A rendered `<th>`/
			   `<td>` gives those fonts exactly; before one exists, the header font is
			   built from the wrapper plus the same weight class the header uses. */
			const style = getComputedStyle(node);
			const family = `${style.fontStyle} ${style.fontSize} ${style.fontFamily}`.trim();
			const weight = (element: Element | null, fallback: string) => {
				if (!element) return fallback;
				const computed = getComputedStyle(element);
				return `${computed.fontWeight} ${computed.fontSize} ${computed.fontFamily}`;
			};
			const bodyFont = weight(node.querySelector("td"), family);
			const headFont = weight(node.querySelector("th"), `${TABLE_HEAD_WEIGHT} ${style.fontSize} ${style.fontFamily}`);
			/* The inline box a column needs beyond its text, read off a real cell
			   (padding + its own borders). A cell that reports NO inline box is not the
			   styled cell — a host with no stylesheet (the unit tree) or a class the
			   build failed to emit — and falling back to the class constant keeps the
			   verdict from ever under-estimating, which is the direction that clips. */
			const cell: HTMLElement | null = node.querySelector("td");
			const measured = cell ? cellInlineOverflow(getComputedStyle(cell)) : 0;
			const overhead = measured > 0 ? measured : CELL_PADDING;
			const measure = (text: string, isHeader: boolean) => {
				probe.font = isHeader ? headFont : bodyFont;
				return probe.measureText(text).width;
			};
			setFits(tableFitsIn(minContentWidths(head, rows, measure, overhead), available));
		};
		decide();
		if (typeof ResizeObserver === "undefined" || !box.current) return;
		const observer = new ResizeObserver(decide);
		observer.observe(box.current);
		return () => observer.disconnect();
		// eslint-disable-next-line react-hooks/exhaustive-deps -- `signature` IS this table's content
	}, [signature]);

	if (!fits) {
		return (
			<div
				ref={box}
				role="table"
				className={cn("max-w-full overflow-hidden rounded-sm border border-hairline", TABLE_FONT_CLASS)}
			>
				{rows.map((row, rowIndex) => (
					<div
						key={rowIndex}
						role="row"
						className={rowIndex > 0 ? "border-t border-hairline px-2 py-1.5" : "px-2 py-1.5"}
					>
						{Array.from({ length: columns }, (_, column) => (
							/* THE HEADER IS PART OF THE CELL (code review round 3, C3-4). The
							   label used to be `aria-hidden`, so a screen reader heard the
							   value alone and nothing tied it to its column — the `<th>` form
							   it replaces supplied that natively, so hiding it was a
							   regression in the form that claims to preserve the table.
							   Visible text in the same cell IS the association. */
							<div key={column} role="cell" className="flex min-w-0 gap-2">
								<span className="shrink-0 break-words text-ink-dim">{markdownTableText(head[column] ?? "")}:</span>
								{/* `break-words` splits ONLY a token too long to fit a line by
								    itself (a URL, a hash). An ordinary word always fits, so it
								    is never cut mid-word. */}
								<span className="min-w-0 break-words text-ink">{renderInline(row[column] ?? "", undefined, askChips)}</span>
							</div>
						))}
					</div>
				))}
			</div>
		);
	}

	return (
		<div ref={box} className={cn("max-w-full overflow-hidden rounded-sm border border-hairline", TABLE_FONT_CLASS)}>
			<table className="w-full table-auto border-collapse">
				<thead>
					<tr className="bg-sunken">
						{Array.from({ length: columns }, (_, column) => (
							<th
								key={column}
								className="border-b border-hairline px-2 py-1 text-start font-semibold text-ink"
							>
								{renderInline(head[column] ?? "", undefined, askChips)}
							</th>
						))}
					</tr>
				</thead>
				<tbody>
					{rows.map((row, rowIndex) => (
						<tr key={rowIndex}>
							{Array.from({ length: columns }, (_, column) => (
								<td
									key={column}
									className="border-b border-hairline px-2 py-1 align-top text-ink-muted last:border-b-0"
								>
									{renderInline(row[column] ?? "", undefined, askChips)}
								</td>
							))}
						</tr>
					))}
				</tbody>
			</table>
		</div>
	);
}

interface ParsedMarkdown {
	text: string;
	blocks: Block[];
}

function parseMarkdown(text: string, previous: ParsedMarkdown | null): Block[] {
	if (previous?.text === text) return previous.blocks;
	if (!previous || !text.startsWith(previous.text)) return splitBlocks(text);
	// An append can change the type of the last line (e.g. a rule becomes
	// prose), which may rejoin its previous block. Keep those two blocks as
	// parser lookbehind; earlier lines already have a fixed following line.
	const boundary = Math.max(0, previous.blocks.length - 2);
	const parsed = splitBlocks(text, previous.blocks[boundary]?.start ?? 0);
	return previous.blocks.slice(0, boundary).concat(parsed.map((next, i) => {
		const old = previous.blocks[boundary + i];
		return old && old.start === next.start && old.kind === next.kind && old.lang === next.lang &&
			old.level === next.level && old.closed === next.closed && old.lines.length === next.lines.length &&
			old.lines.every((line, j) => line === next.lines[j] && old.lineStarts[j] === next.lineStarts[j]) ? old : next;
	}));
}

const MarkdownBlock = memo(function MarkdownBlock({ block: b, from, askChips = false }: {
	block: Block; from?: number; askChips?: boolean;
}) {
	const line = (index: number) => renderInline(b.lines[index], from === undefined || index !== b.lines.length - 1
		? undefined : Math.max(0, from - b.lineStarts[index]), askChips);
	const ref = from !== undefined && b.start >= from ? revealText : undefined;
	// A new block already fades as a unit; only append inside old blocks needs
	// an inline effect. A diagram keys itself by source, not its fence position.
	const inline = (index: number) => ref ? renderInline(b.lines[index], undefined, askChips) : line(index);
	switch (b.kind) {
		case "code": {
			const source = b.lines.join("\n");
			const plain = (
				/* WRAPS, NEVER SCROLLS SIDEWAYS: keep indentation and line breaks
				   without letting a long URL or hash widen the phone column. */
				<pre ref={ref} dir="ltr" className="max-w-full rounded-sm bg-sunken p-2 font-mono text-mono-sm leading-snug whitespace-pre-wrap [overflow-wrap:anywhere]">
					{source}
				</pre>
			);
			if (!shouldDrawDiagram({ lang: b.lang, closed: b.closed === true, source })) return plain;
			return <DiagramBlock source={source} fallback={plain} />;
		}
		case "heading":
			return <div ref={ref} className={HEADING_CLASSES[(b.level ?? 1) - 1]}>{inline(0)}</div>;
		case "hr":
			return <hr ref={ref} className="border-hairline" />;
		case "ul":
			return <ul ref={ref} className="flex list-disc flex-col gap-0.5 ps-5">
				{b.lines.map((_, i) => <li key={b.lineStarts[i]}>{inline(i)}</li>)}
			</ul>;
		case "ol":
			return <ol ref={ref} className="flex list-decimal flex-col gap-0.5 ps-5">
				{b.lines.map((_, i) => <li key={b.lineStarts[i]}>{inline(i)}</li>)}
			</ol>;
		case "quote":
			return <div ref={ref} className="border-s-2 border-control ps-3 text-ink-muted">
				{b.lines.map((_, i) => <div key={b.lineStarts[i]}>{inline(i)}</div>)}
			</div>;
		case "table":
			return <MarkdownTable lines={b.lines} askChips={askChips} />;
		default:
			return <p ref={ref}>
				{b.lines.map((_, i) => <span key={b.lineStarts[i]}>
					{i > 0 ? <br /> : null}{inline(i)}
				</span>)}
			</p>;
	}
});

type SelectionGuardProps = { scope: () => Node | null; children: ReactNode };
type SelectionSnapshot = { selection: Selection; anchor: Node; focus: Node; anchorOffset: number; focusOffset: number };

/** React writes CharacterData for an append inside a word. The browser keeps
 * the node but can reset its Selection offsets to zero. Take the snapshot
 * before that write and restore only ranges whose original endpoints survived;
 * don't guess a new range across a genuine Markdown structure change. */
class SelectionGuard extends Component<SelectionGuardProps> {
	getSnapshotBeforeUpdate(): SelectionSnapshot | null {
		const root = this.props.scope();
		const selection = root?.ownerDocument?.getSelection();
		if (!selection || selection.isCollapsed) return null;
		const { anchorNode: anchor, focusNode: focus, anchorOffset, focusOffset } = selection;
		if (!root || !anchor || !focus || (!root.contains(anchor) && !root.contains(focus))) return null;
		return { selection, anchor, focus, anchorOffset, focusOffset };
	}

	componentDidUpdate(_props: SelectionGuardProps, _state: unknown, snapshot: SelectionSnapshot | null) {
		if (!snapshot) return;
		const { selection, anchor, focus, anchorOffset, focusOffset } = snapshot;
		const length = (node: Node) => node.nodeType === Node.TEXT_NODE ? node.textContent?.length ?? 0 : node.childNodes.length;
		if (!anchor.isConnected || !focus.isConnected || length(anchor) < anchorOffset || length(focus) < focusOffset) return;
		if (selection.anchorNode !== anchor || selection.focusNode !== focus ||
			selection.anchorOffset !== anchorOffset || selection.focusOffset !== focusOffset) {
			selection.setBaseAndExtent(anchor, anchorOffset, focus, focusOffset);
		}
	}

	render() {
		return this.props.children;
	}
}

// Settled block objects are shared by the incremental parser. Reuse the
// closed fence's source and keys instead of joining and hashing it per token.
const diagramOwners = new WeakMap<Block, { source: string; keys: string[] }>();
function diagramOwner(block: Block) {
	if (block.kind !== "code" || !block.closed || block.lang?.toLowerCase() !== "mermaid") return;
	const previous = diagramOwners.get(block);
	if (previous) return previous;
	const source = block.lines.join("\n");
	if (!shouldDrawDiagram({ lang: block.lang, closed: true, source })) return;
	const owner = { source, keys: [] as string[] };
	diagramOwners.set(block, owner);
	return owner;
}

/**
 * `live` marks the answer that is still streaming. It does two things: the
 * received `text` is released at a word-by-word pace (`usePacedText`) instead of
 * in the lumps the transport delivers, and each appended suffix fades in
 * (`stream-reveal.ts`). Everything below renders `shown`, a prefix of `text`
 * that becomes `text` itself the moment `live` goes false, so settled output is
 * byte-identical to the received text.
 */
export const Markdown = memo(function Markdown({ text: received, live = false, askChips = false }: { text: string; live?: boolean; askChips?: boolean }) {
	const root = useRef<HTMLDivElement>(null);
	const text = usePacedText(received, live, root);
	const committed = useRef<ParsedMarkdown | null>(null);
	const previous = committed.current;
	const blocks = parseMarkdown(text, previous);
	const append = previous && text.startsWith(previous.text);
	const grew = !!append && previous.text.length < text.length;
	const from = live && grew && allowsReveal() ? previous.text.length : undefined;
	useLayoutEffect(() => {
		if (!live || (previous && !append)) settleReveals(root.current);
		committed.current = { text, blocks };
	}, [text, live, blocks, previous, append]);
	const diagramSeen = new Map<string, number>();
	return <div ref={root} className="flex min-w-0 flex-col gap-1.5 break-words">
		<SelectionGuard scope={() => grew ? root.current : null}>
			{/* A burst can complete pages in one snapshot. Reveal the tail, not a
			    separate animation on every just-arrived block or line. */}
			{blocks.map((block, i) => {
				let key: string | number = block.start;
				const owner = diagramOwner(block);
				if (owner) {
					// Key the owner, not only the drawing: corrections before the
					// fence must not close its viewer or lose its copy/focus state.
					const occurrence = diagramSeen.get(owner.source) ?? 0;
					diagramSeen.set(owner.source, occurrence + 1);
					key = owner.keys[occurrence] ??= diagramKey(owner.source, occurrence);
				}
				return <MarkdownBlock key={key} block={block} askChips={askChips}
					from={i === blocks.length - 1 && block !== previous?.blocks[i] ? from : undefined} />;
			})}
		</SelectionGuard>
	</div>;
});
