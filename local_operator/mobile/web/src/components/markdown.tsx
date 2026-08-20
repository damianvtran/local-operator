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
import type { ReactNode } from "react";

let keySeq = 0;
function key(): string {
	return `md-${keySeq++}`;
}

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

function renderInline(text: string): ReactNode[] {
	const out: ReactNode[] = [];
	let last = 0;
	for (const m of text.matchAll(INLINE_RE)) {
		const idx = m.index ?? 0;
		if (idx > last) out.push(text.slice(last, idx));
		const [full, code, linkText, href, bold1, bold2, strike, italic1, italic2] = m;
		if (code !== undefined) {
			out.push(
				<code
					key={key()}
					className="rounded-xs bg-sunken px-1 font-mono text-mono-sm"
				>
					{code.slice(1, -1)}
				</code>,
			);
		} else if (linkText !== undefined && href !== undefined) {
			const safe = /^https?:\/\//i.test(href);
			if (safe) {
				out.push(
					<a
						key={key()}
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
				out.push(<span key={key()}>{linkText}</span>);
			}
		} else if (bold1 !== undefined || bold2 !== undefined) {
			out.push(
				<strong key={key()} className="font-semibold">
					{renderInline(bold1 ?? bold2 ?? "")}
				</strong>,
			);
		} else if (strike !== undefined) {
			out.push(
				<s key={key()} className="text-ink-muted">
					{strike}
				</s>,
			);
		} else if (italic1 !== undefined || italic2 !== undefined) {
			out.push(<em key={key()}>{italic1 ?? italic2}</em>);
		} else {
			out.push(full);
		}
		last = idx + full.length;
	}
	if (last < text.length) out.push(text.slice(last));
	return out;
}

/* ------------------------------------------------------------------ */
/* Block                                                              */
/* ------------------------------------------------------------------ */

interface Block {
	kind: "code" | "heading" | "ul" | "ol" | "quote" | "hr" | "para";
	level?: number;
	lang?: string;
	lines: string[];
}

function splitBlocks(text: string): Block[] {
	const blocks: Block[] = [];
	const lines = text.split("\n");
	let i = 0;
	while (i < lines.length) {
		const line = lines[i];

		const fence = line.match(/^```(\w*)\s*$/);
		if (fence) {
			const buf: string[] = [];
			i++;
			while (i < lines.length && !/^```\s*$/.test(lines[i])) {
				buf.push(lines[i]);
				i++;
			}
			i++; /* consume the closing fence (or EOF, which ends the block) */
			blocks.push({ kind: "code", lang: fence[1] || undefined, lines: buf });
			continue;
		}

		const heading = line.match(/^(#{1,4})\s+(.*)$/);
		if (heading) {
			blocks.push({
				kind: "heading",
				level: heading[1].length,
				lines: [heading[2]],
			});
			i++;
			continue;
		}

		if (/^\s*(-{3,}|\*{3,})\s*$/.test(line)) {
			blocks.push({ kind: "hr", lines: [] });
			i++;
			continue;
		}

		if (/^\s*[-*+]\s+/.test(line)) {
			const buf: string[] = [];
			while (i < lines.length && /^\s*[-*+]\s+/.test(lines[i])) {
				buf.push(lines[i].replace(/^\s*[-*+]\s+/, ""));
				i++;
			}
			blocks.push({ kind: "ul", lines: buf });
			continue;
		}

		if (/^\s*\d+[.)]\s+/.test(line)) {
			const buf: string[] = [];
			while (i < lines.length && /^\s*\d+[.)]\s+/.test(lines[i])) {
				buf.push(lines[i].replace(/^\s*\d+[.)]\s+/, ""));
				i++;
			}
			blocks.push({ kind: "ol", lines: buf });
			continue;
		}

		if (/^\s*>\s?/.test(line)) {
			const buf: string[] = [];
			while (i < lines.length && /^\s*>\s?/.test(lines[i])) {
				buf.push(lines[i].replace(/^\s*>\s?/, ""));
				i++;
			}
			blocks.push({ kind: "quote", lines: buf });
			continue;
		}

		/* Paragraph: consume until a blank line or a construct that starts
		   another block. */
		const buf: string[] = [];
		while (
			i < lines.length &&
			lines[i].trim() !== "" &&
			!/^```/.test(lines[i]) &&
			!/^(#{1,4})\s/.test(lines[i]) &&
			!/^\s*[-*+]\s+/.test(lines[i]) &&
			!/^\s*\d+[.)]\s+/.test(lines[i]) &&
			!/^\s*>\s?/.test(lines[i]) &&
			!/^\s*(-{3,}|\*{3,})\s*$/.test(lines[i])
		) {
			buf.push(lines[i]);
			i++;
		}
		if (buf.length > 0) {
			blocks.push({ kind: "para", lines: buf });
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

export function Markdown({ text }: { text: string }) {
	const blocks = splitBlocks(text);
	return (
		<div className="flex flex-col gap-1.5">
			{blocks.map((b) => {
				switch (b.kind) {
					case "code":
						return (
							<pre
								key={key()}
								className="lo-scroll overflow-x-auto rounded-sm bg-sunken p-2 font-mono text-mono-sm leading-snug whitespace-pre"
							>
								{b.lines.join("\n")}
							</pre>
						);
					case "heading":
						return (
							<div
								key={key()}
								className={HEADING_CLASSES[(b.level ?? 1) - 1]}
							>
								{renderInline(b.lines[0])}
							</div>
						);
					case "hr":
						return (
							<hr key={key()} className="border-hairline" />
						);
					case "ul":
						return (
							<ul key={key()} className="flex list-disc flex-col gap-0.5 pl-5">
								{b.lines.map((li) => (
									<li key={key()}>{renderInline(li)}</li>
								))}
							</ul>
						);
					case "ol":
						return (
							<ol
								key={key()}
								className="flex list-decimal flex-col gap-0.5 pl-5"
							>
								{b.lines.map((li) => (
									<li key={key()}>{renderInline(li)}</li>
								))}
							</ol>
						);
					case "quote":
						return (
							<div
								key={key()}
								className="border-l-2 border-control pl-3 text-ink-muted"
							>
								{b.lines.map((q) => (
									<div key={key()}>{renderInline(q)}</div>
								))}
							</div>
						);
					default:
						return (
							<p key={key()}>
								{b.lines.map((line, i) => (
									<span key={key()}>
										{i > 0 ? <br /> : null}
										{renderInline(line)}
									</span>
								))}
							</p>
						);
				}
			})}
		</div>
	);
}
