/**
 * The phone's `$skill` token grammar — a port of the TUI's, not a rewrite.
 *
 * WHY THIS FILE EXISTS. The TUI's composer opens its skill list from
 * `local_operator/tui/widgets/command_picker.py::skill_token` (+ `_active_sigil` and
 * `local_operator/sigils.is_boundary`). The phone must answer the same three
 * questions — is the caret inside a `$name` token, what is the name so far, and
 * what replaces the span on a pick — or the two surfaces will disagree about
 * which keystrokes mean "show me my skills". `skill-token.parity.json` pins the
 * answers case for case: it is GENERATED from the Python functions by
 * `scripts/generate_skill_token_parity.py` and asserted by both
 * `skill-token.test.ts` (here) and `tests/unit/mobile/test_tui_bridge.py`.
 *
 * THE BOUNDARY RULE IS HAND-WRITTEN, NOT `\s`. Python's `str.isspace()` and the
 * JS regex class `\s` disagree on three families that decide real cases:
 *
 *  - `\ufeff` (ZERO WIDTH NO-BREAK SPACE) is `\s` in JS and NOT whitespace in
 *    Python, so a naive `\s` port would open a token after it where the TUI
 *    would not;
 *  - `\x85` (NEL) and `\x1c`–`\x1f` are whitespace in Python and NOT in JS, so
 *    the naive port would leave those tokens closed where the TUI opens them.
 *
 * :data:`PY_SPACE` is Python's exact set (Unicode `Zs` plus the RTL/useful
 * controls that carry bidirectional class WS/B/S). Every question below that
 * Python answers with `.isspace()` / `.strip()` reads it through
 * :func:`isPySpace`, so there is one spelling of the rule on this side too.
 *
 * WHAT IS DELIBERATELY DIFFERENT FROM THE TUI, and why:
 *
 *  - NO COMMAND CLAIM. The TUI's `skill_token` arbitrates against `/`-commands
 *    (`/team ops $research`); the phone keeps the two surfaces apart the other
 *    way: a slash draft closes the skill tray (`isSlashDraft` in
 *    `components/composer.tsx`), so `/team x $skill` is out of scope and no
 *    second copy of the claim rule exists here. One caret, one list.
 *  - NO FUZZY MATCHING. The TUI ranks with its corpus-wide scorer; the phone
 *    picker has no ranked keyboard selection (the arrows walk the list in
 *    catalogue order), so rows are a case-insensitive PREFIX of the typed
 *    name. The lowercase-noise gate is kept (`$PATH`, `$5`, a bare inline `$`
 *    stay closed), because a wrongly opened list is a draft-rewriting hazard,
 *    not merely noise — see `skillMatches`.
 *  - CASE-INSENSITIVE PREFIX, where the TUI's inline rule is case-SENSITIVE.
 *    A named divergence, forced by the platform: phone keyboards auto-capitalise
 *    the first letter of a sentence, so `$Research` typed inline must find
 *    `research` here even though the TUI would stay closed. The evidence rule
 *    only requires a lowercase letter SOMEWHERE in the query, which keeps the
 *    safety property (`$PATH` has none) while accepting `$Research`.
 *
 * THE EVIDENCE RULE'S CHARACTER TEST is Unicode's `Lowercase` property, not
 * the general category `Ll`. The TUI asks Python `any(char.islower())`, and a
 * single character's `str.islower()` is true exactly for `Lowercase` — which is
 * `Ll` PLUS `Other_Lowercase`. `\p{Ll}` disagrees on 311 code points: `ª`,
 * `º`, `ʰ`, `ᵃ`, `ⓐ` are lowercase to Python and are not `Ll`, so `$ªx` opened
 * the TUI's list and stayed shut here (agent review m1). `\p{Lowercase}`
 * matches Python's rule. The residue is Unicode VERSION skew, not a rule
 * difference — Python 3.12 carries Unicode 15.0, the browser runtime 17.0, so
 * 52 code points added since 15.0 are `Lowercase` to the browser alone; no
 * shipped skill name reaches them, and this gate only decides whether a list
 * opens, never which row matches.
 */

import type { SkillEntry } from "../types";

/** Where the active `$name` token sits, and the name typed so far. */
export interface SkillToken {
	/** Index of the `$` itself. */
	start: number;
	/** The name between `$` and the token's end (may be `""`). */
	query: string;
	/** Index of the first cell past the token. */
	end: number;
}

/**
 * Python `str.isspace()`'s character set, exactly.
 *
 * `\t\n\v\f\r \x1c-\x1f \x85\xa0` plus the Unicode spaces `\u1680`,
 * `\u2000-\u200a`, `\u2028`, `\u2029`, `\u202f`, `\u205f`, `\u3000`.
 * `\ufeff` and `\u200b` are NOT in it, which is the whole point of writing it
 * out instead of using `\s`.
 */
const PY_SPACE_SET = new Set([
	"\t",
	"\n",
	"\v",
	"\f",
	"\r",
	"\x1c",
	"\x1d",
	"\x1e",
	"\x1f",
	" ",
	"\x85",
	"\xa0",
	"\u1680",
	"\u2000",
	"\u2001",
	"\u2002",
	"\u2003",
	"\u2004",
	"\u2005",
	"\u2006",
	"\u2007",
	"\u2008",
	"\u2009",
	"\u200a",
	"\u2028",
	"\u2029",
	"\u202f",
	"\u205f",
	"\u3000",
]);

/** Whether `char` is whitespace in Python's sense (see :data:`PY_SPACE_SET`). */
export function isPySpace(char: string): boolean {
	return PY_SPACE_SET.has(char);
}

/**
 * `(line, lineStart, column)` for `caret` within `text`.
 *
 * The port of `command_picker._line_of_cursor`: `caret` is a whole-buffer
 * offset (null means "the end", which is where a field's caret sits before the
 * user has selected anything), clamped into range so a stale caret from a
 * delete cannot index out of the buffer. Lines split on `\n`; a trailing `\r`
 * from a CRLF paste is stripped so the token's `end` never includes it.
 */
function lineOfCursor(
	text: string,
	caret: number | null | undefined,
): { line: string; lineStart: number; column: number } {
	let bounded = caret ?? text.length;
	if (bounded > text.length) bounded = text.length;
	if (bounded < 0) bounded = 0;
	const lineStart = text.slice(0, bounded).lastIndexOf("\n") + 1;
	let lineEnd = text.indexOf("\n", bounded);
	if (lineEnd === -1) lineEnd = text.length;
	let line = text.slice(lineStart, lineEnd);
	if (line.endsWith("\r")) line = line.slice(0, -1);
	return { line, lineStart, column: bounded - lineStart };
}

/** A sigil opens a token only at a word boundary: line start or after space. */
function isBoundary(line: string, index: number): boolean {
	return index === 0 || isPySpace(line[index - 1]);
}

/** Index of the boundary `$` the caret is editing on its line, or null. */
function activeDollar(line: string, column: number): number | null {
	let candidate: number | null = null;
	for (let index = 0; index < line.length; index += 1) {
		if (line[index] === "$" && isBoundary(line, index) && index <= column) candidate = index;
	}
	return candidate;
}

/**
 * The `$skill` token the caret sits in, or `null`.
 *
 * `skill_token`'s rule, verbatim: the LAST boundary `$` at or before the caret
 * on the caret's line; the token runs to the next whitespace; the caret must be
 * inside it (`column <= end` — at the end is inside, one past is not). A `$`
 * glued to a word character is punctuation (`costs$5`), never a sigil.
 *
 * `query === ""` for a bare `$`, which the caller decides what to do with
 * (leading lists everything, inline stays closed).
 */
export function skillToken(text: string, caret?: number | null): SkillToken | null {
	const { line, lineStart, column } = lineOfCursor(text, caret);
	const dollar = activeDollar(line, column);
	if (dollar === null) return null;
	let end = dollar + 1;
	while (end < line.length && !isPySpace(line[end])) end += 1;
	if (column > end) return null;
	return { start: lineStart + dollar, query: line.slice(dollar + 1, end), end: lineStart + end };
}

/**
 * Whether `token` opens `text`, ignoring leading whitespace.
 *
 * `skill_token_is_leading`'s definition, and the bit that decides which
 * matching rule applies: a leading token keeps the "list everything" affordance
 * a bare `$` deserves; an inline one needs positive evidence. `.trim()` is not
 * used because its whitespace set is JS's, not Python's (see the module note).
 */
export function isLeadingSkillToken(text: string, token: SkillToken): boolean {
	return ![...text.slice(0, token.start)].some((char) => !isPySpace(char));
}

/**
 * Rows for `query`, in catalogue order.
 *
 * INLINE keeps the TUI's lowercase-evidence rule: a query with no lowercase
 * letter matches nothing. That single clause closes `$PATH`, `$HOME`, `$5`, a
 * bare inline `$` and every uppercase-prefixed token, because skill names are
 * lowercase by convention while environment variables are uppercase by an
 * equally strong one. Skipping it would let `translate to $LANG` open a list
 * over prose, and Enter — which the open list re-routes to a completion — would
 * rewrite the draft with nothing sent.
 *
 * LEADING has no such rule (the sigil is the first thing typed; nothing else
 * can be meant): a bare `$` lists everything and any prefix narrows it.
 *
 * Matching is a case-insensitive PREFIX, never the TUI's fuzzy scorer — a list
 * walked in catalogue order has no ranking to earn, and a prefix is the rule a
 * user can predict from what they typed. See the module note for the named
 * divergence.
 */
export function skillMatches(
	query: string,
	skills: readonly SkillEntry[],
	leading: boolean,
): SkillEntry[] {
	if (!leading && !/\p{Lowercase}/u.test(query)) return [];
	if (query === "") return leading ? mentionable(skills) : [];
	const needle = query.toLowerCase();
	return mentionable(skills).filter((skill) =>
		skill.name.toLowerCase().startsWith(needle),
	);
}

/**
 * Whether the `$` grammar can express this name at all.
 *
 * The token ends at the first whitespace (this file's `skillToken`, and the
 * server's `skill_name_spans`), so a name containing any Python whitespace can
 * only ever be typed as its first word: the row is offered, the pick inserts
 * `$ux spaced `, the mention parses as `ux`, and the send goes out with no
 * skill attached and no skip receipt — the user believes the skill ran (UX
 * review U3). Offering it is therefore a promise the grammar cannot keep, and
 * the list is the only place that can decline to make it. Empty names are
 * excluded for the same reason: `$` alone is the "list everything" query.
 *
 * The relay's catalogue is the only writer of this field and its own names are
 * filesystem stems, so this rejects what a stray `name: ux spaced` fixture or a
 * future catalogue with free-form names would otherwise advertise.
 */
export function isMentionableSkillName(name: string): boolean {
	if (name === "") return false;
	return ![...name].some((char) => isPySpace(char));
}

/** The catalogue minus every name a mention cannot express. */
function mentionable(skills: readonly SkillEntry[]): SkillEntry[] {
	return skills.filter((skill) => isMentionableSkillName(skill.name));
}

/**
 * Replace the token's span with `$name` — the whole edit.
 *
 * `_skill_span_replacement`'s arithmetic: nothing outside `[start, end)` is
 * rewritten, so prose before and after the token survives byte for byte. A
 * space goes in only when the next character is not whitespace, because that
 * space is the token terminator the list's close depends on; when the user
 * already typed whitespace, that is the separator and it stays.
 *
 * The caret lands one cell past the separator, except that a newline terminates
 * the token without being a place to park the caret — the request continues on
 * the token's line, so it stops after the name.
 */
export function skillSpanReplacement(
	text: string,
	token: SkillToken,
	name: string,
): { text: string; caret: number } {
	const tail = text.slice(token.end);
	const separator = tail === "" || !isPySpace(tail[0]) ? " " : "";
	const completed = `${text.slice(0, token.start)}$${name}${separator}${tail}`;
	if (tail.length > 0 && (tail[0] === "\r" || tail[0] === "\n")) {
		return { text: completed, caret: token.start + name.length + 1 };
	}
	return { text: completed, caret: token.start + name.length + 2 };
}
