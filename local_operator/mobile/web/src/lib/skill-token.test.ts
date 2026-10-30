// The phone's `$skill` grammar against the TUI's own answers.
//
// `skill-token.parity.json` is GENERATED from the Python parser by
// `scripts/generate_skill_token_parity.py` (`skill_token`,
// `skill_token_is_leading` and the SKILL branch of `completion_for`) and
// asserted on the Python side by `tests/unit/mobile/test_tui_bridge.py`. Both
// suites read the one artifact, so neither surface can drift while its own
// tests stay green.
//
// The fixture carries the caret as a `head`/`tail` split and the spans as
// strings — never offsets — because Python counts code points and JavaScript
// counts UTF-16 code units. An emoji before the sigil is exactly the case that
// makes a stored number lie.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
	isLeadingSkillToken,
	skillMatches,
	skillSpanReplacement,
	skillToken,
} from "./skill-token";
import type { SkillEntry } from "../types";

const HERE = __dirname;

interface ParityCase {
	note: string;
	head: string;
	tail: string;
	prefix?: string;
	token: string | null;
	query?: string;
	suffix?: string;
	leading?: boolean;
	insertHead?: string;
	insertTail?: string;
}

const FIXTURE = JSON.parse(
	readFileSync(join(HERE, "skill-token.parity.json"), "utf8"),
) as { cases: ParityCase[] };

const SKILLS: SkillEntry[] = [
	{ name: "planning", description: "Plan the work", hidden: false },
	{ name: "research", description: "Read the sources", hidden: false },
	{ name: "pathfinder", description: "Walk a tree", hidden: true },
	{ name: "code-review", description: "Review a diff", hidden: false },
	{ name: "under_score", description: "Names with an underscore", hidden: false },
	{ name: "dash-name", description: "Names with a dash", hidden: false },
];

describe("the generated token fixture matches the port, case for case", () => {
	it("keeps enough cases to cover every branch the two surfaces share", () => {
		expect(FIXTURE.cases.length).toBeGreaterThanOrEqual(30);
	});

	it.each(FIXTURE.cases.map((c) => [c.note, c] as const))("%s", (_note, c) => {
		const text = c.head + c.tail;
		const caret = c.head.length;
		const token = skillToken(text, caret);
		if (c.token === null) {
			expect(token).toBeNull();
			return;
		}
		expect(token).not.toBeNull();
		if (token === null) return;
		// Strings, not offsets: the same three slices in either language.
		expect(text.slice(0, token.start)).toBe(c.prefix);
		expect(text.slice(token.start, token.end)).toBe(c.token);
		expect(token.query).toBe(c.query);
		expect(text.slice(token.end)).toBe(c.suffix);
		expect(isLeadingSkillToken(text, token)).toBe(c.leading);
		const inserted = skillSpanReplacement(text, token, "planning");
		expect(inserted.text.slice(0, inserted.caret)).toBe(c.insertHead);
		expect(inserted.text.slice(inserted.caret)).toBe(c.insertTail);
	});
});

describe("matching keeps the TUI's positive-evidence rule", () => {
	it("lists the whole catalogue for a bare LEADING $", () => {
		expect(skillMatches("", SKILLS, true)).toEqual(SKILLS);
	});

	it("leaves a bare inline $ closed", () => {
		expect(skillMatches("", SKILLS, false)).toEqual([]);
	});

	it("never opens on an uppercase token — $PATH, $HOME, $LANG", () => {
		for (const query of ["PATH", "HOME", "LANG", "DEBUG"]) {
			expect(skillMatches(query, SKILLS, false)).toEqual([]);
		}
	});

	it("never opens on a caseless token like $5 or $_private", () => {
		for (const query of ["5", "_private", "", "123"]) {
			expect(skillMatches(query, SKILLS, false)).toEqual([]);
		}
	});

	it("matches a case-insensitive prefix, so a phone's auto-capitalised $Research works", () => {
		expect(skillMatches("Research", SKILLS, false).map((s) => s.name)).toEqual(["research"]);
		expect(skillMatches("re", SKILLS, false).map((s) => s.name)).toEqual(["research"]);
	});

	it("is a prefix, never a subsequence — $rsrch finds nothing", () => {
		expect(skillMatches("rsrch", SKILLS, false)).toEqual([]);
	});

	it("accepts dots, underscores and dashes in names", () => {
		expect(skillMatches("code-", SKILLS, false).map((s) => s.name)).toEqual(["code-review"]);
		expect(skillMatches("under_", SKILLS, false).map((s) => s.name)).toEqual(["under_score"]);
		expect(skillMatches("dash-", SKILLS, false).map((s) => s.name)).toEqual(["dash-name"]);
	});

	it("keeps hidden skills in the list — the picker is how they are typed", () => {
		expect(skillMatches("path", SKILLS, false).map((s) => s.name)).toEqual(["pathfinder"]);
		expect(skillMatches("path", SKILLS, false)[0]?.hidden).toBe(true);
	});

	it("a LOWERCASE query with uppercase elsewhere still counts as evidence", () => {
		// `$Re` has one lowercase letter; the prefix rule is case-insensitive.
		expect(skillMatches("Re", SKILLS, false).map((s) => s.name)).toEqual(["research"]);
	});
});

describe("the span replacement never reformats the prose around it", () => {
	const token = skillToken("fix $re\nthe bug", 7);
	it("stops the caret after the name when a newline terminates the token", () => {
		expect(token).not.toBeNull();
		if (token === null) return;
		const inserted = skillSpanReplacement("fix $re\nthe bug", token, "research");
		expect(inserted.text).toBe("fix $research\nthe bug");
		expect(inserted.caret).toBe(13);
	});

	it("keeps whitespace the user already typed as the separator", () => {
		const spaced = skillToken("fix $re the bug", 7);
		expect(spaced).not.toBeNull();
		if (spaced === null) return;
		const inserted = skillSpanReplacement("fix $re the bug", spaced, "research");
		expect(inserted.text).toBe("fix $research the bug");
		expect(inserted.caret).toBe(14);
	});
});
