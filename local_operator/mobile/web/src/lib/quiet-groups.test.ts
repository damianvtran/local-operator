// The shared quiet-group parity fixture, replayed through the relay-web
// derivation (quiet-turn design §5 + §8 S4).
//
// WHY THIS FILE EXISTS. A quiet group is client-derived: there is no wire
// kind and no capability flag, so the risk this suite closes is silent drift
// between surfaces — the desktop's `quietGroupsOf`/`quietGroupOfSegment`
// (local-operator-ui, `scripts/turn-collapse-model.test.mjs`) and this port
// are two implementations of one definition. The fixture is the contract:
// both suites read it, so a change to either side fails in its own tree
// rather than diverging with both green. The copy here is byte-identical and
// hash-pinned below, not a fork.
//
// TWO CASES ARE NOT REPLAYED, and they are named rather than filtered
// silently: the fixture's monitor-only and job-only cases describe `custom`
// rows, and this wire has no equivalent — the relay fold renders a monitor
// prompt and a delivered job result as GENERIC notices, indistinguishably. On
// this surface those rows split and stay visible (pinned below); the fixture
// keeps the cases for the surfaces that can express them.
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
	quietGroupLabel,
	quietGroupOfSegment,
	quietGroupsOf,
	type QuietGroupRecord,
} from "./quiet-groups";

/** One fixture row, as the shared file declares it (“Rows are records in the
 * UI's `TranscriptRecord` vocabulary, carrying only the fields the group
 * definition reads”). The fields the relay mapper does not consume are still
 * typed so a fixture addition fails loudly here rather than being ignored. */
interface FixtureRow {
	kind: "peer" | "wake" | "tool" | "user" | "assistant" | "notice" | "custom" | "compaction";
	id: string;
	ts?: number;
	text?: string;
	body?: string;
	sender?: { conversationName?: string };
	customType?: string;
	level?: string;
	toolName?: string;
	phase?: string;
	isError?: boolean;
	complete?: boolean;
	before?: number;
}

interface FixtureCase {
	name: string;
	span?: [number, number];
	spanHeadLoaded?: boolean;
	open?: boolean;
	rows: FixtureRow[];
	expected: unknown;
}

const FIXTURE_PATH = join(__dirname, "quiet-groups.parity.json");
const fixture: { cases: FixtureCase[] } = JSON.parse(readFileSync(FIXTURE_PATH, "utf8")) as {
	cases: FixtureCase[];
};

/* The fixture cases this surface cannot express: both describe `custom` rows
   (monitor prompts, job results) that the relay fold renders as generic
   notices. Named in full so the replay can assert exactly which cases are
   skipped — a future fixture addition must not fall into a silent gap. */
const NOT_EXPRESSIBLE_CASES = [
	"a monitor-only run states the monitor family",
	"a job-only run states the job family",
];

/**
 * The relay-web MAPPING of one fixture row (the fixture's own instruction: “A
 * client with a different record type maps these onto its own shape”).
 *
 * Deviations from the row, and why:
 * - `sender.conversationName` → the wire's `conversation_name` (the relay
 *   spells its `PeerSender` fields snake-case);
 * - a tool's `isError` → `tool_state: "failed"` — the relay has no isError
 *   boolean; the fold's failure verdict IS the state;
 * - a notice's `complete` is DROPPED, and every notice lands on an empty
 *   `details` — this wire's notices carry no completeness marker and no
 *   custom type, which is exactly why a non-wake notice splits here (see the
 *   module doc). The fixture's completion-receipt case still passes: a
 *   completion receipt splits a group on every surface.
 */
function recordOf(row: FixtureRow): QuietGroupRecord {
	/* The fields every kind carries but only some kinds read: `tool_name` /
	 * `tool_state` are never consulted on a non-tool row, so they take the
	 * wire's own no-outcome default rather than anything meaningful. */
	const base = {
		id: row.id,
		ts: row.ts,
		text: row.text ?? "",
		details: {},
		tool_name: "",
		tool_state: "interrupted" as const,
	};
	switch (row.kind) {
		case "peer": {
			const sender = row.sender?.conversationName;
			return {
				...base,
				kind: "peer_message",
				text: row.body ?? "peer",
				details: sender === undefined ? {} : { sender: { conversation_name: sender } },
			};
		}
		case "wake":
			return {
				...base,
				kind: "notice",
				text: row.text ?? "wake",
				details: { notice_kind: "wake" },
			};
		case "tool":
			return {
				...base,
				kind: "tool",
				tool_name: row.toolName ?? "",
				tool_state: row.isError ? "failed" : "done",
			};
		case "user":
			return { ...base, kind: "user", text: row.text ?? "hi" };
		case "assistant":
			return { ...base, kind: "assistant", text: row.text ?? "text" };
		case "notice":
			return { ...base, kind: "notice", text: row.text ?? "notice" };
		case "compaction":
			return { ...base, kind: "compaction", text: row.text ?? "Context compacted" };
		case "custom":
			// Not a shape this wire can produce; the replay skips these cases
			// before mapping, so reaching here means the skip list drifted.
			throw new Error(`no relay mapping for a custom row (${row.id})`);
	}
}

/** A peer receipt in the relay's own shape. */
function peer(id: string): QuietGroupRecord {
	return {
		kind: "peer_message",
		id,
		text: id,
		details: {},
		tool_name: "",
		tool_state: "interrupted",
	};
}

describe("the quiet-group derivation against the shared parity fixture", () => {
	it("keeps the copied fixture byte-identical to local-operator-ui's", () => {
		/* The copy IS the contract: this suite and local-operator-ui's must
		 * replay the same bytes, so a local edit of the fixture (rather than a
		 * coordinated cross-client re-pin) fails here. The digest pins the copy
		 * taken from local-operator-ui PR #945 (`feat/quiet-turn-ui`),
		 * re-verified byte-identical at the branch head `f9b041f8`. */
		const text = readFileSync(FIXTURE_PATH, "utf8");
		const digest = createHash("sha256").update(text).digest("hex");
		expect(digest).toBe("3e5dca2158ef4103ff0e905e979daae2cb4dd61f4e379fecfb03cb9a53631dac");
	});

	/* ONE TEST PER CASE, named with the fixture's own name: a case that fails
	 * carries its scenario in the report instead of a bare index. */
	for (const entry of fixture.cases) {
		if (NOT_EXPRESSIBLE_CASES.includes(entry.name)) continue;
		it(entry.name, () => {
			const records = entry.rows.map(recordOf);
			const actual = entry.span
				? quietGroupOfSegment(
						records,
						{ from: entry.span[0], to: entry.span[1] },
						{
							spanHeadLoaded: entry.spanHeadLoaded ?? true,
							open: entry.open,
						},
					)
				: quietGroupsOf(records);
			expect(actual).toEqual(entry.expected);
		});
	}

	it("skips exactly the two cases this surface cannot express", () => {
		/* The skip list must name EXACTLY the unexpressible cases — no more
		 * (nothing here proves another case skippable), no fewer (a skipped
		 * case proves nothing). Both halves read the same fixture. */
		const unexpressible = fixture.cases
			.filter((entry) => entry.rows.some((row) => row.kind === "custom"))
			.map((entry) => entry.name);
		expect(unexpressible).toEqual(NOT_EXPRESSIBLE_CASES);
		expect(
			fixture.cases.filter((entry) => !NOT_EXPRESSIBLE_CASES.includes(entry.name)).length +
				NOT_EXPRESSIBLE_CASES.length,
		).toBe(fixture.cases.length);
	});
});

describe("the relay-web vocabulary's own boundaries", () => {
	/** A quiet pair before and after ONE boundary row: two groups of two. */
	const around = (between: QuietGroupRecord): QuietGroupRecord[] => [
		peer("p1"),
		peer("p2"),
		between,
		peer("p4"),
		peer("p5"),
	];

	it("splits on a generic notice: a monitor prompt or job result stays visible", () => {
		/* Those rows reach the relay as plain notices with empty details — no
		 * field distinguishes family or completeness — so they cannot be
		 * folded, and the safe direction is that they split. (The fixture's
		 * two custom cases are the surfaces that CAN classify them; here they
		 * are exactly this shape.) */
		const groups = quietGroupsOf(around({ ...peer("n3"), kind: "notice" }));
		expect(groups.map((group) => group.key)).toEqual(["qg:p1", "qg:p4"]);
	});

	it("splits on a hub relay card (parent or subagent), which stays readable", () => {
		/* The relay renders hub traffic as `parent_message` / `subagent_message`
		 * cards. The shared fixture exercises no hub relay, and this surface
		 * cannot tell one from an ordinary card, so the mapping keeps them
		 * visible rather than folding an unattributable receipt. */
		expect(quietGroupsOf(around({ ...peer("h3"), kind: "parent_message" }))).toHaveLength(2);
		expect(quietGroupsOf(around({ ...peer("h4"), kind: "subagent_message" }))).toHaveLength(2);
	});

	it("sits on an empty assistant row and still splits on visible prose", () => {
		const streaming = quietGroupsOf(around({ ...peer("a3"), kind: "assistant", text: "" }));
		expect(streaming).toHaveLength(1);
		expect(streaming[0].count).toBe(4);
		expect(
			quietGroupsOf(around({ ...peer("a3"), kind: "assistant", text: "on it" })),
		).toHaveLength(2);
	});
});

describe("group facts", () => {
	it("freezes a closed group's facts and lets only the open tail grow", () => {
		const before = quietGroupsOf([peer("p1"), peer("p2"), user("u1")])[0];
		expect(before.open).toBe(false);
		const after = quietGroupsOf([peer("p1"), peer("p2"), user("u1"), peer("p3"), peer("p4")]);
		expect(after).toHaveLength(2);
		expect(after[0]).toEqual(before);
		expect(after[1].key).toBe("qg:p3");
		expect(after[1].open).toBe(true);
		const grown = quietGroupsOf([
			peer("p1"),
			peer("p2"),
			user("u1"),
			peer("p3"),
			peer("p4"),
			peer("p5"),
		]);
		expect(grown[1].key).toBe("qg:p3");
		expect(grown[1].count).toBe(3);
	});

	it("carries no times when the records carry none — this wire's state", () => {
		/* No per-entry timestamp travels on the relay wire, so firstTs/lastTs
		 * are null and no span clause can be stated (the same clauses a
		 * head-cut span nulls). The parity fixture above pins the times path
		 * for surfaces whose records DO carry them. */
		const group = quietGroupsOf([peer("p1"), peer("p2")])[0];
		expect(group.firstTs).toBeNull();
		expect(group.lastTs).toBeNull();
	});

	it("states the family words exactly once, for every family", () => {
		expect(quietGroupLabel("peer")).toBe("Peer messages");
		expect(quietGroupLabel("wake")).toBe("Wake messages");
		expect(quietGroupLabel("monitor")).toBe("Monitor messages");
		expect(quietGroupLabel("job")).toBe("Job results");
		expect(quietGroupLabel("mixed")).toBe("Messages");
	});
});

function user(id: string): QuietGroupRecord {
	return { ...peer(id), kind: "user", text: "steer" };
}
