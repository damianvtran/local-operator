// @vitest-environment happy-dom
//
// The supplement model's pinned half (`lib/supplement.ts`): the frozen reader
// rule, the display copy, the frame-message acceptance rule, the theme reader
// and the mount budget. The renderer's own half is exercised in
// `supplement-block.test.tsx` / `supplement-frame.test.tsx`, and the sandbox
// itself is proven in a real browser on the PR (a happy-dom iframe loads no
// document, so no message wire exists here to test end to end — these tests
// pin the RULES the wire is judged by).
import { afterEach, describe, expect, it, vi } from "vitest";
import {
	COPY_SHOW_FEWER,
	HEIGHT_HINT_MAX,
	HEIGHT_HINT_MIN,
	MORE_MAX,
	acceptFrameMessage,
	beatFromWire,
	beatLive,
	clampFrameHeight,
	clampedHeightHint,
	countClause,
	displayText,
	fileRowParts,
	fileSizeText,
	filesSummaryLine,
	filesSummaryOf,
	frameBudgetPx,
	mintNonce,
	moreText,
	planComponentMounts,
	readSupplementTheme,
	showAllText,
	stateRank,
	supplementDisposition,
	supplementFileClass,
	supplementFrameUrl,
	supplementsFromWire,
	type SupplementBeat,
} from "./supplement";

afterEach(() => {
	vi.restoreAllMocks();
});

/** One beat with every field the readers touch, overridable per case. */
function beat(over: Partial<SupplementBeat> = {}): SupplementBeat {
	return {
		anchor: "a".repeat(32),
		job: "3f9c1a7e5b20",
		version: 1,
		state: "done",
		stage: "",
		elapsedS: 0,
		files: [],
		filesMore: 0,
		more: [],
		components: [],
		error: "",
		dismissed: false,
		...over,
	};
}

describe("the frozen reader rule", () => {
	it("prepares while the job is live and falls to cancelled · Retry when it is not (the stale-row rule)", () => {
		const queued = beat({ state: "queued" });
		expect(supplementDisposition(queued, true)).toBe("preparing");
		expect(supplementDisposition(queued, false)).toBe("cancelled_retry");
	});

	it("carries the files prefix on every arm that has files (they never wait for the generator)", () => {
		const file = {
			path: "reports/latency.md",
			name: "latency.md",
			kind: "markdown",
			sizeBytes: 4120,
			mtime: 0,
			why: "",
		};
		expect(supplementDisposition(beat({ state: "decided", files: [file] }), true)).toBe(
			"files_preparing",
		);
		expect(supplementDisposition(beat({ state: "failed", files: [file] }), false)).toBe(
			"files_failed_retry",
		);
		expect(supplementDisposition(beat({ state: "cancelled", files: [file] }), false)).toBe(
			"files_cancelled_retry",
		);
	});

	it("renders nothing for superseded, dismissed and skipped rows", () => {
		expect(supplementDisposition(beat({ error: "superseded" }), true)).toBe("nothing");
		expect(supplementDisposition(beat({ dismissed: true }), true)).toBe("nothing");
		expect(supplementDisposition(beat({ state: "skipped" }), true)).toBe("nothing");
	});

	it("renders the block for a done row with content and NOTHING for a done row with none", () => {
		expect(supplementDisposition(beat(), false)).toBe("nothing");
		const component = {
			attachment: "f".repeat(32),
			title: "t",
			source: "s",
			mime: "text/html",
			heightHint: 200,
		};
		expect(supplementDisposition(beat({ components: [component] }), false)).toBe("block");
	});

	it("ranks states so a lazy read cannot replay an earlier one", () => {
		expect(stateRank("decided")).toBeLessThan(stateRank("queued"));
		expect(stateRank("queued")).toBeLessThan(stateRank("running"));
		expect(stateRank("running")).toBeLessThan(stateRank("cancelling"));
		expect(stateRank("cancelling")).toBeLessThan(stateRank("done"));
		expect(stateRank("done")).toBe(stateRank("failed"));
		expect(stateRank("unknown")).toBe(0);
	});

	it("reads job liveness from the live-only words and the projection's process identity", () => {
		// `running`/`cancelling` are never journaled, so they are live whatever
		// the projection carries.
		expect(beatLive(beat({ state: "running" }), 0)).toBe(true);
		expect(beatLive(beat({ state: "cancelling" }), 0)).toBe(true);
		// A registrant-stamped projection (pid > 0) is live; the disk rebuild
		// stamps 0 — the stale-row fixture's `state=queued, no live job`.
		expect(beatLive(beat({ state: "queued" }), 4242)).toBe(true);
		expect(beatLive(beat({ state: "queued" }), 0)).toBe(false);
	});
});

describe("reading the wire", () => {
	it("keeps the newest version per anchor and drops malformed rows", () => {
		const map = supplementsFromWire({
			["a".repeat(32)]: { anchor: "a".repeat(32), state: "decided", version: 1, job: "j" },
			["b".repeat(32)]: { anchor: "b".repeat(32), state: "nonsense", version: 1, job: "j" },
			["c".repeat(32)]: { anchor: "WRONG", state: "done", version: 1, job: "j" },
		});
		expect(Object.keys(map)).toEqual(["a".repeat(32)]);
	});

	it("reads files/components defensively and bounds the disclosure tail", () => {
		const parsed = beatFromWire({
			anchor: "a",
			state: "done",
			version: 2,
			files: [{ path: "p.md", name: "p.md", size_bytes: 10, mtime: 1.5, kind: "markdown" }, "junk"],
			components: [
				{ attachment: "f".repeat(32), title: "t", source: "s", mime: "text/html", height_hint: 9999 },
				{ attachment: "not-a-digest" },
			],
			files_more: 25,
			more: Array.from({ length: 30 }, (_, i) => `p${i}`),
			error: "x".repeat(999),
		});
		expect(parsed).not.toBeNull();
		expect(parsed?.files).toHaveLength(1);
		expect(parsed?.components).toHaveLength(1);
		expect(parsed?.more).toHaveLength(MORE_MAX);
		expect(parsed?.filesMore).toBe(25);
		expect(parsed?.error.length).toBe(300);
	});

	it("reads an absent map as no beats", () => {
		expect(supplementsFromWire(undefined)).toEqual({});
		expect(supplementsFromWire(null)).toEqual({});
	});
});

describe("the display copy and the paint boundary", () => {
	it("summarizes files-only rows as the contents themselves", () => {
		expect(filesSummaryLine(["report.md"])).toBe("1 file: report.md");
		expect(filesSummaryLine(["report.md", "bench.csv"])).toBe("2 files: report.md, bench.csv");
		expect(moreText(1)).toBe("1 more file");
		expect(moreText(3)).toBe("3 more files");
		expect(showAllText(1)).toBe("Show all 1 graphic");
		expect(showAllText(2)).toBe("Show all 2 graphics");
		// The open-state half of the disclosure toggle (desktop parity, U2).
		expect(COPY_SHOW_FEWER).toBe("Show fewer");
		expect(countClause(2, "file")).toBe("2 files");
	});

	it("formats sizes with one decimal and drops a trailing .0", () => {
		expect(fileSizeText(912)).toBe("912 B");
		expect(fileSizeText(4600)).toBe("4.5 KB");
		expect(fileSizeText(1024 ** 2 * 1.2)).toBe("1.2 MB");
		expect(fileSizeText(1024 ** 3 * 2)).toBe("2 GB");
		expect(fileSizeText(-1)).toBe("");
		expect(fileSizeText("big")).toBe("");
	});

	it("strips control sequences and collapses whitespace, keeping printable text", () => {
		// The strip REMOVES the escape rather than replacing it with a space (the
		// TUI's `display_text`, `ansi.py`'s grammar): `a\x1b[31mred\x1b[0m\nb`
		// reads `ared b` once the newline collapses.
		expect(displayText("a\u001b[31mred\u001b[0m\nb")).toBe("ared b");
		// Format characters are PRINTABLE to this port (agent review round 1,
		// R1): the TUI's `display_text` and the desktop's `displayText` both
		// preserve ZWJ sequences and RTL overrides, and a name must render
		// here exactly as it renders on the sibling surfaces.
		expect(displayText("a\u202eb\u200bc")).toBe("a\u202eb\u200bc");
		expect(displayText(41)).toBe("");
	});

	it("collapses a row whose name is its path", () => {
		const parts = fileRowParts({
			path: "bench.csv",
			name: "bench.csv",
			kind: "csv",
			sizeBytes: 913,
			mtime: 0,
			why: "",
		});
		expect(parts).toEqual({ name: "bench.csv", path: "", size: "913 B" });
	});

	it("builds the one-liner from files plus the known tail", () => {
		const summary = filesSummaryOf(
			beat({
				files: [
					{ path: "a.md", name: "a.md", kind: "markdown", sizeBytes: 1, mtime: 0, why: "" },
				],
				filesMore: 1,
				more: ["deep/b.md"],
			}),
		);
		// Two files total: the one-liner is owed, tail basenames included.
		expect(summary).toBe("2 files: a.md, b.md");
	});
});

describe("the frame message wire", () => {
	it("accepts ready without a nonce and rejects state-moving types without one", () => {
		expect(acceptFrameMessage({ lo: "supplement", v: 1, t: "ready" }, null)).toMatchObject({
			t: "ready",
		});
		expect(acceptFrameMessage({ lo: "supplement", v: 1, t: "resize", h: 100 }, null)).toBeNull();
		expect(
			acceptFrameMessage({ lo: "supplement", v: 1, t: "resize", h: 100, n: "abc" }, "abc"),
		).toMatchObject({ t: "resize", h: 100 });
		// A stale echo is refused: the nonce binds the browsing context.
		expect(
			acceptFrameMessage({ lo: "supplement", v: 1, t: "pong", n: "old" }, "new"),
		).toBeNull();
	});

	it("bounds the error message and the resize height", () => {
		const nonce = "n".repeat(32);
		expect(
			acceptFrameMessage({ lo: "supplement", v: 1, t: "error", msg: "x".repeat(301), n: nonce }, nonce),
		).toBeNull();
		expect(
			acceptFrameMessage({ lo: "supplement", v: 1, t: "error", msg: "boom", n: nonce }, nonce),
		).toMatchObject({ t: "error", msg: "boom" });
		expect(
			acceptFrameMessage({ lo: "supplement", v: 1, t: "resize", h: -1, n: nonce }, nonce),
		).toBeNull();
		expect(
			acceptFrameMessage({ lo: "supplement", v: 1, t: "resize", h: Number.NaN, n: nonce }, nonce),
		).toBeNull();
		// A nonce longer than the contract's cap is refused (never truncated).
		const long = "n".repeat(65);
		expect(acceptFrameMessage({ lo: "supplement", v: 1, t: "pong", n: long }, long)).toBeNull();
	});

	it("rejects the wrong tag, version or type", () => {
		expect(acceptFrameMessage({ lo: "x", v: 1, t: "ready" }, null)).toBeNull();
		expect(acceptFrameMessage({ lo: "supplement", v: 2, t: "ready" }, null)).toBeNull();
		expect(acceptFrameMessage({ lo: "supplement", v: 1, t: "click" }, null)).toBeNull();
	});

	it("mints a hex nonce within the wire's bound", () => {
		const nonce = mintNonce();
		expect(nonce).toMatch(/^[a-f0-9]{32}$/);
		expect(nonce.length).toBeLessThanOrEqual(64);
	});

	it("builds a UTF-8-safe base64 data URL", () => {
		const html = "<b>π — ✓</b>";
		const url = supplementFrameUrl(html);
		expect(url.startsWith("data:text/html;base64,")).toBe(true);
		const b64 = url.slice("data:text/html;base64,".length);
		const bytes = Uint8Array.from(atob(b64), (char) => char.charCodeAt(0));
		expect(new TextDecoder().decode(bytes)).toBe(html);
	});
});

describe("the theme channel", () => {
	it("reads resolved tokens and the color-scheme mode off the root", () => {
		const root = document.createElement("div");
		vi.spyOn(window, "getComputedStyle").mockReturnValue({
			getPropertyValue: (name: string) => (name === "--lo-ink" ? "  #fff  " : ""),
			colorScheme: "dark",
		} as unknown as CSSStyleDeclaration);
		const theme = readSupplementTheme(root);
		expect(theme.mode).toBe("dark");
		expect(theme.vars).toEqual({ "--lo-ink": "#fff" });
	});

	it("falls back to the product's dark ground", () => {
		vi.spyOn(window, "getComputedStyle").mockReturnValue({
			getPropertyValue: () => "",
			colorScheme: "",
		} as unknown as CSSStyleDeclaration);
		expect(readSupplementTheme(document.createElement("div"))).toEqual({
			mode: "dark",
			vars: {},
		});
	});
});

describe("the mount budget", () => {
	it("is min(480px, 40vh)", () => {
		expect(frameBudgetPx(900)).toBe(360);
		expect(frameBudgetPx(2000)).toBe(480);
		expect(frameBudgetPx(0)).toBe(0);
	});

	it("clamps heights into the contract's range", () => {
		expect(clampedHeightHint({ attachment: "f".repeat(32), title: "", source: "", mime: "", heightHint: 5 })).toBe(
			HEIGHT_HINT_MIN,
		);
		expect(clampedHeightHint({ attachment: "f".repeat(32), title: "", source: "", mime: "", heightHint: 5000 })).toBe(
			HEIGHT_HINT_MAX,
		);
		expect(clampFrameHeight(10)).toBe(HEIGHT_HINT_MIN);
		expect(clampFrameHeight(460)).toBe(460);
	});

	it("admits the first frame regardless of budget, then holds at unmeasured predecessors", () => {
		const mk = (n: number, hint: number) => ({
			attachment: String(n).repeat(32).slice(0, 32),
			title: `t${n}`,
			source: "",
			mime: "",
			heightHint: hint,
		});
		const three = [mk(1, 200), mk(2, 200), mk(3, 200)];
		// Nothing measured: only the first mounts (the rest wait on a rendered
		// height AND the budget).
		expect(planComponentMounts(three, 360).visible.map((c) => c.title)).toEqual(["t1"]);
		// First measured at 200: 200 + 200 > 360, so the second stays hidden.
		expect(
			planComponentMounts(three, 360, new Map([[three[0].attachment, 200]])).visible.map(
				(c) => c.title,
			),
		).toEqual(["t1"]);
		// First measured at 150: 150 + 200 <= 360, but the SECOND is unmeasured —
		// the walk holds there.
		expect(
			planComponentMounts(three, 360, new Map([[three[0].attachment, 150]])).visible.map(
				(c) => c.title,
			),
		).toEqual(["t1", "t2"]);
		// A wider budget with both measured admits all three.
		expect(
			planComponentMounts(
				three,
				1000,
				new Map([
					[three[0].attachment, 200],
					[three[1].attachment, 200],
				]),
			).visible.map((c) => c.title),
		).toEqual(["t1", "t2", "t3"]);
	});
});

describe("the file affordance class", () => {
	it("offers previews only for classes the relay route can serve", () => {
		expect(supplementFileClass("reports/latency.md")).toBe("text");
		expect(supplementFileClass("bench.csv")).toBe("text");
		expect(supplementFileClass("pic.PNG")).toBe("image");
		expect(supplementFileClass("doc.pdf")).toBe("pdf");
		expect(supplementFileClass("slides.key")).toBe("other");
		expect(supplementFileClass("noextension")).toBe("other");
	});
});
