// @vitest-environment happy-dom
//
// The settle's coordinate space, and the frame hand-off that keeps a touch's
// rows stationary (batch 3, project lo-mobile-ux).
//
// WHY THESE PROPERTIES. The row jitter the operator reported was the settle
// re-measuring its OWN in-flight transform: a commit inside the 180ms window
// read the mid-flight offset (a rect includes the CSS transform the settle
// wrote) and wrote it back as a new settle, mirroring the card across its
// slot — sign-alternating per commit and, at the daemon's measured ~24-30Hz
// frame cadence, amplifying. And a frame applied between pointerdown and
// pointerup slid the row out from under the tap, so the synthesised click
// resolved to the container and nothing opened. Both are asserted against the
// REAL screen.
//
// happy-dom lays nothing out, so the geometry is the test's: `offsetTop` is
// the card's LAYOUT position, a stubbed `getBoundingClientRect` is the PAINTED
// box — with the transform the test says is in flight ADDED, which is what
// makes the first case fail against a rect-based measurement — and a Proxy
// over the card's `style` is the write log (one settle writes `transform`
// twice: the invert, then the release to "").
import {
	act,
	cleanup,
	fireEvent,
	render,
	waitFor,
} from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { SessionListScreen } from "./screens/session-list";
import type { SessionSummary } from "./types";

let sessionList: SessionSummary[] = [];
vi.mock("./store", async (importOriginal) => {
	const actual = await importOriginal<typeof import("./store")>();
	return {
		...actual,
		useSessions: () => ({ sessions: sessionList, connected: true }),
		retainSessionListStream: () => () => {},
	};
});
vi.mock("./api", () => ({
	getDirectories: vi.fn(async () => ({ home: "", recent: [] })),
}));

function summary(over: Partial<SessionSummary>): SessionSummary {
	return {
		session_id: "s",
		section: "active",
		conversation_name: "Session",
		cwd: "~",
		model_label: "sonnet-4.5",
		streaming: false,
		needs_attention: false,
		unseen: false,
		pending_kind: "",
		subagents_running: 0,
		todos_open: 0,
		mtime: 0,
		...over,
	};
}

function cardByName(name: string): HTMLElement {
	const card = [...document.querySelectorAll("main button")].find((el) =>
		(el.textContent ?? "").includes(name),
	);
	if (!card) throw new Error(`no card named ${name}`);
	return card as HTMLElement;
}

/** One card's geometry and write log.
 *
 *  `setLayout` is what the code reads through `offsetTop`; `setPainted` is the
 *  transform the mocked rect reports as in flight (a mid-settle card paints at
 *  layout + transform — the value a rect-based measurement mistakes for
 *  movement). The Proxy records every `transform` the settle writes, including
 *  the transient invert that is released to "" within the same synchronous
 *  block — the write a plain style read-back cannot see. */
function instrument(name: string) {
	const card = cardByName(name);
	const writes: Array<[string, string]> = [];
	let layout = 0;
	let painted = 0;
	Object.defineProperty(card, "offsetTop", {
		get: () => layout,
		configurable: true,
	});
	const raw = card.style;
	const style = new Proxy(raw, {
		set(target, property, value) {
			if (property === "transform") {
				writes.push([String(property), String(value)]);
			}
			return Reflect.set(target, property, value);
		},
	});
	Object.defineProperty(card, "style", { get: () => style, configurable: true });
	card.getBoundingClientRect = () =>
		({
			top: layout + painted,
			bottom: layout + painted + 50,
			left: 0,
			right: 100,
			width: 100,
			height: 50,
			x: 0,
			y: layout + painted,
			toJSON: () => ({}),
		}) as DOMRect;
	return {
		writes,
		rawStyle: raw,
		setLayout: (value: number) => {
			layout = value;
		},
		setPainted: (value: number) => {
			painted = value;
		},
	};
}

afterEach(() => {
	cleanup();
	vi.unstubAllGlobals();
});

describe("the settle's own window", () => {
	it("does not re-measure its in-flight transform on a commit inside the window", () => {
		sessionList = [
			summary({ session_id: "a", conversation_name: "Alpha" }),
			summary({ session_id: "b", conversation_name: "Beta" }),
		];
		const view = render(<SessionListScreen />);
		/* Installed after mount, so the mount commit recorded layout 0 for the
		   card; the reorder below is the first move this case is about. */
		const alpha = instrument("Alpha");

		/* A real reorder: Alpha moves down one slot. */
		alpha.setLayout(160);
		act(() => {
			sessionList = [
				summary({ session_id: "b", conversation_name: "Beta" }),
				summary({ session_id: "a", conversation_name: "Alpha" }),
			];
			view.rerender(<SessionListScreen />);
		});
		expect(alpha.writes).toEqual([
			["transform", "translateY(-160px)"],
			["transform", ""],
		]);

		/* The commit INSIDE the 180ms window — the mirror. The card now paints
		   at 160 - 30 (layout plus the in-flight transform); a rect-based
		   re-measurement reads that as movement and writes translateY(30px),
		   mirroring the card across its slot. The settle must write NOTHING:
		   the layout did not move. */
		alpha.setPainted(-30);
		act(() => {
			sessionList = sessionList.map((row) => ({ ...row, mtime: row.mtime + 1 }));
			view.rerender(<SessionListScreen />);
		});
		expect(alpha.writes).toEqual([
			["transform", "translateY(-160px)"],
			["transform", ""],
		]);
	});

	it("holds the order while a pointer is down and applies it on the frame after the release", async () => {
		sessionList = [
			summary({ session_id: "a", conversation_name: "Alpha" }),
			summary({ session_id: "b", conversation_name: "Beta" }),
		];
		const view = render(<SessionListScreen />);
		const order = () =>
			[...document.querySelectorAll("main button")].map((el) =>
				(el.textContent ?? "").includes("Alpha") ? "Alpha" : "Beta",
			);
		expect(order()).toEqual(["Alpha", "Beta"]);

		/* A finger lands on the list. */
		fireEvent.pointerDown(cardByName("Alpha"));

		/* The daemon reorders mid-touch: the frame is buffered, not applied —
		   the row under the finger must not move (this is the "hard to tap"
		   report; measured on the rig, the row used to slide away and the tap
		   resolved to the container). */
		act(() => {
			sessionList = [
				summary({ session_id: "b", conversation_name: "Beta" }),
				summary({ session_id: "a", conversation_name: "Alpha" }),
			];
			view.rerender(<SessionListScreen />);
		});
		expect(order()).toEqual(["Alpha", "Beta"]);

		/* The finger lifts: the buffered frame applies on the next animation
		   frame (after any click the lift produces, never before it). */
		fireEvent.pointerUp(cardByName("Alpha"));
		await waitFor(() => expect(order()).toEqual(["Beta", "Alpha"]));
	});

	it("continues from a settling card's current paint when its layout moves again", () => {
		/* Round-1 review M1: the continuation branch (`prev.top +
		   translateYOf(el) - top`) is what lets a card that is STILL gliding
		   when its layout moves again carry on from where it paints; without a
		   test here a regression would ship silently. `translateYOf` reads the
		   computed transform, so the test stages a mid-flight pose in the
		   element's inline style (what happy-dom's getComputedStyle reports)
		   and stubs the matrix reader to parse it, exactly as a browser would. */
		class FakeMatrixReadOnly {
			f: number;
			constructor(source: string) {
				const translate = /translateY\((-?[\d.]+)px\)/.exec(source);
				const matrix = /matrix\(([^)]+)\)/.exec(source);
				this.f = translate
					? parseFloat(translate[1])
					: matrix
						? parseFloat(matrix[1].split(",")[5] ?? "0")
						: 0;
			}
		}
		vi.stubGlobal("DOMMatrixReadOnly", FakeMatrixReadOnly);

		sessionList = [
			summary({ session_id: "a", conversation_name: "Alpha" }),
			summary({ session_id: "b", conversation_name: "Beta" }),
		];
		const view = render(<SessionListScreen />);
		const alpha = instrument("Alpha");

		/* First move: down one slot, inverted at -160. */
		alpha.setLayout(160);
		act(() => {
			sessionList = [
				summary({ session_id: "b", conversation_name: "Beta" }),
				summary({ session_id: "a", conversation_name: "Alpha" }),
			];
			view.rerender(<SessionListScreen />);
		});
		expect(alpha.writes).toEqual([
			["transform", "translateY(-160px)"],
			["transform", ""],
		]);

		/* Mid-flight: the card paints 30px below its layout slot (the -160
		   inversion has interpolated to -30; a browser reports exactly that
		   through the computed matrix). */
		alpha.rawStyle.transform = "translateY(-30px)";
		alpha.setPainted(-30);

		/* The layout moves AGAIN while that transition is live: the slot moves
		   from 160 to 200. The next settle must start from the paint
		   (160 + -30 = 130) and invert at 130 - 200 = -70. A stale-coordinate
		   regression would write the -40 of `prev.top - top` — the jump a
		   mid-flight re-move would ship. */
		alpha.setLayout(200);
		act(() => {
			sessionList = sessionList.map((row) => ({ ...row, mtime: row.mtime + 1 }));
			view.rerender(<SessionListScreen />);
		});
		expect(alpha.writes.slice(2)).toEqual([
			["transform", "translateY(-70px)"],
			["transform", ""],
		]);
	});

	it("coalesces a burst into one application and paints only the latest frame", async () => {
		/* Round-1 review M2: several frames can arrive between two painted
		   frames (the daemon pushes at ~24-30/s); the hand-off applies at most
		   ONE per animation frame, and a frame still waiting when the next
		   arrives must never paint at all. */
		sessionList = [
			summary({ session_id: "a", conversation_name: "Alpha" }),
			summary({ session_id: "b", conversation_name: "Beta" }),
		];
		const view = render(<SessionListScreen />);
		const order = () =>
			[...document.querySelectorAll("main button")].map((el) =>
				(el.textContent ?? "").includes("Alpha")
					? "Alpha"
					: (el.textContent ?? "").includes("Beta")
						? "Beta"
						: "Gamma",
			);
		const painted = () => document.body.textContent ?? "";

		/* Frame A adds a session; frame B replaces A before the scheduled
		   animation frame can run. Between the two, the DOM must still show
		   the OLD rows — nothing paints mid-burst. */
		act(() => {
			sessionList = [
				summary({ session_id: "a", conversation_name: "Alpha" }),
				summary({ session_id: "b", conversation_name: "Beta" }),
				summary({ session_id: "g", conversation_name: "Gamma" }),
			];
			view.rerender(<SessionListScreen />);
		});
		expect(painted()).not.toContain("Gamma");
		act(() => {
			sessionList = [
				summary({ session_id: "b", conversation_name: "Beta" }),
				summary({ session_id: "a", conversation_name: "Alpha" }),
			];
			view.rerender(<SessionListScreen />);
		});
		expect(painted()).not.toContain("Gamma");
		expect(order()).toEqual(["Alpha", "Beta"]);

		/* ONE application — the latest frame's — on the next animation frame;
		   Gamma (frame A) never painted, not even momentarily. */
		await waitFor(() => expect(order()).toEqual(["Beta", "Alpha"]));
		expect(painted()).not.toContain("Gamma");
	});
});
