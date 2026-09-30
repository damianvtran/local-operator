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
	const style = new Proxy(card.style, {
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
		setLayout: (value: number) => {
			layout = value;
		},
		setPainted: (value: number) => {
			painted = value;
		},
	};
}

afterEach(cleanup);

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
});
