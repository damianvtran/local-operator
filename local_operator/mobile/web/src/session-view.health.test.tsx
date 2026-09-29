// @vitest-environment happy-dom
//
// Session-state honesty on the phone (mobile UX batch 2): U7 (ended/degraded
// are consumed), U11 (the phone's own dropped link says so) and U2 (a refused
// pin says why). Asserted against the REAL SessionScreen, because that is
// where the ladder renders and where the pin refusal lands.
//
// WHY A LADDER AND ONE STRIP. `ended` (the process is gone; resume lives in
// the strip), `degraded` (the relay's dial is down) and `connected === false`
// (the phone's own link) are three different facts, and stacking three strips
// on a 320-wide phone would spend the vertical budget the batch-1 layout work
// bought back. The later rungs are only worth stating while the earlier are
// untrue, so at most one shows — pinned here, since an accidental second strip
// is exactly the kind of regression a single-state assertion misses.
//
// WHAT THIS LAYER CANNOT PROVE: happy-dom does no layout, so "the strip takes
// layout space rather than covering a control" (the reason it is `border-b`
// in-flow rather than an overlay) is a rendered-frame question, proved by the
// batch's headless captures.
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SessionScreen } from "./screens/session-view";
import type { SessionProjection } from "./types";

const mocks = vi.hoisted(() => ({
	setSessionPin: vi.fn(async () => ({ ok: true })),
	resumeSession: vi.fn(async () => ({ ok: true, pid: 42, session_id: "resumed-1" })),
	navigate: vi.fn(),
}));

vi.mock("./router", () => ({ navigate: mocks.navigate }));

vi.mock("./api", () => ({
	getHistory: vi.fn(async () => ({ entries: [], has_more: false })),
	getSubagentHistory: vi.fn(async () => ({ entries: [], has_more: false })),
	getSubagentDetail: vi.fn(async () => null),
	imageUrl: vi.fn(() => ""),
	getCommands: vi.fn(async () => ({ commands: [] })),
	getModels: vi.fn(async () => ({ models: [] })),
	sendCommand: vi.fn(async () => ({ ok: true, detail: "" })),
	markSessionSeen: vi.fn(async () => ({ ok: true })),
	setSessionPin: mocks.setSessionPin,
	resumeSession: mocks.resumeSession,
}));

let slot: { projection: SessionProjection | null; connected: boolean } = {
	projection: null,
	connected: true,
};
vi.mock("./store", async (importOriginal) => {
	const actual = await importOriginal<typeof import("./store")>();
	return {
		...actual,
		useProjection: vi.fn(() => slot),
		retainProjectionStream: vi.fn(() => () => {}),
		retainSessionListStream: vi.fn(() => () => {}),
		useDraft: vi.fn(() => ["", () => {}]),
	};
});

/** The projection shape the REAL SessionScreen tree reads.
    Copied from the sibling render tests rather than trimmed: the Composer
    reads `effort_ladder.length` and friends, and a partial fixture fails
    inside a component rather than in the assertion under test. */
function projection(over: Partial<SessionProjection> = {}): SessionProjection {
	return {
		session_id: "s1",
		pid: 1,
		kind: "tui",
		conversation_name: "health",
		cwd: "",
		model_label: "",
		model_selector: "",
		effort: "",
		effort_ladder: [],
		streaming: false,
		activity: "",
		activity_started_s: 0,
		stop_reason: "",
		queued_count: 0,
		ended: false,
		degraded: false,
		transcript: [],
		todos: [],
		subagents: [],
		pending: null,
		pending_count: 0,
		usage: {},
		version: 1,
		...over,
	} satisfies SessionProjection;
}

beforeEach(() => {
	slot = { projection: projection(), connected: true };
	mocks.setSessionPin.mockClear();
	mocks.setSessionPin.mockImplementation(async () => ({ ok: true }));
	mocks.resumeSession.mockClear();
	mocks.resumeSession.mockImplementation(async () => ({ ok: true, pid: 42, session_id: "resumed-1" }));
	mocks.navigate.mockClear();
});
afterEach(cleanup);

describe("an ended session (U7)", () => {
	it("says so and offers the documented resume, which reopens the conversation", async () => {
		slot = { projection: projection({ ended: true }), connected: true };
		render(<SessionScreen sessionId="s1" />);

		// The reader is told what happened, in the strip's own words.
		expect(screen.getByText("this session has ended — its history is kept")).toBeTruthy();

		fireEvent.click(screen.getByRole("button", { name: "resume" }));
		await waitFor(() => expect(mocks.resumeSession).toHaveBeenCalledWith("s1"));
		// The phone follows the new session the daemon spawned.
		await waitFor(() => expect(mocks.navigate).toHaveBeenCalledWith("/s/resumed-1"));
	});

	it("renders the daemon's sentence when the resume is refused, not a dead control", async () => {
		mocks.resumeSession.mockImplementation(async () => {
			throw new Error("no runtime to resume into");
		});
		slot = { projection: projection({ ended: true }), connected: true };
		render(<SessionScreen sessionId="s1" />);

		fireEvent.click(screen.getByRole("button", { name: "resume" }));
		await waitFor(() => expect(screen.getByRole("alert").textContent).toContain("no runtime to resume into"));
		// The affordance survives the refusal — retrying is the reader's call.
		expect(screen.getByRole("button", { name: "resume" })).toBeTruthy();
		expect(mocks.navigate).not.toHaveBeenCalled();
	});
});

describe("a degraded session (U7)", () => {
	it("states the dial is down and does not claim the conversation ended", () => {
		slot = { projection: projection({ degraded: true }), connected: true };
		render(<SessionScreen sessionId="s1" />);

		expect(screen.getByText("not answering — showing its last synced view")).toBeTruthy();
		// `ended` and `degraded` are different facts; the degraded strip must
		// not drag the resume affordance (and its restart offer) in with it.
		expect(screen.queryByRole("button", { name: "resume" })).toBeNull();
		expect(screen.queryByText(/this session has ended/)).toBeNull();
	});
});

describe("the health ladder renders exactly one rung", () => {
	it("prefers ended over degraded and over the phone's own link", () => {
		slot = {
			projection: projection({ ended: true, degraded: true }),
			connected: false,
		};
		render(<SessionScreen sessionId="s1" />);

		expect(screen.getByText("this session has ended — its history is kept")).toBeTruthy();
		expect(screen.queryByText("not answering — showing its last synced view")).toBeNull();
		expect(screen.queryByText("reconnecting — showing the last synced view")).toBeNull();
	});

	it("prefers degraded over the phone's own link", () => {
		slot = { projection: projection({ degraded: true }), connected: false };
		render(<SessionScreen sessionId="s1" />);

		expect(screen.getByText("not answering — showing its last synced view")).toBeTruthy();
		expect(screen.queryByText("reconnecting — showing the last synced view")).toBeNull();
	});
});

describe("the phone's own dropped link (U11)", () => {
	it("says reconnecting once a projection is on screen, and nothing while it has none", () => {
		slot = { projection: projection(), connected: false };
		const first = render(<SessionScreen sessionId="s1" />);
		expect(screen.getByText("reconnecting — showing the last synced view")).toBeTruthy();
		first.unmount();

		// With no data yet the view already says `connecting to session…`;
		// a second line would be noise about noise.
		slot = { projection: null, connected: false };
		render(<SessionScreen sessionId="s1" />);
		expect(screen.queryByText("reconnecting — showing the last synced view")).toBeNull();
	});
});

describe("a refused pin (U2)", () => {
	it("shows the daemon's reason in the strip instead of a silent flip-back", async () => {
		mocks.setSessionPin.mockImplementation(async () => {
			throw new Error("no saved messages yet — pin it after you send one");
		});
		render(<SessionScreen sessionId="s1" />);

		const star = await screen.findByRole("button", { name: "pin this session" });
		fireEvent.click(star);

		await waitFor(() =>
			expect(screen.getByRole("alert").textContent).toContain(
				"Could not save the pin: no saved messages yet — pin it after you send one",
			),
		);
		// The optimistic mark went back with the refusal — the button offers the
		// press again rather than showing a ★ the daemon never accepted.
		expect(screen.getByRole("button", { name: "pin this session" })).toBeTruthy();
	});
});
