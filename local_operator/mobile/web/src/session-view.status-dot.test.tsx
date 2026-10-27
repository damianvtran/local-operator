// @vitest-environment happy-dom
//
// THE HEADER STATUS DOT replaces the "Session ended · history kept" banner
// (operator request). Asserted against the REAL SessionScreen: the dot's state
// and accessible words, the explanation sheet it opens through the shared
// Sheet primitive, the ended session's Resume path moved into that sheet
// (acceptance line, cap, refusal), a send to an ended session still reopening
// it, and the banner being gone from the composer's top edge.
//
// Imports only the screen and its types, so every assertion here is a real
// failure against a tree without the dot — not a missing-module error.
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { useState } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SessionScreen } from "./screens/session-view";
import type { SessionProjection } from "./types";

/* The round-2 sheet copy (review round 1: U1/D2, D2/D4, N2/U6, U4), the en
   catalogue's exact strings (relay.health.*). */
const RECONNECTING_DETAIL = "This phone lost its connection to your computer and keeps trying. The session may still be running there; sending may fail until the connection is back.";
const DROPPED_DETAIL = "Your computer can't reach this session right now and keeps trying. Until it does, a send fails straight away and your text stays in the message box.";
const QUIET_DETAIL = "This session on your computer has stopped responding — it may be busy or stuck. A send can show a failure and still arrive once it catches up. Retrying won't send it twice.";
const ENDED_DETAIL = "History is kept. Send a message or tap Resume to reopen it in your home folder.";
const ENDED_PLACEHOLDER = "Send to reopen in your home folder…";
const ENDED_SEND_FAILED = "This session has ended and couldn't be reopened just now. Try again, or tap “Ended” at the top and choose Resume.";

const mocks = vi.hoisted(() => ({
	resumeSession: vi.fn(async () => ({ ok: true, pid: 42, session_id: "s1" })),
	sendCommand: vi.fn(async (_sessionId: string, _op: Record<string, unknown>) => ({ ok: true, detail: "" })),
	navigate: vi.fn(),
}));

vi.mock("./router", async (importOriginal) => {
	const actual = await importOriginal<typeof import("./router")>();
	return { ...actual, navigate: mocks.navigate };
});

vi.mock("./api", async () => {
	/* The REAL refusal class so `resumeRefusalText`'s `instanceof` behaves as in
	   production; every function is a stub (a spread would hand unlisted callers
	   a live fetch). */
	const { HttpError } = await vi.importActual<typeof import("./api")>("./api");
	return {
		HttpError,
		getHistory: vi.fn(async () => ({ entries: [], has_more: false })),
		getSubagentHistory: vi.fn(async () => ({ entries: [], has_more: false })),
		getSubagentDetail: vi.fn(async () => null),
		imageUrl: vi.fn(() => ""),
		getCommands: vi.fn(async () => ({ commands: [] })),
		getModels: vi.fn(async () => ({ models: [] })),
		getDirectories: vi.fn(async () => ({ home: "/Users/tester", recent: [], tmp: "" })),
		changeDirectory: vi.fn(async () => ({ ok: true, pid: 1, session_id: "s1" })),
		sendCommand: mocks.sendCommand,
		markSessionSeen: vi.fn(async () => ({ ok: true })),
		setSessionPin: vi.fn(async () => ({ ok: true })),
		resumeSession: mocks.resumeSession,
	};
});

let slot: { projection: SessionProjection | null; connected: boolean } = { projection: null, connected: true };
vi.mock("./store", async (importOriginal) => {
	const actual = await importOriginal<typeof import("./store")>();
	return {
		...actual,
		useProjection: vi.fn(() => slot),
		retainProjectionStream: vi.fn(() => () => {}),
		retainSessionListStream: vi.fn(() => () => {}),
		/* Real draft state, so a send can actually be driven. */
		useDraft: () => useState(""),
	};
});

function projection(over: Partial<SessionProjection> = {}): SessionProjection {
	return {
		session_id: "s1",
		pid: 4242,
		kind: "tui",
		conversation_name: "Toronto mayoral candidates",
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
		transcript: [{ id: "a1", kind: "assistant", text: "Polls are open until 8 p.m." } as SessionProjection["transcript"][number]],
		todos: [],
		subagents: [],
		pending: null,
		pending_count: 0,
		usage: {},
		version: 1,
		...over,
	} satisfies SessionProjection;
}

/** The header's status control: the title block, tagged with its state. */
function control(): HTMLButtonElement {
	const el = document.querySelector<HTMLButtonElement>("header button[data-health]");
	if (!el) throw new Error("no status control in the header");
	return el;
}
function dot(): HTMLElement {
	const el = control().querySelector<HTMLElement>("[data-health-dot]");
	if (!el) throw new Error("status control has no dot");
	return el;
}
function openSheet(): HTMLElement {
	fireEvent.click(control());
	return screen.getByRole("dialog");
}

beforeEach(() => {
	mocks.resumeSession.mockReset();
	mocks.resumeSession.mockImplementation(async () => ({ ok: true, pid: 42, session_id: "s1" }));
	mocks.sendCommand.mockClear();
	mocks.navigate.mockClear();
});
afterEach(() => {
	cleanup();
	localStorage.clear();
	slot = { projection: null, connected: true };
});

describe.each<[string, Partial<SessionProjection>, boolean, string, string, string]>([
	["a live idle owner", {}, true, "live", "bg-success", "Live"],
	["a live owner mid-turn (busy is GREEN)", { streaming: true, activity: "thinking" }, true, "live", "bg-success", "Live"],
	["a stale heartbeat", { heartbeat_stale: true }, true, "not_answering", "bg-warning", "Not answering"],
	["a degraded dial", { degraded: true }, true, "not_answering", "bg-warning", "Not answering"],
	["the phone's own link down", {}, false, "reconnecting", "bg-warning", "Reconnecting"],
	["an ended session", { ended: true }, true, "ended", "bg-ink-dim", "Ended"],
	["a disk rebuild with no running owner", { pid: 0 }, true, "ended", "bg-ink-dim", "Ended"],
])("%s", (_name, over, connected, state, tone, word) => {
	it("paints the dot from the runtime facts and names the state in words", () => {
		slot = { projection: projection(over), connected };
		render(<SessionScreen sessionId="s1" />);
		expect(control().dataset.health).toBe(state);
		expect(dot().className).toContain(tone);
		expect(dot().getAttribute("aria-hidden")).toBe("true");
		// Colour is never the only signal: the state's word is the title, and it
		// is inside the control's accessible text.
		expect(control().getAttribute("title")).toBe(word);
		expect(control().textContent).toContain(word);
		// A 44 px target without growing the 56 px header: the title block is
		// the control.
		expect(control().className).toContain("min-h-11");
		expect(control().getAttribute("aria-haspopup")).toBe("dialog");
	});

	it("opens the explanation in the shared Sheet, titled with the state's word", () => {
		slot = { projection: projection(over), connected };
		render(<SessionScreen sessionId="s1" />);
		const sheet = openSheet();
		expect(sheet.getAttribute("aria-modal")).toBe("true");
		// The shared Sheet names its dialog from its title (`aria-labelledby`).
		expect(screen.getByRole("dialog", { name: word })).toBe(sheet);
		expect(control().getAttribute("aria-expanded")).toBe("true");
		const resume = within(sheet).queryByRole("button", { name: "Resume" });
		// Only an ended session offers Resume.
		if (state === "ended") expect(resume).toBeTruthy();
		else expect(resume).toBeNull();
	});
});

describe("the ended banner is gone", () => {
	it("leaves nothing between the transcript and the composer on an ended session", () => {
		slot = { projection: projection({ ended: true }), connected: true };
		render(<SessionScreen sessionId="s1" />);
		expect(screen.queryByText("Session ended · history kept")).toBeNull();
		expect(screen.queryByText("A send or Resume reopens it in your home folder")).toBeNull();
		expect(screen.queryByRole("button", { name: "Resume" })).toBeNull();
		// No above-composer slot at all when nothing is stacked there.
		expect(document.querySelector("[data-above-composer]")).toBeNull();
		const composer = document.querySelector(".lo-composer");
		const last = screen.getByText("Polls are open until 8 p.m.");
		expect(composer).toBeTruthy();
		expect(last.compareDocumentPosition(composer!) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
	});

	it("no longer paints a degraded warning band — the yellow dot and its sheet carry it", () => {
		slot = { projection: projection({ degraded: true }), connected: true };
		render(<SessionScreen sessionId="s1" />);
		expect(document.querySelector(".bg-warning-wash")).toBeNull();
		expect(screen.queryByText(DROPPED_DETAIL)).toBeNull();
		expect(within(openSheet()).getByText(DROPPED_DETAIL)).toBeTruthy();
	});
});

describe("the ended session's sheet", () => {
	it("explains where it reopens and that a send does it too", () => {
		slot = { projection: projection({ ended: true }), connected: true };
		render(<SessionScreen sessionId="s1" />);
		const sheet = openSheet();
		// ONE paragraph that adds what the title does not (design D2/D4): no
		// "Session ended" re-reading the "Ended" title above it.
		expect(within(sheet).getByText(ENDED_DETAIL)).toBeTruthy();
		expect(within(sheet).queryByText(/session ended/i)).toBeNull();
	});

	it("makes Resume the sheet's primary action, not a tinted slab (design D1)", () => {
		slot = { projection: projection({ ended: true }), connected: true };
		render(<SessionScreen sessionId="s1" />);
		const resume = within(openSheet()).getByRole("button", { name: "Resume" });
		expect(resume.className).toContain("bg-accent");
		expect(resume.className).not.toContain("bg-surface");
	});

	it("resumes through the documented route and lands on the session", async () => {
		slot = { projection: projection({ ended: true }), connected: true };
		render(<SessionScreen sessionId="s1" />);
		fireEvent.click(within(openSheet()).getByRole("button", { name: "Resume" }));
		await waitFor(() => expect(mocks.resumeSession).toHaveBeenCalledWith("s1"));
		await waitFor(() => expect(mocks.navigate).toHaveBeenCalledWith("/s/s1"));
		expect(await screen.findByText("reopening — this can take a few seconds")).toBeTruthy();
		// The same-route no-op leaves the control usable for a retry (U16).
		const again = screen.getByRole("button", { name: "Resume" }) as HTMLButtonElement;
		expect(again.disabled).toBe(false);
	});

	it("says what is true once the reopen outlives its cap (U24)", async () => {
		vi.useFakeTimers();
		try {
			slot = { projection: projection({ ended: true }), connected: true };
			render(<SessionScreen sessionId="s1" />);
			fireEvent.click(within(openSheet()).getByRole("button", { name: "Resume" }));
			await act(async () => {
				await Promise.resolve();
			});
			expect(screen.getByText("reopening — this can take a few seconds")).toBeTruthy();
			await act(async () => {
				vi.advanceTimersByTime(21_000);
			});
			expect(screen.getByText("still reopening — it has not come up yet")).toBeTruthy();
		} finally {
			vi.useRealTimers();
		}
	});

	it("renders a refusal in the one refusal voice and keeps Resume available", async () => {
		const api = await import("./api");
		mocks.resumeSession.mockImplementation(async () => {
			throw new api.HttpError(404, "no such past session: s1");
		});
		slot = { projection: projection({ ended: true }), connected: true };
		render(<SessionScreen sessionId="s1" />);
		const sheet = openSheet();
		fireEvent.click(within(sheet).getByRole("button", { name: "Resume" }));
		await waitFor(() =>
			expect(within(sheet).getByRole("alert").textContent).toBe("Could not resume: this session is no longer saved"),
		);
		expect((within(sheet).getByRole("button", { name: "Resume" }) as HTMLButtonElement).disabled).toBe(false);
		expect(mocks.navigate).not.toHaveBeenCalled();
	});

	it("keeps the acceptance line across closing and reopening the sheet", async () => {
		slot = { projection: projection({ ended: true }), connected: true };
		render(<SessionScreen sessionId="s1" />);
		fireEvent.click(within(openSheet()).getByRole("button", { name: "Resume" }));
		await screen.findByText("reopening — this can take a few seconds");
		fireEvent.click(within(screen.getByRole("dialog")).getByRole("button", { name: /close/i }));
		await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
		expect(within(openSheet()).getByText("reopening — this can take a few seconds")).toBeTruthy();
	});
});

describe("a send to an ended session still reopens it", () => {
	it("goes out as a normal prompt — the composer is not disabled and needs no Resume first", async () => {
		slot = { projection: projection({ ended: true }), connected: true };
		render(<SessionScreen sessionId="s1" />);
		const field = screen.getByRole("textbox", { name: /message/i });
		expect((field as HTMLTextAreaElement).disabled).toBe(false);
		fireEvent.change(field, { target: { value: "who won?" } });
		fireEvent.click(screen.getByRole("button", { name: "send" }));
		await waitFor(() => expect(mocks.sendCommand).toHaveBeenCalled());
		const [sessionId, envelope] = mocks.sendCommand.mock.calls[0]!;
		expect(sessionId).toBe("s1");
		expect(envelope).toMatchObject({ op: "prompt", text: "who won?" });
		// The send path, not the resume route: the daemon wakes a host for a
		// prompt to an ended session (issue #1875).
		expect(mocks.resumeSession).not.toHaveBeenCalled();
	});
});

describe("each yellow and grey sheet says what is TRUE of a send (review round 1)", () => {
	/* The daemon, not a wish: a degraded entry has no writer, so `request()`
	   raises and the command route answers 409 at once (the composer puts the
	   text back); a stale heartbeat with the link up writes the frame, waits
	   15 s and answers 504 — the message may still land. Neither queues. */
	it.each<[string, Partial<SessionProjection>, boolean, string, string]>([
		["a dropped link (degraded)", { degraded: true }, true, "Not answering", DROPPED_DETAIL],
		["a quiet heartbeat with the link up", { heartbeat_stale: true }, true, "Not answering", QUIET_DETAIL],
		["the phone's own link down", {}, false, "Reconnecting", RECONNECTING_DETAIL],
		/* The two precedence pairs a sheet makes visible (copy sign-off): the
		   dial being down is what a send meets first, and an offline phone
		   cannot vouch for either yellow cause from its last frame. */
		["a dropped link AND a quiet beat → dropped", { degraded: true, heartbeat_stale: true }, true, "Not answering", DROPPED_DETAIL],
		["an offline phone on a degraded session → reconnecting", { degraded: true }, false, "Reconnecting", RECONNECTING_DETAIL],
		["an ended session", { ended: true }, true, "Ended", ENDED_DETAIL],
	])("%s", (_name, over, connected, title, body) => {
		slot = { projection: projection(over), connected };
		render(<SessionScreen sessionId="s1" />);
		const sheet = openSheet();
		expect(screen.getByRole("dialog", { name: title })).toBe(sheet);
		const line = within(sheet).getByText(body);
		// One explanation register for every state (design D4).
		expect(line.className).toContain("text-body-sm");
		expect(line.className).toContain("text-ink-muted");
		expect(within(sheet).queryByText(/sends will wait/)).toBeNull();
		expect(within(sheet).queryByText(/Offline · retrying/)).toBeNull();
	});
});

describe("transitions into a non-live state are announced (UX U2)", () => {
	function region(): HTMLElement {
		const el = document.querySelector<HTMLElement>("[data-health-announce]");
		if (!el) throw new Error("no health live region");
		return el;
	}
	it("speaks the state's own word through one polite region outside the control", () => {
		slot = { projection: projection(), connected: true };
		const view = render(<SessionScreen sessionId="s1" />);
		expect(region().getAttribute("aria-live")).toBe("polite");
		// Outside the button, so the words are not read twice as its name.
		expect(control().contains(region())).toBe(false);
		expect(region().textContent).toBe("");
		slot = { projection: projection({ heartbeat_stale: true }), connected: true };
		view.rerender(<SessionScreen sessionId="s1" />);
		expect(region().textContent).toBe("Not answering");
		slot = { projection: projection({ ended: true }), connected: true };
		view.rerender(<SessionScreen sessionId="s1" />);
		expect(region().textContent).toBe("Ended");
		slot = { projection: projection(), connected: false };
		view.rerender(<SessionScreen sessionId="s1" />);
		expect(region().textContent).toBe("Reconnecting");
		slot = { projection: projection(), connected: true };
		view.rerender(<SessionScreen sessionId="s1" />);
		expect(region().textContent).toBe("");
	});
});

describe("Resume is also where actions live: the ⋮ menu (UX U3)", () => {
	function openMenu(): HTMLElement {
		fireEvent.click(screen.getByRole("button", { name: "session menu" }));
		return screen.getByRole("dialog", { name: "Session" });
	}
	it("offers Resume only while ended, and runs the ONE resume path", async () => {
		slot = { projection: projection(), connected: true };
		const live = render(<SessionScreen sessionId="s1" />);
		expect(within(openMenu()).queryByRole("button", { name: "Resume" })).toBeNull();
		live.unmount();

		slot = { projection: projection({ ended: true }), connected: true };
		render(<SessionScreen sessionId="s1" />);
		fireEvent.click(within(openMenu()).getByRole("button", { name: "Resume" }));
		await waitFor(() => expect(mocks.resumeSession).toHaveBeenCalledWith("s1"));
		// The outcome is visible: the health sheet opens with the acceptance line.
		const sheet = await screen.findByRole("dialog", { name: "Ended" });
		expect(await within(sheet).findByText("reopening — this can take a few seconds")).toBeTruthy();
		expect(screen.queryByRole("dialog", { name: "Session" })).toBeNull();
	});

	it("shows a refusal from the menu in the health sheet, not behind it", async () => {
		const api = await import("./api");
		mocks.resumeSession.mockImplementation(async () => {
			throw new api.HttpError(404, "no such past session: s1");
		});
		slot = { projection: projection({ ended: true }), connected: true };
		render(<SessionScreen sessionId="s1" />);
		fireEvent.click(within(openMenu()).getByRole("button", { name: "Resume" }));
		const sheet = await screen.findByRole("dialog", { name: "Ended" });
		await waitFor(() =>
			expect(within(sheet).getByRole("alert").textContent).toBe("Could not resume: this session is no longer saved"),
		);
	});
});

describe("the ended composer says what a send does BEFORE the tap (UX U4)", () => {
	it("swaps the placeholder on an ended session only", () => {
		slot = { projection: projection({ ended: true }), connected: true };
		const ended = render(<SessionScreen sessionId="s1" />);
		expect(screen.getByRole("textbox", { name: /message/i }).getAttribute("placeholder")).toBe(ENDED_PLACEHOLDER);
		ended.unmount();
		slot = { projection: projection(), connected: true };
		render(<SessionScreen sessionId="s1" />);
		expect(screen.getByRole("textbox", { name: /message/i }).getAttribute("placeholder")).toBe("Message…");
	});

	/* ROUND 2, R2-2 = U7: the composer reads the SAME "ended" the header dot
	   does. A conversation opened from history is a disk rebuild with no
	   running owner (`pid: 0`, `ended: false`) — the dot says "Ended" there, and
	   a send to it is the same wake, so the composer must say the same before
	   and after the tap, not the live wording. */
	it("treats a history rebuild with no running owner (pid 0) as ended too", async () => {
		const api = await import("./api");
		mocks.sendCommand.mockRejectedValueOnce(new api.HttpError(502, "bad gateway"));
		slot = { projection: projection({ pid: 0, ended: false }), connected: true };
		render(<SessionScreen sessionId="s1" />);
		expect(control().dataset.health).toBe("ended");
		const field = screen.getByRole("textbox", { name: /message/i });
		expect(field.getAttribute("placeholder")).toBe(ENDED_PLACEHOLDER);
		fireEvent.change(field, { target: { value: "who won?" } });
		fireEvent.click(screen.getByRole("button", { name: "send" }));
		await waitFor(() => expect(screen.getByRole("alert").textContent).toContain(ENDED_SEND_FAILED));
	});
});
