// @vitest-environment happy-dom
//
// THE ACTION CLUSTER AS ONE PAIR OF FIXED SLOTS, driven through the real Composer:
// [A: mic][B: primary], where B holds exactly one of send / steer / stop. happy-dom
// does no layout, so pixel positions are browser-measured on the PR; what is
// pinnable here is the structure that makes them true: slot order, one occupant
// in B, the reserved mic box, the keyed-per-role buttons, and the Stop guard.
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as api from "./api";
import { Composer } from "./components/composer";
import { WorkingLine } from "./components/working-line";
import { GUARD_MS } from "./lib/composer-slots";
import { resetTurnStopForTests } from "./lib/turn-stop";
import type { Capabilities, SessionProjection } from "./types";

vi.mock("./api", async (importOriginal) => {
	const actual = await importOriginal<typeof import("./api")>();
	return {
		...actual,
		getCommands: vi.fn(async () => ({ commands: [] })),
		getSkills: vi.fn(async () => ({ skills: [] })),
		getModels: vi.fn(async () => ({ models: [] })),
		sendCommand: vi.fn(async () => ({ ok: true, detail: "" })),
	};
});

let capabilitySlot: Capabilities | null = null;
vi.mock("./store", async (importOriginal) => {
	const actual = await importOriginal<typeof import("./store")>();
	return { ...actual, useDraft: () => useState(""), useCapabilities: () => capabilitySlot };
});

function projection(streaming: boolean): SessionProjection {
	return {
		session_id: "p1", pid: 1, kind: "tui", conversation_name: "", cwd: "", model_label: "",
		model_selector: "", effort: "", effort_ladder: [], streaming, activity: "", activity_started_s: 0,
		stop_reason: "completed", queued_count: 0, ended: false, degraded: false, transcript: [],
		todos: [], subagents: [], pending: null, pending_count: 0, usage: {}, version: 1,
	} as unknown as SessionProjection;
}

function mount(streaming: boolean) {
	const props = {
		ended: false, pid: "p1", onOpenModels: () => {}, onReturnToModels: () => {},
		onOpenEffort: () => {}, effortOpen: false, onCloseEffort: () => {},
	};
	const view = render(<Composer {...props} projection={projection(streaming)} />);
	return {
		...view,
		setStreaming: (on: boolean) => view.rerender(<Composer {...props} projection={projection(on)} />),
	};
}

const cluster = () => screen.getByTestId("composer-actions");
const roles = () => [...cluster().querySelectorAll("[data-role]")].map((el) => el.getAttribute("data-role"));

beforeEach(() => {
	capabilitySlot = { stt: { available: true, path: "p", reason: "" } };
	vi.stubGlobal("MediaRecorder", class {});
	vi.stubGlobal("isSecureContext", true);
	Object.defineProperty(navigator, "mediaDevices", { configurable: true, value: { getUserMedia: vi.fn() } });
});
afterEach(() => {
	cleanup();
	resetTurnStopForTests();
	vi.useRealTimers();
	vi.clearAllMocks();
	vi.unstubAllGlobals();
	Object.defineProperty(navigator, "mediaDevices", { configurable: true, value: undefined });
});

describe("slot B holds exactly one primary control", () => {
	it("idle + empty: a DISABLED send, no stop", () => {
		mount(false);
		expect(roles()).toEqual(["send"]);
		expect((screen.getByRole("button", { name: "send" }) as HTMLButtonElement).disabled).toBe(true);
	});

	it("idle + text: an enabled send", () => {
		mount(false);
		fireEvent.change(screen.getByPlaceholderText("Ask anything…"), { target: { value: "hi" } });
		expect((screen.getByRole("button", { name: "send" }) as HTMLButtonElement).disabled).toBe(false);
	});

	it("working + empty: stop alone, never beside a send", () => {
		mount(true);
		expect(roles()).toEqual(["stop"]);
		expect(screen.queryByRole("button", { name: "send" })).toBeNull();
		expect(screen.queryByRole("button", { name: "steer" })).toBeNull();
	});

	it("working + text: steer alone, never beside a stop", () => {
		mount(true);
		fireEvent.change(screen.getByPlaceholderText("Steer the agent…"), { target: { value: "also this" } });
		expect(roles()).toEqual(["steer"]);
		expect(screen.queryByRole("button", { name: "stop" })).toBeNull();
	});

	it("the mic precedes the primary slot and is the same size class", () => {
		mount(true);
		const kids = [...cluster().children];
		expect(kids[0]?.getAttribute("aria-label")).toBe("start voice input");
		expect(kids[1]?.getAttribute("data-role")).toBe("stop");
		for (const el of kids) expect(el.className).toContain("size-11");
		expect(cluster().className).toContain("gap-2");
	});

	it("the mic keeps its position in the cluster across a turn starting", () => {
		const { setStreaming } = mount(false);
		const before = [...cluster().children].indexOf(screen.getByRole("button", { name: "start voice input" }));
		setStreaming(true);
		const after = [...cluster().children].indexOf(screen.getByRole("button", { name: "start voice input" }));
		expect(after).toBe(before);
		expect(cluster().children.length).toBe(2);
	});
});

describe("slot A while the capability is unknown", () => {
	it("holds an inert 44 px box, then the mic replaces it in place", () => {
		capabilitySlot = null;
		mount(false);
		expect(cluster().children.length).toBe(2);
		expect(cluster().querySelector("[data-mic-reserve]")?.className).toContain("size-11");
		expect(screen.queryByRole("button", { name: "start voice input" })).toBeNull();
	});

	it("collapses when the daemon says no path can run", () => {
		capabilitySlot = { stt: { available: false, path: null } };
		mount(false);
		expect(cluster().children.length).toBe(1);
	});
});

describe("the Stop guard", () => {
	it("swallows a pointer press inside 500 ms of Stop arriving, then lets one through", async () => {
		vi.useFakeTimers();
		const nowSpy = vi.spyOn(performance, "now");
		nowSpy.mockReturnValue(1000);
		const { setStreaming } = mount(false);
		await act(async () => { setStreaming(true); });
		const stop = screen.getByRole("button", { name: "stop" });
		nowSpy.mockReturnValue(1000 + 150);
		fireEvent.click(stop, { detail: 1 });
		expect(api.sendCommand).not.toHaveBeenCalled();
		// the swallowed press did not disable the control
		expect((stop as HTMLButtonElement).disabled).toBe(false);
		nowSpy.mockReturnValue(1000 + GUARD_MS + 1);
		await act(async () => { fireEvent.click(stop, { detail: 1 }); });
		expect(api.sendCommand).toHaveBeenCalledTimes(1);
		expect(api.sendCommand).toHaveBeenCalledWith("p1", { op: "abort" });
	});

	it("never guards keyboard activation", async () => {
		const nowSpy = vi.spyOn(performance, "now");
		nowSpy.mockReturnValue(1000);
		const { setStreaming } = mount(false);
		await act(async () => { setStreaming(true); });
		await act(async () => { fireEvent.click(screen.getByRole("button", { name: "stop" }), { detail: 0 }); });
		expect(api.sendCommand).toHaveBeenCalledTimes(1);
	});

	it("a second stop while one is pending is not a second abort", async () => {
		const nowSpy = vi.spyOn(performance, "now");
		nowSpy.mockReturnValue(0);
		mount(true);
		nowSpy.mockReturnValue(10_000);
		await act(async () => { fireEvent.click(screen.getByRole("button", { name: "stop" }), { detail: 1 }); });
		const busy = screen.getByRole("button", { name: "stopping" });
		expect(busy.getAttribute("aria-busy")).toBe("true");
		await act(async () => { fireEvent.click(busy, { detail: 1 }); });
		expect(api.sendCommand).toHaveBeenCalledTimes(1);
	});
});

describe("slot B is locked while a send is in flight", () => {
	it("never becomes Stop between the box clearing and the request settling (double-tap race)", async () => {
		let settle!: (v: { ok: boolean; detail: string }) => void;
		vi.mocked(api.sendCommand).mockImplementationOnce(
			() => new Promise((resolve) => { settle = resolve; }),
		);
		mount(true);
		fireEvent.change(screen.getByPlaceholderText("Steer the agent…"), { target: { value: "redirect" } });
		await act(async () => { fireEvent.click(screen.getByRole("button", { name: "steer" })); });
		// The box has been cleared by the send, so alive && empty would read as Stop: the lock must hold.
		await vi.waitFor(() => expect(roles()).toEqual(["connecting"]));
		expect(screen.queryByRole("button", { name: "stop" })).toBeNull();
		expect((screen.getByRole("button", { name: "Connecting" }) as HTMLButtonElement).disabled).toBe(true);
		await act(async () => { settle({ ok: true, detail: "" }); });
		await vi.waitFor(() => expect(roles()).toEqual(["stop"]));
	});
});

describe("focus handoff when a focused role button unmounts", () => {
	it("moves focus to the textarea instead of dropping it to <body>", async () => {
		const { setStreaming } = mount(false);
		const field = screen.getByPlaceholderText("Ask anything…") as HTMLTextAreaElement;
		fireEvent.change(field, { target: { value: "hi" } });
		const send = screen.getByRole("button", { name: "send" });
		await act(async () => { send.focus(); });
		expect(document.activeElement).toBe(send);
		// the turn starts: role send -> steer, the focused element is replaced
		await act(async () => { setStreaming(true); });
		expect(document.activeElement).not.toBe(document.body);
		expect(document.activeElement).toBe(field);
	});

	it("leaves focus alone when the unmounted button was not focused", async () => {
		const { setStreaming } = mount(false);
		const field = screen.getByPlaceholderText("Ask anything…") as HTMLTextAreaElement;
		fireEvent.change(field, { target: { value: "hi" } });
		(document.activeElement as HTMLElement | null)?.blur();
		await act(async () => { setStreaming(true); });
		expect(document.activeElement).toBe(document.body);
	});
});

describe("the composer's Stop and the working line's Stop share one latch", () => {
	it("a tap on the working-line Stop puts the composer's disc into its pressed state, and a second Stop is not a second abort", async () => {
		const nowSpy = vi.spyOn(performance, "now");
		nowSpy.mockReturnValue(0);
		mount(true);
		render(<WorkingLine activity="responding" startedS={0} stopPid="p1" />);
		const lineStop = document.querySelector("[data-working-stop]") as HTMLButtonElement;
		await act(async () => { fireEvent.click(lineStop); });
		expect(api.sendCommand).toHaveBeenCalledTimes(1);
		// BOTH controls show the pressed state: the disc by its role, the line by its label.
		const disc = cluster().querySelector("[data-role='stopping']") as HTMLButtonElement;
		expect(disc.getAttribute("aria-busy")).toBe("true");
		expect(lineStop.getAttribute("aria-busy")).toBe("true");
		nowSpy.mockReturnValue(10_000);
		await act(async () => { fireEvent.click(disc, { detail: 1 }); });
		expect(api.sendCommand).toHaveBeenCalledTimes(1);
	});
});
