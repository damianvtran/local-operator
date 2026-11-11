// @vitest-environment happy-dom
//
// THE BUSY-SEND DELIVERY CHIP: steer / queue / both on the phone composer.
// Visible only while streaming AND the relay advertised `features.busy_delivery`;
// one tap cycles and persists per device; a busy send carries the mode only for
// queue/both; a 400 naming delivery is shown verbatim, never swallowed.
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as api from "./api";
import { Composer } from "./components/composer";
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

const chip = () => document.querySelector<HTMLButtonElement>('[aria-label^="delivery"]');

beforeEach(() => {
	capabilitySlot = { features: { busy_delivery: 1 } } as Capabilities;
	localStorage.clear();
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
	localStorage.clear();
	Object.defineProperty(navigator, "mediaDevices", { configurable: true, value: undefined });
});

describe("the delivery chip", () => {
	it("is hidden when idle, even with the feature", () => {
		mount(false);
		expect(chip()).toBeNull();
	});

	it("is hidden while streaming without the feature key", () => {
		capabilitySlot = { features: {} } as Capabilities;
		mount(true);
		expect(chip()).toBeNull();
	});

	it("is hidden when streaming is false, visible when true", () => {
		const { setStreaming } = mount(false);
		expect(chip()).toBeNull();
		setStreaming(true);
		expect(chip()).not.toBeNull();
		setStreaming(false);
		expect(chip()).toBeNull();
	});

	it("defaults to steer and cycles queue → both → steer", () => {
		mount(true);
		expect(chip()!.textContent).toContain("steer");
		fireEvent.click(chip()!);
		expect(chip()!.textContent).toContain("queue");
		fireEvent.click(chip()!);
		expect(chip()!.textContent).toContain("both");
		fireEvent.click(chip()!);
		expect(chip()!.textContent).toContain("steer");
	});

	it("persists per device and reads back", () => {
		mount(true);
		fireEvent.click(chip()!);
		expect(localStorage.getItem("lo-mobile-delivery-mode")).toBe("queue");
		cleanup();
		mount(true);
		expect(chip()!.textContent).toContain("queue");
	});

	it("a busy send carries delivery only for queue/both, never for steer", async () => {
		mount(true);
		fireEvent.change(screen.getByPlaceholderText("Steer the agent…"), { target: { value: "hi" } });
		await act(async () => { fireEvent.click(screen.getByRole("button", { name: "steer" })); });
		const first = vi.mocked(api.sendCommand).mock.calls[0][1] as Record<string, unknown>;
		expect(first).not.toHaveProperty("delivery");

		fireEvent.click(chip()!);
		expect(chip()!.textContent).toContain("queue");
		fireEvent.change(screen.getByPlaceholderText("Steer the agent…"), { target: { value: "again" } });
		await act(async () => { fireEvent.click(screen.getByRole("button", { name: "steer" })); });
		const second = vi.mocked(api.sendCommand).mock.calls[1][1] as Record<string, unknown>;
		expect(second).toMatchObject({ op: "steer", text: "again", delivery: "queue" });
	});

	it("a 400 naming delivery is shown, not swallowed", async () => {
		vi.mocked(api.sendCommand).mockRejectedValueOnce(
			new api.HttpError(400, "this session's runtime predates delivery modes"),
		);
		mount(true);
		fireEvent.click(chip()!);
		fireEvent.change(screen.getByPlaceholderText("Steer the agent…"), { target: { value: "hi" } });
		await act(async () => { fireEvent.click(screen.getByRole("button", { name: "steer" })); });
		expect(await screen.findByText("this session's runtime predates delivery modes")).not.toBeNull();
	});
});
