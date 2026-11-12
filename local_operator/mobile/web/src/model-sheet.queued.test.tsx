// @vitest-environment happy-dom
//
// Deferred settings picks on the phone (PR #3697 review R1-M2).
//
// A current runtime acks EVERY `defer_ok` switch `queued` before the switch
// runs, so a refusal (a provider with no credential, a model this device does
// not offer) can only arrive later, on the projection's `settings_results`.
// These tests drive that real sequence: the sheet gets the `queued` ack and
// closes, and the composer — still on screen after the sheet is gone — shows
// the refusal correlated by `request_id`, while a success stays silent and
// another device's outcome is ignored.
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useState } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ModelSheet } from "./components/model-sheet";
import {
	reconcileSettingsResults,
	resetSettingsDeferForTests,
	trackSettingsPick,
} from "./lib/settings-defer";
import { SessionScreen } from "./screens/session-view";
import type { ModelEntry, SessionProjection } from "./types";

const mocks = vi.hoisted(() => ({
	sendCommand: vi.fn(async (_sessionId: string, _op: Record<string, unknown>) => ({
		ok: true,
		detail: "queued",
	})),
}));

const ROWS: ModelEntry[] = [
	{
		selector: "openai/gpt-new",
		provider: "openai",
		model_id: "gpt-new",
		name: "GPT New",
		label: "GPT New",
		connected: true,
		aggregated: false,
	},
];

vi.mock("./api", () => ({
	getHistory: vi.fn(async () => ({ entries: [], has_more: false })),
	getSubagentHistory: vi.fn(async () => ({ entries: [], has_more: false })),
	getSubagentDetail: vi.fn(async () => null),
	imageUrl: vi.fn(() => ""),
	getCommands: vi.fn(async () => ({ commands: [] })),
	getSkills: vi.fn(async () => ({ skills: [] })),
	getModels: vi.fn(async () => ({ models: ROWS })),
	getDirectories: vi.fn(async () => ({ home: "/Users/tester", recent: [], tmp: "" })),
	changeDirectory: vi.fn(async () => ({ ok: true, pid: 1, session_id: "s1" })),
	sendCommand: mocks.sendCommand,
	markSessionSeen: vi.fn(async () => ({ ok: true })),
	HttpError: class HttpError extends Error {
		constructor(
			readonly status: number,
			message: string,
			readonly code = "",
		) {
			super(message);
		}
	},
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
		useDraft: () => useState(""),
	};
});

function projection(extra: Partial<SessionProjection> = {}): SessionProjection {
	return {
		session_id: "s1",
		pid: 1,
		kind: "tui",
		conversation_name: "Settings",
		cwd: "",
		model_label: "",
		model_selector: "anthropic/claude-opus-5",
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
		...extra,
	} as SessionProjection;
}

const QUEUED_LINE = /applies at the next step/;

beforeEach(() => {
	mocks.sendCommand.mockReset();
	mocks.sendCommand.mockResolvedValue({ ok: true, detail: "queued" });
	resetSettingsDeferForTests();
});

afterEach(() => {
	cleanup();
	localStorage.clear();
	slot = { projection: null, connected: true };
});

describe("model sheet deferred pick", () => {
	it("asks for a deferred ack and closes on the queued receipt", async () => {
		const onClose = vi.fn();
		render(<ModelSheet open onClose={onClose} pid="s1" projection={projection()} />);
		fireEvent.click(await screen.findByText("GPT New"));
		await waitFor(() => expect(onClose).toHaveBeenCalled());
		const op = mocks.sendCommand.mock.calls[0][1];
		expect(op).toMatchObject({ op: "set_model", provider: "openai", model_id: "gpt-new", defer_ok: true });
		expect(typeof op.request_id).toBe("string");
	});

	it("keeps a synchronous refusal in the sheet, as before", async () => {
		mocks.sendCommand.mockRejectedValue(new Error("no credential for openai"));
		const onClose = vi.fn();
		render(<ModelSheet open onClose={onClose} pid="s1" projection={projection()} />);
		fireEvent.click(await screen.findByText("GPT New"));
		await screen.findByText("no credential for openai");
		expect(onClose).not.toHaveBeenCalled();
	});
});

describe("composer shows the correlated outcome after the sheet closed", () => {
	async function pickThenRepaint(result: (requestId: string) => Record<string, unknown>) {
		const onClose = vi.fn();
		const sheet = render(<ModelSheet open onClose={onClose} pid="s1" projection={projection()} />);
		fireEvent.click(await screen.findByText("GPT New"));
		await waitFor(() => expect(onClose).toHaveBeenCalled());
		sheet.unmount();
		const requestId = String(mocks.sendCommand.mock.calls[0][1].request_id);
		const { rerender } = render(<SessionScreen sessionId="s1" />);
		slot = { projection: projection({ settings_results: result(requestId) as never }), connected: true };
		rerender(<SessionScreen sessionId="s1" />);
		return requestId;
	}

	it("shows a later refusal with the session's reason, and Dismiss clears it", async () => {
		await pickThenRepaint((id) => ({
			model: { request_id: id, ok: false, code: "credential_missing", message: "openai has no credential" },
		}));
		const alert = await screen.findByRole("alert");
		expect(alert.textContent).toBe("openai/gpt-new was not applied: openai has no credential");
		fireEvent.click(screen.getByRole("button", { name: "Dismiss" }));
		await waitFor(() => expect(screen.queryByText(/was not applied/)).toBeNull());
	});

	it("stays silent when the switch applied", async () => {
		await pickThenRepaint((id) => ({ model: { request_id: id, ok: true, code: "applied", message: "model: openai/gpt-new" } }));
		expect(screen.queryByText(/was not applied/)).toBeNull();
		expect(screen.queryByText(QUEUED_LINE)).toBeNull();
	});

	it("ignores another device's refusal", async () => {
		await pickThenRepaint(() => ({ model: { request_id: "someone-else", ok: false, message: "nope" } }));
		expect(screen.queryByText(/was not applied/)).toBeNull();
	});
});

describe("queued status and last-wins", () => {
	it("says queued only while a turn is running", () => {
		trackSettingsPick("s1", { requestId: "r1", kind: "effort", label: "high" });
		const { rerender } = render(<SessionScreen sessionId="s1" />);
		slot = { projection: projection({ streaming: false }), connected: true };
		rerender(<SessionScreen sessionId="s1" />);
		expect(screen.queryByText(QUEUED_LINE)).toBeNull();
		slot = { projection: projection({ streaming: true }), connected: true };
		rerender(<SessionScreen sessionId="s1" />);
		expect(screen.getByText(QUEUED_LINE)).toBeTruthy();
	});

	it("drops an older pick's outcome once a newer pick of the same kind is made", () => {
		trackSettingsPick("s1", { requestId: "old", kind: "model", label: "a/old" });
		trackSettingsPick("s1", { requestId: "new", kind: "model", label: "a/new" });
		trackSettingsPick("s1", { requestId: "eff", kind: "effort", label: "low" });
		const { rerender } = render(<SessionScreen sessionId="s1" />);
		act(() => {
			reconcileSettingsResults("s1", {
				model: { request_id: "old", ok: false, message: "stale" },
				effort: { request_id: "eff", ok: false, message: "no low rung" },
			});
		});
		slot = { projection: projection(), connected: true };
		rerender(<SessionScreen sessionId="s1" />);
		expect(screen.getByRole("alert").textContent).toBe("low was not applied: no low rung");
		expect(screen.queryByText(/stale/)).toBeNull();
	});
});
