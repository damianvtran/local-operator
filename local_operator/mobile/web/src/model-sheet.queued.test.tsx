// @vitest-environment happy-dom
//
// A model pick the session cannot land yet (its loop is mid-step) is answered
// `queued` by the runtime's deferred settings ack: accepted, applied at the next
// step. The sheet must ask for that ack (`defer_ok` + `request_id`) and, on a
// queued answer, stay open and say so — the projection will not show the new
// model until it lands, so a silent close read as "nothing happened". An
// applied answer still closes the sheet as before.
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ModelSheet } from "./components/model-sheet";
import type { ModelEntry, SessionProjection } from "./types";

const sendCommand = vi.fn();

vi.mock("./api", () => ({
	getModels: vi.fn(async () => ({ models: ROWS })),
	sendCommand: (...args: unknown[]) => sendCommand(...args),
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

const projection = {
	model_selector: "anthropic/claude-opus-5",
	effort_ladder: [],
} as unknown as SessionProjection;

const QUEUED_LINE = /applies at the next step/;

describe("model sheet deferred settings ack", () => {
	beforeEach(() => sendCommand.mockReset());
	afterEach(cleanup);

	it("asks for a deferred ack and stays open saying it applies at the next step", async () => {
		sendCommand.mockResolvedValue({ ok: true, detail: "queued" });
		const onClose = vi.fn();
		render(<ModelSheet open onClose={onClose} pid="1" projection={projection} />);
		fireEvent.click(await screen.findByText("GPT New"));
		await waitFor(() => expect(screen.getByRole("status").textContent).toMatch(QUEUED_LINE));
		const op = sendCommand.mock.calls[0][1];
		expect(op).toMatchObject({ op: "set_model", provider: "openai", model_id: "gpt-new", defer_ok: true });
		expect(typeof op.request_id).toBe("string");
		expect(onClose).not.toHaveBeenCalled();
	});

	it("closes as before when the switch applied inline", async () => {
		sendCommand.mockResolvedValue({ ok: true, detail: "model: openai/gpt-new" });
		const onClose = vi.fn();
		render(<ModelSheet open onClose={onClose} pid="1" projection={projection} />);
		fireEvent.click(await screen.findByText("GPT New"));
		await waitFor(() => expect(onClose).toHaveBeenCalled());
		expect(screen.queryByText(QUEUED_LINE)).toBeNull();
	});
});
