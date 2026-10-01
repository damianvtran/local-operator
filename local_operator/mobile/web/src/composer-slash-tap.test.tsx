// @vitest-environment happy-dom
//
// MINOR-3: the phone's slash sheet decides "tap runs it" vs "tap waits for the
// text" from the catalogue's `arguments` field, and the backend moved `/goal`
// and `/loop` from `none` to `optional` when their flag rows landed. So a tap on
// either row now INSERTS `/goal ` / `/loop ` and waits, where it used to run the
// bare word — a shipped surface changed by a remote field, with no edit in this
// repository to notice it.
//
// The behaviour is the one the sheet already gives `/rename`, `/theme`, `/stop`,
// `/mcp` and `/approvals` (all `optional`), and it is the safer half for
// `/loop` (the bare word used to start iterations). This test renders the REAL
// Composer over a mocked catalogue so the decision is pinned on the surface a
// user taps, not on a copy of the expression.
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Composer } from "./components/composer";
import type { SessionProjection, SlashCommand } from "./types";

//: A catalogue that carries one command of each shape — all of them commands the
//: phone's real sheet offers (the daemon serves only the routed set, issue #1869). `goal` is `optional`
//: since the flag work; `compact` stays `none`, the shape whose tap runs.
const CATALOGUE: SlashCommand[] = [
	{
		name: "goal",
		description: "Set the goal and start work; /goal --clear clears it",
		aliases: [],
		arguments: "optional",
	},
	{
		name: "loop",
		description: "Loop toward a goal: /loop <goal>, <n>; --stop cancels",
		aliases: [],
		arguments: "optional",
	},
	{
		name: "compact",
		description: "Compact the conversation context",
		aliases: [],
		arguments: "none",
	},
];

vi.mock("./api", () => ({
	getCommands: vi.fn(async () => ({ commands: CATALOGUE })),
	getModels: vi.fn(async () => ({ models: [] })),
	sendCommand: vi.fn(async (..._args: unknown[]) => ({ ok: true, detail: "" })),
	// The real class: the composer tells a daemon refusal (422 + its own sentence)
	// from a transport failure by `instanceof`.
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

vi.mock("./store", async (importOriginal) => {
	const actual = await importOriginal<typeof import("./store")>();
	return {
		...actual,
		// The draft is the composer's own state, so the mock is REAL state:
		// typing has to re-render for the sheet to open on the `/` keystroke.
		useDraft: () => useState(""),
	};
});

function projection(): SessionProjection {
	return {
		session_id: "s1",
		pid: 1,
		kind: "tui",
		conversation_name: "Goal",
		cwd: "",
		model_label: "",
		model_selector: "",
		effort: "",
		effort_ladder: [],
		streaming: false,
		activity: "",
		activity_started_s: 0,
		stop_reason: "completed",
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
	} as unknown as SessionProjection;
}

async function openSheet() {
	render(
		<Composer
			pid="p1"
			projection={projection()}
			onOpenModels={() => {}}
			onOpenEffort={() => {}}
			effortOpen={false}
			onCloseEffort={() => {}}
		/>,
	);
	const field = screen.getByPlaceholderText("Message…") as HTMLTextAreaElement;
	fireEvent.change(field, { target: { value: "/" } });
	await waitFor(() => expect(screen.getByRole("button", { name: /\/goal/ })).toBeTruthy());
	return field;
}

afterEach(() => {
	cleanup();
	localStorage.clear();
	vi.clearAllMocks();
});

describe("the phone's slash sheet tap", () => {
	it("fills and waits for an optional-argument command instead of running it", async () => {
		const field = await openSheet();
		const api = await import("./api");

		fireEvent.click(screen.getByRole("button", { name: /\/goal/ }));

		// The gap is the tap's one keystroke: the user is left composing the
		// goal, and nothing has been sent.
		expect(field.value).toBe("/goal ");
		expect(vi.mocked(api.sendCommand)).not.toHaveBeenCalled();
	});

	it("still runs a command that takes no argument", async () => {
		const field = await openSheet();
		const api = await import("./api");

		fireEvent.click(screen.getByRole("button", { name: /\/compact/ }));

		expect(field.value).toBe("/compact");
		expect(vi.mocked(api.sendCommand)).toHaveBeenCalledTimes(1);
		// The routed seam, not the off-terminal subset: `slash` answers every word
		// but /goal and /compact with "terminal-only here" (issue #1869).
		expect(vi.mocked(api.sendCommand)).toHaveBeenCalledWith("p1", {
			op: "slash_result",
			command: "compact",
			args: "",
			images: [],
		});
	});

	it("waits for `/loop` too, which is the safer half of the change", async () => {
		const field = await openSheet();
		const api = await import("./api");

		fireEvent.click(screen.getByRole("button", { name: /\/loop/ }));

		// The bare word used to start iterations; a tap no longer spends a turn.
		expect(field.value).toBe("/loop ");
		expect(vi.mocked(api.sendCommand)).not.toHaveBeenCalled();
	});

	it("shows what a routed command did, in the runtime's words", async () => {
		const api = await import("./api");
		vi.mocked(api.sendCommand).mockResolvedValueOnce({ ok: true, detail: "compacting context…" });
		const field = await openSheet();

		fireEvent.click(screen.getByRole("button", { name: /\/compact/ }));

		// A command that ran and said nothing reads as a dead tap (issue #1869).
		await waitFor(() => expect(screen.getByRole("status").textContent).toBe("compacting context…"));
		expect(field.value).toBe("");
	});

	it("carries a runtime refusal to the alert and keeps the draft", async () => {
		const api = await import("./api");
		vi.mocked(api.sendCommand).mockRejectedValueOnce(
			new api.HttpError(422, "A loop is already running"),
		);
		const field = await openSheet();
		fireEvent.change(field, { target: { value: "/goal do the thing" } });
		fireEvent.click(screen.getByRole("button", { name: "send" }));

		await waitFor(() =>
			expect(screen.getByRole("alert").textContent).toBe("A loop is already running"),
		);
		expect(field.value).toBe("/goal do the thing");
		expect(vi.mocked(api.sendCommand)).toHaveBeenCalledWith("p1", {
			op: "slash_result",
			command: "goal",
			args: "do the thing",
			images: [],
		});
	});

	it("paints a refusal neutral, not as a success (D1)", async () => {
		const api = await import("./api");
		vi.mocked(api.sendCommand).mockResolvedValueOnce({
			ok: true,
			detail: "effort is not adjustable on test/e2e-model",
			tone: "neutral",
			refused: true,
		});
		const field = await openSheet();

		fireEvent.click(screen.getByRole("button", { name: /\/compact/ }));

		await waitFor(() =>
			expect(screen.getByRole("status").textContent).toBe(
				"effort is not adjustable on test/e2e-model",
			),
		);
		const notice = screen.getByRole("status");
		// The container follows the daemon's tone: a receipt is not a success, so
		// the green wash (which the delivered-instruction ack owns) stays off it.
		expect(notice.parentElement?.className).toContain("bg-elevated");
		expect(notice.parentElement?.className).not.toContain("bg-success-wash");
		// U3: a refused command keeps the line, so one word can be corrected
		// instead of the whole command retyped.
		expect(field.value).toBe("/compact");
	});

	it("keeps the green wash for a real acknowledgement (D1, the control)", async () => {
		const api = await import("./api");
		vi.mocked(api.sendCommand).mockResolvedValueOnce({
			ok: true,
			detail: "goal set",
			tone: "success",
			refused: false,
		});
		const field = await openSheet();

		fireEvent.click(screen.getByRole("button", { name: /\/compact/ }));

		await waitFor(() => expect(screen.getByRole("status").textContent).toBe("goal set"));
		const notice = screen.getByRole("status");
		expect(notice.parentElement?.className).toContain("bg-success-wash");
		// A command that DID something clears the draft, as before.
		expect(field.value).toBe("");
	});
});
