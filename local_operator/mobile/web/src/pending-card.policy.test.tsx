// @vitest-environment happy-dom
//
// THE CARD'S ESCAPE HATCH (approvals on the phone, global-default stage).
//
// The pending card offers two mode links beneath approve/deny. The invariant this
// file exists for is the one a careful reader would doubt: a mode change is NOT an
// answer. `/approvals auto` loosens the gate for FUTURE calls; the request in
// front of the reader is still pending and must still be answered. A regression
// here does not fail loudly — it silently settles a request the user never
// answered, or leaves the card looking handled while the session stays blocked.
// So the assertions are on the absence of an `approval_answer` frame, on the card
// still being answerable, and on the inline sentence that pairs the mode change
// with "this call still needs your answer".
//
// Rendered through the REAL SessionScreen (the render site that keys the card and
// hands it the pid), not PendingCard in isolation.
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { HttpError } from "./api";
import { SessionScreen } from "./screens/session-view";
import type { PendingRequest, SessionProjection } from "./types";

/* Typed so `mock.calls` carries the FRAME the assertions read. A zero-argument
   `vi.fn` infers an empty tuple and the `calls[0][1]` match fails to typecheck;
   `refused` is the routed op's own field on a receipt. */
type Receipt = { ok: boolean; detail: string; refused?: boolean };
type Frame = Record<string, unknown>;

const mocks = vi.hoisted(() => ({
	sendCommandWithProof: vi.fn(
		async (_sessionId: string, _op: Frame): Promise<Receipt> => ({ ok: true, detail: "" }),
	),
	sendCommand: vi.fn(async (_sessionId: string, _op: Frame) => ({
		ok: true,
		detail: "answer accepted",
	})),
	requestOperatorChallenge: vi.fn(async () => ({
		challenge: "ab",
		expires_s: 30,
		session_id: "s1",
		action: "loosen",
		request_id: "",
	})),
	operatorFieldsFor: vi.fn(async () => ({
		operator_sig: "aa",
		operator_key_id: "bb",
		operator_cert: "cc",
	})),
	storedCertificate: vi.fn(
		(): { certificate: string; keyId: string } | null => ({ certificate: "cert", keyId: "kid" }),
	),
}));

vi.mock("./api", async (importOriginal) => {
	const actual = await importOriginal<typeof import("./api")>();
	return {
		...actual,
		getHistory: vi.fn(async () => ({ entries: [], has_more: false })),
		getSubagentHistory: vi.fn(async () => ({ entries: [], has_more: false })),
		getSubagentDetail: vi.fn(async () => null),
		imageUrl: vi.fn(() => ""),
		getCommands: vi.fn(async () => ({ commands: [] })),
		getModels: vi.fn(async () => ({ models: [] })),
		getDirectories: vi.fn(async () => ({ home: "/Users/tester", recent: [], tmp: "" })),
		changeDirectory: vi.fn(async () => ({ ok: true, pid: 1, session_id: "s1" })),
		markSessionSeen: vi.fn(async () => ({ ok: true })),
		sendCommand: mocks.sendCommand,
		sendCommandWithProof: mocks.sendCommandWithProof,
		requestOperatorChallenge: mocks.requestOperatorChallenge,
	};
});

vi.mock("./lib/operator-device", async (importOriginal) => {
	const actual = await importOriginal<typeof import("./lib/operator-device")>();
	return {
		...actual,
		operatorFieldsFor: mocks.operatorFieldsFor,
		storedCertificate: mocks.storedCertificate,
	};
});

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

function approval(overrides: Partial<PendingRequest> = {}): PendingRequest {
	return {
		request_id: "req-approve",
		kind: "approval",
		title: "run shell command",
		detail: "rm -rf build && make",
		options: [],
		secret: false,
		question_index: 0,
		question_total: 1,
		...overrides,
	};
}

function projection(pending: PendingRequest): SessionProjection {
	return {
		session_id: "s1",
		pid: 1,
		kind: "tui",
		conversation_name: "Approvals",
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
		pending,
		pending_count: 1,
		usage: {},
		version: 1,
	} satisfies SessionProjection;
}

/** Every frame this run sent that ANSWERS a request. Empty is the invariant. */
function answerFrames(): unknown[] {
	return mocks.sendCommand.mock.calls
		.map((call) => call[1])
		.filter((op) => op?.op === "approval_answer" || op?.op === "ask_answer");
}

function renderCard(pending: PendingRequest = approval()) {
	slot = { projection: projection(pending), connected: true };
	render(<SessionScreen sessionId="s1" />);
	return screen.findByRole("button", { name: "approve" });
}

beforeEach(() => {
	mocks.sendCommandWithProof.mockClear();
	mocks.sendCommand.mockClear();
	mocks.requestOperatorChallenge.mockClear();
	mocks.operatorFieldsFor.mockClear();
	mocks.storedCertificate.mockReset();
	mocks.storedCertificate.mockReturnValue({ certificate: "cert", keyId: "kid" });
});

afterEach(() => {
	cleanup();
	localStorage.clear();
	slot = { projection: null, connected: true };
});

describe("the pending card's approval-policy links", () => {
	it("loosens this session without answering the pending approval", async () => {
		await renderCard();
		fireEvent.click(screen.getByRole("button", { name: "Set auto for this session" }));
		await waitFor(() => expect(mocks.sendCommandWithProof).toHaveBeenCalledTimes(1));
		expect(mocks.sendCommandWithProof.mock.calls[0][1]).toMatchObject({
			op: "slash_result",
			command: "approvals",
			args: "auto",
			operator_sig: "aa",
			operator_key_id: "bb",
			operator_cert: "cc",
		});
		// THE WHOLE POINT: a policy write is not an answer.
		expect(answerFrames()).toHaveLength(0);
		// ...the card is still answerable...
		const approve = screen.getByRole("button", { name: "approve" });
		expect((approve as HTMLButtonElement).disabled).toBe(false);
		// ...and the inline line says the mode changed AND the call did not.
		const said = await screen.findByRole("status");
		expect(said.textContent).toContain("this session is now auto");
		expect(said.textContent).toContain("This call still needs your answer.");
	});

	it("sends the global default, proven, from the card too", async () => {
		await renderCard();
		fireEvent.click(screen.getByRole("button", { name: "Set auto for all sessions" }));
		await waitFor(() => expect(mocks.sendCommandWithProof).toHaveBeenCalledTimes(1));
		expect(mocks.sendCommandWithProof.mock.calls[0][1]).toMatchObject({
			op: "slash_result",
			command: "approvals",
			args: "default auto",
			operator_sig: "aa",
		});
		// The same `loosen` challenge the gate sheet mints — one proof protocol,
		// not a second one shaped for this surface.
		expect(mocks.requestOperatorChallenge).toHaveBeenCalledWith("s1", {
			action: "loosen",
			request_id: "",
		});
		expect(answerFrames()).toHaveLength(0);
		const said = await screen.findByRole("status");
		/* m4: the fallback names the DEFAULT, never "all sessions" — a conversation
		   with its own choice keeps it. */
		expect(said.textContent).toContain("the default for sessions that follow it is now auto");
		expect(said.textContent).not.toContain("all sessions");
		expect(said.textContent).toContain("This call still needs your answer.");
	});

	it("shows the runtime's own exception in the confirmation (U1)", async () => {
		/* The defect this replaces: the card printed a fixed "auto for all
		   sessions" sentence and discarded the runtime's account — including the
		   one case that matters, a conversation whose own Ask keeps Ask. */
		mocks.sendCommandWithProof.mockResolvedValueOnce({
			ok: true,
			detail:
				"tool approvals: auto — saved to /Users/tester/.config/config.yml " +
				"(tool_approval_mode); other conversations with their own choice keep it",
		});
		await renderCard();
		fireEvent.click(screen.getByRole("button", { name: "Set auto for all sessions" }));
		const said = await screen.findByRole("status");
		expect(said.textContent).toContain("other conversations with their own choice keep it");
		expect(said.textContent).toContain("This call still needs your answer.");
	});

	it("confirms nothing when the runtime reports the write refused", async () => {
		mocks.sendCommandWithProof.mockResolvedValueOnce({
			ok: true,
			detail: "not on this build — this surface cannot set the default",
			refused: true,
		});
		await renderCard();
		fireEvent.click(screen.getByRole("button", { name: "Set auto for this session" }));
		const refusal = await screen.findByRole("alert");
		expect(refusal.textContent).toContain("cannot set the default");
		expect(screen.queryByRole("status")).toBeNull();
		expect((screen.getByRole("button", { name: "approve" }) as HTMLButtonElement).disabled).toBe(
			false,
		);
	});

	it("leaves the card answerable when the policy write is refused", async () => {
		await renderCard();
		mocks.sendCommandWithProof.mockRejectedValueOnce(
			new HttpError(422, "only the operator can allow it", "operator_authority_required"),
		);
		fireEvent.click(screen.getByRole("button", { name: "Set auto for this session" }));
		// The runtime's own sentence, inline — and NOT the card-wide `error`,
		// which would make every control inert (D7) on a request still waiting
		// for its decision.
		const refusal = await screen.findByRole("alert");
		expect(refusal.textContent).toContain("only the operator can allow it");
		const approve = screen.getByRole("button", { name: "approve" }) as HTMLButtonElement;
		expect(approve.disabled).toBe(false);
		const deny = screen.getByRole("button", { name: "deny" }) as HTMLButtonElement;
		expect(deny.disabled).toBe(false);
		expect(answerFrames()).toHaveLength(0);
	});

	it("does not offer the mode links on an ask card", async () => {
		// A mode change is about the approval gate; an ask (a choice or a
		// credential) has no gate to loosen, and offering it there would imply
		// the question could be skipped.
		slot = {
			projection: projection({
				request_id: "req-ask",
				kind: "ask",
				title: "which colour?",
				detail: "",
				options: [{ label: "blue", description: "" }],
				secret: false,
				question_index: 0,
				question_total: 1,
			}),
			connected: true,
		};
		render(<SessionScreen sessionId="s1" />);
		await screen.findByRole("button", { name: /blue/ });
		expect(screen.queryByRole("button", { name: "Set auto for this session" })).toBeNull();
		expect(screen.queryByRole("button", { name: "Set auto for all sessions" })).toBeNull();
	});
});

describe("the pending card's effective-mode line", () => {
	it("says a session override and the global it overrides", async () => {
		await renderCard(approval({ approval_mode: "ask", approval_provenance: "session" }));
		expect(screen.getByText("Approvals: Ask — session override; global Auto.")).toBeTruthy();
	});

	it("mirrors the session override in the auto direction", async () => {
		await renderCard(approval({ approval_mode: "auto", approval_provenance: "session" }));
		expect(screen.getByText("Approvals: Auto — session override; global Ask.")).toBeTruthy();
	});

	it("says a global default when that is what is in force", async () => {
		await renderCard(approval({ approval_mode: "ask", approval_provenance: "global" }));
		expect(screen.getByText("Approvals: Ask — global default.")).toBeTruthy();
	});

	it("separates a run pin from both saved scopes", async () => {
		await renderCard(approval({ approval_mode: "auto", approval_provenance: "pin" }));
		expect(screen.getByText("Approvals: Auto — this run.")).toBeTruthy();
	});

	it("renders no line when the owner publishes neither field", async () => {
		/* An older runtime owns the session: the card must not guess "global
		   default" — that is the one claim this line exists to stop making. */
		await renderCard(approval({ approval_mode: "", approval_provenance: "" }));
		expect(screen.queryByText(/^Approvals:/)).toBeNull();
	});
});
