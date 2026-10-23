// @vitest-environment happy-dom
//
// UX round 6, D2/D4 and U1/U6 — the phone's gate sheet, on the real render site.
//
// D2: a successful loosen set its notice INSIDE the Sheet and closed in the same
// commit, so the receipt was set and unmounted before anything painted. The fix
// hands it out (`onReceipt`) to the surface the sheet closes back onto, and this
// test asserts the receipt is on screen AFTER the panel is gone — which is the
// whole finding, and the shape a unit test on the sheet alone cannot see.
//
// D4: the unpaired case says what it will refuse BEFORE the tap.
//
// U1/U6: the refusal is keyed on the typed code and passes the runtime's own copy
// through, so the install step the copy names is not reworded here.
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { HttpError, isAuthorityRefusalCode } from "./api";
import { humanizeGateError } from "./components/gate-sheet";
import { isAuthorityRefusal } from "./components/pending-card";
import { SessionScreen } from "./screens/session-view";
import type { SessionProjection } from "./types";

/* The receipt shape these mocks answer with, typed here rather than inferred:
   a zero-argument `vi.fn` gives `mock.calls` an empty tuple, so the assertions
   that read the FRAME (the whole point — scope plus proof fields) would not
   typecheck. `refused` is the routed op's own field. */
type Receipt = { ok: boolean; detail: string; refused?: boolean };
type SignedFrame = Record<string, unknown>;

const mocks = vi.hoisted(() => ({
	sendCommandWithProof: vi.fn(
		async (_sessionId: string, _op: SignedFrame): Promise<Receipt> => ({ ok: true, detail: "" }),
	),
	requestOperatorChallenge: vi.fn(async () => ({
		challenge: "ab",
		expires_s: 30,
		session_id: "s1",
		action: "loosen",
		request_id: "",
	})),
	operatorFieldsFor: vi.fn(async () => ({ operator_sig: "aa", operator_key_id: "bb", operator_cert: "cc" })),
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
		markSessionSeen: vi.fn(async () => ({ ok: true })),
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
		/* Returns the refcount unsubscribe, as the real one does, so a test that
		   unmounts exercises the effect's cleanup path rather than a no-op
		   (review round 2, N2-1). */
		retainSessionListStream: vi.fn(() => () => {}),
		useDraft: vi.fn(() => ["", () => {}]),
	};
});

/** The projection shape the REAL SessionScreen tree reads.
    Copied from the sibling render test rather than trimmed: the Composer reads
    `effort_ladder.length` and friends, and a partial fixture fails inside a
    component rather than in the assertion under test. */
function projection(): SessionProjection {
	return {
		session_id: "s1",
		pid: 1,
		kind: "tui",
		conversation_name: "gate receipt",
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
	} satisfies SessionProjection;
}

async function openSheet() {
	render(<SessionScreen sessionId="s1" />);
	await waitFor(() => expect(screen.getByRole("button", { name: "session menu" })).toBeTruthy());
	fireEvent.click(screen.getByRole("button", { name: "session menu" }));
	fireEvent.click(await screen.findByRole("button", { name: /Approvals/ }));
	await waitFor(() => expect(screen.getByRole("dialog")).toBeTruthy());
}

beforeEach(() => {
	slot = { projection: projection(), connected: true };
	mocks.sendCommandWithProof.mockClear();
	mocks.requestOperatorChallenge.mockClear();
	mocks.operatorFieldsFor.mockClear();
	mocks.storedCertificate.mockReset();
	mocks.storedCertificate.mockReturnValue({ certificate: "cert", keyId: "kid" });
});
afterEach(cleanup);

describe("the phone's gate sheet", () => {
	it("keeps a successful loosen's receipt after the sheet has gone (D2)", async () => {
		await openSheet();
		fireEvent.click(screen.getByText("run without asking (auto)"));
		await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
		// THE ASSERTION THE DEFECT FAILS: the receipt is still on screen, and it is
		// on the surface the sheet closed back onto rather than inside the sheet.
		await waitFor(() => expect(screen.getByRole("status").textContent).toContain("now auto"));
		expect(mocks.sendCommandWithProof).toHaveBeenCalledTimes(1);
	});

	it("clears the receipt when a tighten leaves the gate elsewhere (U8-3)", async () => {
		/* The receipt reports a STATE ("this session's gate is now auto"), and it
		   lives on the header, above the sheet. A `keep asking` from that same sheet
		   is one tap from the receipt's own control, and it used to leave the header
		   asserting a gate the session no longer has — measured as one frame carrying
		   the runtime's refusal and a header claiming the opposite. The round-7
		   comment justified the persistence with "nothing stale to clear"; this is
		   the flow that made that false, so the assertion is on the state, not the
		   sentence. */
		await openSheet();
		fireEvent.click(screen.getByText("run without asking (auto)"));
		await waitFor(() => expect(screen.getByRole("status").textContent).toContain("now auto"));
		fireEvent.click(screen.getByRole("button", { name: "session menu" }));
		fireEvent.click(await screen.findByRole("button", { name: /Approvals/ }));
		await waitFor(() => expect(screen.getByRole("dialog")).toBeTruthy());
		fireEvent.click(screen.getByText("keep asking (ask)"));
		await waitFor(() => expect(screen.queryByRole("status")).toBeNull());
		expect(mocks.sendCommandWithProof).toHaveBeenCalledTimes(2);
	});

	it("says what it will refuse before the tap on an unpaired phone (D4)", async () => {
		mocks.storedCertificate.mockReturnValue(null);
		await openSheet();
		const said = screen.getByText(/will ask you to pair it first/);
		expect(said.textContent).toContain("Keeping approvals at");
		/* BOTH authority-increasing choices are named: the global scope is the one
		   a reader is most likely to assume is exempt because it is "only a
		   default", and it is not. */
		expect(said.textContent).toContain("run without asking in all sessions");
		// ...and it was there BEFORE the tap: no request had been made yet.
		expect(mocks.sendCommandWithProof).not.toHaveBeenCalled();
	});

	it("shows the runtime's own unconfigured copy, setup action and all (U1)", async () => {
		mocks.sendCommandWithProof.mockRejectedValueOnce(
			new HttpError(
				422,
				"this session's gate is still at ask: /approvals auto removes it and needs the operator's own consent — but operator authority is not installed on this machine, so the remedies below cannot work yet. Ask Local Operator to set up operator authority first (one approval and one admin password prompt), then authorise from this machine or your paired phone. /approvals ask still tightens it here.",
				"operator_authority_unconfigured",
			),
		);
		await openSheet();
		fireEvent.click(screen.getByText("run without asking (auto)"));
		const shown = await screen.findByText(/remedies below cannot work yet/);
		expect(shown.textContent).toContain("one approval and one admin password prompt");
		expect(screen.getByRole("dialog")).toBeTruthy();
	});

	it("sends the global default with the same loosen proof, and says what it applied", async () => {
		/* THE ESCAPE HATCH'S WIDER SCOPE. `default auto` changes every session
		   that follows the default, so it must carry the same per-action proof
		   `auto` does — a signature minted for one scope cannot be presented as
		   the other, so the assertion is on the FRAME (scope + proof fields) and
		   on the challenge action, not merely on a call having happened. */
		await openSheet();
		fireEvent.click(screen.getByText("run without asking in all sessions (default auto)"));
		await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
		expect(mocks.sendCommandWithProof).toHaveBeenCalledTimes(1);
		expect(mocks.sendCommandWithProof.mock.calls[0][1]).toMatchObject({
			op: "slash_result",
			command: "approvals",
			args: "default auto",
			operator_sig: "aa",
			operator_key_id: "bb",
			operator_cert: "cc",
		});
		expect(mocks.requestOperatorChallenge).toHaveBeenCalledWith("s1", {
			action: "loosen",
			request_id: "",
		});
		/* The receipt survives the sheet — and states the DEFAULT, which is the
		   scope this write actually reaches. "all sessions are now auto" would be
		   a claim the phone cannot make: a session with its own override is not
		   changed by a default write. */
		await waitFor(() =>
			expect(screen.getByRole("status").textContent).toContain("the default is now auto"),
		);
		expect(screen.getByRole("status").textContent).not.toContain("all sessions are now");
	});

	it("prefers the runtime's account of a global write over the phone's", async () => {
		/* "Do not claim all sessions changed if the runtime receipt says
		   otherwise" — the runtime counts the set, the phone cannot, so its prose
		   passes through unchanged when it has any. */
		mocks.sendCommandWithProof.mockResolvedValueOnce({
			ok: true,
			detail: "the default is now auto (2 of 4 sessions follow it)",
		});
		await openSheet();
		fireEvent.click(screen.getByText("run without asking in all sessions (default auto)"));
		await waitFor(() =>
			expect(screen.getByRole("status").textContent).toContain("2 of 4 sessions follow it"),
		);
	});

	it("keeps a completed-but-refused global write in the sheet", async () => {
		/* The routed op can report `refused` beside `ok`. Closing on that would put
		   a header receipt on screen describing a change the runtime says did not
		   happen; the refusal belongs in the sheet, in the runtime's words. */
		mocks.sendCommandWithProof.mockResolvedValueOnce({
			ok: true,
			detail: "the global default cannot be set from this surface",
			refused: true,
		});
		await openSheet();
		fireEvent.click(screen.getByText("run without asking in all sessions (default auto)"));
		expect(await screen.findByText(/cannot be set from this surface/)).toBeTruthy();
		expect(screen.getByRole("dialog")).toBeTruthy();
		expect(screen.queryByRole("status")).toBeNull();
	});
});

describe("the refusal helpers", () => {
	it("keys the card's re-sign decision on the typed code (U6)", () => {
		expect(isAuthorityRefusal(new HttpError(422, "anything at all", "operator_authority_required"))).toBe(true);
		expect(isAuthorityRefusal(new HttpError(422, "anything at all", "operator_authority_unconfigured"))).toBe(true);
		expect(isAuthorityRefusal(new HttpError(422, "anything at all", "run_capped"))).toBe(false);
		// An older relay carries no code, so the copy is the fallback — and only then.
		expect(isAuthorityRefusal(new Error("this approval is still waiting: only the operator can allow it"))).toBe(true);
		expect(isAuthorityRefusal(new Error("the session went away"))).toBe(false);
	});

	it("adds the step that is left only where this surface can know it (D5)", () => {
		const installed = humanizeGateError(
			new HttpError(422, "authorise it from this machine (Touch ID) or from your paired phone.", "operator_authority_required"),
		);
		expect(installed).toContain("lop operator devices");
		const unconfigured = humanizeGateError(
			new HttpError(422, "…set it up for this machine first…", "operator_authority_unconfigured"),
		);
		expect(unconfigured).toBe("…set it up for this machine first…");
	});

	it("knows the two authority codes and nothing else", () => {
		expect(isAuthorityRefusalCode("operator_authority_required")).toBe(true);
		expect(isAuthorityRefusalCode("operator_authority_unconfigured")).toBe(true);
		expect(isAuthorityRefusalCode("")).toBe(false);
		expect(isAuthorityRefusalCode("session_not_connected")).toBe(false);
	});
});
