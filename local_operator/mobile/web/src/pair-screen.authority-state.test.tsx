// @vitest-environment happy-dom
//
// UX round 6 U2, tested where the reader is (agent review round 7, M-1).
//
// The round-6 remediation comment claimed `pair.tsx`'s branch "is covered by the
// portal suite". It was not: no portal test rendered `PairScreen` at all, and the
// reviewer proved it by grepping. The claim is the thing this round exists to
// remove — evidence outrunning the suite — so the cell now exists, and it drives
// the three states a relay can answer with.
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
	claimPairingCode: vi.fn(async () => ({ ok: true })),
	pairingStatus: vi.fn(),
}));

vi.mock("./api", async (importOriginal) => {
	const actual = await importOriginal<typeof import("./api")>();
	return {
		...actual,
		claimPairingCode: mocks.claimPairingCode,
		pairingStatus: mocks.pairingStatus,
	};
});

import { PairScreen } from "./screens/pair";

async function pairAway(): Promise<void> {
	render(<PairScreen />);
	fireEvent.change(screen.getByPlaceholderText("paste the code"), {
		target: { value: "code-123" },
	});
	fireEvent.click(screen.getByText("pair"));
}

afterEach(() => {
	cleanup();
	localStorage.clear();
	vi.clearAllMocks();
});

describe("the pairing screen and the machine's own state", () => {
	it("does not promise authority the machine cannot honour (U2)", async () => {
		mocks.pairingStatus.mockResolvedValue({
			paired: true,
			device_id: "dev-1",
			certificate: "cert",
			operator_key_id: "kid",
			name: "my phone",
			authority_ready: false,
		});
		await pairAway();
		// Found by the SENTENCE rather than by a prominent inner element: a phrase
		// query would answer with the innermost match and would leave the paragraph's
		// own wording unasserted.
		const said = await screen.findByText(/can sign already/, undefined, { timeout: 5000 });
		expect(said.textContent).toContain("Ask Local Operator on that machine to set up operator authority");
		expect(said.textContent).toContain("cannot check it yet");
		expect(screen.queryByText(/can now approve parked tool calls/)).toBeNull();
	});

	it("makes the promise when the machine can honour it", async () => {
		mocks.pairingStatus.mockResolvedValue({
			paired: true,
			device_id: "dev-1",
			certificate: "cert",
			operator_key_id: "kid",
			name: "my phone",
			authority_ready: true,
		});
		await pairAway();
		const said = await screen.findByText(/can now approve parked tool calls/, undefined, {
			timeout: 5000,
		});
		expect(said.textContent).toContain("loosen");
		expect(screen.queryByText(/set up operator authority on that machine/)).toBeNull();
	});

	it("treats an older relay's missing field as ready, not as unready", async () => {
		mocks.pairingStatus.mockResolvedValue({
			paired: true,
			device_id: "dev-1",
			certificate: "cert",
			operator_key_id: "kid",
			name: "my phone",
		});
		await pairAway();
		await waitFor(
			() => expect(screen.getByText(/can now approve parked tool calls/)).toBeTruthy(),
			{ timeout: 5000 },
		);
	});
});

describe("the pairing screen's failure copy", () => {
	it("routes a revoked device back to the machine through the agent route (U1, D9)", async () => {
		/* The route, in the states it has been through: "Pair it again from
		   `lop pair`" (the step the machine refuses for ever — UX round 8, U8-1), then
		   `lop operator init` + `install` (which the local revocation record defeated —
		   round 9's Q9-1/R9-2, measured 403 either way), then the inverse verb carrying
		   this phone's own id (U1). Design round 2 (D9) then removed the terminal
		   COMMAND from it: a phone reader cannot run one, so the sentence now names the
		   AGENT route the rest of the product names for the same step (§2.9), keeping
		   the id so the reader can hand it over. */
		mocks.claimPairingCode.mockRejectedValueOnce(
			new Error("this device has been revoked"),
		);
		await pairAway();
		const said = await screen.findByText(/revoked on the machine/, undefined, { timeout: 5000 });
		expect(said.textContent).toContain("only there");

		/* D9: no command on a phone surface — the remedy names the action and the id. */
		expect(screen.queryByText(/lop operator devices --authorise/)).toBeNull();
		expect(screen.getByText(/allow this phone again/)).toBeTruthy();
		expect(screen.getByText(/the device id is/)).toBeTruthy();
		expect(screen.queryByText(/this phone's device id/)).toBeNull();
		expect(screen.queryByText(/Pair it again/)).toBeNull();
		expect(screen.queryByText(/create a new operator anchor/)).toBeNull();

		/* D2/D9: the machine-side condition stays its own sentence. */
		expect(screen.getByText(/ask it to set that up first/)).toBeTruthy();
	});

	it("turns a relay fault into a sentence rather than a status code (U8-4)", async () => {
		mocks.claimPairingCode.mockRejectedValueOnce(new Error("500"));
		await pairAway();
		const said = await screen.findByText(/could not answer/, undefined, { timeout: 5000 });
		expect(said.textContent).toContain("lop serve");
	});
});
