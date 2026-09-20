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
		// Found by the SENTENCE rather than by the `<code>` inside it: the default
		// matcher returns the innermost element whose text matches, so a query on
		// "lop operator install" answers with the code element and would leave the
		// paragraph's own wording unasserted.
		const said = await screen.findByText(/can sign already/, undefined, { timeout: 5000 });
		expect(said.textContent).toContain("lop operator install");
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
		expect(screen.queryByText(/lop operator install/)).toBeNull();
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
