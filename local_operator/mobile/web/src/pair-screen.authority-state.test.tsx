// @vitest-environment happy-dom
//
// The code-entry pairing path, now behind "Using a terminal?" (design v2 A1).
//
// The terminal path is what a computer from before terminal-free pairing still
// supports, so it keeps its three answers about the machine's own state (UX
// round 6, U2: never promise authority the machine cannot honour) and its two
// failure arms — a revoked device (now the U15 re-key) and a relay fault (a
// sentence, never a status code; UX round 8, U8-4).
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { HttpError } from "./api";

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
	fireEvent.click(screen.getByText("Using a terminal?"));
	fireEvent.change(screen.getByPlaceholderText("Paste the code"), {
		target: { value: "code-123" },
	});
	fireEvent.click(screen.getByText("Pair with code"));
}

afterEach(() => {
	cleanup();
	localStorage.clear();
	vi.clearAllMocks();
});

describe("the terminal pairing path and the machine's own state", () => {
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
		const said = await screen.findByText(/can’t check this phone’s approvals yet/, undefined, {
			timeout: 5000,
		});
		expect(said.textContent).toContain("Settings → Phone access");
		expect(screen.queryByText("It can now approve actions on your computer.")).toBeNull();
		expect(localStorage.getItem("lo-mobile-operator-device-cert")).toBe("cert");
	});

	it("makes the promise when the machine can honour it", async () => {
		mocks.pairingStatus.mockResolvedValue({
			paired: true,
			device_id: "dev-1",
			certificate: "cert",
			operator_key_id: "kid",
			authority_ready: true,
		});
		await pairAway();
		await screen.findByText("It can now approve actions on your computer.", undefined, {
			timeout: 5000,
		});
		expect(screen.queryByText(/can’t check this phone’s approvals yet/)).toBeNull();
	});

	it("treats an older relay's missing field as ready, not as unready", async () => {
		mocks.pairingStatus.mockResolvedValue({
			paired: true,
			device_id: "dev-1",
			certificate: "cert",
			operator_key_id: "kid",
		});
		await pairAway();
		await screen.findByText("It can now approve actions on your computer.", undefined, {
			timeout: 5000,
		});
	});
});

describe("the terminal pairing path's failure copy", () => {
	it("re-keys a revoked device and says so in plain words (U15)", async () => {
		mocks.claimPairingCode.mockRejectedValueOnce(
			new HttpError(403, "this device has been revoked", "pair_device_revoked"),
		);
		localStorage.setItem("lo-mobile-operator-device-cert", "old-cert");
		localStorage.setItem("lo-mobile-operator-device-key-id", "old-kid");
		await pairAway();
		await screen.findByText(
			"This phone was removed from your computer. Ask again to pair it.",
			undefined,
			{ timeout: 5000 },
		);
		// The old certificate went with the old key.
		expect(localStorage.getItem("lo-mobile-operator-device-cert")).toBeNull();
	});

	it("turns a relay fault into a sentence rather than a status code (U8-4)", async () => {
		mocks.claimPairingCode.mockRejectedValueOnce(new Error("500"));
		await pairAway();
		const said = await screen.findByText(/Couldn’t reach your computer/, undefined, { timeout: 5000 });
		expect(said.textContent).not.toContain("500");
	});
});
