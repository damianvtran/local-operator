// @vitest-environment happy-dom
import { afterEach, describe, expect, it, vi } from "vitest";
import { sendCommand } from "./api";
import { getPendingContinuation, submitContinuation } from "./continuation-command";

vi.mock("./api", () => ({ sendCommand: vi.fn() }));

const mockedSendCommand = vi.mocked(sendCommand);

// The browser API is the producer of protocol identity in production. Pinning
// it here keeps retry assertions deterministic without replacing the shared
// submission path itself.
vi.stubGlobal("crypto", { randomUUID: vi.fn(() => "12345678-1234-4678-9234-567812345678") });

afterEach(() => {
	localStorage.clear();
	mockedSendCommand.mockReset();
});

describe("submitContinuation", () => {
	it("submits steer with a UUID and retires the receipt after acknowledgement", async () => {
		mockedSendCommand.mockResolvedValue({ ok: true, detail: "steering queued" });

		await expect(submitContinuation("root", "steer", "Reach the parent once")).resolves.toEqual({
			ok: true,
			detail: "steering queued",
			envelope: {
				op: "steer",
				command_id: "12345678-1234-4678-9234-567812345678",
				text: "Reach the parent once",
				images: undefined,
			},
		});
		expect(mockedSendCommand).toHaveBeenCalledOnce();
		expect(mockedSendCommand).toHaveBeenCalledWith("root", {
			op: "steer",
			command_id: "12345678-1234-4678-9234-567812345678",
			text: "Reach the parent once",
			images: undefined,
		});
		expect(localStorage.getItem("lo-mobile-command:root")).toBeNull();
	});

	it("retries the admitted envelope even after the caller edits its draft", async () => {
		mockedSendCommand
			.mockRejectedValueOnce(new Error("response lost"))
			.mockResolvedValueOnce({ ok: true, detail: "already admitted" })
			.mockResolvedValueOnce({ ok: true, detail: "steering queued" });

		await expect(submitContinuation("root", "steer", "Original instruction")).rejects.toThrow(
			"response lost",
		);
		const retry = await submitContinuation("root", "prompt", "Edited draft", [
			{ data_b64: "edited-image", mime_type: "image/png" },
		]);
		expect(retry.detail).toBe("already admitted");
		expect(retry.envelope).toEqual(mockedSendCommand.mock.calls[0]?.[1]);
		expect(mockedSendCommand.mock.calls[1]?.[1]).toEqual(mockedSendCommand.mock.calls[0]?.[1]);

		await submitContinuation("root", "prompt", "Edited draft", [
			{ data_b64: "edited-image", mime_type: "image/png" },
		]);
		expect(mockedSendCommand.mock.calls[2]?.[1]).toEqual({
			op: "prompt",
			command_id: "12345678-1234-4678-9234-567812345678",
			text: "Edited draft",
			images: [{ data_b64: "edited-image", mime_type: "image/png" }],
		});
		expect(mockedSendCommand.mock.calls[2]?.[1]).not.toBe(mockedSendCommand.mock.calls[1]?.[1]);
		expect(localStorage.getItem("lo-mobile-command:root")).toBeNull();
	});

	it("recovers a validated envelope after reload and expires unsafe state", async () => {
		mockedSendCommand.mockRejectedValueOnce(new TypeError("response lost"));
		await expect(submitContinuation("root", "prompt", "Original instruction")).rejects.toThrow(
			"response lost",
		);
		expect(getPendingContinuation("root")).toMatchObject({
			op: "prompt",
			text: "Original instruction",
		});

		const raw = localStorage.getItem("lo-mobile-command:root")!;
		const stored = JSON.parse(raw) as { saved_at: number };
		stored.saved_at = Date.now() - 25 * 60 * 60 * 1000;
		localStorage.setItem("lo-mobile-command:root", JSON.stringify(stored));
		/* TTL expiry (24h) is a definitive clear — read-time revalidation drops it. */
		expect(getPendingContinuation("root")).toBeNull();
	});

	it("retires a definitively rejected UUID instead of replaying it", async () => {
		mockedSendCommand.mockRejectedValueOnce(
			Object.assign(new Error("invalid command"), { status: 422 }),
		);
		await expect(submitContinuation("root", "prompt", "Rejected instruction")).rejects.toThrow(
			"invalid command",
		);
		expect(getPendingContinuation("root")).toBeNull();
	});

	it.each([408, 502, 504])(
		"keeps the envelope on ambiguous acknowledgement-loss status %i",
		async (status) => {
			/* 408/502/504 can arrive AFTER the daemon admitted the UUID, so the
			   envelope must survive for a same-UUID retry rather than being retired
			   like a definitive rejection (R9-F2). */
			mockedSendCommand.mockRejectedValueOnce(
				Object.assign(new Error("acknowledgement lost"), { status }),
			);
			await expect(
				submitContinuation("root", "steer", "Ambiguous instruction"),
			).rejects.toThrow("acknowledgement lost");
			expect(getPendingContinuation("root")).toMatchObject({
				op: "steer",
				text: "Ambiguous instruction",
			});

			/* The retry replays the ORIGINAL UUID/body, so an already-admitted
			   instruction is de-duplicated instead of sent twice. */
			mockedSendCommand.mockResolvedValueOnce({ ok: true, detail: "already admitted" });
			const retry = await submitContinuation("root", "steer", "Ambiguous instruction");
			expect(retry.envelope).toEqual(mockedSendCommand.mock.calls[0]?.[1]);
			expect(getPendingContinuation("root")).toBeNull();
		},
	);

	it("scopes envelopes per session and evicts the oldest past the bound", async () => {
		/* Navigating between my own conversations must never drop another route's
		   uncertain envelope (U1): each session key is independent. Storage is
		   bounded by count, evicting the OLDEST, not the route just written. */
		let clock = 1_000;
		const now = vi.spyOn(Date, "now").mockImplementation(() => clock);
		try {
			mockedSendCommand.mockRejectedValue(new Error("response lost"));
			for (let index = 0; index < 8; index++) {
				clock += 1_000;
				await expect(
					submitContinuation(`session-${index}`, "prompt", `Instruction ${index}`),
				).rejects.toThrow("response lost");
			}
			/* All eight distinct sessions retain their own envelope. */
			for (let index = 0; index < 8; index++) {
				expect(getPendingContinuation(`session-${index}`)).not.toBeNull();
			}
			/* A ninth distinct session evicts the OLDEST (session-0), never itself. */
			clock += 1_000;
			await expect(
				submitContinuation("session-8", "prompt", "Instruction 8"),
			).rejects.toThrow("response lost");
			expect(getPendingContinuation("session-0")).toBeNull();
			expect(getPendingContinuation("session-8")).not.toBeNull();
			expect(getPendingContinuation("session-1")).not.toBeNull();
		} finally {
			now.mockRestore();
		}
	});
});
