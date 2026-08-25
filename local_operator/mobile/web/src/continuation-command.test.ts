// @vitest-environment happy-dom
import { afterEach, describe, expect, it, vi } from "vitest";
import { sendCommand } from "./api";
import { submitContinuation } from "./continuation-command";

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

	it("retains one receipt across errors so retry cannot duplicate delivery", async () => {
		mockedSendCommand
			.mockRejectedValueOnce(new Error("connection lost"))
			.mockResolvedValueOnce({ ok: true, detail: "already admitted" });

		await expect(submitContinuation("root", "steer", "Retry safely")).rejects.toThrow(
			"connection lost",
		);
		await expect(submitContinuation("root", "steer", "Retry safely")).resolves.toEqual({
			ok: true,
			detail: "already admitted",
		});
		expect(mockedSendCommand).toHaveBeenCalledTimes(2);
		expect(mockedSendCommand.mock.calls[0]?.[1]).toEqual(mockedSendCommand.mock.calls[1]?.[1]);
		expect(localStorage.getItem("lo-mobile-command:root")).toBeNull();
	});
});
