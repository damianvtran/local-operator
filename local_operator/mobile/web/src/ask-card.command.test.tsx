// @vitest-environment happy-dom
//
// The ask card's suggested terminal command (`AskQuestion.command`): a
// monospace block with a Copy button that copies the EXACT bytes and confirms
// with "Copied". The phone offers nothing about running it — the desktop owns
// the confirmed Insert/Run flow — and a question without a command draws no
// block at all.
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AskCard } from "./components/ask-card";
import type { AskQuestion, PendingAsk } from "./types";

vi.mock("./api", async (importOriginal) => {
	const actual = await importOriginal<typeof import("./api")>();
	return { ...actual, sendCommand: vi.fn(async () => ({ ok: true, detail: "answered" })) };
});

function question(patch: Partial<AskQuestion> = {}): AskQuestion {
	return {
		id: "q1",
		question: "Sign in to GitLab first?",
		options: [
			{ label: "Done", description: "" },
			{ label: "Skip", description: "" },
		],
		multi: false,
		secret: false,
		persist: false,
		...patch,
	};
}

function ask(questions: AskQuestion[]): PendingAsk {
	return {
		ask_id: "ask-1",
		created_at: 1,
		expires_at: Date.now() + 900_000,
		timeout_s: 900,
		urgent: false,
		status: "open",
		delivered: false,
		questions,
	};
}

function renderCard(row: PendingAsk) {
	return render(<AskCard row={row} sessionId="s1" nowMs={Date.now()} />);
}

function stubClipboard(writeText: (text: string) => Promise<void>) {
	Object.defineProperty(navigator, "clipboard", {
		configurable: true,
		value: { writeText: vi.fn(writeText) },
	});
	return navigator.clipboard.writeText as ReturnType<typeof vi.fn>;
}

afterEach(() => {
	cleanup();
	vi.useRealTimers();
	vi.clearAllMocks();
});

describe("AskCard suggested command", () => {
	it("draws no command block for a question without one", () => {
		renderCard(ask([question()]));
		expect(screen.queryByTestId("ask-command")).toBeNull();
	});

	it("shows the command in a labelled LTR monospace block with no run control", () => {
		renderCard(ask([question({ command: "glab auth login", command_target: "agent" })]));
		const group = screen.getByRole("group", { name: "Suggested command" });
		const text = screen.getByTestId("ask-command-text");
		expect(text.textContent).toBe("glab auth login");
		expect(text.getAttribute("dir")).toBe("ltr");
		expect(text.className).toContain("font-mono");
		// Copy is the ONLY action: nothing offers to run or insert it.
		const buttons = Array.from(group.querySelectorAll("button")).map((b) => b.textContent);
		expect(buttons).toEqual(["Copy"]);
		expect(screen.queryByRole("button", { name: /run|insert/i })).toBeNull();
	});

	it("copies the exact bytes and confirms with Copied, then reverts", async () => {
		vi.useFakeTimers();
		const writeText = stubClipboard(async () => undefined);
		renderCard(ask([question({ command: "cd ~/src && make test", command_target: "user" })]));
		await act(async () => {
			fireEvent.click(screen.getByTestId("ask-command-copy"));
		});
		expect(writeText).toHaveBeenCalledWith("cd ~/src && make test");
		expect(screen.getByTestId("ask-command-copy").textContent).toBe("Copied");
		await act(async () => {
			vi.advanceTimersByTime(2_000);
		});
		expect(screen.getByTestId("ask-command-copy").textContent).toBe("Copy");
	});

	it("selects the command and says so when the clipboard refuses", async () => {
		stubClipboard(async () => {
			throw new Error("NotAllowedError");
		});
		renderCard(ask([question({ command: "gh auth status" })]));
		await act(async () => {
			fireEvent.click(screen.getByTestId("ask-command-copy"));
		});
		expect(screen.getByTestId("ask-command-copy").textContent).toBe("Copy");
		expect(screen.getByText(/Couldn't copy/)).toBeTruthy();
		expect(window.getSelection()?.toString()).toBe("gh auth status");
	});
});
