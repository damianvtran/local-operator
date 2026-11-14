// @vitest-environment happy-dom
//
// The ask card's suggested terminal command (`AskQuestion.command`): a
// monospace block with a Copy button that copies the EXACT bytes and confirms
// with "Copied". The phone offers nothing about running it — the desktop owns
// the confirmed Insert/Run flow — and a question without a command draws no
// block at all.
import { act, cleanup, fireEvent, render, screen, within } from "@testing-library/react";
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
		const buttons = Array.from(group.querySelectorAll("button"));
		expect(buttons).toHaveLength(1);
		expect(screen.getByRole("button", { name: "Copy" })).toBe(buttons[0]);
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
		expect(screen.getByTestId("ask-command-copy-label").textContent).toBe("Copied");
		expect(screen.getByRole("button", { name: "Copied" })).toBeTruthy();
		await act(async () => {
			vi.advanceTimersByTime(2_000);
		});
		expect(screen.getByTestId("ask-command-copy-label").textContent).toBe("Copy");
	});

	it("selects the command and says so when the clipboard refuses", async () => {
		stubClipboard(async () => {
			throw new Error("NotAllowedError");
		});
		renderCard(ask([question({ command: "gh auth status" })]));
		await act(async () => {
			fireEvent.click(screen.getByTestId("ask-command-copy"));
		});
		expect(screen.getByTestId("ask-command-copy-label").textContent).toBe("Copy");
		expect(screen.getByText(/Couldn't copy/)).toBeTruthy();
		expect(window.getSelection()?.toString()).toBe("gh auth status");
	});

	it("announces a failed copy through a status live region (m1)", async () => {
		stubClipboard(async () => {
			throw new Error("NotAllowedError");
		});
		renderCard(ask([question({ command: "gh auth status" })]));
		await act(async () => {
			fireEvent.click(screen.getByTestId("ask-command-copy"));
		});
		const group = screen.getByTestId("ask-command");
		const note = within(group).getByRole("status");
		expect(note.textContent).toMatch(/Couldn't copy/);
	});

	it("reserves the longer label's width so the tap never resizes the button (D4)", async () => {
		vi.useFakeTimers();
		stubClipboard(async () => undefined);
		renderCard(ask([question({ command: "glab auth login" })]));
		const button = screen.getByTestId("ask-command-copy");
		// Both labels are always laid out in one cell; only the visible one swaps.
		const before = Array.from(button.querySelectorAll("span.col-start-1")).map((n) => n.textContent);
		expect(new Set(before)).toEqual(new Set(["Copy", "Copied"]));
		await act(async () => {
			fireEvent.click(button);
		});
		const after = Array.from(button.querySelectorAll("span.col-start-1")).map((n) => n.textContent);
		expect(new Set(after)).toEqual(new Set(["Copy", "Copied"]));
		const hidden = button.querySelector('[aria-hidden="true"]');
		expect(hidden?.className).toContain("invisible");
		expect(hidden?.textContent).toBe("Copy");
	});

	it("reads as code, with a $ mark that is never copied or selected (D5/D6)", async () => {
		const writeText = stubClipboard(async () => {
			throw new Error("NotAllowedError");
		});
		renderCard(ask([question({ command: "gh auth status", options: [] })]));
		const block = screen.getByTestId("ask-command");
		// The code surface, not the answer input's `bg-canvas`.
		expect(block.querySelector(".bg-sunken")).toBeTruthy();
		expect(block.querySelector(".bg-canvas")).toBeNull();
		const prompt = screen.getByTestId("ask-command-prompt");
		expect(prompt.textContent?.trim()).toBe("$");
		expect(prompt.getAttribute("aria-hidden")).toBe("true");
		expect(prompt.className).toContain("select-none");
		expect(screen.getByTestId("ask-command-text").contains(prompt)).toBe(false);
		await act(async () => {
			fireEvent.click(screen.getByTestId("ask-command-copy"));
		});
		expect(writeText).toHaveBeenCalledWith("gh auth status");
		expect(window.getSelection()?.toString()).toBe("gh auth status");
	});
});
