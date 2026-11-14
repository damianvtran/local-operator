// @vitest-environment happy-dom
//
// D16 / U19 (design v2 A2): on the phone's pairing and approval paths, NO shell
// command (`lop …`) may reach a person outside the "Using a terminal?"
// disclosure. A non-technical person who tapped Approve on an unpaired phone was
// told to run `lop pair` — Ben's dead end, the reason this redesign exists — and
// a single regressed string would rebuild it. So this scans two things:
//
//   1. the relay.pairing catalogue in EVERY locale (a translation could add a
//      command the English lacks — commands ride in as `{command}` placeholders);
//   2. the RENDERED pairing sheet in each of its states, the gate sheet and an
//      approval card on an unpaired phone, outside `[data-pair-terminal]`.
//
// Negative control: inside the disclosure the terminal path DOES name `lop pair`,
// and the scanner must find it there — a scanner that finds nothing anywhere
// would pass this test vacuously.
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { HttpError } from "./api";
import { GateSheet, humanizeGateError } from "./components/gate-sheet";
import { PendingCard } from "./components/pending-card";
import { PairingSheetHost, PairPromptSheet } from "./screens/pair";
import type { PendingRequest } from "./types";

/** A shell command as user copy: `lop` then a subcommand word. `/mobile …` is a
    slash command typed inside the terminal app, which U17 allows in the
    disclosure only — it is not matched here, and the rendered scan keeps it
    inside the disclosure anyway. */
const SHELL_COMMAND = /\blop\s+[a-z]/i;

const CATALOGUES = join(dirname(fileURLToPath(import.meta.url)), "../../../i18n/catalogues");

type Reply = { status: number; body: unknown };
let handler: (path: string, init?: RequestInit) => Reply;

beforeEach(() => {
	handler = () => ({ status: 200, body: { available: true, reason: "ok" } });
	vi.stubGlobal(
		"fetch",
		vi.fn(async (path: string, init?: RequestInit) => {
			const reply = handler(path, init);
			return { ok: reply.status < 300, status: reply.status, json: async () => reply.body };
		}),
	);
});

afterEach(() => {
	cleanup();
	vi.unstubAllGlobals();
	localStorage.clear();
});

/** All visible text outside the terminal disclosure, plus accessible labels. */
function textOutsideDisclosure(): string {
	const clone = document.body.cloneNode(true) as HTMLElement;
	for (const node of Array.from(clone.querySelectorAll("[data-pair-terminal]"))) node.remove();
	const labels = Array.from(clone.querySelectorAll("[aria-label],[placeholder],[title]")).map(
		(el) => `${el.getAttribute("aria-label") ?? ""} ${el.getAttribute("placeholder") ?? ""} ${el.getAttribute("title") ?? ""}`,
	);
	return `${clone.textContent ?? ""}\n${labels.join("\n")}`;
}

function textInsideDisclosure(): string {
	return Array.from(document.querySelectorAll("[data-pair-terminal]"))
		.map((el) => el.textContent ?? "")
		.join("\n");
}

function openEveryDisclosure() {
	for (const header of screen.queryAllByText("Using a terminal?")) {
		const button = header.closest("button");
		if (button?.getAttribute("aria-expanded") === "false") fireEvent.click(button);
	}
}

describe("no shell command outside 'Using a terminal?' (D16/U19)", () => {
	it("the relay.pairing catalogue carries none, in any locale", () => {
		const locales = readdirSync(CATALOGUES);
		const scanned: string[] = [];
		for (const locale of locales) {
			let raw: string;
			try {
				raw = readFileSync(join(CATALOGUES, locale, "relay.pairing.json"), "utf8");
			} catch {
				continue;
			}
			scanned.push(locale);
			for (const [key, value] of Object.entries(JSON.parse(raw) as Record<string, string>)) {
				expect(`${locale} ${key}: ${value}`).not.toMatch(SHELL_COMMAND);
			}
		}
		expect(scanned).toContain("en");
	});

	it("the pairing sheet's intro, not-available, unsupported and error states", async () => {
		const states: Array<() => Reply> = [
			() => ({ status: 200, body: { available: false, reason: "no_operator_authority" } }),
			() => ({ status: 404, body: { error: "no route" } }),
			() => ({ status: 500, body: { error: "boom" } }),
		];
		// Intro first, for both the generic and the in-place (approval) titles.
		render(<PairPromptSheet open onClose={() => undefined} />);
		openEveryDisclosure();
		expect(textOutsideDisclosure()).not.toMatch(SHELL_COMMAND);
		// The negative control: the scanner does find the command, in its place.
		expect(textInsideDisclosure()).toMatch(SHELL_COMMAND);
		// ...as its own bidi isolate, so RTL copy (ur) cannot detach its punctuation.
		expect(textInsideDisclosure()).toMatch(/\u2068lop [^\u2069]*\u2069/);
		cleanup();
		render(<PairPromptSheet open purpose="approve" onClose={() => undefined} />);
		expect(textOutsideDisclosure()).not.toMatch(SHELL_COMMAND);
		cleanup();
		for (const state of states) {
			handler = state;
			render(<PairPromptSheet open purpose="approve" onClose={() => undefined} />);
			fireEvent.click(screen.getByText("Pair this phone"));
			await screen.findByRole("status").catch(() => undefined);
			await act(async () => {
				await new Promise((resolve) => setTimeout(resolve, 50));
			});
			openEveryDisclosure();
			expect(screen.queryByText("Checking your computer…")).toBeNull();
			expect(textOutsideDisclosure()).not.toMatch(SHELL_COMMAND);
			cleanup();
		}
	});

	it("the gate sheet on an unpaired phone, and its authority refusals", () => {
		render(
			<GateSheet open onClose={() => undefined} sessionId="s1" onReceipt={() => undefined} />,
		);
		render(<PairingSheetHost />);
		expect(textOutsideDisclosure()).not.toMatch(SHELL_COMMAND);
		for (const code of ["operator_authority_required", "operator_authority_unconfigured"]) {
			const said = humanizeGateError(
				new HttpError(422, "authorise it from this machine or from your paired phone.", code),
			);
			expect(said).not.toMatch(SHELL_COMMAND);
		}
	});

	it("an approval card tapped on an unpaired phone opens pairing, not a command", async () => {
		handler = (path) =>
			path.includes("/operator/challenge")
				? { status: 200, body: { challenge: "a".repeat(64), expires_s: 30, session_id: "s1", action: "approve", request_id: "r1" } }
				: path.includes("/command")
					? {
							status: 422,
							body: {
								error: "approving this needs the operator's own consent",
								code: "operator_authority_required",
							},
						}
					: { status: 200, body: { available: true, reason: "ok" } };
		const pending: PendingRequest = {
			request_id: "r1",
			kind: "approval",
			title: "run shell command",
			detail: "make",
			options: [],
			secret: false,
			question_index: 0,
			question_total: 1,
		};
		render(
			<>
				<PendingCard pid="s1" pending={pending} />
				<PairingSheetHost />
			</>,
		);
		fireEvent.click(screen.getByText("approve"));
		await screen.findByText("Pair this phone to approve from here");
		openEveryDisclosure();
		expect(textOutsideDisclosure()).not.toMatch(SHELL_COMMAND);
		// The card stays answerable for the fresh tap after pairing (D15): no
		// error line greys it out, and nothing was approved behind the person.
		expect((screen.getByText("approve").closest("button") as HTMLButtonElement).disabled).toBe(false);
	});
});
