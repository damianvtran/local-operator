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
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { HttpError, isAuthorityRefusalCode } from "./api";
import { clauseSeparator, humanizeGateError } from "./components/gate-sheet";
import { isAuthorityRefusal } from "./components/pending-card";
import { SessionScreen } from "./screens/session-view";
import { messageRuntime } from "./store";
import type { PendingRequest, SessionProjection } from "./types";

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
		getSkills: vi.fn(async () => ({ skills: [] })),
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

/* THE COPY LANGUAGE IS A TEST INPUT, like the projection slot above.

   `useLocaleSnapshot` reports the ACTUAL rendered owned-copy language, and this
   build registers an English bundle only (`BUNDLED_OWNED`), so the real store
   answers "en" under every backend preference — a locale case could never reach
   the sheet through it. The join rule has to be proved per language all the same,
   so the store's runtime is isolated here exactly as `usage-sheet.format.test.tsx`
   isolates it: the same generated English bundle the store registers, one runtime
   whose context a test can move, and the attested catalogue rows registered on top
   for the language under test. This is component/DOM proof of the rule, not a
   claim that the SPA ships those translated bundles yet. */
const copyLocale = vi.hoisted(() => ({ uiLocale: "en", backendLocale: null as string | null }));

vi.mock("./store", async (importOriginal) => {
	const actual = await importOriginal<typeof import("./store")>();
	const { createMessageRuntime } = await import("./i18n/runtime");
	const { BUNDLED_OWNED } = await import("./i18n/bundled.gen");
	const runtime = createMessageRuntime(
		() => ({
			backendLocale: copyLocale.backendLocale,
			uiLocale: copyLocale.uiLocale,
			direction: copyLocale.uiLocale === "ur" ? "rtl" : "ltr",
			revision: 0,
		}),
		() => undefined,
	);
	for (const [namespace, locales] of Object.entries(BUNDLED_OWNED)) {
		for (const [locale, messages] of Object.entries(locales)) {
			runtime.registerBundle(namespace, locale, { ...messages });
		}
	}
	return {
		...actual,
		messageRuntime: runtime,
		useLocaleSnapshot: () => ({
			backendLocale: copyLocale.backendLocale,
			uiLocale: copyLocale.uiLocale,
			direction: copyLocale.uiLocale === "ur" ? ("rtl" as const) : ("ltr" as const),
			revision: 0,
		}),
		useLanguage: () => copyLocale.backendLocale,
		useProjection: vi.fn(() => slot),
		retainProjectionStream: vi.fn(() => () => {}),
		/* Returns the refcount unsubscribe, as the real one does, so a test that
		   unmounts exercises the effect's cleanup path rather than a no-op
		   (review round 2, N2-1). */
		retainSessionListStream: vi.fn(() => () => {}),
		useDraft: vi.fn(() => ["", () => {}]),
	};
});

/* The attested rows, read from the committed catalogues rather than copied into
   this file: the assertions then describe the sentence translators actually
   shipped, and a copy edit cannot leave the join test pinned to stale bytes.
   Resolved from this file's own directory (not `process.cwd()`, which a runner
   may not place at the SPA root) and through `fileURLToPath` rather than a
   `new URL(..., import.meta.url)` literal, which Vite rewrites into an asset
   import it then refuses to serve from outside the project. */
const CATALOGUES = join(dirname(fileURLToPath(import.meta.url)), "../../../i18n/catalogues");
function catalogue(locale: string, namespace: string): Record<string, string> {
	return JSON.parse(
		readFileSync(join(CATALOGUES, locale, `${namespace}.json`), "utf8"),
	) as Record<string, string>;
}

/** Render the sheet's copy in `next`. English needs nothing: the bundle above is
    registered. Every other language registers its relay.approvals rows on top, so
    the sentence under test is the attested one for that locale. */
function copyIn(next: string): void {
	copyLocale.uiLocale = next;
	if (next !== "en") {
		messageRuntime.registerBundle("relay.approvals", next, catalogue(next, "relay.approvals"));
	}
}

/** The clause pieces of `relay.approvals.gate.body_scope`, in composition order. */
const CLAUSES = ["before", "session_clause", "between", "default_clause", "after"] as const;
function clauses(locale: string): string[] {
	const rows = catalogue(locale, "relay.approvals");
	return CLAUSES.map((clause) => rows[`relay.approvals.gate.body_scope.${clause}`]);
}

/** The scope paragraph, found by a phrase rather than by position: the sheet
    renders the body sentence in the same paragraph, and `getByText` on the whole
    element would be satisfied by that sentence alone. */
function scopeParagraph(phrase: string): HTMLElement {
	return screen.getByText(
		(_, element) => element?.tagName === "P" && (element.textContent ?? "").includes(phrase),
	);
}

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
	copyLocale.uiLocale = "en";
	copyLocale.backendLocale = null;
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
		   default", and it is not. It is named in its own sentence —
		   `relay.approvals.gate.unpaired_default_auto` (carried by all seven
		   locales) — rather than as another fragment inside the translated first
		   sentence, so the locales that carry the fragments keep their own word
		   order instead of getting one English clause spliced into the middle. */
		expect(said.textContent).toContain(
			"Run without asking in all sessions also needs a paired phone.",
		);
		// ...and it was there BEFORE the tap: no request had been made yet.
		expect(mocks.sendCommandWithProof).not.toHaveBeenCalled();
	});

	it("keeps main's emphasis on the two blast-radius phrases (M5)", async () => {
		/* KEYING MUST NOT DROP SEMANTIC EMPHASIS. Main renders this sentence with
		   <strong> around "this session" and "the default for all sessions"; one
		   catalogued string can only render them flat, and a translator could
		   never restore emphasis the key dropped. The sentence is therefore
		   composed from five clause keys, and this pins BOTH halves: the exact
		   words (the split changed no copy) and exactly two bold runs at the two
		   scope phrases (a later refactor cannot flatten them silently). */
		await openSheet();
		const scope = screen.getByText(
			(_, element) =>
				element?.tagName === "P" &&
				(element.textContent ?? "").includes("The first two choices change"),
		);
		expect(scope.textContent).toContain(
			"The first two choices change this session only; the last changes the default for all sessions that follow it.",
		);
		const bold = Array.from(scope.querySelectorAll("strong")).map(
			(element) => element.textContent,
		);
		expect(bold).toEqual(["this session", "default for all sessions"]);
		// The five keys are joined with single spaces; a doubled space at a join
		// is invisible in review and visible to the reader.
		expect(scope.textContent).not.toContain("  ");
	});

	it("concatenates the scope clauses in a CJK copy language (zh-CN)", async () => {
		/* THE SPLIT CREATED FOUR SEAMS WHERE THE SINGLE KEY HAD NO CHARACTER AT
		   ALL. Chinese writes Han with no space between clauses, so a U+0020 at
		   each seam is four visible gaps and four cells of width in a sentence
		   that used to be compact — the copy was never wrong, the join was.
		   NOTE ON WHAT THIS DOES NOT PIN: the retired single key read
		   "前两个选项只改变本会话；最后一个改变其后所有会话的默认值。"; the per-locale
		   split reworded the Chinese copy itself (而已 / 则把其后 … 改掉), so the
		   composed sentence is no longer byte-identical to it. That divergence is
		   copy, not spacing, and this test pins the SPACING rule: no separator at
		   any of the four seams. */
		copyIn("zh-CN");
		await openSheet();
		const pieces = clauses("zh-CN");
		const composed = pieces.join("");
		const scope = scopeParagraph(pieces[0]);
		expect(scope.textContent?.endsWith(composed)).toBe(true);
		// No U+0020 anywhere inside the composed scope sentence...
		expect(composed).not.toContain("\u0020");
		expect((scope.textContent ?? "").slice(-composed.length)).toBe(composed);
		// ...and the two comparison units are still the two bold runs.
		expect(Array.from(scope.querySelectorAll("strong")).map((element) => element.textContent)).toEqual([
			pieces[1],
			pieces[3],
		]);
	});

	it("renders the fr fine space, not a breakable U+0020, before the semicolon", async () => {
		/* FRENCH IS NOT COSMETIC HERE. The fr style guide makes U+202F required
		   before `; : ! ?`, and a plain U+0020 is a line-break opportunity: in a
		   narrow sheet the `;` could wrap onto the next line, leaving a line that
		   starts with punctuation. U+202F is a no-break space, so the mark stays
		   with the clause it closes. The pre-split single fr key carried U+202F at
		   this seam; the split replaced it with U+0020, so this composition is
		   byte-for-byte the retired sentence again (verified against 4b40d82512,
		   the last commit carrying the single key for fr). */
		copyIn("fr");
		await openSheet();
		const pieces = clauses("fr");
		const composed = `${pieces[0]} ${pieces[1]}\u202f${pieces[2]} ${pieces[3]} ${pieces[4]}`;
		const scope = scopeParagraph(pieces[0]);
		expect(scope.textContent?.endsWith(composed)).toBe(true);
		const semicolon = composed.indexOf(";");
		expect(semicolon).toBeGreaterThan(0);
		expect(composed.charCodeAt(semicolon - 1)).toBe(0x202f);
		// Neither the breakable U+0020 nor the plain no-break U+00A0 may sit there.
		expect(composed).not.toMatch(/[\u0020\u00a0];/);
		expect(Array.from(scope.querySelectorAll("strong")).map((element) => element.textContent)).toEqual([
			pieces[1],
			pieces[3],
		]);
	});

	it("keeps the two bold runs on the two comparison units in every shipped locale", async () => {
		/* THE EMPHASIS IS THE REASON THE SENTENCE IS COMPOSED AT ALL, so it has to
		   survive the join rule in every language, not just the two with a special
		   separator: the bold runs are the two scope phrases, whatever language
		   surrounds them. */
		for (const language of ["en", "fr", "es", "zh-CN", "ru", "vi", "hi", "ur"]) {
			copyIn(language);
			await openSheet();
			const pieces = clauses(language);
			const scope = scopeParagraph(pieces[0]);
			expect(
				Array.from(scope.querySelectorAll("strong")).map((element) => element.textContent),
			).toEqual([pieces[1], pieces[3]]);
			cleanup();
		}
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
			expect(screen.getByRole("status").textContent).toContain(
				"the default for sessions that follow it is now auto",
			),
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

describe("the clause join rule", () => {
	it("concatenates in CJK languages and keeps one space everywhere else", () => {
		expect(clauseSeparator("zh-CN", "本会话", "而已")).toBe("");
		expect(clauseSeparator("ja", "a", "b")).toBe("");
		expect(clauseSeparator("ko", "a", "b")).toBe("");
		// Every other shipped language separates clauses with one ordinary space.
		for (const language of ["en", "es", "ru", "vi", "hi", "ur"]) {
			expect(clauseSeparator(language, "a", "b")).toBe(" ");
		}
	});

	it("uses the fr fine space before every mark the guide spaces, and after «", () => {
		// The guide's set is `: ; ! ?` plus the inside-quote space before `»` —
		// not `;` alone, which is what the one observed seam would have suggested.
		for (const mark of [";", ":", "!", "?", "»"]) {
			expect(clauseSeparator("fr", "mot", `${mark} la suite`)).toBe("\u202f");
		}
		expect(clauseSeparator("fr", "«", "citation")).toBe("\u202f");
		expect(clauseSeparator("fr", "mot", "ordinaire")).toBe(" ");
		expect(clauseSeparator("fr-CA", "mot", "; la suite")).toBe("\u202f");
	});
});

describe("the sheet's effective-mode line (U2)", () => {
	/* The sheet reads the mode off the projection's pending approval — the only
	   place the runtime publishes it — so each case sets the pending the runtime
	   would send and opens the sheet the reader actually uses. `within(dialog)`
	   because the pending card underneath paints the same line; the assertion is
	   about the SHEET's copy. */
	function pendingApproval(overrides: Partial<PendingRequest>): PendingRequest {
		return {
			request_id: "req-approval",
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

	it("names the REAL global behind a session override (U4)", async () => {
		slot.projection!.pending = pendingApproval({
			approval_mode: "ask",
			approval_provenance: "session",
			approval_global_mode: "ask",
		});
		await openSheet();
		expect(
			within(screen.getByRole("dialog")).getByText(
				"Approvals: Ask — session override; global Ask.",
			),
		).toBeTruthy();
	});

	it("drops the global clause when the owner sends no global mode", async () => {
		/* Half a true sentence beats a whole false one: the sheet must not
		   negate the session mode into a claim about the machine. */
		slot.projection!.pending = pendingApproval({
			approval_mode: "ask",
			approval_provenance: "session",
		});
		await openSheet();
		const dialog = screen.getByRole("dialog");
		expect(within(dialog).getByText("Approvals: Ask — session override.")).toBeTruthy();
		expect(within(dialog).queryByText(/global (Ask|Auto)/)).toBeNull();
	});

	it("says the global default, without the pending card's request clause", async () => {
		slot.projection!.pending = pendingApproval({
			approval_mode: "auto",
			approval_provenance: "global",
		});
		await openSheet();
		const dialog = screen.getByRole("dialog");
		expect(within(dialog).getByText("Approvals: Auto — global default.")).toBeTruthy();
		/* The sheet holds no request. "This request still needs your answer"
		   belongs to the card, and a sheet that said it would name a card that
		   is not on screen. */
		expect(within(dialog).queryByText(/This request still needs your answer/)).toBeNull();
	});

	it("separates a run pin from both saved scopes", async () => {
		slot.projection!.pending = pendingApproval({
			approval_mode: "auto",
			approval_provenance: "pin",
		});
		await openSheet();
		expect(
			within(screen.getByRole("dialog")).getByText("Approvals: Auto — this run."),
		).toBeTruthy();
	});

	it("separates a run pin from both saved scopes", async () => {
		slot.projection!.pending = pendingApproval({
			approval_mode: "auto",
			approval_provenance: "pin",
		});
		await openSheet();
		expect(
			within(screen.getByRole("dialog")).getByText("Approvals: Auto — this run."),
		).toBeTruthy();
	});

	it("omits the line when the owner publishes neither field", async () => {
		/* The default fixture has no pending at all — the state of an owner older
		   than these fields. The sheet must say nothing rather than invent a
		   "global default". */
		await openSheet();
		expect(within(screen.getByRole("dialog")).queryByText(/^Approvals:/)).toBeNull();
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
