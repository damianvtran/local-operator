// @vitest-environment happy-dom
//
// Inbound Team Bridge card bodies on the phone relay (bridge-body design §4).
// The pure model (capability gate, frame parser, state → copy, long-body cut)
// and the card as the transcript draws it: an admitted body inline under the
// untrusted frame, folded when long; held/refused/pending/deleted/offline in
// the server's word; a coalesced card capped at 20 with "N more"; and an older
// relay (no `team_bridge_card_body`) left exactly as today — no read, no panel.
import { act, cleanup, fireEvent, render, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as api from "./api";
import { Transcript } from "./components/transcript";
import { MESSAGE_PRESSED_EVENT } from "./lib/message-fold";
import { CONTENT_RESIZED_EVENT } from "./lib/content-resized";
import {
	BRIDGE_CARD_BODY_COLLAPSE_CHARS,
	bridgeBodyPreview,
	bridgeBodyPreviewText,
	bridgeCardBodyEnabled,
	bridgeCardBodyOf,
	bridgeItemCopyKey,
} from "./lib/bridge-card-body";
import type { Capabilities, TranscriptEntry } from "./types";

vi.mock("./api", async (importOriginal) => ({
	/* The real error class: the panel tells "not yet written" apart from a
	   final failure by its status and body. */
	HttpError: (await importOriginal<typeof import("./api")>()).HttpError,
	getHistory: vi.fn(async () => ({ entries: [], has_more: false })),
	getSubagentHistory: vi.fn(async () => ({ entries: [], has_more: false })),
	imageUrl: vi.fn(() => ""),
	getBridgeCardBody: vi.fn(),
}));

let capabilitySlot: Capabilities | null = null;
vi.mock("./store", async (importOriginal) => {
	const actual = await importOriginal<typeof import("./store")>();
	return { ...actual, useCapabilities: () => capabilitySlot };
});

const mockedBody = vi.mocked(api.getBridgeCardBody);
const WITH_BODIES: Capabilities = { features: { team_bridge_card_body: 1 } } as Capabilities;

beforeEach(() => {
	capabilitySlot = WITH_BODIES;
	mockedBody.mockReset();
});
afterEach(cleanup);

const item = (state: string, extra: Record<string, unknown> = {}) => ({
	message_id: `bcmsg_${state}`,
	conversation_id: "bcv_1",
	state,
	...extra,
});

function card(id = "m1", extra: Partial<TranscriptEntry> = {}): TranscriptEntry {
	return {
		id, kind: "notice", text: "Ada · DM", tool_call_id: "", tool_name: "", tool_state: "done",
		summary: "", intent: "", diff_added: 0, diff_removed: 0, elapsed_s: 0, error: "",
		details: {}, final: true, text_complete: true, custom_type: "team_bridge_message", ...extra,
	} as TranscriptEntry;
}

function draw(entries: TranscriptEntry[]) {
	return render(<Transcript pid="sess_synthetic" entries={entries} streaming={false} />);
}

describe("bridge card body model", () => {
	it("gates on the relay's feature flag and fails closed", () => {
		expect(bridgeCardBodyEnabled(WITH_BODIES)).toBe(true);
		expect(bridgeCardBodyEnabled(null)).toBe(false);
		expect(bridgeCardBodyEnabled({ features: {} } as Capabilities)).toBe(false);
		expect(bridgeCardBodyEnabled({ features: { team_bridge_card_body: "1" } } as unknown as Capabilities)).toBe(false);
		expect(bridgeCardBodyEnabled({ features: { team_bridge_card_body: 0 } } as Capabilities)).toBe(false);
	});

	it("rejects frames that are not the pinned v1 shape", () => {
		expect(bridgeCardBodyOf(null)).toBeNull();
		expect(bridgeCardBodyOf({ v: 2, items: [] })).toBeNull();
		expect(bridgeCardBodyOf({ v: 1 })).toBeNull();
	});

	it("never infers a moderation outcome: unknown or bodiless reads as unavailable", () => {
		const parsed = bridgeCardBodyOf({
			v: 1,
			items: [item("approved"), item("admitted"), item("admitted", { body: "  " }), item("refused", { reason: "abuse" })],
			more: 0,
		})!;
		expect(parsed.items.map((i) => i.state)).toEqual(["unavailable", "unavailable", "unavailable", "refused"]);
		// A refusal never carries a reason to show.
		expect(parsed.items[3].reason).toBeUndefined();
		// A body on a non-admitted item is dropped, never shown.
		expect(bridgeCardBodyOf({ v: 1, items: [item("held", { body: "secret" })], more: 0 })!.items[0].body).toBeUndefined();
	});

	it("drops items without ids and caps the card at 20, counting the rest (dropped ones too)", () => {
		const items = Array.from({ length: 23 }, (_, n) => ({ ...item("pending"), message_id: `bcmsg_${n}` }));
		const parsed = bridgeCardBodyOf({ v: 1, items: [{ state: "admitted", body: "x" }, ...items], more: 5 })!;
		expect(parsed.items).toHaveLength(20);
		// 5 declared + 3 over the cap + 1 dropped for missing ids (review m3).
		expect(parsed.more).toBe(9);
		const three = bridgeCardBodyOf({ v: 1, items: [item("pending"), { state: "pending" }, { ...item("held"), message_id: "x".repeat(129) }], more: 0 })!;
		expect(three.items).toHaveLength(1);
		expect(three.more).toBe(2);
	});

	it("does not append an ellipsis after a sentence-ending stop (design N1)", () => {
		expect(bridgeBodyPreviewText({ text: "item 7.", truncated: true })).toBe("item 7.");
		expect(bridgeBodyPreviewText({ text: "really?", truncated: true })).toBe("really?");
		expect(bridgeBodyPreviewText({ text: "句子。", truncated: true })).toBe("句子。");
		expect(bridgeBodyPreviewText({ text: "mid word", truncated: true })).toBe("mid word…");
		expect(bridgeBodyPreviewText({ text: "whole.", truncated: false })).toBe("whole.");
	});

	it("maps every state to the ruled copy key", () => {
		const key = (raw: Record<string, unknown>) => bridgeItemCopyKey(bridgeCardBodyOf({ v: 1, items: [raw], more: 0 })!.items[0]);
		expect(key(item("held", { hold: { class: "awaiting_review", held_for_s: 3 } }))).toBe("relay.transcript.bridge.held");
		expect(key(item("held", { hold: { class: "transient", held_for_s: 3 } }))).toBe("relay.transcript.bridge.held_transient");
		// An unknown hold class is never softened into "waiting".
		expect(key(item("held", { hold: { class: "mystery" } }))).toBe("relay.transcript.bridge.held");
		expect(key(item("refused"))).toBe("relay.transcript.bridge.refused");
		expect(key(item("pending"))).toBe("relay.transcript.bridge.pending");
		expect(key(item("not_admitted"))).toBe("relay.transcript.bridge.not_admitted");
		expect(key(item("unavailable", { reason: "deleted" }))).toBe("relay.transcript.bridge.deleted");
		expect(key(item("unavailable", { reason: "offline" }))).toBe("relay.transcript.bridge.offline");
		expect(key(item("unavailable", { reason: "signed_out" }))).toBe("relay.transcript.bridge.signed_out");
		expect(key(item("unavailable", { reason: "other_account" }))).toBe("relay.transcript.bridge.other_account");
		expect(key(item("unavailable", { reason: "not_member" }))).toBe("relay.transcript.bridge.not_member");
		expect(key(item("unavailable", { reason: "hub_unsupported" }))).toBe("relay.transcript.bridge.unavailable");
		expect(key(item("unavailable", { reason: "whatever" }))).toBe("relay.transcript.bridge.unavailable");
	});

	it("cuts a long body on a word boundary and never splits a surrogate pair", () => {
		expect(bridgeBodyPreview("short")).toEqual({ text: "short", truncated: false });
		const words = "word ".repeat(200);
		const cut = bridgeBodyPreview(words);
		expect(cut.truncated).toBe(true);
		expect(cut.text.length).toBeLessThanOrEqual(BRIDGE_CARD_BODY_COLLAPSE_CHARS);
		expect(cut.text.endsWith("word")).toBe(true);
		const emoji = "a".repeat(BRIDGE_CARD_BODY_COLLAPSE_CHARS - 1) + "😀" + "b".repeat(10);
		expect(bridgeBodyPreview(emoji).text).toBe("a".repeat(BRIDGE_CARD_BODY_COLLAPSE_CHARS - 1));
	});
});

describe("bridge card on the transcript", () => {
	it("shows an admitted body inline under the untrusted frame", async () => {
		mockedBody.mockResolvedValue({ v: 1, items: [item("admitted", { body: "Hello **team**", source: "local" })], more: 0 });
		const view = draw([card()]);
		expect(view.container.textContent).toContain("Ada · DM");
		expect(view.container.textContent).toContain("Loading message…");
		await waitFor(() => expect(view.container.textContent).toContain("Hello"));
		expect(mockedBody).toHaveBeenCalledWith("sess_synthetic", "m1", expect.anything());
		const panel = view.container.querySelector("[data-bridge-card-body]")!;
		expect(panel.textContent).toContain("External message · untrusted content");
		expect(panel.querySelector("strong")?.textContent).toBe("team");
	});

	it("renders untrusted markdown as data: no raw HTML, no script links", async () => {
		mockedBody.mockResolvedValue({ v: 1, items: [item("admitted", { body: "<img src=x onerror=alert(1)> [go](javascript:alert(1))" })], more: 0 });
		const view = draw([card()]);
		await waitFor(() => expect(view.container.textContent).toContain("<img src=x"));
		const panel = view.container.querySelector("[data-bridge-card-body]")!;
		expect(panel.querySelector("img")).toBeNull();
		expect(panel.querySelector("a[href^='javascript']")).toBeNull();
	});

	it("says held and refused cards are held or refused, not unavailable", async () => {
		mockedBody.mockResolvedValue({
			v: 1,
			items: [item("held", { hold: { class: "awaiting_review", held_for_s: 9 } }), item("refused")],
			more: 0,
		});
		const view = draw([card()]);
		await waitFor(() => expect(view.container.textContent).toContain("Held by moderation"));
		expect(view.container.textContent).toContain("Refused by moderation");
		expect(view.container.textContent).not.toContain("Message body unavailable");
	});

	it("folds a long body behind Show more / Show less", async () => {
		const tail = "TAIL-MARKER";
		mockedBody.mockResolvedValue({ v: 1, items: [item("admitted", { body: "line of text ".repeat(80) + tail })], more: 0 });
		const view = draw([card()]);
		const toggle = await view.findByRole("button", { name: "Show more" });
		expect(view.container.textContent).not.toContain(tail);
		expect(toggle.getAttribute("aria-expanded")).toBe("false");
		fireEvent.click(toggle);
		expect(view.container.textContent).toContain(tail);
		const less = view.getByRole("button", { name: "Show less" });
		expect(less.getAttribute("aria-expanded")).toBe("true");
	});

	it("counts a coalesced card's overflow", async () => {
		mockedBody.mockResolvedValue({ v: 1, items: [item("admitted", { body: "first" })], more: 3 });
		const view = draw([card()]);
		await waitFor(() => expect(view.container.textContent).toContain("3 more messages in this card"));
	});

	it("reads a failed or malformed read as unavailable, never offline", async () => {
		mockedBody.mockRejectedValue(new Error("network"));
		const failed = draw([card("m1")]);
		await waitFor(() => expect(failed.container.textContent).toContain("Message body unavailable"));
		expect(failed.container.textContent).not.toContain("Radient");
		cleanup();
		mockedBody.mockResolvedValue("<html>proxy</html>");
		const malformed = draw([card("m2")]);
		await waitFor(() => expect(malformed.container.textContent).toContain("Message body unavailable"));
	});

	it("reads again on every mount (no cached body, revocation stays live)", async () => {
		mockedBody.mockResolvedValue({ v: 1, items: [item("admitted", { body: "kept" })], more: 0 });
		const first = draw([card()]);
		await waitFor(() => expect(first.container.textContent).toContain("kept"));
		cleanup();
		mockedBody.mockResolvedValue({ v: 1, items: [item("unavailable", { reason: "deleted" })], more: 0 });
		const second = draw([card()]);
		await waitFor(() => expect(second.container.textContent).toContain("This message was deleted"));
		expect(second.container.textContent).not.toContain("kept");
		expect(mockedBody).toHaveBeenCalledTimes(2);
	});

	it("re-asks a card whose row is not written yet, showing Loading, then the body (review M1)", async () => {
		mockedBody
			.mockRejectedValueOnce(new api.HttpError(404, "no such bridge card"))
			.mockResolvedValueOnce({ v: 1, items: [item("admitted", { body: "arrived" })], more: 0 });
		const view = draw([card()]);
		await waitFor(() => expect(mockedBody).toHaveBeenCalledTimes(1));
		await act(async () => {});
		// Between the 404 and the re-ask the card still says Loading, not unavailable.
		expect(view.container.textContent).toContain("Loading message…");
		expect(view.container.textContent).not.toContain("Message body unavailable");
		await waitFor(() => expect(view.container.textContent).toContain("arrived"), { timeout: 2000 });
		expect(mockedBody).toHaveBeenCalledTimes(2);
		expect(view.container.textContent).not.toContain("Message body unavailable");
	});

	it("says unavailable only after the not-yet retries run out, and never retries another 404", async () => {
		mockedBody.mockRejectedValue(new api.HttpError(404, "no such bridge card"));
		const missing = draw([card("m1")]);
		await waitFor(() => expect(missing.container.textContent).toContain("Message body unavailable"), { timeout: 4000 });
		// The first ask plus the two bounded re-asks, no more.
		expect(mockedBody).toHaveBeenCalledTimes(3);
		cleanup();
		mockedBody.mockReset();
		mockedBody.mockRejectedValue(new api.HttpError(404, "unknown session"));
		const gone = draw([card("m2")]);
		await waitFor(() => expect(gone.container.textContent).toContain("Message body unavailable"));
		expect(mockedBody).toHaveBeenCalledTimes(1);
	}, 10_000);

	it("abandons the not-yet re-ask when the row unmounts", async () => {
		mockedBody.mockRejectedValue(new api.HttpError(404, "no such bridge card"));
		const view = draw([card()]);
		await waitFor(() => expect(mockedBody).toHaveBeenCalledTimes(1));
		view.unmount();
		await new Promise((resolve) => setTimeout(resolve, 2300));
		expect(mockedBody).toHaveBeenCalledTimes(1);
	}, 10_000);

	it("aborts the pending read on unmount and writes nothing late (review m2)", async () => {
		let settle: (value: unknown) => void = () => {};
		mockedBody.mockImplementation(() => new Promise((resolve) => { settle = resolve; }));
		const errors = vi.spyOn(console, "error").mockImplementation(() => {});
		try {
			const view = draw([card()]);
			await waitFor(() => expect(mockedBody).toHaveBeenCalledTimes(1));
			const signal = mockedBody.mock.calls[0][2] as AbortSignal;
			expect(signal.aborted).toBe(false);
			view.unmount();
			expect(signal.aborted).toBe(true);
			await act(async () => settle({ v: 1, items: [item("admitted", { body: "late" })], more: 0 }));
			expect(errors).not.toHaveBeenCalled();
		} finally {
			errors.mockRestore();
		}
	});

	it("draws no panel for a pre-receipt card with nothing to resolve (review m1)", async () => {
		mockedBody.mockResolvedValue({ v: 1, items: [], more: 0 });
		const view = draw([card()]);
		await waitFor(() => expect(view.container.querySelector("[data-bridge-card-body]")?.getAttribute("data-state")).toBe("empty"));
		const box = view.container.querySelector<HTMLElement>("[data-bridge-card-body]")!;
		expect(box.hidden).toBe(true);
		expect(box.textContent).toBe("");
		expect(view.container.textContent).toContain("Ada · DM");
		expect(view.container.textContent).not.toContain("Message body unavailable");
		expect(view.container.textContent).not.toContain("External message");
	});

	it("tells the transcript to hold the pressed control around Show more / Show less (design D3)", async () => {
		mockedBody.mockResolvedValue({ v: 1, items: [item("admitted", { body: "line of text ".repeat(80) })], more: 0 });
		const view = draw([card()]);
		const toggle = await view.findByRole("button", { name: "Show more" });
		const seen: string[] = [];
		const scroller = view.container.querySelector(".lo-scroll")!;
		const onPress = (event: Event) => seen.push(`press:${(event as CustomEvent<{ id: string }>).detail.id}:${(event.target as HTMLElement).getAttribute("aria-expanded")}`);
		const onResize = () => seen.push("resize");
		scroller.addEventListener(MESSAGE_PRESSED_EVENT, onPress);
		scroller.addEventListener(CONTENT_RESIZED_EVENT, onResize);
		fireEvent.click(toggle);
		// Pressed BEFORE the flip (still collapsed), then the resize after it.
		expect(seen).toEqual(["press:bridge:m1:0:false", "resize"]);
		// The id names the body the scroller holds when opening, and the control when closing.
		expect(view.container.querySelector('[data-message-id="bridge:m1:0"]')).not.toBeNull();
		expect(view.container.querySelector('[data-message-toggle="bridge:m1:0"]')).toBe(toggle);
	});

	it("holds the body's top on Show more and the control on Show less (design D3)", async () => {
		mockedBody.mockResolvedValue({ v: 1, items: [item("admitted", { body: "line of text ".repeat(80) })], more: 0 });
		const view = draw([card()]);
		const more = await view.findByRole("button", { name: "Show more" });
		const scroller = view.container.querySelector<HTMLElement>(".lo-scroll")!;
		Object.defineProperty(scroller, "scrollHeight", { value: 4000, configurable: true });
		Object.defineProperty(scroller, "clientHeight", { value: 500, configurable: true });
		vi.spyOn(scroller, "getBoundingClientRect").mockReturnValue({ top: 100 } as DOMRect);
		/* Opening: something above moved the body's top by 40; the scroller
		   follows the BODY, not the control at its foot. */
		const body = view.container.querySelector<HTMLElement>('[data-message-id="bridge:m1:0"]')!;
		vi.spyOn(body, "getBoundingClientRect")
			.mockReturnValueOnce({ top: 300 } as DOMRect)
			.mockReturnValue({ top: 340 } as DOMRect);
		scroller.scrollTop = 200;
		fireEvent.click(more);
		expect(scroller.scrollTop).toBe(240);
		/* Closing a long body pulls the control up by 1000; the scroller follows
		   it so the control stays under the thumb. */
		const less = view.getByRole("button", { name: "Show less" });
		vi.spyOn(less, "getBoundingClientRect")
			.mockReturnValueOnce({ top: 1460 } as DOMRect)
			.mockReturnValue({ top: 460 } as DOMRect);
		scroller.scrollTop = 1300;
		fireEvent.click(less);
		expect(scroller.scrollTop).toBe(300);
	});

	it("leaves room under each card so the next receipt heads its own card (design D5)", async () => {
		mockedBody.mockResolvedValue({ v: 1, items: [item("pending")], more: 0 });
		const view = draw([card()]);
		await waitFor(() => expect(view.container.textContent).toContain("Waiting to be delivered"));
		expect(view.container.querySelector("[data-bridge-card-body]")!.className).toContain("mb-2.5");
	});

	it("leaves an older relay's card exactly as today: no read, no panel", async () => {
		capabilitySlot = { features: {} } as Capabilities;
		const view = draw([card()]);
		await act(async () => {});
		expect(mockedBody).not.toHaveBeenCalled();
		expect(view.container.querySelector("[data-bridge-card-body]")).toBeNull();
		expect(view.container.textContent).toContain("Ada · DM");
	});

	it("recognises the live arm's custom_kind and ignores other notices", async () => {
		mockedBody.mockResolvedValue({ v: 1, items: [item("pending")], more: 0 });
		/* Drawn apart: two adjacent receipts fold into one closed quiet group, and
		   a closed group mounts no rows, so nothing is read until it opens. */
		const live = draw([card("live", { custom_type: "", details: { custom_kind: "team_bridge_message" } })]);
		await waitFor(() => expect(live.container.textContent).toContain("Waiting to be delivered to your agent"));
		cleanup();
		const other = draw([card("other", { custom_type: "job_result", text: "job done" })]);
		await act(async () => {});
		expect(other.container.querySelector("[data-bridge-card-body]")).toBeNull();
		expect(mockedBody).toHaveBeenCalledTimes(1);
		expect(mockedBody.mock.calls[0][1]).toBe("live");
	});
});
