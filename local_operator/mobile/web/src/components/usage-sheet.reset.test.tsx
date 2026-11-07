// @vitest-environment happy-dom
/**
 * The `/usage` sheet's reset credit interaction.
 *
 * One test per rule the surface must keep: the affordance is a count, never a
 * name; the confirmation names the account it will spend on; only one card can
 * be confirming at a time; the idempotency id survives every gesture that is
 * not "send that confirmation again"; a refusal spends nothing and says so;
 * and an outcome stays on screen even when it is the LAST credit that was
 * spent, which is the case that removes the badge it was opened from.
 */
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { getUsage, HttpError, refreshUsage, resetUsage, type PhoneUsageEnvelope, type PhoneUsageLimit, type PhoneUsageReport } from "../api";
import { UsageSheet } from "./usage-sheet";
import { messageRuntime } from "../store";
import { BUNDLED_OWNED } from "../i18n/bundled.gen";

vi.mock("../api", async () => {
	// The real HttpError and refusal predicate, so the sheet's own checks see
	// the same classes they would in the bundle.
	const actual = await vi.importActual<typeof import("../api")>("../api");
	return {
		HttpError: actual.HttpError,
		isUsageResetRefusal: actual.isUsageResetRefusal,
		getUsage: vi.fn(),
		refreshUsage: vi.fn(),
		resetUsage: vi.fn(),
	};
});
beforeEach(() => {
	// The on-open refresh never answers, so every test sees the GET's paint.
	vi.mocked(refreshUsage).mockReturnValue(new Promise(() => undefined));
});
afterEach(() => {
	cleanup();
	vi.resetAllMocks();
	vi.useRealTimers();
	messageRuntime.registerBundle("relay.shell", "en", BUNDLED_OWNED["relay.shell"].en);
	messageRuntime.registerBundle("relay.usage_reset", "en", BUNDLED_OWNED["relay.usage_reset"].en);
});

const limit = (over: Partial<PhoneUsageLimit> = {}): PhoneUsageLimit => ({
	amount: { used: null, limit: null, remaining: null, used_fraction: 0.2, unit: "percent" },
	paid_fallback: false, available: null, unlimited: false, supersedes_tiers: false,
	family_cap: false, shared: true, window: "5h", family: "unknown",
	resets_at_ms: null, payable: false, exhausted: false, ...over,
});
const report = (over: Partial<PhoneUsageReport> = {}): PhoneUsageReport => ({
	provider: "openai", provider_name: "ChatGPT", unavailable: false, fetched_at: 1,
	limits: [limit()], resets_available: 2, ...over,
});
const envelope = (reports: PhoneUsageReport[], over: Partial<PhoneUsageEnvelope> = {}): PhoneUsageEnvelope =>
	({ source: "cached", reports, ...over });

/** Two ChatGPT logins with resets on both — the numbering case the copy names. */
const twoAccounts = (resets = 2) => [
	report({ account_index: 0, account_count: 2, resets_available: resets }),
	report({ account_index: 1, account_count: 2, resets_available: resets }),
];

/** The badge's accessible name is the visible count PLUS the account it spends
 *  on (U4), so every query matches the prefix and the account is asserted
 *  where it matters. */
const badgeName = (count: number) => new RegExp(`^Reset · ${count} — `);
const badges = (count: number) => screen.queryAllByRole("button", { name: badgeName(count) });

const paint = async (reports: PhoneUsageReport[]) => {
	vi.mocked(getUsage).mockResolvedValue(envelope(reports));
	render(<UsageSheet open onClose={() => undefined} />);
	await screen.findByText("ChatGPT");
};

it("offers a badge only for an account with spendable credits", async () => {
	await paint([report({ account_index: 0, account_count: 2, resets_available: 0 }), report({ account_index: 1, account_count: 2, resets_available: null })]);
	// One account has none and the other has no trustworthy count: no offer at
	// all, and no row added where it would have been.
	expect(badges(0)).toHaveLength(0);
	expect(screen.queryByRole("button", { name: /^Reset · / })).toBeNull();
	expect(screen.getAllByRole("progressbar").length).toBe(2);
});

it("names the account and the count, and wraps the header instead of scrolling", async () => {
	// A long tone word is what overflowed in ru/fr/es at 320px (D2/U5). happy-dom
	// has no layout engine, so the assertions that bite are the class contract:
	// the row wraps, the label keeps its own line, and the badge never shrinks.
	messageRuntime.registerBundle("relay.shell", "en", {
		...BUNDLED_OWNED["relay.shell"].en,
		"relay.shell.usage.tone_warning": "Proche de la limite avec une marge",
	});
	await paint(twoAccounts(2));
	const badge = badges(2)[1];
	// U4: the accessible name carries the account AND the number of credits.
	expect(badge.getAttribute("aria-label")).toBe("Reset · 2 — ChatGPT · Account 2 of 2");
	const row = badge.closest("div.justify-between") as HTMLElement;
	expect(row).toBeTruthy();
	expect(row.className).toContain("flex-wrap");
	expect(within(row).getByText("Account 2")).toBeTruthy();
	expect(badge.className).toContain("shrink-0");
	// D5: a 44px-wide target even where the text is ~42px (hi/zh-CN).
	expect(badge.className).toContain("min-w-11");
	const label = within(row).getByText("Account 2");
	expect(label.className).toContain("truncate");
	// The rendered frame is design QA's; this keeps the invariant cheap to check.
	for (const node of [row, badge, label]) {
		expect(node.scrollWidth).toBeLessThanOrEqual(node.clientWidth);
	}
});

it("names the account and the consequence before anything is spent", async () => {
	await paint(twoAccounts(2));
	fireEvent.click(badges(2)[1]);
	const panel = screen.getByRole("group", { name: "ChatGPT · Account 2 of 2" });
	expect(within(panel).getByText("ChatGPT · Account 2 of 2")).toBeTruthy();
	// D3: the consequence is body weight, not the weakest text on the card.
	const prompt = within(panel).getByText("Uses 1 of 2 reset credits; clears this account’s rate-limit windows.");
	expect(prompt.className).toContain("text-body-sm");
	expect(within(panel).getByRole("button", { name: "Cancel" })).toBeTruthy();
	// D4: the button that spends is visually distinct, not just differently
	// coloured text -- it is the primary variant whose fill is the accent.
	const confirm = within(panel).getByRole("button", { name: "Reset" });
	expect(confirm.className).toContain("bg-accent");
	// The side-effecting button is never the default focus.
	expect(document.activeElement).toBe(within(panel).getByRole("button", { name: "Cancel" }));
	expect(resetUsage).not.toHaveBeenCalled();
});

it("drops the numbering when the provider has one account", async () => {
	await paint([report({ account_index: 0, account_count: 1, resets_available: 2 })]);
	expect(badges(2)[0].getAttribute("aria-label")).toBe("Reset · 2 — ChatGPT");
	fireEvent.click(badges(2)[0]);
	// U9: "Account 1 of 1" names nothing the operator does not already know.
	expect(screen.getByRole("group", { name: "ChatGPT" })).toBeTruthy();
	expect(screen.queryByText("ChatGPT · Account 1 of 1")).toBeNull();
});

it("keeps ONE confirmation open: a second card's tap replaces the first", async () => {
	await paint(twoAccounts(2));
	const [first, second] = badges(2);
	fireEvent.click(first);
	expect(screen.getByRole("group", { name: "ChatGPT · Account 1 of 2" })).toBeTruthy();
	// Nothing is in flight, so the operator may change their mind: still ONE
	// group, and it is about the card they last opened. (The open card's own
	// badge is hidden while its panel is up, so the remaining badge is the
	// other account's.)
	fireEvent.click(second);
	expect(screen.getAllByRole("group")).toHaveLength(1);
	expect(screen.getByRole("group", { name: "ChatGPT · Account 2 of 2" })).toBeTruthy();
});

it("spends once, shows busy, and keeps the buttons inert while in flight", async () => {
	let settle: ((value: unknown) => void) | undefined;
	vi.mocked(resetUsage).mockReturnValue(new Promise((resolve) => { settle = resolve; }) as never);
	await paint(twoAccounts(2));
	fireEvent.click(badges(2)[1]);
	fireEvent.click(screen.getByRole("button", { name: "Reset" }));
	expect(resetUsage).toHaveBeenCalledTimes(1);
	const call = vi.mocked(resetUsage).mock.calls[0][0];
	expect(call).toMatchObject({ provider: "openai", accountIndex: 1, accountCount: 2 });
	expect(call.redeemRequestId).toMatch(/[0-9a-f-]{36}/);
	// Busy: the buttons stay in the tab order but take no clicks (a native
	// `disabled` would drop keyboard focus out of the confirmation).
	await waitFor(() => expect(screen.getByText("Resetting…")).toBeTruthy());
	const cancel = screen.getByRole("button", { name: "Cancel" });
	const confirm = screen.getByRole("button", { name: "Reset" });
	expect(cancel.getAttribute("aria-disabled")).toBe("true");
	expect(confirm.getAttribute("aria-disabled")).toBe("true");
	fireEvent.click(confirm);
	expect(resetUsage).toHaveBeenCalledTimes(1);
	// U1/R2: while one spend is outstanding EVERY other badge is inert, so the
	// flow showing its outcome cannot be replaced (and its id discarded).
	const sibling = badges(2)[0];
	expect(sibling.getAttribute("aria-disabled")).toBe("true");
	fireEvent.click(sibling);
	expect(screen.getAllByRole("group")).toHaveLength(1);
	expect(screen.getByText("Resetting…")).toBeTruthy();
	expect(resetUsage).toHaveBeenCalledTimes(1);
	// U1/R2: the OTHER card's badge is inert too, so a tap cannot replace the
	// flow that is showing this attempt's outcome and throw its id away.
	const other = badges(2)[0];
	expect(other.getAttribute("aria-disabled")).toBe("true");
	fireEvent.click(other);
	expect(resetUsage).toHaveBeenCalledTimes(1);
	expect(screen.getAllByRole("group")).toHaveLength(1);
	await act(async () => { settle?.({ reports: twoAccounts(1), source: "live", reset: { outcome: "reset", windows_reset: 1 } }); });
});

it("renders the fresh envelope and says only what it observed", async () => {
	vi.mocked(resetUsage).mockResolvedValue({
		reports: twoAccounts(1), source: "live", refreshed_at: 99,
		reset: { outcome: "reset", windows_reset: 1 },
	});
	await paint(twoAccounts(2));
	fireEvent.click(badges(2)[1]);
	fireEvent.click(screen.getByRole("button", { name: "Reset" }));
	await waitFor(() => expect(screen.getByText("Reset applied — 1 window cleared.")).toBeTruthy());
	// The daemon's fresh count is what the card repaints, not the old one.
	expect(badges(1).length).toBe(2);
});

it("keeps the outcome on screen when the last credit was the one spent", async () => {
	// D1/Q2: the badge is gated on a positive count, so this answer repaints the
	// card at zero. The panel must not unmount with the sentence unread.
	vi.mocked(resetUsage).mockResolvedValue({
		reports: [report({ account_index: 0, account_count: 1, resets_available: 0 })],
		source: "live",
		reset: { outcome: "reset", windows_reset: 1 },
	});
	await paint([report({ account_index: 0, account_count: 1, resets_available: 1 })]);
	fireEvent.click(badges(1)[0]);
	fireEvent.click(screen.getByRole("button", { name: "Reset" }));
	await waitFor(() => expect(screen.getByText("Reset applied — 1 window cleared.")).toBeTruthy());
	// No badge: there is nothing left to spend. The panel still says what happened.
	expect(screen.queryByRole("button", { name: /^Reset · / })).toBeNull();
	// U6: and it can be dismissed, which returns focus to the card it was on.
	const dismiss = screen.getByRole("button", { name: "Dismiss" });
	fireEvent.click(dismiss);
	expect(screen.queryByText("Reset applied — 1 window cleared.")).toBeNull();
});

it("says applied when the vendor cleared no window", async () => {
	vi.mocked(resetUsage).mockResolvedValue({ reports: [], reset: { outcome: "reset", windows_reset: 0 } });
	await paint([report({ account_index: 0, account_count: 1, resets_available: 1 })]);
	fireEvent.click(badges(1)[0]);
	fireEvent.click(screen.getByRole("button", { name: "Reset" }));
	await waitFor(() => expect(screen.getByText("Reset applied.")).toBeTruthy());
});

it("does not put a refresh banner beside the outcome it cannot confirm", async () => {
	// R5/Q6: the spend's own re-fetch was throttled. The numbers are old, and
	// the panel says so -- the status line must not add "Just refreshed — you
	// can refresh again in N seconds" next to a credit that is already gone.
	vi.mocked(resetUsage).mockResolvedValue({
		reports: [report({ account_index: 0, account_count: 1, resets_available: 1 })],
		source: "cached",
		refresh: { status: "throttled", forced: true, joined: false, retry_after_ms: 5000 },
		reset: { outcome: "reset", windows_reset: 1 },
	});
	await paint([report({ account_index: 0, account_count: 1, resets_available: 2 })]);
	fireEvent.click(badges(2)[0]);
	fireEvent.click(screen.getByRole("button", { name: "Reset" }));
	await waitFor(() => expect(screen.getByText("Reset applied — 1 window cleared.")).toBeTruthy());
	expect(screen.getByText("Couldn't refresh. Showing last known usage.")).toBeTruthy();
	expect(screen.queryByText(/Just refreshed/)).toBeNull();
});

it("reuses the redeem id after an unconfirmed outcome", async () => {
	// The first attempt's fate is unknown; the retry finds it already applied --
	// which only happens because the retry reused the first attempt's id.
	vi.mocked(resetUsage)
		.mockResolvedValueOnce({ reset: { outcome: "failed", windows_reset: 0 } })
		.mockResolvedValueOnce({ reset: { outcome: "already_redeemed", windows_reset: 0 } });
	await paint([report({ account_index: 0, account_count: 1, resets_available: 1 })]);
	fireEvent.click(badges(1)[0]);
	fireEvent.click(screen.getByRole("button", { name: "Reset" }));
	await waitFor(() => expect(screen.getByText(/Couldn’t confirm the reset/)).toBeTruthy());
	const first = vi.mocked(resetUsage).mock.calls[0][0].redeemRequestId;
	// U7: the retry is labelled to match the copy it sits under ("try again to
	// check"), so it does not read as a second spend.
	const retry = screen.getByRole("button", { name: "Try again" });
	expect(screen.queryByRole("button", { name: "Reset" })).toBeNull();
	fireEvent.click(retry);
	await waitFor(() => expect(screen.getByText("This reset had already been applied.")).toBeTruthy());
	expect(vi.mocked(resetUsage).mock.calls[1][0].redeemRequestId).toBe(first);
	expect(vi.mocked(resetUsage).mock.calls[1][0]).toMatchObject({ accountIndex: 0, accountCount: 1 });
});

it("keeps the pending id across a close and a reopen", async () => {
	// U1/R2 (c): closing the sheet discards the flow, and the retry must still
	// reuse the id it never learned the fate of -- two POSTs, one credit.
	vi.mocked(getUsage).mockResolvedValue(envelope([report({ account_index: 0, account_count: 1, resets_available: 2 })]));
	vi.mocked(resetUsage).mockResolvedValue({ reset: { outcome: "failed", windows_reset: 0 } });
	const view = render(<UsageSheet open onClose={() => undefined} />);
	await screen.findByText("ChatGPT");
	fireEvent.click(badges(2)[0]);
	fireEvent.click(screen.getByRole("button", { name: "Reset" }));
	await waitFor(() => expect(screen.getByText(/Couldn’t confirm the reset/)).toBeTruthy());
	const first = vi.mocked(resetUsage).mock.calls[0][0].redeemRequestId;
	view.rerender(<UsageSheet open={false} onClose={() => undefined} />);
	view.rerender(<UsageSheet open onClose={() => undefined} />);
	await screen.findByText("ChatGPT");
	fireEvent.click(badges(2)[0]);
	fireEvent.click(screen.getByRole("button", { name: "Reset" }));
	await waitFor(() => expect(resetUsage).toHaveBeenCalledTimes(2));
	expect(vi.mocked(resetUsage).mock.calls[1][0].redeemRequestId).toBe(first);
});

it("calls a refusal a refusal, spends nothing, and mints a new id next time", async () => {
	vi.mocked(resetUsage).mockRejectedValueOnce(
		new HttpError(409, "The account list changed", "accounts_changed"),
	).mockResolvedValue({ reset: { outcome: "no_credit", windows_reset: 0 } });
	await paint([report({ account_index: 0, account_count: 1, resets_available: 1 })]);
	const refreshesBefore = vi.mocked(refreshUsage).mock.calls.length;
	fireEvent.click(badges(1)[0]);
	fireEvent.click(screen.getByRole("button", { name: "Reset" }));
	await waitFor(() => expect(screen.getByText("That reset was refused. Refresh the list and try again.")).toBeTruthy());
	// A refusal is not an unconfirmed outcome: no retry button, and the next
	// attempt starts over with a fresh id.
	expect(screen.queryByRole("button", { name: "Try again" })).toBeNull();
	expect(screen.queryByText(/Couldn’t confirm/)).toBeNull();
	// U3: the copy asks the operator to refresh, so the sheet does it.
	await waitFor(() => expect(vi.mocked(refreshUsage).mock.calls.length).toBe(refreshesBefore + 1));
	expect(vi.mocked(refreshUsage).mock.calls.at(-1)?.[0]).toBe(false);
	fireEvent.click(badges(1)[0]);
	fireEvent.click(screen.getByRole("button", { name: "Reset" }));
	await waitFor(() => expect(screen.getByText("No reset credits are left.")).toBeTruthy());
	const [first, second] = vi.mocked(resetUsage).mock.calls;
	expect(second[0].redeemRequestId).not.toBe(first[0].redeemRequestId);
});

it("shows a daemon-decided refusal as a refusal, not as an unknown spend", async () => {
	// Q1/R1: the controller refused before anything was sent. The outcome
	// arrives as `refused`, which is definitive: no retry, no "couldn't confirm".
	vi.mocked(resetUsage).mockResolvedValue({ reset: { outcome: "refused", windows_reset: 0 } });
	await paint([report({ account_index: 0, account_count: 1, resets_available: 1 })]);
	fireEvent.click(badges(1)[0]);
	fireEvent.click(screen.getByRole("button", { name: "Reset" }));
	await waitFor(() => expect(screen.getByText("That reset was refused. Refresh the list and try again.")).toBeTruthy());
	expect(screen.queryByRole("button", { name: "Try again" })).toBeNull();
	expect(screen.getByRole("button", { name: "Dismiss" })).toBeTruthy();
});

it("sends the daemon's ref with the spend, and nothing when it published none", async () => {
	vi.mocked(resetUsage).mockResolvedValue({ reports: [], reset: { outcome: "reset", windows_reset: 1 } });
	await paint([
		report({ account_index: 0, account_count: 2, resets_available: 2, credential_ref: "oauth:7" }),
		report({ account_index: 1, account_count: 2, resets_available: 2 }),
	]);
	fireEvent.click(badges(2)[0]);
	fireEvent.click(screen.getByRole("button", { name: "Reset" }));
	await waitFor(() => expect(resetUsage).toHaveBeenCalledTimes(1));
	expect(vi.mocked(resetUsage).mock.calls[0][0].credentialRef).toBe("oauth:7");
	// The second row has no ref (the daemon could not attribute it); the spend
	// still goes by index, and `credentialRef` is simply absent from the call.
	fireEvent.click(screen.getByRole("button", { name: "Dismiss" }));
	fireEvent.click(badges(2)[1]);
	fireEvent.click(screen.getByRole("button", { name: "Reset" }));
	await waitFor(() => expect(resetUsage).toHaveBeenCalledTimes(2));
	expect(vi.mocked(resetUsage).mock.calls[1][0].credentialRef).toBeNull();
	// The daemon's own numbering travels with it either way.
	expect(vi.mocked(resetUsage).mock.calls[1][0]).toMatchObject({ accountIndex: 1, accountCount: 2 });
});

it("Esc dismisses the confirmation first and the sheet second", async () => {
	const onClose = vi.fn();
	vi.mocked(getUsage).mockResolvedValue(envelope([report({ account_index: 0, account_count: 1, resets_available: 2 })]));
	render(<UsageSheet open onClose={onClose} />);
	await screen.findByText("ChatGPT");
	fireEvent.click(badges(2)[0]);
	expect(screen.getAllByRole("group")).toHaveLength(1);
	// U8: nothing is spent, and the sheet stays where the operator was.
	fireEvent.keyDown(window, { key: "Escape" });
	await waitFor(() => expect(screen.queryAllByRole("group")).toHaveLength(0));
	expect(onClose).not.toHaveBeenCalled();
	fireEvent.keyDown(window, { key: "Escape" });
	await waitFor(() => expect(onClose).toHaveBeenCalledTimes(1));
});

it("stops offering when the daemon says no credit is left", async () => {
	// U6: `no_credit` is a fact about the count even when the answer carries no
	// envelope, so the badge must not survive beside "No reset credits are left."
	vi.mocked(resetUsage).mockResolvedValue({ reset: { outcome: "no_credit", windows_reset: 0 } });
	await paint([report({ account_index: 0, account_count: 1, resets_available: 1 })]);
	fireEvent.click(badges(1)[0]);
	fireEvent.click(screen.getByRole("button", { name: "Reset" }));
	await waitFor(() => expect(screen.getByText("No reset credits are left.")).toBeTruthy());
	expect(screen.queryByRole("button", { name: /^Reset · / })).toBeNull();
});

it("offers nothing on a dead grant", async () => {
	await paint([report({ account_index: 0, account_count: 1, resets_available: 2, credential_invalid: true })]);
	expect(screen.queryByRole("button", { name: /^Reset · / })).toBeNull();
});
