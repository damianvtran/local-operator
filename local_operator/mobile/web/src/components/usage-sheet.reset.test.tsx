// @vitest-environment happy-dom
/**
 * The `/usage` sheet's reset credit interaction.
 *
 * One test per rule the surface must keep: the affordance is a count, never a
 * name; the confirmation names the account it will spend on; only one card can
 * be confirming at a time; an unconfirmed outcome's retry reuses its request
 * id; and a refusal spends nothing and says so. The sheet never claims a reset
 * it did not observe.
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

const paint = async (reports: PhoneUsageReport[]) => {
	vi.mocked(getUsage).mockResolvedValue(envelope(reports));
	render(<UsageSheet open onClose={() => undefined} />);
	await screen.findByText("ChatGPT");
};

it("offers a badge only for an account with spendable credits", async () => {
	await paint([report({ account_index: 0, account_count: 2, resets_available: 0 }), report({ account_index: 1, account_count: 2, resets_available: null })]);
	// One account has none and the other has no trustworthy count: no offer at
	// all, and no row added where it would have been.
	expect(screen.queryByRole("button", { name: /^Reset · / })).toBeNull();
	expect(screen.getAllByRole("progressbar").length).toBe(2);
});

it("puts the badge on the header line without pushing the account label", async () => {
	await paint(twoAccounts(2));
	const second = screen.getByText("Account 2").closest("div.flex") as HTMLElement;
	expect(second).toBeTruthy();
	const badges = screen.getAllByRole("button", { name: /^Reset · 2$/ });
	expect(badges.length).toBe(2);
	// The label and the badge share ONE row: the badge's parent is the same
	// flex row that holds the account number, so no extra row was added.
	const row = badges[0].parentElement as HTMLElement;
	expect(row.className).toContain("items-center");
	expect(row.closest("div")).toBe(badges[0].closest("div"));
});

it("names the account and the consequence before anything is spent", async () => {
	await paint(twoAccounts(2));
	fireEvent.click(screen.getAllByRole("button", { name: /^Reset · 2$/ })[1]);
	const panel = screen.getByRole("group", { name: "ChatGPT · Account 2 of 2" });
	expect(within(panel).getByText("ChatGPT · Account 2 of 2")).toBeTruthy();
	expect(within(panel).getByText("Uses 1 of 2 reset credits; clears this account’s rate-limit windows.")).toBeTruthy();
	expect(within(panel).getByRole("button", { name: "Cancel" })).toBeTruthy();
	// The side-effecting button is never the default focus.
	expect(document.activeElement).toBe(within(panel).getByRole("button", { name: "Cancel" }));
	expect(resetUsage).not.toHaveBeenCalled();
});

it("keeps ONE confirmation open: a second card's tap replaces the first", async () => {
	await paint(twoAccounts(2));
	const badges = screen.getAllByRole("button", { name: /^Reset · 2$/ });
	fireEvent.click(badges[0]);
	expect(screen.getAllByRole("group")).toHaveLength(1);
	fireEvent.click(screen.getAllByRole("button", { name: /^Reset · 2$/ })[0]);
	// Tapping the other card's badge leaves exactly one group, about that card.
	expect(screen.getAllByRole("group")).toHaveLength(1);
	expect(screen.getByRole("group", { name: /Account 1 of 2|Account 2 of 2/ })).toBeTruthy();
});

it("spends once, shows busy, and keeps the buttons inert while in flight", async () => {
	let settle: ((value: unknown) => void) | undefined;
	vi.mocked(resetUsage).mockReturnValue(new Promise((resolve) => { settle = resolve; }) as never);
	await paint(twoAccounts(2));
	fireEvent.click(screen.getAllByRole("button", { name: /^Reset · 2$/ })[1]);
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
	await act(async () => { settle?.({ reports: twoAccounts(1), source: "live", reset: { outcome: "reset", windows_reset: 1 } }); });
});

it("renders the fresh envelope and says only what it observed", async () => {
	vi.mocked(resetUsage).mockResolvedValue({
		reports: twoAccounts(1), source: "live", refreshed_at: 99,
		reset: { outcome: "reset", windows_reset: 1 },
	});
	await paint(twoAccounts(2));
	fireEvent.click(screen.getAllByRole("button", { name: /^Reset · 2$/ })[1]);
	fireEvent.click(screen.getByRole("button", { name: "Reset" }));
	await waitFor(() => expect(screen.getByText("Reset applied — 1 window cleared.")).toBeTruthy());
	// The daemon's fresh count is what the card repaints, not the old one.
	expect(screen.getAllByRole("button", { name: /^Reset · 1$/ }).length).toBe(2);
});

it("says applied when the vendor cleared no window", async () => {
	vi.mocked(resetUsage).mockResolvedValue({ reports: [], reset: { outcome: "reset", windows_reset: 0 } });
	await paint([report({ account_index: 0, account_count: 1, resets_available: 1 })]);
	fireEvent.click(screen.getByRole("button", { name: /^Reset · 1$/ }));
	fireEvent.click(screen.getByRole("button", { name: "Reset" }));
	await waitFor(() => expect(screen.getByText("Reset applied.")).toBeTruthy());
});

it("reuses the redeem id after an unconfirmed outcome", async () => {
	// The first attempt's fate is unknown; the retry finds it already applied --
	// which only happens because the retry reused the first attempt's id.
	vi.mocked(resetUsage)
		.mockResolvedValueOnce({ reset: { outcome: "failed", windows_reset: 0 } })
		.mockResolvedValueOnce({ reset: { outcome: "already_redeemed", windows_reset: 0 } });
	await paint([report({ account_index: 0, account_count: 1, resets_available: 1 })]);
	fireEvent.click(screen.getByRole("button", { name: /^Reset · 1$/ }));
	fireEvent.click(screen.getByRole("button", { name: "Reset" }));
	await waitFor(() => expect(screen.getByText(/Couldn’t confirm the reset/)).toBeTruthy());
	const first = vi.mocked(resetUsage).mock.calls[0][0].redeemRequestId;
	// The retry is offered (and is the only button left, so Cancel is gone).
	const retry = screen.getByRole("button", { name: "Reset" });
	fireEvent.click(retry);
	await waitFor(() => expect(screen.getByText("This reset had already been applied.")).toBeTruthy());
	expect(vi.mocked(resetUsage).mock.calls[1][0].redeemRequestId).toBe(first);
	expect(vi.mocked(resetUsage).mock.calls[1][0]).toMatchObject({ accountIndex: 0, accountCount: 1 });
});

it("calls a refusal a refusal, spends nothing, and mints a new id next time", async () => {
	vi.mocked(resetUsage).mockRejectedValueOnce(
		new HttpError(409, "The account list changed", "accounts_changed"),
	).mockResolvedValue({ reset: { outcome: "no_credit", windows_reset: 0 } });
	await paint([report({ account_index: 0, account_count: 1, resets_available: 1 })]);
	fireEvent.click(screen.getByRole("button", { name: /^Reset · 1$/ }));
	fireEvent.click(screen.getByRole("button", { name: "Reset" }));
	await waitFor(() => expect(screen.getByText("That reset was refused. Refresh the list and try again.")).toBeTruthy());
	// A refusal is not an unconfirmed outcome: no retry button, and the next
	// attempt starts over with a fresh id.
	expect(screen.queryByRole("button", { name: "Reset" })).toBeNull();
	expect(screen.queryByText(/Couldn’t confirm/)).toBeNull();
	fireEvent.click(screen.getByRole("button", { name: /^Reset · 1$/ }));
	fireEvent.click(screen.getByRole("button", { name: "Reset" }));
	await waitFor(() => expect(screen.getByText("No reset credits are left.")).toBeTruthy());
	const [first, second] = vi.mocked(resetUsage).mock.calls;
	expect(second[0].redeemRequestId).not.toBe(first[0].redeemRequestId);
});

it("offers nothing on a dead grant", async () => {
	await paint([report({ account_index: 0, account_count: 1, resets_available: 2, credential_invalid: true })]);
	expect(screen.queryByRole("button", { name: /^Reset · / })).toBeNull();
});
