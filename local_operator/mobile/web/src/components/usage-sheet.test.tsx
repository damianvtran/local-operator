// @vitest-environment happy-dom
import { act, cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { getUsage, HttpError, refreshUsage, type PhoneUsageEnvelope, type PhoneUsageLimit, type PhoneUsageReport } from "../api";
import { UsageSheet } from "./usage-sheet";
import { messageRuntime } from "../store";
import { BUNDLED_OWNED } from "../i18n/bundled.gen";

vi.mock("../api", async () => {
	// The real HttpError, so the sheet's `instanceof` check sees the same class.
	const { HttpError } = await vi.importActual<typeof import("../api")>("../api");
	return { getUsage: vi.fn(), refreshUsage: vi.fn(), HttpError };
});
beforeEach(() => {
	// Default: the on-open refresh never answers, so a test sees the GET paint.
	vi.mocked(refreshUsage).mockReturnValue(new Promise(() => undefined));
});
afterEach(() => {
	cleanup();
	vi.resetAllMocks();
	vi.useRealTimers();
	messageRuntime.registerBundle("relay.shell", "en", BUNDLED_OWNED["relay.shell"].en);
});

const limit = (over: Partial<PhoneUsageLimit> = {}): PhoneUsageLimit => ({
	amount: { used: null, limit: null, remaining: null, used_fraction: 0.2, unit: "percent" },
	paid_fallback: false, available: null, unlimited: false, supersedes_tiers: false,
	family_cap: false, shared: true, window: "5h", family: "unknown",
	resets_at_ms: null, payable: false, exhausted: false, ...over,
});
const report = (over: Partial<PhoneUsageReport> = {}): PhoneUsageReport => ({
	provider: "openai", unavailable: false, fetched_at: 1, limits: [limit()], ...over,
});
const envelope = (reports: PhoneUsageReport[], over: Partial<PhoneUsageEnvelope> = {}): PhoneUsageEnvelope =>
	({ source: "cached", reports, ...over });

it("an empty cache is not a measured zero credit balance", async () => {
	vi.mocked(getUsage).mockResolvedValue(envelope([]));
	render(<UsageSheet open onClose={() => undefined} />);
	expect(await screen.findByText("No providers are reporting usage.")).toBeTruthy();
	expect(screen.queryByText(/0 credits/)).toBeNull();
});

it("paints the cached GET, then refreshes unforced on open", async () => {
	vi.mocked(getUsage).mockResolvedValue(envelope([report({ provider_name: "OpenAI (ChatGPT Plus/Pro)" })]));
	render(<UsageSheet open onClose={() => undefined} />);
	expect(await screen.findByText("OpenAI (ChatGPT Plus/Pro)")).toBeTruthy();
	expect(refreshUsage).toHaveBeenCalledWith(false);
	expect(screen.getByText("Refreshing…")).toBeTruthy();
	// The refresh button is busy, and the cached card is still on screen. It is
	// inert via aria-disabled, never native `disabled`, which would drop focus.
	const button = screen.getByRole("button", { name: "Refresh usage" });
	expect(button.getAttribute("aria-disabled")).toBe("true");
	expect(button.classList.contains("!text-ink-disabled")).toBe(true);
	expect(button.hasAttribute("disabled")).toBe(false);
	fireEvent.click(button);
	expect(refreshUsage).toHaveBeenCalledTimes(1);
	expect(screen.getByRole("progressbar")).toBeTruthy();
});

it("headings use the display name, never the raw provider id", async () => {
	vi.mocked(getUsage).mockResolvedValue(envelope([report({ provider: "anthropic", provider_name: "Anthropic (Claude Pro/Max)" })]));
	render(<UsageSheet open onClose={() => undefined} />);
	expect(await screen.findByRole("heading", { name: "Anthropic (Claude Pro/Max)" })).toBeTruthy();
	expect(screen.queryByText("anthropic")).toBeNull();
});

it("labels accounts only when a provider has several, in login order", async () => {
	vi.mocked(getUsage).mockResolvedValue(envelope([
		report({ provider: "anthropic", account_index: 1, account_count: 2, limits: [limit({ amount: { used: null, limit: null, remaining: null, used_fraction: 1, unit: "percent" }, exhausted: true })] }),
		report({ provider: "anthropic", account_index: 0, account_count: 2 }),
		report({ provider: "deepseek", account_index: 0, account_count: 1 }),
	]));
	render(<UsageSheet open onClose={() => undefined} />);
	const first = await screen.findByText("Account 1");
	// U6: the numbering is explained once per multi-account provider, never
	// under a single-account one.
	expect(screen.getAllByText("Accounts are numbered in the order they were signed in on the computer.")).toHaveLength(1);
	const second = screen.getByText("Account 2");
	// Login order, not health order: Account 1 precedes Account 2 in the DOM.
	expect(first.compareDocumentPosition(second) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
	expect(screen.getAllByText(/^Account \d$/)).toHaveLength(2);
});

it("measured bars carry a tone word; unknown draws no track and says it once", async () => {
	vi.mocked(getUsage).mockResolvedValue(envelope([report({ limits: [
		limit({ tone: "warning", remaining_fraction: 0.1 }),
		limit({ window: "7d", tone: "unknown", remaining_fraction: null, amount: { used: null, limit: null, remaining: null, used_fraction: null, unit: "percent" } }),
	] })]));
	render(<UsageSheet open onClose={() => undefined} />);
	const bars = await screen.findAllByRole("progressbar");
	expect(bars[0].getAttribute("aria-valuenow")).toBe("10");
	expect(bars[0].getAttribute("aria-valuetext")).toMatch(/5-hour: 10% left, Near limit/);
	expect(bars[1].hasAttribute("aria-valuenow")).toBe(false);
	expect(bars[1].className).toContain("border-dotted");
	expect(bars[1].className).not.toContain("bg-sunken");
	expect(screen.getAllByText("Near limit").length).toBeGreaterThan(0);
	// "Amount not reported" already says it: no second "Not reported" (D3).
	expect(screen.getByText("Amount not reported")).toBeTruthy();
	expect(screen.queryByText("Not reported")).toBeNull();
	expect(bars[1].getAttribute("aria-valuetext")).toBe("7-day: Amount not reported");
});

it("a paid balance shows its amount with no tone word; unlimited is worded per kind", async () => {
	vi.mocked(getUsage).mockResolvedValue(envelope([report({ limits: [
		limit({ paid_fallback: true, shared: false, window: "unknown", tone: "unknown", remaining_fraction: null, available: true, payable: true,
			amount: { used: null, limit: null, remaining: 18.4, used_fraction: null, unit: "usd" } }),
		limit({ window: "1mo", unlimited: true, tone: "unknown", remaining_fraction: null }),
	] })]));
	render(<UsageSheet open onClose={() => undefined} />);
	expect(await screen.findByText("18.4 USD left")).toBeTruthy();
	// D1/U5: a known balance has no limit to be "within"; it is not "not reported".
	expect(screen.queryByText("Not reported")).toBeNull();
	// D4: an included unlimited window is plain "Unlimited", not a paid allowance.
	expect(screen.getByText("Unlimited")).toBeTruthy();
	expect(screen.queryByText("Unlimited paid allowance")).toBeNull();
});

it("a blocked wallet keeps its actual amount and does not promise continuation", async () => {
	vi.mocked(getUsage).mockResolvedValue(envelope([report({ limits: [limit({
		amount: { used: null, limit: null, remaining: 123.5, used_fraction: null, unit: "credits" },
		paid_fallback: true, available: false, shared: false, window: "unknown",
	})] })]));
	render(<UsageSheet open onClose={() => undefined} />);
	expect(await screen.findByText("123.5 credits left")).toBeTruthy();
	expect(screen.getByText("Spend unavailable")).toBeTruthy();
	expect(screen.queryByText(/credits will continue/)).toBeNull();
});

it("healthy shared quota with an exhausted family never claims paid continuation", async () => {
	vi.mocked(getUsage).mockResolvedValue(envelope([report({ provider: "anthropic", limits: [
		limit(),
		limit({ shared: false, family_cap: true, family: "opus", window: "7d", exhausted: true,
			amount: { used: null, limit: null, remaining: null, used_fraction: 1, unit: "percent" } }),
		limit({ shared: false, paid_fallback: true, available: true, payable: true,
			amount: { used: null, limit: null, remaining: 25, used_fraction: null, unit: "credits" } }),
	] })]));
	render(<UsageSheet open onClose={() => undefined} />);
	expect(await screen.findByText("Model-family caps still apply.")).toBeTruthy();
	expect(screen.queryByText(/Included quota used|credits will continue/)).toBeNull();
	expect(screen.getByText("Opus · 7-day")).toBeTruthy();
});

it("explicit shared exhaustion continues, but unknown shared metadata does not", async () => {
	const included = limit({ exhausted: true, amount: { used: null, limit: null, remaining: null, used_fraction: 1, unit: "percent" } });
	const wallet = limit({ shared: false, paid_fallback: true, available: true, payable: true,
		amount: { used: null, limit: null, remaining: 25, used_fraction: null, unit: "credits" } });
	vi.mocked(getUsage).mockResolvedValue(envelope([report({ limits: [included, wallet] })]));
	const mounted = render(<UsageSheet open onClose={() => undefined} />);
	expect(await screen.findByText("Included quota used; credits will continue.")).toBeTruthy();
	mounted.unmount();
	vi.mocked(getUsage).mockResolvedValue(envelope([report({ limits: [{ ...included, shared: false }, wallet] })]));
	render(<UsageSheet open onClose={() => undefined} />);
	expect(await screen.findByText("25 credits left")).toBeTruthy();
	expect(screen.queryByText("Included quota used; credits will continue.")).toBeNull();
});

it("reauth shows the remedy and dims last-known fills", async () => {
	vi.mocked(getUsage).mockResolvedValue(envelope([report({ state: "reauth_required", credential_invalid: true,
		limits: [limit({ tone: "ok", remaining_fraction: 0.8 })] })]));
	render(<UsageSheet open onClose={() => undefined} />);
	expect(await screen.findByText("Sign in again on the computer to update this account.")).toBeTruthy();
	expect(screen.getByText("Sign in again")).toBeTruthy();
	const fill = screen.getByRole("progressbar").firstElementChild as HTMLElement;
	expect(fill.className).toContain("bg-ink-dim");
});

it("loading and failed reads remain distinct from empty, and Try again re-reads", async () => {
	vi.mocked(getUsage).mockRejectedValueOnce(new Error("synthetic failure"));
	render(<UsageSheet open onClose={() => undefined} />);
	expect(screen.getByText("Loading usage…")).toBeTruthy();
	expect(await screen.findByRole("alert")).toBeTruthy();
	expect(screen.queryByText("No providers are reporting usage.")).toBeNull();
	vi.mocked(getUsage).mockResolvedValueOnce(envelope([]));
	fireEvent.click(screen.getByRole("button", { name: "Try again" }));
	expect(await screen.findByText("No providers are reporting usage.")).toBeTruthy();
});

it("a failed or timed-out refresh keeps the cached cards and says so", async () => {
	vi.mocked(getUsage).mockResolvedValue(envelope([report({ provider_name: "OpenAI" })]));
	vi.mocked(refreshUsage).mockResolvedValue(envelope([report({ provider_name: "OpenAI" })], {
		refresh: { status: "timeout", forced: false, retry_after_ms: null },
	}));
	render(<UsageSheet open onClose={() => undefined} />);
	expect(await screen.findByText("Refresh timed out. Showing last known usage.")).toBeTruthy();
	expect(screen.getByText("OpenAI")).toBeTruthy();
	cleanup();
	vi.mocked(refreshUsage).mockRejectedValue(new Error("network"));
	render(<UsageSheet open onClose={() => undefined} />);
	expect(await screen.findByText("Couldn't refresh. Showing last known usage.")).toBeTruthy();
	expect(screen.getByText("OpenAI")).toBeTruthy();
});

it("a throttled forced refresh keeps focus, announces once and falls back to Updated", async () => {
	vi.useFakeTimers({ shouldAdvanceTime: true });
	vi.mocked(getUsage).mockResolvedValue(envelope([report()]));
	vi.mocked(refreshUsage).mockResolvedValueOnce(envelope([report()], {
		refresh: { status: "ok", forced: false, retry_after_ms: null }, refreshed_at: Date.now(),
	}));
	render(<UsageSheet open onClose={() => undefined} />);
	const button = await screen.findByRole("button", { name: "Refresh usage" });
	await vi.waitFor(() => expect(button.hasAttribute("aria-disabled")).toBe(false));
	expect(screen.getByText(/^Updated /)).toBeTruthy();
	vi.mocked(refreshUsage).mockResolvedValueOnce(envelope([report()], {
		refresh: { status: "throttled", forced: false, retry_after_ms: 3000 },
	}));
	button.focus();
	fireEvent.click(button);
	expect(refreshUsage).toHaveBeenLastCalledWith(true);
	const live = document.querySelector("[aria-live]")!;
	// U4: a cooldown, not a usage limit.
	expect(await screen.findByText("Just refreshed — you can refresh again in 3 seconds.", { selector: "[aria-hidden] , [aria-hidden] *" })).toBeTruthy();
	expect(button.getAttribute("aria-disabled")).toBe("true");
	expect(button.classList.contains("!text-ink-disabled")).toBe(true);
	// U1: focus stays on the sheet's main action.
	expect(document.activeElement).toBe(button);
	// U3: the spoken sentence is fixed at entry while the visible one ticks.
	const spoken = () => live.querySelector(".sr-only")?.textContent;
	expect(spoken()).toBe("Just refreshed — you can refresh again in 3 seconds.");
	await act(async () => { vi.advanceTimersByTime(1100); });
	expect(live.querySelector("[aria-hidden]")?.textContent).toMatch(/in 2 seconds/);
	expect(spoken()).toBe("Just refreshed — you can refresh again in 3 seconds.");
	await act(async () => { vi.advanceTimersByTime(2500); });
	expect(button.hasAttribute("aria-disabled")).toBe(false);
	expect(button.classList.contains("!text-ink-disabled")).toBe(false);
	expect(document.activeElement).toBe(button);
	// U7: after the cooldown the line shows freshness again, never blank, and
	// says once that refreshing is available.
	expect(live.querySelector("[aria-hidden]")?.textContent).toMatch(/^Updated /);
	expect(spoken()).toMatch(/You can refresh again\.$/);
});

it("a joined refresh is ok, and an older daemon without the route is not a failure", async () => {
	vi.mocked(getUsage).mockResolvedValue(envelope([report({ provider_name: "OpenAI" })], { cached_at: Date.now() }));
	vi.mocked(refreshUsage).mockResolvedValue(envelope([report({ provider_name: "OpenAI" })], {
		refresh: { status: "ok", forced: true, joined: true, retry_after_ms: null }, refreshed_at: Date.now(),
	}));
	render(<UsageSheet open onClose={() => undefined} />);
	expect(await screen.findByText(/^Updated /)).toBeTruthy();
	cleanup();
	// R5: 404/405 from POST /api/usage/refresh means "this daemon predates it".
	vi.mocked(refreshUsage).mockRejectedValue(new HttpError(404, "Not Found"));
	render(<UsageSheet open onClose={() => undefined} />);
	expect(await screen.findByText("OpenAI")).toBeTruthy();
	await vi.waitFor(() => expect(screen.queryByRole("button", { name: "Refresh usage" })).toBeNull());
	expect(screen.queryByText(/Couldn't refresh/)).toBeNull();
	expect(screen.getByText(/^Updated /)).toBeTruthy();
});

it("the reset toggles between relative and exact time", async () => {
	const at = Date.now() + (14 * 60 + 5) * 60_000 + 30_000;
	vi.mocked(getUsage).mockResolvedValue(envelope([report({ limits: [limit({ resets_at_ms: at })] })]));
	render(<UsageSheet open onClose={() => undefined} />);
	// U8: the accessible name is the shown text, and aria-pressed says which.
	const toggle = await screen.findByRole("button", { name: "resets in 14h 6m" });
	expect(within(toggle).getByText("resets in 14h 6m")).toBeTruthy();
	expect(toggle.getAttribute("aria-pressed")).toBe("false");
	expect(toggle.className).toContain("decoration-dotted");
	fireEvent.click(toggle);
	expect(toggle.getAttribute("aria-pressed")).toBe("true");
	expect(toggle.textContent).toMatch(/^Resets at /);
	expect(toggle.textContent).not.toContain(String(new Date().getFullYear()));
});

it("stale cached snapshots disclose freshness", async () => {
	vi.mocked(getUsage).mockResolvedValue(envelope([], { cached_at: 1700000000000, may_be_stale: true }));
	// An older daemon (no refresh route) leaves the GET's snapshot as the answer.
	vi.mocked(refreshUsage).mockRejectedValue(new HttpError(404, "Not Found"));
	render(<UsageSheet open onClose={() => undefined} />);
	// U7: staleness rides the status line's freshness, the footer is fine print.
	expect(await screen.findByText(/^Updated .* May be out of date\.$/)).toBeTruthy();
	// Not today, so the status line carries the date; the footer keeps the
	// full stamp without repeating the stale warning.
	expect(screen.getByText(/^Updated Nov 14, 2023, /)).toBeTruthy();
	expect(screen.getByText(/^As of .*\.$/).textContent).not.toContain("May be out of date");
	expect(screen.queryByText(/No provider refresh is requested/)).toBeNull();
});
