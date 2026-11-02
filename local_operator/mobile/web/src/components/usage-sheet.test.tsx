// @vitest-environment happy-dom
import { act, cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { getUsage, refreshUsage, type PhoneUsageEnvelope, type PhoneUsageLimit, type PhoneUsageReport } from "../api";
import { UsageSheet } from "./usage-sheet";
import { messageRuntime } from "../store";
import { BUNDLED_OWNED } from "../i18n/bundled.gen";

vi.mock("../api", () => ({ getUsage: vi.fn(), refreshUsage: vi.fn() }));
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
	// The refresh button is busy, and the cached card is still on screen.
	const button = screen.getByRole("button", { name: "Refresh usage" });
	expect(button.hasAttribute("disabled")).toBe(true);
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
	const second = screen.getByText("Account 2");
	// Login order, not health order: Account 1 precedes Account 2 in the DOM.
	expect(first.compareDocumentPosition(second) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
	expect(screen.getAllByText(/^Account \d$/)).toHaveLength(2);
});

it("every bar carries a tone word and remaining semantics; unknown draws no track", async () => {
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
	expect(screen.getAllByText("Not reported").length).toBeGreaterThan(0);
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

it("a throttled forced refresh counts down with the button disabled", async () => {
	vi.useFakeTimers({ shouldAdvanceTime: true });
	vi.mocked(getUsage).mockResolvedValue(envelope([report()]));
	vi.mocked(refreshUsage).mockResolvedValueOnce(envelope([report()], {
		refresh: { status: "ok", forced: false, retry_after_ms: null }, refreshed_at: Date.now(),
	}));
	render(<UsageSheet open onClose={() => undefined} />);
	const button = await screen.findByRole("button", { name: "Refresh usage" });
	await vi.waitFor(() => expect(button.hasAttribute("disabled")).toBe(false));
	vi.mocked(refreshUsage).mockResolvedValueOnce(envelope([report()], {
		refresh: { status: "throttled", forced: false, retry_after_ms: 3000 },
	}));
	fireEvent.click(button);
	expect(refreshUsage).toHaveBeenLastCalledWith(true);
	expect(await screen.findByText(/Try again in 3 seconds/)).toBeTruthy();
	expect(button.hasAttribute("disabled")).toBe(true);
	await act(async () => { vi.advanceTimersByTime(3500); });
	expect(button.hasAttribute("disabled")).toBe(false);
});

it("the reset toggles between relative and exact time", async () => {
	const at = Date.now() + (14 * 60 + 5) * 60_000 + 30_000;
	vi.mocked(getUsage).mockResolvedValue(envelope([report({ limits: [limit({ resets_at_ms: at })] })]));
	render(<UsageSheet open onClose={() => undefined} />);
	const toggle = await screen.findByRole("button", { name: /^Resets at/ });
	expect(within(toggle).getByText("resets in 14h 6m")).toBeTruthy();
	fireEvent.click(toggle);
	expect(toggle.textContent).toMatch(/^Resets at/);
});

it("stale cached snapshots disclose freshness", async () => {
	vi.mocked(getUsage).mockResolvedValue(envelope([], { cached_at: 1700000000000, may_be_stale: true }));
	render(<UsageSheet open onClose={() => undefined} />);
	expect(await screen.findByText(/May be out of date/)).toBeTruthy();
	expect(screen.getByText(/As of/)).toBeTruthy();
	expect(screen.queryByText(/No provider refresh is requested/)).toBeNull();
});
