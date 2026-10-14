// @vitest-environment happy-dom
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { getUsage } from "../api";
import { UsageSheet } from "./usage-sheet";

vi.mock("../api", () => ({ getUsage: vi.fn() }));
afterEach(() => { cleanup(); vi.resetAllMocks(); });

it("an empty cache is not a measured zero credit balance", async () => {
	vi.mocked(getUsage).mockResolvedValue({ reports: [], source: "cached" });
	render(<UsageSheet open onClose={() => undefined} />);
	expect(await screen.findByText("No cached usage reported.")).toBeTruthy();
	expect(screen.queryByText(/0 credits/)).toBeNull();
});

it("a blocked wallet keeps its actual amount and does not promise continuation", async () => {
	vi.mocked(getUsage).mockResolvedValue({ source: "cached", reports: [{ provider: "openai",
		unavailable: false, fetched_at: 1, limits: [{
			amount: { used: null, limit: null, remaining: 123.5, used_fraction: null, unit: "credits" },
			paid_fallback: true, available: false, unlimited: false,
			supersedes_tiers: false, family_cap: false, shared: false, window: "unknown",
			family: "unknown", resets_at_ms: null, payable: false, exhausted: false,
		}] }] });
	render(<UsageSheet open onClose={() => undefined} />);
	expect(await screen.findByText(/123.5 credits left/)).toBeTruthy();
	expect(screen.getByText("Spend unavailable")).toBeTruthy();
	expect(screen.queryByText(/credits will continue/)).toBeNull();
});

it("loading and failed cached reads remain distinct from empty", async () => {
	vi.mocked(getUsage).mockRejectedValue(new Error("synthetic failure"));
	render(<UsageSheet open onClose={() => undefined} />);
	expect(screen.getByText("Loading cached usage…")).toBeTruthy();
	expect(await screen.findByRole("alert")).toBeTruthy();
	expect(screen.queryByText("No cached usage reported.")).toBeNull();
});

it("healthy shared quota with an exhausted family never claims paid continuation", async () => {
	const base = { amount: { used: null, limit: null, remaining: null, used_fraction: 0.2, unit: "percent" },
		paid_fallback: false, available: null, unlimited: false, supersedes_tiers: false,
		family_cap: false, shared: true, window: "5h" as const, family: "unknown" as const,
		resets_at_ms: null, payable: false, exhausted: false };
	vi.mocked(getUsage).mockResolvedValue({ source: "cached", reports: [{ provider: "anthropic",
		unavailable: false, fetched_at: 1, limits: [base,
			{ ...base, shared: false, family_cap: true, family: "opus", window: "7d", exhausted: true,
				amount: { ...base.amount, used_fraction: 1 } },
			{ ...base, shared: false, paid_fallback: true, available: true, payable: true,
				amount: { ...base.amount, remaining: 25, used_fraction: null, unit: "credits" } },
		] }] });
	render(<UsageSheet open onClose={() => undefined} />);
	expect(await screen.findByText("Model-family caps still apply.")).toBeTruthy();
	expect(screen.queryByText(/Included quota used|credits will continue/)).toBeNull();
	expect(screen.getByText(/Included allowance.*5-hour/)).toBeTruthy();
	expect(screen.getByText(/Model-family allowance.*opus.*7-day/)).toBeTruthy();
});

it("explicit shared exhaustion continues, but unknown shared metadata does not", async () => {
	const included = { amount: { used: null, limit: null, remaining: null, used_fraction: 1, unit: "percent" },
		paid_fallback: false, available: null, unlimited: false, supersedes_tiers: false,
		family_cap: false, shared: true, window: "5h" as const, family: "unknown" as const,
		resets_at_ms: null, payable: false, exhausted: true };
	const wallet = { ...included, shared: false, paid_fallback: true, available: true,
		payable: true, exhausted: false,
		amount: { ...included.amount, used_fraction: null, remaining: 25, unit: "credits" } };
	vi.mocked(getUsage).mockResolvedValue({ source: "cached", reports: [{ provider: "openai",
		unavailable: false, fetched_at: 1, limits: [included, wallet] }] });
	const mounted = render(<UsageSheet open onClose={() => undefined} />);
	expect(await screen.findByText("Included quota used; credits will continue.")).toBeTruthy();
	mounted.unmount();
	vi.mocked(getUsage).mockResolvedValue({ source: "cached", reports: [{ provider: "openai",
		unavailable: false, fetched_at: 1, limits: [{ ...included, shared: false }, wallet] }] });
	render(<UsageSheet open onClose={() => undefined} />);
	expect(await screen.findByText(/25 credits left/)).toBeTruthy();
	expect(screen.queryByText(/credits will continue/)).toBeNull();
});
