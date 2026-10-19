// @vitest-environment happy-dom
/** Interaction proof with deterministic transport; not a rendered-frame claim. */
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AccountRoutingScreen } from "./account-routing";
import { getAccountRouting, setAccountRouting, HttpError, type AccountRoutingPayload } from "../api";
import { parseHash } from "../router";

vi.mock("../api", async (original) => ({
	...await original<typeof import("../api")>(),
	getAccountRouting: vi.fn(),
	setAccountRouting: vi.fn(),
}));
const payload = (value = "reset-soonest"): AccountRoutingPayload => ({
	capability: "account-routing-v1",
	setting: {
		key: "retry.accountSelectionPolicy", kind: "enum", label: "Account policy",
		help: "Synthetic backend help", value, default: "reset-soonest", is_default: value === "reset-soonest",
		choices: [
			{ value: "reset-soonest", label: "Reset soonest", description: "Synthetic reset help" },
			{ value: "headroom", label: "Most headroom", description: "Synthetic headroom help" },
			{ value: "fair", label: "Fair spread", description: "Synthetic spread help" },
		],
	},
});
beforeEach(() => vi.resetAllMocks());
afterEach(cleanup);

describe("selected-computer account policy", () => {
	it("has a reachable bounded route", () => {
		expect(parseHash("#/settings/account-routing")).toEqual({ name: "account-routing" });
	});
	it("shows loading then backend metadata; pending save disables edits and verified receipt reports success", async () => {
		let load!: (value: AccountRoutingPayload) => void;
		vi.mocked(getAccountRouting).mockReturnValue(new Promise(resolve => { load = resolve; }));
		render(<AccountRoutingScreen />);
		expect(screen.getByRole("status").textContent).toContain("Loading");
		await act(async () => load(payload()));
		expect(screen.getByText("Synthetic backend help")).toBeTruthy();
		fireEvent.change(screen.getByRole("combobox"), { target: { value: "fair" } });
		let finish!: (value: AccountRoutingPayload) => void;
		vi.mocked(setAccountRouting).mockReturnValue(new Promise(resolve => { finish = resolve; }));
		fireEvent.click(screen.getByRole("button", { name: "Save policy" }));
		expect((screen.getByRole("combobox") as HTMLSelectElement).disabled).toBe(true);
		expect(setAccountRouting).toHaveBeenCalledWith("fair");
		await act(async () => finish(payload("fair")));
		expect(screen.getByRole("status").textContent).toBe("Saved on this computer.");
		expect((screen.getByRole("combobox") as HTMLSelectElement).value).toBe("fair");
	});
	it("old relay and empty schema are unavailable, never fake a policy or save", async () => {
		vi.mocked(getAccountRouting).mockRejectedValue(new HttpError(404, "synthetic old relay"));
		const first = render(<AccountRoutingScreen />);
		await waitFor(() => expect(screen.getByRole("alert").textContent).toContain("unavailable"));
		expect(screen.queryByRole("combobox")).toBeNull();
		first.unmount();
		vi.mocked(getAccountRouting).mockResolvedValue({ ...payload(), setting: { ...payload().setting, choices: [] } });
		render(<AccountRoutingScreen />);
		await waitFor(() => expect(screen.getByRole("alert").textContent).toContain("unavailable"));
		expect(screen.queryByRole("combobox")).toBeNull();
		expect(setAccountRouting).not.toHaveBeenCalled();
	});
	it("failed save retains draft and never claims saved; next retry can succeed", async () => {
		vi.mocked(getAccountRouting).mockResolvedValue(payload());
		render(<AccountRoutingScreen />);
		await screen.findByRole("combobox");
		fireEvent.change(screen.getByRole("combobox"), { target: { value: "headroom" } });
		vi.mocked(setAccountRouting).mockRejectedValue(new HttpError(422, "synthetic rejected save"));
		fireEvent.click(screen.getByRole("button", { name: "Save policy" }));
		await waitFor(() => expect(screen.getByRole("alert").textContent).toContain("not saved"));
		expect(screen.queryByText("Saved on this computer.")).toBeNull();
		expect((screen.getByRole("combobox") as HTMLSelectElement).value).toBe("headroom");
		vi.mocked(setAccountRouting).mockResolvedValue(payload("headroom"));
		fireEvent.click(screen.getByRole("button", { name: "Save policy" }));
		await waitFor(() => expect(screen.getByRole("status").textContent).toBe("Saved on this computer."));
	});
	it("shows canonical permission guidance and keeps unknown transport outcomes unconfirmed", async () => {
		vi.mocked(getAccountRouting).mockResolvedValue(payload());
		render(<AccountRoutingScreen />);
		await screen.findByRole("combobox");
		fireEvent.change(screen.getByRole("combobox"), { target: { value: "headroom" } });
		vi.mocked(setAccountRouting).mockRejectedValue(new HttpError(503, "Synthetic configuration permissions guidance", "setting_write_failed"));
		fireEvent.click(screen.getByRole("button", { name: "Save policy" }));
		await waitFor(() => expect(screen.getByRole("alert").textContent).toContain("configuration permissions"));
		vi.mocked(setAccountRouting).mockRejectedValue(new TypeError("synthetic lost reply"));
		fireEvent.click(screen.getByRole("button", { name: "Save policy" }));
		await waitFor(() => expect(screen.getByRole("alert").textContent).toContain("could not be confirmed"));
		expect(screen.queryByText("Saved on this computer.")).toBeNull();
	});
	it("malformed metadata cannot crash the editor or manufacture a choice", async () => {
		vi.mocked(getAccountRouting).mockResolvedValue({ ...payload(), setting: { ...payload().setting, choices: { length: 1 } } } as unknown as AccountRoutingPayload);
		render(<AccountRoutingScreen />);
		await waitFor(() => expect(screen.getByRole("alert").textContent).toContain("unavailable"));
		expect(screen.queryByRole("combobox")).toBeNull();
	});
	it("unmount invalidates late read receipts rather than repainting a later screen", async () => {
		let stale!: (value: AccountRoutingPayload) => void;
		vi.mocked(getAccountRouting).mockReturnValueOnce(new Promise(resolve => { stale = resolve; }));
		const first = render(<AccountRoutingScreen />);
		first.unmount();
		vi.mocked(getAccountRouting).mockResolvedValue(payload("fair"));
		render(<AccountRoutingScreen />);
		await screen.findByRole("combobox");
		await act(async () => stale(payload("headroom")));
		expect((screen.getByRole("combobox") as HTMLSelectElement).value).toBe("fair");
	});
});
