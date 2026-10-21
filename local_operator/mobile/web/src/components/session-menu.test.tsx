// @vitest-environment happy-dom
//
// The session menu: the header's three overflow rows, each naming itself and
// stating its current value. The header controls these replace were the
// operator's "four controls whose purpose is unclear": an unlabelled `? 4`, a
// bordered `[wide]` and a word-shaped `approvals`. These cells pin that every
// row carries a label, that the pin and wide rows state their state through
// both visible text and `aria-pressed`, and that Approvals closes this sheet
// before it asks for the gate sheet.
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ComponentProps } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { SessionMenu } from "./session-menu";

afterEach(() => {
	cleanup();
	localStorage.clear();
	delete document.documentElement.dataset.view;
});

type MenuProps = ComponentProps<typeof SessionMenu>;

function setup(overrides: Partial<MenuProps> = {}) {
	const props: MenuProps = {
		open: true,
		onClose: vi.fn(),
		pinned: false,
		onTogglePin: vi.fn(async () => true),
		onOpenApprovals: vi.fn(),
		...overrides,
	};
	const view = render(<SessionMenu {...props} />);
	return { ...view, props };
}

describe("SessionMenu", () => {
	it("renders nothing while closed", () => {
		setup({ open: false });
		expect(screen.queryByRole("dialog")).toBeNull();
	});

	it("offers three labelled rows, in order, with their states", () => {
		setup();
		const dialog = screen.getByRole("dialog");
		expect(dialog.textContent).toContain("Session");
		/* Query by role/name, not by a layout class: the class is presentation,
		   and a row that stopped being `w-full` would silently drop out of this
		   assertion (R1-8). */
		const rows = screen.getAllByRole("button", {
			name: /^(Pin session|Unpin session|wide view|Approvals…)$/,
		});
		expect(rows.map((row) => row.getAttribute("aria-label") ?? row.textContent?.trim())).toEqual([
			"Pin session",
			"wide view",
			"Approvals…▸",
		]);

		const pin = screen.getByRole("button", { name: "Pin session" });
		expect(pin.textContent).toContain("Pin session");
		expect(pin.textContent).toContain("not pinned");
		expect(pin.getAttribute("aria-pressed")).toBe("false");

		const wide = screen.getByRole("button", { name: "wide view" });
		expect(wide.textContent).toContain("Wide view");
		expect(wide.textContent).toContain("off");
		expect(wide.getAttribute("aria-pressed")).toBe("false");
	});

	it("shows the pinned state as an unpin action, and delegates the press", async () => {
		const onTogglePin = vi.fn(async () => true);
		setup({ pinned: true, onTogglePin });
		const pin = screen.getByRole("button", { name: "Unpin session" });
		expect(pin.textContent).toContain("Unpin session");
		expect(pin.textContent).toContain("pinned");
		expect(pin.getAttribute("aria-pressed")).toBe("true");
		fireEvent.click(pin);
		await waitFor(() => expect(onTogglePin).toHaveBeenCalledTimes(1));
	});

	it("stays open when the pin is accepted", async () => {
		const onClose = vi.fn();
		setup({ onClose, onTogglePin: async () => true });
		fireEvent.click(screen.getByRole("button", { name: "Pin session" }));
		await waitFor(() => expect(onClose).not.toHaveBeenCalled());
	});

	it("steps aside on a refused pin so the header's own receipt is foreground", async () => {
		/* R1-3/U2: the refusal strip lives on the header, behind this modal. If
		   the sheet stayed open, the daemon's sentence was delivered to an inert
		   background layer. */
		const onClose = vi.fn();
		setup({ onClose, onTogglePin: async () => false });
		fireEvent.click(screen.getByRole("button", { name: "Pin session" }));
		await waitFor(() => expect(onClose).toHaveBeenCalledTimes(1));
	});

	it("toggles wide view and states it visibly and to assistive tech", () => {
		setup();
		const wide = screen.getByRole("button", { name: "wide view" });
		fireEvent.click(wide);
		expect(wide.getAttribute("aria-pressed")).toBe("true");
		expect(wide.textContent).toContain("on");
		expect(screen.getByRole("button", { name: "wide view" }).getAttribute("aria-pressed")).toBe(
			"true",
		);
		expect(document.documentElement.dataset.view).toBe("wide");
		expect(localStorage.getItem("lo-mobile-wide-view")).toBe("1");

		fireEvent.click(screen.getByRole("button", { name: "wide view" }));
		expect(document.documentElement.dataset.view).toBeUndefined();
		expect(localStorage.getItem("lo-mobile-wide-view")).toBeNull();
	});

	it("closes itself before asking for the gate sheet", () => {
		const onClose = vi.fn();
		const onOpenApprovals = vi.fn();
		setup({ onClose, onOpenApprovals });
		fireEvent.click(screen.getByRole("button", { name: /Approvals/ }));
		expect(onClose).toHaveBeenCalledTimes(1);
		expect(onOpenApprovals).toHaveBeenCalledTimes(1);
		expect(onClose.mock.invocationCallOrder[0]).toBeLessThan(
			onOpenApprovals.mock.invocationCallOrder[0],
		);
	});

	it("closes on Escape, the sheet's own keyboard escape", () => {
		const onClose = vi.fn();
		setup({ onClose });
		fireEvent.keyDown(window, { key: "Escape" });
		expect(onClose).toHaveBeenCalledTimes(1);
	});
});
