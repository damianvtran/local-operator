// @vitest-environment happy-dom
//
// The session menu: the header's three overflow rows, each naming itself and
// stating its current value. The header controls these replace were the
// operator's "four controls whose purpose is unclear": an unlabelled `? 4`, a
// bordered `[wide]` and a word-shaped `approvals`. These cells pin that every
// row carries a label, that the pin and wide rows state their state through
// both visible text and `aria-pressed`, and that Approvals closes this sheet
// before it asks for the gate sheet.
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
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
		onTogglePin: vi.fn(),
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

	it("offers three labelled full-width rows, in order, with their states", () => {
		setup();
		const dialog = screen.getByRole("dialog");
		expect(dialog.textContent).toContain("Session");
		const rows = Array.from(dialog.querySelectorAll("button")).filter((button) =>
			button.className.includes("w-full"),
		);
		expect(rows.map((row) => row.getAttribute("aria-label") ?? row.textContent?.trim())).toEqual([
			"pin this session",
			"wide view",
			"Approvals…▸",
		]);

		const pin = screen.getByRole("button", { name: "pin this session" });
		expect(pin.textContent).toContain("Pin session");
		expect(pin.textContent).toContain("not pinned");
		expect(pin.getAttribute("aria-pressed")).toBe("false");

		const wide = screen.getByRole("button", { name: "wide view" });
		expect(wide.textContent).toContain("Wide view");
		expect(wide.textContent).toContain("off");
		expect(wide.getAttribute("aria-pressed")).toBe("false");
	});

	it("shows the pinned state as an unpin action, and delegates the press", () => {
		const { props } = setup({ pinned: true });
		const pin = screen.getByRole("button", { name: "unpin this session" });
		expect(pin.textContent).toContain("Unpin session");
		expect(pin.textContent).toContain("pinned");
		expect(pin.getAttribute("aria-pressed")).toBe("true");
		fireEvent.click(pin);
		expect(props.onTogglePin).toHaveBeenCalledTimes(1);
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
