// @vitest-environment happy-dom
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { createHash } from "node:crypto";
import brandIcon from "./assets/brand-icon.png?inline";
import { afterEach, describe, expect, it, vi } from "vitest";
import { SessionListScreen } from "./screens/session-list";

vi.mock("./store", async (importOriginal) => {
	const actual = await importOriginal<typeof import("./store")>();
	return { ...actual, useSessions: () => ({ sessions: [], connected: true, revision: 0 }), retainSessionListStream: () => () => {} };
});
vi.mock("./api", async (importOriginal) => {
	const actual = await importOriginal<typeof import("./api")>();
	return { ...actual, getDirectories: vi.fn(async () => ({ home: "", recent: [] })) };
});
afterEach(() => { cleanup(); vi.clearAllMocks(); document.documentElement.removeAttribute("dir"); });

function expectChromeIcon(button: HTMLElement) {
	expect(button.className).toContain("size-11");
	const icon = button.querySelector("svg");
	expect(icon).not.toBeNull();
	// Operator-reviewed optical size: 20px is larger than the legacy text glyph
	// without the oversized 24px chrome silhouette. The path geometry remains in
	// a 24-unit viewBox, so search's 22-unit stroked diagonal renders at 18.33px.
	expect(icon?.getAttribute("width")).toBe("20");
	expect(icon?.getAttribute("height")).toBe("20");
	expect(icon?.getAttribute("viewBox")).toBe("0 0 24 24");
	expect(icon?.getAttribute("aria-hidden")).toBe("true");
}

describe("conversation chrome", () => {
	it("edge-cases the white disc so the circle is discernible on a light ground", () => {
		// The artwork is a white disc on transparency: on the near-white drawer it
		// reads as a bare dark glyph unless the CONTAINER draws an edge. The bytes
		// stay untouched; the ring uses the existing control-boundary token, which
		// carries the contrast floor on both themes.
		const view = render(<SessionListScreen />);
		const mark = view.container.querySelector("header img[data-brand-mark]")!;
		expect(mark.className).toContain("rounded-full");
		expect(mark.className).toContain("ring-1");
		expect(mark.className).toContain("ring-control");
		expect(mark.getAttribute("width")).toBe("24");
		expect(mark.getAttribute("height")).toBe("24");
	});
	it("uses the exact reviewed white-circle brand bytes, not a recoloured legacy motif", () => {
		expect(brandIcon.startsWith("data:image/png;base64,")).toBe(true);
		const bytes = Buffer.from(brandIcon.split(",")[1], "base64");
		expect(createHash("sha256").update(bytes).digest("hex")).toBe("e438b0430a45a6b42c4e1a49ac8b52c5e5fdefe0955251d94abfbcc0217f7113");
	});

	it("pairs visibly sized search/menu icons with independent 44px targets and a brand mark", () => {
		const view = render(<SessionListScreen />);
		expectChromeIcon(screen.getByRole("button", { name: "Open navigation" }));
		expectChromeIcon(screen.getByRole("button", { name: /search/i }));
		const header = view.container.querySelector("header")!;
		expect(header.className).toContain("52px");
		expect(header.querySelector("img[data-brand-mark]")).not.toBeNull();
		expect(header.querySelector("h1")?.className).toContain("truncate");
	});

	it("keeps the same restrained mark and crisp close control in the modal drawer", () => {
		const view = render(<SessionListScreen />);
		const mark = view.container.querySelector("header img[data-brand-mark]");
		fireEvent.click(screen.getByRole("button", { name: "Open navigation" }));
		const drawer = screen.getByRole("dialog");
		expect(drawer.getAttribute("aria-modal")).toBe("true");
		const drawerMark = drawer.querySelector("img[data-brand-mark]");
		expect(drawerMark).not.toBeNull();
		expect(drawerMark?.getAttribute("src")).toBe(mark?.getAttribute("src"));
		expectChromeIcon(screen.getByRole("button", { name: "Close navigation" }));
	});

	it("keeps the drawer's Escape dismissal and focus return", async () => {
		render(<SessionListScreen />);
		const trigger = screen.getByRole("button", { name: "Open navigation" });
		trigger.focus();
		fireEvent.click(trigger);
		await waitFor(() => expect(document.activeElement).toBe(screen.getByRole("button", { name: "Close navigation" })));
		fireEvent.keyDown(window, { key: "Escape" });
		await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
		expect(document.activeElement).toBe(trigger);
	});

	it("preserves search autofocus, clearing and semantic controls in RTL", async () => {
		document.documentElement.dir = "rtl";
		const view = render(<SessionListScreen />);
		fireEvent.click(screen.getByRole("button", { name: /search/i }));
		const input = view.container.querySelector("header input")!;
		await waitFor(() => expect(document.activeElement).toBe(input));
		fireEvent.change(input, { target: { value: "A long search query" } });
		expectChromeIcon(screen.getByRole("button", { name: /search/i }));
		fireEvent.click(screen.getByRole("button", { name: /search/i }));
		expect(view.container.querySelector("header input")).toBeNull();
		expect(screen.getByRole("button", { name: "Open navigation" }).getAttribute("aria-expanded")).toBe("false");
		fireEvent.click(screen.getByRole("button", { name: /search/i }));
		expect(view.container.querySelector<HTMLInputElement>("header input")?.value).toBe("");
	});
});
