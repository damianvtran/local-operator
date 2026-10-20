// @vitest-environment happy-dom
//
// The conversation-tags sheet: the catalogue's three non-populated states, the
// exact desired-state writes, the cap, the primary fallback, the two-step
// clear, and the RACE RULE the surface exists to make safe.
//
// The race cases are the reason this file is separate from a render snapshot:
// a write's read-back must not paint over a list frame that arrived after the
// write started, and a frame that arrives mid-write must win over the draft.
// Both are only observable by holding a promise open and rerendering with a
// newer revision — which is exactly what the relay does when it wakes the list
// stream after a successful write.
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { SessionTagsSheet } from "./components/session-tags-sheet";
import { BUNDLED_OWNED } from "./i18n/bundled.gen";
import type { TagCatalogueEntry, TagRef } from "./types";

const getTags = vi.fn();
const setSessionTags = vi.fn();

vi.mock("./api", async (importOriginal) => {
	const actual = await importOriginal<typeof import("./api")>();
	return {
		...actual,
		getTags: () => getTags(),
		setSessionTags: (sessionId: string, tags: string[], primary: string | null) =>
			setSessionTags(sessionId, tags, primary),
	};
});

const CATALOGUE: TagCatalogueEntry[] = [
	{ id: "work", name: "Work", order: 0, conversations: 3 },
	{ id: "research", name: "Research", order: 1, conversations: 1 },
];

const tag = (id: string, name: string): TagRef => ({ id, name });
const WORK = tag("work", "Work");
const RESEARCH = tag("research", "Research");

function renderSheet(
	over: Partial<{
		assigned: TagRef[];
		primary: string | null;
		revision: number;
		onClose: () => void;
	}> = {},
) {
	const props = {
		open: true,
		onClose: () => {},
		sessionId: "s1",
		assigned: [] as TagRef[],
		primary: null as string | null,
		revision: 0,
		...over,
	};
	const view = render(<SessionTagsSheet {...props} />);
	return { view, props };
}

async function ready() {
	await screen.findByRole("checkbox", { name: "Work" });
}

beforeEach(() => {
	getTags.mockReset();
	setSessionTags.mockReset();
	getTags.mockResolvedValue({ tags: CATALOGUE });
	setSessionTags.mockImplementation(async (_id: string, tags: string[], primary: string | null) => ({
		session_id: "s1",
		tags: tags.map((id) => ({
			id,
			name: id === "work" ? "Work" : "Research",
		})),
		primary,
	}));
});

afterEach(() => {
	cleanup();
});

describe("the catalogue's states", () => {
	it("says it is loading while the registry read is in flight", () => {
		getTags.mockReturnValue(new Promise(() => {}));
		renderSheet();
		expect(screen.getByText("loading…")).toBeTruthy();
		expect(screen.queryByRole("checkbox")).toBeNull();
	});

	it("distinguishes an empty registry from an unresolved one", async () => {
		getTags.mockResolvedValue({ tags: [] });
		renderSheet();
		expect(await screen.findByText("no tag categories yet")).toBeTruthy();
		expect(
			screen.getByText("create one in the desktop app or the terminal"),
		).toBeTruthy();
		/* The empty state is NOT the loading state: no spinner copy survives it. */
		expect(screen.queryByText("loading…")).toBeNull();
		/* And it is still a state you can FINISH: Done lives outside the row
		   ladder, so an empty catalogue is not a sheet you escape only via ✕. */
		expect(screen.getByRole("button", { name: "done" })).toBeTruthy();
	});

	it("shows the failure and recovers through retry", async () => {
		getTags.mockRejectedValueOnce(new Error("502"));
		getTags.mockResolvedValueOnce({ tags: CATALOGUE });
		renderSheet();
		expect(await screen.findByText("Could not load the tag catalogue.")).toBeTruthy();
		fireEvent.click(screen.getByRole("button", { name: "try again" }));
		expect(await screen.findByRole("checkbox", { name: "Work" })).toBeTruthy();
	});

	it("keeps an assigned category the catalogue no longer carries, still selectable", async () => {
		getTags.mockResolvedValue({ tags: [CATALOGUE[0]] });
		renderSheet({
			assigned: [WORK, tag("gone", "Deleted category")],
			primary: "gone",
		});
		const kept = await screen.findByRole("checkbox", { name: /Deleted category/ });
		expect((kept as HTMLInputElement).checked).toBe(true);
		expect(screen.getByText("not in the catalogue")).toBeTruthy();
		expect(
			(screen.getByRole("radio", {
				name: "make Deleted category the primary tag",
			}) as HTMLInputElement).checked,
		).toBe(true);
	});
});

describe("assignment writes", () => {
	it("sends the complete desired state as {tags, primary} — null primary included", async () => {
		renderSheet();
		await ready();
		fireEvent.click(screen.getByRole("checkbox", { name: "Work" }));
		await waitFor(() =>
			expect(setSessionTags).toHaveBeenCalledWith("s1", ["work"], null),
		);
	});

	it("appends a second category in assignment order", async () => {
		renderSheet({ assigned: [WORK], primary: "work" });
		await ready();
		fireEvent.click(screen.getByRole("checkbox", { name: "Research" }));
		await waitFor(() =>
			expect(setSessionTags).toHaveBeenCalledWith(
				"s1",
				["work", "research"],
				"work",
			),
		);
	});

	it("marks a selected category primary", async () => {
		renderSheet({ assigned: [WORK, RESEARCH], primary: null });
		await ready();
		fireEvent.click(
			screen.getByRole("radio", { name: "make Research the primary tag" }),
		);
		await waitFor(() =>
			expect(setSessionTags).toHaveBeenCalledWith(
				"s1",
				["work", "research"],
				"research",
			),
		);
	});

	it("falls back to the first surviving assignment when the primary is removed", async () => {
		renderSheet({ assigned: [WORK, RESEARCH], primary: "work" });
		await ready();
		fireEvent.click(screen.getByRole("checkbox", { name: "Work" }));
		await waitFor(() =>
			expect(setSessionTags).toHaveBeenCalledWith("s1", ["research"], "research"),
		);
	});

	it("shows the pending mark on the row whose membership is in flight", async () => {
		setSessionTags.mockReturnValue(new Promise(() => {}));
		renderSheet();
		await ready();
		fireEvent.click(screen.getByRole("checkbox", { name: "Work" }));
		expect(await screen.findByText("saving…")).toBeTruthy();
	});

	it("paints the read-back once the write lands", async () => {
		renderSheet();
		await ready();
		const work = screen.getByRole("checkbox", { name: "Work" }) as HTMLInputElement;
		fireEvent.click(work);
		expect(work.checked).toBe(true);
		await waitFor(() => expect(setSessionTags).toHaveBeenCalled());
		await waitFor(() =>
			expect(
				(screen.getByRole("checkbox", { name: "Work" }) as HTMLInputElement).checked,
			).toBe(true),
		);
		expect(screen.queryByText("saving…")).toBeNull();
	});

	it("paints the relay's canonical answer, not an echo of the request", async () => {
		/* A DISCRIMINATING read-back: the mock echoes the request in every other
		   write test, so deleting the `setDraft(answer)` line left them green.
		   Here the store answers with a DIFFERENT assignment (assignment order
		   normalised elsewhere, possibly a fallback primary): the sheet must paint
		   the answer, and the request's own ids must not survive. */
		setSessionTags.mockResolvedValueOnce({
			session_id: "s1",
			tags: [RESEARCH],
			primary: "research",
		});
		renderSheet();
		await ready();
		fireEvent.click(screen.getByRole("checkbox", { name: "Work" }));
		await waitFor(() =>
			expect(
				(screen.getByRole("checkbox", { name: "Research" }) as HTMLInputElement)
					.checked,
			).toBe(true),
		);
		expect(
			(screen.getByRole("checkbox", { name: "Work" }) as HTMLInputElement).checked,
		).toBe(false);
		expect(
			(screen.getByRole("radio", {
				name: "make Research the primary tag",
			}) as HTMLInputElement).checked,
		).toBe(true);
	});
});

describe("the primary column", () => {
	it("labels the column once a tag is selected", async () => {
		renderSheet({ assigned: [WORK], primary: "work" });
		await ready();
		expect(screen.getByText("primary")).toBeTruthy();
	});

	it("hides the legend when nothing is selected", async () => {
		renderSheet();
		await ready();
		expect(screen.queryByText("primary")).toBeNull();
	});

	it("gives the radio a 44px box outside the checkbox label", async () => {
		renderSheet({ assigned: [WORK], primary: "work" });
		await ready();
		const radio = screen.getByRole("radio", {
			name: "make Work the primary tag",
		});
		/* The disc stays 16px; the box around it is the touch target. */
		expect(radio.className).toContain("size-4");
		const box = radio.closest("label");
		expect(box).not.toBeNull();
		expect(box?.className).toContain("size-11");
		/* A press in that box must mean ONE thing: the radio lives outside the
		   checkbox's label, so it cannot also toggle the row's membership. */
		const checkboxLabel = screen
			.getByRole("checkbox", { name: "Work" })
			.closest("label");
		expect(checkboxLabel).not.toBeNull();
		expect(checkboxLabel?.contains(radio)).toBe(false);
	});
});

describe("a target change while a write is in flight", () => {
	function props(over: Partial<{
		open: boolean;
		sessionId: string;
		assigned: TagRef[];
		primary: string | null;
	}> = {}) {
		return {
			open: true,
			onClose: () => {},
			sessionId: "a",
			assigned: [WORK] as TagRef[],
			primary: "work" as string | null,
			revision: 0,
			...over,
		};
	}

	async function openB(view: ReturnType<typeof render>) {
		view.rerender(<SessionTagsSheet {...props({ open: false })} />);
		view.rerender(
			<SessionTagsSheet
				{...props({ sessionId: "b", assigned: [], primary: null })}
			/>,
		);
		await screen.findByRole("checkbox", { name: "Work" });
	}

	it("drops row A's late read-back after the sheet reopens on row B", async () => {
		let resolveWrite!: (value: {
			session_id: string;
			tags: TagRef[];
			primary: string | null;
		}) => void;
		setSessionTags.mockImplementationOnce(
			() =>
				new Promise((resolve) => {
					resolveWrite = resolve;
				}),
		);
		const view = render(<SessionTagsSheet {...props()} />);
		await ready();
		fireEvent.click(screen.getByRole("checkbox", { name: "Research" }));
		expect(await screen.findByText("saving…")).toBeTruthy();

		await openB(view);
		/* A's answer lands only now: it must not set B's draft or B's error. */
		await act(async () => {
			resolveWrite({ session_id: "a", tags: [WORK, RESEARCH], primary: "work" });
		});
		expect(
			(screen.getByRole("checkbox", { name: "Work" }) as HTMLInputElement).checked,
		).toBe(false);
		expect(
			(screen.getByRole("checkbox", { name: "Research" }) as HTMLInputElement)
				.checked,
		).toBe(false);
		expect(screen.getByRole("alert").textContent).toBe("");

		/* And B's next press sends B's state, not A's answer. */
		fireEvent.click(screen.getByRole("checkbox", { name: "Work" }));
		await waitFor(() =>
			expect(setSessionTags).toHaveBeenLastCalledWith("b", ["work"], null),
		);
	});

	it("drops row A's late refusal after the sheet reopens on row B", async () => {
		const { HttpError } = await import("./api");
		let rejectWrite!: (error: unknown) => void;
		setSessionTags.mockImplementationOnce(
			() =>
				new Promise((_resolve, reject) => {
					rejectWrite = reject;
				}),
		);
		const view = render(<SessionTagsSheet {...props()} />);
		await ready();
		fireEvent.click(screen.getByRole("checkbox", { name: "Research" }));
		expect(await screen.findByText("saving…")).toBeTruthy();

		await openB(view);
		await act(async () => {
			rejectWrite(new HttpError(422, "Unknown category id"));
		});
		expect(screen.getByRole("alert").textContent).toBe("");
		expect(
			(screen.getByRole("checkbox", { name: "Research" }) as HTMLInputElement)
				.checked,
		).toBe(false);

		fireEvent.click(screen.getByRole("checkbox", { name: "Work" }));
		await waitFor(() =>
			expect(setSessionTags).toHaveBeenLastCalledWith("b", ["work"], null),
		);
	});
});

describe("the cap", () => {
	const eight = Array.from({ length: 8 }, (_, index) =>
		tag(`t${index}`, `Category ${index}`),
	);
	const nine: TagCatalogueEntry[] = Array.from({ length: 9 }, (_, index) => ({
		id: `t${index}`,
		name: `Category ${index}`,
		order: index,
		conversations: 0,
	}));

	it("disables unchecked rows, explains why, and writes nothing for a blocked press", async () => {
		getTags.mockResolvedValue({ tags: nine });
		renderSheet({ assigned: eight, primary: "t0" });
		const blocked = (await screen.findByRole("checkbox", {
			name: "Category 8",
		})) as HTMLInputElement;
		expect(blocked.disabled).toBe(true);
		const hint = screen.getByText(
			"8 tags is the most a conversation can carry — remove one to add another.",
		);
		expect(blocked.getAttribute("aria-describedby")).toBe(hint.id);
		fireEvent.click(blocked);
		expect(setSessionTags).not.toHaveBeenCalled();
	});
});

describe("the two-step clear", () => {
	it("writes nothing on the first press and the exact empty state on the second", async () => {
		renderSheet({ assigned: [WORK], primary: "work" });
		await ready();
		fireEvent.click(screen.getByRole("button", { name: "remove all tags" }));
		expect(setSessionTags).not.toHaveBeenCalled();
		fireEvent.click(screen.getByRole("button", { name: "remove all tags?" }));
		await waitFor(() =>
			expect(setSessionTags).toHaveBeenCalledWith("s1", [], null),
		);
	});

	it("can cancel the confirmation without writing", async () => {
		renderSheet({ assigned: [WORK], primary: "work" });
		await ready();
		fireEvent.click(screen.getByRole("button", { name: "remove all tags" }));
		fireEvent.click(screen.getByRole("button", { name: "cancel" }));
		expect(setSessionTags).not.toHaveBeenCalled();
		expect(screen.getByRole("button", { name: "remove all tags" })).toBeTruthy();
	});
});

describe("a refusal", () => {
	it("reverts to the newest frame and shows the relay's own sentence", async () => {
		const { HttpError } = await import("./api");
		setSessionTags.mockRejectedValueOnce(new HttpError(422, "Unknown category id"));
		renderSheet();
		await ready();
		const work = screen.getByRole("checkbox", { name: "Work" }) as HTMLInputElement;
		fireEvent.click(work);
		await waitFor(() =>
			expect(screen.getByRole("alert").textContent).toBe(
				"Could not save the tags: Unknown category id",
			),
		);
		expect(
			(screen.getByRole("checkbox", { name: "Work" }) as HTMLInputElement).checked,
		).toBe(false);
	});
});

describe("the frame race", () => {
	it("lets a newer authoritative frame clear the pending draft", async () => {
		let resolveWrite!: (value: {
			session_id: string;
			tags: TagRef[];
			primary: string | null;
		}) => void;
		setSessionTags.mockReturnValue(
			new Promise((resolve) => {
				resolveWrite = resolve;
			}),
		);
		const { view, props } = renderSheet();
		await ready();
		fireEvent.click(screen.getByRole("checkbox", { name: "Work" }));
		expect(
			(screen.getByRole("checkbox", { name: "Work" }) as HTMLInputElement).checked,
		).toBe(true);

		/* The relay woke the list stream: a frame newer than the write started
		   carries the store's answer and the draft must yield to it. */
		view.rerender(
			<SessionTagsSheet
				{...props}
				revision={1}
				assigned={[RESEARCH]}
				primary="research"
			/>,
		);
		expect(
			(screen.getByRole("checkbox", { name: "Work" }) as HTMLInputElement).checked,
		).toBe(false);
		expect(screen.queryByText("saving…")).toBeNull();

		/* The write's own read-back is OLDER than that frame: it must not paint
		   over it. */
		resolveWrite({ session_id: "s1", tags: [WORK], primary: "work" });
		await waitFor(() =>
			expect(
				(screen.getByRole("checkbox", { name: "Work" }) as HTMLInputElement).checked,
			).toBe(false),
		);
		expect(
			(screen.getByRole("checkbox", { name: "Research" }) as HTMLInputElement).checked,
		).toBe(true);
	});

	it("keeps the pending draft when no newer frame arrived", async () => {
		setSessionTags.mockReturnValue(new Promise(() => {}));
		const { view, props } = renderSheet();
		await ready();
		fireEvent.click(screen.getByRole("checkbox", { name: "Work" }));
		/* A rerender from an unrelated reason is not a newer frame: the same
		   revision, so the draft stands. */
		view.rerender(<SessionTagsSheet {...props} />);
		expect(
			(screen.getByRole("checkbox", { name: "Work" }) as HTMLInputElement).checked,
		).toBe(true);
	});
});

describe("copy and storage", () => {
	it("ships every tag key in the English bundle", () => {
		const messages = BUNDLED_OWNED["relay.session"] ?? {};
		const keys = Object.keys(messages).filter((key) =>
			key.startsWith("relay.session.tags."),
		);
		expect(keys).toHaveLength(18);
		for (const key of keys) expect(messages[key]).not.toBe("");
	});

	it("keeps no local store: the relay's assignment is the only one", async () => {
		renderSheet();
		await ready();
		fireEvent.click(screen.getByRole("checkbox", { name: "Work" }));
		await waitFor(() => expect(setSessionTags).toHaveBeenCalled());
		expect(localStorage.length).toBe(0);
		expect(sessionStorage.length).toBe(0);
	});
});
