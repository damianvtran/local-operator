// @vitest-environment happy-dom
//
// The video attachment's three designed states, its fetch shape, and the
// surface's no-spinner rule.
//
// The REAL api module is used (not a mock) so the src assertion pins the
// actual URL builder — the route contract the daemon answers is exactly the
// thing this component exists to reach.
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { AttachmentVideo } from "./attachment-video";

afterEach(cleanup);

function video(): HTMLVideoElement {
	const el = document.querySelector("video");
	if (!el) throw new Error("no video element rendered");
	return el;
}

describe("attachment-video", () => {
	it("mounts <video> with the attachment URL, controls and metadata-only preload", () => {
		render(<AttachmentVideo pid="p9" entryId="e1" index={0} />);
		const box = screen.getByTestId("attachment-video");
		expect(box.getAttribute("data-state")).toBe("resolving");
		const el = video();
		expect(el.getAttribute("src")).toBe(
			"/api/sessions/p9/attachment?entry=e1&i=0&kind=video",
		);
		expect(el.hasAttribute("controls")).toBe(true);
		expect(el.getAttribute("preload")).toBe("metadata");
		/* The sibling image path declares `alt="attachment"`; the player
		   carries the equivalent accessible name (design round 1, D1). */
		expect(el.getAttribute("aria-label")).toBe("attachment");
	});

	it("resolves without a spinner: the reserved box is the whole resolving state", () => {
		const { container } = render(
			<AttachmentVideo pid="p9" entryId="e1" index={0} />,
		);
		expect(container.querySelector(".lo-spinner")).toBeNull();
		expect(screen.queryByText(/loading/i)).toBeNull();
		/* The element itself is hidden while resolving: the media stack draws
		   its own load UI into a buffering <video>, and that spinner must
		   not be this surface's resolving face (the no-spinner rule). */
		expect(video().className).toContain("invisible");
	});

	it("flips to ready when metadata lands", () => {
		render(<AttachmentVideo pid="p9" entryId="e1" index={0} />);
		fireEvent.loadedMetadata(video());
		expect(
			screen.getByTestId("attachment-video").getAttribute("data-state"),
		).toBe("ready");
		expect(video().className).not.toContain("invisible");
	});

	it("missing bytes degrade to the same shape an unavailable image uses", () => {
		render(<AttachmentVideo pid="p9" entryId="e1" index={2} />);
		fireEvent.error(video());
		const box = screen.getByTestId("attachment-video");
		expect(box.getAttribute("data-state")).toBe("error");
		expect(box.textContent).toContain("video unavailable");
		expect(box.textContent).toContain("⊘");
		expect(document.querySelector("video")).toBeNull();
	});
});
