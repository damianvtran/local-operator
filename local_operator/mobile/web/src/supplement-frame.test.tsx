// @vitest-environment happy-dom
//
// The frame's own half: the sandbox attributes, the reserved box, the
// second-`load` teardown, the watchdogs and the fallback line.
//
// WHAT THIS ENVIRONMENT CANNOT PROVE, stated once because it shapes the file:
// happy-dom fires TWO `load` events per iframe mount (measured here: a bare
// probe counts 2 for one `src` assignment), while a real engine fires one. The
// component's second-`load` rule is deliberately unconditional — a blocked
// navigation fires a second `load` when it is attempted after the first load
// (Chrome 155, the desktop lane's measurement, timed by QA round 1 Q2), so a
// rule that tolerated a second event would tolerate the attack — which means
// every happy-dom frame is torn down moments after it
// mounts. Tests that need a LIVE frame (the theme push, the nonce echo, the
// resize clamp applied to the box, a real `message` from `frame.contentWindow`)
// are therefore impossible here and their coverage is the browser rig recorded
// on the PR: the real relay page, a real Chrome, the vendored prelude. What is
// pinned here is each arm that ENDS a frame, and the shapes the host emits.
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { act } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { SupplementFrame } from "./components/supplement-frame";
import type { SupplementComponent } from "./lib/supplement";

afterEach(() => {
	cleanup();
	vi.useRealTimers();
});

const COMPONENT: SupplementComponent = {
	attachment: "a".repeat(32),
	title: "latency by hour",
	source: "data/bench.csv",
	mime: "text/html",
	heightHint: 200,
};

function frameOf(container: HTMLElement): HTMLIFrameElement {
	const iframe = container.querySelector("iframe");
	if (!iframe) throw new Error("the frame did not mount");
	return iframe;
}

describe("the host-side shapes", () => {
	it("mounts `sandbox=\"allow-scripts\"`, a data: URL, and no bridge attributes", () => {
		const { container } = render(
			<SupplementFrame component={COMPONENT} documentHtml="<html>ok</html>" />,
		);
		const iframe = frameOf(container);
		expect(iframe.getAttribute("sandbox")).toBe("allow-scripts");
		expect(iframe.getAttribute("src")?.startsWith("data:text/html;base64,")).toBe(true);
		// No bridge, no navigation aid, no permissions: the memo's relay row
		// forbids every privileged attribute, and the `data:` mount needs none.
		for (const attr of ["allow", "name", "csp", "allowfullscreen", "srcdoc"]) {
			expect(iframe.hasAttribute(attr), attr).toBe(false);
		}
		// `title` is the component's NAME, for tooling and accessibility — never
		// painted chrome.
		expect(iframe.getAttribute("title")).toBe("latency by hour");
	});

	it("holds the reserved box and no iframe until the document arrives", () => {
		const { container } = render(<SupplementFrame component={COMPONENT} documentHtml={null} />);
		expect(container.querySelector("iframe")).toBeNull();
		const box = container.querySelector<HTMLElement>("[data-supplement-frame]");
		expect(box).toBeTruthy();
		// The box holds the clamped hint so the transcript cannot reflow around
		// a document still in flight.
		expect(box?.style.height).toBe("200px");
	});

	it("paints the Source caption beside the frame (one line, display-stripped)", () => {
		// `documentHtml: null` deliberately: the caption belongs to the frame's
		// shape, not to a loaded document, so this asserts it without mounting an
		// iframe whose happy-dom `load` events would tear the component down.
		const { container } = render(<SupplementFrame component={COMPONENT} documentHtml={null} />);
		const caption = container.querySelector("[data-supplement-source]");
		expect(caption?.textContent).toBe("Source: data/bench.csv");
		expect(caption?.getAttribute("title")).toBe("Source: data/bench.csv");
	});
});

describe("the teardown arms", () => {
	it("a second load tears the frame down, paints the fallback and drops later messages", () => {
		const { container } = render(
			<SupplementFrame component={COMPONENT} documentHtml="<html>ok</html>" />,
		);
		const iframe = frameOf(container);
		fireEvent.load(iframe);
		expect(container.querySelector("iframe")).toBeTruthy();
		fireEvent.load(iframe);
		expect(container.querySelector("iframe")).toBeNull();
		const fallback = container.querySelector("[data-supplement-frame-fallback]");
		expect(fallback?.getAttribute("data-supplement-frame-fallback")).toBe("second-load");
		expect(fallback?.textContent).toContain("Couldn't render this graphic");
	});

	it("a new document on the same instance starts its own load count", () => {
		// Agent review round 1, R3: the counter must not carry across a
		// document swap on a reused instance — without the reset, the new
		// document's FIRST load counts as the hostile second one and tears
		// down a frame the host itself mounted.
		const { container, rerender } = render(
			<SupplementFrame component={COMPONENT} documentHtml="<html>one</html>" />,
		);
		fireEvent.load(frameOf(container));
			expect(container.querySelector("iframe")).toBeTruthy();
		rerender(<SupplementFrame component={COMPONENT} documentHtml="<html>two</html>" />);
		fireEvent.load(frameOf(container));
		expect(container.querySelector("iframe")).toBeTruthy();
		expect(container.querySelector("[data-supplement-frame-fallback]")).toBeNull();
	});

	it("fails a frame that never posts ready inside the watchdog", () => {
		vi.useFakeTimers();
		const { container } = render(
			<SupplementFrame component={COMPONENT} documentHtml="<html>ok</html>" />,
		);
		act(() => {
			vi.advanceTimersByTime(5001);
		});
		expect(container.querySelector("[data-supplement-frame-fallback]")).toBeTruthy();
		expect(container.querySelector("iframe")).toBeNull();
	});

	it("a document failure paints the fallback immediately, with Retry only when wired", () => {
		const onRetry = vi.fn();
		const { container } = render(
			<SupplementFrame
				component={COMPONENT}
				documentHtml={null}
				documentFailed={true}
				onRetry={onRetry}
			/>,
		);
		const fallback = container.querySelector("[data-supplement-frame-fallback]");
		expect(fallback?.getAttribute("data-supplement-frame-fallback")).toBe("document");
		fireEvent.click(screen.getByText("Retry"));
		expect(onRetry).toHaveBeenCalledTimes(1);
		cleanup();

		const bare = render(
			<SupplementFrame component={COMPONENT} documentHtml={null} documentFailed={true} />,
		);
		expect(bare.container.textContent).toBe("Couldn't render this graphic");
	});
});
