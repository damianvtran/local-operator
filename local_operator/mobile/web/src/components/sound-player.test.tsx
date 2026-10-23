// @vitest-environment happy-dom
//
// THE shared mobile player's gesture contract, its one-at-a-time registry, and
// its two waveform states.
//
// The real component is used (no mock), because the contract under test is its
// own: a click seeks then plays, a drag only seeks, `ended` resets, and the
// registry pauses whatever was audible before. Media methods are stubbed —
// happy-dom implements the elements' shape, not their playback — so what is
// asserted here is the COMPONENT's decisions, never the browser's decoder.
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SoundPlayer } from "./sound-player";

beforeEach(() => {
	vi.spyOn(HTMLMediaElement.prototype, "play").mockResolvedValue(undefined);
	vi.spyOn(HTMLMediaElement.prototype, "pause").mockImplementation(() => {});
});

afterEach(() => {
	cleanup();
	vi.restoreAllMocks();
});

const PEAKS = new Array(240).fill(0.5);

/** The track reports a 100 px box so the fraction math is legible. */
function track(): HTMLElement {
	const element = document.querySelector<HTMLElement>("[data-sound-track]");
	if (!element) throw new Error("no track rendered");
	element.getBoundingClientRect = () =>
		({ left: 0, top: 0, width: 100, height: 44, right: 100, bottom: 44, x: 0, y: 0 }) as DOMRect;
	return element;
}

function audio(): HTMLAudioElement {
	const element = document.querySelector("audio");
	if (!element) throw new Error("no audio element rendered");
	return element;
}

function clock(): string {
	return document.querySelector("[data-sound-clock]")?.textContent ?? "";
}

describe("sound-player", () => {
	it("seeks then plays on a click", async () => {
		render(<SoundPlayer src="/sound?a" peaks={PEAKS} label="clip" durationHint={10} />);
		fireEvent.pointerDown(track(), { clientX: 25, pointerId: 1 });
		fireEvent.pointerUp(track(), { clientX: 25, pointerId: 1 });

		expect(clock()).toBe("0:02");
		await waitFor(() => expect(HTMLMediaElement.prototype.play).toHaveBeenCalledTimes(1));
	});

	it("seeks on a drag and never starts playback", async () => {
		render(<SoundPlayer src="/sound?a" peaks={PEAKS} label="clip" durationHint={10} />);
		fireEvent.pointerDown(track(), { clientX: 10, pointerId: 1 });
		fireEvent.pointerMove(track(), { clientX: 60, pointerId: 1 });
		fireEvent.pointerUp(track(), { clientX: 60, pointerId: 1 });

		expect(clock()).toBe("0:06");
		expect(HTMLMediaElement.prototype.play).not.toHaveBeenCalled();
	});

	it("a pause press pauses, and a second press resumes", async () => {
		render(<SoundPlayer src="/sound?a" peaks={PEAKS} label="clip" durationHint={10} />);
		const button = screen.getByRole("button");
		fireEvent.click(button);
		await waitFor(() => expect(button.getAttribute("aria-pressed")).toBe("true"));
		fireEvent.click(button);
		expect(HTMLMediaElement.prototype.pause).toHaveBeenCalled();
	});

	it("resets the control when the sound ends", async () => {
		render(<SoundPlayer src="/sound?a" peaks={PEAKS} label="clip" durationHint={10} />);
		const button = screen.getByRole("button");
		fireEvent.click(button);
		await waitFor(() => expect(button.getAttribute("aria-pressed")).toBe("true"));

		fireEvent.ended(audio());
		expect(button.getAttribute("aria-pressed")).toBe("false");
		expect(clock()).toBe("0:00");
	});

	it("pauses the previously playing sound when a second one starts", async () => {
		render(
			<>
				<SoundPlayer src="/sound?a" peaks={PEAKS} label="first" durationHint={10} />
				<SoundPlayer src="/sound?b" peaks={PEAKS} label="second" durationHint={10} />
			</>,
		);
		const [first, second] = screen.getAllByRole("button");
		fireEvent.click(first);
		await waitFor(() => expect(first.getAttribute("aria-pressed")).toBe("true"));
		fireEvent.click(second);
		/* The registry pauses the FIRST element before the second plays. */
		expect(HTMLMediaElement.prototype.pause).toHaveBeenCalled();
	});

	it("marks a null peaks list as the flat scrubber, never a fabricated envelope", () => {
		const { container } = render(
			<SoundPlayer src="/sound?a" peaks={null} label="clip" durationHint={10} />,
		);
		const player = container.querySelector("[data-sound-player]");
		expect(player?.getAttribute("data-waveform")).toBe("flat");
	});
});
