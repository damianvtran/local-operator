// @vitest-environment happy-dom
import { act, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { useKeyboardUp } from "./use-keyboard-up";

const real = Object.getOwnPropertyDescriptor(window, "visualViewport");
afterEach(() => {
	if (real) Object.defineProperty(window, "visualViewport", real);
	else delete (window as { visualViewport?: unknown }).visualViewport;
});

function stub(height: number, scale: number) {
	const listeners: Array<() => void> = [];
	const vv = {
		height,
		scale,
		addEventListener: (_: string, fn: () => void) => listeners.push(fn),
		removeEventListener: () => {},
	};
	Object.defineProperty(window, "visualViewport", { value: vv, configurable: true });
	return { vv, fire: () => act(() => listeners.forEach((fn) => fn())) };
}

describe("useKeyboardUp", () => {
	it("reads a 300 px shorter visual viewport as a keyboard", () => {
		stub(window.innerHeight - 300, 1);
		expect(renderHook(() => useKeyboardUp()).result.current).toBe(true);
	});

	it("does not read a pinch-zoom as a keyboard", () => {
		// At scale 2 the visual viewport is half the page tall with no keyboard in play.
		const { vv, fire } = stub(window.innerHeight / 2, 2);
		const hook = renderHook(() => useKeyboardUp());
		expect(hook.result.current).toBe(false);
		// A zoom change after mount is no keyboard either.
		vv.scale = 3;
		vv.height = window.innerHeight / 3;
		fire();
		expect(hook.result.current).toBe(false);
	});

	it("is false where the browser has no visualViewport", () => {
		Object.defineProperty(window, "visualViewport", { value: undefined, configurable: true });
		expect(renderHook(() => useKeyboardUp()).result.current).toBe(false);
	});
});
