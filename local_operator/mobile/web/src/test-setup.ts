// Shared vitest setup for the mobile web suite.
//
// happy-dom 20.x attaches `localStorage`/`sessionStorage` to the Window via
// non-enumerable own getters. vitest's `populateGlobal` copies keys with
// `Object.keys(win)`, which skips non-enumerable properties — so on newer
// Node versions `window.localStorage` is simply absent and every component
// or module that touches it at mount time crashes before any assertion runs
// (Composer reads it via continuation-command.ts; older Node happened to
// tolerate this). Re-attach happy-dom's own real Storage instances so the
// suite exercises production code paths on any supported Node, instead of
// silently depending on the ambient runtime.
import { Storage } from "happy-dom";

const local = new Storage();
const session = new Storage();

Object.defineProperty(globalThis, "localStorage", {
	value: local,
	configurable: true,
	writable: true,
});
Object.defineProperty(globalThis, "sessionStorage", {
	value: session,
	configurable: true,
	writable: true,
});
if (typeof window !== "undefined") {
	Object.defineProperty(window, "localStorage", { value: local, configurable: true });
	Object.defineProperty(window, "sessionStorage", { value: session, configurable: true });
}
