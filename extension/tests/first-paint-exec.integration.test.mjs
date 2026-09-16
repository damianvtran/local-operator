/* EXECUTE the pre-paint script, rather than pattern-match it.
 *
 * WHY THIS FILE EXISTS. `first-paint.js` is loaded as a CLASSIC render-blocking
 * script in <head> and is copied verbatim into `dist/` (build.mjs), so nothing
 * in the toolchain parses it: no tsc, no bundler, no test. A rebase on this
 * branch joined `var KEY = "lop:pin-hint";` onto the end of the preceding `//`
 * comment, which made the declaration part of the comment. `localStorage.
 * getItem(KEY)` then threw a ReferenceError into the surrounding try/catch
 * (which exists to survive disabled storage) and the script silently painted the
 * pairing pin for every state — the 340px→207px flash the file's own header says
 * it exists to remove. Every gate was green: the jitter test only asserted
 * `source.includes('"lop:pin-hint"')`, which a comment satisfies.
 *
 * So these rows RUN the script against a stubbed DOM and storage, and assert
 * (a) the hint key was actually READ — the sharper signal, because an undeclared
 * identifier means getItem is never called at all — and (b) the min-height rule
 * it emitted. Reverting the declaration turns both red.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const SCRIPT = join(HERE, "..", "src", "popup", "first-paint.js");

/** Run the script the way <head> does: as a classic script, against a stub.
 *
 * Its own guard cannot hide a failure from us — the try/catch swallows a
 * ReferenceError by design — so the observable is what it WROTE and what it
 * READ, which is exactly what a real popup's first frame depends on. */
async function runFirstPaint(storage = {}) {
  const source = await readFile(SCRIPT, "utf8");
  const reads = [];
  const writes = [];
  const styles = [];
  const previous = {
    localStorage: Object.getOwnPropertyDescriptor(globalThis, "localStorage"),
    document: globalThis.document,
  };
  globalThis.localStorage = {
    getItem: (key) => {
      reads.push(key);
      return Object.hasOwn(storage, key) ? storage[key] : null;
    },
    setItem: (key, value) => writes.push([key, value]),
  };
  globalThis.document = {
    createElement: () => ({ textContent: "" }),
    documentElement: { appendChild: (node) => styles.push(node) },
  };
  try {
    // `new Function` rather than `import()`: the file is a classic script with
    // no exports, and importing it would make it a module — which is the very
    // deferral its header explains it must not become.
    new Function(source)();
  } finally {
    if (previous.localStorage) {
      Object.defineProperty(globalThis, "localStorage", previous.localStorage);
    } else {
      delete globalThis.localStorage;
    }
    globalThis.document = previous.document;
  }
  return { reads, writes, css: styles.map((s) => s.textContent).join("") };
}

const minHeight = (css) => /#pending\{min-height:([^}]+)\}/.exec(css)?.[1];

test("the pre-paint honours the stored pin (and actually reads it)", async () => {
  // The connected pin, as popup.ts writes it after a paired render. 148px, not
  // the 86px this test shipped with: the Connected card reserves the update
  // advisory's slot (popup.css), which grew the card 207.16 -> 269.16px, and the
  // pin moved with it. The number is the one popup.ts writes and first-paint.js
  // pre-paints — see PIN_CONNECTED in each.
  const { reads, css } = await runFirstPaint({ "lop:pin-hint": "148px" });
  assert.ok(
    reads.includes("lop:pin-hint"),
    `the script must READ the hint key; it read ${JSON.stringify(reads)} — an undeclared KEY throws before getItem is called and the catch swallows it`,
  );
  assert.equal(
    minHeight(css),
    "148px",
    "a browser pinned to the connected card must pre-paint at that height, not the pairing card's",
  );
});

test("the pre-paint honours the standby pin, a state this branch adds", async () => {
  const { css } = await runFirstPaint({ "lop:pin-hint": "193px" });
  assert.equal(minHeight(css), "193px");
});

test("an unrecognised hint falls back rather than pinning something unmeasured", async () => {
  const { reads, css } = await runFirstPaint({ "lop:pin-hint": "999px" });
  assert.ok(
    reads.includes("lop:pin-hint"),
    "a stale value must still be read and rejected, not skipped by an undeclared identifier",
  );
  assert.equal(minHeight(css), "219px", "the pairing pin is the documented fallback");
});

test("the legacy boolean key still pre-paints the connected card", async () => {
  // Read once so the key rename did not cost an existing paired browser a
  // resize; with B1's undeclared KEY this fallback was dead for the same reason
  // the primary hint was.
  const { reads, css } = await runFirstPaint({ "lop:paired-hint": "1" });
  assert.ok(reads.includes("lop:paired-hint"), `legacy key not read: ${JSON.stringify(reads)}`);
  assert.equal(minHeight(css), "148px");
});

test("no hint at all is the pairing pin, and storage that throws is survived", async () => {
  assert.equal(minHeight((await runFirstPaint({})).css), "219px");
  const source = await readFile(SCRIPT, "utf8");
  globalThis.localStorage = {
    getItem: () => {
      throw new Error("storage is disabled");
    },
    setItem: () => {},
  };
  try {
    globalThis.document = { createElement: () => ({ textContent: "" }), documentElement: { appendChild: () => {} } };
    new Function(source)();
  } finally {
    delete globalThis.localStorage;
    delete globalThis.document;
  }
});
