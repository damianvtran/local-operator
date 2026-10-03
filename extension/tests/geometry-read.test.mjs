/* The read actions' page-side functions: bounds, rounding, and the
 * SELF-CONTAINMENT the two-host sharing arrangement rests on.
 *
 * WHY THESE TESTS EXIST AT THIS LAYER. `driver/geometry-read.ts` is serialized
 * by BOTH hosts before it runs — `chrome.scripting.executeScript({func})` here,
 * `(<fn>.toString())(...)` in the desktop app — so the two properties that
 * matter cannot be seen in a diff: (1) every function must work when its body
 * is detached from this module, and (2) the caps must hold IN the page, because
 * the result is what crosses the boundary. A helper pulled out for tidiness, or
 * a cap lifted to a caller, compiles and passes every other suite and only
 * fails against a serialized copy.
 *
 * The fake DOM here is deliberately a fake and is named as one: it scripts
 * exactly the members the functions read (querySelectorAll, elementsFromPoint,
 * getComputedStyle, rects, inline style declarations), so a cap of 8 or a
 * 120-char class name can be crossed without a browser. A real-browser
 * exercise is the e2e rig's job (AGENTS.md, "Visual validation"); this is the
 * layer below it, where the arithmetic and the serialization contract live.
 */

import assert from "node:assert/strict";
import test from "node:test";
import { build } from "esbuild";
import { pathToFileURL } from "node:url";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

async function load(entry) {
  const dir = await mkdtemp(join(tmpdir(), "lop-geometry-read-"));
  const outfile = join(dir, "module.mjs");
  await build({ entryPoints: [entry], bundle: true, platform: "node", format: "esm", outfile });
  const loaded = await import(pathToFileURL(outfile));
  return { loaded, close: () => rm(dir, { recursive: true, force: true }) };
}

/** A CSSStyleDeclaration over a list of [name, value] pairs. */
function styleMap(entries) {
  return {
    length: entries.length,
    item: (index) => entries[index]?.[0],
    getPropertyValue: (name) => entries.find(([key]) => key === name)?.[1] ?? "",
  };
}

function element(overrides = {}) {
  const rect = overrides.rect ?? {
    x: 1.005,
    y: 2.674,
    top: 1.005,
    right: 11.005,
    bottom: 12.674,
    width: 10,
    height: 10.669,
  };
  return {
    tagName: overrides.tagName ?? "DIV",
    id: overrides.id ?? "",
    parentElement: overrides.parentElement ?? null,
    getAttribute: (name) =>
      name === "class"
        ? overrides.className ?? ""
        : name === "role"
          ? overrides.role ?? ""
          : name === "id"
            ? overrides.id ?? ""
            : null,
    getBoundingClientRect: () => rect,
    style: styleMap(overrides.inline ?? []),
  };
}

/** Install a fake document/getComputedStyle for the duration of `fn`. */
async function withFakeDom(dom, fn) {
  const priorDocument = globalThis.document;
  const priorGetComputedStyle = globalThis.getComputedStyle;
  globalThis.document = dom.document;
  globalThis.getComputedStyle = dom.getComputedStyle;
  try {
    return await fn();
  } finally {
    globalThis.document = priorDocument;
    globalThis.getComputedStyle = priorGetComputedStyle;
  }
}

/** Evaluate a function's `toString()` in GLOBAL scope — where the eval'd body
 *  cannot see this module's bindings — and call the copy. Any reference to a
 *  module-level helper or constant throws here, which is the point. */
function detached(fn) {
  return (0, eval)(`(${fn.toString()})`);
}

const COMPUTED = {
  getPropertyValue(name) {
    if (name === "display") return "block";
    if (name === "background-image") return "u".repeat(900); // pathological value
    return "1px";
  },
};

test("readStyles: caps, rounding, truncation and inline custom properties", async () => {
  const match = element({ tagName: "SECTION", id: "card", className: "x".repeat(200) });
  const seventh = [...Array(7)].map(() => match);
  const html = element({ tagName: "HTML" });
  const dom = {
    document: {
      documentElement: html,
      querySelector: () => match,
      querySelectorAll: () => seventh,
      elementsFromPoint: () => [match],
    },
    getComputedStyle: () => COMPUTED,
  };
  const module = await load("src/driver/geometry-read.ts");
  try {
    const result = await withFakeDom(dom, () => module.loaded.readStyles(".card", ["background-image"]));
    assert.equal(result.count, 5, "matches capped at five");
    assert.equal(result.matches.length, 5);
    assert.equal(result.truncated, true, "seven matches truncate to five");
    assert.equal(result.matches[0].rect.y, 2.67, "rect rounded to two decimals");
    assert.equal(result.matches[0].rect.height, 10.67);
    assert.equal(result.matches[0].className.length, 120, "class name capped at 120 chars");
    assert.ok(result.matches[0].className.endsWith("\u2026"), "truncation is visible");
    assert.equal(result.matches[0].styles["background-image"].length, 200, "values capped at 200");
    assert.equal(result.matches[0].styles.display, "block");
    assert.equal(result.matches[0].tag, "section");
    assert.equal(result.matches[0].id, "card");

    // Defaults + extras dedupe, and the combined list is capped at 30.
    const many = await withFakeDom(dom, () =>
      module.loaded.readStyles(".card", [
        "display", // already a default: deduped
        ...[...Array(19)].map((_, i) => `--extra-${i}`),
      ]),
    );
    const names = Object.keys(many.matches[0].styles);
    assert.equal(names.length, 30, "combined style list capped at 30");
    assert.equal(names.filter((name) => name === "display").length, 1, "deduped");
    // 20 defaults + 10 extras fills the list; later extras are dropped.
    assert.ok(names.includes("--extra-9") && !names.includes("--extra-10"), "extras clipped in order");

    // No matches -> null (the command layer maps it to element_not_found).
    dom.document.querySelectorAll = () => [];
    assert.equal(await withFakeDom(dom, () => module.loaded.readStyles(".none", [])), null);

    // Fewer than the cap -> not truncated.
    dom.document.querySelectorAll = () => [match, match];
    const small = await withFakeDom(dom, () => module.loaded.readStyles(".pair", []));
    assert.equal(small.count, 2);
    assert.equal(small.truncated, false);
  } finally {
    await module.close();
  }
});

test("readStyles: only the element's own inline custom properties are reported", async () => {
  const match = element({
    inline: [
      ["--brand", "#ff0000"],
      ["color", "red"],
      ["--long", "v".repeat(500)],
    ],
  });
  const dom = {
    document: { documentElement: element({ tagName: "HTML" }), querySelectorAll: () => [match] },
    getComputedStyle: () => COMPUTED,
  };
  const module = await load("src/driver/geometry-read.ts");
  try {
    const result = await withFakeDom(dom, () => module.loaded.readStyles(".x", []));
    assert.equal(result.matches[0].inline["--brand"], "#ff0000");
    assert.equal(result.matches[0].inline.color, undefined, "non-custom declarations excluded");
    assert.equal(result.matches[0].inline["--long"].length, 200, "inline values capped at 200");
  } finally {
    await module.close();
  }
});

test("hitTest: topmost-first stack, cap of 8, null off-viewport", async () => {
  const mk = (tag, id) => element({ tagName: tag, id });
  const stack = [mk("BUTTON", "b"), mk("A", "a"), mk("DIV", "d"), mk("UL", "u"), mk("LI", "l"), mk("SPAN", "s"), mk("BODY", ""), mk("HTML", ""), mk("EXTRA", "e")];
  const dom = {
    document: {
      documentElement: mk("HTML", ""),
      elementsFromPoint: (x, y) => (x < 0 ? [] : stack),
    },
    getComputedStyle: () => COMPUTED,
  };
  const module = await load("src/driver/geometry-read.ts");
  try {
    const result = await withFakeDom(dom, () => module.loaded.hitTest(10, 20));
    assert.equal(result.count, 8, "stack capped at eight");
    assert.deepEqual(
      result.elements.map((entry) => entry.id),
      ["b", "a", "d", "u", "l", "s", "", ""],
      "topmost first, in elementsFromPoint order",
    );
    assert.equal(result.elements[0].styles.display, "block");
    assert.ok(!("transform" in result.elements[0].styles), "hitTest carries its own small style set");

    assert.equal(await withFakeDom(dom, () => module.loaded.hitTest(-1, 10)), null, "empty stack -> null");
    assert.equal(await withFakeDom(dom, () => module.loaded.hitTest(Number.NaN, 10)), null, "non-finite -> null");
  } finally {
    await module.close();
  }
});

test("ancestors: chain to html inclusive, depth bound, default 12", async () => {
  const html = element({ tagName: "HTML" });
  const body = element({ tagName: "BODY", parentElement: html });
  const card = element({ tagName: "SECTION", id: "card", parentElement: body });
  const dom = {
    document: { documentElement: html, querySelector: () => card },
    getComputedStyle: () => COMPUTED,
  };
  const module = await load("src/driver/geometry-read.ts");
  try {
    const full = await withFakeDom(dom, () => module.loaded.ancestors("#card", 12));
    assert.deepEqual(full.chain.map((entry) => entry.tag), ["section", "body", "html"], "element -> html inclusive");
    assert.equal(full.count, 3);
    assert.ok("overflow-x" in full.chain[0].styles && "isolation" in full.chain[0].styles, "clip/layout style set");

    const shallow = await withFakeDom(dom, () => module.loaded.ancestors("#card", 2));
    assert.deepEqual(shallow.chain.map((entry) => entry.tag), ["section", "body"], "depth bounds the walk");

    const clamped = await withFakeDom(dom, () => module.loaded.ancestors("#card", 999));
    assert.equal(clamped.count, 3, "over-cap depth still walks to the top");
    const defaulted = await withFakeDom(dom, () => module.loaded.ancestors("#card"));
    assert.equal(defaulted.count, 3, "depth defaults when omitted");

    dom.document.querySelector = () => null;
    assert.equal(await withFakeDom(dom, () => module.loaded.ancestors(".none", 4)), null);
  } finally {
    await module.close();
  }
});

test("oversized tag, id, role and inline keys are bounded in every reader", async () => {
  // Rounds 1 and 3 review findings (inline, geometry-read.ts): `id`/`role`
  // shipped verbatim, then the inline map's property-name key, then `tag` —
  // the last unclipped page-controlled string in all three readers. A page can
  // put megabytes in any of them (a custom-element name has no grammar length
  // bound: `createElement` accepts a 10,000-char name), and the value crosses
  // isolated world -> worker -> daemon into `ToolResult.details`, where
  // BROWSER_TEXT_LIMIT_CHARS cannot bound it — the cap has to exist here.
  const LONG_TAG = "x".repeat(5000); // as `createElement(LONG_TAG)` accepts
  const LONG_ID = "i".repeat(5000);
  const LONG_ROLE = "r".repeat(5000);
  // CSSOM names are page-controlled the same way: whatever `setProperty`
  // accepted becomes a map key in the result.
  const LONG_PROPERTY = "--" + "n".repeat(5000);
  // Two names sharing their first 119 characters, plus a distinct prefix, for
  // the clipped-key dedup guard.
  const SHARED_PREFIX_A = "--" + "s".repeat(5000) + "-a";
  const SHARED_PREFIX_B = "--" + "s".repeat(5000) + "-b";
  const DISTINCT_PREFIX = "--" + "d".repeat(5000);
  const html = element({ tagName: "HTML" });
  const card = element({
    tagName: LONG_TAG,
    id: LONG_ID,
    role: LONG_ROLE,
    inline: [
      [LONG_PROPERTY, "v".repeat(500)],
      [SHARED_PREFIX_A, "first"],
      [SHARED_PREFIX_B, "second"],
      [DISTINCT_PREFIX, "third"],
    ],
    parentElement: html,
  });
  const dom = {
    document: {
      documentElement: html,
      querySelector: () => card,
      querySelectorAll: () => [card],
      elementsFromPoint: () => [card, html],
    },
    getComputedStyle: () => COMPUTED,
  };
  const module = await load("src/driver/geometry-read.ts");
  try {
    const styles = await withFakeDom(dom, () => module.loaded.readStyles(".card", []));
    const hit = await withFakeDom(dom, () => module.loaded.hitTest(5, 6));
    const ancestors = await withFakeDom(dom, () => module.loaded.ancestors("#card", 12));

    const shapes = [
      ["styles.matches[0]", styles.matches[0]],
      ["hit_test.elements[0]", hit.elements[0]],
      ["ancestors.chain[0]", ancestors.chain[0]],
    ];
    for (const [shape, entry] of shapes) {
      for (const field of ["tag", "id", "role", "className"]) {
        assert.ok(entry[field].length <= 120, `${shape}.${field} is bounded`);
      }
      assert.equal(entry.tag.length, 120, `${shape}.tag clipped to the identity cap`);
      assert.equal(entry.id.length, 120, `${shape}.id clipped to the identity cap`);
      assert.equal(entry.role.length, 120, `${shape}.role clipped to the identity cap`);
      assert.ok(entry.tag.endsWith("\u2026"), `${shape}.tag cut is visible`);
      assert.ok(entry.id.endsWith("\u2026"), `${shape}.id cut is visible`);
      assert.ok(entry.role.endsWith("\u2026"), `${shape}.role cut is visible`);
    }

    // The inline map's KEYS are the same class of channel: oversized names are
    // still REPORTED (clipped, not dropped), values stay under the value cap,
    // and two names sharing a cut prefix collapse to ONE deterministic key
    // (the first declaration wins) while a distinct prefix keeps its own.
    const inlineKeys = Object.keys(styles.matches[0].inline);
    assert.equal(inlineKeys.length, 3, "four names -> three keys after the cut");
    for (const key of inlineKeys) {
      assert.equal(key.length, 120, "every inline key sits at the identity cap");
      assert.ok(key.endsWith("\u2026"), "every inline key cut is visible");
    }
    const sharedKeys = inlineKeys.filter((key) => key.startsWith("--" + "s".repeat(117)));
    assert.equal(sharedKeys.length, 1, "names sharing a cut prefix collapse to one key");
    assert.equal(
      styles.matches[0].inline[sharedKeys[0]],
      "first",
      "deterministic: the first declaration wins",
    );
    assert.equal(
      styles.matches[0].inline["--" + "d".repeat(117) + "\u2026"],
      "third",
      "a distinct prefix keeps its own key",
    );
    const longPropertyKey = inlineKeys.find((key) => key.startsWith("--" + "n".repeat(117)));
    assert.equal(
      styles.matches[0].inline[longPropertyKey].length,
      200,
      "inline value still capped at 200",
    );
  } finally {
    await module.close();
  }
});

test("every reader survives a toString() round-trip in global scope", async () => {
  const module = await load("src/driver/geometry-read.ts");
  try {
    const html = element({ tagName: "HTML" });
    const card = element({ tagName: "DIV", id: "card", parentElement: html });
    const dom = {
      document: {
        documentElement: html,
        querySelector: () => card,
        querySelectorAll: () => [card, card],
        elementsFromPoint: () => [card, html],
      },
      getComputedStyle: () => COMPUTED,
    };
    await withFakeDom(dom, () => {
      const direct = {
        styles: module.loaded.readStyles(".card", ["color"]),
        hit: module.loaded.hitTest(5, 6),
        ancestors: module.loaded.ancestors("#card", 12),
      };
      const copy = {
        styles: detached(module.loaded.readStyles)(".card", ["color"]),
        hit: detached(module.loaded.hitTest)(5, 6),
        ancestors: detached(module.loaded.ancestors)("#card", 12),
      };
      // Deep-equality, so a detached copy that reads a module constant (which
      // would throw a ReferenceError in global scope) fails loudly, and one
      // that silently diverges fails too.
      assert.deepEqual(copy.styles, direct.styles);
      assert.deepEqual(copy.hit, direct.hit);
      assert.deepEqual(copy.ancestors, direct.ancestors);
    });
  } finally {
    await module.close();
  }
});
