/* `queryElements`: the read-only DOM answer, its caps, and its ONE safety rule.
 *
 * WHY THIS LAYER. The function is serialized by both hosts before it runs, so
 * two things cannot be seen in a diff: it must work detached from this module
 * (no imports, no closures), and every bound must hold IN the page, because the
 * result is what crosses the boundary. The fake DOM scripts exactly the members
 * the function reads; a real-browser exercise belongs to the host proof.
 *
 * THE SAFETY RULE THIS FILE EXISTS TO PIN: a query can be pointed at a login
 * form, so a password input's VALUE is never returned. "Read-only" is not the
 * same promise as "cannot leak what is already typed".
 */

import assert from "node:assert/strict";
import test from "node:test";
import { build } from "esbuild";
import { pathToFileURL } from "node:url";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

async function load(entry) {
  const dir = await mkdtemp(join(tmpdir(), "lop-element-query-"));
  const outfile = join(dir, "module.mjs");
  await build({ entryPoints: [entry], bundle: true, platform: "node", format: "esm", outfile });
  const loaded = await import(pathToFileURL(outfile));
  return { loaded, close: () => rm(dir, { recursive: true, force: true }) };
}

function styleMap(entries) {
  return {
    getPropertyValue: (name) => entries.find(([key]) => key === name)?.[1] ?? "",
    display: entries.find(([key]) => key === "display")?.[1] ?? "block",
    visibility: entries.find(([key]) => key === "visibility")?.[1] ?? "visible",
    contentVisibility: entries.find(([key]) => key === "content-visibility")?.[1] ?? "visible",
  };
}

function element(overrides = {}) {
  const rect = overrides.rect ?? { x: 1.005, y: 2.674, width: 10, height: 10.669, top: 2.674, bottom: 13.343, left: 1.005, right: 11.005 };
  const attrs = overrides.attrs ?? {};
  return {
    tagName: overrides.tagName ?? "INPUT",
    value: overrides.value,
    checked: overrides.checked,
    disabled: overrides.disabled,
    hasAttribute: (name) => name in attrs,
    getAttribute: (name) => attrs[name] ?? null,
    innerText: overrides.innerText ?? "",
    getBoundingClientRect: () => rect,
  };
}

async function withFakeDom(dom, run) {
  const prior = { document: globalThis.document, getComputedStyle: globalThis.getComputedStyle, window: globalThis.window };
  globalThis.document = dom.document;
  globalThis.getComputedStyle = dom.getComputedStyle;
  globalThis.window = dom.window ?? { innerWidth: 1024, innerHeight: 768 };
  try {
    return await run();
  } finally {
    globalThis.document = prior.document;
    globalThis.getComputedStyle = prior.getComputedStyle;
    globalThis.window = prior.window;
  }
}

/** The function's own source, evaluated in global scope — the same detachment
 *  both hosts perform. */
function detached(fn) {
  return (0, eval)(`(${fn.toString()})`);
}

test("descriptors carry the identity fields an agent asked for", async () => {
  const module = await load("src/driver/element-query.ts");
  const subject = element({
    tagName: "INPUT",
    attrs: { id: "hook_active_55a1", name: "hook[active]", type: "checkbox", role: "checkbox", "aria-label": "Active" },
    checked: true,
    value: "on",
  });
  const dom = {
    document: {
      documentElement: { clientWidth: 1024, clientHeight: 768 },
      querySelectorAll: () => [subject],
    },
    getComputedStyle: () => styleMap([]),
  };
  try {
    const result = await withFakeDom(dom, () => module.loaded.queryElements("input", 20));
    assert.equal(result.count, 1);
    assert.equal(result.truncated, false);
    const [match] = result.matches;
    assert.equal(match.tag, "input");
    assert.equal(match.id, "hook_active_55a1");
    assert.equal(match.name, "hook[active]");
    assert.equal(match.role, "checkbox");
    assert.equal(match.ariaLabel, "Active");
    assert.equal(match.type, "checkbox");
    assert.equal(match.checked, true);
    assert.equal(match.value, "on");
    assert.equal(match.visible, true);
    assert.equal(match.inView, true);
    // Two-decimal rounding, the same convention the geometry readers use.
    assert.equal(match.rect.y, 2.67);
    assert.equal(match.rect.height, 10.67);
  } finally {
    await module.close();
  }
});

test("a password input's value is never returned", async () => {
  const module = await load("src/driver/element-query.ts");
  const secret = element({
    tagName: "INPUT",
    attrs: { id: "pw", type: "password" },
    value: "hunter2-synthetic",
  });
  const dom = {
    document: { documentElement: { clientWidth: 1024, clientHeight: 768 }, querySelectorAll: () => [secret] },
    getComputedStyle: () => styleMap([]),
  };
  try {
    const result = await withFakeDom(dom, () => module.loaded.queryElements("input[type=password]", 20));
    assert.equal(result.matches[0].value, "", "a query must never echo a typed password");
    assert.equal(result.matches[0].type, "password");
    // A password box has no checked STATE; reporting `false` made the tool print
    // "unchecked" on it (measured), which reads as a checkbox.
    assert.equal(result.matches[0].checked, null);
  } finally {
    await module.close();
  }
});

test("visibility is a rendered-box question, not an attribute", async () => {
  const module = await load("src/driver/element-query.ts");
  const zero = element({ rect: { x: 0, y: 0, width: 0, height: 0, top: 0, bottom: 0, left: 0, right: 0 } });
  const hidden = element({ attrs: { id: "h" } });
  const offscreen = element({ rect: { x: 0, y: 5000, width: 10, height: 10, top: 5000, bottom: 5010, left: 0, right: 10 } });
  const dom = {
    document: { documentElement: { clientWidth: 1024, clientHeight: 768 }, querySelectorAll: () => [zero, hidden, offscreen] },
    getComputedStyle: (el) => (el === hidden ? styleMap([["display", "none"]]) : styleMap([])),
  };
  try {
    const result = await withFakeDom(dom, () => module.loaded.queryElements("input", 20));
    assert.equal(result.matches[0].visible, false, "a zero box is not visible");
    assert.equal(result.matches[1].visible, false, "display:none is not visible");
    assert.equal(result.matches[2].visible, true, "scrolled out of view is still rendered");
    assert.equal(result.matches[2].inView, false, "inView is the viewport question");
  } finally {
    await module.close();
  }
});

test("caps hold in the page: 50 elements, clipped text", async () => {
  const module = await load("src/driver/element-query.ts");
  const many = [...Array(80)].map((_, index) => element({ attrs: { id: `e${index}` }, innerText: "x".repeat(500) }));
  const dom = {
    document: { documentElement: { clientWidth: 1024, clientHeight: 768 }, querySelectorAll: () => many },
    getComputedStyle: () => styleMap([]),
  };
  try {
    const result = await withFakeDom(dom, () => module.loaded.queryElements("input", 999));
    assert.equal(result.count, 80, "count is the page's real match count");
    assert.equal(result.matches.length, 50, "at most 50 descriptors");
    assert.equal(result.truncated, true);
    assert.equal(result.matches[0].text.length, 120, "text capped at 120 chars");

    const small = await withFakeDom(dom, () => module.loaded.queryElements("input", 5));
    assert.equal(small.matches.length, 5, "the caller's limit is honoured below the cap");
  } finally {
    await module.close();
  }
});

test("an invalid selector is the caller's error, and an empty one is refused", async () => {
  const module = await load("src/driver/element-query.ts");
  const dom = {
    document: {
      documentElement: { clientWidth: 1024, clientHeight: 768 },
      querySelectorAll: () => {
        throw new Error("SyntaxError: '###' is not a valid selector");
      },
    },
    getComputedStyle: () => styleMap([]),
  };
  try {
    // Reported as a VALUE: a thrown error does not survive the isolated world's
    // transport (Electron answers "Script failed to execute"), so the caller's
    // mistake would reach the tool as `internal`.
    const bad = await withFakeDom(dom, () => module.loaded.queryElements("###", 20));
    assert.equal(bad.error, "invalid_selector");
    assert.deepEqual(bad.matches, []);
    assert.equal(await withFakeDom(dom, () => module.loaded.queryElements("", 20)), null);
    assert.equal(await withFakeDom(dom, () => module.loaded.queryElements("   ", 20)), null);
  } finally {
    await module.close();
  }
});

test("the function survives a toString() round-trip in global scope", async () => {
  const module = await load("src/driver/element-query.ts");
  const source = module.loaded.queryElements.toString();
  assert.ok(source.includes("MAX_ELEMENTS"), "the caps live in the serialized body");
  assert.ok(!/import |require\(/.test(source), "no module references");
  const fn = detached(module.loaded.queryElements);
  const dom = {
    document: { documentElement: { clientWidth: 1024, clientHeight: 768 }, querySelectorAll: () => [element({ attrs: { id: "x" } })] },
    getComputedStyle: () => styleMap([]),
  };
  try {
    const result = await withFakeDom(dom, () => fn("input", 20));
    assert.equal(result.count, 1);
    assert.equal(result.matches[0].id, "x");
  } finally {
    await module.close();
  }
});
