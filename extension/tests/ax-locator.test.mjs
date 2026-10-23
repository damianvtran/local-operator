/* Accessibility locators: the parse, the match, and the candidates.
 *
 * WHY THIS FILE EXISTS AT ALL. On GitHub's "Create GitHub App" page every id is
 * framework-generated, so `#hook_active` missed; what was never ambiguous was
 * the element's accessible identity — a checkbox named "Active". These tests
 * pin the grammar an agent types, the precedence when several nodes match, and
 * the candidates a miss reports.
 *
 * The trees here are shaped like Chromium's `getFullAXTree` payload, including
 * the two spellings of a boolean it really does emit (`expanded` boolean,
 * `checked` string — see ax-window.test.mjs) and nodes with no backend id,
 * which can never be a target.
 */

import assert from "node:assert/strict";
import test from "node:test";
import { build } from "esbuild";
import { pathToFileURL } from "node:url";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

async function load(entry) {
  const dir = await mkdtemp(join(tmpdir(), "lop-ax-locator-"));
  const outfile = join(dir, "module.mjs");
  await build({ entryPoints: [entry], bundle: true, platform: "node", format: "esm", outfile });
  const loaded = await import(pathToFileURL(outfile));
  return { loaded, close: () => rm(dir, { recursive: true, force: true }) };
}

function form() {
  return [
    { nodeId: "1", role: { value: "RootWebArea" }, name: { value: "Create App" }, childIds: ["2"] },
    // The <h1>, which really does come first on the measured page and carries
    // the same words as the submit button.
    { nodeId: "1a", backendDOMNodeId: 9, role: { value: "heading" }, name: { value: "Create GitHub App" }, childIds: [] },
    { nodeId: "2", backendDOMNodeId: 10, role: { value: "textbox" }, name: { value: "GitHub App name" }, childIds: [] },
    { nodeId: "3", backendDOMNodeId: 11, role: { value: "textbox" }, name: { value: "Homepage URL" }, childIds: [] },
    { nodeId: "4", backendDOMNodeId: 12, role: { value: "checkbox" }, name: { value: "Active" },
      properties: [{ name: "checked", value: { value: "true" } }, { name: "focusable", value: { value: true } }], childIds: [] },
    { nodeId: "5", backendDOMNodeId: 13, role: { value: "button" }, name: { value: "Create GitHub App" }, childIds: [] },
    { nodeId: "6", backendDOMNodeId: 14, role: { value: "radio" }, name: { value: "Only on this account" }, childIds: [] },
    { nodeId: "7", backendDOMNodeId: 15, role: { value: "radio" }, name: { value: "Any account" }, childIds: [] },
    { nodeId: "8", backendDOMNodeId: 16, role: { value: "DisclosureTriangle" }, name: { value: "Webhook" },
      properties: [{ name: "expanded", value: { value: false } }], childIds: [] },
    { nodeId: "9", role: { value: "StaticText" }, name: { value: "We will deliver event details" }, childIds: [] },
    { nodeId: "10", backendDOMNodeId: 17, role: { value: "checkbox" }, name: { value: "Active" }, childIds: [] },
  ];
}

test("the three locator forms parse, and CSS does not", async () => {
  const { loaded, close } = await load("src/driver/ax-locator.ts");
  try {
    assert.deepEqual(loaded.parseLocator('role=checkbox[name="Active"]'), {
      kind: "role", role: "checkbox", name: "Active", substring: false,
    });
    assert.deepEqual(loaded.parseLocator("label='Homepage URL'"), {
      kind: "label", name: "Homepage URL", substring: false,
    });
    assert.deepEqual(loaded.parseLocator('text="Create GitHub App"'), {
      kind: "text", name: "Create GitHub App", substring: false,
    });
    assert.deepEqual(loaded.parseLocator('role=checkbox[name*="activ"]'), {
      kind: "role", role: "checkbox", name: "activ", substring: true,
    });
    // A role with no name filter is legal: "the first checkbox".
    assert.deepEqual(loaded.parseLocator("role=checkbox"), {
      kind: "role", role: "checkbox", name: "", substring: false,
    });
    // CSS must stay CSS. Every one of these was tried on the real page.
    for (const css of ["#hook_active", 'input[name="hook_active"]', "button[aria-controls='repository-permissions']"]) {
      assert.equal(loaded.parseLocator(css), null, css);
    }
    // A malformed locator falls through to CSS rather than half-matching.
    assert.equal(loaded.parseLocator('role=[name="Active"]'), null);
    assert.equal(loaded.parseLocator('role=checkbox[name="unterminated'), null);
  } finally {
    await close();
  }
});

test("role+name resolves to the first match and reports ambiguity", async () => {
  const { loaded, close } = await load("src/driver/ax-locator.ts");
  try {
    const hit = loaded.resolveLocator(form(), loaded.parseLocator('role=checkbox[name="Active"]'));
    assert.equal(hit.backendNodeId, 12, "tree order wins: the first Active checkbox");
    assert.equal(hit.role, "checkbox");
    assert.equal(hit.matched, 2, "two matched; the caller is told so");

    const single = loaded.resolveLocator(form(), loaded.parseLocator("role=radio"));
    assert.equal(single.backendNodeId, 14);
    assert.equal(single.matched, 2);
  } finally {
    await close();
  }
});

test("label= resolves a control by its accessible name, not a <label> node", async () => {
  const { loaded, close } = await load("src/driver/ax-locator.ts");
  try {
    // Chromium puts a `<label for>`'s text on the CONTROL as its name; there is
    // no Label node to follow. Matching the control is what makes this work on
    // a page whose ids are generated.
    const hit = loaded.resolveLocator(form(), loaded.parseLocator('label="Homepage URL"'));
    assert.equal(hit.backendNodeId, 11);
    assert.equal(hit.role, "textbox");

    // A label that names nothing is a miss, not a silent first-match.
    assert.equal(loaded.resolveLocator(form(), loaded.parseLocator('label="Nope"')), null);
  } finally {
    await close();
  }
});

test("text= prefers the clickable node over the static text that says the same thing", async () => {
  const { loaded, close } = await load("src/driver/ax-locator.ts");
  try {
    const hit = loaded.resolveLocator(form(), loaded.parseLocator('text="Create GitHub App"'));
    assert.equal(hit.backendNodeId, 13, "the button, not the RootWebArea");
    // A StaticText node carries the same words but has no backend id, so it
    // cannot be a target; it must not win the match either.
    const text = loaded.resolveLocator(form(), loaded.parseLocator('text="We will deliver event details"'));
    assert.equal(text, null);
  } finally {
    await close();
  }
});

test("matching is case-insensitive and whitespace-normalised", async () => {
  const { loaded, close } = await load("src/driver/ax-locator.ts");
  try {
    const hit = loaded.resolveLocator(form(), loaded.parseLocator('role=checkbox[name="  active "]'));
    assert.equal(hit.backendNodeId, 12);
    const label = loaded.resolveLocator(form(), loaded.parseLocator('label="homepage   url"'));
    assert.equal(label.backendNodeId, 11);
  } finally {
    await close();
  }
});

test("a miss reports candidates an agent can paste back", async () => {
  const { loaded, close } = await load("src/driver/ax-locator.ts");
  try {
    // The measured loop: nine CSS guesses, each one a whole tool call.
    const candidates = loaded.locatorSuggestions(form(), "#hook_active");
    assert.ok(candidates.length > 0);
    assert.ok(candidates.some((entry) => entry.name === "Active"), JSON.stringify(candidates));
    for (const entry of candidates) {
      assert.ok(entry.locator.startsWith("role=") || entry.locator.startsWith("text="));
      // Every suggestion must re-parse; a suggestion that cannot be used is
      // worse than none.
      assert.ok(loaded.parseLocator(entry.locator), entry.locator);
    }

    // A locator miss is scored the same way, with the role counting in favour.
    const roleMiss = loaded.locatorSuggestions(form(), 'role=checkbox[name="Actve"]');
    assert.equal(roleMiss[0].name, "Active");
    assert.equal(roleMiss[0].role, "checkbox");
  } finally {
    await close();
  }
});

test("a suggestion resolves back to the element it named", async () => {
  const { loaded, close } = await load("src/driver/ax-locator.ts");
  try {
    const [first] = loaded.locatorSuggestions(form(), "only on this account");
    assert.equal(first.name, "Only on this account");
    const hit = loaded.resolveLocator(form(), loaded.parseLocator(first.locator));
    assert.equal(hit.backendNodeId, 14);
  } finally {
    await close();
  }
});

test("a locator with trailing content is refused, not half-interpreted", async () => {
  // Agent review round 1, R1-1: reading up to the first closing quote DROPPED the
  // caller's constraint — `[checked]` vanished and `please` vanished — so the
  // locator selected something the caller never asked for.
  const { loaded, close } = await load("src/driver/ax-locator.ts");
  try {
    assert.equal(loaded.parseLocator('role=checkbox[name="Active"][checked]'), null);
    assert.equal(loaded.parseLocator('text="Submit" please'), null);
    assert.equal(loaded.parseLocator('label="Home" "Work"'), null);
    // Unterminated quotes and unbalanced brackets are CSS, not half a locator.
    assert.equal(loaded.parseLocator('role=checkbox[name="Active'), null);
    assert.equal(loaded.parseLocator("role=checkbox[name=Active"), null);
    // A bare value may not smuggle a bracket past the grammar.
    assert.equal(loaded.parseLocator("text=Create [GitHub] App"), null);
  } finally {
    await close();
  }
});

test("a suggestion whose name contains a quote still resolves to it", async () => {
  // Agent review round 1, R1-2: JSON.stringify quoting without unescaping made
  // the suggestion parse to a TRUNCATED name and resolve to nothing.
  const { loaded, close } = await load("src/driver/ax-locator.ts");
  try {
    const nodes = [
      { nodeId: "1", role: { value: "RootWebArea" }, name: { value: "" }, childIds: ["2", "3", "4"] },
      { nodeId: "2", backendDOMNodeId: 1, role: { value: "button" }, name: { value: 'Say "yes" now' }, childIds: [] },
      { nodeId: "3", backendDOMNodeId: 2, role: { value: "button" }, name: { value: "back\\\\slash" }, childIds: [] },
      { nodeId: "4", backendDOMNodeId: 3, role: { value: "button" }, name: { value: 'it\'s "both"' }, childIds: [] },
    ];
    for (const node of [nodes[1], nodes[2], nodes[3]]) {
      const locator = loaded.locatorFor(node.role.value, node.name.value);
      const parsed = loaded.parseLocator(locator);
      assert.ok(parsed, `unparseable suggestion: ${locator}`);
      const hit = loaded.resolveLocator(nodes, parsed);
      assert.ok(hit, `suggestion resolved to nothing: ${locator}`);
      assert.equal(hit.name, node.name.value, `${locator} resolved to ${hit.name}`);
    }
  } finally {
    await close();
  }
});

test("label*= and text*= are the documented substring spellings", async () => {
  const { loaded, close } = await load("src/driver/ax-locator.ts");
  try {
    assert.deepEqual(loaded.parseLocator("label*=Home"), { kind: "label", name: "Home", substring: true });
    assert.deepEqual(loaded.parseLocator("text*=Create"), { kind: "text", name: "Create", substring: true });
    // The undocumented pre-review spelling is gone, and falls through to CSS.
    assert.equal(loaded.parseLocator("label=*=Home"), null);
  } finally {
    await close();
  }
});

test("a single-quoted suggestion escapes backslashes like a double-quoted one", async () => {
  // Residual from round 2: the single-quote branch quoted a name containing `"`
  // but did not escape `\`, while `readValue` unescapes under both quote styles,
  // so `C:\path "quoted"` came back as `C:path "quoted"`.
  const { loaded, close } = await load("src/driver/ax-locator.ts");
  try {
    const name = 'C:\\path "quoted"';
    const nodes = [
      { nodeId: "1", role: { value: "RootWebArea" }, name: { value: "" }, childIds: ["2"] },
      { nodeId: "2", backendDOMNodeId: 7, role: { value: "button" }, name: { value: name }, childIds: [] },
    ];
    const locator = loaded.locatorFor("button", name);
    assert.ok(locator.startsWith("role=button[name='"), locator);
    const hit = loaded.resolveLocator(nodes, loaded.parseLocator(locator));
    assert.ok(hit, locator);
    assert.equal(hit.name, name);
  } finally {
    await close();
  }
});
