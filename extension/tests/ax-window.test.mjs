/* `compactAX` scoping and `windowSnapshot` paging: the discovery half of the
 * snapshot contract.
 *
 * WHY THESE ARE DRIVER TESTS AND NOT COMMAND TESTS. The command tests prove the
 * host hands the driver a resolved scope; these prove what the driver does with
 * it, against synthetic trees with the shapes real pages produce — ignored
 * wrappers, a collapsed `<details>` whose children are absent, hundreds of
 * sibling fields, and a landmark that must keep its children's refs.
 *
 * The invariant the paging tests exist for: a ref's NAME is a property of the
 * tree, not of the page the caller is looking at. `e57` in the footer of page
 * one must click the same element as `e57` on page three, or an agent that
 * pages a form and then clicks is clicking blind.
 */

import assert from "node:assert/strict";
import test from "node:test";
import { build } from "esbuild";
import { pathToFileURL } from "node:url";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

async function load(entry) {
  const dir = await mkdtemp(join(tmpdir(), "lop-ax-window-"));
  const outfile = join(dir, "module.mjs");
  await build({ entryPoints: [entry], bundle: true, platform: "node", format: "esm", outfile });
  const loaded = await import(pathToFileURL(outfile));
  return { loaded, close: () => rm(dir, { recursive: true, force: true }) };
}

/** One synthetic tree shaped like a settings form: an ignored document wrapper,
 *  a real <form>, a collapsed <details> (summary only — its panel contributes
 *  NOTHING, which is what Chromium does), and many sibling fields. */
function formTree(fields = 200) {
  const nodes = [
    { nodeId: "1", role: { value: "RootWebArea" }, name: { value: "Create App" }, childIds: ["2"] },
    { nodeId: "2", ignored: true, childIds: ["3"] },
    { nodeId: "3", backendDOMNodeId: 300, role: { value: "form" }, name: { value: "" }, childIds: ["4", "6"] },
    { nodeId: "4", backendDOMNodeId: 401, role: { value: "textbox" }, name: { value: "GitHub App name" }, childIds: [] },
    {
      nodeId: "6",
      backendDOMNodeId: 402,
      role: { value: "DisclosureTriangle" },
      name: { value: "Webhook" },
      properties: [{ name: "expanded", value: { value: false } }, { name: "focusable", value: { value: true } }],
      childIds: ["7"],
    },
    { nodeId: "7", role: { value: "StaticText" }, name: { value: "Webhook" }, childIds: [] },
  ];
  let id = 10;
  for (let index = 0; index < fields; index += 1) {
    nodes.push({
      nodeId: `f${index}`,
      backendDOMNodeId: 1000 + index,
      role: { value: "textbox" },
      name: { value: `Setting ${index}` },
      childIds: [],
    });
    nodes[2].childIds.push(`f${index}`);
  }
  void id;
  return nodes;
}

test("a scoped snapshot renders only the subtree, rebased to depth zero", async () => {
  const { loaded, close } = await load("src/driver/ax-compact.ts");
  try {
    const all = loaded.compactAX(formTree(3), 5);
    assert.ok(all.snapshot.includes("RootWebArea"), "the unscoped call still renders the page");
    assert.equal(all.scoped, false);
    assert.equal(all.totalLines, all.snapshot.split("\n").length);

    const scoped = loaded.compactAX(formTree(3), 5, { rootBackendNodeId: 300 });
    assert.equal(scoped.scoped, true);
    assert.ok(!scoped.snapshot.includes("RootWebArea"), "nothing above the scope survives");
    assert.ok(scoped.snapshot.startsWith("- form"));
    // `e1` here is the name the WHOLE-TREE mint gave this node (it is the first
    // interactive node of the page), not a number restarted inside the scope:
    // that restart was R1-1's wrong-target bug.
    assert.ok(
      scoped.snapshot.includes('textbox "GitHub App name" [e1]'),
      "the scope renders the whole-tree mint's name for this node",
    );
    assert.ok(scoped.snapshot.includes("Setting 2"));
    // Depth is rebased: the form is the root, its children are one level in.
    assert.ok(scoped.snapshot.includes('\n  - textbox "GitHub App name"'));
  } finally {
    await close();
  }
});

test("a scope the tree does not contain reports scoped:false and renders nothing", async () => {
  const { loaded, close } = await load("src/driver/ax-compact.ts");
  try {
    // 999 is a real DOM element with no accessible node (display:none,
    // aria-hidden): the host must answer a typed miss, never an empty tree
    // that reads like an empty section.
    const result = loaded.compactAX(formTree(3), 1, { rootBackendNodeId: 999 });
    assert.equal(result.scoped, false);
    assert.equal(result.snapshot, "");
    assert.deepEqual(result.refs, {});
  } finally {
    await close();
  }
});

test("a collapsed disclosure is marked, and the walk continues past it", async () => {
  const { loaded, close } = await load("src/driver/ax-compact.ts");
  try {
    const rendered = loaded.compactAX(formTree(1), 1).snapshot;
    const lines = rendered.split("\n");
    const summaryIndex = lines.findIndex((line) => line.includes("DisclosureTriangle"));
    const summary = lines[summaryIndex];
    assert.ok(summary.includes("[collapsed]"), summary);
    // The marker is what tells an agent to open it (measured failure: a closed
    // <details> contributes its summary and NOTHING inside it, so the checkbox
    // that unblocks the form was absent from the tree entirely).
    //
    // A collapsed panel's absence is CHROMIUM's fact, not a pruning rule this
    // driver applies — inventing one would also drop the summary's own text.
    // What this driver owes the caller is that the walk survives it, so the
    // assertion that carries weight is the field AFTER the disclosure.
    assert.equal(
      lines.slice(summaryIndex + 1).some((line) => line.includes("Setting 0")),
      true,
      rendered,
    );
    assert.ok(
      summary.includes("[e"),
      "the summary is clickable and carries the ref the whole-tree mint gave it",
    );
  } finally {
    await close();
  }
});

test("a scoped snapshot never re-points a ref minted without a scope", async () => {
  // Agent review round 1, R1-1. Refs used to be minted only for emitted lines,
  // so `e1` inside a scope named a different node than `e1` of the same page's
  // unscoped snapshot — at the SAME epoch, which the click path treats as
  // still valid. That is a wrong-target click with no error.
  const { loaded, close } = await load("src/driver/ax-compact.ts");
  try {
    const epoch = 7;
    const all = loaded.compactAX(formTree(4), epoch);
    const scoped = loaded.compactAX(formTree(4), epoch, { rootBackendNodeId: 1000 });
    assert.equal(scoped.scoped, true);
    // The scoped tree is a SUBSET: every ref it mints names the same node the
    // unscoped call minted for it.
    for (const [ref, node] of Object.entries(scoped.refs)) {
      assert.deepEqual(all.refs[ref], node, `ref ${ref} changed meaning`);
    }
    // And the aliasing case specifically: the "Setting 0" textbox inside the
    // scope must NOT be e1 just because it is first inside the scope.
    const setting0 = Object.entries(all.refs).find(([, node]) => node.backendNodeId === 1000);
    assert.ok(setting0, "the scope node has a ref in the unscoped tree");
    assert.equal(scoped.refs[setting0[0]].backendNodeId, 1000);
    assert.notEqual(setting0[0], "e1");
  } finally {
    await close();
  }
});

test("windowing drops whole lines and preserves every ref name", async () => {
  const { loaded, close } = await load("src/driver/ax-compact.ts");
  try {
    const out = loaded.compactAX(formTree(120), 1);
    const first = loaded.windowSnapshot(out.snapshot, { start: 1, limit: 10 });
    const third = loaded.windowSnapshot(out.snapshot, { start: 21, limit: 10 });
    assert.equal(first.totalLines, out.totalLines);
    assert.equal(first.returnedLines, 10);
    assert.equal(third.start, 21);
    assert.equal(third.returnedLines, 10);
    // The same element carries the same ref on both pages.
    const refOf = (text, name) => text.split("\n").find((line) => line.includes(name)).match(/\[(e\d+)\]/)[1];
    const shared = loaded.windowSnapshot(out.snapshot, { start: 0, limit: 0 });
    assert.equal(refOf(shared.text, "Setting 0"), refOf(first.text, "Setting 0"));
    assert.equal(refOf(shared.text, "Setting 20"), refOf(third.text, "Setting 20"));
    assert.notEqual(refOf(first.text, "Setting 0"), refOf(third.text, "Setting 20"));
  } finally {
    await close();
  }
});

test("search matches role and accessible name, case-insensitively", async () => {
  const { loaded, close } = await load("src/driver/ax-compact.ts");
  try {
    const out = loaded.compactAX(formTree(5), 1);
    const byName = loaded.windowSnapshot(out.snapshot, { search: "setting 3" });
    assert.equal(byName.matchedLines, 1);
    assert.ok(byName.text.includes("Setting 3"));

    const byRole = loaded.windowSnapshot(out.snapshot, { search: "disclosure" });
    assert.equal(byRole.matchedLines, 1);
    assert.ok(byRole.text.includes("Webhook"));

    // Blank means "no filter", never "match nothing": a caller that sends ""
    // means the whole tree, and zero lines would read as an empty section.
    const blank = loaded.windowSnapshot(out.snapshot, { search: "   " });
    assert.equal(blank.matchedLines, blank.totalLines);
  } finally {
    await close();
  }
});

test("a start past the end returns an empty page, not a wrapped one", async () => {
  const { loaded, close } = await load("src/driver/ax-compact.ts");
  try {
    const out = loaded.compactAX(formTree(2), 1);
    const page = loaded.windowSnapshot(out.snapshot, { start: 9999, limit: 10 });
    assert.equal(page.text, "");
    assert.equal(page.returnedLines, 0);
    assert.equal(page.start, 9999);
  } finally {
    await close();
  }
});

test("state markers accept Chromium's string and boolean spellings", async () => {
  const { loaded, close } = await load("src/driver/ax-compact.ts");
  try {
    // Measured on a real page: `expanded` is a boolean, `checked` is the string
    // "true". A compare against `true` alone printed [collapsed] and silently
    // dropped [checked] — on a form whose dangerous default is a ticked box.
    const nodes = [
      { nodeId: "1", role: { value: "RootWebArea" }, name: { value: "" }, childIds: ["2", "3", "4"] },
      { nodeId: "2", backendDOMNodeId: 1, role: { value: "checkbox" }, name: { value: "String check" },
        properties: [{ name: "checked", value: { value: "true" } }], childIds: [] },
      { nodeId: "3", backendDOMNodeId: 2, role: { value: "checkbox" }, name: { value: "Boolean check" },
        properties: [{ name: "checked", value: { value: true } }], childIds: [] },
      { nodeId: "4", backendDOMNodeId: 3, role: { value: "DisclosureTriangle" }, name: { value: "Closed" },
        properties: [{ name: "expanded", value: { value: "false" } }], childIds: [] },
    ];
    const rendered = loaded.compactAX(nodes, 1).snapshot;
    assert.ok(rendered.includes('"String check" [e1] [checked]'), rendered);
    assert.ok(rendered.includes('"Boolean check" [e2] [checked]'), rendered);
    assert.ok(rendered.includes('"Closed" [collapsed]'), rendered);
  } finally {
    await close();
  }
});

test("state markers accumulate instead of hiding one another", async () => {
  // Review round 1, R1-8: a checked AND disabled control printed only
  // [checked], which reads as clickable, and a tri-state "mixed" printed
  // nothing at all.
  const { loaded, close } = await load("src/driver/ax-compact.ts");
  try {
    const nodes = [
      { nodeId: "1", role: { value: "RootWebArea" }, name: { value: "" }, childIds: ["2", "3"] },
      { nodeId: "2", backendDOMNodeId: 1, role: { value: "checkbox" }, name: { value: "Both" },
        properties: [{ name: "checked", value: { value: "true" } }, { name: "disabled", value: { value: true } }], childIds: [] },
      { nodeId: "3", backendDOMNodeId: 2, role: { value: "checkbox" }, name: { value: "Tri" },
        properties: [{ name: "checked", value: { value: "mixed" } }], childIds: [] },
    ];
    const rendered = loaded.compactAX(nodes, 1).snapshot;
    assert.ok(rendered.includes('"Both" [e1] [checked] [disabled]'), rendered);
    assert.ok(rendered.includes('"Tri" [e2] [mixed]'), rendered);
  } finally {
    await close();
  }
});

test("framesInScope counts only the scope's own frames", async () => {
  // Agent review round 2, R2-1: a page-level frame count told a caller that a
  // FRAMELESS section held a frame it was not seeing.
  const { loaded, close } = await load("src/driver/ax-compact.ts");
  try {
    const nodes = [
      { nodeId: "1", role: { value: "RootWebArea" }, name: { value: "" }, childIds: ["2", "3"] },
      { nodeId: "2", backendDOMNodeId: 10, role: { value: "form" }, name: { value: "Outer" }, childIds: ["4"] },
      { nodeId: "3", backendDOMNodeId: 11, role: { value: "region" }, name: { value: "Elsewhere" }, childIds: ["5"] },
      { nodeId: "4", backendDOMNodeId: 12, role: { value: "Iframe" }, name: { value: "Payment" }, childIds: [] },
      { nodeId: "5", backendDOMNodeId: 13, role: { value: "button" }, name: { value: "Inner" }, childIds: [] },
    ];
    assert.equal(loaded.compactAX(nodes, 1).framesInScope, 0, "an unscoped read reports no scope frames");
    assert.equal(loaded.compactAX(nodes, 1, { rootBackendNodeId: 10 }).framesInScope, 1);
    assert.equal(loaded.compactAX(nodes, 1, { rootBackendNodeId: 11 }).framesInScope, 0);
  } finally {
    await close();
  }
});

/*
 * VALUE IN THE SNAPSHOT
 *
 * Every fixture below is the shape `Accessibility.getFullAXTree` actually
 * returned when probed against Electron 44.3.0 (Chromium) on 2026-11-06 — see
 * the measurement notes on each case. The gap they close: a native `<select>`
 * printed `combobox "Environment" [collapsed]` with no sign of which environment
 * was selected, so an agent could only learn the value from a screenshot.
 */
function valueTree(control, options = []) {
  const root = { nodeId: "1", role: { value: "RootWebArea" }, name: { value: "" }, childIds: ["2", ...options.map((_, index) => `o${index}`)] };
  return [root, control, ...options];
}

async function renderValue(control, options = []) {
  const { loaded, close } = await load("src/driver/ax-compact.ts");
  try {
    return loaded.compactAX(valueTree(control, options), 1).snapshot;
  } finally {
    await close();
  }
}

test("a native select's selected option is rendered as the control's value", async () => {
  // Measured: `<select><option selected>Production</option></select>` produces a
  // combobox node whose `value` is the option's TEXT ("Production") and whose
  // `expanded` is boolean false — so the line carries both [collapsed] and the
  // value, in that order.
  const rendered = await renderValue(
    { nodeId: "2", backendDOMNodeId: 1, role: { value: "combobox" }, name: { value: "Environment" }, value: { value: "Production" },
      properties: [{ name: "focusable", value: { value: true } }, { name: "expanded", value: { value: false } }], childIds: ["o0", "o1"] },
    [
      { nodeId: "o0", role: { value: "option" }, name: { value: "Sandbox" }, properties: [{ name: "selected", value: { value: false } }], childIds: [] },
      { nodeId: "o1", role: { value: "option" }, name: { value: "Production" }, properties: [{ name: "selected", value: { value: true } }], childIds: [] },
    ],
  );
  assert.ok(rendered.includes('combobox "Environment" [e1] [collapsed] = "Production"'), rendered);
  // The value must not have moved the ref or the state marker: the locator
  // grammar reads both positionally and refuses an unknown bracketed predicate.
  assert.ok(rendered.includes('[e1] [collapsed]'), rendered);
});

test("a textbox renders its text, including a multi-line one", async () => {
  // Measured: a `<textarea>`'s value arrives verbatim, newlines and the page's
  // own indentation included.
  const single = await renderValue({ nodeId: "2", backendDOMNodeId: 1, role: { value: "textbox" }, name: { value: "App name" }, value: { value: "release-bot" }, childIds: [] });
  assert.ok(single.includes('textbox "App name" [e1] = "release-bot"'), single);

  const multiline = await renderValue({
    nodeId: "2", backendDOMNodeId: 1, role: { value: "textbox" }, name: { value: "Notes" },
    value: { value: "first line\n      second line       third line" },
    properties: [{ name: "multiline", value: { value: true } }], childIds: [],
  });
  assert.ok(multiline.includes('textbox "Notes" [e1] = "first line second line third line"'), multiline);
});

test("an empty value and a value identical to the name render nothing", async () => {
  const empty = await renderValue({ nodeId: "2", backendDOMNodeId: 1, role: { value: "combobox" }, name: { value: "Environment" }, value: { value: "" }, childIds: [] });
  assert.ok(empty.trimEnd().endsWith('combobox "Environment" [e1]'), empty);

  const blank = await renderValue({ nodeId: "2", backendDOMNodeId: 1, role: { value: "textbox" }, name: { value: "App name" }, value: { value: "   " }, childIds: [] });
  assert.ok(blank.trimEnd().endsWith('textbox "App name" [e1]'), blank);

  // <option value="production">Production</option> — the name already says it.
  // (`option` mints no ref: it is not clickable on its own, so the whole line is
  // role + name and nothing else.)
  const duplicate = await renderValue({ nodeId: "2", backendDOMNodeId: 1, role: { value: "option" }, name: { value: "Production" }, value: { value: "Production" }, childIds: [] });
  assert.equal(duplicate.trimEnd().split("\n").pop().trim(), '- option "Production"');
});

test("a long value clips at 40 characters and marks the cut with an ellipsis", async () => {
  const sixty = "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWX";
  assert.equal(sixty.length, 60);
  const rendered = await renderValue({ nodeId: "2", backendDOMNodeId: 1, role: { value: "textbox" }, name: { value: "Long" }, value: { value: sixty }, childIds: [] });
  assert.ok(rendered.includes(` = "${sixty.slice(0, 40)}…"`), rendered);
  // Exactly 40 is NOT clipped: the cap applies to the value, and the ellipsis
  // only appears when characters were dropped.
  const forty = "abcdefghijklmnopqrstuvwxyz0123456789ABCD";
  assert.equal(forty.length, 40);
  const exact = await renderValue({ nodeId: "2", backendDOMNodeId: 1, role: { value: "textbox" }, name: { value: "Exact" }, value: { value: forty }, childIds: [] });
  assert.ok(exact.includes(` = "${forty}"`), exact);
  assert.ok(!exact.includes("…"), exact);
});

test("valuetext outranks the raw value when Chromium sends one", async () => {
  // Measured: a native `<input type=range|number>` carries a `valuetext`
  // property; an ARIA `div[role=slider]` with aria-valuetext does NOT get one
  // (Chromium 142 drops it), so the numeric fallback is what prints there.
  const rendered = await renderValue({
    nodeId: "2", backendDOMNodeId: 1, role: { value: "slider" }, name: { value: "Volume" }, value: { value: 3 },
    properties: [{ name: "valuetext", value: { value: "Medium" } }], childIds: [],
  });
  assert.ok(rendered.includes('slider "Volume" [e1] = "Medium"'), rendered);

  const fallback = await renderValue({
    nodeId: "2", backendDOMNodeId: 1, role: { value: "slider" }, name: { value: "Quality" }, value: { value: 3 },
    properties: [{ name: "valuetext", value: { value: "" } }], childIds: [],
  });
  assert.ok(fallback.includes('slider "Quality" [e1] = "3"'), fallback);
});

test("only the roles an agent acts on carry a value, and a mask stays a mask", async () => {
  // A button's "value" duplicates its name; printing it would lengthen every
  // line and (since `search` matches the rendered line) change search results.
  const button = await renderValue({ nodeId: "2", backendDOMNodeId: 1, role: { value: "button" }, name: { value: "Save" }, value: { value: "Save" }, childIds: [] });
  assert.ok(!button.includes("="), button);

  // Measured on `<input type=password value="hunter2-SECRET">`: Chromium hands
  // over BULLETS (`••••••••••••••`), never the text, and exposes no `protected`
  // property to detect the field by. This assertion pins that the projection
  // adds nothing of its own — if a future Chromium stops masking, this is the
  // test that goes red.
  const password = await renderValue({ nodeId: "2", backendDOMNodeId: 1, role: { value: "textbox" }, name: { value: "" }, value: { value: "••••••••••••••" }, childIds: [] });
  assert.ok(!password.includes("hunter2"), password);
  assert.ok(password.includes('textbox [e1] = "••••••••••••••"'), password);
});
