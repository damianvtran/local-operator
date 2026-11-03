/* Native selection needs no Input.* or active tab: exercise the fixed serialized
 * DOM function and real command wrapper, with a synthetic DOM/CDP boundary.
 * This is not a Chromium/React execution claim; independent real-browser QA
 * must check those. The synthetic own setter trap pins why prototype setters
 * are required, and the wire spy makes accidental layout/popup/focus input fail.
 */
import assert from "node:assert/strict";
import test from "node:test";
import { build } from "esbuild";
import { mkdtemp, rm } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { pathToFileURL } from "node:url";

const root = process.env.LOCAL_OPERATOR_SCRATCHPAD ?? tmpdir();
const dir = await mkdtemp(join(root, "select-extension-"));
const output = join(dir, "native.mjs");
await build({ entryPoints: ["src/commands/select-native.ts"], bundle: true, platform: "node", format: "esm", outfile: output });
const { selectNative, readNativeSelection } = await import(pathToFileURL(output));

class Option {
  constructor(label, value, disabled = false) { this.label = label; this.value = value; this.disabled = disabled; this._selected = false; this.parentElement = null; }
  get selected() { return this._selected; }
  set selected(v) { this._selected = v; if (v && this.owner && !this.owner.multiple) for (const o of this.owner.options) if (o !== this) o._selected = false; }
}
class Select {
  constructor(options, multiple = false) {
    this.options = options; this.multiple = multiple; this.disabled = false; this.events = []; this.isConnected = true;
    this.ownerDocument = { defaultView: { HTMLSelectElement: Select, HTMLOptionElement: Option, Event: class { constructor(type, init) { this.type = type; this.bubbles = init.bubbles; } } } };
    for (const option of options) option.owner = this;
  }
  matches() { return this.disabled; }
  get value() { return this.options.find((o) => o.selected)?.value ?? ""; }
  set value(v) { for (const o of this.options) Object.getOwnPropertyDescriptor(Option.prototype, "selected").set.call(o, o.value === v); }
  dispatchEvent(event) { this.events.push([event.type, event.bubbles]); this.listener?.(event); }
}
// Apply never claims success; only a fresh read on a live node may verify it.
const applyAndRead = (node, requested) => {
  const applied = selectNative.call(node, requested);
  return applied.error ? applied : readNativeSelection.call(node, applied.expected);
};
const fixture = (multiple = false) => new Select([
  new Option(" English (U.S.) ", "en-US"), new Option("French", "fr"),
  new Option("English", "English (U.S.)"), new Option("Disabled", "off", true),
], multiple);

try {
  test("serialized DOM function has no closure and uses prototype setters/input/change", () => {
    const node = fixture();
    Object.defineProperty(node, "value", { set() { throw new Error("framework own setter called"); } });
    Object.defineProperty(node.options[0], "selected", { get() { return this._selected; }, set() { throw new Error("framework option setter called"); } });
    const run = new Function(`return (${selectNative.toString()})`)();
    const applied = run.call(node, ["english (u.s.)"]);
    assert.equal(applied.verified, false);
    const result = readNativeSelection.call(node, applied.expected);
    assert.deepEqual(result, { kind: "native", selected: [{ index: 0, value: "en-US", label: "English (U.S.)" }], verified: true });
    assert.deepEqual(node.events, [["input", true], ["change", true]]);
  });
  test("label precedence then value, and exact index with duplicated values", () => {
    const node = fixture();
    assert.equal(applyAndRead(node, ["English (U.S.)"]).selected[0].index, 0);
    assert.equal(applyAndRead(node, ["fr"]).selected[0].index, 1);
    node.options[1].value = "en-US";
    assert.equal(applyAndRead(node, [1]).selected[0].index, 1);
  });
  test("multiple replaces the set, deduplicates, and [] clears with input/change", () => {
    const node = fixture(true);
    assert.deepEqual(applyAndRead(node, [0, "French", 0]).selected.map((o) => o.index), [0, 1]);
    assert.deepEqual(applyAndRead(node, [1]).selected.map((o) => o.index), [1]);
    const before = node.events.length;
    assert.deepEqual(applyAndRead(node, []), {kind:"native", selected:[], verified:true});
    assert.deepEqual(node.events.slice(before), [["input",true],["change",true]]);
    assert.deepEqual(node.options.flatMap((o, i) => o.selected ? [i] : []), []);
  });
  test("single >1/empty, disabled control/option/optgroup refuse before side effects", () => {
    for (const request of [[], [0, 1]]) {
      const node = fixture(); assert.equal(applyAndRead(node, request).error.data.reason, "single_select"); assert.deepEqual(node.events, []);
    }
    const node = fixture(true);
    node.disabled = true;
    assert.equal(applyAndRead(node, [0]).error.data.reason, "select_disabled");
    node.disabled = false;
    assert.equal(applyAndRead(node, [0, 3]).error.data.reason, "option_disabled");
    assert.equal(node.options[0].selected, false); assert.deepEqual(node.events, []);
    node.options[1].parentElement = { tagName: "OPTGROUP", disabled: true };
    assert.equal(applyAndRead(node, [1]).error.data.reason, "option_disabled");
  });
  test("unknown label/index returns bounded available options without mutation", () => {
    const node = new Select(Array.from({ length: 90 }, (_, i) => new Option("x".repeat(300), `${i}`)), true);
    for (const request of [["missing"], [99]]) {
      const out = applyAndRead(node, request);
      assert.equal(out.error.code, "element_not_found"); assert.equal(out.error.data.reason, "option_not_found");
      assert.equal(out.error.data.available.length, 50); assert.equal(out.error.data.available[0].label.length, 120);
      assert.deepEqual(node.events, []);
    }
  });
  test("a controlled form restoring selection is honestly unverified", () => {
    const node = fixture();
    node.listener = () => { for (const o of node.options) o._selected = false; };
    const out = applyAndRead(node, [0]);
    assert.equal(out.verified, false); assert.deepEqual(out.selected, []);
  });
  test("ARIA/custom target is a typed desktop-only refusal, not a fake native success", () => {
    const node = { ownerDocument: fixture().ownerDocument };
    const out = applyAndRead(node, ["x"]);
    assert.equal(out.error.code, "capability_unsupported");
    assert.equal(out.error.data.reason, "aria_select_unsupported");
  });

  // The real extension command wrapper, with only the host boundary synthetic.
  const commandOutput = join(dir, "command.mjs");
  await build({ entryPoints: ["src/commands/select.ts"], bundle: true, platform: "node", format: "esm", outfile: commandOutput,
    plugins: [{ name: "host-boundary", setup(b) {
      b.onResolve({ filter: /^\.\.\/cdp$/ }, () => ({ path: "cdp", namespace: "fake" }));
      b.onResolve({ filter: /^\.\.\/origins$/ }, () => ({ path: "origins", namespace: "fake" }));
      b.onResolve({ filter: /^\.\/input$/ }, () => ({ path: "input", namespace: "fake" }));
      b.onLoad({ filter: /.*/, namespace: "fake" }, ({ path }) => ({ contents: path === "cdp" ? `
        export class BridgeCommandError extends Error { constructor(code,message,data={}) {super(message);this.code=code;this.data=data;} }
        export const requireSurface=(tab)=>globalThis.peer.requireSurface(tab);
        export const cdp=(...args)=>globalThis.peer.cdp(...args);`
        : path === "origins" ? `export const withOriginGate=(...args)=>globalThis.peer.gate(...args);`
        : `export const resolveTarget=(...args)=>globalThis.peer.resolve(...args);` }));
    } }],
  });
  const { selectOptions } = await import(pathToFileURL(commandOutput));
  test("real command sends Runtime function/args, no Input/layout/popup/focus", async () => {
    const node = fixture(); const calls = [];
    globalThis.chrome = { tabs: { get: async () => ({ url: "https://example.test", title: "Select" }) } };
    globalThis.peer = {
      requireSurface: async () => ({ tabId: 7 }), resolve: async (_, selector) => { assert.equal(selector, "e5"); return { nodeId: 12 }; },
      gate: async (_, requestId, fn) => { assert.equal(requestId, "r1"); return fn(); },
      cdp: async (_, method, args) => {
        calls.push(method);
        if (method === "DOM.resolveNode") return { object: { objectId: "o12" } };
        assert.equal(method, "Runtime.callFunctionOn");
        const value = new Function(`return (${args.functionDeclaration})`)().call(node, args.arguments[0].value);
        return { result: { value } };
      },
    };
    const out = await selectOptions({ tab: "t7", selector: "e5", options: ["French"] }, "r1");
    assert.equal(out.selected[0].value, "fr"); assert.equal(out.kind, "native"); assert.equal(out.verified, true);
    assert.deepEqual(calls, ["DOM.resolveNode", "Runtime.callFunctionOn", "DOM.resolveNode", "Runtime.callFunctionOn"]);
    await assert.rejects(selectOptions({ tab: "t7", selector: "e5", options: ["missing"] }, "r1"), (e) => e.code === "element_not_found" && e.data.available.length === 4);
    node.multiple = true;
    const cleared = await selectOptions({ tab:"t7", selector:"e5", options:[] }, "r1");
    assert.deepEqual(cleared.selected, []); assert.equal(cleared.verified, true);
    assert.deepEqual(node.events.slice(-2), [["input",true],["change",true]]);
    console.log("native multiple []: selected=[], verified=true; input/change fired");
    node.multiple = false;
    const eventCount = node.events.length;
    await assert.rejects(selectOptions({ tab:"t7", selector:"e5", options:[] }, "r1"), (e) => e.code === "element_not_found" && e.data.reason === "single_select");
    assert.equal(node.events.length, eventCount);
    for (const options of [[""],["  "],[true],[1.2],Array(101).fill(0)]) {
      await assert.rejects(selectOptions({tab:"t7",selector:"e5",options},"r1"), (e) => e.data.reason === "invalid_arguments");
    }
    assert.equal(node.events.length, eventCount);
  });
  test("replacement negative control: live default beats detached applied Beta", async () => {
    for (const delayed of [false, true]) {
      let live = fixture(); const original = live;
      const replacement = fixture(); replacement.options[0].selected = true;
      const replace = () => { original.isConnected = false; live = replacement; };
      original.listener = (event) => {
        if (event.type === "change") {
          if (delayed) setTimeout(replace, 10); else replace();
        }
      };
      let resolves = 0; let gateOpen = false;
      globalThis.chrome = { tabs: { get: async () => ({ url: "https://example.test", title: "Controlled" }) } };
      globalThis.peer = {
        requireSurface: async () => ({ tabId: 7 }),
        resolve: async () => { resolves++; return { nodeId: live === original ? 1 : 2 }; },
        gate: async (_, __, fn) => { gateOpen = true; try { return await fn(); } finally { gateOpen = false; } },
        cdp: async (_, method, args) => {
          if (method === "DOM.resolveNode") return { object: { objectId: String(args.nodeId) } };
          assert.equal(method, "Runtime.callFunctionOn"); assert.equal(gateOpen, true);
          const node = args.objectId === "1" ? original : replacement;
          return { result: { value: new Function(`return (${args.functionDeclaration})`)().call(node, args.arguments[0].value) } };
        },
      };
      const out = await selectOptions({ tab: "t", selector: "#s", options: ["French"] }, "r");
      assert.equal(original.isConnected, false);
      assert.equal(original.options[1].selected, true); // detached nodes falsely look successful
      assert.equal(resolves, 2);
      assert.equal(out.verified, false);
      assert.deepEqual(out.selected, [{ index: 0, value: "en-US", label: "English (U.S.)" }]);
      console.log(`replacement ${delayed ? "delayed" : "sync"}: verified=${out.verified}, live=${out.selected[0].value}, detached=${original.options[1].value}`);
    }
  });
  test("a stale snapshot ref cannot verify a detached target", () => {
    const node = fixture(); const applied = selectNative.call(node, ["French"]);
    node.isConnected = false;
    assert.deepEqual(readNativeSelection.call(node, applied.expected), { kind: "native", selected: [], verified: false });
  });
} finally {
  // Node's tests run after module evaluation; defer exact scratch cleanup until
  // the suite is done, not before the tests can load the command module.
  test.after(async () => { delete globalThis.peer; delete globalThis.chrome; await rm(dir, { recursive: true, force: true }); });
}
