// Runs the VENDORED prelude.js (the exact bytes the assembler injects) against a minimal DOM,
// drives it with a scripted conversation and prints what it posted to `parent` as JSON.
//
// WHY A STUB DOM AND NOT A BROWSER: the unit suite cannot assume a browser, but the two
// things the C0 contract asserts about the prelude -- the frame->host message shapes (the
// per-frame nonce echo) and the label geometry it computes -- are decided by its own JS, not
// by layout. Text width comes from a fixed-advance canvas stub (CHAR_PX per character,
// calibrated to the design round's measured `1,000 USD/day` = 83.1 px / 13 chars), so a
// geometry reading here is a MODEL of the browser's, good for "did the layout rule change",
// not for sub-pixel truth. The rendered browser evidence is in the PR, not here.
//
// usage: node prelude_harness.mjs <prelude.js> <scenario.json>
//   scenario = {width, data, steps:[{host:{...}} | {other:{...}} | {flush:true} | {size:true}
//                                    | {error:"msg"} | {draw:"LO.bar(...)"}
//                                    | {script:"..."}]}
//   `script` runs a component's inline script as the browser would: a throw is NOT fatal to
//   the page, it becomes the window `error` event ("Uncaught <error>"). That is how a
//   component that fails during parse -- before the host's theme push -- is reproduced.
// output   = {posts:[...], texts:[{text,x,y,anchor,left,right,top,bottom}...], svgW}
import fs from "node:fs";
import vm from "node:vm";

const [, , preludePath, scenarioPath] = process.argv;
const scenario = JSON.parse(fs.readFileSync(scenarioPath, "utf8"));
const CHAR_PX = 6.4;

class Node {
  constructor(tag, ns) {
    this.tag = tag;
    this.attrs = {};
    this.children = [];
    this.dataset = {};
    this.style = { setProperty() {} };
    this.isConnected = true;
    this._text = "";
    this.clientWidth = 0;
  }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  append(...xs) { for (const x of xs) this.children.push(x); }
  replaceChildren() { this.children = []; }
  set textContent(v) { this._text = String(v); }
  get textContent() { return this._text; }
  getBoundingClientRect() { return { height: 100, width: this.clientWidth }; }
}

const posts = [];
const listeners = {};
const frames = [];
const parent = { postMessage: (m) => posts.push(JSON.parse(JSON.stringify(m))) };
const root = new Node("html");
root.clientWidth = scenario.width ?? 620;
const dataEl = new Node("script");
dataEl._text = JSON.stringify(scenario.data ?? {});
const target = new Node("div");
target.clientWidth = scenario.width ?? 620;

const document = {
  documentElement: root,
  getElementById: (id) => (id === "lo-data" ? dataEl : id === "c" ? target : null),
  createElement: (tag) =>
    tag === "canvas"
      ? { getContext: () => ({ set font(v) {}, measureText: (s) => ({ width: String(s).length * CHAR_PX }) }) }
      : new Node(tag),
  createElementNS: (_ns, tag) => new Node(tag),
};
const ctx = vm.createContext({
  document, parent, console,
  addEventListener: (t, f) => (listeners[t] ||= []).push(f),
  requestAnimationFrame: (f) => frames.push(f) && frames.length,
  cancelAnimationFrame: (id) => { if (id && frames[id - 1]) frames[id - 1] = null; },
  setTimeout: () => 0,
  getComputedStyle: () => ({ getPropertyValue: () => "system-ui" }),
  matchMedia: () => ({ matches: false }),
  CSS: { supports: () => true },
  ResizeObserver: class { observe() {} },
});
ctx.window = ctx;
vm.runInContext(fs.readFileSync(preludePath, "utf8"), ctx);

const fire = (type, ev) => (listeners[type] || []).forEach((f) => f(ev));
const flush = () => { const f = frames.splice(0); f.forEach((x) => x && x()); };
for (const step of scenario.steps ?? []) {
  if (step.host) fire("message", { source: parent, data: step.host });
  else if (step.other) fire("message", { source: {}, data: step.other });
  else if (step.flush) flush();
  else if (step.size) { vm.runInContext("LO.size()", ctx); flush(); }
  else if (step.error) fire("error", { message: step.error });
  else if (step.draw) { vm.runInContext(step.draw, ctx); flush(); }
  else if (step.script) {
    try { vm.runInContext(step.script, ctx); } catch (e) { fire("error", { message: "Uncaught " + String(e) }); }
  }
}

const texts = [];
const walk = (n, dx, dy) => {
  if (n.tag === "g" && n.attrs.transform) {
    const m = /translate\(([-\d.]+),([-\d.]+)\)/.exec(n.attrs.transform);
    if (m) { dx += +m[1]; dy += +m[2]; }
  }
  if (n.tag === "text") {
    const label = n.children.filter((c) => typeof c === "string").join("") || n._text;
    const x = +n.attrs.x + dx, y = +n.attrs.y + dy, w = label.length * CHAR_PX;
    const a = n.attrs["text-anchor"] || "start";
    const left = a === "end" ? x - w : a === "middle" ? x - w / 2 : x;
    // 14 px tall box with the baseline 4 px above its bottom: the design round's own
    // measured boxes (45.7-59.7 for a tick, 49.2-63.2 for a value label).
    texts.push({ text: label, anchor: a, left, right: left + w, top: y - 10, bottom: y + 4 });
  }
  for (const c of n.children) if (typeof c === "object") walk(c, dx, dy);
};
walk(target, 0, 0);
let svgW = 0;
const findSvg = (n) => { if (n.tag === "svg") svgW = +n.attrs.viewBox.split(" ")[2]; for (const c of n.children) if (typeof c === "object") findSvg(c); };
findSvg(target);
process.stdout.write(JSON.stringify({ posts, texts, svgW }));
