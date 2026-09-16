/* Non-rendering DOM host for ordering/IPC tests. Browser screenshots use the
 * real markup via popup-performance-server.mjs; this host proves no geometry. */
export function installPopupDom(html, onMutation = () => {}) {
  const nodes = new Map();
  function make(id, classes = "") {
    const handlers = {};
    const names = new Set(classes.split(/\s+/));
    const props = new Map();
    let text = "";
    const node = { id, value: "", dataset: {}, disabled: false, children: [], scrollTop: 0,
      classList: { contains: c => names.has(c), add(c) { names.add(c); onMutation(); }, remove(c) { names.delete(c); onMutation(); }, toggle(c, force) { const on = force ?? !names.has(c); if (on) names.add(c); else names.delete(c); onMutation(); return on; } },
      style: { setProperty(k, v) { props.set(k, v); onMutation(); }, removeProperty(k) { props.delete(k); onMutation(); }, getPropertyValue: k => props.get(k) ?? "" },
      setAttribute() {}, removeAttribute() {}, querySelectorAll: () => [],
      addEventListener(event, fn) { (handlers[event] ||= []).push(fn); }, click() { handlers.click?.forEach(fn => fn()); },
      focus() { document.activeElement = node; },
      replaceChildren(...children) { node.children = children; node.value = children[0]?.value ?? ""; onMutation(); },
      get options() { return node.children; }, get selectedOptions() { return node.children.filter(c => c.value === node.value); },
      get textContent() { return text; }, set textContent(value) { text = value; onMutation(); },
    };
    return node;
  }
  for (const match of html.matchAll(/<[^>]*\bid="([^"]+)"[^>]*>/g)) nodes.set(match[1], make(match[1], /class="([^"]*)"/.exec(match[0])?.[1] || ""));
  globalThis.document = { getElementById: id => nodes.get(id) ?? null, querySelector: () => null, querySelectorAll: () => [], createElement: () => make("option"), addEventListener() {}, documentElement: make("html"), body: make("body"), scrollingElement: make("scroller"), activeElement: null };
  globalThis.window = { close() {}, matchMedia: () => ({ matches: false, addEventListener() {} }) };
  const storage = new Map();
  globalThis.localStorage = { getItem: k => storage.get(k) ?? null, setItem: (k, v) => storage.set(k, String(v)), removeItem: k => storage.delete(k) };
  const visible = () => [...nodes.values()].filter(n => n.classList.contains("state") && !n.classList.contains("hidden")).map(n => n.id);
  return { nodes, visible };
}
