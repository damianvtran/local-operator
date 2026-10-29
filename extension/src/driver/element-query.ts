/* The read-only element query: CSS in, descriptors out.
 *
 * WHY THIS EXISTS. The measured incident's single most useful unanswered
 * question was "what is the id/name/role of the checkbox next to 'Active'?"
 * Every route the tool had was closed: `read` returns text without attributes,
 * `snapshot` returns an AX tree that was truncated, and there is no `eval`
 * (deliberately — it is arbitrary code execution on a page the agent is
 * driving). An agent that needs to know a control's NAME attribute had to
 * guess, and the diagnosis records nine guesses.
 *
 * This answers that in one call, and it is safe to expose where eval is not:
 * the input is a CSS selector handed to `document.querySelectorAll`, the output
 * is a fixed, bounded record per element, and nothing here writes, navigates,
 * clicks or evaluates caller-supplied code. It runs in the page's isolated
 * world, like `read` and the geometry readers.
 *
 * SELF-CONTAINED BY CONSTRUCTION, because both hosts serialize the function
 * itself (chrome.scripting.executeScript({func}) in the extension, the
 * generator-captured source in the app). No imports, no closures over module
 * scope, no caller text in executable position: the selector is an ARGUMENT,
 * never concatenated into code — the same posture `read` states for its own.
 *
 * BOUNDS, all applied in the page so the result is bounded at the source:
 * at most 50 elements, every identity string clipped to 120 chars, every value
 * to 120, and the values of PASSWORD inputs are never returned at all — a
 * query may be pointed at a login form, and a "read-only" action that echoes a
 * typed password into a transcript is exactly the leak the secret-ref design
 * exists to prevent.
 */

export type QueryDescriptor = {
  tag: string;
  id: string;
  name: string;
  role: string;
  ariaLabel: string;
  text: string;
  type: string;
  visible: boolean;
  inView: boolean;
  disabled: boolean;
  /** `""` for anything that is not a checkable control. */
  checked: boolean | null;
  /** A password input's value is NEVER returned (see the note above). */
  value: string;
  rect: { x: number; y: number; width: number; height: number };
};

export type QueryResult = {
  count: number;
  truncated: boolean;
  matches: QueryDescriptor[];
  /** Set when the selector is not valid CSS — measured, because a THROWN error
   *  does not survive the trip: Electron's isolated-world primitive rejects with
   *  "Script failed to execute", losing the page's own message. Returning the
   *  fact keeps the caller's mistake nameable on both hosts. */
  error?: "invalid_selector";
};

export const queryElements = (selector: string, limit: number): QueryResult | null => {
  const MAX_ELEMENTS = 50;
  const MAX_IDENTITY = 120;
  const MAX_TEXT = 120;
  const MAX_VALUE = 120;
  if (typeof selector !== "string" || !selector.trim()) return null;
  const wanted = typeof limit === "number" && Number.isFinite(limit) ? Math.floor(limit) : 20;
  const bound = Math.max(1, Math.min(MAX_ELEMENTS, wanted));
  const round2 = (value: number): number => {
    const rounded = Math.round(value * 100) / 100;
    return rounded === 0 ? 0 : rounded;
  };
  const clip = (value: string, cap: number): string =>
    value.length > cap ? `${value.slice(0, cap - 1)}\u2026` : value;
  const attrOf = (el: Element, name: string): string =>
    clip(String(el.getAttribute(name) || ""), MAX_IDENTITY);
  const visibleOf = (el: Element, rect: DOMRect): boolean => {
    if (rect.width <= 0 || rect.height <= 0) return false;
    const computed = getComputedStyle(el);
    if (computed.display === "none" || computed.visibility === "hidden") return false;
    // An ancestor's `display:none` gives a zero rect (handled above); opacity 0
    // is still "rendered" and is reported as visible, which is what a layout
    // question means. `content-visibility:hidden` is the one that keeps a
    // non-zero rect while painting nothing, so it is checked explicitly.
    return computed.contentVisibility !== "hidden";
  };
  let nodes: Element[];
  try {
    nodes = Array.from(document.querySelectorAll(selector));
  } catch {
    // Not valid CSS: reported as a VALUE, never thrown (see QueryResult.error).
    return { count: 0, truncated: false, matches: [], error: "invalid_selector" };
  }
  const viewport = {
    width: document.documentElement.clientWidth || window.innerWidth || 0,
    height: document.documentElement.clientHeight || window.innerHeight || 0,
  };
  const matches: QueryDescriptor[] = [];
  for (const el of nodes.slice(0, bound)) {
    const rect = el.getBoundingClientRect();
    const tag = clip(String(el.tagName || "").toLowerCase(), MAX_IDENTITY);
    const type = attrOf(el, "type").toLowerCase();
    const isPassword = tag === "input" && type === "password";
    const rawValue = (el as HTMLInputElement).value;
    // `checked` is a property on EVERY input, so asking it of a text field
    // answers `false` — which the renderer printed as "unchecked" on a password
    // box (measured). Only a checkable control has a checked STATE; everything
    // else reports null and the text says nothing.
    const checkable = tag === "input" && (type === "checkbox" || type === "radio");
    const checked = checkable ? Boolean((el as HTMLInputElement).checked) : null;
    matches.push({
      tag,
      id: attrOf(el, "id"),
      name: attrOf(el, "name"),
      role: attrOf(el, "role"),
      ariaLabel: attrOf(el, "aria-label"),
      // `innerText` is the RENDERED text (a hidden node reports ""), which is
      // what an agent reading a page means by "the text"; `textContent` would
      // report text nobody can see, including inside a collapsed section.
      text: clip(String((el as HTMLElement).innerText || "").replace(/\s+/g, " ").trim(), MAX_TEXT),
      type,
      visible: visibleOf(el, rect),
      inView:
        rect.top < viewport.height &&
        rect.bottom > 0 &&
        rect.left < viewport.width &&
        rect.right > 0,
      disabled: el.hasAttribute("disabled") || (el as HTMLInputElement).disabled === true,
      checked,
      value: isPassword || rawValue === undefined || rawValue === null
        ? ""
        : clip(String(rawValue), MAX_VALUE),
      rect: {
        x: round2(rect.x),
        y: round2(rect.y),
        width: round2(rect.width),
        height: round2(rect.height),
      },
    });
  }
  return { count: nodes.length, truncated: nodes.length > matches.length, matches };
};
