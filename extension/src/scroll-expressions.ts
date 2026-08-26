// Fixed scroll expressions, kept in a chrome-free module so the pure tests can
// load them (cdp.ts registers chrome listeners at module scope and cannot be
// bundled into a Node test). Only constant code ever reaches the page: the
// caller selects a branch or supplies validated numbers, never strings.
//
// CONSTRAINT — behavior:'instant' everywhere: window.scrollBy/scrollTo and
// Element.scrollIntoView honour the page's CSS `scroll-behavior: smooth`, and a
// smooth scroll is a requestAnimationFrame-driven animation. Chrome throttles
// rAF to ZERO in hidden tabs, so in our intentionally-background surface a
// smooth scroll starts an animation that never advances — observed live as
// scrollY stuck at 0 after repeated scrolls. 'instant' overrides the page CSS
// and does not depend on the animation-frame clock.

// A "page" step leaves this much overlap so the agent does not skip a band of
// content between reads — the same courtesy a PageDown key gives.
export const PAGE_OVERLAP_PX = 80;

/** One viewport down (minus overlap): the default "read more" gesture. */
export function defaultScrollExpression(): string {
  return `window.scrollBy({left: 0, top: window.innerHeight - ${PAGE_OVERLAP_PX}, behavior: 'instant'})`;
}

/** Explicit pixel deltas; callers must pass finite numbers, never page input. */
export function deltaScrollExpression(dx: number, dy: number): string {
  return `window.scrollBy({left: ${dx}, top: ${dy}, behavior: 'instant'})`;
}

/** Body for Runtime.callFunctionOn against a resolved node: center the element
 * so it is usable after the scroll rather than jammed against a viewport edge. */
export const SCROLL_INTO_VIEW_FN =
  `function(){ this.scrollIntoView({block:'center', inline:'center', behavior:'instant'}); }`;

/** Fixed expression per direction keyword; unknown directions are a no-op. */
export function scrollExpressionFor(direction: string): string {
  const page = `(window.innerHeight - ${PAGE_OVERLAP_PX})`;
  const across = `(window.innerWidth - ${PAGE_OVERLAP_PX})`;
  switch (direction) {
    case "top":
      return `window.scrollTo({left: window.scrollX, top: 0, behavior: 'instant'})`;
    case "bottom": {
      const de = `(document.scrollingElement||document.documentElement)`;
      return `window.scrollTo({left: window.scrollX, top: ${de}.scrollHeight, behavior: 'instant'})`;
    }
    case "up":
      return `window.scrollBy({left: 0, top: -${page}, behavior: 'instant'})`;
    case "down":
      return `window.scrollBy({left: 0, top: ${page}, behavior: 'instant'})`;
    case "left":
      return `window.scrollBy({left: -${across}, top: 0, behavior: 'instant'})`;
    case "right":
      return `window.scrollBy({left: ${across}, top: 0, behavior: 'instant'})`;
    default:
      return `void 0`;
  }
}
