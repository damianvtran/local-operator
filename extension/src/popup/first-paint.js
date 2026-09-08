/* Size the pre-render placeholder BEFORE the first paint.
 *
 * Plain JavaScript, not TypeScript, and loaded as a CLASSIC render-blocking
 * script from <head> — deliberately, and it is the whole point of the file:
 *
 *   - popup.js is `<script type="module">`, which is DEFERRED. It cannot run
 *     until the document is parsed, so anything it does to layout happens after
 *     the compositor has already had the chance to paint.
 *   - popup.css therefore has to ship SOME default pin, and whichever one it
 *     ships is wrong for the other population. Shipping the unpaired (tall) pin
 *     meant the already-paired user was pinned tall until the module ran, and
 *     the compositor produced a frame inside that window on 9 of 13 opens —
 *     a 340px -> 207px resize on exactly the path this PR set out to settle.
 *   - MV3's default CSP (`script-src 'self'`) forbids an inline <script>, so
 *     the pre-paint work has to come from a file. A classic script in <head>
 *     is fetched and executed before the body is parsed, which closes the
 *     window rather than narrowing it.
 *
 * It is intentionally the smallest thing that can run this early: no imports,
 * no bundler entry, no shared module. Importing anything would make it a module
 * and hand back the deferral this file exists to avoid.
 *
 * The value written here is a LAYOUT HINT and never gates behaviour — render()
 * still paints whatever /health reports. A stale or absent hint costs exactly
 * one resize, which is the behaviour without the file at all, so there is
 * nothing to fail closed about. popup.ts owns the key and the measured pins;
 * the duplication of both is the price of running before the module system.
 */
(function () {
  // Keep in sync with PAIRED_HINT_KEY and applyPendingPin() in popup.ts.
  var KEY = "lop:paired-hint";
  var PAIRED_PIN = "86px";
  var UNPAIRED_PIN = "219px";
  var paired = false;
  try {
    paired = localStorage.getItem(KEY) === "1";
  } catch (error) {
    // Storage unavailable (disabled, quota, partitioned). The unpaired pin is
    // the safe default: it is the state a browser we cannot identify is most
    // likely to be in on its first open.
  }
  // <head> runs before <body> exists, so the pin is applied as a stylesheet
  // rule rather than by reaching for the element. This also keeps it out of the
  // element's inline style, leaving popup.ts's own applyPendingPin() — which
  // runs later and sets an inline style — able to override it without a fight.
  var style = document.createElement("style");
  style.textContent = "#pending{min-height:" + (paired ? PAIRED_PIN : UNPAIRED_PIN) + "}";
  document.documentElement.appendChild(style);
})();
