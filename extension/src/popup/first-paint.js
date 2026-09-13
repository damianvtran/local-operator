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
  // Keep in sync with PIN_HINT_KEY, PIN_CONNECTED/PIN_PAIRING/PIN_UNRESPONSIVE
  // and show() in popup.ts.
  //
  // A PIN PER STATE, not a boolean. Two of the three states this can land on
  // cannot be told apart by a boolean, and the one a boolean has to collapse is
  // the wedged-but-paired card this popup exists for: pinned to the connected
  // height it opened 167.8px short of the card it settled on (design D3-1).
  // popup.ts writes the pin for the card it just rendered, and this script
  // reproduces that height before the module runs.
  var KEY = "lop:pin-hint";
  // The boolean key the previous revision wrote. Read once as a fallback so the
  // rename does not cost an existing paired browser a resize; "1" meant the
  // connected card. Mirrors the same fallback in popup.ts.
  var LEGACY_KEY = "lop:paired-hint";
  var PIN_CONNECTED = "86px";
  var PIN_PAIRING = "219px";
  var PIN_UNRESPONSIVE = "254px";
  var PINS = [PIN_CONNECTED, PIN_PAIRING, PIN_UNRESPONSIVE];
  var pin = null;
  try {
    var stored = localStorage.getItem(KEY);
    if (stored !== null && PINS.indexOf(stored) !== -1) pin = stored;
    else if (localStorage.getItem(LEGACY_KEY) === "1") pin = PIN_CONNECTED;
  } catch (error) {
    // Storage unavailable (disabled, quota, partitioned). No hint is the safe
    // answer: it is the behaviour before this file existed, and it is what a
    // browser we cannot identify must get.
  }
  // <head> runs before <body> exists, so the pin is applied as a stylesheet
  // rule rather than by reaching for the element. This also keeps it out of the
  // element's inline style, leaving popup.ts's own applyPendingPin() — which
  // runs later and sets an inline style — able to override it without a fight.
  var style = document.createElement("style");
  style.textContent = "#pending{min-height:" + (pin || PIN_PAIRING) + "}";
  document.documentElement.appendChild(style);
})();
