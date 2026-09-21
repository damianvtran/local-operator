import { requireSurface } from "../cdp";
import { requireConsent } from "../consent";
import { CHROME_API_DEADLINE_MS, deadline } from "../driver/deadline";
import { safeName } from "../driver/file-transfer-policy";
import { CAPS } from "../driver/file-transfer.tables.gen";
import { click } from "./input";

/** ``scheme://host[:port]`` for a URL, or ``""`` when it is not one.
 *
 * Origin granularity rather than the whole URL, and the SAME granularity Python
 * applies (`browser_files._origin_of`) to the `referrer` it is given: comparing
 * full URLs would drop the association whenever a page moved between two paths of
 * the same site, and comparing nothing at all is how a browser-wide API turns into
 * a browser-wide action.
 *
 * THE SHARED CONTRACT, written down because the two sides cannot share CODE (Python
 * has no URL parser here): one of `http`, `https`, `ws`, `wss`; scheme and host
 * lowercased; userinfo dropped; a default port dropped; `""` for anything else.
 * The scheme list is not decoration — it is what makes the two agree. Round 3 (M2)
 * measured the divergence it closes: `new URL('blob:https://example.test/uuid')`
 * gives `.origin === 'https://example.test'` (the INNER URL's origin), while
 * `urlsplit` sees a scheme with no host and answers `""`. Restricting both sides to
 * the four schemes a driven page can be is the fix; a contract comment claiming
 * agreement the code did not have was the defect.
 */
function originOf(url: string): string {
  if (!url) return "";
  try {
    const parsed = new URL(url);
    const scheme = parsed.protocol.replace(/:$/, "").toLowerCase();
    if (!["http", "https", "ws", "wss"].includes(scheme)) return "";
    // `URL.origin` is used rather than string surgery for the rest: Chrome
    // canonicalises both inputs itself (lowercased host, default port omitted),
    // which is exactly the form the Python side parses.
    return parsed.origin === "null" ? "" : parsed.origin;
  } catch {
    return "";
  }
}

/* Save a file the PAGE offers, by watching what Chrome actually writes.
 *
 * THE PRIMITIVE, and why it is `chrome.downloads` and not the debugger.
 * `Page.setDownloadBehavior` / `Browser.setDownloadBehavior` would let the host
 * choose the destination, and the design was built around them — but a
 * tab-scoped `chrome.debugger` session may not use either on current stable
 * Chrome (measured 2026-09-18, Chrome 153.0.8010.53: `-32000 "Cannot not access
 * browser-level commands"` and `-32601`; no browser target is attachable and no
 * `downloadWillBegin`/`downloadProgress` event is delivered at all — design
 * §17.1). So the extension does not choose the destination: Chrome writes to the
 * user's real download directory and we LEARN the absolute path from the
 * `DownloadItem` Chrome hands us, exactly as design §17.5's probe measured.
 *
 * WHAT THAT COSTS, stated where it bites: the file exists in the user's own
 * `~/Downloads` under the page's own name for the length of the download, before
 * Python relocates it into the session quarantine directory (design §11.5 R7 —
 * the residual the operator accepted with this switch). This handler therefore
 * never claims the file is "saved somewhere": it reports the path Chrome wrote
 * and the state Chrome reported, and the HARNESS is what moves, classifies,
 * renames, chmods and audits it. Nothing here inspects a byte — the extension
 * cannot read a downloaded file's contents at all, which is why every content
 * verdict lives in Python (`driver/file-transfer-policy.ts` says the same from
 * the other side).
 *
 * FOUR RULES ARE STRUCTURAL:
 *
 *   1. THE OPERATOR'S SWITCH IS CHECKED FIRST, and checked here rather than only
 *      in the advertisement: a daemon that predates the advertisement sends
 *      regardless, and a switch flipped off between the advertisement and the
 *      command must still refuse (consent.ts's `requireConsent`).
 *   2. `search` IS THE SOURCE OF TRUTH, events are the fast path. Both are used,
 *      in that order of authority: a missed `onCreated` (a service worker that
 *      was busy, a download Chrome started before our listener was installed)
 *      must not turn a real download into "nothing started", and a `search` that
 *      is a poll interval behind must not let a file run past the ceiling when an
 *      event could have cancelled it. Rule 4 explains the ceiling's shape.
 *   3. NOTHING IS DELETED HERE except by `chrome.downloads.cancel`, which stops a
 *      transfer. The partial file a cancel leaves behind is REPORTED with its
 *      path so Python can remove it — the extension cannot unlink files, and a
 *      report that claimed "nothing is left" would be a claim about the user's
 *      disk that this process cannot make.
 *   4. THE CEILING IS ENFORCED BY CANCELLING NEAR IT, not by refusing after the
 *      fact. A 256 MiB ceiling checked only when a 1 GiB file completes has
 *      already written the 1 GiB. So the first event or poll that sees
 *      `bytesReceived` cross the ceiling cancels the transfer, and the overshoot
 *      is bounded by what arrives between two samples rather than by the file's
 *      size. `totalBytes` is NOT trusted for this: Chrome reports 0 (unknown) for
 *      a chunked or streaming response, and treating "unknown" as "small" is how
 *      a ceiling becomes decoration — an unknown size is reported as unknown and
 *      the harness judges the landed bytes.
 */

/** How long to wait after the last observed change before declaring the set final.
 *
 * A page that starts several downloads in a burst (a multi-file export) creates
 * them in sequence; returning on the first `complete` would report one file and
 * silently drop the rest. 750 ms is longer than the gap between the creations of
 * one user gesture and far shorter than any download worth waiting for.
 */
const SETTLE_MS = 750;

/** How often the poll re-reads `search` while waiting (rule 2). */
const POLL_MS = 200;

interface ObservedItem {
  name: string;
  id: number;
  path: string;
  referrer: string;
  bytes: number;
  totalBytes: number;
  state: string;
  mime: string;
  danger: string;
  exists: boolean;
  paused: boolean;
  error: string;
  cancelled: string;
}

function describe(chromeError: string): string {
  return chromeError.replace(/\s+/g, " ").slice(0, 120).trim();
}

/** One `DownloadItem`, reduced to the facts the harness needs.
 *
 * `bytes` is what Chrome says it received, and it is reported even when the
 * transfer was cancelled: the harness uses it to corroborate the file on disk
 * before it touches anything, and a partial write whose size is not reported
 * cannot be told from a file that was already there (builtin's intake rule).
 */
function reduce(item: chrome.downloads.DownloadItem, cancelled: string): ObservedItem {
  return {
    // The name is the PAGE-SUPPLIED filename, so it goes through the shared
    // sanitiser before it reaches a report, a log or an approval card — the same
    // function Python applies, deliberately, so the name this report is keyed on
    // and the name the harness writes into the quarantine directory are the same
    // string. `safeName` also reduces any directory component to its last
    // segment, which is why `filename` can be passed whole.
    name: safeName(String(item.filename ?? "")),
    id: item.id,
    // The page that caused this download, as Chrome reports it. Carried to the
    // harness because the association is the one thing this API does not give us:
    // a `DownloadItem` has no `tabId` (verified on Chrome 153.0.8010.53 — the field
    // is simply absent from the item), so without this the harness cannot tell the
    // transfer THIS call caused from one the user started by hand in another tab
    // during the same window — and moving the user's own download into quarantine
    // would be the exact outcome the approval card promised would not happen.
    referrer: String(item.referrer ?? ""),
    path: String(item.filename ?? ""),
    bytes: Number(item.bytesReceived ?? 0),
    // -1 is this handler's own sentinel for "Chrome did not say", which the
    // harness must read as unknown rather than as an empty file. Chrome's own
    // unknown is 0 on some responses and -1 on others, and both mean "no answer"
    // here, so they are collapsed into one value the Python side can test for.
    totalBytes: Number(item.totalBytes ?? 0) > 0 ? Number(item.totalBytes) : -1,
    state: String(item.state ?? ""),
    mime: String(item.mime ?? ""),
    danger: String(item.danger ?? ""),
    exists: item.exists !== false,
    paused: item.paused === true,
    error: describe(String(item.error ?? "")),
    cancelled,
  };
}

export async function download(
  params: Record<string, unknown>,
  requestId: string,
): Promise<Record<string, unknown>> {
  // Rule 1: the operator's switch, before anything touches the page.
  await requireConsent("download");
  const surface = await requireSurface(params.tab);
  const tabId = surface.tabId;
  let url = "";
  try {
    const tab = await deadline(chrome.tabs.get(tabId), CHROME_API_DEADLINE_MS, `chrome.tabs.get(${tabId})`);
    url = String(tab?.url ?? "");
  } catch {
    // The URL is audit context, not a gate: a tab that vanished between the
    // handle check and here is reported by `requireSurface`'s own deadline on the
    // NEXT call, and failing the download over a missing label would refuse a
    // working capability for a cosmetic reason.
    url = "";
  }

  /* WHICH DOWNLOADS THIS CALL MAY TOUCH (round-1 R2/R7).
   *
   * `chrome.downloads` is browser-wide and a `DownloadItem` carries no `tabId`
   * (verified on Chrome 153.0.8013.53: the field is simply absent), so the ONLY
   * association available is the item's `referrer` — the page that started it.
   * Everything this handler does to a transfer (cancel it at the ceiling, cancel
   * it at the per-call count, cancel it at the deadline) is destructive, and doing
   * it to a download the user started by hand in another tab would delete their
   * file: the previous shape cancelled on bytes alone, so a large manual download
   * running during a call was stopped by a ceiling that was never about it.
   *
   * An UNKNOWN page origin (`url` empty, because the tab read failed) means nothing
   * is owned: the conservative direction, and the same one `browser_files` takes on
   * the harness side — a save that declines, never a user's file that disappears.
   * An EMPTY referrer is not ownership either (a redirect chain or a user's
   * "Save page as" report none), so such a transfer is reported and never
   * cancelled; the harness still sees it and refuses it if its own checks fail.
   */
  const pageOrigin = originOf(url);
  const owns = (item: chrome.downloads.DownloadItem): boolean =>
    pageOrigin !== "" && originOf(String(item.referrer ?? "")) === pageOrigin;

  const selector = typeof params.selector === "string" ? params.selector.trim() : "";
  const requested = Number(params.timeout_s ?? 0);
  const timeoutS =
    Number.isFinite(requested) && requested > 0
      ? Math.min(requested, CAPS.downloadTimeoutMaxS)
      : CAPS.downloadTimeoutS;
  const ceiling = CAPS.downloadMaxBytes;
  const maxFiles = CAPS.downloadMaxFilesPerCall;

  // What Chrome already knows about, so "new" is a set difference rather than a
  // guess. A download the page started BEFORE this call completes must not be
  // reported as this call's (it is not ours to relocate, and moving a file the
  // user asked for by hand is the worst failure this handler could have).
  const before = new Set(
    (await deadline(
      chrome.downloads.search({}),
      CHROME_API_DEADLINE_MS,
      "chrome.downloads.search(before)",
    )).map((item) => item.id),
  );

  /** Cancellations this call asked for, keyed by download id, so the report can
   *  say WHY a partial file exists (the harness deletes it either way, but the
   *  audit row has to name the cap or the deadline, not just "cancelled"). */
  const cancelled = new Map<number, string>();
  const cancel = (id: number, why: string): void => {
    if (cancelled.has(id)) return;
    cancelled.set(id, why);
    // Fire-and-forget, and never awaited inside the event listener: a cancel
    // that rejects (a download that finished a millisecond ago) must not throw
    // out of a Chrome event handler and take the worker's listener with it.
    void Promise.resolve(chrome.downloads.cancel(id)).catch(() => undefined);
  };

  // The fast path (rule 2). The ceiling is enforced from a FRESH READ of the
  // item rather than from the delta, because `DownloadDelta` carries no byte
  // field at all (`@types/chrome` 0.1.24 models state/totalBytes/filename
  // changes only, and Chrome's own `onChanged` is not a progress meter): the
  // event is the wake-up, `search` is the number. One item by id is a cheap
  // call, and doing it here is what bounds the overshoot by an event instead of
  // by the 200 ms poll.
  const onChange = (delta: chrome.downloads.DownloadDelta): void => {
    if (before.has(delta.id) || cancelled.has(delta.id)) return;
    void chrome.downloads
      .search({ id: delta.id })
      .then(([item]) => {
        // `owns` first: a ceiling this call set does not apply to a transfer this
        // call did not cause (R2/R7).
        if (item && owns(item) && Number(item.bytesReceived ?? 0) > ceiling) {
          cancel(delta.id, "over_cap");
        }
      })
      // A failed or late read is not a reason to throw out of a Chrome event
      // handler; the poll in the wait loop enforces the same ceiling.
      .catch(() => undefined);
  };
  chrome.downloads.onChanged.addListener(onChange);

  try {
    // The trigger is the page's own download, started by the control the caller
    // named. No selector means the page starts it itself (a meta-refresh, a
    // `window.open` to a file URL, a scripted anchor click) — the tool's own copy
    // documents that shape, and this handler simply waits for the event.
    if (selector) {
      // The page's own click path, reused rather than re-implemented: it resolves
      // refs against THIS surface's snapshot, scrolls the node into view and
      // reports a navigation the click started — three behaviours a second copy
      // here would get subtly wrong.
      await click({ tab: params.tab, selector }, requestId);
    }

    const observed = new Map<number, chrome.downloads.DownloadItem>();
    // Ids counted against the per-call ceiling. Over-count transfers are still
    // REPORTED (they were cancelled mid-flight, so a partial file exists and the
    // harness has to remove it) — they are only excluded from the count.
    const accepted = new Set<number>();
    const deadlineAt = Date.now() + timeoutS * 1000;
    let lastChangeAt = Date.now();
    for (;;) {
      // `search({})` is the whole list; the filter is the set difference against
      // `before`, so a download that finished between two polls is still caught
      // (a state event can be missed; the item cannot).
      const items = await deadline(
        chrome.downloads.search({}),
        CHROME_API_DEADLINE_MS,
        "chrome.downloads.search(poll)",
      );
      for (const item of items) {
        if (before.has(item.id)) continue;
        const ours = owns(item);
        if (!observed.has(item.id)) {
          // Only OUR transfers consume the per-call slots: a stranger's download
          // taking one would push our own file over the count and cancel it
          // (round-1 R7), and a stranger's transfer is never cancelled by this
          // call either — it is reported so the harness can decide (and refuse it
          // without touching it).
          if (ours && accepted.size >= maxFiles) {
            // The per-call ceiling, applied to the TRANSFER rather than to the
            // report: letting the extra files finish and then ignoring them
            // would leave the user's disk holding the evidence of a download the
            // session never kept.
            cancel(item.id, "over_count");
          } else if (ours) {
            accepted.add(item.id);
          }
          observed.set(item.id, item);
          lastChangeAt = Date.now();
          continue;
        }
        const previous = observed.get(item.id)!;
        if (
          previous.state !== item.state ||
          previous.bytesReceived !== item.bytesReceived ||
          previous.filename !== item.filename
        ) {
          observed.set(item.id, item);
          lastChangeAt = Date.now();
        }
        if (
          ours &&
          item.state === "in_progress" &&
          Number(item.bytesReceived ?? 0) > ceiling
        ) {
          cancel(item.id, "over_cap");
        }
      }
      const allTerminal = [...observed.values()].every((item) => item.state !== "in_progress");
      const settled = observed.size > 0 && allTerminal && Date.now() - lastChangeAt >= SETTLE_MS;
      if (settled) break;
      if (Date.now() >= deadlineAt) {
        // Never finished: whatever is still moving AND ours is cancelled, and the
        // partial file is reported so the harness can remove it. A transfer left
        // running past the command's own deadline would be a write to the user's
        // disk that nothing is watching and no report mentions — but a stranger's
        // transfer is not ours to stop (R2), so it is only reported.
        for (const item of observed.values()) {
          if (item.state === "in_progress" && owns(item)) cancel(item.id, "unfinished");
        }
        break;
      }
      await new Promise((resolve) => setTimeout(resolve, POLL_MS));
    }

    // A last look, so a cancellation that landed while the loop was exiting is
    // reported with the state Chrome settled on rather than the state observed
    // before the cancel was asked for. BEST-EFFORT, and that is a measured
    // correction: the first headless run had this call exceed its 5 s deadline on
    // a loaded host and take down a command whose download had ALREADY landed — a
    // refinement read failing must not turn a successful transfer into a reported
    // failure, and the poll loop below has already seen everything the report
    // needs. The fallback is the observed set, which is why the report can still
    // name the path.
    const final = new Map<number, chrome.downloads.DownloadItem>();
    try {
      for (const item of await deadline(
        chrome.downloads.search({}),
        CHROME_API_DEADLINE_MS,
        "chrome.downloads.search(final)",
      )) {
        if (observed.has(item.id)) final.set(item.id, item);
      }
    } catch {
      // Nothing to add: `reduce` below falls back to the item the poll saw.
    }

    const files = [...observed.keys()].map((id) => {
      const item = final.get(id) ?? observed.get(id)!;
      return reduce(item, cancelled.get(id) ?? "");
    });
    if (files.some((item) => item.cancelled)) {
      // Said out loud rather than left to the harness to infer: the ceiling and
      // the deadline are this handler's decisions, and the harness's report must
      // be able to name which one stopped a transfer.
      return {
        armed: true,
        url,
        files,
        note:
          `a transfer was cancelled (${[...new Set(files.map((f) => f.cancelled).filter(Boolean))].join(", ")}); ` +
          "a partial file may remain at the path reported here and the harness removes it",
      };
    }
    return { armed: true, url, files };
  } finally {
    // The listener is removed on EVERY exit, including a throw from `click`: a
    // listener left behind would cancel a LATER call's downloads against a
    // ceiling this call never enforced, and the worker outlives the command.
    chrome.downloads.onChanged.removeListener(onChange);
  }
}
