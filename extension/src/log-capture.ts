import type { LogEntry } from "./protocol.gen";

/**
 * Per-tab console/runtime log capture for the `logs` command.
 *
 * WHY a ring buffer in the worker rather than querying the page: the whole
 * point of `logs` is to surface real console output AND uncaught exceptions
 * that already happened — a page's own `console` history is not readable after
 * the fact, so we must be listening from the moment the surface opens. We
 * enable the CDP `Log` and `Runtime` domains on attach and buffer three event
 * streams into one ordered list per tab:
 *   - `Runtime.consoleAPICalled`  — every console.log/warn/error/info/debug
 *   - `Runtime.exceptionThrown`   — uncaught exceptions (the debugging payoff)
 *   - `Log.entryAdded`            — browser-level messages (network, security,
 *                                   deprecations) the page never sees
 *
 * CONSTRAINTS:
 *   - The buffer is module-global and does NOT survive service-worker death.
 *     That is acceptable and honest: `logs` is defined as "since this surface
 *     opened", and a worker restart tears the surface's debugger session down
 *     anyway, so there is no window where we claim history we silently lost.
 *   - Capped two ways (entry count AND total bytes) so a page in a tight
 *     console-spam loop cannot grow the worker's memory without bound; the
 *     oldest entries are dropped first (newest-last is what the tool wants).
 *   - No page-provided string is ever evaluated. We read structured CDP event
 *     payloads and stringify argument previews with a fixed formatter, keeping
 *     the no-remote-code posture.
 */

// Keep the newest N entries; a debugging read wants recent output, and an
// unbounded buffer on a noisy page is a memory leak in a long-lived worker.
const MAX_ENTRIES = 200;
// Independent byte ceiling: 200 short lines is nothing, but 200 lines each
// carrying a stringified megabyte object would still blow up. Trim oldest
// until under this regardless of count.
const MAX_BYTES = 256 * 1024;
// One argument preview is itself capped so a single huge logged object cannot
// dominate the buffer; the tool truncates the whole response again to its own
// text ceiling on the Python side.
const MAX_ARG_CHARS = 2000;

/*
 * ORIGIN TAGS. The buffer outlives navigation (it is "since this surface
 * opened"), and a tab can navigate itself to a site the user never approved,
 * log there, and come back. Each entry is therefore stamped, at capture, with
 * the origin of the page that produced it, and `logs` returns only entries
 * whose origin the tab is admitted to NOW (commands/logs.ts) — the same rule
 * the dispatch gate applies to live page reads (surface-origin.ts). The tag
 * lives beside the entry, not in it, so the wire `LogEntry` shape is
 * unchanged.
 *
 * Where the origin comes from: the CDP Runtime stream itself, so it is
 * ordered with the events it labels (a webNavigation commit arrives on a
 * different channel and could land after the new page's first logs).
 * `Runtime.executionContextCreated` names each context's origin; console calls
 * and exceptions carry their `executionContextId`. `Log.entryAdded` carries no
 * context, so it takes the current DOCUMENT origin: the first default context
 * created after the last `executionContextsCleared` (the main frame's: its
 * document commits before any child frame of it exists). An entry whose origin is unknown
 * is tagged "" and never returned — fail closed.
 */
interface Tagged {
  entry: LogEntry;
  origin: string;
}

interface Buffer {
  entries: Tagged[];
  bytes: number;
  contexts: Map<number, string>;
  docOrigin: string;
  awaitingDoc: boolean;
}

const buffers = new Map<number, Buffer>();

function bufferFor(tabId: number): Buffer {
  let buffer = buffers.get(tabId);
  if (!buffer) {
    buffer = { entries: [], bytes: 0, contexts: new Map(), docOrigin: "", awaitingDoc: true };
    buffers.set(tabId, buffer);
  }
  return buffer;
}

/** Normalize CDP's context origin: opaque contexts report "null" or "". */
function cleanOrigin(value: unknown): string {
  return typeof value === "string" && value !== "null" ? value : "";
}

function contextOrigin(buffer: Buffer, contextId: unknown): string {
  if (typeof contextId === "number" && buffer.contexts.has(contextId)) return buffer.contexts.get(contextId)!;
  return buffer.docOrigin;
}

function push(tabId: number, entry: LogEntry, origin: string): void {
  const buffer = bufferFor(tabId);
  buffer.entries.push({ entry, origin });
  buffer.bytes += entry.text.length;
  // Evict oldest-first until BOTH caps are satisfied. Order matters: the tool
  // contract is newest-last, so we always drop from the front.
  while (buffer.entries.length > MAX_ENTRIES || buffer.bytes > MAX_BYTES) {
    const dropped = buffer.entries.shift();
    if (!dropped) break;
    buffer.bytes -= dropped.entry.text.length;
  }
}

/** Normalize a CDP console API `type` to the tool's level vocabulary. */
function levelForConsole(type: unknown): string {
  switch (type) {
    case "error":
    case "assert":
      return "error";
    case "warning":
      return "warning";
    case "info":
      return "info";
    case "debug":
      return "log";
    default:
      return "log";
  }
}

/** Normalize a CDP `Log.entryAdded` level to the tool vocabulary. */
function levelForLogEntry(level: unknown): string {
  switch (level) {
    case "error":
      return "error";
    case "warning":
      return "warning";
    case "info":
      return "info";
    default:
      return "log";
  }
}

/**
 * Render a `Runtime.RemoteObject` argument to a short string WITHOUT evaluating
 * anything. CDP already sends a value or a preview; we read those fields only.
 */
function renderArg(arg: Record<string, unknown> | undefined): string {
  if (!arg) return "";
  if (arg.type === "string") return String(arg.value ?? "");
  if ("value" in arg && arg.value !== undefined) return String(arg.value);
  if (typeof arg.description === "string") return arg.description;
  if (arg.type === "undefined") return "undefined";
  if (arg.subtype === "null") return "null";
  return String(arg.type ?? "");
}

function clip(text: string): string {
  return text.length > MAX_ARG_CHARS ? `${text.slice(0, MAX_ARG_CHARS)}…` : text;
}

// The single debugger event listener. chrome.debugger delivers every domain
// event for every attached tab through ONE global listener, so we route by the
// source tabId into the right per-tab buffer and ignore events for tabs we are
// not capturing.
function onEvent(source: chrome.debugger.Debuggee, method: string, rawParams?: object): void {
  const tabId = source.tabId;
  if (tabId === undefined || !buffers.has(tabId)) return;
  const params = (rawParams ?? {}) as Record<string, unknown>;
  const buffer = bufferFor(tabId);

  if (method === "Runtime.executionContextCreated") {
    const context = (params.context ?? {}) as Record<string, unknown>;
    const aux = (context.auxData ?? {}) as Record<string, unknown>;
    const origin = cleanOrigin(context.origin);
    if (typeof context.id === "number") buffer.contexts.set(context.id, origin);
    if (aux.isDefault === true && buffer.awaitingDoc) {
      buffer.docOrigin = origin;
      buffer.awaitingDoc = false;
    }
  } else if (method === "Runtime.executionContextDestroyed") {
    if (typeof params.executionContextId === "number") buffer.contexts.delete(params.executionContextId);
  } else if (method === "Runtime.executionContextsCleared") {
    // A main-frame navigation: every old context is gone, and the next
    // document's origin is unknown until its context is created.
    buffer.contexts.clear();
    buffer.docOrigin = "";
    buffer.awaitingDoc = true;
  } else if (method === "Runtime.consoleAPICalled") {
    const args = Array.isArray(params.args) ? (params.args as Record<string, unknown>[]) : [];
    const text = clip(args.map(renderArg).join(" ").trim());
    const frame = topFrame(params.stackTrace);
    push(tabId, {
      level: levelForConsole(params.type),
      text,
      source: "console",
      url: frame.url,
      line: frame.line,
      timestamp: typeof params.timestamp === "number" ? params.timestamp : Date.now(),
    }, contextOrigin(buffer, params.executionContextId));
  } else if (method === "Runtime.exceptionThrown") {
    const details = (params.exceptionDetails ?? {}) as Record<string, unknown>;
    // Prefer the thrown value's description (the Error's message + stack); fall
    // back to the bare exception text CDP always provides.
    const exception = (details.exception ?? {}) as Record<string, unknown>;
    const text = clip(
      String(exception.description ?? details.text ?? "uncaught exception").trim(),
    );
    push(tabId, {
      level: "error",
      text,
      source: "exception",
      url: typeof details.url === "string" ? details.url : "",
      line: typeof details.lineNumber === "number" ? details.lineNumber : 0,
      timestamp: typeof params.timestamp === "number" ? params.timestamp : Date.now(),
    }, contextOrigin(buffer, details.executionContextId));
  } else if (method === "Log.entryAdded") {
    const entry = (params.entry ?? {}) as Record<string, unknown>;
    push(tabId, {
      level: levelForLogEntry(entry.level),
      text: clip(String(entry.text ?? "").trim()),
      source: "log-entry",
      url: typeof entry.url === "string" ? entry.url : "",
      line: typeof entry.lineNumber === "number" ? entry.lineNumber : 0,
      timestamp: typeof entry.timestamp === "number" ? entry.timestamp : Date.now(),
    }, buffer.docOrigin);
  }
}

interface StackFrame {
  url?: unknown;
  lineNumber?: unknown;
}

function topFrame(stackTrace: unknown): { url: string; line: number } {
  const frames =
    stackTrace && typeof stackTrace === "object"
      ? ((stackTrace as Record<string, unknown>).callFrames as StackFrame[] | undefined)
      : undefined;
  const frame = frames?.[0];
  return {
    url: typeof frame?.url === "string" ? frame.url : "",
    line: typeof frame?.lineNumber === "number" ? frame.lineNumber : 0,
  };
}

let listenerBound = false;

/**
 * Begin capturing logs for a tab. Called from `open` right after the debugger
 * attaches, so buffering starts before the destination page runs any script.
 * Idempotent: enabling a domain twice is harmless, and re-registering the
 * global listener is guarded so a worker that survived keeps one listener.
 */
export async function startLogCapture(
  tabId: number,
  cdp: (tabId: number, method: string, params?: Record<string, unknown>) => Promise<unknown>,
): Promise<void> {
  if (!listenerBound) {
    chrome.debugger.onEvent.addListener(onEvent);
    listenerBound = true;
  }
  // Create the buffer BEFORE enabling the domains, so an event that fires
  // between the two enable calls is not dropped by the tabId guard in onEvent.
  bufferFor(tabId);
  await cdp(tabId, "Runtime.enable");
  await cdp(tabId, "Log.enable");
}

/**
 * Filter+limit a list of entries. Pure (no chrome, no buffer state) so the
 * level/limit contract is unit-testable without a debugger session. `level`
 * "all" keeps everything; `limit` > 0 keeps the most recent n, preserving
 * oldest→newest order.
 */
export function filterEntries(entries: LogEntry[], level: string, limit: number): LogEntry[] {
  const filtered = level === "all" ? entries : entries.filter((entry) => entry.level === level);
  return limit > 0 && filtered.length > limit ? filtered.slice(filtered.length - limit) : filtered;
}

/** The distinct origins a tab's buffered entries came from, so the caller can
 * decide admission once per origin (an async storage read) rather than per
 * entry. */
export function loggedOrigins(tabId: number): string[] {
  return [...new Set((buffers.get(tabId)?.entries ?? []).map((tagged) => tagged.origin))];
}

/** Read filtered entries for a tab, newest-last, keeping only entries whose
 * capture origin is in `admitted` (see ORIGIN TAGS above). `level` "all"
 * keeps every level; `limit` applies after the origin filter. */
export function readLogs(tabId: number, level: string, limit: number, admitted: ReadonlySet<string>): LogEntry[] {
  const entries = (buffers.get(tabId)?.entries ?? [])
    .filter((tagged) => admitted.has(tagged.origin))
    .map((tagged) => tagged.entry);
  return filterEntries(entries, level, limit);
}

/** Drop a tab's buffer when its surface closes so a stale tabId cannot leak. */
export function dropLogCapture(tabId: number): void {
  buffers.delete(tabId);
}
