/* Host-free file policy: the name check and the sanitiser, shared with Python.
 *
 * WHY THIS MODULE EXISTS AT ALL. The authoritative policy is Python's
 * (`local_operator/browser_files.py`), because Python is the only side that can
 * see a downloaded file's BYTES and the only side that can delete it. What a
 * browser host can do cheaply, before anything reaches a path or a prompt, is
 * check a NAME — and this module is that check, written once and vendored to the
 * second host rather than written twice (design §5.2, §10.4).
 *
 * WHAT IT MUST NOT GROW. No sniffing. The extension cannot read the bytes it
 * handed to Chrome (a Manifest V3 service worker has no filesystem access, and
 * `chrome.downloads` exposes metadata rather than contents), so a signature table
 * here would be dead code pretending to be a control — design §10.2 says so in as
 * many words. The tables this module reads are generated from Python
 * (`./file-transfer.tables.gen`), which is what keeps the two hosts' lists from
 * drifting, and the extension's own test replays the shared conformance fixture
 * against these functions.
 *
 * NO `chrome.*` AND NO `@types/chrome` TYPE MAY APPEAR HERE: `tests/
 * driver-host-free.test.mjs` enforces it structurally, because this directory is
 * what the desktop app vendors whole.
 */

import {
  CREDENTIAL_COMPONENTS,
  CREDENTIAL_NAME_PATTERNS,
  DENY_EXTS,
} from "./file-transfer.tables.gen";

/** Cap in BYTES, mirroring `MAX_NAME_BYTES` — the filesystem's own limit is
 *  bytes, and a uniquifying suffix has to fit beside the name. */
const MAX_NAME_BYTES = 200;

/** The stem of a generated name, mirroring `_FALLBACK_STEM`. */
const FALLBACK_STEM = "download";

/* C0 and C1 controls: a filename is displayed in an approval card and in the
 * session scrollback, so this is a terminal-escape injection, not cosmetics. */
const CONTROL = /[\u0000-\u001f\u007f-\u009f]/g;

/* RTL and zero-width overrides: `evil.exe` written with an override DISPLAYS as
 * `evilexe.pdf`, which is the whole reason they are enumerated. */
const BIDI_ZERO_WIDTH = /[\u200b-\u200f\u202a-\u202e\u2066-\u2069]/g;

/* Reserved DOS device names. Windows resolves `CON.txt` to the device too, so
 * the STEM is what is tested. */
const WINDOWS_RESERVED_STEMS = new Set([
  "con", "prn", "aux", "nul",
  "com1", "com2", "com3", "com4", "com5", "com6", "com7", "com8", "com9",
  "lpt1", "lpt2", "lpt3", "lpt4", "lpt5", "lpt6", "lpt7", "lpt8", "lpt9",
]);

function bytesOf(text: string): number {
  return new TextEncoder().encode(text).length;
}

/* FNV-1a (32-bit), NOT sha256 — and the reason is not performance.
 *
 * The digest is only ever used to build a generated NAME for a file whose
 * proposed name sanitised to nothing usable. It is not an integrity claim, so
 * the security primitive buys nothing here, and it would cost real weight in
 * this direction: `crypto.subtle.digest` is asynchronous (which would make
 * `safeName` async for every caller) and hand-rolling SHA-256 in a vendored
 * policy module is a second implementation of a security primitive that nobody
 * would review as one. Python computes the same value over the same UTF-8 bytes
 * (`browser_files._fallback_name`), which is what the shared fixture asserts. */
function fallbackDigest(text: string): string {
  let hash = 0x811c9dc5;
  for (const byte of new TextEncoder().encode(text)) {
    /* Math.imul, not `*`: a 32-bit multiply must not round-trip through a
     * double, which silently loses the low bits past 2^53. */
    hash = Math.imul(hash ^ byte, 0x01000193) >>> 0;
  }
  return hash.toString(16).padStart(8, "0");
}

function truncateBytes(name: string, limit: number): string {
  /* The extension is preserved and the STEM is cut, because a long name that
   * loses its extension is a file nobody can open. Truncating a JS string by
   * slice() can split a surrogate pair, so the loop measures bytes as it goes. */
  let head = name;
  let ext = "";
  const dot = name.lastIndexOf(".");
  if (dot > 0 && name.length - dot - 1 <= 12) {
    head = name.slice(0, dot);
    ext = name.slice(dot + 1);
  }
  const budget = limit - (ext ? bytesOf(ext) + 1 : 0);
  if (budget <= 0) return "";
  while (bytesOf(head) > budget && head.length > 0) {
    head = head.slice(0, -1);
    /* A cut that lands mid-surrogate-pair would leave a lone surrogate, which is
     * not encodable text; drop it. */
    if (/[\ud800-\udbff]$/.test(head)) head = head.slice(0, -1);
  }
  return ext ? `${head}.${ext}` : head;
}

/** The generated basename for a name that cannot be used as one. */
export function fallbackName(raw: string, sniffedExt = ""): string {
  const stem = `${FALLBACK_STEM}-${fallbackDigest(raw)}`;
  return sniffedExt ? `${stem}.${sniffedExt}` : stem;
}

/**
 * The sanitised basename a page-supplied (or caller-supplied) name may use.
 *
 * Byte-for-byte the same rules as `browser_files.safe_name`, and the shared
 * fixture (`CONFORMANCE_CASES`) is what keeps them that way: both clauses of a
 * hostile name — the truncation to a basename in EITHER separator flavour, and
 * the stripping of controls, bidi overrides, trailing dots and spaces — are what
 * stop a page from choosing a path, a display, or a directory.
 *
 * `sniffedExt` corrects the extension to what the CONTENT turned out to be. That
 * is the one case where a name is changed rather than merely cleaned, and the
 * extension applies it to the names it reports back so its result cannot show a
 * name the harness would never have written.
 */
export function safeName(raw: string, sniffedExt = ""): string {
  const parts = raw.replace(/\\/g, "/").split("/");
  /* `?? ""` rather than a non-null assertion: this module is vendored into a
   * host compiled with `noUncheckedIndexedAccess`, and an empty basename simply
   * falls through to the generated-name branch below. */
  let name = (parts[parts.length - 1] ?? "").replace(CONTROL, "").replace(BIDI_ZERO_WIDTH, "");
  name = name.normalize("NFC").trim().replace(/[. ]+$/, "");
  if (name === "" || name === "." || name === "..") return fallbackName(raw, sniffedExt);
  const dot = name.lastIndexOf(".");
  const stem = dot > 0 ? name.slice(0, dot) : name;
  let ext = dot > 0 ? name.slice(dot + 1) : "";
  if (WINDOWS_RESERVED_STEMS.has(stem.toLowerCase())) return fallbackName(raw, sniffedExt);
  if (sniffedExt && ext.toLowerCase() !== sniffedExt.toLowerCase()) {
    ext = sniffedExt;
  }
  return truncateBytes(ext ? `${stem}.${ext}` : stem, MAX_NAME_BYTES);
}

/** The lowercased final extension of a SANITISED name, or "". */
export function extensionOf(raw: string): string {
  const name = safeName(raw).toLowerCase();
  const dot = name.lastIndexOf(".");
  if (dot <= 0) return "";
  const ext = name.slice(dot + 1);
  return ext.length > 12 || ext.includes("/") ? "" : ext;
}

/**
 * Whether a name promises an executable/script — the name-only half of the
 * download refusal (design §10.2's fourth row).
 *
 * It is a NAME check and never a verdict: a `.txt` whose bytes are a PE is
 * Python's to catch (content wins), and a `.exe` over PDF bytes is allowed as a
 * PDF. The desktop app's download path uses this to refuse an obviously-unwanted
 * file BEFORE it lands, where the platform lets it; the extension cannot serve
 * downloads at all (see `EXTENSION_CANNOT_SERVE` in
 * `local_operator/browser_bridge/protocol.py`), so this function's caller lives
 * in the other host.
 */
export function executableName(raw: string): boolean {
  const ext = extensionOf(raw);
  return ext !== "" && DENY_EXTS.includes(ext);
}

function globToRegExp(pattern: string): RegExp {
  /* `fnmatch` semantics: `*` spans anything, `?` one character, everything else
   * is literal (the patterns in the generated table are lowercase, which is why
   * both sides lowercase the candidate before matching). */
  const escaped = pattern.replace(/[.+^${}()|[\]\\]/g, "\\$&").replace(/\*/g, ".*").replace(/\?/g, ".");
  return new RegExp(`^${escaped}$`);
}

const CREDENTIAL_GLOBS = CREDENTIAL_NAME_PATTERNS.map((pattern) => ({
  pattern,
  regex: globToRegExp(pattern),
}));

/**
 * Why this path may not be uploaded, or "" when it may — the same rule as
 * `browser_files._credential_refusal`, matched on the basename AND on every path
 * component.
 *
 * Both halves are needed because either alone is escapable: an innocent basename
 * inside `secrets/` is still a secret, and a file copied out of `~/.ssh/` keeps
 * a name that says so. This is the host-side half of the upload policy and is NOT
 * the control — Python's `check_upload` is, and it runs unconditionally.
 */
export function credentialRefusal(pathish: string): string {
  const parts = pathish.split("/").filter((part) => part !== "");
  const base = (parts[parts.length - 1] ?? "").toLowerCase();
  for (const { pattern, regex } of CREDENTIAL_GLOBS) {
    if (regex.test(base)) {
      return `refused: '${base}' matches the credential deny-list (${pattern})`;
    }
  }
  for (const part of parts) {
    if (CREDENTIAL_COMPONENTS.includes(part.toLowerCase())) {
      return `refused: '${base}' is inside a '${part}' directory, which holds credentials`;
    }
  }
  return "";
}
