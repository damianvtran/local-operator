import { BridgeCommandError, cdp, requireSurface } from "../cdp";
import { requireConsent } from "../consent";
import { credentialRefusal, safeName } from "../driver/file-transfer-policy";
import { nodeIdFor } from "./input";

/* Attach local files to a page's file input.
 *
 * The primitive is `DOM.setFileInputFiles` over the tab-scoped debugger session
 * this extension already holds — the same one Puppeteer and Playwright use, and
 * the reason the existing `debugger` permission is sufficient for uploads (no
 * new permission, no store re-review). The browser process reads the bytes off
 * disk; this worker never sees them, which is also why it can check a NAME but
 * never a CONTENT (see driver/file-transfer-policy.ts).
 *
 * THE OPERATOR'S SWITCH GATES IT, and it is the only control this direction has:
 * no new permission is involved, so `allowUploads` in the extension's options is
 * what decides whether a local file may leave this machine at all. It is OFF by
 * default, and `requireConsent` is checked here and not only in the
 * advertisement — see consent.ts.
 *
 * TWO RULES ARE STRUCTURAL, not stylistic:
 *
 *   1. READ BACK, always. `type`'s rule (`docs/BROWSER.md`): the field's value is
 *      read back and COMPARED, because a call that silently did nothing must not
 *      be reported as a filled input. A mismatch is an error naming both sides —
 *      a form that ignores the attach is exactly the failure a page would like us
 *      to report as success. A read that could not be TAKEN at all is the other
 *      case, and it is reported as an UNVERIFIED attach rather than as a failure:
 *      the attach has already resolved by then, so "nothing was sent" would be a
 *      claim about bytes that have gone (and a double-send invitation). See the
 *      catch in `upload`.
 *   2. `accept=` IS REPORTED AND NEVER OBEYED. A site's own filter protects
 *      nothing (it exists to help a human pick a file) and honouring it would let
 *      the page steer which local files we try to attach. It rides the result as
 *      a fact for the audit row.
 *
 * A refused path is a RESULT, not a wire error. `ErrorDetail.code` is validated
 * against the daemon's enum, so an extension that emitted a new code would have
 * its frame dropped by an already-released daemon (protocol.py's C3 rule); the
 * refusal therefore travels as `ok: true` with a `refused` list, and PYTHON
 * decides how the model is told (design 6.2). The checks here are defence in
 * depth: `local_operator/browser_files.py::check_upload` is the control, and it
 * runs unconditionally on the harness side.
 */

interface ResolvedNode { object: { objectId: string } }
interface CallResult { result: { value?: unknown } }
interface FileReadback { count: number; names: string[]; sizes: number[]; accept: string }

/* CDP failures that mean the page NAVIGATED out from under the read-back.
 *
 * Used for WORDING only, never as the gate: every read-back failure is reported as
 * an unverified attach (see the catch in `upload`), and this list decides whether
 * the note blames the page's navigation or names the error instead. Matched on the
 * MESSAGE because `-32000` is CDP's generic "something went wrong"; the strings are
 * the concrete ones Chrome 153 returns for a destroyed/replaced execution context
 * or a detached node.
 */
const CONTEXT_GONE = [
  "Cannot find context with specified id",
  "Cannot find execution context",
  "Execution context was destroyed",
  "Inspected target navigated or closed",
  "Node with given id does not belong to the document",
  "No node with given id found",
  "Could not find node with given id",
];

function isContextGone(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error);
  return CONTEXT_GONE.some((marker) => message.includes(marker));
}

/* One readable line for an error that failed a READ, for the note and the row.
 *
 * Trimmed to a single line and capped: the text comes from Chrome, lands in the
 * operator's transcript and in the audit file, and a multi-line CDP payload there
 * would push the rest of the row out of sight.
 */
function describeError(error: unknown): string {
  const message = (error instanceof Error ? error.message : String(error)).replace(/\s+/g, " ");
  return message.slice(0, 120).trim() || "no detail";
}

/* The DOM's answer for what the input now holds, or null when it holds no files.
 *
 * READ THROUGH THE PROTOTYPE'S OWN GETTER, never `this.files`. The page is the
 * adversary this read-back exists for, and `this.files` is an own property a
 * hostile page can shadow with `Object.defineProperty` to answer with anything
 * — a real gap, since the page can run between the attach and the read.
 */
async function readInputFiles(tabId: number, nodeId: number): Promise<FileReadback | null> {
  const resolved = await cdp<ResolvedNode>(tabId, "DOM.resolveNode", { nodeId });
  const read = await cdp<CallResult>(tabId, "Runtime.callFunctionOn", {
    objectId: resolved.object.objectId,
    functionDeclaration: `function(){
      const input = this;
      if(!input || !('files' in input)) return null;
      const descriptor = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'files');
      // A non-input element has no business here; the getter throws for one and
      // that is a null read, not a fabricated count.
      let files;
      try { files = descriptor && descriptor.get ? descriptor.get.call(input) : input.files; }
      catch (err) { return null; }
      if(!files) return null;
      const list = [...files];
      return JSON.stringify({
        count: list.length,
        names: list.map((f) => f.name),
        sizes: list.map((f) => f.size),
        accept: String(input.getAttribute('accept') || ''),
      });
    }`,
    returnByValue: true,
  });
  return typeof read.result?.value === "string"
    ? (JSON.parse(read.result.value) as FileReadback)
    : null;
}

export async function upload(params: Record<string, unknown>): Promise<Record<string, unknown>> {
  // The operator's switch, before the page is touched at all (consent.ts). The
  // refusal names the switch rather than the build, because the capability is
  // present and only its consent is missing — and this is the LAST gate, after
  // the advertisement, so a page's file input is never reached on a switched-off
  // capability even if an old daemon sent the command anyway.
  await requireConsent("upload");
  const surface = await requireSurface(params.tab);
  const selector = params.selector ?? params.ref;
  const raw = Array.isArray(params.paths) ? params.paths : [];
  const paths: string[] = [];
  const refused: { path: string; reason: string }[] = [];
  for (const entry of raw) {
    if (typeof entry !== "string" || entry === "") {
      refused.push({ path: "", reason: "refused: every upload path must be a non-empty string" });
      continue;
    }
    // The name shown back is the SANITISED one — this string reaches a log line
    // and a tool result, so a control character or an RTL override in it would be
    // an injection into the operator's next approval card.
    const shown = safeName(entry);
    const reason = credentialRefusal(entry);
    if (reason) {
      refused.push({ path: shown, reason });
      continue;
    }
    if (!entry.startsWith("/")) {
      // The tool composes absolute paths; a relative one would be resolved
      // against Chrome's own cwd, which is not the session's.
      refused.push({ path: shown, reason: `refused: '${shown}' is not an absolute path` });
      continue;
    }
    paths.push(entry);
  }
  if (paths.length === 0) {
    // Nothing to attach: report the refusals rather than driving the input with
    // an empty list, which a page would see as "the agent cleared my field".
    return { inputs: [], accepted: [], refused, accept: "", readback: "" };
  }

  const nodeId = await nodeIdFor(surface, selector);
  await cdp(surface.tabId, "DOM.setFileInputFiles", { nodeId, files: paths });

  /* The read-back is mandatory, but a read that could not be TAKEN is not a
   * failed attach.
   *
   * A page that submits itself from the change handler — the standard "pick a
   * file and it uploads" form — navigates in the same tick as the attach, which
   * destroys the execution context the read-back runs in. The attach itself has
   * already happened (the `DOM.setFileInputFiles` above RESOLVED) and the bytes
   * have already gone. Reporting the raw CDP error for that is wrong twice: the
   * model cannot learn the files were sent (so it retries and double-sends) and
   * the audit row for a real egress is never written.
   *
   * So the rule is stated once and stated generally: **the calls in this `try`
   * are READS, and a read that failed is evidence of nothing.** What the
   * read-back exists to catch — a page that ignored the attach — is a MISMATCH,
   * found by a read that SUCCEEDED, and every mismatch still throws below. A
   * narrower catch keyed on "context gone" strings would leave the same false
   * failure reachable through our own per-call deadline (a stall on the read,
   * which QA measured once as a flake) or any other Chrome error, and each of
   * those would report "nothing was sent" over bytes that had gone. The wording
   * still distinguishes the navigation, and Python's own re-stat + digest is what
   * the facts are built from either way.
   */
  let payload: FileReadback | null = null;
  let readback = "";
  try {
    payload = await readInputFiles(surface.tabId, nodeId);
  } catch (error) {
    readback = isContextGone(error)
      ? "unavailable — the page navigated out of the change event before the input could be read back"
      : `unavailable — the read-back failed (${describeError(error)})`;
  }
  const expected = paths.map((path) => safeName(path));
  if (readback) {
    // Checked BEFORE the count comparison below, because there is no payload to
    // compare against: the read is what failed, not the attach. Nothing is
    // claimed ABOUT the DOM — the paths are reported as attached because the
    // attach is what was just sent, `bytes` is -1 for "unknown here" (Python
    // stats every path from disk, and that is the verification which survives
    // the page), and the marker travels so Python can report the attach as
    // unverified instead of turning a completed egress into an error.
    return {
      inputs: [String(selector ?? "")],
      accepted: paths.map((path, index) => ({
        name: expected[index],
        path,
        bytes: -1,
        mime: "",
        sniffed: "",
        sha256: "",
      })),
      refused,
      accept: "",
      readback,
    };
  }
  if (!payload || payload.count !== paths.length) {
    throw new BridgeCommandError(
      "internal",
      `the file input did not take the attach: it holds ${payload ? payload.count : 0} file(s) `
        + `after ${paths.length} were set`,
    );
  }
  if (payload.names.join("\u0000") !== expected.join("\u0000")) {
    throw new BridgeCommandError(
      "internal",
      `the file input holds different files than were attached: it reports `
        + `${payload.names.join(", ")} for ${expected.join(", ")}`,
    );
  }
  return {
    inputs: [String(selector ?? "")],
    // `bytes` is what the DOM reports; Python re-stats every path from disk and
    // computes the digest itself, because a host's word about a file it did not
    // measure is exactly what the post-hoc verification exists to remove.
    accepted: paths.map((path, index) => ({
      name: expected[index],
      path,
      bytes: payload!.sizes[index] ?? 0,
      mime: "",
      sniffed: "",
      sha256: "",
    })),
    refused,
    accept: payload!.accept,
    readback: "",
  };
}
