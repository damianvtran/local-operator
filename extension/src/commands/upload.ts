import { BridgeCommandError, cdp, requireSurface } from "../cdp";
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
 * TWO RULES ARE STRUCTURAL, not stylistic:
 *
 *   1. READ BACK, always. `type`'s rule (`docs/BROWSER.md`): the field's value is
 *      read back and COMPARED, because a call that silently did nothing must not
 *      be reported as a filled input. A mismatch is an error naming both sides —
 *      a form that ignores the attach is exactly the failure a page would like us
 *      to report as success.
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

export async function upload(params: Record<string, unknown>): Promise<Record<string, unknown>> {
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
    return { inputs: [], accepted: [], refused, accept: "" };
  }

  const nodeId = await nodeIdFor(surface, selector);
  await cdp(surface.tabId, "DOM.setFileInputFiles", { nodeId, files: paths });

  const resolved = await cdp<ResolvedNode>(surface.tabId, "DOM.resolveNode", { nodeId });
  const read = await cdp<CallResult>(surface.tabId, "Runtime.callFunctionOn", {
    objectId: resolved.object.objectId,
    functionDeclaration: `function(){
      if(!this || !('files' in this)) return null;
      return JSON.stringify({
        count: this.files.length,
        names: [...this.files].map((f) => f.name),
        sizes: [...this.files].map((f) => f.size),
        accept: String(this.getAttribute('accept') || ''),
      });
    }`,
    returnByValue: true,
  });
  const payload = typeof read.result?.value === "string" ? (JSON.parse(read.result.value) as FileReadback) : null;
  if (!payload || payload.count !== paths.length) {
    throw new BridgeCommandError(
      "internal",
      `the file input did not take the attach: it holds ${payload ? payload.count : 0} file(s) `
        + `after ${paths.length} were set`,
    );
  }
  const expected = paths.map((path) => safeName(path));
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
      bytes: payload.sizes[index] ?? 0,
      mime: "",
      sniffed: "",
      sha256: "",
    })),
    refused,
    accept: payload.accept,
  };
}
