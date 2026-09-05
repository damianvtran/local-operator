# Agent import and one-shot edit filesystem boundaries

These are bounded C2 defensive fixes to the existing trusted-local HTTP API,
not a new authentication model or a sandbox for local harness tools.

## Import is a copy, never a restore over an identity

`POST /v1/agents/import` and `AgentRegistry.import_agent` treat ZIP metadata as
an instruction profile. Every import gets a fresh UUID, including a repeated
import of an export from this same registry. Archive IDs are ignored, not
normalized into local paths. The validated, normalized metadata is written with
the new ID; the profile name and permitted instruction files are preserved.
The workspace is reset to the local agent home. Private history, schedules,
learnings and pickled context remain excluded by the existing profile rules.

Metadata must validate before registry mutation. Destination `mkdir` is
exclusive: an existing directory, regular file, symlink (including a dangling
one), or in-memory identity collision fails without replacement. Cleanup after
a failed copy is limited to the directory that import exclusively reserved.
ZIP member traversal and symlink rejection remain in the extraction layer.

Callers must use the ID returned in the 201 response. Import does not offer an
implicit overwrite option. Exported profiles remain importable, including those
from older versions and those with names that are not valid path components.

## Two edit input modes

`POST /v1/chat/agents/{agent_id}/edit` accepts optional `file_content`:

- **Supplied string, including `""`:** edits are generated from the caller's
  live buffer. `file_path` is display identity only. It is not expanded,
  resolved, statted or opened. This preserves desktop editing of unsaved files
  and files outside the server's working directory.
- **Omitted or `null`:** the server reads a UTF-8 regular file within the
  agent's configured workspace. Relative paths are interpreted against that
  workspace, not the server process working directory. Absolute paths must be
  within it too. Canonical, component-aware containment rejects traversal,
  sibling-prefix paths and symlink/junction escapes; links that resolve inside
  the workspace remain supported. A missing, blank, relative, nonexistent or
  non-directory workspace fails closed. Invalid/non-text targets return 400,
  unavailable workspaces/outside targets return 403, and missing contained
  targets return 404. Denials occur before model creation.

Both modes preserve the response envelope and return diffs without writing the
file. Desktop CodeEditor and WysiwygMarkdownEditor must send their live buffer,
not a disk reread or a stale document prop. Deploy the companion desktop update
first: the additive field is ignored by older servers, while old desktop builds
against a patched server can edit only workspace-contained files. An unknown
agent still returns 404. Edit attachments retain their existing behavior; the
one-shot executor currently consumes conversation text only, not record files.

## Trust model and residual limits

`lop serve` now defaults to `127.0.0.1`. An explicit `--host` still works, but
this legacy API has **no authentication or tenant separation**. Loopback is a
safer exposure default, not authorization: do not expose it to untrusted local
or remote clients. Put access controls in front before intentionally widening
the bind. Other API operations can configure an agent workspace; a workspace
explicitly set to the filesystem root grants that scope. The edit check is not
a security boundary against a caller already authorized to reconfigure agents
or use the unrestricted local harness.

Canonical resolution followed immediately by opening the canonical target is
portable, but not race-free against a concurrent local process replacing path
components. Such a process is outside this remote-input confinement threat
model. No POSIX-only descriptor sandbox or Windows feature disable is implied.
Native Windows drive and junction regressions run in a dedicated CI job; macOS
and Linux run the regular unit suites. Trusted TUI/mobile filesystem tools,
mobile authentication and the mobile daemon's mandatory loopback bind are
unchanged.

## Verification scope

Regressions cover fresh/repeated IDs, directory/file/symlink collisions,
malformed metadata and failed-copy cleanup, ZIP extraction validation,
workspace-relative and absolute success, traversal and link escapes, missing
workspace/target, invalid/non-text targets, unknown agents, and explicit empty
and nonempty buffers. The desktop integration story exercises the production
editors, HTTP client and diff review against a disposable real API, substituting
only native saves and the model provider boundary. It is renderer/API evidence,
not evidence of Electron packaging or a live third-party model.
