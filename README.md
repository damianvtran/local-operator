# Rendered evidence for PR #909

Frames of the real `OperatorApp` (Textual, headless `run_test` export) over the
production session stack: production handle class, production runtime server on
a real loopback socket, production `RemoteSession` client — the class
`cli.viewer_factory` returns on every path.

Captured on branch `fix/inline-credential-through-session` at 45c019c44
(2026-09-10), worktree `~/workspace/repos/lo-cred-pr`, by pid 1184's session
during end-to-end validation of the inline `/credential` fix.

| file | state |
|---|---|
| `before/typed.png` | base behaviour: a credential typed in the composer reaches no store; the notice ends "the secret is stored on the owner" and advises pasting `/credential <KEY>` — advice the code's own comment documents as un-followable (the space opens the masked capture) |
| `after/typed-alive.png` | runtime alive: the typed credential is stored through the session and reported as injected into every bash command, value unreadable by the agent |
| `after/typed-dead.png` | runtime dead: loud refusal naming the failure, with a followable retry (paste the value again after `/credential`) |

This branch is evidence storage only; it is never merged.
