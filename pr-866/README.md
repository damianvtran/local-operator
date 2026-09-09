# PR #866 evidence — MCP auth propagation

Rendered frames for the user-visible half of
`fix(mcp): propagate a peer session's re-auth to servers this session gave up on`.

Captured from the real `OperatorApp` via `scripts.visual_capture.save_capture`
(isolated HOME/config, `env -u NO_COLOR TERM=xterm-256color`), same seeded
state in both: two configured servers, `notion`'s OAuth grant expiring
mid-session. Screen 98x24, virtual 98x24, no scrollbar, identical in both —
only the status column changes.

| file | tree | `/mcp` row for notion |
| --- | --- | --- |
| `before-mcp-block.png` | `origin/main` a8f98be3b | `notion  disconnected` |
| `after-mcp-block.png`  | `fix/mcp-auth-propagation` | `notion  auth-required` |

Kept on this orphan branch rather than in the tree, per AGENTS.md
"Evidence goes on the PR, never into the repository".
