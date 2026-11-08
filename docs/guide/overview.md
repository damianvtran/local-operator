<!-- Moved verbatim from the project README (2026-11-08 README refresh); the README now links here. -->

# Why Local Operator

- **Agent organizations with enforced roles.** Roles are capability
  boundaries (a `reviewer` loses the tools to edit what it reviews),
  specialists carry their own standing instructions, and a team is a saved
  roster (a manager plus members) with two short briefs: how the group works
  together and what product it owns. Reuse the same roster on a different
  product by swapping one brief. Nested teams show in the chart today; a
  manager delegating into a nested team's manager is coming.
- **Agents that talk to each other.** A manager peeks at, questions, steers,
  pauses, and resumes its workers through `hub`; independent sessions in
  different repos message each other with `send`, choosing whether to leave a
  note, wake an idle peer, or redirect one mid-turn. Loopback only, your OS
  account only.
- **Always on.** Wakes persist and fire on schedule even after the terminal is
  closed; long commands and subagents run as background jobs whose results
  auto-deliver when the session is idle; `lop exec --background` detaches a
  whole task; a paused worker resumes after a process restart; the mobile
  daemon keeps every session reachable from your phone.
- **Every subscription you already pay for.** Sign in to ChatGPT, Claude,
  Kimi, Grok, Z.AI, and Qwen with the accounts you already have. Have two
  Claude or ChatGPT accounts? They're pooled: eligible accounts default to
  cached reset-soonest selection, a rate-limit tries a sibling, and
  only when both are exhausted does it fall back to the next model you've
  listed.
- **Cheaper long sessions.** The prompt is laid out so your provider's prompt
  cache keeps hitting turn after turn: stale tool output is cleared without
  invalidating the cache, large contexts automatically get the longer
  Anthropic cache window, skills and MCP tools load only when used, and
  agents wait for events instead of polling. The result: a long session
  doesn't re-pay for its history every few minutes.
- **Approval-gated by default.** Reads run; writes and shell commands show the
  exact command and ask. Opting out is an explicit act: `/approvals auto` or
  `--yolo`. Every tool call leaves a visible receipt in the transcript.
- **Reach beyond the terminal.** Watch and steer sessions from your phone,
  and drive the Chromium browser you already use, with your real logins,
  through the published browser extension. The
  [desktop app](desktop-app.md#desktop-app-local-operator-ui) puts the same sessions in a
  desktop window, with a browser pane, a file canvas, and — coming soon — a
  terminal agents can use.
