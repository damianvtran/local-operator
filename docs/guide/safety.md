<!-- Moved verbatim from the project README (2026-11-08 README refresh); the README now links here. -->

# Safety Model

- **Approval tiers.** Read-only tools run automatically; anything that writes
  files or executes commands prompts first, showing the exact command.
  `/approvals auto` or `--yolo` disables prompts only when you say so.
  Scheduling a `wake` is a write-tier action too: it is the one tool that
  arms unattended future execution, so it prompts like a mutation. If a wake
  fires unattended and a tool needs approval, the turn stalls at the prompt
  until you answer from the phone or `/resume`; `/approvals auto` or
  `--yolo` is how you opt into unattended execution.
- **Visible receipts.** Every tool call leaves a card or one-line receipt in
  the transcript. There is no invisible action, and a peer message from
  another session is a distinct inbound card, never disguised as your turn.
- **Roles that cannot escalate.** A tool-restricted role keeps read-only reach
  but cannot enable new MCP tools, and neither can anything it delegates to.
- **Loopback only.** Peer messaging, the mobile daemon, and the browser bridge
  all bind to loopback and authenticate; the trust boundary is your OS account.
- **Local-first options.** Run a local model and disable web search
  (`lop search off`) and fetch (`lop fetch set enabled off`) for a setup
  where no prompt leaves the machine. Web search is on by default.
- **MCP trust model.** Project-supplied MCP configs are treated as trusted
  input and warned about on first connect. See [docs/mcp.md](../../docs/mcp.md)
  for the trust model.
- **Credential hygiene.** Provider keys live in a local file, are entered
  through masked prompts, and are kept out of transcripts. A secret handed over
  with `/credential` becomes a chip rather than text; the composer's own paste
  key (`Ctrl+V` on macOS) is not a hand-over, and it inserts the value as
  ordinary text. Provider logins stay in `auth.db`, and everything you save with
  `lop secret` lives in the encrypted long-term store. See
  [Credentials and secrets](configuration.md#credentials-and-secrets).
- **A shell's environment is a policy, not an accident.** A command the model
  runs sees your environment by default (`shell_environment.mode: inherit`); a
  server-owned run sets `allowlist`, so the provider key that session launched
  with is not readable from that shell. See
  [What the agent's shell may see](configuration.md#what-the-agents-shell-may-see).
- **Endpoint protection.** A full install registers per-user services, which
  behaviour-based EDR can read as persistence. See
  [docs/ENDPOINT_PROTECTION.md](../../docs/ENDPOINT_PROTECTION.md) for the inventory
  of what is installed and persisted, how the artefacts are signed, and how to
  allow-list by path.
