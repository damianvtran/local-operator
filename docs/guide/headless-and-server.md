<!-- Moved verbatim from the project README (2026-11-08 README refresh); the README now links here. -->

# Headless & Server Modes

**Headless sessions** for scripts, saved teams and goal loops:

```bash
lop exec "summarize the failures in ./test.log"
lop exec "long migration" --background   # detach with a log file
lop exec "audit deps" --json             # one JSON line per event
lop exec "review the change" --team release --background
lop exec --goal "Finish the checklist" --loop 3
lop exec --status JOB_ID                 # durable outcome and session ID
lop --resume SESSION_ID                  # live TUI attachment or cold resume
```

Foreground exit code 0 means success, so `lop exec` composes in a pipeline.
Background launch prints a readiness receipt; use `--status` for the eventual
outcome. Saved teams, reusable profiles, goals and conversation names persist
with the ordinary session. Non-TTY approvals still deny unless explicitly
changed; `--control` opts into supervisor gates, not automatic approval.
See [the full exec guide](../../docs/EXEC.md) for option combinations, stdin,
resume precedence, loop counting, approvals and lifecycle semantics.

**Server mode** exposes the agent as a FastAPI service (used by the optional
[desktop UI](https://github.com/damianvtran/local-operator-ui)):

```bash
pip install "local-operator[server]"
lop serve                 # http://127.0.0.1:1111, docs at /docs
```

The legacy HTTP API has no authentication and defaults to loopback. Keep it
away from untrusted clients; an explicit `--host` can widen the bind only when
you provide access controls in front. This is separate from the authenticated
mobile relay. See
[API filesystem boundaries](../../docs/API_FILESYSTEM_BOUNDARIES.md) for
fresh-ID profile imports, workspace-confined edit reads and live editor
buffers.
