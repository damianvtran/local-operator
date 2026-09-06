# Herdr: live agent state in the Agents panel

[Herdr](https://herdr.dev) is a different kind of integration from the resume
binding in [multiplexer-resume.md](./multiplexer-resume.md): instead of *where* a
session lives, `lop` reports *what it is doing*. Herdr's Agents panel shows one
row per pane with a lifecycle state, and
by default it fills that row by screen detection, which cannot tell "the model
is thinking" from "a tool approval is waiting on you". `lop` already knows,
because it is the same three-valued state the terminal title carries — so
inside a Herdr pane it reports that state through Herdr's documented
custom-agent CLI:

| `lop` | Herdr Agents row |
|---|---|
| at the prompt (`lo ›`) | `idle` |
| a turn running (`lo ⣻`) | `working` |
| a tool approval or `ask` waiting (`lo !`) | `blocked` |
| exit | released (`pane release-agent`) |

Detection is `HERDR_ENV=1` plus `HERDR_PANE_ID`, both of which Herdr injects
into every pane process. Reporting additionally needs the `herdr` CLI:
`HERDR_BIN_PATH` is preferred (it names the binary that spawned the pane, so
its protocol matches the running server), with `herdr` on `PATH` as the
fallback. Outside Herdr, or with neither resolvable, nothing runs.

The row appears as source `custom:local-operator`, agent `local-operator`:

```sh
herdr agent list
herdr agent get "$HERDR_PANE_ID"
```

Every report carries a `--seq` so Herdr drops a stale one that lands out of
order; the session id goes along as `--agent-session-id` metadata.

**The session id is sent but not visible on Herdr 0.8.2.** `--agent-session-id`
is accepted (`rc=0`), but `agent get` / `agent list` only populate the
read-only `agent_session` object for Herdr's *official* integrations, so you
will not find the id in the output above for a `custom:` source. It is sent
anyway — it costs nothing and is correct the moment Herdr opens that field to
custom sources — but do not go looking for it today. Herdr-managed session
restoration from that id is out of scope either way.

Subagent
child sessions never report — they run inside their parent's pane and would
overwrite its row. Like the resume binding, every call is best-effort: a
missing binary, a dead socket, a non-zero exit or a timeout is logged at debug
and otherwise ignored, off the event loop, and can never slow or fail a session.

Turning it off:

```sh
export LOCAL_OPERATOR_NO_HERDR=1
```

It is independent of `LOCAL_OPERATOR_NO_TERMINAL_TITLE`: turning off the
window title does not silence the Agents row, and vice versa.
