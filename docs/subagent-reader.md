# The subagent reader

Opening a worker from the subagent dock attaches a read-only reader to that
child. This page records what the reader shows, what its keys do, and the
limits behind its empty states — detail the README used to carry and that a
first-time reader does not need, but that matters when a plan says
**Todos unavailable** and you want to know why.

## Scope

The reader shows the selected worker's own live todo list and its direct
children, not the root session's plan or unrelated workers. Click a child to
inspect it; `Esc` returns one parent at a time, `[` / `]` cycle peers, and `c`
opens the first child. The reader stays read-only.

At shorter terminal heights the disabled composer is collapsed so the
transcript, child controls, and plan remain useful; `ctrl+t` and `ctrl+g`
expose the scrollable full plan and child list.

## Empty states

Attached terminals fetch child details only while that worker is open, so a
plan has three distinct "nothing to show" states:

| Label | Meaning |
| --- | --- |
| **Loading todos** | The plan has been requested and has not arrived yet. |
| **No todos** | The plan arrived and is empty. |
| **Todos unavailable** | The owner is too old to serve plans, its history is unavailable, or the child plan exceeds the wire cap below. |

**Todos unavailable** is shown rather than a partial list: a truncated plan
would read as a complete one.

## Limits

- **Wire cap.** A child plan larger than 128 KiB of serialized JSON
  (`JOB_TODOS_WIRE_BYTES` in `local_operator/session/frontend_state.py`) is
  not sent to the reader. The full plan stays on the owner; the cap does not
  truncate the todo store and does not prevent attaching to the root session.
- **Not checkpointed.** Child plans are not copied into the root's durable
  frontend checkpoint.
- **After an owner restart.** Saved child plans and the recorded child
  hierarchy and status remain inspectable, but the in-memory tool-event window
  is gone. Attached child pages do not yet page the full saved transcript and
  report that history limitation.
