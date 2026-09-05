# Forking a conversation

`/fork` branches the conversation you are in into a **new session that starts
with a copy of its history**, leaving the original running and untouched. It is
the `git checkout -b` of sessions: keep everything established so far, then try
a direction without spending or contaminating the conversation that got you
there.

```
/fork                       follow an idle branch in this terminal
/fork try the streaming parser instead    follow a branch and start that instruction
/fork --window              open the branch elsewhere for this invocation
/fork --switch              follow the branch here for this invocation
/fork -- --window is literal prompt text
```

The expensive part of a long session is the context inside it — files read,
decisions made, conventions established. Before `/fork`, exploring a risky
direction from that state meant carrying on and polluting the conversation with
a dead end that keeps being re-billed, `/new` and rebuilding from scratch, or
`/compact`, which is lossy and not reversible.

## What happens

1. The authoritative runtime copies the **last committed, paired history** into
   a fresh session id, under the transcript writer's lock. It does not interrupt
   a tool or wait for a running tool to finish. Work still in progress is not
   copied; a partial multi-tool suffix is omitted as one unit. Malformed interior
   history fails explicitly rather than silently deleting completed work.
2. By default, **this terminal follows the fork**. An explicit saved
   `fork.mode: window` preference remains unchanged. The one-off destination
   flags override that preference without writing it.
3. **The original's work keeps running in its runtime.** Tools, subagents, wakes,
   and unanswered approvals stay with the original, even when
   `runtime.background_on_resume` is false. The receipt gives `/resume <parent-id>`
   to return in this same terminal, attaching to the same active owner rather
   than restarting it. An idle owner can naturally reap; its saved conversation
   remains resumable. This is not a promise to keep an idle process alive forever.
4. If you passed a message, the fork starts working on it as its first turn.
   Normal reattachment does not replay that opening instruction. This does not
   promise exactly-once delivery across a crash while consuming its sidecar.

Press **Esc while copying** to stay in the original. A filesystem copy already
in progress must settle before its writer lock is released; the resulting saved
fork is reported with a `/resume` command and is not automatically opened.
Repeated fork commands during a copy are refused, so they cannot create hidden
extra branches.

The argument is a **message, not a title**: `/fork try the streaming parser`
spends a real model call in the fork the moment it opens, and that turn is
billed like any other. A bare `/fork` opens the branch idle and costs nothing
until you send it something.

While a `/compact` is running, `/fork` refuses rather than defers: compaction
rewrites the head of the transcript, and a copy taken mid-rewrite can capture a
half-written file. Wait for it to finish.

## What the fork inherits

| | |
|---|---|
| The conversation | **yes** — retained transcript rows are byte-identical; incomplete in-flight tool suffixes are excluded |
| Images and attachments | **yes** — they live in a shared content-addressed store |
| The `/team`, `/agent` and `/goal` you had attached | **yes** |
| The provider cache | **yes, deliberately** — see below |
| The parent's **title** | **no** — the fork names itself (see below) |
| Subagents the parent launched | **no** — those jobs belong to the parent's process |

The subagent roster is dropped on purpose. It records children launched by the
parent's process, and `hub` resolves them against that process's own registry —
a fork inheriting the roster would list jobs it cannot peek, steer or cancel.
The parent keeps its children; `hub op='peek'` in the parent still reaches them.

## The fork gets its own name

A fork is **named for what you forked it to do**, not for what the parent was
doing. `/fork rewrite this with a state machine` produces a session titled after
that instruction; a bare `/fork` stays unnamed until your first message in it and
is named from that.

This matters because forks are how you end up with several related sessions at
once, and a picker listing four rows all called "Refactor the loader" is not a
picker. Until the fork's own title lands (a second or two), its `/resume` row
shows the parent's name so it is never a blank row — tagged `[fork]` ahead of
the borrowed title, so the branch and the trunk are never two identical rows:

```
❯ [fork] Refactor the YAML loader to stream a…  just now  a1b2c3d4e5f6
        Refactor the YAML loader to stream a…   just now  9f8e7d6c5b4a
```

The tag is searchable — typing `fork` in `/resume` lists the forks that are
still wearing an inherited title. The same mark prefixes the window/tab title
and the phone's history list, and all three drop it the moment the fork names
itself.

## Cache warmth

The fork's first request is designed to hit the same provider cache the parent
had warm — this is why the transcript is copied verbatim rather than
re-serialised, why the fork opens in the parent's working directory, and why the
attached agent profile comes across.

- **Anthropic** keys its cache on prefix *content*, so a fork's identical prefix
  is eligible to hit. That cache has a five-minute TTL, so a fork taken from a
  session you were just working in hits; a fork of a session idle for an hour is
  cold — but so is the parent's own next turn.
- **OpenAI / Responses** routes on a `prompt_cache_key`. A fork inherits its
  parent's key rather than minting a new one, so the branch stays on the warm
  prefix. Credential stickiness is scoped separately and is *not* shared.

## Where the fork opens

Two settings, in `/settings` under **Fork**, or via `lop config edit`:

| Key | Default | What it does |
|---|---|---|
| `fork.mode` | `switch` | `switch` follows the fork here; `window` opens it elsewhere. Both preserve the original's work |
| `fork.cmux_placement` | `workspace` | Under cmux, `workspace` gives the fork a sidebar row; `surface` gives it a tab here |

Both take effect on the very next `/fork` in the same session. Use `/settings`,
then **Fork**, to edit them live. Flags are accepted only before the prompt;
`--` ends option parsing, and unknown or conflicting flags fail without copying
or sending a model instruction.

The terminal-specific launchers below apply to **window mode only**:

| Terminal | What opens |
|---|---|
| cmux | a new workspace (or a surface in this one) — never focused, so it does not interrupt you |
| Ghostty | a new window (`open -na Ghostty.app` on macOS, the CLI on Linux) |
| kitty | a new OS window, via remote control when it is enabled |
| WezTerm | a new window via the mux, else a fresh instance |
| iTerm2 / Terminal.app | a new window, via AppleScript |
| anything else | see below |

## When no window can be opened

In **window mode**, over plain SSH, on a bare tty, or in an unrecognised
terminal, `/fork` prints a receipt instead:

```
forked to a1b2c3d4e5f6 — no window server over ssh; run `lop --resume a1b2c3d4e5f6`
```

**This is a normal outcome, not a failure.** The fork is created and durable
*before* any window is attempted, so it always exists: it is in `/resume` like
any other conversation and the command above reaches it. The same receipt
appears if a window was attempted and could not be opened.

The one hard failure is the copy itself failing (a read-only volume, a full
disk). That says so, and no fork is created.

## Notes

- Forks appear in `/resume` as ordinary conversations, because they *are* your
  own work. A bare `lop --resume` with no id resolves to the most recent
  session, which after a fork is the fork.
- A failed switch leaves the saved fork reachable with `/resume <fork-id>`;
  it never silently opens an external window as a fallback. Unsupported legacy
  in-process launchers refuse a busy switch before creating a fork, because
  disposing those launchers would destroy the work they own.
- Forking a runtime on a different machine is refused: this terminal cannot
  resume that machine's saved fork through its local session store.
- Ghostty on macOS receives a shell-quoted `--initial-command` configuration
  value through `open -na Ghostty.app --args`. Passing the CLI `-e` shortcut
  through `open` can cause duplicate execution in Ghostty 1.3.1 (#572).
  Linux retains its supported `ghostty -e` CLI form. Ghostty's execution-approval
  dialog is a security boundary and is not bypassed by this fix.
- `^F` inside a `/btw` aside is a different, older gesture: it folds an aside
  into the chat. It has nothing to do with forking a session.
