# Forking a conversation

`/fork` branches the conversation you are in into a **new session that starts
with a copy of its history**, leaving the original running and untouched. It is
the `git checkout -b` of sessions: keep everything established so far, then try
a direction without spending or contaminating the conversation that got you
there.

```
/fork                       branch this conversation; the fork opens idle
/fork try the streaming parser instead    branch it, and start the fork on that
```

The expensive part of a long session is the context inside it — files read,
decisions made, conventions established. Before `/fork`, exploring a risky
direction from that state meant carrying on and polluting the conversation with
a dead end that keeps being re-billed, `/new` and rebuilding from scratch, or
`/compact`, which is lossy and not reversible.

## What happens

1. The conversation is copied at the next **safe boundary** into a fresh session
   id. At idle that is immediate. Mid-turn it is the next tool-loop boundary,
   and a long-running interruptible tool (`wait`, `bash`, an MCP call) is
   interrupted so the boundary arrives in about 250 ms rather than minutes.
2. The fork opens **in a new window** — a new cmux workspace by default under
   cmux — already running `lop --resume <fork-id>`.
3. **This session keeps running.** No transition, no reset; the original is not
   modified in any way.
4. If you passed a message, the fork starts working on it as its first turn.

The argument is a **message, not a title**: `/fork try the streaming parser`
spends a real model call in the new window the moment it opens, and that turn is
billed like any other. A bare `/fork` opens the branch idle and costs nothing
until you send it something.

While a `/compact` is running, `/fork` refuses rather than defers: compaction
rewrites the head of the transcript, and a copy taken mid-rewrite can capture a
half-written file. Wait for it to finish.

## What the fork inherits

| | |
|---|---|
| The conversation | **yes** — a byte-identical copy of the transcript |
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
| `fork.mode` | `window` | `window` opens the fork elsewhere; `switch` moves this terminal onto it |
| `fork.cmux_placement` | `workspace` | Under cmux, `workspace` gives the fork a sidebar row; `surface` gives it a tab here |

Both take effect on the very next `/fork` in the same session.

| Terminal | What opens |
|---|---|
| cmux | a new workspace (or a surface in this one) — never focused, so it does not interrupt you |
| Ghostty | a new window (`open -na Ghostty.app` on macOS, the CLI on Linux) |
| kitty | a new OS window, via remote control when it is enabled |
| WezTerm | a new window via the mux, else a fresh instance |
| iTerm2 / Terminal.app | a new window, via AppleScript |
| anything else | see below |

## When no window can be opened

Over plain SSH, on a bare tty, or in an unrecognised terminal, `/fork` prints a
receipt instead:

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
- Nothing rate-limits `/fork`. Ten forks are ten windows and ten live sessions.
- `^F` inside a `/btw` aside is a different, older gesture: it folds an aside
  into the chat. It has nothing to do with forking a session.
