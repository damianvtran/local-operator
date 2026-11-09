---
name: permissions
description: macOS privacy permissions (Screen Recording, Files & Folders, Accessibility) for Local Operator runtimes — which row to enable and how to clean up stale rows.
---

# macOS privacy permissions (`lop permissions`)

macOS asks for consent before a process records the screen, reads protected
folders, controls another app or drives the UI. Each grant is a row in
**System Settings › Privacy & Security › <service>**. This guide explains which
"Local Operator" row governs which part of Local Operator, how to grant it once,
and how to clean up the stale rows older versions left behind.

Start with:

```sh
lop permissions status
```

It is read-only and never prompts. It prints which identity this host's runtimes
use, whether the desktop app is installed, whether team signing is on, and
whether a runtime would be allowed to record the screen (a *preflight*: it asks
macOS for the current answer without ever triggering a permission dialog).

## The rows you will see

| Row in System Settings | What it is | Who uses it |
|---|---|---|
| **Local Operator** (app icon) | The desktop app, bundle id `com.local-operator` | The desktop app, everything it starts directly (`lop serve`, the app console) — and, on a host where team signing is on, **every runtime too** |
| **Local Operator Runtime** | The runtime bundle, `com.local-operator.runtime`, signed ad hoc on this machine | Every runtime on a host **without** team signing |
| Your terminal (Terminal, iTerm, Ghostty, …) | The terminal app | The `lop` TUI process itself |
| Old "Local Operator" rows with a generic icon, or an `exec` icon | Leftovers from older versions, keyed by a file path that no longer exists or is no longer used | Nothing — safe to remove |

**One row, where possible.** When team signing is on (`lop permissions setup`),
runtimes are signed with the same Developer ID team as the desktop app and carry
its bundle id, so macOS treats them as the app: enabling **Local Operator** once
covers the app *and* every runtime. Without team signing, runtimes share one
**Local Operator Runtime** row instead — still one row for all of them, no
matter how many sessions, worktrees or updates you run.

## Which surface is governed by which row

| You are using | Process that needs the permission | Row |
|---|---|---|
| Desktop app | The app and its `lop serve` backend | Local Operator |
| Desktop app or TUI sessions | Session runtimes (the agent doing the work, started by the session host) | Local Operator (team signing on) or Local Operator Runtime |
| TUI | The TUI process you typed `lop` into | Your terminal's row |
| `lop exec`, headless runs | The exec worker | Local Operator (team signing on) or Local Operator Runtime |
| `eval` cells, the secrets broker, browser opener, background daemons (mobile, browser bridge) | Their own runtime processes | Local Operator (team signing on) or Local Operator Runtime |
| SDK in your own Python | Your interpreter, or the terminal/IDE that started it | That interpreter's or app's own row; runtimes the SDK starts through `lop` follow the rows above |

## How it works

Runtimes are deliberately **not** children of the desktop app in macOS's eyes:
they are started "disclaimed" (or owned by launchd), so quitting, force-quitting
or crashing the app never takes running sessions down with it. The cost of that
independence is that runtimes cannot borrow the app's permissions by ancestry,
so they need an identity of their own.

macOS keys a plain executable's permissions by its **file path**, which is why
older versions accumulated one row per Python build, update and worktree. Every
runtime now executes from a small bundle, `Local Operator.lopruntime`, which
macOS keys by its **bundle id** instead — the same row at any path, across
updates and worktrees.

- **Team-signed** (after `lop permissions setup`): bundle id
  `com.local-operator`, signed with your Developer ID. Before it is used, its
  signature is checked offline against the desktop app's designated requirement.
  Measured on a host with the app's Screen Recording grant on: a disclaimed
  runtime from this bundle was evaluated by macOS as
  `requesting={identifier=com.local-operator, binary_path=…/Local Operator.lopruntime/Contents/MacOS/Local Operator}`
  with no responsible app, `subject=com.local-operator`, `authValue=2`
  (allowed) — the app's own grant, with no new row.
- **Ad hoc** (every host, the default): bundle id `com.local-operator.runtime`,
  signed on this machine without any key, shown as **Local Operator Runtime**.
  It never uses the app's id, so it can never disturb the app's own grant.

The runtime bundle is never registered as an application: it does not appear in
the Dock or the app switcher, and nothing that quits or updates the desktop app
selects it.

## Grant once

1. Run `lop permissions status` and note the identity line.
2. Open **System Settings › Privacy & Security** and the service you need
   (for example **Screen Recording**).
3. Turn on **Local Operator** if the status says the runtimes share the
   desktop app's row; otherwise turn on **Local Operator Runtime**. If that row
   is not listed yet, it appears the first time a runtime asks.
4. Run `lop permissions status` again; the preflight line should say granted.
   Restart running sessions if they were started before the grant.

### Turning team signing on

```sh
lop permissions setup
```

Interactive and terminal-only. It needs the desktop app installed and a
"Developer ID Application" certificate for the app's team in your keychain.
macOS may show one dialog asking whether `codesign` may use that key: **Allow**
works for this run; **Always Allow** lets later `lop update` runs sign new
Python builds without asking. Setup records your choice; an update running
without a terminal signs only if you chose Always Allow, otherwise runtimes keep
using the Local Operator Runtime row until you run setup again.

Setup refuses to run in a sandboxed or redirected home directory, and it never
runs as part of starting a session.

## What turning a permission off does

- Turning **Local Operator** off revokes it for the desktop app **and**, with
  team signing on, for every runtime. That is the point of one shared
  permission: one switch.
- Turning **Local Operator Runtime** off revokes it for every runtime on a host
  without team signing; the desktop app is unaffected.
- Neither stops running sessions; the affected processes simply lose access
  the next time they use the resource.

## On a Python update (ad hoc hosts)

The ad hoc signature is tied to the exact bytes of the Python build. When
`lop update` moves to a new Python build, macOS still shows the same
**Local Operator Runtime** row (no new row), but may ask once more or show the
switch off — toggle it on again. Team-signed hosts are not affected: the app's
grant follows the team signature, not the bytes.

## Removing stale rows

Older versions left one "Local Operator" row per path. To clean up, by hand:

1. Open the service in **System Settings › Privacy & Security**.
2. Select a stale "Local Operator" row (generic icon, or an `exec` icon), click
   **−**, and confirm.
3. Keep the desktop app's **Local Operator** row (app icon) and, on hosts without
   team signing, **Local Operator Runtime**.

Agents must never run `tccutil reset` for you: it wipes grants (including ones
you want to keep) and cannot target the old path-keyed rows anyway.

## Troubleshooting

- **Preflight says not granted, but the switch is on.** Check which identity
  `lop permissions status` reports and that you enabled *that* row. Restart
  sessions started before the change.
- **"Runtime identity: not created yet".** No runtime has started from this
  install yet; start a session, then check again.
- **"a path-keyed image".** The runtime bundle could not be built on this host
  (for example, no `codesign`); runtimes still work, with the older per-build
  rows. It is retried after a day.
- **"Team signing: unavailable in this process".** You are in a sandboxed or
  redirected environment (the reason code says which); run the command from
  your own login shell.
- **The TUI itself is denied.** The TUI process is your terminal's client:
  grant your terminal app, not Local Operator.
