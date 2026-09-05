---
name: mobile
description: Install and operate Local Operator phone access — the mobile daemon, portal password, LaunchAgent, and how to hand the password to the user securely.
---

# Mobile: phone access to every lop session

Read this guide before installing, rotating, or troubleshooting `lop mobile`.
The design contract lives in `docs/mobile.md`; this file is the operational
playbook an agent follows when the user asks to set phone access up.

The daemon binds **loopback only** (`127.0.0.1:4098`). Remote access is
something the user puts in front (a tunnel plus an identity proxy). Never
widen the bind.

## When the user asks to set it up

Do the work. Do not dump the commands and wait. The sequence is:

1. Ask how they want the portal password delivered (below). Do this **before**
   `install` so you are not holding a secret with nowhere to put it.
2. Run `lop mobile install`.
3. Confirm health and the closed auth gate.
4. Deliver the password only through the channel they chose.
5. Tell them the local URL and that every interactive `lop` TUI they start
   after this will appear on the phone automatically.

### 1. Ask how to deliver the password

The portal password is the whole credential for the phone UI. It must not
land in the **transcript or any other model-visible context** — a reply,
a tool argument, a notice, a commit, a ticket, Slack, or a log. Printing
it "just this once" still writes it into the conversation the next turn
re-sends. Use the `ask` tool **before** install, with one question and
these options (consequence in the description; mark the one you recommend
— it is moved to the top of the card):

- **Leave it in the Keychain only** — you never print it. They retrieve it
  themselves with `lop mobile password` (or Keychain Access, service
  `lop-mobile`). Best default when they will type it into the phone once.
- **Copy it to the clipboard** — you run `pbcopy` (macOS) / `wl-copy` /
  `xclip` and tell them it is on the clipboard, not what it is. The clipboard
  is transient; still not for a shared screen.
- **Write it to a 0600 file they name** — they give a path (for example a
  1Password import file). You write that path and nothing else. Never a path
  inside a repo.

If they answer nothing, take **Leave it in the Keychain only** and say so in
one line.

Never invent a fourth channel (email, Slack, a gist, a Linear comment, or
"I'll just paste it here"). If they ask to see it in chat, refuse: the
transcript is the context window. Re-offer the three above.

### 2. Install

```bash
lop mobile install
```

Idempotent. Generates a Keychain password if none exists (keeps an existing
one — rotation never happens behind their back), writes the LaunchAgent,
loads it, waits for `/healthz` and a closed auth gate.

Install also makes sure the phone UI is actually servable. The wheel ships
`local_operator/mobile/web/dist/` pre-built, but a source checkout or an
editable install has none (`dist/` is gitignored). When the bundle is
missing, `install` builds it in place (`pnpm install --frozen-lockfile &&
pnpm build`, via corepack when there is no global pnpm) and **fails the
install with a clear error if it cannot** — rather than leaving the daemon
up with every authed GET answering 503 "bundle not built". `lop mobile
status` reports the bundle state (`built` / `buildable` /
`missing-sources`). If a machine shows that 503, the fix is `lop mobile
install` again, not a hand-built bundle.

On a machine without launchd (Linux, a container, a CI runner):

```bash
# Foreground. Pair with LOP_MOBILE_PASSWORD in the environment, never argv.
LOP_MOBILE_PASSWORD='…' lop mobile serve --port 4098
```

Do not generate a password onto the command line (`ps` can read it). For a
foreground run, put it in the environment of that one process or use
`lop mobile password` after a Keychain-backed install.

### 3. Confirm

```bash
lop mobile status
```

Expect `healthy: yes` and `auth gate: closed`. An open gate is a boundary
failure — do not hand the user a URL until it is closed.

Open `http://127.0.0.1:4098` only after that. Sign-in is the password from
step 1; there is no username.

### 4. Deliver the password

Match the choice from step 1 exactly:

| Choice | What you do |
|---|---|
| Keychain only | Print nothing. Tell them: `lop mobile password` shows it; Keychain Access → service `lop-mobile`. |
| Clipboard | `printf '%s' "$pw" \| pbcopy` (macOS). Confirm "it's on the clipboard", never the value. |
| 0600 file | Write the path they named, `chmod 0600`, tell them the path. |

`install` never prints the password (it would land in the tool result,
which is model-visible). For **clipboard** or **0600 file**, retrieve it
out of band — `security find-generic-password -s lop-mobile -w` on macOS,
piped straight into `pbcopy` or the file, never through a `print` or a
tool argument the model will see. `lop mobile password` itself refuses
to print when stdout is not a TTY, so an agent cannot slurp it.

### 5. What you tell them afterwards

- Local URL: `http://127.0.0.1:4098`
- Every interactive `lop` they start now publishes itself; the phone list
  updates live. No extra flag.
- Remote access is their tunnel (Cloudflare, Tailscale, …) in front of
  loopback. The daemon will not bind a LAN address.
- Rotate with `lop mobile password` then `lop mobile restart` (rotation
  invalidates every live cookie).

## Day-to-day commands

```bash
lop mobile status
lop mobile start | stop | restart
lop mobile logs --lines 100
lop mobile logs --follow
lop mobile password          # show or rotate (interactive)
lop mobile uninstall         # keep the password
lop mobile uninstall --purge # also delete the Keychain item
```

`status` lists live / wedged / stale sessions. A TUI that is running but
missing from the list has not published — that is a bug, not a missing
flag; check `lop mobile logs`.

## Troubleshooting

- **`install` says it needs macOS launchd.** Use `lop mobile serve` in the
  foreground. Supervision is the only macOS-specific piece.
- **`auth gate: OPEN`.** The daemon is serving `/api` without a cookie.
  Stop. Do not share the URL. Check `lop mobile logs`; reinstall if the
  password never landed in the Keychain.
- **Phone or login shows "mobile web bundle not built" (503).** The install
  has no `dist/` (source checkout or editable install, where it is
  gitignored). Run `lop mobile install` — it builds the bundle. Do not
  hand-build and forget it; the next `install` is the self-healing path.
- **Phone shows "connecting to session…".** The TUI (or a phone-spawned
  child) is not publishing, or the daemon has not redialed yet. `status`
  should show the pid; if it is `wedged`, the process is alive but its
  heartbeat stopped.
- **Login rejects a password they just set.** Cookies are keyed on the
  password; a rotation invalidates them, but a *new* password they have not
  typed yet will 401. They are typing the old one, or `LOP_MOBILE_PASSWORD`
  is overriding the Keychain in the daemon's environment.
- **Port 4098 already taken.** Another `lop mobile serve`, or a leftover
  LaunchAgent. `lop mobile status` / `lop mobile stop`. Do not pick a
  random port unless they ask — the phone bookmark and the docs assume 4098.

## What not to do

- Do not print the password into a commit, a PR, a ticket, or a log file.
- Do not pass it on a command line (`lop mobile serve --password …` does
  not exist, and inventing one would be `ps`-readable).
- Do not bind `0.0.0.0`. Both daemons default to loopback, but unlike
  `lop serve`, the mobile daemon does not permit widening its bind.
- Do not skip the delivery ask and paste the password "so they have it".
- Do not treat `docs/mobile.md` as the playbook — it is the architecture.
  This guide is what you execute.
