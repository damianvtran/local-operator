---
name: team-bridge
description: Set up and check the agent messages (Team Bridge) receiver on macOS, Linux or Windows, including a backend-only machine with no desktop app or TUI open.
---

# Agent messages (Team Bridge): the background receiver

Read this guide before installing, checking or troubleshooting the receiver
that delivers agent messages to the user's chief of staff. The user-visible
feature is called **agent messages**; "Team Bridge" is its internal name. The
**receiver** is one always-on background process per user
(`python -m local_operator.team_bridge.connector`), supervised by the
platform's own user service manager:

| Platform | Supervisor | Registration |
|---|---|---|
| macOS | launchd | LaunchAgent `com.local-operator.bridge-connector` in `~/Library/LaunchAgents` |
| Linux | `systemctl --user` | unit `local-operator-bridge-connector.service` in `~/.config/systemd/user` |
| Windows | Task Scheduler | task `Local Operator bridge connector-<per-user suffix>` (logon trigger, current user, no elevation) |

It is user scope everywhere: no admin rights, no system service.

## What it does when nothing is open

The receiver long-polls the user's Radient inbox. When a message arrives it
screens it and hands it to the designated **chief of staff** session:

- the session is running: the message is delivered into it as a wake;
- the session is not running (no desktop app, no TUI, no `lop serve`): the
  receiver **starts that session's runtime headlessly** and delivers the
  message to it. No window is needed.

The chief of staff can then read the message and reply over the bridge (its
`bridge` tool's `reply` answers the received message's sender in the same
thread).

**One limit to tell the user about:** a reply or send is a write-tier tool call.
Under the default approval mode (`tool_approval_mode: ask`), a headless chief of
staff parks that call for a human approval. The approval is waiting in the
session; nobody sees it until they open the session in the desktop app or the
TUI (`lop --resume <session-id>`), or answer it from the phone. If the user
wants replies to go out with nobody at the machine, that is their decision
about their approval mode. Never change it for them.

## Prerequisites and setup without the desktop app

Check these in order. The receiver is not installed until all of them hold.

1. **Radient sign-in.** Run `lop login-status`. If Radient is missing or
   expired, run `lop login radient` and finish its browser flow yourself. Never
   accept account terms for the user or ask for their provider password/token.
2. **A Team plan organization** where agent messages are available. The setup
   preview refuses if the organization is not eligible. Plan changes belong to
   the user in the Radient console.
3. **Preview the exact served consent**, using the credential row id from
   `login-status`:

   ```bash
   lop team-bridge setup --credential-id <id>
   ```

   This is read-only: it does not enroll, enable, designate or install. Its
   JSON includes the accepted account id, tenant id and the full consent text.
   Show that text to the user and let them decide. The session id is the id of
   their existing chief-of-staff session (plain `lop sessions` shows session
   ids; `lop sessions --help` documents the listing filters). The consent
   version is data, not approval by itself.
4. **Acceptance is the user's, at their own terminal.** Prepare the exact
   command from the preview and hand it to them to run:

   ```bash
   lop team-bridge setup --credential-id <id> \
     --account-id <preview.account_id> --tenant-id <preview.tenant_id> \
     --accept-consent-version <preview.consent.version> --session <session-id>
   ```

   The command prints the served consent text in full, names the account,
   organization and chief-of-staff session, and asks for an explicit `y`. It
   refuses when it is not run at a terminal (`human_required`) and when it
   descends from an agent's shell.

   **That is a speed bump, not a security boundary**, and this guide will not
   claim otherwise: the marker is an environment variable and a TTY can be
   simulated, so an agent that deliberately works around both can still run the
   command. The real control is that the *user* accepts: run everything up to
   and including the preview, then hand the acceptance command over. Do not run
   the acceptance yourself, and do not strip the marker to make it "just work".

   It re-reads the served consent and refuses if the account, tenant, version
   or existing session differs (`account_mismatch`, `tenant_mismatch`,
   `served_consent_required`, `consent_version_mismatch`, `session_not_found`).
   It then uses the same production enrollment and designation functions as the
   desktop flow. This path needs no desktop bearer and no running desktop app.
5. The switch `team_bridge.enabled` is set by that explicit setup. Check
   current state with `lop team-bridge receiver status` or `lop config list`.
6. **The default configuration folder** (`~/.local-operator`). The receiver is
   shared per user, so it is only installed for that folder. A run with a
   different `LOCAL_OPERATOR_CONFIG_DIR` is refused (see troubleshooting).

## When the user asks you to set it up

Normally there is nothing to install by hand. The receiver is placed
automatically whenever agent messages are enrolled or turned on, and at every
start of `lop serve`, the desktop app, the TUI and the mobile daemon. On a
backend-only machine, or to check:

1. Set up consent and the chief of staff explicitly as above if this is a new
   machine. Run `lop team-bridge receiver status`.
2. If `Agent messages:` is not `on for this machine`, work through the
   prerequisites above. `install` never enrols or turns anything on.
3. Run `lop team-bridge receiver install`. It is idempotent and safe to repeat.
4. Run `lop team-bridge receiver status` again and expect
   `Receiver: running (the service manager holds it)`.
5. On Linux, read the `Runs after logout:` line (see "Linux: lingering").
6. Tell the user it is running, where its log is, and the approval limit above.

The same commands work on all three platforms. `--json` on `status`,
`install` and `remove` prints machine-readable output (`state`, `code`,
`detail`, `log`, `last_check`, `linger`).

## Checking status and logs

```bash
lop team-bridge receiver status
lop team-bridge receiver status --json
lop team-bridge receiver logs
lop team-bridge receiver logs --lines 200
```

`status` prints the service manager and registration name, the receiver state,
whether agent messages are on, the configuration folder, the log path, the last
recorded inbox check, and on Linux whether lingering is on.

The receiver writes its own rotating log at
`~/.local-operator/logs/bridge-connector.log` on every platform. It contains
content-free lines such as `stage=… code=…` and never message text. Anything
that escapes it (a crash traceback) goes to
`~/.local-operator/logs/bridge-connector.stderr.log`, or to the journal on a
Linux systemd older than 240. `status` then prints the `journalctl --user -u
local-operator-bridge-connector.service` command to read it.

"Last inbox check" moves when the receiver merges an inbox page or finishes a
resync, not on every quiet poll, so an old time on a quiet account is normal.
`logs` is the better liveness signal.

## Linux: lingering

Without lingering, systemd only runs a user's services while that user is
logged in. `install` **tries** `loginctl enable-linger` for the user (most
distributions let a user do this for themselves). It then **reads the result
back** and never assumes it worked:

- `Runs after logout: yes`: the receiver starts at boot and keeps running
  after logout.
- `Runs after logout: no …`: the attempt was refused (polkit or distribution
  policy). The receiver runs only while the user is logged in. To change that,
  the user (or an admin) runs the printed `loginctl enable-linger <user>`.
  It needs their own approval and may need admin rights, so ask before running
  anything with `sudo`.

## Turning it off

- **Stop receiving agent messages on this machine:** turn agent messages off
  in the desktop app (Turn off), or run `lop config edit team_bridge.enabled
  false`. Either removes the receiver through the same path, and it is not
  placed again.
- **Remove only the service:** `lop team-bridge receiver remove`. Agent
  messages stay on, so the next start of any Local Operator surface installs it
  again. The command says so. Use this for a reset, not to stop messages.

## Troubleshooting by status

`Receiver:` lines (`status --json` field `state`):

| State | Meaning | What to do |
|---|---|---|
| `running` (`loaded`) | The supervisor holds it | Nothing. Check `logs` if messages do not arrive |
| `installed, but not running` (`installed`) | Registered, not running | `lop team-bridge receiver install` starts it again. If it stops again, read `logs` |
| `not installed` (`missing`) | Nothing registered | Check prerequisites, then `install` |
| `no service manager on this machine` (`no_supervisor`) | No launchd, `systemctl` or `schtasks` (OpenRC, Alpine, a minimal container) | Run it in the foreground or under the host's own process manager: `python -m local_operator.team_bridge.connector --config-dir ~/.local-operator` |
| `systemd has no user session for this login` (`no_user_manager`) | systemd is installed but no user instance is reachable (SSH without lingering, a container or WSL without systemd as init) | Enable lingering (above) and run `install` again; where systemd is not the init, use the foreground command |
| `not available for this configuration folder` (`isolated`) | A non-default `LOCAL_OPERATOR_CONFIG_DIR` | Expected for test and scratch folders. The shared receiver serves only `~/.local-operator`; for another folder, run the printed foreground command |

`install`/`remove` outcomes (`--json` field `code`), beyond the states above:

- `installed`, `current` or `registered`: success. `registered` means the
  registration was already correct and was only started again.
- `human_required` (from `setup`): the acceptance command must be run by the
  user at their own terminal, not from an agent. Have them run the exact command
  printed in the preview.
- `not_wanted` (`--json`: `not_wanted: true`): agent messages are not set up or
  are off. Nothing was installed. Follow the printed steps.
- `source_checkout`: this `lop` runs from a source checkout and will not point
  the user's service at it. Use the installed `lop`.
- `not_addressable`: the process's home folder is redirected (a sandbox), so
  the real service manager was not touched. Run it from the user's own shell.
- `unknown`: settings or the bridge store could not be read, so nothing was
  changed. Fix the config file (`lop config list` reports parse errors), then
  retry.
- `failed`: the supervisor refused. The detail carries its own words
  (`launchctl`, `systemctl` or `schtasks`). Read it and `logs`.

Two connectors for one account cannot run. A second one exits on its own
("another bridge connector holds this account's machine-wide lock"), so a
foreground run next to an installed receiver is harmless but pointless.

## What not to do

- Never enrol, consent, choose a chief of staff or change the approval mode on
  the user's behalf. Those are their decisions, made in the desktop app.
- Never hand-write the plist, unit or task, and never point it at a
  worktree's Python. `install` is the only supported writer.
- Never run `install` with a scratch `HOME` or a non-default config folder to
  "test" it on the user's machine. It refuses, and that refusal is the safety.
