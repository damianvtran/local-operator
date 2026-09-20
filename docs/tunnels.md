# Radient personal tunnels

A personal tunnel connects a phone or remote browser to harnesses running on
your computer. Local Operator and OpenCode have separate HTTPS hostnames, such
as `<random>-lop.radienthq.com` and `<random>-oc.radienthq.com`. Radient allocates
the names atomically and keeps deleted names reserved. Your model login does
not publish anything: creation and startup are explicit actions.

The remote device signs in with Radient at the cloud edge. It can then view
Local Operator's running sessions, start a new session, send or steer a turn,
answer prompts, or return to a previous conversation. Your computer and its
harnesses must remain running and connected to the internet.

## First setup

1. Sign in or create an account using `/login radient` or `lop login radient`.
   New users review and acknowledge the linked terms and privacy policy in
   the browser, then return through native authorization. Existing users sign
   in with their current account. `lop login-status` shows the credential ID.
2. Install `cloudflared` 2025.4.0 or later through your platform's package
   manager. The connector uses its token-file option; no Cloudflare account
   login or administrative API key is needed on your computer.
3. Run `lop tunnel billing --credential-id <id> --json` to verify that exact
   login and a strictly positive credit balance. `/usage` displays Radient
   credits too, but its cache is not used to authorize setup. If credit is
   zero or negative, add credit in the console and repeat the fresh check.
   Existing starter credit counts; there is no required purchase when the
   balance is already positive. Check the current monthly price (currently
   **USD 0/month**, subject to the live quote). Billing uses
   Radient's configured allocation of actual infrastructure cost with an 80%
   gross margin, rather than a Cloudflare Access seat fee. The quote is one
   owner-level monthly amount; it is not multiplied by the number of tunnels.
4. Run `lop tunnel create --credential-id <id>`, then `lop tunnel install`.
   If activation is required, add `--accept-monthly-price <quoted-USD-amount>`
   only after accepting that exact quote. Positive credit is shared account
   credit, not an additional tunnel fee. Initial setup requires positive
   credit; existing subscriptions keep their −USD 1 suspension floor.
5. Run `lop tunnel status`. Open the printed harness URL on your phone and
   sign in with the same Radient account.

In the TUI, `/mobile billing` shows the quote and `/mobile enable <amount>`
creates or reactivates the tunnel, then installs its connector. `/mobile status`, `/mobile stop`, and
`/mobile start` use the same lifecycle as the CLI. Billing is never activated
silently by a provider login.

macOS installation prepares the existing mobile relay and installs a user
LaunchAgent for the connector. Linux installs a user systemd unit; start the
mobile relay separately with `lop mobile serve` and supply its password through
the existing private environment configuration. An explicit
`LOP_MOBILE_PASSWORD` supplied during connector installation is retained in
private local connector configuration so the user service can authenticate the
relay; macOS Keychain passwords need no extra copy. On other platforms, run
`lop tunnel serve` in the foreground. User services run while that user is
logged in; the OS's own user-service policy controls availability after logout.

If you created a tunnel in the [Radient console](https://console.radienthq.com/dashboard/tunnels),
use `lop tunnel connect <id>`
on the computer that will host it. This prepares the relay and starts the
connector service. Use `--no-start` to save its configuration only. A local connector
is pinned to the selected Radient OAuth credential. With several Radient
logins, run `lop login-status` and pass the desired Radient row's bracketed ID
as `--credential-id`; model quota routing never chooses the owner of your tunnel.

## Harness configuration

Local Operator uses port 4098 by default. To also expose an already-running
OpenCode web server on loopback port 4096:

```sh
lop tunnel configure --opencode-port 4096
lop tunnel restart
```

Use `--mobile-port`, `--opencode-port`, `--no-mobile`, or `--no-opencode` to
configure the harness set. The gateway listens on loopback port 4100 by
default, leaving 4099 for the browser bridge and 4098 for the mobile relay.
`--gateway-port` selects it at creation. It must differ from every
harness port and remains fixed for that tunnel. Revoke and recreate to change
it, so a running connector never targets a port before the proof gateway owns
its listener. Ports below 1024 and arbitrary upstream URLs are rejected: a
harness is always dialed on loopback at a numeric port, never at a URL the
cloud supplies. Harness ports are additionally pinned on this device to the
record the last local `create`, `connect`, or `configure` stored, so changing a
harness port in the console alone stops the connector until you re-approve it
locally. That keeps a console session from silently repointing a harness at an
unrelated loopback service and handing it this device's relay credentials. The
connector records the reason, the remedy and the local state file, and writes a
timestamped line to `~/.local-operator/tunnel/service.log`:

```
2026-09-19T14:32:07-07:00 local_operator.tunnels.service: connector parked reason=reenrolment_required attempts=1 — run lop tunnel connect: Harness port for local-operator changed in the console. Run lop tunnel connect again.
```

Two things are load-bearing in that line. It is **parked**, not stopped: the
connector exits successfully so its supervisor stops retrying it (see
[A dead Radient login](#a-dead-radient-login-parked-connector) for why a retry
loop was the bug), and it keeps the reason and the remedy in
`~/.local-operator/tunnel/state.json` where every surface can read them. And it
is rate-limited: a repeat of the same park is silent, so a supervisor that does
restart the unit (a reload, a reboot loop) cannot fill this log again.

`lop tunnel status` now leads with the connector's own state, so the condition
is visible there rather than only in the log:

```
Connector: parked — needs re-enrolment (since 14:32): Harness port for local-operator changed in the console. Run lop tunnel connect again.
Tunnel: tunnel-1
Status: active
```

Run `lop tunnel connect` to accept the console's current ports on this device;
`lop tunnel status`, `lop tunnel install` and `lop tunnel start` each re-arm a
parked connector once the record matches. Harnesses run
separately; the tunnel does not install OpenCode or change its server's bind
address.

For an OpenCode server that requires Basic authentication, create a private
file outside any repository containing `{"username":"...","password":"..."}`,
set its permissions to `0600`, and pass `--opencode-auth-file <path>` to
`create`, `connect`, or `configure`. Those credentials stay in private local
connector configuration and are never uploaded to Radient or printed.

Local Operator's existing password gate remains active on loopback. After
verifying the edge proof, the gateway supplies its local authentication cookie
internally, so the phone does not need to enter a second password. Neither that
cookie nor an OpenCode password is sent to the browser.

## Stop, billing suspension, and revocation

- `lop tunnel stop` stops this computer's connector; local sessions continue.
- `lop tunnel configure --disable` disables the cloud tunnel.
- `lop tunnel configure --enable --accept-monthly-price <amount>` reactivates
  the cloud configuration, subject to billing eligibility.
- `lop tunnel revoke` stops the local connector and deletes its cloud routes.
- `lop tunnel uninstall` removes the local service while keeping cloud
  configuration; it is not a billing cancellation or cloud revocation.

The balance hard floor is USD -1. Billing suspension blocks remote access and
preserves tunnel configuration. Add credit and reactivate through the
[Radient console](https://console.radienthq.com/dashboard/tunnels); the installed service
retries every 10 seconds and reconnects once
eligible. `lop tunnel billing` reports the amount due and current quote. A
stopped or suspended tunnel does not stop work already running locally.

## When the relay refuses a request

Once the 30-second authorization lease lapses the gateway answers `503` with a
body whose `detail` names the cause, so a lost network is not mistaken for a
withdrawn authorization. `detail` leads the body, then the machine-readable
`reason`, then the long-standing `error` value: a phone renders this JSON in its
browser with no viewer to fold it, so the sentence has to be the first thing
read.

- `control_plane_unreachable` — "This computer could not reach Radient to renew
the relay authorization (its network may be down, or Radient may be
unreachable). It reauthorizes by itself once the control plane answers again —
check this computer's network connection if it does not clear." No local command
is needed; the connector retries every 10 seconds.
- `authorization_refused` — Radient answered and refused the check, so the login
may have expired or this tunnel's billing may be inactive. Both are fixed at
<https://console.radienthq.com/dashboard/tunnels>.
- `tunnel_not_authorized` — the tunnel is revoked, suspended, disabled, stopped
on this computer, or its configuration changed.
- `authorization_lease_pending` — the lease has not been renewed yet, which
normally clears by itself within a few seconds.

`lop tunnel status` prints the same cause beside the connector state, worded for
a terminal: it names the commands a phone cannot run (`lop login radient`, `lop
tunnel install`), which the relay's own sentence leaves out, and links the
console wherever the console is the remedy. The one sentence that travels to
more than one surface — a park's — names no command at all, because it is also
written into `state.json`, forwarded to the desktop as `connector.detail` and
rendered by the terminal's own card: each of those appends the command in the
spelling it can run (`lop login radient` in a shell, `/login radient` in the
app's composer). The states it reports are
`connected`, `connecting`, `not serving` (the gateway answered and is refusing,
with cloudflared possibly still attached to the edge), `stopped` — with the line
saying so when nothing answered on the gateway port at all — and `parked`, the
connector's own state, described next.

## A dead Radient login (parked connector)

The connector owns the tunnel with one Radient login, and no retry can restore
a login the identity provider has stopped accepting. Before, the connector
exited 1 on that failure and its supervisor restarted it every 10 seconds for
ever: 870 identical lines in `service.log` with no timestamp and no state beside
them, while `lop tunnel status` kept reading `Status: active` off its cached
cloud record, and the phone was simply unreachable.

A login the credential store judges dead is now a **park**. The connector writes
why to `~/.local-operator/tunnel/state.json` (0600, written atomically), logs one
timestamped line, and exits 0 — the one exit both launchd and systemd read as
"do not restart me". A park covers the three failures a retry cannot fix: a dead
login (`login_required`), a missing local prerequisite (`local_prerequisite`:
cloudflared, the mobile relay), and an enrolment the console invalidated
(`reenrolment_required`). Everything else keeps the old behaviour, including a
suspended or disabled tunnel, which comes back by itself at the 10-second floor
once the operator tops up credit or reactivates it.

```console
$ lop tunnel status
Connector: parked — login required (since 14:32)
  The connector's Radient login is no longer valid, so it stopped and will not retry by itself. Signing in again starts it again on its own.
Login: sign-in expired — run lop login radient
Tunnel: tunnel-1
Status: active (cached — cloud read failed)
Cloud status: unavailable — showing the record stored at the last connect.
mobile: https://lo-divine-frost.radient.run
```

The state line carries the state and its age; the park's own sentence is a
continuation row beneath it; the command appears once, on `Login:`, which is the
line this device answers for. A tunnel the operator stopped is told no command
at all (`Login: sign-in expired (not in use — tunnel stopped)`, and no cloud
line), which is what `--json`'s `null` remedy has always said.

Three things are being said separately there, deliberately. `Connector:` is this
machine. `Login:` is this device's credential store, checked locally, which is
the only thing that can answer while the login is dead. `Status:` is the cloud's
record, marked **cached** whenever this command could not read it fresh — reading
`active` off a cached copy is exactly how a withdrawn tunnel looked healthy. The
line under it states that provenance and no cause: a verdict from the relay's own
vocabulary, printed by a command that has just said it could not read the cloud,
is what used to sit there. The cause stays available as data — `cloud.reason`.

`--json` emits the same three as data, for the desktop app and for scripts:

```json
{
  "tunnel_id": "tunnel-1",
  "cloud": {"status": "active", "source": "cached", "reason": "control_plane_unreachable"},
  "connector": {
    "state": "parked",
    "reason": "login_required",
    "detail": "The connector's Radient login is no longer valid, …",
    "since": 1789857517,
    "remedy": {"command": "lop login radient", "url": "https://console.radienthq.com/dashboard/tunnels"}
  },
  "login": {"credential_id": 50, "state": "login_required"},
  "remedy": {"command": "lop login radient", "url": "https://console.radienthq.com/dashboard/tunnels"}
}
```

`connector.state` is one of `parked`, `connected`, `connecting`, `not serving`,
`stopped`; `login.state` is one of `ok`, `login_required`, `unknown` — and
`unknown` means the check itself could not run (a refresh could not reach
Radient), never "your login is dead". `remedy` is an OBJECT
(`{"command": str, "url": str}`), not a bare command string, and `connector`
carries the same shape for a park. `connector.detail` is the park's own sentence:
it names NO command, because it is read by a shell, by the TUI and by the desktop
app, so each of those appends `remedy.command` in its own spelling.
`connector.since` and `.first_at`-style stamps are epoch seconds. The desktop app
reads the same shape from `GET /v1/desktop/tunnel`; see
[DESKTOP_API.md](DESKTOP_API.md).

**Signing in restores it, without touching the service.** A successful Radient
login on this machine re-arms a connector parked for its login: the credential
write path kickstarts the unit (launchd `kickstart`, or `systemctl --user start`
on Linux) as soon as the grant is stored, so `/login radient` is the whole
remedy. It is guarded to touch nothing else — only this tunnel's own credential,
only while it is parked for its login, only for a tunnel that is configured and
not deliberately stopped. `lop tunnel status` and `lop tunnel start` re-arm too,
for a machine whose login was fixed somewhere else. The terminal also raises one
`! /login radient — Radient sign-in expired` card when it finds a park on startup
or sees the state change while it is open, and withdraws it when the park clears.
A card is ten seconds and a park is hours, so the card is not the whole story: the
status band keeps a standing `! remote access off` for as long as the park lasts,
which is the same treatment the MCP alarm gets (a toast AND a segment) and the
only thing still saying it to an operator who was not at the machine when it
parked. Both read the same local state file, and neither appears for a machine
with no tunnel or one the operator stopped.

## Trust boundaries and transport

The path is browser → Radient authentication Worker → Cloudflare Tunnel →
local gateway → harness. Cloudflare Access applications and paid Access seats
are not used. Agent-server is the control plane and does not carry transcript
or steering traffic. Requests and SSE bodies are streamed with backpressure.

The Worker strips browser cloud credentials and signs a separate RS256 origin
assertion lasting at most 30 seconds. The gateway pins its public keys from
the authenticated connector response and verifies owner, tunnel, harness,
configuration version, exact hostname, method, encoded path and query, and a
SHA-256 digest of the body. Mutations reject replayed assertion IDs. Merely
reaching loopback or providing an identity header grants no access.

Browser mutations require an exact matching HTTPS Origin. This matters because
cookie SameSite rules do not isolate two users' sibling subdomains. The relay
also independently checks browser mutation origins. Requests over 10 MiB are
rejected. No authorization cookies or proof headers are forwarded to harnesses;
origin-specific authentication is supplied from private local configuration.

The connector checks control-plane eligibility every 10 seconds and fails
closed after 30 seconds without a successful check. SSE and WebSocket
connections have a maximum 60-second lease, then reconnect through the edge.
Revoking an already-open connection is therefore bounded, not instantaneous.
The Local Operator mobile client already reconnects its SSE streams; other
harness clients must likewise reconnect WebSockets. Arbitrary redirect targets
and public upstream addresses are not accepted.

The connector token lives in a `0600` file under the private tunnel directory
only while the service runs. It is passed to cloudflared by file path, never
argv or logs, and cannot administer the Cloudflare account. Debug logging of
cloudflared traffic is intentionally disabled.
