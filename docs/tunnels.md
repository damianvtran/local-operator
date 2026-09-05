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

1. Sign in using `/login radient` or `lop login radient`.
2. Install `cloudflared` 2025.4.0 or later through your platform's package
   manager. The connector uses its token-file option; no Cloudflare account
   login or administrative API key is needed on your computer.
3. Run `lop tunnel billing` to see the current monthly price. Billing uses
   Radient's configured allocation of actual infrastructure cost with an 80%
   gross margin, rather than a Cloudflare Access seat fee. The quote is one
   owner-level monthly amount; it is not multiplied by the number of tunnels.
4. Run `lop tunnel create --accept-monthly-price <quoted-USD-amount>`, then
   `lop tunnel install`. The amount must match the server's current quote.
5. Run `lop tunnel status`. Open the printed harness URL on your phone and
   sign in with the same Radient account.

In the TUI, `/mobile billing` shows the quote and `/mobile enable <amount>`
performs creation and installation. `/mobile status`, `/mobile stop`, and
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
logins, pass `--credential-id` explicitly; model quota routing never chooses
the owner of your tunnel.

## Harness configuration

Local Operator uses port 4098 by default. To also expose an already-running
OpenCode web server on loopback port 4096:

```sh
lop tunnel configure --opencode-port 4096
lop tunnel restart
```

Use `--mobile-port`, `--opencode-port`, `--no-mobile`, or `--no-opencode` to
configure the harness set. The gateway listens on loopback port 4099 by
default; `--gateway-port` selects it at creation. It must differ from every
harness port and remains fixed for that tunnel. Revoke and recreate to change
it, so a running connector never targets a port before the proof gateway owns
its listener. Ports below 1024 and arbitrary upstream URLs are
rejected. Harnesses run separately; the tunnel does not install OpenCode or
change its server's bind address.

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
