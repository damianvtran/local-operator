---
name: tunnel
description: Set up Radient-authenticated phone access, personal Cloudflare tunnels, Local Operator or OpenCode harness routes, billing quotes, connector lifecycle, and revocation.
---

# Radient personal tunnels

Use `lop tunnel` when the user asks to reach local harnesses from a phone or
another device. The connection is generic: Local Operator and OpenCode each
receive a collision-protected personal hostname. A Radient login is required;
it does not expose the machine by itself.

1. Check `lop login-status` and have the user complete `/login radient` if
   needed. It supports existing-account login and new-account signup; new
   users personally acknowledge the linked terms/privacy policy in the browser.
   With several Radient logins, select the intended owner's bracketed ID and
   use the same `--credential-id` on billing, create and connect. Do not read
   auth.db or print credentials.
2. Run `lop tunnel setup --credential-id <id> --json` to observe a derived
   plan. No action argument (or `plan`) is read-only, even with price flags.
   It reads the selected account and billing fresh; cached usage is insufficient.
   Positive starter credit counts, and setup never purchases credit.
3. Follow its local prerequisite blockers before any cloud mutation. Install
   cloudflared >=2025.4.0 using the supported package manager (normal tool
   approval; macOS `brew install cloudflared`). A supported user supervisor is
   required. A phone origin that is not already healthy/protected also needs
   its interface bundle ready; use the explicit approved `lop mobile install`
   prerequisite when reported. The plan never builds/installs it.
   The runner does not silently install software or change listener
   binds. A deliberately stopped setup stays stopped, and a dead/different
   owning credential cannot silently repoint an existing configuration.
4. Run `lop tunnel setup apply --credential-id <id>` only when instructed.
   If activation is required, obtain acceptance of the exact fresh monthly USD
   price and add `--accept-monthly-price <amount>`. Already eligible means no
   new price-consent step. TUI equivalents: `/mobile status --credential-id <id>`
   and `/mobile enable [amount] --credential-id <id>`; all use the same runner.
   Partial failures retain completed receipts; retry reuses the saved create
   key and original payload. Healthy existing setup is a no-op. Give the URL
   only when `verified:true`: running connector, answering gateway, healthy
   mobile with a closed gate and a fresh active route. The URL is taken from
   that route, never guessed. This does not prove phone-side sign-in/access.
5. The phone signs in through Radient. Do not ask the user to copy a relay
   password into chat: the local gateway authenticates the relay internally.

For a tunnel already created in the console use
`lop tunnel connect <id> --credential-id <credential-id>` after the fresh check.
For OpenCode use `lop tunnel configure --opencode-port <port>`. If origin
Basic auth is necessary, accept a user-named private `0600` file with username
and password and pass `--opencode-auth-file <path>`; never read that file into
the conversation. Credentials remain on the local device.

`lop tunnel stop` withdraws the connector without stopping sessions.
`lop tunnel revoke` also removes the cloud routes. `uninstall` only removes
the local user service, not cloud resources or billing. Suspended billing
preserves configuration; top up and reactivate in the Radient console.
The positive-balance enrollment check does not change the −USD 1 suspension
floor for already-running subscriptions.

Keep every listener on loopback. Do not run quick anonymous tunnels, widen
binds, bypass proof verification, print connector tokens, or enable cloudflared
debug logging. The gateway rejects unsigned requests, cross-user identities,
sibling Origin mutations, altered bodies, and configuration-version mismatches.

On Linux the connector supports a user systemd service; prepare the mobile
relay separately with its documented private password environment. On other
platforms `lop tunnel serve` runs in the foreground. The computer and harnesses
must stay running for remote access.
