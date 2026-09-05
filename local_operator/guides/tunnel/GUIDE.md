---
name: tunnel
description: Set up Radient-authenticated phone access, personal Cloudflare tunnels, Local Operator or OpenCode harness routes, billing quotes, connector lifecycle, and revocation.
---

# Radient personal tunnels

Use `lop tunnel` when the user asks to reach local harnesses from a phone or
another device. The connection is generic: Local Operator and OpenCode each
receive a collision-protected personal hostname. A Radient login is required;
it does not expose the machine by itself.

1. Check `lop login-status` and ask the user to complete `/login radient` if
   needed. With several Radient logins, use the bracketed ID shown in that
   listing as `--credential-id`. Do not read auth.db or print credentials.
2. Run `lop tunnel billing` and show its actual monthly price. Only activate
   billing at the exact amount the user has accepted; do not invent a fixed
   price or silently accept a changed quote. Supply that amount with
   `lop tunnel create --accept-monthly-price <amount>`.
3. Install cloudflared 2025.4.0 or newer using the platform's supported
   package manager if it is missing. Run `lop tunnel install`, then verify
   `lop tunnel status` before giving the user the printed URL.
4. The phone signs in through Radient. Do not ask the user to copy a relay
   password into chat: the local gateway authenticates the relay internally.

For a tunnel already created in the console use `lop tunnel connect <id>`.
For OpenCode use `lop tunnel configure --opencode-port <port>`. If origin
Basic auth is necessary, accept a user-named private `0600` file with username
and password and pass `--opencode-auth-file <path>`; never read that file into
the conversation. Credentials remain on the local device.

`lop tunnel stop` withdraws the connector without stopping sessions.
`lop tunnel revoke` also removes the cloud routes. `uninstall` only removes
the local user service, not cloud resources or billing. Suspended billing
preserves configuration; top up and reactivate in the Radient console.

Keep every listener on loopback. Do not run quick anonymous tunnels, widen
binds, bypass proof verification, print connector tokens, or enable cloudflared
debug logging. The gateway rejects unsigned requests, cross-user identities,
sibling Origin mutations, altered bodies, and configuration-version mismatches.

On Linux the connector supports a user systemd service; prepare the mobile
relay separately with its documented private password environment. On other
platforms `lop tunnel serve` runs in the foreground. The computer and harnesses
must stay running for remote access.
