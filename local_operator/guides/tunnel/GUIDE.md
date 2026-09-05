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
2. Run `lop tunnel billing --credential-id <id> --json`. Require a fresh valid
   account and strictly positive balance (`account_valid` and `positive_balance`).
   `/usage` shows Radient credits but its cache cannot authorize setup. For
   zero/negative/unavailable credit, direct the user to the console to top up
   or repair their login, then repeat the fresh check. Starter credit counts;
   a purchase is not needed if the balance is already positive.
3. Show the actual monthly price in `billing.monthly_price_usd`. The current
   allocation is USD 0/month, subject to that live quote. Positive account
   credit is not a separate tunnel fee. `setup_ready` is true when positive
   credit and billing eligibility both hold. Only activate
   billing at the exact amount the user has accepted; do not invent a fixed
   price or silently accept a changed quote. Supply that amount with
   `lop tunnel create --credential-id <id> --accept-monthly-price <amount>`.
   If already eligible, `lop tunnel create --credential-id <id>` suffices.
4. Install cloudflared 2025.4.0 or newer using the platform's supported
   package manager if it is missing. Run `lop tunnel install`, then verify
   `lop tunnel status` before giving the user the printed URL.
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
