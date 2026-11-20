---
name: google-vertex
description: Google Vertex sign-in keeps expiring (lop login google-vertex) — Google Cloud session control, invalid_rapt, admin exemption, own OAuth client, service account.
---

# Google Vertex AI sign-in, and why it expires

Use this when a user signed in with `lop login google-vertex` (or `/login
google-vertex` in the app) and, hours later, Vertex requests stop working, the
turn falls back to another model, or `lop login-status` shows the credential as
`sign-in required`. `local_operator/providers/oauth/vertex.py` and the shared
refresh classifier in `local_operator/providers/oauth/callback_server.py` are
authoritative; this guide is a reading of them that can fall behind.

## The three ways to authenticate

| Login | What it stores | Expires? |
|---|---|---|
| `google-vertex` | A Google Cloud OAuth sign-in (browser, Workspace SSO works). Refreshed automatically. | Only if the grant is revoked, or a Workspace session policy forces a re-sign-in (below). |
| `google-vertex-key` | A Vertex API key. | When the key is deleted or restricted. |
| `google-vertex-sa` | A service-account JSON key; lop signs a fresh token itself every hour. | No. Service accounts are not subject to Google Cloud session control. |

## Symptom: "Your Google Workspace requires you to sign in to Vertex AI again"

The browser sign-in worked, then 1–24 hours later:

- a turn prints `provider failure: auth — falling back to <model>`, followed
  once by `Your Google Workspace requires you to sign in to Vertex AI again
  (Google Cloud session control) …`;
- `lop login-status` lists the row as `oauth, sign-in required (Workspace policy)`;
- `gcloud auth print-access-token` for the same account fails with
  "Reauthentication failed. cannot prompt during non-interactive execution".

Cause: the organisation's Google Workspace admin has turned on **Google Cloud
session control** (Admin console › Security › Access and data control ›
Google Cloud session control; Google's article:
<https://support.google.com/a/answer/9368756>). It makes users re-authenticate
every 1–24 hours for the Cloud console, gcloud, and **any third-party app that
uses Google Cloud scopes**. When the window passes, Google refuses the refresh
with HTTP 400 `invalid_grant` / `error_subtype: invalid_rapt` ("reauth related
error"). The refresh token is not revoked, but no client can get past the
policy without an interactive sign-in. lop records the refusal on the stored
row, stops presenting the token, and keeps any configured fallback serving the
turn.

## Fixes

**Sign in again (the user can do this now).** `/login google-vertex` in the
TUI or desktop app, or `lop login google-vertex` in a terminal. This lasts until
the policy's next re-authentication window.

**Ask the Workspace admin to exempt lop** (stops the expiry). In Google Cloud
session control, either:

1. **Exempt a trusted app**: add the OAuth client lop signs in with to the
   Trusted apps list (Security › Access and data control › API controls ›
   Manage third-party app access), then tick **Exempt Trusted apps** in Google
   Cloud session control. By default lop uses the Google Cloud SDK's public
   client (`32555940559.apps.googleusercontent.com`, the one gcloud uses), so
   trusting it exempts gcloud too. To exempt only lop, register your own
   client (next section) and trust that one instead.
2. **Change the re-authentication frequency**: lengthen it (up to 24 hours) or
   set the policy to never require re-authentication, for the whole
   organisation or for the organisational unit the user is in.

**Use a service account** for unattended work: `lop login google-vertex-sa`
with a service-account JSON key. Session control does not apply to service
accounts; follow the organisation's key-handling policy.

## Using your organisation's own OAuth client

Register an OAuth client of type **Desktop app** in a Google Cloud project
(APIs & Services › Credentials), with the Cloud Platform scope available to it.
Then point lop at it — the client ID is not secret, the client secret is kept in
lop's encrypted store:

```sh
lop credential update GOOGLE_VERTEX_OAUTH_CLIENT_ID      # or config.yml, below
lop credential update GOOGLE_VERTEX_OAUTH_CLIENT_SECRET
lop login google-vertex
```

`lop credential update` warns that these names are not provider API keys and
stores them anyway; that warning is expected here.

The client ID can live in `config.yml` instead:

```yaml
values:
  providers:
    google-vertex:
      oauth_client_id: 1234567890-abc.apps.googleusercontent.com
```

Precedence for the client ID: the `GOOGLE_VERTEX_OAUTH_CLIENT_ID` environment
variable, then the secret store, then `config.yml`, then the Google Cloud SDK
client. The secret comes from `GOOGLE_VERTEX_OAUTH_CLIENT_SECRET` (environment
first, then the store). A custom client ID with no secret is refused before any
request is sent; gcloud's secret is never paired with another client.

A refresh token belongs to the client that issued it, so a sign-in remembers its
client ID and keeps refreshing with it. After you switch clients, sign in again
so the stored login uses the new one.
