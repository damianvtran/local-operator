---
name: credentials
description: How to use a stored secret from bash and eval without reading it, when to store one and when not to, and what to never do with a credential (echo, save, commit, paste it).
---

# Working with credentials and secrets

This is for an agent in the middle of a task that has just been handed a
credential, or needs one that is already stored. It says how to use a secret
without ever reading it, and what the store does and does not protect against.

## The three places a secret can live

| Where | Lifetime | You can read the value | Reach it with |
|---|---|---|---|
| **Session credential** — `/credential`, `ask secret=true` | this session | no, never | `$NAME` in `bash` |
| **Long-term store** — `lop secret`, the `secret` tool | forever, encrypted | only through the paths below | `$(lop secret get NAME)`, `secrets["NAME"]` |
| Provider API keys | forever, plaintext | not yours to touch | the harness reads them itself |

A session credential is injected into every `bash` child's environment, so it
is already `$NAME` there and needs nothing from you. Everything below is about
the long-term store.

## From bash

Interpolate it **directly into the command that needs it**:

```bash
curl -H "Authorization: Bearer $(lop secret get GITHUB_TOKEN)" https://api.github.com/user
```

The value crosses a pipe into the child's argv. You never see it, and it never
enters the transcript.

Do **not** do any of these — each one puts the secret somewhere it outlives the
command:

```bash
echo "$(lop secret get GITHUB_TOKEN)"          # prints it
TOKEN=$(lop secret get GITHUB_TOKEN); echo $TOKEN   # same, one step later
lop secret get GITHUB_TOKEN > /tmp/token       # writes it to disk, unencrypted
```

`lop secret get` writes the value to stdout with no trailing newline and exits
non-zero with empty stdout on any failure, which is what makes `$( )` safe: a
missing secret gives you an empty string and a failed command, never a partial
or a diagnostic in place of a value.

For a secret that is a FILE (a service-account JSON, a PEM), use the form that
materialises it for one command and removes it afterwards:

```bash
lop secret file GCP_SA_JSON -- gcloud auth activate-service-account --key-file "$GOOGLE_APPLICATION_CREDENTIALS"
```

To hand several secrets to one command as environment variables:

```bash
lop secret run --secret GITHUB_TOKEN --secret NPM_TOKEN=NODE_AUTH_TOKEN -- npm publish
```

## From eval

```python
token = secrets["GITHUB_TOKEN"]
requests.get(url, headers={"Authorization": f"Bearer {token}"})
```

`secrets` is already bound in the kernel namespace — no import needed. Each
lookup is one round trip and one audit entry. `"NAME" in secrets` checks
existence without retrieving. `import secrets` still gets the stdlib module, as
usual.

The value is a real `str`, so f-strings, concatenation and `.encode()` all work.
Its `repr` shows `[redacted]`, and anything it does reach — stdout, stderr,
`display`, the cell's result — is scrubbed before the model sees it. **Do not
rely on that as permission to print it.** It is a safety net for accidents, not
a channel: a value you deliberately write to a file or post to a service has
left the harness entirely.

## When to store a secret

`secret` tool, `op="store"` — or `lop secret set NAME` from bash. Storing is
your decision to make.

**Store it when:** you have just minted a token that will be needed after this
session; the user pasted a credential they will clearly need again; a setup step
produced a key. Prefer this store over writing to a plaintext `.env`.

**Do not store:** anything the user said not to keep; a value needed for exactly
one command; provider API keys the harness already manages; anything you are not
sure the user wants persisted — ask instead, with `ask` and `secret=true`
(add `persist=true` and the answer is saved long-term as well as for the
session).

To promote a credential the user already handed to this session, they run
`/credential --persist NAME`.

## What you must never do

- **Never echo, print, or log a secret**, in any surface — bash output, an eval
  cell, a notice, a commit message.
- **Never write one to a file** you do not immediately delete, and never into
  the repository. Not `.env`, not a config file, not a scratch note.
- **Never put one in a commit, a PR body, an issue, or a message** to anyone.
  If you think you have, say so immediately — a leaked credential must be
  rotated, and only the user can do that.
- **Never paste one into a tool argument** that is echoed back, including your
  own reasoning text.
- **Never move a secret between stores** without being asked. Reading one out of
  the long-term store to put it somewhere more convenient defeats the store.

## What is recorded

Every store, retrieval, update, delete and failed authorization is appended to a
hash-chained audit trail with a timestamp, the secret's id (never its value),
the session id, and the calling process's pid and executable path.
`lop secret audit --verify` checks the chain. So "I just read one to check" is
visible to the user afterwards.

## What this actually protects against — do not overstate it

Be accurate if the user asks, because the honest answer is the useful one.

The store is AES-256-GCM encrypted with blind-indexed names, so **the "scan the
disk for credential files" malware that a bad link actually drops finds
nothing** — no readable `.env`, no greppable `API_KEY=`, not even the names. In
the default mode the master key is a file beside the store, so an attacker who
knows to look for it reads both and decrypts; the opt-in passphrase mode holds
the key only in a running broker's memory, where reading it needs a macOS
authorization prompt the user would see.

**But anything running as the user that is willing to run `lop` can read these
secrets**, exactly as you do — `lop` is on `PATH` and a script that spawns a
session is a legitimate caller. Ancestry proves lineage, not intent. That is the
fundamental limit and no design confined to one user account removes it.

So: **a large and worthwhile increase in the cost of stealing these credentials,
not a vault.** Never tell the user their secrets are "safe" or "protected"
without that qualification.
