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
the long-term store. The sharpest instance of the provider-key row is the
QwenCloud console ticket — plaintext, captured by the *user* from a browser,
and never yours to read; `guide://qwencloud` carries the capture ritual.

## From bash

Interpolate it **directly into the command that needs it**:

```bash
curl -H "Authorization: Bearer $(lop secret get GITHUB_TOKEN)" https://api.github.com/user
```

The value crosses a pipe into the child's argv. You never see it, and it never
enters the transcript.

Do **not** do either of these — the transcript is the one place a value must
never land, because the transcript is the model's context:

```bash
echo "$(lop secret get GITHUB_TOKEN)"          # prints it
TOKEN=$(lop secret get GITHUB_TOKEN); echo $TOKEN   # same, one step later
```

`lop secret get` writes the value to stdout with no trailing newline and exits
non-zero with empty stdout on any failure, which is what makes `$( )` safe: a
missing secret gives you an empty string and a failed command, never a partial
or a diagnostic in place of a value.

A FILE is a different matter. `lop secret get NAME > /tmp/token` puts the value
on disk unencrypted, which is **not** treated as a compromise — the model never
sees it — but it is a cleanup debt: delete the copy afterwards **without
reading it**, because reading it is what would turn a contained value into a
leaked one. Prefer a form that never leaves a copy at all:

For a secret that is a FILE (a service-account JSON, a PEM), use the form that
materialises it for one command and removes it afterwards:

```bash
lop secret file GCP_SA_JSON -- gcloud auth activate-service-account --key-file "$GOOGLE_APPLICATION_CREDENTIALS"
```

To hand several secrets to one command as environment variables:

```bash
lop secret run --secret GITHUB_TOKEN --secret NPM_TOKEN=NODE_AUTH_TOKEN -- npm publish
```

## Identifying a secret without reading it

Most of what an agent needs from a secret is its IDENTITY, not its bytes: is the
value I am about to use the one that is stored, and is this the same secret the
runbook means? `describe` answers that without printing anything:

```bash
lop secret describe GITHUB_TOKEN --length --fingerprint
```

```
name         GITHUB_TOKEN
kind         string
...
length       40 bytes
fingerprint  hmac-sha256:191abdba80c8f80cbd072522719bf381
```

`length` is the value's exact size in bytes — compare it with the `stored NAME
(N bytes)` line from when it went in, and a mismatch says the wrong thing was
written. `fingerprint` is an HMAC-SHA256 over the value, keyed on the store's
master key and truncated to 16 bytes: two calls, two names holding the same
bytes, and two sessions all produce the same digest, while a different value
produces a different one. Neither flag prints the value, and no flag on any verb
does.

**What a fingerprint proves, and what it does not.**

- It proves the two values you compared are the same bytes.
- It is **not** a hash you can compare against a copy you already hold: a plain
  `shasum -a 256` of that copy gives a different string, because the fingerprint
  is keyed.
- It is **not** stable across a master-key rotation (`lop secret rotate`), and
  **not** comparable between two stores — the same value fingerprints
  differently under a different master key, so a changed fingerprint means "the
  store's key changed" as often as it means "the secret changed".
- It is **not** proof that a value is secret or unguessable. Whoever holds the
  master key can confirm a guessed value against it (and can decrypt the store
  anyway, so gains nothing); whoever does not, cannot.

The round-trip check above — `lop secret get NAME | shasum -a 256` against the
original — still works and is still the right tool when you have a plaintext
copy to compare with. The fingerprint is for when you do not: it answers the
same question with no value-bearing pipeline to get wrong.

## When a human has to see the bytes

There is exactly one path for that, and it is the user's own terminal:

```bash
lop secret get GITHUB_TOKEN --reveal      # asks at the terminal, then prints it
```

It is refused — exit 3, **empty stdout**, and the reason on stderr — unless stdin
AND stdout are both a terminal. So an agent's `bash` call, a script and a
pipeline are refused by construction: there is nobody to ask, and nobody to ask
is not permission. There is deliberately no flag, environment variable or config
setting that stands in for the prompt, because anything the model can set in the
same call it uses is a default, not an opt-in. Answering `n` at the prompt
reveals nothing and exits non-zero.

If you are an agent wondering whether you need this: you almost certainly need
`describe --length --fingerprint` above, or one of the forms in "From bash" that
hand the value to a consumer without printing it. The reveal belongs to the user,
and it is recorded as theirs (see "What is recorded").

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

**List before asking.** When you need a credential, `list` the store (or `lop
secret list`) before asking the user for something they may have already given
you — re-asking for a value already stored is the friction this store exists to
remove.

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

**Storing a multi-line secret: use `--from-file`, not stdin.** `lop secret set`
and `lop secret update` read stdin and strip ONE trailing newline, because
`echo` and every heredoc add one and `echo hunter2` means five characters, not
six. That is right for a token on a single line and silently wrong for anything
that legitimately ends in a newline — a PEM private key, a service-account
JSON, an SSH key. Piping one in costs it its final byte, and the value comes
back one byte short with no error at any point; you find out when the key fails
to verify, far from the cause.

```bash
lop secret set DEPLOY_KEY --from-file ./deploy_key.pem   # exact bytes, newline kept
printf %s "$TOKEN" | lop secret set API_TOKEN            # single-line: stdin is fine
```

`--from-file` reads the file's exact bytes and strips nothing. It does not
remove the file, so delete the plaintext afterwards. Verify a round trip with
`lop secret get NAME | shasum -a 256` against the original when the value has
to be byte-exact.

## What actually counts as a compromise

The harness tells you when it masks a credential (`[credential redaction] …`),
and the notice classifies itself. The distinction is worth knowing, because a
rotation is the user's work and a false alarm spends it:

- **A value in the MODEL'S CONTEXT is compromised.** Something of it is readable
  there — the mask did not remove every copy, whether it fell short or a rule
  kept a run by design — so it is in the transcript in plain text, replays into
  later requests, and may be in training data. Nothing can undo that: the
  credential has to be rotated, and only the user can do it. The notice says
  `rotate it — … its value is readable in this session's context`, and it means
  it.
- **A value that reached `bash` is not compromised.** A command's `argv`, a
  child's environment, a pipeline, an output pipe: the value was *used*, not
  *read*, and the model never saw it. **Do not ask for a rotation for this** —
  the harness masks it, reports that it happened, and asks only for cleanup.
- **A value in this process's memory is not compromised.** Same reason.
- **A plaintext file on disk is not compromised either** — but it is the one
  case with work in it: delete the copy without reading it (`rm -f`), then
  carry on. Reading it to "check" is exactly what would create the compromise.

So when a notice arrives, read its second half: a `rotate it` head is a real
incident, and the contained wording is the harness saying the event happened,
it was handled, and there is no exposure.

## What you must never do

- **Never echo, print, or log a secret**, in any surface — bash output, an eval
  cell, a notice, a commit message.
- **Never write one into the repository** — not `.env`, not a config file, not a
  scratch note, not a commit. And never leave a plaintext copy outside it
  either: delete it as soon as the command that needed it is done, **without
  reading it**.
- **Never put one in a commit, a PR body, an issue, or a message** to anyone.
  If you think you have, say so immediately. Whether it has to be rotated
  depends on where it went: a value that reached the model's context must be
  rotated by the user, while one that only reached a shell or a file needs the
  copy deleted — unread — and nothing more.
- **Never paste one into a tool argument** that is echoed back, including your
  own reasoning text.
- **Never move a secret between stores** without being asked. Reading one out of
  the long-term store to put it somewhere more convenient defeats the store.

## What is recorded

Every store, retrieval, update, delete and failed authorization is appended to a
hash-chained audit trail with a timestamp, the secret's id (never its value),
the session id, and the calling process's pid and executable path. An audited
reveal is recorded as `reveal` — `tty` when the bytes went to a terminal,
`refused` when nobody could be asked, `cancelled` when the human declined — so
it is never mistaken for an ordinary retrieval, and an identity check that reads
the value (`describe --length`/`--fingerprint`) is recorded as `describe`. A
plain `describe` reads no value and writes no row. `lop secret audit --verify`
checks the chain. So "I just read one to check" is visible to the user
afterwards.

## What this actually protects against — do not overstate it

Be accurate if the user asks, because the honest answer is the useful one.

The store is AES-256-GCM encrypted with blind-indexed names, so **the "scan the
disk for credential files" malware that a bad link actually drops finds
nothing** — no readable `.env`, no greppable `API_KEY=`, not even the names. In
the default mode the master key is a file beside the store, so an attacker who
knows to look for it reads both and decrypts; the opt-in passphrase mode holds
the key only in a running broker's memory, where reading it needs a macOS
authorization prompt the user would see.

A credential the user hands you (an API key for a service, a token for a
script) is a NORMAL thing to `store`, with a good `description` so it is findable
later — and a non-technical user hands one over through the `/credential`
gesture rather than a shell command.

**But anything running as the user that is willing to run `lop` can read these
secrets**, exactly as you do — `lop` is on `PATH` and a script that spawns a
session is a legitimate caller. Ancestry proves lineage, not intent. That is the
fundamental limit and no design confined to one user account removes it.

So: **a large and worthwhile increase in the cost of stealing these credentials,
not a vault.** Never tell the user their secrets are "safe" or "protected"
without that qualification.
