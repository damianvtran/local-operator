---
name: credentials
description: Pass secrets to remote tools without leaking them — why credential flags echo their values, the Authorization-header recipe, probing a container's wget, what the harness scrubs.
---

# Credentials: getting a secret into a remote tool without leaking it

Read this before running an authenticated HTTP call inside a container or over
`kubectl exec`/`ssh`, and whenever you see `«REDACTED:flag=...»` in tool output.

## The failure this guide exists for

An agent ran, inside a pod:

```sh
kubectl exec "$POD" -- sh -c \
  'wget -q -O - --user="$OPENSEARCH_USERNAME" --password="$OPENSEARCH_PASSWORD" \
     "$OPENSEARCH_URL/_cat/thread_pool/write?v"' 2>&1 | head -15
```

Every secret-hygiene rule was followed. The value was an env var *inside* the
pod, never interpolated by the calling shell, never read or printed by the
agent, and correctly quoted. It leaked anyway:

```
exit code: 0
--- stdout ---
wget: unrecognized option: password=<the actual password>
BusyBox v1.37.0 (2025-11-21 22:40:56 UTC) multi-call binary.
```

Three things make this worth a guide rather than a rule:

1. **The tool printed the secret, not the agent.** The image ships BusyBox
   wget, which does not implement `--user`/`--password`. BusyBox's
   `getopt_long` failure path echoes the rejected option *with its value*. GNU
   wget accepts the same flags and prints nothing — so the pattern passes
   review and survives a long time before biting.
2. **It is environment-dependent, not code-dependent.** The identical command
   is safe or unsafe depending on which `wget` the container ships. Across the
   stored corpus, six sessions used `wget --password`; three leaked and three
   did not, purely on image contents.
3. **The exit code was 0.** `2>&1 | head` turned a failed command into a
   successful-looking one, so nothing in the result signalled a problem. Any
   check keyed on failure would have missed it.

## The safe pattern

Build the `Authorization` header **inside** the target and pass it as a header
value, never as a credential-typed flag:

```sh
kubectl exec "$POD" -- sh -c '
  AUTH=$(printf "%s:%s" "$OPENSEARCH_USERNAME" "$OPENSEARCH_PASSWORD" | base64 | tr -d "\n")
  wget -q -O - --no-check-certificate \
       --header="Authorization: Basic $AUTH" \
       "$OPENSEARCH_URL/_cluster/health"
'
```

This is safe for structural reasons, not stylistic ones:

- `--header` is supported by **both** BusyBox and GNU wget — it is the
  intersection — so it is never rejected and never echoed.
- Even if it *were* rejected, the echoed value is a base64 blob of
  `user:pass`, not the plaintext password.
- The credential never appears in `argv`, so it is not visible in `ps` output
  on the node.

The same reasoning applies to `curl -H "Authorization: ..."`, and to any tool
where a header or a credentials file is available instead of a flag.

## Probing, when you must use a flag

If a flag really is the only option, find out first whether the binary
implements it. This costs one round trip and removes the
environment-dependence:

```sh
kubectl exec "$POD" -- sh -c 'wget --help 2>&1 | grep -q -- "--password" \
  && echo GNU_WGET || echo BUSYBOX'
```

Check for `curl` too before assuming it exists — slim images frequently ship
neither `curl` nor GNU wget:

```sh
kubectl exec "$POD" -- sh -c 'command -v curl || echo NO_CURL'
```

Prefer `--header` regardless of what the probe says. The probe tells you
whether the unsafe path would have leaked; it does not make it a good path.

## What the harness does for you, and what it does not

- **Value-keyed redaction** (`variables.redact_secret_values`) replaces
  secrets the harness *holds* — credential-store entries, `ask(secret=true)`
  answers — with `[redacted]`. It cannot help when the value lived only inside
  a remote container, because there is nothing to match on.
- **Shape-based scrubbing** (`local_operator/redaction.py`) catches the case
  above: a parser's option-rejection sentence naming a credential-typed flag
  with its value glued on by `=`. The value is replaced with
  `«REDACTED:flag=password»` — which names the flag, because the flag is all
  that was actually observed. The variable name was expanded by a shell inside
  the pod and appears nowhere in the output.
- **A pre-flight note** appears on the bash result when a credential-typed
  flag is passed to `wget` or `curl`. It warns; it never blocks and never
  rewrites your command.

Seeing `«REDACTED:flag=...»` means **the command failed** — the flag was
rejected — and that the value was removed on its way to the transcript. Treat
it as a signal to switch to the header form, and do not trust the exit code if
the output was piped.

The marker means a value was removed from *that* output. It does not mean the
secret is safe elsewhere: assume the value also reached anything else that
command's output was fed into.
