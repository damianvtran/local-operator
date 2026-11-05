# `account.json`: the current login, for terminal hosts

A terminal host that wants to attribute a terminal's usage to an account (Tower,
whose usage feed goes to the company's Adopty tracker) reads one small JSON file
from lop's config directory, the same way it reads Claude Code's
`~/.claude.json` (`oauthAccount.organizationUuid` / `organizationType`). It never opens
`auth.db` and never sees a credential.

## Location and guarantees

`<config dir>/account.json`: `~/.local-operator/account.json`, or
`$LOCAL_OPERATOR_CONFIG_DIR/account.json` when that is set.

- Mode `0600`, replaced atomically (same-directory temp file + rename), so a
  reader never sees a torn document.
- Publication is serialized: every rewrite takes an exclusive lock on
  `<config dir>/.account-file.lock` (across processes and threads) and, while
  holding it, reads the current credential rows and config and builds the file. A
  slower or older publisher therefore cannot overwrite a newer logout with a
  snapshot it read earlier; the last one to run always reflects the latest
  committed state.
- Best-effort under contention: a publisher waits at most 5 seconds for the lock
  and otherwise leaves the previous document in place (logged, never an error for
  the login/logout that triggered it) until the next publish.
- Contains no token, key, email, organization name or raw provider id. A ref is
  not a stored identifier in cleartext and cannot be enumerated without already
  knowing the candidate identity; it is not secret against a guess, since a
  64-bit truncated digest of a low-entropy id (an email-shaped `sub`) can be
  confirmed by someone who guesses that id.
- Rewritten after login, logout, a credential being disabled, a credential block
  being set or cleared, and a change of the configured default model (`hosting`),
  and at every `lop` start when missing or stale. A host should read it at terminal launch and again when the file changes.
- Absent until lop has stored a credential or been started in a config directory
  that has one. A missing file means the account is unknown.

## Shape (stable, `version: 1`)

```json
{
  "version": 1,
  "default": {
    "provider": "anthropic",
    "account_ref": "0123456789abcdef",
    "kind": "oauth",
    "org_type": null
  },
  "providers": {
    "anthropic": {
      "account_ref": "0123456789abcdef",
      "kind": "oauth",
      "org_type": null,
      "accounts": [{"account_ref": "0123456789abcdef", "kind": "oauth", "org_type": null}]
    },
    "openai": {
      "account_ref": null,
      "kind": null,
      "org_type": null,
      "accounts": [
        {"account_ref": "fedcba9876543210", "kind": "oauth", "org_type": "plus"},
        {"account_ref": "1111222233334444", "kind": "oauth", "org_type": "pro"}
      ]
    }
  }
}
```

- `account_ref`: the same opaque 16-hex id stamped on every call as
  `Usage.account_ref` and recorded in `calls.account_ref` (derivation in
  `docs/USAGE_ANALYTICS_DESIGN.md` §1.1). Stable across re-logins of one account.
  An OAuth identity ref is the same on every host. A ref for a stored API key, or
  for an OAuth login with no IdP identity, is namespaced to that machine's auth
  store (a random per-store id), so it is equal across hosts only through the
  owner's ref on a mesh grant.
- `kind`: `"oauth"` (a subscription login) or `"api_key"` (a stored key).
- `org_type`: a short lowercase plan slug when lop already holds one from the
  login exchange, else `null`. Today only OpenAI/ChatGPT logins carry one
  (`chatgpt_plan_type`, e.g. `plus`, `pro`). Anthropic logins store only the
  organization's name, and lop makes no profile call to learn the type, so
  Anthropic is `null`. `null` means the plan is unknown.
- `providers`: one entry per provider with at least one enabled stored credential.
  **Determinability rule.** `account_ref`, `kind` and `org_type` of an entry name
  the serving account only when lop's credential cascade can be certain of it,
  and are all `null` otherwise. The cascade tries OAuth rows, then API keys saved
  by `login`, then an environment variable, then any other stored key. The winning
  tier is the first non-empty of OAuth rows, login keys and other stored keys, and
  the fields are set only when that tier holds exactly one enabled row. They are
  `null` when it holds several (the pick is made per session: sticky, usage-aware
  or round-robin, not by row order) and when only a plain stored key exists (an
  environment variable outranks it and this file cannot see one).

  **Blocks.** They are also `null` when the sole winning row has an unexpired
  credential block of any scope (a quota or auth backoff in
  `auth_credential_blocks`, including a model-scoped one, which makes the answer
  depend on the model). The cascade then serves a lower tier until the block
  expires, and nothing is written when it does, so the serving account is
  time-dependent; the file says `null` rather than a ref that turns wrong on
  expiry. Setting or clearing a block republishes. After a block expires the
  entry stays `null` until the next publish (lop start, login/logout, a block
  change, a `hosting` change): an under-claim, never a wrong account.

  **Demotions are not in the file.** A recently failing credential is demoted by
  a per-process, in-memory reordering lasting at most 120 seconds. It is not
  reflected here (it cannot be shared across processes), so a host may see the
  durable winner while one process briefly prefers a sibling.
- `accounts`: every enabled stored row of the provider as
  `{account_ref, kind, org_type}`, in cascade order (OAuth, login keys, other
  keys; then row id). An inventory of accounts lop may use, not a claim about
  which one serves. Rows without a durable ref are left out. Which account served
  an individual call is what `Usage.account_ref` records.
- `default`: `{provider, account_ref, kind, org_type}` (the entry without
  `accounts`) for the configured default model's
  provider (`hosting` in `config.yml`, mapped to its credential storage id), so
  `default.account_ref` is `null` exactly when that provider's account is not
  determinable. `default` itself is `null` when no default is configured or that
  provider has no enabled stored credential. Agent and per-request model overrides
  are not reflected.
- Credentials from an environment variable, a config override or a runtime
  override have no stored row and are not represented; a provider that has one
  may still show a stored account that the environment variable outranks.

## Host configuration

Tower: `UsageAccount{File: "account.json", IDKey: "default.account_ref", KindKey: "default.kind"}`.
