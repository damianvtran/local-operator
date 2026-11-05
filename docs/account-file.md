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
- Contains no token, key, email, organization name or raw provider id.
- Rewritten after login, logout, a credential being disabled, and a change of the
  configured default model (`hosting`), and at every `lop` start when missing or
  stale. A host should read it at terminal launch and again when the file changes.
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
    "anthropic": {"account_ref": "0123456789abcdef", "kind": "oauth", "org_type": null},
    "openai": {"account_ref": "fedcba9876543210", "kind": "api_key", "org_type": null}
  }
}
```

- `account_ref`: the same opaque 16-hex id stamped on every call as
  `Usage.account_ref` and recorded in `calls.account_ref` (derivation in
  `docs/USAGE_ANALYTICS_DESIGN.md` §1.1). Stable across re-logins of one account.
- `kind`: `"oauth"` (a subscription login) or `"api_key"` (a stored key).
- `org_type`: a short lowercase plan slug when lop already holds one from the
  login exchange, else `null`. Today only OpenAI/ChatGPT logins carry one
  (`chatgpt_plan_type`, e.g. `plus`, `pro`). Anthropic logins store only the
  organization's name, and lop makes no profile call to learn the type, so
  Anthropic is `null`. `null` means the plan is unknown.
- `default`: the account lop uses for the configured default model's provider
  (`hosting` in `config.yml`, mapped to its credential storage id). `null` when no
  default is configured or that provider has no enabled stored credential. Agent
  and per-request model overrides are not reflected.
- `providers`: one entry per provider with an enabled stored credential. With
  several accounts for a provider the entry is the first lop's cascade would reach:
  the lowest-id enabled OAuth row, else the lowest-id enabled API-key row. Which
  account served an individual call is what `Usage.account_ref` records.
- Environment-variable credentials have no stored row and are not listed.

## Host configuration

Tower: `UsageAccount{File: "account.json", IDKey: "default.account_ref", KindKey: "default.kind"}`.
