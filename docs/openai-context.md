# OpenAI context windows

ChatGPT OAuth and API-key OpenAI calls can have different context limits even
when they use the same model id. Local Operator reads the selected ChatGPT
account's Codex catalogue and keeps three values distinct:

- **Provider default:** the catalogue's `context_window`.
- **Supported maximum:** its optional `max_context_window`, the ceiling for a
  local context override. It is not a request flag.
- **Active window:** the limit used by the session, composer, and child-job usage
  displays. The maximum is selected by default when the account states one;
  otherwise the positive provider default remains active.

The model picker labels different values as `872k max · provider default 272k`.
Narrow terminals retain `872k max` rather than dropping capacity with prices.
Equal values collapse to one number. Percentages remain honest above 100%.

To use the provider default instead, turn off **Use maximum OpenAI context** in
`/settings`, or set:

```yaml
providers:
  openai:
    use_max_context_window: false
```

Only an explicit boolean `false` opts out. This takes effect on the next request
without changing model selection, sampling, or the public API route. The picker
then identifies the provider default as active. Actual dispatch resolves against
the selected credential, including same-model account rotation, rather than the
newest credential stored on disk. Metadata caches isolate account identities
without storing bearer tokens in their keys. Old catalogue captures are refreshed;
an unavailable account catalogue does not substitute the public API's ceiling.

## Compaction is independent

No personal setting or application-wide compaction default is changed. The
existing threshold rule still uses the smaller of the percentage and absolute
triggers. With an active 872,000-token window and explicit settings of `0.8` and
`400000`, the trigger remains `min(697600, 400000) = 400000`. For example, 300,000
input tokens display as `34.4%/872k`, not `110.3%/272k`. Child-job usage uses the
same active limit.

## Boundaries

API-key public limits remain separate from ChatGPT OAuth metadata. Catalogue
values can change by account or model; a published maximum is metadata, not
proof that a particular large request was accepted. Provider entitlement,
request size, output reservations, or other provider constraints can still
reject a request. This change neither sends a special maximum-context wire
parameter nor performs a huge paid generation to test the catalogue.

Pricing is unchanged. Long-context requests may have provider-specific
surcharges or allowance multipliers; a larger supported context is not a claim
that those requests cost the same as shorter ones. The existing pricing engine
is outside this change's scope.
