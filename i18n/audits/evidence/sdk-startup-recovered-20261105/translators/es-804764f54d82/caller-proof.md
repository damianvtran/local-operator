# Caller and producer proof — 0e7cde33ca8008d561e74f4aed774aa5f76ff7d5

Derived here from the pinned blobs (not from a moving checkout), then compared
with the retained architect assessment. Independent re-derivation is marked
"translated-by-me" where it was performed; the assessment is named where it is
the source.

## Producer map (all four keys, one call site each)

`git grep -n wire_errors_sdk_startup <rev>` finds producer references only in
`local_operator/sdk.py` (lines 624, 654, 667, 687) plus the generated-key
re-export (`local_operator/i18n/keys/__init__.py:101,213`) and the type-only
relay declaration (`local_operator/mobile/web/src/i18n/keys.gen.ts:1`). There is
no second Python producer and no runtime sink in the relay type file.

| Key | Producer (pinned) | Trigger | Rendered through |
|---|---|---|---|
| attach_hosting_unconfigured | `sdk.py:618-632` `except HostingNotConfiguredError` in `open_session` | attach mode, root has no hosting | `startup_refusal_text(...)` |
| attach_runtime_missing | `sdk.py:648-660` after `await session.dispose()` when `owns_runtime` | attach mode, cold id fell through to an in-process owner | `startup_refusal_text(...)` |
| profile_attach_failed | `sdk.py:663-673` when `attach_agent_profile(spec.profile)` returns falsy | named profile could not attach | `startup_refusal_text(...)` |
| tool_inventory_mismatch | `sdk.py:678-694` when `tuple(inventory) != policy_tools` in `declared` mode | spec.tools and ApprovalPolicy.declared disagree | `startup_refusal_text(...)` |

## The render boundary (this is the part that differs from the 21 sdk_spec keys)

`startup_refusal_text` (pinned `sdk.py:587-598`) reads `language` from the
already-scoped `ConfigManager`, resolves it with `resolve_language`, and calls
`local_operator.i18n.messages.render(message.code, message.params, locale=locale)`.

`render_record` (`local_operator/i18n/messages.py`, pinned) then:

1. resolves the requested locale through `resolve.eligible_locale`, which
   coerces anything unshipped to `en`;
2. looks the message up in that locale's catalogue and falls back to `en`;
3. renders with the SOURCE locale's plural/format rules.

Consequences for this namespace, stated exactly:

- The exception is still a plain one-argument `SessionSpecError(...)`: type,
  `args` arity, and the absence of code/params attributes are unchanged. No
  producer, `str`, `args`, API or runtime change is authorized or made here.
- Unlike the 21 `wire.errors.sdk_spec` producers (which pass `locale="en"`), the
  four startup refusals ALREADY localize per resolved root. Once an `es`
  catalogue for this namespace is committed, audited and `es` is eligible, the
  single message string carried by `str(SessionSpecError)` for an es-resolved
  root becomes Spanish. That is the designed path the ru/vi/ur/hi catalogues
  already exercise; it is not a new contract this change introduces.

## Ship-set caveat discovered while staging (must land together)

`scripts/i18n/generate.py::_shipped_locales` (pinned, lines 440-481) ships a
locale only when EVERY namespace it CARRIES has a `status: passed` ledger row
with a fresh `source_sha256`. Adding
`local_operator/i18n/catalogues/es/wire.errors.sdk_startup.json` makes `es`
carry a new namespace. Committing the file without its passed, fresh ledger row
would therefore remove `es` from `shipped.json` entirely — a locale-wide
English regression, not a localized-diagnostic gain. The catalogue file and its
independent audit row belong to the same landing step. This staging packet
changes neither.

## Exposure

No desktop or mobile SDK sink for these exceptions was found by the architect
assessment (assessment §"Actual interface exposure": desktop session creation
uses `server/routes/desktop_sessions.py`; mobile `spawn_session` is a different
function in `mobile/daemon.py`; mobile's `humanizeError` matches pending-answer
and connection phrases only). That is the assessment's finding, not my runtime
proof; I did not add a UI sink and claim no interface-level localization.

## Embedded English

`attach_hosting_unconfigured` embeds `error=str(error)`, whose real producer is
`session_factory.py` `raise HostingNotConfiguredError("Hosting platform is not
configured.")`. The catalogue keeps `{error}` opaque, so that English cause
prose still appears inside a Spanish sentence. The assessment recommends a
display-specific owned message instead (§3), but that needs a new key in every
locale plus a producer change — outside this staged, catalogue-only packet. It
is recorded as a caveat, not fixed, and not hidden.
