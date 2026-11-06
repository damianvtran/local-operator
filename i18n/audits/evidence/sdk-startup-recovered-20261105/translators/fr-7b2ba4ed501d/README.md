# French `wire.errors.sdk_startup` — staged translation packet

4/4 keys translated at the immutable core commit `0e7cde33ca8008d561e74f4aed774aa5f76ff7d5`.
Staged only: not audited, not integrated, not activated. No source code, ledger,
receipt, generation, baseline, PR or commit was changed.

## Values

| Key | French |
|---|---|
| `attach_hosting_unconfigured` | aucun runtime actif ne sert {resume}, et cette racine ne peut en démarrer un : {error} Configurez la plateforme d’hébergement de cette racine (son config.yml), puis démarrez l’identifiant avec spawn_session()/deliver() et rattachez-vous dès qu’un runtime existe. |
| `attach_runtime_missing` | aucun runtime actif ne sert {resume} ; ouvrez-le avec mode='own' pour l’exécuter dans ce processus, ou démarrez-le d’abord avec spawn_session()/deliver(), puis rattachez-vous dès qu’un runtime existe. |
| `profile_attach_failed` | Impossible de rattacher le profil {profile} |
| `tool_inventory_mismatch` | spec.tools {spec_tools} et ApprovalPolicy.declared({policy_tools}) désignent des inventaires différents ; ne déclarez la portée de l’exécution qu’une seule fois |

(The table shows rendered text; the catalogue file stores ICU-escaped `mode=''own''`.)

## Locale behaviour — do not confuse with `wire.errors.sdk_spec`

The 17/9 SDK-spec guards each request English explicitly. **These four do not.**
`sdk.py::startup_refusal_text` reads the already-scoped configuration manager,
resolves the language, and renders through `messages.render(..., locale=locale)`.
English here is the default-fallback outcome, so a carried French catalogue with a
resolved `fr` locale is the intended path. Nothing in this packet asserts that the
four refusals are English-pinned, and nothing claims they were executed.

`attach_hosting_unconfigured` embeds `{error}` = `str(HostingNotConfiguredError)`,
whose producer sentence is English. It is carried as opaque data; no producer,
exception-string, attribute or API migration is proposed by this packet.

## Actual validation

`validation.log` is the passing run: exit 0, 0.55 s real, 256 MB heap cap,
`cd /Users/damian/local-operator-ui && /usr/bin/time -p env NODE_OPTIONS=--max-old-space-size=256 node validate.mjs`.
It checks exact key parity/order, argument names and types, technical literals,
all four messages formatted through `IntlMessageFormat`/`fr` with hostile opaque
parameters (braces, percent signs, ICU-looking syntax) preserved verbatim, no
English duplicates, and no bidi controls. Four negative controls are rejected:
missing key, renamed argument, English duplicate, translated technical literal.
Two earlier failures were validator bugs, corrected before the recorded run; the
French target values were never altered by them.

## Provenance

`provenance.json`: agent `i18n-translator`, session `7b2ba4ed501d`. The persisted
usage for the translation call exposes only the opaque route `provider=radient,
model_id=auto`; the underlying model is undisclosed and **no GPT/OpenAI fallback is
claimed**. Two `active_model_route` events in `fallback_events_in_session` belong to
earlier turns of this session under a different primary route and are not attributed
to this packet.

## Gaps

French file absent at the pinned commit: 4 missing keys. Staged packet: 0 missing.
The repository gap stays 4 until integration. `es` and `zh-CN` are also absent;
`hi`, `ru`, `ur` and `vi` carry 4/4 distinct values. That census is a file/key count,
not a semantic audit of those locales.

Independent auditors are required and have not run. Pseudo-localization, RTL,
rendered frames, real SDK guard execution and full validator/build/scanner were NOT RUN.
