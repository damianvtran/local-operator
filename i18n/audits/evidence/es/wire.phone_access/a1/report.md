# i18n audit a1 — es / wire.phone_access (branch fix/tunnel-remedies-auto-up @ 1fce7ecb1f5f)

Head audited: 1fce7ecb1f5fb587e7b91ef2f9c7fe658521e0d0 (worktree .worktrees/tunnel-remedies-auto-up, branch fix/tunnel-remedies-auto-up). Date 2026-11-14.
Auditor: i18n-auditor, independent `task` subagent on node lop-remote-node-2, session 110287e9c4f3; not the translator (7af08d89b259) nor the English-drafting coder (b0300e2c2c7f); did not edit any value; model anthropic/claude-opus-5-5 (harness system prompt of session 110287e9c4f3 (`Model: anthropic/claude-opus-5-5`); no provider fallback disclosed during the audit).
Translator: i18n-translator subagent, Local Operator session 7af08d89b259 (values in commit 1fce7ecb1f5f; English source drafted by coder session b0300e2c2c7f); model anthropic/claude-opus-5-5 (as stated by the coordinator brief; not independently verified); 2026-11-14.
Source: en sha256 2979b226db50db768fcad68f5f7376854f2185b08e6b0af08eb8e72bdb470f58 (4 keys); target sha256 f3a749cfd1ac561794389efaf229eb2b4f3dbf828bb11f05645c4ea855bafcae. Prior rounds on disk: none.

**Verdict: passed** — 0 unresolved blocker/major; 2 minor/nit finding(s).

Scope: the whole namespace at the new English source hash. The 4 keys new in 1fce7ecb1f5f were judged in full; the namespace is new; every key is new.

Read: docs/I18N.md 'Add or extend a locale', i18n/glossary.md (no row for phone access/tunnel/billing/credit), i18n/style-guides/es.md, en + context sidecar, corpus precedents (mobile.setup.balance/consent/billing_activated, tunnel.status.server_unavailable.*).

## Per-key judgement (new keys)

| key | en | es | judgement |
|---|---|---|---|
| `wire.phone_access.title` | Phone access | Acceso telefónico | OK — matches corpus «acceso telefónico» (mobile.setup.title_*); see ES-03 nit. |
| `wire.phone_access.description` | How this computer keeps its Radient tunnel for the phone app working. Changes apply to the next sign-in. | Cómo mantiene este equipo en funcionamiento su túnel de Radient para la aplicación del teléfono. Los cambios se aplican en el próximo inicio de sesión. | OK on meaning; Radient verbatim. ES-01 minor («aplicación del teléfono»). |
| `wire.phone_access.auto_setup.label` | Reconnect phone access after signing in | Reconectar el acceso telefónico tras iniciar sesión | OK — infinitive per es guide L13 (action/toggle labels); answers translator question: endorsed. |
| `wire.phone_access.auto_setup.help` | When you sign in to Radient on a computer that already has phone access, reconnect or repair its tunnel in the background. It never activates billing or spends credit; any price or missing credit is shown to you instead. | Cuando inicias sesión en Radient en un equipo que ya tiene acceso telefónico, su túnel se reconecta o se repara en segundo plano. Nunca activa la facturación ni gasta crédito; en su lugar, se te muestra cualquier precio o la falta de crédito. | OK — «Nunca activa la facturación ni gasta crédito» intact; price/missing credit shown instead. |

## Safety meanings

No terminal command is named in any new key (only `/mobile enable`, verbatim, in mobile.setup.login_offer). Money clauses survive where the English has them (nothing charged; setup never purchases credit; never activates billing or spends credit; price shown before billing) — regex presence logged per key in audit.log, meaning judged above. {url} and the console URL are byte-identical to en.

## Findings

| ID | severity | keys | issue | suggested fix |
|---|---|---|---|---|
| ES-01 | minor | `wire.phone_access.description` | «la aplicación del teléfono» can read as the built-in Phone (dialer) app, which is literally named «Teléfono» on iOS/Android in Spanish (answers translator open question). | «la aplicación móvil» (pan-regional; the es guide's region warning is about «móvil» as the noun for the handset) or «la app para el teléfono». |
| ES-03 | nit | `wire.phone_access.title`, `wire.phone_access.description` | Corpus-level, not introduced here: «acceso telefónico» is also the established es term for dial-up access, and «equipo» for computer collides with the es guide's «equipo (team)»; the corpus is mixed (máquina 16 / equipo 13 / ordenador 6 for 'computer'). New keys follow the already-audited tunnel.status/mobile.setup precedent, so no change is requested in this PR. | Record a glossary/style-guide row for 'phone access' and 'computer' in a later wave. |

Minor/nit findings are advisory and do not block; the coordinator may batch them into a translator revision.

## Not verified

No native-speaker review, no rendered terminal/desktop/phone frame. Judgement by reasoning only; no MT service used. Translator model identity is as stated by the coordinator, not independently verified.
