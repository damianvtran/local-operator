# Spanish provider-state author notes

Classification: `audit_notes_not_execution_log`. Author receipt only; status **translated**, never an independent audit or pass. Actual execution is retained separately in `validation.log`.

## Scope and provenance

Five `tui.provider_state` human labels only, authored by i18n-translator, session `d37edc1334b7`. Source revision read: `a54ec02de48a94b68d6c3dc92e1307c97ea776ad`. English and context were read using the approved `read_public_copy` reader; neither was changed. English source author is session `93b7` per the delegation, not this translator.

The session model-switch notice states temporary fallback from `anthropic/claude-opus-5-5` to `openai/gpt-6.1-sol` because of provider failure: quota. This is notice-based provenance, not independent per-response served-model telemetry; none was available. Translation was reasoned here without translation services, APIs, SDKs or external MT.

Read inputs: `docs/I18N.md`, accepted RFC `docs/design/i18n.md` catalogue/format contract, `i18n/style-guides/es.md`, `i18n/glossary.md`, English catalogue and context. Existing Spanish `wire.settings` was read for terminology consistency. No glossary/style edits.

## Native wording choices and ambiguity

- `logged_in` → **sesión iniciada**: affirmative, completed login status; does not tell the user to log in or claim server connectivity. Follows normal Spanish “iniciar sesión” terminology rather than the misleading connectivity adjective “conectado”.
- `env_key` → **clave del entorno**: credential-source label. “Entorno” means the environment-variable source in this context, not a named environment or server. It is human copy, not a translated variable identifier. “Credenciales” remains the glossary's collective term; “clave” is the specific key here.
- `needs_login` → **requiere iniciar sesión**: authentication requirement as a status. Not the action-only “iniciar sesión”, and not a claim of successful login. The longer wording retains the requirement unambiguously.
- `local_ready` → **servidor configurado**: completed configuration, not a health check, verified reachability or login state. The local-provider context supplies “local”; adding it to the label would unnecessarily lengthen the source's equally implicit wording.
- `local_unconfigured` → **configurar servidor**: infinitive instruction in the Spanish style guide's action-label register. Contrasts deliberately with the completed-state participle above.

Lowercase fragments are intentional. Zero placeholders, plural/select arms, markup or technical identifiers occur in these five values. Keys and key order are unchanged.

## Mechanical validation and width limitations

`validator.py.txt` executes the official `scripts.i18n.check.check_catalogues()` with only namespace/locale enumeration scoped to `en`/`es` and this namespace. No eligibility or validator-result bypass. Real catalogue, English context, ICU parser, render API and provenance are exercised. Imports are asserted to originate in the assigned worktree, using the primary repository's `.venv/bin/python`, isolated scratch HOME/config, stripped inherited environment/CMUX variables, and disabled bytecode writes. Git commands used normal HOME.

Actual result: exit 0; official catalogue problems `[]`; five identically ordered keys; zero ICU arguments and plural/select branches; NFC; no manual bidi controls; direct ICU and `messages.render_record` checks with English controls. In-memory missing-key and added-argument counterfactuals are both rejected by the official checker. Pseudo coverage is explicitly a synthetic bracketed/padded expansion through the real ICU renderer, not a shipped pseudo-locale or full pseudo UI run.

Spanish Rich cell lengths, in key order: **15, 17, 23, 20, 19**. No source `maxCells` is declared. Context-only diagnostics use `max(31 - ID cells - 4, 3)` for a 31-cell pane, not a universal 17-cell cap. With illustrative `openrouter` (17 cells available), the first two labels fit; requirement and server labels truncate. With illustrative `ollama` (21 available), both server labels fit while the authentication requirement truncates. English controls fit in both examples. Longer labels retain the explicit requirement and configured/action distinction rather than lose meaning to satisfy an invented cap. These diagnostics identify clipping, not native-layout approval.

Plain Rich text fitting only: no real settings screen, rendered frame, native fonts, app startup, network-isolation certification, production wiring or release proof. Existing provider visuals remain offline/uncertified; the current producer remains English, and this author contribution does not claim to fix it. No shared ledger/history, generated code, baseline, caps, eligibility or consumer changes. No staging, commit or push. Independent A/B auditing is pending and must be performed by other sessions after the root commits the input bytes.

## Glossary proposals (not applied)

- logged in → sesión iniciada
- sign in / log in → iniciar sesión
- login required → requiere iniciar sesión (status, distinct from an action)
- environment key → clave del entorno (human credential-source label)
- configured server → servidor configurado; configure server → configurar servidor

These are namespace proposals for currently unlisted terms, not changes to the shared glossary or style guide.
