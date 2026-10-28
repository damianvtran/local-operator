# Independent audit — French `wire.errors.files`

| | |
|---|---|
| locale | `fr` |
| namespace (file group) | `wire.errors.files` — 11 keys |
| audit id | `wf26a1` |
| audit date | 2026-10-26 |
| audited revision | `1c23634501a5a65a610f4d428e27aadcd52e5e09` (worktree `i18n-fr-wire-errors-files`; the target catalogue is an untracked working-tree file, so the audited target bytes are bound by hash) |
| English source | `local_operator/i18n/catalogues/en/wire.errors.files.json` — sha256 `ccfeb1ea55e0fed62213846cb669f28075b20e91e361f380456772d433983f64` |
| French target | `local_operator/i18n/catalogues/fr/wire.errors.files.json` — sha256 `9a2da0e48a75c746889a6ff0fec7d1e6530a68b6ccbc408c1062c6786b4ac6d4` |
| context sidecar | none exists for this namespace (`en/wire.errors.files.context.json` absent), so no `maxCells`/`maxChars` budgets apply |
| auditor | `i18n-auditor` (lopdev team), session `2674212bee8a`, independent of the translator session |
| model | `radient/auto` — the concrete upstream model behind the auto route is not surfaced to the session; no pinned model and no fallback claim |
| **verdict** | **passed** — 0 BLOCKER, 0 MAJOR, 1 MINOR, 0 NIT |

## Scope and method

Strings are HTTP error details raised by the local static-file router
(`local_operator/server/routes/static.py`) and by the signed-URL endpoint, so
they are read by the desktop app and by anyone scripting the local API —
developer-facing error copy, short form.

1. Read `docs/I18N.md`, `i18n/glossary.md`, `i18n/style-guides/fr.md`, and the
   shipped `fr` `wire.*` catalogues (`wire.errors.desktop`, `wire.settings`,
   `wire.memory`, `wire.slash`, `wire.incidents`, `wire.notices`,
   `wire.remote_compute`) as terminology controls.
2. Read the code that raises each message (file:line in `audit.log`, section
   `[10]` and the `_wf.sign_*` grep), including `desktop.py::require_desktop`
   for the "desktop bearer" credential and `SignRequest.ttl_s: int` for the TTL
   bound.
3. Ran the repository checks (`audit.log`): `python3 scripts/i18n/check.py
   --catalogues-only` → `check.py: catalogue checks: 0 problem(s)`, exit 0; the
   full `python3 scripts/i18n/check.py` → `check.py: ok — 184 file(s) scanned,
   2258 literal(s) at or below the baseline, catalogues consistent; 37 advisory
   finding(s)`, exit 0. Nothing failed, so nothing here is implicated in an
   unrelated failure.
4. Re-derived every mechanical claim myself with a read-only script
   (`verify_wire_errors_files.py.txt`): ICU argument tuples (name, order, kind)
   via the real parser and formatter, JSON shape, NFC, typography codepoints,
   and an `en`/`fr` render of all 11 strings.
5. Second pass (see below): reverse-read each French string for anything it
   implies that the English does not, and re-checked it against the raising
   condition.

## Findings

### F1 — MINOR — `wire.errors.files.sign_auth`

- **Current**: `Un token d’accès local ou un bearer de bureau est requis.`
- **Problem**: the second credential is named by a bare borrowed noun,
  "un bearer". The meaning is right — the gate really does want
  `Authorization: Bearer <posture token>` (or a valid `X-Local-Token`) — and
  "de bureau" is the shipped `fr` word for the desktop surface
  (`le bureau`, `application de bureau`, `Notifications de bureau`,
  `origines de bureau configurées`). What is missing is the noun: the `fr`
  guide keeps `token` English and the shipped catalogues use `token`/`tokens`
  throughout (15 occurrences in `wire.settings`, 2 in `wire.slash`, 1 in
  `wire.memory`; counted as occurrences, not lines), but never `bearer`, and `bearer` is not on the guide's
  kept-English list. A reader who does not already know the HTTP auth scheme
  has no anchor for "un bearer de bureau", in a message whose whole job is to
  tell them what is missing.
- **Suggested wording**: `Un token d’accès local ou un bearer token de bureau est requis.`
  (keeps the scheme name verbatim, as `zh-CN` does with `桌面bearer令牌`, and
  makes the thing concrete). Freely acceptable alternative: keep the string and
  record `bearer` in the `fr` guide's kept-English list — the present text is
  not wrong, only under-specified.
- **Severity rationale**: comprehension polish in an auth refusal, no meaning,
  polarity, register or terminology *conflict*; does not block.

## Per-key assessment

| key | en | fr | verdict |
|---|---|---|---|
| `audio_type` | File is not an allowed audio type: {mime_type} | Type de fichier audio non autorisé : {mime_type} | correct — nominalisation is standard `fr` UI error form; "non autorisé" keeps the allowlist (policy) sense rather than drifting to "non pris en charge" (unsupported), which would be wrong here |
| `heic_conversion` | Failed to convert HEIC/HEIF file: {error} | Échec de la conversion du fichier HEIC/HEIF : {error} | correct — same shape as shipped `tui.team_authoring.save_error` ("Échec de l’enregistrement : {reason}"); HEIC/HEIF verbatim |
| `html_type` | File is not an allowed HTML type: {mime_type} | Type de fichier HTML non autorisé : {mime_type} | correct |
| `html_utf8` | HTML file is not valid UTF-8 text. | Le fichier HTML n’est pas du texte UTF-8 valide. | correct — partitive after "n’est pas" is idiomatic for content, and UTF-8 stays verbatim; matches the `UnicodeDecodeError` condition (the file failed UTF-8 decoding, not "another encoding") |
| `image_type` | File is not an allowed image type: {mime_type} | Type de fichier image non autorisé : {mime_type} | correct |
| `internal_error` | Internal Server Error: {error} | Erreur interne du serveur : {error} | correct — the standard French form of the HTTP 500 reason phrase; sentence case per guide |
| `sign_auth` | A local access token or desktop bearer is required. | Un token d’accès local ou un bearer de bureau est requis. | **F1 (MINOR)** — meaning exact, wording under-specified |
| `sign_path` | path must not be empty. | path ne doit pas être vide. | correct — `path` is the request-body field name and stays verbatim; lowercase sentence start matches the identifier, as in `sign_ttl` |
| `sign_route` | route must be one of: {routes} | route doit être l’une des valeurs suivantes : {routes} | correct — adds the generic noun "valeurs" the English leaves implicit; no trailing period, matching en; U+202F before the colon |
| `sign_ttl` | ttl_s must be between 1 and {maximum}. | ttl_s doit être compris entre 1 et {maximum}. | correct — and accurate to the code: `ttl_s` is an `int`, so `0 < ttl_s <= 3600` really is 1..3600 inclusive |
| `video_type` | File is not an allowed video type: {mime_type} | Type de fichier vidéo non autorisé : {mime_type} | correct |

## Mechanical results (all clean)

- **Placeholders**: `{mime_type}` ×4, `{error}` ×2, `{routes}` ×1, `{maximum}`
  ×1 — names, order and kinds identical to `en` for all 11 keys (checked with
  `runtime.message_arguments`). All are bare simple arguments; no
  `{x, number}`/date formatting was introduced. That matters here: the caller
  passes `str(TTL_MAX_S)` and a `", ".join(...)` for `{routes}`, so a typed
  formatter would raise at render time. Absent.
- **ICU structure**: no plural or select in this namespace, in `en` or `fr`, so
  the `fr` CLDR category set (`one`/`many`/`other`) is vacuously satisfied; no
  `'{}'` escapes or doubled braces.
- **Markup / code**: the English carries no Rich tags or backtick spans, and the
  French adds none.
- **Verbatim tokens**: `HEIC/HEIF`, `HTML`, `UTF-8`, `path`, `route`, `ttl_s`,
  `token` present in every string that needs them; MIME types arrive only
  through the argument.
- **Typography**: U+202F before the colon in all six strings that contain one;
  no ASCII space and no U+00A0 before `: ; ! ?` anywhere; U+2019 for the three
  apostrophes and no straight `'`; sentence case; no ellipsis or dash misuse.
  No BOM; whole file NFC; no bidi control characters; no non-Latin homoglyphs
  (scan over every character, `audit.log` section `[7]`).
- **JSON**: parses; 2-space indent on every member line; exactly one trailing
  LF; key set and key ORDER identical to `en` (11/11, no additions, no drops).
- **Meaning vs code**: every key's condition re-read at the raise site; no
  omission, addition, negation or polarity error. The three type refusals are
  400s on a MIME allowlist membership test; `heic_conversion` is the 500 from a
  failed HEIC/HEIF→PNG conversion; `html_utf8` is the 400 from a failed
  `read_text(encoding="utf-8")`; `internal_error` is the catch-all 500;
  `sign_auth` is the 401 taken when neither a valid `X-Local-Token` nor a
  passing `require_desktop` is presented; `sign_route`/`sign_path`/`sign_ttl`
  are the 400s on the sign body.
- **Fit**: no `maxCells`/`maxChars` exists for this namespace (no context
  sidecar). Longest French string is 57 characters (`sign_auth`) against 51 in
  English (+12%, under the guide's expected 15–25% growth — these are short
  fragments, not prose), and far below any
  toast or log-line budget this product pins. No overflow risk, and no CJK/wide
  glyph or combining-mark concern in a Latin-script locale.
- **RTL**: not applicable (Latin script).

## Second pass

High-visibility group (user-facing errors), so a second pass was run: every
string was re-read backwards (what does the French claim, and does the English
claim anything else?), then re-checked against the code path. It changed no
verdict and added no finding. Caveat, recorded rather than hidden: the second
pass was done by the same auditor in the same session, so this is a second
*pass*, not an independent second reviewer; no independent-rater agreement is
claimed.

## Notes, not defects

- The `fr` coverage of this namespace is new; nothing here contradicts shipped
  French copy. The only terminology gap found is the missing `bearer` entry
  (`F1`).
- `sign_auth` describes the sign endpoint's own two accept paths. The desktop
  refusal string it wraps (`desktop.py`, hard-coded English, `i18n: ignore`) is
  out of scope for this namespace and was not assessed as copy.

## Limitations

- This is an LLM audit, not a native-speaker review; the record claims no native
  review.
- Repository checks were run on the audited worktree at
  `1c23634501a`; the target catalogue is untracked there, so `check.py` sees it
  as a present `fr` catalogue (it is read from disk, not from git) — which is
  what makes the `--catalogues-only` pass meaningful for these bytes.
- No rendered-UI capture was taken: these strings are JSON error bodies, and no
  surface renders them at a pinned width. Fit is judged from the strings and
  the absence of a budget, not from a screenshot.
