# Redaction policy

**Owner:** Damian Tran (`@damianvtran`) — the operator of this harness.
**Status:** active.
**Version:** 1, 2026-10-22.
**Applies to:** every surface that runs the credential redactor — the loop's
`redact_tool_result` / `redact_tool_arguments` hooks, `VariableStore.redact`,
`VariableStore.redact_with_report`, `VariableStore.redact_arguments`, the bash
pipe filter, background-job peeks, persisted transcripts, and replayed history.

This is the prose definition of what the redactor treats as a secret. The shape
table in `local_operator/redaction_shapes.py` is the implementation, not the
policy; when the two disagree, this document and its tests are the authority,
and the code is wrong.

## 1. Why this document exists

Two operator reports about this redactor over-firing were each answered with a
narrow rule patch, and the class survived both:

| Report | What was asked | What actually landed | Where it died |
| --- | --- | --- | --- |
| 2026-10-16, session `03b9caa1027e` | Do not mask key names or package/crate names | PR **#2504** fixed one rule (`client-inline-password` anchoring on a filename stem) | Its own PR body deferred the `credential-assignment` strong-name class and the `is_registerable_component` amplifier as "pre-existing and orthogonal". Both are the class the operator reported. |
| 2026-10-18, LOH-28, session `aefe6e48b331` | Identifier fields (project key/name, session title/id, agent/team key/name, skill name) must never be masked | PR **#2628** added identifier-span markers at render sites; PR **#2751** released keywords/identifiers from the argument pass and the registration gate | The fix was scoped to *rendered* identifiers and to the argument pass. A raw `key: <slug>` from an unmarked render site, and a store name after `--secret`, still masked. |
| 2026-10-19, i18n audit, commit `ebb70df3a8` | Read the public catalogue key `wire.slash.credential` to audit its translation | The catalogue was withheld at 51/52 and the commit records "no redaction bypass was used" | There was no sanctioned route for redacted-but-public copy, so an auditor's only options were to work around the control or to stop. |

The common cause is not a regex. It is that the policy had **no owner, no prose
definition and no negative-test table**, so every report became an instance fix
and every fix was free to widen again. This document, its owner and
`tests/unit/secrets/test_redaction_policy.py` are the missing amendment path.

## 2. Definitions

### 2.1 A secret

A secret is a **value**, not the name of a value. The following are secrets and
are masked wherever they appear on a model-visible surface:

* credential values: passwords, passphrases, API keys, access/refresh tokens,
  session cookies, private keys and key material;
* the password/userinfo of a connection string (DSN), a bearer/basic
  authorization header, and an issuer-prefixed token (`ghp_`, `glpat-`,
  `sk-`, `xoxb-`, …);
* any value the session was explicitly told is secret: stored credentials,
  broker-delivered values, and values registered by a shape hit;
* a value in a credential position that carries **value evidence** — issuer
  prefix, entropy, a key header, a DSN structure, or a known stored value.

### 2.2 Not a secret

The following are **never** masked and never registered into the session-wide
redaction set, even when they sit next to a credential word:

* **Key names and entry names** — `OPENAI_API_KEY`, `GITHUB_TOKEN`,
  `MINERVA_UI_NPROD_USERNAME`, `wire.slash.credential`, a store entry after
  `--secret` when it has name structure (dotted, hyphenated, or the store's
  caps/underscore spelling). A name is a reference; a reference must stay
  readable or the lookup cannot be written. A single lowercase word after
  `--secret` remains masked (§6).
* **Package, crate and module names** — `minerva-core-engine`,
  `mda-category-sources`, `local-operator`, `cargo test -p …`,
  `pip install …`.
* **Public catalogue keys and copy** — everything under
  `local_operator/i18n/catalogues/**`, `local_operator/i18n/keys/**`,
  `local_operator/mobile/web/src/i18n/keys.gen.ts` and
  `bundled.gen.ts`. These strings are shipped to every user; they are not
  secret by construction.
* **Documentation, help text and release notes** — prose that *describes*
  credentials, including the guard's own source and test corpus. An **actual
  credential pasted into prose** is still a secret: this bullet is about the
  words, not a value that happens to sit in a document.
* **Identifiers and slugs at their render sites** — project keys, session ids,
  agent/team keys, skill names, DNS-1123 object names, file paths, dotted key
  paths. A renderer that knows a field is an identifier wraps it in
  `VariableStore.identifier_span` (LOH-28); a raw `key: <long phrase>` is not
  that route and stays masked (§6).
* **Placeholders and references** — `${TOKEN}`, `$GITHUB_TOKEN`, `<password>`,
  `%s`, `{{ … }}`, `changeme`, `your-token-here`.
* **Explicitly fake test values** — fixtures labelled as fake in their own
  source, and the existing credential-shape corpus.

### 2.3 The boundary

A **name-shaped value in a credential position** is read as a name when it
carries name structure and no value evidence. "Name structure" is DNS-1123 or
store grammar: a dotted/path identifier (`wire.slash.credential`,
`auth-svc.cluster.local`), a multi-segment hyphen identifier with no digits
(`mda-internal-contribution-delivery`), or the store's caps/underscore spelling
(`OPENAI_API_KEY`). A **strong** credential name (`SECRET_KEY`, `DB_PASSWORD`,
`token:`) keeps its mask for a value with no name structure. A weak name
(`key:`) releases only the base readable-phrase class up to 32 characters; a
longer raw phrase is ambiguous and stays masked. Identifier **fields** are
released by the render-site span (`VariableStore.identifier_span`), not by
label strength — that is the LOH-28 contract and the reason a raw
`key: <long phrase>` remains a keep-masking control.

`--secret` is the one flag whose argument is a store **entry name**, not a
value (`secrets/cli.py`'s `NAME[=VAR]` metavar). Its argument is therefore
released when it has name structure: a dotted or hyphenated DNS-1123 name with
no digits, or the store's caps/underscore spelling. A single lowercase word or
a digit-bearing hyphenated token after `--secret` stays masked (§6), and
`--password`, `--token` and `--api-key` bind values and keep masking.

## 3. The rule

1. A **value** is masked when it is a secret by §2.1.
2. A **name** is never masked and never registered, even when a shape rule
   matched it. Registration is the amplifier: one false hit rewrites every later
   occurrence in the session, including the model's own tool arguments.
3. A stream that cannot read the redaction sink must **refuse to emit**, not
   emit unfiltered bytes. `_REDACTION_SEAM_BROKEN` in
   `local_operator/tools/builtin.py` is that control and this policy does not
   weaken it for any route.
4. The sanctioned public-copy route (§4) is the only place a result may reach
   the model without the normal result pass. It is keyed to the tool identity
   `read_public_copy`, never to an argument a caller supplies.

## 4. Sanctioned read path for redacted-but-public copy

`read_public_copy` (in `local_operator/tools/public_copy.py`) is the deliberate,
narrow route for §2.2 public copy that the shape pass would otherwise mask. It
is **not** a redaction toggle:

* It opens only these roots: `local_operator/i18n/catalogues/*/*.json`,
  `local_operator/i18n/keys/*.py`,
  `local_operator/mobile/web/src/i18n/keys.gen.ts`,
  `local_operator/mobile/web/src/i18n/bundled.gen.ts`. There is no directory
  rule and no extension rule; adding a root is an amendment under §5.
* It serves **the committed blob only**: the file must be tracked and its
  working tree clean against `HEAD` (`git ls-files --error-unmatch` plus
  `git diff --quiet HEAD --`). A working-tree difference or an untracked file is
  refused, and the bytes returned are `HEAD`'s blob — so a clean CRLF checkout
  still reads correctly. Commit the public copy before auditing it; that keeps
  the route from becoming a two-call bypass (write a secret into a catalogue,
  read it back). The route requires `git` on `PATH`.
* It returns a named catalogue key, or the lines of a key module / generated
  map that name the key. A whole catalogue is returned only when it is small
  and no key was given.
* Every successful read appends a JSON line to
  `<config dir>/audit/public-copy-reads.jsonl` **before** the content is
  returned: timestamp, session, path, key, character count, SHA-256 and this
  document. If the record cannot be written, the read is refused.
* Nothing else is exempt. `bash`, `read`, `grep` and every other tool still go
  through the normal redaction pass; a genuine secret read with any of them is
  masked exactly as before.

The exemption is wired in `local_operator/harness/redaction.py`
(`PUBLIC_COPY_TOOLS`) and consumed by `_redact_content` / `_redact_details` in
`local_operator/harness/loop.py`. A unit test asserts the set is exactly
`{"read_public_copy"}`.

**Trust boundary.** The allowlisted roots are public by policy; a secret
committed into a catalogue is a policy violation, not something the reader can
detect (a credential scan there would reproduce the false positive the route
exists to avoid). Review of changes under those roots is the control.

## 5. Amendment

1. Propose the change against this document first: which §2 entry moves, and
   why. A rule change without a policy change is not a policy change.
2. Add a row to `tests/unit/secrets/test_redaction_policy.py` for every new
   non-secret class, and a positive control proving a genuine secret still
   masks on the same surface. Both directions are required; a widening with no
   positive control is rejected.
3. The owner approves. The lopdev manager may draft and run the review, but the
   owner is the only one who can change §2.
4. The test file is the gate: a future widening that breaks a non-secret row
   fails loudly instead of silently blocking an auditor.

## 6. Known residuals

Stated rather than implied away; each is pinned by a test row or a source
comment.

* A raw `key: <long hyphen phrase>` stays masked: it is indistinguishable from
  a passphrase and is not a render-site identifier. The sanctioned route for an
  identifier field is `VariableStore.identifier_span`, which the project tool
  already uses. The one base release that still applies is a value that repeats
  its own name (`_repeats_its_own_name`); that is pre-existing and not widened
  here.
* A single lowercase word after `--secret` still masks in both passes. A
  digit-bearing hyphenated name after `--secret` is released by the **argument**
  pass (the flag is a name carrier and the command must execute) while the
  **result** pass masks it; the exact-value pass covers it once the session
  knows it. Both directions are pinned in the policy test.
* A credential that is genuinely spelled as a DNS-1123 identifier with a
  workload/environment qualifier is released by name-driven rules. It is still
  masked by DSN, bearer, PEM and vendor rules, and by exact value once known.
* `read_public_copy` serves the allowlisted roots **at `HEAD`**. `HEAD` is the
  committed copy, not necessarily a reviewed one: a secret committed into a
  catalogue by a mistaken commit will be returned verbatim, so review of
  changes under those roots is the control. The route refuses working-tree
  changes precisely so an agent cannot create that state silently. Git's
  cleanliness check honours `assume-unchanged`/`skip-worktree`: a file marked
  so can hide a working-tree change, and the route then serves the `HEAD` blob
  without the refusal sentence — no working-tree secret is exposed, because the
  bytes still come from `HEAD`. A hard link is another working-tree name and
  does not change `HEAD`; it is served for the same reason.

## 7. Enforcement map

| Concern | Where |
| --- | --- |
| Shape table (values) | `local_operator/redaction_shapes.py` |
| Store values + registration | `local_operator/variables.py` (`VariableStore.redact`, `redact_with_report`, `redact_arguments`, `_register_shape_hits`) |
| Live stream / seam | `local_operator/tools/builtin.py` (`_stream_redaction_values`, `_REDACTION_SEAM_BROKEN`) |
| Result/argument hooks | `local_operator/harness/loop.py` (`_redact_content`, `_redact_details`) |
| Public-copy route | `local_operator/tools/public_copy.py`, `local_operator/tools/registry.py` |
| Policy tests | `tests/unit/secrets/test_redaction_policy.py` |
