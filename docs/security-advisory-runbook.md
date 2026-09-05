# Security advisory response runbook

Maintainer- and agent-facing procedure for handling a reported vulnerability in
`local-operator`, from the private report to the moment downstream scanners
(`pip-audit`, Dependabot, Snyk, PyPI) actually warn users. The public-facing
summary of this procedure lives in [`SECURITY.md`](../SECURITY.md); this file is
the one you execute from.

**The advisory is not done when it is published.** It is done when the
propagation checks in phase 7 pass, or when the +3d/+14d follow-up and
escalation in phase 8 are recorded. This was the lesson of the September 2026
advisories (GHSA-22mg-8gw7-636x, GHSA-3xjw-9qpc-53mh, fixed in 0.47.5): the
reporter pointed out that a repository advisory alone does not warn anyone
running a vulnerable version in their CI/CD, because those tools read from the
vulnerability databases, not from our repository page. Hours after publishing
(and after requesting CVEs), the GitHub *global* Advisory Database still had 0
entries for us. Propagation is asynchronous and unannounced, so it has to be
verified (phase 7) and chased (phase 8), never assumed from any single action.

Placeholders below: `<GHSA>` is the advisory id (`GHSA-xxxx-xxxx-xxxx`),
`<X>` the fixed version, `<vulnerable>` any version in the affected range, and
`<owner>/<repo>` is `damianvtran/local-operator`.

## 0. Ground rules

- Work the advisory through the **private** GitHub Security Advisory the reporter
  opened (or open one yourself if the report arrived by email). Do not discuss
  details in public issues, PR titles, or commit messages until publication.
- Never reproduce against real user data. Every reproduction and every
  before/after capture runs under an isolated `HOME` **and**
  `LOCAL_OPERATOR_CONFIG_DIR` with disposable sentinel files — see
  "Isolating a run" in [`AGENTS.md`](../AGENTS.md) for why the config dir alone
  is not enough.
- Redact secrets, tokens, and real paths from everything that lands on the
  advisory, the PR, or the release notes.
- Use `~/` in anything committed; never an absolute home path.

## 1. Triage and acknowledge (within 48 hours)

1. Read the report and confirm the affected surface, the attacker model, and the
   affected version range. Check whether `main` and the latest PyPI release are
   both affected (`pip index versions local-operator`).
2. Reply on the private advisory within 48 hours, even if only to say the report
   is received and being reproduced. A reporter who hears nothing for a week
   reasonably assumes the project is unmaintained and may disclose publicly.
3. Decide severity (CVSS 3.1/4.0) provisionally. Prefer the reporter's vector
   unless you have a stated, written reason to differ — you will put the reason
   on the advisory.
4. Read this runbook to the end before starting the fix. The release and
   publication ordering in phases 4 and 5 is not obvious and getting it wrong
   cannot be undone (an advisory published before the fix is on PyPI produces
   Dependabot alerts with no safe version to upgrade to).

## 2. Private fix branch and reproduction

1. Branch from `origin/main` in a fresh worktree — never from a dirty checkout:

   ```sh
   git -C ~/local-operator fetch origin
   git -C ~/local-operator worktree add ../local-operator-<topic> -b dev-<topic> origin/main
   ```

   For a critical issue, use the advisory's temporary private fork
   (`Start a temporary private fork` on the advisory page) so the diff is not
   visible before publication.

2. **Reproduce before fixing.** Build a minimal reproduction under an isolated
   environment and capture the actual output:

   ```sh
   ISO=$(mktemp -d)
   mkdir -p "$ISO/.local-operator"
   echo SENTINEL-$(date +%s) > "$ISO/sentinel.txt"   # the thing the exploit should NOT reach
   env HOME="$ISO" LOCAL_OPERATOR_CONFIG_DIR="$ISO/.local-operator" <reproduction command>
   ```

   Record the command and its response verbatim. This capture becomes the
   "before" row of the advisory's Verified-fix table, so it has to be real.
   Delete `$ISO` when done.

3. Write the regression test first so the fix has something to turn green, and
   keep it in the normal unit suite (do not gate it behind a marker that CI
   skips).

## 3. Fix PR with the review and QA gate

The fix PR is a normal change and goes through the full gate in
[`AGENTS.md`](../AGENTS.md) and the Minerva software development skill: agent
review round(s) recorded on the PR, QA evidence showing the reproduction failing
before and passing after, and design review only if a rendered surface changed.
Security fixes do **not** get a "hotfix" exemption from review — a rushed
security patch that breaks the CLI is a second incident.

- Change class is normally C2 (bounded security patch). It is C6 only if the
  issue is being actively exploited; even then the round is posted
  retrospectively, not skipped.
- Put the before/after reproduction (commands and actual responses) on the PR.
  The same evidence is reused on the advisory in phase 5, so capture it once,
  properly.
- Keep the PR title and description free of exploit detail until publication if
  the fix lands in the public repository rather than the private fork. "Harden
  path validation in X" is fine; the payload is not.
- The PR body carries the `Release: patch — <one-line user impact>` line
  under its summary (see "What the release owner does" in `AGENTS.md`); the
  release owner writes the release notes from it, and a security fix whose
  impact line is missing is the one most likely to be described wrongly.
  Keep the impact line free of exploit detail; "fixes a path-traversal in X"
  is enough. Never touch `pyproject.toml` in the fix PR.
- If the fix changes an API contract the desktop app relies on, the companion
  repository (`local-operator-ui`) needs a matching change and its own release
  (phase 4, step 6). Enumerate callers rather than recalling them.

## 4. Release

Order matters here. The **fix must be installable from PyPI before the advisory
is published**, because Dependabot and `pip-audit` will otherwise tell users
they are vulnerable with no version to move to.

1. Merge the fix PR to `main` once the gate is clean.
2. **Get the fix into a release window; never bump `pyproject.toml` yourself.**
   Releases here are combined and cut by one release owner per window, and the
   lock on a window is an open `chore(release):` PR — see "One release owner
   per window" in `AGENTS.md` for the mechanics, which this runbook does not
   repeat. What the advisory handler does:
   - Look for the lock:
     `gh pr list --search '"chore(release)" in:title' --state open`.
   - If a window is owned, `send` the owner (the pid named in that PR's body)
     the fix PR number, its merge SHA and its `Release:` line, and wait for
     that window's tag and PyPI upload. Tell the owner it is a security fix so
     the release notes link the GHSA(s) (phase 5 publishes their ids).
   - If nothing is open, take the lock per that section (claim PR on
     `release-next`) and cut the window yourself; a security fix on its own is
     a patch bump.

   The fixed version `<X>` is **whatever that window's version turns out to
   be**, not a number chosen in advance; `patched_versions` in phase 5 is
   filled in from the tag, and the advisory is published only after that
   version is on PyPI.
3. The window's **GitHub Release** `v<X>` must link the GHSA(s) and name the
   fixed vulnerability class in one line each; the release is what most users
   read. Publishing the release triggers `.github/workflows/publish.yml`
   (`release: published`), which uploads to PyPI.
4. Confirm PyPI has it before going further:

   ```sh
   curl -s https://pypi.org/pypi/local-operator/json | jq -r '.info.version'
   pip index versions local-operator | head -1
   ```

5. Run `lop-update` so the stable local runtime is on the fixed version, and
   smoke test `lop` from outside the repository (see "Releasing the stable
   `lop` runtime" in `AGENTS.md`).
6. If the API contract changed, cut the companion `local-operator-ui` release
   now, so desktop users are not left on an incompatible client.

## 5. Advisory publication — field checklist

Fill every field on the advisory **before** publishing. Missing fields are why
an advisory is rejected or de-prioritised in curation, and an advisory with no
`patched_versions` is what makes Dependabot tell users there is no fix.

Draft in the UI or via the API; either way, check against this list:

| Field | What goes in it | Why |
| --- | --- | --- |
| `summary` | One line: what an attacker can do, in which component | This is the title in every downstream database |
| `description` | Impact, affected surface, then a **"Verified fix"** section: fixed version, fixing PR, merge SHA, and a before/after table with the actual command output from phase 2 | Curators and users need to see the fix is real, not asserted |
| `vulnerabilities[]` | `{"package":{"ecosystem":"pip","name":"local-operator"},"vulnerable_version_range":"< X","patched_versions":"X"}` | The range and patched version are what `pip-audit`/Dependabot match against. Ecosystem is `pip`, not `PyPI`, on this API |
| `cwe_ids` | e.g. `["CWE-22"]` | Required for a reviewed global entry |
| `cvss_vector_string` | The reporter's vector, unless you documented a reason to differ | Consistency with the public write-up; unexplained downgrades look like minimisation |
| `credits` | The reporter, with type `reporter` | Credit is part of responsible disclosure and is what makes the next reporter come to us privately |
| `state` | `published` — set last | There is no separate publish endpoint; publication is a PATCH of `state` |

The API shape catches people out: `vulnerable_version_range` and
`patched_versions` are **not** accepted as top-level keys (the PATCH is rejected).
They live inside each element of the `vulnerabilities` array. Severity is
derived from the CVSS vector, so set the vector rather than a separate severity
label.

```sh
# Set the affected package/range/fix, CWE, CVSS and credit in one PATCH.
gh api -X PATCH "repos/<owner>/<repo>/security-advisories/<GHSA>" \
  --input - <<'EOF'
{
  "summary": "<one line>",
  "description": "<impact>\n\n## Verified fix\n\nFixed in <X> (PR #<n>, merge <sha>).\n\n| Case | Before (<vulnerable>) | After (<X>) |\n| --- | --- | --- |\n| <exploit command> | <actual response> | <actual response> |",
  "cvss_vector_string": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
  "cwe_ids": ["CWE-22"],
  "vulnerabilities": [
    {
      "package": {"ecosystem": "pip", "name": "local-operator"},
      "vulnerable_version_range": "< <X>",
      "patched_versions": "<X>"
    }
  ],
  "credits": [{"login": "<reporter>", "type": "reporter"}]
}
EOF

# Read it back and check every field above is populated before publishing.
gh api "repos/<owner>/<repo>/security-advisories/<GHSA>" \
  | jq '{state, summary, severity, cvss: .cvss.vector_string, cvss_score: .cvss.score, cwe_ids, vulnerabilities, credits}'
# The GET shape differs from the PATCH shape: the vector is read back at
# `.cvss.vector_string` (there is no top-level `cvss_vector_string` on a GET,
# so selecting it prints null and looks like the field was never set).

# Publish. Only after PyPI has <X> (phase 4 step 4).
gh api -X PATCH "repos/<owner>/<repo>/security-advisories/<GHSA>" -f state=published
```

Credits are shown publicly only once the credited user accepts them, so ask
the reporter to accept in the thank-you message in phase 9.

## 6. CVE request

Request a CVE for every published advisory, immediately after publishing. It
is free, it gives scanners and users a stable identifier that outlives the
GHSA, and reporters ask for it. GitHub reviews published repository advisories
for the Advisory Database on its own schedule (their docs say usually within
72 hours, not guaranteed); the CVE request is not what triggers that review,
and neither action is evidence of propagation — only the phase 7 checks are.

```sh
gh api -X POST "repos/<owner>/<repo>/security-advisories/<GHSA>/cve"
# Expected: HTTP 202 Accepted, empty body. The `cve_id` field on the advisory
# stays null until GitHub's CNA assigns one; that can take days.
gh api "repos/<owner>/<repo>/security-advisories/<GHSA>" | jq '{state, cve_id}'
```

Record the timestamp of the 202 on the advisory thread or in the tracking
conversation; the +3d/+14d follow-ups in phase 8 are counted from it.

## 7. Propagation verification

Each hop below is a distinct system with its own ingestion delay, and a
downstream tool only warns once its *own* source has the entry. Check every hop
rather than the first one that turns green. Expected outputs are what "done"
looks like; anything else means keep waiting (phase 8).

```sh
# 7a. GitHub Advisory Database (global, reviewed). Both must succeed:
gh api "advisories?ghsa_id=<GHSA>" | jq 'length'       # expected: 1 (0 = not yet curated)
curl -s -o /dev/null -w '%{http_code}\n' "https://github.com/advisories/<GHSA>"   # expected: 200 (404 = not yet)

# 7b. OSV (osv.dev) — what pip-audit -s osv and PyPI read:
curl -s -X POST https://api.osv.dev/v1/query \
  -d '{"package":{"name":"local-operator","ecosystem":"PyPI"},"version":"<vulnerable>"}' \
  | jq '[.vulns[]?.id, .vulns[]?.aliases[]?]'          # expected: contains "<GHSA>" (and the CVE once assigned)

# 7c. PyPI per-release vulnerability data:
curl -s "https://pypi.org/pypi/local-operator/<vulnerable>/json" \
  | jq '.vulnerabilities[] | {id, aliases, fixed_in}'   # expected: an entry whose aliases include <GHSA>, fixed_in ["<X>"]

# 7d. pip-audit, both sources, against the vulnerable version only (no deps):
TMP=$(mktemp -d); echo "local-operator==<vulnerable>" > "$TMP/req.txt"
pipx run pip-audit --no-deps -s pypi -r "$TMP/req.txt"  # expected: 1 known vulnerability, Fix Versions <X>
pipx run pip-audit --no-deps -s osv  -r "$TMP/req.txt"  # expected: same
rm -rf "$TMP"
```

- **Dependabot**: check a repository that pins a vulnerable version (a throwaway
  private repo with `requirements.txt` containing `local-operator==<vulnerable>`
  is enough). An alert appears once 7a shows a reviewed entry; Dependabot reads
  the GitHub Advisory Database directly.
- **Snyk**: search `https://security.snyk.io/package/pip/local-operator`. Snyk
  curates on its own schedule; if the advisory is still missing at +14d, submit
  it directly (phase 8).

Record the outputs (or the "not yet" results) on the advisory thread or the
release PR so the next person can see how far propagation had got.

## 8. Follow-ups at +3 days and +14 days, and escalation

Propagation is asynchronous and nobody is notified when it completes, so the
follow-ups have to be scheduled explicitly. When an agent handles the advisory,
it records a wake/reminder for each checkpoint; a human puts them in a calendar.
Do not consider the advisory closed until one of these outcomes is recorded.

- **+3 days**: rerun phase 7. Typical outcome: 7a is green (curated), 7b–7d are
  filling in. If `cve_id` is still null and 7a is still 0, check the advisory
  for a curation comment from GitHub asking for changes (they can request a
  narrower range or a clearer description) and answer it.
- **+14 days**: rerun phase 7. Everything should be green. If it is not:

  1. **PYSEC submission** — open a PR to
     <https://github.com/pypa/advisory-database>, which OSV, PyPI and
     `pip-audit -s pypi` ingest directly. PyPI has **no** direct submission
     path; this is the only way to reach it other than GitHub curation. Add
     `vulns/local-operator/PYSEC-0000-<anything>.yaml` in the
     [OSV schema](https://ossf.github.io/osv-schema/) — the maintainers assign
     the real PYSEC id on merge — with `aliases: [<GHSA>]` so the entry merges
     with the GitHub one rather than duplicating it, `affected[].package`
     `{ecosystem: PyPI, name: local-operator}`, `affected[].ranges` of type
     `ECOSYSTEM` with `introduced: "0"` and `fixed: "<X>"`, `summary`,
     `details`, and a `references` list pointing at the GHSA, the fixing PR and
     the release. Validate before pushing:

     ```sh
     pipx run check-jsonschema \
       --schemafile https://raw.githubusercontent.com/ossf/osv-schema/main/validation/schema.json \
       vulns/local-operator/PYSEC-0000-<anything>.yaml
     ```

  2. **Snyk** — submit through <https://snyk.io/vulnerability-disclosure/>, or
     ask them to ingest the GHSA by id. Include the GHSA link, the affected
     range and the fixed version.
  3. **GitHub curation** — if 7a is still 0 at +14d, comment on the advisory
     and contact GitHub Support referencing the GHSA and the CVE request date.

  Record which escalations were filed, with links, on the advisory thread.

## 9. Close-out

An advisory is closed when all of the following are true and recorded:

- Phase 7 is fully green, **or** phase 8 escalations are filed with links and a
  further follow-up is scheduled.
- The reporter has been thanked on the advisory, told the fixed version and
  the CVE id (once assigned), and asked to accept the credit.
- The GitHub Release notes and `SECURITY.md` "Past advisories" table list the
  GHSA, publication date, fixed version, and reporter.
- The temporary worktree(s), isolated `HOME` directories and any throwaway
  Dependabot test repository are deleted.
- This runbook is updated with anything that surprised you. The September 2026
  lesson — publishing was not the end; hours later nothing had propagated, so
  the phase 7 checks and phase 8 follow-ups were added — landed exactly this
  way, and the next gap will be found the same way.

## Appendix: why the inbound scanners in this repository are not enough

`.github/workflows/ci.yml` runs `pypa/gh-action-pip-audit` on *our*
dependencies, and `.github/dependabot.yml` keeps our pins and actions current.
Both are **inbound**: they tell us when something we depend on is vulnerable.
Neither publishes anything about `local-operator` itself, so a green `pip-audit`
job here says nothing about whether our users' scanners know about our
advisory. Outbound propagation is phases 5–8 of this runbook, and it is manual.
