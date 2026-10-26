# Building and admitting an arm (no guest, no model, no spend)

An **arm** is the dedicated interpreter the OSWorld adapter runs in: the harness
wheel, the adapter wheel, the adapter's locked `osworld` extra, and a recorded
verification of what was installed. `scripts/build_osworld_arm.py` builds one
from a committed SHA and `scripts/osworld_dependency_admission.py` decides
whether its dependency set conforms to the reviewed
[dependency policy](dependency-policy.md). Until now that recipe existed only as
operator scratch; this is its durable, tested form.

The two commands are a pair, and the order matters: **plan writes a small
receipt and no arm directory**, so the same `--out` is free for the build.

```sh
# 1. plan: resolve the closure and record the digest. No venv, no wheel, no
#    arm directory -- a plan costs kilobytes, not the ~220 MB a full arm does.
python scripts/build_osworld_arm.py --source-sha <40-hex-main-commit> --plan ~/worktrees/osworld/plans/arm-plan.json

# 2. build: wheels, venv, install, admission, installed-identity verification.
#    --plan binds the build to the resolution that was reviewed.
python scripts/build_osworld_arm.py --source-sha <40-hex-main-commit> \
    --out ~/worktrees/osworld/arms/<name> --execute --plan ~/worktrees/osworld/plans/arm-plan.json
```

`--out` belongs to `--execute` alone: handing one to a plan run is refused rather
than ignored, because a caller who asked for an arm would otherwise believe one
had been prepared. `--plan` is optional for the build, but a build given one
re-resolves and **refuses unless the plan digest still matches** (the digest
covers the installed set and the override policy, so a moved index, a new
upstream release inside a range or a policy edit cannot slip in between the two
commands). A plan is only meaningful on a `3.12` interpreter, which the plan run
checks.

Exit codes: `2` refused (short, unmerged or branch-only SHA, existing arm
directory, volatile output root, plan digest drift, unusable interpreter, failed
command), `4` built but dependency admission or installed-identity verification
refused. Receipts are written in both cases — a refusal without its evidence is
not actionable. Note that the docstring's "merged" means **on `origin/main`'s
first-parent history**: an intermediate commit of a merged branch is an ancestor
of `main` and is refused, because it can be a state review rejected.

## What it proves

For the exact commit named:

- the wheels were built from `git archive <sha>`, never from a working tree, so
  a dirty checkout cannot leak into the artifact;
- the installed closure is the one the receipts describe: every installed
  distribution is either in the pinned plan at the pinned version, or one of the
  two wheels built here (`--execute` refuses otherwise);
- the harness core modules executing in the arm hash identically to the same
  files in the extracted commit (the stable source identity);
- the adapter's installed file set hashes to the `package_digest` the selector
  pins and the worker re-verifies, and to a `source_record_digest` that is
  comparable across builds (see the identity table below);
- dependency admission passed **under the policy** — see below.

## Dependency admission is not a green `uv pip check`

`[tool.uv].override-dependencies` in `benchmarks/osworld_v2_adapter/pyproject.toml`
is authoritative: it overrides two stale upstream bounds instead of downgrading
a security library.

| upstream declaration | reviewed override |
| --- | --- |
| `osworld` requires `requests~=2.31.0` | `requests>=2.32.0` |
| `zhipuai` requires `pyjwt>=2.8.0,<2.9.0` | `pyjwt[crypto]>=2.10,<3` |

A uv override is a resolver rule; it is not written into the installed wheels'
metadata. A correctly built arm therefore still exits `1` from a raw
`uv pip check` with exactly those two findings. The builder keeps that raw text
and rc verbatim (`raw-pip-check.log`, `raw-pip-check.rc`) and admission is a
separate verdict (`policy-admission.json`, `plain_pip_check_green: false`).

Admission refuses, fail-closed:

- a third conflict of any kind (main16 had one: a missing `miniaudio`);
- a missing dependency, which `--no-deps` must never be used to conceal;
- a superseded bound whose upstream specifier changed or disappeared;
- installed metadata that does not actually declare the admitted bound;
- a fabricated or genuinely green report (the two bounds are real metadata, so
  green means the upstream moved and the policy needs re-review);
- unrecognised checker output, or a finding count that disagrees with its lines;
- a committed override list that no longer covers exactly the reviewed pair, or
  an installed version outside it;
- a harness requirement that is absent or unsatisfied.

## The three construction constraints

1. **`venv --copies`, never `uv venv`.** `discovery._symlink_free` refuses a
   launch path whose real path differs from the named one, and `uv venv` writes
   `bin/python3.12` as a symlink at every flag combination: measured
   2026-10-25, every such arm was refused pre-dispatch with
   `adapter launch path has a symlink or lexical alias`. The builder uses the
   stdlib `venv --copies --without-pip` and refuses a venv whose interpreter
   resolves through a symlink.
2. **The dylib copy is required on macOS.** `--copies` copies the interpreter
   but not the shared library it links; the copied binary keeps
   `@rpath/libpython3.12.dylib` with an rpath of `<venv>/lib`, so without
   `libpython3.12.dylib` beside it every invocation dies in dyld. A missing
   dylib to copy is tolerated (an interpreter linking by absolute path needs
   none); an interpreter that cannot start is a refusal.
3. **A wheel archive hash is not an identity — and neither is `package_digest`.**
   Zip member timestamps differ between two builds of the same commit, so the
   `.whl` sha256 is not reproducible (main16's recorded archive mismatch was
   this, not tampering); archive hashes are kept under
   `wheel_sha256_informational_only`. One level down, the same trap applies to
   **`package_digest`**: it hashes the installed wheel RECORD, and RECORD holds
   entries the *installer* writes — `direct_url.json` carries the absolute wheel
   path and `uv_cache.json` a timestamp — so two installs of the same wheel bytes
   give different digests (measured: `9a0728015b14…` and `762103891899…` for one
   wheel). It is exactly right for what the selector pins and the worker
   re-verifies, and it is documented as a **this-install** identity, never as a
   cross-build drift detector. The cross-build identity is the source commit,
   the pinned core module hashes and `source_record_digest` (the same RECORD rows
   with `INSTALLER`, `REQUESTED`, `direct_url.json` and `uv_cache.json`
   removed). `discovery.distribution_digest` keeps its existing semantics
   unchanged: the adapter guard and the recorded arms depend on them.

### Identities, and which question each answers

| value (in `receipts/`) | answers | stable across two builds of one commit? |
| --- | --- | --- |
| `source_commit` | which tree was built | yes, by definition |
| `core_source[*].hash`, `core_module_digest` | which harness code executes | yes |
| `installed_distributions` | the exact installed set | yes for equal lock + index |
| `adapter.package_digest` | what the selector pins for THIS install | **no** (installer-written RECORD rows) |
| `adapter.source_record_digest` | drift between two installs of one commit | yes |
| `wheel_sha256_informational_only` | provenance of the local artifact only | **no** (zip timestamps) |

## What it does NOT prove

- **Not offline.** A build fetches the locked dependency set from PyPI and the
  pinned OSWorld git ref; only the *checks* are local. Nothing here contacts a
  model provider, a guest, or AWS.
- **No guest, no VM, no episode.** Nothing here allocates, boots, or talks to a
  remote desktop, and the adapter's action path is not exercised. A green build
  says nothing about whether a task can be driven.
- **No model call, no scoring, no leaderboard claim.** The arm is an
  environment, not a result.
- **Not a substitute for a paid smoke run.** The first paid episode of a new
  arm is still the first real test of it.
- **Reproducibility of the *set*, not of the bytes.** Two builds of one commit
  produce the same distribution names and versions and the same source/module
  identity; the wheel archives differ, by design (constraint 3).
- **Not a live SDK compatibility proof.** PyJWT is overridden past ZhipuAI's
  declared upper bound on the strength of the offline signing probe in
  [dependency-policy.md](dependency-policy.md), which is a separate check this
  script does not run.

## Receipts

Written under `<out>/receipts/`: `policy-provenance.json` (the exact policy
bytes, hashed from the commit), `requirements.locked.txt` (the frozen adapter
export) and `requirements.dropped.txt` (the harness row the wheel replaces),
`resolution-project/` (the throwaway uv project that re-resolves the closure
under the overrides) with its `uv.lock`, `policy-pins.txt` (what was installed),
`installation-overrides.txt` (the policy narrowed to exact pins for pip),
`interpreter.json` (dylib copy and interpreter identity), `raw-pip-check.log` /
`.rc`, `policy-admission.json`, `installed-verification.json`, `plan-binding.json` (when a plan was given) and
`build-record.json`. The verifier writes its report **before** deciding its exit
code, so a refusal carries its evidence too.

**The build record is one document at two paths.** `<arm>/build-record.json` is
the canonical caller contract — `scripts/materialize_workspace_21.sh` reads
`package_digest` from it before it will materialise a workspace — and
`receipts/build-record.json` is kept because the earlier arms and this page name
it. Both are serialised from the same dict, re-read, and compared byte for byte
after writing; a mismatch is a refusal, not a warning. Neither path is ever
overwritten (exclusive create), so an adopted or hand-repaired arm cannot be
relabelled by a later run — before this, an arm built by the tool had to be
hard-linked by hand before the launcher would accept it.

The record's identity fields are taken from the verification report of **that
run**, never from the arguments alone: `verified_commit` and
`verified_venv_prefix` are what the verifier actually saw, and a report naming a
different commit or a different interpreter is refused, so a record cannot cite a previous source.

`installed-verification.json` enumerates every installed distribution, so a
later comparison needs no re-install. The tests for all of this are
`tests/unit/evaluation/adapters/osworld/test_osworld_arm_build.py`; the
executable spec of the admission refusals is that file's negative controls.
