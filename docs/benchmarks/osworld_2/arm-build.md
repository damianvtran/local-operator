# Building and admitting an arm (no guest, no model, no spend)

An **arm** is the dedicated interpreter the OSWorld adapter runs in: the harness
wheel, the adapter wheel, the adapter's locked `osworld` extra, and a recorded
verification of what was installed. `scripts/build_osworld_arm.py` builds one
from a committed SHA and `scripts/osworld_dependency_admission.py` decides
whether its dependency set conforms to the reviewed
[dependency policy](dependency-policy.md). Until now that recipe existed only as
operator scratch; this is its durable, tested form.

```sh
# plan: resolve and record, install nothing
python scripts/build_osworld_arm.py --source-sha <40-hex-merged-sha> --out ~/worktrees/osworld/arms/<name>

# build: wheels, venv, install, admission, installed-identity verification
python scripts/build_osworld_arm.py --source-sha <40-hex-merged-sha> --out ~/worktrees/osworld/arms/<name> --execute
```

Exit codes: `2` refused (short or unmerged SHA, existing arm directory, volatile
output root, unusable interpreter, failed command), `4` built but dependency
admission or installed-identity verification refused. Receipts are written in
both cases — a refusal without its evidence is not actionable.

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
  pins and the worker re-verifies;
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
3. **A wheel archive hash is not an identity.** Zip member timestamps differ
   between two builds of the same commit, so the `.whl` sha256 is not
   reproducible and must not be used as a drift detector (main16's recorded
   archive mismatch was this, not tampering). The record keeps archive hashes
   under `wheel_sha256_informational_only`; the identities are the source
   commit, the pinned module hashes, the installed file set and the package
   digest.

## What it does NOT prove

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
`.rc`, `policy-admission.json`, `installed-verification.json`, and
`build-record.json`.

`installed-verification.json` enumerates every installed distribution, so a
later comparison needs no re-install. The tests for all of this are
`tests/unit/evaluation/adapters/osworld/test_osworld_arm_build.py`; the
executable spec of the admission refusals is that file's negative controls.
