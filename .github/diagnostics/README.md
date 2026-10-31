# Pinned selector diagnostic (not a CI fix)

This is a disposable native experiment for publisher PR #3209, run
41230666425/job 124627633024, also observed on FIFO PR #3237. Historical reports
identify an xdist worker exit in
`tests/unit/test_ci_scope_scoping.py::test_a_repo_python_file_a_test_names_is_always_selected`;
the 240-second report-only watchdog did not capture exit/signal/cgroup evidence.
Neither a local macOS pass nor descendant relay #3228 establishes native proof.

## Preflight and lease

- Exactly two **serial** payloads: base
  `87a3741ceb062979989d134acd18d3320c372313`, then head
  `9db425ea9b3c1f6c6cdfe236644d88c7b0e94ebd`, even if base fails.
- One hosted job, original runner label `ubuntu-latest` (confirmed from that job's
  API metadata). The driver rejects anything except Linux, Ubuntu 26.04 and
  Python 3.12.15; actual point release/kernel/Python are retained in provenance.
- One dependency install at base using CI's `pip install -e '.[dev]'`, with
  setup-python's pip cache keyed on `pyproject.toml`. There is no root Python
  lockfile. Byte-identical base/head `pyproject.toml` is mandatory; both checkouts
  use the same path, so the single editable install follows the checkout.
- No product/source/test changes at either pinned ref, no matrix, no broad-suite
  reruns, no service/production calls. Isolated HOME/config/TMPDIR, allowlisted
  environment and disabled credential helpers keep host credentials out.
- Local controls use five tiny payloads (success, exit 23, SIGKILL, timeout with
  ignored TERM and a descendant, 80 MiB allocation). No actual graph test runs
  locally. Use the existing repository interpreter; install nothing locally.

Each ref executes the existing `scripts/run_bounded.py` CLI with its actual
supported flags:

```sh
python scripts/run_bounded.py --timeout 600 --grace 5 --memory-mb 2048 -- \
  python -X faulthandler -m pytest \
  tests/unit/test_ci_scope_scoping.py::test_a_repo_python_file_a_test_names_is_always_selected \
  -n 1 -vv --cov --cov-fail-under=0 --cov-report= --junit-xml=<ref-artifact>/junit.xml
```

The reviewed driver observes the bounded interpreter's `Popen` wait status via
`runpy`; it records the **raw child exit code/signal** without modifying pytest,
xdist or either ref's bounded script. The original report-only stall watchdog
stays at 240 seconds. It captures full separate stdout/stderr, raw watchdog files
under each ref's `tmp/lo-shard-stall`, raw memory dumps, per-second
PID/PPID/PGID/RSS samples, command/SHA, JUnit, and cgroup v2 memory events before
and after (including ancestors). Cgroup counters are shared context, not proof
that this child caused a cgroup OOM. Worker crashes remain in full pytest stderr
and watchdog artifacts, distinct from the controller's raw exit receipt.

The 2048 MiB guard is the shipped wrapper's **sampled owned-process-group**
ceiling, not a kernel cgroup limit and not an aggregate job limit; transient
overshoot is possible. Timeout is 600 seconds plus 5-second TERM/KILL grace.
The driver's emergency monitor stops after 615 seconds per payload and allows
8 seconds for wrapper cleanup, then exact owned-group KILL/reap (3 seconds).
The step cap is 22 minutes; the whole job cap is **35 runner-minutes**, including
one <=8-minute installation and artifact upload. At most one job per activation,
no matrix. Queue time is outside the runner-minute lease. Output has a 32 MiB
per-file `RLIMIT_FSIZE` (hitting it fails, never silently truncates) and a sampled
256 MiB per-ref disk guard. Artifacts expire after seven days.

A diagnostic cap firing does **not** prove the original hit that cap. Two PASS
results leave historical shard/history interaction **unknown**. No automatic
suite escalation, bypass, main merge or CI-fix claim follows either outcome.

## Review first; activation is a separate, explicit action

Implementation commits carry `[skip ci]` because `.github/**` intentionally
selects **all** normal CI gates; auto-running that matrix would violate this
experiment's narrow lease. This is not a merge candidate and must not be merged
with skipped required checks. The new workflow has no PR or dispatch trigger and
runs only on `diag/ci-selector-native-activate`. Nothing native runs on the
implementation branch. New workflow dispatches may require main registration;
this branch-push mechanism needs no main merge.

After independent code review and manager authorization, create a **tree-identical
empty activation commit** without `[skip ci]`, based on the exact reviewed head,
and push ONLY the activation ref (do not push it to the PR branch):

```sh
reviewed=<full-reviewed-head-sha>
activation=$(printf 'ci: activate reviewed selector diagnostic\n' | \
  git commit-tree "$(git rev-parse "$reviewed^{tree}")" -p "$reviewed")
test "$(git rev-parse "$activation^{tree}")" = "$(git rev-parse "$reviewed^{tree}")"
git push origin "$activation:refs/heads/diag/ci-selector-native-activate"
```

No force push, merge or workflow dispatch is required. If that branch already
exists, stop and inspect it rather than overwrite it. Review/authorization is
still required before repeating this action; concurrency does not grant it.
The upload includes `diagnostic-source-sha` identifying this tree-identical
activation commit. Inspect provenance, both `result.json`/`child-status.json`,
raw stderr/watchdogs and JUnit before interpreting a pair as PASS.

Local orchestration check (not native selector proof):

```sh
python -m unittest discover -s .github/diagnostics -p 'test_ci_selector_native.py' -v
```
