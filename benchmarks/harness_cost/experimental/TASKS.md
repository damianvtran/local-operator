# External public-task materialization and offline evaluator

## Scope and attribution

Task definitions and metric implementations come from **OSWorld original v1**, full
commit `8903ebeb0efdf74331bd7dc37ed91ccbbff22317` (not OSWorld-Verified):

- Repository: https://github.com/xlang-ai/OSWorld
- Definitions: https://github.com/xlang-ai/OSWorld/tree/8903ebeb0efdf74331bd7dc37ed91ccbbff22317/evaluation_examples/examples
- Metrics: https://github.com/xlang-ai/OSWorld/tree/8903ebeb0efdf74331bd7dc37ed91ccbbff22317/desktop_env/evaluators/metrics
- License: https://github.com/xlang-ai/OSWorld/blob/8903ebeb0efdf74331bd7dc37ed91ccbbff22317/LICENSE
  (unaltered copy in `licenses/OSWorld-LICENSE`).
- Public fixtures: https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache

No task text, fixtures, gold artifacts, generated tasks.json, outputs, or credentials
are redistributed here. Public availability does not change an asset's license;
check upstream fixture terms before use or redistribution. The selection metadata
below is not a preregistered validated suite. Tier labels are inherited benchmark
selection metadata, not upstream difficulty claims.

## Exact setup: one DEVELOPMENT ONLY smoke task

These commands require Python 3.11+ and git. They are instructions, not an assertion
that dependencies or downloads have been run. Start in this `experimental` directory. Network occurs only in the
optional clone/install commands or the explicit `--download` fetch command.

```sh
export BENCH_KIT="$PWD"
export BENCH_PUBLIC_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/osworld-public.XXXXXX")"
export BENCH_OSWORLD_ROOT="$BENCH_PUBLIC_ROOT/OSWorld"
export BENCH_TASKS_DIR="$BENCH_PUBLIC_ROOT/data"
export BENCH_TASKS_FILE="$BENCH_TASKS_DIR/tasks.json"
export BENCH_TASK_CACHE="$BENCH_TASKS_DIR/cache"
export BENCH_EVALUATOR="$BENCH_KIT/evaluate.py"
export BENCH_OSWORLD_PIN=8903ebeb0efdf74331bd7dc37ed91ccbbff22317

# Optional explicit network setup; use an existing clean pinned checkout instead.
git clone --no-checkout https://github.com/xlang-ai/OSWorld.git "$BENCH_OSWORLD_ROOT"
git -C "$BENCH_OSWORLD_ROOT" checkout --detach "$BENCH_OSWORLD_PIN"

# Offline metadata/URL planning only. Choose a fresh output directory for each fetch.
python3 "$BENCH_KIT/fetch_task.py" \
  --osworld-root "$BENCH_OSWORLD_ROOT" --pin "$BENCH_OSWORLD_PIN" \
  --output-dir "$BENCH_PUBLIC_ROOT/plan-only" \
  --task libreoffice_calc:37608790-6147-45d0-9f20-1137bb35703d:short

# Explicit public input/gold download, external to both source repositories.
python3 "$BENCH_KIT/fetch_task.py" \
  --osworld-root "$BENCH_OSWORLD_ROOT" --pin "$BENCH_OSWORLD_PIN" \
  --output-dir "$BENCH_TASKS_DIR" \
  --task libreoffice_calc:37608790-6147-45d0-9f20-1137bb35703d:short \
  --download
```

Repeat `--task DOMAIN:UUID:short|long` to materialize an explicitly chosen subset.
No selection defaults, implicit clone, install, setup-command execution, model
calls, or network requests exist in the fetcher's planning path. Checkout HEAD
must equal the supported full pin, and task/metric trees must be clean, including
untracked files. Output must be empty and outside both source repositories.

The development task is marked `_development_only: true`. Keep it out of confirmatory
counts. Fetching it does **not** qualify its oracle or make it a production task.

## Dependencies and evaluator invocation

Choose the interpreter explicitly; there is no account-specific interpreter or
checkout fallback. The following is optional manual environment setup and performs
package downloads. It is not a reproducible dependency lock or a validated install.
Dependency compatibility must be checked on the intended platform.

```sh
python3 -m venv "$BENCH_PUBLIC_ROOT/evalenv"
export BENCH_EVAL_PY="$BENCH_PUBLIC_ROOT/evalenv/bin/python"
"$BENCH_EVAL_PY" -m pip install \
  openpyxl pandas numpy rapidfuzz formulas lxml cssselect xmltodict tldextract \
  Pillow python-docx odfpy python-pptx PyPDF2

# Checks pin, task definition, all selected task fixture hashes and metric import.
# Does not run the checker, start a model, or assert oracle validity.
"$BENCH_EVAL_PY" "$BENCH_EVALUATOR" \
  37608790-6147-45d0-9f20-1137bb35703d "$BENCH_PUBLIC_ROOT/work" --preflight

# Only after the runner has prepared inputs and saved the actual result in work:
"$BENCH_EVAL_PY" "$BENCH_EVALUATOR" \
  37608790-6147-45d0-9f20-1137bb35703d "$BENCH_PUBLIC_ROOT/work"
```

`evaluate.py TASK_ID WORKDIR` uses `BENCH_OSWORLD_ROOT`, `BENCH_TASKS_FILE`, and
`BENCH_TASK_CACHE` (the **cache directory**, not its parent). `BENCH_TASKS_DIR`
provides `tasks.json` and `cache/` defaults, overridden by explicit FILE/CACHE env
vars. CLI `--osworld-root`, `--tasks-file`, `--task-cache` take highest precedence.
The runner chooses `BENCH_EVAL_PY` and `BENCH_EVALUATOR`; the evaluator does not
re-exec itself. A VS Code-only metric environment requires `PyPDF2` because that
pinned module imports it even for JSON checks.

### Runner integration contract

- Load `BENCH_TASKS_FILE` (or `$BENCH_TASKS_DIR/tasks.json`), never a packaged list.
- Read inputs from `$BENCH_TASK_CACHE/<id>/input/<basename(config path)>`.
- Read gold from `$BENCH_TASK_CACHE/<id>/gold/<expected.dest>`.
- Preserve upstream `id`, `instruction`, `config`, and `evaluator` maps. Generated
  definitions add `_domain`, `_tier`, and the smoke flag; setup entries are not
  rewritten. Basename collisions fail rather than silently replacing inputs.
- Map `/home/user/<relative>` to `WORKDIR/<relative>`; reject path escapes.
  The fetcher preserves `open`/`execute` config but never executes any of it.
  Runner_v2 handles downloads and its existing unzip adaptation; it does not
  reproduce arbitrary VM or GUI setup. Audit each selected task's setup before use.
- Run the evaluator as `[BENCH_EVAL_PY, BENCH_EVALUATOR, TASK_ID, WORKDIR]`,
  forwarding source/data env vars even when the agent HOME is isolated.
- Treat exit 2 / `{"error": ...}` as a setup failure, **not** a zero-scored task.
  Missing saved result is an ordinary `score: 0.0` only after setup/import gates.
- Require this evaluator's `--preflight` before costly execution. Preserve the
  generated manifest and fixture hashes with the run's frozen inputs.

## Provenance and honest limitations

`task-manifest.json` records source URL/full commit, each raw upstream definition's
SHA256, generated tasks.json SHA256, and each input/gold source URL, destination,
and SHA256. Without `--download`, asset hashes remain null and
`downloads_complete` is false when assets are needed. Such a plan cannot score.
Downloads use the public Hugging Face collection, permit its CDN redirects, and
send no authorization headers. URLs may point at mutable `main`: the recorded
hash is the observed artifact identity, not a claim that upstream published an
immutable fixture version. No expected hash is invented.

The evaluator rechecks the clean pinned source, compares the task definition to
its git object, and verifies selected task input/gold hashes before metric import.
Only ten explicitly mapped file/rule metric functions are supported. It skips
package initializers that would initialize VM machinery. OCR is a fail-closed
stub: any request for an OCR reader fails rather than fetching model weights.
GUI save postconfig is not run: agents must save in place. This is an offline
file-task adaptation, not full OSWorld execution or proof of benchmark parity.
Archive execution, applications, and external converters remain runner/operator
responsibilities. Evaluate untrusted artifacts in an isolated environment.

**Verification actually performed for this change:** offline generation of all
24 selected definitions against the local pinned checkout; 47 public asset-map
entries planned, zero fixture downloads; CLI help; rejection of short pin,
nonempty/in-repository output, unfetched/unhashed fixtures, unsafe paths/URLs,
and tampered task hashes. A patched network opener confirmed planning makes no
HTTP calls; explicit environment overrides and directory fallback were checked.
The copied license matched the pinned git object byte for byte. A rule-only public task's
preflight reached its upstream import and correctly failed with missing `PyPDF2`
(exit 2). Existing interpreters lacked metric dependencies; no install, model,
network, harness run, or full oracle-validation matrix was performed. Passing,
wrong-result, and malformed-artifact checker controls remain required before
claiming any task qualified.

## Selection references (metadata only)

These original-v1 IDs reproduce the inspected source selection, not private task
text. `37608790-...` is **DEVELOPMENT ONLY** even though it occurs in that selection.

| Domain | Tier | Upstream ID |
|---|---|---|
| libreoffice_calc | short / DEVELOPMENT ONLY | 37608790-6147-45d0-9f20-1137bb35703d |
| libreoffice_calc | short | abed40dc-063f-4598-8ba5-9fe749c0615d |
| libreoffice_calc | short | 01b269ae-2111-4a07-81fd-3fcd711993b0 |
| libreoffice_calc | short | 1e8df695-bd1b-45b3-b557-e7d599cf7597 |
| libreoffice_calc | short | eb03d19a-b88d-4de4-8a64-ca0ac66f426b |
| libreoffice_calc | short | 7a4e4bc8-922c-4c84-865c-25ba34136be1 |
| libreoffice_calc | short | 51b11269-2ca8-4b2a-9163-f21758420e78 |
| libreoffice_calc | short | 0cecd4f3-74de-457b-ba94-29ad6b5dafb6 |
| libreoffice_writer | short | 0e763496-b6bb-4508-a427-fad0b6c3e195 |
| libreoffice_writer | short | e528b65e-1107-4b8c-8988-490e4fece599 |
| libreoffice_writer | short | d53ff5ee-3b1a-431e-b2be-30ed2673079b |
| libreoffice_impress | short | 73c99fb9-f828-43ce-b87a-01dc07faa224 |
| libreoffice_impress | short | edb61b14-a854-4bf5-a075-c8075c11293a |
| libreoffice_impress | short | 9cf05d24-6bd9-4dae-8967-f67d88f5d38a |
| vs_code | short | 276cc624-87ea-4f08-ab93-f770e3790175 |
| vs_code | short | 930fdb3b-11a8-46fe-9bac-577332e2640e |
| multi_apps | long | eb303e01-261e-4972-8c07-c9b4e7a4922a |
| multi_apps | long | aceb0368-56b8-4073-b70e-3dc9aee184e0 |
| multi_apps | long | 81c425f5-78f3-4771-afd6-3d2973825947 |
| multi_apps | long | bc2b57f3-686d-4ec9-87ce-edf850b7e442 |
| multi_apps | long | 51f5801c-18b3-4f25-b0c3-02f85507a078 |
| multi_apps | long | 5bc63fb9-276a-4439-a7c1-9dc76401737f |
| multi_apps | long | 20236825-b5df-46e7-89bf-62e1d640a897 |
| multi_apps | long | 8df7e444-8e06-4f93-8a1a-c5c974269d82 |
