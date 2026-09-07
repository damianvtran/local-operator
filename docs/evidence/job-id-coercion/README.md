# Job ids survive argument coercion as opaque text

Evidence behind the fix for `_coerce_job_targets`, which reinterpreted a
numeric-looking job **id** as a JSON **number** and then dropped it.

Job ids are `uuid4().hex[:12]`. A measured **0.65%** of them are also
well-formed JSON numbers — ~0.31% all digits (`920883861377`) and ~0.33%
exponent-shaped (`7019316393e2`, `177650473e52`). `json.loads('[920883861377]')`
returns `[920883861377]`, a list of `int`; the old `isinstance(item, str)`
filter dropped every element, so the function returned the raw bracketed string
and the caller reported `unknown job [920883861377]` — an error in which the id
*looks* correct, which is what made the failure expensive to diagnose.

The script is renamed `.py.txt` so the repo's linters skip one-off tooling, and
it runs against the worktree it is launched from (`sys.path` is the cwd). It
prints `module.__file__`, the sha256 of that exact file, and the git HEAD, and
asserts the import came from the current tree — this repo's editable venv has
produced false "no difference" A/B results for reviewers who skipped that check.

```sh
# From the root of a worktree, with its own venv:

# Unit-level: every id shape through _coerce_job_targets, plus the measured
# mangle rate over 200k real uuid4 ids.
.venv/bin/python docs/evidence/job-id-coercion/repro_job_id.py.txt

# End-to-end: the REAL wait/jobs tools against a REAL AsyncJobManager whose
# minted job ids are forced to be all digits. Copy it to the repo root first,
# so the editable package resolves from the tree under test.
cp docs/evidence/job-id-coercion/e2e_jobid.py.txt e2e_jobid.py
.venv/bin/python e2e_jobid.py && rm e2e_jobid.py
```

## End-to-end result (real tools, real job manager)

`e2e_jobid.py.txt` patches `uuid.uuid4` so `AsyncJobManager` mints the
all-digit ids `920883861377` and `468698086935`, registers two real jobs
through the `task` tool, then drives `jobs` and `wait` exactly as a model would.

On `origin/main` — the reported symptom, verbatim:

```
=== jobs(op='peek') with an all-digit id ===
  [FAIL] bracketed string  '[920883861377]'   -> unknown job [920883861377]
  [FAIL] bare int           920883861377      -> invalid arguments: - job_id: Input should be a valid string
=== wait with all-digit ids (single + list) ===
  [FAIL] single bracketed  '[468698086935]'   -> unknown job [468698086935]
  [FAIL] LIST of two digit ids (string form)  -> unknown job [920883861377, 468698086935]
RESULT: 4 FAILED
```

On this branch all nine cases pass, e.g.

```
  [PASS] bracketed string  '[920883861377]'   -> job 920883861377 [running] seq=0 (no new output since last peek)
  [PASS] LIST of two digit ids (string form)  -> job 920883861377 (alpha) [completed] done:920883861377
```

The last case is a **negative control**: a genuinely unknown id must still
fail. It does, on both arms — and the message is now `unknown job 111111111111`
rather than `unknown job [111111111111]`, so the id in the error is the id the
caller can look up.

## A/B result

Same file, same venv, same cwd; the differing sha256 proves the two arms
imported different code.

| arm | sha256(builtin.py)[:16] | repro | `tests/unit/tools/test_job_id_coercion.py` |
| --- | --- | --- | --- |
| `origin/main` @ `d017115dd` | `f2bbcd0afc917fe8` | 9 failing groups, **1304/200000 (0.65%)** ids mangled | **16 failed**, 13 passed |
| this branch | `576cdae74f6737f7` | ALL PASS, **0/200000** mangled | **29 passed** |

On `origin/main` the end-to-end arm reproduces the reported symptom exactly:

```
[FAIL] resolve('[920883861377]') -> None err='unknown job [920883861377]'
```

and on this branch:

```
[PASS] resolve('[920883861377]') -> '920883861377' err=None
```

## Shapes covered

Every row is exercised by the repro and pinned in the unit test.

| shape | example | old | new |
| --- | --- | --- | --- |
| all-digit id, bracketed | `[920883861377]` | ✗ returned input | `920883861377` |
| all-digit id, bare | `920883861377` | ok | ok |
| all-digit id, JSON-quoted | `["920883861377"]` | ok | ok |
| exponent-shaped id | `[12e345678901]`, `[177650473e52]` | ✗ returned input | id |
| leading-zero id | `[000123456789]`, `[007]` | ok (parse error path) | ok, digits preserved |
| float-looking id | `[1e5]`, `[1.50]` | ✗ returned input | text preserved, not `100000.0` |
| bare int from the outer decode | `920883861377` (int) | ✗ stayed `int` | `"920883861377"` |
| int inside a real list | `[920883861377]` (list) | ✗ stayed `[int]` | `"920883861377"` |
| JSON bare literals | `[true]`, `[null]` | ✗ returned input | `"true"`, `"null"` |
| **multi-id, numeric** | `[920883861377, 468698086935]` | ✗ returned input | `['920883861377', '468698086935']` |
| multi-id, mixed | `[920883861377, a1b2c3d4e5f6]` | ok | ok |
| multi-id, JSON | `["a1b2c3d4e5f6", "0f1e2d3c4b5a"]` | ok | ok |
| nested list | `[["920883861377"], "a1b2c3d4e5f6"]` | ✗ lost the first id | both ids |
| ordinary hex id / label / `all` | `a1b2c3d4e5f6`, `reviewer`, `all` | ok | ok |

Note the multi-id and nested rows: the defect was **not** confined to the
single-id path. A genuine list-of-ids call whose ids happened to be numeric was
broken too, so the fix had to repair the list case rather than trade it away.

## Why the fix is not a digit special case

The root cause is that an opaque token was round-tripped through a grammar that
assigns meaning to its characters. So the repair is at the grammar level:
`parse_int`, `parse_float` and `parse_constant` hand back the **raw source
literal** instead of a parsed value, which keeps JSON's structure handling
(nesting, quoting, escapes) while making every scalar opaque.

Two alternatives were rejected:

- **`str(parsed)` after a normal parse** — lossy and silently corrupting. The
  source text is already gone by then: `007` normalises to `"7"` and `1e5` to
  `"100000.0"`, so an id would be "recovered" as a different id. Pinned by the
  `[007]` and `[1.50]` cases.
- **Skip the JSON parse when the payload looks numeric** — a digit special case
  that leaves the class of bug alive; it fixes all-digit ids while exponent and
  float shapes keep failing, and it makes the numeric multi-id list worse.

`bool` is checked before `int` in `_job_target_text` because `True` *is* an
`int` in Python and would otherwise stringify as `"1"`.
