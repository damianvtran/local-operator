# Arm 1830 measurement records

The records behind the figures [the methods doc](../README.md) cites. They were
written during the OSWorld 2.0 campaign as loose files under
`~/worktrees/osworld/scripts/logs/`, which is **not a git worktree**: they were
unversioned, unshareable and unbacked, so a reader of the methods doc could not
check a single figure in it. They are copied here verbatim so that the citations
resolve inside the repository.

**They belong to one arm: 1830** — harness **0.64.10** at commit
`302a061e5d3f29c952c99cb7109adf9358e85ca1`, route
`openrouter/qwen/qwen3.8-max-0902`, ten tasks of the 108 in OSWorld 2.0, two
runs each (r1/r2), 500 steps, `--max-usd 3.00`, 18000 s wall. Arm 1830's
identity, digests and protocol are stated in `TRANCHE-arm-1830.md`; the records
below are the evidence, not a re-measurement.

## What is here

| Record | What it is |
| --- | --- |
| `TRANCHE-arm-1830.md` | The primary record. Arm identity and digests, the ten-pair r1/r2 table, the binary flip rate (2 of 8 paired tasks), the cost distribution (r1 $35.26 / r2 $37.28), per-fix exercise status, every zero classified, incidents, and the parity caveats. |
| `COST-FORENSICS-1830-task_003.md` | Records-only analysis of task_003 r1's $14.32: 86 calls, 53.6% cache-read, `cache_write` 5,022,937 tokens, 40 re-writes. Separates that from TTL expiry, retries/fallback and image churn, and names the arm-1796 counterpart (`1796-c1-004`). |
| `EVAL-JUDGE-FINDING-1830-task_003.md` | Why task_003's `LLM` sub-check can never pass: the driver pins `OSWORLD_EVAL_MODEL_PROVIDER=openrouter`, the vendored registry has no such backend, so `create_backend()` raises before the call and the task's bare `except Exception: return False` turns it into a silent `LLM: False`. |
| `ZERO-CLASSIFICATION-1830-009-013.md` | Classifies two r1 zeros: task_009's evaluator read a directory-listing tab instead of the app page (**apparatus**), task_013's comparison ran and was wrong (**capability**). Also answers the general "scored but silent" question. |
| `PROBE-RULE4-README.md` | The README for the nine-case prefix-cache probe: what it varies, the two rows the methods doc quotes (18,572/18,566/0 and 18,575/0/18,569), and the caveat that every row was served by Alibaba. |
| `probe_rule4.py.txt` | The probe script itself, byte-identical to the arm's `probe_rule4.py`. See "Why the probe is a `.txt`" below. |

## What these records do NOT establish

Each record states its own limits. They are carried forward here rather than
flattened, because a reader who takes a number without them will over-read it:

- **Nothing here supports a parity claim.** `TRANCHE-arm-1830.md` says so in its
  own words: these are ten tasks of 108, OSWorld 2.0's leaderboard figures are
  full-suite, and "a domain-selected subset can outscore a full-suite average by
  drawing easier tasks". A ten-task arm cannot carry a capability claim — its
  binary flip rate (2 of 8 paired tasks, in opposite directions on identical
  build/route/budgets) is the number that matters.
- **The arm does not validate a fix that never fired.** Three of the seven fixes
  under test were not exercised in any record; the signal scan covered the r1
  records and was explicitly "not an exhaustive pass over all 20 episodes".
- **The cost figures are lower bounds where noted**, because the provider
  reported a price on only a subset of calls (e.g. 001 r1 on 62 of 144).
- **The tranche is incomplete in two places**: r2 for task_016 and task_017 were
  never run (blocked on the disk floor), and the first 005/006/009 attempt was
  voided by a disk stop and excluded from every aggregate. 001 r1 scored through
  an `ENOSPC` (`record_incomplete: true`, labelled).
- **task_009's exclusion rests on the wrong page, not on the model being right.**
  The record: it proves the zero came from the evaluator reading a directory
  listing, it does **not** prove the model's answers were correct — "not
  evaluated as intended", not "verified wrong".
- **task_013 is a capability miss whose cause is not recoverable**: `evaluate()`
  returned a bare `0.0` that names no failing checkpoint, so the record
  distinguishes ran-and-failed from never-ran but not *why* it failed.
- **The cost-forensics correlation is not a proof**, in its own words: longer
  preceding gaps accompany re-writes, but the opposite case exists in-arm
  (004 r1 ran 45 action batches with **0** re-writes), and the proposed fix —
  appending the observation frame so the cached prefix survives an action turn
  — was out of that tranche's scope and is **not applied**. The
  `cache_write_5m_tokens`/`cache_write_1h_tokens` fields are zero on this route,
  so no cache-tier signal is recoverable from the records.
- **The judge finding proves the judge cannot be constructed**, no more: it does
  not establish that the images the model produced were correct. It records the
  affected class as "any task whose evaluator calls
  `desktop_env.evaluators.model_client`" and names task_003 as the one such task
  *within the ten*; the count of affected tasks out of the 108 is **not** in the
  record.
- **The probe rows were not re-measured.** `PROBE-RULE4-README.md` records that
  they are reproduced from the probe's own output as captured in the arm's
  transcript, and every case was served by **Alibaba**, so the observation is
  provider-specific. Treat them as recorded observations, not a re-runnable
  result, unless the probe is executed again.
- **Several figures the methods doc quotes are not in these records at all.**
  That is not a contradiction, only a boundary: e.g. the provider-reported call
  counts for task_005 and task_016, task_016's `cache_write` 1.69M, the
  "18 of the 108 tasks" scope of the judge defect, and the "nine ended in a real
  `finish`, the tenth (`task_004`) ended `agent_stop`" line come from the sealed
  episode bundles, which are not committed here.

## Copies and provenance

Every file above is a **byte-identical copy** of the arm's record; nothing was
edited to arrive here. sha256 of the copies as committed:

```
76e878efc26d13130a8edeccc4b5935414396a309696a49a0e4fbaf25acb507e  TRANCHE-arm-1830.md
593981b4c90d1665f54c06efd5dfce3251c3fe54b39b39b4ddb16846dcf4b708  COST-FORENSICS-1830-task_003.md
53054d1f56ba06c185cb39dfabe94dde8e7ced42b47df287bea6298b14315870  EVAL-JUDGE-FINDING-1830-task_003.md
5cc0766509b69fea30328cf6f41b3d3187dc5736ab85964a74bd868288ff1d95  ZERO-CLASSIFICATION-1830-009-013.md
52ea010d1f538e1b7c19d1924a819359d18d21bae5e62f09617bcc62028d7cd3  PROBE-RULE4-README.md
98f8295e97f13ca242dc2004edbecf8668ebe579aac01de5ddb5e0839853a916  probe_rule4.py.txt
```

No credential, hostname, instance id or session id appears in any of them, so
nothing needed scrubbing. One operator-home path does appear, and is harmless:
`probe_rule4.py.txt:11` (and `:4`) builds `Path.home() / ".local-operator" /
"auth.db"` — an *unexpanded* home construction carrying no username and no
secret, and resolving on whatever machine reads it — and it is left as-is so the
file stays byte-identical to the arm's copy. The `/home/user/...`
paths in `ZERO-CLASSIFICATION-1830-009-013.md` are the **Ubuntu guest's** paths
and are part of the evidence — they are deliberately left alone.

### Why the probe is a `.txt`

`probe_rule4.py` is preserved under its own name plus `.txt` because the tree's
lint gate reads every tracked `.py` outside `.venv` (`flake8 .`, `black --check .`
and `isort --check .` walk the whole tree) and this record fails it in 23 places
— it is a scratch script, never formatted. Storing it as a `.py` would redden CI's
lint job; the exclusion that would prevent that belongs in `.flake8` (flake8's
config, not `pyproject.toml`), which this change deliberately does not touch.
Copying it verbatim and renaming
the extension keeps the record unmodified *and* keeps the gate green. It is not
importable and nothing reads it; it is here to be read.

## Records still outside the repository

The methods doc cites two further records that this change does **not** bring in,
so those citations remain home paths:

- `~/worktrees/osworld/scripts/logs/CONTROL-repeat-1796.md` and
  `TRANCHE-arm-1796.md` — arm **1796** records (its control repeats and the
  −46% arm-to-arm delta the control put inside within-build variance).
  `COST-FORENSICS-1830-task_003.md` also quotes the arm-1796 episode
  `1796-c1-004` (111 calls, 58.0% read, `cache_write` 6,023,682, $17.1957).
- The arm **1748** and arm **1796** tranches generally, and the sealed episode
  bundles under `~/worktrees/osworld/runs/a<arm>-*` that several figures in the
  methods doc and in `TRANCHE-arm-1830.md` come from.
