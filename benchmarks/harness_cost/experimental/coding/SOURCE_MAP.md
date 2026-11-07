# Source → destination map

All destinations below are relative to
`benchmarks/harness_cost/experimental/coding/` in the destination repository.
The input root is the local `harness-cost-bench/coding/` lane;
no absolute home path is embedded in runtime code.

| Source relative to input root | Destination | Treatment |
|---|---|---|
| adapters.py | adapters.py | Ported; reviewed-source LOP label, shared loadout path, generated Dsh config |
| driver.py | driver.py | Reworked portable dispatcher; frozen evidence/hashes, explicit auth/tier/caps |
| budget_guard.py | budget_guard.py | Reworked durable conservative trial + total reservations |
| test_guard.py | test_guard.py | Reworked stdlib offline guard/CLI negative controls |
| codex_bridge.py | codex_bridge.py | Preserved translation; removed private YAML auth coupling |
| lop_launch.py | lop_launch.py | Process-local endpoint launch, no recommendation credential/auth DB |
| lop_shim.py | lop_shim.py | Source shim preserved |
| oracle_sweep.py | oracle_sweep.py | Portable disk/external paths; explicit execution gate |
| subset.json | subset.json | Sanitized public selection metadata; removed oracle rewards, disk measurements and host Python version |
| codex-model-catalog.json | codex-model-catalog.json | Byte-identical source catalog; external Codex template attributed |
| README.md | README.md | Replaced private runbook with portable contracts and validation boundary |
| STATUS.md | STATUS.md | Sanitized historical status; no fresh native-validation claim |
| PILOT.md | PILOT.md | Explicit no-paid-pilot boundary; no copied results |
| dsh-anthropic.yml | embedded config in adapters.py | Exact source config generated per container; no extra payload file |

Additional in-scope documentation: `THIRD_PARTY_NOTICES.md` and this map.
`CODEX-LICENSE` and `CODEX-NOTICE` are verbatim copies of LICENSE and NOTICE
from a locally available Codex checkout, examined without fetching.

The source rig's `codex_toolsearch_bridge.py` is the attributed ancestor of the
coding source's `codex_bridge.py`; no extra bridge implementation was added.
The only public schema reference points to `../../public_loadout/`; no copied
schema, tasks, fixtures, toolchains, jobs, logs, results, auth files, archives,
build scripts or model/harness pilot outputs are included. Parent manifests,
publication files and Git commits are outside this slice.
