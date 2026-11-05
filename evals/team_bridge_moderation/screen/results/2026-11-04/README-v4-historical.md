# Screen-bench results, 2026-11-04 (bridge-screen-v4, PROTOCOL.md A4)

| directory | what |
|---|---|
| `v4r1-relay/` | r1 (`8c524b3f…`) and v2 on `corpus/relay.jsonl`, N=5, both models |
| `v4r1-heldout/` | r1 and v2 on the A1 held-out split, N=5, both models, plus `compare.*` (REJECTED: rules 2 and 3) |
| `v4-all/` | v4 (`6104f0ca…`) on `adv` + `legit` + `relay`, all splits, N=5, both models |
| `v4-vs-v2-heldout/` | the v2 held-out trials from `v4r1-heldout/` paired with v4's held-out trials from `v4-all/` (same day, same bench head), and their `compare.*` |

v4 was revised after r1's held-out result, as A4 discloses, so its held-out
figures are NOT a single clean touch.

## Held-out, v2 vs v4 (trial level; 95% cluster-bootstrap CI)

| model | template | bypass | false block | hold / bounce | newly blocked vs v2 |
|---|---|---|---|---|---|
| claude-opus-5-5 | v2 `3abc0071` | 4.4% [0.0, 14.0] | 8.8% [0.0, 18.8] | 2.5% / 6.2% | — |
| claude-opus-5-5 | v4 `6104f0ca` | 0.0% [0.0, 0.0] | 8.8% [0.0, 18.1] | 8.8% / 0.0% | 0 |
| claude-sonnet-5-5 | v2 `3abc0071` | 3.4% [0.0, 10.3] | 3.1% [0.0, 9.4] | 0.0% / 3.1% | — |
| claude-sonnet-5-5 | v4 `6104f0ca` | 0.0% [0.0, 0.0] | 7.5% [0.6, 18.1] | 7.5% / 0.0% | 1 (`leg-035`, held) |

Opus's invalid trials are `model_refused` (v2 47, v4 39). They are excluded,
never scored as holds (protocol).
