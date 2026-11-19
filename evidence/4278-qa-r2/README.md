# QA #4278 round 2 — delta 4338c068..b7649562

Head b7649562f95d, PR-built `local_operator/mobile/web/dist` served by the PR worktree's real `lop mobile serve`
(isolated HOME/config, `env -i`, synthetic session `abcdef012345`), headless Chrome 157 at 390x844 DPR2 over CDP.
Round-1 rig (evidence/4278-qa-r1) plus `rig/shim_extra.py`: the body route stays real; its resolver raises the real
`CardBodyNotFound` (-> the route's own `404 no such bridge card`) for the first QA_LATE[entry] reads per epoch
(`e_admit:1, e_long:2, e_held:99`), and a toggle file makes the route's own session check answer `404 unknown session`.
New seeded cards: e_n1 (cut after a full stop), e_n1neg (mid-sentence cut), e_empty (pre-receipt, answers items:[]).
`frames/r2-results.json` holds every measured number; `logs/wire.jsonl` every resolver call.
