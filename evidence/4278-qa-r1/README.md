# QA #4278 round 1 — phone web relay Team Bridge card body

Head 4338c0684b33c2eb9d239b7d6d65da67250d0dcd, base 9907020f. Isolated HOME/config under the
QA session scratchpad, synthetic session `abcdef012345`, hub wire faked (rig/lop_fake.py),
everything else real: `lop mobile serve` from the PR worktree venv serving the PR-built
`local_operator/mobile/web/dist`, headless Chrome 157 at 390x844 DPR2 over CDP.

- `frames/after-*`: PR relay, every state, light + dark; `after-net.json` = every body request.
- `frames/before-*`: control, installed lop 0.107.3 relay + its own web app.
- `frames/zh-*`: PR relay with `language: zh-CN` (host has no CJK font: glyphs tofu; DOM text in zh-geom.json).
- `frames/nocap-*`: PR relay with the capability removed (negative control), `nocap-net.json`.
- `frames/m1-*`: review M1 live-delivery attempts (live runtime, core peer-delivery primitive).
- `rig/`: seeding (seed.py via core stores, seed_journal.py via core Transcript writer), fake, capture scripts.
- `logs/`: build log, run_bounded completion records, fake-hub wire logs.
