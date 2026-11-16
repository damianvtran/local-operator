# PR2 bounded preflight — 2026-11-16

Fix-class allocation: implementation started 06:35Z; checkpoint 08:35Z. Worktree: `/home/ec2-user/local-operator/.worktrees/node-browser-relay-harness` (base `7dbd515864c9e6d7549b1255367a28171ab08754`). No installs, builds, downloads, new venv or worktree.

## Plan before native dispatch

1. Fresh all-running scope/process census and disk >=1 GiB; stable owners ce618964c88e and d5188ee174d4 clear (manager relayed their receipts). START/DONE around each bounded window.
2. Existing file-provenance Python only; isolated HOME/config/TMPDIR, clean env, no inherited CMUX/LOP/XPC flags. Assert `local_operator.__file__` is this worktree before importing relay/route/session dispatch; AST parse edited Python.
3. Offline existing Pyright/flake8 against the changed files only; targeted `tests/unit/scripts/test_node_browser_harness.py` only, serial, no pytest cache or bytecode writes.
4. Check existing Storybook iframe/story and installed signed CRX/update manifest (read only). Check Chrome executable and unshare/mount/ip/ss availability.
5. Private user+mount+net namespace, bind-mask policy and anchor, only lo, real :4099 unreachable. Bootstrap and close two real loopback relays using existing invite/SAS fixture, real synthetic signed onboarding binding, and a synthetic inbox. Mount the real desktop router and assert the actual /answers path exists. This preflight does not launch Chrome or answer any card.
6. STOP on any prerequisite failure; report bounded repair. Do not dispatch the full harness until all preflight rows are green.

## One discriminating native run

Run the existing harness with the reused Storybook build, under its existing scope MemoryMax=1536M, MemorySwapMax=0, timeout600. Production Session/browser tool creates real site:v1 asks. Catalogue transport + RemoteAttentionWatcher must create exactly one full remote-notice key in the synthetic operator notifications.db. A real TCP HTTP POST to the actual desktop /answers handler must resolve a real mesh-backed remote session and sign through production bridge_decide. Verify node sites output, open succeeds, rendered light/dark stills, deny, wrong-key, unsigned remote facade and node-local unsigned refusals, cancellation/late approval and member notice settlement. Compare real anchor/policy hashes, real paired status identity and live :4099 owner before/after. Keep frames/results/logs only; existing exact scope/group cleanup removes Chrome profile and short tmp.

Full matrix is independent QA/review later on one pushed head; no merge by this coder.
