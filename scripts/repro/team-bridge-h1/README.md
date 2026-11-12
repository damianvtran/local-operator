# team-bridge-h1 reproduction tools

Reusable sources retained from a historical validation run. These are development
rigs, not current performance results or an instruction to run live-provider or
native checks without their normal isolation and approval gates.

- [`capture-rig.py`](capture-rig.py)
- [`proactive-rig.py`](proactive-rig.py)
- [`wire-captures.json`](wire-captures.json)

Read each source's module documentation and arguments before running it. Run
from the repository root unless the source says otherwise; put all generated
frames, logs and reports in session scratch storage, not beside these sources.

`capture-rig.py` replays the adjacent synthetic `wire-captures.json` input;
keep that fixture with the rig. The safety prologue is regression-tested by
`tests/unit/team_bridge/test_evidence_rigs.py`.

[Original run context and invocation examples](https://github.com/damianvtran/local-operator/blob/60a908c247c4c5067488356da9c935a505759b87/docs/evidence/team-bridge-h1/README.md)
are historical; they do not certify the current revision.
