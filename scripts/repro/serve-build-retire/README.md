# serve-build-retire reproduction tools

Reusable sources retained from a historical validation run. These are development
rigs, not current performance results or an instruction to run live-provider or
native checks without their normal isolation and approval gates.

- [`drive.py`](drive.py)
- [`inject.py`](inject.py)
- [`run.sh`](run.sh)
- [`ungated.py`](ungated.py)

Read each source's module documentation and arguments before running it. Run
from the repository root unless the source says otherwise; put all generated
frames, logs and reports in session scratch storage, not beside these sources.

[Original run context and invocation examples](https://github.com/damianvtran/local-operator/blob/60a908c247c4c5067488356da9c935a505759b87/docs/evidence/serve-build-retire/README.md)
are historical; they do not certify the current revision.
