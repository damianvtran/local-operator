"""Make the example's modules importable without packaging them.

The example ships as loose files copied into a container (``app/``) and run as
scripts (``scripts/``), not as an installed package, so the tests put those two
directories on ``sys.path`` exactly as the Dockerfile's ``PYTHONPATH=/app`` and
``python3 scripts/invoke.py`` do.

These tests are offline: no AWS credentials, no network, no AgentCore SDK. Run:

    python -m pytest examples/agentcore/tests -q

with ``boto3``, ``pytest`` and ``local-operator`` installed. They are not part
of the repository's own ``tests/`` suite (pytest ``testpaths``), because the
example pins a released ``local-operator`` from PyPI rather than the checkout.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("app", "scripts"):
    path = str(ROOT / sub)
    if path not in sys.path:
        sys.path.insert(0, path)
