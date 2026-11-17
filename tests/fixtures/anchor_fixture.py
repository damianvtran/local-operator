"""The one synthetic OS-anchor boundary shared by process-level fixtures.

WHY A MODULE OF ITS OWN. Two process-level rigs need the same seam: the
two-process signed-onboarding fixture (``mesh_process_node``) and the isolated
node-browser harness (``scripts/e2e/node_browser_harness.py``), whose operator
and node roles must both trust a SYNTHETIC operator key without ever reading the
host's root-owned ``/etc/local-operator/operators/<uid>.json``. One copy means a
fix to the guard (it also fences the native path and counts any attempt) reaches
both rigs, rather than one silently drifting back to a host fallback.

CONSTRAINTS. This lives under ``tests/`` and is never packaged (the wheel ships
only ``local_operator*``), and nothing under ``local_operator/`` imports it or
reads an environment variable to enable it: a caller installs it in-process,
explicitly, inside a process whose config root is already isolated.
"""

from __future__ import annotations

import json
from pathlib import Path


def install_fixture_anchor(root: Path) -> dict[str, int]:
    """One synthetic OS boundary for BOTH loader bindings, never host fallback.

    operator imports/reexports load_anchor before this helper runs. Patching
    only trust left its authority report reading the real host anchor: Darwin
    could pass via an installed anchor while clean Linux truthfully refused.
    Native aliases must instead trip this guard BEFORE any OS path is read.
    """
    from local_operator import operator
    from local_operator.operator import trust

    public_anchor = root / "fixture-owned-anchor.json"

    def os_anchor(uid=None):
        # Hardware/root-owned-OS installation is explicitly synthetic. Signature
        # math and all network authority verification remain production code.
        anchor = (
            trust.OperatorAnchor.from_json(json.loads(public_anchor.read_text()))
            if public_anchor.exists()
            else None
        )
        return trust.AnchorLoad(
            anchor=anchor,
            path=public_anchor,
            root_owned=anchor is not None,
            reason="fixture OS boundary",
            exists=anchor is not None,
        )

    reads = {"native_anchor_reads": 0}
    # Existing test-only path seam also fences any path-construction alias;
    # it is never a product environment override or an installed host anchor.
    trust._ANCHOR_ROOT_OVERRIDE = root / "forbidden-native-anchor"

    def forbidden_native_path(uid=None):
        reads["native_anchor_reads"] += 1
        raise AssertionError("fixture attempted a native OS anchor read")

    trust.anchor_path = forbidden_native_path
    trust.load_anchor = os_anchor
    operator.load_anchor = os_anchor
    assert operator.authority_level_load.__globals__["load_anchor"] is os_anchor
    return reads
