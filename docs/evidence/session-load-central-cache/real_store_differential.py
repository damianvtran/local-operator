"""Differential over REAL journals: the one-row reader vs the resident Transcript.

Read-only against the operator's store (every journal is opened ``"rb"``, and the
comparison object is built with ``defer_materialise=True`` so nothing is created
or written). This is the real-store counterpart to the synthetic differential in
``tests/unit/session/test_transcript.py``: the unit matrix pins the contract, this
pins it against the journals the operator actually has, including the ones with
thousands of rows above the matching row.

    .venv/bin/python docs/evidence/session-load-central-cache/real_store_differential.py

Every custom type present in each sampled journal is compared, both projections
(``latest_custom`` and ``latest_custom_entry``), and a mismatch prints both
answers. The exit status is non-zero if any row disagrees, so it can be wired
into a check rather than read by eye.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent.parent.parent))  # the repo root, wherever this file moves

from local_operator.session.transcript import (  # noqa: E402
    TRANSCRIPT_FILENAME,
    Transcript,
    read_latest_custom,
    read_latest_custom_entry,
)

#: The journals the diagnosis was taken on, plus the small one that is the
#: control: 261 / 108 / 96 / 87 / 32 / 9 MB.
SESSIONS = (
    "bda7b76d34e0",
    "2f95e374dd22",
    "9f8e5b652ac7",
    "4140ee201ce1",
    "29435655756c",
    "03f18d75b736",
)


def custom_types(path: Path) -> list[str]:
    """Every ``custom_type`` the journal carries, in first-appearance order."""
    found: list[str] = []
    with path.open("rb") as handle:
        for raw in handle:
            if b'"custom"' not in raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if row.get("type") != "custom":
                continue
            name = str(row.get("payload", {}).get("custom_type", ""))
            if name not in found:
                found.append(name)
    return found


def main() -> int:
    store = Path.home() / ".local-operator" / "sessions"
    comparisons = 0
    mismatches = 0
    for session_id in SESSIONS:
        directory = store / session_id
        path = directory / TRANSCRIPT_FILENAME
        if not path.is_file():
            print(f"{session_id}: absent, skipped")
            continue
        resident = Transcript(directory, defer_materialise=True)
        types = custom_types(path)
        for custom_type in [*types, "never_written"]:
            for name, scanned, reference in (
                ("latest_custom", read_latest_custom, resident.latest_custom),
                ("latest_custom_entry", read_latest_custom_entry, resident.latest_custom_entry),
            ):
                comparisons += 1
                try:
                    expected = ("value", reference(custom_type))
                except Exception as exc:  # noqa: BLE001 — the failure mode is compared too
                    expected = ("raised", type(exc).__name__)
                try:
                    actual = ("value", scanned(directory, custom_type))
                except Exception as exc:  # noqa: BLE001
                    actual = ("raised", type(exc).__name__)
                if expected != actual:
                    mismatches += 1
                    print(
                        f"MISMATCH {session_id} {name}({custom_type!r}): "
                        f"resident={expected!r} scan={actual!r}"
                    )
        print(f"{session_id}: {len(types)} custom types compared, {mismatches} mismatches so far")

    print(f"\ncomparisons={comparisons} mismatches={mismatches}")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
