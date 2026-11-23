"""Native Windows checks for the handoff containment boundary."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from local_operator.network import mobility


@pytest.mark.skipif(os.name != "nt", reason="Requires native Windows reparse-point semantics")
@pytest.mark.parametrize("entry", ["parent", "leaf"])
def test_handoff_paths_refuse_reparse_points(tmp_path: Path, entry: str) -> None:
    root = tmp_path / "config"
    backing = root / "backing"
    backing.mkdir(parents=True)
    parent = root / "sessions"
    leaf = parent / "abc123"
    if entry == "parent":
        (backing / leaf.name).mkdir()
        link = parent
    else:
        parent.mkdir()
        link = leaf
    subprocess.run(
        ["cmd", "/d", "/c", "mklink", "/J", str(link), str(backing)],
        check=True,
        capture_output=True,
    )

    control = root / "plain" / leaf.name
    control.mkdir(parents=True)
    expected = control.resolve()
    with (
        patch("os.rename") as rename,
        patch("os.replace") as replace,
        patch("os.unlink") as unlink,
    ):
        assert (
            mobility._contained_handoff_path(  # noqa: SLF001
                root, control, parent_parts=("plain",), leaf=leaf.name, require_leaf=True
            )
            == expected
        )
        assert (
            mobility._contained_handoff_path(  # noqa: SLF001
                root, leaf, parent_parts=("sessions",), leaf=leaf.name, require_leaf=True
            )
            is None
        )
        missing = control.with_name("missing")
        assert mobility._contained_handoff_path(  # noqa: SLF001
            root, missing, parent_parts=("plain",), leaf=missing.name
        ) == expected.with_name(missing.name)
        rename.assert_not_called()
        replace.assert_not_called()
        unlink.assert_not_called()
