from __future__ import annotations

import os

import pytest

from local_operator.library import identity


def test_asset_id_is_stable_and_opaque() -> None:
    key = "p:/tmp/a.png"
    first = identity.asset_id(key)
    assert first == identity.asset_id(key)
    assert first.startswith("la_") and len(first) == 19
    assert identity.is_asset_id(first)
    assert not identity.is_asset_id("p:/tmp/a.png")
    assert not identity.is_asset_id("la_zzzz")
    assert identity.asset_id(key) != identity.asset_id("p:/tmp/b.png")


def test_path_keys_resolve_symlinks_but_keep_a_missing_spelling(tmp_path) -> None:
    real = tmp_path / "real.txt"
    real.write_text("x")
    alias = tmp_path / "alias.txt"
    alias.symlink_to(real)
    assert identity.path_key(str(real)) == identity.path_key(str(alias))
    missing = tmp_path / "missing.txt"
    assert identity.path_key(str(missing)) == f"p:{os.path.abspath(missing)}"


def test_digest_key_and_key_for_precedence() -> None:
    assert identity.digest_key("a" * 32) == "a:" + "a" * 32
    assert identity.key_for("/tmp/x", "b" * 32) == "a:" + "b" * 32
    assert identity.key_for("/tmp/x", None).startswith("p:/")
    with pytest.raises(ValueError):
        identity.key_for(None, None)
    with pytest.raises(ValueError):
        identity.path_key("")
