"""Boundary tests for the sanctioned public-copy reader.

The reader is a result-redaction exemption, so its security boundary is the
FILE it opens, not the masking pass.  Review round 1 reproduced the writable-root
bypass — an agent writes a credential into a catalogue and reads it back
verbatim — so these tests pin the answers: serve the committed blob at
``HEAD``, refuse a working-tree difference, a symlink escape and an untracked
file, and record every read before returning it.  A hard link is another
working-tree name and does not change ``HEAD``, so it is served.

The fixture is a real miniature git checkout: ``HEAD`` is the whole control, so
a test that never commits is not testing the route.

Three fixtures exist because the allowlist spans three checkouts: ``_public_repo``
models this repository's copy roots, ``_ui_repo`` models the companion UI
checkout's and ``_mobile_repo`` models the companion mobile checkout's
(``src/i18n/...`` in both consumer apps).  A consumer path is only ever read
from a checkout whose working directory is that checkout and whose ``origin``
names the repository that ships that root, which is what the last two fixtures
pin — including that neither app can read the other's generated map.

Review round 2 (security) measured five boundary holes these tests now pin: a
``src/i18n/`` tree in any other checkout served unredacted (S-1: checkout
identity), a pattern widening with the glob tuple untouched (S-2: the patterns
are pinned too), a symlinked audit log suppressing the record (S-3), a
``git replace`` ref substituting the served blob (S-4), and a workspace below
the checkout root validating one file while ``git show`` served another (S-5).
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import ToolContext
from local_operator.tools import public_copy as public_copy_module
from local_operator.tools.public_copy import (
    _PUBLIC_COPY_PATTERNS,
    CORE_REPOSITORY,
    MOBILE_REPOSITORY,
    PUBLIC_COPY_GLOBS,
    PUBLISHING_REPOSITORIES,
    UI_REPOSITORY,
    _matches_public_copy,
    _origin_identity,
    _publishing_checkouts,
    execute_read_public_copy,
)

FAKE_GITHUB = "ghp_" + "A" * 36
FAKE_API_KEY = "sk_live_" + "B" * 24
UI_ORIGIN_SSH = "git@github.com:damianvtran/local-operator-ui.git"
UI_ORIGIN_HTTPS = "https://github.com/damianvtran/local-operator-ui.git"

#: Every reviewed root, with one representative file.  ``PUBLIC_COPY_GLOBS`` is
#: asserted against :data:`ACCEPTED_PUBLIC_COPY_GLOBS` below, and each sample is
#: run through the enforcement patterns: an added root or a widened pattern
#: moves this set and the diff is what review sees.
UI_CATALOGUE = "src/i18n/catalogues/en/wire.slash.json"
UI_KEYS = "src/i18n/keys.gen.ts"
UI_VALUE = "Type or paste a secret after a space; masked"
MOBILE_ORIGIN_SSH = "git@github.com:damianvtran/local-operator-mobile.git"
MOBILE_ORIGIN_HTTPS = "https://github.com/damianvtran/local-operator-mobile.git"
MOBILE_CATALOGUE = "src/i18n/catalogues/en/mobile.home.json"
MOBILE_MAP = "src/i18n/catalogue.gen.ts"
MOBILE_VALUE = "What can I help you with today?"
CORE_ORIGIN_SSH = "git@github.com:damianvtran/local-operator.git"
CORE_CATALOGUE = "local_operator/i18n/catalogues/en/wire.slash.json"

ACCEPTED_PUBLIC_COPY_GLOBS: tuple[str, ...] = (
    "local_operator/i18n/catalogues/*/*.json",
    "local_operator/i18n/keys/*.py",
    "local_operator/mobile/web/src/i18n/keys.gen.ts",
    "local_operator/mobile/web/src/i18n/bundled.gen.ts",
    "src/i18n/catalogues/*/*.json",
    "src/i18n/keys.gen.ts",
    "src/i18n/catalogue.gen.ts",
)

ACCEPTED_PUBLIC_COPY_PATHS: tuple[str, ...] = (
    "local_operator/i18n/catalogues/en/wire.slash.json",
    "local_operator/i18n/keys/wire_slash.py",
    "local_operator/mobile/web/src/i18n/keys.gen.ts",
    "local_operator/mobile/web/src/i18n/bundled.gen.ts",
    UI_CATALOGUE,
    UI_KEYS,
    MOBILE_CATALOGUE,
    MOBILE_MAP,
)

#: The enforcement list, verbatim.  The glob tuple alone is not the gate: a
#: pattern can widen while the globs stay untouched (review S-2 measured exactly
#: that), so both lists are pinned and a mutation has to move this tuple.
ACCEPTED_PUBLIC_COPY_PATTERNS: tuple[str, ...] = (
    r"local_operator/i18n/catalogues/[^/]+/[^/]+\.json",
    r"local_operator/i18n/keys/[^/]+\.py",
    r"local_operator/mobile/web/src/i18n/keys\.gen\.ts",
    r"local_operator/mobile/web/src/i18n/bundled\.gen\.ts",
    r"src/i18n/catalogues/[^/]+/[^/]+\.json",
    r"src/i18n/keys\.gen\.ts",
    r"src/i18n/catalogue\.gen\.ts",
)


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _public_repo(
    tmp_path: Path,
    *,
    name: str = "repo",
    value: str = "Type or paste a secret after a space; masked",
    origin: str | None = None,
) -> Path:
    """A committed miniature checkout with a catalogue and a key module.

    ``name``, ``value`` and ``origin`` exist so one test can stand two DISTINCT
    core trees side by side with different bytes for the same key — the only
    shape in which "the named checkout was read" cannot pass by reading the
    session's own workspace.  Every existing caller passes none of them and gets
    the original fixture: an origin-less checkout, which is exactly what the
    session-workspace arm must keep serving now that a NAMED checkout has to
    prove its identity.
    """
    root = tmp_path / name
    catalogue = root / "local_operator/i18n/catalogues/en"
    catalogue.mkdir(parents=True)
    (catalogue / "wire.slash.json").write_text(
        json.dumps({"wire.slash.credential": value}, ensure_ascii=False),
        encoding="utf-8",
    )
    keys = root / "local_operator/i18n/keys"
    keys.mkdir(parents=True)
    (keys / "wire_slash.py").write_text(
        'return Msg("wire.slash.credential", {})\n', encoding="utf-8"
    )
    (root / "README.md").write_text("not public copy\n", encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")
    if origin is not None:
        _git(root, "remote", "add", "origin", origin)
    _git(root, "add", "-A")
    _git(root, "-c", "commit.gpgsign=false", "commit", "-qm", "fixture")
    return root


def _context(root: Path, tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=str(root), session_dir=str(tmp_path / "sessions" / "abc123"))


def _ui_repo(tmp_path: Path, origin: str | None = UI_ORIGIN_SSH) -> Path:
    """A committed miniature UI checkout with the companion repo's copy roots."""
    root = tmp_path / "ui"
    catalogue = root / "src/i18n/catalogues/en"
    catalogue.mkdir(parents=True)
    (catalogue / "wire.slash.json").write_text(
        json.dumps({"wire.slash.credential": UI_VALUE}), encoding="utf-8"
    )
    (root / "src/i18n/keys.gen.ts").write_text(
        "export const keys = {\n" f'  "wire.slash.credential": "{UI_VALUE}",\n' "} as const;\n",
        encoding="utf-8",
    )
    # Committed lookalikes: each must be refused by the allowlist even though it
    # exists at HEAD, so a refusal cannot be an accident of a missing file.
    (root / "src/i18n/other.json").write_text(
        json.dumps({"wire.other": "not an allowlisted catalogue"}), encoding="utf-8"
    )
    (root / "src/i18n/x.json").write_text(
        json.dumps({"wire.x": "not an allowlisted catalogue"}), encoding="utf-8"
    )
    (root / "README.md").write_text("not public copy\n", encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")
    if origin is not None:
        _git(root, "remote", "add", "origin", origin)
    _git(root, "add", "-A")
    _git(root, "-c", "commit.gpgsign=false", "commit", "-qm", "ui fixture")
    return root


def _mobile_repo(tmp_path: Path, origin: str | None = MOBILE_ORIGIN_SSH) -> Path:
    """A committed miniature mobile checkout with that app's copy roots.

    The app ships ``src/i18n/catalogue.gen.ts`` (not a ``keys.gen.ts``), so the
    fixture carries the map the real checkout commits.  It also commits the UI's
    generated map as a decoy holding a secret-shaped value: the refusal for that
    path is then the identity gate, not the absence of a file, which is what
    makes the mobile/UI separation a measurement rather than a coincidence.
    """
    root = tmp_path / "mobile"
    catalogue = root / "src/i18n/catalogues/en"
    catalogue.mkdir(parents=True)
    (catalogue / "mobile.home.json").write_text(
        json.dumps({"mobile.home.greeting": MOBILE_VALUE}), encoding="utf-8"
    )
    (root / MOBILE_MAP).write_text(
        "export const catalogue = {\n"
        f'  "mobile.home.greeting": "{MOBILE_VALUE}",\n'
        "} as const;\n",
        encoding="utf-8",
    )
    # The other consumer app's generated map, committed here on purpose.
    (root / "src/i18n/keys.gen.ts").write_text(
        "export const keys = {\n" f'  "wire.slash.credential": "{FAKE_API_KEY}",\n' "} as const;\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text("not public copy\n", encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")
    if origin is not None:
        _git(root, "remote", "add", "origin", origin)
    _git(root, "add", "-A")
    _git(root, "-c", "commit.gpgsign=false", "commit", "-qm", "mobile fixture")
    return root


def test_the_allowlist_is_exactly_the_reviewed_set() -> None:
    """A future widening has to edit these tuples and show up as a reviewed diff."""
    assert PUBLIC_COPY_GLOBS == ACCEPTED_PUBLIC_COPY_GLOBS
    # The enforcement list, not just the human-readable one: a pattern is what
    # actually admits a path (review S-2).
    assert (
        tuple(pattern.pattern for pattern in _PUBLIC_COPY_PATTERNS) == ACCEPTED_PUBLIC_COPY_PATTERNS
    )


def test_the_consumer_identity_binding_is_per_checkout() -> None:
    """A consumer root names its own checkout, and never another's.

    Both apps keep a ``src/i18n/catalogues/`` tree, so that root belongs to both
    identities; the generated maps differ per app, so each one names only the
    app that ships it.  A core root names no checkout at all: the identity gate
    is the consumer roots' control, and the committed-blob check is the core
    roots' — the mobile identity must not add a core root to its own set.
    """
    ui = (UI_REPOSITORY, "companion UI checkout")
    mobile = (MOBILE_REPOSITORY, "mobile app checkout")

    assert _publishing_checkouts(UI_KEYS) == (ui,)
    assert _publishing_checkouts(MOBILE_MAP) == (mobile,)
    assert _publishing_checkouts(UI_CATALOGUE) == (ui, mobile)
    assert _publishing_checkouts(MOBILE_CATALOGUE) == (ui, mobile)
    for core in (
        "local_operator/i18n/catalogues/en/wire.slash.json",
        "local_operator/i18n/keys/wire_slash.py",
        "local_operator/mobile/web/src/i18n/keys.gen.ts",
    ):
        assert _publishing_checkouts(core) == ()


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        # The three shapes git hands back for a GitHub remote.
        ("git@github.com:damianvtran/local-operator-mobile.git", MOBILE_REPOSITORY),
        ("ssh://git@github.com/damianvtran/local-operator-mobile.git", MOBILE_REPOSITORY),
        ("https://github.com/damianvtran/local-operator-mobile", MOBILE_REPOSITORY),
        ("ssh://git@github.com/damianvtran/local-operator-ui.git", UI_REPOSITORY),
        # Case-insensitive, and the host is not part of the identity: an alias or
        # a mirror with the repository path is the same repository.
        ("git@gitlab.example:DamianVTran/Local-Operator-Mobile.git", MOBILE_REPOSITORY),
        # Another owner, and a shorter path, are not a consumer checkout.
        (
            "git@github.com:someone-else/local-operator-mobile.git",
            "someone-else/local-operator-mobile",
        ),
        ("/Users/me/src", "me/src"),
        ("", ""),
    ],
)
def test_the_origin_identity_is_the_last_two_path_components(url: str, expected: str) -> None:
    """Identity is `owner/repo` from the URL's tail, never the path shape."""
    assert _origin_identity(url) == expected


@pytest.mark.parametrize("path", ACCEPTED_PUBLIC_COPY_PATHS)
def test_every_reviewed_root_is_accepted(path: str) -> None:
    assert _matches_public_copy(path)


@pytest.mark.parametrize(
    "path",
    [
        # A file beside the catalogues, not inside them.
        "src/i18n/other.json",
        # Traversal that resolves out of a catalogue directory.
        "src/i18n/catalogues/en/../x.json",
        # One level too deep: ``[^/]+`` does not span a second separator.
        "src/i18n/catalogues/en/deep/x.json",
        # A name that merely ends like the generated map.
        "src/i18n/keys.gen.ts.bak",
        # The mobile app's generated map, misspelled and misplaced.
        "src/i18n/catalogue.gen.ts.bak",
        "src/i18n/catalogues.gen.ts",
        "vendor/src/i18n/catalogue.gen.ts",
        # A name beside the generated map; a widening to ``src/i18n/*`` would
        # admit it, which is why the compiled patterns are pinned (review S-2).
        "src/i18n/private.secrets.ts",
        # The same spelling nested somewhere else in the tree.
        "vendor/src/i18n/keys.gen.ts",
    ],
)
def test_paths_outside_the_reviewed_roots_are_not_matched(path: str) -> None:
    assert not _matches_public_copy(path)


@pytest.mark.asyncio
async def test_a_ui_catalogue_key_is_served(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _ui_repo(tmp_path)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    result = await execute_read_public_copy(
        "call-ui-1",
        {"path": UI_CATALOGUE, "key": "wire.slash.credential"},
        None,
        None,
        _context(root, tmp_path),
    )

    assert not result.is_error, result.text
    assert UI_VALUE in result.text
    assert result.details is not None
    assert result.details["public_copy"] is True
    assert result.details["path"] == UI_CATALOGUE


@pytest.mark.asyncio
async def test_a_ui_generated_key_map_is_served(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # HTTPS is the other spelling the identity parser must accept.
    root = _ui_repo(tmp_path, origin=UI_ORIGIN_HTTPS)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    result = await execute_read_public_copy(
        "call-ui-2",
        {"path": UI_KEYS, "key": "wire.slash.credential"},
        None,
        None,
        _context(root, tmp_path),
    )

    assert not result.is_error, result.text
    assert UI_VALUE in result.text
    assert result.details is not None
    assert result.details["path"] == UI_KEYS


@pytest.mark.asyncio
async def test_a_mobile_catalogue_key_is_served(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mobile app's committed copy is public, like the UI's, so it is served."""
    root = _mobile_repo(tmp_path)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    result = await execute_read_public_copy(
        "call-mobile-1",
        {"path": MOBILE_CATALOGUE, "key": "mobile.home.greeting"},
        None,
        None,
        _context(root, tmp_path),
    )

    assert not result.is_error, result.text
    assert MOBILE_VALUE in result.text
    assert result.details is not None
    assert result.details["public_copy"] is True
    assert result.details["path"] == MOBILE_CATALOGUE


@pytest.mark.asyncio
async def test_a_mobile_generated_catalogue_map_is_served(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # HTTPS is the other spelling the identity parser must accept.
    root = _mobile_repo(tmp_path, origin=MOBILE_ORIGIN_HTTPS)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    result = await execute_read_public_copy(
        "call-mobile-2",
        {"path": MOBILE_MAP, "key": "mobile.home.greeting"},
        None,
        None,
        _context(root, tmp_path),
    )

    assert not result.is_error, result.text
    assert MOBILE_VALUE in result.text
    assert result.details is not None
    assert result.details["path"] == MOBILE_MAP


@pytest.mark.asyncio
async def test_a_ui_working_tree_change_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The write-a-secret bypass is refused in the UI checkout too."""
    root = _ui_repo(tmp_path)
    (root / UI_CATALOGUE).write_text(
        json.dumps({"wire.slash.credential": "[redacted]" + FAKE_GITHUB}),
        encoding="utf-8",
    )
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    result = await execute_read_public_copy(
        "call-ui-3",
        {"path": UI_CATALOGUE, "key": "wire.slash.credential"},
        None,
        None,
        _context(root, tmp_path),
    )

    assert result.is_error
    assert "differs from HEAD" in result.text
    assert FAKE_GITHUB not in result.text


@pytest.mark.asyncio
async def test_an_untracked_ui_copy_file_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _ui_repo(tmp_path)
    (root / "src/i18n/catalogues/en/new.json").write_text(
        json.dumps({"wire.slash.new": "public"}), encoding="utf-8"
    )
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    result = await execute_read_public_copy(
        "call-ui-4",
        {"path": "src/i18n/catalogues/en/new.json", "key": "wire.slash.new"},
        None,
        None,
        _context(root, tmp_path),
    )

    assert result.is_error
    assert "not committed at HEAD" in result.text


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["src/i18n/other.json", "src/i18n/catalogues/en/../x.json"])
async def test_a_ui_path_outside_the_allowlist_is_refused(
    path: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _ui_repo(tmp_path)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    result = await execute_read_public_copy(
        "call-ui-5", {"path": path, "key": "wire.other"}, None, None, _context(root, tmp_path)
    )

    assert result.is_error
    assert "not a public-copy path" in result.text


@pytest.mark.asyncio
async def test_an_absolute_ui_path_outside_the_checkout_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _ui_repo(tmp_path)
    elsewhere = tmp_path / "elsewhere" / UI_KEYS
    elsewhere.parent.mkdir(parents=True)
    elsewhere.write_text('"wire.slash.credential": "public"\n', encoding="utf-8")
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    result = await execute_read_public_copy(
        "call-ui-6",
        {"path": str(elsewhere), "key": "wire.slash.credential"},
        None,
        None,
        _context(root, tmp_path),
    )

    assert result.is_error
    assert "not inside this session's workspace" in result.text


@pytest.mark.asyncio
async def test_a_src_shaped_tree_in_another_checkout_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S-1: a ``src/i18n/`` tree is not the UI checkout until its origin says so.

    The reproduced bypass: any repository containing these spellings served an
    unredacted ``sk_live_``-shaped value.  The fixture is otherwise identical to
    the accepted UI repo, so only the origin differs.
    """
    root = _ui_repo(tmp_path, origin="git@github.com:damianvtran/other-product.git")
    (root / UI_CATALOGUE).write_text(
        json.dumps({"wire.slash.credential": FAKE_API_KEY}), encoding="utf-8"
    )
    _git(root, "add", "-A")
    _git(root, "-c", "commit.gpgsign=false", "commit", "-qm", "decoy")
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    result = await execute_read_public_copy(
        "call-s1-1",
        {"path": UI_CATALOGUE, "key": "wire.slash.credential"},
        None,
        None,
        _context(root, tmp_path),
    )

    assert result.is_error
    assert "companion UI checkout" in result.text
    assert "damianvtran/local-operator-ui" in result.text
    assert FAKE_API_KEY not in result.text
    assert not (tmp_path / "config" / "audit" / "public-copy-reads.jsonl").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "origin",
    [
        None,
        # A lookalike name in another namespace is not the UI repository.
        "git@github.com:someone-else/local-operator-ui.git",
    ],
    ids=["no-origin", "lookalike-owner"],
)
async def test_a_ui_tree_without_the_ui_identity_is_refused(
    origin: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _ui_repo(tmp_path, origin=origin)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    result = await execute_read_public_copy(
        "call-s1-2",
        {"path": UI_CATALOGUE, "key": "wire.slash.credential"},
        None,
        None,
        _context(root, tmp_path),
    )

    assert result.is_error
    assert "companion UI checkout" in result.text
    assert UI_VALUE not in result.text


#: A credential a clone URL can carry in its userinfo.  Built from parts so no secret
#: scanner reads the literal as a real token; the test only needs a string to search for.
_ORIGIN_CREDENTIAL = "origin" + "-userinfo-" + "credential"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("maker", "path"),
    [
        (_ui_repo, UI_CATALOGUE),
        (_mobile_repo, MOBILE_MAP),
        (_mobile_repo, UI_KEYS),
        (_ui_repo, MOBILE_MAP),
    ],
    ids=[
        "shared-root-from-ui",
        "mobile-map-from-mobile",
        "ui-map-from-mobile",
        "mobile-map-from-ui",
    ],
)
@pytest.mark.parametrize(
    "origin",
    [
        f"https://user:{_ORIGIN_CREDENTIAL}@github.com/someone/else.git",
        f"https://{_ORIGIN_CREDENTIAL}@github.com/someone/else",
    ],
    ids=["user-and-password", "token-as-user"],
)
async def test_the_refusal_never_echoes_a_credential_bearing_origin_url(
    maker, path: str, origin: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal sentence reaches the model, so it must not carry the raw origin.

    An origin URL can embed a credential in its userinfo.  The refusal names only
    the parsed ``owner/repo``; a mutation that interpolated the raw URL into the
    sentence left every other test green, so the "never echo it" promise was
    unpinned.  Every refusing root is covered, including the shared one whose
    sentence names two identities.
    """
    root = maker(tmp_path, origin=origin)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    result = await execute_read_public_copy(
        "call-origin-echo",
        {"path": path, "key": "wire.slash.credential"},
        None,
        None,
        _context(root, tmp_path),
    )

    assert result.is_error
    assert "someone/else" in result.text  # the parsed identity is what is named
    assert _ORIGIN_CREDENTIAL not in result.text
    assert origin not in result.text
    assert "github.com" not in result.text


@pytest.mark.asyncio
async def test_a_mobile_checkout_refuses_the_ui_generated_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mobile identity does not unlock the UI's map, even when committed.

    The fixture commits ``src/i18n/keys.gen.ts`` holding a secret-shaped value,
    so a refusal here cannot be the file being absent: it is the identity gate
    keeping the two consumer checkouts apart.  A widened shared identity ("any
    consumer checkout may serve any consumer root") would serve it.
    """
    root = _mobile_repo(tmp_path)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    result = await execute_read_public_copy(
        "call-mobile-3",
        {"path": UI_KEYS, "key": "wire.slash.credential"},
        None,
        None,
        _context(root, tmp_path),
    )

    assert result.is_error
    assert "companion UI checkout" in result.text
    assert UI_REPOSITORY in result.text
    assert FAKE_API_KEY not in result.text
    assert not (tmp_path / "config" / "audit" / "public-copy-reads.jsonl").exists()


@pytest.mark.asyncio
async def test_a_mobile_checkout_has_no_route_to_a_core_catalogue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mobile identity adds its own roots only — never this repository's.

    Core roots carry no identity gate: they are gated by being committed in the
    checkout, so a mobile checkout has no route to a core catalogue it does not
    itself commit.  :func:`test_the_consumer_identity_binding_is_per_checkout`
    pins the structural half — no core root names a consumer identity.
    """
    root = _mobile_repo(tmp_path)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    result = await execute_read_public_copy(
        "call-mobile-4",
        {
            "path": "local_operator/i18n/catalogues/en/wire.slash.json",
            "key": "wire.slash.credential",
        },
        None,
        None,
        _context(root, tmp_path),
    )

    assert result.is_error
    assert "not committed at HEAD" in result.text
    assert not (tmp_path / "config" / "audit" / "public-copy-reads.jsonl").exists()


@pytest.mark.asyncio
async def test_a_core_checkout_does_not_serve_a_mobile_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Symmetric: this repository's own roots cannot borrow the mobile map.

    The core fixture carries no origin, which is the realistic case for a clone
    an auditor is reading from: a ``src/i18n/`` root is refused as unowned
    rather than served, even though the file is committed there.
    """
    root = _public_repo(tmp_path)
    (root / MOBILE_MAP).parent.mkdir(parents=True, exist_ok=True)
    (root / MOBILE_MAP).write_text(
        "export const catalogue = {\n"
        f'  "mobile.home.greeting": "{FAKE_API_KEY}",\n'
        "} as const;\n",
        encoding="utf-8",
    )
    _git(root, "add", "-A")
    _git(root, "-c", "commit.gpgsign=false", "commit", "-qm", "mobile map decoy")
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    result = await execute_read_public_copy(
        "call-mobile-5",
        {"path": MOBILE_MAP, "key": "mobile.home.greeting"},
        None,
        None,
        _context(root, tmp_path),
    )

    assert result.is_error
    assert "mobile app checkout" in result.text
    assert MOBILE_REPOSITORY in result.text
    assert FAKE_API_KEY not in result.text
    assert not (tmp_path / "config" / "audit" / "public-copy-reads.jsonl").exists()


@pytest.mark.asyncio
async def test_a_workspace_below_the_checkout_root_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S-5: ``show HEAD:<path>`` is toplevel-relative, ``ls-files`` is cwd-relative.

    The measured split: a workspace that is a subdirectory of a larger checkout
    validates the workspace-relative file with ``ls-files``/``diff`` but serves
    the toplevel-relative blob of the same spelling.  The fixture puts a secret
    in the toplevel file and public copy in the workspace file.
    """
    parent = tmp_path / "monorepo"
    workspace = parent / "sub"
    workspace_catalogue = workspace / "local_operator/i18n/catalogues/en"
    workspace_catalogue.mkdir(parents=True)
    (workspace_catalogue / "wire.slash.json").write_text(
        json.dumps({"wire.slash.credential": "workspace copy"}), encoding="utf-8"
    )
    top_catalogue = parent / "local_operator/i18n/catalogues/en"
    top_catalogue.mkdir(parents=True)
    (top_catalogue / "wire.slash.json").write_text(
        json.dumps({"wire.slash.credential": FAKE_API_KEY}), encoding="utf-8"
    )
    _git(parent, "init", "-q")
    _git(parent, "config", "user.email", "test@example.com")
    _git(parent, "config", "user.name", "Test")
    _git(parent, "add", "-A")
    _git(parent, "-c", "commit.gpgsign=false", "commit", "-qm", "monorepo")
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    result = await execute_read_public_copy(
        "call-s5-1",
        {
            "path": "local_operator/i18n/catalogues/en/wire.slash.json",
            "key": "wire.slash.credential",
        },
        None,
        None,
        _context(workspace, tmp_path),
    )

    assert result.is_error
    assert "not the checkout root" in result.text
    assert FAKE_API_KEY not in result.text


@pytest.mark.asyncio
async def test_a_replace_ref_cannot_change_the_served_blob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S-4: ``git replace`` must not substitute the blob the route serves."""
    root = _public_repo(tmp_path)
    catalogue = "local_operator/i18n/catalogues/en/wire.slash.json"
    replacement = json.dumps({"wire.slash.credential": FAKE_GITHUB}).encode("utf-8")
    replacement_blob = (
        subprocess.run(
            ["git", "-C", str(root), "hash-object", "-w", "--stdin"],
            input=replacement,
            check=True,
            capture_output=True,
        )
        .stdout.strip()
        .decode()
    )
    original_blob = subprocess.run(
        ["git", "-C", str(root), "rev-parse", f"HEAD:{catalogue}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _git(root, "replace", original_blob, replacement_blob)
    plain = subprocess.run(
        ["git", "-C", str(root), "show", f"HEAD:{catalogue}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    # The control: without the flag, this checkout really does serve the
    # replacement — so the passing read below is the flag doing its work.
    assert FAKE_GITHUB in plain
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    result = await execute_read_public_copy(
        "call-s4-1",
        {"path": catalogue, "key": "wire.slash.credential"},
        None,
        None,
        _context(root, tmp_path),
    )

    assert not result.is_error, result.text
    assert "Type or paste a secret after a space; masked" in result.text
    assert FAKE_GITHUB not in result.text


@pytest.mark.asyncio
async def test_a_working_tree_change_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reproduced bypass: write a credential into a catalogue, read it back."""
    root = _public_repo(tmp_path)
    catalogue = root / "local_operator/i18n/catalogues/en/wire.slash.json"
    catalogue.write_text(
        json.dumps({"wire.slash.credential": "GITHUB_TOKEN=" + FAKE_GITHUB}),
        encoding="utf-8",
    )
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    result = await execute_read_public_copy(
        "call-1",
        {
            "path": "local_operator/i18n/catalogues/en/wire.slash.json",
            "key": "wire.slash.credential",
        },
        None,
        None,
        _context(root, tmp_path),
    )

    assert result.is_error
    assert "differs from HEAD" in result.text
    assert FAKE_GITHUB not in result.text


@pytest.mark.asyncio
async def test_an_untracked_public_copy_file_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _public_repo(tmp_path)
    new_file = root / "local_operator/i18n/catalogues/en/new.json"
    new_file.write_text(json.dumps({"wire.slash.new": "public"}), encoding="utf-8")
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    result = await execute_read_public_copy(
        "call-2",
        {"path": "local_operator/i18n/catalogues/en/new.json", "key": "wire.slash.new"},
        None,
        None,
        _context(root, tmp_path),
    )

    assert result.is_error
    assert "not committed at HEAD" in result.text


@pytest.mark.asyncio
async def test_a_clean_crlf_checkout_is_served(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Git for Windows' default: CRLF on disk, LF in the blob, clean index.

    Round 2 measured the raw byte-equality version refusing every read here. The
    route now compares with ``git diff --quiet`` and serves the blob, so the
    platform's line endings cannot kill the audit.
    """
    root = _public_repo(tmp_path)
    _git(root, "config", "core.autocrlf", "true")
    catalogue = root / "local_operator/i18n/catalogues/en/wire.slash.json"
    body = json.dumps(
        {"wire.slash.credential": "Type or paste a secret after a space; masked"},
        ensure_ascii=False,
        indent=2,
    )
    catalogue.write_bytes(body.replace("\n", "\r\n").encode("utf-8") + b"\r\n")
    _git(root, "add", "local_operator/i18n/catalogues/en/wire.slash.json")
    _git(root, "-c", "commit.gpgsign=false", "commit", "-qm", "crlf")
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    result = await execute_read_public_copy(
        "call-crlf",
        {
            "path": "local_operator/i18n/catalogues/en/wire.slash.json",
            "key": "wire.slash.credential",
        },
        None,
        None,
        _context(root, tmp_path),
    )

    assert not result.is_error, result.text
    assert "Type or paste a secret after a space; masked" in result.text


@pytest.mark.asyncio
async def test_a_hard_link_does_not_change_the_served_blob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hard link is another name, not different committed bytes.

    Round 2 measured that the old nlink refusal was a false refuse once the
    route serves the HEAD blob: the link cannot change what HEAD holds. The
    test keeps the old fixture (a second link to a clean catalogue) and asserts
    the read still succeeds.
    """
    root = _public_repo(tmp_path)
    catalogue = root / "local_operator/i18n/catalogues/en/wire.slash.json"
    os.link(catalogue, catalogue.with_name("hard.json"))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    result = await execute_read_public_copy(
        "call-3",
        {
            "path": "local_operator/i18n/catalogues/en/wire.slash.json",
            "key": "wire.slash.credential",
        },
        None,
        None,
        _context(root, tmp_path),
    )

    assert not result.is_error, result.text
    assert "Type or paste a secret after a space; masked" in result.text


@pytest.mark.asyncio
async def test_a_non_git_directory_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The route requires a git checkout; the refusal names that, not 'untracked'."""
    root = tmp_path / "plain"
    catalogue = root / "local_operator/i18n/catalogues/en"
    catalogue.mkdir(parents=True)
    (catalogue / "wire.slash.json").write_text(
        json.dumps({"wire.slash.credential": "public"}), encoding="utf-8"
    )
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    result = await execute_read_public_copy(
        "call-3b",
        {
            "path": "local_operator/i18n/catalogues/en/wire.slash.json",
            "key": "wire.slash.credential",
        },
        None,
        None,
        _context(root, tmp_path),
    )

    assert result.is_error
    assert "not a git checkout" in result.text


@pytest.mark.asyncio
async def test_a_symlink_to_a_non_public_file_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _public_repo(tmp_path)
    catalogue = root / "local_operator/i18n/catalogues/en/wire.slash.json"
    catalogue.with_name("link.json").symlink_to(root / "README.md")
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    result = await execute_read_public_copy(
        "call-4",
        {"path": "local_operator/i18n/catalogues/en/link.json", "key": "x"},
        None,
        None,
        _context(root, tmp_path),
    )

    assert result.is_error
    assert "not a public-copy path" in result.text


@pytest.mark.asyncio
async def test_an_unkeyed_oversized_catalogue_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _public_repo(tmp_path)
    catalogue = root / "local_operator/i18n/catalogues/en/big.json"
    catalogue.write_text(
        json.dumps({f"wire.big.{index}": "x" * 100 for index in range(2100)}),
        encoding="utf-8",
    )
    _git(root, "add", "local_operator/i18n/catalogues/en/big.json")
    _git(root, "-c", "commit.gpgsign=false", "commit", "-qm", "big")
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    result = await execute_read_public_copy(
        "call-5",
        {"path": "local_operator/i18n/catalogues/en/big.json"},
        None,
        None,
        _context(root, tmp_path),
    )

    assert result.is_error
    assert "larger than" in result.text


@pytest.mark.asyncio
async def test_an_empty_key_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _public_repo(tmp_path)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    result = await execute_read_public_copy(
        "call-6",
        {"path": "local_operator/i18n/keys/wire_slash.py", "key": "   "},
        None,
        None,
        _context(root, tmp_path),
    )

    assert result.is_error
    assert "key must not be empty" in result.text


@pytest.mark.asyncio
async def test_every_read_appends_and_the_audit_is_not_truncated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _public_repo(tmp_path)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    context = _context(root, tmp_path)
    request = {
        "path": "local_operator/i18n/catalogues/en/wire.slash.json",
        "key": "wire.slash.credential",
    }

    first = await execute_read_public_copy("call-7", request, None, None, context)
    second = await execute_read_public_copy("call-8", request, None, None, context)

    assert not first.is_error, first.text
    assert not second.is_error, second.text
    log = tmp_path / "config" / "audit" / "public-copy-reads.jsonl"
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(records) == 2
    assert records[0]["sha256"] == records[1]["sha256"]
    assert os.stat(log).st_mode & 0o777 == 0o600
    assert os.stat(log.parent).st_mode & 0o777 == 0o700


@pytest.mark.asyncio
async def test_a_symlinked_audit_log_refuses_the_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S-3: a log symlink (measured with ``/dev/null``) suppressed the record."""
    root = _public_repo(tmp_path)
    audit = tmp_path / "config" / "audit"
    audit.mkdir(parents=True)
    redirect = tmp_path / "redirect.jsonl"
    redirect.write_text("", encoding="utf-8")
    (audit / "public-copy-reads.jsonl").symlink_to(redirect)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    result = await execute_read_public_copy(
        "call-s3-1",
        {
            "path": "local_operator/i18n/catalogues/en/wire.slash.json",
            "key": "wire.slash.credential",
        },
        None,
        None,
        _context(root, tmp_path),
    )

    assert result.is_error
    assert "audit record could not be written" in result.text
    assert "Type or paste a secret" not in result.text
    assert redirect.read_text() == ""


@pytest.mark.asyncio
async def test_a_symlinked_audit_dir_refuses_the_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S-3: a symlinked ``audit`` directory redirected the records."""
    root = _public_repo(tmp_path)
    config = tmp_path / "config"
    config.mkdir()
    redirected = tmp_path / "redirected-audit"
    redirected.mkdir()
    (config / "audit").symlink_to(redirected, target_is_directory=True)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))

    result = await execute_read_public_copy(
        "call-s3-2",
        {
            "path": "local_operator/i18n/catalogues/en/wire.slash.json",
            "key": "wire.slash.credential",
        },
        None,
        None,
        _context(root, tmp_path),
    )

    assert result.is_error
    assert "audit record could not be written" in result.text
    assert "Type or paste a secret" not in result.text
    assert list(redirected.iterdir()) == []


@pytest.mark.asyncio
@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits are not the Windows control")
async def test_an_existing_permissive_audit_log_is_tightened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S-3: an existing 0644 log is made 0600 (and the directory 0700)."""
    root = _public_repo(tmp_path)
    audit = tmp_path / "config" / "audit"
    audit.mkdir(parents=True)
    log = audit / "public-copy-reads.jsonl"
    log.write_text("", encoding="utf-8")
    os.chmod(log, 0o644)
    os.chmod(audit, 0o755)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    result = await execute_read_public_copy(
        "call-s3-3",
        {
            "path": "local_operator/i18n/catalogues/en/wire.slash.json",
            "key": "wire.slash.credential",
        },
        None,
        None,
        _context(root, tmp_path),
    )

    assert not result.is_error, result.text
    assert os.stat(log).st_mode & 0o777 == 0o600
    assert os.stat(audit).st_mode & 0o777 == 0o700
    assert len(log.read_text().splitlines()) == 1


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFOs are POSIX-only")
def test_a_fifo_at_the_audit_path_refuses_without_blocking(
    tmp_path: Path,
) -> None:
    """S-8: a FIFO at the log path must fail the open, not block it.

    The open runs in a worker thread, so a regression would hang the pytest
    worker with no way for the test to fail.  Run the real tool in a child
    process instead: ``subprocess.run(timeout=…)`` kills it and reports a
    failure if the open blocks again.  Without ``O_NONBLOCK`` the measured
    behaviour was no return within 6 s.
    """
    root = _public_repo(tmp_path)
    audit = tmp_path / "config" / "audit"
    audit.mkdir(parents=True)
    os.mkfifo(audit / "public-copy-reads.jsonl")
    script = (
        "import asyncio, sys\n"
        "from local_operator.harness.types import ToolContext\n"
        "from local_operator.tools.public_copy import execute_read_public_copy\n"
        "ctx = ToolContext(cwd=sys.argv[1], session_dir=sys.argv[2])\n"
        "r = asyncio.run(execute_read_public_copy(sys.argv[3], {\n"
        "    'path': 'local_operator/i18n/catalogues/en/wire.slash.json',\n"
        "    'key': 'wire.slash.credential',\n"
        "}, None, None, ctx))\n"
        "print('REFUSED' if r.is_error else 'SERVED')\n"
        "print(r.text)\n"
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(root),
            str(tmp_path / "sessions" / "abc123"),
            "call-s8-1",
        ],
        capture_output=True,
        text=True,
        timeout=20,
        env=dict(os.environ, LOCAL_OPERATOR_CONFIG_DIR=str(tmp_path / "config")),
    )

    assert proc.returncode == 0, proc.stderr
    assert "REFUSED" in proc.stdout
    assert "audit record could not be written" in proc.stdout
    assert "Type or paste a secret" not in proc.stdout


# --- ``repo_root``: reading copy from a NAMED checkout -----------------------
#
# An independent translation audit reads a SPECIFIC branch from a checkout the
# session is not standing in — the session cwd is wherever the agent is, often
# the home directory.  ``repo_root`` names that directory; the directory is
# admitted only when its git origin is one of the publishing repositories.  The
# tests below pin both halves: the named tree is the one read, and every
# neighbouring directory is refused by name.


@pytest.mark.asyncio
async def test_a_named_checkout_serves_that_checkouts_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two distinct committed trees, different bytes for the same key.

    The assertion cannot pass by accident: a read that quietly fell back to the
    session's own workspace returns the OTHER string, which is asserted absent.
    """
    workspace = _public_repo(
        tmp_path, name="workspace", value="the workspace's copy", origin=CORE_ORIGIN_SSH
    )
    audited = _public_repo(
        tmp_path, name="audited", value="the audited branch's copy", origin=CORE_ORIGIN_SSH
    )
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    context = _context(workspace, tmp_path)
    request = {"path": CORE_CATALOGUE, "key": "wire.slash.credential"}

    named = await execute_read_public_copy(
        "call-root-1", {**request, "repo_root": str(audited)}, None, None, context
    )

    assert not named.is_error, named.text
    assert "the audited branch's copy" in named.text
    assert "the workspace's copy" not in named.text

    # CONTROL: the identical request without ``repo_root`` still reads the
    # session workspace, so the difference above is the named checkout itself.
    own = await execute_read_public_copy("call-root-2", request, None, None, context)
    assert not own.is_error, own.text
    assert "the workspace's copy" in own.text


@pytest.mark.asyncio
async def test_a_named_consumer_checkout_still_carries_its_identity_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The per-checkout binding survives a named root: the UI's catalogue is
    served from it, the mobile app's generated map is still refused."""
    workspace = _public_repo(tmp_path, origin=CORE_ORIGIN_SSH)
    ui = _ui_repo(tmp_path)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    context = _context(workspace, tmp_path)

    served = await execute_read_public_copy(
        "call-root-ui",
        {"path": UI_CATALOGUE, "key": "wire.slash.credential", "repo_root": str(ui)},
        None,
        None,
        context,
    )
    assert not served.is_error, served.text
    assert UI_VALUE in served.text

    other_app = await execute_read_public_copy(
        "call-root-ui-mobile-map",
        {"path": MOBILE_MAP, "key": "mobile.home.greeting", "repo_root": str(ui)},
        None,
        None,
        context,
    )
    assert other_app.is_error
    assert "mobile app checkout's copy" in other_app.text


@pytest.mark.asyncio
async def test_a_traversal_out_of_the_named_checkout_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _public_repo(tmp_path, name="workspace", origin=CORE_ORIGIN_SSH)
    audited = _public_repo(tmp_path, name="audited", origin=CORE_ORIGIN_SSH)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    result = await execute_read_public_copy(
        "call-root-escape-1",
        {"path": "../../../etc/passwd", "repo_root": str(audited)},
        None,
        None,
        _context(workspace, tmp_path),
    )

    assert result.is_error
    assert "the path is not inside the named checkout" in result.text


@pytest.mark.asyncio
async def test_an_absolute_path_outside_the_named_checkout_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _public_repo(tmp_path, name="workspace", origin=CORE_ORIGIN_SSH)
    audited = _public_repo(tmp_path, name="audited", origin=CORE_ORIGIN_SSH)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    result = await execute_read_public_copy(
        "call-root-escape-2",
        {"path": "/etc/passwd", "repo_root": str(audited)},
        None,
        None,
        _context(workspace, tmp_path),
    )

    assert result.is_error
    assert "the path is not inside the named checkout" in result.text


@pytest.mark.asyncio
async def test_a_symlink_out_of_the_named_checkout_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A COMMITTED symlink whose allowlist-shaped name resolves outside.

    Committed deliberately: the refusal must be the confinement, not a missing
    file at ``HEAD``, so the negative control is a link git tracks and the
    reader would otherwise serve.
    """
    workspace = _public_repo(tmp_path, name="workspace", origin=CORE_ORIGIN_SSH)
    audited = _public_repo(tmp_path, name="audited", origin=CORE_ORIGIN_SSH)
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"wire.slash.credential": "outside the tree"}), encoding="utf-8")
    link = audited / "local_operator/i18n/catalogues/en/escape.json"
    link.symlink_to(outside)
    _git(audited, "add", "-A")
    _git(audited, "-c", "commit.gpgsign=false", "commit", "-qm", "symlinked copy")
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    result = await execute_read_public_copy(
        "call-root-escape-3",
        {
            "path": "local_operator/i18n/catalogues/en/escape.json",
            "key": "wire.slash.credential",
            "repo_root": str(audited),
        },
        None,
        None,
        _context(workspace, tmp_path),
    )

    assert result.is_error
    assert "the path is not inside the named checkout" in result.text
    assert "outside the tree" not in result.text


@pytest.mark.asyncio
async def test_a_named_directory_outside_git_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _public_repo(tmp_path, name="workspace", origin=CORE_ORIGIN_SSH)
    plain = tmp_path / "plain"
    plain.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    result = await execute_read_public_copy(
        "call-root-plain",
        {"path": CORE_CATALOGUE, "key": "wire.slash.credential", "repo_root": str(plain)},
        None,
        None,
        _context(workspace, tmp_path),
    )

    assert result.is_error
    assert f"repo_root {plain.resolve()} is not inside a git working tree" in result.text


@pytest.mark.asyncio
async def test_a_named_checkout_without_an_origin_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _public_repo(tmp_path, name="workspace", origin=CORE_ORIGIN_SSH)
    originless = _public_repo(tmp_path, name="originless")
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    result = await execute_read_public_copy(
        "call-root-originless",
        {"path": CORE_CATALOGUE, "key": "wire.slash.credential", "repo_root": str(originless)},
        None,
        None,
        _context(workspace, tmp_path),
    )

    assert result.is_error
    assert f"repo_root {originless.resolve()} has no usable git origin" in result.text
    assert "must be a clone of one of: " + ", ".join(PUBLISHING_REPOSITORIES) in result.text


@pytest.mark.asyncio
async def test_a_named_checkout_of_another_repository_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _public_repo(tmp_path, name="workspace", origin=CORE_ORIGIN_SSH)
    lookalike = _public_repo(tmp_path, name="lookalike", origin="git@github.com:someone/else.git")
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    result = await execute_read_public_copy(
        "call-root-lookalike",
        {"path": CORE_CATALOGUE, "key": "wire.slash.credential", "repo_root": str(lookalike)},
        None,
        None,
        _context(workspace, tmp_path),
    )

    assert result.is_error
    assert f"repo_root {lookalike.resolve()} is a checkout of someone/else" in result.text


@pytest.mark.asyncio
async def test_a_named_symlink_or_a_nul_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _public_repo(tmp_path, name="workspace", origin=CORE_ORIGIN_SSH)
    real = _public_repo(tmp_path, name="real", origin=CORE_ORIGIN_SSH)
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    context = _context(workspace, tmp_path)

    linked = await execute_read_public_copy(
        "call-root-link",
        {"path": CORE_CATALOGUE, "key": "wire.slash.credential", "repo_root": str(link)},
        None,
        None,
        context,
    )
    assert linked.is_error
    assert f"repo_root {link} is a symlink; name the checkout directory itself" in linked.text

    nul = await execute_read_public_copy(
        "call-root-nul",
        {"path": CORE_CATALOGUE, "key": "wire.slash.credential", "repo_root": "a\x00b"},
        None,
        None,
        context,
    )
    assert nul.is_error
    assert "repo_root must not contain NUL" in nul.text


@pytest.mark.asyncio
async def test_an_omitted_repo_root_still_reads_the_session_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REGRESSION: the identity admission is scoped to a NAMED checkout only.

    The fixture has NO origin — an ordinary cloned workspace or worktree of an
    unrelated repository is still the session's own tree, and the tool must keep
    serving it without asking for a git identity it never needed.
    """
    workspace = _public_repo(tmp_path, name="workspace", value="the workspace's copy")
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    result = await execute_read_public_copy(
        "call-root-absent",
        {"path": CORE_CATALOGUE, "key": "wire.slash.credential"},
        None,
        None,
        _context(workspace, tmp_path),
    )

    assert not result.is_error, result.text
    assert "the workspace's copy" in result.text
    assert result.details is not None
    assert "checkout" not in result.details
    records = [
        json.loads(line)
        for line in (tmp_path / "config" / "audit" / "public-copy-reads.jsonl")
        .read_text()
        .splitlines()
    ]
    assert records[-1]["path"] == CORE_CATALOGUE
    assert "checkout" not in records[-1]
    assert "checkout_root" not in records[-1]


@pytest.mark.asyncio
async def test_the_audit_record_names_the_checkout_that_was_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A receipt says WHICH tree served the bytes.

    Identity alone cannot: two clones of the same repository share it, which is
    exactly the pair above, so the record carries the root path as well.
    """
    workspace = _public_repo(
        tmp_path, name="workspace", value="the workspace's copy", origin=CORE_ORIGIN_SSH
    )
    audited = _public_repo(
        tmp_path, name="audited", value="the audited branch's copy", origin=CORE_ORIGIN_SSH
    )
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    context = _context(workspace, tmp_path)
    request = {"path": CORE_CATALOGUE, "key": "wire.slash.credential"}

    named = await execute_read_public_copy(
        "call-root-audit-1", {**request, "repo_root": str(audited)}, None, None, context
    )
    own = await execute_read_public_copy("call-root-audit-2", request, None, None, context)
    assert not named.is_error, named.text
    assert not own.is_error, own.text

    records = [
        json.loads(line)
        for line in (tmp_path / "config" / "audit" / "public-copy-reads.jsonl")
        .read_text()
        .splitlines()
    ]
    assert len(records) == 2
    named_record, own_record = records
    assert named_record["checkout"] == CORE_REPOSITORY
    assert Path(named_record["checkout_root"]) == audited.resolve()
    assert named_record["path"] == CORE_CATALOGUE
    assert named_record["policy"] == own_record["policy"]
    # The session-workspace read keeps exactly the fields it always had.
    assert "checkout" not in own_record
    assert "checkout_root" not in own_record
    assert named.details is not None
    assert named.details["checkout"] == CORE_REPOSITORY
    assert Path(named.details["checkout_root"]) == audited.resolve()


# --- round-1 remediation: what the review measured, pinned ------------------
#
# M-1: admission shells out to `git`; it must not run on the harness event loop.
# m-2: a relative `repo_root` is ambiguous (it resolves against the PROCESS cwd).
# m-3: the receipt must name the revision it served, or it cannot be reconciled.
# m-4: the recorded identity must be true at the moment the bytes are served.
# n-1: a named subdirectory must be told to name the checkout root, not to "run
#      from the checkout root" — a thing a named-root caller cannot do.


async def _with_heartbeat(call: Any, *, interval: float = 0.01) -> tuple[Any, float]:
    """Await ``call()`` while sampling the loop; return the result and the worst gap.

    A blocking probe inside the awaited call shows up here as a gap of one git
    invocation, because nothing else on the loop runs while it blocks.
    """
    gaps: list[float] = []
    stop = asyncio.Event()
    last = time.monotonic()

    async def beat() -> None:
        nonlocal last
        while not stop.is_set():
            await asyncio.sleep(interval)
            now = time.monotonic()
            gaps.append(now - last)
            last = now

    task = asyncio.create_task(beat())
    try:
        result = await call()
    finally:
        stop.set()
        await task
    return result, max(gaps or [0.0])


@pytest.mark.asyncio
async def test_admitting_a_checkout_does_not_freeze_the_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M-1: a stalled ``git`` stalls the loop only if the probe runs inline.

    Every `git` invocation is slowed by the stub below, so an inline admission
    probe shows one heartbeat gap per call; the shipped route admits on a worker
    thread and the loop keeps ticking. This is the measurement the review made,
    pinned as a test: the gap is the verdict, not the wall time.
    """
    workspace = _public_repo(tmp_path, name="workspace", origin=CORE_ORIGIN_SSH)
    audited = _public_repo(tmp_path, name="audited", origin=CORE_ORIGIN_SSH)
    real_git = shutil.which("git")
    assert real_git, "this test needs a real git on PATH"
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    stub = stub_dir / "git"
    stub.write_text(f'#!/bin/sh\nsleep 0.8\nexec "{real_git}" "$@"\n', encoding="utf-8")
    stub.chmod(0o755)
    monkeypatch.setenv("PATH", str(stub_dir) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    context = _context(workspace, tmp_path)

    async def call() -> Any:
        return await execute_read_public_copy(
            "call-loop",
            {
                "path": CORE_CATALOGUE,
                "key": "wire.slash.credential",
                "repo_root": str(audited),
            },
            None,
            None,
            context,
        )

    result, gap = await _with_heartbeat(call)

    assert not result.is_error, result.text
    assert gap < 0.4, (
        f"the event loop stalled for {gap:.2f}s while admitting the checkout; "
        "admission must run off the loop (asyncio.to_thread)"
    )


@pytest.mark.asyncio
async def test_a_relative_repo_root_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """m-2: a relative spelling would resolve against the PROCESS cwd."""
    workspace = _public_repo(tmp_path, name="workspace", origin=CORE_ORIGIN_SSH)
    audited = _public_repo(tmp_path, name="audited", origin=CORE_ORIGIN_SSH)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    context = _context(workspace, tmp_path)
    request = {"path": CORE_CATALOGUE, "key": "wire.slash.credential"}

    relative = await execute_read_public_copy(
        "call-relative", {**request, "repo_root": "audited"}, None, None, context
    )
    assert relative.is_error
    assert "repo_root must be an absolute path" in relative.text
    assert "harness process's working directory" in relative.text

    # The same checkout, named absolutely, is admitted — the refusal is about the
    # spelling, not the tree.
    absolute = await execute_read_public_copy(
        "call-absolute", {**request, "repo_root": str(audited)}, None, None, context
    )
    assert not absolute.is_error, absolute.text


@pytest.mark.asyncio
async def test_the_receipt_records_the_revision_it_served(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """m-3: identity + path cannot distinguish two commits of one worktree."""
    workspace = _public_repo(tmp_path, name="workspace", origin=CORE_ORIGIN_SSH)
    audited = _public_repo(tmp_path, name="audited", value="first revision", origin=CORE_ORIGIN_SSH)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    context = _context(workspace, tmp_path)
    request = {
        "path": CORE_CATALOGUE,
        "key": "wire.slash.credential",
        "repo_root": str(audited),
    }

    first = await execute_read_public_copy("call-head-1", request, None, None, context)
    first_head = subprocess.run(
        ["git", "-C", str(audited), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    (audited / CORE_CATALOGUE).write_text(
        json.dumps({"wire.slash.credential": "second revision"}), encoding="utf-8"
    )
    _git(audited, "add", "-A")
    _git(audited, "-c", "commit.gpgsign=false", "commit", "-qm", "second revision")
    second = await execute_read_public_copy("call-head-2", request, None, None, context)
    second_head = subprocess.run(
        ["git", "-C", str(audited), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    assert not first.is_error, first.text
    assert not second.is_error, second.text
    assert first_head != second_head
    assert first.details is not None and second.details is not None
    assert first.details["checkout_head"] == first_head
    assert second.details["checkout_head"] == second_head
    assert "second revision" in second.text

    records = [
        json.loads(line)
        for line in (tmp_path / "config" / "audit" / "public-copy-reads.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [record["checkout_head"] for record in records] == [first_head, second_head]
    # The path and the identity are IDENTICAL across the two reads: only the
    # revision and the body digest distinguish them, which is the finding.
    assert {record["path"] for record in records} == {CORE_CATALOGUE}
    assert {record["checkout"] for record in records} == {CORE_REPOSITORY}
    assert records[0]["sha256"] != records[1]["sha256"]


@pytest.mark.asyncio
async def test_a_checkout_whose_origin_changed_after_admission_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """m-4: the identity the receipt records is re-derived where the bytes are read.

    The stale admission below is exactly what a directory swapped between the
    admission probe and the read looks like to the reader. A receipt whose
    identity came only from the pre-read probe would serve this read and record
    an identity the serving repository does not have.
    """
    workspace = _public_repo(tmp_path, name="workspace", origin=CORE_ORIGIN_SSH)
    audited = _public_repo(tmp_path, name="audited", origin=CORE_ORIGIN_SSH)

    def stale_admission(raw: str) -> tuple[Path, str, None]:
        return audited.resolve(), UI_REPOSITORY, None

    monkeypatch.setattr(public_copy_module, "_admitted_checkout", stale_admission)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    result = await execute_read_public_copy(
        "call-swap",
        {
            "path": CORE_CATALOGUE,
            "key": "wire.slash.credential",
            "repo_root": str(audited),
        },
        None,
        None,
        _context(workspace, tmp_path),
    )

    assert result.is_error
    assert f"the origin of {audited.resolve()} is now damianvtran/local-operator" in result.text
    assert f"not the {UI_REPOSITORY} this call admitted" in result.text
    assert "record could not be written" not in result.text
    assert not (tmp_path / "config" / "audit" / "public-copy-reads.jsonl").exists()


@pytest.mark.asyncio
async def test_a_named_subdirectory_is_refused_with_actionable_advice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """n-1: the caller named a checkout, so "run from the checkout root" is advice
    they cannot act on. The copy root below EXISTS and is committed; the refusal
    is the toplevel rule, and it must say "the named checkout root"."""
    workspace = _public_repo(tmp_path, name="workspace", origin=CORE_ORIGIN_SSH)
    root = tmp_path / "outer"
    (root / "sub" / "local_operator/i18n/catalogues/en").mkdir(parents=True)
    (root / "sub" / CORE_CATALOGUE).write_text(
        json.dumps({"wire.slash.credential": "a committed copy below the root"}),
        encoding="utf-8",
    )
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")
    _git(root, "remote", "add", "origin", CORE_ORIGIN_SSH)
    _git(root, "add", "-A")
    _git(root, "-c", "commit.gpgsign=false", "commit", "-qm", "fixture")
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    result = await execute_read_public_copy(
        "call-subdir",
        {
            "path": CORE_CATALOGUE,
            "key": "wire.slash.credential",
            "repo_root": str(root / "sub"),
        },
        None,
        None,
        _context(workspace, tmp_path),
    )

    assert result.is_error
    assert "the named checkout root" in result.text
    assert "not one below it" in result.text
    assert "run read_public_copy from the checkout root" not in result.text
    assert "workspace" not in result.text
    assert "a committed copy below the root" not in result.text
