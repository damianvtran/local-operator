"""Catalogue loader: names, hashes, traversal refusal, shape validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from local_operator.i18n import catalogues, runtime

#: The `wire.*` namespaces still awaiting their extraction slice: stub files
#: that M0 shipped and no slice has populated yet (RFC §5: later slices
#: populate them). An extraction slice removes its namespace from this tuple
#: in its own PR, so the set shrinks until P5 and the file-list assertion
#: below stays the part that must never change.
_PENDING_EXTRACTION = (
    "wire.errors",
    "wire.incidents",
    "wire.notices",
)


def test_shipped_wire_namespaces_exist() -> None:
    # M0 shipped the six `wire.*` namespace files as STUBS (RFC §5): the files
    # exist so later slices populate them. `wire.slash` was populated by the S2
    # extraction slice, `wire.settings` by S1, and `wire.errors.settings` (a
    # CHILD of `wire.errors`, RFC §2.2) by the E1 slice; E2 added
    # `wire.errors.desktop` (also a child of `wire.errors`) the same way,
    # `wire.exec` by E2b, `wire.errors.agents` by E2c, and `wire.errors.media`
    # by E2d; the files slice adds populated `wire.errors.files`. The rest stay empty
    # until their own slices land. The console-capacity slice adds populated
    # `wire.console` (its `wire.console.context` is an en-only context SIDECAR,
    # not a namespace, so it is absent from `namespaces()` by construction).
    # The TUI wave writes `tui.*` files instead —
    # the first (`tui.welcome`, T1) is checked by PRESENCE here and by content
    # in its own slice test, so this list does not churn per TUI slice.
    # The held Chinese candidate lives on a separate remote ref, not in the
    # releasable package. Pin the whole set so partial restoration cannot
    # accidentally admit nine passed groups without its failed slash audit.
    assert catalogues.locales() == ("en", "es", "fr", "hi", "ru", "vi")
    assert {name for name in catalogues.namespaces("en") if name.startswith("wire.")} == {
        "wire.errors",
        "wire.errors.agents",
        "wire.errors.auth",
        "wire.errors.desktop",
        "wire.errors.files",
        "wire.errors.media",
        "wire.errors.sdk_spec",
        "wire.errors.sdk_startup",
        "wire.errors.settings",
        "wire.exec",
        "wire.incidents",
        "wire.notices",
        "wire.settings",
        "wire.settings.hub_defaults",
        # Account-routing policy copy is an approved English-only child, not
        # an edit to the already-audited wire.settings or translated ledgers.
        "wire.settings.account_policy",
        "wire.slash",
        # The Library slice's two slash doors (`/library`, `/librarian`)
        # resolve through this English-only CHILD of `wire.slash`, the
        # `wire.settings.account_policy` shape: their copy must not move the
        # already-audited `wire.slash` locale hashes.
        "wire.slash.library",
        "wire.aida_cadence",
        # The OpenRouter host-speed settings row (English-only, own namespace so
        # the audited wire.settings hash is untouched).
        "wire.host_speed",
        # Populated by the console-capacity slice (this PR), the same way the
        # extraction slices added wire.settings/wire.errors.*/wire.exec.
        "wire.console",
        # The desktop phone-setup refusals (this PR's own namespace).
        "wire.tunnel_setup",
        # Memory settings copy (LOH-25): the registry rows resolve through it.
        "wire.memory",
        # The tier-1 remote-compute CLI copy is its own English-only namespace,
        # the `wire.host_speed` shape: `lop exec --remote` adds keys to `wire.exec`,
        # which fr/hi/vi CARRY, and an English edit to a carried namespace
        # invalidates that locale's ledger row and unships the whole locale.
        "wire.remote_compute",
    }
    for namespace in _PENDING_EXTRACTION:
        assert catalogues.load_catalogue("en", namespace) == {}
    assert catalogues.load_catalogue("en", "wire.errors.agents") != {}
    assert catalogues.load_catalogue("en", "wire.errors.settings") != {}
    assert catalogues.load_catalogue("en", "wire.errors.media") != {}
    assert catalogues.load_catalogue("en", "wire.errors.files") != {}
    # Specification diagnostics are an approved populated English source child,
    # not a stub or an exemption from the exact namespace contract.
    assert catalogues.load_catalogue("en", "wire.errors.sdk_spec") != {}
    assert catalogues.load_catalogue("en", "wire.exec") != {}
    assert catalogues.load_catalogue("en", "wire.remote_compute") != {}
    # Approved source namespace, not a prefix-wide wildcard. Mutation controls
    # below still reject undeclared additions, removals and emptied namespaces.
    assert catalogues.load_catalogue("en", "wire.settings.hub_defaults") != {}
    assert catalogues.load_catalogue("en", "wire.settings.account_policy") != {}
    assert catalogues.load_catalogue("en", "wire.slash") != {}
    assert catalogues.load_catalogue("en", "wire.slash.library") != {}
    assert catalogues.load_catalogue("en", "wire.settings") != {}
    # The Aida cadence interval row is a small English-only addition carried in
    # its own `wire.*` namespace so already-audited `wire.settings` locale
    # hashes stay valid; pinned populated here for the same reason the other
    # populated namespaces are.
    assert catalogues.load_catalogue("en", "wire.aida_cadence") != {}
    # Populated, never a stub, so it is NOT in _PENDING_EXTRACTION.
    assert catalogues.load_catalogue("en", "wire.console") != {}
    assert "tui.welcome" in catalogues.namespaces("en")
    assert catalogues.load_catalogue("en", "tui.welcome") != {}


@pytest.mark.parametrize(
    "target_namespace",
    ["wire.errors.files", "wire.errors.sdk_spec", "wire.settings.account_policy"],
)
@pytest.mark.parametrize("mutation", ["unexpected-namespace", "missing-declared", "empty-declared"])
def test_exact_wire_namespace_contract_rejects_mutations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str, target_namespace: str
) -> None:
    # Exercise the real exact-set/population assertion against synthetic roots,
    # so an extraction cannot silently weaken it to a subset or an empty stub.
    locales = catalogues.locales()
    namespaces = catalogues.namespaces("en")
    for variant in ("pristine", "mutated"):
        root = tmp_path / variant
        for locale in locales:
            directory = root / locale
            directory.mkdir(parents=True)
            if locale != "en":
                (directory / "fixture.json").write_text("{}", encoding="utf-8")
                continue
            for namespace in namespaces:
                if variant == "mutated" and mutation == "missing-declared":
                    if namespace == target_namespace:
                        continue
                empty = namespace in _PENDING_EXTRACTION or (
                    variant == "mutated"
                    and mutation == "empty-declared"
                    and namespace == target_namespace
                )
                body = {} if empty else {"fixture": "populated"}
                (directory / f"{namespace}.json").write_text(json.dumps(body), encoding="utf-8")
            if variant == "mutated" and mutation == "unexpected-namespace":
                (directory / "wire.errors.unexpected.json").write_text("{}", encoding="utf-8")
        monkeypatch.setattr(catalogues, "_CATALOGUES", root)
        if variant == "pristine":
            test_shipped_wire_namespaces_exist()
        else:
            with pytest.raises(AssertionError):
                test_shipped_wire_namespaces_exist()


def test_wave3_hindi_carries_its_namespace_snapshot_with_parity() -> None:
    # Pin only the original ten Hindi groups plus each later reviewed slice.
    # Analytics/desktop backfill is explicitly unfinished; mechanical parity is
    # not full-surface coverage. `tui.review` is the LOH-19 slice's audited
    # namespace, added here with its own ledger row; `tui.schedule` is P1's,
    # carried only now that its audit row exists (the carried-namespace gate
    # un-ships a locale for an unaudited file); `tui.projects_edit` is P2's
    # audited namespace, carried the same way on top of the union. P3a adds
    # `tui.authoring` only after its independent 40-key audit passed. P3b adds
    # `tui.team_authoring` after its independent 35-key audit passed. P4a adds
    # `tui.viewer` after its independent audit and keyboard-hint audit refresh.
    # The remote-compute wave (S3e) adds `cli.exec_remote_help` and
    # `wire.remote_compute`, each carried only once its independent audit row
    # passed against the current en bytes.
    locale = "hi"
    namespaces = (
        "cli.exec_remote_help",
        "tui.authoring",
        "tui.projects_edit",
        "tui.review",
        "tui.schedule",
        "tui.status",
        "tui.team_authoring",
        "tui.transcript",
        "tui.viewer",
        "tui.welcome",
        "wire.errors",
        "wire.errors.settings",
        "wire.exec",
        "wire.incidents",
        "wire.notices",
        "wire.remote_compute",
        "wire.settings",
        "wire.slash",
    )
    assert catalogues.namespaces(locale) == namespaces
    for namespace in namespaces:
        source = catalogues.load_catalogue("en", namespace)
        translated = catalogues.load_catalogue(locale, namespace)
        assert translated.keys() == source.keys(), (locale, namespace)
        for key in source:
            assert runtime.message_arguments(translated[key]) == runtime.message_arguments(
                source[key]
            ), (locale, namespace, key)


def test_hash_is_over_file_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "catalogues"
    (root / "en").mkdir(parents=True)
    raw = b'{\n  "demo.words.hello": "Hello"\n}\n'
    (root / "en" / "demo.words.json").write_bytes(raw)
    monkeypatch.setattr(catalogues, "_CATALOGUES", root)
    assert catalogues.catalogue_sha256("en", "demo.words") == hashlib.sha256(raw).hexdigest()
    # The map is parsed, the hash is NOT: re-serialising the same map would
    # hash differently, which would break the ledger's staleness rule.
    assert catalogues.load_catalogue("en", "demo.words") == {"demo.words.hello": "Hello"}


def test_missing_locale_and_namespace_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "catalogues"
    (root / "en").mkdir(parents=True)
    (root / "en" / "demo.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(catalogues, "_CATALOGUES", root)
    with pytest.raises(catalogues.CatalogueNotFound):
        catalogues.load_catalogue("fr", "demo")
    with pytest.raises(catalogues.CatalogueNotFound):
        catalogues.load_catalogue("en", "nope")


@pytest.mark.parametrize("namespace", ["..", "../etc/passwd", "a/b", ".hidden", "x\x00y"])
def test_traversal_shaped_names_are_refused(namespace: str) -> None:
    with pytest.raises(catalogues.CatalogueNotFound):
        catalogues.load_catalogue("en", namespace)


def test_invalid_shapes_are_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "catalogues"
    (root / "en").mkdir(parents=True)
    (root / "en" / "list.json").write_text("[1, 2]", encoding="utf-8")
    (root / "en" / "value.json").write_text('{"k": 3}', encoding="utf-8")
    monkeypatch.setattr(catalogues, "_CATALOGUES", root)
    with pytest.raises(catalogues.CatalogueInvalid):
        catalogues.load_catalogue("en", "list")
    with pytest.raises(catalogues.CatalogueInvalid):
        catalogues.load_catalogue("en", "value")


def test_malformed_file_raises_catalogue_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Core hardening (S2 M1): decode/parse failures surface as the ONE
    # catalogue exception the render path handles, so a garbled file degrades
    # a render rather than raising a bare ValueError through it.
    root = tmp_path / "catalogues"
    (root / "en").mkdir(parents=True)
    (root / "en" / "demo.json").write_text("{not json", encoding="utf-8")
    (root / "en" / "demo.bad.json").write_bytes(b"\xff\xfe{}")
    monkeypatch.setattr(catalogues, "_CATALOGUES", root)
    with pytest.raises(catalogues.CatalogueInvalid):
        catalogues.load_catalogue("en", "demo")
    with pytest.raises(catalogues.CatalogueInvalid):
        catalogues.load_catalogue("en", "demo.bad")


def test_context_sidecar_is_not_a_namespace_and_reads_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "catalogues"
    (root / "en").mkdir(parents=True)
    (root / "en" / "demo.json").write_text(json.dumps({"demo.x": "y"}), encoding="utf-8")
    sidecar = {"demo.x": {"role": "help text", "maxCells": 43}}
    (root / "en" / "demo.context.json").write_text(json.dumps(sidecar), encoding="utf-8")
    monkeypatch.setattr(catalogues, "_CATALOGUES", root)
    assert catalogues.namespaces("en") == ("demo",)
    assert catalogues.context("en", "demo") == sidecar
    # Absent sidecar is a valid state ({}), not an error.
    assert catalogues.context("en", "missing") == {}
