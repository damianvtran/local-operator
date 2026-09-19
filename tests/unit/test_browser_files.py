"""Policy tests for the browser tool's file transfer (`local_operator/browser_files.py`).

The classifier, the sanitiser and the upload gate are the SECURITY surface of
`download`/`upload`: the approval gate is not the protection (it is one callback
for both tiers, and `--yolo` installs none), so every rule asserted here is one
that runs unconditionally on a path a hostile page or a confused-deputy agent
would use.

The classifier's expectations are not written here a second time: they live in
`browser_files.CONFORMANCE_CASES`, which the TypeScript generator re-derives
before it will emit the shared tables. This module tests the RULES AROUND that
fixture — the caps, the containment, the upload gate, the audit's best-effort
guarantee — plus the fixture's own self-check.
"""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from local_operator import browser_files as bf


@pytest.fixture(autouse=True)
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A config root of our own: `session_dir` and `check_upload` both read it."""
    root = tmp_path / "config"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir(exist_ok=True)
    return root


# --- the sanitiser -----------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("../../etc/passwd", "passwd"),
        ("..\\..\\Windows\\evil.txt", "evil.txt"),
        ("/absolute/report.pdf", "report.pdf"),
        ("re\u0000port\u001b[31m.pdf", "report[31m.pdf"),
        ("report.pdf ", "report.pdf"),
        ("report.pdf...", "report.pdf"),
        ("installer\u202egnp.exe", "installergnp.exe"),
    ],
)
def test_safe_name_keeps_only_a_usable_basename(raw: str, expected: str) -> None:
    assert bf.safe_name(raw) == expected


def test_safe_name_falls_back_for_names_that_cannot_be_used() -> None:
    for raw in (".", "..", "   ", "", "CON.txt", "lpt9"):
        name = bf.safe_name(raw)
        assert name.startswith(bf._FALLBACK_STEM)
        assert "/" not in name and "\\" not in name
        assert name != raw


def test_safe_name_caps_the_byte_length_and_keeps_the_extension() -> None:
    capped = bf.safe_name("a" * 400 + ".pdf")
    assert len(capped.encode("utf-8")) <= bf.MAX_NAME_BYTES
    assert capped.endswith(".pdf")


def test_safe_name_corrects_the_extension_from_the_content() -> None:
    assert bf.safe_name("invoice.zip", sniffed_ext="pdf") == "invoice.pdf"
    assert bf.safe_name("handout", sniffed_ext="zip") == "handout.zip"
    # A name that sanitised to nothing still takes the sniffed extension, and the
    # generated stem is the ONLY thing that can name it.
    corrected = bf.safe_name("   ", sniffed_ext="pdf")
    assert corrected.startswith(bf._FALLBACK_STEM) and corrected.endswith(".pdf")


def test_a_refused_name_is_redacted_for_the_audit_row() -> None:
    assert bf.redact_name("id_rsa") == "i\u2026"
    assert bf.redact_name("") == ""


# --- the classifier, against the shared fixture ------------------------------


def test_the_hand_written_fixture_matches_the_classifier() -> None:
    """The generator's own gate, asserted here so a local run sees it too.

    `gen_ts` refuses to WRITE when this is non-empty; failing here means the
    generated tables in the tree describe behaviour the classifier no longer has.
    """
    assert bf.fixture_mismatches() == []


def test_classify_download_reads_the_file_it_is_given(tmp_path: Path) -> None:
    pdf = tmp_path / "receipt.pdf"
    pdf.write_bytes(b"%PDF-1.4\n1 0 obj\n%%EOF\n")
    verdict = bf.classify_download(pdf)
    assert (verdict.kind, verdict.sniffed) == ("allow", "pdf")

    exe = tmp_path / "holiday.jpg"
    exe.write_bytes(b"MZ\x90\x00\x03\x00\x00\x00")
    denied = bf.classify_download(exe)
    assert denied.kind == "deny" and denied.sniffed == "pe"
    assert "executable" in denied.reason


def test_classify_download_refuses_a_file_over_the_cap(tmp_path: Path) -> None:
    """The cap is a refusal, not a truncation: nothing partial is ever kept."""
    small = bf.Policy(download_max_bytes=8)
    big = tmp_path / "big.pdf"
    big.write_bytes(b"%PDF-1.4 and then some more bytes")
    verdict = bf.classify_download(big, policy=small)
    assert verdict.kind == "deny"
    assert "over the 8 byte limit" in verdict.reason


def test_a_dmg_is_caught_by_its_footer(tmp_path: Path) -> None:
    """`koly` closes the file, so the head alone cannot identify it."""
    image = tmp_path / "installer.dmg"
    image.write_bytes(b"\x00" * 4096 + b"koly")
    verdict = bf.classify_download(image)
    assert verdict.kind == "deny" and verdict.sniffed == "dmg"


def test_a_truncated_head_still_classifies_by_signature(tmp_path: Path) -> None:
    elf = tmp_path / "notes"
    elf.write_bytes(b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 4096)
    assert bf.classify_download(elf).sniffed == "elf"


# --- the quarantine root -----------------------------------------------------


def test_the_session_directory_is_private_and_timestamped() -> None:
    directory = bf.session_dir("abcdef0123456789")
    assert directory.is_dir()
    assert directory.name.startswith("20") and directory.name.endswith("-abcdef01")
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert stat.S_IMODE(bf.downloads_root().stat().st_mode) == 0o700
    assert bf.session_dir("abcdef0123456789") != directory


def test_session_bytes_spans_every_stamped_directory_for_that_session() -> None:
    first = bf.session_dir("sess0001")
    (first / "a.pdf").write_bytes(b"x" * 10)
    second = bf.session_dir("sess0001")
    (second / "b.pdf").write_bytes(b"x" * 5)
    other = bf.session_dir("other999")
    (other / "c.pdf").write_bytes(b"x" * 1000)
    assert bf.session_bytes("sess0001") == 15
    assert bf.session_bytes("nobody") == 0


def test_is_within_is_the_one_public_containment_rule(tmp_path: Path) -> None:
    """Both halves of the feature share this; a private copy in the caller drifts."""
    root = tmp_path / "root"
    (root / "inner").mkdir(parents=True, exist_ok=True)
    assert bf.is_within(root, root)
    assert bf.is_within(root / "inner" / "x.pdf", root)
    assert not bf.is_within(tmp_path / "outside.pdf", root)
    # A sibling whose NAME merely starts with the root's is not inside it, which
    # is the bug a string-prefix test would have.
    assert not bf.is_within(tmp_path / "root-elsewhere" / "x.pdf", root)


def test_a_kept_artifact_can_be_tightened_to_the_private_mode(tmp_path: Path) -> None:
    """§4.1's "files 0600": the chmod helper the download half calls (Q-2)."""
    path = tmp_path / "receipt.pdf"
    path.write_bytes(b"%PDF-1.4\n")
    path.chmod(0o644)
    assert bf.chmod_private(path) is True
    assert stat.S_IMODE(path.stat().st_mode) == bf.PRIVATE_FILE_MODE == 0o600
    # Best-effort: a path that is not there reports the failure instead of raising.
    assert bf.chmod_private(tmp_path / "gone.pdf") is False


def test_dir_size_and_snapshot_survive_a_missing_directory(tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    assert bf.dir_size(missing) == 0
    assert bf.snapshot(missing) == {}
    assert bf.session_bytes("sess", root=missing) == 0


# --- the copy the model reads (R3) ------------------------------------------


def test_declared_mime_label_drops_generic_types_and_sanitises_the_rest() -> None:
    """The declared type is quoted into the transcript and the audit row (R3).

    It is a string from OUTSIDE, so it gets the treatment a name gets: control
    characters and overrides out, length capped. A generic type is dropped rather
    than quoted, because quoting it would read as a signal the server never sent.
    """
    assert bf.declared_mime_label("application/zip") == "application/zip"
    assert bf.declared_mime_label("  text/html \n") == "text/html"
    assert bf.declared_mime_label("application/octet-stream") == ""
    assert bf.declared_mime_label("") == ""
    assert bf.declared_mime_label("application\u202eevil") == "applicationevil"
    assert bf.declared_mime_label("text/\u0000html") == "text/html"
    assert len(bf.declared_mime_label("x/" + "a" * 400)) <= bf.MAX_MIME_BYTES


def test_the_rename_sentence_names_the_server_type_when_there_is_one() -> None:
    """Design §7.4's "the server called it …": the parameter is READ, not dropped."""
    verdict = bf.classify_bytes(
        "invoice.zip", b"%PDF-1.4\n1 0 obj\n", declared_mime="application/zip"
    )
    assert verdict.kind == "allow"
    assert verdict.safe_name == "invoice.pdf"
    assert "the server said 'application/zip'" in verdict.reason
    assert "the name said 'zip'" in verdict.reason
    # Still a HINT: the declared type can lie without changing the verdict, and a
    # generic one is not quoted at all.
    generic = bf.classify_bytes(
        "invoice.zip", b"%PDF-1.4\n1 0 obj\n", declared_mime="application/octet-stream"
    )
    assert generic.safe_name == "invoice.pdf"
    assert "the server said" not in generic.reason


# --- uploads -----------------------------------------------------------------


@pytest.fixture
def uploadable(tmp_path: Path) -> Path:
    path = tmp_path / "deck.pptx"
    path.write_bytes(b"PK\x03\x04fake deck")
    return path


def test_check_upload_accepts_a_real_file(uploadable: Path, tmp_path: Path) -> None:
    resolved, reason = bf.check_upload(str(uploadable), cwd=str(tmp_path))
    assert reason == ""
    assert resolved == uploadable.resolve()


def test_check_upload_resolves_relative_paths_against_the_session_cwd(
    uploadable: Path, tmp_path: Path
) -> None:
    resolved, reason = bf.check_upload("deck.pptx", cwd=str(tmp_path))
    assert reason == "" and resolved == uploadable.resolve()


def test_check_upload_refuses_the_harness_config_root(isolated_config: Path) -> None:
    """Unconditional and first: the secret store and config.yml live there."""
    isolated_config.mkdir(parents=True, exist_ok=True)
    secret = isolated_config / "notes.pdf"
    secret.write_bytes(b"%PDF-1.4\n")
    resolved, reason = bf.check_upload(str(secret), cwd=str(secret.parent))
    assert resolved is None
    assert "config directory" in reason


def test_check_upload_judges_a_symlink_by_its_target(tmp_path: Path) -> None:
    """A symlink called `handout.pdf` must not smuggle a private key out."""
    key = tmp_path / ".ssh" / "id_rsa"
    key.parent.mkdir(parents=True)
    key.write_text("-----BEGIN OPENSSH PRIVATE KEY-----\n")
    handout = tmp_path / "handout.pdf"
    handout.symlink_to(key)
    resolved, reason = bf.check_upload(str(handout), cwd=str(tmp_path))
    assert resolved is None
    assert "credential deny-list" in reason


@pytest.mark.parametrize(
    "name",
    [
        "id_rsa",
        "id_ed25519.pub",
        "server.pem",
        "server.key",
        "keystore.p12",
        "client.pfx",
        "tls.jks",
        "login.keystore",
        ".netrc",
        ".env",
        ".env.production",
        "credentials",
        "credentials.json",
        "service-account-prod.json",
        "login.keychain-db",
        ".git-credentials",
        ".npmrc",
        ".pypirc",
        ".pgpass",
        ".my.cnf",
        ".dockercfg",
    ],
)
def test_check_upload_refuses_the_credential_deny_list(name: str, tmp_path: Path) -> None:
    path = tmp_path / name
    path.write_bytes(b"secret")
    resolved, reason = bf.check_upload(str(path), cwd=str(tmp_path))
    assert resolved is None, name
    assert "credential deny-list" in reason


@pytest.mark.parametrize("component", [".ssh", ".gnupg", ".aws", ".azure", ".kube", "secrets"])
def test_check_upload_refuses_a_credential_directory(component: str, tmp_path: Path) -> None:
    path = tmp_path / component / "notes.pdf"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"%PDF-1.4\n")
    resolved, reason = bf.check_upload(str(path), cwd=str(tmp_path))
    assert resolved is None
    assert "holds credentials" in reason


def test_check_upload_refuses_a_directory_named_as_a_file(tmp_path: Path) -> None:
    folder = tmp_path / "folder.pdf"
    folder.mkdir()
    resolved, reason = bf.check_upload(str(folder), cwd=str(tmp_path))
    assert resolved is None
    assert "not a regular file" in reason


def test_check_upload_refuses_a_missing_path_and_an_empty_string(tmp_path: Path) -> None:
    for raw in (str(tmp_path / "nope.pdf"), "", "   "):
        resolved, reason = bf.check_upload(raw, cwd=str(tmp_path))
        assert resolved is None
        assert "refused" in reason


def test_check_upload_refuses_an_empty_file(tmp_path: Path) -> None:
    empty = tmp_path / "empty.pdf"
    empty.write_bytes(b"")
    resolved, reason = bf.check_upload(str(empty), cwd=str(tmp_path))
    assert resolved is None
    assert "empty" in reason


def test_check_upload_refuses_a_file_over_the_cap(tmp_path: Path) -> None:
    big = tmp_path / "big.pdf"
    big.write_bytes(b"x" * 32)
    resolved, reason = bf.check_upload(
        str(big), cwd=str(tmp_path), policy=bf.Policy(upload_max_bytes=8)
    )
    assert resolved is None
    assert "over the 8 byte limit" in reason


def test_check_upload_leaves_the_long_tail_attachable(tmp_path: Path) -> None:
    """A deny list, never an allow list: the real work must still go through."""
    for name in ("deck.pptx", "notes.txt", "diagram.drawio", "model.stl", "mail.msg"):
        path = tmp_path / name
        path.write_bytes(b"content")
        resolved, reason = bf.check_upload(str(path), cwd=str(tmp_path))
        assert reason == "", name
        assert resolved == path.resolve()


# --- the audit writer --------------------------------------------------------


def test_audit_appends_one_private_jsonl_row_per_decision() -> None:
    import json
    import stat as stat_module

    bf.audit({"session_id": "s1", "action": "download", "verdict": "allow", "name": "a.pdf"})
    bf.audit({"session_id": "s1", "action": "download", "verdict": "deny", "name": "b.exe"})
    path = bf.downloads_root() / bf.AUDIT_FILENAME
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["verdict"] for row in rows] == ["allow", "deny"]
    assert all("ts_ms" in row for row in rows)
    assert stat_module.S_IMODE(path.stat().st_mode) == 0o600


def test_audit_never_raises_into_a_turn(tmp_path: Path) -> None:
    """A failed append must cost the audit, never the user's download."""
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory")
    bf.audit({"session_id": "s1"}, root=blocker)  # must not raise


def test_stat_fact_hashes_from_this_side(tmp_path: Path) -> None:
    """The digest is computed by PYTHON: a host's word is not evidence."""
    import hashlib

    path = tmp_path / "receipt.pdf"
    payload = b"%PDF-1.4\n" + b"x" * 5000
    path.write_bytes(payload)
    fact = bf.stat_fact("receipt.pdf", path, declared_mime="application/pdf")
    assert fact["bytes"] == len(payload)
    assert fact["sha256"] == hashlib.sha256(payload).hexdigest()
    assert fact["path"] == str(path)


def test_stat_fact_on_an_unreadable_file_reports_zeroes(tmp_path: Path) -> None:
    fact = bf.stat_fact("gone.pdf", tmp_path / "gone.pdf")
    assert fact["bytes"] == 0 and fact["sha256"] == ""


# --- the emitted tables ------------------------------------------------------


def test_tables_for_ts_carries_every_list_the_extension_reads() -> None:
    tables = bf.tables_for_ts()
    assert tables["denyExts"] == sorted(bf.DENY_EXTS)
    assert set(tables["credentialComponents"]) == set(bf.CREDENTIAL_COMPONENTS)
    assert tables["caps"]["uploadMaxFiles"] == bf.UPLOAD_MAX_FILES
    assert len(tables["cases"]) == len(bf.CONFORMANCE_CASES)
    # Every case carries the correction it expects, so the TypeScript replay is
    # an exact call rather than a guess between two candidates.
    for case in tables["cases"]:
        assert "safeNameSniffedExt" in case and "nameIsDenyListed" in case


def test_a_new_signature_class_must_be_declared_on_one_side_only() -> None:
    """The deny/allow split is the table's shape, not a comment: every class has
    a label (the model reads it) and every deny class has an extension."""
    for item in bf.DENY_CLASSES:
        assert item.label and item.ext and item.name
        assert item in bf.DENY_CLASSES and item not in bf.ALLOW_CLASSES
    for item in bf.ALLOW_CLASSES:
        assert item.label and item.ext
        assert item not in bf.DENY_CLASSES


def test_the_sniff_never_confuses_a_deny_class_with_an_allow_class() -> None:
    """Deny classes are tested first, so a collision resolves to the refusal."""
    pe = bf.sniff(b"MZ\x90\x00")
    pdf = bf.sniff(b"%PDF-1.4")
    assert pe is not None and pe.name == "pe"
    assert pdf is not None and pdf.name == "pdf"
    assert bf.sniff(b"\x00\x01\x02\x03") is None


def test_policy_defaults_match_the_module_constants() -> None:
    assert bf.DEFAULT.download_max_bytes == bf.DOWNLOAD_MAX_BYTES
    assert bf.DEFAULT.upload_max_files == bf.UPLOAD_MAX_FILES
    assert bf.DOWNLOAD_TIMEOUT_MAX_S > bf.DOWNLOAD_TIMEOUT_S


def test_new_call_id_is_unique_and_prefixed() -> None:
    ids = {bf.new_call_id() for _ in range(50)}
    assert len(ids) == 50
    assert all(item.startswith("bf-") for item in ids)


def test_the_audit_writer_creates_its_directory_private() -> None:
    bf.audit({"session_id": "s"})
    root = bf.downloads_root()
    assert root.is_dir()
    assert stat.S_IMODE(root.stat().st_mode) == 0o700


def test_check_upload_does_not_confine_the_caller_to_the_workspace(tmp_path: Path) -> None:
    """A deliberate NON-refusal, pinned so it cannot drift back (§7.4, §17.7 #8).

    An earlier draft of the design listed "outside the workspace" among the
    refusal reasons, and PR B's QA compared the sentence to the behaviour. The
    behaviour is the one that is right: this session's own quarantine root is
    outside the workspace, and so is the user's Downloads folder, so a
    containment rule would refuse the file the user asked the agent to send. The
    workspace is MARKED in the approval row (`[outside workspace]`, the `write`
    convention) and the controls stay the config-root refusal, the credential
    deny-list on the RESOLVED path, and the cap.
    """
    workspace = tmp_path / "workspace"
    elsewhere = tmp_path / "elsewhere"
    workspace.mkdir()
    elsewhere.mkdir()
    deck = elsewhere / "deck.pptx"
    deck.write_bytes(b"PK\x03\x04a deck")

    resolved, reason = bf.check_upload(str(deck), cwd=str(workspace))
    assert reason == "", "outside-workspace is not a refusal"
    assert resolved == deck.resolve()

    # The neighbouring refusal does NOT need the workspace to fire, and it still
    # does: only the deny-list and the config root stop a resolved path.
    key = elsewhere / ".ssh" / "id_rsa"
    key.parent.mkdir()
    key.write_text("-----BEGIN OPENSSH PRIVATE KEY-----\n")
    assert bf.check_upload(str(key), cwd=str(workspace))[0] is None
