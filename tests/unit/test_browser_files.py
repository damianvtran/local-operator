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

import os
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


# --- intake: the extension host's landed file, moved into quarantine ----------
#
# The extension cannot write into the session directory (Chrome refuses it a
# download path outside the user's own download folder), so its file exists in
# `~/Downloads` under the page's name until the harness moves it. These are the
# rules that decide which reported path may be believed, and — the half no later
# step could do — that the ORIGINAL is gone afterwards, including when the
# content is refused.


def _quarantine(tmp_path: Path) -> Path:
    directory = tmp_path / "config" / "browser" / "downloads" / "20260919-120000-sess0001"
    directory.mkdir(parents=True)
    return directory


def _downloaded(tmp_path: Path, name: str = "receipt.pdf", body: bytes = b"%PDF-1.4\n") -> Path:
    """A file in the user's real download directory, freshly written."""
    landing = tmp_path / "Downloads"
    landing.mkdir(exist_ok=True)
    path = landing / name
    path.write_bytes(body)
    return path


#: The page this file's downloads are attributed to, matching the driven tab.
#:
#: It exists because ownership is now a GATE rather than a footnote: an item whose
#: referrer names another origin is refused before anything is moved or removed, and
#: an item reported while the session does not know which page it is driving is
#: refused too (round-1 R2/R3). `_item` therefore carries a referrer by default —
#: the shape the extension actually reports — and `_intake` supplies the origin.
_PAGE_ORIGIN = "http://127.0.0.1:9/page"


def _item(path: Path, **overrides: object) -> dict[str, object]:
    item: dict[str, object] = {
        "name": path.name,
        "path": str(path),
        "bytes": path.stat().st_size,
        "state": "complete",
        "cancelled": "",
        "referrer": f"{_PAGE_ORIGIN}/index.html",
    }
    item.update(overrides)
    return item


def _intake(items: object, directory: Path, **overrides: object) -> "bf.IntakeOutcome":
    """`intake_landed` for a download this session's own page started.

    The tests that are ABOUT the ownership gate call `bf.intake_landed` directly, so
    the default here cannot hide the behaviour they assert.
    """
    overrides.setdefault("page_origin", _PAGE_ORIGIN)
    return bf.intake_landed(items, directory, **overrides)  # type: ignore[arg-type]


def test_intake_moves_the_landed_file_in_and_leaves_nothing_behind(tmp_path: Path) -> None:
    """The operator's decision, asserted literally: saved here, gone from there."""
    directory = _quarantine(tmp_path)
    source = _downloaded(tmp_path)

    outcome = _intake([_item(source)], directory)

    assert outcome.refused == ()
    assert outcome.moved == ("receipt.pdf",)
    assert (directory / "receipt.pdf").read_bytes() == b"%PDF-1.4\n"
    assert not source.exists(), "the original must be gone from the user's Downloads"


def test_intake_skips_an_item_with_no_path_which_is_the_app_host(tmp_path: Path) -> None:
    """The app host writes into `directory` itself and reports no path at all.

    Skipping is what keeps the two hosts' shapes apart: the app host's files are
    found by the ordinary directory diff, and a rule that demanded a path would
    have refused every download on the host where the destination works.
    """
    directory = _quarantine(tmp_path)
    (directory / "receipt.pdf").write_bytes(b"%PDF-1.4\n")

    outcome = _intake([{"name": "receipt.pdf", "bytes": 9}], directory)

    assert outcome == bf.IntakeOutcome((), ())
    assert (directory / "receipt.pdf").exists()


def test_intake_deletes_a_cancelled_partial_and_says_which_cancel(tmp_path: Path) -> None:
    """An over-cap transfer leaves a partial file in the user's Downloads.

    Nothing later in the pipeline ever looks outside the quarantine root, so if
    this step does not remove it the operator's own decision ("the original is
    gone afterwards") is simply not implemented for the case that most needs it.
    """
    directory = _quarantine(tmp_path)
    source = _downloaded(tmp_path, "big.bin", b"x" * 64)

    outcome = _intake([_item(source, state="interrupted", cancelled="over_cap")], directory)

    assert not source.exists()
    assert outcome.moved == ()
    (refusal,) = outcome.refused
    assert refusal.disposition == bf.DELETED
    # The reason names the CEILING'S VALUE as the constant defines it, not a
    # rounded figure a second table could contradict: the cap is data
    # (`DOWNLOAD_MAX_BYTES`), and a message that said "256 MiB" would keep saying
    # it after someone raised the cap.
    assert str(bf.DOWNLOAD_MAX_BYTES) in refusal.reason and "cancelled" in refusal.reason


def test_intake_never_deletes_a_file_it_cannot_corroborate(tmp_path: Path) -> None:
    """A size mismatch means the file is NOT this download, so it is not ours to touch.

    The asymmetry is the control: a path we can corroborate is ours and a failed
    one is probably the user's, and removing a file we cannot show we watched
    arrive is the one outcome worse than a stray file.
    """
    directory = _quarantine(tmp_path)
    source = _downloaded(tmp_path, "thesis.pdf", b"the user's own file")

    outcome = _intake([_item(source, bytes=999_999)], directory)

    assert source.exists(), "an uncorroborated file must survive"
    (refusal,) = outcome.refused
    assert refusal.disposition == bf.KEPT
    assert "999999" in refusal.reason and "19 bytes" in refusal.reason


def test_intake_refuses_a_stale_file_but_leaves_it_alone(tmp_path: Path) -> None:
    """An old mtime is not this call's download, however the host labelled it."""
    directory = _quarantine(tmp_path)
    source = _downloaded(tmp_path, "screenshot.png")
    stale = source.stat().st_mtime - (bf.DOWNLOAD_TIMEOUT_MAX_S + 600)
    os.utime(source, (stale, stale))

    outcome = _intake([_item(source)], directory)

    assert source.exists()
    (refusal,) = outcome.refused
    assert refusal.disposition == bf.KEPT
    assert "not written during this call" in refusal.reason


def test_intake_unlinks_a_symlink_entry_and_not_its_target(tmp_path: Path) -> None:
    """R1's rule, applied to the one path source a page cannot steer but a build could."""
    directory = _quarantine(tmp_path)
    secret = tmp_path / "secret.txt"
    secret.write_text("private key material")
    link = tmp_path / "Downloads"
    link.mkdir()
    entry = link / "innocent.pdf"
    entry.symlink_to(secret)

    outcome = _intake([_item(entry)], directory)

    assert not entry.exists()
    assert secret.read_text() == "private key material", "the target is untouched"
    (refusal,) = outcome.refused
    assert refusal.disposition == bf.DELETED
    assert "not a regular file" in refusal.reason


def test_intake_refuses_a_relative_path_and_a_missing_file(tmp_path: Path) -> None:
    directory = _quarantine(tmp_path)

    outcome = _intake(
        [
            {"name": "a.pdf", "path": "Downloads/a.pdf", "state": "complete"},
            {"name": "b.pdf", "path": str(tmp_path / "Downloads" / "b.pdf"), "state": "complete"},
        ],
        directory,
    )

    assert outcome.moved == ()
    assert "not absolute" in outcome.refused[0].reason
    assert "not there" in outcome.refused[1].reason
    assert outcome.refused[0].disposition == bf.KEPT
    assert outcome.refused[1].disposition == bf.ABSENT, "nothing was there to remove"


def test_intake_refuses_a_path_inside_our_own_config_root(tmp_path: Path) -> None:
    """Local Operator's own files must never be relocated by a host's claim."""
    directory = _quarantine(tmp_path)
    ours = Path(os.environ["LOCAL_OPERATOR_CONFIG_DIR"]) / "config.yml"
    ours.parent.mkdir(parents=True, exist_ok=True)
    ours.write_text("models: {}\n")

    outcome = _intake([{"name": "config.yml", "path": str(ours), "state": "complete"}], directory)

    assert outcome.moved == ()
    assert ours.exists()
    assert "config directory" in outcome.refused[0].reason


def test_intake_refuses_an_alias_spelling_of_our_own_config_root(tmp_path: Path) -> None:
    """A case-variant (or symlinked) spelling of the config dir is still the config dir.

    Intake MOVES then deletes the host-named source, so a spelling-only refusal would let a
    host relocate `config.yml` out of the config directory. On a case-folding volume the
    alias is the upper-case spelling; elsewhere it is a symlink to the same directory.
    """
    directory = _quarantine(tmp_path)
    cfg = Path(os.environ["LOCAL_OPERATOR_CONFIG_DIR"])
    cfg.mkdir(parents=True, exist_ok=True)
    ours = cfg / "config.yml"
    ours.write_text("models: {}\n")
    probe = tmp_path / "CaseProbe"
    probe.mkdir()
    if (tmp_path / "caseprobe").exists():
        alias = cfg.parent / cfg.name.upper()
    else:
        alias = tmp_path / "config-alias"
        alias.symlink_to(cfg)

    outcome = _intake(
        [{"name": "config.yml", "path": str(alias / "config.yml"), "state": "complete"}], directory
    )

    assert outcome.moved == ()
    assert ours.read_text() == "models: {}\n"
    assert "config directory" in outcome.refused[0].reason


def test_intake_refuses_a_name_the_session_already_holds(tmp_path: Path) -> None:
    """No silent overwrite (§11.4): both files survive and the caller is told."""
    directory = _quarantine(tmp_path)
    (directory / "receipt.pdf").write_bytes(b"%PDF-1.4\nthe first one")
    source = _downloaded(tmp_path)

    outcome = _intake([_item(source)], directory)

    assert outcome.moved == ()
    assert (directory / "receipt.pdf").read_bytes() == b"%PDF-1.4\nthe first one"
    (refusal,) = outcome.refused
    assert "already holds a file with that name" in refusal.reason
    assert refusal.disposition == bf.DELETED, "the duplicate is removed rather than piled up"


def test_intake_accepts_a_file_already_in_the_session_directory(tmp_path: Path) -> None:
    """A host that DID write into the directory is reported as moved, not moved again."""
    directory = _quarantine(tmp_path)
    landed = directory / "receipt.pdf"
    landed.write_bytes(b"%PDF-1.4\n")

    outcome = _intake([_item(landed)], directory)

    assert outcome.moved == ("receipt.pdf",)
    assert outcome.refused == ()
    assert landed.exists()


def test_intake_refuses_a_download_another_page_started(tmp_path: Path) -> None:
    """A download the USER started by hand must not be relocated as if it were ours.

    The association is the one thing the extension's API cannot state (a
    `DownloadItem` has no `tabId`), so the harness uses the referrer: an item whose
    referrer is a different origin from the page this session is driving is the
    user's own download, and it is refused WITHOUT being deleted — a file we never
    asked for is not ours to remove either.
    """
    directory = _quarantine(tmp_path)
    source = _downloaded(tmp_path, "my-tax-return.pdf")

    outcome = _intake(
        [_item(source, referrer="https://elsewhere.example/account")],
        directory,
        page_origin="https://example.test/export",
    )

    assert outcome.moved == ()
    assert source.exists(), "the user's own download must survive"
    (refusal,) = outcome.refused
    assert refusal.disposition == bf.KEPT
    assert "a page this session is not driving" in refusal.reason


def test_intake_accepts_an_absent_referrer_and_the_driven_pages_own(tmp_path: Path) -> None:
    """Absent is not a mismatch, and same-origin-by-different-path is not either.

    Both are the shapes a legitimate page-initiated download takes (a redirect
    chain or a blob download reports no referrer; a page at another path of the
    same site reports one), so a rule that refused them would break the feature to
    close a window this check only narrows.
    """
    directory = _quarantine(tmp_path)
    blob = _downloaded(tmp_path, "export.csv")
    linked = _downloaded(tmp_path, "statement.pdf")

    outcome = _intake(
        [
            _item(blob, referrer=""),
            _item(linked, referrer="http://127.0.0.1:9/form"),
        ],
        directory,
        page_origin="http://127.0.0.1:9/export",
    )

    assert outcome.refused == ()
    assert sorted(outcome.moved) == ["export.csv", "statement.pdf"]


def test_a_cancelled_transfer_whose_partial_is_already_gone_says_so(tmp_path: Path) -> None:
    """Chrome removes a cancelled download's own partial file, which the first
    end-to-end run measured — and the old sentence called that "the file the host
    named is not there", which reads like a corroboration failure when the outcome
    is exactly the one the operator asked for."""
    directory = _quarantine(tmp_path)

    outcome = _intake(
        [
            {
                "name": "stall.bin",
                "path": str(tmp_path / "Downloads" / "stall.bin"),
                "state": "interrupted",
                "cancelled": "unfinished",
            }
        ],
        directory,
    )

    (refusal,) = outcome.refused
    assert refusal.disposition == bf.ABSENT
    assert "the transfer was stopped" in refusal.reason
    assert "already gone" in refusal.reason
    assert "not there" not in refusal.reason


def test_intake_refuses_a_cancelled_transfer_that_is_not_ours(tmp_path: Path) -> None:
    """Ownership is decided BEFORE the destructive branches (round-1 R2).

    The order used to be the other way round, so a cancelled transfer from ANOTHER
    origin — a download the user started by hand, cancelled by the peer at the byte
    ceiling — was DELETED by a refusal that was never about it, and the promised
    "refused without deleting" held only for a transfer that had completed.
    """
    directory = _quarantine(tmp_path)
    source = _downloaded(tmp_path, "the-users-own.iso", b"x" * 64)

    outcome = _intake(
        [
            _item(
                source,
                state="interrupted",
                cancelled="over_cap",
                referrer="https://elsewhere.example/account",
            )
        ],
        directory,
    )

    assert source.exists(), "a cancelled transfer we do not own must not be deleted"
    assert outcome.moved == ()
    (refusal,) = outcome.refused
    assert refusal.disposition == bf.KEPT
    assert "a page this session is not driving" in refusal.reason


def test_intake_refuses_everything_when_it_cannot_name_the_driven_page(tmp_path: Path) -> None:
    """An unknown origin must REFUSE, not skip the check (round-1 R3).

    Repro before the fix: with `page_origin=""` — what the tool passes when it could
    not read the driven tab's URL — a COMPLETED download from another origin was
    moved into quarantine and removed from the user's Downloads, with no refusal row
    and a success result. The safe direction is a save that declines.
    """
    directory = _quarantine(tmp_path)
    foreign = _downloaded(tmp_path, "their-export.pdf")
    ours = _downloaded(tmp_path, "our-export.pdf")

    outcome = _intake(
        [
            _item(foreign, referrer="https://elsewhere.example/account"),
            _item(ours),
        ],
        directory,
        page_origin="",
    )

    assert outcome.moved == ()
    assert foreign.exists() and ours.exists(), "nothing is moved or removed without ownership"
    assert {refusal.disposition for refusal in outcome.refused} == {bf.KEPT}
    assert all("could not confirm which page" in r.reason for r in outcome.refused)


def test_intake_does_not_read_an_absent_state_as_a_stopped_transfer(tmp_path: Path) -> None:
    """No `state` at all is not evidence that a transfer stopped (round-1 blocker).

    The regression this pins: the cancelled branch tested `state != "complete"`, so
    an item that reported no state at all fell into it — the sentence the assertion
    expects became unreachable, and the copy it emitted instead nested one
    parenthesis in another ("the transfer was stopped (the transfer did not complete
    (state unknown))"). At the previous head this case fell through to the sentence
    below.
    """
    directory = _quarantine(tmp_path)

    outcome = _intake(
        [
            {
                "name": "receipt.pdf",
                "path": str(tmp_path / "Downloads" / "receipt.pdf"),
                "bytes": 10,
            }
        ],
        directory,
    )

    (refusal,) = outcome.refused
    assert "the file the host named is not there" in refusal.reason
    assert "was stopped" not in refusal.reason
    # `absent`: nothing was removed because nothing was there, which is what stops
    # the row saying "could NOT be removed — it is still on disk" (round-1 Q2).
    assert refusal.disposition == bf.ABSENT


def test_intake_reports_an_already_gone_partial_as_nothing_to_remove(tmp_path: Path) -> None:
    """The disposition, not just the sentence: `ABSENT`, so no caller can spell it
    as "could NOT be removed — it is still on disk" (round-1 Q2)."""
    directory = _quarantine(tmp_path)

    outcome = _intake(
        [
            {
                "name": "stall.bin",
                "path": str(tmp_path / "Downloads" / "stall.bin"),
                "state": "interrupted",
                "cancelled": "unfinished",
            }
        ],
        directory,
    )

    (refusal,) = outcome.refused
    assert refusal.disposition == bf.ABSENT


def test_intake_refuses_a_path_reached_through_a_symlinked_parent(tmp_path: Path) -> None:
    """The RESOLVED containment rule, with round-2 M2's own repro (round-3 m4).

    The lexical check is not enough, and this is the measurement that says so: a
    download directory holding a symlinked PARENT (`<downloads>/link ->
    <config>/sessions/<id>`) passes it, and the file the host names is then OUR
    session record — `shutil.move` relocated `session.json` into the quarantine
    directory and `os.unlink` deleted it.

    Both sides are asserted because two separate mistakes have to stay fixed: the
    file must not be MOVED (the resolved-path branch), and it must not be DELETED
    either — it is Local Operator's own store, so removing it would be the destructive
    half of the same error. `tmp_path` is under `/private/var/...` on this host while
    `config_dir()` is spelled through `/var/...`, which is what made the first attempt
    at this fix pass locally and fail the real case; comparing both spellings of the
    root is what closes it.
    """
    directory = _quarantine(tmp_path)
    record = bf.config_dir() / "sessions" / "sess0001" / "session.json"
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text('{"private": true}')
    landing = tmp_path / "Downloads"
    landing.mkdir(exist_ok=True)
    link = landing / "link"
    link.symlink_to(record.parent, target_is_directory=True)

    outcome = _intake(
        [_item(link / "session.json", name="session.json", bytes=record.stat().st_size)],
        directory,
    )

    assert outcome.moved == ()
    assert record.exists(), "our own session record must survive a host's claim on it"
    assert record.read_text() == '{"private": true}'
    (refusal,) = outcome.refused
    assert refusal.disposition == bf.KEPT
    assert "resolves inside Local Operator's own config directory" in refusal.reason
    assert list(directory.iterdir()) == [], "nothing may land in quarantine"


def test_the_origin_predicate_names_only_the_schemes_the_extension_agrees_on(
    tmp_path: Path,
) -> None:
    """`_origin_of`'s contract, pinned against `download.ts::originOf` (round-3 M2).

    The two cannot share code — Python has no URL parser here — so the agreement is a
    written contract with a test on one side and the same cases checked by hand on the
    other. `blob:` is the case that was WRONG: `new URL('blob:https://h/uuid').origin`
    is the inner URL's origin (`https://h`), while `urlsplit` sees a scheme with no
    host — so the contract's claim that both answered `""` was false. Both sides now
    answer `""` for anything outside the four schemes a driven page can be.
    """
    assert bf._origin_of("http://User@Host:80/x") == "http://host"
    assert bf._origin_of("https://Example.test/y") == "https://example.test"
    assert bf._origin_of("https://h:8443/z") == "https://h:8443"
    assert bf._origin_of("ws://h:80/a") == "ws://h"
    for url in ("blob:https://example.test/uuid", "data:text/plain,x", "/relative", "ftp://h/p"):
        assert bf._origin_of(url) == "", f"{url} has no origin the extension agrees on"


# --- upload: the session's OWN scratchpad, and the honest refusals -----------
#
# The reported incident: a session wrote an icon to its scratchpad (where the
# system prompt tells it to write files) and `upload` refused it as "inside Local
# Operator's own config directory". The allow is derived from the session id and
# lifts the blanket config-root refusal for that ONE subtree; every test below
# either pins the allow or pins a way of abusing it that must stay refused.

SESSION = "abc123def456"


def _png(path: Path, fill: bytes = b"0") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + fill * 64)
    return path


def _pad(config: Path, session_id: str = SESSION) -> Path:
    return config / "sessions" / session_id / "scratchpad"


def test_check_upload_allows_the_sessions_own_scratchpad(isolated_config: Path) -> None:
    icon = _png(_pad(isolated_config) / "grafana-icon" / "slack-icon-1024.png")
    resolved, reason = bf.check_upload(str(icon), cwd="/", session_id=SESSION)
    assert reason == ""
    assert resolved == icon.resolve()


def test_check_upload_without_a_session_id_keeps_the_blanket_refusal(
    isolated_config: Path,
) -> None:
    """Empty id = no allow: every existing caller keeps the strict behaviour."""
    icon = _png(_pad(isolated_config) / "x.png")
    resolved, reason = bf.check_upload(str(icon), cwd="/")
    assert resolved is None
    assert "config directory" in reason


def test_check_upload_refuses_a_scratchpad_symlink_escaping_to_config_yml(
    isolated_config: Path,
) -> None:
    config_yml = isolated_config / "config.yml"
    config_yml.parent.mkdir(parents=True, exist_ok=True)
    config_yml.write_text("api_key: nope\n")
    pad = _pad(isolated_config)
    pad.mkdir(parents=True)
    (pad / "icon.png").symlink_to(config_yml)
    resolved, reason = bf.check_upload(str(pad / "icon.png"), cwd="/", session_id=SESSION)
    assert resolved is None
    assert "config directory" in reason


def test_check_upload_refuses_a_scratchpad_symlink_to_the_secrets_store(
    isolated_config: Path,
) -> None:
    store = isolated_config / "secrets" / "store.db"
    store.parent.mkdir(parents=True)
    store.write_bytes(b"SQLite format 3\x00" + b"0" * 64)
    pad = _pad(isolated_config)
    pad.mkdir(parents=True)
    (pad / "notes.png").symlink_to(store)
    resolved, reason = bf.check_upload(str(pad / "notes.png"), cwd="/", session_id=SESSION)
    assert resolved is None
    assert "config directory" in reason


def test_check_upload_refuses_dotdot_traversal_out_of_the_scratchpad(
    isolated_config: Path,
) -> None:
    pad = _pad(isolated_config)
    pad.mkdir(parents=True)
    (isolated_config / "config.yml").write_text("k: v\n")
    sibling = _png(isolated_config / "sessions" / SESSION / "transcript.png")
    for escape in (
        pad / ".." / ".." / ".." / "config.yml",
        pad / ".." / "transcript.png",
    ):
        resolved, reason = bf.check_upload(str(escape), cwd="/", session_id=SESSION)
        assert resolved is None, escape
        assert "config directory" in reason
    assert sibling.exists()


@pytest.mark.parametrize(
    ("own", "neighbour"),
    [("abc", "abcdef"), ("abc", "abc-evil"), ("abc", "abc.evil"), ("abc", "ab")],
)
def test_check_upload_refuses_a_sibling_whose_name_shares_the_id_prefix(
    isolated_config: Path, own: str, neighbour: str
) -> None:
    """Containment is on path components, never on a string prefix."""
    _png(_pad(isolated_config, own) / "own.png")
    theirs = _png(_pad(isolated_config, neighbour) / "theirs.png")
    resolved, reason = bf.check_upload(str(theirs), cwd="/", session_id=own)
    assert resolved is None
    assert "config directory" in reason


def test_check_upload_refuses_siblings_inside_the_same_session_dir(
    isolated_config: Path,
) -> None:
    """Neighbours that a STRING-prefix test on the pad would admit must stay refused.

    The pad's own path is ``.../sessions/<id>/scratchpad``, so only a sibling INSIDE the
    session directory (``scratchpad-evil``, ``scratchpad.bak``) or an id-prefix sibling
    (``<id>x/scratchpad``) shares its string prefix. Mutation-checked: replacing
    ``is_within`` by ``str.startswith`` in the allow makes the first two fail.
    """
    _png(_pad(isolated_config) / "own.png")
    for evil in (
        isolated_config / "sessions" / SESSION / "scratchpad-evil" / "x.png",
        isolated_config / "sessions" / SESSION / "scratchpad.bak" / "x.png",
        isolated_config / "sessions" / (SESSION + "x") / "scratchpad" / "x.png",
    ):
        _png(evil)
        resolved, reason = bf.check_upload(str(evil), cwd="/", session_id=SESSION)
        assert resolved is None, evil
        assert "config directory" in reason


def test_check_upload_refuses_the_config_dir_reached_through_a_symlink(
    tmp_path: Path, isolated_config: Path
) -> None:
    """The root is compared RESOLVED: an alias of the config dir is still the config dir.

    Mutation-checked: dropping ``.resolve()`` on the config root lets this through when the
    configured root is itself a symlink.
    """
    real = tmp_path / "real-config"
    (real / "sessions").mkdir(parents=True)
    (real / "config.yml").write_text("api_key: nope\n")
    isolated_config.symlink_to(real)  # the configured root is an alias of the real directory
    alias = tmp_path / "alias"
    alias.symlink_to(real)
    for candidate in (real / "config.yml", alias / "config.yml", isolated_config / "config.yml"):
        resolved, reason = bf.check_upload(str(candidate), cwd="/", session_id=SESSION)
        assert resolved is None, candidate
        assert "config directory" in reason


def test_check_upload_allows_the_own_pad_when_the_configured_root_is_a_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The allow root is derived from the RESOLVED config root, like the candidate.

    Mutation-checked: building it from the unresolved root leaves a resolved candidate that
    is never "inside" it, so the session's own pad would be refused again.
    """
    real = tmp_path / "real-config"
    icon = _png(real / "sessions" / SESSION / "scratchpad" / "icon.png")
    link = tmp_path / "link-config"
    link.symlink_to(real)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(link))
    resolved, reason = bf.check_upload(
        str(link / "sessions" / SESSION / "scratchpad" / "icon.png"), cwd="/", session_id=SESSION
    )
    assert reason == ""
    assert resolved == icon.resolve()


def _folds_case(directory: Path) -> bool:
    """Whether this volume treats ``Name`` and ``NAME`` as one entry (APFS/NTFS default)."""
    probe = directory / "CaseProbe"
    probe.mkdir()
    return (directory / "caseprobe").exists()


def test_check_upload_refuses_a_case_variant_spelling_of_the_config_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``<home>/.LOCAL-OPERATOR/config.yml`` IS ``.local-operator/config.yml`` on APFS.

    The refusal used to compare spellings, so the upper-case spelling attached
    ``config.yml`` and ``auth.db``. On a case-sensitive volume there is no such alias, so the
    test builds the same situation with a symlinked spelling instead of skipping, which
    exercises the same identity rule.
    """
    home = tmp_path / "ci-home"
    home.mkdir()
    cfg = home / ".local-operator"
    (cfg / "sessions" / SESSION / "scratchpad").mkdir(parents=True)
    (cfg / "config.yml").write_text("api_key: nope\n")
    (cfg / "auth.db").write_bytes(b"SQLite format 3\x00" + b"0" * 64)
    _png(cfg / "sessions" / "other-session" / "scratchpad" / "theirs.png")
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(cfg))
    if _folds_case(tmp_path):
        variant = home / ".LOCAL-OPERATOR"
    else:
        variant = home / "alias-config"
        variant.symlink_to(cfg)
    assert variant.exists()
    for name in (
        "config.yml",
        "auth.db",
        "sessions/other-session/scratchpad/theirs.png",
    ):
        resolved, reason = bf.check_upload(str(variant / name), cwd="/", session_id=SESSION)
        assert resolved is None, name
        assert "config directory" in reason


def test_check_upload_own_scratchpad_allow_is_case_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A spelling the filesystem merely folds is not promoted to "the agent's own folder".

    The exact spelling attaches; the same file reached via ``SCRATCHPAD`` is refused (fail
    closed). Only meaningful where the volume folds case, otherwise that spelling is a
    different, non-existent path and the gate's "does not exist" answers instead.
    """
    cfg = tmp_path / "home" / ".local-operator"
    icon = _png(cfg / "sessions" / SESSION / "scratchpad" / "icon.png")
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(cfg))
    assert bf.check_upload(str(icon), cwd="/", session_id=SESSION)[1] == ""
    if not _folds_case(tmp_path):
        pytest.skip("case-sensitive volume: no folded spelling of the pad exists")
    folded = cfg / "sessions" / SESSION / "SCRATCHPAD" / "icon.png"
    resolved, reason = bf.check_upload(str(folded), cwd="/", session_id=SESSION)
    assert resolved is None
    assert "config directory" in reason


def test_is_inside_config_matches_by_identity_and_by_spelling(tmp_path: Path) -> None:
    root = tmp_path / "cfg"
    (root / "a").mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(root)
    assert bf.is_inside_config(root / "a", root)
    assert bf.is_inside_config(root, root)
    assert bf.is_inside_config(alias / "a", root)  # same directory, different spelling
    assert not bf.is_inside_config(tmp_path / "elsewhere", root)
    # A root that does not exist yet falls back to the spelling rule instead of raising.
    assert bf.is_inside_config(tmp_path / "gone" / "x", tmp_path / "gone")
    assert not bf.is_inside_config(tmp_path / "x", tmp_path / "gone")


def test_check_upload_does_not_exist_hint_only_when_the_folder_exists(tmp_path: Path) -> None:
    folder = tmp_path / "work"
    folder.mkdir()
    _, same_turn = bf.check_upload(str(folder / "late.png"), cwd="/")
    assert "does not exist" in same_turn and "same turn" in same_turn
    _, typo = bf.check_upload(str(tmp_path / "wrok" / "late.png"), cwd="/")
    assert "does not exist" in typo and "same turn" not in typo


def test_check_upload_refuses_another_sessions_scratchpad(isolated_config: Path) -> None:
    _png(_pad(isolated_config) / "mine.png")
    theirs = _png(_pad(isolated_config, "zzz999zzz999") / "theirs.png")
    resolved, reason = bf.check_upload(str(theirs), cwd="/", session_id=SESSION)
    assert resolved is None
    assert "config directory" in reason


def test_check_upload_refuses_the_sessions_own_directory_outside_the_scratchpad(
    isolated_config: Path,
) -> None:
    """Only `<session>/scratchpad` is the agent's; the transcript beside it is not."""
    transcript = _png(isolated_config / "sessions" / SESSION / "transcript.png")
    resolved, reason = bf.check_upload(str(transcript), cwd="/", session_id=SESSION)
    assert resolved is None
    assert "config directory" in reason


def test_check_upload_refuses_when_the_scratchpad_itself_is_a_symlink_elsewhere(
    isolated_config: Path,
) -> None:
    """A planted `scratchpad -> <other session>` must not widen the allow."""
    theirs = _png(_pad(isolated_config, "zzz999zzz999") / "theirs.png")
    own_dir = isolated_config / "sessions" / SESSION
    own_dir.mkdir(parents=True)
    (own_dir / "scratchpad").symlink_to(_pad(isolated_config, "zzz999zzz999"))
    resolved, reason = bf.check_upload(
        str(own_dir / "scratchpad" / "theirs.png"), cwd="/", session_id=SESSION
    )
    assert resolved is None
    assert theirs.exists()
    assert "config directory" in reason


@pytest.mark.parametrize("bad_id", ["../x", "a/b", "", "..", ".", "a\\b", "abc\x00", " abc", "a b"])
def test_check_upload_grants_nothing_for_a_malformed_session_id(
    isolated_config: Path, bad_id: str
) -> None:
    # Planted at the places a malformed id could be coerced to point at.
    _png(isolated_config / "x" / "scratchpad" / "x.png")
    _png(isolated_config / "sessions" / "x" / "scratchpad" / "x.png")
    _png(isolated_config / "sessions" / "scratchpad" / "x.png")
    for candidate in (
        isolated_config / "x" / "scratchpad" / "x.png",
        isolated_config / "sessions" / "x" / "scratchpad" / "x.png",
        isolated_config / "sessions" / "scratchpad" / "x.png",
    ):
        resolved, reason = bf.check_upload(str(candidate), cwd="/", session_id=bad_id)
        assert resolved is None, (bad_id, candidate)
        assert "config directory" in reason
    assert bf.own_scratchpad_root(bad_id) is None


# One test per credential CLASS, each with a valid session id supplied: the allow
# must be invisible to every one of them.


def test_check_upload_still_refuses_the_secrets_store_with_a_session_id(
    isolated_config: Path,
) -> None:
    store = isolated_config / "secrets" / "store.db"
    store.parent.mkdir(parents=True)
    store.write_bytes(b"SQLite format 3\x00" + b"0" * 64)
    resolved, reason = bf.check_upload(str(store), cwd="/", session_id=SESSION)
    assert resolved is None
    assert "config directory" in reason


def test_check_upload_still_refuses_config_yml_with_a_session_id(isolated_config: Path) -> None:
    isolated_config.mkdir(parents=True, exist_ok=True)
    (isolated_config / "config.yml").write_text("k: v\n")
    resolved, reason = bf.check_upload(
        str(isolated_config / "config.yml"), cwd="/", session_id=SESSION
    )
    assert resolved is None
    assert "config directory" in reason


@pytest.mark.parametrize("name", ["auth.db", "credentials.env", "mcp.json"])
def test_check_upload_still_refuses_the_auth_and_token_stores_with_a_session_id(
    isolated_config: Path, name: str
) -> None:
    isolated_config.mkdir(parents=True, exist_ok=True)
    (isolated_config / name).write_bytes(b"token")
    resolved, reason = bf.check_upload(str(isolated_config / name), cwd="/", session_id=SESSION)
    assert resolved is None
    assert "config directory" in reason


def test_check_upload_still_refuses_a_keychain_with_a_session_id(
    isolated_config: Path, tmp_path: Path
) -> None:
    """Outside the config root the name/component deny-list is what stops it..."""
    outside = tmp_path / "Library" / "Keychains" / "login.keychain-db"
    outside.parent.mkdir(parents=True)
    outside.write_bytes(b"kych" + b"0" * 64)
    resolved, reason = bf.check_upload(str(outside), cwd="/", session_id=SESSION)
    assert resolved is None
    assert "credential deny-list" in reason or "holds credentials" in reason

    # ...and INSIDE the allowed scratchpad too: the allow lifts the blanket
    # config-root refusal and nothing else, so a keychain copied into the pad is
    # still a keychain.
    pad = _pad(isolated_config)
    for planted in (pad / "login.keychain-db", pad / "keychains" / "innocent.png"):
        planted.parent.mkdir(parents=True, exist_ok=True)
        planted.write_bytes(b"kych" + b"0" * 64)
        resolved, reason = bf.check_upload(str(planted), cwd="/", session_id=SESSION)
        assert resolved is None, planted
        assert "credential deny-list" in reason or "holds credentials" in reason


def test_check_upload_still_applies_the_name_and_shape_checks_inside_the_scratchpad(
    isolated_config: Path,
) -> None:
    pad = _pad(isolated_config)
    pad.mkdir(parents=True)
    (pad / ".env").write_text("SECRET=1\n")
    (pad / "empty.png").write_bytes(b"")
    (pad / "folder").mkdir()
    for name, needle in (
        (".env", "credential deny-list"),
        ("empty.png", "empty"),
        ("folder", "regular"),
    ):
        resolved, reason = bf.check_upload(str(pad / name), cwd="/", session_id=SESSION)
        assert resolved is None, name
        assert needle in reason, (name, reason)


def test_check_upload_still_refuses_another_sessions_directory_with_a_session_id(
    isolated_config: Path,
) -> None:
    transcript = _png(isolated_config / "sessions" / "zzz999zzz999" / "transcript.png")
    resolved, reason = bf.check_upload(str(transcript), cwd="/", session_id=SESSION)
    assert resolved is None
    assert "config directory" in reason


# --- honest refusals: the cause is named, never collapsed to "does not exist" --


def test_check_upload_a_missing_file_names_the_full_path_and_the_nearest_parent(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "workspace"
    existing.mkdir()
    missing = existing / "grafana-slack-icon" / "slack-icon-1024.png"
    resolved, reason = bf.check_upload(str(missing), cwd=str(tmp_path))
    assert resolved is None
    assert reason.startswith("refused: ")
    assert "does not exist" in reason
    # The FULL path, not just the basename: the old text hid which folder was looked in.
    assert str(missing) in reason
    assert f"nearest existing folder: {existing}" in reason


def test_check_upload_shows_what_a_relative_path_was_resolved_against(tmp_path: Path) -> None:
    resolved, reason = bf.check_upload("sub/nope.png", cwd=str(tmp_path))
    assert resolved is None
    assert str(tmp_path / "sub" / "nope.png") in reason


def test_check_upload_sanitises_the_path_it_echoes(tmp_path: Path) -> None:
    hostile = str(tmp_path / "ev\x1b[31mil\u202e.png\nINJECTED")
    resolved, reason = bf.check_upload(hostile, cwd=str(tmp_path))
    assert resolved is None
    assert "\x1b" not in reason and "\u202e" not in reason and "\n" not in reason


def test_check_upload_a_path_through_a_regular_file_is_not_called_missing(
    tmp_path: Path,
) -> None:
    afile = tmp_path / "afile"
    afile.write_text("x")
    resolved, reason = bf.check_upload(str(afile / "x.png"), cwd=str(tmp_path))
    assert resolved is None
    assert "could not be opened" in reason
    assert "does not exist" not in reason


def test_check_upload_a_symlink_loop_is_reported_not_raised(tmp_path: Path) -> None:
    """Python 3.12 raises RuntimeError (not OSError) for a loop; it escaped the tool."""
    (tmp_path / "a").symlink_to(tmp_path / "b")
    (tmp_path / "b").symlink_to(tmp_path / "a")
    resolved, reason = bf.check_upload(str(tmp_path / "a"), cwd=str(tmp_path))
    assert resolved is None
    assert "could not be opened" in reason
    assert "does not exist" not in reason


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_check_upload_a_permission_error_is_not_called_missing(tmp_path: Path) -> None:
    locked = tmp_path / "locked"
    locked.mkdir()
    target = _png(locked / "x.png")
    locked.chmod(0)
    try:
        resolved, reason = bf.check_upload(str(target), cwd=str(tmp_path))
    finally:
        locked.chmod(0o700)
    assert resolved is None
    assert "could not be opened" in reason
    assert "Permission denied" in reason
    assert "does not exist" not in reason
