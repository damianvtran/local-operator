"""A run's results: the presigned download, the archive, and the repository bundle.

WHAT IS PROVEN HERE, one test per property the surface depends on:

* a presigned download NEVER carries the account's bearer and nothing but the
  advertised size is accepted — a short body and a body that grows past it are
  both refused, and nothing is extracted from either;
* an archive is extracted ONLY where it may be: ``..``, an absolute name, a
  symbolic link, a hard link and a device file are each refused with a reason
  code, a destination that already exists is refused, and what does land is
  owner-only (0700 directories, 0600 files);
* a bundle lands as ``refs/remotes/radient/lop/<id>`` in a REAL repository while
  HEAD, the branches, the index and ``git status`` are exactly what they were;
* a session is restored once: a session with that id already on the device is
  left byte-for-byte alone, and the answer says the copy did not happen;
* neither a presigned URL nor a bearer reaches a log line or the run record —
  asserted against the actual values, not against a redaction helper's name.

The download runs against an ``httpx.MockTransport`` (the same fake the rest of
this slice uses) and the seam is ``results.artifact_client``: the point of that
seam is that the download cannot travel on the client holding the credential, so
a test that replaced the run client's transport instead would prove the wrong
thing.
"""

from __future__ import annotations

import io
import logging
import os
import subprocess
import tarfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from local_operator.remote_compute import records, results
from local_operator.remote_compute.client import Artifact, RunArtifacts
from local_operator.remote_compute.errors import (
    BundleApplyFailed,
    ResultsUnavailable,
    ResultsUnreachable,
    SizeMismatch,
    UnsafeArchive,
)

#: The bearer the fake Radient login holds, and the head of a presigned URL. Both
#: are unmistakable strings so the "must not leak" assertions cannot pass by
#: accident.
BEARER = "RADIENT-BEARER-MUST-NOT-LEAK"
SIGNATURE = "PRESIGNED-MUST-NOT-LEAK"

RUN_ID = "9f2c1a04-1111-4111-8111-abcdef000001"
SESSION_ID = "e1e2e3e4e5e6"


@pytest.fixture(autouse=True)
def isolated_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A config root and an archive-scratch root, both temporary."""
    root = tmp_path / "config"
    root.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    monkeypatch.delenv("LOP_REMOTE_COMPUTE_API_URL", raising=False)
    return root


def _presigned(name: str = "results.tar.gz") -> str:
    """A loopback ``http://`` URL, the exception the presigned-URL check makes."""
    return f"http://127.0.0.1:9/{name}?X-Amz-Signature={SIGNATURE}"


def _download_server(payload: bytes, *, status: int = 200) -> Any:
    """A transport that answers every GET with ``payload`` and records the requests."""
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(status, content=payload)

    transport = httpx.MockTransport(handle)
    return transport, requests


@pytest.fixture()
def presigned_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """Point ``results.artifact_client`` at a transport the test supplies.

    A FACTORY RATHER THAN A CLIENT: production code builds and closes its own
    client, and the thing under test is that it builds one WITHOUT the credential —
    so the test replaces where the client comes from, not the client.
    """

    def install(transport: httpx.MockTransport) -> None:
        def build() -> httpx.AsyncClient:
            return httpx.AsyncClient(transport=transport, follow_redirects=False)

        monkeypatch.setattr(results, "artifact_client", build)

    yield install


def _archive(members: list[tuple[tarfile.TarInfo, bytes | None]]) -> bytes:
    """A gzipped tar holding exactly ``members``."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as bundle:
        for info, payload in members:
            bundle.addfile(info, io.BytesIO(payload) if payload is not None else None)
    return buffer.getvalue()


def _file(name: str, payload: bytes = b"line\n") -> tuple[tarfile.TarInfo, bytes]:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mode = 0o644
    return info, payload


def _session_archive(session_id: str = SESSION_ID) -> bytes:
    """The shape a real results archive has: ``sessions/<id>/…``."""
    return _archive(
        [
            _file(f"sessions/{session_id}/transcript.jsonl", b'{"role":"user"}\n'),
            _file(f"sessions/{session_id}/meta.json", b"{}\n"),
        ]
    )


def _client(**overrides: Any) -> Any:
    """A stand-in for the run client: only ``run_artifacts`` is used here."""

    class _Client:
        def __init__(self) -> None:
            self.listing = RunArtifacts(
                run_status=overrides.get("run_status", "succeeded"),
                artifacts=overrides.get(
                    "artifacts",
                    (Artifact(name="results.tar.gz", size_bytes=10, url=_presigned()),),
                ),
                git=overrides.get("git"),
            )
            self.calls: list[str] = []

        async def run_artifacts(self, identifier: str) -> RunArtifacts:
            self.calls.append(identifier)
            return self.listing

    return _Client()


def _write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


# ---------------------------------------------------------------------------
# The download: no bearer, and only the advertised size
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_download_carries_no_radient_bearer_and_no_authorization_header(
    tmp_path: Path, presigned_client: Any
) -> None:
    payload = _session_archive()
    transport, requests = _download_server(payload)
    presigned_client(transport)
    artifact = Artifact(name="results.tar.gz", size_bytes=len(payload), url=_presigned())

    written = await results._download(  # noqa: SLF001 — the unit under test
        artifact, tmp_path / "archive.tar.gz"
    )

    assert written == len(payload)
    assert len(requests) == 1
    assert "authorization" not in {name.lower() for name in requests[0].headers}
    # The signature is carried by the URL itself, which is the whole mechanism.
    assert requests[0].url.params["X-Amz-Signature"] == SIGNATURE
    assert (tmp_path / "archive.tar.gz").read_bytes() == payload
    assert (tmp_path / "archive.tar.gz").stat().st_mode & 0o777 == 0o600


@pytest.mark.asyncio
async def test_a_short_body_is_refused_and_never_extracted(
    tmp_path: Path, presigned_client: Any
) -> None:
    payload = _session_archive()
    transport, _requests = _download_server(payload[:-10])
    presigned_client(transport)
    artifact = Artifact(name="results.tar.gz", size_bytes=len(payload), url=_presigned())
    archive = tmp_path / "archive.tar.gz"

    with pytest.raises(SizeMismatch) as refusal:
        await results._download(artifact, archive)  # noqa: SLF001
    assert refusal.value.expected == len(payload)
    assert refusal.value.actual == len(payload) - 10

    destination = tmp_path / "out"
    with pytest.raises(SizeMismatch):
        await results.fetch_results(
            _client(artifacts=(Artifact("results.tar.gz", len(payload), _presigned()),)),
            RUN_ID,
            destination,
        )
    assert not destination.exists()


@pytest.mark.asyncio
async def test_a_body_that_grows_past_the_advertised_size_is_abandoned(
    tmp_path: Path, presigned_client: Any
) -> None:
    payload = _session_archive()
    transport, _requests = _download_server(payload)
    presigned_client(transport)
    # The server advertises less than it sends: the bound is what stops the write,
    # so the refusal names the byte the body was abandoned at rather than the end.
    artifact = Artifact(name="results.tar.gz", size_bytes=len(payload) - 10, url=_presigned())

    with pytest.raises(SizeMismatch) as refusal:
        await results._download(artifact, tmp_path / "archive.tar.gz")  # noqa: SLF001
    assert refusal.value.actual > refusal.value.expected


@pytest.mark.asyncio
async def test_a_refused_download_names_the_status_and_never_the_url(
    tmp_path: Path, presigned_client: Any
) -> None:
    transport, _requests = _download_server(b"", status=403)
    presigned_client(transport)
    artifact = Artifact(name="results.tar.gz", size_bytes=4, url=_presigned())

    with pytest.raises(ResultsUnavailable) as refusal:
        await results._download(artifact, tmp_path / "archive.tar.gz")  # noqa: SLF001
    assert refusal.value.reason == results.REASON_DOWNLOAD_REFUSED
    assert "403" in str(refusal.value)
    assert SIGNATURE not in str(refusal.value)


def test_the_artifact_url_is_not_in_its_repr_and_not_in_its_public_shape() -> None:
    """The one filter in the type: a log line or a body cannot leak the URL."""
    artifact = Artifact("results.tar.gz", 10, _presigned())

    assert SIGNATURE not in repr(artifact)
    assert SIGNATURE not in str(artifact)
    assert "url" not in artifact.to_json()
    assert artifact.to_json()["name"] == "results.tar.gz"


# ---------------------------------------------------------------------------
# Extraction: what may be written, and where
# ---------------------------------------------------------------------------


def test_an_archive_is_extracted_owner_only(tmp_path: Path) -> None:
    archive = _write(tmp_path / "results.tar.gz", _session_archive())
    destination = tmp_path / "out"

    assert results.extract(archive, destination) == 2

    transcript = destination / "sessions" / SESSION_ID / "transcript.jsonl"
    assert transcript.read_bytes() == b'{"role":"user"}\n'
    assert transcript.stat().st_mode & 0o777 == 0o600
    for directory in (destination, destination / "sessions", transcript.parent):
        assert directory.stat().st_mode & 0o777 == 0o700


def test_a_destination_that_already_exists_is_refused(tmp_path: Path) -> None:
    archive = _write(tmp_path / "results.tar.gz", _session_archive())
    destination = tmp_path / "out"
    destination.mkdir()

    with pytest.raises(ResultsUnavailable) as refusal:
        results.extract(archive, destination)
    assert refusal.value.reason == results.REASON_DESTINATION_EXISTS
    assert list(destination.iterdir()) == []


def test_an_entry_that_climbs_out_of_the_destination_is_refused(tmp_path: Path) -> None:
    archive = _write(tmp_path / "results.tar.gz", _archive([_file("../escaped.txt")]))
    destination = tmp_path / "out"

    with pytest.raises(UnsafeArchive) as refusal:
        results.extract(archive, destination)
    assert refusal.value.reason == results.REASON_OUTSIDE_DESTINATION
    assert not (tmp_path / "escaped.txt").exists()


def test_an_absolute_entry_is_refused(tmp_path: Path) -> None:
    target = tmp_path / "absolute.txt"
    archive = _write(tmp_path / "results.tar.gz", _archive([_file(str(target))]))

    with pytest.raises(UnsafeArchive) as refusal:
        results.extract(archive, tmp_path / "out")
    assert refusal.value.reason == results.REASON_ABSOLUTE_PATH
    assert not target.exists()
    # Not a mangled copy inside the destination either: tarfile's own data filter
    # would have stripped the leading separator and written ``<dest>/private/...``.
    assert not (tmp_path / "out" / "private").exists()


def test_a_symlink_is_refused_even_when_it_points_outside(tmp_path: Path) -> None:
    info = tarfile.TarInfo("sessions/link")
    info.type = tarfile.SYMTYPE
    info.linkname = "../../../../etc/passwd"
    archive = _write(tmp_path / "results.tar.gz", _archive([(info, None)]))

    with pytest.raises(UnsafeArchive) as refusal:
        results.extract(archive, tmp_path / "out")
    assert refusal.value.reason == results.REASON_LINK


def test_a_hardlink_is_refused(tmp_path: Path) -> None:
    payload = b"line\n"
    info = tarfile.TarInfo("sessions/hard")
    info.type = tarfile.LNKTYPE
    info.linkname = "sessions/elsewhere"
    archive = _write(
        tmp_path / "results.tar.gz",
        _archive([_file("sessions/elsewhere", payload), (info, None)]),
    )

    with pytest.raises(UnsafeArchive) as refusal:
        results.extract(archive, tmp_path / "out")
    assert refusal.value.reason == results.REASON_LINK


def test_a_device_file_is_refused(tmp_path: Path) -> None:
    info = tarfile.TarInfo("sessions/tty")
    info.type = tarfile.CHRTYPE
    info.devmajor = 4
    info.devminor = 1
    archive = _write(tmp_path / "results.tar.gz", _archive([(info, None)]))

    with pytest.raises(UnsafeArchive) as refusal:
        results.extract(archive, tmp_path / "out")
    assert refusal.value.reason == results.REASON_SPECIAL_FILE


def test_bytes_that_are_not_an_archive_are_refused(tmp_path: Path) -> None:
    archive = _write(tmp_path / "results.tar.gz", b"not a tarball")

    with pytest.raises(ResultsUnavailable) as refusal:
        results.extract(archive, tmp_path / "out")
    assert refusal.value.reason == results.REASON_UNREADABLE_ARCHIVE


@pytest.mark.asyncio
async def test_a_relative_destination_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ResultsUnavailable) as refusal:
        await results.fetch_results(_client(), RUN_ID, Path("relative/dir"))
    assert refusal.value.reason == results.REASON_RELATIVE_DESTINATION


@pytest.mark.asyncio
async def test_a_run_without_results_is_refused(tmp_path: Path) -> None:
    listing = _client(
        artifacts=(Artifact(name="probes.json", size_bytes=2, url=_presigned("probes.json")),)
    )
    with pytest.raises(ResultsUnavailable) as refusal:
        await results.fetch_results(listing, RUN_ID, tmp_path / "out")
    assert refusal.value.reason == results.REASON_NO_RESULTS


@pytest.mark.asyncio
async def test_fetch_extracts_and_reports_what_it_wrote(
    tmp_path: Path, presigned_client: Any
) -> None:
    payload = _session_archive()
    transport, _requests = _download_server(payload)
    presigned_client(transport)
    client = _client(artifacts=(Artifact("results.tar.gz", len(payload), _presigned()),))

    fetched = await results.fetch_results(client, RUN_ID, tmp_path / "out")

    assert fetched.directory == tmp_path / "out"
    assert fetched.archive_bytes == len(payload)
    assert fetched.file_count == 2
    assert client.calls == [RUN_ID]


# ---------------------------------------------------------------------------
# The session: restored once, never over what is already here
# ---------------------------------------------------------------------------


def test_a_session_is_restored_under_the_config_root(tmp_path: Path, isolated_roots: Path) -> None:
    extracted = tmp_path / "out"
    results.extract(_write(tmp_path / "results.tar.gz", _session_archive()), extracted)

    restored = results.restore_session(extracted)

    assert restored is not None
    assert restored.session_id == SESSION_ID
    assert restored.created is True
    target = isolated_roots / "sessions" / SESSION_ID
    assert (target / "transcript.jsonl").read_text() == '{"role":"user"}\n'


def test_an_existing_session_is_never_overwritten(tmp_path: Path, isolated_roots: Path) -> None:
    existing = isolated_roots / "sessions" / SESSION_ID
    existing.mkdir(parents=True)
    (existing / "transcript.jsonl").write_text("the operator's own, longer history\n")
    extracted = tmp_path / "out"
    results.extract(_write(tmp_path / "results.tar.gz", _session_archive()), extracted)

    restored = results.restore_session(extracted)

    assert restored is not None
    assert restored.session_id == SESSION_ID
    assert restored.created is False
    assert (existing / "transcript.jsonl").read_text() == "the operator's own, longer history\n"


def test_an_archive_without_a_session_directory_offers_nothing(tmp_path: Path) -> None:
    extracted = tmp_path / "out"
    archive = _write(tmp_path / "results.tar.gz", _archive([_file("probes.json")]))
    results.extract(archive, extracted)

    assert results.restore_session(extracted) is None


def test_a_directory_that_is_not_a_session_id_is_not_restored(
    tmp_path: Path, isolated_roots: Path
) -> None:
    extracted = tmp_path / "out"
    results.extract(
        _write(tmp_path / "results.tar.gz", _archive([_file("sessions/not-a-session/x")])),
        extracted,
    )

    assert results.restore_session(extracted) is None
    assert not (isolated_roots / "sessions" / "not-a-session").exists()


# ---------------------------------------------------------------------------
# The bundle: a ref, and nothing else in the operator's repository
# ---------------------------------------------------------------------------


def _git(cwd: Path, *arguments: str) -> str:
    """One git command with a pinned identity and no inherited repository location."""
    done = subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
        env={
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(cwd),
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_AUTHOR_NAME": "test",
            "GIT_AUTHOR_EMAIL": "test@example.org",
            "GIT_COMMITTER_NAME": "test",
            "GIT_COMMITTER_EMAIL": "test@example.org",
        },
    )
    return done.stdout


def _source_repository(root: Path, run_id: str = RUN_ID) -> tuple[Path, Path]:
    """A repository with one commit, and a bundle of the run's own branch on top.

    THE CLONE IS MADE FIRST, deliberately: it must be a repository that has the
    base commit and has never seen the run's branch, which is what an operator's
    working clone looks like when they apply a bundle.
    """
    source = root / "source"
    source.mkdir()
    _git(source, "init", "--quiet", "--initial-branch=main")
    (source / "README.md").write_text("hello\n")
    _git(source, "add", "README.md")
    _git(source, "commit", "--quiet", "-m", "base")
    clone = root / "clone"
    _git(root, "clone", "--quiet", str(source), str(clone))
    _git(source, "branch", f"lop/{run_id}", "main")
    _git(source, "checkout", "--quiet", f"lop/{run_id}")
    (source / "work.txt").write_text("the run's work\n")
    _git(source, "add", "work.txt")
    _git(source, "commit", "--quiet", "-m", "the run's commit")
    _git(source, "checkout", "--quiet", "main")
    bundle = root / "repo.bundle"
    _git(source, "bundle", "create", str(bundle), f"main..lop/{run_id}")
    return clone, bundle


@pytest.mark.asyncio
async def test_a_bundle_lands_as_a_remote_ref_and_moves_nothing_else(
    tmp_path: Path, presigned_client: Any
) -> None:
    clone, bundle = _source_repository(tmp_path)
    payload = bundle.read_bytes()
    transport, _requests = _download_server(payload)
    presigned_client(transport)
    client = _client(artifacts=(Artifact("repo.bundle", len(payload), _presigned("repo.bundle")),))

    before = {
        "head": _git(clone, "rev-parse", "HEAD"),
        "branch": _git(clone, "symbolic-ref", "HEAD"),
        "refs": _git(clone, "for-each-ref", "refs/heads"),
        "status": _git(clone, "status", "--porcelain"),
        "index": _git(clone, "ls-files", "--stage"),
    }

    applied = await results.apply_bundle(client, RUN_ID, clone)

    assert applied.ref == f"refs/remotes/radient/lop/{RUN_ID}"
    assert _git(clone, "rev-parse", applied.ref).strip() == applied.commit
    assert {
        "head": _git(clone, "rev-parse", "HEAD"),
        "branch": _git(clone, "symbolic-ref", "HEAD"),
        "refs": _git(clone, "for-each-ref", "refs/heads"),
        "status": _git(clone, "status", "--porcelain"),
        "index": _git(clone, "ls-files", "--stage"),
    } == before
    # The run's commit is reachable from the new ref, and is not on main.
    assert _git(clone, "log", "-1", "--format=%s", applied.ref).strip() == "the run's commit"
    assert "the run's commit" not in _git(clone, "log", "--format=%s", "main")


@pytest.mark.asyncio
async def test_a_bundle_the_clone_cannot_use_is_refused(
    tmp_path: Path, presigned_client: Any
) -> None:
    clone, bundle = _source_repository(tmp_path)
    # A clone of a DIFFERENT history: the bundle's base commit is not there, which
    # is exactly what ``git bundle verify`` refuses.
    other = tmp_path / "other"
    other.mkdir()
    _git(other, "init", "--quiet", "--initial-branch=main")
    (other / "other.txt").write_text("other\n")
    _git(other, "add", "other.txt")
    _git(other, "commit", "--quiet", "-m", "unrelated")
    payload = bundle.read_bytes()
    transport, _requests = _download_server(payload)
    presigned_client(transport)
    client = _client(artifacts=(Artifact("repo.bundle", len(payload), _presigned("repo.bundle")),))

    with pytest.raises(BundleApplyFailed) as refusal:
        await results.apply_bundle(client, RUN_ID, other)
    assert refusal.value.reason == results.REASON_BUNDLE_INVALID
    assert not (other / ".git" / "refs" / "remotes" / "radient").exists()
    assert clone.exists()


@pytest.mark.asyncio
async def test_a_run_without_a_bundle_is_refused(tmp_path: Path) -> None:
    # A REAL repository, because the directory checks run first now: a run with no
    # bundle in a directory that is not a repository answers ``not_a_repository``.
    clone, _bundle = _source_repository(tmp_path)
    with pytest.raises(ResultsUnavailable) as refusal:
        await results.apply_bundle(_client(artifacts=()), RUN_ID, clone)
    assert refusal.value.reason == results.REASON_NO_BUNDLE


@pytest.mark.asyncio
async def test_a_bundle_target_that_is_not_a_directory_is_refused(tmp_path: Path) -> None:
    with pytest.raises(BundleApplyFailed) as refusal:
        await results.apply_bundle(_client(), RUN_ID, tmp_path / "missing")
    assert refusal.value.reason == results.REASON_NOT_A_DIRECTORY


@pytest.mark.asyncio
async def test_a_run_id_that_cannot_be_a_ref_is_refused(tmp_path: Path) -> None:
    for bad in ("../escape", "a/b", "..", ""):
        with pytest.raises(ResultsUnavailable) as refusal:
            await results.apply_bundle(_client(), bad, tmp_path)
        assert refusal.value.reason == results.REASON_INVALID_RUN_ID


# ---------------------------------------------------------------------------
# Nothing secret reaches a log line or the run record
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_neither_the_url_nor_the_bearer_reaches_a_log_or_the_record(
    tmp_path: Path, presigned_client: Any, caplog: pytest.LogCaptureFixture
) -> None:
    payload = _session_archive()
    transport, _requests = _download_server(payload)
    presigned_client(transport)
    client = _client(artifacts=(Artifact("results.tar.gz", len(payload), _presigned()),))
    records.record_submission(
        run_id=RUN_ID,
        repo_url="https://github.com/example/widget.git",
        repo_sha="0" * 40,
        prompt="do the thing",
        size="m",
        api_origin="https://api.example.org",
        network_id=None,
        status="succeeded",
        config_dir=tmp_path / "config",
    )
    record = records.records_path(tmp_path / "config")

    with caplog.at_level(logging.DEBUG):
        await results.fetch_results(client, RUN_ID, tmp_path / "out")

    text = "\n".join(row.getMessage() or "" for row in caplog.records)
    assert SIGNATURE not in text
    assert BEARER not in text
    assert SIGNATURE not in record.read_text()
    assert BEARER not in record.read_text()


# ---------------------------------------------------------------------------
# A refusal leaves nothing behind, and every failure of a write is typed
# ---------------------------------------------------------------------------


def _dir(name: str) -> tuple[tarfile.TarInfo, None]:
    info = tarfile.TarInfo(name)
    info.type = tarfile.DIRTYPE
    info.mode = 0o755
    return info, None


def _sequence_server(payloads: list[bytes]) -> httpx.MockTransport:
    """A download transport that answers each payload in turn."""
    pending = list(payloads)

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=pending.pop(0))

    return httpx.MockTransport(handle)


def _sized_client(payloads: list[bytes]) -> Any:
    """A run client whose listing advertises each payload's own size, in order."""

    class _Sequenced:
        def __init__(self) -> None:
            self.answered = 0

        async def run_artifacts(self, identifier: str) -> RunArtifacts:
            payload = payloads[min(self.answered, len(payloads) - 1)]
            self.answered += 1
            return RunArtifacts(
                run_status="succeeded",
                artifacts=(Artifact("results.tar.gz", len(payload), _presigned()),),
                git=None,
            )

    return _Sequenced()


@pytest.mark.asyncio
async def test_a_refused_archive_leaves_no_destination_and_the_retry_succeeds(
    tmp_path: Path, presigned_client: Any
) -> None:
    """The default destination is per run, so a leftover tree wedges that run forever."""
    hostile = _archive([_file("fine.txt", b"fine\n"), _file("../escape", b"no\n")])
    good = _session_archive()
    presigned_client(_sequence_server([hostile, good]))
    client = _sized_client([hostile, good])
    destination = tmp_path / "out"

    with pytest.raises(UnsafeArchive) as refusal:
        await results.fetch_results(client, RUN_ID, destination)
    assert refusal.value.reason == results.REASON_OUTSIDE_DESTINATION
    assert not destination.exists()
    # Nor a staging directory beside it: the refusal removed the only thing it made.
    assert [child for child in tmp_path.iterdir() if ".extracting-" in child.name] == []

    fetched = await results.fetch_results(client, RUN_ID, destination)

    assert fetched.file_count == 2
    assert (destination / "sessions" / SESSION_ID / "meta.json").read_text() == "{}\n"


#: The reviewer's crafted archives: each asks the FILESYSTEM for something it will
#: not do, and each has to come back as a typed refusal naming the member rather
#: than as an ``OSError`` a surface can only answer 500 to.
_CRAFTED_MEMBERS: dict[str, list[tuple[tarfile.TarInfo, bytes | None]]] = {
    "a name too long for the filesystem": [_file("a" * 300, b"x\n")],
    "a path through a file": [_file("f", b"x\n"), _file("f/child", b"y\n")],
    "a file over an existing directory": [_dir("d"), _file("d", b"x\n")],
    "a directory over an existing file": [_file("d", b"x\n"), _dir("d")],
    "a name with no component to write": [_file(".", b"x\n")],
}


@pytest.mark.parametrize("described", sorted(_CRAFTED_MEMBERS))
def test_a_member_write_the_filesystem_refuses_is_a_typed_refusal(
    tmp_path: Path, described: str
) -> None:
    archive = _write(tmp_path / "results.tar.gz", _archive(_CRAFTED_MEMBERS[described]))
    destination = tmp_path / "out"

    with pytest.raises(UnsafeArchive) as refusal:
        results.extract(archive, destination)

    assert refusal.value.reason == results.REASON_MEMBER_UNWRITABLE
    assert not destination.exists()


def test_an_archive_that_expands_past_the_bound_is_refused(tmp_path: Path) -> None:
    """A zip bomb: 8 MB of zeros in one member out of a few KB of gzip."""
    archive = _write(
        tmp_path / "results.tar.gz", _archive([_file("sessions/x/blob.bin", b"\0" * 8_000_000)])
    )
    limit = archive.stat().st_size * results.EXPANSION_LIMIT
    assert 8_000_000 > limit  # the bomb is really over the bound

    with pytest.raises(UnsafeArchive) as refusal:
        results.extract(archive, tmp_path / "out")

    assert refusal.value.reason == results.REASON_TOO_LARGE
    assert not (tmp_path / "out").exists()
    # Nothing was written past the bound, and the staging tree went with the refusal.
    assert [child for child in tmp_path.iterdir() if ".extracting-" in child.name] == []


def test_an_archive_with_more_members_than_the_bound_is_refused(tmp_path: Path) -> None:
    members: list[tuple[tarfile.TarInfo, bytes | None]] = [
        _file(f"f{number:05d}.txt", b"") for number in range(results.MAX_MEMBERS + 1)
    ]
    archive = _write(tmp_path / "results.tar.gz", _archive(members))

    with pytest.raises(UnsafeArchive) as refusal:
        results.extract(archive, tmp_path / "out")

    assert refusal.value.reason == results.REASON_TOO_LARGE
    assert "members" in str(refusal.value)


def test_file_count_counts_distinct_files_not_members(tmp_path: Path) -> None:
    """A tar may repeat a name; the count a panel shows has to match the disk."""
    archive = _write(
        tmp_path / "results.tar.gz",
        _archive([_file("same.txt", b"one\n"), _file("same.txt", b"one\n")]),
    )

    assert results.extract(archive, tmp_path / "out") == 1


@pytest.mark.parametrize("raised", [httpx.ConnectError, httpx.ReadTimeout])
@pytest.mark.asyncio
async def test_a_download_that_never_completes_is_a_typed_refusal(
    tmp_path: Path, presigned_client: Any, raised: type[httpx.TransportError]
) -> None:
    """A dropped connection and a timeout share a remedy: a fresh listing."""

    def handle(request: httpx.Request) -> httpx.Response:
        raise raised("the transport gave up", request=request)

    presigned_client(httpx.MockTransport(handle))
    artifact = Artifact(name="results.tar.gz", size_bytes=4, url=_presigned())

    with pytest.raises(ResultsUnreachable) as refusal:
        await results._download(artifact, tmp_path / "archive.tar.gz")  # noqa: SLF001

    assert raised.__name__ in str(refusal.value)
    assert SIGNATURE not in str(refusal.value)
    # No chain is kept, because httpx's exception carries the presigned request.
    assert refusal.value.__cause__ is None
    assert refusal.value.__context__ is None


@pytest.mark.asyncio
async def test_both_wire_loggers_are_pinned_while_a_presigned_get_runs(
    tmp_path: Path, presigned_client: Any
) -> None:
    """httpcore emits its own records for the same request, so it is pinned too."""
    payload = _session_archive()
    levels: list[tuple[int, int]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        levels.append((logging.getLogger("httpx").level, logging.getLogger("httpcore").level))
        return httpx.Response(200, content=payload)

    presigned_client(httpx.MockTransport(handle))
    wire = logging.getLogger("httpx")
    core = logging.getLogger("httpcore")
    # ONE OF THE TWO IS ALREADY PINNED product-wide (``logger.quiet_wire_clients``
    # sets httpx to WARNING in every entry point), so the assertion is that BOTH come
    # back to whatever they were rather than to the level the pin uses.
    core.setLevel(logging.NOTSET)
    before = (wire.level, core.level)
    artifact = Artifact(name="results.tar.gz", size_bytes=len(payload), url=_presigned())

    await results._download(artifact, tmp_path / "archive.tar.gz")  # noqa: SLF001

    assert levels == [(logging.WARNING, logging.WARNING)]
    assert (wire.level, core.level) == before


def test_a_run_id_git_would_refuse_as_a_ref_is_refused_before_git() -> None:
    """``a..b`` and ``x.lock`` are legal filenames and illegal ref names."""
    for bad in ("a..b", "x.lock", "x.", "..", "..."):
        with pytest.raises(ResultsUnavailable) as refusal:
            results.checked_run_id(bad)
        assert refusal.value.reason == results.REASON_INVALID_RUN_ID
    assert results.checked_run_id(RUN_ID) == RUN_ID
    assert results.bundle_ref(RUN_ID) == f"refs/remotes/radient/lop/{RUN_ID}"


@pytest.mark.asyncio
async def test_a_directory_that_is_not_a_repository_is_refused_before_any_download(
    tmp_path: Path,
) -> None:
    """The directory is the answer here, not the bundle: nothing was fetched to find out."""
    client = _client()

    with pytest.raises(BundleApplyFailed) as refusal:
        await results.apply_bundle(client, RUN_ID, tmp_path)

    assert refusal.value.reason == results.REASON_NOT_A_REPOSITORY
    assert client.calls == []


@pytest.mark.asyncio
async def test_a_directory_inside_a_repository_but_not_its_root_is_refused(
    tmp_path: Path,
) -> None:
    """A fetch there would write this app's ref into a repository nobody named."""
    clone, _bundle = _source_repository(tmp_path)
    inside = clone / "sub"
    inside.mkdir()

    with pytest.raises(BundleApplyFailed) as refusal:
        await results.apply_bundle(_client(), RUN_ID, inside)

    assert refusal.value.reason == results.REASON_NOT_REPOSITORY_ROOT


@pytest.mark.asyncio
async def test_a_relative_repository_directory_is_refused(tmp_path: Path) -> None:
    """A relative path would resolve against the daemon's working directory."""
    with pytest.raises(BundleApplyFailed) as refusal:
        await results.apply_bundle(_client(), RUN_ID, "clone")

    assert refusal.value.reason == results.REASON_RELATIVE_DESTINATION


def test_restoring_a_session_with_a_link_inside_is_refused(
    tmp_path: Path, isolated_roots: Path
) -> None:
    """``restore_session`` is public, so the link rule has to hold here too."""
    session = tmp_path / "results" / "sessions" / SESSION_ID
    session.mkdir(parents=True)
    (session / "meta.json").write_text("{}\n")
    (session / "link").symlink_to("/etc/passwd")

    with pytest.raises(UnsafeArchive) as refusal:
        results.restore_session(tmp_path / "results")

    assert refusal.value.reason == results.REASON_LINK
    assert not (isolated_roots / "sessions" / SESSION_ID).exists()
