"""A run's results: the session archive, and the repository bundle beside it.

WHAT A FINISHED RUN LEAVES BEHIND. The container publishes its work as named
files — ``results.tar.gz`` (the lop session directory), ``repo.bundle`` (the
commits it made), ``probes.json``, ``git.json`` — and the run's artifact listing
answers a presigned GET for each. This module is what turns those two facts into
three things a surface can offer: extract the archive somewhere, restore the
session so ``lop --resume <id>`` can reach it, and fetch the bundle into a clone
so the run's commits are visible there. It PRINTS NOTHING and raises typed errors
from :mod:`local_operator.remote_compute.errors`; every sentence a person reads
belongs to the surface.

A PRESIGNED URL IS A BEARER, AND IT IS TREATED AS ONE. The URL authorises its own
download for a few minutes with no Radient header at all. So the download is made
by a FRESH client (:func:`artifact_client`) that never carries the account's
bearer: S3 has no use for it, ignores or rejects it, and sending it would hand a
credential to a second service for no gain. And the URL never reaches a log, an
error or the run record — the refusals here name the ARTIFACT, never its URL, so
there is nothing to remember to redact at a call site.

WHAT THIS MODULE DOES NOT DO. It does not record anything locally: the run record
(:mod:`local_operator.remote_compute.records`) is about submissions and status,
and a results fetch writes files, not rows. It does not verify the archive's
contents beyond refusing what must never be written — a session directory is the
far end's own format, and a decision about what belongs in it would be a second
copy of that format here. And it does not cap the EXPANDED size of the archive:
the advertised ``size_bytes`` bounds the download, and the expansion ratio of a
session directory is not a number this repository has measured, so a cap would be
a guess that fails a legitimate large run.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import os
import re
import shutil
import tarfile
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import IO, TYPE_CHECKING

import httpx

from local_operator.remote_compute import repo as repo_module
from local_operator.remote_compute.errors import (
    BundleApplyFailed,
    ResultsUnavailable,
    SizeMismatch,
    UnsafeArchive,
)
from local_operator.remote_files.cache import is_session_id

if TYPE_CHECKING:  # the artifact type only, so this module keeps its light imports
    from local_operator.remote_compute.client import Artifact, RemoteComputeClient

#: The artifact that carries the run's session directory.
RESULTS_ARTIFACT = "results.tar.gz"

#: The artifact that carries the commits the run made.
BUNDLE_ARTIFACT = "repo.bundle"

#: Where a restored session lands, relative to the config directory.
SESSIONS_DIRECTORY = "sessions"

#: Where extracted results land by default, relative to the config directory.
#:
#: A SIBLING of the run record (``remote-compute/runs.jsonl``), because both belong
#: to this feature and one directory is one thing to clean up.
RESULTS_SUBDIRECTORY: tuple[str, ...] = ("remote-compute", "results")

#: The ref a run's bundle is fetched to. DELIBERATELY NOT a branch: nothing checks
#: it out, so a run's commits are visible to ``git log`` without moving HEAD or
#: touching the operator's index.
BUNDLE_REF_PREFIX = "refs/remotes/radient/lop/"

#: The ref the bundle is expected to carry inside itself (``lop/<run id>``).
BUNDLE_SOURCE_PREFIX = "lop/"

#: The shape a run id must have to become a path component or a git ref name.
#:
#: TIGHTER THAN THE WIRE NEEDS, deliberately: the id is written into a directory
#: name and into a ref, and git's ref grammar refuses ``..``, ``~``, ``^``, spaces
#: and control bytes. The server's ids are UUID-shaped, so a value outside this
#: set is a value this build should not be asked to turn into a path.
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")

#: What this module refuses, as the reason codes its errors carry.
REASON_NO_RESULTS = "no_results"
REASON_NO_BUNDLE = "no_bundle"
REASON_DOWNLOAD_REFUSED = "download_refused"
REASON_DESTINATION_EXISTS = "destination_exists"
REASON_RELATIVE_DESTINATION = "relative_destination"
REASON_UNREADABLE_ARCHIVE = "unreadable_archive"
REASON_INVALID_RUN_ID = "invalid_run_id"
REASON_ABSOLUTE_PATH = "absolute_path"
REASON_OUTSIDE_DESTINATION = "outside_destination"
REASON_LINK = "link"
REASON_SPECIAL_FILE = "special_file"
REASON_NOT_A_DIRECTORY = "not_a_directory"
REASON_BUNDLE_INVALID = "bundle_invalid"
REASON_FETCH_REFUSED = "fetch_refused"

#: Modes a session directory and its files are written with. A transcript can
#: quote an operator's prompt, so both are owner-only, as the run record and the
#: transcript itself already are.
DIRECTORY_MODE = 0o700
FILE_MODE = 0o600

#: How much of a download is read at a time.
CHUNK_BYTES = 64 * 1024

#: How long one presigned download may take.
#:
#: SIZED FROM THE URL'S OWN LIFE: a presigned GET is good for about five minutes,
#: so a transfer that has not finished inside two is a transfer that is almost
#: certainly not going to — the URL cannot be re-used to resume it, and a client
#: that waited longer would hold a GUI request open past the point where the only
#: useful remedy is to ask for a fresh listing.
ARTIFACT_TIMEOUT_SECONDS = 120.0


@dataclasses.dataclass(frozen=True)
class FetchedResults:
    """Where a run's results were extracted, and how much of them there is."""

    directory: Path
    archive_bytes: int
    file_count: int


@dataclasses.dataclass(frozen=True)
class AppliedBundle:
    """The ref a run's bundle now answers to, and the commit it points at."""

    ref: str
    commit: str


@dataclasses.dataclass(frozen=True)
class RestoredSession:
    """The run's session id, and whether this call is what put it on this device."""

    session_id: str
    created: bool


def artifact_client() -> httpx.AsyncClient:
    """A FRESH client for one presigned download: no Radient bearer, no redirects.

    THE SEPARATION IS THE POINT, and it is why this is a seam a test replaces: the
    download must not travel on the client that holds the account's credential.
    Redirects are refused because a presigned GET that answers ``302`` is a URL
    life expiring or a bucket moved, and following one would send this run's
    signature wherever the answer pointed.
    """
    return httpx.AsyncClient(timeout=ARTIFACT_TIMEOUT_SECONDS, follow_redirects=False)


def results_directory(identifier: str, *, config_dir: Path | None = None) -> Path:
    """``<config dir>/remote-compute/results/<run id>``: where a fetch lands by default.

    Resolved per call exactly as :func:`~local_operator.remote_compute.records.records_path`
    does, so a relocated config root (a daemon started against one) is honoured
    without rebuilding anything.
    """
    if config_dir is None:
        from local_operator.paths import config_dir as default_config_dir

        config_dir = default_config_dir()
    return Path(config_dir).joinpath(*RESULTS_SUBDIRECTORY, checked_run_id(identifier))


def checked_run_id(identifier: str) -> str:
    """``identifier`` when it is safe to write into a path or a ref, else a refusal.

    ONE SPELLING OF THE QUESTION for both uses, because they fail together: an id
    that cannot be a path component is also one that cannot be a ref name.
    """
    if not isinstance(identifier, str) or RUN_ID_PATTERN.fullmatch(identifier) is None:
        raise ResultsUnavailable(REASON_INVALID_RUN_ID, detail=str(identifier)[:64])
    return identifier


async def fetch_results(
    client: "RemoteComputeClient", identifier: str, destination: Path
) -> FetchedResults:
    """Download ``results.tar.gz`` and extract it into a NEW ``destination``.

    THE ORDER IS THE SAFETY. The archive is downloaded into a private temp file
    and checked against the size the listing advertised BEFORE anything is
    extracted, and the extraction target is created only once that check passed —
    so a truncated, padded or refused download leaves no half-populated results
    directory for a surface to offer.
    """
    directory = Path(destination)
    if not directory.is_absolute():
        raise ResultsUnavailable(REASON_RELATIVE_DESTINATION, detail=str(directory))
    listing = await client.run_artifacts(identifier)
    artifact = listing.artifact(RESULTS_ARTIFACT)
    if artifact is None:
        raise ResultsUnavailable(REASON_NO_RESULTS, detail=RESULTS_ARTIFACT)
    with tempfile.TemporaryDirectory(prefix="lop-results-") as scratch:
        archive = Path(scratch) / RESULTS_ARTIFACT
        written = await _download(artifact, archive)
        file_count = extract(archive, directory)
    return FetchedResults(directory=directory, archive_bytes=written, file_count=file_count)


async def apply_bundle(
    client: "RemoteComputeClient", identifier: str, repo_dir: Path | str
) -> AppliedBundle:
    """Fetch the run's ``repo.bundle`` into ``repo_dir`` as ``refs/remotes/radient/lop/<id>``.

    ``git bundle verify`` RUNS FIRST AND IN THE REPOSITORY, because that is what
    it checks: a bundle records the commits it assumes are already present, and a
    clone that does not hold the run's base refuses here instead of fetching a
    broken history. Only then is the bundle fetched.

    NOTHING IN THE OPERATOR'S CHECKOUT MOVES. The refspec writes one
    ``refs/remotes/radient/lop/<id>`` and nothing else, ``--no-tags`` keeps the
    bundle's tags out of their repository (fetch follows tags by default, and a
    tag is a name in their namespace this feature has no business claiming), and
    no checkout, branch or index write happens here at all. ``FETCH_HEAD`` is the
    one file git updates by itself; it is neither HEAD, a branch nor the index.

    THE ENVIRONMENT IS ``repo._git``'s, so every inherited ``GIT_*`` variable is
    dropped for the same reason that module documents: a shell's ``GIT_DIR`` would
    otherwise point this fetch at another checkout. Reused rather than restated,
    because a second copy of that scrub is a second thing to keep right — and run
    OFF THE EVENT LOOP by :func:`_git`, because a fetch may take up to its timeout
    and the daemon serves every other request on the loop it is awaited from.
    """
    run = checked_run_id(identifier)
    root = Path(repo_dir)
    if not root.is_dir():
        raise BundleApplyFailed(REASON_NOT_A_DIRECTORY, detail=str(root))
    listing = await client.run_artifacts(identifier)
    artifact = listing.artifact(BUNDLE_ARTIFACT)
    if artifact is None:
        raise ResultsUnavailable(REASON_NO_BUNDLE, detail=BUNDLE_ARTIFACT)
    ref = BUNDLE_REF_PREFIX + run
    refspec = f"{BUNDLE_SOURCE_PREFIX}{run}:{ref}"
    with tempfile.TemporaryDirectory(prefix="lop-bundle-") as scratch:
        bundle = Path(scratch) / BUNDLE_ARTIFACT
        await _download(artifact, bundle)
        code, _ = await _git(root, "bundle", "verify", str(bundle))
        if code != 0:
            raise BundleApplyFailed(REASON_BUNDLE_INVALID, detail=refspec)
        code, _ = await _git(root, "fetch", "--no-tags", str(bundle), refspec)
        if code != 0:
            raise BundleApplyFailed(REASON_FETCH_REFUSED, detail=refspec)
    code, commit = await _git(
        root, "rev-parse", "--verify", ref, timeout=repo_module.GIT_TIMEOUT_SECONDS
    )
    if code != 0 or not commit:
        raise BundleApplyFailed(REASON_FETCH_REFUSED, detail=ref)
    return AppliedBundle(ref=ref, commit=commit)


async def _git(
    root: Path, *arguments: str, timeout: float = repo_module.GIT_FETCH_TIMEOUT_SECONDS
) -> tuple[int, str]:
    """One git command in ``root``, run in a worker thread.

    ``repo._git`` IS THE RUNNER, and the private name is deliberate: it is where the
    ``GIT_*`` scrub and the ``(returncode, stdout)`` shape live, and a second runner
    here would be a second copy of the one thing this package must not get wrong.
    ``to_thread`` because these calls block for as long as git takes.
    """
    runner = repo_module._git  # noqa: SLF001 — the one git runner, with its GIT_* scrub
    return await asyncio.to_thread(runner, root, *arguments, timeout=timeout)


def restore_session(extracted: Path, *, config_dir: Path | None = None) -> RestoredSession | None:
    """Place the run's session directory in this device's sessions store, once.

    NEVER OVERWRITES. A session with that id already on this device is the
    operator's own history — it may be longer than the archive's copy, since the
    run's archive stops when the run stopped — so the copy is skipped and the
    answer says so (``created=False``). That is also why the tree is COPIED rather
    than moved: the extracted results stay complete for whoever looks at them.

    THE RUN ID IS NOT A PARAMETER, and that is not an omission: the run id names
    the run on the server, while the session directory inside the archive names
    the session, and the contract does not tie the two spellings together. The id
    a caller offers ``lop --resume`` is the one read out of the archive.

    ``None`` means the archive carried no session directory, which is a legitimate
    outcome (a run that published results but no session, or an archive packed
    differently) and not an error: the caller offers no resume and nothing else
    about the fetch changes.

    A FORK RIDES ALONG. If the archive carries more than one session directory,
    every one is restored and the run's own is the answer — by name order, because
    nothing in the archive says which is the parent, and restoring only one would
    silently drop a conversation.
    """
    root = Path(config_dir) if config_dir is not None else _default_config_dir()
    sources = _session_directories(extracted)
    if not sources:
        return None
    restored: list[RestoredSession] = []
    for source in sources:
        restored.append(_restore_one(source, root))
    return restored[0]


def extract(archive: Path, destination: Path) -> int:
    """Extract a results archive into a NEW directory, refusing every unsafe entry.

    THE DESTINATION MUST NOT EXIST. It is a directory this call creates: writing
    into one that is already there is how a fetch comes to overwrite a file that
    was not its own, and there is no local provenance a read could check. The
    refusal names the path and the caller picks another.

    ONE MEMBER AT A TIME, WITH THIS MODULE'S OWN WRITES. ``extractall`` is not
    used for two reasons: a filter that raises from inside it has already written
    the members before it, and these refusals have to be TYPED and reason-coded so
    a surface can say what the archive tried to do. ``tarfile``'s own ``data``
    filter answers the PATH questions (an absolute name, a name that climbs out, a
    link target that escapes) and this module answers the TYPE questions it is
    deliberately softer about — any link at all, and anything that is neither a
    regular file nor a directory — before that filter is consulted, so its
    refusals are only ever about paths.

    A LINK IS REFUSED EVEN WHEN IT STAYS INSIDE. A session directory is a
    transcript and its sidecars; a symbolic or hard link buys nothing in it, and a
    link that is inside the tree now can point outside it after the next write
    into that tree. Refusing the family is one rule instead of a containment proof
    that has to be re-derived per member — and it is why the ``data`` filter's
    "a link that resolves inside is fine" arm is never reached.

    A FILE IS CREATED 0600 AND EVERY DIRECTORY IS 0700. The file's mode is set at
    creation, because the window between a default-mode create and a chmod is a
    window something else can read through, and a transcript can quote an
    operator's prompt. The directories are swept once at the end (and again on the
    refusal path) rather than chmod-ed as they are made: ``mkdir(parents=True)``
    creates the intermediate levels with the umask's mode, so a per-call chmod would
    leave every parent of a deep member at 0755 — the sweep is what makes the rule
    true of the whole tree.
    """
    if destination.exists():
        raise ResultsUnavailable(REASON_DESTINATION_EXISTS, detail=str(destination))
    destination.mkdir(parents=True, mode=DIRECTORY_MODE, exist_ok=True)
    _restrict_directories(destination)
    files = 0
    try:
        opened = tarfile.open(archive, "r:gz")
    except (tarfile.TarError, OSError) as failure:
        raise ResultsUnavailable(REASON_UNREADABLE_ARCHIVE, detail=archive.name) from failure
    with opened:
        try:
            for member in opened:
                _checked(member, destination)
                target = destination / member.name
                if member.isdir():
                    target.mkdir(parents=True, mode=DIRECTORY_MODE, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, mode=DIRECTORY_MODE, exist_ok=True)
                source = opened.extractfile(member)
                if source is None:
                    # Unreachable for a regular file (checked above); refused
                    # rather than skipped so a member this module cannot read is
                    # never silently dropped from the results.
                    raise UnsafeArchive(REASON_SPECIAL_FILE, detail=member.name)
                with source, _private_file(target) as sink:
                    shutil.copyfileobj(source, sink, CHUNK_BYTES)
                files += 1
        finally:
            # Runs on the refusal path too: a bare ``mkdir`` above creates
            # intermediates with the umask's mode, and a half-extracted tree must
            # not be left group- or world-readable.
            _restrict_directories(destination)
    return files


async def _download(artifact: "Artifact", destination: Path) -> int:
    """Stream ``artifact`` to ``destination``, refusing a body of the wrong size.

    THE ADVERTISED ``size_bytes`` IS A BOUND AS WELL AS A CHECK. A body that grows
    past it is abandoned at the first chunk that crosses it, so a server or a
    proxy answering with an unbounded stream cannot fill this disk; a body that
    ends short is refused after the transfer. Either way nothing is extracted from
    it.

    NEITHER THE URL NOR THE BODY REACHES AN ERROR: the reasons name the artifact
    and the status, never the request.
    """
    written = 0
    with _unlogged_request():
        async with artifact_client() as client:
            async with client.stream("GET", artifact.url) as response:
                if response.status_code != 200:
                    raise ResultsUnavailable(
                        REASON_DOWNLOAD_REFUSED,
                        detail=f"{artifact.name} (HTTP {response.status_code})",
                    )
                with _private_file(destination) as sink:
                    async for chunk in response.aiter_bytes(CHUNK_BYTES):
                        written += len(chunk)
                        if written > artifact.size_bytes:
                            raise SizeMismatch(
                                artifact=artifact.name,
                                expected=artifact.size_bytes,
                                actual=written,
                            )
                        sink.write(chunk)
    if written != artifact.size_bytes:
        raise SizeMismatch(artifact=artifact.name, expected=artifact.size_bytes, actual=written)
    return written


def _private_file(path: Path) -> IO[bytes]:
    """Create ``path`` owner-only and open it for writing."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, FILE_MODE)
    return os.fdopen(descriptor, "wb")


def _restrict_directories(root: Path) -> None:
    """Give every directory under ``root`` (and ``root``) mode 0700."""
    for directory, _subdirectories, _files in os.walk(root):
        os.chmod(directory, DIRECTORY_MODE)


def _checked(member: tarfile.TarInfo, destination: Path) -> None:
    """Refuse a member this module will not write, with a reason a surface can name."""
    if member.issym() or member.islnk():
        raise UnsafeArchive(REASON_LINK, detail=member.name)
    if not (member.isfile() or member.isdir()):
        raise UnsafeArchive(REASON_SPECIAL_FILE, detail=member.name)
    # AN ABSOLUTE NAME IS REFUSED BEFORE THE FILTER SEES IT, and this is the one
    # place where ``data_filter`` is not the right judge: it STRIPS a leading
    # separator and extracts the entry under a mangled path INSIDE the destination
    # (measured on 3.12 — ``/etc/passwd`` lands as ``<dest>/etc/passwd``). That is
    # not an escape, but it is not an archive this device should read either, and a
    # refusal is what the contract asks for.
    if _absolute(member.name):
        raise UnsafeArchive(REASON_ABSOLUTE_PATH, detail=member.name)
    filter_ = getattr(tarfile, "data_filter", None)
    if filter_ is None:  # pragma: no cover - requires-python >= 3.12 provides it
        _refuse_escape(destination, member.name)
        return
    try:
        filter_(member, str(destination))
    except tarfile.TarError as refusal:
        raise UnsafeArchive(_filter_reason(refusal), detail=member.name) from None


def _absolute(name: str) -> bool:
    """Whether an archive member's name is absolute, on any platform's spelling.

    THREE SPELLINGS, because a tar written on Windows is read here: a leading ``/``
    or ``\\`` (tar's separator), and a drive prefix (``C:/x``, which
    ``os.path.isabs`` answers ``False`` for on this host).
    """
    if name.startswith(("/", "\\")):
        return True
    return len(name) > 2 and name[1] == ":" and name[2] in "/\\" and name[0].isalpha()


@contextlib.contextmanager
def _unlogged_request() -> Iterator[None]:
    """Keep ``httpx``'s request line out of the log while a presigned GET runs.

    THE LINE IS THE CREDENTIAL: httpx logs ``HTTP Request: GET <url> ``"HTTP/1.1
    200 OK"`` at INFO, and a presigned URL carries its signature in the query
    string. The product already pins this logger to WARNING in every entry point
    (``logger.quiet_wire_clients``), so today it is never emitted — but a secret
    that is safe only because of a level set somewhere else is not protected by
    this module.

    A LEVEL PIN FOR THE WINDOW rather than a log filter: a filter that rewrites
    ``record.args`` would have to know httpx's format string, and the cost of the
    pin is only that another request's line in the same window is not logged,
    which is a diagnostic the product turns off anyway.
    """
    logger = logging.getLogger("httpx")
    previous = logger.level
    logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        logger.setLevel(previous)


def _filter_reason(refusal: tarfile.TarError) -> str:
    """The reason code for a refusal from ``tarfile``'s own data filter.

    THE TWO PATH ARMS ARE NAMED AND THE REST ARE NOT: a link or a special file is
    already refused above, so what the filter can still refuse here is a path that
    is absolute or one that leaves the destination — and an arm added later is
    reported as "not writable here" rather than as a case this build pretends to
    have understood.
    """
    absolute = getattr(tarfile, "AbsolutePathError", None)
    if absolute is not None and isinstance(refusal, absolute):
        return REASON_ABSOLUTE_PATH
    return REASON_OUTSIDE_DESTINATION


def _refuse_escape(destination: Path, name: str) -> None:  # pragma: no cover
    """The containment check for an interpreter without ``tarfile.data_filter``."""
    candidate = Path(name)
    if candidate.is_absolute():
        raise UnsafeArchive(REASON_ABSOLUTE_PATH, detail=name)
    resolved = (destination / candidate).resolve()
    if not resolved.is_relative_to(destination.resolve()):
        raise UnsafeArchive(REASON_OUTSIDE_DESTINATION, detail=name)


def _session_directories(extracted: Path) -> list[Path]:
    """Every session directory the archive carried, in name order."""
    sessions = extracted / SESSIONS_DIRECTORY
    if not sessions.is_dir():
        return []
    return sorted(
        child
        for child in sessions.iterdir()
        if child.is_dir() and not child.is_symlink() and is_session_id(child.name)
    )


def _restore_one(source: Path, config_dir: Path) -> RestoredSession:
    """Copy one session directory into ``config_dir`` unless that id is already there.

    ``copytree`` creates the destination with ``exist_ok=False``, so the "already
    there" arm is the directory's own atomic refusal — which is also why a racing
    restore of the same id answers ``created=False`` instead of half-copying over
    the other one. ``symlinks=True`` copies a link as a link: extraction refused
    every link already, so this is a second refusal rather than a dereference of
    something unexpected.
    """
    target = config_dir / SESSIONS_DIRECTORY / source.name
    if target.exists():
        return RestoredSession(session_id=source.name, created=False)
    target.parent.mkdir(parents=True, mode=DIRECTORY_MODE, exist_ok=True)
    try:
        shutil.copytree(source, target, symlinks=True)
    except FileExistsError:
        return RestoredSession(session_id=source.name, created=False)
    return RestoredSession(session_id=source.name, created=True)


def _default_config_dir() -> Path:
    from local_operator.paths import config_dir as default_config_dir

    return default_config_dir()


def bundle_ref(identifier: str) -> str:
    """The ref a run's bundle is fetched to, for a caller that only wants to name it."""
    return BUNDLE_REF_PREFIX + checked_run_id(identifier)


__all__ = [
    "ARTIFACT_TIMEOUT_SECONDS",
    "BUNDLE_ARTIFACT",
    "BUNDLE_REF_PREFIX",
    "BUNDLE_SOURCE_PREFIX",
    "REASON_ABSOLUTE_PATH",
    "REASON_BUNDLE_INVALID",
    "REASON_DESTINATION_EXISTS",
    "REASON_DOWNLOAD_REFUSED",
    "REASON_FETCH_REFUSED",
    "REASON_INVALID_RUN_ID",
    "REASON_LINK",
    "REASON_NO_BUNDLE",
    "REASON_NO_RESULTS",
    "REASON_NOT_A_DIRECTORY",
    "REASON_OUTSIDE_DESTINATION",
    "REASON_RELATIVE_DESTINATION",
    "REASON_SPECIAL_FILE",
    "REASON_UNREADABLE_ARCHIVE",
    "RESULTS_ARTIFACT",
    "RESULTS_SUBDIRECTORY",
    "SESSIONS_DIRECTORY",
    "AppliedBundle",
    "FetchedResults",
    "RestoredSession",
    "apply_bundle",
    "artifact_client",
    "bundle_ref",
    "checked_run_id",
    "extract",
    "fetch_results",
    "restore_session",
    "results_directory",
]
