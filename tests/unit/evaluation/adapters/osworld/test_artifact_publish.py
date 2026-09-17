"""Atomic publication of content-addressed artifacts.

Every artifact here is addressed by the sha256 of its own bytes and the parent
reopens ``<root>/<name>`` with ``O_NOFOLLOW``, re-hashes it and compares size,
digest and media type against what the event declared
(``adapters.supervisor.verify_artifact``). So "the name exists" and "the bytes
are there" have to be ONE fact, and a create-then-write cannot make them one:
the name lands at the create, and a write that fails in between leaves a 0-byte
file sitting permanently under a valid address, because every later attempt sees
the name present and skips its own write.

That is the failure a live episode died of, and it is why these tests exist
rather than a test of the happy path. Its evidence is preserved at
``runs/batch-deepseek-flash-canary9/task_012`` of the OSWorld worktree: the
publish raised ``OSError: [Errno 28] No space left on device``, the retry found
the create's 0-byte file, declared the frame's true ``byte_count`` against it,
and the parent refused the artifact as "not a matching regular file" -- a
corrupted-bundle refusal, ``retryable: false``, which killed an episode that had
done 88 steps of real work.

The planted leftovers below are byte-for-byte the shape that run left on disk:
the published name, with none or only part of its bytes. The real
``verify_artifact`` is the judge of the result, because it is the same call the
parent's HostVerifier makes.
"""

from __future__ import annotations

import errno
import hashlib
import signal
from pathlib import Path

import pytest
from lop_osworld_v2_adapter import scoring
from lop_osworld_v2_adapter.observation import (
    NATIVE_SCREEN,
    ObservationBuilder,
    write_png_rgb,
)

from local_operator.evaluation.adapters.supervisor import verify_artifact
from local_operator.evaluation.protocol import Observation


def _raw(shade: int = 1) -> dict[str, object]:
    width, height = NATIVE_SCREEN.width, NATIVE_SCREEN.height
    return {
        "screenshot": write_png_rgb(width, height, bytes((shade, shade, shade)) * (width * height)),
        "accessibility_tree": None,
        "terminal": None,
        "instruction": "do the thing",
    }


def _build(root: Path, shade: int) -> Observation:
    """One observation built by a FRESH builder, i.e. a retried worker."""

    return ObservationBuilder(root).build(_raw(shade), task_id="t", episode_id="e", sequence=0)


def _entries(root: Path) -> set[str]:
    return {entry.name for entry in root.iterdir()}


@pytest.mark.parametrize("leftover", ["none_written", "partially_written"])
def test_a_leftover_at_the_content_address_is_healed(tmp_path: Path, leftover: str) -> None:
    """A retry over a poisoned address must end in real bytes, not a refusal.

    ``none_written`` is the live 0-byte case exactly. ``partially_written`` is
    the same poisoning one failure window later -- a truncated file is refused
    by the parent just as fatally, since it declares the full ``byte_count``.
    Both must be REPLACED: an address left behind by an interrupted publish is
    not evidence that those bytes are published.
    """

    first = _build(tmp_path, shade=11)
    artifact = first.frames[0].artifact
    path = tmp_path / artifact.sha256
    payload = path.read_bytes()

    path.write_bytes(b"" if leftover == "none_written" else payload[: len(payload) // 2])

    retry = _build(tmp_path, shade=11)
    assert retry.frames[0].artifact.sha256 == artifact.sha256
    # The size half of the contract first, because it is the half the live
    # refusal collapsed into "artifact is not a matching regular file".
    assert path.stat().st_size == artifact.byte_count
    # ...and then the verifier that raised it, on the same bytes.
    assert verify_artifact(tmp_path, retry.frames[0].artifact) == payload
    assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact.sha256


def test_a_publish_that_fails_mid_write_leaves_nothing_at_the_address(tmp_path: Path) -> None:
    """The write itself must fail without landing the name.

    The failure is a REAL one from the kernel rather than a patched writer:
    ``RLIMIT_FSIZE`` caps how many bytes this process may put in a file, so the
    write of a frame larger than the cap fails partway through exactly as a full
    disk makes it fail (``EFBIG`` here, ``ENOSPC`` on the live run -- a
    file-size limit is what a test can impose portably, and it is enforced by
    the same syscall path). ``SIGXFSZ`` is ignored because its default action is
    to kill the process outright instead of returning the error.

    On the pre-fix tree this test fails at the first assertion: the create had
    already landed the NAME under the digest, and the write that followed it
    stopped at the limit rather than at zero. What the retry then finds is a
    partial file -- ``byte_count // 4`` bytes here, 25% of the frame, while a
    limit of 0 leaves the 0-byte shape the live run left. The poisoning is the
    name landing without its bytes, so no particular length is part of it.
    """

    resource = pytest.importorskip("resource", reason="RLIMIT_FSIZE is POSIX-only")

    builder = ObservationBuilder(tmp_path)
    raw = _raw(shade=13)
    # Publish once so the payload's true size is known, then take the artifact
    # away: the failure has to land on a real publish of the bytes the frame
    # declares, not on a fixture this test invented.
    published = builder.build(raw, task_id="t", episode_id="e", sequence=0)
    artifact = published.frames[0].artifact
    (tmp_path / artifact.sha256).unlink()

    soft, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
    previous_handler = signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    try:
        # A quarter of the payload, derived from the payload rather than a fixed
        # number of bytes: the frame's own write has to be what trips the limit,
        # and a constant would be overtaken by any future change to the bounded
        # frame's size, quietly turning this into a test of nothing.
        resource.setrlimit(resource.RLIMIT_FSIZE, (artifact.byte_count // 4, hard))
        with pytest.raises(OSError) as raised:
            builder.build(raw, task_id="t", episode_id="e", sequence=0)
        assert raised.value.errno is not None
    finally:
        resource.setrlimit(resource.RLIMIT_FSIZE, (soft, hard))
        signal.signal(signal.SIGXFSZ, previous_handler)

    assert not (
        tmp_path / artifact.sha256
    ).exists(), "a failed publish left a name at the content address"
    assert _entries(tmp_path) == set(), "a failed publish left its temporary file behind"

    # The same bytes must publish cleanly once the disk is not full again.
    healed = builder.build(raw, task_id="t", episode_id="e", sequence=0)
    assert (tmp_path / artifact.sha256).stat().st_size == artifact.byte_count
    assert verify_artifact(tmp_path, healed.frames[0].artifact)


def test_a_second_publish_of_identical_content_is_correct_and_leaves_no_temp_file(
    tmp_path: Path,
) -> None:
    """The dedup survives the fix, and only ever skips a COMPLETE file."""

    first = _build(tmp_path, shade=17)
    artifact = first.frames[0].artifact
    path = tmp_path / artifact.sha256
    published_inode = path.stat().st_ino

    retry = _build(tmp_path, shade=17)
    assert retry.frames[0].artifact.sha256 == artifact.sha256
    assert _entries(tmp_path) == {artifact.sha256}
    # Reused, not rewritten: a repeat of an identical frame (a ``wait``, a
    # no-op click) still costs no write, which is the whole point of the check.
    assert path.stat().st_ino == published_inode
    assert verify_artifact(tmp_path, retry.frames[0].artifact) == path.read_bytes()


def test_a_link_at_the_address_is_replaced_rather_than_written_through(tmp_path: Path) -> None:
    """A non-regular entry is repaired instead of aimed at.

    ``verify_artifact`` opens the name with ``O_NOFOLLOW`` and refuses anything
    that is not a matching regular file, so a link at a digest name is a fatal
    bundle refusal -- and a plain ``write_bytes`` aims the frame's bytes at
    whatever that link points to, outside the artifact root, while leaving the
    refusal in place.
    """

    elsewhere = tmp_path / "elsewhere"
    elsewhere.write_bytes(b"not the frame")

    first = _build(tmp_path, shade=19)
    artifact = first.frames[0].artifact
    path = tmp_path / artifact.sha256
    payload = path.read_bytes()
    path.unlink()
    path.symlink_to(elsewhere)

    retry = _build(tmp_path, shade=19)
    assert not path.is_symlink()
    assert elsewhere.read_bytes() == b"not the frame"
    assert verify_artifact(tmp_path, retry.frames[0].artifact) == payload


def test_a_directory_at_the_address_is_refused_rather_than_skipped(tmp_path: Path) -> None:
    """A broken address that cannot be repaired is REPORTED, not hidden.

    A symlink or a FIFO is repaired in place (above); a DIRECTORY is the shape
    that cannot be, because ``os.replace`` will not overwrite one. It is
    deliberately not treated as published: the name existing is not the same
    fact as the bytes being there, and the pre-fix dedup -- which asked only
    whether the name existed -- declared the frame's true ``byte_count``
    against a directory that the parent then refused as an unsafe artifact
    path, long after the publish that should have reported it had returned.

    So the publish raises the kernel's own error, the address is left exactly
    as it was found, and no temp file survives the attempt. This pins that as
    the contract rather than an accident of ``os.replace``.
    """

    first = _build(tmp_path, shade=23)
    artifact = first.frames[0].artifact
    path = tmp_path / artifact.sha256
    path.unlink()
    path.mkdir()

    with pytest.raises(OSError) as raised:
        _build(tmp_path, shade=23)
    assert raised.value.errno == errno.EISDIR

    assert path.is_dir(), "the refusal consumed the directory it refused"
    assert _entries(tmp_path) == {artifact.sha256}, "the failed publish left a temp file"


def test_a_leftover_score_detail_is_healed_too(tmp_path: Path) -> None:
    """The score detail publishes through the same primitive, same contract.

    Its blast radius is smaller than a frame's -- a refused detail loses the
    episode's score rather than the episode -- but a create-then-write leaves the
    same permanently refused address, so it is fixed by the same code path.
    """

    first = scoring.score_to_artifact(0.5, artifact_root=tmp_path)
    assert first.details is not None
    path = tmp_path / first.details.sha256
    payload = path.read_bytes()
    path.write_bytes(b"")

    retry = scoring.score_to_artifact(0.5, artifact_root=tmp_path)
    assert retry.details is not None
    assert retry.details.sha256 == first.details.sha256
    assert path.stat().st_size == retry.details.byte_count
    assert path.read_bytes() == payload
    assert _entries(tmp_path) == {first.details.sha256}
